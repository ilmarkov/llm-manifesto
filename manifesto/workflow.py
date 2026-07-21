"""Higher-level workflow helpers for the manifesto CLI."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .cluster import load_cluster
from .instance import Instance
from .render import render, render_to_yaml
from .spec import RoutingKind, load_spec


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR_NAME = "llm-manifesto"


class WorkflowError(RuntimeError):
    """Expected error that should be printed without a traceback."""

    def __init__(self, message: str, *, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RuntimeConfig:
    user: str
    namespace: str
    cluster_path: str | None
    render_out: Path

    @classmethod
    def from_args(cls, args, *, require_cluster: bool = True) -> "RuntimeConfig":
        load_dotenv()
        user = resolve_user(getattr(args, "user", None))
        namespace = resolve_namespace(getattr(args, "namespace", None))
        cluster_path = resolve_cluster(getattr(args, "cluster", None)) if require_cluster else getattr(args, "cluster", None)
        render_out = Path(
            getattr(args, "output", None)
            or os.environ.get("MANIFESTO_RENDER_OUT", f"/tmp/{user}-manifesto.yaml")
        )
        return cls(user=user, namespace=namespace, cluster_path=cluster_path, render_out=render_out)

    def kubectl(self) -> list[str]:
        return ["kubectl", "-n", self.namespace]


def config_home() -> Path:
    if configured := os.environ.get("MANIFESTO_CONFIG_HOME"):
        return Path(configured).expanduser()
    if xdg_home := os.environ.get("XDG_CONFIG_HOME"):
        return Path(xdg_home).expanduser() / CONFIG_DIR_NAME
    return Path.home() / ".config" / CONFIG_DIR_NAME


def load_dotenv(path: Path | None = None) -> None:
    paths = [path] if path is not None else [config_home() / ".env", ROOT / ".env"]
    for env_path in paths:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line.removeprefix("export ").strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


def resolve_model(value: str) -> str:
    return _resolve_catalog_path(value, "models")


def _resolve_catalog_path(value: str, catalog: str) -> str:
    path = Path(value).expanduser()
    variants = [path]
    if not path.suffix:
        variants.append(path.with_suffix(".yaml"))

    for candidate in variants:
        if candidate.exists():
            return str(candidate)
    if path.is_absolute():
        return str(path)

    for root in (config_home() / catalog, ROOT / catalog):
        for candidate in variants:
            resolved = root / candidate
            if resolved.exists():
                return str(resolved)
    return value


def resolve_user(explicit: str | None = None) -> str:
    return explicit or os.environ.get("USER") or "dev"


def resolve_namespace(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if os.environ.get("MANIFESTO_NAMESPACE"):
        return os.environ["MANIFESTO_NAMESPACE"]
    namespace = capture(["kubectl", "config", "view", "--minify", "-o", "jsonpath={..namespace}"], check=False)
    return namespace.strip() or "default"


def resolve_cluster(explicit: str | None = None) -> str:
    if explicit:
        return _resolve_catalog_path(explicit, "clusters")
    if os.environ.get("MANIFESTO_CLUSTER"):
        return _resolve_catalog_path(os.environ["MANIFESTO_CLUSTER"], "clusters")
    mapping = os.environ.get("MANIFESTO_CLUSTER_MAP", "")
    context = capture(["kubectl", "config", "current-context"], check=False).strip()
    kube_cluster = capture(
        ["kubectl", "config", "view", "--minify", "-o", "jsonpath={.clusters[0].name}"],
        check=False,
    ).strip()
    if mapping:
        for entry in mapping.split(","):
            key, sep, value = entry.partition("=")
            if sep and key.strip() in {context, kube_cluster}:
                return _resolve_catalog_path(value.strip(), "clusters")
    for name in (context, kube_cluster):
        if not name:
            continue
        candidate = _resolve_catalog_path(name, "clusters")
        if Path(candidate).exists():
            return candidate
    raise WorkflowError(
        "No cluster profile configured. Pass --cluster, set MANIFESTO_CLUSTER, "
        "add the current kube context to MANIFESTO_CLUSTER_MAP, or create "
        f"{config_home() / 'clusters' / '<context>.yaml'}.",
        code=2,
    )


def load_runtime_cluster(config: RuntimeConfig, args):
    if not config.cluster_path:
        raise WorkflowError("No cluster profile configured.", code=2)
    return load_cluster_with_overrides(config.cluster_path, args)


def load_cluster_with_overrides(cluster_path: str, args):
    return load_cluster(cluster_path).with_path_overrides(
        user_root=getattr(args, "user_root", None),
        log_root=getattr(args, "log_root", None),
        cache_root=getattr(args, "cache_root", None),
        dev_venv=getattr(args, "dev_venv", None),
        dev_source=getattr(args, "dev_source", None),
    )


def apply_runtime_overrides(spec, args, config: RuntimeConfig) -> None:
    spec.namespace = config.namespace
    if getattr(args, "dev", False):
        spec.runtime.dev = True
    if getattr(args, "dev_venv", None):
        spec.runtime.dev_venv = args.dev_venv
    spec.runtime.pre_launch.extend(getattr(args, "pre_launch", None) or [])


def render_manifest(args, config: RuntimeConfig, *, routing_only: bool = False) -> str:
    cluster = load_runtime_cluster(config, args)
    spec = load_spec(resolve_model(args.spec), cluster)
    apply_runtime_overrides(spec, args, config)
    return render_to_yaml(
        render(spec, user=config.user, cluster=cluster, routing_only=routing_only),
        header=manifest_header(args, config, routing_only=routing_only),
    )


def manifest_header(args, config: RuntimeConfig, *, routing_only: bool) -> list[str]:
    if not config.cluster_path:
        raise WorkflowError("No cluster profile configured.", code=2)
    command = [
        "manifesto",
        "render-routing" if routing_only else "render",
        resolve_model(args.spec),
        "--cluster",
        config.cluster_path,
        "--namespace",
        config.namespace,
        "--user",
        config.user,
    ]
    if getattr(args, "dev", False):
        command.append("--dev")
    for name in ("user_root", "log_root", "cache_root", "dev_venv", "dev_source"):
        value = getattr(args, name, None)
        if value:
            command.extend([f"--{name.replace('_', '-')}", value])
    for hook in getattr(args, "pre_launch", None) or []:
        command.extend(["--pre-launch", hook])
    return [
        "Generated by:",
        f"  {shlex.join(command)}",
        "Source: https://github.com/tlrmchlsmth/llm-manifesto",
        "Safe to edit before applying.",
    ]


def render_to_file(args) -> Path:
    config = RuntimeConfig.from_args(args)
    config.render_out.parent.mkdir(parents=True, exist_ok=True)
    config.render_out.write_text(render_manifest(args, config))
    return config.render_out


def deploy(args, *, routing_only: bool = False) -> int:
    config = RuntimeConfig.from_args(args)
    manifest = render_manifest(args, config, routing_only=routing_only)
    return run([*config.kubectl(), "apply", "-f", "-"], input_text=manifest)


def stop(args) -> int:
    config = RuntimeConfig.from_args(args)
    manifest = render_manifest(args, config)
    cmd = [*config.kubectl(), "delete", "-f", "-", "--ignore-not-found=true"]
    if args.now:
        cmd.extend(["--grace-period=0", "--force"])
    return run(cmd, input_text=manifest)


def diff_file(args) -> int:
    config = RuntimeConfig.from_args(args, require_cluster=False)
    return run([*config.kubectl(), "diff", "-f", str(config.render_out)])


def apply_file(args) -> int:
    config = RuntimeConfig.from_args(args, require_cluster=False)
    return run([*config.kubectl(), "apply", "-f", str(config.render_out)])


def delete_file(args) -> int:
    config = RuntimeConfig.from_args(args, require_cluster=False)
    cmd = [*config.kubectl(), "delete", "-f", str(config.render_out), "--ignore-not-found=true"]
    if args.now:
        cmd.extend(["--grace-period=0", "--force"])
    return run(cmd)


def ready(args) -> int:
    config = RuntimeConfig.from_args(args, require_cluster=False)
    spec = load_spec(resolve_model(args.spec))
    instance = Instance(user=config.user, release=spec.release)
    epp = instance.name("infpool-epp")
    routing_enabled = spec.routing.kind != RoutingKind.DISABLED

    gateway = ""
    if routing_enabled:
        cluster = load_cluster_with_overrides(resolve_cluster(config.cluster_path), args)
        gateway_name = instance.name("gateway", max_length=63 - len(cluster.gateway.class_name) - 1)
        gateway = f"{gateway_name}-{cluster.gateway.class_name}"

    print("Waiting for model pods and endpoint picker...")
    waits = [
        [
            *config.kubectl(),
            "wait",
            "--for=condition=Ready",
            "pod",
            "-l",
            ",".join(f"{key}={value}" for key, value in instance.pod_selector(role.name).items()),
            "--timeout=1200s",
        ]
        for role in spec.roles
    ]
    if routing_enabled:
        waits.append([*config.kubectl(), "wait", "--for=condition=Available", f"deploy/{epp}", "--timeout=120s"])
    procs = [subprocess.Popen(cmd) for cmd in waits]
    rc = max(proc.wait() for proc in procs)
    if rc:
        return rc
    if not routing_enabled:
        print("Ready.")
        return 0

    print("Checking gateway...")
    url = f"http://{gateway}:80/v1/models"
    deadline = time.monotonic() + args.gateway_timeout
    while time.monotonic() < deadline:
        out = capture(["curl", "-sf", "--max-time", "5", url], check=False)
        if '"id"' in out:
            print("Ready.")
            return 0
        out = capture(
            [*config.kubectl(), "exec", f"deploy/{epp}", "--", "curl", "-sf", "--max-time", "5", url],
            check=False,
        )
        if '"id"' in out:
            print("Ready.")
            return 0
        time.sleep(2)
    print(f"Gateway did not become ready within {args.gateway_timeout}s.", file=sys.stderr)
    return 1


def run(cmd: list[str], *, input_text: str | None = None) -> int:
    return subprocess.run(cmd, input=input_text, text=True).returncode


def capture(cmd: list[str], *, check: bool = True) -> str:
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        if check:
            raise WorkflowError(f"command not found: {cmd[0]}")
        return ""
    if check and proc.returncode != 0:
        raise WorkflowError(proc.stderr.strip() or f"command failed ({proc.returncode}): {shlex.join(cmd)}")
    return proc.stdout
