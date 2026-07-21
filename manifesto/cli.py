"""Command-line entrypoints for rendering manifests and printing derived names/paths."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

from .instance import Instance
from .render import render_to_yaml
from .render.devpod import render_dev_pod
from .spec import load_spec
from .workflow import (
    RuntimeConfig,
    WorkflowError,
    apply_file,
    delete_file,
    deploy,
    diff_file,
    load_cluster_with_overrides,
    load_dotenv,
    ready,
    render_manifest,
    render_to_file,
    resolve_cluster,
    resolve_model,
    resolve_user,
    stop,
)


def _render(args: argparse.Namespace, *, routing_only: bool = False) -> int:
    config = RuntimeConfig.from_args(args)
    sys.stdout.write(render_manifest(args, config, routing_only=routing_only))
    return 0


def _add_render_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("spec")
    parser.add_argument("--cluster")
    parser.add_argument("--namespace")
    parser.add_argument("--user")
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--user-root")
    parser.add_argument("--log-root")
    parser.add_argument("--cache-root")
    parser.add_argument("--dev-venv")
    parser.add_argument("--dev-source")
    parser.add_argument("--pre-launch", action="append", default=[])


def _render_file(args: argparse.Namespace) -> int:
    print(render_to_file(args))
    return 0


def _edit_file(args: argparse.Namespace) -> int:
    path = render_to_file(args)
    editor = os.environ.get("EDITOR", "vi")
    return subprocess.run([*shlex.split(editor), str(path)]).returncode


def _add_file_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--namespace")
    parser.add_argument("--user")
    parser.add_argument("--output")


def _add_cluster_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cluster")
    parser.add_argument("--user")
    parser.add_argument("--user-root")
    parser.add_argument("--log-root")
    parser.add_argument("--cache-root")
    parser.add_argument("--dev-venv")
    parser.add_argument("--dev-source")


def _add_ready_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("spec")
    parser.add_argument("--cluster")
    parser.add_argument("--namespace")
    parser.add_argument("--user")
    parser.add_argument("--gateway-timeout", type=int, default=120)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="manifesto")
    sub = parser.add_subparsers(dest="command", required=True)

    render_parser = sub.add_parser("render", help="render Kubernetes YAML to stdout")
    _add_render_args(render_parser)
    render_parser.set_defaults(func=lambda args: _render(args, routing_only=False))

    routing_parser = sub.add_parser("render-routing", help="render routing-only Kubernetes YAML to stdout")
    _add_render_args(routing_parser)
    routing_parser.set_defaults(func=lambda args: _render(args, routing_only=True))

    instance_parser = sub.add_parser("instance-id", help="print the instance label value for a spec")
    instance_parser.add_argument("spec")
    instance_parser.add_argument("--user")
    instance_parser.set_defaults(func=_instance_id)

    name_parser = sub.add_parser("name", help="print a generated component name for a spec")
    name_parser.add_argument("spec")
    name_parser.add_argument("component")
    name_parser.add_argument("--user")
    name_parser.set_defaults(func=_name)

    cache_parser = sub.add_parser("cache-path", help="print the resolved compile cache path")
    cache_parser.add_argument("spec")
    _add_cluster_args(cache_parser)
    cache_parser.set_defaults(func=_cache_path)

    log_parser = sub.add_parser("log-path", help="print the resolved persisted log directory")
    log_parser.add_argument("spec")
    log_parser.add_argument("--role", required=True)
    _add_cluster_args(log_parser)
    log_parser.set_defaults(func=_log_path)

    dev_path_parser = sub.add_parser("dev-path", help="print a resolved dev workflow path")
    dev_path_parser.add_argument("kind", choices=["venv", "source", "user-root"])
    _add_cluster_args(dev_path_parser)
    dev_path_parser.set_defaults(func=_dev_path)

    dev_pod_parser = sub.add_parser("render-dev-pod", help="render the persistent dev pod to stdout")
    _add_cluster_args(dev_pod_parser)
    dev_pod_parser.set_defaults(func=_render_dev_pod)

    render_file_parser = sub.add_parser("render-file", help="render a full manifest to the workflow file")
    _add_render_args(render_file_parser)
    render_file_parser.add_argument("-o", "--output")
    render_file_parser.set_defaults(func=_render_file)

    edit_file_parser = sub.add_parser("edit-file", help="render to the workflow file and open it in $EDITOR")
    _add_render_args(edit_file_parser)
    edit_file_parser.add_argument("-o", "--output")
    edit_file_parser.set_defaults(func=_edit_file)

    diff_parser = sub.add_parser("diff", help="kubectl diff the workflow file")
    _add_file_args(diff_parser)
    diff_parser.set_defaults(func=diff_file)

    apply_parser = sub.add_parser("apply", help="kubectl apply the workflow file")
    _add_file_args(apply_parser)
    apply_parser.set_defaults(func=apply_file)

    delete_parser = sub.add_parser("delete", help="kubectl delete objects from the workflow file")
    _add_file_args(delete_parser)
    delete_parser.add_argument("--now", action="store_true")
    delete_parser.set_defaults(func=delete_file)

    deploy_parser = sub.add_parser("deploy", help="render a spec and apply it to the cluster")
    _add_render_args(deploy_parser)
    deploy_parser.set_defaults(func=lambda args: deploy(args, routing_only=False))

    deploy_routing_parser = sub.add_parser("deploy-routing", help="render and apply routing objects only")
    _add_render_args(deploy_routing_parser)
    deploy_routing_parser.set_defaults(func=lambda args: deploy(args, routing_only=True))

    stop_parser = sub.add_parser("stop", help="render a spec and delete its objects from the cluster")
    _add_render_args(stop_parser)
    stop_parser.add_argument("--now", action="store_true")
    stop_parser.set_defaults(func=stop)

    ready_parser = sub.add_parser("ready", help="wait for model pods and gateway readiness")
    _add_ready_args(ready_parser)
    ready_parser.set_defaults(func=ready)

    try:
        args = parser.parse_args(argv)
        return args.func(args)
    except WorkflowError as exc:
        print(str(exc), file=sys.stderr)
        return exc.code


def _instance_id(args: argparse.Namespace) -> int:
    spec = load_spec(resolve_model(args.spec))
    print(Instance(user=resolve_user(args.user), release=spec.release).instance_id)
    return 0


def _name(args: argparse.Namespace) -> int:
    spec = load_spec(resolve_model(args.spec))
    print(Instance(user=resolve_user(args.user), release=spec.release).name(args.component))
    return 0


def _cache_path(args: argparse.Namespace) -> int:
    load_dotenv()
    cluster = load_cluster_with_overrides(resolve_cluster(args.cluster), args)
    spec = load_spec(resolve_model(args.spec))
    instance = Instance(user=resolve_user(args.user), release=spec.release)
    print(
        cluster.cache_root(
            user=instance.user_slug,
            release=instance.release_slug,
            gpu_arch=spec.cache.gpu_arch,
            cuda=spec.cache.cuda,
            cache_key=spec.cache_key,
        )
    )
    return 0


def _log_path(args: argparse.Namespace) -> int:
    load_dotenv()
    cluster = load_cluster_with_overrides(resolve_cluster(args.cluster), args)
    spec = load_spec(resolve_model(args.spec))
    instance = Instance(user=resolve_user(args.user), release=spec.release)
    print(f"{cluster.log_root(user=instance.user_slug, release=instance.release_slug)}/{args.role}")
    return 0


def _dev_path(args: argparse.Namespace) -> int:
    load_dotenv()
    cluster = load_cluster_with_overrides(resolve_cluster(args.cluster), args)
    user_slug = Instance(user=resolve_user(args.user), release="dev").user_slug
    paths = {
        "venv": cluster.dev_venv,
        "source": cluster.dev_source,
        "user-root": cluster.user_root,
    }
    print(paths[args.kind](user=user_slug, release=""))
    return 0


def _render_dev_pod(args: argparse.Namespace) -> int:
    load_dotenv()
    cluster = load_cluster_with_overrides(resolve_cluster(args.cluster), args)
    sys.stdout.write(render_to_yaml([render_dev_pod(cluster, resolve_user(args.user))]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
