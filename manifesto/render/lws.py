"""Workload renderer for vLLM roles, including sidecars and fabric mounts."""

from __future__ import annotations

import re
import shlex
from typing import Any

from .common import env_list, field_ref_env, secret_env
from .mooncake import _mooncake_device_name
from .sidecars import sidecars
from ..cluster import Cluster
from ..instance import Instance
from ..launch import build_launch_script
from ..parallelism import parallel_layout
from ..resolve import resolve_role
from ..spec import DeploymentSpec, DpLoadBalancing, RoleSpec


def _mooncake_init_container(instance: Instance) -> dict:
    """Init container that waits for mooncake-master to be ready (srt-slurm approach)."""
    master_host = instance.name("mooncake-master")
    return {
        "name": "wait-for-mooncake",
        "image": "busybox:1.36",
        "command": [
            "sh",
            "-c",
            f"""echo "Waiting for {master_host}:50051 (timeout 120s)..."
TIMEOUT=120
ELAPSED=0
while ! nc -z {master_host} 50051; do
  if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "ERROR: mooncake-master not ready after ${{TIMEOUT}}s"
    exit 1
  fi
  echo "mooncake-master not ready, sleeping 5s... (${{ELAPSED}}s elapsed)"
  sleep 5
  ELAPSED=$((ELAPSED + 5))
done
echo "mooncake-master is ready!"
""",
        ],
    }


def _mooncake_volume(instance: Instance) -> dict:
    """Volume definition for Mooncake config ConfigMap."""
    return {
        "name": "mooncake-config",
        "configMap": {"name": instance.name("mooncake-config")},
    }


def _mooncake_volume_mount() -> dict:
    """VolumeMount for Mooncake config."""
    return {
        "name": "mooncake-config",
        "mountPath": "/etc/mooncake",
        "readOnly": True,
    }


# Mooncake client version to install - must match the master image version
MOONCAKE_CLIENT_VERSION = "0.3.12.post1"


def _mooncake_upgrade_prefix() -> str:
    """Returns shell command prefix to upgrade mooncake client before vLLM starts.
    
    Uses mooncake-transfer-engine-cuda13 for CUDA 13.x compatibility (GB200/Blackwell).
    """
    return f"pip install --upgrade mooncake-transfer-engine-cuda13=={MOONCAKE_CLIENT_VERSION} && "


# Fixed per-process overhead added on top of a mooncake_client's own
# --global_segment_size when sizing its container memory limit (RPC/transfer-
# engine buffers, DirectIO staging buffer when offload is enabled, etc.) --
# same order of magnitude as the master's own 8Gi limit for its (much
# lighter) control-plane-only footprint.
_MOONCAKE_CLIENT_MEMORY_OVERHEAD_GIB = 8

_SIZE_MULTIPLIERS = {
    "": 1,
    "B": 1,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
    "KIB": 1024,
    "MIB": 1024**2,
    "GIB": 1024**3,
    "TIB": 1024**4,
}


def _parse_size_bytes(value: str) -> int:
    """Parse a human size string like '160GB' or '4GiB' into bytes."""
    match = re.fullmatch(r"\s*([\d.]+)\s*([A-Za-z]*)\s*", value)
    if not match:
        raise ValueError(f"Cannot parse size string: {value!r}")
    number, unit = match.groups()
    unit = unit.upper()
    if unit not in _SIZE_MULTIPLIERS:
        raise ValueError(f"Unknown size unit in {value!r}")
    return int(float(number) * _SIZE_MULTIPLIERS[unit])


def _mooncake_client_global_segment_bytes(spec: DeploymentSpec, role: RoleSpec) -> int:
    """Total bytes this pod's mooncake_client sidecar must contribute.

    spec.mooncake.global_segment_size is the *per-rank* contribution (same
    meaning it always had): in the old embedded mode, each of a pod's local
    ranks independently called store.setup() with this value, so a 4-local-
    rank pod registered 4 separate segments (see the mooncake-master admin
    metrics comment in the model YAML: "8 ranks x 160GB" is the whole role's
    aggregate, i.e. per-pod ranks x pods). One shared sidecar now covers all
    of that pod's local ranks, so it must contribute local_ranks x that
    per-rank value, not just one rank's worth.
    """
    layout = parallel_layout(role)
    local_ranks = layout.tp_local_size * layout.dp_local_size
    return _parse_size_bytes(spec.mooncake.global_segment_size) * local_ranks


def _mooncake_client_memory_limit(segment_bytes: int) -> str:
    """Container memory limit for a mooncake_client sidecar: enough to hold
    its full --global_segment_size (the memory it contributes to the pool)
    plus a fixed overhead margin, rounded up to whole GiB for the k8s
    resource string.
    """
    segment_gib = -(-segment_bytes // (1024**3))  # ceil
    return f"{segment_gib + _MOONCAKE_CLIENT_MEMORY_OVERHEAD_GIB}Gi"


def _mooncake_client_container(
    spec: DeploymentSpec, instance: Instance, cluster: Cluster, role: RoleSpec, security_context: dict
) -> dict:
    """Sidecar running mooncake_client (Mooncake's Method C 'resource-owning
    real client'). Colocated with the vLLM container so it can be reached over
    127.0.0.1 (MOONCAKE_PREFERRED_SEGMENT) -- one sidecar serves *all* of this
    pod's local ranks (Method C explicitly supports multiple dummy/application
    clients talking to one real client), contributing local_ranks x
    --global_segment_size to the pool and, when enable_offload is set, owning
    this node's local-nvme SSD tier for it.
    """
    master_name = instance.name("mooncake-master")
    segment_bytes = _mooncake_client_global_segment_bytes(spec, role)
    command = [
        "mooncake_client",
        "--port",
        str(spec.mooncake.client_port),
        "--global_segment_size",
        str(segment_bytes),
        "--master_server_address",
        f"{master_name}.{spec.namespace}.svc.cluster.local:50051",
        "--metadata_server",
        "P2PHANDSHAKE",
        "--protocol",
        cluster.mooncake.protocol,
        "--device_names",
        _mooncake_device_name(cluster),
        # Defaults to 1 -- this single sidecar's RPC control plane serves
        # *all* of the pod's local ranks (dummy clients) concurrently, so one
        # thread can serialize their batch_put/batch_get dispatch. Match the
        # CPU request below so control-plane concurrency scales with it.
        "--threads",
        "4",
        "--enable_http_server=true",
        "--http_port",
        "9300",
    ]
    env = [
        # --host defaults to 0.0.0.0, and mooncake_client registers that
        # literal value with the master (MountSegment/P2PHANDSHAKE) as its
        # own reachable address for other clients' transfers. Left unset,
        # every sidecar advertises "0.0.0.0:<port>", which peers can't
        # actually dial -- causing intermittent TRANSFER_FAIL/"Connection
        # refused" on batch_put/get that can cascade into a fatal engine
        # crash. Point it at this pod's real IP instead.
        field_ref_env("MOONCAKE_CLIENT_HOST", "status.podIP"),
        # This cluster grants RDMA NIC access (/dev/infiniband/*) via the
        # NVIDIA Container Runtime's legacy prestart hook, keyed off these
        # env vars -- not a k8s device-plugin extended resource (see
        # cluster.rdma.resource_name below for clusters that use that path
        # instead). NVIDIA_VISIBLE_DEVICES=none opts into the hook without
        # granting actual GPU device access (this sidecar never touches GPU
        # memory), same pattern as the dcgm-exporter sidecar in sidecars.py.
        # NVIDIA_MOFED=enabled makes the hook additionally discover/inject
        # the Mellanox OFED InfiniBand devices mooncake_client needs for
        # --protocol=rdma.
        {"name": "NVIDIA_VISIBLE_DEVICES", "value": "none"},
        {"name": "NVIDIA_MOFED", "value": "enabled"},
    ]
    mkdir_prefix = ""
    if spec.mooncake.enable_offload:
        command.append("--enable_offload=true")
        # MOONCAKE_OFFLOAD_FILE_STORAGE_PATH must already exist and be an
        # absolute, writable, non-symlink directory -- the client does not
        # create it itself.
        env.append({"name": "MOONCAKE_OFFLOAD_FILE_STORAGE_PATH", "value": "/mnt/local/mooncake-offload"})
        mkdir_prefix = "mkdir -p /mnt/local/mooncake-offload && "
        if spec.mooncake.quota_bytes is not None:
            # The master's --quota_bytes (mooncake.py) is only wired to the
            # legacy --root_fs_dir persistence path in client_service.cpp --
            # it's silently dropped for the real --enable_offload/bucket-
            # storage path we use here. This env var is the actual knob
            # BucketBackendConfig::FromEnvironment() reads for max_total_size
            # (eviction policy already defaults to FIFO, not NONE).
            env.append(
                {
                    "name": "MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE",
                    "value": str(spec.mooncake.quota_bytes),
                }
            )

    container: dict[str, Any] = {
        "name": "mooncake-client",
        "image": cluster.mooncake.master_image,
        "command": [
            "/bin/sh",
            "-c",
            # --host is appended outside the shlex.quote()'d args above --
            # it must stay unquoted so the shell expands $MOONCAKE_CLIENT_HOST
            # (the env var isn't known until pod scheduling, via status.podIP).
            mkdir_prefix + " ".join(shlex.quote(arg) for arg in command) + ' --host="$MOONCAKE_CLIENT_HOST"',
        ],
        "ports": [
            {"containerPort": spec.mooncake.client_port, "name": "mc-rpc"},
            {"containerPort": 9300, "name": "metrics"},
        ],
        "env": env,
        "securityContext": security_context,
        "volumeMounts": cluster.volume_mounts(),
        "readinessProbe": {
            "tcpSocket": {"port": spec.mooncake.client_port},
            "initialDelaySeconds": 5,
            "periodSeconds": 5,
            "timeoutSeconds": 3,
            "failureThreshold": 6,
        },
        "resources": {
            "requests": {"cpu": "4", "memory": _mooncake_client_memory_limit(segment_bytes)},
            "limits": {"cpu": "8", "memory": _mooncake_client_memory_limit(segment_bytes)},
        },
    }
    if cluster.rdma.resource_name:
        for key in ("requests", "limits"):
            container["resources"][key][cluster.rdma.resource_name] = cluster.rdma.value
    return container


def _readiness_probe_cmd(role: RoleSpec, readiness_ports: list[int]) -> str:
    """Build the readiness probe shell command.

    For multi-node TP the worker pods run --headless (no HTTP server), so the
    probe checks the vllm process instead.  The leader still gets the normal
    /v1/models HTTP probe.
    """
    layout = parallel_layout(role)
    http_check = " && ".join(
        f"curl -sf http://localhost:{port}/v1/models | grep -q '\"id\"'"
        for port in readiness_ports
    )
    if layout.tp_world_size <= layout.tp_local_size:
        return http_check
    return (
        'if [ "${LWS_WORKER_INDEX:-0}" = "0" ]; then '
        + http_check
        + "; else pgrep -f 'vllm' > /dev/null; fi"
    )


def render_workload(spec: DeploymentSpec, instance: Instance, cluster: Cluster, role: RoleSpec) -> dict:
    resolved = resolve_role(spec, instance, cluster, role)
    external_dp = role.parallelism.dp_enabled and role.dp_load_balancing == DpLoadBalancing.EXTERNAL
    workload_name = instance.user_scoped_name(role.workload_name) if role.workload_name else instance.name(role.name)

    containers, extra_volumes = sidecars(
        spec.runtime.sidecars,
        dcgm_config_name=instance.name("dcgm-metrics"),
    )
    volumes = cluster.base_volumes()
    if role.shm_size:
        volumes[0]["emptyDir"]["sizeLimit"] = role.shm_size
    volumes.extend(extra_volumes)
    if spec.mooncake.enabled:
        volumes.append(_mooncake_volume(instance))

    container_ports = [
        {"containerPort": port, "name": f"vllm-{idx}", "protocol": "TCP"}
        for idx, port in enumerate(resolved.ports.backend)
    ]
    if external_dp:
        container_ports.insert(0, {"containerPort": 8100, "name": "dp-supervisor", "protocol": "TCP"})
    if role.p2p_config:
        # Same fixed local range logic as KV_EVENTS -- see P2P_BASE comment in
        # launch.py.  Every pod in the LWS exposes the same port range because
        # P2P_BASE already compensates for START_RANK.
        p2p_port = role.p2p_config.get("port", 7777)
        local_size = parallel_layout(role).dp_local_size if role.parallelism.dp_enabled else 1
        container_ports.extend(
            {"containerPort": p2p_port + idx, "name": f"p2p-{idx}", "protocol": "TCP"}
            for idx in range(local_size)
        )
    readiness_ports = resolved.ports.public if role.routing_proxy else resolved.ports.backend

    init_containers = []
    if spec.mooncake.enabled:
        init_containers.append(_mooncake_init_container(instance))
    if role.routing_proxy:
        init_containers.append(
            {
                "name": "routing-proxy",
                "image": cluster.llm_d.routing_sidecar,
                "imagePullPolicy": "Always",
                "args": [
                    f"--port={resolved.ports.public[0]}",
                    f"--vllm-port={resolved.ports.backend[0]}",
                    f"--data-parallel-size={resolved.ports.rank_count}",
                    "--secure-proxy=false",
                    # v0.10.0 removed the deprecated `--connector` alias
                    # (llm-d-router options.go); use --kv-connector, its
                    # replacement, or pd-sidecar exits immediately on an
                    # unrecognized flag (pflag defaults to ExitOnError),
                    # crash-looping the decode pod's routing-proxy sidecar.
                    "--kv-connector=nixlv2",
                ],
                "ports": [
                    {"containerPort": port, "name": f"rank{idx}", "protocol": "TCP"}
                    for idx, port in enumerate(resolved.ports.public)
                ],
                "restartPolicy": "Always",
                "resources": {
                    "requests": {"cpu": 8, "memory": "16Gi"},
                    "limits": {"cpu": 8, "memory": "16Gi"},
                },
                "securityContext": {"allowPrivilegeEscalation": False},
            }
        )

    security_context = cluster.pod_defaults.container_security_context
    if security_context is None:
        security_context = {
            "capabilities": {"add": ["IPC_LOCK", "SYS_RAWIO"]},
            "runAsGroup": 0,
            "runAsUser": 0,
        }
    container_env = [
        secret_env("HF_TOKEN", "hf-secret", "HF_TOKEN"),
        *env_list(resolved.env),
    ]
    if (
        _uses_nixl_connector(role.kv_transfer_config)
        and "VLLM_NIXL_SIDE_CHANNEL_HOST" not in resolved.env
    ):
        container_env.append(
            field_ref_env("VLLM_NIXL_SIDE_CHANNEL_HOST", "status.podIP")
        )
    if role.p2p_config:
        if "POD_IP" not in resolved.env:
            container_env.append(field_ref_env("POD_IP", "status.podIP"))
        if "VLLM_P2P_SIDE_CHANNEL_HOST" not in resolved.env:
            container_env.append(field_ref_env("VLLM_P2P_SIDE_CHANNEL_HOST", "status.podIP"))
    if spec.mooncake.enabled:
        container_env.append({"name": "MOONCAKE_CONFIG_PATH", "value": "/etc/mooncake/mooncake_config.json"})
    uses_mooncake_client_sidecar = (
        spec.mooncake.enabled
        and spec.mooncake.mode == "standalone-store"
        and _uses_mooncake_store_connector(role.kv_transfer_config)
    )
    if uses_mooncake_client_sidecar:
        # Reach this pod's own mooncake_client sidecar (below) instead of
        # contributing memory in-process -- see mooncake.py for why
        # global_segment_size is forced to 0 in this mode.
        container_env.append(
            {"name": "MOONCAKE_PREFERRED_SEGMENT", "value": f"127.0.0.1:{spec.mooncake.client_port}"}
        )
        containers.append(_mooncake_client_container(spec, instance, cluster, role, security_context))

    launch_script = build_launch_script(
        spec,
        role,
        resolved.ports,
        log_dir=resolved.log_dir,
        dev_source=resolved.dev_source,
        vllm_args=resolved.vllm_args,
    )
    # Prepend mooncake client upgrade when enabled to ensure version matches master
    if spec.mooncake.enabled:
        launch_script = _mooncake_upgrade_prefix() + launch_script

    vllm_container = {
        "name": "vllm",
        "image": spec.model.image,
        "imagePullPolicy": "Always",
        "securityContext": security_context,
        "command": ["/bin/bash", "-c"],
        "args": [launch_script],
        "env": container_env,
        "ports": container_ports,
        "readinessProbe": {
            "exec": {
                "command": [
                    "/bin/bash",
                    "-c",
                    _readiness_probe_cmd(role, readiness_ports),
                ]
            },
            "periodSeconds": 5,
            "failureThreshold": 120,
        },
        "resources": {
            "requests": {
                "cpu": role.resources.cpu,
                "memory": role.resources.memory,
                "ephemeral-storage": role.resources.ephemeral_storage,
                "nvidia.com/gpu": str(role.resources.gpus),
            },
            "limits": {
                "memory": role.resources.memory,
                "ephemeral-storage": role.resources.ephemeral_storage,
                "nvidia.com/gpu": str(role.resources.gpus),
            },
        },
        "volumeMounts": cluster.volume_mounts() + ([_mooncake_volume_mount()] if spec.mooncake.enabled else []),
        "workingDir": "/code",
    }
    if external_dp:
        vllm_container["startupProbe"] = {
            "httpGet": {"path": "/health", "port": "dp-supervisor"},
            "periodSeconds": 1,
            "timeoutSeconds": 5,
            "failureThreshold": 1800,
        }
    if cluster.rdma.resource_name:
        for resources in ("requests", "limits"):
            vllm_container["resources"][resources][cluster.rdma.resource_name] = cluster.rdma.value
    if resolved.resource_claims:
        vllm_container["resources"]["claims"] = [{"name": claim["name"]} for claim in resolved.resource_claims]

    pod_labels = instance.labels("model-server", role.name) | {
        "llm-d.ai/inferenceServing": "true",
        "llm-d.ai/model": spec.model.label_value,
        "llm-d.ai/deployment": spec.topology.value,
    }
    pod_metadata = {"labels": pod_labels}
    if cluster.pod_defaults.annotations:
        pod_metadata["annotations"] = cluster.pod_defaults.annotations

    pod_spec = {
        "serviceAccountName": instance.name("model-server"),
        "terminationGracePeriodSeconds": 0,
        "volumes": volumes,
        "containers": [vllm_container, *containers],
    }
    if cluster.pod_defaults.affinity:
        pod_spec["affinity"] = cluster.pod_defaults.affinity
    if cluster.pod_defaults.tolerations:
        pod_spec["tolerations"] = cluster.pod_defaults.tolerations
    if init_containers:
        pod_spec["initContainers"] = init_containers
    if resolved.resource_claims:
        pod_spec["resourceClaims"] = resolved.resource_claims

    if role.lws.size == 1:
        selector = instance.pod_selector(role.name)
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": workload_name,
                "labels": instance.labels("model-server", role.name),
            },
            "spec": {
                "replicas": role.lws.replicas,
                "selector": {"matchLabels": selector},
                "strategy": {
                    "type": "RollingUpdate",
                    "rollingUpdate": {"maxSurge": 0, "maxUnavailable": "100%"},
                },
                "template": {
                    "metadata": pod_metadata,
                    "spec": pod_spec,
                },
            },
        }

    return {
        "apiVersion": "leaderworkerset.x-k8s.io/v1",
        "kind": "LeaderWorkerSet",
        "metadata": {
            "name": workload_name,
            "labels": instance.labels("lws", role.name)
            | {
                "llm-d.ai/inferenceServing": "true",
                "llm-d.ai/model": spec.model.label_value,
                "llm-d.ai/deployment": spec.topology.value,
            },
        },
        "spec": {
            "replicas": role.lws.replicas,
            "rolloutStrategy": {
                "type": "RollingUpdate",
                "rollingUpdateConfiguration": {"maxUnavailable": "100%"},
            },
            "leaderWorkerTemplate": {
                "size": role.lws.size,
                "workerTemplate": {
                    "metadata": pod_metadata,
                    "spec": pod_spec,
                },
            },
        },
    }


def _uses_connector(value: Any, name_substr: str) -> bool:
    """Recursively search a kv_transfer_config tree (including nested
    MultiConnector `connectors` lists) for a kv_connector name containing
    name_substr, case-insensitively.
    """
    if isinstance(value, dict):
        connector = value.get("kv_connector")
        if isinstance(connector, str) and name_substr in connector.casefold():
            return True
        return any(_uses_connector(item, name_substr) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_uses_connector(item, name_substr) for item in value)
    return False


def _uses_nixl_connector(value: Any) -> bool:
    return _uses_connector(value, "nixl")


def _uses_mooncake_store_connector(value: Any) -> bool:
    return _uses_connector(value, "mooncakestore")
