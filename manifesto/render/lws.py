"""Workload renderer for vLLM roles, including sidecars and fabric mounts."""

from __future__ import annotations

from typing import Any

from .common import env_list, field_ref_env, secret_env
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
                    "--connector=nixlv2",
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
    if spec.mooncake.enabled:
        container_env.append({"name": "MOONCAKE_CONFIG_PATH", "value": "/etc/mooncake/mooncake_config.json"})

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


def _uses_nixl_connector(value: Any) -> bool:
    if isinstance(value, dict):
        connector = value.get("kv_connector")
        if isinstance(connector, str) and "nixl" in connector.casefold():
            return True
        return any(_uses_nixl_connector(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_uses_nixl_connector(item) for item in value)
    return False
