"""Workload renderer for vLLM roles, including sidecars and fabric mounts."""

from __future__ import annotations

from typing import Any

from .common import env_list, field_ref_env, secret_env
from .sidecars import sidecars
from ..cluster import Cluster
from ..instance import Instance
from ..launch import build_launch_script
from ..resolve import resolve_role
from ..spec import DeploymentSpec, DpLoadBalancing, RoleSpec


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

    container_ports = [
        {"containerPort": port, "name": f"vllm-{idx}", "protocol": "TCP"}
        for idx, port in enumerate(resolved.ports.backend)
    ]
    if external_dp:
        container_ports.insert(0, {"containerPort": 8100, "name": "dp-supervisor", "protocol": "TCP"})
    readiness_ports = resolved.ports.public if role.routing_proxy else resolved.ports.backend

    init_containers = []
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

    vllm_container = {
        "name": "vllm",
        "image": spec.model.image,
        "imagePullPolicy": "Always",
        "securityContext": security_context,
        "command": ["/bin/bash", "-c"],
        "args": [
            build_launch_script(
                spec,
                role,
                resolved.ports,
                log_dir=resolved.log_dir,
                dev_source=resolved.dev_source,
                vllm_args=resolved.vllm_args,
            )
        ],
        "env": container_env,
        "ports": container_ports,
        "readinessProbe": {
            "exec": {
                "command": [
                    "/bin/bash",
                    "-c",
                    # TODO(readiness): compare with upstream llm-d probes as these templates mature.
                    " && ".join(
                        f"curl -sf http://localhost:{port}/v1/models | grep -q '\"id\"'"
                        for port in readiness_ports
                    ),
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
        "volumeMounts": cluster.volume_mounts(),
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
