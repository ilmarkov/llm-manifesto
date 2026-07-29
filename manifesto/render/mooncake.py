"""Mooncake master service deployment and configuration for distributed KV cache."""

from __future__ import annotations

import json

from ..cluster import Cluster
from ..instance import Instance
from ..spec import DeploymentSpec


def _mooncake_device_name(cluster: Cluster) -> str:
    """Derive Mooncake device_name from cluster's UCX NIC list.

    fabric.ucx_net_devices is formatted like "mlx5_0:1,mlx5_1:1" (UCX wants
    the port suffix); Mooncake's device_name wants bare device names.
    """
    return ",".join(
        device.split(":")[0] for device in cluster.fabric.ucx_net_devices.split(",")
    )


def render_mooncake(
    spec: DeploymentSpec, instance: Instance, cluster: Cluster
) -> list[dict]:
    """Render Mooncake master Deployment, Service, and ConfigMap.

    Returns an empty list when mooncake is disabled. Raises ValueError if
    mooncake is enabled but the cluster lacks configuration or RDMA devices.
    """
    if not spec.mooncake.enabled:
        return []

    if cluster.mooncake is None:
        raise ValueError(
            f"spec.mooncake.enabled is true but cluster {cluster.name!r} has no `mooncake:` section"
        )
    if not cluster.fabric.ucx_net_devices:
        raise ValueError(
            f"Mooncake requires RDMA devices but cluster {cluster.name!r} has empty ucx_net_devices"
        )

    master_name = instance.name("mooncake-master")
    config_name = instance.name("mooncake-config")

    store_config = {
        "mode": spec.mooncake.mode,
        "metadata_server": "P2PHANDSHAKE",
        "master_server_address": f"{master_name}.{spec.namespace}.svc.cluster.local:50051",
        "global_segment_size": spec.mooncake.global_segment_size,
        "local_buffer_size": spec.mooncake.local_buffer_size,
        "protocol": cluster.mooncake.protocol,
        "device_name": _mooncake_device_name(cluster),
        "enable_offload": spec.mooncake.enable_offload,
    }

    return [
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": config_name,
                "labels": instance.labels("mooncake"),
            },
            "data": {"mooncake_config.json": json.dumps(store_config)},
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": master_name,
                "labels": instance.labels("mooncake"),
            },
            "spec": {
                "selector": instance.labels("mooncake"),
                "ports": [{"name": "grpc", "port": 50051, "protocol": "TCP"}],
            },
        },
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": master_name,
                "labels": instance.labels("mooncake"),
            },
            "spec": {
                "replicas": spec.mooncake.replicas,
                "selector": {"matchLabels": instance.labels("mooncake")},
                "template": {
                    "metadata": {"labels": instance.labels("mooncake")},
                    "spec": {
                        "containers": [
                            {
                                "name": "mooncake-master",
                                "image": cluster.mooncake.master_image,
                                "command": ["mooncake_master", "--port", "50051"],
                                "ports": [{"containerPort": 50051, "name": "grpc"}],
                                "resources": {
                                    "requests": {"cpu": "2", "memory": "4Gi"},
                                    "limits": {"cpu": "4", "memory": "8Gi"},
                                },
                                "readinessProbe": {
                                    "tcpSocket": {"port": 50051},
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 5,
                                    "timeoutSeconds": 3,
                                    "failureThreshold": 3,
                                },
                                "livenessProbe": {
                                    "tcpSocket": {"port": 50051},
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10,
                                    "timeoutSeconds": 5,
                                    "failureThreshold": 6,
                                },
                            }
                        ],
                        "affinity": {
                            "nodeAffinity": {
                                "requiredDuringSchedulingIgnoredDuringExecution": {
                                    "nodeSelectorTerms": [
                                        {
                                            "matchExpressions": [
                                                {
                                                    "key": "kubernetes.io/arch",
                                                    "operator": "In",
                                                    "values": ["amd64"],
                                                }
                                            ]
                                        }
                                    ]
                                }
                            }
                        },
                    },
                },
            },
        },
    ]
