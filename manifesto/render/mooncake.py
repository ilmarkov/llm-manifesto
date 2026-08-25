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
        # standalone-store requires 0 here (vLLM's MooncakeStoreConfig rejects
        # a nonzero value): the pool's memory is contributed by each pod's
        # mooncake_client sidecar instead (see lws.py), not by the vLLM rank
        # itself. spec.mooncake.global_segment_size still governs that
        # sidecar's --global_segment_size in that mode.
        "global_segment_size": (
            spec.mooncake.global_segment_size if spec.mooncake.mode == "embedded" else 0
        ),
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
                "ports": [
                    {"name": "grpc", "port": 50051, "protocol": "TCP"},
                    # Master's native Prometheus admin endpoint (/metrics,
                    # /metrics/summary, /health) -- exposes real eviction
                    # counters (Eviction: Success/Attempts, AllocFail, keys,
                    # size) that are otherwise only visible in the periodic
                    # stdout log. Already scraped cluster-wide by the shared
                    # Prometheus's `mooncake-master` job (hardcoded to
                    # <pod_ip>:9003), which discovers by container name, not
                    # by this Service -- this port is declared here mainly
                    # for documentation and manual `port-forward` access.
                    {"name": "metrics", "port": 9003, "protocol": "TCP"},
                ],
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
                                "command": [
                                    "mooncake_master",
                                    "--port",
                                    "50051",
                                    # Explicit even though 9003 is the gflags
                                    # default -- makes the Prometheus admin
                                    # endpoint's port a documented contract
                                    # rather than an implicit default that
                                    # could silently move on a version bump.
                                    "--metrics_port",
                                    "9003",
                                    *(
                                        # Master-side orchestration flag: queues
                                        # completed memory writes for async SSD
                                        # persistence. The master never holds
                                        # cache bytes itself -- actual disk I/O
                                        # happens on whichever mooncake_client
                                        # (sidecar, see lws.py) owns the segment
                                        # being evicted, via that client's own
                                        # MOONCAKE_OFFLOAD_FILE_STORAGE_PATH.
                                        ["--enable_offload=true"]
                                        if spec.mooncake.enable_offload
                                        else []
                                    ),
                                    *(
                                        # Lazy SSD writes on eviction instead
                                        # of eager writes after every Put --
                                        # reduces SSD write amplification.
                                        ["--offload_on_evict=true"]
                                        if spec.mooncake.offload_on_evict
                                        else []
                                    ),
                                    *(
                                        # Allow hot SSD-only objects to be
                                        # promoted back to DRAM on repeated
                                        # reads (uses the master's default
                                        # --promotion_admission_threshold=2).
                                        ["--promotion_on_hit=true"]
                                        if spec.mooncake.promotion_on_hit
                                        else []
                                    ),
                                    # Deliberately NOT passing --quota_bytes here:
                                    # client_service.cpp only forwards the master's
                                    # quota_bytes (from GetStorageConfig()) into
                                    # PrepareStorageBackend() on the legacy
                                    # --root_fs_dir/fsdir path, which we never use
                                    # (see comment on global_segment_size above --
                                    # standalone-store + enable_offload never sets
                                    # root_fs_dir). It's silently dropped on our
                                    # real --enable_offload/bucket-storage path, so
                                    # setting it here would be a no-op that implies
                                    # false enforcement. The actual per-client disk
                                    # cap is spec.mooncake.quota_bytes, applied via
                                    # MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE on the
                                    # mooncake_client sidecar itself (see lws.py).
                                ],
                                "ports": [
                                    {"containerPort": 50051, "name": "grpc"},
                                    {"containerPort": 9003, "name": "metrics"},
                                ],
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
