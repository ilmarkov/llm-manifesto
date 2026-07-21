"""Render the persistent per-user dev pod for building vLLM from source."""

from __future__ import annotations

from .common import env_list, secret_env
from ..cluster import Cluster
from ..images import DEFAULT_IMAGES
from ..instance import Instance


def render_dev_pod(cluster: Cluster, user: str) -> dict:
    instance = Instance(user=user, release="dev")
    name = instance.user_scoped_name("vllm-dev")
    user_root = cluster.user_root(user=instance.user_slug, release="")
    dev_source = cluster.dev_source(user=instance.user_slug, release="")

    volumes = [
        {
            "name": "dshm",
            "emptyDir": {"medium": "Memory", "sizeLimit": cluster.pod_defaults.shm_size},
        }
    ]
    mounts = [{"name": "dshm", "mountPath": "/dev/shm"}]
    if cluster.storage.shared_volume:
        volumes.append({"name": "shared-storage", **cluster.storage.shared_volume})
        mounts.append({"name": "shared-storage", "mountPath": cluster.storage.shared_mount_path})
        hf_home = f"{cluster.storage.shared_mount_path}/hf_cache"
    elif cluster.cache.hf_host_path and cluster.cache.jit_host_path:
        volumes.extend(
            [
                {"name": "hf-cache", "hostPath": {"path": cluster.cache.hf_host_path, "type": "DirectoryOrCreate"}},
                {"name": "jit-cache", "hostPath": {"path": cluster.cache.jit_host_path, "type": "DirectoryOrCreate"}},
            ]
        )
        mounts.extend(
            [
                {"name": "hf-cache", "mountPath": "/var/cache/huggingface"},
                {"name": "jit-cache", "mountPath": "/var/cache/vllm"},
            ]
        )
        hf_home = cluster.cache.hf_home
    else:
        raise ValueError("cluster profile has neither shared storage nor host caches for the dev pod")

    env = {
        "HF_HOME": hf_home,
        "VLLM_CACHE_ROOT": f"{user_root}/dev-caches/vllm",
        "FLASHINFER_CACHE_DIR": f"{user_root}/dev-caches/flashinfer",
        "TORCH_CUDA_ARCH_LIST": "9.0a;10.0+PTX",
        "CCACHE_DIR": f"{user_root}/ccache",
        "CMAKE_CXX_COMPILER_LAUNCHER": "ccache",
        "CMAKE_C_COMPILER_LAUNCHER": "ccache",
    }

    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "labels": {"app": name, **instance.labels("dev")},
        },
        "spec": {
            "restartPolicy": "Always",
            "affinity": {
                "nodeAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": {
                        "nodeSelectorTerms": [
                            {
                                "matchExpressions": [
                                    {"key": "nvidia.com/gpu.present", "operator": "Exists"}
                                ]
                            }
                        ]
                    }
                }
            },
            "securityContext": {"runAsUser": 0, "runAsGroup": 0},
            "containers": [
                {
                    "name": "dev",
                    "image": DEFAULT_IMAGES.get("dev.image"),
                    "imagePullPolicy": "Always",
                    "command": ["sleep", "infinity"],
                    "resources": {
                        "requests": {"cpu": "32", "memory": "512Gi", "nvidia.com/gpu": "1"},
                        "limits": {"cpu": "32", "memory": "512Gi", "nvidia.com/gpu": "1"},
                    },
                    "env": [secret_env("HF_TOKEN", "hf-secret", "HF_TOKEN"), *env_list(env)],
                    "volumeMounts": mounts,
                    "workingDir": dev_source,
                }
            ],
            "volumes": volumes,
        },
    }
