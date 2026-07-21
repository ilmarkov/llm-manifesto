"""Cluster profile schema and concrete cluster facts used while rendering pods.

The pydantic models mirror the sections of a cluster YAML profile. Unknown
keys are rejected so a typo'd profile fails at load instead of silently
falling back to defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .equations import render_mapping
from .images import DEFAULT_IMAGES


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shared_volume: dict[str, Any] | None = None
    shared_mount_path: str = "/mnt/shared"
    local_nvme_path: str | None = "/mnt/numa0"


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_root: str | None = None
    log_root: str | None = None
    cache_root: str | None = None


class DevConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    venv: str | None = None
    source: str | None = None


class CacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hf_host_path: str | None = None
    jit_host_path: str | None = None
    hf_home: str | None = None


class RdmaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resource_name: str | None = None
    value: str = "1"

    @field_validator("value", mode="before")
    @classmethod
    def coerce_value(cls, value: Any) -> str:
        return str(value)


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pvc: str | None = None
    mount_path: str = "/mnt/logs"
    root: str | None = None


class PodDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shm_size: str = "2Gi"
    annotations: dict[str, str] = Field(default_factory=dict)
    affinity: dict[str, Any] = Field(default_factory=dict)
    tolerations: list[dict[str, Any]] = Field(default_factory=list)
    extra_volumes: list[dict[str, Any]] = Field(default_factory=list)
    extra_volume_mounts: list[dict[str, Any]] = Field(default_factory=list)
    container_security_context: dict[str, Any] | None = None


class ModelServerResourcesConfig(BaseModel):
    """Per-GPU requests and optional node capacities used for pod packing."""

    model_config = ConfigDict(extra="forbid")

    cpu_per_gpu: str
    memory_per_gpu: str
    node_allocatable_cpu: str | None = None
    node_allocatable_memory: str | None = None

    @field_validator(
        "cpu_per_gpu",
        "memory_per_gpu",
        "node_allocatable_cpu",
        "node_allocatable_memory",
        mode="before",
    )
    @classmethod
    def coerce_quantities(cls, value: Any) -> str | None:
        return None if value is None else str(value)


class GatewayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_name: str = "istio"
    service_type: str = "ClusterIP"


class FabricProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env: dict[str, Any] = Field(default_factory=dict)
    computed_env: dict[str, Any] = Field(default_factory=dict)


class FabricConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ucx_net_devices: str
    default_profile: str = "standard"
    expert_parallel_profiles: dict[str, str] = Field(default_factory=dict)
    default_env: dict[str, Any] = Field(default_factory=dict)
    profiles: dict[str, FabricProfileConfig] = Field(default_factory=dict)
    imex_resource_claim_template: str | None = None


class LlmdConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    release: str | None = None
    images: dict[str, str] = Field(default_factory=dict)

    @property
    def resolved_release(self) -> str:
        return self.release or DEFAULT_IMAGES.get("llm_d.release")

    @property
    def epp(self) -> str:
        return self._image("epp")

    @property
    def routing_sidecar(self) -> str:
        return self._image("routing_sidecar")

    def _image(self, name: str) -> str:
        release = self.resolved_release
        template = self.images.get(name, DEFAULT_IMAGES.get(f"llm_d.{name}", release=release))
        return template.format(release=release)


class OpenShiftConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scc: str | None = None


class Cluster(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    gpus_per_node: int = Field(ge=1)
    platform: Literal["kubernetes", "openshift"] = "kubernetes"
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    dev: DevConfig = Field(default_factory=DevConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    rdma: RdmaConfig = Field(default_factory=RdmaConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    pod_defaults: PodDefaults = Field(default_factory=PodDefaults)
    model_server_resources: ModelServerResourcesConfig
    fabric: FabricConfig
    llm_d: LlmdConfig = Field(default_factory=LlmdConfig)
    openshift: OpenShiftConfig = Field(default_factory=OpenShiftConfig)

    @model_validator(mode="after")
    def default_hf_home(self) -> "Cluster":
        if self.cache.hf_home is not None:
            return self
        if self.cache.hf_host_path and self.cache.jit_host_path:
            self.cache.hf_home = "/var/cache/huggingface"
        elif self.storage.local_nvme_path:
            self.cache.hf_home = "/mnt/local/hf_cache"
        else:
            self.cache.hf_home = f"{self.storage.shared_mount_path}/hf_cache"
        return self

    # Path templates. Explicit profile values win; defaults derive from the
    # shared mount so they are declared exactly once.

    @property
    def user_root_template(self) -> str:
        return self.paths.user_root or f"{self.storage.shared_mount_path}/{{user}}"

    @property
    def log_root_template(self) -> str:
        return self.logging.root or self.paths.log_root or f"{self.user_root_template}/logs"

    @property
    def cache_root_template(self) -> str:
        return (
            self.paths.cache_root
            or f"{self.storage.shared_mount_path}/{{user}}/jit-cache/{{gpu_arch}}/{{cuda}}/{{cache_key}}/{{release}}"
        )

    @property
    def dev_venv_template(self) -> str:
        return self.dev.venv or f"{self.storage.shared_mount_path}/{{user}}/vllm-venv"

    @property
    def dev_source_template(self) -> str:
        return self.dev.source or f"{self.storage.shared_mount_path}/{{user}}/vllm-dev"

    def user_root(self, *, user: str, release: str) -> str:
        return self.user_root_template.format(user=user, release=release)

    def log_root(self, *, user: str, release: str) -> str:
        return self.log_root_template.format(user=user, release=release)

    def cache_root(self, *, user: str, release: str, gpu_arch: str, cuda: str, cache_key: str) -> str:
        return self.cache_root_template.format(
            user=user, release=release, gpu_arch=gpu_arch, cuda=cuda, cache_key=cache_key
        )

    def dev_venv(self, *, user: str, release: str) -> str:
        return self.dev_venv_template.format(user=user, release=release)

    def dev_source(self, *, user: str, release: str) -> str:
        return self.dev_source_template.format(user=user, release=release)

    def with_path_overrides(
        self,
        *,
        user_root: str | None = None,
        log_root: str | None = None,
        cache_root: str | None = None,
        dev_venv: str | None = None,
        dev_source: str | None = None,
    ) -> "Cluster":
        cluster = self.model_copy(deep=True)
        if user_root:
            cluster.paths.user_root = user_root
        if log_root:
            cluster.logging.root = log_root
        if cache_root:
            cluster.paths.cache_root = cache_root
        if dev_venv:
            cluster.dev.venv = dev_venv
        if dev_source:
            cluster.dev.source = dev_source
        return cluster

    def base_volumes(self) -> list[dict]:
        volumes = [
            {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": self.pod_defaults.shm_size}},
        ]
        if self.cache.hf_host_path and self.cache.jit_host_path:
            volumes.extend(
                [
                    {
                        "name": "hf-cache",
                        "hostPath": {"path": self.cache.hf_host_path, "type": "DirectoryOrCreate"},
                    },
                    {
                        "name": "jit-cache",
                        "hostPath": {"path": self.cache.jit_host_path, "type": "DirectoryOrCreate"},
                    },
                ]
            )
        else:
            if not self.storage.shared_volume:
                raise ValueError("storage.shared_volume is required when host caches are not configured")
            volumes.append({"name": "shared-storage", **self.storage.shared_volume})
            if self.storage.local_nvme_path:
                volumes.append(
                    {
                        "name": "local-nvme",
                        "hostPath": {"path": self.storage.local_nvme_path, "type": "Directory"},
                    }
                )
        if self.logging.pvc and self.logging.mount_path not in self._base_mount_paths():
            volumes.append({"name": "logs", "persistentVolumeClaim": {"claimName": self.logging.pvc}})
        volumes.extend(self.pod_defaults.extra_volumes)
        return volumes

    def volume_mounts(self) -> list[dict]:
        mounts = self._base_volume_mounts()
        if self.logging.pvc and self.logging.mount_path not in self._base_mount_paths():
            mounts.append({"name": "logs", "mountPath": self.logging.mount_path})
        mounts.extend(self.pod_defaults.extra_volume_mounts)
        return mounts

    def _base_mount_paths(self) -> set[str]:
        return {mount["mountPath"] for mount in self._base_volume_mounts()}

    def _base_volume_mounts(self) -> list[dict]:
        if self.cache.hf_host_path and self.cache.jit_host_path:
            return [
                {"name": "dshm", "mountPath": "/dev/shm"},
                {"name": "hf-cache", "mountPath": "/var/cache/huggingface"},
                {"name": "jit-cache", "mountPath": "/var/cache/vllm"},
            ]
        mounts = [
            {"name": "dshm", "mountPath": "/dev/shm"},
            {"name": "shared-storage", "mountPath": self.storage.shared_mount_path},
        ]
        if self.storage.local_nvme_path:
            mounts.append({"name": "local-nvme", "mountPath": "/mnt/local"})
        return mounts

    def fabric_profile_for(self, *, topology: str, role_name: str, expert_parallel: bool) -> str:
        if not expert_parallel:
            return self.fabric.default_profile
        return self.fabric.expert_parallel_profiles.get(role_name, self.fabric.default_profile)

    def fabric_env(self, profile: str, context: dict | None = None) -> dict[str, str]:
        format_context = {"ucx_net_devices": self.fabric.ucx_net_devices}
        env = {key: str(value).format(**format_context) for key, value in self.fabric.default_env.items()}
        profile_config = self.fabric.profiles.get(profile, FabricProfileConfig())
        env |= {key: str(value) for key, value in profile_config.env.items()}
        if profile_config.computed_env:
            env |= {
                key: str(value)
                for key, value in render_mapping(profile_config.computed_env, context or {}).items()
            }
        return env


def load_cluster(path: str | Path) -> Cluster:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Cluster.model_validate(data)
