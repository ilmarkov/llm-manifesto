"""Resolve a spec role into concrete ports, paths, env vars, and vLLM arguments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cluster import Cluster
from .equations import render_mapping
from .instance import Instance
from .dp_ports import RolePorts, derive_ports
from .parallelism import ParallelLayout, parallel_layout
from .spec import DeploymentSpec, RoleSpec


@dataclass(frozen=True)
class ResolvedRole:
    ports: RolePorts
    log_dir: str
    dev_source: str
    fabric_profile: str
    env: dict[str, str]
    vllm_args: dict[str, Any]
    resource_claims: list[dict[str, str]]


def resolve_role(spec: DeploymentSpec, instance: Instance, cluster: Cluster, role: RoleSpec) -> ResolvedRole:
    layout = parallel_layout(role)
    ports = derive_ports(
        rank_count=layout.dp_local_size,
        public_base=role.serving_port_base,
        backend_base=role.backend_port_base,
    )
    context = _variable_context(spec, role, layout)
    computed_env = render_mapping(role.computed.get("env", {}), context)
    context |= computed_env
    computed_vllm_args = render_mapping(role.computed.get("vllm", {}), context)

    fabric_profile = role.fabric_profile or cluster.fabric_profile_for(
        topology=spec.topology.value,
        role_name=role.name,
        expert_parallel=role.parallelism.ep,
    )

    cache_prefix = cluster.cache_root(
        user=instance.user_slug,
        release=instance.release_slug,
        gpu_arch=spec.cache.gpu_arch,
        cuda=spec.cache.cuda,
        cache_key=spec.cache_key,
    )
    dev_venv = spec.runtime.dev_venv or (
        cluster.dev_venv(user=instance.user_slug, release=instance.release_slug) if spec.runtime.dev else ""
    )
    env = _base_env(spec, cache_prefix, dev_venv=dev_venv, platform=cluster.platform)
    env |= cluster.fabric_env(fabric_profile, context)
    env |= spec.runtime.env
    env |= role.env
    env |= {key: str(value) for key, value in computed_env.items()}

    return ResolvedRole(
        ports=ports,
        log_dir=f"{cluster.log_root(user=instance.user_slug, release=instance.release_slug)}/{role.name}",
        dev_source=cluster.dev_source(user=instance.user_slug, release=instance.release_slug),
        fabric_profile=fabric_profile,
        env=env,
        vllm_args=role.vllm_args | computed_vllm_args,
        resource_claims=_resource_claims(cluster, fabric_profile),
    )


def _variable_context(spec: DeploymentSpec, role: RoleSpec, layout: ParallelLayout) -> dict[str, Any]:
    return {
        **spec.vars,
        **role.vars,
        "gpus_per_pod": role.gpus_per_pod,
        "tp": layout.tp_world_size,
        "tp_world_size": layout.tp_world_size,
        "tp_local_size": layout.tp_local_size,
        "dp_enabled": role.parallelism.dp_enabled,
        "dp_local_size": layout.dp_local_size,
        "dp_world_size": layout.dp_world_size,
        "lws_size": role.lws.size,
        "lws_replicas": role.lws.replicas,
    }


def _base_env(spec: DeploymentSpec, cache_prefix: str, *, dev_venv: str, platform: str) -> dict[str, str]:
    env = {
        "HF_HOME": spec.model.hf_home,
        "HOME": f"{cache_prefix}/home",
        "XDG_CACHE_HOME": f"{cache_prefix}/xdg",
        "MANIFESTO_VLLM_DEV_VENV": dev_venv,
        "VLLM_NO_USAGE_STATS": "1",
        "TQDM_DISABLE": "1",
        "VLLM_LOGGING_LEVEL": "INFO",
        "VLLM_CACHE_ROOT": f"{cache_prefix}/vllm",
        "FLASHINFER_CACHE_DIR": f"{cache_prefix}/flashinfer",
        "FLASHINFER_WORKSPACE_BASE": f"{cache_prefix}/flashinfer-workspace",
        "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
        "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": f"{cache_prefix}/fa-cute-dsl",
        "TRITON_CACHE_DIR": f"{cache_prefix}/triton",
        "TORCHINDUCTOR_CACHE_DIR": f"{cache_prefix}/torchinductor",
        "TILELANG_CACHE_DIR": f"{cache_prefix}/tilelang",
    }
    if platform == "openshift":
        # OpenShift commonly assigns an arbitrary UID absent from /etc/passwd.
        # Python getpass (used by torch during import) honors USER first.
        env["USER"] = "vllm"
    return env


def _resource_claims(cluster: Cluster, fabric_profile: str) -> list[dict[str, str]]:
    if cluster.fabric.imex_resource_claim_template and fabric_profile.startswith("deepep"):
        return [
            {
                "name": "compute-domain-channel",
                "resourceClaimTemplateName": cluster.fabric.imex_resource_claim_template,
            }
        ]
    return []
