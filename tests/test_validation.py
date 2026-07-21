"""Tests for hard validation of impossible or contradictory role configurations."""

import warnings
from pathlib import Path

import pytest
from pydantic import ValidationError

from manifesto.cluster import load_cluster
from manifesto.parallelism import parallel_layout
from manifesto.spec import DeploymentSpec

ROOT = Path(__file__).resolve().parents[1]


def _spec_with_role(role: dict) -> dict:
    return {
        "release": "bad",
        "topology": "aggregated",
        "model": {"id": "model", "image": "image"},
        "routing": {"kind": "disabled"},
        "roles": [role],
    }


def _role(role: dict) -> DeploymentSpec:
    return DeploymentSpec.model_validate(_spec_with_role(role)).role(role["name"])


def test_uneven_global_dp_split_is_an_error():
    role = _role(
        {
            "name": "decode",
            "lws": {"size": 4},
            "parallelism": {"gpus_per_node": 4, "tp": 1, "dp": 10, "ep": True},
        }
    )

    with pytest.raises(ValueError, match="dp=10 does not divide evenly across 4 LWS nodes"):
        parallel_layout(role)


def test_uneven_global_tp_split_is_an_error():
    role = _role(
        {
            "name": "prefill",
            "lws": {"size": 3},
            "parallelism": {"gpus_per_node": 4, "tp": 10, "dp": False, "ep": True},
        }
    )

    with pytest.raises(ValueError, match="tp=10 does not divide evenly across 3 LWS nodes"):
        parallel_layout(role)


def test_idle_gpus_without_dp_is_an_error():
    role = _role(
        {
            "name": "prefill",
            "lws": {"size": 1},
            "parallelism": {"gpus_per_node": 4, "tp": 2, "dp": False, "ep": True},
        }
    )

    with pytest.raises(ValueError, match="DP is disabled but local TP 2 leaves 2 of 4 GPUs idle"):
        parallel_layout(role)


def test_dp_tp_gpu_partition_mismatch_is_an_error():
    role = _role(
        {
            "name": "decode",
            "lws": {"size": 4},
            "parallelism": {"gpus_per_node": 4, "tp": 2, "dp": 4, "ep": True},
        }
    )

    with pytest.raises(ValueError, match="needs 2 GPUs per pod, got 4"):
        parallel_layout(role)


def test_gpus_not_divisible_by_local_tp_is_an_error():
    role = _role(
        {
            "name": "decode",
            "lws": {"size": 1},
            "parallelism": {"gpus_per_node": 4, "tp": 3, "dp": False, "ep": True},
        }
    )

    with pytest.raises(ValueError, match="4 GPUs per pod is not divisible by local TP 3"):
        parallel_layout(role)


def test_dp_true_is_rejected():
    with pytest.raises(ValidationError, match="dp: true is ambiguous"):
        DeploymentSpec.model_validate(
            _spec_with_role({"name": "decode", "parallelism": {"tp": 1, "dp": True}})
        )


def test_routing_proxy_sets_default_port_bases():
    spec = DeploymentSpec.model_validate(
        _spec_with_role(
            {
                "name": "decode",
                "lws": {"size": 4},
                "parallelism": {"gpus_per_node": 4, "tp": 1, "dp": 16, "ep": True},
                "dp_load_balancing": "external",
                "routing_proxy": True,
            }
        )
    )

    role = spec.role("decode")
    assert role.routing_proxy is True
    assert role.serving_port_base == 8000
    assert role.backend_port_base == 8200


def test_routing_proxy_with_internal_dp_is_an_error():
    with pytest.raises(ValidationError, match="routing_proxy requires dp_load_balancing: external"):
        DeploymentSpec.model_validate(
            _spec_with_role(
                {
                    "name": "decode",
                    "lws": {"size": 4},
                    "parallelism": {"gpus_per_node": 4, "tp": 1, "dp": 16, "ep": True},
                    "routing_proxy": True,
                }
            )
        )


def test_external_dp_with_multiple_api_servers_is_an_error():
    with pytest.raises(ValidationError, match="api_server_count > 1 is incompatible"):
        DeploymentSpec.model_validate(
            _spec_with_role(
                {
                    "name": "decode",
                    "lws": {"size": 4},
                    "parallelism": {"gpus_per_node": 4, "tp": 1, "dp": 16, "ep": True},
                    "dp_load_balancing": "external",
                    "vllm": {"api_server_count": 4},
                }
            )
        )


def test_pd_topology_sets_decode_proxy_without_role_flag():
    spec = DeploymentSpec.model_validate(
        {
            "release": "pd",
            "topology": "pd",
            "model": {"id": "model", "image": "image"},
            "roles": [
                {
                    "name": "decode",
                    "lws": {"size": 4},
                    "parallelism": {"gpus_per_node": 4, "tp": 1, "dp": 16, "ep": True},
                },
                {
                    "name": "prefill",
                    "lws": {"size": 2},
                    "parallelism": {"gpus_per_node": 4, "tp": 8, "dp": False, "ep": True},
                },
            ],
        }
    )

    assert spec.role("decode").routing_proxy is True
    assert spec.role("decode").dp_load_balancing == "external"
    assert spec.role("decode").backend_port_base == 8200
    assert spec.role("prefill").routing_proxy is False


def test_parallelism_gpus_alias_still_parses():
    spec = DeploymentSpec.model_validate(
        _spec_with_role(
            {
                "name": "decode",
                "lws": {"size": 4},
                "parallelism": {"gpus": 4, "tp": 1, "dp": 16, "ep": True},
            }
        )
    )

    role = spec.role("decode")
    assert role.lws.size == 4
    assert role.gpus_per_pod == 4


def test_unknown_role_keys_are_rejected():
    with pytest.raises(ValidationError):
        DeploymentSpec.model_validate(
            _spec_with_role({"name": "decode", "tensor_parallel_size": 4})
        )
    with pytest.raises(ValidationError):
        DeploymentSpec.model_validate(
            _spec_with_role({"name": "decode", "parallelism": {"tp": 1, "dp_load_balancing": "external"}})
        )


def test_warns_when_dp_replicas_fit_by_gpu_but_not_aggregate_cpu():
    spec = DeploymentSpec.model_validate(
        _spec_with_role(
            {
                "name": "prefill",
                "lws": {"size": 1, "replicas": 4},
                "parallelism": {"tp": 1, "dp": 2},
            }
        )
    )
    cluster = load_cluster(ROOT / "clusters" / "cks-h200.yaml")
    cluster.model_server_resources.node_allocatable_cpu = "63"

    with pytest.warns(UserWarning, match=r"4 pods fit.*16 CPU each.*64 total.*allocatable CPU 63"):
        spec.apply_cluster_defaults(cluster)


def test_no_warning_when_dp_replicas_fit_aggregate_cpu():
    spec = DeploymentSpec.model_validate(
        _spec_with_role(
            {
                "name": "prefill",
                "lws": {"size": 1, "replicas": 4},
                "parallelism": {"tp": 1, "dp": 2},
            }
        )
    )
    cluster = load_cluster(ROOT / "clusters" / "cks-h200.yaml")
    cluster.model_server_resources.node_allocatable_cpu = "64"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec.apply_cluster_defaults(cluster)


def test_warns_when_dp_replicas_fit_by_gpu_but_not_aggregate_memory():
    spec = DeploymentSpec.model_validate(
        _spec_with_role(
            {
                "name": "prefill",
                "lws": {"size": 1, "replicas": 4},
                "parallelism": {"tp": 1, "dp": 2},
            }
        )
    )
    cluster = load_cluster(ROOT / "clusters" / "cks-h200.yaml")
    cluster.model_server_resources.node_allocatable_memory = "1023Gi"

    with pytest.warns(UserWarning, match=r"4 pods fit.*256Gi memory each.*1024Gi total"):
        spec.apply_cluster_defaults(cluster)
