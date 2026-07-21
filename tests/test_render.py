"""Structural tests for rendered Kubernetes objects and YAML serialization."""

from pathlib import Path

import yaml

from manifesto.cluster import load_cluster
from manifesto.images import DEFAULT_IMAGES
from manifesto.render import render, render_to_yaml
from manifesto.render.devpod import render_dev_pod
from manifesto.spec import load_spec


ROOT = Path(__file__).resolve().parents[1]
CLUSTER = load_cluster(ROOT / "clusters" / "oci-gb200.yaml")
CKS_H200 = load_cluster(ROOT / "clusters" / "cks-h200.yaml")
DEEPSEEK = "deepseek-v4/1P-EP8-1D-EP8.yaml"


def _objects(config: str) -> list[dict]:
    spec = load_spec(ROOT / "models" / config, CLUSTER)
    return render(spec, user="tester", cluster=CLUSTER)


def _find(objects: list[dict], kind: str, name_suffix: str | None = None) -> dict:
    for obj in objects:
        if obj["kind"] != kind:
            continue
        if name_suffix is None or obj["metadata"]["name"].endswith(name_suffix):
            return obj
    raise AssertionError(f"missing {kind} {name_suffix or ''}")


def test_rendered_yaml_parses():
    objects = _objects(DEEPSEEK)
    parsed = list(yaml.safe_load_all(render_to_yaml(objects)))

    assert len(parsed) == len(objects)


def test_rendered_launch_script_uses_literal_yaml_block():
    rendered = render_to_yaml(_objects(DEEPSEEK))

    assert "args:\n          - |-" in rendered
    assert "exec \\\n              vllm \\\n              serve \\" in rendered
    assert "\\nexec vllm serve" not in rendered


def test_nixl_roles_advertise_their_pod_ip():
    objects = _objects(DEEPSEEK)

    for role in ("decode", "prefill"):
        workload = _find(objects, "LeaderWorkerSet", role)
        container = workload["spec"]["leaderWorkerTemplate"]["workerTemplate"][
            "spec"
        ]["containers"][0]
        env = {item["name"]: item for item in container["env"]}

        assert env["VLLM_NIXL_SIDE_CHANNEL_HOST"] == {
            "name": "VLLM_NIXL_SIDE_CHANNEL_HOST",
            "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}},
        }


def test_explicit_nixl_side_channel_host_is_preserved():
    spec = load_spec(ROOT / "models" / DEEPSEEK, CLUSTER)
    spec.role("decode").env["VLLM_NIXL_SIDE_CHANNEL_HOST"] = "nixl.example.test"
    objects = render(spec, user="tester", cluster=CLUSTER)
    workload = _find(objects, "LeaderWorkerSet", "decode")
    container = workload["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"][
        "containers"
    ][0]
    matching = [
        item
        for item in container["env"]
        if item["name"] == "VLLM_NIXL_SIDE_CHANNEL_HOST"
    ]

    assert matching == [{"name": "VLLM_NIXL_SIDE_CHANNEL_HOST", "value": "nixl.example.test"}]


def test_non_nixl_role_does_not_get_side_channel_host():
    spec = load_spec(ROOT / "models" / "qwen" / "aggregated.yaml", CLUSTER)
    spec.role("decode").kv_transfer_config = {
        "kv_connector": "LMCacheConnectorV1",
    }
    objects = render(spec, user="tester", cluster=CLUSTER)
    workload = _find(objects, "Deployment", "decode")
    container = workload["spec"]["template"]["spec"]["containers"][0]

    assert not any(
        item["name"] == "VLLM_NIXL_SIDE_CHANNEL_HOST"
        for item in container["env"]
    )


def test_dp_ports_feed_container_readiness_and_inferencepool():
    objects = _objects(DEEPSEEK)
    lws = _find(objects, "LeaderWorkerSet", "decode")
    container = lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]
    infpool = _find(objects, "InferencePool")

    assert [p["containerPort"] for p in container["ports"]] == [8100, 8200, 8201, 8202, 8203]
    assert container["resources"]["requests"]["cpu"] == "32"
    assert container["resources"]["requests"]["memory"] == "512Gi"
    readiness = container["readinessProbe"]["exec"]["command"][-1]
    assert "localhost:8000" in readiness
    assert "localhost:8003" in readiness
    assert container["startupProbe"]["httpGet"]["port"] == "dp-supervisor"
    assert infpool["apiVersion"] == "inference.networking.k8s.io/v1"
    assert infpool["spec"]["targetPorts"] == [{"number": 8000}, {"number": 8001}, {"number": 8002}, {"number": 8003}]
    assert infpool["spec"]["endpointPickerRef"]["name"] == "tester-wide-ep-1p-ep8-1d-ep8-infpool-epp"
    script = container["args"][0]
    assert "DP_SIZE=8" in script
    assert "DP_SIZE=$((LWS_GROUP_SIZE * DP_SIZE_LOCAL))" not in script
    assert "--data-parallel-multi-port-external-lb" in script
    assert "--data-parallel-supervisor-port 8100" in script
    assert "--data-parallel-start-rank $START_RANK" in script
    assert "--data-parallel-rank" not in script
    assert "exec \\\n  vllm \\\n  serve \\\n  deepseek-ai/DeepSeek-V4-Pro \\" in script
    assert "  --port 8200 \\" in script
    assert "  --max-num-seqs 1024 \\" in script
    assert "  --enable-eplb False" not in script


def test_deepseek_lws_uses_short_workload_names_with_full_instance_labels():
    objects = _objects(DEEPSEEK)
    decode = _find(objects, "LeaderWorkerSet", "decode")
    prefill = _find(objects, "LeaderWorkerSet", "prefill")

    assert decode["metadata"]["name"] == "tester-vllm-ep8-decode"
    assert prefill["metadata"]["name"] == "tester-vllm-ep8-prefill"
    assert decode["metadata"]["labels"]["app.kubernetes.io/instance"] == "tester-wide-ep-1p-ep8-1d-ep8"
    assert prefill["metadata"]["labels"]["app.kubernetes.io/instance"] == "tester-wide-ep-1p-ep8-1d-ep8"


def test_deepseek_ep16_decode_name_keeps_decode_width():
    spec = load_spec(ROOT / "models" / "deepseek-v4" / "3P-EP8-1D-EP16.yaml", CLUSTER)
    objects = render(spec, user="tester", cluster=CLUSTER)
    decode = _find(objects, "LeaderWorkerSet", "decode")
    prefill = _find(objects, "LeaderWorkerSet", "prefill")

    assert decode["metadata"]["name"] == "tester-vllm-ep16-decode"
    assert prefill["metadata"]["name"] == "tester-vllm-ep8-prefill"


def test_logs_persist_to_cluster_log_root():
    objects = _objects(DEEPSEEK)
    lws = _find(objects, "LeaderWorkerSet", "decode")
    pod_spec = lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]
    container = pod_spec["containers"][0]
    script = container["args"][0]
    volumes = {volume["name"]: volume for volume in pod_spec["volumes"]}
    mounts = {mount["name"]: mount["mountPath"] for mount in container["volumeMounts"]}

    assert volumes["shared-storage"]["persistentVolumeClaim"]["claimName"] == "lustre-pvc-vllm"
    assert mounts["shared-storage"] == "/mnt/lustre"
    assert "LOG_DIR=/mnt/lustre/tester/logs/decode" in script


def test_shared_storage_accepts_non_pvc_volume_sources():
    cluster = CLUSTER.model_copy(deep=True)
    cluster.storage.shared_volume = {"emptyDir": {}}
    spec = load_spec(ROOT / "models" / DEEPSEEK, cluster)
    objects = render(spec, user="tester", cluster=cluster)
    lws = _find(objects, "LeaderWorkerSet", "decode")
    pod_spec = lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]
    volume = next(volume for volume in pod_spec["volumes"] if volume["name"] == "shared-storage")

    assert volume == {"name": "shared-storage", "emptyDir": {}}


def test_gateway_class_comes_from_cluster_profile():
    cluster = CLUSTER.model_copy(deep=True)
    cluster.gateway.class_name = "platform-gateway"
    spec = load_spec(ROOT / "models" / DEEPSEEK, cluster)
    objects = render(spec, user="tester", cluster=cluster)
    gateway = _find(objects, "Gateway")
    gateway_options = _find(objects, "ConfigMap", "gateway-options")

    assert gateway["spec"]["gatewayClassName"] == "platform-gateway"
    assert len(f"{gateway['metadata']['name']}-platform-gateway") <= 63
    assert not any(
        obj["kind"] == "Service" and obj["metadata"]["name"].startswith(gateway["metadata"]["name"])
        for obj in objects
    )
    assert yaml.safe_load(gateway_options["data"]["service"])["spec"]["type"] == "ClusterIP"


def test_dedicated_logging_pvc_is_mounted_when_configured():
    cluster = CLUSTER.model_copy(deep=True)
    cluster.logging.pvc = "logs-pvc"
    cluster.logging.mount_path = "/mnt/logs"
    cluster.logging.root = "/mnt/logs/{user}/{release}"
    spec = load_spec(ROOT / "models" / DEEPSEEK, cluster)
    objects = render(spec, user="tester", cluster=cluster)
    lws = _find(objects, "LeaderWorkerSet", "decode")
    pod_spec = lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]
    container = pod_spec["containers"][0]
    volumes = {volume["name"]: volume for volume in pod_spec["volumes"]}
    mounts = {mount["name"]: mount["mountPath"] for mount in container["volumeMounts"]}

    assert volumes["logs"]["persistentVolumeClaim"]["claimName"] == "logs-pvc"
    assert mounts["logs"] == "/mnt/logs"
    assert "LOG_DIR=/mnt/logs/tester/wide-ep-1p-ep8-1d-ep8/decode" in container["args"][0]


def test_no_dp_qwen_uses_single_port_and_no_dp_flags():
    spec = load_spec(ROOT / "models" / "qwen" / "aggregated.yaml", CLUSTER)
    spec.role("decode").lws.replicas = 2
    objects = render(spec, user="tester", cluster=CLUSTER)
    deployment = _find(objects, "Deployment", "decode")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    script = container["args"][0]
    infpool = _find(objects, "InferencePool")

    assert not any(obj["kind"] == "LeaderWorkerSet" for obj in objects)
    assert deployment["spec"]["replicas"] == 2
    assert deployment["spec"]["selector"]["matchLabels"] == {
        "app.kubernetes.io/instance": "tester-qwen",
        "llm-d.ai/role": "decode",
    }
    assert deployment["spec"]["template"]["metadata"]["labels"].items() >= deployment["spec"]["selector"][
        "matchLabels"
    ].items()
    assert [p["containerPort"] for p in container["ports"]] == [8000]
    assert "--data-parallel-size" not in script
    assert "startupProbe" not in container
    assert infpool["spec"]["targetPorts"] == [{"number": 8000}]


def test_single_node_dp_uses_deployment_without_lws_environment():
    spec = load_spec(ROOT / "models" / "qwen" / "h200-aggregated.yaml", CKS_H200)
    objects = render(spec, user="tester", cluster=CKS_H200)
    deployment = _find(objects, "Deployment", "decode")
    script = deployment["spec"]["template"]["spec"]["containers"][0]["args"][0]

    assert "LWS_" not in script
    assert "START_RANK=0" in script
    assert "--data-parallel-address 127.0.0.1" in script


def test_pd_inferencepool_selector_includes_prefill_and_decode_roles():
    objects = _objects(DEEPSEEK)
    infpool = _find(objects, "InferencePool")

    selector = infpool["spec"]["selector"]["matchLabels"]
    assert selector["app.kubernetes.io/instance"] == "tester-wide-ep-1p-ep8-1d-ep8"
    assert selector["llm-d.ai/deployment"] == "pd"
    assert selector["llm-d.ai/inferenceServing"] == "true"
    assert "llm-d.ai/role" not in selector


def test_non_pd_inferencepool_selector_targets_decode_role():
    objects = _objects("qwen/aggregated.yaml")
    infpool = _find(objects, "InferencePool")

    selector = infpool["spec"]["selector"]["matchLabels"]
    assert selector["app.kubernetes.io/instance"] == "tester-qwen"
    assert selector["llm-d.ai/role"] == "decode"


def test_inferencepool_references_epp_service():
    objects = _objects(DEEPSEEK)
    service = _find(objects, "Service", "infpool-epp")

    assert service["spec"]["selector"]["app.kubernetes.io/component"] == "epp"
    assert service["spec"]["ports"][0]["targetPort"] == 9002


def test_epp_uses_current_config_file_flag():
    objects = _objects(DEEPSEEK)
    deployment = _find(objects, "Deployment", "infpool-epp")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    args = container["args"]

    assert container["image"] == DEFAULT_IMAGES.get("llm_d.epp", release=DEFAULT_IMAGES.get("llm_d.release"))
    assert "--config-file=/etc/epp/plugins.yaml" in args
    assert "--pool-name=tester-wide-ep-1p-ep8-1d-ep8-infpool" in args
    assert "--pool-namespace=default" in args
    assert not any(arg.startswith("--plugins-config-file") for arg in args)


def test_epp_uses_dedicated_service_account_and_rbac():
    objects = _objects(DEEPSEEK)
    service_account = _find(objects, "ServiceAccount", "infpool-epp")
    role = _find(objects, "Role", "infpool-epp-rbac")
    binding = _find(objects, "RoleBinding", "infpool-epp-rbac")
    deployment = _find(objects, "Deployment", "infpool-epp")

    assert deployment["spec"]["template"]["spec"]["serviceAccountName"] == service_account["metadata"]["name"]
    assert binding["subjects"] == [
        {"kind": "ServiceAccount", "name": service_account["metadata"]["name"], "namespace": "default"}
    ]
    assert binding["roleRef"]["name"] == role["metadata"]["name"]

    rules = {(tuple(rule["apiGroups"]), tuple(rule["resources"])): rule["verbs"] for rule in role["rules"]}
    assert rules[(("",), ("pods",))] == ["get", "list", "watch"]
    assert rules[(("inference.networking.k8s.io",), ("inferencepools",))] == ["get", "list", "watch"]
    assert rules[
        (
            ("inference.networking.x-k8s.io",),
            ("inferencemodelrewrites", "inferencemodels", "inferenceobjectives", "inferencepoolimports"),
        )
    ] == ["get", "list", "watch"]


def test_cks_h200_cluster_uses_coreweave_cache_and_rdma_settings():
    spec = load_spec(ROOT / "models" / "qwen" / "h200-aggregated.yaml", CKS_H200)
    assert spec.model.hf_home == "/var/cache/huggingface"
    objects = render(spec, user="tester", cluster=CKS_H200)
    deployment = _find(objects, "Deployment", "decode")
    pod_spec = deployment["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]
    script = container["args"][0]
    env = {item["name"]: item["value"] for item in container["env"] if "value" in item}

    volumes = {volume["name"]: volume for volume in pod_spec["volumes"]}
    mounts = {mount["name"]: mount["mountPath"] for mount in container["volumeMounts"]}

    assert volumes["hf-cache"]["hostPath"]["path"] == "/mnt/local/hf-cache"
    assert volumes["jit-cache"]["hostPath"]["path"] == "/mnt/local/jit-cache"
    assert mounts["hf-cache"] == "/var/cache/huggingface"
    assert mounts["jit-cache"] == "/var/cache/vllm"
    assert env["NCCL_IB_HCA"] == "ibp"
    assert env["NVSHMEM_HCA_PREFIX"] == "ibp"
    assert env["HF_HUB_CACHE"] == "/var/cache/huggingface"
    assert env["FLASHINFER_WORKSPACE_BASE"] == "/var/cache/vllm/flashinfer"
    assert "MAX_TOKENS" not in env
    assert "--max-num-batched-tokens" not in script
    assert "--max-num-seqs" not in script
    assert "--max-cudagraph-capture-size" not in script
    assert container["resources"]["requests"]["rdma/ib"] == "1"
    assert container["resources"]["limits"]["rdma/ib"] == "1"


def test_dev_pod_derives_storage_and_paths_from_cluster_profile():
    pod = render_dev_pod(CLUSTER, "Tester.Name")
    container = pod["spec"]["containers"][0]
    env = {item["name"]: item["value"] for item in container["env"] if "value" in item}
    volumes = {volume["name"]: volume for volume in pod["spec"]["volumes"]}

    assert pod["metadata"]["name"] == "tester-name-vllm-dev"
    assert container["image"] == DEFAULT_IMAGES.get("dev.image")
    assert volumes["shared-storage"]["persistentVolumeClaim"]["claimName"] == "lustre-pvc-vllm"
    assert container["workingDir"] == "/mnt/lustre/tester-name/vllm-dev"
    assert env["HF_HOME"] == "/mnt/lustre/hf_cache"
    assert env["CCACHE_DIR"] == "/mnt/lustre/tester-name/ccache"

    h200_pod = render_dev_pod(CKS_H200, "tester")
    h200_volumes = {volume["name"]: volume for volume in h200_pod["spec"]["volumes"]}
    h200_env = {
        item["name"]: item["value"] for item in h200_pod["spec"]["containers"][0]["env"] if "value" in item
    }
    assert h200_volumes["hf-cache"]["hostPath"]["path"] == "/mnt/local/hf-cache"
    assert h200_env["HF_HOME"] == "/var/cache/huggingface"


def test_lws_uses_cluster_routing_sidecar_image():
    objects = _objects(DEEPSEEK)
    lws = _find(objects, "LeaderWorkerSet", "decode")
    init_container = lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["initContainers"][0]

    assert init_container["image"] == DEFAULT_IMAGES.get(
        "llm_d.routing_sidecar",
        release=DEFAULT_IMAGES.get("llm_d.release"),
    )


def test_routing_plugin_config_can_be_inline_override():
    spec = load_spec(ROOT / "models" / "qwen" / "aggregated.yaml", CLUSTER)
    spec.routing.plugin_config = {
        "apiVersion": "inference.networking.x-k8s.io/v1alpha1",
        "kind": "EndpointPickerConfig",
        "plugins": [{"type": "weighted-random-picker", "name": "custom-picker"}],
    }
    objects = render(spec, user="tester", cluster=CLUSTER)
    config = _find(objects, "ConfigMap", "epp-config")

    assert "custom-picker" in config["data"]["plugins.yaml"]
    assert "active-request-scorer" not in config["data"]["plugins.yaml"]


def test_prefill_launch_uses_global_tp_and_local_gpu_span():
    objects = _objects(DEEPSEEK)
    lws = _find(objects, "LeaderWorkerSet", "prefill")
    container = lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]
    script = container["args"][0]

    assert "--tensor-parallel-size 1" in script
    assert "DP_SIZE=8" in script
    assert "--data-parallel-multi-port-external-lb" in script
    assert "--data-parallel-start-rank $START_RANK" in script
    assert "--data-parallel-rank" not in script


def test_deepseek_v4_nested_attention_config_preserves_official_flag_spelling():
    objects = _objects(DEEPSEEK)
    for role in ("decode", "prefill"):
        lws = _find(objects, "LeaderWorkerSet", role)
        script = lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]["args"][0]

        assert "--attention_config.use_fp4_indexer_cache=True" in script
        assert "attention-config.use-fp4-indexer-cache" not in script
