"""Gateway API, InferencePool, and EPP manifests for instance-scoped routing."""

from __future__ import annotations

import yaml

from .common import secret_env
from ..instance import Instance
from ..cluster import Cluster
from ..parallelism import parallel_layout
from ..resolve import resolve_role
from ..spec import DeploymentSpec, RoutingKind, RoutingSpec


def _plugin_config(
    routing: RoutingSpec,
    *,
    dp_enabled: bool = False,
    model_name: str | None = None,
    namespace: str = "default",
    prefill_lws_name: str | None = None,
    enable_p2p: bool = False,
    tokenizer_url: str = "http://localhost:8000",
) -> str:
    if routing.plugin_config is not None:
        return yaml.safe_dump(routing.plugin_config, sort_keys=False)
    if routing.kind == RoutingKind.PD:
        pod_label_selector = (
            f"leaderworkerset.sigs.k8s.io/name={prefill_lws_name},llm-d.ai/role=prefill"
            if prefill_lws_name
            else "llm-d.ai/role=prefill"
        )
        config = {
            "apiVersion": "llm-d.ai/v1alpha1",
            "kind": "EndpointPickerConfig",
            "plugins": [
                {"type": "disagg-headers-handler"},
                {"type": "always-disagg-pd-decider"},
                {
                    "type": "disagg-profile-handler",
                    "parameters": {"deciderPluginName": "always-disagg-pd-decider"},
                },
                {
                    # Real tokenizer, required by precise-prefix-cache-producer
                    # (the estimate backend's byte-packed IDs don't correlate
                    # with vLLM's own KV-block hashes). Calls a CPU-only
                    # `vllm launch render` sidecar in this same EPP pod (see
                    # render_routing) rather than a serving prefill/decode
                    # pod, so tokenization never competes with GPU work.
                    "type": "token-producer",
                    "parameters": {
                        "modelName": model_name,
                        "vllm": {"url": tokenizer_url, "timeout": "15s"},
                    },
                },
                {
                    # Feeds precise-prefix-cache-producer's per-pod ZMQ
                    # subscriber lifecycle (connect on pod add, disconnect on
                    # pod delete).
                    "type": "endpoint-notification-source",
                },
                {"type": "prefill-filter"},
                {"type": "decode-filter"},
                {
                    # Subscribes to vLLM's real KV-block store/remove event
                    # stream (--kv-events-config on the prefill role) and
                    # builds an exact per-pod-per-tier index. Named
                    # "approx-prefix-cache-producer" so downstream plugins
                    # reference it by that stable name regardless of the
                    # underlying implementation. blockSizeTokens must match
                    # --block-size on the engine so EPP-recomputed hashes
                    # align with KV-event block boundaries.
                    # kvCacheBackendConfigs tracks both the GPU tier
                    # (weight 1.0) and the SimpleCPUOffload tier (weight 0.5).
                    # fullReportRepair requests a full index snapshot from a
                    # prefill pod when the EPP detects too many missing blocks,
                    # closing gaps from missed ZMQ events.
                    "type": "precise-prefix-cache-producer",
                    "name": "approx-prefix-cache-producer",
                    "parameters": {
                        "tokenProcessorConfig": {"blockSizeTokens": 256},
                        "speculativeIndexing": False,
                        "indexerConfig": {
                            "kvBlockIndexConfig": {
                                "enableMetrics": True,
                                "inMemoryConfig": {"size": 10000000, "podCacheSize": 1024},
                            },
                            "kvCacheBackendConfigs": [
                                {"name": "gpu", "weight": 1.0},
                                {"name": "cpu", "weight": 0.5},
                            ],
                        },
                        "kvEventsConfig": {
                            "topicFilter": "kv@",
                            "concurrency": 64,
                            "discoverPods": True,
                            "podDiscoveryConfig": {
                                "podNamespace": namespace,
                                "podLabelSelector": pod_label_selector,
                                "socketPort": 5557,
                            },
                        },
                        "fullReportRepair": {
                            "fullReportThreshold": 0.8,
                            "minMissingBlocks": 32,
                            "prefillProfileName": "prefill",
                        },
                    },
                },
                *(
                    [
                        {
                            # Reads the precise-prefix-cache index and, when a
                            # request's cached-token delta on a prefill exceeds
                            # minCachedTokenDelta, adds x-kv-cache-source-host-port
                            # so the decode sidecar (--enable-p2p-pull) injects
                            # remote_kv_source into the prefill leg, triggering a
                            # cross-prefill P2P pull from the source's CPU tier.
                            # minCachedTokenDelta is provisional; measure the
                            # pull-vs-recompute crossover for DeepSeek-V4 on GB200
                            # before finalising (spec §4 note).
                            "type": "p2p-source-producer",
                            "parameters": {
                                "prefixMatchInfoProducerName": "approx-prefix-cache-producer",
                                "prefillProfileName": "prefill",
                                "minCachedTokenDelta": 2048,
                            },
                        }
                    ]
                    if enable_p2p
                    else []
                ),
                {
                    # Named explicitly so gpu-prefix-cache-affinity-filter
                    # and token-load-scorer can reference it by name.
                    # addEstimatedOutputTokens=false: output tokens aren't
                    # known at scheduling time for prefill, so adding an
                    # estimate would skew the inflight load signal.
                    "type": "inflight-load-producer",
                    "name": "inflight-load-producer",
                    "parameters": {
                        "addEstimatedOutputTokens": False,
                        "prefixMatchInfoProducerName": "approx-prefix-cache-producer",
                    },
                },
                {
                    # Hard filter: blocks a prefill endpoint when its in-flight
                    # request count exceeds maxConcurrency * (1 + headroom) = 64.
                    # With 8 prefill endpoints and input concurrency 256,
                    # steady-state average is 32 per endpoint — well below 64,
                    # so no endpoint is ever blocked in steady state. The guard
                    # only fires during a cold-start hot-spot where one rank
                    # accumulates requests before the affinity filter can
                    # differentiate endpoints. All 8 endpoints would be blocked
                    # simultaneously only at 8 × 64 = 512 concurrent in-flight
                    # requests, which is above the benchmark concurrency of 256.
                    "type": "concurrency-detector",
                    "name": "prefill-concurrency-guard",
                    "parameters": {
                        "concurrencyMode": "requests",
                        "maxConcurrency": 32,
                        "headroom": 1.0,
                        "inFlightLoadProducerName": "inflight-load-producer",
                    },
                },
                {
                    # TTFT-gate affinity filter: excludes an endpoint only when
                    # routing to it would incur >maxTTFTPenaltyMs extra latency
                    # vs the best cached endpoint. peakPrefillThroughput
                    # calibrated via llm-d's calibrate.sh against the live
                    # deployment: sequential 8192-token random-token-ID
                    # requests, guaranteed cache miss, median TTFT 1.7124s →
                    # 4783 tok/s (v16 run). maxTTFTPenaltyMs=30000 derived from
                    # live gate telemetry over the v16 c192 benchmark (see git
                    # history on imarkov/dsv4-bench for full derivation).
                    # affinityThreshold intentionally omitted: match-ratio
                    # hard-exclusion caused cold ranks to be permanently starved.
                    # explorationProbability intentionally omitted: no-hit-lru-scorer
                    # handles cold-start starvation directly (steers zero-hit requests
                    # to the LRU rank), so random exploration of cache-hit requests
                    # would only waste throughput in steady state.
                    "type": "prefix-cache-affinity-filter",
                    "name": "gpu-prefix-cache-affinity-filter",
                    "parameters": {
                        "inFlightLoadProducerName": "inflight-load-producer",
                        "maxTTFTPenaltyMs": 30000,
                        "peakPrefillThroughput": 4783,
                        "prefixMatchInfoProducerName": "approx-prefix-cache-producer",
                    },
                },
                {
                    "type": "token-load-scorer",
                    "parameters": {"inFlightLoadProducerName": "inflight-load-producer"},
                },
                {
                    # Cold-burst distributor: for requests where no candidate
                    # endpoint has any matching KV blocks (cold requests), scores
                    # endpoints by reverse-LRU of cold-request assignments so that
                    # ranks that have never (or least recently) received a cold
                    # request are preferred. On cache-hit requests it returns
                    # neutral scores (0.5) and defers fully to token-load-scorer.
                    # Weight 1:1 with token-load-scorer so it dominates cold-start
                    # but does not override the load signal in steady state.
                    "type": "no-hit-lru-scorer",
                    "parameters": {
                        "prefixMatchInfoProducerName": "approx-prefix-cache-producer",
                    },
                },
                {
                    "type": "active-request-scorer",
                    "parameters": {"inFlightLoadProducerName": "inflight-load-producer"},
                },
                {"type": "max-score-picker"},
            ],
            "dataLayer": {
                "injectDefaults": False,
                "sources": [
                    {
                        "pluginRef": "endpoint-notification-source",
                        "extractors": [{"pluginRef": "approx-prefix-cache-producer"}],
                    }
                ],
            },
            "schedulingProfiles": [
                {
                    "name": "prefill",
                    "plugins": [
                        {"pluginRef": "prefill-filter"},
                        {"pluginRef": "prefill-concurrency-guard"},
                        {"pluginRef": "gpu-prefix-cache-affinity-filter"},
                        {"pluginRef": "token-load-scorer", "weight": 1},
                        {"pluginRef": "no-hit-lru-scorer", "weight": 1},
                        {"pluginRef": "max-score-picker"},
                    ],
                },
                {
                    "name": "decode",
                    "plugins": [
                        {"pluginRef": "decode-filter"},
                        {"pluginRef": "active-request-scorer"},
                        {"pluginRef": "max-score-picker"},
                    ],
                },
            ],
        }
    elif dp_enabled:
        config = {
            "apiVersion": "llm-d.ai/v1alpha1",
            "kind": "EndpointPickerConfig",
            "plugins": [
                {
                    "type": "approx-prefix-cache-producer",
                    "parameters": {
                        "blockSizeTokens": 256,
                        "maxPrefixTokensToMatch": 1048576,
                        "maxPrefixBlocksToMatch": 4096,
                    },
                },
                {
                    "type": "prefix-cache-affinity-filter",
                    "parameters": {"peakPrefillThroughput": 80000},
                },
                {"type": "prefix-cache-scorer"},
                {"type": "active-request-scorer"},
                {"type": "queue-scorer"},
                {"type": "weighted-random-picker"},
            ],
            "schedulingProfiles": [
                {
                    "name": "default",
                    "plugins": [
                        {"pluginRef": "prefix-cache-affinity-filter"},
                        {"pluginRef": "prefix-cache-scorer", "weight": 10},
                        {"pluginRef": "active-request-scorer", "weight": 2},
                        {"pluginRef": "queue-scorer", "weight": 2},
                        {"pluginRef": "weighted-random-picker"},
                    ],
                }
            ],
        }
    else:
        config = {
            "apiVersion": "inference.networking.x-k8s.io/v1alpha1",
            "kind": "EndpointPickerConfig",
            "plugins": [
                {"type": "active-request-scorer"},
                {"type": "queue-scorer"},
                {"type": "weighted-random-picker"},
            ],
            "schedulingProfiles": [
                {
                    "name": "default",
                    "plugins": [
                        {"pluginRef": "active-request-scorer", "weight": 2},
                        {"pluginRef": "queue-scorer", "weight": 2},
                        {"pluginRef": "weighted-random-picker"},
                    ],
                }
            ],
        }
    return yaml.safe_dump(config, sort_keys=False)


def render_routing(spec: DeploymentSpec, instance: Instance, cluster: Cluster) -> list[dict]:
    assert spec.routing.kind is not None
    assert spec.routing.target_role is not None
    if spec.routing.kind == RoutingKind.DISABLED:
        return []

    role = spec.role(spec.routing.target_role)
    ports = resolve_role(spec, instance, cluster, role).ports
    infpool_name = instance.name("infpool")
    epp_name = instance.name("infpool-epp")
    epp_role_name = instance.name("infpool-epp-rbac")
    gateway_name_limit = 63 - len(cluster.gateway.class_name) - 1
    gateway_name = instance.name("gateway", max_length=gateway_name_limit)

    selector = instance.pod_selector(None if spec.routing.kind == RoutingKind.PD else spec.routing.target_role) | {
        "llm-d.ai/inferenceServing": "true",
        "llm-d.ai/deployment": spec.topology.value,
    }

    layout = parallel_layout(role)
    if layout.tp_world_size > layout.tp_local_size:
        selector["leaderworkerset.sigs.k8s.io/worker-index"] = "0"

    prefill_lws_name: str | None = None
    tokenizer_svc_name: str | None = None
    if spec.routing.kind == RoutingKind.PD:
        try:
            prefill_role = spec.role("prefill")
            prefill_lws_name = (
                instance.user_scoped_name(prefill_role.workload_name)
                if prefill_role.workload_name
                else instance.name(prefill_role.name)
            )
        except KeyError:
            pass
        tokenizer_svc_name = instance.name("vllm-tokenize")

    tokenizer_url = (
        f"http://{tokenizer_svc_name}.{spec.namespace}.svc.cluster.local:8000"
        if tokenizer_svc_name
        else "http://localhost:8000"
    )

    return [
        {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {"name": epp_name, "labels": instance.labels("epp")},
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": {"name": epp_role_name, "labels": instance.labels("epp")},
            "rules": [
                {
                    "apiGroups": [""],
                    "resources": ["pods"],
                    "verbs": ["get", "list", "watch"],
                },
                {
                    "apiGroups": ["inference.networking.k8s.io"],
                    "resources": ["inferencepools"],
                    "verbs": ["get", "list", "watch"],
                },
                {
                    "apiGroups": ["inference.networking.x-k8s.io"],
                    "resources": [
                        "inferencemodelrewrites",
                        "inferencemodels",
                        "inferenceobjectives",
                        "inferencepoolimports",
                    ],
                    "verbs": ["get", "list", "watch"],
                },
            ],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": {"name": epp_role_name, "labels": instance.labels("epp")},
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": epp_name,
                    "namespace": spec.namespace,
                }
            ],
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": epp_role_name,
            },
        },
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": instance.name("epp-config"), "labels": instance.labels("routing")},
            "data": {
                "plugins.yaml": _plugin_config(
                    spec.routing,
                    dp_enabled=role.parallelism.dp_enabled,
                    model_name=spec.model.served_name or spec.model.id,
                    namespace=spec.namespace,
                    prefill_lws_name=prefill_lws_name,
                    enable_p2p=any(r.p2p_config for r in spec.roles),
                    tokenizer_url=tokenizer_url,
                )
            },
        },
        *(
            [
                {
                    # Tokenizer Service: load-balances EPP token-producer calls
                    # across all serving pods (prefill + decode). Uses the named
                    # port "vllm-0" so kube-proxy resolves per pod: 8000 on
                    # prefill (vLLM direct) and 8200 on decode (vLLM behind the
                    # routing sidecar). No role filter needed — all pods expose
                    # vllm-0 under the same name via lws.py port assignment.
                    "apiVersion": "v1",
                    "kind": "Service",
                    "metadata": {
                        "name": tokenizer_svc_name,
                        "labels": instance.labels("tokenizer"),
                    },
                    "spec": {
                        "selector": instance.pod_selector() | {"llm-d.ai/inferenceServing": "true"},
                        "ports": [{"name": "http", "port": 8000, "targetPort": "vllm-0", "protocol": "TCP"}],
                    },
                }
            ]
            if tokenizer_svc_name
            else []
        ),
        {
            "apiVersion": "inference.networking.k8s.io/v1",
            "kind": "InferencePool",
            "metadata": {"name": infpool_name, "labels": instance.labels("routing")},
            "spec": {
                "targetPorts": [{"number": port} for port in ports.public],
                "selector": {"matchLabels": selector},
                "endpointPickerRef": {"name": epp_name, "kind": "Service", "port": {"number": 9002}},
            },
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": epp_name, "labels": instance.labels("epp")},
            "spec": {
                "selector": instance.labels("epp"),
                "ports": [{"name": "grpc", "port": 9002, "protocol": "TCP", "targetPort": 9002}],
            },
        },
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": epp_name, "labels": instance.labels("epp")},
            "spec": {
                "replicas": spec.routing.replicas,
                "selector": {"matchLabels": instance.labels("epp")},
                "template": {
                    "metadata": {"labels": instance.labels("epp") | {"inferencepool": epp_name}},
                    "spec": {
                        "serviceAccountName": epp_name,
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
                        "containers": [
                            {
                                "name": "epp",
                                "image": spec.routing.epp_image or cluster.llm_d.epp,
                                "imagePullPolicy": "Always",
                                "args": [
                                    "--config-file=/etc/epp/plugins.yaml",
                                    "--grpc-port=9002",
                                    "--grpc-health-port=9003",
                                    f"--pool-name={infpool_name}",
                                    f"--pool-namespace={spec.namespace}",
                                    "--pool-group=inference.networking.k8s.io",
                                    "--allow-experimental-plugins",
                                    "--refresh-metrics-interval=500ms",
                                    "--zap-log-level=debug",
                                ],
                                "ports": [{"containerPort": 9002, "name": "grpc"}],
                                "volumeMounts": [
                                    {"name": "config", "mountPath": "/etc/epp/plugins.yaml", "subPath": "plugins.yaml"}
                                ],
                                "resources": {
                                    "requests": {"cpu": "8", "memory": "16Gi"},
                                    "limits": {"cpu": "8", "memory": "16Gi"},
                                },
                            },
                            # vllm-render sidecar disabled: token-producer now
                            # calls the vllm-tokenize Service (prefill pods,
                            # port 8000) which scales with the prefill fleet and
                            # avoids a separate CPU container on the EPP node.
                            # Re-enable by reverting tokenizer_url to localhost
                            # and restoring the sidecar block below.
                            #
                            # *(
                            #     [{"name": "vllm-render", "image": spec.routing.render_image or "vllm/vllm-openai-cpu:latest", ...}]
                            #     if spec.routing.kind == RoutingKind.PD
                            #     else []
                            # ),
                        ],
                        "volumes": [{"name": "config", "configMap": {"name": instance.name("epp-config")}}],
                    },
                },
            },
        },
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": instance.name("gateway-options"), "labels": instance.labels("gateway")},
            "data": {
                "deployment": yaml.safe_dump(
                    {
                        "spec": {
                            "template": {
                                "spec": {
                                    "containers": [
                                        {
                                            "name": "istio-proxy",
                                            "resources": {
                                                "requests": {"cpu": "8", "memory": "64Gi"},
                                                "limits": {"cpu": "8", "memory": "64Gi"},
                                            },
                                        }
                                    ]
                                }
                            }
                        }
                    },
                    sort_keys=False,
                ),
                "service": yaml.safe_dump({"spec": {"type": cluster.gateway.service_type}}, sort_keys=False),
            },
        },
        {
            "apiVersion": "gateway.networking.k8s.io/v1",
            "kind": "Gateway",
            "metadata": {
                "name": gateway_name,
                "labels": instance.labels("gateway") | {"istio.io/enable-inference-extproc": "true"},
            },
            "spec": {
                "infrastructure": {
                    "parametersRef": {
                        "group": "",
                        "kind": "ConfigMap",
                        "name": instance.name("gateway-options"),
                    }
                },
                "gatewayClassName": cluster.gateway.class_name,
                "listeners": [
                    {
                        "name": "default",
                        "port": 80,
                        "protocol": "HTTP",
                        "allowedRoutes": {"namespaces": {"from": "Same"}},
                    }
                ],
            },
        },
        {
            "apiVersion": "gateway.networking.k8s.io/v1",
            "kind": "HTTPRoute",
            "metadata": {"name": instance.name("route"), "labels": instance.labels("route")},
            "spec": {
                "parentRefs": [{"group": "gateway.networking.k8s.io", "kind": "Gateway", "name": gateway_name}],
                "rules": [
                    {
                        "backendRefs": [
                            {
                                "group": "inference.networking.k8s.io",
                                "kind": "InferencePool",
                                "name": infpool_name,
                                "port": ports.public[0],
                                "weight": 1,
                            }
                        ],
                        "matches": [{"path": {"type": "PathPrefix", "value": "/"}}],
                        "timeouts": {"backendRequest": "0s", "request": "0s"},
                    }
                ],
            },
        },
        {
            "apiVersion": "networking.istio.io/v1",
            "kind": "DestinationRule",
            "metadata": {"name": epp_name, "labels": instance.labels("epp")},
            "spec": {
                "host": epp_name,
                "trafficPolicy": {
                    "connectionPool": {
                        "tcp": {
                            "connectTimeout": "900s",
                            "maxConnectionDuration": "1800s",
                            "maxConnections": 256000,
                        },
                        "http": {
                            "http1MaxPendingRequests": 256000,
                            "http2MaxRequests": 256000,
                            "idleTimeout": "900s",
                            "maxRequestsPerConnection": 256000,
                        },
                    },
                    "tls": {"insecureSkipVerify": True, "mode": "SIMPLE"},
                },
            },
        },
        {
            "apiVersion": "networking.istio.io/v1",
            "kind": "DestinationRule",
            "metadata": {"name": instance.name("infpool-backend"), "labels": instance.labels("routing")},
            "spec": {
                "host": f"{infpool_name}-ip",
                "trafficPolicy": {
                    "connectionPool": {
                        "tcp": {"maxConnections": 256000},
                        "http": {"idleTimeout": "300s"},
                    }
                },
            },
        },
    ]
