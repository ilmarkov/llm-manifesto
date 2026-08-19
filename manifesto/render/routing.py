"""Gateway API, InferencePool, and EPP manifests for instance-scoped routing."""

from __future__ import annotations

import yaml

from ..instance import Instance
from ..cluster import Cluster
from ..parallelism import parallel_layout
from ..resolve import resolve_role
from ..spec import DeploymentSpec, RoutingKind, RoutingSpec


def _plugin_config(routing: RoutingSpec, *, dp_enabled: bool = False) -> str:
    if routing.plugin_config is not None:
        return yaml.safe_dump(routing.plugin_config, sort_keys=False)
    if routing.kind == RoutingKind.PD:
        config = {
            "apiVersion": "llm-d.ai/v1alpha1",
            "kind": "EndpointPickerConfig",
            "plugins": [
                {"type": "prefill-filter"},
                {"type": "decode-filter"},
                {
                    "type": "approx-prefix-cache-producer",
                    "parameters": {
                        "blockSizeTokens": 256,
                        "maxPrefixTokensToMatch": 1048576,
                        "maxPrefixBlocksToMatch": 4096,
                    },
                },
                # Explicit instance (not auto-injected): llm-d-router v0.9.0 only registers
                # inflight-load-producer as the default producer for InFlightLoadDataKey, not
                # for UncachedRequestTokensDataKey, even though it produces both. Without this
                # explicit declaration, EPP fails to start when token-load-scorer (the only
                # consumer of UncachedRequestTokensDataKey) is configured.
                {"type": "inflight-load-producer"},
                {
                    # peakPrefillThroughput: log-derived from results/dsv4-pro's
                    # -v1 prefill_0.log (Reqs Running:1, Deferred:0 solo/unchunked
                    # windows) -- real sustained per-rank peak is ~57-75k tok/s;
                    # matches the dp_enabled path's already-calibrated 80000 below.
                    # maxTTFTPenaltyMs pinned at the llm-d-router default (5000ms)
                    # rather than left implicit; NOT yet validated at c160-256 --
                    # a static ms budget doesn't scale with concurrency, so this
                    # gate could under-fire (pileup, low peakPrefillThroughput) or
                    # over-fire (loses affinity right when we need it) at that
                    # range. Re-check local_hit_pct + TTFT tails together on the
                    # next c160-256 sweep before adjusting further.
                    "type": "prefix-cache-affinity-filter",
                    "parameters": {"peakPrefillThroughput": 80000, "maxTTFTPenaltyMs": 5000},
                },
                {"type": "active-request-scorer"},
                # queueThresholdTokens: anchored to this deployment's own per-rank
                # GPU KV cache capacity (logged: 1,992,761 tokens for EP8 prefill
                # DP ranks on DSV4-Pro/GB200). The prior 750000 was ~38% of that
                # capacity, so InFlightLoad+UncachedRequestTokens saturated the
                # score to 0 for ~all endpoints almost immediately (confirmed via
                # EPP logs on the c160 v6 sweep: 99% of TokenLoadScorer decisions
                # were tokenLoad clamped at 750000 -> score 0), destroying the
                # scorer's discriminating power and leaving prefill routing close
                # to a random pick among whatever survived the affinity filter.
                # 2000000 puts our typical 8-12 req/rank @ ~80k p50 ISL operating
                # point (640k-960k tokens) in the middle of the score range
                # (~0.52-0.68) instead of clamped at the ceiling, while still
                # saturating to 0 once backlog genuinely exceeds what a rank can
                # physically hold resident. Re-check the tokenLoad/score
                # distribution in EPP logs on the next sweep before adjusting
                # further.
                {"type": "token-load-scorer", "parameters": {"queueThresholdTokens": 2000000}},
                {"type": "always-disagg-pd-decider"},
                {"type": "disagg-profile-handler", "parameters": {"deciders": {"prefill": "always-disagg-pd-decider"}}},
                {"type": "weighted-random-picker", "name": "prefill-picker"},
                {"type": "max-score-picker", "name": "decode-picker"},
            ],
            "schedulingProfiles": [
                {
                    "name": "prefill",
                    "plugins": [
                        {"pluginRef": "prefill-filter"},
                        {"pluginRef": "prefix-cache-affinity-filter"},
                        {"pluginRef": "token-load-scorer"},
                        {"pluginRef": "prefill-picker"},
                    ],
                },
                {
                    "name": "decode",
                    "plugins": [
                        {"pluginRef": "decode-filter"},
                        {"pluginRef": "active-request-scorer", "weight": 2},
                        {"pluginRef": "decode-picker"},
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
            "data": {"plugins.yaml": _plugin_config(spec.routing, dp_enabled=role.parallelism.dp_enabled)},
        },
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
                                    f"--pool-name={infpool_name}",
                                    f"--pool-namespace={spec.namespace}",
                                ],
                                "ports": [{"containerPort": 9002, "name": "grpc"}],
                                "volumeMounts": [
                                    {"name": "config", "mountPath": "/etc/epp/plugins.yaml", "subPath": "plugins.yaml"}
                                ],
                                "resources": {
                                    "requests": {"cpu": "8", "memory": "16Gi"},
                                    "limits": {"cpu": "8", "memory": "16Gi"},
                                },
                            }
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
