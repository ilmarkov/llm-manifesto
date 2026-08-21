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
                    # Required explicitly: token-load-scorer consumes both
                    # InFlightLoadDataKey (auto-injectable -- registered as
                    # the default producer for that key) and
                    # UncachedRequestTokensDataKey (NOT auto-injectable --
                    # no default producer is registered for it anywhere in
                    # llm-d-router). Without this declaration EPP fails at
                    # startup: "failed to create missing data producers - no
                    # default producer found for missing data key:
                    # UncachedRequestTokensDataKey/inflight-load-producer,
                    # which is consumed by: token-load-scorer" (confirmed via
                    # live CrashLoopBackOff on the v9 rollout). No name or
                    # parameters needed: the default (unnamed) instance
                    # produces both keys under the plugin's type name
                    # "inflight-load-producer", which is exactly what
                    # token-load-scorer looks up when its own
                    # inFlightLoadProducerName is left unset.
                    "type": "inflight-load-producer",
                },
                {
                    # autoTune disabled: it reads real per-pod capacity from vLLM's
                    # vllm:cache_config_info metric, but that metric is never emitted
                    # by our vLLM build (checked live: 54 metric families on
                    # /metrics, none matching cache_config_info/num_gpu_blocks) --
                    # autoTune was silently falling back to the plugin's static
                    # default (31250 blocks * our 256 blockSizeTokens = 8M tokens)
                    # regardless. Even if the metric worked, autoTune only ever
                    # reads num_gpu_blocks -- it has no concept of CPU-offloaded
                    # capacity, so it would still only model the GPU tier.
                    #
                    # lruCapacityPerServer derived from live prefill logs
                    # (ilmarkov-vllm-ep8-prefill-dspark-0, gpu-memory-utilization
                    # 0.97, --block-size 256):
                    #   GPU KV cache:  3,089,495 tokens/rank (kv_cache_utils.py log)
                    #   CPU offload:  34,952 blocks x 256 tok/block = 8,947,712
                    #     tokens/rank (SimpleCPUOffloadConnector, cpu_bytes_to_use=
                    #     42949672960 = 40 GiB/rank; unchanged by gpu-memory-utilization)
                    #   combined:     12,037,207 tokens/rank -> /256 ~= 47,020 blocks
                    # Mooncake (also configured, enable_offload=false) excluded --
                    # it's an external/read-only tier for this pod, not local
                    # retention. Decode has no CPU-offload connector at all, but
                    # the decode profile doesn't consume prefix-cache-affinity-filter
                    # /prefix-cache-scorer, so this value only matters for prefill.
                    # Re-derive if gpu-memory-utilization, cpu_bytes_to_use, or
                    # --block-size change.
                    "type": "approx-prefix-cache-producer",
                    "parameters": {
                        "autoTune": False,
                        "blockSizeTokens": 256,
                        "maxPrefixTokensToMatch": 1048576,
                        "maxPrefixBlocksToMatch": 4096,
                        "lruCapacityPerServer": 47020,
                    },
                },
                {
                    #
                    # peakPrefillThroughput re-calibrated with
                    # llm-d's official recipe (guides/recipes/router/
                    # calibration/calibrate.sh) run against this exact live
                    # deployment: sequential 8192-token (= our prefill
                    # max_num_batched_tokens) random-token-ID requests via the
                    # real gateway path, guaranteed cache miss, median TTFT
                    # 1.7124s -> 4783 tok/s. The old 200000 (and v6's 80000)
                    # were both derived from active_prefill_throughput, an
                    # *aggregate concurrent-batched* metric -- a fundamentally
                    # different regime from this plugin's intended semantics
                    # (single in-flight request draining a backlog). The
                    # calibrated value is ~15-40x lower than either prior
                    # guess. Re-run calibrate.sh if max_num_batched_tokens or
                    # hardware changes.
                    #
                    # maxTTFTPenaltyMs re-derived from live gate
                    # telemetry: ran a 900s c192 benchmark against this exact
                    # deployment with EPP at -v=4 and captured every
                    # PrefixCacheAffinityFilter decision via continuous
                    # `kubectl logs -f` (retroactive `kubectl logs --since`
                    # doesn't work here -- at -v=4 the pod emits ~270
                    # lines/sec and container log rotation evicts anything
                    # older than a few minutes). Across 11,767 prefill
                    # scheduling decisions: 15.7% had no sticky candidate,
                    # 24.2% held stickiness, and 60.1% broke it via the TTFT
                    # load gate at the plugin's default 5000ms -- i.e. cache
                    # affinity was actually honored only ~1 in 4 times.
                    # Penalty-when-broken distribution: median 17.4s over
                    # budget, p75 26.1s, p90 36.0s, p99 202s (max 527s).
                    # Recomputed breaking rate at higher thresholds: 15000ms
                    # -> 42.1%, 25000ms -> 19.5%, 30000ms -> 12.6%, 40000ms
                    # -> 5.0%, 60000ms -> 1.9%. Picked 30000ms to keep the
                    # gate as a safety valve for genuinely severe (p75+)
                    # imbalances while letting affinity hold for the routine
                    # variance that was previously discarding it most of the
                    # time. When gate held, avg sticky candidates was 1.009
                    # of 8 -- each conversation really does have one clear
                    # home rank, so honoring stickiness is high-value here.
                    # Re-capture live telemetry (same method) if
                    # peakPrefillThroughput or the affinity/load-scoring mix
                    # changes materially.
                    "type": "prefix-cache-affinity-filter",
                    "parameters": {"peakPrefillThroughput": 4783, "maxTTFTPenaltyMs": 30000},
                },
                {
                    # Re-added 2026-08-21 after v9 regressed hard vs v7
                    # (c160: TTFT p50 31.4s vs v7's 5.9s, throughput 4530
                    # vs 6902 tok/s -- despite v9 having *more* prefill KV
                    # cache than v7, ruling out capacity as the cause).
                    # Root cause: removing this scorer left every decision
                    # outside strict stickiness with zero cache-awareness.
                    # Live-captured EPP logs (default verbosity -- token-
                    # load-scorer already logs one line per candidate at
                    # info level, no -v=4 needed) over 123 real prefill
                    # decisions from the live c192 run: only 26.8% narrowed
                    # to 1 sticky candidate; 73.2% saw 2-8 candidates, incl.
                    # 30.9% seeing all 8 (full fallback to token-load-scorer
                    # alone). Token-load spread across candidates in those
                    # non-narrowed decisions: median 213k, p90 625k tokens --
                    # both above the current gate's break threshold
                    # (maxTTFTPenaltyMs 30000 * peakPrefillThroughput 4783 /
                    # 1000 = ~143k tokens), confirming the gate is firing
                    # well inside ambient load variance, not just on extreme
                    # imbalances. So most decisions were being made by
                    # token-load-scorer alone, with no partial-cache-match
                    # tiebreak, actively scattering conversation continuity
                    # across ranks. Weight 6 (vs token-load-scorer's 2) is
                    # the last live-validated ratio from before the removal
                    # -- high enough to dominate when candidates differ in
                    # cache match, but not so high it re-fights the load
                    # scorer's signal outright. NOT re-touching
                    # maxTTFTPenaltyMs/peakPrefillThroughput in this change;
                    # revisit those separately if this alone doesn't recover
                    # v7-level local hit rates.
                    "type": "prefix-cache-scorer",
                },
                {
                    # queueThresholdTokens re-derived from the same live
                    # prefill logs used for lruCapacityPerServer above
                    # (gpu-memory-utilization 0.97): GPU KV cache 3,089,495
                    # tokens/rank. Rounded to 3,000,000 tokens as the "fully
                    # loaded" normalization point for scoring. NOT yet
                    # live-calibrated against real gate/queue telemetry the
                    # way maxTTFTPenaltyMs was -- worth re-checking with the
                    # same kubectl-logs-capture method if scores look
                    # degenerate (e.g. many ties at score 0 the way the old
                    # 750000 queueThresholdTokens did before).
                    "type": "token-load-scorer",
                    "parameters": {"queueThresholdTokens": 3000000},
                },
                {"type": "active-request-scorer"},
                {"type": "always-disagg-pd-decider"},
                {"type": "disagg-profile-handler", "parameters": {"deciders": {"prefill": "always-disagg-pd-decider"}}},
                # max-score on both: deterministic, always honors the best
                # cache-match/load score instead of diluting it with randomness.
                # Decode has no cache locality to protect, but live per-rank
                # metrics (queue depth ~0.1, KV util 31-36%, running requests
                # within +-6% across all 8 ranks) show it's already flat under
                # weighted-random and isn't the bottleneck, so the theoretical
                # hot-spotting risk of greedy picking isn't materializing here
                # (single EPP replica, only 8 decode endpoints, scorer signals
                # already well-behaved) -- switched to max-score for decode too.
                {"type": "max-score-picker", "name": "prefill-picker"},
                {"type": "max-score-picker", "name": "decode-picker"},
            ],
            "schedulingProfiles": [
                {
                    "name": "prefill",
                    "plugins": [
                        {"pluginRef": "prefill-filter"},
                        {"pluginRef": "prefix-cache-affinity-filter"},
                        {"pluginRef": "prefix-cache-scorer", "weight": 6},
                        {"pluginRef": "token-load-scorer", "weight": 2},
                        {"pluginRef": "prefill-picker"},
                    ],
                },
                {
                    "name": "decode",
                    "plugins": [
                        {"pluginRef": "decode-filter"},
                        {"pluginRef": "active-request-scorer"},
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
