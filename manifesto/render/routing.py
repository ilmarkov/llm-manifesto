"""Gateway API, InferencePool, and EPP manifests for instance-scoped routing."""

from __future__ import annotations

import yaml

from .common import secret_env
from ..instance import Instance
from ..cluster import Cluster
from ..parallelism import parallel_layout
from ..resolve import resolve_role
from ..spec import DeploymentSpec, RoutingKind, RoutingSpec


def _plugin_config(routing: RoutingSpec, *, dp_enabled: bool = False, model_name: str | None = None) -> str:
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
                    # Real tokenizer, required by precise-prefix-cache-producer
                    # (the estimate backend's byte-packed IDs don't correlate
                    # with vLLM's own KV-block hashes). Calls a CPU-only
                    # `vllm launch render` sidecar in this same EPP pod (see
                    # render_routing) rather than a serving prefill/decode
                    # pod, so tokenization never competes with GPU work.
                    "type": "token-producer",
                    "parameters": {
                        "modelName": model_name,
                        "vllm": {"url": "http://localhost:8000"},
                    },
                },
                {
                    # Feeds precise-prefix-cache-producer's per-pod ZMQ
                    # subscriber lifecycle (connect on pod add, disconnect on
                    # pod delete).
                    "type": "endpoint-notification-source",
                },
                # Removed 2026-08-21 (prefill-queue-limiter / utilization-
                # filter, cap 40): live v12 c192 data showed the queue-
                # balancing changes above (queue-scorer, higher
                # queueThresholdTokens) succeeded in eliminating the old
                # single-rank pileup -- but pushed *typical* per-rank
                # waiting-queue depth up to 24-27 avg across all 8 ranks
                # (vs v10's 7-12 on 7 healthy ranks), so this filter's cap=40
                # was now being crossed 17-22% of the time on *every* rank
                # instead of just the one pathological rank it was designed
                # for. Placed before prefix-cache-affinity-filter (by design,
                # to exclude overloaded ranks before stickiness locks in), it
                # was routinely excluding the cache-affine rank itself,
                # breaking locality broadly: prefill local hit rate collapsed
                # 77.8% (v10 c192) -> 19.2% (v12 c192), true recompute nearly
                # doubled (6.9% -> 14.6%), TTFT avg roughly tripled (22.5s ->
                # 66.4s), throughput dropped 42% (7147 -> 4142 tok/s) -- all
                # while ITL stayed flat, confirming decode/capacity weren't
                # the cause. Re-add only with a cap re-derived from the
                # *current* per-rank operating range (not v10's stale
                # baseline) if single-rank pileups reappear.
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
                    # live CrashLoopBackOff on the v9 rollout). No name
                    # needed: the default (unnamed) instance produces both
                    # keys under the plugin's type name
                    # "inflight-load-producer", which is exactly what
                    # token-load-scorer looks up when its own
                    # inFlightLoadProducerName is left unset.
                    "type": "inflight-load-producer",
                    "parameters": {
                        # prefixMatchInfoProducerName (2026-08-22): this
                        # producer's own Consumes() treats PrefixCacheMatchInfo
                        # as *optional*, defaulting to the approximate-prefix
                        # producer's key when unset (llm-d-router
                        # pkg/.../dataproducer/inflightload/producer.go). Now
                        # that approx-prefix-cache-producer is gone, leaving
                        # this unset would silently resolve to no data (no
                        # startup error, since it's optional) and always
                        # apply a zero cached-prefix discount -- so
                        # UncachedRequestTokensDataKey, and therefore
                        # token-load-scorer's tokenLoad, would never benefit
                        # from the precise producer at all. Matches
                        # precise-routing.values.yaml in the llm-d wide-ep-lws
                        # guide. Verified this field and behavior exist as
                        # described at our pinned router tag v0.10.0, not just
                        # on a newer commit.
                        "prefixMatchInfoProducerName": "precise-prefix-cache-producer",
                    },
                },
                {
                    # Replaces approx-prefix-cache-producer (2026-08-22): that
                    # producer only ever *estimated* per-pod retention via a
                    # static lruCapacityPerServer we had to hand-derive and
                    # re-derive from log snapshots every time gpu-memory-
                    # utilization/cpu_bytes_to_use changed (last value: 45272
                    # blocks, see git history). This producer instead
                    # subscribes to vLLM's real KV-block store/remove event
                    # stream (--kv-events-config, enabled on the prefill role
                    # only -- see ix-disagg-base.yaml) and builds an exact
                    # per-pod-per-tier index, so no capacity guessing is
                    # needed at all. Requires our EPP image >= v0.10.0
                    # (llm-d-router#2233: without it, KV-index identity is
                    # derived from the ZMQ topic string, which collapses all
                    # local DP ranks of one pod into a single index entry --
                    # exactly our topology, 4 local ranks/pod). blockSizeTokens
                    # matches --block-size 256 on the engine (required for the
                    # EPP's independently-recomputed hashes to align with the
                    # KV-event block boundaries). kvCacheBackendConfigs gives
                    # combined GPU+CPU-offload visibility -- confirmed
                    # SimpleCPUOffloadConnector's block pool emits the same
                    # KVCacheEvents gated by the same enable_kv_cache_events
                    # flag (vllm/v1/simple_kv_offload/manager.py), so the CPU
                    # tier is precisely tracked too, not just guessed at.
                    # Mooncake stays untracked here deliberately: it's a
                    # cluster-shared distributed store reachable from any
                    # rank regardless of routing, so it doesn't need (or
                    # benefit from) per-pod cache-location precision the way
                    # the two local-only tiers do.
                    "type": "precise-prefix-cache-producer",
                    "parameters": {
                        "tokenProcessorConfig": {"blockSizeTokens": 256},
                        "indexerConfig": {
                            "kvBlockIndexConfig": {"inMemoryConfig": {"podCacheSize": 128}},
                            "kvCacheBackendConfigs": [
                                {"name": "gpu", "weight": 1.0},
                                {"name": "cpu", "weight": 0.4},
                            ],
                        },
                        "kvEventsConfig": {
                            "topicFilter": "kv@",
                            "discoverPods": True,
                            "podDiscoveryConfig": {"socketPort": 5557},
                        },
                    },
                },
                # Removed 2026-08-22 (prefix-cache-affinity-filter): this
                # hard elimination gate had a structural flaw exposed by its
                # own calibration telemetry (see prior comment history, now
                # superseded) -- at its 0.80 default affinityThreshold, once
                # a sticky candidate existed it narrowed to *only* that
                # endpoint (avg 1.009 of 8) with zero regard for its current
                # queue/load, since narrowing to a single candidate leaves
                # nothing for token-load-scorer/queue-scorer to score. The
                # only release valve was the filter's own coarse, binary
                # TTFT-estimate gate (peakPrefillThroughput/maxTTFTPenaltyMs),
                # which per that same telemetry didn't trip until a severe
                # (p75+, ~17s+ over budget) imbalance -- a likely structural
                # cause of the persistent per-rank waiting-queue skew, since
                # a cache-affine-but-currently-busy rank got 100% of matching
                # traffic until things got extreme. prefix-cache-scorer below
                # reads PrefixCacheMatchInfo directly from
                # precise-prefix-cache-producer (Consumes() has no coupling
                # to this filter having run) and gives a continuous
                # match-ratio reward on every decision, blended via weights
                # with token-load-scorer/queue-scorer -- graceful, graduated
                # trade-offs instead of a bimodal fully-sticky-vs-full-
                # fallback switch. Note the 6:3:3 weight ratio below was
                # calibrated for the fallback-only regime this filter used to
                # gate into (~15-30% of decisions); it now governs 100% of
                # decisions, so re-observe/retune if live behavior warrants.
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
                    #
                    # prefixMatchInfoProducerName added 2026-08-22 to point at
                    # precise-prefix-cache-producer instead of the removed
                    # approx producer -- same weight/rationale above still
                    # applies unchanged.
                    "type": "prefix-cache-scorer",
                    "parameters": {"prefixMatchInfoProducerName": "precise-prefix-cache-producer"},
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
                {
                    # Added 2026-08-21 to prefill after diagnosing a persistent
                    # 2-rank pileup (v10 c192: ranks 1/4 stuck at 40-90 waiting
                    # while the other 6 drained to ~0). Root cause: prefix-cache-
                    # scorer's MatchBlocks/TotalBlocks score is continuous and not
                    # gated by the filter's 0.80 affinityThreshold, so even before
                    # any endpoint is "sticky" it creates a rich-get-richer pull
                    # toward whichever rank first captures a slice of a shared
                    # prefix (this workload is agentic subagent trees with a
                    # common root-context prefix across many branches). v8 had
                    # the same mechanism but kept it smaller-scale (worst rank
                    # ~55 vs v10's ~93) because its combined load-scorer weight
                    # (kv-util 2 + active-request 2 + queue 2 = 6) roughly matched
                    # its cache weight (10), a 1.67:1 ratio -- vs v10's 3:1
                    # (prefix-cache-scorer 6 : token-load-scorer 2 alone) before
                    # this change. queue-scorer specifically restores a signal
                    # sourced directly from vLLM's own num_requests_waiting,
                    # immune to the inflight-load-producer 5-minute PluginState
                    # staleness reaping that silently zeroes out token-load-
                    # scorer's view of long-queued (not-yet-first-token)
                    # requests -- exactly the case on the pathological ranks.
                    #
                    # Note: this weight only matters in the "fallback" case
                    # where prefix-cache-affinity-filter's own 0.80
                    # affinityThreshold gate found no fully-sticky candidate
                    # and handed all 8 endpoints to the scorers -- exactly the
                    # partial-match regime (this workload's subagent trees
                    # share large root-context prefixes well before any one
                    # branch crosses 80%) where the rich-get-richer
                    # concentration originates. When the filter does find a
                    # sticky candidate, it narrows to ~1 endpoint before
                    # scorers run, so these weights don't affect that (already
                    # working well) path at all.
                    #
                    # Weight raised to 3 (both token-load-scorer and
                    # queue-scorer) after weight=2 was deployed and diagnosed
                    # live on the v10 c192 run: cache:load ratio of 6:4 (1.5:1)
                    # was still cache-leaning vs v8's 1.67:1, but v8 *itself*
                    # showed the same pathology at smaller scale (worst rank
                    # ~55 vs v10's ~93 before this scorer was even added) --
                    # so matching v8's ratio alone wasn't expected to be
                    # enough. Went to 3+3=6, a 6:6 (1:1) ratio, giving load-
                    # balancing equal say against cache-affinity in the
                    # contested fallback regime. Trade-off: some legitimate
                    # partial-prefix reuse across subagent branches will now
                    # get scattered instead of consolidated, but the observed
                    # downside (severe per-rank queue skew) outweighed that.
                    "type": "queue-scorer",
                },
                {"type": "always-disagg-pd-decider"},
                {"type": "disagg-profile-handler", "parameters": {"deciders": {"prefill": "always-disagg-pd-decider"}}},
                {"type": "max-score-picker", "name": "prefill-picker"},
                {"type": "max-score-picker", "name": "decode-picker"},
            ],
            "dataLayer": {
                "sources": [
                    {
                        "pluginRef": "endpoint-notification-source",
                        "extractors": [{"pluginRef": "precise-prefix-cache-producer"}],
                    }
                ]
            },
            "schedulingProfiles": [
                {
                    "name": "prefill",
                    "plugins": [
                        {"pluginRef": "prefill-filter"},
                        {"pluginRef": "prefix-cache-scorer", "weight": 6},
                        {"pluginRef": "token-load-scorer", "weight": 3},
                        {"pluginRef": "queue-scorer", "weight": 3},
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
            "data": {
                "plugins.yaml": _plugin_config(
                    spec.routing,
                    dp_enabled=role.parallelism.dp_enabled,
                    model_name=spec.model.served_name or spec.model.id,
                )
            },
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
                            },
                            *(
                                [
                                    {
                                        # CPU-only tokenizer sidecar for
                                        # token-producer's vllm backend, so
                                        # precise-prefix-cache-producer gets
                                        # real token IDs without adding load
                                        # to GPU-serving pods. This EPP
                                        # Deployment is pinned to amd64 nodes
                                        # (see nodeAffinity above), separate
                                        # from the arm64 GPU nodepool, so it
                                        # can't use spec.model.image if that's
                                        # an arm64-only custom build. Doesn't
                                        # need our patches though: `--tokenizer
                                        # -mode` and `vllm launch render` are
                                        # both plain upstream vLLM features,
                                        # so any reasonably current public
                                        # image produces the same token IDs
                                        # (and thus the same KV-block hashes)
                                        # as the real engines, as long as our
                                        # branch hasn't locally patched
                                        # tokenizer/renderer/parser code for
                                        # this model (verified true as of
                                        # 2026-08-22). Must be the dedicated
                                        # `-cpu` build, not the CUDA-targeted
                                        # vllm-openai image: vLLM's platform
                                        # auto-detection only activates
                                        # CpuPlatform if the installed
                                        # package's version string is tagged
                                        # `+cpu` (or on macOS) -- on a plain
                                        # CUDA build with no GPU present, it
                                        # resolves to UnspecifiedPlatform and
                                        # crashes with "Failed to infer
                                        # device type" instead of falling
                                        # back to CPU.
                                        "name": "vllm-render",
                                        "image": spec.routing.render_image or "vllm/vllm-openai-cpu:latest",
                                        "imagePullPolicy": "Always",
                                        "command": ["vllm", "launch", "render"],
                                        "args": [
                                            spec.model.id,
                                            "--port=8000",
                                            *(["--trust-remote-code"] if role.vllm_args.get("trust_remote_code") else []),
                                            *(
                                                [f"--tokenizer-mode={role.vllm_args['tokenizer_mode']}"]
                                                if role.vllm_args.get("tokenizer_mode")
                                                else []
                                            ),
                                        ],
                                        "env": [secret_env("HF_TOKEN", "hf-secret", "HF_TOKEN")],
                                        "ports": [{"containerPort": 8000, "name": "render"}],
                                        "readinessProbe": {
                                            "httpGet": {"path": "/health", "port": 8000},
                                            "periodSeconds": 5,
                                        },
                                        "resources": {
                                            "requests": {"cpu": "4", "memory": "16Gi"},
                                            "limits": {"cpu": "4", "memory": "16Gi"},
                                        },
                                    }
                                ]
                                if spec.routing.kind == RoutingKind.PD
                                else []
                            ),
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
