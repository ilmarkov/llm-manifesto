# DeepSeek-V4 config diff: `llm-manifesto` vs InferenceX PR #2260

**Local:** `llm-manifesto/models/deepseek-v4/`  
**Reference:** [SemiAnalysisAI/InferenceX#2260](https://github.com/SemiAnalysisAI/InferenceX/pull/2260) — *[AgentX] vLLM DeepSeek-V4 GB200 disagg* (branch `agentx/dsv4-gb200-pd`, head `a7b253a`)

PR #2260 adds GB200 AgentX srt-slurm recipes under `benchmarks/multi_node/srt-slurm-recipes/vllm/deepseek-v4/agentic/` plus master-config entries in `configs/nvidia-master.yaml`. Our local configs are Kubernetes/LWS manifests in the llm-manifesto schema. This document compares the **model serving topology and vLLM tuning**, not the benchmark harness or Slurm/Dynamo plumbing.

---

## Topology mapping

Concurrencies come from PR `configs/nvidia-master.yaml` → `scenarios.agentic-coding[].conc-list` (scenario: **agentic-coding**, spec-decoding: **none**). Local manifests do not define benchmark conc lists; `wide-ep-base.yaml` sets a serving ceiling via `vars.max_concurrency: 1024`.

| Local config | Local nodes / GPUs | PR recipe | PR master-config key | PR conc-list | PR nodes / GPUs | Match quality |
|---|---:|---|---|---|---:|---|
| *(none)* | — | `agg-gb200-tp8-agentic.yaml` | `dsv4-fp4-gb200-dynamo-vllm-agentic-agg` | 1, 4, 8, 16 | 2 / 8 (TP8 aggregate) | **PR only** — no local aggregate recipe |
| `1P-EP8-1D-EP8.yaml` | 4 / 16 | `disagg-gb200-1p1d-dep8-dep8-agentic.yaml` | `dsv4-fp4-gb200-dynamo-vllm-agentic-1p1d-dep8-dep8` | 64, 128, 192, 256 | 4 / 16 | **Strong** — same P/D worker counts and DEP8/DEP8 |
| *(none)* | — | `disagg-gb200-1p1d-dep8-dep12-agentic.yaml` | `dsv4-fp4-gb200-dynamo-vllm-agentic-1p1d-dep8-dep12` | 384, 512 | 5 / 20 | **PR only** — DEP12 decode |
| `2P-EP8-1D-EP8.yaml` | 6 / 24 | `disagg-gb200-2p1d-dep8-dep12-agentic.yaml` | `dsv4-fp4-gb200-dynamo-vllm-agentic-2p1d-dep8-dep12` | 640, 720, 768 | 7 / 28 | **Partial** — same 2P prefill count; decode EP differs (EP8 vs EP12) and GPU count differs |
| `3P-EP8-1D-EP8.yaml` | 8 / 32 | *(no PR recipe)* | — | — | — | **Local only** |
| `3P-EP8-1D-EP16.yaml` | 10 / 40 | `disagg-gb200-3p1d-dep8-dep16-agentic.yaml` | `dsv4-fp4-gb200-dynamo-vllm-agentic-3p1d-dep8-dep16` | 800, 960, 1024, 1280 | 10 / 40 | **Strong** — same 3P EP8 + 1D EP16 layout |

**Summary:** PR sweeps low concurrency on aggregate TP8, mid/high on smaller disagg topologies, and up to 1280 on the largest 3P/1D EP16 layout. We have two local topologies (2P/1D EP8/EP8 and 3P/1D EP8/EP8) with no PR counterpart.

### Scheduler limits vs benchmark `conc-list` (all PR disagg topologies)

`max-num-batched-tokens` is **8192 on every prefill** and **256 on every decode** in PR #2260. Values below are from the recipe YAMLs at `a7b253a`.

| Topology | `conc-list` | P `max-seqs` | P `max-batched` | D `max-seqs` | D `max-batched` | Caps vs max conc |
|---|---:|---:|---:|---:|---:|---|
| 1P/1D DEP8/DEP8 | 64–256 | 128 | 8192 | 64 | 256 | P seqs OK ≤128; **fails at conc 192, 256**; D seqs fail at conc ≥64 |
| **1P/1D DEP8/DEP12** | **384, 512** | **256** | **8192** | **128** | **256** | **P seqs fail at conc 384, 512**; D seqs fail at conc ≥128 |
| 2P/1D DEP8/DEP12 | 640–768 | 192 | 8192 | 128 | 256 | All caps below min conc |
| 3P/1D DEP8/DEP16 | 800–1280 | 214 | 8192 | 160 | 256 | All caps below min conc |

So for **1P/1D DEP8/DEP12** specifically: our `deepseek-v4-ix` manifests **match the PR** on `max-num-batched-tokens`, but the PR itself pairs high benchmark conc (**384/512**) with fixed batched-token budgets identical to the DEP8/DEP8 tier and `max-num-seqs` below the lowest conc point. That is likely what looks inconsistent — it is in the InferenceX recipe/master-config pairing, not a transcription error in `deepseek-v4-ix/`.

### `perf-changelog.yaml` description vs actual `conc-list`

PR #2260 appends a changelog entry whose `description` still lists **stale** concurrency values. The **`nvidia-master.yaml` `conc-list` fields are authoritative** for what CI actually sweeps.

| Topology | `perf-changelog.yaml` description | Actual `nvidia-master.yaml` `conc-list` | Match? |
|---|---|---|---|
| Agg TP8 | 1, 4, 8, 16 | 1, 4, 8, 16 | ✓ |
| 1P/1D DEP8/DEP8 | 64, 128, **256, 320** | 64, 128, **192, 256** | ✗ |
| 1P/1D DEP8/DEP12 | **128, 256, 320** | **384, 512** | ✗ |
| 2P/1D DEP8/DEP12 | **480**, 640, **768** | 640, **720**, 768 | ✗ |
| 3P/1D DEP8/DEP16 | **640**, 800, 960, 1280 | 800, 960, **1024**, 1280 | ✗ |

Changelog text (PR `perf-changelog.yaml`):

> Add GB200 Dynamo-vLLM AgentX aggregate TP8 at conc [1,4,8,16] and disaggregated topologies: 1P/1D DEP8/DEP8 at [64,128,256,320], 1P/1D DEP8/DEP12 at [128,256,320], 2P/1D DEP8/DEP12 at [480,640,768], and 3P/1D DEP8/DEP16 at [640,800,960,1280].

Likely cause: the description was written from an earlier tuning iteration (commit `453e342` / `a94e310` adjusted conc lists) and not updated when `nvidia-master.yaml` changed. **Treat the topology mapping table above, not the changelog prose, as the source of truth.**

---

## Configs present in one side only

### Only in PR #2260

- **Aggregate TP8** (`agg-gb200-tp8-agentic.yaml`) — single worker spanning 2×4-GPU nodes, `tensor-parallel-size: 8`, no disaggregation.
- **1P/1D DEP8/DEP12** — decode uses `data-parallel-size: 12` on 3 nodes (12 GPUs).
- **2P/1D DEP8/DEP12** — two DEP8 prefill workers, one DEP12 decode worker.
- **Mooncake KV store** — embedded RDMA store with IB NIC pinning (`mlx5_0,mlx5_1,mlx5_3,mlx5_4`).
- **MultiConnector KV path** (disagg) — `NixlConnector` + `MooncakeStoreConnector` in one `kv-transfer-config`.
- **Dynamo frontend** — `router-mode: kv`, `tokenizer: fastokens` (disagg), health/Slurm integration.
- **Agentic benchmark env** — `AIPERF_*`, `WEKA_LOADER_OVERRIDE`, `agentic_srt.sh` command.
- **Master-config conc lists** — see topology mapping table; changelog description is stale (four of five disagg topologies wrong).

### Only in local `llm-manifesto`

- **`2P-EP8-1D-EP8.yaml`** — 2 prefill + 1 decode, both EP8 (6 nodes / 24 GPUs).
- **`3P-EP8-1D-EP16.yaml`** with decode EP8 — 3 prefill + 1 decode EP8 (8 nodes / 32 GPUs).
- **EPLB (expert-parallel load balancing)** on base decode EP16 — full `eplb_config` with NIXL communicator, redundant experts, async rebalance.
- **`all2all_backend: deepep_v2`** on base decode EP16.
- **`dp_load_balancing: external`** on prefill.
- **Runtime sidecars** — `dcgm-exporter`, `node-exporter`.
- **Kubernetes/LWS fields** — `workload_name`, `lws.size/replicas`, `routing`, `release` names.
- **`enable_force_include_usage: true`** on both roles.
- **Env:** `ACTIVATION_V2_GRID_Y`, `DEEPEP_METRICS_ENABLED`, `VLLM_HTTP_TIMEOUT_KEEP_ALIVE`, `VLLM_COMPUTE_NANS_IN_LOGITS`, `VLLM_USE_DEEP_GEMM`.

---

## Cross-cutting differences

| Area | Local (`wide-ep-base` + variants) | PR #2260 (AgentX GB200 recipes) |
|---|---|---|
| **Deployment** | Kubernetes LeaderWorkerSet (llm-manifesto) | Slurm + srt-slurm + Dynamo |
| **Container** | `image_ref: vllm.standard` (abstract) | `vllm/vllm-openai:nightly-dev-arm64-cu13.0.1-c188b96`, `precision: fp4` |
| **Model path** | `deepseek-ai/DeepSeek-V4-Pro` (id) | `deepseek-v4-pro` + canonical Lustre checkpoint for AgentX |
| **Dynamo** | Not configured | Pinned hash `1f74ef8c…`, router `1.3.0.dev20260618` |
| **KV connector** | `NixlConnector` only | Disagg: `MultiConnector` (Nixl + Mooncake); Agg: Mooncake only |
| **DP launch** | Per-GPU LWS pods | `dp_launch_mode: per_node` (one vLLM process per node) |
| **Load balancing** | `dp_load_balancing: external` (prefill) | `data-parallel-hybrid-lb: true` |
| **EP weight filter** | Not set | `enable-ep-weight-filter: true` |
| **NUMA** | Not set | `numa-bind: true` (disagg); commented out on agg (srt-slurm cgroup conflict) |
| **Max context** | Not explicit in base | `max-model-len: 1048576` |
| **Tokenizer** | Not set | `tokenizer-mode: deepseek_v4` |
| **Attention backend** | Only `use_fp4_indexer_cache` | `FLASHINFER_MLA_SPARSE_DSV4` + `use_prefill_query_quantization` |
| **MoE backend** | `deep_gemm_mega_moe` (prefill / EP8 decode) | PR sets `deep_gemm_amxf4_mega_moe` — **not a valid upstream vLLM `--moe-backend` value** (see below) |
| **GPU memory util** | `0.85` both roles | Prefill `0.9`, decode `0.95`; agg sets explicit `kv-cache-memory: 35 GiB` |
| **EPLB** | Enabled on base EP16 decode; disabled in EP8 decode variants | Not used |
| **DeepEP / all2all** | `deepep_v2` on base EP16 decode | Not used |
| **CUDA graphs** | Decode: `custom_ops: [all]`; `VLLM_USE_BREAKABLE_CUDAGRAPH: "1"` | Decode: `compilation-config mode: 0` (no custom_ops); prefill sets `VLLM_USE_BREAKABLE_CUDAGRAPH: "0"` |
| **Stream interval** | `50` (decode) | `10` (agg only) |
| **Uvicorn access log** | `disable_uvicorn_access_log: true` | Removed (commit `e366444`) |
| **Concurrency model** | `vars.max_concurrency: 1024` drives computed `max_num_seqs` / batched tokens / cudagraph size | Fixed per-recipe values (see per-topology table below) |

---

## Per-topology vLLM tuning diffs

### 1P / 1D EP8 / EP8

| Setting | Local | PR (`disagg-gb200-1p1d-dep8-dep8-agentic`) |
|---|---|---|
| Prefill `max-num-seqs` | 1024 (computed) | 128 |
| Prefill `max-num-batched-tokens` | 1024 (computed) | 8192 |
| Decode `max-num-seqs` | 1024 (computed) | 64 |
| Decode `max-num-batched-tokens` | 1024 (computed) | 256 |
| Decode `max-cudagraph-capture-size` | 1024 (computed) | 64 |
| Prefill GPU util | 0.85 | 0.9 |
| Decode GPU util | 0.85 | 0.95 |
| Decode MoE | `deep_gemm_mega_moe` | PR: `deep_gemm_amxf4_mega_moe` (invalid upstream name) |
| KV connector | Nixl only | MultiConnector (Nixl + Mooncake) |

### 1P / 1D EP8 / EP12

| Setting | PR (`disagg-gb200-1p1d-dep8-dep12-agentic`) | `deepseek-v4-ix/1P-EP8-1D-EP12.yaml` | Match? |
|---|---|---|---|
| Benchmark `conc-list` | 384, 512 | *(comment only)* | — |
| Prefill `max-num-seqs` | 256 | 256 | ✓ |
| Prefill `max-num-batched-tokens` | 8192 | 8192 *(from `ix-disagg-base`)* | ✓ |
| Decode `max-num-seqs` | 128 | 128 | ✓ |
| Decode `max-num-batched-tokens` | 256 | 256 *(from `ix-disagg-base`)* | ✓ |
| Decode `max-cudagraph-capture-size` | 128 | 128 | ✓ |

**PR internal tension on this topology:** `conc-list` goes to **512**, but prefill `max-num-seqs` is **256** and decode `max-num-seqs` is **128**. Prefill/decode `max-num-batched-tokens` (**8192 / 256**) are unchanged from the lower-tier **1P/1D DEP8/DEP8** recipe even though concurrency targets roughly doubled — so the scheduler caps do not scale with the benchmark conc sweep.

### 3P / 1D EP8 / EP16

| Setting | Local (base EP16 decode) | PR (`disagg-gb200-3p1d-dep8-dep16-agentic`) |
|---|---|---|
| Prefill replicas | 3 | 3 (`prefill_workers: 3`, 6 nodes) |
| Decode DP | 16 (4 nodes) | 16 (4 nodes) |
| Prefill `max-num-seqs` | 1024 (computed) | 214 |
| Decode `max-num-seqs` | 1024 (computed) | 160 |
| Decode `max-cudagraph-capture-size` | 1024 (computed) | 160 |
| EPLB | Enabled with full config | Not present |
| `all2all_backend` | `deepep_v2` | Not present |
| Decode compilation | `custom_ops: [all]` | `mode: 0` only |

### Aggregate TP8 (PR only)

PR adds a non-disaggregated recipe with no local equivalent:

- `tensor-parallel-size: 8` across 2 nodes (8 GPUs total).
- Single `MooncakeStoreConnector` (`kv_both`), not MultiConnector.
- `max-num-seqs: 32`, `max-num-batched-tokens: 8192`, `max-cudagraph-capture-size: 32`.
- `VLLM_USE_NCCL_SYMM_MEM: "0"` (agg) vs `"1"` (disagg) — NCCL symmetric mem disabled due to CUDA graph capture bug in container.
- `VLLM_USE_RUST_FRONTEND: "1"`, `DYN_REQUEST_PLANE: tcp`, `ETCD_LEASE_TTL: 600`.
- Benchmark env: `IS_MULTINODE: false`, `TP: 8`.

---

## Invalid / questionable PR setting: `moe-backend: deep_gemm_amxf4_mega_moe`

PR #2260 disagg recipes set `moe-backend: "deep_gemm_amxf4_mega_moe"` on both prefill and decode. **This string is not a recognized vLLM MoE backend** in upstream vLLM — there is no `deep_gemm_amxf4_mega_moe` in `vllm.config.kernel.MoEBackend` or the public CLI/docs.

Known upstream DeepSeek-V4 MoE backends include:

| Backend | Typical use |
|---|---|
| `deep_gemm_mega_moe` | Blackwell MegaMoE kernel (FP8 activations × MXFP4 weights) — what our local manifests use |
| `flashinfer_trtllm` | Default fallback; required for NVFP4-quantized expert layouts |
| *(default / auto)* | Used when MegaMoE is incompatible with checkpoint format |

**Implications:**

1. **Our local configs are aligned with upstream vLLM** — `deep_gemm_mega_moe` is the documented Blackwell path for native DeepSeek-V4-Pro (MXFP4 MoE weights).
2. **PR #2260 likely has a typo or fork-only alias** — the name does not appear in vllm-project/vllm. If passed verbatim to stock vLLM, it should fail at argument parsing / backend dispatch.
3. **PR AgentX runs the canonical checkpoint, not NVFP4** — `launch_gb200-nv.sh` points AgentX at `/mnt/lustre01/models/DeepSeek-V4-Pro/` (native artifact). That is exactly the checkpoint type `deep_gemm_mega_moe` targets; `deep_gemm_mega_moe` explicitly **does not** work with NVFP4 ModelOpt layouts ([vllm#43454](https://github.com/vllm-project/vllm/issues/43454)).
4. **PR may rely on a private nightly fork** — container `vllm/vllm-openai:nightly-dev-arm64-cu13.0.1-c188b96` could register a custom backend name, but that is not portable to our manifests unless we pin the same fork.
5. **Related PR env: `VLLM_DSV4_MEGA_FP8_COMBINE=1`** — likely toggles DSV4 MegaMoE FP8-combine behavior inside vLLM; this is separate from `--moe-backend` and does not justify a new backend string.

**For llm-manifesto:** keep `moe_backend: deep_gemm_mega_moe`. Do **not** copy `deep_gemm_amxf4_mega_moe` from PR #2260 unless verified against the exact vLLM build you deploy.

---

## Environment variable diff (high level)

### In PR but not in local manifests

| Variable | Typical PR value | Role / notes |
|---|---|---|
| `ETCD_LEASE_TTL` | `"600"` | Survives long cold-start / weight load |
| `DG_JIT_CACHE_DIR` | `/tmp/dg-cache-dsv4-agentx` | DeepGEMM JIT cache |
| `TORCH_SYMMMEM` | `NVSHMEM` | Symmetric memory backend |
| `VLLM_USE_NCCL_SYMM_MEM` | `"1"` disagg / `"0"` agg | Agg disabled for cudagraph bug |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `"3600"` | Startup timeout |
| `VLLM_RPC_TIMEOUT` | `"600000"` | RPC timeout |
| `VLLM_MOONCAKE_LOAD_RECV_THREADS` | `"4"` agg / `"20"` disagg | Mooncake loader threads |
| `VLLM_CONNECTOR_PREFETCH_*` | depth 8, cap 0.65 | KV prefetch tuning |
| `VLLM_DSV4_MEGA_FP8_COMBINE` | `"1"` | DSV4-specific combine path |
| `VLLM_ALLREDUCE_USE_SYMM_MEM` | `"0"` | Allreduce symmetric mem off |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | `"32768"` | Prefix cache |
| `UCX_*` / `NCCL_*` | various | Network / collective tuning |
| `MC_ENABLE_DEST_DEVICE_AFFINITY` | `"1"` | Mooncake device affinity |
| `AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING` | `"0"` | Workaround for dynamo 400 on session_control |

### In local but not in PR recipes

| Variable | Local value | Role / notes |
|---|---|---|
| `ACTIVATION_V2_GRID_Y` | `"512"` | Decode activation grid |
| `DEEPEP_METRICS_ENABLED` | `"1"` | DeepEP metrics |
| `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` | `"120"` | HTTP keep-alive |
| `VLLM_COMPUTE_NANS_IN_LOGITS` | `"1"` | NaN detection |
| `VLLM_USE_DEEP_GEMM` | `"1"` | DeepGEMM enable (PR uses moe-backend string instead) |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | `"1"` decode | PR prefill sets `"0"`; decode omits |

---

## PR infra changes outside recipe YAML (for context)

These are not represented in llm-manifesto but affect how PR configs run:

1. **`configs/nvidia-master.yaml`** — five new keys: `dsv4-fp4-gb200-dynamo-vllm-agentic-agg`, `-1p1d-dep8-dep8`, `-1p1d-dep8-dep12`, `-2p1d-dep8-dep12`, `-3p1d-dep8-dep16` with conc lists and `CONFIG_FILE` pointers.
2. **`runners/launch_gb200-nv.sh`** — AgentX uses canonical `DeepSeek-V4-Pro/` checkpoint; switches srt-slurm from `cquil11/srt-slurm-nv` fork to `NVIDIA/srt-slurm@v1.0.27` with cherry-picks (per-node DP, health count fix, VLLM_PORT patch).
3. **`benchmarks/benchmark_lib.sh`** — opt-out env `AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING=0`.
4. **Patches** — `srt-slurm-vllm-per-node-health.patch`, `srt-slurm-vllm-port-single-gpu.patch`.

---

## Recommended follow-ups

1. **Decide topology parity** — Add local variants for DEP12 decode (1P/1D and 2P/1D) and/or aggregate TP8 if AgentX parity is required; or document intentional omission.
2. **Do not adopt PR MoE backend string** — PR's `deep_gemm_amxf4_mega_moe` is not upstream vLLM; local `deep_gemm_mega_moe` is correct for native DeepSeek-V4-Pro on Blackwell. Flag PR #2260 for correction or document fork dependency.
3. **KV connector strategy** — PR disagg relies on Mooncake + Nixl MultiConnector; local is Nixl-only. Mooncake store config (RDMA devices, segment sizes) would need a manifest equivalent if adopting PR tuning.
4. **Fix PR changelog text** — `perf-changelog.yaml` description lists wrong concurrencies for 4/5 configs; update to match `nvidia-master.yaml` or drop prose in favor of `config-keys` only.
5. **Concurrency caps** — PR uses much lower per-role `max-num-seqs` than local’s 1024 computed ceiling; reconcile with AgentX session fan-out comments in PR agg recipe (`max-num-seqs: 32`).
6. **EPLB / DeepEP** — Local base EP16 decode enables EPLB + `deepep_v2`; PR drops both. Confirm whether EP16 production should keep EPLB or follow PR’s simpler decode path.

---

*Generated 2026-07-21 from local files in `llm-manifesto/models/deepseek-v4/` and InferenceX PR #2260 at commit `a7b253a`.*
