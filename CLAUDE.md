# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
this repository.

## Preferences

- Use `podman` instead of `docker` for container builds.
- Prefer `just` commands when available. Do not run `kubectl apply` directly if
  a Justfile recipe exists for the workflow, because recipes handle user
  namespacing and render-time defaults.
- Use the renderer-first commands: `just start SPEC *ARGS`, `just restart SPEC
  *ARGS`, `just stop SPEC [NOW]`, and `just ready SPEC`.

## Repository Overview

Manifesto renders shareable Kubernetes manifests for llm-d/vLLM deployments.
The current target is GB200 NVL72-style clusters with large MoE deployments,
including aggregated and prefill/decode topologies.

The repository contains:

- `manifesto/` - Python renderer implementation and CLI.
- `models/` - Model deployment specs.
- `clusters/` - Cluster profiles.
- `dev/` - Dev image Containerfile; the persistent dev pod for building vLLM
  from source is rendered by `manifesto render-dev-pod`.
- `monitoring/` - Namespace-scoped Prometheus and Grafana stack.
- `tests/` - Renderer, validation, and UX regression tests.

## Architecture

Manifesto takes a model spec and a cluster profile, then emits raw Kubernetes
objects:

- Deployment or LeaderWorkerSet model-server workloads, depending on node count.
- InferencePool and endpoint picker deployment.
- Gateway API and HTTPRoute objects.
- Per-pod monitoring sidecars.
- Instance-scoped names, labels, selectors, and cache paths.

Key components:

- **vLLM** - Model server and inference engine.
- **Inference Gateway** - Request scheduler and balancer through Gateway API
  InferencePool.
- **Kubernetes** - Infrastructure orchestrator and workload control plane.
- **LeaderWorkerSet** - Multi-host inference coordination.
- **NIXL** - Fast interconnect library for KV cache transfer.

## Common Commands

The Justfile requires a `.env` file with:

- `HF_TOKEN` - HuggingFace token for model access.
- `GH_TOKEN` - GitHub token.
- `KUBECONFIG` - Path to kubeconfig.
- `MANIFESTO_CLUSTER` or `MANIFESTO_CLUSTER_MAP` - Explicit renderer cluster
  profile, or local kube context/cluster to profile mapping.
- `MANIFESTO_NAMESPACE` - Optional namespace override; defaults to the current
  kube context namespace or `default`.

Renderer workflow:

```bash
just render models/qwen/aggregated.yaml
just render-file models/deepseek-v4/1P-EP8-1D-EP8.yaml --dev
just diff-file
just apply-file
just start models/deepseek-v4/1P-EP8-1D-EP8.yaml --dev
just ready models/deepseek-v4/1P-EP8-1D-EP8.yaml
just restart models/deepseek-v4/1P-EP8-1D-EP8.yaml --dev
just stop models/deepseek-v4/1P-EP8-1D-EP8.yaml
just logs models/deepseek-v4/1P-EP8-1D-EP8.yaml decode
just logs models/deepseek-v4/1P-EP8-1D-EP8.yaml decode -f
```

Monitoring and diagnostics:

```bash
just get-decode-pods
just print-gpus
just cks-nodes
just start-monitoring
just grafana
just prometheus
just load-dashboards
just stop-monitoring
```

Dev vLLM workflow:

```bash
just dev-start
just dev
just dev-build
just dev-build-log
just dev-stop
just flush-cache models/deepseek-v4/1P-EP8-1D-EP8.yaml --dev
```

Nyann benchmark workflow:

```bash
just nyann
just nyann-stairs
just nyann-logs load
just nyann-eval-gsm8k
just nyann-eval-gpqa
just nyann-eval
just nyann-prep-gpqa
just nyann-stop
```

## Key Configuration Files

- `pyproject.toml` - Python package metadata and test configuration.
- `Justfile` - Local automation, deployment, dev, monitoring, and benchmark
  commands.
- `config/images.yaml` - Central image catalog for model, llm-d, sidecar, and
  dev images.
- `clusters/oci-gb200.yaml` - GB200 cluster profile.
- `clusters/cks-h200.yaml` - CoreWeave H200 cluster profile.
- `models/qwen/aggregated.yaml` - Aggregated Qwen example.
- `models/deepseek-v4/wide-ep-base.yaml` - Shared DeepSeek V4 wide-EP base.
- `models/deepseek-v4/1P-EP8-1D-EP8.yaml` - DeepSeek V4 P/D EP8 example.
- `models/deepseek-v4/3P-EP8-1D-EP16.yaml` - DeepSeek V4 wider decode example.
- `monitoring/` - Prometheus/Grafana Helm values and dashboards.

## Development Workflow

1. Render a manifest with `just render` or `just render-file`.
2. Inspect or edit the generated YAML.
3. Apply with `just apply-file` or deploy directly with `just start`.
4. Wait with `just ready`.
5. Use `just dev-start` and `just dev-build` for vLLM source iteration.
6. Use `just restart SPEC --dev` after changing runtime code or render inputs.

## Important Notes

- Rendered objects are scoped by `{user}-{release}` so multiple users can share
  a namespace.
- Decode pods may expose multiple vLLM ports when data parallel fanout is
  enabled.
- vLLM API servers can take several minutes to start for large MoE models.
- Decode pod information is cached in `.tmp/decode_pods.txt` by
  `just get-decode-pods`.

## Just Variable Expansion Notes

- Just strips outer quotes during expansion. If you define `VAR := "value with
  spaces"`, then `{{VAR}}` expands to `value with spaces`.
- Add quotes in bash assignments. Use `BASH_VAR="{{JUST_VAR}}"` to preserve
  values with spaces.
