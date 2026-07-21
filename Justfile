# Lifecycle, renderer, dev, monitoring, and Nyann benchmark commands for manifesto.
set dotenv-load
set dotenv-required

NAMESPACE := env("MANIFESTO_NAMESPACE", `namespace=$(kubectl config view --minify -o jsonpath='{..namespace}' 2>/dev/null || true); printf '%s' "${namespace:-default}"`)
HF_TOKEN := "$HF_TOKEN"
GH_TOKEN := "$GH_TOKEN"

KN := "kubectl -n " + NAMESPACE

NAME_PREFIX := env("USER", "dev")
DEV_POD_NAME := NAME_PREFIX + "-vllm-dev"

MONITORING_DIR := "monitoring"
RENDER_OUT := env("MANIFESTO_RENDER_OUT", "/tmp/" + NAME_PREFIX + "-manifesto.yaml")

default:
  just --list

# === Spec-based renderer (thin wrappers over the manifesto CLI) ===

render SPEC *ARGS='':
  uv run manifesto render {{SPEC}} --namespace {{NAMESPACE}} --user {{NAME_PREFIX}} {{ARGS}}

render-file SPEC *ARGS='':
  uv run manifesto render-file {{SPEC}} --output "{{RENDER_OUT}}" --namespace {{NAMESPACE}} --user {{NAME_PREFIX}} {{ARGS}}

edit-file SPEC *ARGS='':
  uv run manifesto edit-file {{SPEC}} --output "{{RENDER_OUT}}" --namespace {{NAMESPACE}} --user {{NAME_PREFIX}} {{ARGS}}

diff-file FILE=RENDER_OUT:
  uv run manifesto diff --namespace {{NAMESPACE}} --output "{{FILE}}"

apply-file FILE=RENDER_OUT:
  uv run manifesto apply --namespace {{NAMESPACE}} --output "{{FILE}}"

delete-file FILE=RENDER_OUT NOW='false':
  uv run manifesto delete --namespace {{NAMESPACE}} --output "{{FILE}}" {{ if NOW == 'true' { '--now' } else { '' } }}

render-routing SPEC *ARGS='':
  uv run manifesto render-routing {{SPEC}} --namespace {{NAMESPACE}} --user {{NAME_PREFIX}} {{ARGS}}

start SPEC *ARGS='':
  uv run manifesto deploy {{SPEC}} --namespace {{NAMESPACE}} --user {{NAME_PREFIX}} {{ARGS}}

deploy-routing SPEC *ARGS='':
  uv run manifesto deploy-routing {{SPEC}} --namespace {{NAMESPACE}} --user {{NAME_PREFIX}} {{ARGS}}

stop SPEC NOW='false':
  uv run manifesto stop {{SPEC}} --namespace {{NAMESPACE}} --user {{NAME_PREFIX}} {{ if NOW == 'true' { '--now' } else { '' } }}

restart SPEC *ARGS='':
  #!/usr/bin/env bash
  set -euo pipefail
  INSTANCE=$(uv run manifesto instance-id {{SPEC}} --user {{NAME_PREFIX}})
  {{KN}} delete lws -l app.kubernetes.io/instance=$INSTANCE --ignore-not-found=true --grace-period=0 --force &
  {{KN}} delete pod -l app.kubernetes.io/instance=$INSTANCE --ignore-not-found=true --grace-period=0 --force &
  wait
  {{KN}} wait --for=delete pod -l app.kubernetes.io/instance=$INSTANCE --timeout=60s 2>/dev/null || true
  just start {{SPEC}} {{ARGS}}

# Wait for the rendered stack to be ready.
ready SPEC:
  uv run manifesto ready {{SPEC}} --namespace {{NAMESPACE}} --user {{NAME_PREFIX}}

flush-cache SPEC *ARGS='':
  #!/usr/bin/env bash
  set -euo pipefail
  CACHE_PATH=$(uv run manifesto cache-path {{SPEC}} --user {{NAME_PREFIX}} {{ARGS}})
  {{KN}} exec {{DEV_POD_NAME}} -- bash -c "rm -rf '$CACHE_PATH' && echo 'Compile cache flushed: $CACHE_PATH'"

@print-gpus:
  kubectl get pods -A -o json | jq -r ' \
    .items \
    | map(select(.status.phase=="Running")) \
    | map({ \
        ns: .metadata.namespace, \
        pod: .metadata.name, \
        node: .spec.nodeName, \
        gpus: ([ .spec.containers[]? \
                 | ( (.resources.limits."nvidia.com/gpu" \
                      // .resources.requests."nvidia.com/gpu" \
                      // "0" | tonumber) \
                   + (.resources.limits."amd.com/gpu" \
                      // .resources.requests."amd.com/gpu" \
                      // "0" | tonumber) ) ] | add) \
      }) \
    | map(select(.gpus>0 and .node != null)) \
    | sort_by(.node, .ns, .pod) \
    | group_by(.node) \
    | .[] as $grp \
    | "== Node: \($grp[0].node) ==", \
      "NAMESPACE\tPOD\tGPUs", \
      ( $grp[] | "\(.ns)\t\(.pod)\t\(.gpus)" ), \
      "" \
  ' | column -t -s $'\t' \
  | awk 'NR==1{print; next} /^== /{print ""; print; next} {print}'


# Print table of nodes on Coreweave. Quiet recipe since the command is messy
@cks-nodes:
  kubectl get nodes -o=custom-columns="NAME:metadata.name,IP:status.addresses[?(@.type=='InternalIP')].address,TYPE:metadata.labels['node\.coreweave\.cloud\/type'],LINK:metadata.labels['ethernet\.coreweave\.cloud/speed'],READY:status.conditions[?(@.type=='Ready')].status,CORDON:spec.unschedulable,TAINT:spec.taints[?(@.key=='qos.coreweave.cloud/interruptable')].effect,RELIABILITY:metadata.labels['node\.coreweave\.cloud\/reliability'],LG:metadata.labels['ib\.coreweave\.cloud\/leafgroup'],VERSION:metadata.labels['node\.coreweave\.cloud\/version'],IB:metadata.labels['ib\.coreweave\.cloud\/speed'],STATE:metadata.labels['node\.coreweave\.cloud\/state'],RESERVED:metadata.labels['node\.coreweave\.cloud\/reserved']"

create-secrets:
  kubectl create secret generic hf-secret --from-literal=HF_TOKEN={{HF_TOKEN}} -n {{NAMESPACE}} \
  && kubectl create secret generic gh-token-secret --from-literal=GH_TOKEN={{GH_TOKEN}} -n {{NAMESPACE}}

# Fetch decode pod names and IPs and cache them
get-decode-pods:
  #!/usr/bin/env bash
  set -euo pipefail
  mkdir -p ./.tmp
  echo "Fetching decode pod information..."
  {{KN}} get pods -l llm-d.ai/owner={{NAME_PREFIX}},llm-d.ai/role=decode -o json | jq -r '.items[] | "\(.metadata.name) \(.status.podIP)"' > .tmp/decode_pods.txt
  echo "Decode pods:"
  cat .tmp/decode_pods.txt

VLLM_DEV_REMOTE := "https://github.com/vllm-project/vllm.git"
VLLM_DEV_BRANCH := "main"
VLLM_BUILD_JOBS := "16"

# Show persisted model-server logs (survives LWS pod recreation when backed by shared storage)
# Usage: just logs models/deepseek-v4/1P-EP8-1D-EP8.yaml decode
#        just logs models/deepseek-v4/1P-EP8-1D-EP8.yaml decode -f
logs SPEC ROLE='decode' *ARGS='':
  #!/usr/bin/env bash
  set -euo pipefail
  LOG_DIR=$(uv run manifesto log-path {{SPEC}} --user {{NAME_PREFIX}} --role {{ROLE}})
  if [[ "{{ARGS}}" == *"-f"* ]]; then
    # Follow the latest log file
    LATEST=$({{KN}} exec {{DEV_POD_NAME}} -- bash -c "ls -t $LOG_DIR/*.log 2>/dev/null | head -1")
    if [ -z "$LATEST" ]; then
      echo "No logs found in $LOG_DIR"
      exit 1
    fi
    echo "Following $LATEST"
    {{KN}} exec {{DEV_POD_NAME}} -- tail -f "$LATEST"
  else
    # List recent log files with size and last line
    {{KN}} exec {{DEV_POD_NAME}} -- bash -c "
      ls -lt $LOG_DIR/*.log 2>/dev/null | head -20 || echo 'No logs found in $LOG_DIR'
    "
  fi

# Clean up old persisted model-server logs
logs-clean SPEC ROLE='decode' KEEP='5':
  #!/usr/bin/env bash
  set -euo pipefail
  LOG_DIR=$(uv run manifesto log-path {{SPEC}} --user {{NAME_PREFIX}} --role {{ROLE}})
  {{KN}} exec {{DEV_POD_NAME}} -- bash -c "
    cd $LOG_DIR 2>/dev/null || { echo 'No logs directory'; exit 0; }
    FILES=(\$(ls -t *.log 2>/dev/null))
    if [ \${#FILES[@]} -le {{KEEP}} ]; then
      echo 'Nothing to clean ({{KEEP}} or fewer logs)'
      exit 0
    fi
    echo \"Removing \$((\${#FILES[@]} - {{KEEP}})) old logs (keeping {{KEEP}})...\"
    for f in \"\${FILES[@]:{{KEEP}}}\"; do
      echo \"  rm \$f\"
      rm \"\$f\"
    done
  "

# === Dev Environment ===

# Deploy the persistent dev pod (on a GPU node, for editing/compiling vLLM on shared storage)
dev-start:
  uv run manifesto render-dev-pod --user {{NAME_PREFIX}} | {{KN}} apply -f -
  {{KN}} wait --for=condition=Ready pod/{{DEV_POD_NAME}} --timeout=300s

# Exec into the dev pod
dev:
  {{KN}} exec -it {{DEV_POD_NAME}} -- /bin/zsh

# Build vLLM from source on shared storage (runs in background, survives disconnects)
dev-build REMOTE=VLLM_DEV_REMOTE BRANCH=VLLM_DEV_BRANCH JOBS=VLLM_BUILD_JOBS:
  #!/usr/bin/env bash
  set -euo pipefail
  DEV_VENV=$(uv run manifesto dev-path venv --user {{NAME_PREFIX}})
  DEV_SRC=$(uv run manifesto dev-path source --user {{NAME_PREFIX}})
  USER_ROOT=$(uv run manifesto dev-path user-root --user {{NAME_PREFIX}})
  {{KN}} exec -i {{DEV_POD_NAME}} -- bash <<EOF
  set -euo pipefail

  # ensure venv python points to /usr/bin/python3.12 (works in both dev and runtime images)
  VENV_PYTHON="$DEV_VENV/bin/python"
  if [ "\$(readlink "\$VENV_PYTHON")" != "/usr/bin/python3.12" ]; then
    echo "Repointing venv python: \$(readlink "\$VENV_PYTHON") -> /usr/bin/python3.12"
    ln -sf /usr/bin/python3.12 "\$VENV_PYTHON"
  fi

  source "$DEV_VENV/bin/activate"
  cd "$DEV_SRC"

  # fetch and reset to requested branch
  git remote set-url origin {{REMOTE}}
  git fetch origin
  git checkout {{BRANCH}}
  git reset --hard origin/{{BRANCH}}

  # Flush JIT/compile caches invalidated by the rebuild
  if [ -d "$USER_ROOT/jit-cache" ]; then
    mv "$USER_ROOT/jit-cache" "$USER_ROOT/jit-cache.old.\$(date +%s)" && echo "Flushed jit-cache"
  fi

  # Find the merge-base with upstream main for precompiled C++ extensions
  git remote add upstream https://github.com/vllm-project/vllm.git 2>/dev/null || true
  git fetch upstream main --no-tags 2>/dev/null
  PRECOMPILED_COMMIT=\$(git merge-base HEAD upstream/main 2>/dev/null || git rev-parse HEAD)
  echo "Precompiled wheel commit: \${PRECOMPILED_COMMIT:0:12}"

  # build in background
  VLLM_USE_PRECOMPILED=1 \
  VLLM_PRECOMPILED_WHEEL_COMMIT=\$PRECOMPILED_COMMIT \
  VLLM_PRECOMPILED_WHEEL_VARIANT="" \
  MAX_JOBS={{JOBS}} \
  nohup uv pip install --no-build-isolation -e . \
    > "$USER_ROOT/build.log" 2>&1 &

  echo "Build started ({{REMOTE}} {{BRANCH}}, jobs={{JOBS}}), follow with: just dev-build-log"
  EOF

# Tail the dev build log
dev-build-log:
  #!/usr/bin/env bash
  set -euo pipefail
  USER_ROOT=$(uv run manifesto dev-path user-root --user {{NAME_PREFIX}})
  {{KN}} exec {{DEV_POD_NAME}} -- tail -f "$USER_ROOT/build.log"

# Delete the dev pod
dev-stop:
  uv run manifesto render-dev-pod --user {{NAME_PREFIX}} | {{KN}} delete -f - --ignore-not-found=true

NYANN_BENCH_DIR := env("NYANN_BENCH_DIR", "")
# Spec whose gateway the nyann jobs target; override with NYANN_SPEC in .env or the environment.
NYANN_SPEC := env("NYANN_SPEC", "models/deepseek-v4/1P-EP8-1D-EP8.yaml")
NYANN_LOAD_JOB := NAME_PREFIX + "-sharegpt-load"
NYANN_EVAL_JOB := NAME_PREFIX + "-nyann-eval"
NYANN_GSM8K_JOB := NAME_PREFIX + "-eval-gsm8k"
NYANN_GPQA_JOB := NAME_PREFIX + "-eval-gpqa"

# Wait for stack readiness, then launch nyann-bench load + eval jobs
nyann-stairs SWEEP_MIN='1600' SWEEP_MAX='14400' STEPS='10' STEP_DURATION='300s' ISL='500' OSL='1500' EVAL_CONCURRENCY='16':
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "{{NYANN_BENCH_DIR}}" ]; then
      echo "Error: NYANN_BENCH_DIR is not set. Add it to .env or export it." >&2
      exit 1
    fi
    STEP_DURATION_VAL="{{STEP_DURATION}}"
    STEP_SECS="${STEP_DURATION_VAL%s}"
    EVAL_DURATION="$(( {{STEPS}} * STEP_SECS ))s"
    GATEWAY=$(cd {{justfile_directory()}} && uv run manifesto name {{NYANN_SPEC}} inference-gateway-istio --user {{NAME_PREFIX}})
    EVAL_BASE_URL="http://${GATEWAY}.{{NAMESPACE}}.svc.cluster.local/v1"
    cd "{{NYANN_BENCH_DIR}}"
    go run ./cmd/nyann-bench/ generate \
      --target "$EVAL_BASE_URL" \
      --config '{"load":{"concurrency":128},"warmup":{"duration":"60s","stagger":true},"sweep":{"min":{{SWEEP_MIN}},"max":{{SWEEP_MAX}},"steps":{{STEPS}},"step_duration":"{{STEP_DURATION}}"},"workload":{"type":"corpus","corpus_path":"{{LUSTRE_DATA}}/corpus/sharegpt.txt","isl":{{ISL}},"osl":{{OSL}},"turns":1}}' \
      --workers auto \
      --kube --kube.name {{NYANN_LOAD_JOB}} --kube.volume lustre --kube.image {{NYANN_IMAGE_TAG}} --kube.namespace {{NAMESPACE}} &
    go run ./cmd/nyann-bench/ generate \
      --target "$EVAL_BASE_URL" \
      --config "{\"load\":{\"concurrency\":{{EVAL_CONCURRENCY}},\"duration\":\"${EVAL_DURATION}\"},\"workload\":{\"type\":\"gsm8k\",\"gsm8k_path\":\"{{LUSTRE_DATA}}/gsm8k_test.jsonl\",\"gsm8k_train_path\":\"{{LUSTRE_DATA}}/gsm8k_train.jsonl\"}}" \
      --kube --kube.name {{NYANN_EVAL_JOB}} --kube.volume lustre --kube.image {{NYANN_IMAGE_TAG}} --kube.namespace {{NAMESPACE}} &
    wait
    echo "nyann-bench jobs submitted. Use 'just nyann-logs load' or 'just nyann-logs eval' to follow."

nyann CONCURRENCY='14400' DURATION='600s' ISL='500' OSL='1500' EVAL_CONCURRENCY='16':
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "{{NYANN_BENCH_DIR}}" ]; then
      echo "Error: NYANN_BENCH_DIR is not set. Add it to .env or export it." >&2
      exit 1
    fi
    GATEWAY=$(cd {{justfile_directory()}} && uv run manifesto name {{NYANN_SPEC}} inference-gateway-istio --user {{NAME_PREFIX}})
    EVAL_BASE_URL="http://${GATEWAY}.{{NAMESPACE}}.svc.cluster.local/v1"
    cd "{{NYANN_BENCH_DIR}}"
    go run ./cmd/nyann-bench/ generate \
      --target "$EVAL_BASE_URL" \
      --config '{"load":{"concurrency":{{CONCURRENCY}},"duration":"{{DURATION}}","rampup":"30s"},"warmup":{"duration":"60s","stagger":true},"workload":{"type":"corpus","corpus_path":"{{LUSTRE_DATA}}/corpus/sharegpt.txt","isl":{{ISL}},"osl":{{OSL}},"turns":1}}' \
      --workers auto \
      --kube --kube.name {{NYANN_LOAD_JOB}} --kube.volume lustre --kube.image {{NYANN_IMAGE_TAG}} --kube.namespace {{NAMESPACE}} &
    go run ./cmd/nyann-bench/ generate \
      --target "$EVAL_BASE_URL" \
      --config '{"load":{"concurrency":{{EVAL_CONCURRENCY}},"duration":"{{DURATION}}"},"workload":{"type":"gsm8k","gsm8k_path":"{{LUSTRE_DATA}}/gsm8k_test.jsonl","gsm8k_train_path":"{{LUSTRE_DATA}}/gsm8k_train.jsonl"}}' \
      --kube --kube.name {{NYANN_EVAL_JOB}} --kube.volume lustre --kube.image {{NYANN_IMAGE_TAG}} --kube.namespace {{NAMESPACE}} &
    wait
    echo "nyann-bench jobs submitted. Use 'just nyann-logs load' or 'just nyann-logs eval' to follow."

LUSTRE_DATA := "/mnt/lustre/" + NAME_PREFIX

NYANN_IMAGE_TAG := env("NYANN_IMAGE_TAG", "latest")

# Run GSM8K eval (1319 math problems, ~30 min)
nyann-eval-gsm8k CONCURRENCY='64':
    #!/usr/bin/env bash
    set -euo pipefail
    GATEWAY=$(uv run manifesto name {{NYANN_SPEC}} inference-gateway-istio --user {{NAME_PREFIX}})
    cd {{NYANN_BENCH_DIR}} && NYANN_NAME_PREFIX={{NAME_PREFIX}} go run ./cmd/nyann-bench/ eval gsm8k \
      --target "http://${GATEWAY}.{{NAMESPACE}}.svc.cluster.local/v1" \
      --gsm8k-path {{LUSTRE_DATA}}/gsm8k_test.jsonl \
      --gsm8k-train-path {{LUSTRE_DATA}}/gsm8k_train.jsonl \
      --concurrency {{CONCURRENCY}} \
      --timeout 2h \
      --kube --kube.volume lustre --kube.image {{NYANN_IMAGE_TAG}} --kube.namespace {{NAMESPACE}}

# Run GPQA Diamond eval (198 grad-level science questions)
nyann-eval-gpqa CONCURRENCY='64':
    #!/usr/bin/env bash
    set -euo pipefail
    GATEWAY=$(uv run manifesto name {{NYANN_SPEC}} inference-gateway-istio --user {{NAME_PREFIX}})
    cd {{NYANN_BENCH_DIR}} && NYANN_NAME_PREFIX={{NAME_PREFIX}} go run ./cmd/nyann-bench/ eval gpqa \
      --target "http://${GATEWAY}.{{NAMESPACE}}.svc.cluster.local/v1" \
      --gpqa-path {{LUSTRE_DATA}}/gpqa_diamond.jsonl \
      --concurrency {{CONCURRENCY}} \
      --timeout 2h \
      --kube --kube.volume lustre --kube.image {{NYANN_IMAGE_TAG}} --kube.namespace {{NAMESPACE}}

# Run both evals in parallel
nyann-eval: (nyann-eval-gsm8k) (nyann-eval-gpqa)

# Prep GPQA dataset on Lustre (if missing or empty)
nyann-prep-gpqa:
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{NYANN_BENCH_DIR}}"
    just prep-gpqa "{{LUSTRE_DATA}}" {{NAMESPACE}}

# Stop nyann-bench benchmark jobs
nyann-stop:
  {{KN}} delete jobset -l app={{NYANN_LOAD_JOB}} --ignore-not-found=true &
  {{KN}} delete jobset -l app={{NYANN_EVAL_JOB}} --ignore-not-found=true &
  {{KN}} delete jobset -l app={{NYANN_GSM8K_JOB}} --ignore-not-found=true &
  {{KN}} delete jobset -l app={{NYANN_GPQA_JOB}} --ignore-not-found=true &
  wait

# Tail nyann-bench job logs
nyann-logs TARGET='load':
  #!/usr/bin/env bash
  set -euo pipefail
  case "{{TARGET}}" in
    load) APP="{{NYANN_LOAD_JOB}}" ;;
    eval) APP="{{NYANN_EVAL_JOB}}" ;;
    gsm8k) APP="{{NYANN_GSM8K_JOB}}" ;;
    gpqa) APP="{{NYANN_GPQA_JOB}}" ;;
    *) APP="{{TARGET}}" ;;
  esac
  {{KN}} logs -l app="$APP" -c nyann-bench --tail=50 -f --max-log-requests=20

# Query Prometheus for per-stage benchmark metrics (requires port-forward: just prometheus)
query-prometheus CLIENT_JOB=(NAME_PREFIX + "-sharegpt-load") DEPLOYMENT=(NAME_PREFIX + "-wide-ep") EVAL_JOB=(NAME_PREFIX + "-nyann-eval") *ARGS='':
  #!/usr/bin/env bash
  set -euo pipefail
  if [ -z "{{NYANN_BENCH_DIR}}" ]; then
    echo "Error: NYANN_BENCH_DIR is not set. Add it to .env or export it." >&2
    exit 1
  fi
  cd "{{NYANN_BENCH_DIR}}"
  just query-prometheus {{CLIENT_JOB}} {{DEPLOYMENT}} {{NAMESPACE}} '' {{EVAL_JOB}} {{ARGS}}

# === Monitoring ===

# Install Prometheus and Grafana (namespace-scoped, no cluster permissions needed)
start-monitoring:
  helm repo add prometheus-community https://prometheus-community.github.io/helm-charts --force-update && helm repo add grafana https://grafana.github.io/helm-charts --force-update && helm repo update && kubectl apply -f {{MONITORING_DIR}}/prometheus-rbac.yaml && kubectl apply -f {{MONITORING_DIR}}/grafana-rbac.yaml && helm upgrade --install prometheus prometheus-community/prometheus -n {{NAMESPACE}} -f {{MONITORING_DIR}}/prometheus-values.yaml --set rbac.create=false --set serviceAccounts.server.create=false && helm upgrade --install grafana grafana/grafana -n {{NAMESPACE}} -f {{MONITORING_DIR}}/grafana-values.yaml --set rbac.create=false

# Uninstall monitoring stack
stop-monitoring:
  helm uninstall prometheus -n {{NAMESPACE}} --ignore-not-found && helm uninstall grafana -n {{NAMESPACE}} --ignore-not-found && kubectl delete -f {{MONITORING_DIR}}/prometheus-rbac.yaml --ignore-not-found && kubectl delete -f {{MONITORING_DIR}}/grafana-rbac.yaml --ignore-not-found

# Port-forward Grafana to localhost:3000 (background)
grafana:
  kubectl port-forward -n {{NAMESPACE}} svc/grafana 3000:80 > /dev/null 2>&1 &

# Port-forward Prometheus to localhost:9090 (background)
prometheus:
  kubectl port-forward -n {{NAMESPACE}} svc/prometheus-server 9090:80 > /dev/null 2>&1 &

# Load Grafana dashboards
load-dashboards:
  #!/usr/bin/env bash
  set -euo pipefail
  for f in "{{MONITORING_DIR}}"/*.json; do
    [ -f "$f" ] || continue
    NAME=$(basename "$f" .json)
    CM_NAME=$(printf 'grafana-dashboard-%s' "$NAME" | tr '[:upper:]' '[:lower:]')
    echo "Creating ConfigMap for dashboard: $NAME ($CM_NAME)"
    {{KN}} create configmap "$CM_NAME" --from-file="$NAME.json=$f" --dry-run=client -o yaml | \
      {{KN}} label -f - --local -o yaml grafana_dashboard=1 | \
      {{KN}} apply -f -
  done
  echo "Dashboards loaded. Grafana will auto-discover them within 30 seconds."
