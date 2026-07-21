# Lifecycle, renderer, dev, monitoring, and Nyann benchmark commands for manifesto.
set dotenv-load
set dotenv-required

NAMESPACE := env("MANIFESTO_NAMESPACE", `namespace=$(kubectl config view --minify -o jsonpath='{..namespace}' 2>/dev/null || true); printf '%s' "${namespace:-default}"`)
HF_TOKEN := "$HF_TOKEN"
GH_TOKEN := "$GH_TOKEN"

KN := "kubectl -n " + NAMESPACE

NAME_PREFIX := env("USER", "dev")
DEPLOY_NAME := NAME_PREFIX + "-wide-ep"
DEV_POD_NAME := NAME_PREFIX + "-vllm-dev"

DEV_DIR := "dev"
MONITORING_DIR := "monitoring"
RENDER_OUT := env("MANIFESTO_RENDER_OUT", "/tmp/" + NAME_PREFIX + "-manifesto.yaml")
EDITOR := env("EDITOR", "vi")

default:
  just --list

# === v2 spec-based renderer ===

@_cluster:
  #!/usr/bin/env bash
  set -euo pipefail
  CLUSTER="${MANIFESTO_CLUSTER:-}"
  if [ -z "$CLUSTER" ] && [ -z "${MANIFESTO_CLUSTER_MAP:-}" ]; then
    echo "MANIFESTO_CLUSTER is required, or set MANIFESTO_CLUSTER_MAP in .env." >&2
    exit 2
  fi
  if [ -z "$CLUSTER" ]; then
    context=$(kubectl config current-context)
    cluster=$(kubectl config view --minify -o jsonpath='{.clusters[0].name}')
    IFS=',' read -ra entries <<< "${MANIFESTO_CLUSTER_MAP:-}"
    for entry in "${entries[@]}"; do
      key="${entry%%=*}"
      value="${entry#*=}"
      if [ "$key" = "$context" ] || [ "$key" = "$cluster" ]; then
        CLUSTER="$value"
        break
      fi
    done
  fi
  if [ -z "$CLUSTER" ]; then
    echo "No cluster mapping found for current kube context or cluster. Set MANIFESTO_CLUSTER, or add it to MANIFESTO_CLUSTER_MAP in .env." >&2
    exit 2
  fi
  echo "$CLUSTER"

render SPEC *ARGS='':
  #!/usr/bin/env bash
  set -euo pipefail
  CLUSTER="$(just --quiet _cluster)"
  uv run manifesto render {{SPEC}} --cluster "$CLUSTER" --namespace {{NAMESPACE}} --user {{NAME_PREFIX}} {{ARGS}}

render-file SPEC *ARGS='':
  #!/usr/bin/env bash
  set -euo pipefail
  CLUSTER="$(just --quiet _cluster)"
  uv run manifesto render {{SPEC}} --cluster "$CLUSTER" --namespace {{NAMESPACE}} --user {{NAME_PREFIX}} {{ARGS}} > "{{RENDER_OUT}}"
  echo "{{RENDER_OUT}}"

edit-file SPEC *ARGS='':
  #!/usr/bin/env bash
  set -euo pipefail
  just render-file {{SPEC}} {{ARGS}}
  "{{EDITOR}}" "{{RENDER_OUT}}"

diff-file FILE=RENDER_OUT:
  {{KN}} diff -f "{{FILE}}"

apply-file FILE=RENDER_OUT:
  {{KN}} apply -f "{{FILE}}"

delete-file FILE=RENDER_OUT NOW='false':
  #!/usr/bin/env bash
  set -euo pipefail
  FORCE=""
  if [ "{{NOW}}" = "true" ]; then
    FORCE="--grace-period=0 --force"
  fi
  {{KN}} delete -f "{{FILE}}" --ignore-not-found=true $FORCE

render-routing SPEC *ARGS='':
  #!/usr/bin/env bash
  set -euo pipefail
  CLUSTER="$(just --quiet _cluster)"
  uv run manifesto render-routing {{SPEC}} --cluster "$CLUSTER" --namespace {{NAMESPACE}} --user {{NAME_PREFIX}} {{ARGS}}

start SPEC *ARGS='':
  #!/usr/bin/env bash
  set -euo pipefail
  CLUSTER="$(just --quiet _cluster)"
  uv run manifesto render {{SPEC}} --cluster "$CLUSTER" --namespace {{NAMESPACE}} --user {{NAME_PREFIX}} {{ARGS}} | {{KN}} apply -f -

deploy-routing SPEC *ARGS='':
  #!/usr/bin/env bash
  set -euo pipefail
  CLUSTER="$(just --quiet _cluster)"
  uv run manifesto render-routing {{SPEC}} --cluster "$CLUSTER" --namespace {{NAMESPACE}} --user {{NAME_PREFIX}} {{ARGS}} | {{KN}} apply -f -

stop SPEC NOW='false':
  #!/usr/bin/env bash
  set -euo pipefail
  FORCE=""
  if [ "{{NOW}}" = "true" ]; then
    FORCE="--grace-period=0 --force"
  fi
  CLUSTER="$(just --quiet _cluster)"
  uv run manifesto render {{SPEC}} --cluster "$CLUSTER" --namespace {{NAMESPACE}} --user {{NAME_PREFIX}} | {{KN}} delete -f - --ignore-not-found=true $FORCE

restart SPEC *ARGS='':
  #!/usr/bin/env bash
  set -euo pipefail
  INSTANCE=$(uv run manifesto instance-id {{SPEC}} --user {{NAME_PREFIX}})
  {{KN}} delete lws -l app.kubernetes.io/instance=$INSTANCE --ignore-not-found=true --grace-period=0 --force &
  {{KN}} delete pod -l app.kubernetes.io/instance=$INSTANCE --ignore-not-found=true --grace-period=0 --force &
  wait
  {{KN}} wait --for=delete pod -l app.kubernetes.io/instance=$INSTANCE --timeout=60s 2>/dev/null || true
  just start {{SPEC}} {{ARGS}}

# Wait for the v2-rendered stack to be ready.
ready SPEC:
  #!/usr/bin/env bash
  set -euo pipefail
  INSTANCE=$(uv run manifesto instance-id {{SPEC}} --user {{NAME_PREFIX}})
  EPP=$(uv run manifesto name {{SPEC}} infpool-epp --user {{NAME_PREFIX}})
  GATEWAY_SVC=$(uv run manifesto name {{SPEC}} inference-gateway-istio --user {{NAME_PREFIX}})
  {{KN}} wait --for=condition=Ready pod -l app.kubernetes.io/instance=$INSTANCE,llm-d.ai/role=decode --timeout=1200s &
  ({{KN}} wait --for=condition=Ready pod -l app.kubernetes.io/instance=$INSTANCE,llm-d.ai/role=prefill --timeout=1200s 2>/dev/null || true) &
  {{KN}} wait --for=condition=Available deploy/$EPP --timeout=120s &
  echo "Waiting for v2 model pods and EPP..."
  wait
  echo "Checking gateway..."
  until curl -sf --max-time 5 http://$GATEWAY_SVC:80/v1/models 2>/dev/null | grep -q '"id"' || \
    {{KN}} exec deploy/$EPP -- curl -sf --max-time 5 http://$GATEWAY_SVC:80/v1/models 2>/dev/null | grep -q '"id"'
  do
    sleep 2
  done
  echo "Ready."

flush-cache SPEC *ARGS='':
  #!/usr/bin/env bash
  set -euo pipefail
  CLUSTER="$(just --quiet _cluster)"
  CACHE_PATH=$(uv run manifesto cache-path {{SPEC}} --cluster "$CLUSTER" --user {{NAME_PREFIX}} {{ARGS}})
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

VLLM_DEV_VENV := "/mnt/lustre/" + NAME_PREFIX + "/vllm-venv"
VLLM_DEV_SRC := "/mnt/lustre/" + NAME_PREFIX + "/vllm-dev"
VLLM_DEV_REMOTE := "https://github.com/vllm-project/vllm.git"
VLLM_DEV_BRANCH := "main"
VLLM_BUILD_JOBS := "16"
IMAGE_CATALOG := "config/images.yaml"

# Show persisted model-server logs (survives LWS pod recreation when backed by shared storage)
# Usage: just logs models/deepseek-v4/1P-EP8-1D-EP8.yaml decode
#        just logs models/deepseek-v4/1P-EP8-1D-EP8.yaml decode -f
logs SPEC ROLE='decode' *ARGS='':
  #!/usr/bin/env bash
  set -euo pipefail
  CLUSTER="$(just --quiet _cluster)"
  LOG_DIR=$(uv run manifesto log-path {{SPEC}} --cluster "$CLUSTER" --user {{NAME_PREFIX}} --role {{ROLE}})
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
  CLUSTER="$(just --quiet _cluster)"
  LOG_DIR=$(uv run manifesto log-path {{SPEC}} --cluster "$CLUSTER" --user {{NAME_PREFIX}} --role {{ROLE}})
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

# Deploy the persistent dev pod (CPU-only, for editing/compiling vLLM on Lustre)
dev-start:
  VLLM_DEV_IMAGE=$(python3 -c 'import yaml; print(yaml.safe_load(open("{{IMAGE_CATALOG}}"))["dev"]["image"])') envsubst < {{DEV_DIR}}/dev-pod.yaml | {{KN}} apply -f -
  {{KN}} wait --for=condition=Ready pod/{{DEV_POD_NAME}} --timeout=300s

# Exec into the dev pod
dev:
  {{KN}} exec -it {{DEV_POD_NAME}} -- /bin/zsh

# Build vLLM from source on Lustre (runs in background, survives disconnects)
dev-build REMOTE=VLLM_DEV_REMOTE BRANCH=VLLM_DEV_BRANCH JOBS=VLLM_BUILD_JOBS:
  #!/usr/bin/env bash
  set -euo pipefail
  {{KN}} exec -i {{DEV_POD_NAME}} -- bash <<'EOF'
  set -euo pipefail

  # ensure venv python points to /usr/bin/python3.12 (works in both dev and runtime images)
  VENV_PYTHON="{{VLLM_DEV_VENV}}/bin/python"
  if [ "$(readlink "$VENV_PYTHON")" != "/usr/bin/python3.12" ]; then
    echo "Repointing venv python: $(readlink "$VENV_PYTHON") -> /usr/bin/python3.12"
    ln -sf /usr/bin/python3.12 "$VENV_PYTHON"
  fi

  source {{VLLM_DEV_VENV}}/bin/activate
  cd {{VLLM_DEV_SRC}}

  # fetch and reset to requested branch
  git remote set-url origin {{REMOTE}}
  git fetch origin
  git checkout {{BRANCH}}
  git reset --hard origin/{{BRANCH}}

  # Flush stale JIT/compile caches (AOT graphs, flashinfer kernels, FA cute DSL)
  for d in vllm_cache_extdp flashinfer_cache_extdp fa_cute_dsl_cache; do
    [ -d "/mnt/lustre/{{NAME_PREFIX}}/$d" ] && mv "/mnt/lustre/{{NAME_PREFIX}}/$d" "/mnt/lustre/{{NAME_PREFIX}}/${d}.old.$(date +%s)" && echo "Flushed $d"
  done

  # Find the merge-base with upstream main for precompiled C++ extensions
  git remote add upstream https://github.com/vllm-project/vllm.git 2>/dev/null || true
  git fetch upstream main --no-tags 2>/dev/null
  PRECOMPILED_COMMIT=$(git merge-base HEAD upstream/main 2>/dev/null || git rev-parse HEAD)
  echo "Precompiled wheel commit: ${PRECOMPILED_COMMIT:0:12}"

  # build in background
  VLLM_USE_PRECOMPILED=1 \
  VLLM_PRECOMPILED_WHEEL_COMMIT=$PRECOMPILED_COMMIT \
  VLLM_PRECOMPILED_WHEEL_VARIANT="" \
  MAX_JOBS={{JOBS}} \
  nohup uv pip install --no-build-isolation -e . \
    > /mnt/lustre/{{NAME_PREFIX}}/build.log 2>&1 &

  echo "Build started ({{REMOTE}} {{BRANCH}}, jobs={{JOBS}}), follow with: just dev-build-log"
  EOF

# Tail the dev build log
dev-build-log:
  {{KN}} exec {{DEV_POD_NAME}} -- tail -f /mnt/lustre/{{NAME_PREFIX}}/build.log

# Delete the dev pod
dev-stop:
  VLLM_DEV_IMAGE=$(python3 -c 'import yaml; print(yaml.safe_load(open("{{IMAGE_CATALOG}}"))["dev"]["image"])') envsubst < {{DEV_DIR}}/dev-pod.yaml | {{KN}} delete -f - --ignore-not-found=true

NYANN_BENCH_DIR := env("NYANN_BENCH_DIR", "")
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
    cd "{{NYANN_BENCH_DIR}}"
    go run ./cmd/nyann-bench/ generate \
      --target "{{EVAL_BASE_URL}}" \
      --config '{"load":{"concurrency":128},"warmup":{"duration":"60s","stagger":true},"sweep":{"min":{{SWEEP_MIN}},"max":{{SWEEP_MAX}},"steps":{{STEPS}},"step_duration":"{{STEP_DURATION}}"},"workload":{"type":"corpus","corpus_path":"{{LUSTRE_DATA}}/corpus/sharegpt.txt","isl":{{ISL}},"osl":{{OSL}},"turns":1}}' \
      --workers auto \
      --kube --kube.name {{NYANN_LOAD_JOB}} --kube.volume lustre --kube.image {{NYANN_IMAGE_TAG}} --kube.namespace {{NAMESPACE}} &
    go run ./cmd/nyann-bench/ generate \
      --target "{{EVAL_BASE_URL}}" \
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
    cd "{{NYANN_BENCH_DIR}}"
    go run ./cmd/nyann-bench/ generate \
      --target "{{EVAL_BASE_URL}}" \
      --config '{"load":{"concurrency":{{CONCURRENCY}},"duration":"{{DURATION}}","rampup":"30s"},"warmup":{"duration":"60s","stagger":true},"workload":{"type":"corpus","corpus_path":"{{LUSTRE_DATA}}/corpus/sharegpt.txt","isl":{{ISL}},"osl":{{OSL}},"turns":1}}' \
      --workers auto \
      --kube --kube.name {{NYANN_LOAD_JOB}} --kube.volume lustre --kube.image {{NYANN_IMAGE_TAG}} --kube.namespace {{NAMESPACE}} &
    go run ./cmd/nyann-bench/ generate \
      --target "{{EVAL_BASE_URL}}" \
      --config '{"load":{"concurrency":{{EVAL_CONCURRENCY}},"duration":"{{DURATION}}"},"workload":{"type":"gsm8k","gsm8k_path":"{{LUSTRE_DATA}}/gsm8k_test.jsonl","gsm8k_train_path":"{{LUSTRE_DATA}}/gsm8k_train.jsonl"}}' \
      --kube --kube.name {{NYANN_EVAL_JOB}} --kube.volume lustre --kube.image {{NYANN_IMAGE_TAG}} --kube.namespace {{NAMESPACE}} &
    wait
    echo "nyann-bench jobs submitted. Use 'just nyann-logs load' or 'just nyann-logs eval' to follow."

LUSTRE_DATA := "/mnt/lustre/" + NAME_PREFIX
EVAL_BASE_URL := "http://" + DEPLOY_NAME + "-inference-gateway-istio." + NAMESPACE + ".svc.cluster.local/v1"

NYANN_IMAGE_TAG := env("NYANN_IMAGE_TAG", "latest")

# Run GSM8K eval (1319 math problems, ~30 min)
nyann-eval-gsm8k CONCURRENCY='64':
    cd {{NYANN_BENCH_DIR}} && NYANN_NAME_PREFIX={{NAME_PREFIX}} go run ./cmd/nyann-bench/ eval gsm8k \
      --target "{{EVAL_BASE_URL}}" \
      --gsm8k-path {{LUSTRE_DATA}}/gsm8k_test.jsonl \
      --gsm8k-train-path {{LUSTRE_DATA}}/gsm8k_train.jsonl \
      --concurrency {{CONCURRENCY}} \
      --timeout 2h \
      --kube --kube.volume lustre --kube.image {{NYANN_IMAGE_TAG}} --kube.namespace {{NAMESPACE}}

# Run GPQA Diamond eval (198 grad-level science questions)
nyann-eval-gpqa CONCURRENCY='64':
    cd {{NYANN_BENCH_DIR}} && NYANN_NAME_PREFIX={{NAME_PREFIX}} go run ./cmd/nyann-bench/ eval gpqa \
      --target "{{EVAL_BASE_URL}}" \
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
query-prometheus CLIENT_JOB=(NAME_PREFIX + "-sharegpt-load") DEPLOYMENT=DEPLOY_NAME EVAL_JOB=(NAME_PREFIX + "-nyann-eval") *ARGS='':
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
