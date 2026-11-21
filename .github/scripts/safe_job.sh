#!/usr/bin/env bash
# Runs a job's GPU steps in order on one 8-GPU Primus-SaFE workload; each step returns its exit code.
# usage: safe_job.sh start <script> (submits the workload) | <script> | run <command> | delete
# GPU_WORKSPACES is "workspace:priority ..."; a workload still queued after QUEUE_MINUTES moves to the
# next one. PASS_ENV names the variables the workload gets; lines a step appends to $SAFE_ENV are
# sourced before the later steps.
set -euo pipefail
shopt -s nullglob
# Workloads reach NFS as root or as the submitting user, depending on the workspace.
umask 000

: "${SAFE_API_KEY:?is empty: the SAFE_API_KEY secret is not available to this run}"
read -ra entries <<< "$GPU_WORKSPACES"
workspaces=("${entries[@]%%:*}")
priorities=("${entries[@]#*:}")
# The runner image ships no curl.
command -v curl > /dev/null || { apt-get update -qq && apt-get install -y -qq curl > /dev/null; }
certs=https://raw.githubusercontent.com/AMD-AGI/Primus-SaFE/main/Scripts/setup-certs
[[ -s $RUNNER_TEMP/amd-ca.pem ]] ||
  curl -fsSL --retry 3 "$certs/amd-root-ca.crt" "$certs/amd-issuing-ca.crt" > "$RUNNER_TEMP/amd-ca.pem"

api() {
  curl -sS --fail-with-body --max-time 60 --cacert "$RUNNER_TEMP/amd-ca.pem" -X "$1" \
    -H "Authorization: Bearer $SAFE_API_KEY" -H 'Content-Type: application/json' \
    "https://global.primus-safe.amd.com/api/v1/workloads$2" "${@:3}"
}

name=turbo-ci-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT
export RUN_DIR=/shared_nfs/github-runners/_runs-turbo/$name

if [[ ${1:-} == delete ]]; then
  rm -rf "$RUN_DIR"
  for ws in "${workspaces[@]}"; do
    ids=$(api GET "?workspaceId=$ws&kind=Deployment&limit=100" |
      jq -r --arg name "$name-" '.items[]?.workloadId | select(startswith($name))')
    for id in $ids; do
      echo "🧹 Deleting the workload $id"
      api DELETE "/$id" > /dev/null || echo "::warning::Could not delete the workload $id"
    done
  done
  exit
fi

start=false
[[ ${1:-} == start ]] && { start=true; shift; }
if [[ ${1:-} == run ]]; then
  printf '%s\n' "${2:?usage: safe_job.sh run <command>}" > "$RUNNER_TEMP/safe-step.sh"
  set -- "$RUNNER_TEMP/safe-step.sh"
fi
script=${1:?usage: safe_job.sh [start] <script> | run <command> | delete}

pod_command=$(
  cat << 'EOF'
umask 000
if [ -e "$RUN_DIR/started" ]; then
  touch "$RUN_DIR/restarted"
  exec sleep infinity
fi
touch "$RUN_DIR/started"
work=$(mktemp -d)
export REPORTS=$work/reports SAFE_ENV=$work/env
mkdir "$work/src" "$REPORTS"
touch "$SAFE_ENV"
cd "$work/src"
tar -xf "$RUN_DIR/src.tar"
n=1
while :; do
  if [ -e "$RUN_DIR/$n.sh" ]; then
    (set -a; . "$SAFE_ENV"; set +a; exec bash "$RUN_DIR/$n.sh") >> "$RUN_DIR/$n.log" 2>&1
    code=$?
    tar -C "$REPORTS" -cf "$RUN_DIR/reports.tar" .
    echo "$code" > "$RUN_DIR/$n.exit"
    n=$((n + 1))
  else
    sleep 2
  fi
done
EOF
)

# No timeout: SaFE counts it from submission, queueing included.
submit() {
  workload=$(jq -n --arg name "$name" --arg ws "${workspaces[i]}" --argjson priority "${priorities[i]}" \
    --arg command "$(base64 -w0 <<< "$pod_command")" --arg pass "${PASS_ENV:-}" '{
    displayName: $name,
    workspaceId: $ws,
    groupVersionKind: {kind: "Deployment", version: "v1"},
    images: [env.IMAGE],
    entryPoints: [$command],
    resources: [{replica: 1, gpu: "8", cpu: "96", memory: "512Gi", sharedMemory: "64Gi", ephemeralStorage: "300Gi"}],
    env: ({RUN_DIR: env.RUN_DIR} + ($ENV | with_entries(select(.key | IN($pass | splits(" +")))))),
    priority: $priority,
    useWorkspaceStorage: true
  }' | api POST '' -d @- | jq -r '.workloadId // error(tostring)')
  submitted=$SECONDS
  echo "🚀 $(date -u +%T) Workload $workload in ${workspaces[i]} at priority ${priorities[i]}, image $IMAGE"
}

last_state='' lost=0
check_workload() {
  if ! status=$(api GET "/$workload"); then
    ((++lost < 8)) && return
    echo "::error::Workload $workload is gone: deleted or stopped by the platform"
    exit 1
  fi
  lost=0
  local phase state
  phase=$(jq -r .phase <<< "$status")
  state="$phase: $(jq -r '.message // "" | .[:200]' <<< "$status")"
  [[ $state == "$last_state" ]] || echo "⏳ $(date -u +%T) Workload $workload $state"
  last_state=$state
  if [[ $phase == Failed || $phase == Stopped ]]; then
    echo "::error::Workload $workload $phase"
    exit 1
  fi
}

if $start; then
  mkdir -p "$RUN_DIR"
  tar -C "$GITHUB_WORKSPACE" --exclude=./.git --exclude=./build -cf "$RUN_DIR/src.tar" .
  i=0
  submit
  until [[ -e $RUN_DIR/started ]]; do
    sleep 15
    if ((i + 1 < ${#workspaces[@]} && SECONDS - submitted > ${QUEUE_MINUTES:-10} * 60)); then
      echo "↪️ $(date -u +%T) Still queued in ${workspaces[i]}; moving to ${workspaces[i + 1]}"
      api DELETE "/$workload" > /dev/null || echo "::warning::Could not delete the workload $workload"
      rm -f "$RUN_DIR/started"
      i=$((i + 1))
      submit
      last_state=''
      continue
    fi
    check_workload
  done
  echo "✅ $(date -u +%T) Workload $workload started in ${workspaces[i]} after $(((SECONDS - submitted) / 60)) minutes"
  echo "$workload" > "$RUN_DIR/workload"
fi
[[ -s $RUN_DIR/workload ]] || { echo "::error::No GPU workload: the step that starts it did not succeed"; exit 1; }
workload=$(< "$RUN_DIR/workload")

steps=("$RUN_DIR"/[0-9]*.sh)
n=$((${#steps[@]} + 1))
cat "$script" > "$RUN_DIR/.$n.sh"
mv "$RUN_DIR/.$n.sh" "$RUN_DIR/$n.sh"
tail --pid=$$ -F "$RUN_DIR/$n.log" 2> /dev/null &
until [[ -s $RUN_DIR/$n.exit ]]; do
  sleep 15
  if [[ -e $RUN_DIR/restarted ]]; then
    echo "::error::The GPU pod restarted and lost its state"
    exit 1
  fi
  check_workload
done
sleep 5
kill $!

mkdir -p "$RUNNER_TEMP/gpu-reports"
tar -C "$RUNNER_TEMP/gpu-reports" -xf "$RUN_DIR/reports.tar"
exit "$(< "$RUN_DIR/$n.exit")"
