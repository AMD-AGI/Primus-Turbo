#!/bin/bash
# Llama-3.1-8B BF16 pretraining on one MI455X (gfx1250) with Primus (torchtitan) + Primus-Turbo
# branch dev/lhz/llama31_attn_opt (FlyDSL attention). MBS = GBS = 4, seq 8192.
#
#   run_l8b_e2e.sh [steps=20] [tag]
#
# Runs on the HOST; the training runs inside the container $CT (image fa-tune:deps). Paths are
# host paths and must be visible at the same path inside the container (the container mounts
# /home/<user> at the same place). Override with env: CT PRIMUS TURBO HF_ASSETS WORKSPACE.
set -euo pipefail
STEPS=${1:-20}
TAG=${2:-l8b_$(date +%m%d_%H%M%S)}
HERE=$(cd "$(dirname "$0")" && pwd)
CT=${CT:-fa-repro}
PRIMUS=${PRIMUS:-/home/lihuzhan/code/2026_0828__primus/Primus}          # + primus-nkfix-hook.patch
TURBO=${TURBO:-/home/lihuzhan/code/2026_0903__turbo/wt-llama31}           # dev/lhz/llama31_attn_opt
HF_ASSETS=${HF_ASSETS:-/home/lihuzhan/_hfassets/llama31_8B}
WORKSPACE=${WORKSPACE:-/home/lihuzhan/_l8b_runs}
# The hipBLASLt library shipped INSIDE the image. Never point this at a host copy
# (e.g. ~/.local/hipblaslt-gfx1250): that produced NaNs and a GPU wedge on this card.
BLAS_LIB=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250

mkdir -p "$WORKSPACE"
CFG=$WORKSPACE/$TAG.yaml
LOG=$WORKSPACE/$TAG.log
sed -e "s#@STEPS@#$STEPS#" -e "s#@HF_ASSETS@#$HF_ASSETS#" -e "s#@WORKSPACE@#$WORKSPACE#" \
  "$HERE/l8b_mi455x.yaml.in" > "$CFG"

# One GPU client at a time on this card: refuse to start next to another KFD user.
if [ -n "$(ls /sys/class/kfd/kfd/proc 2>/dev/null)" ]; then
  echo "another process holds the GPU (/sys/class/kfd/kfd/proc is not empty); not starting" >&2
  exit 1
fi

echo "[$(date +%T)] $TAG: steps=$STEPS turbo=$TURBO primus=$PRIMUS log=$LOG"
T0=$(date +%s)
# bash -c, not bash -lc: a login shell reads /etc/profile.d, which forces TORCH_BLAS_PREFER_HIPBLASLT=0.
set +e
docker exec \
  -e GPUS_PER_NODE=1 -e NNODES=1 -e NODE_RANK=0 -e MASTER_PORT=$((20000 + RANDOM % 20000)) \
  -e PRIMUS_GPU_MODEL=MI455X -e PRIMUS_EXP_NAME="$TAG" \
  "$CT" bash -c "
    ulimit -c 0
    export TORCH_BLAS_PREFER_HIPBLASLT=1 HIPBLASLT_TENSILE_LIBPATH=$BLAS_LIB
    export NKFIX_ENABLE=1 NKFIX_CHECK=\${NKFIX_CHECK:-${NKFIX_CHECK:-0}}
    export FLYDSL_RUNTIME_CACHE_DIR=/tmp/flydsl_cache_$TAG TRITON_CACHE_DIR=/tmp/triton_cache_l8b
    export PYTHONPATH=$HERE/nkfix:$TURBO
    (cd /tmp && python -c 'import primus_turbo, flydsl; print(\"primus_turbo\", primus_turbo.__file__, \"flydsl\", flydsl.__version__)')
    cd $PRIMUS && exec bash runner/primus-cli direct --log_file $WORKSPACE/$TAG.launcher.log -- \
      train pretrain --config $CFG
  " > "$LOG" 2>&1
RC=$?
set -e
T1=$(date +%s)

# Summary: step 1-5 include warmup (FlyDSL JIT, allocator growth); report steps 6..N.
sed 's/\x1b\[[0-9;]*m//g' "$LOG" | python3 -c "
import re, statistics, sys
rows = []
prev = None
for line in sys.stdin:
    m = re.search(r'^\[(\d{8}) (\d\d):(\d\d):(\d\d)\].*step:\s*(\d+)\s+loss:\s*([-\w.]+).*memory:\s*([\d.]+)GiB\(([\d.]+)%\).*tps:\s*([\d,]+)', line)
    if m:
        rows.append((int(m.group(5)), float(m.group(6)), float(m.group(8)), int(m.group(9).replace(',', ''))))
if not rows:
    print('no training steps found in the log'); sys.exit(0)
steady = [r for r in rows if r[0] >= 6] or rows
tps = statistics.median(r[3] for r in steady)
print(f'steps logged: {len(rows)}  last loss: {rows[-1][1]}  peak memory: {max(r[2] for r in rows):.2f}%')
print(f'steady state (steps {steady[0][0]}-{steady[-1][0]}): median {tps:,.0f} tokens/s = {4 * 8192 / tps * 1000:.1f} ms/step')
print('all losses finite:', all(r[1] == r[1] and abs(r[1]) != float('inf') for r in rows))
"
echo "rc=$RC wall=$((T1 - T0)) s (container start to exit, incl. model init and JIT) log=$LOG"
exit $RC
