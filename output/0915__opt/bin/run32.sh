#!/bin/bash
# The 32-layer question, run in the order that fails cheapest first.
#
# That config sits at 88% memory and has taken the card down twice, so this does NOT just run
# it. b=2 first (half the activations) to prove the wgrad rule survives a real 32-layer model
# at all; then the unpatched 32L baseline, which is needed anyway because the existing one is
# unseeded; then 32L patched. Every run aborts the whole queue if memory crosses 85% or a
# card-fault signature appears -- the SIGBUS that cost an AC-cycle came from +0.32% on a run
# already at 87.98%, so there is no margin to spend on optimism.
set -u
cd /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo
D=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250
PP=/home/lihuzhan/_dbg_l8b/patch:/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo:/home/lihuzhan/code/aiter-src
M0=$(sudo -n dmesg 2>/dev/null | grep -cE "ring buffer is full|failed to respond|wait for reset ack|SIGBUS")

run() { # tag cfg nkfix(0|1)
  local tag=$1 cfg=$2 nk=$3 env=""
  [ "$nk" = "1" ] && env="-e NKFIX_ENABLE=1 -e NKFIX_STATS_FILE=/home/lihuzhan/_dbg_l8b/$tag.stats"
  mkdir -p "/home/lihuzhan/_dbg_l8b/$tag"
  E2E_TIMEOUT=1500 E2E_INNER=1440 E2E_COOLDOWN=25 \
  BLAS_ENV="-e HIPBLASLT_TENSILE_LIBPATH=$D" \
  E2E_ENV="-e PYTHONPATH=$PP $env" \
    bash output/0915__opt/bin/e2e.sh "$tag" "$cfg" >/dev/null 2>&1
  local L=output/0915__opt/logs/e2e.$tag.log
  local TPS MEM MES
  TPS=$(sed 's/\x1b\[[0-9;]*m//g' "$L" 2>/dev/null | grep -oE "tps: +[0-9,]+" | tail -1)
  MEM=$(sed 's/\x1b\[[0-9;]*m//g' "$L" 2>/dev/null | grep -oE "\([0-9.]+%\)" | tr -d '()%' | sort -rn | head -1)
  MES=$(sudo -n dmesg 2>/dev/null | grep -cE "ring buffer is full|failed to respond|wait for reset ack|SIGBUS")
  echo "$tag: ${TPS:-NO-STEPS}  peakmem=${MEM:-?}%  MES=$MES"
  if [ "$MES" -gt "$M0" ]; then echo ">>> CARD FAULT, aborting queue"; return 1; fi
  if [ -n "$MEM" ] && awk "BEGIN{exit !($MEM>85)}"; then echo ">>> MEM ${MEM}% > 85%, aborting queue"; return 1; fi
  return 0
}

run b2-nk   repro_l8b_turbo_conv_b2.yaml 1 || exit 1
run 32-base repro_l8b_turbo_conv.yaml    0 || exit 1
run 32-nk   repro_l8b_turbo_conv.yaml    1 || exit 1
echo "=== 32L queue done ==="
