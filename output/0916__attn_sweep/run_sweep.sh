#!/bin/bash
# Triton TWO-KERNEL attention sweep, gfx1250. Run INSIDE the container, on an exclusive GPU.
#   docker exec -it fa-repro bash -lc 'bash /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0916__attn_sweep/run_sweep.sh r0'
# Rounds are separate arguments so a round can be re-run without touching the others.
set -eu
REPO=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo
OUT=$REPO/output/0916__attn_sweep
SHAPE=${SHAPE:-llama31-8b-s4096}
cd "$REPO"

export PYTHONPATH=$REPO:${PYTHONPATH:-}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache_attn}
export TORCH_BLAS_PREFER_HIPBLASLT=0
export GPU=${GPU:-0}
# Every one of these is read at import or at dispatch and would silently change WHICH
# kernel is measured. None of them may be inherited from the shell that launches this.
unset PRIMUS_TURBO_ATTN_TRITON_TUNE || true
unset PRIMUS_TURBO_ATTN_ENABLE_ASM_BWD || true
unset PRIMUS_TURBO_ATTN_DKDV_BLOCKS || true
unset PRIMUS_TURBO_FUSED_MHA_BWD_TUNE || true
unset PRIMUS_TURBO_ATTN_DISABLE_ASM_FWD || true

python -c 'import primus_turbo,sys; p=primus_turbo.__file__; sys.exit(0 if p.startswith("'"$REPO"'") else (print("WRONG CHECKOUT:",p) or 1))'
mkdir -p "$OUT"

# --bwd-path twokernel is MANDATORY: without it flash_attn_interface.py:387 sends the
# backward to dense_fused_backward and every bwd: candidate returns the same time.
# --asm-fwd off is belt and braces for the forward half.
HA="--harness-arg --bwd-path --harness-arg twokernel --harness-arg --asm-fwd --harness-arg off"
SW="python3 tools/gfx1250/sweep_attention.py --shape $SHAPE"

case "${1:?usage: run_sweep.sh r0|r1|r2|r3|r4|r5 [pins]}" in
r0)  # plumbing gate: bwd:num_warps must show a large spread, or STOP.
  $SW --baseline --axis bwd:num_warps=2,4,8 $HA --ledger "$OUT/r0_gate.jsonl" ;;
r1)  # backward scheduling grid
  $SW --axis bwd:num_warps=1,2,4,8,16 --axis bwd:num_stages=1,2 $HA --ledger "$OUT/r1_bwd.jsonl" ;;
r2)  # forward scheduling grid
  $SW --axis fwd:num_warps=1,2,4,8 --axis fwd:num_stages=1,2,3 $HA --ledger "$OUT/r2_fwd.jsonl" ;;
r3)  # waves_per_eu on the winners. BW/BS from r1, FW/FS from r2.
  : "${BW:?set BW/BS from r1}" "${BS:?}" "${FW:?set FW/FS from r2}" "${FS:?}"
  $SW --axis bwd:num_warps=$BW --axis bwd:num_stages=$BS --axis bwd:waves_per_eu=0,1,2 \
      $HA --ledger "$OUT/r3_bwd_wpe.jsonl"
  $SW --axis fwd:num_warps=$FW --axis fwd:num_stages=$FS --axis fwd:waves_per_eu=0,1,2 \
      $HA --ledger "$OUT/r3_fwd_wpe.jsonl" ;;
r4)  # PRE_LOAD_V is FORWARD ONLY: an unprefixed axis puts it on the bwd kernels, which
     # have no such parameter, and triton 3.6 raises KeyError at launch.
  : "${FW:?}" "${FS:?}"
  $SW --axis fwd:num_warps=$FW --axis fwd:num_stages=$FS --axis fwd:PRE_LOAD_V=True,False \
      $HA --ledger "$OUT/r4_preloadv.jsonl" ;;
r5)  # dk/dv tile. NOT an --axis: it is a separate env var that the ledger does not record,
     # so each value needs its own ledger or the resume key (tune string) collides.
  : "${BW:?}" "${BS:?}"
  for TILE in 64x64 32x128 32x64 64x128 128x64; do
    M=${TILE%x*}; N=${TILE#*x}
    PRIMUS_TURBO_ATTN_DKDV_BLOCKS="M=$M,N=$N" \
      $SW --axis bwd:num_warps=$BW --axis bwd:num_stages=$BS $HA \
          --ledger "$OUT/r5_dkdv_${TILE}.jsonl"
  done ;;
*) echo "unknown round"; exit 2 ;;
esac
