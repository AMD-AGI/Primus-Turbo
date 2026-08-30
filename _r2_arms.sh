#!/usr/bin/env bash
# Campaign 20260830_154052, optimize round 1: the arms that ride ON TOP of the swapped-operand
# native dQ (which `base` now is). All of them are sub-1% items by construction, so they are
# priced here and only the survivors go on the scored ruler as part of the stack.
#
#   mod:_EXP_IGLP_AT=1   hand MFMAExpInterleave the whole head-step instead of one q-half
#   kw:mfma_tie=0/1/2    the accvgpr staging tax, re-priced on the current allocation
#   kw:dv_pin=1          dV accumulator home pinned back into the arch heap
#
# usage: _r2_arms.sh <B> <Hq> <Hkv> <S> <D> <reps>
set -u
B=$1; HQ=$2; HKV=$3; S=$4; D=$5; R=$6
export FLYDSL_RUNTIME_ENABLE_CACHE=0

run() {
  printf '%-34s ' "$1"
  python -u _probe_bwdcfg.py "$B" "$HQ" "$HKV" "$S" "$D" "$1" "$R" 2>/dev/null | tail -1
}

echo "=== cell ${B}/${HQ}/${HKV}/${S}/${D} reps=${R} ==="
for arm in 'mod:_EXP_IGLP_AT=1' 'kw:mfma_tie=0' 'kw:dv_pin=1' 'mod:_A16_BLOCK_KV=128' \
           'kw:dv_pin=1' 'kw:mfma_tie=0' 'mod:_EXP_IGLP_AT=1'; do
  run base
  run "$arm"
done
run base
