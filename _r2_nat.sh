#!/usr/bin/env bash
# Campaign 20260830_154052, optimize round 1: price the SWAPPED-OPERAND native dQ against the
# deployed image path on the clean probe channel. `base` is now native (_A16_NAT default 2),
# so the control arm is the subtraction `mod:_A16_NAT=0` -- the image plus its un-permute pass.
#
# usage: _r2_nat.sh <B> <Hq> <Hkv> <S> <D> <reps>
#
# Palindromic with a control between every pair so the box's within-batch drift interpolates
# out, and one canary so a dead arm cannot read as a tie.
set -u
B=$1; HQ=$2; HKV=$3; S=$4; D=$5; R=$6
export FLYDSL_RUNTIME_ENABLE_CACHE=0

run() {
  printf '%-34s ' "$1"
  python -u _probe_bwdcfg.py "$B" "$HQ" "$HKV" "$S" "$D" "$1" "$R" 2>/dev/null | tail -1
}

echo "=== cell ${B}/${HQ}/${HKV}/${S}/${D} reps=${R} ==="
for arm in base base 'mod:_A16_BLOCK_KV=128' base base; do
  run 'mod:_A16_NAT=0'
  run "$arm"
done
run 'mod:_A16_NAT=0'
