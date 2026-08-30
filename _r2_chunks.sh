#!/usr/bin/env bash
# Campaign 20260830_154052, optimize round 1: RE-PRICE the batch-chunk cut now that native dQ
# has deleted the un-permute pass. The deployed count (4 at b4) was chosen when each chunk
# after the first hosted an un-permute AND a delta; with the un-permute gone, chunks 2..4 host
# nothing, so the cut is paying its boundary cost for a rider that no longer exists.
#
# usage: _r2_chunks.sh <B> <Hq> <Hkv> <S> <D> <reps>
set -u
B=$1; HQ=$2; HKV=$3; S=$4; D=$5; R=$6
export FLYDSL_RUNTIME_ENABLE_CACHE=0

run() {
  printf '%-34s ' "$1"
  python -u _probe_bwdcfg.py "$B" "$HQ" "$HKV" "$S" "$D" "$1" "$R" 2>/dev/null | tail -1
}

echo "=== cell ${B}/${HQ}/${HKV}/${S}/${D} reps=${R} ==="
for arm in 'mod:_A16_BAT_CHUNKS=2' 'mod:_A16_BAT_CHUNKS=1' 'mod:_A16_BLOCK_KV=128' \
           'mod:_A16_BAT_CHUNKS=1' 'mod:_A16_BAT_CHUNKS=2'; do
  run base
  run "$arm"
done
run base
