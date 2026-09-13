#!/bin/bash
# Self-perpetuating sweep queue. Runs until killed, never runs dry.
#
# The earlier queues were finite (48, 27, 59 candidates) and each went idle ~15 min after
# launch, which left the GPU doing nothing until someone noticed. This one has an outer
# loop: when the planned phases are exhausted it keeps exploring, so "the queue finished"
# stops being a failure mode.
set -u
cd /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo
LEDGER=${LEDGER:-/tmp/forever.jsonl}; touch "$LEDGER"
H=tools/gfx1250/tune_attention.py
export PYTHONPATH=/home/lihuzhan/code/aiter-src

# Graceful stop. Taking an exclusive measurement window by pkill -9 on this queue kills a
# process with work in flight on the GPU, and doing that repeatedly is a plausible
# contributor to the wedge on 2026-09-13 (see phase2/INCIDENT-2026-09-13-wedge.md).
# Touch $STOP_FILE instead: the loop notices BETWEEN candidates, so nothing is killed
# mid-kernel. Remove the file and relaunch to resume -- the ledger makes it idempotent.
STOP_FILE=${STOP_FILE:-/tmp/wq.stop}
stop_requested(){ [ -f "$STOP_FILE" ]; }
wait_if_stopped(){
  while stop_requested; do
    echo "{\"paused\":1,\"t\":$(date +%s)}" >> "$LEDGER"
    sleep 10
  done
}

d(){ grep -qF "\"tag\":\"$1\"" "$LEDGER" 2>/dev/null; }
health(){ dmesg 2>/dev/null|tail -60|grep -qE 'MES\(|GPU Hang|wait for reset ack' && return 1||return 0; }
run(){ local t="$1";shift; wait_if_stopped; d "$t"&&return 0
  health||{ echo "{\"tag\":\"$t\",\"halt\":\"dmesg fault\",\"t\":$(date +%s)}">>"$LEDGER"; sleep 300; return 0; }
  local o;o=$(timeout 1200 python3 $H "$@" 2>/dev/null|tail -1)
  case "$o" in '{'*) echo "{\"tag\":\"$t\",\"r\":$o}">>"$LEDGER";; *) echo "{\"tag\":\"$t\",\"r\":null}">>"$LEDGER";; esac; }

SHAPES="gate-s1024 gate-s2048 llama31-8b-s4096 gate-b1 llama31-8b-b2 llama31-8b gate-b8 gate-s16384"
ROUND=0
while true; do
  ROUND=$((ROUND+1))
  # P1 -- gate table, vendored path, every shape x every N1 x BLK_SLICE
  for S in $SHAPES; do for N in 64 128 256; do for B in 1 2; do
    run "P1|$S|N$N|B$B" --shape $S --impl fused --tune "fwd:num_stages=2" \
        --fused-tune "BLOCK_N1=$N,BLOCK_M2=$N,BLK_SLICE_FACTOR=$B"
  done; done; done
  # P2 -- M1/N2 at the winning N1, per shape
  for S in $SHAPES; do for M in 16 32 64; do
    run "P2|$S|M$M" --shape $S --impl fused --tune "fwd:num_stages=2" \
        --fused-tune "BLOCK_M1=$M,BLOCK_N2=$M,BLOCK_N1=256,BLOCK_M2=256,BLK_SLICE_FACTOR=1"
  done; done
  # P3 -- forward space, both implementations
  for ST in 1 2 3 4; do for W in 2 4 8; do
    run "P3|turbo|st$ST|w$W" --shapes llama31-8b,llama31-8b-b2 --tune "fwd:num_stages=$ST,num_warps=$W;bwd:num_warps=2"
    run "P3|aiter|st$ST|w$W" --shape llama31-8b --impl aiter --aiter-fwd-stages $ST \
        --aiter-bwd-cfg "BLOCK_M1=32,BLOCK_N1=256,BLOCK_M2=256,BLOCK_N2=32,BLK_SLICE_FACTOR=1"
  done; done
  # P4 -- periodic determinism + repeatability on the shipping path
  run "P4|det|r$ROUND" --shape llama31-8b --impl fused --tune "fwd:num_stages=2" --determinism-reps 200
  for i in 1 2 3; do run "P4|rep|r$ROUND|$i" --shapes llama31-8b,llama31-8b-b2,llama31-8b-s4096 \
      --impl fused --tune "fwd:num_stages=2"; done
  echo "{\"round_complete\":$ROUND,\"t\":$(date +%s)}" >> "$LEDGER"
  # Nothing new was added this round -> widen rather than exit.
  SHAPES="$SHAPES smoke"
done
