#!/bin/bash
# Per-GPU sweep queue. Idempotent, resumable, never runs dry, stops BETWEEN kernels.
#
#   GPU=0 PHASE=e4 bash queue.sh &
#
# Everything here is a lesson from 2026-09-13: the ledger makes a restart free, the STOP
# sentinel means an exclusive window never has to SIGKILL a process with work in flight
# (a suspected contributor to that day's fifth GPU wedge), and the outer loop exists
# because every finite queue that day went idle ~15 min after launch.
set -u
GPU=${GPU:?set GPU}
PHASE=${PHASE:-e4}
ROOT=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo
OUT=$ROOT/output/0914__campaign
LEDGER=${LEDGER:-$OUT/ledgers/g${GPU}.${PHASE}.jsonl}
STOP=${STOP:-/tmp/campaign.g${GPU}.stop}
PIDF=${LEDGER%.jsonl}.pid
CTR=${CTR:-fa-repro}
touch "$LEDGER"; echo $$ > "$PIDF"; trap 'rm -f "$PIDF"' EXIT

# Liveness and health use only calls that cannot block on the driver. rocm-smi, ps and
# pgrep all hang on a wedged card -- dmesg read alone, under a hard timeout, does not.
health(){ timeout 15 dmesg 2>/dev/null | tail -80 \
          | grep -qE 'failed to respond to msg|GPU Hang|wait for reset ack|Memory access fault' && return 1 || return 0; }
paused(){ [ -f "$STOP" ]; }

done_tag(){ grep -qF "\"tag\":\"$1\"" "$LEDGER" 2>/dev/null; }

run(){ # tag, then tune_attention.py args
  local tag="$1"; shift
  while paused; do sleep 10; done
  done_tag "$tag" && return 0
  health || { echo "{\"tag\":\"$tag\",\"halt\":\"dmesg fault\",\"t\":$(date +%s)}" >> "$LEDGER"; sleep 300; return 0; }
  local o
  o=$(timeout 1800 docker exec \
        -e GPU=$GPU \
        -e PYTHONPATH=/home/lihuzhan/code/aiter-src \
        -e TRITON_CACHE_DIR=/tmp/triton_cache_g${GPU} \
        "$CTR" bash -lc 'cd "$0" && exec python3 tools/gfx1250/tune_attention.py "$@"' \
        "$ROOT" "$@" 2>>"$OUT/logs/g${GPU}.${PHASE}.err" | tail -1)
  case "$o" in
    '{'*) echo "{\"tag\":\"$tag\",\"gpu\":$GPU,\"t\":$(date +%s),\"r\":$o}" >> "$LEDGER" ;;
    *)    echo "{\"tag\":\"$tag\",\"gpu\":$GPU,\"t\":$(date +%s),\"r\":null}" >> "$LEDGER" ;;
  esac
}

ROUND=0
while true; do
  ROUND=$((ROUND+1))
  case "$PHASE" in
  combo)
    # The forward grid says num_warps=2 at num_stages=2 beats the shipped 4 by 5.6% on the
    # forward (2.574 vs 2.726). Confirm it on the shipping path, with repeats, because 0.72%
    # on the total is inside the 0.78% run-to-run spread even though the forward half is not.
    for i in 1 2 3 4 5; do
      run "combo|fwd_w2|$i"  --shape llama31-8b --impl fused --tune "fwd:num_stages=2,num_warps=2"
      run "combo|champ|$i"   --shape llama31-8b --impl fused --tune "fwd:num_stages=2"
    done
    # Does it hold at the other production shapes, or is it an s=8192 artefact?
    for S in llama31-8b-b2 llama31-8b-s4096 gate-b1 gate-s16384; do
      run "combo|shape|$S|w2" --shape $S --impl fused --tune "fwd:num_stages=2,num_warps=2"
      run "combo|shape|$S|w4" --shape $S --impl fused --tune "fwd:num_stages=2"
    done
    ;;
  tile)
    # The dkdv/dq tile family at the winning schedule. The pairing constraint is
    # BLOCK_N1==BLOCK_M2 and BLOCK_M1==BLOCK_N2; breaking it does not fault, it returns a dq
    # that is half computed and times faster for it.
    for M in 16 32 64; do
      run "tile|M$M" --shape llama31-8b --impl fused --tune "fwd:num_stages=2" \
          --fused-tune "BLOCK_M1=$M,BLOCK_N2=$M,BLOCK_N1=256,BLOCK_M2=256,BLK_SLICE_FACTOR=1"
    done
    # waves_per_eu is worth one honest pass here and nowhere else: its refutation was measured
    # at 1100 MHz, and memory latency in CYCLES is 2.14x larger at this clock.
    for W in 0 1 2 3; do
      run "tile|wpe$W" --shape llama31-8b --impl fused --tune "fwd:num_stages=2" \
          --fused-tune "waves_per_eu=$W"
    done
    ;;
  sweep2)
    # Fresh grid: the champion's remaining scheduling knobs on the fused backward, at the
    # config that now ships (ASM forward + in-thread transpose).
    for W in 1 2 4; do for S in 1 2; do
      run "s2|bwdw$W|s$S" --shape llama31-8b --impl asm --fused-tune "num_warps=$W,num_stages=$S"
    done; done
    for M in 16 32 64 128; do
      run "s2|nkdim$M" --shape llama31-8b --impl asm --fused-tune "matrix_instr_nonkdim=$M"
    done
    ;;
  shapes2)
    # The production-adjacent shapes on the shipping path, for the dispatcher table.
    for S in gate-s1024 gate-s2048 llama31-8b-s4096 gate-b1 llama31-8b-b2 llama31-8b gate-b8 gate-s16384; do
      run "sh2|$S|asm" --shape $S --impl asm
    done
    ;;
  gate)
    # Shape-gate coverage for the dispatcher: the tile that wins at s=8192 loses at s=1024,
    # so the gate needs the crossover, not the endpoints.
    for S in gate-s1024 gate-s2048 llama31-8b-s4096 gate-b1 llama31-8b-b2 llama31-8b gate-b8 gate-s16384; do
      run "gate|$S|fused" --shape $S --impl fused --tune "fwd:num_stages=2"
      run "gate|$S|turbo" --shape $S --tune "fwd:num_stages=2"
    done
    ;;
  e4)
    # E4: the fused backward's own num_warps x num_stages. Written into day1.sh yesterday
    # and never run. Zero code change, and the offline ISA screen says warps=8 drops VGPR
    # from a pinned 1024 to 512 at the cost of more spill -- which way that lands is
    # exactly what has to be measured rather than argued.
    for W in 2 4 8 16; do for S in 1 2; do
      run "e4|w$W|s$S" --shape llama31-8b --impl fused --tune "fwd:num_stages=2" \
          --fused-tune "num_warps=$W,num_stages=$S"
    done; done
    # Tile family at the winning schedule, respecting the pairing constraint
    # (BLOCK_N1==BLOCK_M2, BLOCK_M1==BLOCK_N2): breaking it is "faster but dq half computed".
    for N in 128 256; do for B in 1 2; do
      run "e4|tile|N$N|B$B" --shape llama31-8b --impl fused --tune "fwd:num_stages=2" \
          --fused-tune "BLOCK_N1=$N,BLOCK_M2=$N,BLK_SLICE_FACTOR=$B"
    done; done
    # Re-anchor the champion each round: a drifting machine shows up here first.
    ;;
  fwd)
    # Forward space. More than half the remaining gap to aiter now lives here
    # (2.734 vs 2.095 ms = 1.31x), which is a today measurement, not yesterday's ranking.
    for ST in 1 2 3 4; do for W in 2 4 8; do
      run "fwd|st$ST|w$W" --shape llama31-8b --impl fused --tune "fwd:num_stages=$ST,num_warps=$W"
    done; done
    ;;
  esac
  echo "{\"round_complete\":$ROUND,\"phase\":\"$PHASE\",\"t\":$(date +%s)}" >> "$LEDGER"
  # Grid exhausted: every tag is now a no-op, so back off instead of spinning.
  [ $ROUND -ge 2 ] && sleep 120 || sleep 5
done
