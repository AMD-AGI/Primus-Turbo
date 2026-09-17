#!/bin/bash
# Long-running unattended sweep queue. Keeps the GPU busy for hours without supervision.
# Appends one JSON line per candidate to $LEDGER; re-running skips what is already there.
set -u
cd /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo
LEDGER=${LEDGER:-/tmp/wq.jsonl}
touch "$LEDGER"
H=tools/gfx1250/tune_attention.py
CH="BLOCK_M1=32,BLOCK_N1=256,BLOCK_M2=256,BLOCK_N2=32,BLK_SLICE_FACTOR=1"

done_already() { grep -qF "\"tag\":\"$1\"" "$LEDGER" 2>/dev/null; }
health() { dmesg 2>/dev/null | tail -80 | grep -qE 'MES\(|GPU Hang|wait for reset ack' && return 1 || return 0; }

run() {  # tag, then harness args
  local tag="$1"; shift
  done_already "$tag" && return 0
  health || { echo "{\"tag\":\"$tag\",\"halt\":\"gpu fault in dmesg\"}" >> "$LEDGER"; exit 3; }
  local out; out=$(timeout 600 env PYTHONPATH=/home/lihuzhan/code/aiter-src python3 $H "$@" 2>/dev/null | tail -1)
  case "$out" in '{'*) echo "{\"tag\":\"$tag\",\"r\":$out}" >> "$LEDGER" ;;
                  *)   echo "{\"tag\":\"$tag\",\"r\":null}"  >> "$LEDGER" ;; esac
}

# --- A. shape gate: where does the champion win, and where does it lose? -------------
for S in llama31-8b llama31-8b-s4096 llama31-8b-b2 smoke; do
  run "gate|$S|champ"   --shape $S --impl aiter --aiter-fwd-stages 2 --aiter-bwd-cfg "$CH"
  run "gate|$S|ship"    --shape $S --impl aiter
  run "gate|$S|fused"   --shape $S --impl fused --tune "fwd:num_stages=2"
done

# --- B. N1 vs sequence length: find the crossover that the shape gate needs ----------
for N in 64 128 256; do
  for S in smoke llama31-8b-s4096 llama31-8b; do
    run "n1|$S|$N" --shape $S --impl aiter --aiter-fwd-stages 2 \
        --aiter-bwd-cfg "BLOCK_M1=32,BLOCK_N1=$N,BLOCK_M2=$N,BLOCK_N2=32,BLK_SLICE_FACTOR=1"
  done
done

# --- C. forward: turbo's 4.15 ms vs aiter's 3.26 ms, both Triton ---------------------
for ST in 1 2 3 4; do
  run "fwd|turbo|st$ST"  --shape llama31-8b --tune "fwd:num_stages=$ST;bwd:num_warps=2"
  run "fwd|aiter|st$ST"  --shape llama31-8b --impl aiter --aiter-fwd-stages $ST --aiter-bwd-cfg "$CH"
done
for W in 1 2 4 8; do
  run "fwd|turbo|w$W"    --shape llama31-8b --tune "fwd:num_stages=2,num_warps=$W;bwd:num_warps=2"
done
for PV in true false; do
  run "fwd|turbo|pv$PV"  --shape llama31-8b --tune "fwd:num_stages=2,PRE_LOAD_V=$PV;bwd:num_warps=2"
done

# --- D. fine sweep around the champion on the fused backward ------------------------
for M1 in 16 32 48 64; do
  run "fine|m1$M1" --shape llama31-8b --impl aiter --aiter-fwd-stages 2 \
      --aiter-bwd-cfg "BLOCK_M1=$M1,BLOCK_N1=256,BLOCK_M2=256,BLOCK_N2=$M1,BLK_SLICE_FACTOR=1"
done
for PB in 64 128 256; do
  run "fine|preblock$PB" --shape llama31-8b --impl aiter --aiter-fwd-stages 2 \
      --aiter-bwd-cfg "$CH,PRE_BLOCK=$PB"
done

# --- E. determinism + repeatability of the champion, many reps ----------------------
for i in 1 2 3 4 5; do
  run "rep|$i" --shape llama31-8b --impl aiter --aiter-fwd-stages 2 --aiter-bwd-cfg "$CH"
done
run "det|champ" --shape llama31-8b --impl aiter --aiter-fwd-stages 2 --aiter-bwd-cfg "$CH" --determinism-reps 300

echo "WORK_QUEUE_DONE" >> "$LEDGER"
