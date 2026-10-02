#!/bin/bash
# Post-run analysis of the 2026-10-02 A0 e2e (CPU only, host python3, no torch, no GPU, no lock).
#   bash analyze.sh <tags file | tag ...>        (drive.sh calls it with runs/tags.<stamp>.txt)
#   OP="asm_bwd=..,s6=..,r29=..,asm_fwd=..,r16=.." OP_LABEL="real dumps after GEMM burst" bash analyze.sh ...
# Per tag: steady_arms3 (step ms, ratios), attn_events (fwd/bwd attention ms per step from CUDA events),
# trace_breakdown2 (kineto, if the traces are valid), clk_summary; then make_table -> runs/TABLE.<stamp>.md
set -u
KIT=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__e2e/e2e
T=$KIT/tools; R=$KIT/runs
if [ $# = 1 ] && [ -f "$1" ]; then TAGS=$(cat "$1"); STAMP=$(basename "$1" .txt | sed 's/^tags\.//'); else TAGS="$*"; STAMP=$(date +%m%d_%H%M); fi
for tag in $TAGS; do
  SPEC=$(sed -n 1p $R/$tag.spec); PF=$(sed -n 3p $R/$tag.spec); [ "$PF" = 0 ] && PF=0
  echo "################ $tag  spec '$SPEC' pfreq $PF"
  echo "---- post-run summary"; cat $R/$tag.post.txt 2>/dev/null
  echo "---- nkfix"; head -3 $KIT/logs/nkfix.$tag.txt 2>/dev/null
  echo "---- env evidence"; cat $KIT/logs/env.$tag.txt 2>/dev/null
  echo "---- steady state (step ms from tps)"
  python3 $T/steady_arms3.py $KIT/logs/e2e.$tag.log "$SPEC" $PF --json $R/$tag.steady.json
  echo "---- attention per step (CUDA events)"
  [ -s $KIT/logs/attn_ev.$tag.jsonl ] && python3 $T/attn_events.py $KIT/logs/attn_ev.$tag.jsonl "$SPEC" $PF \
    --log $KIT/logs/e2e.$tag.log --json $R/$tag.events.json || echo "!! no attention events for $tag"
  echo "---- kineto traces"
  TR=$(ls $KIT/traces/$tag/iteration_*/*.json* 2>/dev/null)
  [ -n "$TR" ] && python3 $T/trace_breakdown2.py --json $R/$tag.trace.json $TR | tee $R/$tag.breakdown.txt | grep -E "^===|VALID|FA path|arm ranges|per stream|^!!" || echo "(no traces)"
  echo "---- clocks"
  python3 $T/clk_summary.py $KIT/logs/clk.$tag.csv --json $R/$tag.clk.json
done
python3 $T/make_table.py $R $TAGS ${OP:+--op "$OP"} ${OP:+--op-label "${OP_LABEL:-op-level}"} > $R/TABLE.$STAMP.md
echo "################ table -> $R/TABLE.$STAMP.md"; cat $R/TABLE.$STAMP.md
