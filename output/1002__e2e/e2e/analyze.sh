#!/bin/bash
# Post-run analysis of the 2026-10-02 A0 e2e (CPU only, host python3, no torch, no GPU, no lock).
#   bash analyze.sh <tags file | tag ...>        (drive.sh calls it with runs/tags.<stamp>.txt)
#   OP="asm_bwd=..,s6=..,r29=..,asm_fwd=..,r16=.." OP_LABEL="real dumps after GEMM burst" bash analyze.sh ...
# Per tag: steady_arms3 (step ms, ratios), attn_events (fwd/bwd attention ms per step from CUDA events),
# trace_breakdown2 (kineto, if the traces are valid), clk_summary; the files that ran (logs/tree.<tag>.sha256)
# against the first tag's -> runs/<tag>.treecmp.txt; then make_table -> runs/TABLE.<stamp>.md
# Arm order is fixed (ORDER, default asm,flyr29,fly): every pair is later/earlier in every process, so p1 and p2
# both report fly/flyr29 = fly - flyr29 (with first-appearance order the two processes reported opposite pairs).
set -u
KIT=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__e2e/e2e
T=$KIT/tools; R=$KIT/runs
ORDER=${ORDER:-asm,flyr29,fly}
if [ $# = 1 ] && [ -f "$1" ]; then TAGS=$(cat "$1"); STAMP=$(basename "$1" .txt | sed 's/^tags\.//'); else TAGS="$*"; STAMP=$(date +%m%d_%H%M); fi
REF=""; REFC=""
for tag in $TAGS; do
  SPEC=$(sed -n 1p $R/$tag.spec); PF=$(sed -n 3p $R/$tag.spec); [ "$PF" = 0 ] && PF=0
  echo "################ $tag  spec '$SPEC' pfreq $PF"
  echo "---- post-run summary"; cat $R/$tag.post.txt 2>/dev/null
  echo "---- nkfix"; head -3 $KIT/logs/nkfix.$tag.txt 2>/dev/null
  echo "---- env evidence"; cat $KIT/logs/env.$tag.txt 2>/dev/null
  echo "---- code that ran (logs/tree.$tag.sha256, config excluded) vs the first process"
  TF=$KIT/logs/tree.$tag.sha256
  if [ ! -s $TF ]; then echo "!! 无快照 logs/tree.$tag.sha256" > $R/$tag.treecmp.txt
  else
    CUR=$(grep -v "/configs/run\.$tag\.yaml\$" $TF | sort -k2)
    if [ -z "$REF" ]; then REF=$tag; REFC=$CUR; echo "基准（$(echo "$CUR" | wc -l) 个文件）" > $R/$tag.treecmp.txt
    else
      D=$(diff <(echo "$REFC") <(echo "$CUR") | grep '^[<>]' | awk '{print $NF}' | sort -u | tr '\n' ' ')
      if [ -z "$D" ]; then echo "同 $REF（$(echo "$CUR" | wc -l) 个文件）" > $R/$tag.treecmp.txt
      else echo "!! 与 $REF 不同：$D" > $R/$tag.treecmp.txt; fi
    fi
  fi
  cat $R/$tag.treecmp.txt
  echo "---- steady state (step ms from tps)"
  python3 $T/steady_arms3.py $KIT/logs/e2e.$tag.log "$SPEC" $PF --order $ORDER --json $R/$tag.steady.json
  echo "---- attention per step (CUDA events)"
  [ -s $KIT/logs/attn_ev.$tag.jsonl ] && python3 $T/attn_events.py $KIT/logs/attn_ev.$tag.jsonl "$SPEC" $PF \
    --log $KIT/logs/e2e.$tag.log --order $ORDER --json $R/$tag.events.json || echo "!! no attention events for $tag"
  echo "---- kineto traces"
  TR=$(ls $KIT/traces/$tag/iteration_*/*.json* 2>/dev/null)
  [ -n "$TR" ] && python3 $T/trace_breakdown2.py --json $R/$tag.trace.json $TR | tee $R/$tag.breakdown.txt | grep -E "^===|VALID|FA path|arm ranges|per stream|^!!" || echo "(no traces)"
  echo "---- clocks"
  python3 $T/clk_summary.py $KIT/logs/clk.$tag.csv --json $R/$tag.clk.json
done
python3 $T/make_table.py $R $TAGS ${OP:+--op "$OP"} ${OP:+--op-label "${OP_LABEL:-op-level}"} > $R/TABLE.$STAMP.md
echo "################ table -> $R/TABLE.$STAMP.md"; cat $R/TABLE.$STAMP.md
