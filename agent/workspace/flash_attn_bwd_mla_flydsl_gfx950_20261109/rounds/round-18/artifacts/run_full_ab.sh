#!/bin/bash
# Full set, rotated: r18 (working tree), r16 (round-16 package), r18, r16, r18. GPU 0, n01-33.
C=/perf_apps/xiaoming/Primus-Turbo/agent/workspace/flash_attn_bwd_mla_flydsl_gfx950_20261109
A=$C/rounds/round-18/artifacts
export HIP_VISIBLE_DEVICES=0
for t in r18:1 r16:1 r18:2 r16:2 r18:3; do
  n=${t%%:*}; i=${t#*:}
  if [ $n = r16 ]; then root=$C/rounds/round-16/pkg; else root=/perf_apps/xiaoming/megaattention_overlap_bench; fi
  MLA_PKG_ROOT=$root python $C/quick_test_bench.py --set full --summary-csv $A/full_${n}_run${i}.csv > $A/full_${n}_run${i}.raw 2>&1
  echo "$(date +%H:%M:%S) $n run$i rc=$?"
done
echo ALL_DONE
