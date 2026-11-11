#!/bin/bash
# Full set, rotated: r19 (working tree), r18 (round-18 package), r19, r18, r19. GPU 0, n02-29.
C=/perf_apps/xiaoming/Primus-Turbo/agent/workspace/flash_attn_bwd_mla_flydsl_gfx950_20261109
A=$C/rounds/round-19/artifacts
export HIP_VISIBLE_DEVICES=0
for t in r19:1 r18:1 r19:2 r18:2 r19:3; do
  n=${t%%:*}; i=${t#*:}
  if [ $n = r18 ]; then root=$C/rounds/round-18/pkg; else root=/perf_apps/xiaoming/megaattention_overlap_bench; fi
  MLA_PKG_ROOT=$root python $C/quick_test_bench.py --set full --summary-csv $A/full_${n}_run${i}.csv > $A/full_${n}_run${i}.raw 2>&1
  echo "$(date +%H:%M:%S) $n run$i rc=$?"
done
echo ALL_DONE
