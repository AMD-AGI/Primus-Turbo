#!/bin/bash
# Round 18 A/B: r17 baseline vs XCD_Q_GROUPS (32, 16, 8) (working copy) vs (16, 8), full set with aiter,
# forward order then reversed, same session.
C=/perf_apps/xiaoming/Primus-Turbo/agent/workspace/flash_attn_fwd_mla_flydsl_gfx950_20261109
A=$C/rounds/round-18/artifacts
declare -A PKG=(
    [r17]=$C/profiles/pkg_r17
    [g32]=/perf_apps/xiaoming/megaattention_overlap_bench
    [g16]=$C/profiles/variants/r18_g16_8
)
n=0
for order in "r17 g32 g16" "g16 g32 r17"; do
    n=$((n + 1))
    for tag in $order; do
        echo "=== $tag run $n $(date +%H:%M:%S)"
        MLA_PKG_ROOT=${PKG[$tag]} FLYDSL_RUNTIME_CACHE_DIR=/tmp/abcache18_$tag HIP_VISIBLE_DEVICES=0 \
            python $C/quick_test_bench.py --set full --summary-csv $A/ab_full_${tag}_$n.csv \
            > $A/ab_full_${tag}_$n.log 2>&1
        grep -E "PASS|FAIL|Geomean" $A/ab_full_${tag}_$n.log
    done
done
echo "=== done $(date +%H:%M:%S)"
