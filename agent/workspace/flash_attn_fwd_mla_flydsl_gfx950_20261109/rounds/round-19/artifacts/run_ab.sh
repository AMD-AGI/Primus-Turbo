#!/bin/bash
# Round 19 A/B: r18 baseline vs independent-section rewrite (working) vs r18's (16,8) build
# (codegen reference at S4K / S8K where CTA order matches). Full set with aiter.
C=/perf_apps/xiaoming/Primus-Turbo/agent/workspace/flash_attn_fwd_mla_flydsl_gfx950_20261109
A=$C/rounds/round-19/artifacts
declare -A PKG=(
    [r18]=$C/profiles/pkg_r18
    [r19]=/perf_apps/xiaoming/megaattention_overlap_bench
    [g16]=$C/profiles/variants/r18_g16_8
)
n=0
for order in "r18 r19 g16" "g16 r19 r18"; do
    n=$((n + 1))
    for tag in $order; do
        echo "=== $tag run $n $(date +%H:%M:%S)"
        MLA_PKG_ROOT=${PKG[$tag]} FLYDSL_RUNTIME_CACHE_DIR=/tmp/abcache19_$tag HIP_VISIBLE_DEVICES=0 \
            python $C/quick_test_bench.py --set full --summary-csv $A/ab_full_${tag}_$n.csv \
            > $A/ab_full_${tag}_$n.log 2>&1
        grep -E "PASS|FAIL|Geomean" $A/ab_full_${tag}_$n.log
    done
done
echo "=== done $(date +%H:%M:%S)"
