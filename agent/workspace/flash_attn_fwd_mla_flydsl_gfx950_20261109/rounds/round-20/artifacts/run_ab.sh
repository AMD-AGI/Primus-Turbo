#!/bin/bash
# Round 20 A/B: r18 baseline vs diagonal dead-half skip on max_m4 HI / max_m3 / max_m2 LO (working).
# Quick set rotated three times (no aiter), then the full set (with aiter) alternated twice.
C=/perf_apps/xiaoming/Primus-Turbo/agent/workspace/flash_attn_fwd_mla_flydsl_gfx950_20261109
A=$C/rounds/round-20/artifacts
declare -A PKG=(
    [r18]=$C/profiles/pkg_r18
    [r20]=/perf_apps/xiaoming/megaattention_overlap_bench
)
n=0
for order in "r18 r20" "r20 r18" "r18 r20"; do
    n=$((n + 1))
    for tag in $order; do
        MLA_PKG_ROOT=${PKG[$tag]} FLYDSL_RUNTIME_CACHE_DIR=/tmp/abcache20_$tag HIP_VISIBLE_DEVICES=0 \
            python $C/quick_test_bench.py --set quick --no-aiter 2>&1 |
            awk -v t=$tag -v r=$n '/PASS|FAIL/ {printf "%s r%s %s %s %s\n", t, r, $1, $2, $4} /^Geomean Fwd/ {printf "%s r%s geomean %s\n", t, r, $NF}'
    done
done | tee $A/ab_quick.txt
n=0
for order in "r18 r20" "r20 r18"; do
    n=$((n + 1))
    for tag in $order; do
        echo "=== $tag run $n $(date +%H:%M:%S)"
        MLA_PKG_ROOT=${PKG[$tag]} FLYDSL_RUNTIME_CACHE_DIR=/tmp/abcache20_$tag HIP_VISIBLE_DEVICES=0 \
            python $C/quick_test_bench.py --set full --summary-csv $A/ab_full_${tag}_$n.csv \
            > $A/ab_full_${tag}_$n.log 2>&1
        grep -E "PASS|FAIL|Geomean" $A/ab_full_${tag}_$n.log
    done
done | tee $A/ab_full.txt
echo "=== done $(date +%H:%M:%S)"
