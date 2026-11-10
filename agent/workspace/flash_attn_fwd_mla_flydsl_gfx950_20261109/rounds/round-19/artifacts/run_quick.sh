#!/bin/bash
C=/perf_apps/xiaoming/Primus-Turbo/agent/workspace/flash_attn_fwd_mla_flydsl_gfx950_20261109
A=$C/rounds/round-19/artifacts
declare -A PKG=(
    [r18]=$C/profiles/pkg_r18
    [r19]=/perf_apps/xiaoming/megaattention_overlap_bench
    [g16]=$C/profiles/variants/r18_g16_8
)
n=0
for order in "r18 r19 g16" "r19 g16 r18" "g16 r18 r19"; do
    n=$((n + 1))
    for tag in $order; do
        MLA_PKG_ROOT=${PKG[$tag]} FLYDSL_RUNTIME_CACHE_DIR=/tmp/abcache19_$tag HIP_VISIBLE_DEVICES=0 \
            python $C/quick_test_bench.py --set quick --no-aiter 2>&1 |
            awk -v t=$tag -v r=$n '/PASS|FAIL/ {printf "%s r%s %s %s %s\n", t, r, $1, $2, $4} /^Geomean Fwd/ {printf "%s r%s geomean %s\n", t, r, $NF}'
    done
done
