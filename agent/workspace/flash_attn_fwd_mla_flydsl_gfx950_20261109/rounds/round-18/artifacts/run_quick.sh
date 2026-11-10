#!/bin/bash
# Round 18 quick-set rotations (no aiter): each package takes every position once.
C=/perf_apps/xiaoming/Primus-Turbo/agent/workspace/flash_attn_fwd_mla_flydsl_gfx950_20261109
A=$C/rounds/round-18/artifacts
declare -A PKG=(
    [r17]=$C/profiles/pkg_r17
    [g32]=/perf_apps/xiaoming/megaattention_overlap_bench
    [g16]=$C/profiles/variants/r18_g16_8
)
n=0
for order in "r17 g32 g16" "g32 g16 r17" "g16 r17 g32"; do
    n=$((n + 1))
    for tag in $order; do
        MLA_PKG_ROOT=${PKG[$tag]} FLYDSL_RUNTIME_CACHE_DIR=/tmp/abcache18_$tag HIP_VISIBLE_DEVICES=0 \
            python $C/quick_test_bench.py --set quick --no-aiter 2>&1 |
            awk -v t=$tag -v r=$n '/PASS|FAIL/ {printf "%s r%s %s %s %s\n", t, r, $1, $2, $4} /^Geomean Fwd/ {printf "%s r%s geomean %s\n", t, r, $NF}'
    done
done
