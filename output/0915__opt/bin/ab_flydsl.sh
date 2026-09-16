#!/bin/bash
# Order-alternated A/B for nkfix rule 3: FlyDSL TN for wgrad, versus rule 2's two copies.
#
# Both arms carry NKFIX_ENABLE=1 and the ASM backward, so the ONLY difference is which path
# the wgrad mm takes. Rule 3 needs the FlyDSL gfx1250 GEMM, so this must run with
# tmp/flydsl-gemm checked out -- the training imports primus_turbo from the working tree.
#
# Operator-level measurement said rule 3 should be worth about 58 ms of a 760 ms step (7.6%),
# from 216.6 ms of wgrad becoming 158.6 ms. That is an estimate built from per-shape
# throughput, not a measurement of the step, which is what this script is for. Note the
# estimate already cost one downward revision (77 -> 58 ms) when lm_head was measured instead
# of extrapolated.
set -u
cd /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo
OUT=output/0915__opt
D=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250
PP=/home/lihuzhan/_dbg_l8b/patch:/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo:/home/lihuzhan/code/aiter-src
REPS=${REPS:-3}
CFG=${CFG:-repro_l8b_turbo_conv_8L.yaml}

for r in $(seq 1 "$REPS"); do
  for arm in fly copy; do
    tag="fly${r}-${arm}"
    mkdir -p "/home/lihuzhan/_dbg_l8b/$tag"
    extra=""
    # The table is built offline by flydsl_table.py on an idle card. Rule 3 reads it and
    # never tunes; a shape with no entry falls through to rule 2 (counted as
    # flydsl_no_config in the stats file, so a silent fallback is visible afterwards).
    [ "$arm" = "fly" ] && extra="-e NKFIX_FLYDSL_WGRAD=1 -e NKFIX_FLYDSL_TABLE=/home/lihuzhan/_dbg_l8b/flydsl_tn.json"
    echo "=== $(date +%H:%M:%S) $tag ==="
    BLAS_ENV="-e HIPBLASLT_TENSILE_LIBPATH=$D" \
    E2E_ENV="-e PYTHONPATH=$PP -e NKFIX_ENABLE=1 -e PRIMUS_TURBO_ATTN_ENABLE_ASM_BWD=1 $extra \
             -e NKFIX_STATS_FILE=/home/lihuzhan/_dbg_l8b/$tag.stats" \
      bash $OUT/bin/e2e.sh "$tag" "$CFG" >/dev/null 2>&1
    L=$OUT/logs/e2e.$tag.log
    echo "  tps=$(sed 's/\x1b\[[0-9;]*m//g' $L 2>/dev/null | grep -oE 'tps: +[0-9,]+' | tail -1) \
stats=$(grep -o "'wgrad_flydsl': [0-9]*" /home/lihuzhan/_dbg_l8b/$tag.stats 2>/dev/null)"
  done
done
echo "=== done ==="
