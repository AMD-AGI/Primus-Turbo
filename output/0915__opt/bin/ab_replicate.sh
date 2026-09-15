#!/bin/bash
# Replicated, order-alternated A/B for the ASM backward.
#
# WHY THIS EXISTS. The first 8-layer A/B (fix2-on / fix2-off) reported ON +3.42% and that
# number is not usable, for three reasons found by adversarial review afterwards:
#
#   1. No seed. torchtitan's set_determinism returns WITHOUT seeding when world_size == 1 and
#      debug.seed is None, and the config set none -- so model init was fresh-random per run.
#      Five runs gave five different step-1 losses (12.12861 / 12.31076 / 12.32626 / 12.33148
#      / 12.29155). The two arms differed in weight init as well as in the switch under test.
#      Fixed by pinning debug.seed in the config.
#   2. n=1 per arm. The 1.20% / 0.77% dispersions quoted as the noise floor are WITHIN-run step
#      jitter. Run-to-run variance -- the thing a 3.42% separation has to clear -- was never
#      measured at all. Fixed by replicating.
#   3. Order confound. ON ran first from cold (its step 1 took 247 s); OFF started 43 s after
#      ON's last step, on an already-hot VR-limited 1100 MHz part. Clock/power drift on this
#      box is of the same order as the effect. Fixed by alternating and reporting per-order.
#
# Alternation does not REMOVE a thermal trend, it balances its first-order term across arms.
# If run k of each arm drifts monotonically, that shows up as a within-arm trend across
# replicates, which is why every replicate's number is printed and not just the mean.
set -u
cd /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo
OUT=output/0915__opt
D=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250
REPS=${REPS:-3}
CFG=${CFG:-repro_l8b_turbo_conv_8L.yaml}

for r in $(seq 1 "$REPS"); do
  for arm in on off; do
    tag="rep${r}-${arm}"
    mkdir -p "/home/lihuzhan/_dbg_l8b/$tag"
    extra=""
    [ "$arm" = "on" ] && extra="-e PRIMUS_TURBO_ATTN_ENABLE_ASM_BWD=1"
    echo "=== $(date +%H:%M:%S) $tag ==="
    BLAS_ENV="-e HIPBLASLT_TENSILE_LIBPATH=$D" \
    E2E_ENV="$extra -e PRIMUS_TURBO_ASM_BWD_TRACE=1 -e PRIMUS_TURBO_ASM_BWD_TRACE_FILE=/home/lihuzhan/_dbg_l8b/$tag/asmbwd.trace" \
      bash $OUT/bin/e2e.sh "$tag" "$CFG"
  done
done
echo "=== all replicates done ==="
