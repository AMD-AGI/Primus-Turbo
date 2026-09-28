#!/bin/bash
# usage: run.sh <arm> <variant: prod|nc> <shape: prod|proxy|fast|mha>   (compile-only, fa-g2, no flock: no GPU)
ARM=$1; VAR=$2; SHAPE=$3
ROOT=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab2/L22_L23
OUT=$ROOT/isa/$ARM/${VAR}_${SHAPE}
rm -rf "${OUT:?}"; mkdir -p "$OUT"
docker exec -u 12850:12850 -e HOME=/tmp fa-g2 bash -c "cd $ROOT/tools && env COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250 FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_l2223_${ARM}_${VAR}_${SHAPE} FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=$OUT /opt/venv/bin/python3 compile_isa.py $ROOT/$ARM $OUT $VAR $SHAPE" > "$OUT/compile.log" 2>&1
RC=$?
S=$(find "$OUT" -name '*final_isa.s' | head -1)
if [ -n "$S" ]; then
  V=$(grep -m1 '\.vgpr_count:' "$S" | awk '{print $2}'); SG=$(grep -m1 '\.sgpr_count:' "$S" | awk '{print $2}')
  VS=$(grep -m1 '\.vgpr_spill_count:' "$S" | awk '{print $2}'); SS=$(grep -m1 '\.sgpr_spill_count:' "$S" | awk '{print $2}')
  SC=$(grep -m1 '\.private_segment_fixed_size:' "$S" | awk '{print $2}'); MD=$(md5sum "$S" | cut -c1-8)
  echo "$ARM $VAR $SHAPE rc=$RC vgpr=$V sgpr=$SG vspill=$VS sspill=$SS scratch=$SC md5=$MD"
else
  echo "$ARM $VAR $SHAPE rc=$RC NO_ISA $(tail -1 $OUT/compile.log)"
fi
