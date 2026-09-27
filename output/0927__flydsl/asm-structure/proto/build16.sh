#!/bin/bash
# usage: build.sh <gate:on|off> <tag> [prod|thd|win_sink]; env SHAPE GQA CAUSAL D DT pass through.
# Copies ../op to variants/<gate> with ASM_STRUCT set, compiles (COMPILE_ONLY, GPU hidden), prints resources.
P=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__flydsl/asm-structure/proto
G=$1; TAG=$2; VAR=${3:-prod}
V=$P/variants/$G
if [ $G != champ ] && { [ ! -d $V ] || [ -n "$REFRESH" ]; }; then
  rm -rf $V; mkdir -p $P/variants; cp -r $P/../op $V; rm -rf $V/__pycache__ $V/flydsl_fwd/__pycache__
  if [ $G = on ]; then val=True; else val=False; fi
  sed -i "s/^ASM_STRUCT = \(True\|False\)$/ASM_STRUCT = $val/" $V/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py
fi
[ $G = champ ] || grep -q "^ASM_STRUCT = " $V/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py || { echo "gate line missing"; exit 1; }
rm -rf $P/isa/$TAG
docker exec fa-repro bash -c "cd $P && env COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250 HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_asmp2_$TAG SHAPE=${SHAPE:-prod} GQA=${GQA:-4} CAUSAL=${CAUSAL:-1} D=${D:-128} DT=${DT:-bf16} /opt/venv/bin/python3 ${RC:-rc.py} $V isa/$TAG $VAR" > $P/logs/$TAG.log 2>&1
rc=$?
f=$(ls $P/isa/$TAG/kn_*/22_final_isa.s 2>/dev/null | head -1)
[ -n "$f" ] && cp $f $P/isa/$TAG.s
echo "$TAG rc=$rc $(grep -E '^\s+\.(vgpr_count|sgpr_count|vgpr_spill_count|sgpr_spill_count|private_segment_fixed_size|group_segment_fixed_size|max_flat_workgroup_size):' $f 2>/dev/null | tr -s ' ' | tr '\n' ' ') $(tail -1 $P/logs/$TAG.log)"
