#!/bin/bash
# Compile-only matrix, run INSIDE fa-g2 WITHOUT flock (no GPU). Usage: build_all.sh <arm> [l5]
L=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab2/L5
arm=$1; tag=${arm}${2:+_$2}${SEED:+_s$SEED}
mkdir -p $L/isa
for sc in ${CFGS:-prod:causal proxy:causal fast:causal prod:nc gqa1:causal gqa2:causal}; do
  shape=${sc%%:*}; cfg=${sc##*:}; out=$L/isa/${tag}_${shape}_${cfg}
  rm -rf $out /tmp/flycache_l5_${tag}_${shape}_${cfg}
  PYTHONHASHSEED=${SEED:-0} COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250 FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_l5_${tag}_${shape}_${cfg} \
    timeout 900 /opt/venv/bin/python3 $L/tools/compile_l5.py $L/$arm $out $shape $cfg $2 > $out.log 2>&1
  echo "$tag $shape $cfg rc=$? $(tail -1 $out.log)"
done
