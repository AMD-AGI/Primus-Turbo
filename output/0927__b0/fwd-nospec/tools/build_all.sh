#!/bin/bash
# Compile-only matrix INSIDE fa-g0 WITHOUT flock (no GPU). Usage: build_all.sh <arm>
L=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/fwd-nospec
arm=$1
for c in ${CFGS:-m32x8:4:causal m32x8:4:nc m32x8:1:causal m32x8:2:causal m32x2:4:causal m32x2:4:nc m32x2:1:causal m32x2:2:causal}; do
  IFS=: read mod g m <<<"$c"; out=$L/isa/${arm}_${mod}_g${g}_${m}; cache=/tmp/flycache_ns_${arm}_${mod}_${g}_${m}
  rm -rf $out $cache
  PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250 FLYDSL_RUNTIME_CACHE_DIR=$cache \
    timeout 900 /opt/venv/bin/python3 $L/tools/compile_arm.py $L/arms/$arm $out $mod $g $m > $out.log 2>&1
  echo "$arm $mod g$g $m rc=$? $(tail -1 $out.log)"
done
