#!/bin/bash
# Repeat the prod causal compile N times per arm to measure backend nondeterminism. In fa-g2, no flock.
L=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab2/L5
arm=$1; n=$2; extra=$3; mkdir -p $L/isa_rep
for i in $(seq 1 $n); do out=$L/isa_rep/${arm}${extra:+_$extra}_r$i; rm -rf $out /tmp/flycache_l5rep
  COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250 FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_l5rep \
    timeout 900 ${WRAP} /opt/venv/bin/python3 $L/tools/compile_l5.py $L/$arm $out prod causal $extra > $out.log 2>&1
  f=$(ls $out/*/22_final_isa.s); echo "$arm$extra r$i $(grep -v '^\s*\.\(file\|ident\)' $f | md5sum | cut -c1-12) $(md5sum < $(ls $out/*/21_llvm_ir.ll) | cut -c1-12)"
done
