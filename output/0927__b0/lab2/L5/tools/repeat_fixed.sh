#!/bin/bash
# Same as repeat.sh but the dump dir is the SAME path every run (tests path-dependence).
L=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab2/L5
arm=$1; n=$2; out=$L/isa_rep/${arm}_fixed
for i in $(seq 1 $n); do rm -rf $out /tmp/flycache_l5rep
  COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250 FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_l5rep \
    timeout 900 /opt/venv/bin/python3 $L/tools/compile_l5.py $L/$arm $out prod causal > $out.log 2>&1
  f=$(ls $out/*/22_final_isa.s); echo "$arm fixed r$i $(grep -v '^\s*\.\(file\|ident\)' $f | md5sum | cut -c1-12)"
done
