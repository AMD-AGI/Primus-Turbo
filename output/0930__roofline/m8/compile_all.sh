#!/bin/bash
# COMPILE_ONLY builds of every arm (causal prod path, gqa 4) plus base non-causal; ISA + resource census. CPU only.
M=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0930__roofline/m8
for a in base noexp nosm nomask nobar wl; do
  for c in causal nc; do
    [ $c = nc ] && [ $a != base ] && continue
    D=$M/dump/${a}_$c; rm -rf $D; mkdir -p $D
    docker exec -e COMPILE_ONLY=1 -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 -e FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_m8c_${a}_$c -e HIP_VISIBLE_DEVICES= \
      fa-repro bash -c "cd $M && rm -rf /tmp/flycache_m8c_${a}_$c && timeout 600 /opt/venv/bin/python3 compile_arm.py arms/$a $D m32x8 4 $c" > $D.log 2>&1
    S=$(find $D -name "*final_isa.s" | head -1)
    echo "$a $c $(grep -c COMPILE_OK $D.log) isa=$( [ -n "$S" ] && echo yes || echo no) $( [ -n "$S" ] && grep -hE '\.(vgpr_count|vgpr_spill_count|sgpr_spill_count|private_segment_fixed_size):' $S | tr -s ' ' | tr '\n' ' ')"
  done
done
