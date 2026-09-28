#!/bin/bash
# usage: compile.sh ARM [kernels...] -- COMPILE_ONLY in fa-g3 (no flock, no GPU), summary line per kernel
L=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab-kdq
A=$1; shift
rm -rf "${L:?}/compile/dump/$A"
docker exec -e COMPILE_ONLY=1 -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 -e FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_kdq_c_$A -e FLYDSL_DUMP_IR=1 -e FLYDSL_DUMP_DIR=$L/compile/dump/$A fa-g3 bash -c "cd $L/compile && timeout 1500 python3 compile_bwd.py $L/OP/$A $*" > $L/compile/$A.log 2>&1
echo "RC=$?" >> $L/compile/$A.log
tail -1 $L/compile/$A.log
for f in $(find $L/compile/dump/$A -name 21_final_isa.s | sort); do
  printf "%-40s " "${f#$L/compile/dump/}"; grep -oE '\.(vgpr_count|vgpr_spill_count|sgpr_spill_count|private_segment_fixed_size|group_segment_fixed_size):\s*[0-9]+' $f | sort -u | tr '\n' ' '; echo
done
