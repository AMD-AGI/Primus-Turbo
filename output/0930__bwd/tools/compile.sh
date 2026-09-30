#!/bin/bash
# Compile-only screen of an arm (NO GPU: HIP_VISIBLE_DEVICES=-1, no lock needed). Prints per-kernel resource line.
#   compile.sh <arm_dir> [dkdv] [dqg]      -> dump in <arm_dir>/.dump, log <arm_dir>/.compile.log
set -u
T=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0930__bwd/tools
A=$(readlink -f "$1"); shift
timeout 60 docker exec fa-repro rm -rf "$A/.dump" 2>/dev/null; rm -rf "$A/.dump" 2>/dev/null; mkdir -p "$A/.dump"
timeout 1800 docker exec -e COMPILE_ONLY=1 -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 -e HIP_VISIBLE_DEVICES=-1 \
  -e FLYDSL_RUNTIME_ENABLE_CACHE=0 -e FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_c_$(basename $A)_$$ -e FLYDSL_DUMP_IR=1 \
  -e FLYDSL_DUMP_DIR=$A/.dump -e PYTHONDONTWRITEBYTECODE=1 fa-repro \
  bash -c "cd $A && /opt/venv/bin/python3 $T/compile_arm.py $A $*" > $A/.compile.log 2>&1
echo "RC=$?" >> $A/.compile.log
tail -1 $A/.compile.log
for f in $(find $A/.dump -name "2[12]_final_isa.s" | sort); do
  printf "%-12s " "$(echo ${f#$A/.dump/} | cut -d/ -f1)"
  grep -oE '\.(vgpr_count|vgpr_spill_count|sgpr_spill_count|private_segment_fixed_size|group_segment_fixed_size):\s*[0-9]+' $f | sort -u | tr '\n' ' '
  printf " wmma=%s ds=%s " $(grep -c v_wmma $f) $(grep -cE '^\s*ds_' $f); echo
done
