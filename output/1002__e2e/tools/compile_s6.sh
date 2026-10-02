#!/bin/bash
# Compile-only screen of an s6-family bwd arm (NO GPU: HIP_VISIBLE_DEVICES=-1, COMPILE_ONLY=1, no lock needed).
#   compile_s6.sh <arm_dir> <out_dir> <expect_flydsl_version> [delta] [dkdv] [dkdv_sp] [dqg]
# Dumps every IR stage + final ISA under <out_dir>/<kernel>/, log in <out_dir>/compile.log; fresh JIT cache dir,
# disk cache off. Prints the per-kernel resource line. Pattern of output/0930__bwd/tools/compile.sh.
set -u
T=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__e2e/tools
A=$(readlink -f "$1"); O=$(readlink -f "$2"); V=$3; shift 3
mkdir -p "$O"
timeout 1800 docker exec -e COMPILE_ONLY=1 -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 -e HIP_VISIBLE_DEVICES=-1 \
  -e FLYDSL_RUNTIME_ENABLE_CACHE=0 -e FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_c1002_$(basename $O)_$$ -e FLYDSL_DUMP_IR=1 \
  -e FLYDSL_DUMP_DIR=$O -e PYTHONDONTWRITEBYTECODE=1 -e EXPECT_FLYDSL=$V fa-repro \
  bash -c "cd $A && /opt/venv/bin/python3 $T/compile_s6.py $A $*" > $O/compile.log 2>&1
echo "RC=$?" >> $O/compile.log
timeout 120 docker exec fa-repro chown -R $(id -u):$(id -g) "$O"
tail -1 $O/compile.log
for f in $(find $O -name "2[12]_final_isa.s" | sort); do
  printf "%-10s " "$(echo ${f#$O/} | cut -d/ -f1)"
  grep -oE '\.(vgpr_count|sgpr_count|vgpr_spill_count|sgpr_spill_count|private_segment_fixed_size|group_segment_fixed_size):\s*[0-9]+' $f | sort -u | tr '\n' ' '
  printf " wmma=%s ds=%s " $(grep -c v_wmma $f) $(grep -cE '^\s*ds_' $f); echo
done
