#!/bin/bash
# usage: run.sh <arm> <config>   compile-only inside fa-g2, no flock (no GPU use; LAB-RULES rule 1)
ARM=$1; CFG=$2
L=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab2-L12
SRC=$L/arms/$ARM; case $ARM in base|base_r6) SRC=$L/$ARM;; esac
OUT=$L/isa/$ARM/$CFG
rm -rf $OUT; mkdir -p $OUT
docker exec -u $(id -u):$(id -g) -e HOME=/tmp fa-g2 bash -c "cd $L/tools && env COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250 FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_L12_${ARM}_${CFG} FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=$OUT /opt/venv/bin/python3 compile_isa.py $SRC $CFG" > $OUT/compile.log 2>&1
echo "$ARM $CFG rc=$? $(tail -1 $OUT/compile.log)"
