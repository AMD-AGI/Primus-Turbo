#!/bin/bash
# usage: cc.sh <arm> <config>   compile-only inside fa-g3, no flock (no GPU use; LAB-RULES rule 1)
ARM=$1; CFG=$2
L=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab3-L21
OUT=$L/isa/$ARM/$CFG
rm -rf $OUT; mkdir -p $OUT
docker exec -u $(id -u):$(id -g) -e HOME=/tmp fa-g3 bash -c "cd $L/tools && env COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250 FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_L21cc_${ARM}_${CFG} FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=$OUT timeout 1200 /opt/venv/bin/python3 compile_isa.py $L/arms/$ARM $CFG" > $OUT/compile.log 2>&1
echo "$ARM $CFG rc=$? $(tail -1 $OUT/compile.log)"
