#!/bin/bash
# compile-only under flydsl 0.3.2 in fa-g3 (no flock: no GPU).  usage: cc.sh <label> <tree> <F5_ATOM> <jobs...>
L=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab3
T=$1; TREE=$2; AT=$3; shift 3
docker exec fa-g3 bash -c "cd $L && COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250 FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_cc_$T FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=$L/isa032/$T F5_KT=lds F5_ATOM=$AT timeout 1500 /opt/venv/bin/python3 compile_f5_032.py $TREE $* > $L/isa032/$T.log 2>&1; echo RC=\$? >> $L/isa032/$T.log"
tail -2 $L/isa032/$T.log
