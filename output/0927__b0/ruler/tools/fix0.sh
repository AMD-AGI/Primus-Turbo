#!/bin/bash
# usage: fix0.sh TAG shape arm1 arm2 ...  -- the PATCHED harness (harness_op/benchmark.py == fix/benchmark.py) on GPU0
W=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/ruler
TAG=$1; SH=$2; shift 2
ARGS=""; for a in "$@"; do ARGS="$ARGS --arm-path $a=$W/arms/$a"; done
$W/tools/g0.sh $TAG "cd harness_op && export PYTHONPATH=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/tools FLYDSL_RUNTIME_CACHE_DIR=/tmp/rc_$TAG ${FIXENV} && timeout 900 /opt/venv/bin/python3 benchmark.py $ARGS --shapes $SH --iters 101 ${FIXARGS} --json $W/runs0/$TAG.json" | sed -E 's/ (min_ms|max_ms|tflops|shape)=[^ ]*//g'
exit ${PIPESTATUS[0]}
