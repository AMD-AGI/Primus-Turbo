#!/bin/bash
# usage: r4.sh TAG "extra env" arm1 arm2 ...   (arms under ruler/arms; bench4 via g0.sh; per-process JIT cache dir)
W=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/ruler
TAG=$1; EXTRA=$2; shift 2
ARGS=""; for a in "$@"; do ARGS="$ARGS --arm-path $a=$W/arms/$a"; done
$W/tools/g0.sh $TAG "export FLYDSL_RUNTIME_CACHE_DIR=/tmp/rc_$TAG $EXTRA && timeout 900 /opt/venv/bin/python3 tools/bench4.py $ARGS --shape ${SHAPE:-prod} --modes ${MODES:-std,blocked} --json runs0/$TAG.json ${B4X}" | sed "s/ by_pred=.*//" ; exit ${PIPESTATUS[0]}
