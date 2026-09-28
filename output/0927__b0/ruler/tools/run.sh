#!/bin/bash
# usage: run.sh TAG "extra env" arm1 arm2 ...   (arms are names under ruler/arms)
W=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/ruler
TAG=$1; EXTRA=$2; shift 2
ARGS=""; for a in "$@"; do ARGS="$ARGS --arm-path $a=$W/arms/$a"; done
mkdir -p $W/runs
OUT=$W/runs/$TAG
D0=$(timeout 20 sudo -n dmesg | wc -l)
flock /tmp/b0-gpu2.lock docker exec -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 fa-g2 bash -c \
  "cd $W && export PYTHONDONTWRITEBYTECODE=1 $EXTRA && timeout 900 /opt/venv/bin/python3 $W/tools/${BENCH:-bench2}.py $ARGS --shape ${SHAPE:-prod} --iters 101 --json $OUT.json ${BENCH_EXTRA}" > $OUT.log 2>&1
RC=$?
D1=$(timeout 20 sudo -n dmesg | wc -l)
timeout 20 sudo -n dmesg | tail -n +$((D0+1)) > $OUT.dmesg
echo "rc=$RC newdmesg=$((D1-D0))" >> $OUT.log
grep -E "^RESULT|rc=" $OUT.log | sed "s/^/$TAG /"
grep -iE "amdgpu|fault|gpu reset|hang" $OUT.dmesg | head -5
