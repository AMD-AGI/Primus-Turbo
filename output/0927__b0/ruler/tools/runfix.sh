#!/bin/bash
# usage: runfix.sh TAG shape arm1 arm2 ...  -- runs the PATCHED harness (fix/benchmark.py)
W=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/ruler
TAG=$1; SH=$2; shift 2
ARGS=""; for a in "$@"; do ARGS="$ARGS --arm-path $a=$W/arms/$a"; done
OUT=$W/runs/$TAG
D0=$(timeout 20 sudo -n dmesg | wc -l)
flock /tmp/b0-gpu2.lock docker exec -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 fa-g2 bash -c \
  "cd $W/harness_op && export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/tools && timeout 900 /opt/venv/bin/python3 benchmark.py $ARGS --shapes $SH --iters 101 --json $OUT.json" > $OUT.log 2>&1
RC=$?
D1=$(timeout 20 sudo -n dmesg | wc -l)
timeout 20 sudo -n dmesg | tail -n +$((D0+1)) > $OUT.dmesg
echo "rc=$RC newdmesg=$((D1-D0))" >> $OUT.log
grep -E "^# arm|^RESULT|rc=" $OUT.log | sed -E "s/^/$TAG /; s/ (stat|iters|min_ms|max_ms|bw_gbs|flop|bytes_min|causal)=[^ ]*//g"
