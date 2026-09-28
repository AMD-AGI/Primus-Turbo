#!/bin/bash
# usage: g0.sh TAG "shell command run inside fa-g0 (cwd = ruler/)"   -- GPU 0 only, flock'ed
W=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/ruler
TAG=$1; CMD=$2; OUT=$W/runs0/$TAG; mkdir -p $W/runs0
D0=$(timeout 20 sudo -n dmesg | wc -l)
flock /tmp/b0-gpu0.lock docker exec -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 fa-g0 bash -c \
  "cd $W && export PYTHONDONTWRITEBYTECODE=1 TORCH_BLAS_PREFER_HIPBLASLT=1 && $CMD" > $OUT.log 2>&1
RC=$?
timeout 20 sudo -n dmesg | tail -n +$((D0+1)) | grep -v '0002:04:00.0' > $OUT.dmesg
timeout 20 sudo -n dmesg | tail -20 | grep -v '0002:04:00.0' >> $OUT.dmesg.tail
echo "rc=$RC newdmesg_g0=$(wc -l < $OUT.dmesg)" >> $OUT.log
grep -E "^RESULT|rc=" $OUT.log | sed -E "s/^/$TAG /; s/ (stat|iters|bw_gbs|flop|bytes_min|causal)=[^ ]*//g"
if grep -q '0001:04:00.0' $OUT.dmesg; then echo "!!! NEW GPU0 DMESG in $TAG"; head $OUT.dmesg; exit 9; fi
[ $RC = 0 ] || exit $RC
