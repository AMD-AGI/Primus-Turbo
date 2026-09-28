#!/bin/bash
# usage: run1.sh TAG SCRIPT ARGS... -- one card process on GPU3 (fa-g3), cwd = lab OP, then dmesg check
TAG=$1; shift
L=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab-kdq
OP=$L/oe/artifacts/job/job_context/op; R=$L/runs
N0=$(timeout 20 sudo -n dmesg | wc -l)
flock /tmp/b0-gpu3.lock docker exec -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 -e OE_PHYS_GPU=3 fa-g3 bash -c "export TORCH_BLAS_PREFER_HIPBLASLT=1 HIPBLASLT_TENSILE_LIBPATH=/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250 FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_labkdq; cd $OP && timeout 1500 python3 $*" > $R/$TAG.log 2>&1
echo "rc=$?" >> $R/$TAG.log
timeout 20 sudo -n dmesg | tail -n +$((N0+1)) > $R/$TAG.dmesg
grep -v '0002:04:00' $R/$TAG.dmesg | grep -iE 'amdgpu|fault|gcvm|mes|hang|reset' > $R/$TAG.dmesg.bad
echo "$TAG $(tail -1 $R/$TAG.log) newdmesg=$(wc -l < $R/$TAG.dmesg) bad=$(wc -l < $R/$TAG.dmesg.bad)"
grep -E '^(RESULT|CORR|DET|XARM|LABVAL|KB )' $R/$TAG.log | sed -E -e 's/^RESULT .*arm=([^ ]+).*latency_ms=([^ ]+).*sclk_start=([^ ]+) sclk_end=([^ ]+).*order=([^ ]+).*/  \1 \2 sclk \3-\4/' -e 's/^KB shape=([^ ]+) mode=([^ ]+) what=([^ ]+) arm=([^ ]+) median_ms=([^ ]+).* sclk_med=([^ ]+).*/  \2 \3 \4 \5 sclk \6/'
[ -s $R/$TAG.dmesg.bad ] && { echo "!!! DMESG BAD"; cat $R/$TAG.dmesg.bad; exit 9; }
grep -q '^rc=0' $R/$TAG.log || exit 8
exit 0
