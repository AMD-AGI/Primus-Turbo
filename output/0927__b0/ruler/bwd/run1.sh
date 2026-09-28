#!/bin/bash
# usage: run1.sh TAG [benchmark args...]   -- one prod benchmark process on GPU0 (fa-g0), then dmesg check
TAG=$1; shift
OP=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/ruler/bwd/oe/artifacts/job/job_context/op; R=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/ruler/bwd/runs
N0=$(timeout 20 sudo -n dmesg | wc -l)
flock /tmp/b0-gpu0.lock docker exec -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 fa-g0 bash -c "export TORCH_BLAS_PREFER_HIPBLASLT=1 HIPBLASLT_TENSILE_LIBPATH=/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250; cd $OP && timeout 900 python3 benchmark.py --shapes prod --iters 101 --json $R/$TAG.json $*" > $R/$TAG.log 2>&1
echo "rc=$?" >> $R/$TAG.log
timeout 20 sudo -n dmesg | tail -n +$((N0+1)) > $R/$TAG.dmesg
grep -v '0002:04:00' $R/$TAG.dmesg | grep -iE 'amdgpu|fault|gcvm|mes|hang|reset' > $R/$TAG.dmesg.bad
echo "$TAG $(tail -1 $R/$TAG.log) newdmesg=$(wc -l < $R/$TAG.dmesg) bad=$(wc -l < $R/$TAG.dmesg.bad)"
grep ^RESULT $R/$TAG.log | sed -E 's/.*arm=([^ ]+).*latency_ms=([^ ]+).*sclk_start=([^ ]+) sclk_end=([^ ]+).*order=([^ ]+).*/  \1 \2 sclk \3-\4 \5/'
