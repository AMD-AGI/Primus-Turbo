#!/bin/bash
# usage: rc.sh <tag> <serialize 0|3> <cache-suffix> <cmd...>   One card process on GPU 3 (fa-g3), under its lock.
L=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab3
OP=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/job_context/op
T=$1; SER=$2; CS=$3; shift 3
O=$L/run/$T
d0=$(timeout 20 sudo -n dmesg | wc -l)
( while true; do echo "$(date +%T) kfd=$(ls /sys/class/kfd/kfd/proc/ | tr "\n" ,) $(grep '\*' /sys/class/drm/card24/device/pp_dpm_sclk) busy=$(cat /sys/class/drm/card24/device/gpu_busy_percent 2>/dev/null)"; sleep 1; done ) > $O.sclk 2>&1 &
W=$!
t0=$(date +%s)
echo "# $(date -u +%FT%TZ) cmd: $*" > $O.log
flock /tmp/b0-gpu3.lock docker exec -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 fa-g3 bash -c "export TORCH_BLAS_PREFER_HIPBLASLT=1 HIPBLASLT_TENSILE_LIBPATH=/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250 AMD_SERIALIZE_KERNEL=$SER FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_lab3_$CS; $*" >> $O.log 2>&1
rc=$?
kill $W
echo "RC=$rc wall=$(( $(date +%s)-t0 ))s" >> $O.log
echo "--- dmesg new lines since start:" >> $O.log
timeout 20 sudo -n dmesg | tail -n +$((d0+1)) >> $O.log
echo "--- kfd after: $(ls /sys/class/kfd/kfd/proc/ | tr '\n' ,)" >> $O.log
grep -E "correctness|f5 vs|determinism|VERDICT|RESULT|RC=|Error|error|Traceback" $O.log | tail -30; tail -3 $O.log
