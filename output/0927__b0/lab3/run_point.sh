#!/bin/bash
# usage: run_point.sh <tag> <ATOM_MODE> <ATOM_LDS> <KVG> [iters] [serialize]
# One P2 point = one process on GPU 3 (fa-g3) under its lock. sclk witness from host sysfs (card24).
L=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab3
T=$1; M=$2; S=$3; K=$4; IT=${5:-21}; SER=${6:-0}
O=$L/run/$T
d0=$(timeout 20 sudo -n dmesg | wc -l)
( while true; do echo "$(date +%T) kfd=$(ls /sys/class/kfd/kfd/proc/ | tr "\n" ,) $(grep '\*' /sys/class/drm/card24/device/pp_dpm_sclk) busy=$(cat /sys/class/drm/card24/device/gpu_busy_percent 2>/dev/null)"; sleep 0.5; done ) > $O.sclk 2>&1 &
W=$!
t0=$(date +%s)
flock /tmp/b0-gpu3.lock docker exec -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 fa-g3 bash -c "cd $L && AMD_SERIALIZE_KERNEL=$SER FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_p2run_${M}_$S ATOM_MODE=$M ATOM_LDS=$S /opt/venv/bin/python3 atomprobe/run_probe.py $K $IT" > $O.log 2>&1
rc=$?
kill $W
echo "RC=$rc wall=$(( $(date +%s)-t0 ))s" >> $O.log
echo "--- dmesg new lines since start:" >> $O.log
timeout 20 sudo -n dmesg | tail -n +$((d0+1)) >> $O.log
tail -4 $O.log
