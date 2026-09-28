#!/bin/bash
# usage: run_card.sh SHAPE CAUSAL [kinds]  -- one (shape,causal) per card process, GPU 3 / fa-g3 under the shared lock
V=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/verify_r6
tag=$1_$([ "$2" = 1 ] && echo causal || echo full)
log=$V/logs/card_$tag.log
flock /tmp/b0-gpu3.lock docker exec -e PYTHONDONTWRITEBYTECODE=1 -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 \
  -e FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_vr6 fa-g3 bash -c "cd $V && timeout 1800 python card_run.py $1 $2 $3" > $log 2>&1
rc=$?
echo "rc=$rc" >> $log
timeout 20 sudo -n dmesg | tail -20 > $V/logs/dmesg_after_$tag.txt
echo "rc=$rc"; grep -c '^CARD ' $log; tail -n 2 $log; grep -i "amdgpu" $V/logs/dmesg_after_$tag.txt | tail -5
