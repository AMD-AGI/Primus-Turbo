#!/bin/bash
# usage: card.sh <tag> <python args...>   ONE card process on GPU 3 / fa-g3 under the lock; dmesg check after.
W=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab3-L21
TAG=$1; shift
mkdir -p $W/run
[ -f $W/run/STOP ] && { echo "STOP sentinel"; exit 3; }
base=$(timeout 20 sudo -n dmesg | wc -l)
echo "$(date +%T) START $TAG"
flock /tmp/b0-gpu3.lock docker exec -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 \
  -e FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_L21_$TAG ${EXTRA_ENV} fa-g3 bash -c \
  "cd /home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-fwd-b0-20260927/job_context/op && timeout ${TMO:-1500} /opt/venv/bin/python3 $*" \
  > $W/run/$TAG.log 2>&1
rc=$?
new=$(timeout 20 sudo -n dmesg | tail -n +$((base+1)))
{ echo "== rc=$rc dmesg-new-lines:"; echo "$new"; echo "== dmesg tail -20:"; timeout 20 sudo -n dmesg | tail -20; } > $W/run/$TAG.dmesg
echo "$(date +%T) END $TAG rc=$rc newdmesg=$(echo -n "$new" | grep -c .)"
if echo "$new" | grep -qiE 'amdgpu|gcvm|kfd|reset|ring .*timeout|hang'; then
  echo "FAULT after $TAG:"; echo "$new" | grep -iE 'amdgpu|gcvm|kfd|reset|timeout|hang' | head -20; touch $W/run/STOP; exit 2; fi
exit $rc
