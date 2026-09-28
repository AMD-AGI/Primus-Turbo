#!/bin/bash
# usage: card.sh <tag> <serialize 0|3> <python args...>   one card process on GPU 2 / fa-g2, dmesg checked after.
# exit 2 on any new amdgpu/fault dmesg line (caller must stop all card work).
W=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab2-L12
M=$W/card; TAG=$1; SER=$2; shift 2
[ -f $M/STOP ] && { echo "STOP sentinel"; exit 3; }
base=$(timeout 20 sudo -n dmesg | wc -l)
echo "$(date +%T) START $TAG"
EXTRA=""; [ "$SER" = 3 ] && EXTRA="-e AMD_SERIALIZE_KERNEL=3"; [ -n "$CHAMP_DIR" ] && EXTRA="$EXTRA -e CHAMP_DIR=$CHAMP_DIR"
timeout 1800 flock /tmp/b0-gpu2.lock docker exec -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 $EXTRA \
  -e FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_L12_$TAG fa-g2 bash -c \
  "cd /home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-fwd-b0-20260927/job_context/op && /opt/venv/bin/python3 $*" \
  > $M/$TAG.log 2>&1
rc=$?
new=$(timeout 20 sudo -n dmesg | tail -n +$((base+1)))
{ echo "== rc=$rc dmesg-new-lines:"; echo "$new"; echo "== dmesg tail -20:"; timeout 20 sudo -n dmesg | tail -20; } > $M/$TAG.dmesg
echo "$(date +%T) END $TAG rc=$rc $(grep -hE '^(CORR|RATIO)' $M/$TAG.log | tr '\n' ' ')"
if echo "$new" | grep -qiE 'amdgpu|gcvm|kfd|reset|ring .*timeout|hang|fault'; then
  echo "FAULT after $TAG:"; echo "$new" | grep -iE 'amdgpu|gcvm|kfd|reset|timeout|hang|fault' | head -20; exit 2; fi
[ $rc -ne 0 ] && { echo "NONZERO rc=$rc after $TAG"; tail -20 $M/$TAG.log; exit 1; }
exit 0
