#!/bin/bash
# wait_gpu_free.sh <gpu> [max_wait_s] [quiet_s]: read-only watcher. Returns 0 once the card has had no process and
# <= 1% VRAM for quiet_s seconds in a row (default 60), 1 after max_wait_s (default 6 h). Polls every 30 s and prints
# who is on the card whenever that changes. Use it before tools/run_gpu.sh while someone else's job runs (run_gpu.sh
# still makes every one of its own checks at start).
set -uo pipefail
GPU=${1:?gpu index}; MAX=${2:-21600}; QUIET=${3:-60}
CT=lhz-hkl-g$GPU
PCI=$(cat "/tmp/$CT.pci" 2>/dev/null) || { echo "no /tmp/$CT.pci" >&2; exit 3; }
t0=$(date +%s); free_since=""; last=""
while :; do
  USED=$(rocm-smi --showbus --showmemuse --json 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(next((v.get('GPU Memory Allocated (VRAM%)', '?') for v in d.values() if isinstance(v, dict) and v.get('PCI Bus') == '$PCI'), '?'))")
  PIDS=$(amd-smi process -g "$GPU" 2>/dev/null | awk '/PID:/{print $2}' | tr '\n' ' ')
  who=""
  for p in $PIDS; do who="$who $p:$(ps -o user=,comm= -p "$p" 2>/dev/null | tr -s ' ' | cut -c1-40)"; done
  now=$(date +%s)
  state="vram=${USED}% pids=[${who# }]"
  [ "$state" != "$last" ] && { echo "[$(date -u +%H:%M:%S) UTC] gpu $GPU: $state"; last=$state; }
  if [ -z "${PIDS// /}" ] && { [ "$USED" = 0 ] || [ "$USED" = 1 ]; }; then
    [ -z "$free_since" ] && free_since=$now
    [ $((now - free_since)) -ge "$QUIET" ] && { echo "[$(date -u +%H:%M:%S) UTC] gpu $GPU free for ${QUIET}s"; exit 0; }
  else
    free_since=""
  fi
  [ $((now - t0)) -ge "$MAX" ] && { echo "gave up after ${MAX}s" >&2; exit 1; }
  sleep 30
done
