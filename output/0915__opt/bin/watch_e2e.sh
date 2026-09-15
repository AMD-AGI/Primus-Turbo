#!/bin/bash
# Silent-until-broken watchdog for an e2e run. Polls every ~30 s and prints NOTHING while the
# run is healthy; speaks only for a stall, a card fault, or the end of the run.
#
# The point is catching a hang EARLY, not watching progress. On 0915 a hang burned 16 minutes
# of wall clock before anyone looked, and a wedged card on this box needs a human AC-cycle --
# while a per-step monitor, tried earlier the same day, buried every useful line on screen.
#
#   watch_e2e.sh <log-path> [stall_seconds] [poll_seconds]
set -u
LOG=${1:?log path}; STALL=${2:-120}; POLL=${3:-30}
BASE_MES=$(sudo -n dmesg 2>/dev/null | grep -cE "ring buffer is full|failed to respond|SIGBUS|wait for reset ack")
warned_stall=0
while true; do
  sleep "$POLL"
  # Card faults first: these decide whether a stall is recoverable or needs a power cycle.
  MES=$(sudo -n dmesg 2>/dev/null | grep -cE "ring buffer is full|failed to respond|SIGBUS|wait for reset ack")
  if [ "$MES" -gt "$BASE_MES" ]; then
    echo "CARD FAULT: $((MES-BASE_MES)) new dmesg fault line(s) -- $(sudo -n dmesg 2>/dev/null | grep -E 'ring buffer is full|failed to respond|SIGBUS|wait for reset ack' | tail -1)"
    exit 2
  fi
  [ -f "$LOG" ] || continue
  # Finished? Report the steady tps and stop.
  if grep -q "Training completed" "$LOG" 2>/dev/null; then
    echo "DONE: $(sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -oE 'tps: +[0-9,]+' | tail -1)"
    exit 0
  fi
  AGE=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
  if [ "$AGE" -gt "$STALL" ]; then
    if [ "$warned_stall" -eq 0 ]; then
      echo "STALLED: no log output for ${AGE}s (last: $(sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -oE 'step: +[0-9]+' | tail -1 || echo 'no step yet'))"
      warned_stall=1
    fi
  else
    warned_stall=0
  fi
done
