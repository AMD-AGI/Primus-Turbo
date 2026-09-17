#!/bin/bash
# Take an exclusive window on one GPU without killing anything mid-kernel.
#   exclusive.sh <gpu> <command...>
#
# The queue polls a STOP sentinel BETWEEN candidates, so it parks without a signal ever
# reaching a process that has work in flight on the card. That matters twice over: a
# contended measurement does not raise, it just records a wrong number and that number
# becomes evidence; and repeatedly SIGKILLing running kernels is the practice that
# plausibly contributed to yesterday's fifth GPU wedge.
set -u
G=${1:?gpu}; shift
STOP=/tmp/campaign.g${G}.stop
OUT=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0914__campaign
# Release before anything that can be interrupted: installed after the wait instead, a
# timeout during the wait leaves the sentinel behind and the queue stays parked forever.
trap 'rm -f "$STOP"' EXIT INT TERM
touch "$STOP"
# Wait for quiet: no new ledger line across one interval means the queue is parked.
led=$(ls -t "$OUT"/ledgers/g${G}.*.jsonl 2>/dev/null | head -1)
for _ in $(seq 1 18); do
  [ -n "$led" ] || break
  a=$(stat -c %Y "$led" 2>/dev/null || echo 0); sleep 10
  b=$(stat -c %Y "$led" 2>/dev/null || echo 0)
  [ "$a" = "$b" ] && break
done
"$@"
