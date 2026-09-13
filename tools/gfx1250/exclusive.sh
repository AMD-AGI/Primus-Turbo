#!/bin/bash
# Take an exclusive GPU window WITHOUT killing anything mid-kernel, then release it.
#
#   tools/gfx1250/exclusive.sh <command...>
#
# Signals the sweep queue to pause between candidates, waits for the current one to finish,
# runs the command, then releases. Replaces the pkill -9 pattern that is a suspected
# contributor to the 2026-09-13 wedge.
set -u
STOP_FILE=${STOP_FILE:-/tmp/wq.stop}
LEDGER=${LEDGER:-/tmp/forever.jsonl}
touch "$STOP_FILE"
# Wait for the queue to go quiet: no new ledger line for 20 s, or no queue running at all.
for _ in $(seq 1 60); do
  pgrep -f forever_queue >/dev/null 2>&1 || break
  a=$(stat -c %Y "$LEDGER" 2>/dev/null || echo 0); sleep 12
  b=$(stat -c %Y "$LEDGER" 2>/dev/null || echo 0)
  [ "$a" = "$b" ] && break
done
trap 'rm -f "$STOP_FILE"' EXIT
"$@"
