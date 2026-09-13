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
PID_FILE=${PID_FILE:-${LEDGER%.jsonl}.pid}

# Release the STOP file before anything that can block or be killed. Installed after the
# wait loop instead, a timeout or Ctrl-C during the wait leaves it behind and the sweep
# queue stays paused forever -- which is how this script failed its first test.
trap 'rm -f "$STOP_FILE"' EXIT INT TERM
touch "$STOP_FILE"

# Liveness via a PID file and kill -0, NOT pgrep.
#
# pgrep walks /proc, and on a wedged card that blocks on the processes stuck in the driver
# -- pgrep itself hung when this was first tested, the same way ps and rocm-smi do. kill -0
# is a single signal check against one pid and cannot block on the driver.
queue_alive(){ [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; }

# Wait for the queue to go quiet: no new ledger line for one interval, or no queue at all.
# Bounded at ~2 minutes -- a candidate takes seconds, so longer means it is not stopping.
for _ in $(seq 1 12); do
  queue_alive || break
  a=$(stat -c %Y "$LEDGER" 2>/dev/null || echo 0)
  sleep 10
  b=$(stat -c %Y "$LEDGER" 2>/dev/null || echo 0)
  [ "$a" = "$b" ] && break
done

"$@"
