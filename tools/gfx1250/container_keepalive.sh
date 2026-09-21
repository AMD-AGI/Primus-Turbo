#!/bin/bash
# Keep the op-evolve job's container running. One line of output per intervention, nothing
# on the happy path.
#
#   container_keepalive.sh [container] [interval_s]     # foreground; run under setsid
#
# WHY. The job spec declares this container `owned: false`, which means op-evolve attaches
# and will never create or start it. The container is currently in Exited(137) -- it has
# already been OOM-killed once. If that happens again overnight, supervise_job.sh cannot
# recover: every `docker exec` fails, it backs off and restarts the loop, and gpu_ok()
# still passes because the CARD is fine. It burns its MAX_RESTARTS=60 budget, paying for
# an agent each time, and dies silently before morning. Nothing in dmesg.
#
# Starting a stopped container is a zero-GPU-risk action, so this is safe to automate.
set -u
C=${1:-fa-repro}
INTERVAL=${2:-60}
LEDGER=${KEEPALIVE_LEDGER:-/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0921__flydsl/keepalive.jsonl}
mkdir -p "$(dirname "$LEDGER")"

note(){ printf '{"t":"%s","container":"%s","event":%s}\n' "$(date -Is)" "$C" "$1" >> "$LEDGER"; }

# A PID file, because `pgrep -f container_keepalive.sh` matches the command line of
# whoever is asking -- measured: a liveness check run from an interactive shell reported
# "already running" against its own 0-second-old invocation, on a machine that had been
# power-cycled minutes earlier and could not have had a survivor. Same family as the
# `pkill -f <pattern>` trap: a pattern that names the target also names the asker.
# Check liveness with `kill -0 "$(cat <pidfile>)"`, never by pattern.
PID_FILE=${KEEPALIVE_PID_FILE:-/tmp/gfx1250-container-keepalive.pid}
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
  echo "already running as pid $(cat "$PID_FILE")"; exit 0
fi
echo $$ > "$PID_FILE"
# The handler must EXIT. A trap on TERM whose body does not exit makes bash run the body
# and then CARRY ON with the loop -- the signal is swallowed and the process becomes
# un-stoppable by the ordinary means. Measured: this script ignored two SIGTERMs and kept
# restarting a container it was supposed to have stopped guarding.
trap 'rm -f "$PID_FILE"' EXIT
trap 'rm -f "$PID_FILE"; exit 0' INT TERM

while true; do
  state=$(timeout 30 docker inspect -f '{{.State.Running}}|{{.State.ExitCode}}' "$C" 2>/dev/null)
  case "$state" in
    true\|*) : ;;                       # running: say nothing
    false\|*)
      code=${state#false|}
      echo "[$(date -Is)] $C is down (exit $code); starting it"
      note "{\"down_exit\":$code}"
      if timeout 120 docker start "$C" >/dev/null 2>&1; then
        note '{"started":true}'
      else
        echo "[$(date -Is)] $C failed to start"
        note '{"started":false}'
      fi
      ;;
    "")
      # No such container, or dockerd is not answering. Do not spin: say it once per
      # interval into the ledger and keep checking, because this is recoverable by a human
      # and unrecoverable by us.
      note '{"inspect_failed":true}'
      ;;
  esac
  sleep "$INTERVAL"
done
