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
