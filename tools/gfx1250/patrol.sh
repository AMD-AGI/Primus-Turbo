#!/bin/bash
# One pass of op-evolve supervision. Silent on the happy path; prints only what a person
# would act on. Meant for cron every ~7 minutes.
#
#   patrol.sh <job-ref> [stall_minutes]      exit 0 quiet | 1 attention | 2 wedged
#
# WHY THIS EXISTS. supervise_job.sh has three terminal states and ALL THREE ARE SILENT,
# and in all three dmesg is completely clean:
#
#   1. MAX_RESTARTS=60 exhausted  -> one line into supervisor.log, then exit 1
#   2. job_finished (max_rounds / max_timeout reached) -> a normal exit, and the card
#      then sits idle until somebody notices
#   3. gpu_ok() fails             -> infinite 900 s backoff, waiting for a human forever
#
# A watchdog that only greps dmesg sees none of these. It also cannot see the failure mode
# this project produces most: a candidate that HANGS rather than crashes, which leaves the
# loop alive and state.yaml frozen. supervise_job.sh only supervises the loop DYING.
#
# Note also that supervise_job.sh's backoff only ever grows -- it is never reset after a
# successful round -- so a job that restarted a few times ends up idling 900 s between
# rounds while looking perfectly healthy.
set -u
JOB=${1:?usage: patrol.sh <job-ref> [stall_minutes]}
STALL_MIN=${2:-90}

OE=${OE_ROOT:-/home/lihuzhan/code/2026_0910__op-evolve/op-evolve}
ART="$OE/artifacts"
STATE="$ART/$JOB/job_context/state.yaml"
SUPLOG="$ART/supervisor.log"
MARK=${PATROL_MARK:-/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0921__flydsl/.patrol}
mkdir -p "$(dirname "$MARK")"

rc=0
alert(){ echo "[$(date -Is)] $*"; rc=1; }

# --- the wedge latch --------------------------------------------------------------------
# A wedge is a STATE, not an event, and this file used to report it as an event. The dmesg
# check below is deliberately differential -- it baselines the count each pass and reacts
# only to what is new -- because a buffer still holding an old, recovered fault would
# otherwise make every pass shout. But that means the wedge alarm fires exactly ONCE: the
# next pass has already folded those lines into its baseline, sees no new ones, and
# returns 0.
#
# Observed on 2026-09-21. Two unrecoverable signatures appeared, the pass reported WEDGED
# and exit 2, and the very next pass came back clean -- on a card that was still wedged
# and still needed a human. A monitor whose alarm cancels itself is worse than no monitor:
# the second reading looks like recovery.
#
# It already wrote this flag. It just never read it back. Now it latches: once set, every
# pass keeps reporting until a person clears it, and the only thing that clears it is the
# power cycle it was asking for.
if [ -f "$MARK.wedged" ]; then
  BOOT=$(date -d "$(uptime -s)" +%s 2>/dev/null || echo 0)
  SET=$(stat -c %Y "$MARK.wedged" 2>/dev/null || echo 0)
  if [ "$SET" -gt "$BOOT" ]; then
    echo "[$(date -Is)] STILL WEDGED (latched $(date -Is -d @"$SET"))."
    echo "  Nothing has power-cycled this machine since the wedge was recorded."
    echo "  Every GPU loop must stay stopped. Clear with: rm $MARK.wedged"
    exit 2
  fi
  # The box booted after the flag was written, so the cycle happened. Clear it and carry
  # on -- re-arming the dmesg baseline below against the fresh buffer.
  rm -f "$MARK.wedged" "$MARK.hard"
  echo "[$(date -Is)] wedge cleared: the machine booted at $(uptime -s), after the flag was set."
fi

# --- unrecoverable card state, measured against a baseline taken when this armed --------
# Only NEW lines matter. A dmesg buffer already containing an old, recovered fault would
# otherwise make every single pass shout.
HARD=$(timeout -k 5 25 sudo -n dmesg 2>/dev/null \
       | grep -cE 'wait for reset ack|ring gfx timeout|GPU reset begin' || true)
HARD=$(echo "$HARD" | head -1); HARD=${HARD:-0}
PREV_HARD=$(cat "$MARK.hard" 2>/dev/null || echo "$HARD")
echo "$HARD" > "$MARK.hard"
if [ "$HARD" -gt "$PREV_HARD" ]; then
  echo "[$(date -Is)] WEDGED: $((HARD - PREV_HARD)) new unrecoverable signature(s)."
  echo "  Stopping is now the only safe action; an AC cycle is needed."
  touch "$MARK.wedged"
  exit 2
fi

# --- supervisor's own terminal lines -----------------------------------------------------
if [ -f "$SUPLOG" ]; then
  LINES=$(wc -l < "$SUPLOG")
  PREV=$(cat "$MARK.suplog" 2>/dev/null || echo "$LINES")
  echo "$LINES" > "$MARK.suplog"
  if [ "$LINES" -gt "$PREV" ]; then
    NEW=$(tail -n "$((LINES - PREV))" "$SUPLOG")
    echo "$NEW" | grep -qE 'NOT restarting|MAX_RESTARTS|giving up|FATAL' \
      && alert "supervisor is not restarting: $(echo "$NEW" | grep -E 'NOT restarting|MAX_RESTARTS|giving up|FATAL' | tail -2 | tr '\n' ' ')"
  fi
fi

# --- the loop finished, and nothing else will happen -------------------------------------
if [ -f "$STATE" ]; then
  if grep -qE '^\s*-?\s*event:\s*finished' "$STATE" 2>/dev/null \
     || grep -qE 'reason:\s*(target_met|max_rounds|max_timeout)' "$STATE" 2>/dev/null; then
    alert "job $JOB has FINISHED ($(grep -oE 'reason:\s*\w+' "$STATE" | tail -1)). The card is now idle."
  fi

  # --- state not advancing: a hung candidate, which the supervisor cannot see ------------
  AGE=$(( ( $(date +%s) - $(stat -c %Y "$STATE") ) / 60 ))
  if [ "$AGE" -ge "$STALL_MIN" ]; then
    alert "state.yaml has not advanced in ${AGE} min (>= ${STALL_MIN}). A candidate may be hung; the loop would still look alive."
  fi
else
  alert "no state.yaml at $STATE"
fi

# --- the container the runner attaches to ------------------------------------------------
if ! timeout 30 docker inspect -f '{{.State.Running}}' fa-repro 2>/dev/null | grep -q true; then
  alert "container fa-repro is not running; op-evolve attaches and cannot start it itself."
fi

exit $rc
