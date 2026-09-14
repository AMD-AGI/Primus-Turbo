#!/bin/bash
# Fleet watchdog. Emits ONE line per actionable event on stdout and rewrites fleet_status.json
# every cycle. Never exits on its own.
#
# Three rules learned the hard way on 2026-09-13:
#  - Liveness is a PID file plus `kill -0`, never pgrep/ps/rocm-smi: all of those walk device
#    or process state and HANG on a wedged card, so the natural diagnostic is the one that
#    dies first.
#  - A wedged GPU means STOP DISPATCHING to it and say so. It does not mean reboot, and it
#    does not mean retry -- the driver's own reset does not complete.
#  - Restart a dead stream by relaunching it. The ledger makes that free: every candidate is
#    tagged and an existing tag is skipped, so a restart resumes rather than repeats.
set -u
OUT=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0914__campaign
STATUS=$OUT/fleet_status.json
STALL=${STALL:-900}          # seconds without a new ledger line before a stream is "stalled"
PERIOD=${PERIOD:-60}

declare -A PHASE=( [0]=gate_asm [2]=gate )   # 1 = op-evolve (self-supervised), 3 = e2e loop (below)

# Own PID file, and refuse to start twice. Stopping this by `pkill -f watchdog.sh` matches the
# killer's OWN command line and takes the caller's shell with it -- that is a documented trap
# and it fired here once already. Stop it with: kill $(cat <this file>).
SELF_PID=$OUT/watchdog.pid
if [ -f "$SELF_PID" ] && kill -0 "$(cat "$SELF_PID" 2>/dev/null)" 2>/dev/null; then
  echo "watchdog already running as $(cat "$SELF_PID"); exiting"; exit 0
fi
echo $$ > "$SELF_PID"; trap 'rm -f "$SELF_PID"' EXIT

now(){ date +%s; }
alive(){ local f=$1; [ -f "$f" ] && kill -0 "$(cat "$f" 2>/dev/null)" 2>/dev/null; }
# grep -c always PRINTS a count and exits 1 when that count is zero, so `|| echo 0` appended
# a second zero and every arithmetic test downstream failed. Let grep print; ignore its status.
faults(){ timeout 15 dmesg 2>/dev/null | tail -200 \
          | grep -cE 'MES\(|GPU Hang|wait for reset ack|Memory access fault'; }

while true; do
  T=$(now); F=$(faults); WEDGE=0
  [ "${F:-0}" -gt 0 ] && WEDGE=1
  rows=""
  # Iterate the map's own keys. A hardcoded GPU list plus `set -u` means retiring a stream
  # from the map kills the watchdog on its next cycle with "PHASE[0]: unbound variable" --
  # which it did, silently, leaving a stale status file that still read "ok".
  for g in "${!PHASE[@]}"; do
    ph=${PHASE[$g]}; led=$OUT/ledgers/g${g}.${ph}.jsonl; pid=${led%.jsonl}.pid
    age=$(( T - $(stat -c %Y "$led" 2>/dev/null || echo 0) ))
    st=ok
    if ! alive "$pid"; then
      st=dead
      if [ "$WEDGE" = 1 ]; then
        echo "WEDGE: gpu$g stream '$ph' is dead and dmesg shows GPU faults -- NOT restarting, card needs a human"
      else
        echo "RESTART: gpu$g stream '$ph' died, relaunching (ledger makes it resume, not repeat)"
        setsid nohup env GPU=$g PHASE=$ph bash "$OUT/bin/queue.sh" \
          > "$OUT/logs/g${g}.${ph}.log" 2>&1 < /dev/null & disown
      fi
    elif [ "$age" -gt "$STALL" ]; then
      st=stalled
      echo "STALL: gpu$g stream '$ph' has written nothing for ${age}s (threshold ${STALL}s)"
    fi
    rows="$rows{\"gpu\":$g,\"stream\":\"$ph\",\"state\":\"$st\",\"ledger_age_s\":$age},"
  done
  # gpu3 reports alongside the others, or "all four cards are busy" cannot be read off the
  # status file -- which is how eight idle minutes went unnoticed.
  e3age=$(( T - $(stat -c %Y "$OUT/ledgers/e2e_ab.jsonl" 2>/dev/null || echo 0) ))

  # GPU3's e2e loop. Not in PHASE because its ledger and restart command differ, but it is
  # watched for the same reason: a one-shot run leaves the card idle the moment it finishes.
  e3=$OUT/ledgers/g3.e2e.pid
  e3st=ok; alive "$e3" || e3st=dead
  rows="$rows{\"gpu\":3,\"stream\":\"e2e_ab\",\"state\":\"$e3st\",\"ledger_age_s\":$e3age},"
  if ! alive "$e3"; then
    if [ "$WEDGE" = 1 ]; then
      echo "WEDGE: gpu3 e2e loop is dead and dmesg shows GPU faults -- NOT restarting"
    else
      echo "RESTART: gpu3 e2e loop died, relaunching"
      setsid nohup bash "$OUT/bin/e2e_loop.sh" > "$OUT/logs/g3.e2e.log" 2>&1 < /dev/null & disown
    fi
  fi

  # op-evolve supervises itself; the watchdog only reports whether its loop is advancing.
  OE=$(ls -dt /home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-* 2>/dev/null | head -1)
  oe_age=-1; oe_ref="none"
  if [ -n "$OE" ]; then
    oe_ref=$(basename "$OE")
    oe_age=$(( T - $(stat -c %Y "$OE/job_context/state.yaml" 2>/dev/null || stat -c %Y "$OE" 2>/dev/null || echo 0) ))
    # A deep round can reason for ~56 min with almost no file writes, so the bar is high.
    [ "$oe_age" -gt 5400 ] && echo "STALL: op-evolve $oe_ref state.yaml unchanged for ${oe_age}s"
  fi

  cat > "$STATUS" <<JSON
{"t":$T,"iso":"$(date -Is)","dmesg_faults":${F:-0},"wedge":$WEDGE,
 "streams":[${rows%,}],
 "op_evolve":{"ref":"$oe_ref","state_age_s":$oe_age}}
JSON
  sleep "$PERIOD"
done
