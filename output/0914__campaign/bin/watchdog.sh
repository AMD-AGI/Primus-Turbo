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
STALL=${STALL:-900}
E2E_STALL=${E2E_STALL:-1500}   # one e2e round = inductor compile + 20 steps; 6 min is normal          # seconds without a new ledger line before a stream is "stalled"
PERIOD=${PERIOD:-60}

# Real utilisation, per card. Ledger mtime is not a liveness signal: an exhausted grid
# still appends round_complete every cycle and keeps the file looking fresh.
gpu_use(){ timeout 30 docker exec fa-repro rocm-smi --showuse 2>/dev/null \
           | grep -F "GPU[$1]" | grep -F "GPU use" | awk '{print $NF}' | head -1; }

declare -A PHASE=( [0]=sweep2 [2]=shapes2 [3]=fwd )   # 1 = op-evolve (self-supervised), 3 = e2e loop (below)

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
          | grep -cE 'failed to respond to msg|GPU Hang|wait for reset ack|Memory access fault'; }
# Unrecoverable only. The driver's reset failing to complete is what needed a reboot on
# 2026-09-13; MES timeouts alone are a degraded card that still finishes kernels.
hard_faults(){ timeout 15 dmesg 2>/dev/null | tail -400 \
          | grep -cE 'wait for reset ack|ring gfx timeout|GPU reset begin'; }

while true; do
  T=$(now); F=$(faults); H=$(hard_faults); WEDGE=0; DEGRADED=0
  [ "${F:-0}" -gt 0 ] && DEGRADED=1
  [ "${H:-0}" -gt 0 ] && WEDGE=1
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
    elif [ -f "/tmp/campaign.g${g}.stop" ]; then
      # Parked on purpose by exclusive.sh -- not a stall.
      st=parked
    elif [ "$age" -gt "$STALL" ]; then
      st=stalled
      echo "STALL: gpu$g stream '$ph' has written nothing for ${age}s (threshold ${STALL}s)"
    fi
    u=$(gpu_use "$g"); u=${u:-?}
    # An "ok" stream on a 0%-busy card is the failure this field exists to expose.
    if [ "$st" = ok ] && [ "$u" = "0" ] && [ "$age" -gt 180 ]; then
      st=idle
      echo "IDLE: gpu$g stream '$ph' reports ok but the card is 0% busy and its ledger is ${age}s old -- the grid is probably exhausted"
    fi
    rows="$rows{\"gpu\":$g,\"stream\":\"$ph\",\"state\":\"$st\",\"ledger_age_s\":$age,\"gpu_use\":\"$u\"},"
  done
  # gpu3 reports alongside the others, or "all four cards are busy" cannot be read off the
  # status file -- which is how eight idle minutes went unnoticed.
  e3age=$(( T - $(stat -c %Y "$OUT/ledgers/e2e_ab2.jsonl" 2>/dev/null || echo "$T") ))

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
{"t":$T,"iso":"$(date -Is)","dmesg_faults":${F:-0},"hard_faults":${H:-0},"degraded":$DEGRADED,"wedge":$WEDGE,
 "streams":[${rows%,}],
 "op_evolve":{"ref":"$oe_ref","state_age_s":$oe_age}}
JSON
  sleep "$PERIOD"
done
