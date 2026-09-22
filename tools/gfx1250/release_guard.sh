#!/bin/bash
# Release the node on the first anomaly. Never restart anything.
#
#   release_guard.sh <job-ref> [interval_s]
#
# This is the INVERSE of supervise_job.sh, and it is meant to outrank it. The supervisor's
# whole purpose is to keep a job alive: any death and it resumes, backing off 60s -> 900s,
# up to MAX_RESTARTS=60. That is the right behaviour while the node is ours. It is the
# wrong behaviour the moment the node stops being ours.
#
# The operator is asleep and a colleague takes this node next. Their first move -- docker
# stop, a reboot, taking the card -- looks to the supervisor exactly like a failure to
# recover from, so it would fight them for the machine and burn an agent per attempt while
# doing it. So: on the first sign that the node has changed hands or broken, stop our side
# cleanly, drop the card token, write a note, and exit. A human decides what happens next.
#
# Stopping is done at a MODULE BOUNDARY via `op-evolve stop`, not with a signal. Killing a
# process with work in flight on the GPU leaves an unkillable D-state process behind each
# time, which pushes a recoverable card toward one that needs a power cycle -- the exact
# outcome we are trying to spare whoever gets this node next.
#
# WHY IT SLEPT THROUGH 2026-09-21. Committed in dd1aec29 with no launcher anywhere: no
# cron, no systemd unit, no call from patrol.sh. It never ran, and since it wrote no pid
# file, no log and no sentinel, nothing on disk could show that it had not. patrol.sh now
# starts it (opt-in, PATROL_ARM_GUARD=1) and it now writes $PIDFILE. Separately, its four
# original checks would not have caught that wedge anyway: the box did not reboot, dmesg
# carried no reset signature at all, fa-repro stayed up, and the supervisor stayed alive --
# parked forever in gpu_ok()'s 900 s backoff. Check 5 exists for exactly that state.
#
# TESTABILITY. NOTE/TOKEN/MARK are overridable. They used to be hardcoded to production
# paths, so smoke-testing this script performed a real node release -- which is how a test
# on 2026-09-22 deleted the live card token and wrote a bogus RELEASED.md. To exercise it:
#   GUARD_NOTE=/tmp/t.md GUARD_TOKEN=/tmp/t.owner GUARD_MARK=/tmp/t GUARD_PID_FILE=/tmp/t.pid \
#     bash release_guard.sh <job> 5
set -u
JOB=${1:?usage: release_guard.sh <job-ref> [interval_s]}
INTERVAL=${2:-60}

OE=${OE_ROOT:-/home/lihuzhan/code/2026_0910__op-evolve/op-evolve}
REPO=${REPO:-/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo}
NOTE=${GUARD_NOTE:-$REPO/output/0921__flydsl/RELEASED.md}
TOKEN=${GUARD_TOKEN:-/home/lihuzhan/gfx1250.owner}
MARK=${GUARD_MARK:-$REPO/output/0921__flydsl/.patrol}
PIDFILE=${GUARD_PID_FILE:-/tmp/gfx1250-release-guard.pid}
STATE="$OE/artifacts/$JOB/job_context/state.yaml"
SUPLOG="$OE/artifacts/supervisor.log"
BOOT_AT=$(date -d "$(uptime -s)" +%s)

# A guard that RELEASES a node on a false positive is expensive -- it stops the job and
# hands the machine away. So its staleness trigger is deliberately looser than patrol.sh's
# alert threshold: opt rounds on this job have run 66 and 70 minutes, so 90 would fire
# inside a healthy deep round.
STALE_MIN=${GUARD_STALE_MIN:-180}
PARKED_STREAK=0

log(){ echo "[$(date -Is)] $*"; }

release(){
  local why="$1" detail="$2"
  log "RELEASING THE NODE: $why"

  # Module-boundary stop first. --force adds a SIGTERM to the loop if it is wedged inside
  # a module, which is still not a signal to anything holding a kernel.
  ( cd "$OE" && export PATH="$OE/.venv/bin:$PATH"
    timeout 300 .venv/bin/op-evolve stop --job "$JOB" 2>&1 | tail -3
    sleep 20
    timeout 120 .venv/bin/op-evolve stop --job "$JOB" --force 2>&1 | tail -2 ) || true

  # Then the supervisor itself, by process group, so it cannot resume behind us. Its PID
  # is found by matching the job ref rather than the script name, because a pattern that
  # names the script also names whoever is asking -- that has already produced a false
  # "already running" on this box today.
  for pid in $(ps -eo pid,args | awk -v j="$JOB" '$0 ~ "supervise_job.sh --job " j {print $1}'); do
    kill -TERM -"$(ps -o pgid= "$pid" | tr -d ' ')" 2>/dev/null || true
  done

  rm -f "$TOKEN"

  local rounds best
  rounds=$(grep -c '^- round:' "$STATE" 2>/dev/null || echo '?')
  best=$(grep -m1 'best_round' "$STATE" 2>/dev/null | tr -d ' ')

  cat > "$NOTE" <<EOF
# 节点已释放 -- $(date -Is)

**没有做任何自动重启。** 这是按操作者睡前的明确要求：一旦出现挂卡、容器被停、机器重启
或其它异常，就把节点让给同事，不要自动把任务拉起来。

## 触发原因

$why

\`\`\`
$detail
\`\`\`

## 作业账目（停下来的那一刻）

- 作业：\`$JOB\`
- 轮数：$rounds
- $best  —— 被接受的候选已落盘在 \`rounds/<best_round>/op/\` 与 \`job_context/op/current/\`

停止走的是 **module 边界**（\`op-evolve stop\`），不是信号，所以没有任何进程是在 kernel
在途时被杀的。持卡令牌 \`$TOKEN\` 已删除。

## 给接手这台机器的人

这边**不会**再自动启动任何东西。supervisor 已停，容器守护已停。
如果 dmesg 里有 \`wait for reset ack\` / \`ring gfx timeout\` / \`GPU reset begin\`，
那张卡需要一次 AC-cycle 才能用；只有带 \`timeout\` 的 \`dmesg\` 读和 \`/sys\` 读是安全的，
\`rocm-smi\` / \`pgrep\` / \`ps -o wchan\` 在挂卡时会自己挂住。

## 要恢复这个作业（由人决定，不要自动做）

\`\`\`bash
cd $OE && export PATH="\$PWD/.venv/bin:\$PATH"
bash $REPO/tools/gfx1250/card_ok.sh            # 先确认卡真的能算
docker start fa-repro
setsid nohup env PATH="\$PATH" tools/supervise_job.sh --job $JOB >/dev/null 2>&1 &
\`\`\`
EOF
  log "wrote $NOTE"
  exit 0
}

# Capture the supervisor by job ref, using awk so the pattern is a variable rather than a
# literal in this process's command line.
find_sup(){ ps -eo pid,args | awk -v j="supervise_job.sh --job $JOB" 'index($0,j)>0 && $2 ~ /bash|sh$/ {print $1; exit}'; }

# Bounded retry rather than a silent `exit 1`. Launched the way this repo launches
# background work (>/dev/null 2>&1 &), an immediate refusal to arm is indistinguishable
# from a healthy guard. A guard that declines to arm must leave a sentinel on disk, not a
# line on a discarded stdout.
SUP_PID=""
for _try in $(seq 1 10); do
  SUP_PID=$(find_sup)
  [ -n "${SUP_PID:-}" ] && break
  log "no supervisor for $JOB yet (try $_try/10); retrying in ${INTERVAL}s"
  sleep "$INTERVAL"
done
if [ -z "${SUP_PID:-}" ]; then
  touch "$MARK.guard-unarmed"
  log "GAVE UP ARMING: no supervisor for $JOB after 10 tries; wrote $MARK.guard-unarmed"
  exit 1
fi
rm -f "$MARK.guard-unarmed"

log "release guard armed for $JOB (supervisor pid $SUP_PID; checks every ${INTERVAL}s; releases on first anomaly, never restarts)"

# Liveness observable from disk. Without this, "is the guard running?" had no answer, which
# is why its absence went unnoticed for fourteen hours.
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"; exit 0' INT TERM
trap 'rm -f "$PIDFILE"' EXIT

while true; do
  # 1. The machine rebooted under us.
  now_boot=$(date -d "$(uptime -s)" +%s)
  [ "$now_boot" -ne "$BOOT_AT" ] && release "机器重启了" "boot time moved: $BOOT_AT -> $now_boot"

  # 2. Unrecoverable card state. Bounded standalone dmesg read is the only safe probe.
  hard=$(timeout -k 5 25 sudo -n dmesg 2>/dev/null \
         | grep -cE 'wait for reset ack|ring gfx timeout|GPU reset begin' || true)
  hard=$(echo "$hard" | head -1)
  if [ "${hard:-0}" -gt 0 ]; then
    touch "$MARK.wedged"
    release "GPU 挂死（不可恢复签名）" "$(timeout -k 5 25 sudo -n dmesg 2>/dev/null | grep -E 'wait for reset ack|ring gfx timeout|GPU reset begin' | tail -3)"
  fi

  # 3. Somebody stopped the container. op-evolve attaches and cannot start it; restarting
  #    it ourselves is precisely what we were told not to do.
  if ! timeout 30 docker inspect -f '{{.State.Running}}' fa-repro 2>/dev/null | grep -q true; then
    release "容器 fa-repro 被停掉了" "docker inspect fa-repro -> $(timeout 30 docker inspect -f '{{.State.Running}}|{{.State.ExitCode}}' fa-repro 2>&1 | head -1)"
  fi

  # 4. Our own supervisor is gone.
  #
  #    NOT `ps -eo args | grep -q "supervise_job.sh --job $JOB"`. The grep process's own
  #    command line contains that exact string, so depending on whether ps snapshots it
  #    the test can be permanently true -- a guard that never fires on the one condition
  #    it was armed for. A PID captured once and checked with kill -0 cannot self-match,
  #    and kill -0 cannot block on the driver the way pgrep can.
  if ! kill -0 "$SUP_PID" 2>/dev/null; then
    release "supervisor 自己没了" "pid $SUP_PID (captured at arm time) is gone"
  fi

  # 5. The supervisor is ALIVE but parked forever in gpu_ok()'s 900 s backoff. This is the
  #    state the 2026-09-21 16:01:54 wedge actually produced, and the reason checks 1-4 all
  #    stayed false while the card was unusable for fourteen hours. patrol.sh's own header
  #    calls this terminal state #3. Two consecutive observations, since one could be
  #    transient, and only the last few lines, so a recovered old backoff cannot re-fire.
  if tail -3 "$SUPLOG" 2>/dev/null \
     | grep -qE 'GPU not visible to rocminfo|NOT restarting into a dead device|MAX_RESTARTS|giving up'; then
    PARKED_STREAK=$((PARKED_STREAK+1))
  else
    PARKED_STREAK=0
  fi
  if [ "$PARKED_STREAK" -ge 2 ]; then
    touch "$MARK.wedged"
    release "supervisor 停在 gpu_ok() 无限 backoff（卡对 rocminfo 不可见）" "$(tail -4 "$SUPLOG" 2>/dev/null)"
  fi

  # 6. patrol.sh latched a wedge. Honour another watcher's verdict instead of re-deriving
  #    it -- but only if the latch postdates this boot, since patrol clears it by boot time.
  if [ -f "$MARK.wedged" ] && [ "$(stat -c %Y "$MARK.wedged" 2>/dev/null || echo 0)" -gt "$BOOT_AT" ]; then
    release "patrol 已置位 wedge latch" "$MARK.wedged @ $(date -Is -d @"$(stat -c %Y "$MARK.wedged")")"
  fi

  # 7. A candidate hung: the loop is alive and looks healthy, state.yaml is frozen. This is
  #    the failure mode this project produces most, and supervise_job.sh only supervises the
  #    loop DYING, so nothing else catches it.
  if [ -f "$STATE" ]; then
    age_min=$(( ( $(date +%s) - $(stat -c %Y "$STATE") ) / 60 ))
    if [ "$age_min" -ge "$STALE_MIN" ]; then
      release "state.yaml 冻结 ${age_min} 分钟（>= ${STALE_MIN}），候选疑似挂起" \
              "$STATE last modified $(date -Is -d @"$(stat -c %Y "$STATE")")"
    fi
  fi

  sleep "$INTERVAL"
done
