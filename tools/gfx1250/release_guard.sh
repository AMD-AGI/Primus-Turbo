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
set -u
JOB=${1:?usage: release_guard.sh <job-ref> [interval_s]}
INTERVAL=${2:-60}

OE=${OE_ROOT:-/home/lihuzhan/code/2026_0910__op-evolve/op-evolve}
REPO=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo
NOTE="$REPO/output/0921__flydsl/RELEASED.md"
TOKEN=/home/lihuzhan/gfx1250.owner
MARK="$REPO/output/0921__flydsl/.patrol"
BOOT_AT=$(date -d "$(uptime -s)" +%s)

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
  rounds=$(grep -c '^- round:' "$OE/artifacts/$JOB/job_context/state.yaml" 2>/dev/null || echo '?')
  best=$(grep -m1 'best_round' "$OE/artifacts/$JOB/job_context/state.yaml" 2>/dev/null | tr -d ' ')

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

# Capture the supervisor once, by job ref, using awk so the pattern is a variable rather
# than a literal in this process's command line.
SUP_PID=$(ps -eo pid,args | awk -v j="supervise_job.sh --job $JOB" 'index($0,j)>0 && $2 ~ /bash|sh$/ {print $1; exit}')
if [ -z "${SUP_PID:-}" ]; then
  log "no supervisor running for $JOB -- refusing to arm, since there is nothing to guard"
  exit 1
fi

log "release guard armed for $JOB (supervisor pid $SUP_PID; checks every ${INTERVAL}s; releases on first anomaly, never restarts)"

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
  #    it was armed for. This box has already produced two self-match bugs today from the
  #    same family. A PID captured once and checked with kill -0 cannot self-match, and
  #    kill -0 is a single signal check that cannot block on the driver the way pgrep can.
  if ! kill -0 "$SUP_PID" 2>/dev/null; then
    release "supervisor 自己没了" "pid $SUP_PID (captured at arm time) is gone"
  fi

  sleep "$INTERVAL"
done
