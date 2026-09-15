#!/bin/bash
# Supervisor: restart sched.py if it CRASHES. It cannot detect a hang -- the only detector
# for that is a stale heartbeat in status.json, which is why sched.py rewrites it every 5 s
# even while a job is running. Accepted risk: sched.py can only hang if Popen itself blocks,
# which needs a wedged docker daemon.
#
# Stop with: kill $(cat output/0915__opt/sched.pid)
# Pause with: touch output/0915__opt/STOP
# Never `pkill -f run.sh` or `pkill -f sched.py` -- the pattern matches the killer's own
# command line, which has already taken a shell down once in this campaign.
cd /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo || exit 1
while true; do
  python3 tools/gfx1250/sched.py
  rc=$?
  [ $rc -eq 3 ] && { echo "already running, exiting supervisor"; exit 3; }
  echo "sched.py exited rc=$rc at $(date -Is); restarting in 5s"
  sleep 5
done
