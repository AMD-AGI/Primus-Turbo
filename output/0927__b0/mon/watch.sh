#!/bin/bash
# B0 4-card campaign watcher. Silent on the happy path; prints one line per problem or round close.
# usage: watch.sh  (reads $M/jobs: lines "<tag> <job-dir> <loop-pid>")
M=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/mon
KFD=(34992 30548 57865 51359)
T0=$(cut -d' ' -f1 /proc/uptime | cut -d. -f1)
declare -A CID2NAME SEEN LASTROUND
refresh_cids() { while read -r id name; do CID2NAME[$id]=$name; done < <(docker ps --no-trunc --format '{{.ID}} {{.Names}}'); }
refresh_cids
while read -r tag dir pid; do LASTROUND[$tag]=$(python3 -c "import yaml;s=yaml.safe_load(open('$dir/job_context/state.yaml'));print(len([r for r in s['rounds'] if 'reflect' in (r.get('progress') or []) or r.get('outcome')=='failed']))"); done < $M/jobs
n=0
while true; do
  # 1. loops alive
  while read -r tag dir pid; do
    kill -0 $pid 2>/dev/null || { [ -z "${SEEN[dead$tag]}" ] && { echo "LOOP GONE $tag pid=$pid: $(tail -2 /home/lihuzhan/code/2026_0910__op-evolve/op-evolve/LOG.$tag 2>/dev/null | tr '\n' ' ' | cut -c1-200)"; SEEN[dead$tag]=1; }; }
  done < $M/jobs
  # 0. our containers: if anyone stopped one, the operator must shut everything down (user rule, 2026-09-27)
  for c in fa-g0 fa-g2 fa-g3; do
    st=$(docker inspect -f '{{.State.Status}}' $c 2>/dev/null || echo absent)
    [ "$st" != running ] && [ -z "${SEEN[ctr$c]}" ] && { echo "OUR CONTAINER GONE: $c is $st -- report to user; recreate after 30 min without reply (day-2 rule)"; SEEN[ctr$c]=1; }
  done
  # 1b. foreign containers (anything running that is not ours)
  for c in $(docker ps --format '{{.Names}}'); do
    # user rule 2026-09-28: during our booked time, stop foreign containers immediately and report
    case $c in fa-g0|fa-g2|fa-g3) ;; *) [ -z "${SEEN[fc$c]}" ] && { info=$(docker inspect -f '{{.Config.Image}} started {{.State.StartedAt}} mounts={{range .Mounts}}{{.Source}} {{end}}' $c 2>/dev/null); docker stop -t 10 $c >/dev/null 2>&1; echo "FOREIGN CONTAINER STOPPED: $c ($info) rc=$?"; SEEN[fc$c]=1; } ;; esac
  done
  # 2. new amdgpu faults since the watcher started
  timeout 20 sudo -n dmesg 2>/dev/null | awk -v t0=$T0 '{ts=substr($1,2)+0} ts>t0' | grep -v "0002:04:00.0" | grep -iE "ring buffer is full|failed to respond|GCVM_L2_PROTECTION|page fault|gpu reset|hang" | head -3 | while read -r l; do k=$(echo "$l" | cut -c1-40); [ -z "${SEEN[$k]}" ] && echo "DMESG: $l"; done
  timeout 20 sudo -n dmesg 2>/dev/null | awk -v t0=$T0 '{ts=substr($1,2)+0} ts>t0' | grep -v "0002:04:00.0" | grep -qiE "ring buffer is full|failed to respond|GCVM_L2_PROTECTION|gpu reset" && T0=$(cut -d' ' -f1 /proc/uptime | cut -d. -f1)
  # 3. per-card KFD ownership: a process on card N must live in container fa-gN
  (( n % 10 == 0 )) && refresh_cids
  for p in /sys/class/kfd/kfd/proc/*; do
    [ -d "$p" ] || continue; pid=$(basename $p)
    cid=$(grep -oE '[0-9a-f]{64}' /proc/$pid/cgroup 2>/dev/null | head -1)
    who=${CID2NAME[$cid]:-host}
    for g in $(cat $p/queues/*/gpuid 2>/dev/null | sort -u); do
      card=-1; for i in 0 1 2 3; do [ "${KFD[$i]}" = "$g" ] && card=$i; done
      [ $card -lt 0 ] && continue
      if [ "$who" != "fa-g$card" ] && [ "$who" != "host" ] || { [ "$who" = host ] && [ "$(ps -o user= -p $pid 2>/dev/null)" != lihuzhan ]; }; then
        k="own$pid$g"; [ -z "${SEEN[$k]}" ] && { echo "FOREIGN/WRONG GPU PROCESS: pid $pid user=$(ps -o user= -p $pid 2>/dev/null) container=$who card=$card cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-80)"; SEEN[$k]=1; }
        continue
      fi
      if [ "$who" != "fa-g$card" ]; then k="own$pid$g"; [ -z "${SEEN[$k]}" ] && { echo "WRONG CARD: pid $pid ($who, $(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-80)) has queues on card $card"; SEEN[$k]=1; }; fi
    done
  done
  # 4. round closes (so the operator can send the summary table)
  while read -r tag dir pid; do
    c=$(python3 -c "import yaml;s=yaml.safe_load(open('$dir/job_context/state.yaml'));print(len([r for r in s['rounds'] if 'reflect' in (r.get('progress') or []) or r.get('outcome')=='failed']))" 2>/dev/null)
    [ -n "$c" ] && [ "$c" != "${LASTROUND[$tag]}" ] && { echo "ROUND CLOSED $tag: $(python3 -c "import yaml;s=yaml.safe_load(open('$dir/job_context/state.yaml'));r=s['rounds'][-1];print('r%s %s accepted=%s gain=%s prod=%s'%(r['round'],r.get('mode'),r.get('accepted'),r.get('gain'),(r.get('tflops') or {}).get('prod')))")"; LASTROUND[$tag]=$c; }
  done < $M/jobs
  n=$((n+1)); sleep 30
done
