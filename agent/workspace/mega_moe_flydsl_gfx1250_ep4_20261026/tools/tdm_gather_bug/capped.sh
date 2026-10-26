#!/bin/bash
# capped.sh SECONDS LOG cmd...: run cmd in its own process group; when the time is up, kill the group
# and every process still running one of cmd's .py scripts (torchrun workers start their own
# sessions), so a hung collective never keeps the GPUs spinning.
limit=$1; log=$2; shift 2
ulimit -c 0
scripts=()
for a in "$@"; do [[ $a == *.py ]] && scripts+=("$a"); done
setsid "$@" > "$log" 2>&1 &
pid=$!
(
    sleep "$limit"
    kill -9 -- -"$pid" 2>/dev/null && echo "KILLED after ${limit}s" >> "$log"
    for s in "${scripts[@]}"; do pkill -9 -f "$s"; done
) &
watchdog=$!
wait "$pid"; rc=$?
kill "$watchdog" 2>/dev/null
pkill -9 -g "$pid" 2>/dev/null
for s in "${scripts[@]}"; do
    pgrep -f "$s" | grep -vx -e "$$" -e "$watchdog" | xargs -r kill -9 2>/dev/null
done
echo "exit $rc"
