#!/bin/bash
# Closing sequence. The whole box goes quiet first, because every op figure today was taken
# with four cards loaded and the shared board power budget moves a single-card measurement by
# up to 50%. This is the number tomorrow starts from.
set -u
OUT=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0914__campaign
R=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo
L=$OUT/ledgers/closing.jsonl; : > "$L"

# Wait for quiet. grep -c prints a number and exits 1 on zero matches, so read its stdout and
# ignore its status -- the "0\n0" trap cost two debugging rounds today.
for i in $(seq 1 90); do
  # Count via ps with a bracketed pattern. `pgrep -cf tune_attention` matches its OWN
  # command line -- and the bash -lc wrapper carrying it -- so the count never reaches zero
  # and the loop spins forever. Same family as the pkill -f trap that killed a shell earlier
  # today; third time this pattern has cost something.
  n=$(docker exec fa-repro bash -lc 'ps -eo args | grep -c "[t]une_attention\.py"; true' 2>/dev/null | head -1 | tr -dc 0-9)
  [ "${n:-0}" = "0" ] && { echo "box quiet after $((i*20))s"; break; }
  sleep 20
done
echo "cooling 150s"; sleep 150

for i in 1 2 3 4 5; do
  for impl in asm fused; do
    o=$(timeout 1800 docker exec -e GPU=0 -e PYTHONPATH=/home/lihuzhan/code/aiter-src \
          -e TRITON_CACHE_DIR=/tmp/triton_cache_g0 \
          fa-repro bash -lc "cd $R && exec python3 tools/gfx1250/tune_attention.py --shape llama31-8b --impl $impl" 2>/dev/null | tail -1)
    case "$o" in '{'*) echo "{\"tag\":\"closing|$impl|$i\",\"r\":$o}" >> "$L" ;;
                 *)    echo "{\"tag\":\"closing|$impl|$i\",\"r\":null}" >> "$L" ;; esac
  done
done
echo "closing measurement done"
