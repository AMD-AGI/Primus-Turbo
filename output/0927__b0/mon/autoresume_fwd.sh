#!/bin/bash
# Wait for the fwd loop to exit after an operator stop, then resume it -- unless any fa-gN was stopped by someone.
OLD=$1; M=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/mon
while kill -0 $OLD 2>/dev/null; do sleep 30; done
for c in fa-g0 fa-g1 fa-g2 fa-g3; do [ "$(docker inspect -f '{{.State.Status}}' $c 2>/dev/null)" = running ] || { echo "NOT RESUMING: $c not running"; exit 1; }; done
cd /home/lihuzhan/code/2026_0910__op-evolve/op-evolve && setsid nohup env PATH="$PWD/.venv/bin:$HOME/bin:$PATH" op-evolve resume --job gfx1250-flydsl-attn-fwd-b0-20260927 >> LOG.fwd-b0 2>&1 < /dev/null &
sleep 20
P=$(ps -eo pid,cmd | grep "[.]venv/bin/op-evolve resume --job gfx1250-flydsl-attn-fwd-b0" | awk '{print $1}')
sed -i "s/^fwd-b0 \(.*\) $OLD$/fwd-b0 \1 $P/" $M/jobs
echo "fwd resumed pid=$P"; tail -2 /home/lihuzhan/code/2026_0910__op-evolve/op-evolve/LOG.fwd-b0
