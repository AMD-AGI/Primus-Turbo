#!/bin/bash
B=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/ruler/bwd
until [ -f $B/runs/BI1.dmesg.bad ] && grep -q rc= $B/runs/BI1.log; do sleep 10; done
[ -s $B/runs/BI1.dmesg.bad ] && { echo "STOP: BI1 dmesg"; exit 1; }
grep -q "rc=0" $B/runs/BI1.log || { echo "STOP: BI1 rc"; exit 1; }
echo "BI1 rc=0 (via batch1)"
r() { out=$($B/run1.sh "$@"); echo "$out"; echo "$out" | head -1 | grep -q "rc=0 .* bad=0" || { echo "STOP: $1 failed"; exit 1; }; }
r C01 --arms current,r024,r025,r019
r BB1 --arms current,beat
r C02 --arms r019,r025,r024,current
r BI2 --arms beat,current --block 1
r C03 --arms r024,current,r019,r025
r BB2 --arms beat,current
r CI1 --arms current,r024,r025,r019 --block 1
r BI3 --arms current,beat --block 1
r BB3 --arms beat,current
r CI2 --arms r019,r025,r024,current --block 1
echo BATCH-DONE
