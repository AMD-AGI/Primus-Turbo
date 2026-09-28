#!/bin/bash
B=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/ruler/bwd
r() { out=$($B/run1.sh "$@"); echo "$out"; echo "$out" | head -1 | grep -q "rc=0 .* bad=0" || { echo "STOP: $1 failed"; exit 1; }; }
r A02 --arms current_copy,current
r A03 --arms current,current_copy
r A04 --arms current_copy,current
r I01 --arms current,current_copy --block 1
r I02 --arms current_copy,current --block 1
r BI1 --arms current,beat --block 1
r BB1 --arms current,beat
r BI2 --arms beat,current --block 1
r BB2 --arms beat,current
r BI3 --arms current,beat --block 1
r BB3 --arms beat,current
r C01 --arms current,r024,r025,r019
r C02 --arms r019,r025,r024,current
r C03 --arms r024,current,r019,r025
r CI1 --arms current,r024,r025,r019 --block 1
r CI2 --arms r019,r025,r024,current --block 1
echo BATCH-DONE
