#!/bin/bash
# 3 rotated processes per shape (A/A current_copy in each), then beat alone per shape.
L=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab-bwd-r19; OP=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab-bwd-r19/oe/artifacts/job/job_context/op
A() { for a in "$@"; do printf -- "--arm-path %s=%s/%s " $a $OP $a; done; }
O1="current r019 r19h current_copy"; O2="r19h current_copy current r019"; O3="r019 current current_copy r19h"
for s in fast proxy prod; do
  i=1; for o in "$O1" "$O2" "$O3"; do
    $L/run1.sh t_${s}_p$i benchmark.py --shapes $s --iters 101 --json $L/runs/t_${s}_p$i.json $(A $o) || exit 1
    { [ -s $L/runs/t_${s}_p$i.dmesg.bad ] || ! grep -q "^rc=0" $L/runs/t_${s}_p$i.log; } && { echo STOP; exit 1; }
    i=$((i+1)); done
  $L/run1.sh beat_${s} benchmark.py --shapes $s --iters 101 --json $L/runs/beat_${s}.json --arms beat || exit 1
  { [ -s $L/runs/beat_${s}.dmesg.bad ] || ! grep -q "^rc=0" $L/runs/beat_${s}.log; } && { echo STOP; exit 1; }
done
echo BATCH DONE
