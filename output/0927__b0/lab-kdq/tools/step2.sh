#!/bin/bash
L=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab-kdq; P=$(readlink -f $L/OP); KB=$(readlink -f $L/tools/kbench.py)
A() { for a in "$@"; do printf "%s=%s/%s " $a $P $a; done; }
i=1; for o in "r19h q1 u2 u2n ku2 qu2ku2" "qu2ku2 ku2 u2n u2 q1 r19h" "u2 r19h qu2ku2 q1 ku2 u2n"; do
  $L/tools/run1.sh kb3_prod_blk_p$i $KB prod blk 45 $(A $o) || { echo STOP; exit 1; }; i=$((i+1)); done
echo STEP2 DONE
