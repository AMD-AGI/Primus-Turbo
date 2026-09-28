#!/bin/bash
# re-run of kb5_prod_blk_p1/p2 (overlapped the foreign hipblaslt-bench on GPU0, 09:00:31-~09:10 UTC)
L=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab-kdq; P=$(readlink -f $L/OP); KB=$(readlink -f $L/tools/kbench.py)
A() { for a in "$@"; do printf "%s=%s/%s " $a $P $a; done; }
until grep -qE "STEP5 DONE|STOP" $L/runs/step5.out; do sleep 10; done
O1="cur r19h c_u2nb c_u2b c_u2f c_u2fb c_epi c_ku2b"
O2="c_ku2b c_epi c_u2fb c_u2f c_u2b c_u2nb r19h cur"
$L/tools/run1.sh kb5_prod_blk_p1r $KB prod blk 45 $(A $O1) || { echo STOP; exit 1; }
$L/tools/run1.sh kb5_prod_blk_p2r $KB prod blk 45 $(A $O2) || { echo STOP; exit 1; }
echo STEP5R DONE
