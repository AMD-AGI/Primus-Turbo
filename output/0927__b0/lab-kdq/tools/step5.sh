#!/bin/bash
# round 2 on GPU0 (user-approved): rebased arms vs champion `cur` (r29 = r19h + u2n + g86), prod + proxy,
# 3 rotated processes each, per-kernel (k_dq) + full op, blocked ruler.
L=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab-kdq; P=$(readlink -f $L/OP); KB=$(readlink -f $L/tools/kbench.py)
A() { for a in "$@"; do printf "%s=%s/%s " $a $P $a; done; }
O1="cur r19h c_u2nb c_u2b c_u2f c_u2fb c_epi c_ku2b"
O2="c_ku2b c_epi c_u2fb c_u2f c_u2b c_u2nb r19h cur"
O3="c_u2f cur c_ku2b c_u2nb r19h c_epi c_u2b c_u2fb"
for s in prod proxy; do i=1; for o in "$O1" "$O2" "$O3"; do
  $L/tools/run1.sh kb5_${s}_blk_p$i $KB $s blk 45 $(A $o) || { echo STOP; exit 1; }; i=$((i+1)); done; done
echo STEP5 DONE
