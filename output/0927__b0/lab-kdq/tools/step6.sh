#!/bin/bash
# after step5r: (a) gate c_k1/c_k1f/c_ku2 (fast/proxy/prod); (b) blocked A/B prod x3 of k_dkdv arms;
# (c) lowered clock (4 bounded bf16 GEMM 32768x4096x14336 before each timed call) prod x3: r19h vs cur.
L=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab-kdq; P=$(readlink -f $L/OP); KB=$(readlink -f $L/tools/kbench.py)
R=$L/tools/run1.sh
A() { for a in "$@"; do printf "%s=%s/%s " $a $P $a; done; }
until grep -qE "STEP5R DONE|STOP" $L/runs/step5r.out; do sleep 10; done
grep -q STOP $L/runs/step5r.out && exit 1
for s in proxy prod fast; do n=$([ $s = fast ] && echo 200 || ([ $s = proxy ] && echo 100 || echo 50))
  $R val6_$s lab_validate.py $s $n $(A cur c_k1 c_k1f c_ku2) || { echo STOP; exit 1; }; done
i=1; for o in "cur c_k1 c_k1f c_ku2" "c_ku2 c_k1f c_k1 cur" "c_k1 cur c_ku2 c_k1f"; do
  $R kb6_prod_blk_p$i $KB prod blk 45 $(A $o) || { echo STOP; exit 1; }; i=$((i+1)); done
i=1; for o in "r19h cur c_epi" "c_epi cur r19h" "cur r19h c_epi"; do
  KB_NG=4 $R kb6_prod_gb_p$i $KB prod gb 45 $(A $o) || { echo STOP; exit 1; }; i=$((i+1)); done
echo STEP6 DONE
