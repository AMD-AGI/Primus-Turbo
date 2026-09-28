#!/bin/bash
# NOT RUN. Remaining card plan (my GPU0 launch of step A was refused by the harness at ~07:40;
# left for the operator). run1.sh = fa-g0 / gpu0 lock; one shape per process; stops on bad dmesg.
set -e
L=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab-kdq; P=$(readlink -f $L/OP)
R=$L/tools/run1.sh; KB=$(readlink -f $L/tools/kbench.py)
A() { for a in "$@"; do printf "%s=%s/%s " $a $P $a; done; }
B() { for a in "$@"; do printf -- "--arm-path %s=%s/%s " $a $P $a; done; }
# A. screen the 4 compile-only follow-ups (1 process), then rotate the best 2 x3
$R kb4_prod_blk_s1 $KB prod blk 45 $(A r19h q1 u2n u2 u2b u2nb u2f qf)
# B. correctness of whatever is kept (expect u2nb bitwise == r19h; u2b == q1)
for s in proxy prod fast; do n=$([ $s = fast ] && echo 200 || ([ $s = proxy ] && echo 100 || echo 50))
  $R val4_$s lab_validate.py $s $n $(A r19h u2nb u2b u2f qf); done
# C. official ruler, 3 rotated processes per shape, r19h + A/A copy (u2n only touches the
#    non-split path: fast runs r19h code, expect ~1.00 there)
cp -r $P/r19h $P/r19h_copy 2>/dev/null || true
for s in prod proxy fast; do i=1
  for o in "r19h u2n q1 r19h_copy" "u2n r19h_copy r19h q1" "r19h_copy q1 u2n r19h"; do
    $R bm_${s}_p$i benchmark.py --shapes $s --iters 101 --json $L/runs/bm_${s}_p$i.json $(B $o); i=$((i+1)); done; done
# D. lowered clock: 4 bf16 GEMMs 32768x4096x14336 before every timed call (bounded, ~6 s of GEMM
#    per process, not a burn loop -- confirm LAB-RULES rule 6 first), 3 rotated processes
i=1; for o in "r19h u2n q1 n1d kq1d" "kq1d n1d q1 u2n r19h" "q1 r19h kq1d u2n n1d"; do
  KB_NG=4 $R kb_prod_gb_p$i $KB prod gb 45 $(A $o); i=$((i+1)); done
