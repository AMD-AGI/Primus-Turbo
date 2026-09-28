#!/bin/bash
# NOT RUN (card handed to the op-evolve job at 06:45). Card plan for the compile-only arms,
# for the operator to schedule. Every step: one shape per process, dmesg check (run1.sh),
# stops at the first bad dmesg / nonzero rc.
set -e
L=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab-kdq; P=$(readlink -f $L/OP)
R=$L/tools/run1.sh; KB=$(readlink -f $L/tools/kbench.py)
A() { for a in "$@"; do printf "%s=%s/%s " $a $P $a; done; }
B() { for a in "$@"; do printf -- "--arm-path %s=%s/%s " $a $P $a; done; }
# 1. correctness + determinism (expect XARM bitwise: u2n=r19h? no (dqg vs dq ISA same math) ->
#    u2n/ku2 bitwise vs r19h, u2/qu2ku2 bitwise vs q1)
for s in proxy prod fast; do n=$([ $s = fast ] && echo 200 || ([ $s = proxy ] && echo 100 || echo 50))
  $R val3_$s lab_validate.py $s $n $(A r19h u2n ku2 q1 u2 qu2ku2); done
# 2. per-kernel + full-op, prod, 3 rotated processes (blocked ruler)
O1="r19h q1 u2 u2n ku2 qu2ku2"; O2="qu2ku2 ku2 u2n u2 q1 r19h"; O3="u2 r19h qu2ku2 q1 ku2 u2n"
i=1; for o in "$O1" "$O2" "$O3"; do $R kb3_prod_blk_p$i $KB prod blk 45 $(A $o); i=$((i+1)); done
# 3. official ruler (benchmark.py), champion = r19h + A/A copy, 3 rotated processes per shape
cp -r $P/r19h $P/r19h_copy 2>/dev/null || true
for s in prod proxy fast; do i=1
  for o in "r19h q1 qu2ku2 r19h_copy" "qu2ku2 r19h_copy r19h q1" "r19h_copy qu2ku2 q1 r19h"; do
    $R bm_${s}_p$i benchmark.py --shapes $s --iters 101 --json $L/runs/bm_${s}_p$i.json $(B $o); i=$((i+1)); done; done
# 4. lowered clock (bounded GEMM bursts: KB_NG=4 x 32768x4096x14336 before each timed call,
#    ~6 s of GEMM per process -- NOT a burn loop; check LAB-RULES rule 6 with the operator)
i=1; for o in "r19h q1 qu2ku2 kq1d n1d" "n1d kq1d qu2ku2 q1 r19h" "qu2ku2 r19h n1d q1 kq1d"; do
  KB_NG=4 $R kb3_prod_gb_p$i $KB prod gb 45 $(A $o); i=$((i+1)); done
