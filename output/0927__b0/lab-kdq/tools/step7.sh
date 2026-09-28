#!/bin/bash
# lowered clock with the IMAGE hipBLASLt library (as e2e / profile gburst): 10 bf16 GEMM
# 32768x4096x14336 (~25 ms, ~1.5 PF) before EVERY timed call -> sclk ~1280-1350 MHz, the
# training operating point. Bounded: ~7 s of GEMM per process. 3 rotated processes.
L=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab-kdq; P=$(readlink -f $L/OP); KB=$(readlink -f $L/tools/kbench.py)
A() { for a in "$@"; do printf "%s=%s/%s " $a $P $a; done; }
export KB_BLASLIB=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250 KB_NG=10
i=1; for o in "r19h cur c_epi c_k1" "c_k1 c_epi cur r19h" "cur c_k1 r19h c_epi"; do
  $L/tools/run1.sh kb7_prod_gbimg_p$i $KB prod gb 30 $(A $o) || { echo STOP; exit 1; }; i=$((i+1)); done
echo STEP7 DONE
