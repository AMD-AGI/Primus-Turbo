#!/bin/bash
# usage: mkknob.sh <name> 'sed-expr' ...  -> variants/<name> = variants/on + sed edits on the kernel file
P=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__flydsl/asm-structure/proto
N=$1; shift
rm -rf $P/variants/$N; cp -r $P/../op $P/variants/$N; rm -rf $P/variants/$N/__pycache__ $P/variants/$N/flydsl_fwd/__pycache__
K=$P/variants/$N/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py
sed -i "s/^ASM_STRUCT = False$/ASM_STRUCT = True/" $K
for e in "$@"; do sed -i "$e" $K; done
grep -nE "^(ASM_STRUCT|P2_[A-Z_]+) = " $K | tr '\n' ' '; echo
