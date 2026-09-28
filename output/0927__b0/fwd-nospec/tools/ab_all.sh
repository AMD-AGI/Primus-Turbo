#!/bin/bash
# 3 conditions x 3 rotated arm orders, one card process each (tools/run_op.sh: lock + clock sampler + dmesg).
D=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/fwd-nospec; cd $D
A13=$D/arms/fwd_r13; ANS=$D/arms/fwd_r13ns; AS=$D/harness/beat
i=0
for ord in r13,r13ns,asm r13ns,asm,r13 asm,r13,r13ns; do
  i=$((i+1))
  ap=""; for a in ${ord//,/ }; do case $a in r13) ap="$ap --arm-path r13=$A13";; r13ns) ap="$ap --arm-path r13ns=$ANS";; asm) ap="$ap --arm-path asm=$AS";; esac; done
  tools/run_op.sh i_randn_blk_p$i 900 OE_PHYS_GPU=0 -- harness/benchmark.py $ap --shapes prod --json $D/runs/i_randn_blk_p$i.json || exit 1
  tools/run_op.sh ii_real_blk_p$i 1200 AB_COND=blk AB_ORDER=$ord AB_JSON=$D/runs/ii_real_blk_p$i.json -- tools/ab.py || exit 1
  tools/run_op.sh iii_real_gb_p$i 1200 AB_COND=gb AB_ORDER=$ord AB_JSON=$D/runs/iii_real_gb_p$i.json -- tools/ab.py || exit 1
done
echo ALL_DONE
