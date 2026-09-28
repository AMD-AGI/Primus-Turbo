#!/bin/bash
P=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/profile; cd $P
declare -A G=(
 [g1]=SQ_WAVES,SQ_CYCLES,SQ_BUSY_CYCLES,SQ_WAVE_CYCLES,GRBM_GUI_ACTIVE,GRBM_COUNT
 [g2]=SQ_INST_CYCLES_VALU_WMMA,SQ_INSTS_VEC32_VALU_WMMA,SQ_VALU_WMMA_FLOP_BF16,SQ_ITEMS
 [g3]=SQC_ICACHE_REQ,SQC_ICACHE_MISSES,SQC_ICACHE_MISSES_DUPLICATE,SQC_ICACHE_HITS
 [g4]=CHC_REQ_READ,CHC_REQ_READ_128B,CHA_BUSY,CHA_CYCLE
 [g5]=GL1C_GL2_REQ_READ_LEVEL,GL1C_GL2_REQ_WRITE_LEVEL,GL1A_BUSY,GL1A_CYCLE
 [g6]=SPI_RA_REQ_NO_ALLOC,SQ_INSTS_SENDMSG,CPC_CPC_STAT_BUSY
)
for spec in g1:L16 g2:L16 g2:randn g3:L16 g3:randn g4:L16 g5:L16 g6:L16; do
  g=${spec%%:*}; set_=${spec#*:}; T=pmc_${g}_$set_
  RUN_BIN=/opt/venv/bin/rocprofv3 ./tools/run_op.sh $T 400 PM_SET=$set_ PM_N=5 -- --pmc ${G[$g]} --output-format csv --output-file pmc -d $P/runs/pmc/$T -- /opt/venv/bin/python3 tools/pmcrun.py 2>&1 | grep -E "rc=|NEW|dmesg"
  ls $P/runs/pmc/$T 2>/dev/null | tr '\n' ' '; echo
  sleep 5
done
