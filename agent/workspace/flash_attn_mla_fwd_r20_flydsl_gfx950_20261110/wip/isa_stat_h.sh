#!/bin/bash
# Like isa_stat.sh with a chosen shape: wip/isa_stat_h.sh PKG_DIR TAG B,S,H
set -e
cd /home/xiaompen/Turbo/agent/workspace/flash_attn_mla_fwd_r20_flydsl_gfx950_20261110
PKG=$1
TAG=$2
SHAPE=$3
docker exec -w "$PWD" -e PYTHONPATH=/tmp/aiter_0124:/home/xiaompen/Turbo:/tmp/turbo_overlay -e HIP_VISIBLE_DEVICES=0 \
  -e FLYDSL_DUMP_IR=1 -e FLYDSL_DEBUG_DUMP_ASM=1 -e FLYDSL_DUMP_DIR=/tmp/isa_$TAG xiaoming-dev bash -c \
  "rm -rf /tmp/isa_$TAG; python wip/smoke.py $PKG $SHAPE 2>&1 | grep -v '^\[aiter\]\|flydsl.compile' | tail -1; \
   f=\$(find /tmp/isa_$TAG -name '*final_isa*' | head -1); mkdir -p wip/isa_r3; cp \$f wip/isa_r3/$TAG.s; chmod a+rw wip/isa_r3/$TAG.s; \
   grep -E '^\s+\.(vgpr_count|agpr_count|vgpr_spill_count|private_segment_fixed_size):' \$f | tr -s ' ' | tr '\n' ' '; echo"
