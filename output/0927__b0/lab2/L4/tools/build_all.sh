#!/bin/bash
# Compile-only builds for every L4 arm, inside fa-g2, NO flock (no GPU touched).
L=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab2/L4
CH=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-fwd-b0-20260927/job_context/op/current
cfgs="prod:prod proxy:prod fast:prod prod:nc prodg1:prod prodg1:nc fastg1:prod"
for arm in champ ctrl bmajor spread; do
  dir=$L/$arm; [ $arm = champ ] && dir=$CH
  for c in $cfgs; do
    sh=${c%%:*}; var=${c##*:}; tag=${arm}_${sh}_${var}
    rm -rf "${L:?}/isa/${tag:?}"
    docker exec fa-g2 bash -c "cd $L && env COMPILE_ONLY=1 FLYDSL_DUMP_IR=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250 FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_L4b_$tag /opt/venv/bin/python3 tools/compile_fwd_isa.py $dir isa/$tag $var $sh" > $L/logs/$tag.log 2>&1
    rc=$?
    echo "$tag rc=$rc $(tail -n1 $L/logs/$tag.log | cut -c1-40) $(bash $L/tools/isa_stats.sh $L/isa/$tag 2>/dev/null)"
  done
done
