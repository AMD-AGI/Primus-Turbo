#!/bin/bash
# M8 card runs: toy (proxy) first in its own process, then prod timing, prod non-causal timing, prod PMC cycles.
set -u
R=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0930__roofline; M=$R/m8; A=$M/arms
H=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/fwd-nospec/harness
ENVS="env OE_PHYS_GPU=0 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250 PYTHONDONTWRITEBYTECODE=1 TORCH_BLAS_PREFER_HIPBLASLT=1 HIPBLASLT_TENSILE_LIBPATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250"
AP=""; for a in base noexp nosm nomask nobar wl; do AP="$AP --arm-path $a=$A/$a"; done; AP="$AP --arm-path asm=$H/beat"
cool(){ sleep 20; }
AMD_SERIALIZE_KERNEL=3 $R/tools/run_mb.sh m8_toy 900 $H -- $ENVS FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_m8_toy /opt/venv/bin/python3 benchmark.py $AP --shapes proxy --iters 21 --warmup-seconds 1 --json $R/runs/m8_toy.json || exit 1
cool
$R/tools/run_mb.sh m8_prod 1500 $H -- $ENVS FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_m8_prod /opt/venv/bin/python3 benchmark.py $AP --shapes prod --iters 27 --warmup-seconds 2 --json $R/runs/m8_prod.json || exit 1
cool
$R/tools/run_mb.sh m8_prod_nc 900 $H -- $ENVS FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_m8_nc /opt/venv/bin/python3 benchmark.py --arm-path base=$A/base --arm-path asm=$H/beat --shapes prod --non-causal --iters 27 --warmup-seconds 2 --json $R/runs/m8_prod_nc.json || exit 1
cool
DA=""; for a in base noexp nosm nomask nobar wl; do DA="$DA $a=$A/$a"; done; DA="$DA asm=$H/beat base_nc=$A/base:nc asm_nc=$H/beat:nc"
$R/tools/run_mb.sh m8_pmc 1200 $M -- $ENVS FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_m8_pmc /opt/venv/bin/rocprofv3 --pmc GRBM_GUI_ACTIVE,SQ_WAVES,GRBM_COUNT --output-format csv --output-file pmc -d $R/runs/pmc/m8 -- /opt/venv/bin/python3 m8drive.py prod 7 $DA || exit 1
echo M8ALLDONE
