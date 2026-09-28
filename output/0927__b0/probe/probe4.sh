#!/bin/bash
# 4-way: fwd harness on g0 and g2, bwd harness on g1 and g3, all at once.
P=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/probe
FJ=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-fwd-20260925-114644
BJ=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934
FWD="cd $FJ/job_context/op && /opt/venv/bin/python3 benchmark.py --arms baseline,beat --arm-path r4=$FJ/rounds/004/op --shapes prod --iters 101"
BWD="cd $BJ/job_context/op && export TORCH_BLAS_PREFER_HIPBLASLT=1 HIPBLASLT_TENSILE_LIBPATH=/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250 && /opt/venv/bin/python3 benchmark.py --arms current,beat --shapes prod --iters 51"
ex() { timeout -s INT 2400 docker exec -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 fa-g$1 bash -c "$2 --json $P/$3.json" > $P/$3.log 2>&1; echo "$3 rc=$? $(grep -c RESULT $P/$3.log) results"; }
ex 0 "$FWD" fwd_Q1g0 & ex 1 "$BWD" bwd_Q1g1 & ex 2 "$FWD" fwd_Q1g2 & ex 3 "$BWD" bwd_Q1g3 & wait
echo PROBE4_DONE
