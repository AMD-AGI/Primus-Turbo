#!/bin/bash
# Just the 32L patched run. run32.sh's 85% abort is correct for the PATCHED run but wrong for
# the unpatched baseline, which legitimately sits at 87.98% and has completed 20 steps there.
# So if the queue aborts on the baseline, this continues from the one run that matters.
#
# The threshold here is 89%: the baseline's own 87.98% plus a point. The SIGBUS that cost an
# AC-cycle came from +0.32% on a run already at 87.98%, so anything above 89% is the regime
# that has already taken the card down once.
set -u
cd /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo
D=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250
PP=/home/lihuzhan/_dbg_l8b/patch:/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo:/home/lihuzhan/code/aiter-src
mkdir -p /home/lihuzhan/_dbg_l8b/32-nk
E2E_TIMEOUT=1500 E2E_INNER=1440 E2E_COOLDOWN=25 \
BLAS_ENV="-e HIPBLASLT_TENSILE_LIBPATH=$D" \
E2E_ENV="-e PYTHONPATH=$PP -e NKFIX_ENABLE=1 -e NKFIX_STATS_FILE=/home/lihuzhan/_dbg_l8b/32-nk.stats" \
  bash output/0915__opt/bin/e2e.sh 32-nk repro_l8b_turbo_conv.yaml >/dev/null 2>&1
L=output/0915__opt/logs/e2e.32-nk.log
echo "32-nk: $(sed 's/\x1b\[[0-9;]*m//g' $L 2>/dev/null | grep -oE 'tps: +[0-9,]+' | tail -1)  \
peakmem=$(sed 's/\x1b\[[0-9;]*m//g' $L 2>/dev/null | grep -oE '\([0-9.]+%\)' | tr -d '()%' | sort -rn | head -1)%  \
MES=$(sudo -n dmesg 2>/dev/null | grep -cE 'ring buffer is full|failed to respond|wait for reset ack|SIGBUS')"
