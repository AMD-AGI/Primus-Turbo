#!/bin/bash
# One card process on GPU0 / fa-g0 under the lock, with the host clock sampler and a dmesg check.
#   run_op.sh <tag> <timeout_s> <env assignments...> -- <python args...>
set -u
P=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/profile
TAG=$1; TMO=$2; shift 2
EF=$P/runs/$TAG.env; : > $EF; while [ "$1" != "--" ]; do echo "$1" >> $EF; shift; done; shift
BLAS="export TORCH_BLAS_PREFER_HIPBLASLT=1 HIPBLASLT_TENSILE_LIBPATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250"
mkdir -p $P/runs
mark=$(timeout 20 sudo -n dmesg | grep -v 0002:04:00.0 | tail -1 | sed 's/^\[\s*\([0-9.]*\)\].*/\1/')
echo "[$(date +%T)] $TAG waiting for lock"
flock /tmp/b0-gpu0.lock bash -c "
  $P/tools/clksamp2.sh $P/runs/$TAG.clk $((TMO+30)) & SP=\$!
  timeout $TMO docker exec -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 -e FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_prof \
     -e TRITON_CACHE_DIR=/tmp/triton_cache_e2e --env-file $EF fa-g0 bash -c 'ulimit -c 0; $BLAS; cd $P && exec timeout -k 20 $((TMO-30)) ${RUN_BIN:-/opt/venv/bin/python3} $*' > $P/runs/$TAG.log 2>&1
  echo rc=\$? >> $P/runs/$TAG.log
  kill \$SP"
echo "[$(date +%T)] $TAG $(tail -1 $P/runs/$TAG.log)"
NEW=$(timeout 20 sudo -n dmesg | grep -v 0002:04:00.0 | awk -v m="$mark" '{t=$0; sub(/^\[ */,"",t); sub(/\].*/,"",t); if (t+0>m+0) print}' | grep -iE "amdgpu|reset|MES|GCVM|fault|timeout|Queues" )
[ -n "$NEW" ] && { echo "!! NEW dmesg:"; echo "$NEW"; exit 99; }
timeout 20 docker top fa-g0 -eo pid,args | grep -v "sleep infinity" | tail -n +2
echo "dmesg clean"
