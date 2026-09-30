#!/bin/bash
# One card process on A0 (fa-repro, the only card): lock, KFD-empty check, fresh FlyDSL JIT cache,
# image hipBLASLt env, sclk sampler, dmesg delta classification.
#   run.sh <tag> <timeout_s> <workdir> -- <shell command run inside fa-repro...>
# Env passthrough: SERIAL=1 -> AMD_SERIALIZE_KERNEL=3; EXTRA_ENV="-e K=V ..."
set -u
R=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0930__bwd
TAG=$1; TMO=$2; WD=$3; shift 3; [ "$1" = "--" ] && shift
BLAS="export TORCH_BLAS_PREFER_HIPBLASLT=1 HIPBLASLT_TENSILE_LIBPATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250"
LOG=$R/runs/$TAG.log
SER=0; [ "${SERIAL:-0}" = 1 ] && SER=3
N0=$(timeout 20 sudo -n dmesg | wc -l)
[ -z "$(ls /sys/class/kfd/kfd/proc)" ] || { echo "$TAG: KFD holders present: $(ls /sys/class/kfd/kfd/proc)"; exit 7; }
flock /tmp/a0-gpu0.lock bash -c "
  $R/tools/clksamp.sh $R/runs/$TAG.clk $((TMO+30)) & SP=\$!
  timeout $TMO docker exec -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 -e FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_$TAG.\$\$ \
     -e AMD_SERIALIZE_KERNEL=$SER -e HIP_VISIBLE_DEVICES=0 -e PYTHONDONTWRITEBYTECODE=1 ${EXTRA_ENV:-} fa-repro \
     bash -c 'ulimit -c 0; $BLAS; cd $WD && exec timeout -k 20 $((TMO-30)) $*' > $LOG 2>&1
  echo rc=\$? >> $LOG
  kill \$SP 2>/dev/null"
timeout 20 sudo -n dmesg | tail -n +$((N0+1)) | grep -v "Hardware Error\|mce:\|correctable hardware errors\|apparmor\|pcie_pl\|hogged CPU" | grep -iE "amdgpu|GCVM|MES\(|Queues reset|ring .*timeout|gpu reset|page fault|hang" > $R/runs/$TAG.dmesg.bad
echo "$TAG $(tail -1 $LOG) dmesg_bad=$(wc -l < $R/runs/$TAG.dmesg.bad) kfd=[$(ls /sys/class/kfd/kfd/proc | tr '\n' ' ')]"
[ -s $R/runs/$TAG.dmesg.bad ] && { cat $R/runs/$TAG.dmesg.bad; exit 9; }
grep -q '^rc=0' $LOG || exit 8
exit 0
