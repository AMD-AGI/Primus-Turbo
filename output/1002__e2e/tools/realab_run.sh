#!/bin/bash
# One card process on A0 (fa-repro, the only card): lock, KFD-empty check, fresh FlyDSL JIT cache,
# image hipBLASLt env, sclk sampler, dmesg delta classification.
# Copy of output/0930__bwd/tools/run.sh (2026-09-30) for output/1002__e2e (tools/realab.sh). Changes:
#   - R = output/1002__e2e, sampler = tools/realab_clksamp.sh (byte copy of 0930__bwd/tools/clksamp.sh, card1)
#   - ENV_FILE=<file of KEY=VALUE lines> -> docker exec --env-file (values with quotes/braces, e.g. JSON)
#   - flock -w 900 (exit 6 if the lock is not free within 15 min)
#   - the new dmesg lines are classified (skill gfx1250-card-safety section 2a) before exit 9
#   run.sh <tag> <timeout_s> <workdir> -- <shell command run inside fa-repro...>
# Env passthrough: SERIAL=1 -> AMD_SERIALIZE_KERNEL=3; EXTRA_ENV="-e K=V ..."; ENV_FILE=...
# Exit: 0 ok | 6 lock timeout | 7 KFD holders before start | 8 command rc != 0 | 9 new GPU lines in dmesg
set -u
R=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__e2e
TAG=$1; TMO=$2; WD=$3; shift 3; [ "$1" = "--" ] && shift
BLAS="export TORCH_BLAS_PREFER_HIPBLASLT=1 HIPBLASLT_TENSILE_LIBPATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250"
mkdir -p $R/runs
LOG=$R/runs/$TAG.log
SER=0; [ "${SERIAL:-0}" = 1 ] && SER=3
EFOPT=""; [ -n "${ENV_FILE:-}" ] && EFOPT="--env-file $ENV_FILE"
N0=$(timeout 20 sudo -n dmesg | wc -l)
[ -z "$(ls /sys/class/kfd/kfd/proc)" ] || { echo "$TAG: KFD holders present: $(ls /sys/class/kfd/kfd/proc)"; exit 7; }
flock -w 900 -E 66 /tmp/a0-gpu0.lock bash -c "
  [ -z \"\$(ls /sys/class/kfd/kfd/proc)\" ] || { echo rc=77 KFD-holders-after-lock >> $LOG; exit 77; }
  $R/tools/realab_clksamp.sh $R/runs/$TAG.clk $((TMO+30)) & SP=\$!
  timeout $TMO docker exec -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 -e FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_$TAG.\$\$ \
     -e AMD_SERIALIZE_KERNEL=$SER -e HIP_VISIBLE_DEVICES=0 -e PYTHONDONTWRITEBYTECODE=1 ${EXTRA_ENV:-} $EFOPT fa-repro \
     bash -c 'ulimit -c 0; $BLAS; cd $WD && exec timeout -k 20 $((TMO-30)) $*' > $LOG 2>&1
  echo rc=\$? >> $LOG
  kill \$SP 2>/dev/null; true"
[ $? = 66 ] && { echo "$TAG: /tmp/a0-gpu0.lock not free within 900 s -- not started"; exit 6; }
timeout 20 sudo -n dmesg | tail -n +$((N0+1)) | grep -v "Hardware Error\|mce:\|correctable hardware errors\|apparmor\|pcie_pl\|hogged CPU" | grep -iE "amdgpu|GCVM|MES\(|Queues reset|ring .*timeout|gpu reset|page fault|hang" > $R/runs/$TAG.dmesg.bad
echo "$TAG $(tail -1 $LOG) dmesg_bad=$(wc -l < $R/runs/$TAG.dmesg.bad) kfd=[$(ls /sys/class/kfd/kfd/proc | tr '\n' ' ')]"
if [ -s $R/runs/$TAG.dmesg.bad ]; then
  cat $R/runs/$TAG.dmesg.bad
  if grep -qiE "wait for reset ack|GPU reset begin|ring .*timeout|unrecoverable" $R/runs/$TAG.dmesg.bad; then
    echo "$TAG: dmesg CLASS=unrecoverable -- needs an AC power cycle by the user; stop ALL card work"
  elif grep -qiE "Queues reset on process|PROTECTION_FAULT|page fault" $R/runs/$TAG.dmesg.bad; then
    echo "$TAG: dmesg CLASS=process-fault -- the process died, the card probably did not; verify (KFD empty + one 4096^3 GEMM) before more card work"
  else
    echo "$TAG: dmesg CLASS=other -- read the lines above before any more card work"
  fi
  exit 9
fi
grep -q '^rc=0' $LOG || exit 8
exit 0
