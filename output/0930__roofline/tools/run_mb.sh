#!/bin/bash
# One card process on A0 (fa-repro, the only card): lock, KFD-empty check, sclk/power sampler, dmesg delta.
#   run_mb.sh <tag> <timeout_s> <workdir> -- <command...>
set -u
R=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0930__roofline
TAG=$1; TMO=$2; WD=$3; shift 3; [ "$1" = "--" ] && shift
LOG=$R/runs/$TAG.log
N0=$(timeout 20 sudo -n dmesg | wc -l)
[ -z "$(ls /sys/class/kfd/kfd/proc)" ] || { echo "$TAG: KFD holders present: $(ls /sys/class/kfd/kfd/proc)"; exit 7; }
flock /tmp/a0-gpu0.lock bash -c "
  $R/tools/clksamp.sh $R/runs/$TAG.clk $((TMO+30)) & SP=\$!
  timeout $TMO docker exec -e AMD_SERIALIZE_KERNEL=\${AMD_SERIALIZE_KERNEL:-0} -e HIP_VISIBLE_DEVICES=0 fa-repro bash -c 'ulimit -c 0; cd $WD && exec timeout -k 20 $((TMO-30)) $*' > $LOG 2>&1
  echo rc=\$? >> $LOG
  kill \$SP 2>/dev/null"
timeout 20 sudo -n dmesg | tail -n +$((N0+1)) | grep -v "Hardware Error\|mce:\|correctable hardware errors" | grep -iE "amdgpu|GCVM|MES\(|Queues reset|ring .*timeout|gpu reset|page fault|hang" > $R/runs/$TAG.dmesg.bad
echo "$TAG $(tail -1 $LOG) dmesg_bad=$(wc -l < $R/runs/$TAG.dmesg.bad) kfd=[$(ls /sys/class/kfd/kfd/proc | tr '\n' ' ')]"
[ -s $R/runs/$TAG.dmesg.bad ] && { cat $R/runs/$TAG.dmesg.bad; exit 9; }
grep -q '^rc=0' $LOG || exit 8
exit 0
