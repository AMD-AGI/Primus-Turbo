#!/bin/bash
# A0 e2e repro: 3 attention arms in ONE training process (aiter ASM / FlyDSL old = fwd r6 + bwd r20 /
# FlyDSL new = fwd r16 (r13ns) + bwd r29), per-step interleave, 90 steps, profile every 11 steps.
# Two processes with rotated orders. Stops on the first dmesg hit (card rule: never re-probe).
#   RUNS="opc p3a p3b" bash e2e_drive.sh
set -u
R=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0928__a0_repro
E=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/e2e
TREES="{\"fly\":{\"fwd\":\"$E/arms/fwd_r16\",\"bwd\":\"$E/arms/bwd_r29_0341\"},\"new\":{\"fwd\":\"$E/arms/fwd_r16\",\"bwd\":\"$E/arms/bwd_r29_0341\"},\"old\":{\"fwd\":\"$E/arms/fwd_r6\",\"bwd\":\"$E/arms/bwd_r20_0341\"}}"
mkdir -p $R/e2e

idle() {  # KFD empty, then >= 120 s cool-down
  until [ -z "$(ls /sys/class/kfd/kfd/proc)" ]; do sleep 5; done
  sleep ${COOL:-300}
  [ -z "$(ls /sys/class/kfd/kfd/proc)" ] || { echo "KFD busy after cool-down: $(ls /sys/class/kfd/kfd/proc)"; exit 97; }
}
check() {  # $1 rc $2 out
  { [ $1 = 99 ] || grep -q "!! NEW dmesg" $2; } && { echo "DMESG_STOP"; exit 99; }
  grep -q "watchdog fired" $2 && { echo "WATCHDOG_STOP"; exit 98; }
  grep -qE "DISCARD|non-finite|nonfinite_events: \[\(" $2 && { echo "NONFINITE_STOP"; exit 96; }
  return 0
}
for r in ${RUNS:-opc p3a p3b}; do
  idle
  echo "[$(date +%T)] start $r"
  case $r in
    opc)
      E2E_FLYCACHE=/tmp/flycache_a0opc_$(date +%H%M%S) OPCHECK_ENV="-e AMD_SERIALIZE_KERNEL=3 -e E2E_FLY_TREES=$TREES" \
        bash $R/run_e2e_a0.sh opcheck fly fast > $R/e2e/opc.out 2>&1; rc=$?
      echo "[$(date +%T)] end $r rc=$rc"; tail -3 $R/e2e/opc.out; check $rc $R/e2e/opc.out
      [ $rc = 0 ] || { echo "OPCHECK_FAIL"; exit 1; } ;;
    p3a|p3b)
      [ $r = p3a ] && S="asm,old,new,asm,new,old;asm,old,new,asm,new,old" || S="new,old,asm,new,asm,old;new,old,asm,new,asm,old"
      TAG=a0_$r; echo "$S" > $R/e2e/$TAG.spec
      E2E_FLYCACHE=/tmp/flycache_${TAG}_$(date +%H%M%S) E2E_NKFIX=1 E2E_MEM_STOP=89.5 NKFIX_CHECK=1 E2E_TIMEOUT=4200 E2E_INNER=4100 \
        E2E_ENV="-e E2E_FLY_TREES=$TREES" bash $R/run_e2e_a0.sh train $TAG "$S" 90 11 > $R/e2e/e2e.$TAG.out 2>&1; rc=$?
      echo "[$(date +%T)] end $r rc=$rc"; tail -4 $R/e2e/e2e.$TAG.out; check $rc $R/e2e/e2e.$TAG.out ;;
    p4a|p4b)   # 24 layers (A0 SIGBUS/wedge at ~88-89% memory with 32L); 2x2 factorial fwd {r6,r16} x bwd {r20,r29} + asm; "f/b" token = fwd from arm f, bwd from arm b; no profiler (A0 kineto drops GPU records)
      [ $r = p4a ] && S="asm,old,new,new/old,old/new,old/new,new/old,new,old,asm" || S="old/new,new/old,new,old,asm,asm,old,new,new/old,old/new"
      TAG=a0_$r; echo "$S" > $R/e2e/$TAG.spec
      E2E_FLYCACHE=/tmp/flycache_${TAG}_$(date +%H%M%S) E2E_NKFIX=1 E2E_MEM_STOP=80 NKFIX_CHECK=1 E2E_TIMEOUT=4200 E2E_INNER=4100 \
        E2E_ENV="-e E2E_FLY_TREES=$TREES" E2E_NLAYERS=${NL:-24} bash $R/run_e2e_a0.sh train $TAG "$S" 100 0 > $R/e2e/e2e.$TAG.out 2>&1; rc=$?
      echo "[$(date +%T)] end $r rc=$rc"; tail -4 $R/e2e/e2e.$TAG.out; check $rc $R/e2e/e2e.$TAG.out ;;
  esac
done
echo E2E_DRIVE_DONE
