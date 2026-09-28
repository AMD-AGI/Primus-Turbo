#!/bin/bash
# e2e P2 runs on GPU0 (run_e2e.sh holds the lock + dmesg check). The "fly" arm's fwd tree is overridden per run
# through E2E_FLY_TREES (bwd stays e2e/arms/bwd_r20_0341). 32 layers, 62 steps, profile every 10 steps.
D=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/fwd-nospec; E=$D/../e2e
BW=$E/arms/bwd_r20_0341
run() {  # tag fwdtree schedule
  echo "[$(date +%T)] start $1"
  E2E_NKFIX=1 E2E_MEM_STOP=89.5 NKFIX_CHECK=1 E2E_ENV="-e E2E_FLY_TREES={\"fly\":{\"fwd\":\"$2\",\"bwd\":\"$BW\"}}" \
    bash $E/run_e2e.sh train $1 "$3" 62 10 > $D/runs/e2e.$1.out 2>&1
  rc=$?; echo "[$(date +%T)] end $1 rc=$rc"; tail -4 $D/runs/e2e.$1.out
  [ $rc = 99 ] && { echo "DMESG_STOP"; exit 99; }
  grep -q "!! NEW dmesg" $D/runs/e2e.$1.out && { echo "DMESG_STOP"; exit 99; }
  return 0
}
for r in ${RUNS:-ns_p2a ns_p2b r13_p2a}; do
  case $r in
    ns_p2a)  run ns_p2a_asmfly  $D/arms/fwd_r13ns "asm,fly,asm,fly;asm,fly,fly,asm" ;;
    ns_p2b)  run ns_p2b_flyasm  $D/arms/fwd_r13ns "fly,asm,fly,asm;fly,asm,asm,fly" ;;
    r13_p2a) run r13_p2a_asmfly $D/arms/fwd_r13   "asm,fly,asm,fly;asm,fly,fly,asm" ;;
  esac
done
echo E2E_ALL_DONE
