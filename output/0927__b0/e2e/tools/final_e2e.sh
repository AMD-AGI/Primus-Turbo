#!/bin/bash
# Final e2e: FlyDSL arm = fwd r16 (== r13ns) + bwd r29 (u2n, re-pinned 0.3.4.1) vs aiter ASM, P2 same-process,
# 32 layers, 62 steps, profile every 10 steps, fresh FlyDSL JIT cache dir per run (JIT-cache hazard h46/h72).
E=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/e2e; . $E/tools/final_trees.sh
run() {  # tag schedule
  echo "[$(date +%T)] start $1"
  E2E_FLYCACHE=/tmp/flycache_$1_$(date +%H%M%S) E2E_NKFIX=1 E2E_MEM_STOP=89.5 NKFIX_CHECK=1 \
    E2E_ENV="-e E2E_FLY_TREES=$FLY_TREES_JSON" bash $E/run_e2e.sh train $1 "$2" 62 10 > $E/runs_final/e2e.$1.out 2>&1
  rc=$?; echo "[$(date +%T)] end $1 rc=$rc"; tail -4 $E/runs_final/e2e.$1.out
  { [ $rc = 99 ] || grep -q "!! NEW dmesg" $E/runs_final/e2e.$1.out; } && { echo "DMESG_STOP"; exit 99; }
  return 0
}
for r in ${RUNS:-fin_p2a fin_p2b}; do
  case $r in
    fin_p2a) run fin_p2a_asmfly "asm,fly,asm,fly;asm,fly,fly,asm" ;;
    fin_p2b) run fin_p2b_flyasm "fly,asm,fly,asm;fly,asm,asm,fly" ;;
  esac
done
echo FINAL_E2E_DONE
