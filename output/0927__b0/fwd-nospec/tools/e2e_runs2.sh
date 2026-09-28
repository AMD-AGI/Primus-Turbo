#!/bin/bash
# second batch: r13 (spec) BAAB, and a direct r13 vs r13ns run (both FlyDSL, bwd r20 both; fly=r13ns, fly13=r13)
D=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/fwd-nospec; E=$D/../e2e; BW=$E/arms/bwd_r20_0341
run() {  # tag trees-json schedule
  echo "[$(date +%T)] start $1"
  E2E_NKFIX=1 E2E_MEM_STOP=89.5 NKFIX_CHECK=1 E2E_ENV="-e E2E_FLY_TREES=$2" bash $E/run_e2e.sh train $1 "$3" 62 10 > $D/runs/e2e.$1.out 2>&1
  rc=$?; echo "[$(date +%T)] end $1 rc=$rc"; tail -2 $D/runs/e2e.$1.out
  { [ $rc = 99 ] || grep -q "!! NEW dmesg" $D/runs/e2e.$1.out; } && { echo "DMESG_STOP"; exit 99; }
  return 0
}
run r13_p2b_flyasm "{\"fly\":{\"fwd\":\"$D/arms/fwd_r13\",\"bwd\":\"$BW\"}}" "fly,asm,fly,asm;fly,asm,asm,fly"
run nsr13_p2c "{\"fly\":{\"fwd\":\"$D/arms/fwd_r13ns\",\"bwd\":\"$BW\"},\"fly13\":{\"fwd\":\"$D/arms/fwd_r13\",\"bwd\":\"$BW\"}}" "fly13,fly,fly13,fly;fly13,fly,fly,fly13"
echo E2E2_ALL_DONE
