#!/bin/bash
# CPU dry-run stand-in for run_e2e_a0.sh (NO docker, NO GPU): lets drive.sh's control flow, stop rules
# and analysis be exercised. Replays the A0 09-28 a0_p3a training log (90 steps, real tps) under the
# new tag and synthesises an attention-event file. STUB_FAIL=nonfinite|hang|blas|memguard|dmesg|opcheck
# makes it emit that failure. Usage (see E2E-PLAN.md 5.0):
#   LOCKFILE=/tmp/e2e_dry.lock LAUNCHER=$K/tools/stub_launcher.sh SKIP_PREFLIGHT=1 COOL=1 IDLE_MAX=5 bash $K/drive.sh
set -u
KIT=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__e2e/e2e
SRC=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/e2e/logs/e2e.a0_p3a.log
MODE=$1; shift
case $MODE in
opcheck)
  J=$KIT/runs/opcheck.$1.$2.STUB.json
  if [ "${STUB_FAIL:-}" = opcheck ]; then dq='"UNCOVERED 1/2"'; else dq=51.7; fi
  echo "{\"arm\":\"$1\",\"shape\":\"$2\",\"sqnr_chain_db\":{\"o\":51.23,\"dq\":$dq,\"dk\":51.7,\"dv\":52.32},\"sqnr_kbwd_refolse_db\":{\"dq\":52.6,\"dk\":52.65,\"dv\":52.83},\"blas_repoints\":[],\"attn_timer\":{\"ok\":true},\"adapter_copies\":{\"copies\":0}}" > $J
  echo "rc=0 log=stub json=$J"; exit 0 ;;
train)
  TAG=$1; SPEC=$2; STEPS=$3
  LOG=$KIT/logs/e2e.$TAG.log; rm -f $LOG.watchdog $LOG.memguard   # as the real launcher does
  head -c 2000000 $SRC | awk -v n=$STEPS '/step: *[0-9]+ +loss/ { match($0, /step: *[0-9]+/); s = substr($0, RSTART + 5) + 0; if (s > n) next } { print }' > $LOG
  python3 - "$KIT/logs/attn_ev.$TAG.jsonl" "$SPEC" "$STEPS" <<'EOF'
import json, random, sys
sys.path.insert(0, "/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__e2e/e2e/tools")
from steady_arms3 import arm
base = {"asm": (1.22, 7.35), "fly": (1.52, 7.05), "flyr29": (1.52, 8.75)}
random.seed(1)
with open(sys.argv[1], "w") as f:
    for s in range(1, int(sys.argv[3]) + 1):
        a = arm(sys.argv[2], s); b = base.get(a, (1.5, 9.0))
        fw = [b[0] + random.gauss(0, .02) for _ in range(32)]; bw = [b[1] + random.gauss(0, .05) for _ in range(32)]
        f.write(json.dumps({"step": s, "arm_fwd": a, "arm_bwd": a, "fwd_ms": sum(fw), "bwd_ms": sum(bw),
                            "n_fwd": 32, "n_bwd": 32, "fwd": fw, "bwd": bw}) + "\n")
EOF
  cp /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/e2e/logs/clk.a0_p3a.csv $KIT/logs/clk.$TAG.csv
  case "${STUB_FAIL:-}" in
    nonfinite) echo "08:00:00 WATCHDOG fired: nonfinite: step: 4 loss: 10.2 grad_norm: inf" > $LOG.watchdog ;;
    hang) echo "08:00:00 WATCHDOG fired: hang: step 9 running 300s > 240s" > $LOG.watchdog ;;
    blas) echo "08:00:00 WATCHDOG fired: blas: BLAS-REPOINT by load fly/fwd" > $LOG.watchdog ;;
    memguard) echo "!! memory 89.7% > 89.5% -- SIGTERM torchrun pid 1" > $LOG.memguard ;;
  esac
  printf 'tag: %s\nspec: %s\nsteps: %s\nrc: 0\nnkfix_events:  []\nbwd_stream_mode: [flydsl_bwd bwd_s6_0341] DQ_SIDE_STREAM=1 DQ_SIDE_RECORD=0\n' "$TAG" "$SPEC" "$STEPS" > $KIT/runs/$TAG.post.txt
  echo "rc=0 tag=$TAG log=$LOG steps_logged=$STEPS (STUB)"
  if [ "${STUB_FAIL:-}" = dmesg ]; then echo "!! NEW dmesg FAULT lines on a live card -- stop all card work:"; echo "[1.0] amdgpu: MES failed to respond"; exit 99; fi
  exit 0 ;;
esac
