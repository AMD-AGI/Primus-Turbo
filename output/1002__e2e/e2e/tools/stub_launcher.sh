#!/bin/bash
# CPU dry-run stand-in for run_e2e_a0.sh (NO docker, NO GPU): lets drive.sh's control flow, guards, stop rules
# and analysis be exercised. Replays the A0 09-28 a0_p3a training log (90 steps, real tps) under the new tag and
# synthesises an attention-event file. Usage (see E2E-PLAN.md 5.0):
#   LOCKFILE=/tmp/e2e_dry.lock LAUNCHER=$K/tools/stub_launcher.sh SKIP_PREFLIGHT=1 COOL=1 IDLE_MAX=5 bash $K/drive.sh
# STUB_FAIL=<what> makes it emit one failure (expected driver exit in brackets):
#   in opcheck fast:  opcheck [1]  refuse [97: launcher pre-exec guard]  kfdbusy [97: fake KFD holder in the
#                     driver's private KFD dir]  oe [97: fake "op-evolve resume" loop]  realab [97: fake realab.sh]
#                     drift [93: baseline sha256 no longer matches]  stop [90: STOP sentinel]
#   in opcheck prod:  opcprod [1]  foreignopc [91]
#   in training p1:   nonfinite [96]  hang [98]  blas [94]  memguard [95]  dmesg [99]  foreign [91]  treerun [93]
# Fakes live under /tmp/e2e_dry_*; the fake client processes exit by themselves after 30 s.
set -u
. /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__e2e/e2e/trees.sh   # KIT, run_files_sha
SRC=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/e2e/logs/e2e.a0_p3a.log
F=${STUB_FAIL:-}
fake_client() {   # $1 = script name: a bash script that sleeps 30 s, comm = its name, cmdline carries e2e_dry
  local d=/tmp/e2e_dry_bin; mkdir -p $d
  printf '#!/bin/bash\nsleep 30\n' > $d/$1; chmod +x $d/$1
  if [ "$1" = op-evolve ]; then setsid $d/$1 resume --job e2e_dry_$$ > /dev/null 2>&1 < /dev/null &
  else setsid bash $d/$1 e2e_dry_$$ > /dev/null 2>&1 < /dev/null & fi
}
MODE=$1; shift
case $MODE in
opcheck)
  J=$KIT/runs/opcheck.$1.$2.STUB.json
  echo "[$(date +%T)] opcheck.$1.$2.STUB start (env: ${OPCHECK_ENV:-none})"
  if [ "$2" = fast ]; then
    case "$F" in
      refuse) echo "!! refusing to start opcheck.$1.$2.STUB: KFD holders 424242[]"; exit 97 ;;
      kfdbusy) mkdir -p "$KFD_DIR/424242" ;;
      oe) fake_client op-evolve ;;
      realab) fake_client realab.sh ;;
      drift) echo "0000000000000000000000000000000000000000000000000000000000000000  /e2e_dry/changed.py" >> "$E2E_TREE_BASE" ;;
      stop) touch "$STOPFILE" ;;
    esac
  fi
  dq=51.7
  { [ "$F" = opcheck ] && [ "$2" = fast ]; } || { [ "$F" = opcprod ] && [ "$2" = prod ]; } && dq='"UNCOVERED 1/2"'
  echo "{\"arm\":\"$1\",\"shape\":\"$2\",\"sqnr_chain_db\":{\"o\":51.23,\"dq\":$dq,\"dk\":51.7,\"dv\":52.32},\"sqnr_kfwd_db\":{\"o\":51.23,\"lse\":85.5,\"lse_shape\":[1,8,1024]},\"sqnr_kbwd_refolse_db\":{\"dq\":52.6,\"dk\":52.65,\"dv\":52.83},\"blas_repoints\":[],\"attn_timer\":{\"ok\":true},\"adapter_copies\":{\"copies\":0},\"timing_ms\":{\"k_bwd\":{\"med\":5.3},\"m_bwd\":{\"med\":5.4}}}" > $J
  echo "rc=0 log=stub json=$J"
  if [ "$F" = foreignopc ] && [ "$2" = prod ]; then
    echo "08:00:00 !! FOREIGN KFD 424242 python3 bench.py (no E2E_RUN_MARKER=opcheck.$1.$2.STUB: a second GPU client) -- run INVALID, stopping it"
    echo "!! opcheck.$1.$2.STUB INVALID: a second GPU client held KFD during the run"; exit 91
  fi
  exit 0 ;;
train)
  TAG=$1; SPEC=$2; STEPS=$3
  LOG=$KIT/logs/e2e.$TAG.log; rm -f $LOG.watchdog $LOG.memguard $LOG.foreign   # as the real launcher does
  { run_files_sha; echo "0000000000000000000000000000000000000000000000000000000000000000  $KIT/configs/run.$TAG.yaml"; } > $KIT/logs/tree.$TAG.sha256
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
  case "$F" in
    nonfinite) echo "08:00:00 WATCHDOG fired: nonfinite: step: 4 loss: 10.2 grad_norm: inf" > $LOG.watchdog ;;
    hang) echo "08:00:00 WATCHDOG fired: hang: step 9 running 300s > 240s" > $LOG.watchdog ;;
    blas) echo "08:00:00 WATCHDOG fired: blas: BLAS-REPOINT by load fly/fwd" > $LOG.watchdog ;;
    memguard) echo "!! memory 89.7% > 89.5% -- SIGTERM torchrun pid 1" > $LOG.memguard ;;
    foreign) echo "08:00:00 !! FOREIGN KFD 424242 python3 bench.py (no E2E_RUN_MARKER=$TAG: a second GPU client) -- run INVALID, stopping it" > $LOG.foreign ;;
  esac
  V=yes; [ -n "$F" ] && V="INVALID (stub $F )"
  printf 'tag: %s\nspec: %s\nsteps: %s\nrc: 0\nnkfix_events:  []\nforeign_kfd: %s\ntree_changed_during_run: %s\nvalid: %s\nbwd_stream_mode: [flydsl_bwd bwd_s6_0341] DQ_SIDE_STREAM=1 DQ_SIDE_RECORD=0\n' \
    "$TAG" "$SPEC" "$STEPS" "$(head -1 $LOG.foreign 2>/dev/null)" "$([ "$F" = treerun ] && echo /e2e_dry/changed.py || echo no)" "$V" > $KIT/runs/$TAG.post.txt
  echo "rc=0 tag=$TAG log=$LOG steps_logged=$STEPS (STUB)"
  [ "$F" = treerun ] && echo "!! TREE CHANGED DURING RUN $TAG (run INVALID): /e2e_dry/changed.py"
  if [ "$F" = dmesg ]; then echo "!! NEW dmesg FAULT lines on a live card -- stop all card work:"; echo "[1.0] amdgpu: MES failed to respond"; exit 99; fi
  [ "$F" = foreign ] && exit 91
  exit 0 ;;
esac
