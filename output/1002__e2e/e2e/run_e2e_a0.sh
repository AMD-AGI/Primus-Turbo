#!/bin/bash
# A0 e2e launcher, 2026-10-02 (output/1002__e2e/e2e). Adapted copy of output/0928__a0_repro/run_e2e_a0.sh
# (itself the A0 copy of output/0927__b0/e2e/run_e2e.sh). Llama-3.1-8B BF16, MBS=GBS=4, seq 8192, 1 GPU.
#
#   run_e2e_a0.sh train   <tag> <E2E_ATTN spec> [steps=30] [profile_freq=10|0=off]
#   run_e2e_a0.sh opcheck <arm> <shape: fast|prod> [extra opcheck.py args]
#
# Normally called by drive.sh, which holds /tmp/a0-gpu0.lock for the whole sequence and exports
# E2E_LOCK_HELD=1 (then this script does not take the lock itself). Standalone it takes the lock.
#
# Differences from the 09-28 A0 copy (each one a 09-28 lesson or a 10-02 review item; E2E-PLAN.md section 3):
#   * adapter = this kit's attn_backends (BLAS guard + per-step CUDA-event attention timing,
#     E2E_ATTN_EVENTS) and FlyDSL trees only from E2E_FLY_TREES (no silent default)
#   * E2E_EXPECT_BLAS_LIB = the IMAGE hipBLASLt library, assigned before python starts (bash -c, not -lc)
#   * watchdog also stops the run (ONE SIGTERM to torchrun by PID) on the first non-finite
#     loss / grad_norm, the first "[nkfix] NON-FINITE", or a "!! BLAS-REPOINT" line
#   * dmesg is classified: FAULT (stop) vs INFO (CPU MCE, GPU RAS *correctable* pcie_pl reports, which
#     A0 logs during normal work, e.g. 2026-10-02 07:58:50) -- the 09-28 filter would have stopped on INFO
#   * evidence: worker /proc/<pid>/environ (E2E_FLY_TREES, FLYDSL_RUNTIME_CACHE_DIR, BLAS env), sha256 of
#     every file that runs, a key: value post-run summary runs/<tag>.post.txt, the [e2e_attn]/[nkfix]/
#     [flydsl_bwd]/BLAS-REPOINT lines of the rank-0 debug.log (outside the repo) -> runs/<tag>.dbg.txt
#   * right before the docker exec (both modes): KFD empty, no op-evolve run/resume loop and no realab driver
#     (neither takes the lock), the three trees at their pinned md5, and every file that runs unchanged since
#     the driver started (E2E_TREE_BASE) -- else refuse (97 other client / 93 drift)
#   * during the run (both modes, every 5 s): every KFD holder must carry E2E_RUN_MARKER=<this tag> in its
#     environ (docker exec -e; torchrun and its workers inherit it). Any other holder is a second GPU client:
#     "!! FOREIGN KFD <pid> <cmd>", ONE SIGTERM to this run's torchrun / opcheck python, run INVALID, exit 91
#   * after a training run the files that ran are hashed again: a change during the run -> INVALID, the
#     driver stops (93)
# Signals: only SIGTERM, by PID, at most once per process (watchdog, memguard, foreign-KFD watch). The one
# SIGKILL anywhere is `timeout -k 20` as the last backstop of a hard timeout (opcheck 600 s, training 2600 s)
# when the process still runs 20 s after timeout's own SIGTERM.
set -u
. /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__e2e/e2e/trees.sh   # KIT, B0E, trees, guards
KFD_DIR=$KFD_PROC                     # the real card, always (KFD_DIR is a dry-run override of drive.sh only)
PRIMUS=/home/lihuzhan/code/2026_0828__primus/Primus
FLY=/home/lihuzhan/.local/flydsl0341
AITER=/home/lihuzhan/code/aiter-src
SHIM=$KIT/attn_backends/shim
LOCK=/tmp/a0-gpu0.lock
CT=fa-repro
CARD=/sys/class/drm/card1/device
# hipBLASLt: the IMAGE library, never the host ~/.local/hipblaslt-gfx1250 (A0 09-28: NaN + wedge when it was live)
BLAS_LIB=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250
BLAS_EXPORT="export TORCH_BLAS_PREFER_HIPBLASLT=1 HIPBLASLT_TENSILE_LIBPATH=$BLAS_LIB"
PP="$SHIM:$KIT/attn_backends:$FLY:$AITER"
mkdir -p "$KIT/logs" "$KIT/runs" "$KIT/configs" "$KIT/traces"

LOCKCMD="flock $LOCK"
# under drive.sh the lock is the driver's fd 9: drop our inherited copy so no long-lived child can keep it
if [ "${E2E_LOCK_HELD:-0}" = 1 ]; then LOCKCMD=""; exec 9>&-; fi

dmesg_mark() { timeout 20 sudo -n dmesg 2>/dev/null | tail -1 | sed 's/^\[\s*\([0-9.]*\)\].*/\1/'; }
dmesg_new() {   # $1 = mark; every line newer than the mark
  timeout 20 sudo -n dmesg 2>/dev/null | awk -v m="$1" '{ t=$0; sub(/^\[ */,"",t); sub(/\].*/,"",t); if (t+0 > m+0) print }'
}
# INFO: CPU MCE / [Hardware Error] register dumps, GPU RAS correctable-error counts, MES ring-full
# backpressure, apparmor/audit noise, 'workqueue: svm_range_restore_work [amdgpu] hogged CPU' (seen 09-30 a06). FAULT: anything else that names the GPU or a known fault signature.
DM_INFO='mce:|\[Hardware Error\]|[0-9]+ (new )?correctable hardware errors detected|ring buffer is full|audit: |apparmor|workqueue: .* hogged CPU'
DM_FAULT='amdgpu|gpu reset|reset ack|ring .* timeout|MES|GCVM|page fault|Queues reset|SIGBUS|general protection|uncorrectable|kfd|drm'
dmesg_fault() { dmesg_new "$1" | grep -vE "$DM_INFO" | grep -iE "$DM_FAULT" || true; }
dmesg_info()  { dmesg_new "$1" | grep -E "correctable hardware errors detected|mce: \[Hardware Error\]: Machine check" | grep -v uncorrectable || true; }
leftovers() {   # our own container only
  timeout 20 docker top $CT -eo pid,etime,args 2>/dev/null | grep -E "torchrun|primus/cli/main.py|opcheck.py" | grep -v grep || true
}

# Immediately before the docker exec: no other GPU client, the pinned trees, nothing changed since the driver
# started. (drive.sh ran the same checks after its cool-down; this closes the gap to the exec itself.)
pre_exec_guard() {
  local x
  x=$(kfd_holders); [ -n "$x" ] && { echo "!! refusing to start $TAG: KFD holders $(pid_desc $x)"; exit 97; }
  x=$(oe_loops); [ -n "$x" ] && { echo "!! refusing to start $TAG: op-evolve loop running $(pid_desc $x)"; exit 97; }
  x=$(realab_drivers); [ -n "$x" ] && { echo "!! refusing to start $TAG: realab driver running $(pid_desc $x)"; exit 97; }
  if [ "${PREFLIGHT_ALLOW_SRC_DIFF:-0}" != 1 ]; then
    x=$(trees_check); [ -n "$x" ] && { echo "!! refusing to start $TAG: tree drift: $x"; exit 93; }
  fi
  if [ -n "${E2E_TREE_BASE:-}" ]; then
    x=$(run_files_sha | diff "$E2E_TREE_BASE" - | grep '^[<>]' | awk '{print $NF}' | sort -u | head -6 | tr '\n' ' ')
    [ -n "$x" ] && { echo "!! refusing to start $TAG: run files changed since the driver started: $x"; exit 93; }
  fi
  echo "[$(date +%T)] pre-exec guard ok: KFD empty, no op-evolve/realab, trees pinned${E2E_TREE_BASE:+, run files = $E2E_TREE_BASE}"
}

# Foreign-KFD watch (background, both modes). $1 = file for the "!! FOREIGN KFD" lines, $2 = docker-top args
# pattern of the process to stop (torchrun / opcheck.py). Every KFD holder must be this run's process; the first
# foreign one is reported (each pid once) and then each of this run's matching processes gets ONE SIGTERM --
# also one that only starts after the detection (a foreign holder can appear before our exec).
kfd_watch() {
  local p o c t own=" " sent=" "
  while :; do
    for p in $(kfd_holders); do
      case "$own" in *" $p "*) continue ;; esac
      o=$(kfd_owner "$p" "$TAG" $CT)
      case $o in
        ours|unverified) own="$own$p " ;;
        foreign)
          own="$own$p "
          c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-200)
          echo "$(date +%T) !! FOREIGN KFD $p ${c:-?} (no E2E_RUN_MARKER=$TAG: a second GPU client) -- run INVALID, stopping it" >> "$1" ;;
      esac
    done
    if [ -s "$1" ]; then
      for t in $(timeout 20 docker top $CT -eo pid,args 2>/dev/null | grep -E "$2" | grep -v -E "timeout |bash " | awk '{print $1}'); do
        case "$sent" in *" $t "*) continue ;; esac
        [ "$(kfd_owner "$t" "$TAG" $CT)" = ours ] || continue
        echo "$(date +%T) SIGTERM pid $t (this run's; foreign KFD holder present)" >> "$1"
        sudo -n kill -TERM "$t" 2>/dev/null; sent="$sent$t "
      done
    fi
    sleep 5
  done
}

MODE=${1:?mode: train|opcheck}; shift
case "$MODE" in
opcheck)
  ARM=${1:?arm}; SHAPE=${2:?shape}; shift 2
  TAG=opcheck.$ARM.$SHAPE.$(date +%m%d_%H%M%S)
  OUT=$KIT/runs; FOREIGN=$OUT/$TAG.foreign; rm -f "$FOREIGN"
  OTMO=${E2E_OPC_TIMEOUT:-600}        # B0 09-28: fast 9 s, prod 12 s of python wall (+ imports / JIT)
  echo "[$(date +%T)] $TAG start (lock: ${LOCKCMD:-held by driver}; env: ${OPCHECK_ENV:-none}; timeout ${OTMO}s)"
  MARK=$(dmesg_mark)
  kfd_watch "$FOREIGN" "opcheck.py" 2>/dev/null & KWPID=$!
  trap 'kill $KWPID 2>/dev/null' EXIT
  pre_exec_guard
  $LOCKCMD timeout $((OTMO + 60)) docker exec \
    -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 -e FLYDSL_RUNTIME_CACHE_DIR=${E2E_FLYCACHE:?fresh dir per process} \
    -e TRITON_CACHE_DIR=/tmp/triton_cache_e2e -e E2E_EXPECT_BLAS_LIB=$BLAS_LIB -e E2E_RUN_MARKER=$TAG \
    -e E2E_FLY_TREES="${E2E_FLY_TREES_JSON:-}" \
    -e E2E_ATTN_EVENTS=$OUT/$TAG.attn_ev.jsonl ${OPCHECK_ENV:-} $CT bash -c "ulimit -c 0; $BLAS_EXPORT; \
      export PYTHONPATH=$PP; cd $KIT/attn_backends && exec timeout --foreground -k 20 $OTMO \
      /opt/venv/bin/python3 opcheck.py --arm $ARM --shape $SHAPE --json $OUT/$TAG.json $*" \
    > "$OUT/$TAG.log" 2>&1
  RC=$?
  kill $KWPID 2>/dev/null
  echo "rc=$RC log=$OUT/$TAG.log json=$OUT/$TAG.json"
  INFO=$(dmesg_info "$MARK"); [ -n "$INFO" ] && { echo "dmesg INFO (not a stop):"; echo "$INFO" | head -5; }
  NEW=$(dmesg_fault "$MARK")
  if [ -n "$NEW" ]; then echo "!! NEW dmesg FAULT lines -- stop all card work:"; echo "$NEW" | head -40; exit 99; fi
  L=$(leftovers); [ -n "$L" ] && echo "!! leftover processes in $CT (kill by PID only): $L"
  if [ -s "$FOREIGN" ]; then cat "$FOREIGN"; echo "!! $TAG INVALID: a second GPU client held KFD during the run"; exit 91; fi
  exit $RC ;;
train)
  TAG=${1:?tag}; ATTN=${2:?E2E_ATTN}; STEPS=${3:-30}; PFREQ=${4:-10}
  CFG=$KIT/configs/run.$TAG.yaml
  PROFILE=true; [ "$PFREQ" = 0 ] && { PROFILE=false; PFREQ=1000; }
  NL_SED=""; [ -n "${E2E_NLAYERS:-}" ] && NL_SED="/^ *flavor: 8B_flex\$/a\\        n_layers: ${E2E_NLAYERS}"
  sed -e "s/@STEPS@/$STEPS/" -e "s/@PFREQ@/$PFREQ/g" -e "s/@PROFILE@/$PROFILE/" ${NL_SED:+-e "$NL_SED"} \
    "$B0E/configs/l8b_e2e.template.yaml" > "$CFG"
  RUNDIR=/home/lihuzhan/_dbg_l8b/$TAG; mkdir -p "$RUNDIR"
  LOG=$KIT/logs/e2e.$TAG.log; CLK=$KIT/logs/clk.$TAG.csv; EVF=$KIT/logs/attn_ev.$TAG.jsonl
  POST=$KIT/runs/$TAG.post.txt; WDF=$LOG.watchdog; FOREIGN=$LOG.foreign; TREEF=$KIT/logs/tree.$TAG.sha256
  rm -f "$EVF" "$WDF" "$LOG.memguard" "$POST" "$FOREIGN"
  L=$(leftovers); [ -n "$L" ] && { echo "!! refusing to start: leftovers in $CT: $L"; exit 98; }
  KFD=$(kfd_holders); [ -n "$KFD" ] && { echo "!! refusing to start: KFD holders: $KFD"; exit 97; }
  # snapshot of every file that runs (the arm trees can move under us); hashed again after the run
  { run_files_sha; sha256sum "$CFG"; } > "$TREEF"
  echo "t,busy_pct,sclk_mhz,fclk_mhz,power_uw,temp_mc,kfd" > "$CLK"
  ( H=$(ls -d $CARD/hwmon/hwmon* 2>/dev/null | head -1)
    while :; do
      printf '%s,%s,%s,%s,%s,%s,%s\n' "$(date +%s)" "$(timeout 5 cat $CARD/gpu_busy_percent 2>/dev/null)" \
        "$(grep '\*' $CARD/pp_dpm_sclk 2>/dev/null | grep -oE '[0-9]+Mhz' | tr -d Mhz)" \
        "$(grep '\*' $CARD/pp_dpm_fclk 2>/dev/null | grep -oE '[0-9]+Mhz' | tr -d Mhz)" \
        "$(cat $H/power1_average 2>/dev/null || cat $H/power1_input 2>/dev/null)" \
        "$(cat $H/temp2_input 2>/dev/null || cat $H/temp1_input 2>/dev/null)" \
        "$(kfd_holders | tr ' ' ';')"
      sleep 5
    done ) >> "$CLK" 2>/dev/null &
  CLKPID=$!
  kfd_watch "$FOREIGN" "torchrun.*primus|pt_elastic" 2>/dev/null &
  KWPID=$!
  # Memory guard: peak reserved above E2E_MEM_STOP (89.5, B0 final recipe) -> ONE SIGTERM to torchrun.
  ( STOP=${E2E_MEM_STOP:-89.5}; sent=0
    while [ $sent = 0 ]; do
      sleep 10
      P=$(sed 's/\x1b\[[0-9;]*m//g' "$LOG" 2>/dev/null | grep -oE 'memory: *[0-9.]+GiB\([0-9.]+%\)' | tail -1 | grep -oE '[0-9.]+%' | tr -d %)
      if [ -n "$P" ] && awk -v p="$P" -v s="$STOP" 'BEGIN{exit !(p>s)}'; then
        for pid in $(timeout 20 docker top $CT -eo pid,args 2>/dev/null | grep -E "torchrun.*primus|pt_elastic" | grep -v grep | awk '{print $1}'); do
          echo "!! memory ${P}% > ${STOP}% -- SIGTERM torchrun pid $pid" | tee -a "$LOG.memguard"
          sudo -n kill -TERM "$pid" 2>/dev/null
        done
        sent=1
      fi
    done ) &
  MEMPID=$!
  # Watchdog. Fires ONCE: evidence (docker top, busy/sclk, dmesg tail, one gdb bt of the worker), then ONE
  # SIGTERM to this run's torchrun by host PID. Reasons:
  #   hang:      no first attention fwd E2E_WD_PRE s (900) after "Training starts"; step 1 not printed
  #              E2E_WD_FIRST s (360) after the first attention fwd; a step running > max(3 x previous, 240 s)
  #   nonfinite: first non-finite loss / grad_norm in the main log, or "[nkfix] NON-FINITE" in the rank-0 log
  #              (A0 09-28 a0_p3b ran 86 NaN steps and the next startup wedged; stop at the first sign)
  #   blas:      "!! BLAS-REPOINT" in the rank-0 log (a tree touched the hipBLASLt env mid-run)
  DBG=/home/lihuzhan/_dbg_l8b/output/amd/root/$TAG/logs/pre_trainer/rank-0/debug.log
  [ -e "$(dirname "$DBG")" ] && mv "/home/lihuzhan/_dbg_l8b/output/amd/root/$TAG" "/home/lihuzhan/_dbg_l8b/output/amd/root/$TAG.old.$(date +%H%M%S)"
  ( WD_PRE=${E2E_WD_PRE:-900}; WD_FIRST=${E2E_WD_FIRST:-360}; WD_FLOOR=${E2E_WD_FLOOR:-240}
    t_train=""; t_attn=""; last_n=0; t_last=""; prev_d=""; envdone=0
    ts() { date -d "$(echo "$1" | grep -oE '^\[[0-9]{8} [0-9:]{8}\]' | tr -d '[]')" +%s 2>/dev/null; }
    fire() {
      F=$KIT/logs/hang.$TAG
      echo "$(date +%T) WATCHDOG fired: $1" | tee -a "$F.actions.txt" >> "$WDF"
      timeout 20 docker top $CT -eo pid,ppid,stat,etime,time,args > "$F.dockertop.txt" 2>&1
      for i in 1 2 3; do
        echo "$(date +%T) busy=$(timeout 5 cat $CARD/gpu_busy_percent) sclk=$(grep '\*' $CARD/pp_dpm_sclk)" >> "$F.actions.txt"; sleep 1
      done
      timeout 20 sudo -n dmesg 2>/dev/null | tail -80 > "$F.dmesg.txt"
      case "$1" in hang*)
        W=$(grep -E "python.*primus/cli/main.py" "$F.dockertop.txt" | grep -v -E "torchrun|timeout|bash" | awk '{print $1}' | tail -1)
        [ -n "$W" ] && timeout 90 sudo -n gdb -p "$W" -batch -ex "thread apply 1 bt" > "$F.gdb.txt" 2>&1 ;;
      esac
      for pid in $(grep -E "torchrun.*primus|pt_elastic" "$F.dockertop.txt" | grep -v -E "timeout|bash " | awk '{print $1}'); do
        echo "$(date +%T) SIGTERM torchrun pid $pid" | tee -a "$F.actions.txt" >> "$WDF"
        sudo -n kill -TERM "$pid" 2>/dev/null
      done
    }
    while :; do
      sleep 10; now=$(date +%s)
      CL=$(sed 's/\x1b\[[0-9;]*m//g' "$LOG" 2>/dev/null)
      [ -z "$t_train" ] && echo "$CL" | grep -q "Training starts at step" && t_train=$now
      [ -z "$t_attn" ] && grep -q "\[e2e_attn\] BLAS" "$DBG" 2>/dev/null && t_attn=$now
      if [ $envdone = 0 ] && [ -n "$t_train" ]; then     # evidence: what the worker really got
        W=$(timeout 20 docker top $CT -eo pid,args 2>/dev/null | grep -E "python.*primus/cli/main.py" | grep -v -E "torchrun|timeout|bash" | awk '{print $1}' | tail -1)
        if [ -n "$W" ]; then
          sudo -n cat /proc/$W/environ 2>/dev/null | tr '\0' '\n' | grep -E '^(E2E_|FLYDSL_|FLY_BWD_|HIPBLASLT_|TORCH_BLAS|NKFIX_|PYTHONPATH=|AMD_SERIALIZE|GPU_MAX_HW_QUEUES|CUDA_DEVICE_MAX_CONNECTIONS|HSA_|HIP_)' | sort > "$KIT/logs/env.$TAG.txt"
          envdone=1
        fi
      fi
      SL=$(echo "$CL" | grep -E "step: *[0-9]+ +loss" | tail -2)
      n=$(echo "$SL" | tail -1 | grep -oE "step: *[0-9]+" | grep -oE "[0-9]+"); n=${n:-0}
      if [ "$n" -gt "$last_n" ]; then
        t_last=$(ts "$(echo "$SL" | tail -1)")
        if [ "$(echo "$SL" | wc -l)" -ge 2 ] && [ "$n" -ge 2 ]; then prev_d=$(( t_last - $(ts "$(echo "$SL" | head -1)") ))
        else prev_d=$(( t_last - ${t_attn:-$t_last} )); fi
        last_n=$n
      fi
      BADL=$(echo "$CL" | grep -E "step: *[0-9]+ +loss" | grep -iE "loss: *-?(nan|inf)\b|grad_norm: *-?(nan|inf)\b" | head -1)
      if [ -n "$BADL" ]; then
        fire "nonfinite: $(echo "$BADL" | grep -oE 'step: *[0-9]+.*grad_norm: *[^ ]+')"; break; fi
      if grep -q "\[nkfix\] NON-FINITE" "$DBG" 2>/dev/null; then
        fire "nonfinite: $(grep -m1 "\[nkfix\] NON-FINITE" "$DBG" | sed 's/.*\[nkfix\]/[nkfix]/') (last logged step $n)"; break; fi
      if grep -q "!! BLAS-REPOINT" "$DBG" 2>/dev/null; then
        fire "blas: $(grep -m1 "!! BLAS-REPOINT" "$DBG" | sed 's/.*!! BLAS-REPOINT/BLAS-REPOINT/')"; break; fi
      [ "$n" -ge "$STEPS" ] && break
      if [ -z "$t_attn" ] && [ -n "$t_train" ] && [ $((now - t_train)) -gt "$WD_PRE" ]; then
        fire "hang: no first attention fwd ${WD_PRE}s after training start"; break; fi
      if [ -n "$t_attn" ] && [ "$n" = 0 ] && [ $((now - t_attn)) -gt "$WD_FIRST" ]; then
        fire "hang: step 1 not printed ${WD_FIRST}s after first attention fwd"; break; fi
      if [ "$n" -ge 1 ] && [ -n "$t_last" ]; then
        thr=$(( 3 * ${prev_d:-0} )); [ "$thr" -lt "$WD_FLOOR" ] && thr=$WD_FLOOR
        if [ $((now - t_last)) -gt "$thr" ]; then
          fire "hang: step $((n+1)) running $((now - t_last))s > ${thr}s (prev step ${prev_d}s)"; break; fi
      fi
    done ) &
  WDPID=$!
  trap 'kill $CLKPID $KWPID $MEMPID $WDPID 2>/dev/null' EXIT
  # nkfix (E2E_NKFIX=1): every bwd GEMM off hipBLASLt's MT32x16x32 fallback; NKFIX_CHECK=1 = non-finite check
  NKFIX_ENV=""; NKSTATS=""
  if [ "${E2E_NKFIX:-0}" != 0 ]; then
    NKSTATS=$KIT/logs/nkfix.$TAG.txt; rm -f "$NKSTATS"
    NKFIX_ENV="-e E2E_NKFIX=1 -e NKFIX_STATS_FILE=$NKSTATS"
    for v in NKFIX_MIN_BYTES NKFIX_CHUNK_BYTES NKFIX_CHECK NKFIX_CHECK_EVERY NKFIX_RULES NKFIX_SHADOW NKFIX_TRANSPOSE; do
      [ -n "${!v:-}" ] && NKFIX_ENV="$NKFIX_ENV -e $v=${!v}"; done
  fi
  MARK=$(dmesg_mark); MCE0=$(timeout 20 sudo -n dmesg 2>/dev/null | grep -c "mce: \[Hardware Error\]")
  T0=$(date +%s)
  echo "[$(date +%T)] $TAG E2E_ATTN=$ATTN steps=$STEPS pfreq=$PFREQ nlayers=${E2E_NLAYERS:-32} (lock: ${LOCKCMD:-held by driver})"
  pre_exec_guard
  $LOCKCMD timeout ${E2E_TIMEOUT:-2700} docker exec \
    -e GPUS_PER_NODE=1 -e NNODES=1 -e NODE_RANK=0 -e PRIMUS_GPU_MODEL=MI455X \
    -e MASTER_PORT=$((20000 + RANDOM % 20000)) -e PRIMUS_EXP_NAME=$TAG -e E2E_RUN_MARKER=$TAG \
    -e TRITON_CACHE_DIR=/tmp/triton_cache_e2e -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 \
    -e FLYDSL_RUNTIME_CACHE_DIR=${E2E_FLYCACHE:?fresh dir per process} -e E2E_ATTN="$ATTN" \
    -e E2E_FLY_TREES="${E2E_FLY_TREES_JSON:?trees json}" -e E2E_EXPECT_BLAS_LIB=$BLAS_LIB -e E2E_ATTN_EVENTS=$EVF \
    $NKFIX_ENV ${E2E_ENV:-} \
    $CT bash -c "ulimit -c 0; $BLAS_EXPORT; export PYTHONPATH=$PP; \
      echo E2E_ENV PREFER=\$TORCH_BLAS_PREFER_HIPBLASLT LIB=\$HIPBLASLT_TENSILE_LIBPATH E2E_ATTN=\$E2E_ATTN; \
      cd $PRIMUS && exec timeout --foreground -k 20 ${E2E_INNER:-2600} bash runner/primus-cli direct \
        --log_file $RUNDIR/launcher.log -- train pretrain --config $CFG" \
    > "$LOG" 2>&1
  RC=$?
  T1=$(date +%s)
  kill $CLKPID $KWPID $MEMPID $WDPID 2>/dev/null
  [ -s "$WDF" ] && { echo "!! watchdog fired:"; cat "$WDF"; }
  [ -s "$LOG.memguard" ] && { echo "!! memguard fired:"; cat "$LOG.memguard"; }
  [ -s "$FOREIGN" ] && { cat "$FOREIGN"; echo "!! $TAG INVALID: a second GPU client held KFD during the run"; }
  # the files that ran, hashed again: a change during the run means the code that ran is unknown
  TCHG=$(diff <(grep -vF " $CFG" "$TREEF") <(run_files_sha) | grep '^[<>]' | awk '{print $NF}' | sort -u | tr '\n' ' ')
  [ -n "$TCHG" ] && echo "!! TREE CHANGED DURING RUN $TAG (run INVALID): $TCHG"
  # torchtitan writes traces to the shared $PRIMUS/outputs/profile_traces/iteration_N: keep this run's copy
  for d in $(find "$PRIMUS/outputs/profile_traces" -mindepth 1 -maxdepth 1 -type d -newer "$CFG" 2>/dev/null); do
    mkdir -p "$KIT/traces/$TAG" && cp -r "$d" "$KIT/traces/$TAG/"; done
  # rank-0 debug.log lives outside the repo: keep its evidence lines with the run
  grep -aE "\[e2e_attn\]|\[nkfix\]|\[flydsl_bwd|BLAS-REPOINT" "$DBG" 2>/dev/null | head -2000 > "$KIT/runs/$TAG.dbg.txt"
  CLEAN=$(sed 's/\x1b\[[0-9;]*m//g' "$LOG")
  NSTEP=$(echo "$CLEAN" | grep -cE "step: *[0-9]+ +loss")
  NONFIN=$(echo "$CLEAN" | grep -E "step: *[0-9]+ +loss" | grep -ciE "loss: *-?(nan|inf)\b|grad_norm: *-?(nan|inf)\b")
  NKNF=$(grep -c "\[nkfix\] NON-FINITE" "$DBG" 2>/dev/null); NKNF=${NKNF:-0}
  REPT=$(grep -c "!! BLAS-REPOINT" "$DBG" 2>/dev/null); REPT=${REPT:-0}
  NEV=0; [ -f "$EVF" ] && NEV=$(wc -l < "$EVF")
  [ "${NONFIN:-0}" -gt 0 ] && echo "!! $TAG: $NONFIN step lines with a non-finite loss or grad_norm -- DISCARD this run"
  [ -n "$NKSTATS" ] && { echo "nkfix stats ($NKSTATS):"; head -3 "$NKSTATS" 2>/dev/null || echo "!! no nkfix stats file"; }
  echo "rc=$RC tag=$TAG log=$LOG steps_logged=$NSTEP nonfinite_lines=$NONFIN nkfix_nonfinite=$NKNF blas_repoints=$REPT event_steps=$NEV wall_s=$((T1-T0))"
  INFO=$(dmesg_info "$MARK"); FAULT=$(dmesg_fault "$MARK")
  MCE1=$(timeout 20 sudo -n dmesg 2>/dev/null | grep -c "mce: \[Hardware Error\]")
  L=$(leftovers)
  INV=""
  [ -s "$FOREIGN" ] && INV="$INV foreign-KFD"; [ -n "$TCHG" ] && INV="$INV tree-changed"
  [ "${NONFIN:-0}" -gt 0 ] || [ "${NKNF:-0}" -gt 0 ] && INV="$INV non-finite"
  [ "${REPT:-0}" -gt 0 ] && INV="$INV BLAS-repoint"; [ -s "$WDF" ] && INV="$INV watchdog"; [ -n "$FAULT" ] && INV="$INV dmesg-FAULT"
  { echo "tag: $TAG"; echo "spec: $ATTN"; echo "steps: $STEPS"; echo "pfreq: $PFREQ"; echo "nlayers: ${E2E_NLAYERS:-32}"
    echo "rc: $RC"; echo "wall_s: $((T1-T0))"; echo "steps_logged: $NSTEP"; echo "nonfinite_lines: $NONFIN"
    echo "nkfix_nonfinite_lines: $NKNF"; echo "nkfix_events: $(grep -m1 '^nonfinite_events' "$NKSTATS" 2>/dev/null | cut -d: -f2-)"
    echo "blas_repoints: $REPT"; echo "event_steps: $NEV"; echo "watchdog: $(cat "$WDF" 2>/dev/null | head -1)"
    echo "memguard: $(cat "$LOG.memguard" 2>/dev/null | head -1)"; echo "dmesg_fault_lines: $(echo -n "$FAULT" | grep -c .)"
    echo "dmesg_info_lines: $(echo -n "$INFO" | grep -c .)"; echo "cpu_mce_new: $((MCE1-MCE0))"
    echo "foreign_kfd: $(grep -m1 'FOREIGN KFD' "$FOREIGN" 2>/dev/null)"; echo "tree_changed_during_run: ${TCHG:-no}"
    echo "valid: $([ -n "$INV" ] && echo "INVALID ($INV )" || echo yes)"
    echo "leftovers: $(echo -n "$L" | tr '\n' ' ')"; echo "flycache: ${E2E_FLYCACHE}"
    echo "bwd_stream_mode: $(grep -m1 -oE '\[flydsl_bwd [^]]+\] DQ_SIDE_STREAM=[01] DQ_SIDE_RECORD=[01]' "$DBG" 2>/dev/null)"
    echo "log: $LOG"; echo "events: $EVF"; echo "clk: $CLK"; echo "env: $KIT/logs/env.$TAG.txt"; echo "tree: $TREEF"
    echo "dbg: $KIT/runs/$TAG.dbg.txt ($(wc -l < "$KIT/runs/$TAG.dbg.txt") lines of $DBG)"; } > "$POST"
  [ -n "$INFO" ] && { echo "dmesg INFO (RAS correctable / MCE; not a stop):"; echo "$INFO" | head -5; }
  if [ -n "$FAULT" ]; then echo "!! NEW dmesg FAULT lines on a live card -- stop all card work:"; echo "$FAULT" | head -40; exit 99; fi
  [ -n "$L" ] && echo "!! leftover processes in $CT (kill by PID only): $L"
  [ -s "$FOREIGN" ] && exit 91
  exit $RC ;;
*) echo "unknown mode $MODE"; exit 2 ;;
esac
