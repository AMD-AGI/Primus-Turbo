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
# Differences from the 09-28 A0 copy (each one a 09-28 lesson; E2E-PLAN.md section 3):
#   * adapter = this kit's attn_backends (BLAS guard + per-step CUDA-event attention timing,
#     E2E_ATTN_EVENTS) and FlyDSL trees only from E2E_FLY_TREES (no silent default)
#   * E2E_EXPECT_BLAS_LIB = the IMAGE hipBLASLt library, assigned before python starts (bash -c, not -lc)
#   * watchdog also stops the run (ONE SIGTERM to torchrun, never SIGKILL) on the first non-finite
#     loss / grad_norm, the first "[nkfix] NON-FINITE", or a "!! BLAS-REPOINT" line
#   * dmesg is classified: FAULT (stop) vs INFO (CPU MCE, GPU RAS *correctable* pcie_pl reports, which
#     A0 logs during normal work, e.g. 2026-10-02 07:58:50) -- the 09-28 filter would have stopped on INFO
#   * evidence: worker /proc/<pid>/environ (E2E_FLY_TREES, FLYDSL_RUNTIME_CACHE_DIR, BLAS env), sha256 of
#     every file that runs, a key: value post-run summary runs/<tag>.post.txt
set -u
KIT=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__e2e/e2e
B0E=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/e2e
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

MODE=${1:?mode: train|opcheck}; shift
case "$MODE" in
opcheck)
  ARM=${1:?arm}; SHAPE=${2:?shape}; shift 2
  TAG=opcheck.$ARM.$SHAPE.$(date +%m%d_%H%M%S)
  OUT=$KIT/runs; MARK=$(dmesg_mark)
  echo "[$(date +%T)] $TAG start (lock: ${LOCKCMD:-held by driver})"
  $LOCKCMD timeout 1500 docker exec \
    -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 -e FLYDSL_RUNTIME_CACHE_DIR=${E2E_FLYCACHE:?fresh dir per process} \
    -e TRITON_CACHE_DIR=/tmp/triton_cache_e2e -e E2E_EXPECT_BLAS_LIB=$BLAS_LIB \
    -e E2E_FLY_TREES="${E2E_FLY_TREES_JSON:-}" \
    -e E2E_ATTN_EVENTS=$OUT/$TAG.attn_ev.jsonl ${OPCHECK_ENV:-} $CT bash -c "ulimit -c 0; $BLAS_EXPORT; \
      export PYTHONPATH=$PP; cd $KIT/attn_backends && exec timeout --foreground -k 20 1400 \
      /opt/venv/bin/python3 opcheck.py --arm $ARM --shape $SHAPE --json $OUT/$TAG.json $*" \
    > "$OUT/$TAG.log" 2>&1
  RC=$?
  echo "rc=$RC log=$OUT/$TAG.log json=$OUT/$TAG.json"
  INFO=$(dmesg_info "$MARK"); [ -n "$INFO" ] && { echo "dmesg INFO (not a stop):"; echo "$INFO" | head -5; }
  NEW=$(dmesg_fault "$MARK")
  if [ -n "$NEW" ]; then echo "!! NEW dmesg FAULT lines -- stop all card work:"; echo "$NEW" | head -40; exit 99; fi
  L=$(leftovers); [ -n "$L" ] && echo "!! leftover processes in $CT (kill by PID only): $L"
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
  POST=$KIT/runs/$TAG.post.txt; WDF=$LOG.watchdog
  rm -f "$EVF" "$WDF" "$LOG.memguard" "$POST"
  L=$(leftovers); [ -n "$L" ] && { echo "!! refusing to start: leftovers in $CT: $L"; exit 98; }
  KFD=$(ls /sys/class/kfd/kfd/proc 2>/dev/null | tr '\n' ' '); [ -n "$KFD" ] && { echo "!! refusing to start: KFD holders: $KFD"; exit 97; }
  # snapshot of every file that runs (the arm trees can move under us)
  TREES=$(python3 -c 'import json,os,sys; t=json.loads(os.environ.get("E2E_FLY_TREES_JSON","{}")); print(" ".join(sorted({d for a in t.values() for d in a.values()})))')
  ( cd / && find $KIT/attn_backends $B0E/arms/asm $TREES -name '*.py' -type f -not -path '*/__pycache__/*' | sort | xargs sha256sum
    sha256sum /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/gemm/nkfix_b0.py "$CFG" ) > "$KIT/logs/tree.$TAG.sha256"
  echo "t,busy_pct,sclk_mhz,fclk_mhz,power_uw,temp_mc" > "$CLK"
  ( H=$(ls -d $CARD/hwmon/hwmon* 2>/dev/null | head -1)
    while :; do
      printf '%s,%s,%s,%s,%s,%s\n' "$(date +%s)" "$(timeout 5 cat $CARD/gpu_busy_percent 2>/dev/null)" \
        "$(grep '\*' $CARD/pp_dpm_sclk 2>/dev/null | grep -oE '[0-9]+Mhz' | tr -d Mhz)" \
        "$(grep '\*' $CARD/pp_dpm_fclk 2>/dev/null | grep -oE '[0-9]+Mhz' | tr -d Mhz)" \
        "$(cat $H/power1_average 2>/dev/null || cat $H/power1_input 2>/dev/null)" \
        "$(cat $H/temp2_input 2>/dev/null || cat $H/temp1_input 2>/dev/null)"
      sleep 5
    done ) >> "$CLK" 2>/dev/null &
  CLKPID=$!
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
  trap 'kill $CLKPID $MEMPID $WDPID 2>/dev/null' EXIT
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
  kill $CLKPID $MEMPID $WDPID 2>/dev/null
  [ -s "$WDF" ] && { echo "!! watchdog fired:"; cat "$WDF"; }
  [ -s "$LOG.memguard" ] && { echo "!! memguard fired:"; cat "$LOG.memguard"; }
  # torchtitan writes traces to the shared $PRIMUS/outputs/profile_traces/iteration_N: keep this run's copy
  for d in $(find "$PRIMUS/outputs/profile_traces" -mindepth 1 -maxdepth 1 -type d -newer "$CFG" 2>/dev/null); do
    mkdir -p "$KIT/traces/$TAG" && cp -r "$d" "$KIT/traces/$TAG/"; done
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
  { echo "tag: $TAG"; echo "spec: $ATTN"; echo "steps: $STEPS"; echo "pfreq: $PFREQ"; echo "nlayers: ${E2E_NLAYERS:-32}"
    echo "rc: $RC"; echo "wall_s: $((T1-T0))"; echo "steps_logged: $NSTEP"; echo "nonfinite_lines: $NONFIN"
    echo "nkfix_nonfinite_lines: $NKNF"; echo "nkfix_events: $(grep -m1 '^nonfinite_events' "$NKSTATS" 2>/dev/null | cut -d: -f2-)"
    echo "blas_repoints: $REPT"; echo "event_steps: $NEV"; echo "watchdog: $(cat "$WDF" 2>/dev/null | head -1)"
    echo "memguard: $(cat "$LOG.memguard" 2>/dev/null | head -1)"; echo "dmesg_fault_lines: $(echo -n "$FAULT" | grep -c .)"
    echo "dmesg_info_lines: $(echo -n "$INFO" | grep -c .)"; echo "cpu_mce_new: $((MCE1-MCE0))"
    echo "leftovers: $(echo -n "$L" | tr '\n' ' ')"; echo "flycache: ${E2E_FLYCACHE}"
    echo "bwd_stream_mode: $(grep -m1 -oE '\[flydsl_bwd [^]]+\] DQ_SIDE_STREAM=[01] DQ_SIDE_RECORD=[01]' "$DBG" 2>/dev/null)"
    echo "log: $LOG"; echo "events: $EVF"; echo "clk: $CLK"; echo "env: $KIT/logs/env.$TAG.txt"; } > "$POST"
  [ -n "$INFO" ] && { echo "dmesg INFO (RAS correctable / MCE; not a stop):"; echo "$INFO" | head -5; }
  if [ -n "$FAULT" ]; then echo "!! NEW dmesg FAULT lines on a live card -- stop all card work:"; echo "$FAULT" | head -40; exit 99; fi
  [ -n "$L" ] && echo "!! leftover processes in $CT (kill by PID only): $L"
  exit $RC ;;
*) echo "unknown mode $MODE"; exit 2 ;;
esac
