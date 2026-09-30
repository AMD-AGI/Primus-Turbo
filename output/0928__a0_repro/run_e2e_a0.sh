#!/bin/bash
# A0 copy (output/0928__a0_repro) of the B0 e2e launcher: Llama-3.1-8B, BF16, MBS=GBS=4, seq 8192, 1 GPU (GPU 0 / container fa-g0).
# The attention arm is selected by ONE env var, E2E_ATTN (see attn_backends/e2e_attn/__init__.py):
#
#   run_e2e.sh train   <tag> <E2E_ATTN> [steps=30] [profile_freq=14|0=off]
#   run_e2e.sh opcheck <arm: turbo|asm|fly> <shape: fast|prod> [extra opcheck.py args]
#
#   E2E_ATTN=turbo                          P1: real Primus-Turbo @ wt-bakeoff (Triton), no FlyDSL
#   E2E_ATTN=asm | fly | asm/fly | "W;C"    P2: shim primus_turbo + flydsl 0.3.4.1 + aiter
#
# Nothing in the Primus or Primus-Turbo trees is edited: the arm is injected by putting
# attn_backends/shim (a package named primus_turbo) first on PYTHONPATH. In P1 the shim re-points
# itself at the real package and only wraps TurboAttention; in P2 it is the whole package.
#
# Card discipline (LAB-RULES, gfx1250-card-safety): the whole run holds /tmp/b0-gpu0.lock; bash -c
# (NOT -lc: /etc/profile.d/zz-gfx1250.sh would force TORCH_BLAS_PREFER_HIPBLASLT=0); BLAS env is
# exported INSIDE the command; no HIP_VISIBLE_DEVICES; no global KFD reaping (GPU2/3 jobs hold
# KFD) -- only fa-g0's own leftovers are named; clocks from /sys/class/drm/card0, never rocm-smi;
# dmesg checked after every run, ignoring 0002:04:00.0 (GPU 1, already dead).
set -u
E2E=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/e2e
PRIMUS=/home/lihuzhan/code/2026_0828__primus/Primus
WT=/home/lihuzhan/code/2026_0903__turbo/wt-bakeoff
FLY=/home/lihuzhan/.local/flydsl0341
AITER=/home/lihuzhan/code/aiter-src
SHIM=$E2E/attn_backends/shim
LOCK=/tmp/a0-gpu0.lock
CT=fa-repro
# hipBLASLt library: the IMAGE library (A0: 32L, ~17 s/step, 2,027 tps -- 0915__opt/BLAS-FINDING.md).
# The host library ~/.local/hipblaslt-gfx1250 (only square GEMMs ever validated) was used by the
# first run p1a_turbo, which hung in step 1; override with E2E_BLAS_LIB for opcheck-style reuse.
BLAS_LIB=${E2E_BLAS_LIB:-/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250}
BLAS_EXPORT="export TORCH_BLAS_PREFER_HIPBLASLT=1 HIPBLASLT_TENSILE_LIBPATH=$BLAS_LIB"
mkdir -p "$E2E/logs"

arm_pythonpath() {   # $1 = E2E_ATTN value
  if [ "$1" = "turbo" ]; then echo "$SHIM:$E2E/attn_backends:$WT:$AITER"
  else echo "$SHIM:$E2E/attn_backends:$FLY:$AITER"; fi
}

dmesg_mark() { timeout 20 sudo -n dmesg 2>/dev/null | grep -v "0002:04:00.0" | tail -1 | sed 's/^\[\s*\([0-9.]*\)\].*/\1/'; }
dmesg_check() {   # $1 = timestamp mark from before the run; prints NEW non-GPU1 amdgpu lines
  timeout 20 sudo -n dmesg 2>/dev/null | grep -v "0002:04:00.0" | awk -v m="$1" '
    { t=$0; sub(/^\[ */,"",t); sub(/\].*/,"",t); if (t+0 > m+0) print }' |
    grep -v "Hardware Error\|mce:" | grep -iE "amdgpu|gpu reset|MES\(|GCVM|page fault|reset ack|ring .* timeout|SIGBUS|Queues reset|general protection" || true
}
leftovers() {   # our own container only
  timeout 20 docker top $CT -eo pid,etime,args 2>/dev/null | grep -E "torchrun|primus/cli/main.py|opcheck.py" | grep -v grep || true
}

MODE=${1:?mode: train|opcheck}; shift
case "$MODE" in
opcheck)
  ARM=${1:?arm}; SHAPE=${2:?shape}; shift 2
  PP=$(arm_pythonpath "$ARM")
  TAG=opcheck.$ARM.$SHAPE.$(date +%m%d_%H%M%S)
  OUT=$E2E/opcheck; mkdir -p "$OUT"
  MARK=$(dmesg_mark)
  echo "[$(date +%T)] $TAG waiting for $LOCK"
  flock $LOCK timeout 1500 docker exec \
    -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 -e FLYDSL_RUNTIME_CACHE_DIR=${E2E_FLYCACHE:-/tmp/flycache_e2e} \
    -e TRITON_CACHE_DIR=/tmp/triton_cache_e2e ${OPCHECK_ENV:-} $CT bash -c "ulimit -c 0; $BLAS_EXPORT; \
      export PYTHONPATH=$PP; cd $E2E && exec timeout --foreground -k 20 1400 \
      /opt/venv/bin/python3 attn_backends/opcheck.py --arm $ARM --shape $SHAPE --json $OUT/$TAG.json $*" \
    > "$OUT/$TAG.log" 2>&1
  RC=$?
  echo "rc=$RC log=$OUT/$TAG.log"
  NEW=$(dmesg_check "$MARK")
  if [ -n "$NEW" ]; then echo "!! NEW dmesg lines on a live card:"; echo "$NEW"; exit 99; fi
  L=$(leftovers); [ -n "$L" ] && echo "!! leftover processes in $CT: $L"
  exit $RC ;;
train)
  TAG=${1:?tag}; ATTN=${2:?E2E_ATTN}; STEPS=${3:-30}; PFREQ=${4:-14}
  PP=$(arm_pythonpath "$ATTN")
  CFG=$E2E/configs/run.$TAG.yaml
  PROFILE=true; [ "$PFREQ" = 0 ] && { PROFILE=false; PFREQ=1000; }
  # E2E_NLAYERS (optional): smoke runs with fewer layers via Primus' torchtitan.model_override patch
  NL_SED=""; [ -n "${E2E_NLAYERS:-}" ] && NL_SED="/^ *flavor: 8B_flex\$/a\\        n_layers: ${E2E_NLAYERS}"
  sed -e "s/@STEPS@/$STEPS/" -e "s/@PFREQ@/$PFREQ/g" -e "s/@PROFILE@/$PROFILE/" ${NL_SED:+-e "$NL_SED"} \
    "$E2E/configs/l8b_e2e.template.yaml" > "$CFG"
  RUNDIR=/home/lihuzhan/_dbg_l8b/$TAG; mkdir -p "$RUNDIR"
  LOG=$E2E/logs/e2e.$TAG.log; CLK=$E2E/logs/clk.$TAG.csv
  L=$(leftovers); [ -n "$L" ] && { echo "!! refusing to start: leftovers in $CT: $L"; exit 98; }
  # snapshot what this run executes: the arm trees move when the op-evolve jobs advance
  (cd "$E2E" && find arms attn_backends -name '*.py' -type f | sort | xargs sha256sum) > "$E2E/logs/tree.$TAG.sha256"
  echo "t,busy_pct,sclk_mhz,power_uw,temp_mc" > "$CLK"
  ( H=$(ls -d /sys/class/drm/card1/device/hwmon/hwmon* 2>/dev/null | head -1)
    while :; do
      printf '%s,%s,%s,%s,%s\n' "$(date +%s)" \
        "$(timeout 5 cat /sys/class/drm/card1/device/gpu_busy_percent 2>/dev/null)" \
        "$(grep '\*' /sys/class/drm/card1/device/pp_dpm_sclk 2>/dev/null | grep -oE '[0-9]+Mhz' | tr -d Mhz)" \
        "$(cat $H/power1_average 2>/dev/null || cat $H/power1_input 2>/dev/null)" \
        "$(cat $H/temp2_input 2>/dev/null || cat $H/temp1_input 2>/dev/null)"
      sleep 5
    done ) >> "$CLK" 2>/dev/null &
  CLKPID=$!
  # Memory guard: A0 SIGBUS'd at 88.30% and leaked its KFD context (AC cycle). torchtitan logs
  # "memory: <x>GiB(<p>%)" every step; above E2E_MEM_STOP (default 88.5) send ONE SIGTERM to
  # this run's torchrun (never SIGKILL -- card-safety #8) and let it wind down.
  ( STOP=${E2E_MEM_STOP:-88.5}; sent=0
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
  # Step-time watchdog (added after p1a_turbo hung 14 min in step 1 with no evidence trail).
  # Anchors: "Training starts at step" (main log), the first attention fwd = "[e2e_attn] BLAS"
  # (rank-0 debug.log), and the timestamps of "step: N loss" lines. Fires ONCE when
  #   - no first attention fwd within E2E_WD_PRE s (900) after training start, or
  #   - step 1 not printed within E2E_WD_FIRST s (360) after the first attention fwd, or
  #   - the running step exceeds max(3 x previous step, E2E_WD_FLOOR s (240: profiler export)).
  # Firing = collect evidence (docker top, busy%/sclk, dmesg, one gdb bt of the worker), then ONE
  # SIGTERM to this run's torchrun by host PID (never SIGKILL -- card-safety #8).
  DBG=/home/lihuzhan/_dbg_l8b/output/amd/root/$TAG/logs/pre_trainer/rank-0/debug.log
  [ -e "$(dirname "$DBG")" ] && mv "/home/lihuzhan/_dbg_l8b/output/amd/root/$TAG" "/home/lihuzhan/_dbg_l8b/output/amd/root/$TAG.old.$(date +%H%M%S)"
  ( WD_PRE=${E2E_WD_PRE:-900}; WD_FIRST=${E2E_WD_FIRST:-360}; WD_FLOOR=${E2E_WD_FLOOR:-240}
    t_train=""; t_attn=""; last_n=0; t_last=""; prev_d=""
    ts() { date -d "$(echo "$1" | grep -oE '^\[[0-9]{8} [0-9:]{8}\]' | tr -d '[]')" +%s 2>/dev/null; }
    fire() {
      F=$E2E/logs/hang.$TAG
      echo "$(date +%T) WATCHDOG fired: $1" | tee -a "$F.actions.txt" >> "$LOG.watchdog"
      timeout 20 docker top $CT -eo pid,ppid,stat,etime,time,args > "$F.dockertop.txt" 2>&1
      for i in 1 2 3 4 5; do
        echo "$(date +%T) busy=$(timeout 5 cat /sys/class/drm/card1/device/gpu_busy_percent) sclk=$(grep '\*' /sys/class/drm/card1/device/pp_dpm_sclk)" >> "$F.actions.txt"
        sleep 1
      done
      timeout 20 sudo -n dmesg 2>/dev/null | tail -80 > "$F.dmesg.txt"
      W=$(grep -E "python.*primus/cli/main.py" "$F.dockertop.txt" | grep -v -E "torchrun|timeout|bash" | awk '{print $1}' | tail -1)
      [ -n "$W" ] && timeout 90 sudo -n gdb -p "$W" -batch -ex "thread apply 1 bt" > "$F.gdb.txt" 2>&1
      for pid in $(grep -E "torchrun.*primus|pt_elastic" "$F.dockertop.txt" | grep -v -E "timeout|bash " | awk '{print $1}'); do
        echo "$(date +%T) SIGTERM torchrun pid $pid" | tee -a "$F.actions.txt" >> "$LOG.watchdog"
        sudo -n kill -TERM "$pid" 2>/dev/null
      done
    }
    while :; do
      sleep 10; now=$(date +%s)
      CL=$(sed 's/\x1b\[[0-9;]*m//g' "$LOG" 2>/dev/null)
      [ -z "$t_train" ] && echo "$CL" | grep -q "Training starts at step" && t_train=$now
      [ -z "$t_attn" ] && grep -q "\[e2e_attn\] BLAS" "$DBG" 2>/dev/null && t_attn=$now
      SL=$(echo "$CL" | grep -E "step: *[0-9]+ +loss" | tail -2)
      n=$(echo "$SL" | tail -1 | grep -oE "step: *[0-9]+" | grep -oE "[0-9]+"); n=${n:-0}
      if [ "$n" -gt "$last_n" ]; then
        t_last=$(ts "$(echo "$SL" | tail -1)")
        if [ "$(echo "$SL" | wc -l)" -ge 2 ] && [ "$n" -ge 2 ]; then prev_d=$(( t_last - $(ts "$(echo "$SL" | head -1)") ))
        else prev_d=$(( t_last - ${t_attn:-$t_last} )); fi
        last_n=$n
      fi
      [ "$n" -ge "$STEPS" ] && break
      if [ -z "$t_attn" ] && [ -n "$t_train" ] && [ $((now - t_train)) -gt "$WD_PRE" ]; then
        fire "no first attention fwd ${WD_PRE}s after training start"; break; fi
      if [ -n "$t_attn" ] && [ "$n" = 0 ] && [ $((now - t_attn)) -gt "$WD_FIRST" ]; then
        fire "step 1 not printed ${WD_FIRST}s after first attention fwd"; break; fi
      if [ "$n" -ge 1 ] && [ -n "$t_last" ]; then
        thr=$(( 3 * ${prev_d:-0} )); [ "$thr" -lt "$WD_FLOOR" ] && thr=$WD_FLOOR
        if [ $((now - t_last)) -gt "$thr" ]; then
          fire "step $((n+1)) running $((now - t_last))s > ${thr}s (prev step ${prev_d}s)"; break; fi
      fi
    done ) &
  WDPID=$!
  trap 'kill $CLKPID $MEMPID $WDPID 2>/dev/null' EXIT
  # Opt-in GEMM layout workaround (../gemm/nkfix_b0.py, installed by the shim): E2E_NKFIX=1.
  # NKFIX_* knobs pass through; per-run rule/non-finite stats land in ../gemm/logs/nkfix.<tag>.txt.
  NKFIX_ENV=""; NKSTATS=""
  if [ "${E2E_NKFIX:-0}" != 0 ]; then
    NKSTATS=$E2E/../gemm/logs/nkfix.$TAG.txt; rm -f "$NKSTATS"
    NKFIX_ENV="-e E2E_NKFIX=1 -e NKFIX_STATS_FILE=$NKSTATS"
    for v in NKFIX_MIN_BYTES NKFIX_CHUNK_BYTES NKFIX_CHECK NKFIX_CHECK_EVERY NKFIX_RULES NKFIX_SHADOW NKFIX_TRANSPOSE; do
      [ -n "${!v:-}" ] && NKFIX_ENV="$NKFIX_ENV -e $v=${!v}"; done
  fi
  MARK=$(dmesg_mark)
  echo "[$(date +%T)] $TAG E2E_ATTN=$ATTN steps=$STEPS pfreq=$PFREQ waiting for $LOCK"
  flock $LOCK timeout ${E2E_TIMEOUT:-2700} docker exec \
    -e GPUS_PER_NODE=1 -e NNODES=1 -e NODE_RANK=0 -e PRIMUS_GPU_MODEL=MI455X \
    -e MASTER_PORT=$((20000 + RANDOM % 20000)) -e PRIMUS_EXP_NAME=$TAG -e E2E_RUN_MARKER=$TAG \
    -e TRITON_CACHE_DIR=/tmp/triton_cache_e2e -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 \
    -e FLYDSL_RUNTIME_CACHE_DIR=${E2E_FLYCACHE:-/tmp/flycache_e2e} -e E2E_ATTN="$ATTN" $NKFIX_ENV ${E2E_ENV:-} \
    $CT bash -c "ulimit -c 0; $BLAS_EXPORT; export PYTHONPATH=$PP; \
      echo E2E_ENV PREFER=\$TORCH_BLAS_PREFER_HIPBLASLT LIB=\$HIPBLASLT_TENSILE_LIBPATH E2E_ATTN=\$E2E_ATTN; \
      cd $PRIMUS && exec timeout --foreground -k 20 ${E2E_INNER:-2600} bash runner/primus-cli direct \
        --log_file $RUNDIR/launcher.log -- train pretrain --config $CFG" \
    > "$LOG" 2>&1
  RC=$?
  kill $CLKPID $MEMPID $WDPID 2>/dev/null
  [ -s "$LOG.watchdog" ] && { echo "!! watchdog fired:"; cat "$LOG.watchdog"; }
  # torchtitan writes traces to the shared $PRIMUS/outputs/profile_traces/iteration_N (overwritten by
  # the next run): keep this run's copy next to its log.
  for d in $(find "$PRIMUS/outputs/profile_traces" -mindepth 1 -maxdepth 1 -type d -newer "$CFG" 2>/dev/null); do
    mkdir -p "$E2E/traces/$TAG" && cp -r "$d" "$E2E/traces/$TAG/"; done
  NANS=$(sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -ci "loss: *nan")
  NONFIN=$(sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -ciE "loss: *(nan|inf)|grad_norm: *(nan|inf)")
  [ "${NONFIN:-0}" -gt 0 ] && echo "!! $TAG: $NONFIN step lines with a non-finite loss or grad_norm"
  [ -n "$NKSTATS" ] && { echo "nkfix stats ($NKSTATS):"; head -3 "$NKSTATS" 2>/dev/null || echo "!! no nkfix stats file"; }
  echo "rc=$RC tag=$TAG log=$LOG nan_steps=$NANS"
  [ "${NANS:-0}" -gt 0 ] && echo "!! $TAG: loss went nan on $NANS steps -- DISCARD this run"
  NEW=$(dmesg_check "$MARK")
  if [ -n "$NEW" ]; then echo "!! NEW dmesg lines on a live card -- stop all card work:"; echo "$NEW"; exit 99; fi
  L=$(leftovers); [ -n "$L" ] && echo "!! leftover processes in $CT (kill by PID only): $L"
  exit $RC ;;
*) echo "unknown mode $MODE"; exit 2 ;;
esac
