#!/bin/bash
# CPU-only preflight for the 2026-10-02 A0 e2e (never touches the GPU: sysfs/dmesg reads, md5, greps,
# host python without torch, and ONE container python with HIP_VISIBLE_DEVICES=-1 that imports the
# adapter but loads no arm and makes no device call). Prints PREFLIGHT_OK, or PREFLIGHT_FAIL <n> and
# exits 93. WARN lines do not fail. Usage: MODE=3arm|2arm bash preflight.sh
# drive.sh runs it again while holding /tmp/a0-gpu0.lock (E2E_LOCK_HELD=1). Run standalone while another
# process holds that lock (another GPU client is mid-run), the container python is skipped (WARN) -- it is
# never skipped under the driver. The lock is only read (/proc/locks), never taken.
set -u
. /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__e2e/e2e/trees.sh
KFD_DIR=$KFD_PROC                     # the real card (KFD_DIR is a dry-run override of drive.sh only)
MODE=${MODE:-3arm}
CT=fa-repro; CARD=/sys/class/drm/card1/device
PT=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo
BLAS_LIB=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250
FAIL=0
ok()   { echo "  ok   $*"; }
warn() { echo "  WARN $*"; }
bad()  { echo "  FAIL $*"; FAIL=$((FAIL+1)); }
md5of() { md5sum "$1" 2>/dev/null | cut -d' ' -f1; }
lock_holders() {   # PIDs holding a flock on $1 -- read-only (/proc/locks; waiters show "->" and are skipped)
  local ino; ino=$(stat -c %i "$1" 2>/dev/null) || return 0
  awk -v ino="$ino" '$2 == "FLOCK" { n = split($6, a, ":"); if (a[n] == ino) print $5 }' /proc/locks | sort -u | tr '\n' ' ' | sed 's/ $//'
}

echo "== host / card / container"
[ "$(hostname)" = heliosr-1b114-c07-1 ] && ok "host A0" || bad "host is $(hostname), not A0"
SMAX=$(grep -oE '[0-9]+Mhz' $CARD/pp_dpm_sclk 2>/dev/null | tr -d Mhz | sort -n | tail -1)
[ "${SMAX:-0}" -ge 2000 ] && ok "sclk DPM table $(tr -s ' \n' ' ' < $CARD/pp_dpm_sclk) (top ${SMAX} MHz, not VR-capped)" \
  || warn "sclk top level ${SMAX:-?} MHz -- the 1100 MHz VR cap is back? A0 numbers then not comparable with 09-30/10-02"
echo "  info driver $(dpkg -l amdgpu-dkms 2>/dev/null | awk '/^ii/{print $3}') vbios $(cat $CARD/vbios_version 2>/dev/null)"
K=$(kfd_holders)
[ -z "$K" ] && ok "KFD empty" || warn "KFD holders now: $(pid_desc $K)(drive.sh waits for empty + cool-down; every holder must be nameable)"
OE=$(oe_loops)
[ -z "$OE" ] && ok "no op-evolve run/resume loop" || bad "op-evolve loop running: $(pid_desc $OE)-- stop it (op-evolve stop --job ...) and do not resume it before E2E_DRIVE_DONE/STOP -- it does not take the lock"
RA=$(realab_drivers)
[ -z "$RA" ] && ok "no realab driver" || bad "realab driver running: $(pid_desc $RA)-- wait for REALAB_DRIVER_DONE (it releases the lock between its processes)"
LH=$(lock_holders /tmp/a0-gpu0.lock)
if [ -z "$LH" ]; then ok "/tmp/a0-gpu0.lock free"
elif [ "${E2E_LOCK_HELD:-0}" = 1 ]; then ok "/tmp/a0-gpu0.lock held by the driver (pid $LH)"
else warn "/tmp/a0-gpu0.lock held by $(pid_desc $LH)(another GPU client; drive.sh waits for it)"; fi
[ -e $KIT/STOP ] && warn "STOP sentinel $KIT/STOP present: drive.sh would stop (90) before its first launch"
timeout 20 docker ps --format '{{.Names}}' 2>/dev/null | grep -qx $CT && ok "container $CT up" || bad "container $CT not running"
L=$(timeout 20 docker top $CT -eo pid,args 2>/dev/null | grep -E "torchrun|primus/cli/main.py|opcheck.py|benchmark.py|kbench.py|ab.py" | grep -v grep)
[ -z "$L" ] && ok "no GPU workload in $CT" || bad "processes in $CT: $L"
timeout 30 docker exec $CT test -d $BLAS_LIB && ok "image hipBLASLt library present in $CT" || bad "image hipBLASLt library missing: $BLAS_LIB"
F=$(df -BG --output=avail /home/lihuzhan | tail -1 | tr -dc 0-9); [ "${F:-0}" -ge 20 ] && ok "disk ${F}G free" || bad "disk ${F}G free < 20G"

echo "== dmesg since amdgpu load (fault class only; RAS correctable / MCE are INFO on A0)"
LOADT=$(timeout 20 sudo -n dmesg 2>/dev/null | grep -m1 -E "\[drm\] amdgpu version|amdgpu: Initialized|Initialized amdgpu" | sed 's/^\[\s*\([0-9.]*\)\].*/\1/')
if [ -z "$LOADT" ]; then warn "cannot read dmesg (sudo -n dmesg)"; else
  DM_INFO='mce:|\[Hardware Error\]|[0-9]+ (new )?correctable hardware errors detected|ring buffer is full|audit: |apparmor|workqueue: .* hogged CPU'
  DM_FAULT='gpu reset|reset ack|ring .* timeout|MES.*(failed|unrecoverable)|GCVM|page fault|Queues reset|SIGBUS|general protection|uncorrectable|hang'
  FL=$(timeout 20 sudo -n dmesg 2>/dev/null | awk -v m="$LOADT" '{t=$0; sub(/^\[ */,"",t); sub(/\].*/,"",t); if (t+0 > m+15) print}' | grep -vE "$DM_INFO" | grep -iE "$DM_FAULT")
  [ -z "$FL" ] && ok "no fault-class dmesg line since amdgpu load (t=$LOADT)" || bad "fault-class dmesg lines since load: $(echo "$FL" | head -3)"
  NI=$(timeout 20 sudo -n dmesg 2>/dev/null | grep -cE "new correctable hardware errors detected")
  echo "  info $NI GPU RAS correctable reports since boot (pcie_pl; INFO, not a stop)"
fi

echo "== trees"
# fwd: all files but _env.py byte-identical to the champion; _env.py without BLAS assignments
( cd $FWD_TREE && md5sum -c --quiet MD5SUMS ) >/dev/null 2>&1 && ok "fwd_r16_imglib matches its MD5SUMS" || bad "fwd_r16_imglib differs from its MD5SUMS"
D=$(diff <(cd $FWD_TREE && find . -name '*.py' -not -path '*/__pycache__/*' | sort | xargs md5sum) \
         <(cd $PT/output/0927__b0/champions/fwd_r16_r13ns && find . -name '*.py' -not -path '*/__pycache__/*' | sort | xargs md5sum) | grep '^[<>]' | awk '{print $3}' | sort -u | tr '\n' ' ')
[ "$D" = "./_env.py " ] && ok "fwd copy differs from champions/fwd_r16_r13ns only in _env.py" || bad "fwd copy vs champion differs in: $D"
if [ -f "$S6_SRC_MD5SUMS" ]; then   # the gitignored e2e copy must equal the committed source dir of the re-pin agent
  M=$(cd $BWD_S6 && awk '!/^#/ && $2 !~ /\// {print}' "$S6_SRC_MD5SUMS" | md5sum -c --quiet 2>&1)
  [ -z "$M" ] && ok "bwd_s6_0341 e2e copy == output/1002__e2e/arms_src/bwd_s6_0341 (MD5SUMS)" || bad "bwd_s6_0341 e2e copy differs from arms_src: $M"
else warn "no $S6_SRC_MD5SUMS"; fi
TC=$(trees_check)   # the same check drive.sh and the launcher repeat before every launch
[ -z "$TC" ] && ok "trees_check: fwd MD5SUMS, s6 ${S6_KERNELS_MD5:0:8}/${S6_IMPL_MD5:0:8}/_env ${PIN0341_ENV_MD5:0:8} = arms_src, r29 ${R29_KERNELS_MD5:0:8}/${R29_IMPL_MD5:0:8}" \
  || { [ "${PREFLIGHT_ALLOW_SRC_DIFF:-0}" = 1 ] && warn "trees_check: $TC (override set)" || bad "trees_check: $TC"; }
for t in "s6:$BWD_S6:$S6_KERNELS_MD5:$S6_IMPL_MD5" "r29:$BWD_R29:$R29_KERNELS_MD5:$R29_IMPL_MD5"; do
  IFS=: read -r n d km im <<< "$t"
  if [ ! -d "$d" ]; then bad "bwd $n tree missing: $d"; continue; fi
  [ "$(md5of $d/kernels.py)" = "$km" ] && [ "$(md5of $d/impl.py)" = "$im" ] && ok "bwd $n kernels.py/impl.py = champion ($km / $im)" \
    || { [ "${PREFLIGHT_ALLOW_SRC_DIFF:-0}" = 1 ] && warn "bwd $n source differs from the champion (override set)" || bad "bwd $n kernels.py $(md5of $d/kernels.py) impl.py $(md5of $d/impl.py) != champion"; }
  [ "$(md5of $d/_env.py)" = "$PIN0341_ENV_MD5" ] && ok "bwd $n _env.py = the 0.3.4.1 pin ($PIN0341_ENV_MD5)" \
    || { grep -q 'flydsl0341' $d/_env.py && ! grep -q 'flydsl032' $d/_env.py && warn "bwd $n _env.py is a different 0.3.4.1 pin ($(md5of $d/_env.py))" || bad "bwd $n _env.py does not pin flydsl 0.3.4.1"; }
done
[ -f $B0E/arms/asm/_asm_bwd_kernargs.py ] && ok "asm launcher present" || bad "asm launcher missing"

echo "== no tree touches the hipBLASLt env (09-28 a0_p3b NaN / a0_p4b wedge)"
HITS=$(grep -rnE "environ\[[\"']HIPBLASLT_TENSILE_LIBPATH[\"']\]\s*=|setdefault\(\s*[\"']HIPBLASLT_TENSILE_LIBPATH|putenv\(\s*[\"']HIPBLASLT_TENSILE_LIBPATH|environ\.update\(.*HIPBLASLT_TENSILE_LIBPATH" \
       --include=*.py $FWD_TREE $BWD_S6 $BWD_R29 $B0E/arms/asm 2>/dev/null | grep -vE '^[^:]+:[0-9]+:\s*#')
[ -z "$HITS" ] && ok "no assignment/setdefault of HIPBLASLT_TENSILE_LIBPATH in any tree" || bad "HIPBLASLT_TENSILE_LIBPATH touched: $HITS"
PH=$(grep -rnE "environ\[[\"']TORCH_BLAS_PREFER_HIPBLASLT[\"']\]\s*=\s*[\"'](0|)[\"']|putenv\([\"']TORCH_BLAS_PREFER" --include=*.py $FWD_TREE $BWD_S6 $BWD_R29 $B0E/arms/asm 2>/dev/null)
[ -z "$PH" ] && ok "no tree ASSIGNS TORCH_BLAS_PREFER_HIPBLASLT=0 (the bwd trees' setdefault(...,'0') is a no-op: the launcher assigns 1 first)" || bad "TORCH_BLAS_PREFER_HIPBLASLT forced off: $PH"

echo "== s6 ISA identical under flydsl 0.3.2 and 0.3.4.1 (compile-only evidence in output/1002__e2e/isa)"
I=$PT/output/1002__e2e/isa
if [ -d $I/s6_032 ] && [ -d $I/s6_0341 ]; then
  for k in delta dkdv dqg; do
    a=$(find $I/s6_032/$k -name '2[12]_final_isa.s' | head -1); b=$(find $I/s6_0341/$k -name '2[12]_final_isa.s' | head -1)
    if [ -z "$a" ] || [ -z "$b" ]; then bad "ISA dump missing for $k"; continue; fi
    f() { grep -vE '^\s*(\.|;|//|$)' "$1" | grep -vE '^[A-Za-z_.$][A-Za-z0-9_.$]*:' | sed 's/;.*//;s/\s\+$//'; }
    n=$(diff <(f "$a") <(f "$b") | grep -c '^[<>]')
    sp=$(grep -hoE '\.(vgpr_spill_count|sgpr_spill_count|private_segment_fixed_size):\s*[0-9]+' "$b" | awk '{s+=$2} END{print s+0}')
    [ "$n" = 0 ] && [ "$sp" = 0 ] && ok "$k: $(f "$b" | wc -l) instructions, 0 differing, 0 spill/scratch" || bad "$k: $n differing ISA lines, spill/scratch sum $sp"
  done
else bad "no ISA evidence under $I (run output/1002__e2e/tools/compile_s6.sh for s6 under 0.3.2 and bwd_s6_0341 under 0.3.4.1)"; fi

echo "== dependencies"
for p in /home/lihuzhan/code/2026_0828__primus/Primus/runner/primus-cli /home/lihuzhan/code/aiter-src/aiter /home/lihuzhan/_hfassets/llama31_8B \
         $B0E/configs/l8b_e2e.template.yaml $PT/output/0927__b0/gemm/nkfix_b0.py $PT/output/0927__b0/gemm/transpose_triton.py \
         /home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/job_context/op/refcache/fast.pt \
         /home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/job_context/op/refcache/prod.pt; do
  [ -e "$p" ] && ok "$p" || bad "missing $p"
done
V=$(grep -m1 -oE '__version__ *= *"[^"]+"' /home/lihuzhan/.local/flydsl0341/flydsl/__init__.py /home/lihuzhan/.local/flydsl0341/flydsl/_version.py 2>/dev/null | head -1)
echo "  info flydsl0341 $V"

echo "== adapter code (host python, no torch)"
python3 -m py_compile $KIT/attn_backends/e2e_attn/__init__.py $KIT/attn_backends/e2e_attn/arms.py \
  $KIT/attn_backends/e2e_attn/attn_timer.py $KIT/attn_backends/opcheck.py $KIT/attn_backends/shim/primus_turbo/__init__.py \
  $KIT/tools/*.py 2>&1 && ok "py_compile" || bad "py_compile"
find $KIT -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
python3 -B $KIT/tools/selftest_timer.py | grep -q SELFTEST_OK && ok "attn timer self-test" || bad "attn timer self-test"
find $KIT -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
bash -n $KIT/run_e2e_a0.sh && bash -n $KIT/drive.sh && bash -n $KIT/trees.sh && bash -n $KIT/analyze.sh && bash -n $KIT/tools/stub_launcher.sh \
  && ok "bash -n launcher/driver/trees/analyze/stub" || bad "bash -n"

echo "== schedules ($MODE)"
if [ "$MODE" = 3arm ]; then SP="$SPEC_3ARM_P1|$SPEC_3ARM_P2"; ST=$STEPS_3ARM; PF=$PFREQ_3ARM
else SP="$SPEC_2ARM_P1|$SPEC_2ARM_P2"; ST=$STEPS_2ARM; PF=$PFREQ_2ARM; fi
python3 - "$SP" "$ST" "$PF" <<'EOF' || FAIL=$((FAIL+1))
import json, os, sys
specs, steps, pf = sys.argv[1].split("|"), int(sys.argv[2]), int(sys.argv[3])
trees = json.loads(os.environ["E2E_FLY_TREES_JSON"])
bad = 0
for spec in specs:
    w, _, c = spec.rpartition(";")
    w = [x.strip() for x in w.split(",") if x.strip()]; c = [x.strip() for x in c.split(",") if x.strip()]
    arm = lambda s: w[s - 1] if s - 1 < len(w) else c[(s - 1 - len(w)) % len(c)]
    names = {t for x in w + c for t in x.split("/")}
    unknown = sorted(n for n in names if n != "asm" and n not in trees)
    excl = set(range(1, 8)) | {x for f in range(pf, steps + 2, pf) for x in (f - 1, f, f + 1)}
    win = [s for s in range(1, steps + 1) if s not in excl]
    per = {a: sum(1 for s in win if arm(s) == a) for a in dict.fromkeys(c)}
    prof = {a: [f for f in range(pf, steps + 1, pf) if arm(f) == a] for a in dict.fromkeys(c)}
    first_fly = next(s for s in range(1, steps + 1) if arm(s) != "asm")
    ok = arm(1) == "asm" and not unknown and min(per.values()) >= 12
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} {spec!r}: step1={arm(1)} first FlyDSL step={first_fly} unknown={unknown} "
          f"window steps/arm={per} profiled steps/arm={prof} first 8: {[arm(s) for s in range(1, 9)]}")
sys.exit(1 if bad else 0)
EOF

echo "== container import check (HIP_VISIBLE_DEVICES=-1: imports the adapter, builds E2EAttention, loads no arm)"
if [ "${E2E_LOCK_HELD:-0}" != 1 ] && [ -n "$(lock_holders /tmp/a0-gpu0.lock)" ]; then
  warn "container checks SKIPPED: /tmp/a0-gpu0.lock is held by another GPU client -- drive.sh re-runs them under its lock"
else
K0=$(kfd_holders)
SPEC1=$([ "$MODE" = 3arm ] && echo "$SPEC_3ARM_P1" || echo "$SPEC_2ARM_P1")
OUT=$(timeout 180 docker exec -e HIP_VISIBLE_DEVICES=-1 -e CUDA_VISIBLE_DEVICES=-1 -e E2E_ATTN="$SPEC1" \
  -e E2E_FLY_TREES="$E2E_FLY_TREES_JSON" -e E2E_EXPECT_BLAS_LIB=$BLAS_LIB -e PYTHONDONTWRITEBYTECODE=1 $CT bash -c \
  "export TORCH_BLAS_PREFER_HIPBLASLT=1 HIPBLASLT_TENSILE_LIBPATH=$BLAS_LIB PYTHONPATH=$KIT/attn_backends/shim:$KIT/attn_backends; \
   cd $KIT/tools && /opt/venv/bin/python3 import_check_1002.py" 2>&1)
echo "$OUT" | grep -q IMPORT_CHECK_OK && ok "$(echo "$OUT" | grep IMPORT_CHECK_OK | cut -c1-200)..." || bad "import check: $(echo "$OUT" | tail -5)"
OUT2=$(timeout 180 docker exec -e HIP_VISIBLE_DEVICES=-1 -e CUDA_VISIBLE_DEVICES=-1 -e PYTHONDONTWRITEBYTECODE=1 $CT bash -c \
  "cd $KIT/tools && /opt/venv/bin/python3 functest_adapter_cpu.py" 2>&1)
echo "$OUT2" | grep -q FUNCTEST_OK && ok "adapter functional test (fake trees, Function called without the autograd engine, no HIP init)" \
  || bad "adapter functional test: $(echo "$OUT2" | tail -3)"
K1=$(kfd_holders)
[ "$K0" = "$K1" ] && ok "KFD holders unchanged by the container checks" || warn "KFD holders changed during import check: [$K0] -> [$K1] (another client?)"
fi

echo "== reminder: commit + push output/1002__e2e before the card run (code + configs; KIT/logs is tracked -- KIT/.gitignore re-includes it -- only traces/ is ignored); no git operation while the e2e runs -- A0 host crashed again before the 10-01 20:41 reboot (BERT fatal L3)"
if [ $FAIL = 0 ]; then echo PREFLIGHT_OK; else echo "PREFLIGHT_FAIL $FAIL"; exit 93; fi
