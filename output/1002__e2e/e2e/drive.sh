#!/bin/bash
# Driver for the 2026-10-02 A0 e2e (E2E-PLAN.md). Holds /tmp/a0-gpu0.lock for the WHOLE sequence (including
# cool-downs, so no other client slips in between processes), runs:
#   [lock] preflight (CPU) -> [idle] opcheck fast (fly, AMD_SERIALIZE_KERNEL=3) -> [idle] opcheck prod (fly, NOT
#   serialized: the prod kernels k_dkdv + k_dqg with the side stream really concurrent and record_stream off)
#   -> [idle] p1 -> [idle] p2 -> analysis (CPU)
# and stops at the first bad sign without starting another process (a startup is what wedges this card).
# [idle] = KFD empty, 300 s cool-down, then the guard: STOP sentinel, no op-evolve run/resume loop and no realab
# driver (neither takes the lock), KFD still empty, the three trees at their pinned md5, every file that runs
# unchanged since the driver started. The launcher repeats the guard right before its docker exec and, during
# the run, stops it on any KFD holder that is not the run's own process ("!! FOREIGN KFD", run INVALID).
# op-evolve must not be started or resumed before this driver prints E2E_DRIVE_DONE or E2E_DRIVE_STOP.
#
#   MODE=3arm (default: asm / fly=r16+s6 / flyr29=r16+r29, 92 steps, profile every 11)
#   MODE=2arm (B0 final recipe: asm / fly, 62 steps, profile every 10)
#   RUNS="opc opcprod p1 p2" (default)   other token: p3 (= p1's order again)
#   COOL=300 (s after KFD empty)  IDLE_MAX=1800 (s to wait for KFD empty)  E2E_NLAYERS=24 (fallback only)
#   touch $KIT/STOP    -> stop before the next launch (a running process is never killed); remove it to rerun
# Launch (from the host, detached):
#   setsid nohup bash .../1002__e2e/e2e/drive.sh > .../1002__e2e/e2e/runs/driver.$(date +%m%d_%H%M).log 2>&1 &
# CPU dry run (stub launcher, no docker, no card; private KFD dir / STOP file / lock, real GPU clients are only
# reported): LOCKFILE=/tmp/e2e_dry.lock LAUNCHER=$KIT/tools/stub_launcher.sh SKIP_PREFLIGHT=1 COOL=1 IDLE_MAX=5 bash drive.sh
# Exit codes: 0 done | 93 preflight / tree drift | 97 card not idle / op-evolve or realab running | 1 opcheck failed |
#   91 foreign KFD holder during a run (INVALID) | 90 STOP sentinel | 99 dmesg FAULT | 98 watchdog hang |
#   96 non-finite loss/grad/nkfix | 95 memguard | 94 BLAS re-point | 92 rc!=0 | 2 usage
set -u
. /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__e2e/e2e/trees.sh
MODE=${MODE:-3arm}; RUNS=${RUNS:-opc opcprod p1 p2}; COOL=${COOL:-300}; IDLE_MAX=${IDLE_MAX:-1800}
REAL_LAUNCHER=$KIT/run_e2e_a0.sh; STUB=$KIT/tools/stub_launcher.sh
LOCKFILE=${LOCKFILE:-/tmp/a0-gpu0.lock}; LAUNCHER=${LAUNCHER:-$REAL_LAUNCHER}   # overrides only for the CPU dry run
STAMP=$(date +%m%d_%H%M%S); TAGS=$KIT/runs/tags.$STAMP.txt
case $MODE in
  3arm) SP1=$SPEC_3ARM_P1; SP2=$SPEC_3ARM_P2; ST=${STEPS:-$STEPS_3ARM}; PF=${PFREQ:-$PFREQ_3ARM} ;;
  2arm) SP1=$SPEC_2ARM_P1; SP2=$SPEC_2ARM_P2; ST=${STEPS:-$STEPS_2ARM}; PF=${PFREQ:-$PFREQ_2ARM} ;;
  *) echo "MODE must be 3arm or 2arm"; exit 2 ;;
esac
for r in $RUNS; do case $r in opc|opcprod|p1|p2|p3) ;; *) echo "unknown run token '$r' in RUNS='$RUNS'"; exit 2 ;; esac; done
say() { echo "[$(date +%T)] $*"; }
stop() { say "E2E_DRIVE_STOP $1 $2"; exit "$1"; }

# The real card: KFD_PROC and $KIT/STOP, no override. CPU dry run (stub launcher only): a private empty KFD dir and
# STOP file, and only fake clients (cmdline containing "e2e_dry", see STUB_FAIL=oe) count -- real ones are reported.
DRY=0
if [ "$LAUNCHER" != "$REAL_LAUNCHER" ]; then
  [ "$LAUNCHER" = "$STUB" ] || { echo "LAUNCHER may only be overridden with $STUB (CPU dry run)"; exit 2; }
  [ "$LOCKFILE" != /tmp/a0-gpu0.lock ] || { echo "dry run: use LOCKFILE=/tmp/e2e_dry.lock, not the GPU lock"; exit 2; }
  DRY=1; KFD_DIR=$(mktemp -d /tmp/e2e_dry_kfd.XXXXXX); STOPFILE=${STOPFILE:-/tmp/e2e_dry_STOP}
  trap 'rm -rf "$KFD_DIR"' EXIT
else
  KFD_DIR=$KFD_PROC; STOPFILE=$KIT/STOP
fi
export KFD_DIR STOPFILE
dry_only() {   # dry run: only the fakes count as clients
  if [ $DRY = 0 ]; then echo "$*"; return; fi
  local p; for p in "$@"; do tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q e2e_dry && echo -n "$p "; done
}
[ $DRY = 1 ] && say "DRY RUN (stub launcher, no card): KFD dir $KFD_DIR, STOP file $STOPFILE, lock $LOCKFILE; real clients now (ignored): op-evolve [$(oe_loops)] realab [$(realab_drivers)] KFD [$(ls $KFD_PROC 2>/dev/null | tr '\n' ' ')]"

exec 9>$LOCKFILE
say "waiting for $LOCKFILE"
flock -w ${LOCK_WAIT:-3600} 9 || stop 97 "lock not free after ${LOCK_WAIT:-3600}s (another GPU client is running)"
export E2E_LOCK_HELD=1
say "lock held by driver pid $$ (mode $MODE runs '$RUNS' steps $ST pfreq $PF; s6 stream env: $FLY_BWD_ENV)"
if [ "${SKIP_PREFLIGHT:-0}" != 1 ]; then   # under the lock: no other client can start while its container python runs
  MODE=$MODE bash $KIT/preflight.sh > $KIT/runs/preflight.$STAMP.log 2>&1 9>&-
  tail -1 $KIT/runs/preflight.$STAMP.log | grep -q PREFLIGHT_OK || { grep -E "FAIL" $KIT/runs/preflight.$STAMP.log; stop 93 "preflight ($KIT/runs/preflight.$STAMP.log)"; }
  say "preflight ok ($KIT/runs/preflight.$STAMP.log)"
fi
# every file that runs, as of now; compared before every launch (here and in the launcher), per-run copies in logs/
E2E_TREE_BASE=$KIT/runs/tree.$STAMP.sha256; export E2E_TREE_BASE
run_files_sha > $E2E_TREE_BASE
grep -vq '^[0-9a-f]\{64\}  /' $E2E_TREE_BASE && stop 93 "cannot hash the run files: $(grep -v '^[0-9a-f]\{64\}  /' $E2E_TREE_BASE | head -2)"
D=$(trees_check); [ -n "$D" ] && [ "${PREFLIGHT_ALLOW_SRC_DIFF:-0}" != 1 ] && stop 93 "trees not at their pinned md5: $D"
say "run files: $(wc -l < $E2E_TREE_BASE) hashed -> $E2E_TREE_BASE; trees pinned (s6 impl ${S6_IMPL_MD5:0:8}, kernels ${S6_KERNELS_MD5:0:8}; r29 ${R29_KERNELS_MD5:0:8}/${R29_IMPL_MD5:0:8}; fwd MD5SUMS)"

clients_check() {   # $1 = what is next
  [ -e "$STOPFILE" ] && stop 90 "STOP sentinel $STOPFILE present -- not starting $1 (remove it to run again)"
  local x
  x=$(dry_only $(oe_loops)); [ -n "$x" ] && stop 97 "op-evolve run/resume loop running before $1: $(pid_desc $x)"
  x=$(dry_only $(realab_drivers)); [ -n "$x" ] && stop 97 "realab driver running before $1: $(pid_desc $x)"
  return 0
}
guard() {   # $1 = what is next; runs after the cool-down, i.e. right before the launch
  clients_check "$1"
  local x
  x=$(kfd_holders); [ -n "$x" ] && stop 97 "KFD busy after the cool-down, before $1: $(pid_desc $x)"
  if [ "${PREFLIGHT_ALLOW_SRC_DIFF:-0}" != 1 ]; then
    x=$(trees_check); [ -n "$x" ] && stop 93 "tree drift before $1: $x"
  fi
  x=$(run_files_sha | diff $E2E_TREE_BASE - | grep '^[<>]' | awk '{print $NF}' | sort -u | head -6 | tr '\n' ' ')
  [ -n "$x" ] && stop 93 "run files changed since the driver started, before $1: $x"
  return 0
}
idle() {   # $1 = what is next. KFD empty (every holder must be nameable -- we name none, so none may exist),
           # cool down (STOP checked every second), then the guard once more
  local t0=$(date +%s) end
  clients_check "$1"
  until [ -z "$(kfd_holders)" ]; do
    [ $(( $(date +%s) - t0 )) -gt $IDLE_MAX ] && stop 97 "KFD not empty after ${IDLE_MAX}s: $(pid_desc $(kfd_holders))"
    sleep 5
  done
  end=$(( $(date +%s) + COOL ))
  while [ $(date +%s) -lt $end ]; do
    [ -e "$STOPFILE" ] && stop 90 "STOP sentinel $STOPFILE present -- not starting $1 (remove it to run again)"
    sleep 1
  done
  guard "$1"
  say "idle before $1: KFD empty, cooled ${COOL}s, no op-evolve/realab, trees unchanged, sclk $(grep '\*' /sys/class/drm/card1/device/pp_dpm_sclk | tr -s ' ')"
}
refused() {   # $1 = out file, $2 = rc, $3 = what: the launcher's own pre-exec guard said no
  grep -q "^!! refusing to start" $1 && stop $([ $2 = 93 ] && echo 93 || echo 97) "launcher refused $3: $(grep -m1 '^!! refusing to start' $1 | cut -c1-240)"
  return 0
}

opcheck() {   # $1 shape. fast: AMD_SERIALIZE_KERNEL=3 (a fault is attributable to its kernel; s6's split path
              # k_dq_sp+k_redsp_q / k_dkdv_sp+k_redsp). prod: NOT serialized -- the training's kernels k_dkdv (main
              # stream) + k_dqg (side stream) really concurrent, FLY_BWD_RECORD_STREAM=0 reuse, E2EAttention + autograd
  idle "opcheck $1"
  local out=$KIT/runs/opc.$1.$STAMP.out ser=""
  [ "$1" = fast ] && ser="-e AMD_SERIALIZE_KERNEL=3"
  E2E_FLYCACHE=/tmp/flycache_opc_$1_$(date +%H%M%S) OPCHECK_ENV="$ser $FLY_BWD_ENV" \
    bash $LAUNCHER opcheck fly $1 > $out 2>&1 9>&-
  local rc=$?
  say "opcheck fly $1 rc=$rc ($out)"; tail -3 $out
  [ $rc = 99 ] && stop 99 "dmesg FAULT after opcheck $1"
  { [ $rc = 91 ] || grep -q "!! FOREIGN KFD" $out; } && stop 91 "FOREIGN KFD holder during opcheck $1 (INVALID): $(grep -m1 '!! FOREIGN KFD' $out | cut -c1-200)"
  refused $out $rc "opcheck $1"
  [ $rc = 0 ] || stop 1 "opcheck $1 rc=$rc"
  local js=$(grep -oE 'json=\S+' $out | cut -d= -f2)
  python3 - "$js" "$1" <<'EOF' || stop 1 "opcheck $1 gate"
# gate: every output finite and >= 47 dB against the fp32 reference (refcache, the same one the ASM arm is scored
# against: ASM at prod 50.6-53.2 dB, r29 50.8-52.8 dB, B0 09-28), no BLAS re-point, attention timer working
import json, sys
r, shape = json.load(open(sys.argv[1])), sys.argv[2]
vals = dict(("chain." + k, v) for k, v in r["sqnr_chain_db"].items())
vals.update(("kbwd." + k, v) for k, v in r.get("sqnr_kbwd_refolse_db", {}).items())
vals.update(("kfwd." + k, v) for k, v in r.get("sqnr_kfwd_db", {}).items() if k != "lse_shape")
bad = [f"{k}={v}" for k, v in vals.items() if not isinstance(v, (int, float)) or v < 47.0]
if r.get("arm") != "fly" or r.get("shape") != shape:
    bad.append(f"json is arm {r.get('arm')} shape {r.get('shape')}, expected fly {shape}")
if not r.get("sqnr_kbwd_refolse_db"):
    bad.append("no kernel-bwd SQNR")
if r.get("blas_repoints"):
    bad.append(f"BLAS re-point {r['blas_repoints']}")
if not (r.get("attn_timer") or {}).get("ok"):
    bad.append(f"attn timer {r.get('attn_timer')}")
t = r.get("timing_ms") or {}
tm = " ".join(f"{k} {t[k]['med']}" for k in ("k_fwd", "k_bwd", "m_fwd", "m_bwd") if isinstance(t.get(k), dict))
print(f"  opcheck {shape} gate:", "PASS" if not bad else "FAIL " + "; ".join(bad), json.dumps(vals),
      "copies", r.get("adapter_copies"), f"| ms {tm}")
sys.exit(1 if bad else 0)
EOF
}

train() {   # $1 tag-suffix $2 spec
  local tag=a0e2e_${MODE}_$1_$STAMP
  idle "train $tag"
  local out=$KIT/runs/e2e.$tag.out
  printf '%s\n%s\n%s\n' "$2" "$ST" "$PF" > $KIT/runs/$tag.spec
  echo "$tag" >> $TAGS
  E2E_FLYCACHE=/tmp/flycache_${tag}_$(date +%H%M%S) E2E_NKFIX=1 E2E_MEM_STOP=${E2E_MEM_STOP:-89.5} NKFIX_CHECK=${NKFIX_CHECK:-1} E2E_ENV="$FLY_BWD_ENV" \
    bash $LAUNCHER train $tag "$2" $ST $PF > $out 2>&1 9>&-
  local rc=$?
  say "train $tag rc=$rc ($out)"; tail -6 $out
  local wd=$(grep -h "WATCHDOG fired" $KIT/logs/e2e.$tag.log.watchdog 2>/dev/null | head -1)
  [ $rc = 99 ] || grep -q "!! NEW dmesg FAULT" $out && stop 99 "dmesg FAULT after $tag"
  [ -s $KIT/logs/e2e.$tag.log.foreign ] || [ $rc = 91 ] && stop 91 "FOREIGN KFD holder during $tag (INVALID): $(grep -m1 'FOREIGN KFD' $KIT/logs/e2e.$tag.log.foreign 2>/dev/null | cut -c1-200)"
  refused $out $rc "$tag"
  grep -q "!! TREE CHANGED DURING RUN" $out && stop 93 "run files changed during $tag (INVALID): $(grep -m1 '!! TREE CHANGED' $out | cut -c1-200)"
  case "$wd" in *nonfinite*) stop 96 "$wd" ;; *blas*) stop 94 "$wd" ;; *hang*) stop 98 "$wd" ;; esac
  [ -s $KIT/logs/e2e.$tag.log.memguard ] && stop 95 "memguard: $(head -1 $KIT/logs/e2e.$tag.log.memguard)"
  grep -qE "DISCARD|non-finite" $out && stop 96 "non-finite steps in $tag"
  grep -qE "^nkfix_events: *\[\(" $KIT/runs/$tag.post.txt && stop 96 "nkfix non-finite events in $tag"
  [ $rc = 0 ] || stop 92 "rc=$rc for $tag"
  return 0
}

for r in $RUNS; do
  case $r in
    opc) opcheck fast ;;
    opcprod) opcheck prod ;;
    p1) train p1 "$SP1" ;;
    p2) train p2 "$SP2" ;;
    p3) train p3 "$SP1" ;;
    *) stop 2 "unknown run token $r" ;;
  esac
done
flock -u 9; exec 9>&-
say "card work done; lock released"
[ -s $TAGS ] && bash $KIT/analyze.sh $TAGS > $KIT/runs/analysis.$STAMP.txt 2>&1 && say "analysis -> $KIT/runs/analysis.$STAMP.txt, table -> $KIT/runs/TABLE.$STAMP.md"
say E2E_DRIVE_DONE
