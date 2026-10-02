#!/bin/bash
# Driver for the 2026-10-02 A0 e2e (E2E-PLAN.md). Holds /tmp/a0-gpu0.lock for the WHOLE sequence (including
# cool-downs, so no other client slips in between processes), runs:
#   preflight (CPU) -> [idle] opcheck fast (fly, AMD_SERIALIZE_KERNEL=3) -> [idle] p1 -> [idle] p2 -> analysis (CPU)
# and stops at the first bad sign without starting another process (a startup is what wedges this card).
#
#   MODE=3arm (default: asm / fly=r16+s6 / flyr29=r16+r29, 92 steps, profile every 11)
#   MODE=2arm (B0 final recipe: asm / fly, 62 steps, profile every 10)
#   RUNS="opc p1 p2" (default)   other tokens: opcprod (opcheck fly prod), p3 (= p1's order again)
#   COOL=300 (s after KFD empty)  IDLE_MAX=1800 (s to wait for KFD empty)  E2E_NLAYERS=24 (fallback only)
# Launch (from the host, detached):
#   setsid nohup bash .../1002__e2e/e2e/drive.sh > .../1002__e2e/e2e/runs/driver.$(date +%m%d_%H%M).log 2>&1 &
# Exit codes: 0 done | 93 preflight | 97 card not idle / op-evolve running | 1 opcheck failed |
#   99 dmesg FAULT | 98 watchdog hang | 96 non-finite loss/grad/nkfix | 95 memguard | 94 BLAS re-point | 92 rc!=0
set -u
. /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__e2e/e2e/trees.sh
MODE=${MODE:-3arm}; RUNS=${RUNS:-opc p1 p2}; COOL=${COOL:-300}; IDLE_MAX=${IDLE_MAX:-1800}
LOCKFILE=${LOCKFILE:-/tmp/a0-gpu0.lock}; LAUNCHER=${LAUNCHER:-$KIT/run_e2e_a0.sh}   # overrides only for the CPU dry run
STAMP=$(date +%m%d_%H%M%S); TAGS=$KIT/runs/tags.$STAMP.txt
case $MODE in
  3arm) SP1=$SPEC_3ARM_P1; SP2=$SPEC_3ARM_P2; ST=${STEPS:-$STEPS_3ARM}; PF=${PFREQ:-$PFREQ_3ARM} ;;
  2arm) SP1=$SPEC_2ARM_P1; SP2=$SPEC_2ARM_P2; ST=${STEPS:-$STEPS_2ARM}; PF=${PFREQ:-$PFREQ_2ARM} ;;
  *) echo "MODE must be 3arm or 2arm"; exit 2 ;;
esac
say() { echo "[$(date +%T)] $*"; }
stop() { say "E2E_DRIVE_STOP $1 $2"; exit "$1"; }

if [ "${SKIP_PREFLIGHT:-0}" != 1 ]; then
  MODE=$MODE bash $KIT/preflight.sh > $KIT/runs/preflight.$STAMP.log 2>&1
  tail -1 $KIT/runs/preflight.$STAMP.log | grep -q PREFLIGHT_OK || { grep -E "FAIL" $KIT/runs/preflight.$STAMP.log; stop 93 "preflight ($KIT/runs/preflight.$STAMP.log)"; }
  say "preflight ok ($KIT/runs/preflight.$STAMP.log)"
fi

exec 9>$LOCKFILE
say "waiting for $LOCKFILE"
flock -w ${LOCK_WAIT:-3600} 9 || stop 97 "lock not free after ${LOCK_WAIT:-3600}s (another GPU client is running)"
export E2E_LOCK_HELD=1
say "lock held by driver pid $$ (mode $MODE runs '$RUNS' steps $ST pfreq $PF; s6 stream env: $FLY_BWD_ENV)"

idle() {   # KFD empty (every holder must be nameable -- we name none, so none may exist), no op-evolve, then cool down
  local t0=$(date +%s)
  until [ -z "$(ls /sys/class/kfd/kfd/proc 2>/dev/null)" ]; do
    [ $(( $(date +%s) - t0 )) -gt $IDLE_MAX ] && stop 97 "KFD not empty after ${IDLE_MAX}s: $(ls /sys/class/kfd/kfd/proc | tr '\n' ' ')"
    sleep 5
  done
  OE=$(ps -eo pid,comm,args | awk '$2 != "bash" && $2 != "awk" && /op-evolve/ && /(resume|run)/ {print $1}' | tr '\n' ' ')
  [ -n "$OE" ] && stop 97 "op-evolve loop running ($OE)"
  sleep $COOL
  [ -z "$(ls /sys/class/kfd/kfd/proc 2>/dev/null)" ] || stop 97 "KFD busy after cool-down: $(ls /sys/class/kfd/kfd/proc | tr '\n' ' ')"
  say "idle: KFD empty, cooled ${COOL}s, sclk $(grep '\*' /sys/class/drm/card1/device/pp_dpm_sclk | tr -s ' ')"
}

opcheck() {   # $1 shape
  idle
  local out=$KIT/runs/opc.$1.$STAMP.out
  E2E_FLYCACHE=/tmp/flycache_opc_$1_$(date +%H%M%S) OPCHECK_ENV="-e AMD_SERIALIZE_KERNEL=3 $FLY_BWD_ENV" \
    bash $LAUNCHER opcheck fly $1 > $out 2>&1
  local rc=$?
  say "opcheck fly $1 rc=$rc ($out)"; tail -3 $out
  [ $rc = 99 ] && stop 99 "dmesg FAULT after opcheck"
  [ $rc = 0 ] || stop 1 "opcheck rc=$rc"
  local js=$(grep -oE 'json=\S+' $out | cut -d= -f2)
  python3 - "$js" <<'EOF' || stop 1 "opcheck gate"
import json, sys
r = json.load(open(sys.argv[1]))
vals = dict(("chain." + k, v) for k, v in r["sqnr_chain_db"].items())
vals.update(("kbwd." + k, v) for k, v in r.get("sqnr_kbwd_refolse_db", {}).items())
bad = [f"{k}={v}" for k, v in vals.items() if not isinstance(v, (int, float)) or v < 47.0]
if r.get("blas_repoints"):
    bad.append(f"BLAS re-point {r['blas_repoints']}")
if not (r.get("attn_timer") or {}).get("ok"):
    bad.append(f"attn timer {r.get('attn_timer')}")
print("  opcheck gate:", "PASS" if not bad else "FAIL " + "; ".join(bad), json.dumps(vals), "copies", r.get("adapter_copies"))
sys.exit(1 if bad else 0)
EOF
}

train() {   # $1 tag-suffix $2 spec
  idle
  local tag=a0e2e_${MODE}_$1_$STAMP out=$KIT/runs/e2e.a0e2e_${MODE}_$1_$STAMP.out
  printf '%s\n%s\n%s\n' "$2" "$ST" "$PF" > $KIT/runs/$tag.spec
  echo "$tag" >> $TAGS
  E2E_FLYCACHE=/tmp/flycache_${tag}_$(date +%H%M%S) E2E_NKFIX=1 E2E_MEM_STOP=${E2E_MEM_STOP:-89.5} NKFIX_CHECK=${NKFIX_CHECK:-1} E2E_ENV="$FLY_BWD_ENV" \
    bash $LAUNCHER train $tag "$2" $ST $PF > $out 2>&1
  local rc=$?
  say "train $tag rc=$rc ($out)"; tail -6 $out
  local wd=$(grep -h "WATCHDOG fired" $KIT/logs/e2e.$tag.log.watchdog 2>/dev/null | head -1)
  [ $rc = 99 ] || grep -q "!! NEW dmesg FAULT" $out && stop 99 "dmesg FAULT after $tag"
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
