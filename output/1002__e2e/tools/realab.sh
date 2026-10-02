#!/bin/bash
# realab.sh -- run tools/realab.py on the card: op-level A/B of fwd r16 and bwd s6 / r29 (the flydsl 0.3.4.1 e2e
# trees) vs aiter ASM on the 6 REAL training q/k/v dumps + randn, ONE card process per condition (default: blk,
# then gb). Every process goes through tools/realab_run.sh (copy of output/0930__bwd/tools/run.sh):
# /tmp/a0-gpu0.lock, KFD-empty check, fresh FlyDSL JIT cache dir, image hipBLASLt env assigned in the exec, host
# sclk sampler, dmesg delta check + classification.
#
#   bash tools/realab.sh                   blk + gb, arm order asm first (rounds are palindromic inside a process)
#   RA_CONDS=gb bash tools/realab.sh       one condition only
#   RA_ROT=1   ...                         plus one process per condition with the arm order reversed (as many as
#                                          fit the card budget)
#   RA_AA=0    ...                         no A/A copies (fly_aa = copy of the fwd tree, s6_aa = copy of s6)
#   RA_KINETO=0 ...                        no kineto pass (per-kernel attribution, summarised with profile/tools/opana.py)
#   RA_FWD=/abs RA_S6=/abs RA_R29=/abs     other trees (every FlyDSL tree must pin flydsl 0.3.4.1; kernel md5s are
#                                          checked against the champions, RA_ALLOW_MD5=1 overrides)
#   FLY_BWD_SIDE_STREAM=0 / FLY_BWD_RECORD_STREAM=0   forwarded to the process (bwd_s6_0341 impl switches, read
#                                          at import); RA_* timing knobs of realab.py are forwarded too
#   touch output/1002__e2e/STOP            stop before the next process (a running one is never killed)
#   RA_DRYRUN=1 ...                        preflight only (md5, ISA evidence, busy check, A/A copies), no card
#
# Card budget: < 15 min of card time in total. A process is started only if (card time so far + its hard timeout
# RA_TMO, default 360 s) <= RA_BUDGET_S (default 870 s). Expected ~2.5-3 min per process (load + JIT ~1.5 min).
# The driver stops at the first non-zero rc. rc: 9 new GPU lines in dmesg (STOP ALL CARD WORK; the line says
# whether it is the unrecoverable class), 7 KFD holders, 6 lock timeout, 8 python rc != 0 (realab.py: 3 = an arm
# failed the correctness pass, 4 = gb ruler void), 5 another GPU client (op-evolve loop / e2e training) is running,
# 2 preflight refused (tree md5 / ISA evidence), 1 other.
# Outputs: runs/realab_<cond>_<o1|o2>_<stamp>.{log,json,clk,env,dmesg.bad,opana.txt}, traces/<same>.trace.json,
#          runs/realab_<stamp>.driver.txt (this script's output), runs/realab_<stamp>.summary.txt (--sum of all runs).
set -u
E=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__e2e
T=$E/tools
B=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0
OPANA=$B/profile/tools/opana.py
LOCK=/tmp/a0-gpu0.lock
CONDS=${RA_CONDS:-blk gb}
STAMP=${RA_STAMP:-$(date +%m%d_%H%M%S)}
TMO=${RA_TMO:-360}
BUDGET=${RA_BUDGET_S:-870}
COOL=${RA_COOL:-30}
AA=${RA_AA:-1}
ROT=${RA_ROT:-0}
KIN=${RA_KINETO:-1}
RUNNER=${RA_RUNNER:-$T/realab_run.sh}   # test hook only: a stand-in runner with the same interface
mkdir -p $E/runs $E/traces
exec > >(tee -a $E/runs/realab_$STAMP.driver.txt) 2>&1

say() { echo "[$(date +%T)] $*"; }
die() { say "REALAB_ABORT: $1"; exit "${2:-1}"; }
md5() { md5sum "$1" 2>/dev/null | cut -c1-32; }
expect_md5() {  # file expected label
  local got; got=$(md5 "$1")
  [ "$got" = "$2" ] && return 0
  say "md5 MISMATCH $3: $1 = ${got:-missing}, expected $2"
  [ "${RA_ALLOW_MD5:-0}" = 1 ] || die "tree differs from the champion (RA_ALLOW_MD5=1 to measure it anyway)" 2
}
kfd() { ls /sys/class/kfd/kfd/proc 2>/dev/null | tr '\n' ' '; }
busy_check() {   # other GPU clients that do not take $LOCK
  local p
  p=$(pgrep -af "op-evolve (resume|start|run)|op_evolve.*(resume|start)" 2>/dev/null | grep -v pgrep)
  [ -n "$p" ] && { say "an op-evolve loop is running (stop it first: op-evolve stop --job ...): $p"; return 1; }
  p=$(timeout 20 docker top fa-repro -eo pid,args 2>/dev/null | grep -E "primus/cli/main.py|torchrun" | grep -v grep)
  [ -n "$p" ] && { say "e2e training is running in fa-repro: $p"; return 1; }
  return 0
}
wait_idle() {    # KFD empty and the lock free, up to 10 min
  local i
  for i in $(seq 1 120); do
    if [ -z "$(kfd)" ] && flock -n $LOCK true; then return 0; fi
    [ $i = 1 ] && say "waiting for the card (KFD: [$(kfd)], lock $LOCK)"
    sleep 5
  done
  return 1
}
mkaa() {         # src dst: fresh byte copy of the .py files of a tree, verified
  rm -rf "$2" && mkdir -p "$2" || return 1
  (cd "$1" && find . -name '*.py' -not -path '*/__pycache__/*' -not -path './.*' | cpio -pdm --quiet "$2") || return 1
  [ "$(cd "$1" && find . -name '*.py' -not -path '*/__pycache__/*' -not -path './.*' | sort | xargs md5sum)" = \
    "$(cd "$2" && find . -name '*.py' | sort | xargs md5sum)" ]
}
rev() { echo "$1" | tr ',' '\n' | tac | paste -sd, -; }

# ------------------------------------------------------------------ preflight (host only, no card)
say "realab $STAMP: conds=[$CONDS] rot=$ROT aa=$AA kineto=$KIN timeout=${TMO}s budget=${BUDGET}s cool=${COOL}s"
[ -e $E/STOP ] && die "STOP sentinel $E/STOP present"
for c in 672 673 674 680 688 703; do
  [ -s /home/lihuzhan/_prof_dump/qkv_call0$c.pt ] || die "missing real dump /home/lihuzhan/_prof_dump/qkv_call0$c.pt"
done
FWD=${RA_FWD:-$B/e2e/arms/fwd_r16}
R29=${RA_R29:-$B/e2e/arms/bwd_r29_0341}
S6=${RA_S6:-}
if [ -z "$S6" ]; then
  for c in $B/e2e/arms/bwd_s6_0341 $E/arms/bwd_s6_0341 $E/arms_src/bwd_s6_0341; do
    [ -f $c/impl.py ] && { S6=$c; break; }
  done
fi
[ -n "$S6" ] && [ -f "$S6/impl.py" ] || die "no s6_0341 tree (looked in 0927__b0/e2e/arms, 1002__e2e/arms, 1002__e2e/arms_src; set RA_S6)"
expect_md5 $FWD/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py 370769c94c0f951655f66e10c39947d8 "fwd r16 m32x8"
expect_md5 $FWD/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x2.py b09721562d696d5c1edc7eba29857a88 "fwd r16 m32x2"
expect_md5 $FWD/impl.py 35f4247ad75b0907b83d467b3f8bf33f "fwd r16 impl"
expect_md5 $R29/kernels.py 37f37052eb739579555d6bdf62a829cc "r29 kernels"
expect_md5 $R29/impl.py 08cb8533d75e82198fabb9514fd18ba5 "r29 impl"
expect_md5 $S6/kernels.py 5e61678d53260a2fc17c68286298f4b9 "s6 kernels (0930__bwd/armsrc/s6)"
for t in $FWD $R29 $S6; do grep -q "flydsl0341" $t/_env.py || die "$t/_env.py does not pin flydsl 0.3.4.1" 2; done
say "trees: fwd=$FWD  r29=$R29  s6=$S6 (impl md5 $(md5 $S6/impl.py | cut -c1-8), _env $(md5 $S6/_env.py | cut -c1-8))"
if [ -f $E/arms_src/bwd_s6_0341/impl.py ] && [ "$(readlink -f $S6)" != "$(readlink -f $E/arms_src/bwd_s6_0341)" ]; then
  for f in impl.py kernels.py _env.py; do
    [ "$(md5 $S6/$f)" = "$(md5 $E/arms_src/bwd_s6_0341/$f)" ] || die "$S6/$f differs from arms_src/bwd_s6_0341/$f -- refusing to time a drifted s6"
  done
fi
# s6 under 0.3.4.1 must be proven to run the same code objects as the card-proven 0.3.2 build (compile-only dumps)
ISA_OK=1
for k in delta dkdv dqg; do
  a=$(ls $E/isa/s6_032/$k/*/2[12]_final_isa.s 2>/dev/null | head -1); b=$(ls $E/isa/s6_0341/$k/*/2[12]_final_isa.s 2>/dev/null | head -1)
  if [ -z "$a" ] || [ -z "$b" ]; then say "ISA evidence missing for s6 $k (isa/s6_032, isa/s6_0341)"; ISA_OK=0; continue; fi
  n=$(diff <(grep -v '^\s*\.\(file\|ident\)' $a) <(grep -v '^\s*\.\(file\|ident\)' $b) | grep -c '^[<>]')
  say "ISA s6 $k flydsl 0.3.2 vs 0.3.4.1: $n differing lines"
  [ "$n" = 0 ] || ISA_OK=0
done
[ $ISA_OK = 1 ] || [ "${RA_ALLOW_ISA:-0}" = 1 ] || die "s6_0341 ISA not proven identical to s6 0.3.2 (RA_ALLOW_ISA=1 overrides)" 2
busy_check || die "another GPU client is active" 5
FWD_AA=""; S6_AA=""
if [ "$AA" = 1 ]; then
  FWD_AA=$E/arms/aa/fwd_r16_aa; S6_AA=$E/arms/aa/bwd_s6_0341_aa
  mkaa $FWD $FWD_AA || die "A/A copy $FWD -> $FWD_AA failed"
  mkaa $S6 $S6_AA || die "A/A copy $S6 -> $S6_AA failed"
  say "A/A copies: $FWD_AA $S6_AA (byte-identical .py)"
fi
O1F="asm,fly"; O1B="asm,s6,r29"
[ "$AA" = 1 ] && { O1F="$O1F,fly_aa"; O1B="$O1B,s6_aa"; }
PLAN=""
for c in $CONDS; do PLAN="$PLAN $c:o1"; done
[ "$ROT" = 1 ] && for c in $CONDS; do PLAN="$PLAN $c:o2"; done
say "plan:$PLAN  (o1 fwd $O1F bwd $O1B; o2 reversed)"
write_env() {   # tag cond fwd_order bwd_order -> the KEY=VALUE lines of realab_run.sh --env-file
  echo "RA_COND=$2"
  echo "RA_JSON=$E/runs/$1.json"
  echo "RA_TRACE=$E/traces/$1.trace.json"
  echo "RA_KINETO=$KIN"
  echo "RA_DEADLINE_S=$(( TMO > 150 ? TMO - 75 : TMO / 2 ))"   # soft: no new set / no kineto after this
  echo "RA_FWD=$FWD"; echo "RA_S6=$S6"; echo "RA_R29=$R29"
  [ -n "$FWD_AA" ] && echo "RA_FWD_AA=$FWD_AA"
  [ -n "$S6_AA" ] && echo "RA_S6_AA=$S6_AA"
  echo "RA_ORDER_FWD=$3"; echo "RA_ORDER_BWD=$4"
  local v
  for v in RA_WARM RA_BLOCK RA_LEAD RA_BLK_ITERS_FWD RA_BLK_ITERS_BWD RA_GB_N RA_GB_ROUNDS RA_GB_NG RA_GB_MAXMS \
           RA_KINETO_REPS RA_SETS RA_DIRS RA_DO_SCALE RA_SQNR_WARN RA_FWD_ARMS RA_BWD_ARMS RA_SCLK_FILE \
           FLY_BWD_SIDE_STREAM FLY_BWD_RECORD_STREAM; do
    [ -n "${!v:-}" ] && echo "$v=${!v}"
  done
  return 0
}
if [ "${RA_DRYRUN:-0}" = 1 ]; then
  say "RA_DRYRUN=1: preflight passed, no card process started. env file of the first process:"
  c0=${PLAN# }; c0=${c0%%:*}
  write_env realab_${c0}_o1_$STAMP $c0 "$O1F" "$O1B" | sed 's/^/    /'
  exit 0
fi

# ------------------------------------------------------------------ card processes
SPENT=0; RC_ALL=0; JSONS=""
for item in $PLAN; do
  c=${item%%:*}; o=${item##*:}
  if [ $o = o1 ]; then OF=$O1F; OB=$O1B; else OF=$(rev $O1F); OB=$(rev $O1B); fi
  tag=realab_${c}_${o}_$STAMP
  [ -e $E/STOP ] && { say "STOP sentinel -- not starting $tag"; break; }
  if [ $((SPENT + TMO)) -gt $BUDGET ]; then say "BUDGET: not starting $tag (card ${SPENT}s + timeout ${TMO}s > ${BUDGET}s)"; break; fi
  busy_check || { RC_ALL=5; break; }
  wait_idle || { say "card not idle within 10 min -- not starting $tag"; RC_ALL=7; break; }
  say "cool-down ${COOL}s (KFD empty, lock free) before $tag"
  sleep $COOL
  wait_idle || { RC_ALL=7; break; }
  busy_check || { RC_ALL=5; break; }
  EF=$E/runs/$tag.env
  write_env $tag $c "$OF" "$OB" > $EF
  say "start $tag (fwd $OF | bwd $OB)"
  t0=$(date +%s)
  ENV_FILE=$EF $RUNNER $tag $TMO $T -- "/opt/venv/bin/python3 $T/realab.py"
  rc=$?
  t1=$(date +%s); SPENT=$((SPENT + t1 - t0))
  say "end $tag rc=$rc card $((t1 - t0))s (total ${SPENT}s of ${BUDGET}s)"
  timeout 60 docker exec fa-repro chown "$(id -u):$(id -g)" $E/runs/$tag.json $E/traces/$tag.trace.json 2>/dev/null
  grep -E "^(CORR (FAIL|WARN)|EXCLUDED|RULER VOID)|BURST|DEADLINE|KINETO|REALAB_DONE|PHASE|OP_STRING|Traceback|Error" $E/runs/$tag.log | sed 's/^/    /'
  sed -n '/^== SUMMARY/,/^== END SUMMARY/p' $E/runs/$tag.log
  [ -s $E/runs/$tag.json ] && JSONS="$JSONS $E/runs/$tag.json"
  if [ "$KIN" = 1 ] && [ -s $E/traces/$tag.trace.json ]; then
    if timeout 900 python3 $OPANA $E/traces/$tag.trace.json $E/runs/$tag.clk > $E/runs/$tag.opana.txt 2>&1; then
      say "opana -> $E/runs/$tag.opana.txt ($(wc -l < $E/runs/$tag.opana.txt) lines)"
    else
      say "opana failed, see $E/runs/$tag.opana.txt"
    fi
  fi
  if [ $rc != 0 ]; then
    RC_ALL=$rc
    case $rc in
      9) say "!! NEW GPU dmesg lines after $tag -- STOP ALL CARD WORK (classification above; skill gfx1250-card-safety 2a)" ;;
      8) say "!! realab.py $(grep -E '^rc=' $E/runs/$tag.log | tail -1) -- see $E/runs/$tag.log"; tail -25 $E/runs/$tag.log ;;
      7) say "!! KFD holders appeared before $tag" ;;
      6) say "!! lock $LOCK not free for 15 min" ;;
      *) say "!! realab_run.sh rc=$rc" ;;
    esac
    break
  fi
done
if [ -n "$JSONS" ]; then
  python3 $T/realab.py --sum $JSONS > $E/runs/realab_$STAMP.summary.txt 2>&1
  cat $E/runs/realab_$STAMP.summary.txt
fi
say "REALAB_DRIVER_DONE rc=$RC_ALL card_time=${SPENT}s runs:$JSONS"
exit $RC_ALL
