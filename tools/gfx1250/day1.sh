#!/bin/bash
# Unattended day-one run. Executes the ordered plan and stops at the first gate that fails.
#
#   tools/gfx1250/day1.sh [GPU_ID]
#
# Everything here has an accept/reject rule that is arithmetic, not judgement. Results go to
# $OUT/day1.jsonl; a human reads the summary at the end. Nothing is committed automatically.
set -u
GPU=${1:-0}
REPO=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo
OUT=${OUT:-$REPO/output/0913__opt_plan__claude/phase2/day1}
LEDGER=$OUT/day1.jsonl
H=tools/gfx1250/tune_attention.py
CHAMP_MS=24.435          # today's in-tree champion, VR-throttled card
NOISE_PCT=2              # measured run-to-run spread is 0.87%; 2% is the accept threshold
mkdir -p "$OUT"; touch "$LEDGER"; cd "$REPO"
export PYTHONPATH=/home/lihuzhan/code/aiter-src

say(){ printf "\n=== %s ===\n" "$1"; }
rec(){ echo "$1" >> "$LEDGER"; }
run(){ # tag, args... -> echoes total_ms or empty
  local tag="$1"; shift
  local o; o=$(GPU=$GPU timeout 1200 python3 $H "$@" 2>/dev/null | tail -1)
  case "$o" in '{'*) rec "{\"tag\":\"$tag\",\"r\":$o}"; python3 -c "
import json,sys
r=json.loads('''$o''')
print('%.3f' % r['total_ms'] if r.get('correct') else 'WRONG')" 2>/dev/null ;;
    *) rec "{\"tag\":\"$tag\",\"r\":null}"; echo "" ;; esac
}

# ---- GATE 0: the node must be fit. Everything downstream is meaningless otherwise. ----
say "GATE 0  node bring-up"
bash tools/gfx1250/bringup.sh || { echo "NODE NOT READY -- stopping."; exit 1; }

# ---- STEP 1: re-baseline. Today's numbers were VR-throttled; every target shifts. ----
say "STEP 1  re-baseline the champion on GPU $GPU"
base=$(run "base|champ" --shape llama31-8b --impl fused --tune "fwd:num_stages=2")
[ -z "$base" ] && { echo "baseline FAILED -- stopping."; exit 1; }
echo "  champion here: $base ms   (was $CHAMP_MS ms on the throttled card)"
echo "  ALL comparisons below use THIS number, not today's."

# ---- STEP 2: E4, num_warps on the fused kernel. Zero code change, highest untested EV. ----
say "STEP 2  E4  num_warps x num_stages on the FUSED backward"
best=$base; bestcfg="shipped"
for W in 2 4 8 16; do for S in 1 2; do
  t=$(PRIMUS_TURBO_FUSED_MHA_BWD_TUNE="num_warps=$W,num_stages=$S" \
      run "E4|w$W|s$S" --shape llama31-8b --impl fused --tune "fwd:num_stages=2")
  printf "  num_warps=%-3s num_stages=%s -> %s ms\n" "$W" "$S" "${t:-FAILED}"
  [ -n "$t" ] && [ "$t" != WRONG ] && awk -v a="$t" -v b="$best" 'BEGIN{exit !(a<b)}' && { best=$t; bestcfg="num_warps=$W,num_stages=$S"; }
done; done
echo "  BEST: $bestcfg at $best ms"
awk -v b="$best" -v z="$base" -v n="$NOISE_PCT" 'BEGIN{
  g=(z-b)/z*100; printf "  verdict: %.2f%% vs baseline -- %s\n", g, (g>n?"ACCEPT":"reject (within noise)")}'

# ---- STEP 3: prepared kernel patches, one at a time, each against the CURRENT best. ----
for E in E1-hoist-Di E3-fold-log2-scale-dq; do
  say "STEP 3  $E"
  P=$OUT/../prepared-experiments/$E.patch
  [ -f "$P" ] || { echo "  patch missing, skipping"; continue; }
  # No git stash here. The kernel is clean at this point (a rejected patch is reverted
  # below, an accepted one stays), so stash/pop would either be a no-op or would fight the
  # checkout used to revert. Verified offline: both prepared patches apply, parse and revert
  # cleanly in sequence.
  patch -p0 -s < "$P" || { echo "  patch did not apply -- skipping"; continue; }
  python3 -c "import ast;ast.parse(open('primus_turbo/triton/attention/fused_mha_bwd_kernel.py').read())" \
    || { echo "  patch produced unparseable python -- reverting"; git checkout -- primus_turbo/triton/attention/fused_mha_bwd_kernel.py; continue; }
  t=$(PRIMUS_TURBO_FUSED_MHA_BWD_TUNE="$([ "$bestcfg" = shipped ] && echo "" || echo "$bestcfg")" \
      run "$E" --shape llama31-8b --impl fused --tune "fwd:num_stages=2")
  echo "  $E -> ${t:-FAILED} ms   (best so far $best)"
  if [ -n "$t" ] && [ "$t" != WRONG ] && awk -v a="$t" -v b="$best" -v n="$NOISE_PCT" 'BEGIN{exit !((b-a)/b*100>n)}'; then
    echo "  ACCEPT -- keeping $E"; best=$t
  else
    echo "  reject -- reverting"; git checkout -- primus_turbo/triton/attention/fused_mha_bwd_kernel.py
  fi
done

# ---- STEP 4: confirm the winner generalises and is deterministic. ----
say "STEP 4  generalisation + determinism"
for S in llama31-8b-b2 llama31-8b-s4096 gate-s1024 gate-s16384; do
  t=$(run "gen|$S" --shape $S --impl fused --tune "fwd:num_stages=2")
  printf "  %-20s %s ms\n" "$S" "${t:-FAILED}"
done
run "det" --shape llama31-8b --impl fused --tune "fwd:num_stages=2" --determinism-reps 300 >/dev/null
python3 -c "
import json
for l in open('$LEDGER'):
    d=json.loads(l)
    if d['tag']=='det' and d.get('r'):
        x=d['r'].get('determinism',{})
        print('  determinism: %d reps, mismatches %s, non-finite %d' % (x.get('reps',0), x.get('bitwise_mismatch_reps'), x.get('nonfinite_reps',0)))
" 2>/dev/null

say "DONE"
echo "ledger: $LEDGER"
echo "baseline $base ms -> best $best ms"
awk -v a="$best" -v b="$base" 'BEGIN{printf "overall: %.3fx\n", b/a}'
echo
echo "NOT run automatically (they need judgement, see PLAN-4GPU-TOMORROW.md):"
echo "  - torch.compile GEMM probe  (decides whether end-to-end is valid at all)"
echo "  - aiter prebuilt gfx1250 ASM backward probe"
echo "  - E2 tl.trans elimination   (do the 15-min ISA check first)"
