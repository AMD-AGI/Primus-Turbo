#!/bin/bash
# swap_deep_prompts.sh -- switch op-evolve's deep-round prompts between the fwd and the bwd job variant.
#
#   bash swap_deep_prompts.sh status        which variant the OE working tree holds now (read-only)
#   bash swap_deep_prompts.sh fwd           install the fwd variant (before resuming gfx1250-flydsl-attn-fwd-b0-20260927)
#   bash swap_deep_prompts.sh bwd           install the bwd variant (before resuming gfx1250-flydsl-attn-bwd-20260917-115934)
#   bash swap_deep_prompts.sh head          back to OE HEAD (the committed prompts; bwd-specific 09-25 corrections)
#
# Why: the deep profiling/plan/act prompts do not read the job's hint.md, so each job's corrections are inlined in
# three _preamble.md files, plus job-specific lines in profiling/prompts/0[1-4]_*.md. The OE prompts are shared by
# every job, so they must hold the variant of the job that runs next. Seven files, all under
# op_evolve/modules/deep_loop/, change; nothing else in OE is touched.
#   bwd variant = PT/output/0930__bwd/oejob/oe_uncommitted_0930.diff (= deep_att_enable.patch), a diff against HEAD
#   fwd variant = PT/output/1002__oe/fwdjob/oe_deep_fwd_1002.diff, a diff against HEAD (built by
#                 oe_prompts/build_fwd_variant.py)
# The swap reverse-applies the installed variant and applies the wanted one, with `git apply` (never checkout or
# reset). It refuses while an op-evolve loop runs (a deep step renders its prompt when it starts, so a swap under a
# running loop changes what the next step reads), when the seven files hold anything other than HEAD / bwd / fwd,
# when they are staged, or when a patch file differs from the md5 recorded here. The pre-swap state of the seven
# files is saved as a tar under PT/output/1002__oe/fwdjob/oe_prompt_backups/ first. CPU only; no GPU, no job dir.
#
# Env: OE=<op-evolve checkout> (default the real one; set it to a scratch clone to test), FORCE=1 skips the
# running-loop refusal (do not use it while a loop runs a deep round).
set -u
OE=${OE:-/home/lihuzhan/code/2026_0910__op-evolve/op-evolve}
PT=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo
BWD_DIFF=$PT/output/0930__bwd/oejob/oe_uncommitted_0930.diff
FWD_DIFF=$PT/output/1002__oe/fwdjob/oe_deep_fwd_1002.diff
BWD_MD5=d55b862d51fff39889f16ec5e676983f
FWD_MD5=4b787f48f0133a634dbc3d00fd4ec175
BK=${BK:-$PT/output/1002__oe/fwdjob/oe_prompt_backups}
D=op_evolve/modules/deep_loop
FILES="$D/act/prompts/_preamble.md $D/plan/prompts/_preamble.md $D/profiling/prompts/01_select.md
$D/profiling/prompts/02_counters.md $D/profiling/prompts/03_metrics.md $D/profiling/prompts/04_thread_trace.md
$D/profiling/prompts/_preamble.md"

say() { echo "[swap_deep_prompts] $*"; }
die() { say "REFUSED: $1"; exit "${2:-1}"; }

patch_of() { case $1 in bwd) echo "$BWD_DIFF" ;; fwd) echo "$FWD_DIFF" ;; esac; }

state() {   # prints head | bwd | fwd | staged | unknown
  git -C "$OE" diff --cached --quiet -- $FILES || { echo staged; return; }
  if git -C "$OE" diff --quiet -- $FILES; then echo head; return; fi
  git -C "$OE" apply --reverse --check "$BWD_DIFF" 2>/dev/null && { echo bwd; return; }
  git -C "$OE" apply --reverse --check "$FWD_DIFF" 2>/dev/null && { echo fwd; return; }
  echo unknown
}

show() {
  local f
  for f in act plan profiling; do
    printf '  %-9s %s\n' "$f" "$(grep -m1 -o 'CAMPAIGN CORRECTIONS[^—]*' "$OE/$D/$f/prompts/_preamble.md" 2>/dev/null)"
  done
}

want=${1:-status}
case $want in status|fwd|bwd|head) ;; *) die "usage: $0 status|fwd|bwd|head" 2 ;; esac
[ -d "$OE/.git" ] || die "$OE is not a git checkout" 2
[ "$(md5sum < "$BWD_DIFF" | cut -c1-32)" = $BWD_MD5 ] || die "$BWD_DIFF changed (md5 != $BWD_MD5)" 2
[ "$(md5sum < "$FWD_DIFF" | cut -c1-32)" = $FWD_MD5 ] || die "$FWD_DIFF changed (md5 != $FWD_MD5)" 2

cur=$(state)
say "OE $OE: deep prompts = $cur"
show
[ "$want" = status ] && exit 0
case $cur in staged|unknown) die "the seven prompt files are '$cur' -- inspect 'git -C $OE diff -- $D' by hand" 3 ;; esac
if [ "$cur" = "$want" ]; then say "already $want; nothing to do"; exit 0; fi

loops() {  # live op-evolve loops: by process image (python .../op-evolve run|resume) and by each job's .pid file.
  # Not `pgrep -f`: it also matches any shell whose command line merely contains the words "op-evolve resume".
  ps -eo pid=,args= | awk '{ for (i = 2; i <= 3; i++) if ($i ~ /(^|\/)op-evolve$/ && ($(i+1) == "run" || $(i+1) == "resume") \
                              && (i == 2 || $2 ~ /python[0-9.]*$/)) { print; break } }'
  local f p
  for f in "$OE"/artifacts/*/.pid; do
    [ -f "$f" ] || continue
    p=$(tr -dc 0-9 < "$f")
    [ -n "$p" ] && kill -0 "$p" 2>/dev/null && grep -qa op-evolve /proc/$p/cmdline 2>/dev/null && echo "$p (from $f)"
  done
}
loop=$(loops | sort -u | tr '\n' ';')
if [ -n "$loop" ] && [ "${FORCE:-0}" != 1 ]; then
  die "an op-evolve loop is running: $loop -- stop it (op-evolve stop --job <job>) or wait for it to end" 4
fi

mkdir -p "$BK" || die "cannot create $BK" 5
tarf=$BK/oe_deep_prompts.$(date -u +%Y%m%dT%H%M%SZ).was-$cur.tar
(cd "$OE" && tar -cf "$tarf" $FILES) || die "backup tar failed" 5
say "backup of the current files: $tarf"

if [ "$cur" != head ]; then
  git -C "$OE" apply --reverse "$(patch_of $cur)" || die "reverse-apply of the $cur variant failed (tree unchanged)" 6
fi
if [ "$want" != head ]; then
  if ! git -C "$OE" apply "$(patch_of $want)"; then
    say "apply of the $want variant failed; restoring the $cur variant"
    [ "$cur" != head ] && git -C "$OE" apply "$(patch_of $cur)"
    die "swap aborted; tree is back to '$(state)'" 6
  fi
fi
now=$(state)
[ "$now" = "$want" ] || die "after the swap the state reads '$now', not '$want' (backup: $tarf)" 7
say "done: $cur -> $now"
show
git -C "$OE" diff --stat -- $D | tail -1
