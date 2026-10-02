# sourced by drive.sh / preflight.sh / run_e2e_a0.sh -- the arms of the 2026-10-02 A0 e2e, and the
# guards every launch goes through (one GPU client at a time, the pinned trees and nothing else).
#   fly    = fwd r16 (r13ns, e2e copy without the hipBLASLt re-point) + bwd s6 (re-pinned to flydsl 0.3.4.1)
#   flyr29 = fwd r16 (same copy)                                      + bwd r29 (0.3.4.1 pin, the B0 09-28 e2e arm)
#   asm    = aiter ASM fwd + hand-launched ASM bwd (+ host GQA sum)
KIT=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__e2e/e2e
B0E=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/e2e
FWD_TREE=$KIT/arms/fwd_r16_imglib
BWD_S6=$B0E/arms/bwd_s6_0341          # created 2026-10-02 08:01 by the s6 re-pin agent (source copy: output/1002__e2e/arms_src/bwd_s6_0341)
BWD_R29=$B0E/arms/bwd_r29_0341
E2E_FLY_TREES_JSON="{\"fly\":{\"fwd\":\"$FWD_TREE\",\"bwd\":\"$BWD_S6\"},\"flyr29\":{\"fwd\":\"$FWD_TREE\",\"bwd\":\"$BWD_R29\"}}"
export E2E_FLY_TREES_JSON
# expected md5 of the files that carry kernels (preflight refuses on a mismatch; PREFLIGHT_ALLOW_SRC_DIFF=1 overrides)
S6_KERNELS_MD5=5e61678d53260a2fc17c68286298f4b9       # output/0930__bwd/armsrc/s6/kernels.py (champion s6)
S6_IMPL_MD5=46e0e8127bd57c888944bbc6ee96249b          # e2e port of output/0930__bwd/armsrc/s6/impl.py (e8015b95) + two env
                                                      # switches, defaults = s6 (output/1002__e2e/patches/s6_impl_env_switch.diff):
                                                      #   FLY_BWD_SIDE_STREAM=0   dQ chain serial on the caller's stream (r29 order)
                                                      #   FLY_BWD_RECORD_STREAM=0 keep the side stream, skip record_stream() at the join
S6_SRC_MD5SUMS=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__e2e/arms_src/bwd_s6_0341/MD5SUMS   # the e2e copy must equal it
# fly arm stream mode for the e2e: side stream ON (as benchmarked), record_stream OFF. The join
# (caller stream waits for the side stream before returning; every block the side stream touches is
# referenced until then) already orders any reuse after the side stream's work; record_stream would
# only defer reuse of ~1.1 GiB per call while the host runs ahead (up to ~4.6 layers between nkfix
# syncs, i.e. up to ~5 GiB) against 2.2 GiB of memguard headroom at 32L. GPU work is identical.
FLY_BWD_ENV="-e FLY_BWD_SIDE_STREAM=${FLY_BWD_SIDE_STREAM:-1} -e FLY_BWD_RECORD_STREAM=${FLY_BWD_RECORD_STREAM:-0}"
R29_KERNELS_MD5=37f37052eb739579555d6bdf62a829cc      # job round 29 kernels.py (B0 09-28 e2e)
R29_IMPL_MD5=08cb8533d75e82198fabb9514fd18ba5
PIN0341_ENV_MD5=7df61bba26309ffc14b84a432fb451a7      # the 0.3.4.1 _env.py shared by bwd_r20/r29/s6 _0341 trees
# schedules: step 1 (and 2 in 3arm) is ASM -- no FlyDSL in step 1 (09-28: both FlyDSL-in-step-1 runs went bad)
SPEC_3ARM_P1="asm;asm,fly,flyr29,asm,flyr29,fly"
SPEC_3ARM_P2="asm;asm,flyr29,fly,asm,fly,flyr29"
STEPS_3ARM=92; PFREQ_3ARM=11          # 11 is coprime with the 6-step cycle: every arm gets profiled
SPEC_2ARM_P1="asm,fly,asm,fly;asm,fly,fly,asm"   # B0 final recipe, process a (ABBA)
SPEC_2ARM_P2="asm,fly,asm,fly;fly,asm,asm,fly"   # B0's process b (BAAB) with the warm-up made ASM-first
STEPS_2ARM=62; PFREQ_2ARM=10

# ------------------------------------------------------------------------------------------------ guards
# Shared by drive.sh (after every cool-down = right before every launch), run_e2e_a0.sh (right before its
# docker exec, and the foreign-KFD watch during the run) and preflight.sh. All read-only: ps, /proc, sysfs, md5.
KFD_PROC=/sys/class/kfd/kfd/proc      # KFD holders, named by HOST pid (= docker top's pid column)
NKFIX_PY=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/gemm/nkfix_b0.py
TRANSPOSE_PY=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/gemm/transpose_triton.py   # imported by nkfix
# KFD_DIR replaces KFD_PROC only in drive.sh's CPU dry run (stub launcher); run_e2e_a0.sh and preflight.sh reset it.
kfd_holders() { ls "${KFD_DIR:-$KFD_PROC}" 2>/dev/null | tr '\n' ' ' | sed 's/ $//'; }

# op-evolve run/resume loops. op-evolve does NOT take /tmp/a0-gpu0.lock, so a loop that is alive can put a
# benchmark on the card at any time. Matched on each process's OWN comm and argv -- comm "op-evolve" (the console
# script .venv/bin/op-evolve, whose python gets the script's name as comm) or a python -- whose script argument
# (interpreter options skipped) is .../op-evolve or "-m op_evolve[.cli]" and whose next word, the subcommand, is
# "run" or "resume". A shell or tool whose command line merely mentions op-evolve (this check's own shell, a
# `bash -c '... op-evolve resume ...'` wrapper, `tail -F LOG.fwd`) has comm bash/tail/... and never matches;
# "op-evolve status|stop|tune" (short-lived, no card) do not match either. No pgrep -f, no self-match.
oe_loops() {
  ps -eo pid=,comm=,args= 2>/dev/null | awk '
    $2 == "op-evolve" || $2 ~ /^python[0-9.]*$/ {
      i = 4; s = 0
      while (i <= NF) {
        if ($i == "-m") { if ($(i + 1) ~ /^op_evolve(\.cli)?$/) s = i + 2; break }
        if ($i == "-X" || $i == "-W") { i += 2; continue }
        if ($i ~ /^-/) { i++; continue }
        if ($i ~ /(^|\/)op-evolve$/) s = i + 1
        break
      }
      if (s && ($s == "run" || $s == "resume")) print $1
    }' | tr '\n' ' ' | sed 's/ $//'
}
# realab drivers (output/1002__e2e/tools/realab.sh, realab_run.sh): the other agent's op-level A/B. Each of its
# processes takes the lock, but the driver releases it between processes, so an e2e started while it is alive
# would interleave with it. Matched like oe_loops: comm, or argv[1] of a bash/sh (options skipped).
realab_drivers() {
  ps -eo pid=,comm=,args= 2>/dev/null | awk '
    $2 ~ /^realab(_run)?\.sh$/ { print $1; next }
    $2 ~ /^(ba)?sh$/ { for (i = 4; i <= NF; i++) { if ($i ~ /^-/) continue; if ($i ~ /(^|\/)realab(_run)?\.sh$/) print $1; break } }' \
    | tr '\n' ' ' | sed 's/ $//'
}
pid_desc() { local p; for p in "$@"; do printf '%s[%s] ' "$p" "$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-120)"; done; }

# Is KFD holder $1 a process of run $2? docker exec -e E2E_RUN_MARKER=<tag> puts the marker in the exec'd
# process's environ and torchrun / its workers / their subprocesses inherit it; a holder without it is a second
# GPU client (another exec into fa-repro -- realab, an op-evolve benchmark -- or a host process).
# -> ours | foreign | gone | unverified (environ unreadable but inside container $3: not called foreign)
kfd_owner() {
  local env
  [ -d /proc/$1 ] || { echo gone; return; }
  env=$(sudo -n cat /proc/$1/environ 2>/dev/null | tr '\0' '\n')
  if [ -n "$env" ]; then
    echo "$env" | grep -qx "E2E_RUN_MARKER=$2" && echo ours || echo foreign
    return
  fi
  [ -d /proc/$1 ] || { echo gone; return; }
  timeout 20 docker top "$3" -eo pid 2>/dev/null | awk 'NR > 1 {print $1}' | grep -qx "$1" && echo unverified || echo foreign
}

# Every file that runs in an e2e process (the three arm trees, the adapter, the ASM launcher, nkfix): drive.sh
# snapshots it at start (runs/tree.<stamp>.sha256) and compares before every launch; the launcher snapshots it
# per process (logs/tree.<tag>.sha256, + the config) and re-hashes after the run.
run_files() {
  find $KIT/attn_backends $B0E/arms/asm $FWD_TREE $BWD_S6 $BWD_R29 -name '*.py' -type f -not -path '*/__pycache__/*' 2>/dev/null | sort
  echo $NKFIX_PY; echo $TRANSPOSE_PY
}
run_files_sha() { run_files | xargs sha256sum 2>&1; }
md5is() { [ "$(md5sum < "$1" 2>/dev/null | cut -c1-32)" = "$2" ]; }
# The pinned code of the three trees (E2E-PLAN.md section 2). Prints what differs; nothing when all match.
trees_check() {
  local m=""
  (cd $FWD_TREE 2>/dev/null && md5sum -c --quiet MD5SUMS) >/dev/null 2>&1 || m="$m fwd_r16_imglib!=its-MD5SUMS"
  md5is $BWD_S6/kernels.py $S6_KERNELS_MD5 || m="$m s6/kernels.py!=${S6_KERNELS_MD5:0:8}"
  md5is $BWD_S6/impl.py $S6_IMPL_MD5 || m="$m s6/impl.py!=${S6_IMPL_MD5:0:8}"
  md5is $BWD_S6/_env.py $PIN0341_ENV_MD5 || m="$m s6/_env.py!=${PIN0341_ENV_MD5:0:8}"
  if [ -f $S6_SRC_MD5SUMS ]; then
    (cd $BWD_S6 2>/dev/null && awk '!/^#/ && $2 !~ /\// {print}' $S6_SRC_MD5SUMS | md5sum -c --quiet) >/dev/null 2>&1 || m="$m s6!=arms_src/MD5SUMS"
  else m="$m missing:$S6_SRC_MD5SUMS"; fi
  md5is $BWD_R29/kernels.py $R29_KERNELS_MD5 || m="$m r29/kernels.py!=${R29_KERNELS_MD5:0:8}"
  md5is $BWD_R29/impl.py $R29_IMPL_MD5 || m="$m r29/impl.py!=${R29_IMPL_MD5:0:8}"
  md5is $BWD_R29/_env.py $PIN0341_ENV_MD5 || m="$m r29/_env.py!=${PIN0341_ENV_MD5:0:8}"
  echo "${m# }"
}
