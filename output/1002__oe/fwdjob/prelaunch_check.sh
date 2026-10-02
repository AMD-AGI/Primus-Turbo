#!/bin/bash
# prelaunch_check.sh -- read-only pre-launch checks for resuming the fwd op-evolve job on A0.
#
#   bash prelaunch_check.sh            all checks (touches the container with `docker exec fa-repro` only to read)
#   bash prelaunch_check.sh --no-card  skip every check that needs docker / dmesg / GPU sysfs (CPU-only subset)
#
# Changes nothing: no swap, no cache clear, no lock, no launch. Exit 0 only when every check passes; WARN lines do
# not fail it. See PT/output/1002__oe/FWDJOB.md for what to do about each FAIL.
set -u
OE=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve
JN=gfx1250-flydsl-attn-fwd-b0-20260927
J=$OE/artifacts/$JN
F=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__oe/fwdjob
CARD=1; [ "${1:-}" = --no-card ] && CARD=0
fails=0
ok()   { echo "PASS  $*"; }
bad()  { echo "FAIL  $*"; fails=$((fails + 1)); }
warn() { echo "WARN  $*"; }

# 1. no op-evolve loop anywhere (process image, not command-line text), no live job .pid
loops=$(ps -eo pid=,args= | awk '{ for (i = 2; i <= 3; i++) if ($i ~ /(^|\/)op-evolve$/ && ($(i+1) == "run" || $(i+1) == "resume") && (i == 2 || $2 ~ /python[0-9.]*$/)) { print; break } }')
[ -z "$loops" ] && ok "no op-evolve loop running" || bad "op-evolve loop(s) running: $(echo "$loops" | cut -c1-160 | tr '\n' ';') -- stop: op-evolve stop --job <job>"
for p in "$OE"/artifacts/*/.pid; do
  [ -f "$p" ] || continue; pid=$(tr -dc 0-9 < "$p")
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && bad "live .pid $p ($pid)"
done
# 2. no e2e / hand-campaign GPU client, nobody holds the hand-campaign lock
other=$(ps -eo pid=,args= | grep -E 'e2e/drive\.sh|run_e2e\.sh|torchrun|primus/cli/main\.py|1002__e2e/tools/realab|0930__bwd/tools/(run|arm|abl)\.sh' | grep -v -e grep -e prelaunch_check)
[ -z "$other" ] && ok "no e2e / realab / hand-campaign process" || bad "other GPU clients: $(echo "$other" | cut -c1-160 | tr '\n' ';')"
holders=$(fuser /tmp/a0-gpu0.lock 2>/dev/null | tr -s ' ')
[ -z "$holders" ] && ok "/tmp/a0-gpu0.lock not held" || bad "/tmp/a0-gpu0.lock held by PID(s)$holders"
k=$(ls /sys/class/kfd/kfd/proc 2>/dev/null | tr '\n' ' ')
[ -z "$k" ] && ok "KFD process list empty" || bad "KFD processes on the card: $k"

# 3. job directory state
S=$J/job_context/state.yaml
grep -q '^  spec_version: v000' $S && ok "spec_version v000" || bad "spec_version is not v000 (setup would re-run)"
[ -e $J/.stop ] && warn "$J/.stop exists (run_loop removes it on resume)" || ok "no .stop"
awk '/^- round: 20$/{f=1} f && /module: act/{m=1} f && /step: 01_implement/{s=1} END{exit !(m && s)}' $S \
  && ok "round 20 (deep) open at act/01_implement -> resume redoes the act" || warn "round 20 is not where FWDJOB.md expects it (open at act/01_implement)"
grep -A40 '^- round: 20$' $S | grep -q '^  sessions:' && bad "round 20 still carries B0-only agent sessions" || ok "round 20 has no B0 agent sessions"
check_md5() { [ "$(md5sum < "$J/job_context/$1" | cut -c1-32)" = "$2" ] && ok "md5 $1" || bad "md5 $1 differs from the 2026-10-02 edit ($2)"; }
check_md5 gfx1250-flydsl-attn-fwd_final.yaml 39e3c33cbe48304a217af0c24a8065db
check_md5 hint.md d464f93ac00c2c3bf2ba7b292f8519b7
check_md5 op/eager/impl.py e85c795dcf9c9e932fce2934ac8fbc39
check_md5 op/refcache_util.py 66f9d39864f627a0a1dc892df3309911
check_md5 op/current/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py 370769c94c0f951655f66e10c39947d8
check_md5 ../rounds/019/op/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py e4485ba08d49bdbaef36987c94ae1c1f
grep -q 'container: fa-repro' $J/job_context/gfx1250-flydsl-attn-fwd_final.yaml && ok "final.yaml container fa-repro" || bad "final.yaml container is not fa-repro"

# 4. hints, refcache, no fp32 reference on the card (CPU-only tests)
(cd $OE && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c "
from op_evolve.core import hints
h = {x.id: x for x in hints.parse(open('artifacts/$JN/job_context/hint.md').read())}
p = next((x for x in h.values() if x.kind == 'refactor' and x.outstanding), None)
assert p is not None and p.id == 'h48' and 'A0 amendment' in p.body, 'pending refactor is not the amended h48'
assert h['h49'].standing and not h['h29'].standing and not h['h38'].standing, 'h49/h29/h38 types'
") >/dev/null 2>&1 && ok "hint.md: h48 pending (amended), h49 standing, h29/h38 retired" || bad "hint.md parse check"
python3 -B $F/tools/refcache_prov.py $J/job_context/op $J/job_context/op/eager/impl.py.bak.pre-a0-1002 | grep -q 'RESULT: ALL MATCH' \
  && ok "refcache provenance matches (pre-guard eager sha)" || bad "refcache provenance mismatch"
python3 -B $F/tools/test_refcache_util.py $J/job_context/op | grep -q 'RESULT: PASS' \
  && ok "refcache_util/eager guard test (CPU fallbacks, proxy/prod refusals)" || bad "test_refcache_util.py failed"

# 5. OE deep prompts must be the fwd variant before a fwd deep round
st=$(bash $F/swap_deep_prompts.sh status 2>/dev/null | head -1 | sed 's/.*deep prompts = //')
[ "$st" = fwd ] && ok "OE deep prompts = fwd variant" || bad "OE deep prompts = '$st' -- run: bash $F/swap_deep_prompts.sh fwd"

# 6. card checks (read-only)
if [ $CARD = 1 ]; then
  timeout 20 docker exec fa-repro true && ok "docker exec fa-repro responsive" || bad "docker exec fa-repro did not answer in 20 s"
  docker exec fa-repro printenv FLYDSL_RUNTIME_CACHE_DIR OE_PHYS_GPU >/dev/null 2>&1 && warn "fa-repro sets FLYDSL_RUNTIME_CACHE_DIR/OE_PHYS_GPU now (FWDJOB.md assumes neither)"
  c=$(timeout 20 docker exec fa-repro bash -c 'du -s /root/.flydsl/cache 2>/dev/null | cut -f1; ls -d /tmp/flycache 2>/dev/null' | tr '\n' ' ')
  case "$c" in ""|"0 ") ok "FlyDSL cache in fa-repro empty" ;; *) warn "FlyDSL cache in fa-repro not empty ($c KB / dirs) -- move it aside before launch (FWDJOB.md step 3)" ;; esac
  [ "$(sysctl -n kernel.dmesg_restrict 2>/dev/null)" = 0 ] && ok "kernel.dmesg_restrict 0" || warn "kernel.dmesg_restrict is not 0 (agents' dmesg checks read empty): sudo sysctl -w kernel.dmesg_restrict=0"
  s=$(cat /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | tr -s ' \n' ' ')
  echo "INFO  pp_dpm_sclk: $s  (post-09-29 A0 table: 500 / 2355 / 2400 MHz)"
  echo "INFO  last amdgpu dmesg lines:"; sudo -n dmesg 2>/dev/null | grep -iE 'amdgpu|MES|GCVM' | tail -3 | sed 's/^/      /'
else
  echo "INFO  --no-card: docker / dmesg / sysfs checks skipped"
fi
echo "RESULT: $([ $fails = 0 ] && echo "READY (resume with the command in FWDJOB.md)" || echo "NOT READY ($fails FAIL)")"
exit $([ $fails = 0 ] && echo 0 || echo 1)
