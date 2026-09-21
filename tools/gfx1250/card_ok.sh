#!/bin/bash
# Is this card able to do work RIGHT NOW? Three direct signals, no lying instruments.
#
#   card_ok.sh [--quiet]     exit 0 usable | 1 busy/degraded | 2 wedged, needs a human
#
# WHY NOT supervise_job.sh's gpu_ok(). That gate is `timeout 90 rocminfo` plus a grep for
# the arch. The failure this box actually produces most often is "the card is alive but a
# hung D-state process owns it" -- and in that state rocminfo still lists the agent and
# gpu_health.sh still prints HEALTHY. The gate passes, MAX_RESTARTS=60 fires a new loop
# into a card that cannot take one, sixty times, each one paying for an agent.
#
# WHY NOT rocm-smi. Two reasons, both measured here. It HANGS on a wedged card, along with
# ps -o wchan, pgrep and torch.cuda.device_count(). And its busy percentage lies: on 0917
# it read "GPU use (%): 13" with zero KFD holders and zero VRAM in use.
#
# So: KFD holders from /sys, VRAM from /sys, and one real (tiny) matmul in its own process
# under a hard timeout. A plain file read cannot block on the driver, and the matmul is the
# only thing that actually answers "can this card execute".
set -u

QUIET=0; [ "${1:-}" = "--quiet" ] && QUIET=1
say(){ [ "$QUIET" = 1 ] || echo "$*"; }

CONTAINER=${CARD_OK_CONTAINER:-fa-repro}
VRAM_MAX_PCT=${CARD_OK_VRAM_MAX_PCT:-80}
MATMUL_TIMEOUT=${CARD_OK_MATMUL_TIMEOUT:-60}

# --- 1. unrecoverable kernel signatures ------------------------------------------------
# Three tiers, and the distinction is load-bearing. `MES(...) ring buffer is full` is
# routine backpressure -- matching a bare `MES(` reports a healthy card as faulted, which
# has cost a day of a working card here. Only these three mean the driver's own reset
# never completed, and that needs a person at the machine.
HARD=$(timeout -k 5 25 sudo -n dmesg 2>/dev/null \
       | grep -cE 'wait for reset ack|ring gfx timeout|GPU reset begin' || true)
HARD=$(echo "$HARD" | head -1)
if [ "${HARD:-0}" -gt 0 ]; then
  say "WEDGED: $HARD unrecoverable signature(s) in dmesg. Stop every GPU loop and ask for an AC cycle."
  exit 2
fi

# --- 2. who holds the card -------------------------------------------------------------
# A directory listing. It cannot block on the driver, unlike every tool that asks the
# driver. Every holder must be nameable; an unnameable one invalidates the measurement.
HOLDERS=$(ls /sys/class/kfd/kfd/proc/ 2>/dev/null | tr '\n' ' ')
NH=$(echo "$HOLDERS" | wc -w)
if [ "$NH" -gt 0 ]; then
  names=""
  for p in $HOLDERS; do
    n=$(tr -d '\0' < "/proc/$p/comm" 2>/dev/null || echo "<gone>")
    names="$names $p:$n"
  done
  say "BUSY: $NH KFD holder(s):$names"
  exit 1
fi

# --- 3. how much memory is already gone ------------------------------------------------
# The SIGBUS that cost an AC cycle came from +0.32% on a run already sitting at 87.98%,
# and it leaked the KFD context so a container restart could not get it back -- with dmesg
# completely clean throughout. There is no margin to spend on optimism here.
U=$(cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | head -1)
T=$(cat /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null | head -1)
if [ -n "${U:-}" ] && [ -n "${T:-}" ] && [ "${T:-0}" -gt 0 ]; then
  PCT=$(( U * 100 / T ))
  if [ "$PCT" -ge "$VRAM_MAX_PCT" ]; then
    say "BUSY: VRAM ${PCT}% >= ${VRAM_MAX_PCT}%"
    exit 1
  fi
else
  PCT="?"
fi

# --- 4. the only signal that answers the question --------------------------------------
if ! timeout 20 docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true; then
  say "BUSY: container $CONTAINER is not running (card may be fine; we cannot use it)"
  exit 1
fi

# Toy shape on purpose. This is a liveness probe, not a benchmark -- a 4096^3 GEMM would
# make every health check a measurable load on a card we are trying to keep free.
OUT=$(timeout -k 10 "$MATMUL_TIMEOUT" docker exec \
        -e TORCH_BLAS_PREFER_HIPBLASLT=0 -e AMD_SERIALIZE_KERNEL=3 "$CONTAINER" \
        python3 -c '
import torch, time
t0 = time.time()
a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
c = (a @ a).float()
torch.cuda.synchronize()
assert torch.isfinite(c).all(), "matmul produced non-finite values"
print("MATMUL_OK %.3f" % (time.time() - t0))
' 2>&1 | tail -3)
RC=$?
case "$OUT" in
  *MATMUL_OK*) ;;
  *) say "DEGRADED: the card did not complete a toy matmul (rc=$RC): $(echo "$OUT" | tr '\n' ' ')"; exit 1 ;;
esac

SCLK=$(awk '/\*/{print $2}' /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | head -1)
say "OK: holders=0 vram=${PCT}% sclk=${SCLK:-?} $(echo "$OUT" | grep -o 'MATMUL_OK.*')"
exit 0
