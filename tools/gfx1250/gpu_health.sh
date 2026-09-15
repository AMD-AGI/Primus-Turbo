#!/bin/bash
# The single definition of "is this card healthy", and the only probes that are safe to ask.
#
# Four copies of these patterns exist in this tree and they disagree. work_queue.sh:12,
# forever_queue.sh:49 and bringup.sh:31 still grep for a bare `MES\(`, which matches routine
# "MES(0,0) ring buffer is full" backpressure -- a healthy card halts the queue. That pattern
# was retracted; queue.sh:23-24 and watchdog.sh:40-45 carry the corrected set. This file is
# meant to replace all four.
#
# WHAT MAY NOT BE USED HERE. On a wedged card rocm-smi, `ps ... wchan`, pgrep, docker exec
# and torch.cuda.device_count() all hang, and `timeout` does not rescue `sudo rocm-smi` --
# sudo's parent is killed while the rocm-smi child keeps holding the pipe, so the command
# substitution never returns. Everything below is either a bounded dmesg read or a sysfs
# read, neither of which can block on the driver.
#
# Usage:  gpu_health.sh [--json]     -> prints HEALTHY | DEGRADED | WEDGED, exit 0|1|2
set -u

# Tier 1: the card still completes kernels. Not a reason to stop.
DEGRADED_RE='failed to respond to msg|GPU Hang|Memory access fault'
# Tier 2: the driver's own reset never completes. A human and an AC cycle are required.
WEDGED_RE='wait for reset ack|ring gfx timeout|GPU reset begin'

# dmesg must be read over a WIDE window: apparmor spam pushes real faults out of `tail -80`
# and reports a wedged card clean. 4000 lines is ~15x the spam burst seen on 0913.
DMESG_LINES=${DMESG_LINES:-4000}

_dmesg() { timeout 15 dmesg 2>/dev/null | tail -n "$DMESG_LINES"; }

degraded_count() { _dmesg | grep -cE "$DEGRADED_RE"; }
wedged_count()   { _dmesg | grep -cE "$WEDGED_RE"; }

# Number of GPUs, read from sysfs rather than rocm-smi. Pure read, cannot block on the driver.
gpu_count() {
  grep -l 'gfx_target_version' /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null \
    | xargs -r grep -l 'simd_count [1-9]' 2>/dev/null | wc -l
}

# Processes currently holding the KFD. `ls` of a sysfs directory, not a process-table walk.
kfd_holders() { ls /sys/class/kfd/kfd/proc/ 2>/dev/null | wc -l; }

# The clock the card is actually pinned to. This is a CORRECTNESS probe as much as a health
# one: a silent un-throttle makes every number in the session incomparable with every other.
sclk() { timeout 15 cat /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | awk '/\*/{gsub(/[^0-9]/,"",$2); print $2; exit}'; }

# Process liveness. Shell BUILTIN kill -0 against a pid file -- never `timeout 5 kill -0`,
# which runs /bin/kill with different semantics and has false-reported a live process gone,
# and never pgrep, which hangs alongside everything else.
pid_alive() { local f=$1; [ -f "$f" ] && kill -0 "$(cat "$f" 2>/dev/null)" 2>/dev/null; }

health_state() {
  local base_d=${1:-0} base_w=${2:-0}
  local d w
  d=$(degraded_count); w=$(wedged_count)
  if [ "$w" -gt "$base_w" ]; then echo WEDGED; return 2; fi
  if [ "$d" -gt "$base_d" ]; then echo DEGRADED; return 1; fi
  echo HEALTHY; return 0
}

# A dmesg grep is necessary and NOT sufficient, and 0915 is the demonstration: after an e2e
# run hung, three independent jobs failed to complete while every fault pattern above read
# zero and rocm-smi answered promptly with VRAM free. The card was not wedged and not
# faulting -- torch.cuda.init() had gone from about 1 s to 45.6 s, so everything downstream
# blew its timeout. "dmesg is clean" and "the card computes" are different questions.
#
# compute_probe STAGES the answer, because the two failure modes need opposite responses:
#   init completes but slowly -> DEGRADED: widen timeouts and keep working
#   init never returns        -> WEDGED:   stop dispatching, report, switch to CPU work
#
# On WEDGED, do NOT reload the amdgpu module. On 2026-09-15 that was tried on a card that
# could create a context but not execute work, and the machine became unreachable -- SSH
# included -- until a human AC-cycled it. PROGRESS.md already records that the driver's own
# reset never completes on a wedged card. The downside of a driver-level action here is not
# "no improvement", it is "the whole box is gone".
# A single end-to-end matmul probe cannot tell them apart; it just times out either way.
compute_probe() {
  local budget=${1:-600} ctr=${CTR:-fa-repro}
  timeout $((budget + 30)) docker exec -e HIP_VISIBLE_DEVICES=0 "$ctr" bash -c \
    "exec timeout -k 10 $budget python3 -u -c '
import time,sys
t=time.time()
import torch
print(\"import %.1f\"%(time.time()-t),flush=True)
t=time.time(); torch.cuda.init(); print(\"init %.1f\"%(time.time()-t),flush=True)
t=time.time(); a=torch.randn(4096,4096,device=\"cuda\",dtype=torch.bfloat16); torch.cuda.synchronize()
print(\"alloc %.1f\"%(time.time()-t),flush=True)
t=time.time(); b=a@a; torch.cuda.synchronize(); print(\"matmul %.2f\"%(time.time()-t),flush=True)
'" 2>/dev/null | grep -E '^(import|init|alloc|matmul) '
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  if [ "${1:-}" = "--probe" ]; then compute_probe "${2:-600}"; exit $?; fi
  st=$(health_state "${BASE_DEGRADED:-0}" "${BASE_WEDGED:-0}"); rc=$?
  if [ "${1:-}" = "--json" ]; then
    printf '{"state":"%s","degraded":%s,"wedged":%s,"gpus":%s,"kfd_holders":%s,"sclk_mhz":%s}\n' \
      "$st" "$(degraded_count)" "$(wedged_count)" "$(gpu_count)" "$(kfd_holders)" "$(sclk)"
  else
    echo "$st  degraded=$(degraded_count) wedged=$(wedged_count) gpus=$(gpu_count) kfd_holders=$(kfd_holders) sclk=$(sclk)MHz"
  fi
  exit $rc
fi
