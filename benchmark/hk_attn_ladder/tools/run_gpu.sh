#!/bin/bash
# run_gpu.sh <tag> <timeout_s> -- <command...>
# The only way the ladder runs anything on a GPU. Env HKL_GPU = gpu index (container lhz-hkl-g$HKL_GPU, started by
# start_gpu_container.sh). Rules (learned the hard way on a08-1, see the README "GPU safety"):
#   * one job per GPU at a time (flock), never two processes on the card; refuse when the card already has
#     VRAM allocated (someone else's job: other containers may map every GPU);
#   * refuse while logs/GPU<n>.FAULT exists (only a person removes it, after reading the kernel log);
#   * refuse when the kernel log already holds amdgpu errors for this card since boot (a degraded card);
#   * hard timeout; afterwards any NEW amdgpu error line for this card (hang, page fault, ring / MES timeout,
#     uncorrectable or UMC ECC, reset) writes the FAULT file and fails the job. Correctable PCIe ACA lines are
#     not faults (a08-1 rule) but are kept in the job's .dmesg file.
# Output: logs/<UTC date>_<tag>.log (command output) and .dmesg (kernel lines of the job window).
set -uo pipefail
[ $# -ge 4 ] && [ "$3" = "--" ] || { sed -n '2,12p' "$0"; exit 2; }
TAG=$1; TMO=$2; shift 3
HERE=$(readlink -f "$(dirname "$0")/..")
GPU=${HKL_GPU:?set HKL_GPU to the gpu index}
CT=lhz-hkl-g$GPU
LOGS=$HERE/logs; mkdir -p "$LOGS"
FAULT=$LOGS/GPU$GPU.FAULT
PCI=$(cat "/tmp/$CT.pci" 2>/dev/null) || { echo "no /tmp/$CT.pci: run tools/start_gpu_container.sh $GPU" >&2; exit 3; }
[ -e "$FAULT" ] && { echo "refusing: $FAULT exists ($(head -1 "$FAULT"))" >&2; exit 4; }
docker inspect -f '{{.State.Running}}' "$CT" 2>/dev/null | grep -q true || { echo "container $CT not running" >&2; exit 3; }
NDEV=$(docker inspect -f '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' "$CT" | tr ' ' '\n' | grep -c '/dev/dri/')
[ "$NDEV" = 1 ] || { echo "$CT maps $NDEV render nodes, want exactly 1" >&2; exit 3; }
# MES / reset / ECC wording seen on gfx1250 (a08-1 and e15-02 logs); the bare word MES also appears in normal init
# lines ("MES: vmid_mask ..."), so only its failure forms count
ERR_RE='amdgpu.*('"$PCI"').*(hang|page fault|GCVM_L2_PROTECTION|VM_L2_PROTECTION|timeout|ring .* (test )?failed|MES.*(fail|timeout|full|not respond|hang)|(gpu|mode[0-9]?) reset|reset (begin|succeeded|failed)|uncorrect|UMC|poison|fatal)'
ACA_RE='correctable.*(ACA|pcie)|ACA.*correctable'
klog() { sudo -n dmesg --time-format iso 2>/dev/null; }
if klog | grep -E "$ERR_RE" | grep -v -i -E "$ACA_RE" | grep -q .; then
  echo "refusing: the kernel log already has amdgpu errors for $PCI since boot (degraded card?):" >&2
  klog | grep -E "$ERR_RE" | grep -v -i -E "$ACA_RE" | tail -5 >&2; exit 5
fi
# someone else's job on this card (another container may map every GPU): its VRAM shows up here
USED=$(rocm-smi --showbus --showmemuse --json 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(next((v.get('GPU Memory Allocated (VRAM%)', '?') for v in d.values() if isinstance(v, dict) and v.get('PCI Bus') == '$PCI'), '?'))")
case "$USED" in 0|1) ;; *) echo "refusing: $PCI has ${USED}% VRAM allocated (someone else is on this card?)" >&2; exit 8 ;; esac
exec 9>"/tmp/lhz-hkl-gpu$GPU.lock"
flock -n 9 || { echo "gpu $GPU busy (lock /tmp/lhz-hkl-gpu$GPU.lock held)" >&2; exit 6; }
STAMP=$(date -u +%Y%m%d_%H%M%S); LOG=$LOGS/${STAMP}_$TAG.log
T0=$(date -u +%Y-%m-%dT%H:%M:%S)
echo "[$T0 UTC] gpu $GPU ($PCI) $CT: $*" | tee "$LOG"
# primary group = the host group owning /dev/kfd (render): docker exec -u does not apply the container's
# --group-add, and the image's own render group has another gid
timeout -k 30 "$TMO" docker exec -u "$(id -u):$(stat -c %g /dev/kfd)" -e HOME=/tmp -w "$HERE" "$CT" "$@" >>"$LOG" 2>&1
RC=$?
sleep 2
klog | awk -v t0="$T0" '$1 >= t0' > "$LOGS/${STAMP}_$TAG.dmesg"
BAD=$(grep -E "$ERR_RE" "$LOGS/${STAMP}_$TAG.dmesg" | grep -v -i -E "$ACA_RE")
if [ -n "$BAD" ]; then
  { echo "$(date -u) job $TAG: new amdgpu error lines for $PCI"; echo "$BAD"; } > "$FAULT"
  echo "FAULT: kernel log errors during $TAG; wrote $FAULT -- stop and look before anything else runs" | tee -a "$LOG" >&2
  exit 7
fi
[ $RC -eq 124 ] && echo "TIMEOUT after $TMO s" | tee -a "$LOG" >&2
echo "[$(date -u +%H:%M:%S) UTC] rc=$RC log $LOG" | tee -a "$LOG"
exit $RC
