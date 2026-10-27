#!/bin/bash
# start_gpu_container.sh <gpu> [name] -- a container that sees exactly ONE GPU: /dev/kfd plus that GPU's render
# node only (gpu N = the N-th AMD render node in PCI order, renderD128 + 8 N on ctheliosr-rck-g02-e15-02).
# Default name lhz-hkl-g<gpu>. Image amdprimus/amdprimus:tas-20261019 (torch, flydsl 0.3.4.1, rocprofv3).
set -euo pipefail
[ $# -ge 1 ] || { sed -n '2,4p' "$0"; exit 2; }
GPU=$1
NAME=${2:-lhz-hkl-g$GPU}
IMAGE=${HKL_IMAGE:-amdprimus/amdprimus:tas-20261019}
ROOT=$(readlink -f "$(dirname "$0")/../../../..")
mapfile -t NODES < <(for r in /sys/class/drm/renderD*; do
  [ "$(cat "$r/device/vendor" 2>/dev/null)" = 0x1002 ] && echo "$(basename "$(readlink -f "$r/device")") $(basename "$r")"; done | sort)
[ "$GPU" -ge 0 ] && [ "$GPU" -lt "${#NODES[@]}" ] || { echo "gpu $GPU: only ${#NODES[@]} AMD render nodes" >&2; exit 1; }
PCI=${NODES[$GPU]% *}; NODE=${NODES[$GPU]#* }
for c in $(docker ps --format '{{.Names}}'); do  # warn when someone else's container also maps this node
  [ "$c" = "$NAME" ] && continue
  docker inspect -f '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}{{.HostConfig.Privileged}}' "$c" 2>/dev/null |
    grep -q -E "/dev/dri/$NODE|true$|/dev/dri( |$)" && echo "warning: running container $c can also use $NODE ($PCI)" >&2
done
if docker inspect "$NAME" >/dev/null 2>&1; then
  docker start "$NAME" >/dev/null
else
  docker run -d --name "$NAME" --device /dev/kfd --device "/dev/dri/$NODE" --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g "/dev/dri/$NODE")" \
    --shm-size 16g --ipc private -v "$ROOT:$ROOT" -w "$ROOT" --entrypoint sleep "$IMAGE" infinity >/dev/null
fi
echo "$NAME up: gpu $GPU = $PCI /dev/dri/$NODE"
echo "$PCI" > "/tmp/$NAME.pci"
