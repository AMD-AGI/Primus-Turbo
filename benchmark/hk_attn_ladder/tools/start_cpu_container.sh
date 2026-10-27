#!/bin/bash
# start_cpu_container.sh [name] -- the device-less build container hkl.build uses (default lhz-hk-cpu): image
# amdprimus/amdprimus:tas-20261019 (AMD clang 862306352beb, the toolchain every sha256 pin was made with), no
# /dev/kfd, no /dev/dri, no network; mounts the directory three levels above this ladder (the projects tree).
set -euo pipefail
NAME=${1:-lhz-hk-cpu}
IMAGE=${HKL_IMAGE:-amdprimus/amdprimus:tas-20261019}
ROOT=$(readlink -f "$(dirname "$0")/../../../..")
if docker inspect "$NAME" >/dev/null 2>&1; then
  docker start "$NAME" >/dev/null
else
  docker run -d --name "$NAME" --network none -v "$ROOT:$ROOT" -w "$ROOT" --entrypoint sleep "$IMAGE" infinity >/dev/null
fi
DEVS=$(docker inspect -f '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' "$NAME")
[ -z "${DEVS// /}" ] || { echo "$NAME maps devices ($DEVS): not a build container" >&2; exit 1; }
echo "$NAME up (no devices), mounts $ROOT; $(docker exec "$NAME" amdclang++ --version | head -1)"
