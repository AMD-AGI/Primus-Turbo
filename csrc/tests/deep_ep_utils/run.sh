#!/usr/bin/env bash
# Build and run the utils.cuh verification harness on the local GPU.
#   - test_utils      : native gfx1250 path (HW named barrier + TDM async)
#   - test_utils_emul : -DDISABLE_GFX1250_FEATURES (LDS-atomic emulated barrier)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
INC="$REPO/csrc/include"
KERN="$REPO/csrc/kernels"
ARCH="${ARCH:-gfx1250}"

CXXFLAGS=(-std=c++17 --offload-arch="$ARCH" -O2 -I"$INC" -I"$KERN")

# Make the ROCm runtime that hipcc links against discoverable at run time.
ROCM_LIB="$(dirname "$(dirname "$(readlink -f "$(command -v hipcc)")")")/lib"
export LD_LIBRARY_PATH="$ROCM_LIB:${LD_LIBRARY_PATH:-}"

echo ">>> arch=$ARCH  include=$INC"

echo ">>> compiling test_utils (native)"
hipcc "${CXXFLAGS[@]}" "$HERE/test_utils.hip" -o "$HERE/test_utils" || exit 1

echo ">>> compiling test_utils_emul (emulated barrier)"
hipcc "${CXXFLAGS[@]}" -DDISABLE_GFX1250_FEATURES "$HERE/test_utils.hip" \
    -o "$HERE/test_utils_emul" || exit 1

rc=0
echo; echo ">>> running EMULATED build (production path on gfx1250)"
timeout 120 "$HERE/test_utils_emul" || rc=$?

echo; echo ">>> running NATIVE build (HW named barrier + TDM async)"
timeout 120 "$HERE/test_utils" || rc=$?

echo; echo ">>> overall rc=$rc"
exit $rc
