#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Build the TensileLite toolkit of the TENSILELITE Ulysses backend into PREFIX:
#
#   PREFIX/lib/        libtensilelite-host (linked into the backend's extension)
#   PREFIX/include/    TensileLite headers (source and generated) the extension compiles against
#   PREFIX/rocisa/, PREFIX/origami/   the other include roots it needs
#   PREFIX/tensilelite TensileLite's Python tools (Tensile, TensileCreateLibrary, TensileMergeLibrary)
#   PREFIX/python/     their Python dependencies and rocisa (put on PYTHONPATH)
#   PREFIX/bin/        tensilelite-client (tuning)
#   PREFIX/STAMP.json  the rocm-libraries commit and the compiler it was built with
#
# The commit is the one pinned in csrc/ulysses_tensilelite/manifest.json; setup.py builds the
# backend only against a toolkit whose STAMP.json matches it. The prefix has no absolute paths
# inside, so it can be built in an image or copied to shared storage.
#
#     tools/build_tensilelite.sh [PREFIX]          # default $TENSILELITE_HOME, else /opt/tensilelite
#
# SKIP_APT=1 skips the system packages (installing them needs root); WORK (default a temporary
# directory) holds the checkout and the build and is removed at the end unless KEEP_WORK=1.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MANIFEST=$HERE/../csrc/ulysses_tensilelite/manifest.json
PREFIX=$(realpath -m "${1:-${TENSILELITE_HOME:-/opt/tensilelite}}")
JOBS=${JOBS:-$(nproc)}
COMMIT=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['rocm_libraries_commit'])" "$MANIFEST")
ARCH=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['arch'])" "$MANIFEST")

if [ -z "${ROCM_PATH:-}" ]; then
    ROCM_PATH=$(rocm-sdk path --root 2>/dev/null || echo /opt/rocm)
fi
export ROCM_PATH
# TensileLite's CMake preset expects ROCm under /opt/rocm
if [ ! -e /opt/rocm ]; then
    ln -s "$ROCM_PATH" /opt/rocm
fi
CXX=/opt/rocm/bin/amdclang++
CC=/opt/rocm/bin/amdclang
TOOLCHAIN=$("$CXX" --version | head -n 1)

if [ "${SKIP_APT:-0}" != 1 ]; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git cmake ccache python3-venv \
        libmsgpack-cxx-dev libboost-dev libnuma-dev >/dev/null
fi

WORK=${WORK:-$(mktemp -d)}
mkdir -p "$WORK"
SRC=$WORK/rocm-libraries
HB=$SRC/projects/hipblaslt
TL=$HB/tensilelite
BUILD=$WORK/build

echo "[tensilelite] rocm-libraries $COMMIT for $ARCH into $PREFIX (work $WORK)"
if [ ! -d "$SRC/.git" ]; then
    git init -q "$SRC"
    git -C "$SRC" remote add origin https://github.com/ROCm/rocm-libraries.git
fi
git -C "$SRC" sparse-checkout set --cone projects/hipblaslt shared cmake
git -C "$SRC" fetch -q --depth 1 --filter=blob:none origin "$COMMIT"
git -C "$SRC" checkout -q FETCH_HEAD

# a build venv: TensileLite's CMake and the rocisa build need its Python requirements
python3 -m venv "$WORK/venv"
# shellcheck disable=SC1091
source "$WORK/venv/bin/activate"
python3 -m pip install -q --upgrade pip
python3 -m pip install -q invoke "scikit-build-core>=0.10" "nanobind>=2.0,!=3.0.0" -r "$TL/requirements.txt"
python3 -m pip wheel -q --no-build-isolation --no-deps -w "$WORK/wheels" "$TL/rocisa"
python3 -m pip install -q --no-deps "$WORK"/wheels/rocisa-*.whl

SYSDEPS=$ROCM_PATH/lib/rocm_sysdeps/lib
LDFLAGS_SYSDEPS="-L$SYSDEPS -Wl,-rpath,$SYSDEPS"
cmake --preset tensilelite -S "$HB" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS="$ARCH" \
    -DTENSILELITE_ENABLE_SDMA=ON -DHIPBLASLT_ENABLE_YAML=OFF -DTENSILELITE_CLIENT_ENABLE_ROCPROFSDK=OFF \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_EXE_LINKER_FLAGS="$LDFLAGS_SYSDEPS" -DCMAKE_SHARED_LINKER_FLAGS="$LDFLAGS_SYSDEPS" >/dev/null
cmake --build "$BUILD" -j "$JOBS"

# empty PREFIX in place (it may be a mount point), but only if it is empty or an earlier toolkit
mkdir -p "$PREFIX"
if [ -n "$(ls -A "$PREFIX")" ] && [ ! -f "$PREFIX/STAMP.json" ]; then
    echo "[tensilelite] $PREFIX is not empty and holds no STAMP.json; refusing to overwrite it" >&2
    exit 1
fi
find "$PREFIX" -mindepth 1 -delete
mkdir -p "$PREFIX"/{lib,include,bin,python}
cp -a "$BUILD"/tensilelite/libtensilelite-host.so* "$PREFIX/lib/"
cp -a "$BUILD"/tensilelite/client/tensilelite-client "$PREFIX/bin/"
cp -a "$TL/include/." "$PREFIX/include/"
cp -a "$BUILD/tensilelite/include/." "$PREFIX/include/"
# the include roots the attn-a2a-demo extension uses: rocisa's source root, origami's two include dirs
cp -a "$TL/rocisa" "$PREFIX/rocisa"
rm -rf "$PREFIX/rocisa/build"
mkdir -p "$PREFIX/origami"
cp -a "$SRC/shared/origami/include" "$PREFIX/origami/include"
cp -a "$BUILD/origami/include" "$PREFIX/origami/build-include"
cp -a "$TL/tensilelite" "$PREFIX/tensilelite"
python3 -m pip install -q --target "$PREFIX/python" -r "$TL/requirements.txt" "$WORK"/wheels/rocisa-*.whl
deactivate

python3 - "$PREFIX/STAMP.json" "$COMMIT" "$ARCH" "$TOOLCHAIN" <<'PY'
import json, sys
path, commit, arch, toolchain = sys.argv[1:]
json.dump({"rocm_libraries_commit": commit, "arch": arch, "toolchain": toolchain}, open(path, "w"), indent=2)
PY

if [ "${KEEP_WORK:-0}" != 1 ]; then
    rm -rf "$WORK"
fi
echo "[tensilelite] done: $(du -sh "$PREFIX" | cut -f1) in $PREFIX"
