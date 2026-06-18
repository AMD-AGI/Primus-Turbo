#!/usr/bin/env bash
# Run the EP8 cross-rank (XGMI) dispatch + grouped-GEMM fp8 test inside dev_primus.
#
# Sets up the env so the SOURCE flydsl (this repo) imports together with the built
# primus_turbo.pytorch._C (SymmetricMemory): symlinks the compiled _C extension next
# to the source pytorch package and puts its lib dir on LD_LIBRARY_PATH.
#
# Usage (from anywhere):
#   ./run_ep8_test.sh                         # DeepSeek defaults (8 procs, heavy)
#   ./run_ep8_test.sh --num-processes 8 --hidden 7168 --inter 2048 \
#                     --num-experts 8 --per-peer 512 --iters 10
#   DEV_PRIMUS_CONTAINER=dev_primus ./run_ep8_test.sh ...   # override container name
set -euo pipefail

CONTAINER="${DEV_PRIMUS_CONTAINER:-dev_primus}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST="tests/test_dispatch_grouped_gemm_fp8_ep8.py"
# Fresh rendezvous port each run: a stale process from a prior killed/hung run
# holding the default port wedges every subsequent launch (looks like 卡死).
PORT="${MASTER_PORT:-$(( 20000 + RANDOM % 20000 ))}"

echo "[run_ep8_test] container=$CONTAINER repo=$REPO port=$PORT"
echo "[run_ep8_test] test=$TEST args=$*"

docker exec -e PYTORCH_ROCM_ARCH=gfx950 -e REPO="$REPO" -e TEST="$TEST" -e MASTER_PORT="$PORT" \
  "$CONTAINER" bash -lc '
set -e
# site-packages holding the built _C + libprimus_turbo_kernels.so
SP="$(python -c "import site; print(site.getsitepackages()[0])")"
SO="$(ls "$SP"/primus_turbo/pytorch/_C*.so 2>/dev/null | head -1)"
if [ -z "$SO" ]; then
    echo "[run_ep8_test] error: built primus_turbo.pytorch._C not found under $SP" >&2
    exit 1
fi
# make the source tree import-complete (idempotent symlink of the compiled extension)
ln -sf "$SO" "$REPO/primus_turbo/pytorch/$(basename "$SO")"
export PYTHONPATH="$REPO"
export LD_LIBRARY_PATH="$SP${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$REPO"
exec python "$TEST" "$@"
' bash "$@"
