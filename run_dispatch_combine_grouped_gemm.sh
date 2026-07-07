#!/usr/bin/env bash
# Run the EP8 end-to-end fused mega-MoE test (prologue + dispatch + swiglu +
# combine over HIP-IPC symmetric memory) inside the dev_primus container.
#
# Uses THIS repo's own compiled extension (primus_turbo/pytorch/_C*.so +
# primus_turbo/lib/libprimus_turbo_kernels.so), NOT the older one bundled in the
# image's site-packages. The image build lags the source (e.g. it lacks the
# turbo moe permute_preprocessing op the test's baseline needs), so we put the
# source tree first on PYTHONPATH and its lib dir on LD_LIBRARY_PATH.
#
# Usage (from anywhere):
#   ./run_dispatch_combine_grouped_gemm.sh                 # EP8 defaults, correctness
#   ./run_dispatch_combine_grouped_gemm.sh --perf          # add perf timing
#   ./run_dispatch_combine_grouped_gemm.sh --mode load_balanced --num-tokens 8192
#   DEV_PRIMUS_CONTAINER=dev_primus ./run_dispatch_combine_grouped_gemm.sh ...
set -euo pipefail

CONTAINER="${DEV_PRIMUS_CONTAINER:-dev_primus}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST="tests/pytorch/ops/test_mega_moe_dispatch_combine_grouped_gemm.py"
NP="${NP:-8}"
# Fresh rendezvous port each run: a stale process from a prior killed/hung run
# holding the default port wedges every subsequent launch (looks like 卡死).
PORT="${MASTER_PORT:-$(( 20000 + RANDOM % 20000 ))}"

echo "[run_dispatch_combine_grouped_gemm] container=$CONTAINER repo=$REPO port=$PORT np=$NP"
echo "[run_dispatch_combine_grouped_gemm] test=$TEST args=$*"

docker exec -e PYTHONUNBUFFERED=1 -e PYTORCH_ROCM_ARCH=gfx950 -e REPO="$REPO" -e TEST="$TEST" \
  -e MASTER_PORT="$PORT" -e NP="$NP" \
  -e MEGA_COMB_WRITE_CACHE="${MEGA_COMB_WRITE_CACHE:-}" -e MEGA_COMB_READ_CACHE="${MEGA_COMB_READ_CACHE:-}" \
  -e MEGA_GEMM_LEGACY_VIEW="${MEGA_GEMM_LEGACY_VIEW:-}" -e MEGA_CLEAR_CACHE="${MEGA_CLEAR_CACHE:-}" \
  -e MEGA_POOL_HEAD="${MEGA_POOL_HEAD:-}" \
  "$CONTAINER" bash -lc '
set -e
[ -n "${MEGA_CLEAR_CACHE:-}" ] && rm -rf /root/.flydsl /tmp/flydsl_autotune
ulimit -l unlimited 2>/dev/null || true
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ROCSHMEM_HEAP_SIZE=8GB
# require the source tree to be self-contained (built in-place)
SO="$(ls "$REPO"/primus_turbo/pytorch/_C*.so 2>/dev/null | head -1)"
if [ -z "$SO" ] || [ -L "$SO" ]; then
    echo "[run_dispatch_combine_grouped_gemm] error: source-built _C not found at" \
         "$REPO/primus_turbo/pytorch/ (or it is a symlink to the image build)." >&2
    echo "    restore it with: cp build/lib/primus_turbo/pytorch/_C*.so primus_turbo/pytorch/" >&2
    exit 1
fi
# import the source package first; resolve its kernel lib from the source tree
export PYTHONPATH="$REPO"
export LD_LIBRARY_PATH="$REPO/primus_turbo/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$REPO"
exec python "$TEST" --num-processes "$NP" "$@"
' bash "$@"
