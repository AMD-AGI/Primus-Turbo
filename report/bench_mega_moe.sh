#!/usr/bin/env bash
#
# Run benchmark/ops/training/bench_mega_moe.py inside the rootless-runc ROCm
# container, once per fused mega-MoE mode.
#
# The container is the one started by ~/start_container.sh (rootless runc,
# $HOME bind-mounted at /io). A running container is reused; otherwise one is
# started here with ATTACH=0.
#
# Usage:
#   ./bench_mega_moe.sh [extra args forwarded to bench_mega_moe.py]
#
# Env knobs (all optional):
#   MODES               modes to sweep. Default: "dispatch_grouped_gemm grouped_gemm_combine"
#   MODELS              --models value.        Default: DeepSeek-V3
#   ITERS               --iters value.         Default: 50
#   NUM_PROCESSES       --num-processes value. Default: 8
#   OUT_DIR             host dir for CSV + logs. Default: <repo>/report/runs/<timestamp>
#   CONTAINER_NAME      runc container to use. Default: first running one
#   ROOTLESS_RUNC_HOME  runc state/bundles root. Default: $HOME/.local/share/rootless-runc
#   START_CONTAINER     launcher used when nothing is running. Default: $HOME/start_container.sh
#   PYTORCH_ROCM_ARCH   target arch. Default: gfx950
#
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODES=${MODES:-"dispatch_grouped_gemm grouped_gemm_combine"}
MODELS=${MODELS:-DeepSeek-V3}
ITERS=${ITERS:-50}
NUM_PROCESSES=${NUM_PROCESSES:-8}
PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-gfx950}

ROOTLESS_RUNC_HOME=${ROOTLESS_RUNC_HOME:-"$HOME/.local/share/rootless-runc"}
RUNC_STATE="$ROOTLESS_RUNC_HOME/state"
RUNC_BIN=${RUNC_BIN:-$(command -v runc || true)}
START_CONTAINER=${START_CONTAINER:-"$HOME/start_container.sh"}
CONTAINER_NAME=${CONTAINER_NAME:-}
# $HOME is bind-mounted at /io by start_container.sh
CONTAINER_HOME=${CONTAINER_HOME:-/io}

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUT_DIR=${OUT_DIR:-"$SCRIPT_DIR/runs/$TIMESTAMP"}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

runc_do() {
    "$RUNC_BIN" --root "$RUNC_STATE" "$@"
}

container_status() {
    runc_do state "$1" 2>/dev/null | jq -r '.status' 2>/dev/null || true
}

# First running container in the rootless state dir (empty if there is none).
first_running_container() {
    local id
    for id in $(runc_do list --quiet 2>/dev/null); do
        if [[ "$(container_status "$id")" == "running" ]]; then
            printf '%s\n' "$id"
            return
        fi
    done
}

ensure_container() {
    if [[ -n "$CONTAINER_NAME" ]]; then
        [[ "$(container_status "$CONTAINER_NAME")" == "running" ]] ||
            die "container $CONTAINER_NAME is not running (start it with: runc start $CONTAINER_NAME)"
        return
    fi

    CONTAINER_NAME=$(first_running_container)
    if [[ -n "$CONTAINER_NAME" ]]; then
        printf 'Reusing running container %s\n' "$CONTAINER_NAME"
        return
    fi

    [[ -x "$START_CONTAINER" ]] || die "no running container and $START_CONTAINER is not executable"
    printf 'No running container; starting one via %s\n' "$START_CONTAINER"
    ATTACH=0 "$START_CONTAINER"

    CONTAINER_NAME=$(first_running_container)
    [[ -n "$CONTAINER_NAME" ]] || die "container failed to start"
    printf 'Started container %s\n' "$CONTAINER_NAME"
}

# Host path under $HOME -> the path the container sees under /io.
to_container_path() {
    local host_path=$1
    [[ "$host_path" == "$HOME"/* ]] ||
        die "$host_path is outside \$HOME, so it is not visible in the container"
    printf '%s/%s\n' "$CONTAINER_HOME" "${host_path#"$HOME"/}"
}

[[ -n "$RUNC_BIN" && -x "$RUNC_BIN" ]] || die "runc is not available"
command -v jq >/dev/null 2>&1 || die "jq is not available"

ensure_container
mkdir -p "$OUT_DIR"
CONTAINER_REPO=$(to_container_path "$REPO_DIR")
CONTAINER_OUT=$(to_container_path "$OUT_DIR")

printf '\n'
printf 'container : %s\n' "$CONTAINER_NAME"
printf 'repo      : %s (host: %s)\n' "$CONTAINER_REPO" "$REPO_DIR"
printf 'models    : %s | iters: %s | ranks: %s\n' "$MODELS" "$ITERS" "$NUM_PROCESSES"
printf 'modes     : %s\n' "$MODES"
printf 'output    : %s\n\n' "$OUT_DIR"

failed=()
for mode in $MODES; do
    log="$OUT_DIR/$mode.log"
    printf '=== %s ===\n' "$mode"
    printf '    log: %s\n' "$log"
    set +e
    runc_do exec \
        --cwd "$CONTAINER_REPO" \
        --env "PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH" \
        "$CONTAINER_NAME" \
        python benchmark/ops/training/bench_mega_moe.py \
        --mode "$mode" \
        --models $MODELS \
        --iters "$ITERS" \
        --num-processes "$NUM_PROCESSES" \
        --output "$CONTAINER_OUT/$mode.csv" \
        "$@" 2>&1 | tee "$log"
    status=${PIPESTATUS[0]}
    set -e
    if [[ "$status" -ne 0 ]]; then
        printf '    FAILED (exit %s)\n' "$status"
        failed+=("$mode")
    fi
    printf '\n'
done

printf 'Results in %s\n' "$OUT_DIR"
ls -1 "$OUT_DIR" | sed 's/^/  /'
printf '\n'

# baseline comparison (the host has no python, so run the extractor in the container too)
runc_do exec \
    --cwd "$CONTAINER_REPO" \
    "$CONTAINER_NAME" \
    python report/extract_performance.py "$CONTAINER_OUT" 2>&1 | tee "$OUT_DIR/compare.txt" || true

if [[ ${#failed[@]} -gt 0 ]]; then
    printf 'Failed modes: %s\n' "${failed[*]}" >&2
    exit 1
fi
