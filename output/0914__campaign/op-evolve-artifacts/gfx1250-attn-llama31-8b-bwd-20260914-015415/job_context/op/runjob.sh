#!/usr/bin/env bash
# Detached runner for long GPU jobs. Survives its own caller dying.
#
# Usage:  runjob.sh <key> <command ...>
#
# The in-container command is a child of dockerd, not of this shell, so if the
# caller times out or is killed the command keeps running and keeps the GPU.
# Therefore: launch detached under setsid, write the exit code to a sentinel
# file, and poll for the sentinel.
#
# Run-directory semantics, which are the whole point:
#   op/_runs/<key>/rc      exists -> the run FINISHED. Re-invoking RE-RUNS it
#                          (the directory is wiped first). A finished run is
#                          never served from cache; that would report a stale
#                          number as if it were fresh.
#   op/_runs/<key>/pid     exists, no rc, process alive -> the run is IN FLIGHT.
#                          Re-invoking ATTACHES to it and keeps polling. It does
#                          not start a second copy competing for the same GPU.
#   neither                -> fresh launch.
#
# Always exits with the command's own exit code, and always prints its output.

set -u
JC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEY="$1"; shift
D="$JC/op/_runs/$KEY"
CONTAINER=op-evolve-gfx1250-attn-llama31-8b-bwd
# runtime.gpu_id in the spec. Every measurement must land on the same physical
# card: "reproduce every claim on the same device, and treat a result that
# appears only after changing device as no result." Unpinned runs default to
# device 0, which on this node is another tenant's job.
GPU_ID=${GPU_ID:-1}

attach=0
if [ -f "$D/rc" ]; then
    rm -rf "$D"                       # FINISHED -> re-run, never cache
elif [ -f "$D/pid" ] && kill -0 "$(cat "$D/pid")" 2>/dev/null; then
    attach=1                          # IN FLIGHT -> attach
else
    rm -rf "$D"                       # stale/aborted -> fresh
fi

if [ "$attach" = 0 ]; then
    mkdir -p "$D"
    printf '%s\n' "$*" > "$D/cmd"
    # docker exec, not ssh: zero ssh hops on this node. Same shape either way.
    setsid bash -c "docker exec -e HIP_VISIBLE_DEVICES='$GPU_ID' -w '$JC' '$CONTAINER' bash -lc \"\$(cat '$D/cmd')\" > '$D/out' 2>&1; echo \$? > '$D/rc'" \
        </dev/null >/dev/null 2>&1 &
    echo $! > "$D/pid"
    echo "[runjob] launched key=$KEY pid=$(cat "$D/pid")" >&2
else
    echo "[runjob] attaching to in-flight key=$KEY pid=$(cat "$D/pid")" >&2
fi

# Poll for the sentinel. The caller may die here; the run does not.
for _ in $(seq 1 7200); do
    [ -f "$D/rc" ] && break
    sleep 1
done
if [ ! -f "$D/rc" ]; then
    echo "[runjob] TIMEOUT waiting for key=$KEY; run is still in flight" >&2
    exit 124
fi
cat "$D/out"
exit "$(cat "$D/rc")"
