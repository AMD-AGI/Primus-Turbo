#!/bin/bash
# snapshot.sh N: copy the campaign's kernel sources into rounds/round-N/kernel_snapshot/
set -e
C=$(cd "$(dirname "$0")" && pwd)
R=/home/xiaompen/Turbo
D=$C/rounds/round-$1/kernel_snapshot
mkdir -p "$D"
for f in \
    primus_turbo/flydsl/mega/bf16_gfx1250/__init__.py \
    primus_turbo/flydsl/mega/bf16_gfx1250/dispatch_gemm_kernel.py \
    primus_turbo/flydsl/mega/bf16_gfx1250/gemm_combine_kernel.py \
    primus_turbo/flydsl/mega/bf16_gfx1250/gemm_nt_tile.py \
    primus_turbo/flydsl/mega/bf16_gfx1250/x_redispatch_kernel.py \
    primus_turbo/flydsl/mega/bf16/ep_intranode.py \
    primus_turbo/flydsl/mega/bf16/dispatch_prologue_kernel.py \
    primus_turbo/flydsl/mega/bf16/symm_buffer.py \
    primus_turbo/flydsl/grouped_gemm/grouped_gemm_bf16_kernel_gfx1250.py \
    primus_turbo/pytorch/kernels/fused_mega_moe/fused_mega_moe_gfx1250_impl.py; do
    mkdir -p "$D/$(dirname "$f")"
    cp "$R/$f" "$D/$f"
done
(cd "$R" && git rev-parse HEAD && git status --short) > "$D/git_state.txt"
echo "snapshot -> $D"
