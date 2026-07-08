###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""fp8 cross-rank dispatch PUSH (milestone-2 comm): quantize-then-push.

Pushes per-token fp8 activations + their raw E8M0 block scales into the peer
``pool_fp8`` / ``pool_scale`` symmetric regions over XGMI, halving the dispatch
comm bytes vs the bf16 token push. Comm-only (no fused GEMM): the caller runs the
prologue (which fills the group layout / origin tables), then this push, a host
sync + group barrier (cross-rank completion), then the standalone grouped mxfp8 L1
GEMM over ``pool_fp8`` / ``pool_scale``.

Token geometry mirrors ``ep_intranode._make_dispatch_tile`` (warp-per-token, 16B/lane
vectorized), specialized to fp8 (1 byte/elem, so hidden % 1024 == 0). The E8M0 scale
row (hidden//32 bytes) is pushed alongside as int32 words (hidden//128 i32/row).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.compiler.ast_rewriter import InsertEmptyYieldForSCFFor
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
    create_buffer_resource_from_addr,
)

from primus_turbo.flydsl.mega.prims import l2_invalidate
from primus_turbo.flydsl.mega.sym_layout import SymLayout
from primus_turbo.flydsl.utils.gemm_helper import _emit_if_then, make_value_attrs

_WARP = 64
_BLOCK_THREADS = 512
_VEC_I32 = 4  # 4 x i32 = 16B per lane (b128 XGMI)


def _peer_addr(local_base, offsets_resource, dst_rank):
    delta = buffer_load(offsets_resource, dst_rank, vec_width=1, dtype=fx.T.i64())
    return local_base + delta


def _emit_for(stop, body):
    InsertEmptyYieldForSCFFor.scf_for_dispatch(
        fx.Int32(0), stop, fx.Int32(1), lambda iv, _names: body(fx.arith.ArithValue(iv, signed=True))
    )


@functools.lru_cache(maxsize=64)
def _compile_fp8_push(hidden, num_max_pool_tokens, num_ranks):
    import os

    push_scale = os.environ.get("MXFP8_PUSH_SCALE", "1") != "0"  # diag toggle (closure const)
    assert hidden % 1024 == 0, f"fp8 token push needs hidden % 1024 == 0, got {hidden}"
    n_warps = _BLOCK_THREADS // _WARP
    hidden_i32 = hidden // 4  # fp8 row: hidden bytes -> i32 words
    cols_per_warp_i32 = _WARP * _VEC_I32  # 256
    chunk_count = hidden_i32 // cols_per_warp_i32  # = hidden // 1024
    scale_i32 = hidden // 128  # E8M0 row: hidden//32 bytes -> i32 words
    assert scale_i32 <= _WARP, f"scale row {scale_i32} i32 > warp {_WARP} (hidden {hidden} > 8192 unsupported)"
    pool_tok_bytes = num_max_pool_tokens * hidden  # fp8 pool records (bytes)
    pool_scale_bytes = num_max_pool_tokens * (hidden // 32)  # E8M0 pool records (bytes)

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def dispatch_fp8_push(
        XQ: fx.Tensor,  # fp8 tokens viewed int32 [T, hidden//4] flattened
        XS: fx.Tensor,  # raw E8M0 scales viewed int32 [T, hidden//128] flattened
        EXPERT_SEND_DST_RANK: fx.Tensor,
        EXPERT_SEND_DST_ROW: fx.Tensor,
        EXPERT_SEND_COUNT: fx.Tensor,
        EXPERT_SEND_OFFSET: fx.Tensor,
        DISPATCHED_TOKEN_IDX: fx.Tensor,
        sym_layout: SymLayout,
    ):
        thread_index = fx.thread_idx.x
        task_index = fx.block_idx.x  # one block per comm task (grid = num_comm)
        warp_id = thread_index // fx.Int32(_WARP)
        lane_id = thread_index % fx.Int32(_WARP)

        xq_res = create_buffer_resource(XQ, max_size=True)
        xs_res = create_buffer_resource(XS, max_size=True)
        esr = create_buffer_resource(EXPERT_SEND_DST_RANK, max_size=True)
        esrow = create_buffer_resource(EXPERT_SEND_DST_ROW, max_size=True)
        escnt = create_buffer_resource(EXPERT_SEND_COUNT, max_size=True)
        esoff = create_buffer_resource(EXPERT_SEND_OFFSET, max_size=True)
        dti = create_buffer_resource(DISPATCHED_TOKEN_IDX, max_size=True)

        pool_off = create_buffer_resource_from_addr(
            sym_layout.offsets_ptr, num_records_bytes=num_ranks * 8
        )

        destination_rank = buffer_load(esr, task_index, vec_width=1, dtype=fx.T.i32())
        dest_row_start = buffer_load(esrow, task_index, vec_width=1, dtype=fx.T.i32())
        source_offset = buffer_load(esoff, task_index, vec_width=1, dtype=fx.T.i32())
        token_count = buffer_load(escnt, task_index, vec_width=1, dtype=fx.T.i32())

        pool_addr = _peer_addr(sym_layout.pool_fp8_ptr, pool_off, destination_rank)
        peer_pool = create_buffer_resource_from_addr(pool_addr, num_records_bytes=pool_tok_bytes)
        pscale_addr = _peer_addr(sym_layout.pool_scale_ptr, pool_off, destination_rank)
        peer_pscale = create_buffer_resource_from_addr(pscale_addr, num_records_bytes=pool_scale_bytes)

        # warp-per-token over [0, token_count)
        local_count = (token_count - warp_id + fx.Int32(n_warps - 1)) // fx.Int32(n_warps)

        def _row(i):
            row_index = warp_id + i * fx.Int32(n_warps)
            source_row = buffer_load(dti, source_offset + row_index, vec_width=1, dtype=fx.T.i32())
            dest_row = dest_row_start + row_index
            # token (fp8): 16B/lane vectorized i32x4
            vals = []
            for c in fx.range_constexpr(chunk_count):
                col = fx.Int32(c * cols_per_warp_i32) + lane_id * fx.Int32(_VEC_I32)
                vals.append(
                    buffer_load(
                        xq_res, source_row * fx.Int32(hidden_i32) + col, vec_width=_VEC_I32, dtype=fx.T.i32()
                    )
                )
            for c in fx.range_constexpr(chunk_count):
                col = fx.Int32(c * cols_per_warp_i32) + lane_id * fx.Int32(_VEC_I32)
                buffer_store(vals[c], peer_pool, dest_row * fx.Int32(hidden_i32) + col)
            # E8M0 scale row: hidden//128 i32 words, lanes [0, scale_i32) each push one i32.
            # Control-flow guard (not a buffer mask): masked lanes must not form an OOB address.
            if push_scale:
                def _one_scale():
                    sv = buffer_load(
                        xs_res, source_row * fx.Int32(scale_i32) + lane_id, vec_width=1, dtype=fx.T.i32()
                    )
                    buffer_store(sv, peer_pscale, dest_row * fx.Int32(scale_i32) + lane_id)

                _emit_if_then(lane_id < fx.Int32(scale_i32), _one_scale)

        _emit_for(local_count, _row)
        # release fence: drain the peer XGMI stores so they are globally visible before
        # the kernel completes (host sync + group.barrier then gate the pool read).
        fx.rocdl.s_waitcnt(0)

    @flyc.jit
    def launch_fp8_push(
        XQ, XS, EXPERT_SEND_DST_RANK, EXPERT_SEND_DST_ROW, EXPERT_SEND_COUNT,
        EXPERT_SEND_OFFSET, DISPATCHED_TOKEN_IDX, sym_layout, num_comm: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        dispatch_fp8_push(
            XQ, XS, EXPERT_SEND_DST_RANK, EXPERT_SEND_DST_ROW, EXPERT_SEND_COUNT,
            EXPERT_SEND_OFFSET, DISPATCHED_TOKEN_IDX, sym_layout,
            value_attrs=make_value_attrs(2, 0, "512,512"),
        ).launch(grid=(num_comm, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch_fp8_push


def dispatch_fp8_push_launch(xq, xs, handle, sym_layout, num_max_pool_tokens, num_ranks):
    """Launch the fp8 dispatch PUSH given a prologue ``handle`` + active ``sym_layout``.

    ``xq`` [T, H] fp8 tokens, ``xs`` [T, H//32] raw E8M0 scales. Pushes into the peer
    ``pool_fp8`` / ``pool_scale`` regions. Caller must host-sync + group-barrier after
    for cross-rank completion before reading the pool.
    """
    T, H = xq.shape
    esdr, esdr_row, esc, eso, dti = handle[0], handle[1], handle[2], handle[3], handle[4]
    num_comm = int(esdr.numel())
    XQ = xq.contiguous().view(torch.uint8).view(torch.int32).reshape(-1)
    XS = xs.contiguous().view(torch.uint8).view(torch.int32).reshape(-1)
    launch = _compile_fp8_push(int(H), int(num_max_pool_tokens), int(num_ranks))
    launch(
        XQ, XS, esdr, esdr_row, esc, eso, dti, sym_layout, num_comm,
        stream=torch.cuda.current_stream(),
    )


@functools.lru_cache(maxsize=1)
def _compile_l2_inv():
    @flyc.kernel(known_block_size=[_WARP, 1, 1])
    def _inv():
        l2_invalidate()

    @flyc.jit
    def launch_inv(n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        _inv().launch(grid=(n, 1, 1), block=(_WARP, 1, 1), stream=stream)

    return launch_inv


def l2_invalidate_all(num_cu: int = 320):
    """Device-wide L2 invalidate so a cached-heap region written cross-rank (pool_fp8 /
    pool_scale) is re-fetched from DRAM by the subsequent reader (the L1 GEMM)."""
    _compile_l2_inv()(int(num_cu), stream=torch.cuda.current_stream())
