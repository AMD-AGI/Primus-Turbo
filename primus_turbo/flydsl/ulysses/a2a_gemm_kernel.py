###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Ulysses A2AGemm: an all-to-all from head-sharded to sequence-sharded rows fused into the GEMM
that consumes them (push mode), with an optional residual epilogue.

Every rank holds ``O`` = ``[n * M, K_r]`` rows of the full sequence for its ``K_r`` columns (its
heads). Push work moves the rows of destination ``dst``'s sequence slice into ``dst``'s symmetric
recv bank at column block ``rank`` and raises ``dst``'s flag ``(m_tile, rank)``. The recv bank is
row-major ``[M, n * K_r]``: the GEMM's ``A``. A GEMM tile waits for the flags of its m-tile from all
sources, then runs the plain K loop, so the accumulation order is fixed (bitwise deterministic).

The first ``num_push`` blocks claim push units (m-major, so early m-tiles complete first) until none
are left, then compute the last tiles. A block only waits once every unit is claimed by a running
block, so the launch cannot deadlock whatever the grid size.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.deep_ep.launch_cache import run_launcher
from primus_turbo.flydsl.mega.bf16.gemm_bf16_kernel import (
    _make_shared_storage,
    gemm_bf16_tile,
)
from primus_turbo.flydsl.ulysses.common import (
    BLOCK_THREADS,
    SYS_NT,
    VEC_BF16,
    VEC_I32,
    claim_unit,
    exit_and_bump_epoch,
    i64_base,
    spin_flag,
)
from primus_turbo.flydsl.ulysses.workspace import (
    FLAG_BYTES,
    FLAG_REGION_BYTES,
    FLAGS_PER_REGION,
)
from primus_turbo.flydsl.utils.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
    create_buffer_resource_from_addr,
    extract_base_index,
)
from primus_turbo.flydsl.utils.gemm_helper import (
    StoreCBf16,
    _pack_out_pair,
    make_bf16_fp16_tile_tensor,
    make_row_band_resource,
    make_value_attrs,
)
from primus_turbo.flydsl.utils.prims import ceildiv, ld, memory_fence, st

# 16 B vectors whose loads are issued before their stores
_BATCH = 8


def _residual_store_c(residual):
    """A ``StoreCBf16`` writing ``C = acc + residual``. Defined in a function so the kernel cache key
    (which hashes the source of functions the launcher reaches) covers the class body."""

    class _StoreCBf16Residual(StoreCBf16):
        """``C = acc + R`` with the add in fp32 before the one bf16 rounding (an unfused ``+ residual``
        after a bf16 GEMM output rounds twice). Only the paired-column store of a full ``BLOCK_N = 256``
        tile is implemented: the GEMM asserts ``N % BLOCK_N == 0``."""

        def __init__(self, C, c_rows, c_cols, out_ty, cache_modifier=0):
            super().__init__(C, c_rows, c_cols, out_ty, cache_modifier=cache_modifier)
            self.r_base = extract_base_index(residual)

        def store_band_pair16(self, frag_even, frag_odd, base_row, base_col, n_tiles_b, mask_cols=True):
            rsrc = make_row_band_resource(self.c_base, base_row, self.c_rows, self.c_cols, 2)
            res_rsrc = make_row_band_resource(self.r_base, base_row, self.c_rows, self.c_cols, 2)
            n_tiles_a = len(frag_even) // n_tiles_b
            lane_col = self.lane_id % 16
            row_bytes = self.c_cols * 2
            col_ok = [
                (base_col + (j * 16 + lane_col) * 2 < self.c_cols) if mask_cols else None
                for j in range_constexpr(n_tiles_b)
            ]
            base_off = ((self.lane_id // 16) * 4) * row_bytes + (base_col + lane_col * 2) * 2
            for ti in range_constexpr(n_tiles_a):
                offs = [base_off + (ti * 16 + r) * row_bytes for r in range_constexpr(4)]
                res = [
                    [
                        Vec.from_elements(
                            [
                                buffer_load(
                                    res_rsrc, (off + j * 64) // 4, vec_width=1, dtype=fx.T.i32, mask=col_ok[j]
                                )
                            ],
                            fx.Int32,
                        ).bitcast(fx.BFloat16)
                        for j in range_constexpr(n_tiles_b)
                    ]
                    for off in offs
                ]
                for r in range_constexpr(4):
                    for j in range_constexpr(n_tiles_b):
                        t = ti * n_tiles_b + j
                        packed = _pack_out_pair(
                            Vec(frag_even[t])[r] + res[r][j][0].to(fx.Float32),
                            Vec(frag_odd[t])[r] + res[r][j][1].to(fx.Float32),
                            self.out_ty,
                        )
                        buffer_store(
                            Vec.from_elements([packed], fx.Int32).bitcast(self.out_ty),
                            rsrc,
                            offs[r] + j * 64,
                            mask=col_ok[j],
                            cache_modifier=self.cache_modifier,
                            offset_is_bytes=True,
                        )

        def _unsupported(self, *args, **kwargs):
            raise NotImplementedError("the residual epilogue only covers full 256-wide column tiles")

        store = store_band16 = store_trans16 = _unsupported

    return _StoreCBf16Residual


def _copy_rows(thread_index, num_vecs, vec_coords, src_res, dst_res, src_cm, dst_cm):
    """Copy ``num_vecs`` 16 B vectors; ``vec_coords(v)`` gives ``(src_word, dst_word, valid)``."""
    per_iter = BLOCK_THREADS * _BATCH
    for it in range_constexpr(ceildiv(num_vecs, per_iter)):
        coords = [
            vec_coords(fx.Int32(it * per_iter + j * BLOCK_THREADS) + thread_index)
            for j in range_constexpr(_BATCH)
            if it * per_iter + j * BLOCK_THREADS < num_vecs
        ]
        vals = [
            buffer_load(src_res, src, vec_width=VEC_I32, dtype=fx.T.i32, mask=valid, cache_modifier=src_cm)
            for src, _, valid in coords
        ]
        for val, (_, dst, valid) in zip(vals, coords):
            buffer_store(val, dst_res, dst, mask=valid, cache_modifier=dst_cm)


def _o_row(t, dst, M, seq_local, num_ranks, is_bshd):
    """Row of ``O`` (full sequence) holding token ``t`` of ``dst``'s sequence slice."""
    if is_bshd:
        bi = t // fx.Int32(seq_local)
        return t + bi * fx.Int32(seq_local * (num_ranks - 1)) + dst * fx.Int32(seq_local)
    return dst * fx.Int32(M) + t


@ASTRewriter.transform
def _push_unit(thread_index, m_tile, dst, dst_heap, bank_offset, o_res, epoch, **c):
    """Rows of ``m_tile`` of ``dst``'s sequence slice into ``dst``'s recv bank, then ``dst``'s flag."""
    M, K, K_r, BM, rank, num_ranks = c["M"], c["K"], c["K_r"], c["BM"], c["rank"], c["num_ranks"]
    vecs_per_row = K_r // VEC_BF16
    recv_res = create_buffer_resource_from_addr(dst_heap + bank_offset, num_records_bytes=c["M_pad"] * K * 2)

    def coords(v):
        r = v // fx.Int32(vecs_per_row)
        col = (v % fx.Int32(vecs_per_row)) * fx.Int32(VEC_I32)
        t = m_tile * fx.Int32(BM) + r
        o_row = _o_row(t, dst, M, c["seq_local"], num_ranks, c["is_bshd"])
        src_word = o_row * fx.Int32(K_r // 2) + col
        dst_word = t * fx.Int32(K // 2) + fx.Int32(rank * K_r // 2) + col
        return src_word, dst_word, (r < fx.Int32(BM)) & (t < fx.Int32(M))

    _copy_rows(thread_index, BM * vecs_per_row, coords, o_res, recv_res, 0, SYS_NT)
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    if thread_index == fx.Int32(0):
        flag_base = dst_heap + fx.Int64(c["flag_region"] * FLAG_REGION_BYTES)
        st(flag_base, m_tile * fx.Int32(num_ranks) + fx.Int32(rank), epoch, scope="sys", order="release")


@ASTRewriter.transform
def _push_loop(thread_index, state_base, claim_slot, peers_base, bank_offset, o_res, epoch, **c):
    """Claim ``(m_tile, dst = rank + i)`` units until none are left and push each one."""
    num_ranks = c["num_ranks"]
    num_units = num_ranks * ceildiv(c["M"], c["BM"])
    unit = claim_unit(thread_index, state_base, claim_slot)
    while unit < fx.Int32(num_units):
        m_tile = unit // fx.Int32(num_ranks)
        dst = (fx.Int32(c["rank"]) + unit % fx.Int32(num_ranks)) % fx.Int32(num_ranks)
        dst_heap = ld(peers_base, dst, dtype=fx.T.i64)
        _push_unit(thread_index, m_tile, dst, dst_heap, bank_offset, o_res, epoch, **c)
        unit = claim_unit(thread_index, state_base, claim_slot)


@ASTRewriter.transform
def _wait_m_tile(thread_index, flag_base, m_tile, epoch, rank, num_ranks):
    """Every source's rows of ``m_tile`` have landed in this rank's recv bank."""
    if thread_index == fx.Int32(0):
        for src in range_constexpr(num_ranks):
            spin_flag(flag_base, m_tile * fx.Int32(num_ranks) + fx.Int32(src), epoch, rank, src, "a2a_gemm")
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    # recv rows are rewritten by peers every 2 epochs: drop anything this CU still caches
    memory_fence(order="acquire", scope="sys")


@functools.lru_cache(maxsize=64)
def _make_kernel(
    M,
    K,
    N,
    BM,
    BN,
    rank,
    num_ranks,
    seq_local,
    is_bshd,
    has_residual,
    save_o,
    num_push,
    flag_region,
    bank_bytes,
    nt_vmcnt,
):
    assert K % num_ranks == 0 and N % VEC_BF16 == 0
    K_r = K // num_ranks
    assert K_r % VEC_BF16 == 0
    m_tiles = ceildiv(M, BM)
    M_pad = m_tiles * BM
    n_tiles = ceildiv(N, BN)
    num_tiles = m_tiles * n_tiles
    assert m_tiles * num_ranks <= FLAGS_PER_REGION, "too many m-tiles for one flag region"
    assert M_pad * K < 2**31 and M * N < 2**31 and K * N < 2**31, "i32 element offsets overflow"
    assert M_pad * K * 2 <= bank_bytes
    assert not has_residual or (N % BN == 0 and BN == 256), "the residual epilogue needs N % 256 == 0"
    num_push = min(num_push, num_tiles)
    grid = num_tiles
    gemm_tile = functools.partial(gemm_bf16_tile, "nt", pair_n=True)
    # one extra i32: the block-wide broadcast of a claimed push unit
    SharedStorage = _make_shared_storage(BM, BN, num_lds_row_idx_entries=1)
    consts = dict(
        M=M,
        M_pad=M_pad,
        K=K,
        K_r=K_r,
        BM=BM,
        rank=rank,
        num_ranks=num_ranks,
        seq_local=seq_local,
        is_bshd=is_bshd,
        flag_region=flag_region,
    )
    # O_gathered copy-out: each of an m-tile's n_tiles blocks takes one column chunk of its rows
    og_vecs_per_row = K // VEC_BF16
    og_chunk = ceildiv(og_vecs_per_row, n_tiles)

    @flyc.kernel(name="ulysses_a2a_gemm_kernel", known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        O: fx.Tensor,
        W: fx.Tensor,
        RES: fx.Tensor,
        Y: fx.Tensor,
        OG: fx.Tensor,
        PEER_PTRS: fx.Tensor,
        STATE: fx.Tensor,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        claim_slot = fx.Int32(fx.ptrtoint(lds.lds_row_idx.ptr))
        state_base = i64_base(STATE)
        peers_base = i64_base(PEER_PTRS)
        epoch = ld(state_base, fx.Int32(0), dtype=fx.T.i64)
        bank_offset = fx.Int64(FLAG_BYTES) + (epoch & fx.Int64(1)) * fx.Int64(bank_bytes)
        my_base = ld(peers_base, fx.Int32(rank), dtype=fx.T.i64)

        if block_index < fx.Int32(num_push):
            o_res = create_buffer_resource(O, num_records_bytes=num_ranks * M * K_r * 2)
            _push_loop(thread_index, state_base, claim_slot, peers_base, bank_offset, o_res, epoch, **consts)
        # pushers take the last tiles: their rows are the last to arrive anyway
        tile_index = arith.select(
            block_index < fx.Int32(num_push),
            block_index + fx.Int32(num_tiles - num_push),
            block_index - fx.Int32(num_push),
        )
        tile_m = tile_index // fx.Int32(n_tiles)
        tile_n = tile_index % fx.Int32(n_tiles)
        flag_base = my_base + fx.Int64(flag_region * FLAG_REGION_BYTES)
        _wait_m_tile(thread_index, flag_base, tile_m, epoch, rank, num_ranks)

        recv_addr = my_base + bank_offset
        epilogue = {}
        if const_expr(has_residual):
            epilogue["store_c_cls"] = _residual_store_c(RES)
        if const_expr(save_o):
            recv_res = create_buffer_resource_from_addr(recv_addr, num_records_bytes=M_pad * K * 2)
            og_res = create_buffer_resource(OG, num_records_bytes=M * K * 2)

            def og_coords(v):
                r = v // fx.Int32(og_chunk)
                t = tile_m * fx.Int32(BM) + r
                col = tile_n * fx.Int32(og_chunk) + v % fx.Int32(og_chunk)
                word = t * fx.Int32(K // 2) + col * fx.Int32(VEC_I32)
                valid = (r < fx.Int32(BM)) & (t < fx.Int32(M)) & (col < fx.Int32(og_vecs_per_row))
                return word, word, valid

            _copy_rows(thread_index, BM * og_chunk, og_coords, recv_res, og_res, SYS_NT, 0)

        gemm_tile(
            make_bf16_fp16_tile_tensor(recv_addr, fx.Int64(0), M_pad * K),
            make_bf16_fp16_tile_tensor(i64_base(W), fx.Int64(0), K * N),
            make_bf16_fp16_tile_tensor(i64_base(Y), fx.Int64(0), M * N),
            fx.Int32(M),
            fx.Int32(N),
            lds,
            tile_m,
            tile_n,
            K=K,
            BLOCK_M=BM,
            BLOCK_N=BN,
            out_fp16=False,
            nt_vmcnt=nt_vmcnt,
            n_tail=N % BN,
            **epilogue,
        )
        exit_and_bump_epoch(thread_index, state_base, epoch, grid)

    return kernel, grid


@flyc.jit
def _launch(
    O,
    W,
    RES,
    Y,
    OG,
    PEER_PTRS,
    STATE,
    M: fx.Constexpr[int],
    K: fx.Constexpr[int],
    N: fx.Constexpr[int],
    BM: fx.Constexpr[int],
    BN: fx.Constexpr[int],
    rank: fx.Constexpr[int],
    num_ranks: fx.Constexpr[int],
    seq_local: fx.Constexpr[int],
    is_bshd: fx.Constexpr[bool],
    has_residual: fx.Constexpr[bool],
    save_o: fx.Constexpr[bool],
    num_push: fx.Constexpr[int],
    flag_region: fx.Constexpr[int],
    bank_bytes: fx.Constexpr[int],
    nt_vmcnt: fx.Constexpr[int],
    stream: fx.Stream,
):
    kernel, grid = _make_kernel(
        M,
        K,
        N,
        BM,
        BN,
        rank,
        num_ranks,
        seq_local,
        is_bshd,
        has_residual,
        save_o,
        num_push,
        flag_region,
        bank_bytes,
        nt_vmcnt,
    )
    kernel(
        O,
        W,
        RES,
        Y,
        OG,
        PEER_PTRS,
        STATE,
        value_attrs=make_value_attrs(2, 0, "512,512"),
    ).launch(grid=(grid, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)


def a2a_gemm_bank_bytes(M, K, BM=256):
    return ceildiv(M, BM) * BM * K * 2


def ulysses_a2a_gemm(
    o,
    w,
    residual,
    y,
    o_gathered,
    workspace,
    *,
    seq_local,
    qkv_format,
    flag_region,
    BM=256,
    BN=256,
    num_push=32,
):
    """``o`` ``[n * M, K_r]`` this rank's heads over the full sequence (``qkv_format`` row order);
    ``w`` ``[N, K]`` with ``K = n * K_r``; ``y = A @ w^T (+ residual)`` ``[M, N]`` where ``A`` is the
    gathered ``[M, K]``, also copied to ``o_gathered`` unless it is ``None``."""
    num_ranks = workspace.num_ranks
    rows, K_r = o.shape
    N, K = w.shape
    M = rows // num_ranks
    assert K == num_ranks * K_r and tuple(y.shape) == (M, N)
    assert o.is_contiguous() and w.is_contiguous() and y.is_contiguous()
    assert residual is None or (residual.is_contiguous() and residual.shape == y.shape)
    assert o_gathered is None or (o_gathered.is_contiguous() and tuple(o_gathered.shape) == (M, K))
    run_launcher(
        _launch,
        o,
        w,
        y if residual is None else residual,
        y,
        y if o_gathered is None else o_gathered,
        workspace.peer_ptrs,
        workspace.state,
        M,
        K,
        N,
        BM,
        BN,
        workspace.rank,
        num_ranks,
        seq_local,
        qkv_format == "bshd",
        residual is not None,
        o_gathered is not None,
        num_push,
        flag_region,
        workspace.bank_bytes,
        3,
        torch.cuda.current_stream(),
    )
