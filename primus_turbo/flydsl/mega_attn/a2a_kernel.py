###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The two Ulysses all-to-alls of the fused attention backward (single node, symmetric memory).

* ``seq -> heads`` (dO): every block stages this rank's rows ``[M, n * h, d]`` into its bank as one
  column block per destination rank, then pulls this rank's column block of every source into
  ``[S, h_r, d]`` in sequence order (the forward kernel's stage / pull role);
* ``heads -> seq`` (dq / dk / dv): every block pushes rows ``[BM)`` of ``[S, h_r, d]`` into the recv bank
  of the rank owning those tokens and raises that rank's flag ``(m-tile, rank)``; once every push unit is
  claimed the blocks claim gather tiles, wait for every source's flag and copy the recv rows into
  ``[M, n * h_r, d]``.

Several tensors move in one launch (their heads side by side in a column block). The sequence split is
the forward kernel's (``seq_chunk``).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr

from primus_turbo.flydsl.deep_ep.launch_cache import run_launcher
from primus_turbo.flydsl.mega_attn.a2a_attn_kernel import BM, _pull_role
from primus_turbo.flydsl.ulysses.a2a_gemm_kernel import _copy_rows
from primus_turbo.flydsl.ulysses.common import (
    BLOCK_THREADS,
    STATE_ASSIST,
    STATE_CLAIM,
    STATE_TILE,
    SYS_NT,
    VEC_BF16,
    VEC_I32,
    claim_unit,
    exit_and_bump_epoch,
    i64_base,
    opaque_tid,
    spin_flags,
)
from primus_turbo.flydsl.ulysses.workspace import (
    FLAG_BYTES,
    FLAG_REGION_BYTES,
    FLAGS_PER_REGION,
)
from primus_turbo.flydsl.utils.buffer_ops import (
    create_buffer_resource,
    create_buffer_resource_from_addr,
)
from primus_turbo.flydsl.utils.prims import ld, memory_fence, st

REGION_BWD_STAGE = 3
REGION_BWD_READY = 4
REGION_BWD_PUSH = 5

MAX_TENSORS = 3


def a2a_bank_bytes(M, num_ranks, heads, head_dim):
    """Bank bytes of one exchange of ``heads`` (per-rank head counts of each tensor)."""
    return M * num_ranks * sum(heads) * head_dim * 2


@ASTRewriter.transform
def _push_unit(thread_index, unit, peers_base, bank_offset, in_res, epoch, **c):
    """Rows ``[unit * BM, +BM)`` of the sequence into the owner's recv bank (this rank's column block),
    then the owner's flag ``(m-tile, rank)``."""
    n, d, rank, cs = c["num_ranks"], c["head_dim"], c["rank"], c["seq_chunk"]
    heads, offs = c["heads"], c["head_offs"]
    row_vecs = n * c["N_r"] // VEC_BF16
    blk_vecs = c["N_r"] // VEC_BF16
    g0 = unit * fx.Int32(BM)
    chunk = g0 // fx.Int32(cs)
    owner = chunk % fx.Int32(n)
    t0 = chunk // fx.Int32(n) * fx.Int32(cs) + g0 % fx.Int32(cs)
    owner_heap = ld(peers_base, owner, dtype=fx.T.i64)
    recv_res = create_buffer_resource_from_addr(owner_heap + bank_offset, num_records_bytes=c["bank_used"])
    for x in range_constexpr(len(heads)):
        h_vecs = heads[x] * d // VEC_BF16
        off = rank * blk_vecs + offs[x] * d // VEC_BF16

        def coords(v, h_vecs=h_vecs, off=off):
            r = v // fx.Int32(h_vecs)
            col = v % fx.Int32(h_vecs)
            src_word = ((g0 + r) * fx.Int32(h_vecs) + col) * fx.Int32(VEC_I32)
            dst_word = ((t0 + r) * fx.Int32(row_vecs) + fx.Int32(off) + col) * fx.Int32(VEC_I32)
            return src_word, dst_word, r < fx.Int32(BM)

        _copy_rows(thread_index, BM * h_vecs, coords, in_res[x], recv_res, 0, SYS_NT)
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    if thread_index == fx.Int32(0):
        flag_base = owner_heap + fx.Int64(REGION_BWD_PUSH * FLAG_REGION_BYTES)
        st(flag_base, t0 // fx.Int32(BM) * fx.Int32(n) + fx.Int32(rank), epoch, scope="sys", order="release")


@ASTRewriter.transform
def _gather_unit(thread_index, m_tile, my_base, bank_offset, out_res, epoch, **c):
    """Recv rows ``[m_tile * BM, +BM)`` into the ``[M, n * h_r, d]`` outputs once every source pushed them."""
    n, d = c["num_ranks"], c["head_dim"]
    heads, offs = c["heads"], c["head_offs"]
    row_vecs = n * c["N_r"] // VEC_BF16
    blk_vecs = c["N_r"] // VEC_BF16
    if thread_index == fx.Int32(0):
        flag_base = my_base + fx.Int64(REGION_BWD_PUSH * FLAG_REGION_BYTES)
        flag_idxs = [m_tile * fx.Int32(n) + fx.Int32(i) for i in range(n)]
        spin_flags(flag_base, flag_idxs, epoch, c["rank"], -1, "a2a heads->seq gather")
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    # recv rows are rewritten by peers every 2 epochs: drop what this CU / XCD still caches of them
    memory_fence(order="acquire", scope="sys")
    recv_res = create_buffer_resource_from_addr(my_base + bank_offset, num_records_bytes=c["bank_used"])
    t0 = m_tile * fx.Int32(BM)
    for x in range_constexpr(len(heads)):
        h_vecs = heads[x] * d // VEC_BF16
        row_vecs_out = n * h_vecs
        off = offs[x] * d // VEC_BF16

        def coords(v, h_vecs=h_vecs, row_vecs_out=row_vecs_out, off=off):
            r = v // fx.Int32(row_vecs_out)
            col = v % fx.Int32(row_vecs_out)
            src_col = col // fx.Int32(h_vecs) * fx.Int32(blk_vecs) + fx.Int32(off) + col % fx.Int32(h_vecs)
            src_word = ((t0 + r) * fx.Int32(row_vecs) + src_col) * fx.Int32(VEC_I32)
            dst_word = ((t0 + r) * fx.Int32(row_vecs_out) + col) * fx.Int32(VEC_I32)
            return src_word, dst_word, r < fx.Int32(BM)

        _copy_rows(thread_index, BM * row_vecs_out, coords, recv_res, out_res[x], SYS_NT, 0)
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()


@ASTRewriter.transform
def _push_gather(
    thread_index, state_base, claim_slot, peers_base, my_base, bank_offset, in_res, out_res, epoch, **c
):
    """Push units until none are left, then gather tiles."""
    m_tiles, num_push = c["m_tiles"], c["m_tiles"] * c["num_ranks"]
    more_push = fx.Int32(1)
    while more_push == fx.Int32(1):
        unit = claim_unit(opaque_tid(thread_index), state_base, claim_slot, STATE_CLAIM)
        if unit < fx.Int32(num_push):
            _push_unit(thread_index, unit, peers_base, bank_offset, in_res, epoch, **c)
        more_push = arith.select(unit < fx.Int32(num_push), fx.Int32(1), fx.Int32(0))
    more_gather = fx.Int32(1)
    while more_gather == fx.Int32(1):
        m_tile = claim_unit(opaque_tid(thread_index), state_base, claim_slot, STATE_TILE)
        if m_tile < fx.Int32(m_tiles):
            _gather_unit(thread_index, m_tile, my_base, bank_offset, out_res, epoch, **c)
        more_gather = arith.select(m_tile < fx.Int32(m_tiles), fx.Int32(1), fx.Int32(0))


@functools.lru_cache(maxsize=64)
def _make_kernel(direction, M, rank, num_ranks, h0, h1, h2, head_dim, seq_chunk, num_blocks, bank_bytes):
    n, d = num_ranks, head_dim
    heads = tuple(h for h in (h0, h1, h2) if h)
    cs = seq_chunk if seq_chunk else M
    assert M % cs == 0 and cs % BM == 0, f"need M % seq_chunk == 0 and seq_chunk % {BM} == 0"
    assert 1 <= len(heads) <= MAX_TENSORS
    S = M * n
    m_tiles = M // BM
    N_r = sum(heads) * d
    bank_used = a2a_bank_bytes(M, n, heads, d)
    assert bank_used <= bank_bytes and bank_used < 2**31 and S * max(heads) * d * 2 < 2**31, (
        "i32 byte offsets overflow"
    )
    assert m_tiles * n <= FLAGS_PER_REGION
    head_offs = tuple(sum(heads[:x]) for x in range(len(heads)))
    c = dict(
        M=M,
        rank=rank,
        num_ranks=n,
        head_dim=d,
        heads=heads,
        head_offs=head_offs,
        N_r=N_r,
        seq_chunk=cs,
        m_tiles=m_tiles,
        stage_bytes=bank_used,
        bank_used=bank_used,
        stage_region=REGION_BWD_STAGE,
        ready_region=REGION_BWD_READY,
    )

    @fx.struct
    class SharedStorage:
        claim: fx.Array[fx.Int32, 4, 16]

    @flyc.kernel(name=f"ulysses_a2a_{direction}_kernel", known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        X0: fx.Tensor,
        X1: fx.Tensor,
        X2: fx.Tensor,
        Y0: fx.Tensor,
        Y1: fx.Tensor,
        Y2: fx.Tensor,
        PEER_PTRS: fx.Tensor,
        STATE: fx.Tensor,
    ):
        thread_index = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        claim_slot = fx.Int32(fx.ptrtoint(lds.claim.ptr))
        state_base = i64_base(STATE)
        peers_base = i64_base(PEER_PTRS)
        epoch = ld(state_base, fx.Int32(0), dtype=fx.T.i64)
        bank_offset = fx.Int64(FLAG_BYTES) + (epoch & fx.Int64(1)) * fx.Int64(bank_bytes)
        my_base = ld(peers_base, fx.Int32(rank), dtype=fx.T.i64)
        xs = (X0, X1, X2)[: len(heads)]
        ys = (Y0, Y1, Y2)[: len(heads)]
        # [M, n * h, d] and [S, h, d] are the same bytes
        in_res = [create_buffer_resource(t, num_records_bytes=S * h * d * 2) for t, h in zip(xs, heads)]
        out_res = [create_buffer_resource(t, num_records_bytes=S * h * d * 2) for t, h in zip(ys, heads)]
        if const_expr(direction == "seq_to_heads"):
            _pull_role(
                thread_index,
                state_base,
                claim_slot,
                peers_base,
                my_base,
                bank_offset,
                in_res,
                out_res,
                epoch,
                None,
                **c,
            )
        else:
            _push_gather(
                thread_index,
                state_base,
                claim_slot,
                peers_base,
                my_base,
                bank_offset,
                in_res,
                out_res,
                epoch,
                **c,
            )
        exit_and_bump_epoch(thread_index, state_base, epoch, num_blocks, extra_counters=(STATE_ASSIST,))

    value_attrs = {"rocdl.flat_work_group_size": f"{BLOCK_THREADS},{BLOCK_THREADS}"}
    return kernel, value_attrs


@flyc.jit
def _launch(
    X0,
    X1,
    X2,
    Y0,
    Y1,
    Y2,
    PEER_PTRS,
    STATE,
    direction: fx.Constexpr[str],
    M: fx.Constexpr[int],
    rank: fx.Constexpr[int],
    num_ranks: fx.Constexpr[int],
    h0: fx.Constexpr[int],
    h1: fx.Constexpr[int],
    h2: fx.Constexpr[int],
    head_dim: fx.Constexpr[int],
    seq_chunk: fx.Constexpr[int],
    num_blocks: fx.Constexpr[int],
    bank_bytes: fx.Constexpr[int],
    stream: fx.Stream,
):
    kernel, value_attrs = _make_kernel(
        direction, M, rank, num_ranks, h0, h1, h2, head_dim, seq_chunk, num_blocks, bank_bytes
    )
    kernel(X0, X1, X2, Y0, Y1, Y2, PEER_PTRS, STATE, value_attrs=value_attrs).launch(
        grid=(num_blocks, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream
    )


def _run(direction, xs, ys, workspace, seq_chunk, num_blocks):
    n = workspace.num_ranks
    assert 1 <= len(xs) == len(ys) <= MAX_TENSORS
    assert all(t.is_contiguous() and t.dtype == torch.bfloat16 for t in (*xs, *ys))
    d = xs[0].shape[-1]
    if direction == "seq_to_heads":
        M = xs[0].shape[0]
        heads = tuple(x.shape[-2] // n for x in xs)
        assert all(x.shape[-2] % n == 0 and x.shape[0] == M for x in xs)
        assert all(y.shape == (M * n, h, d) for y, h in zip(ys, heads))
    else:
        M = ys[0].shape[0]
        heads = tuple(x.shape[-2] for x in xs)
        assert all(x.shape == (M * n, h, d) for x, h in zip(xs, heads))
        assert all(y.shape == (M, n * h, d) for y, h in zip(ys, heads))
    if num_blocks is None:
        num_blocks = torch.cuda.get_device_properties(xs[0].device).multi_processor_count
    pad = (xs[0],) * (MAX_TENSORS - len(xs))
    run_launcher(
        _launch,
        *xs,
        *pad,
        *ys,
        *((ys[0],) * (MAX_TENSORS - len(ys))),
        workspace.peer_ptrs,
        workspace.state,
        direction,
        M,
        workspace.rank,
        n,
        *heads,
        *((0,) * (MAX_TENSORS - len(heads))),
        d,
        seq_chunk,
        num_blocks,
        workspace.bank_bytes,
        torch.cuda.current_stream(),
    )


def ulysses_a2a_seq_to_heads(xs, ys, workspace, *, seq_chunk=0, num_blocks=None):
    """``xs[i]`` ``[M, n * h_i, d]`` (this rank's tokens, every head) into ``ys[i]`` ``[S, h_i, d]`` (this
    rank's heads over the sequence, sequence order). The bank must hold ``a2a_bank_bytes``."""
    _run("seq_to_heads", tuple(xs), tuple(ys), workspace, seq_chunk, num_blocks)


def ulysses_a2a_heads_to_seq(xs, ys, workspace, *, seq_chunk=0, num_blocks=None):
    """``xs[i]`` ``[S, h_i, d]`` (this rank's heads over the sequence) into ``ys[i]`` ``[M, n * h_i, d]``."""
    _run("heads_to_seq", tuple(xs), tuple(ys), workspace, seq_chunk, num_blocks)
