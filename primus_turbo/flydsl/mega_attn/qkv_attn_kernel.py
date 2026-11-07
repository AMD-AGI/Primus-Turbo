###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""QKV GemmA2A and causal forward attention in one kernel (single node, symmetric memory).

Grid, in dispatch order: GEMM tiles and pull blocks exactly as ``ulysses.gemm_a2a_kernel`` (the QKV
projection scattered into this rank's q / k / v heads over the full sequence), then one persistent
attention worker per CU. A pull unit ``(m_tile, src)`` publishes a local ready flag once its rows are
written; a worker claims ``(head group, q block)`` items and waits for the flags covering the q block's
causal KV prefix before running the dual-wave tile on it. Workers are dispatched after every GEMM tile
and pull block, so none of them holds a CU that a producer still needs.

With a striped sequence split the pull units arrive m-major, i.e. in sequence order, so the prefix a
q block needs is complete early for small q blocks: workers take a short ascending run of q blocks
(they overlap the exchange tail), then the rest longest first.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith

from primus_turbo.flydsl.attention.flash_attn_fwd import (
    DUALWAVE_SWP_COMPILE_HINTS,
    make_dualwave_swp_tile,
)
from primus_turbo.flydsl.deep_ep.common import _asm
from primus_turbo.flydsl.deep_ep.launch_cache import run_launcher
from primus_turbo.flydsl.mega.bf16.gemm_bf16_kernel import (
    _make_shared_storage,
    gemm_bf16_tile,
)
from primus_turbo.flydsl.mega_attn.attn_fwd_persistent import make_traits
from primus_turbo.flydsl.ulysses.common import (
    BLOCK_THREADS,
    SPIN_TIMEOUT,
    STATE_TILE,
    SYS_NT,
    claim_unit,
    exit_and_bump_epoch,
    i64_base,
)
from primus_turbo.flydsl.ulysses.gemm_a2a_kernel import _pull_unit
from primus_turbo.flydsl.ulysses.workspace import (
    FLAG_BYTES,
    FLAG_REGION_BYTES,
    FLAGS_PER_REGION,
    REGION_QKV_GEMM_A2A,
    REGION_QKV_READY,
)
from primus_turbo.flydsl.utils.attn_helper import (
    DualwaveKernelContext,
    dtype_to_elem_type,
)
from primus_turbo.flydsl.utils.buffer_ops import create_buffer_resource
from primus_turbo.flydsl.utils.gemm_helper import make_bf16_fp16_tile_tensor
from primus_turbo.flydsl.utils.prims import (
    ceildiv,
    ld,
    read_clock,
    spin_timed_out,
    st,
)


@ASTRewriter.transform
def _pull_loop(
    thread_index,
    state_base,
    claim_slot,
    peers_base,
    ready_base,
    bank_offset,
    out_res,
    epoch,
    trace_base,
    trace_pull_at=0,
    **c,
):
    """``gemm_a2a_kernel._pull_loop`` plus a local ready flag per finished unit."""
    num_ranks, rank = c["num_ranks"], c["rank"]
    num_units = num_ranks * ceildiv(c["M"], c["BM"])
    unit = claim_unit(thread_index, state_base, claim_slot)
    while unit < fx.Int32(num_units):
        m_tile = unit // fx.Int32(num_ranks)
        src = (fx.Int32(rank) + unit % fx.Int32(num_ranks)) % fx.Int32(num_ranks)
        src_heap = ld(peers_base, src, dtype=fx.T.i64)
        _pull_unit(thread_index, src, m_tile, src_heap, bank_offset, out_res, epoch, **c)
        fx.rocdl.s_waitcnt(0)
        fx.gpu.barrier()
        if thread_index == fx.Int32(0):
            st(ready_base, m_tile * fx.Int32(num_ranks) + src, epoch, scope="agent", order="release")
            if const_expr(trace_base is not None):
                st(trace_base, fx.Int32(trace_pull_at) + unit, read_clock())
        unit = claim_unit(thread_index, state_base, claim_slot)


@ASTRewriter.transform
def _wait_ready(thread_index, ready_base, num_flags, epoch, rank):
    """Every thread polls its own share of flags ``[0, num_flags)``; then the block rendezvous."""
    idx = thread_index
    spin_start = read_clock()
    while idx < num_flags:
        flag = ld(ready_base, idx, scope="agent", order="acquire", dtype=fx.T.i64)
        while flag < epoch:
            fx.rocdl.s_sleep(fx.Int32(1))
            if spin_timed_out(spin_start, SPIN_TIMEOUT):
                fx.printf(f"[mega_attn rank={rank}] ready flag timeout: idx={{}} flag={{}}\n", idx, flag)
                _asm("s_trap 2")
            flag = ld(ready_base, idx, scope="agent", order="acquire", dtype=fx.T.i64)
        idx = idx + fx.Int32(BLOCK_THREADS)
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()


@ASTRewriter.transform
def _attn_loop(thread_index, state_base, claim_slot, ready_base, epoch, run_tile, trace_base, **c):
    """Claim ``(head group, q block)`` items: ``num_ascending`` q blocks in sequence order, then the
    rest longest first; wait for the KV prefix of each, then run its tile."""
    nqb, hg = c["num_q_blocks"], c["head_groups"]
    num_units = nqb * hg
    span = c["seq_chunk"] * c["num_ranks"]
    unit = claim_unit(thread_index, state_base, claim_slot, counter=STATE_TILE)
    while unit < fx.Int32(num_units):
        k = unit // fx.Int32(hg)
        asc = k < fx.Int32(c["num_ascending"])
        qb = arith.select(asc, k, fx.Int32(nqb - 1 + c["num_ascending"]) - k)
        q_end = (qb + fx.Int32(1)) * fx.Int32(c["block_m"])
        # local tokens of every source rank that the prefix [0, q_end) can touch
        local = (q_end + fx.Int32(span - 1)) // fx.Int32(span) * fx.Int32(c["seq_chunk"])
        local = arith.select(local < fx.Int32(c["seq_local"]), local, fx.Int32(c["seq_local"]))
        m_need = (local + fx.Int32(c["BM"] - 1)) // fx.Int32(c["BM"])
        if const_expr(trace_base is not None):
            if thread_index == fx.Int32(0):
                st(trace_base, fx.Int32(c["trace_attn_at"]) + unit * fx.Int32(3), read_clock())
        _wait_ready(thread_index, ready_base, m_need * fx.Int32(c["num_ranks"]), epoch, c["rank"])
        if const_expr(trace_base is not None):
            if thread_index == fx.Int32(0):
                st(trace_base, fx.Int32(c["trace_attn_at"]) + unit * fx.Int32(3) + fx.Int32(1), read_clock())
        run_tile((unit % fx.Int32(hg), fx.Int32(nqb - 1) - qb, fx.Int32(0)))
        if const_expr(trace_base is not None):
            fx.rocdl.s_waitcnt(0)
            fx.gpu.barrier()
            if thread_index == fx.Int32(0):
                st(trace_base, fx.Int32(c["trace_attn_at"]) + unit * fx.Int32(3) + fx.Int32(2), read_clock())
        unit = claim_unit(thread_index, state_base, claim_slot, counter=STATE_TILE)


@functools.lru_cache(maxsize=32)
def _make_kernel(
    M,
    K,
    N,
    BM,
    BN,
    rank,
    num_ranks,
    num_pull_cu,
    num_tail_pull,
    num_lead,
    num_attn,
    head_dim,
    G,
    nq,
    seq_local,
    seq_chunk,
    bank_bytes,
    num_ascending,
    trace=False,
):
    hq, hkv = G * nq, G
    N_r = N // num_ranks
    assert N_r == G * (nq + 2) * head_dim
    m_tiles = ceildiv(M, BM)
    M_pad = m_tiles * BM
    n_tiles = ceildiv(N, BN)
    num_gemm_tiles = m_tiles * n_tiles
    assert num_gemm_tiles <= FLAGS_PER_REGION and m_tiles * num_ranks <= FLAGS_PER_REGION
    assert M_pad * N < 2**31 and M * K < 2**31 and K * N < 2**31, "i32 element offsets overflow"
    assert M == seq_local, "batch 1 only"
    num_lead = min(num_lead, num_gemm_tiles)
    seq = seq_local * num_ranks
    traits = make_traits(hq, hkv, head_dim, emit_lse=True)
    assert traits.BLOCK_SIZE == BLOCK_THREADS
    attn_tile = make_dualwave_swp_tile(traits)
    num_q_blocks = ceildiv(seq, traits.BLOCK_M)
    num_ascending = min(num_ascending, num_q_blocks)
    gemm_tile = functools.partial(gemm_bf16_tile, "nt", pair_n=True)
    first_attn = num_gemm_tiles + num_pull_cu + num_tail_pull
    grid = first_attn + num_attn

    GemmStorage = _make_shared_storage(BM, BN, num_lds_row_idx_entries=1)
    elem = dtype_to_elem_type(traits.DTYPE_STR)

    @fx.struct
    class AttnStorage:
        kv: fx.Array[elem, traits.LDS_KV_TOTAL_SIZE, 16]

    # the claim slot sits past the attention buffers, so a worker's claim never touches them
    assert traits.LDS_KV_TOTAL_SIZE * 2 <= 8 * (BM // 16 + BN // 16) * 512

    @fx.union
    class SharedStorage:
        gemm: GemmStorage
        attn: AttnStorage

    pull_consts = dict(
        M=M,
        M_pad=M_pad,
        N=N,
        N_r=N_r,
        BM=BM,
        BN=BN,
        head_dim=head_dim,
        G=G,
        nq=nq,
        nk=1,
        nv=1,
        seq_local=seq_local,
        is_bshd=False,
        seq_chunk=seq_chunk,
        rank=rank,
        num_ranks=num_ranks,
        flag_region=REGION_QKV_GEMM_A2A,
    )
    # debug timeline (s_memrealtime): GEMM tile end, pull unit end, then [claim, ready, end] per attention item
    trace_pull_at = num_gemm_tiles
    trace_attn_at = trace_pull_at + m_tiles * num_ranks
    attn_consts = dict(
        trace_attn_at=trace_attn_at,
        num_q_blocks=num_q_blocks,
        head_groups=hq // traits.Q_HEADS_PER_WG,
        num_ascending=num_ascending,
        block_m=traits.BLOCK_M,
        seq_local=seq_local,
        seq_chunk=seq_chunk if seq_chunk else seq_local,
        num_ranks=num_ranks,
        BM=BM,
        rank=rank,
    )

    @flyc.kernel(name="mega_attn_qkv_attn_kernel", known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        OUT_Q: fx.Tensor,
        OUT_K: fx.Tensor,
        OUT_V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        LSE: fx.Tensor,
        PEER_PTRS: fx.Tensor,
        STATE: fx.Tensor,
        TRACE: fx.Tensor,
    ):
        thread_index = fx.thread_idx.x
        trace_base = i64_base(TRACE) if trace else None
        block_index, _b, _c = fx.block_idx
        lds = fx.SharedAllocator().allocate(SharedStorage)
        gemm_lds, attn_lds = lds.gemm.peek(), lds.attn.peek()
        claim_slot = fx.Int32(fx.ptrtoint(gemm_lds.lds_row_idx.ptr))
        state_base = i64_base(STATE)
        peers_base = i64_base(PEER_PTRS)
        epoch = ld(state_base, fx.Int32(0), dtype=fx.T.i64)
        bank_offset = fx.Int64(FLAG_BYTES) + (epoch & fx.Int64(1)) * fx.Int64(bank_bytes)
        my_base = ld(peers_base, fx.Int32(rank), dtype=fx.T.i64)
        ready_base = my_base + fx.Int64(REGION_QKV_READY * FLAG_REGION_BYTES)

        def run_tile(block_ids):
            ctx = DualwaveKernelContext(
                traits,
                OUT_Q,
                OUT_K,
                OUT_V,
                O,
                LSE,
                O,
                O,
                O,
                fx.Int32(seq),
                fx.Int32(seq),
                fx.Int32(hq * head_dim),
                fx.Int32(hkv * head_dim),
                fx.Int32(head_dim),
                fx.Int32(0),
                SINK=O,
                ScoreBound=O,
            )
            ctx._setup(None, block_ids=block_ids, lds=attn_lds)
            attn_tile(ctx)

        if block_index >= fx.Int32(first_attn):
            _attn_loop(
                thread_index, state_base, claim_slot, ready_base, epoch, run_tile, trace_base, **attn_consts
            )
        else:
            is_pull = (
                (block_index >= fx.Int32(num_lead)) & (block_index < fx.Int32(num_lead + num_pull_cu))
            ) | (block_index >= fx.Int32(num_gemm_tiles + num_pull_cu))
            if is_pull:
                out_res = [
                    create_buffer_resource(t, num_records_bytes=num_ranks * M * h * head_dim * 2)
                    for t, h in ((OUT_Q, hq), (OUT_K, hkv), (OUT_V, hkv))
                ]
                _pull_loop(
                    thread_index,
                    state_base,
                    claim_slot,
                    peers_base,
                    ready_base,
                    bank_offset,
                    out_res,
                    epoch,
                    trace_base,
                    trace_pull_at=trace_pull_at,
                    **pull_consts,
                )
            else:
                gemm_index = arith.select(
                    block_index < fx.Int32(num_lead), block_index, block_index - fx.Int32(num_pull_cu)
                )
                tile_m = gemm_index // fx.Int32(n_tiles)
                tile_n = gemm_index % fx.Int32(n_tiles)
                A_t = make_bf16_fp16_tile_tensor(i64_base(A), fx.Int64(0), M * K)
                B_t = make_bf16_fp16_tile_tensor(i64_base(B), fx.Int64(0), K * N)
                C_t = make_bf16_fp16_tile_tensor(my_base, bank_offset, M_pad * N)
                gemm_tile(
                    A_t,
                    B_t,
                    C_t,
                    fx.Int32(M_pad),
                    fx.Int32(N),
                    gemm_lds,
                    tile_m,
                    tile_n,
                    K=K,
                    BLOCK_M=BM,
                    BLOCK_N=BN,
                    out_fp16=False,
                    nt_vmcnt=3,
                    n_tail=N % BN,
                    c_cache_modifier=SYS_NT,
                )
                fx.rocdl.s_waitcnt(0)
                fx.gpu.barrier()
                # keep a separator: LLVM folds adjacent barriers, but two rendezvous are required
                fx.rocdl.s_waitcnt(0)
                fx.gpu.barrier()
                if thread_index == fx.Int32(0):
                    flag_base = my_base + fx.Int64(REGION_QKV_GEMM_A2A * FLAG_REGION_BYTES)
                    st(flag_base, tile_m * fx.Int32(n_tiles) + tile_n, epoch, scope="sys", order="release")
                    if const_expr(trace):
                        st(trace_base, tile_m * fx.Int32(n_tiles) + tile_n, read_clock())

        exit_and_bump_epoch(thread_index, state_base, epoch, grid)

    value_attrs = {
        "rocdl.waves_per_eu": traits.WAVES_PER_EU,
        "rocdl.flat_work_group_size": f"{BLOCK_THREADS},{BLOCK_THREADS}",
        "llvm.passthrough": [
            ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
            ["no-nans-fp-math", "true"],
            ["unsafe-fp-math", "true"],
        ],
    }
    return kernel, grid, value_attrs


@flyc.jit
def _launch(
    A,
    B,
    OUT_Q,
    OUT_K,
    OUT_V,
    O,  # noqa: E741
    LSE,
    PEER_PTRS,
    STATE,
    TRACE,
    M: fx.Constexpr[int],
    K: fx.Constexpr[int],
    N: fx.Constexpr[int],
    BM: fx.Constexpr[int],
    BN: fx.Constexpr[int],
    rank: fx.Constexpr[int],
    num_ranks: fx.Constexpr[int],
    num_pull_cu: fx.Constexpr[int],
    num_tail_pull: fx.Constexpr[int],
    num_lead: fx.Constexpr[int],
    num_attn: fx.Constexpr[int],
    head_dim: fx.Constexpr[int],
    G: fx.Constexpr[int],
    nq: fx.Constexpr[int],
    seq_local: fx.Constexpr[int],
    seq_chunk: fx.Constexpr[int],
    bank_bytes: fx.Constexpr[int],
    num_ascending: fx.Constexpr[int],
    trace: fx.Constexpr[bool],
    stream: fx.Stream,
):
    kernel, grid, value_attrs = _make_kernel(
        M,
        K,
        N,
        BM,
        BN,
        rank,
        num_ranks,
        num_pull_cu,
        num_tail_pull,
        num_lead,
        num_attn,
        head_dim,
        G,
        nq,
        seq_local,
        seq_chunk,
        bank_bytes,
        num_ascending,
        trace,
    )
    kernel(A, B, OUT_Q, OUT_K, OUT_V, O, LSE, PEER_PTRS, STATE, TRACE, value_attrs=value_attrs).launch(
        grid=(grid, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream
    )


def ulysses_qkv_attn(
    a,
    w,
    q,
    k,
    v,
    o,
    lse,
    workspace,
    *,
    head_dim,
    groups,
    nq,
    seq_local,
    seq_chunk,
    BM=128,
    BN=256,
    num_pull_cu=24,
    num_tail_pull=8,
    num_ascending=None,
    trace=None,
):
    """``a`` [M, K] this rank's tokens (batch 1), ``w`` [N, K] interleaved QKV rows (``groups`` kv heads
    of this rank, ``nq`` q heads each). Writes q / k / v ``[n * M, h, d]`` (sequence order) and the causal
    attention output ``o`` (like ``q``) with its LSE ``[n * M, hq]`` fp32."""
    M, K = a.shape
    N = w.shape[0]
    assert all(t.is_contiguous() for t in (a, w, q, k, v, o, lse))
    num_cus = torch.cuda.get_device_properties(a.device).multi_processor_count
    if num_ascending is None:
        num_ascending = ceildiv(seq_local * workspace.num_ranks, 128) // 2
    with CompilationContext.compile_hints(DUALWAVE_SWP_COMPILE_HINTS):
        run_launcher(
            _launch,
            a,
            w,
            q,
            k,
            v,
            o,
            lse,
            workspace.peer_ptrs,
            workspace.state,
            workspace.state if trace is None else trace,
            M,
            K,
            N,
            BM,
            BN,
            workspace.rank,
            workspace.num_ranks,
            num_pull_cu,
            num_tail_pull,
            num_cus,
            num_cus,
            head_dim,
            groups,
            nq,
            seq_local,
            seq_chunk,
            workspace.bank_bytes,
            num_ascending,
            trace is not None,
            torch.cuda.current_stream(),
        )
