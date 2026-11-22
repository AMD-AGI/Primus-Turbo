###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""The all-gather-KV attention forward as one kernel: K / V exchange and both causal calls (gfx950).

One launch on one stream does what ``mega`` spreads over push, wait, attention and merge kernels on three
streams. Grid ``(Hq / Q_HEADS_PER_WG, c / BLOCK_M, 1 + 2B)``:

``z = 0``
    the first ``push_blocks`` blocks (lowest block ids, so dispatched before any attention block) store
    this rank's K / V halves into the peers' forward banks (``agkv_comm_kernel._push_loop``, the ``mega``
    units and flags), then copy what the backward needs, ``[ remote | own first | own second ]``, into
    ordinary memory: the own chunks at once, each remote chunk once its flag is up. The other blocks of
    the slice exit.
``z = 1 .. B``
    q1 (chunk 2n-1-r) of batch ``z - 1``: keys ``[ remote (rem) | own first | own second ]``.
``z = B + 1 .. 2B``
    q0 (chunk r): keys ``[ first halves of 0..r-1 (low) | own first ]``.

An attention block is the DUALWAVE_SWP flash-attention body with its key tiles reordered: own tiles first
(they need no wait, and the diagonal ones are the only masked ones), then the remote tiles from the last
to the first, so a chunk is entered at its last tile. Before loading the K tile a chunk is entered at,
every wave polls that chunk's flag (sender, half) until it reaches ``epoch``. Every remote key is visible
to every query, so the online softmax over the reordered tiles is the causal attention over all keys and
the block stores the final ``out`` / ``lse``; nothing merges afterwards.

The banks are ordinary (cached) memory. Every block runs a system-scope acquire before its first read, so
no XCD's L2 keeps lines of a bank a peer rewrote for this call: after that a line is only cached by a read
that observed its chunk's flag.
"""

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu
from flydsl.expr.utils.arith import ArithValue

from primus_turbo.flydsl.attention.agkv_comm_kernel import (
    PUSH_BATCH,
    UNIT_BYTES,
    WAIT_TIMEOUT,
    _push_loop,
    _unit_field,
)
from primus_turbo.flydsl.attention.flash_attn_fwd_stock import emit_dualwave_swp_body
from primus_turbo.flydsl.deep_ep.common import _asm
from primus_turbo.flydsl.fused_collective_gemm.common import BLOCK_THREADS, VEC_I32, copy_rows, i64_base
from primus_turbo.flydsl.utils.attn_helper_stock import (
    DualwaveKernelContext,
    _make_dualwave_swp_traits,
    dtype_to_elem_type,
)
from primus_turbo.flydsl.utils.buffer_ops import create_buffer_resource_from_addr
from primus_turbo.flydsl.utils.prims import cast, ld, memory_fence, read_clock, spin_timed_out, st

# i64 fields of a save unit, at the push unit's positions (fields 5 / 6 unused, flag -1: none)
S_SRC, S_SRC_OFF, S_DST, S_DST_OFF, S_BYTES = range(5)
S_FLAG = 7
# save unit sources: bank K, bank V, local K, local V; destinations: saved K, saved V
SAVE_SRC_BANK_K, SAVE_SRC_BANK_V, SAVE_SRC_K, SAVE_SRC_V = range(4)
# this rank's flag array, ``agkv_comm`` order: [first halves | second halves], indexed by sender
_FLAG_FIRST, _FLAG_SECOND = 0, 1


def supported(num_heads, num_kv_heads, head_dim, c):
    """Whether the fused forward handles this shape (else the caller keeps the unfused path)."""
    if head_dim != 128 or num_heads % num_kv_heads or (num_heads // num_kv_heads) % 2:
        return False
    return c % 128 == 0 and c >= 256


def bank_chunk_flag(n, r, k):
    """Flag index (into receiver ``r``'s array) of the sender half at chunk ``k`` of its forward bank,
    laid out ``[ second halves of r+1..n-1 | first halves of r+1..n-1 | first halves of 0..r-1 ]``."""
    n1 = n - 1 - r
    if k < n1:
        return _FLAG_SECOND * n + r + 1 + k
    if k < 2 * n1:
        return _FLAG_FIRST * n + r + 1 + k - n1
    return _FLAG_FIRST * n + k - 2 * n1


class _Geometry:
    def __init__(self, n, r, c, batch):
        self.n, self.r, self.c, self.batch = n, r, c, batch
        self.rem, self.low = (2 * n - 2 - r) * c, r * c


@ASTRewriter.transform
def _wait_flag(need, flag_base, flag_idx, epoch):
    """Poll flag ``flag_idx`` until it reaches ``epoch`` where ``need``; trap after ``WAIT_TIMEOUT``."""
    if need:
        spin_start = read_clock()
        flag = ld(flag_base, flag_idx, scope="sys", dtype=fx.T.i64)
        while flag < epoch:
            fx.rocdl.s_sleep(fx.Int32(1))
            if spin_timed_out(spin_start, WAIT_TIMEOUT):
                _asm("s_trap 2")
            flag = ld(flag_base, flag_idx, scope="sys", dtype=fx.T.i64)


@ASTRewriter.transform
def _spin_flag(flag_base, flag_idx, epoch):
    spin_start = read_clock()
    flag = ld(flag_base, flag_idx, scope="sys", dtype=fx.T.i64)
    while flag < epoch:
        fx.rocdl.s_sleep(fx.Int32(1))
        if spin_timed_out(spin_start, WAIT_TIMEOUT):
            _asm("s_trap 2")
        flag = ld(flag_base, flag_idx, scope="sys", dtype=fx.T.i64)


def _save_unit(thread_index, unit, srcs, dsts, units, block_threads):
    sel = _unit_field(units, unit, S_SRC)
    src_base = srcs[0]
    for i in range(1, len(srcs)):
        src_base = fx.arith.select(sel == fx.Int32(i), srcs[i], src_base)
    dst_base = fx.arith.select(_unit_field(units, unit, S_DST) == fx.Int32(1), dsts[1], dsts[0])
    nbytes = _unit_field(units, unit, S_BYTES)
    src_res = create_buffer_resource_from_addr(
        src_base + _unit_field(units, unit, S_SRC_OFF, wide=True), num_records_bytes=nbytes
    )
    dst_res = create_buffer_resource_from_addr(
        dst_base + _unit_field(units, unit, S_DST_OFF, wide=True), num_records_bytes=nbytes
    )
    nvec = nbytes // fx.Int32(16)

    def coords(v):
        word = v * fx.Int32(VEC_I32)
        return word, word, v < nvec

    for k in range(BLOCK_THREADS // block_threads):
        copy_rows(thread_index + fx.Int32(k * block_threads), UNIT_BYTES // 16, coords, src_res, dst_res, 0, 0, PUSH_BATCH)


@ASTRewriter.transform
def _save_loop(thread_index, block_index, srcs, dsts, units, flag_base, epoch, num_units, grid, block_threads):
    unit = block_index
    while unit < fx.Int32(num_units):
        flag = _unit_field(units, unit, S_FLAG)
        if flag >= fx.Int32(0):
            if thread_index == fx.Int32(0):
                _spin_flag(flag_base, flag, epoch)
            fx.gpu.barrier()
        _save_unit(thread_index, unit, srcs, dsts, units, block_threads)
        unit = unit + fx.Int32(grid)


@ASTRewriter.transform
def _comm_role(
    thread_index,
    block_index,
    srcs,
    heaps,
    flags,
    units,
    counters,
    bank_off,
    epoch,
    num_units,
    save_srcs,
    save_dsts,
    save_units,
    flag_base,
    num_save_units,
    grid,
    block_threads,
):
    if block_index < fx.Int32(grid):
        # before the first bank read, at kernel start: later it would drop the attention's lines too
        if const_expr(num_save_units > 0):
            memory_fence(order="acquire", scope="sys")
        if const_expr(not _DBG_NOPUSH):
            _push_loop(
                thread_index, block_index, srcs, heaps, flags, units, counters, bank_off, epoch, num_units, grid, None, 0,
                block_threads,
            )
        if const_expr(num_save_units > 0 and not _DBG_NOPUSH):
            _save_loop(
                thread_index, block_index, save_srcs, save_dsts, save_units, flag_base, epoch, num_save_units, grid,
                block_threads,
            )


class _AgkvFwdContext(DualwaveKernelContext):
    """DualwaveKernelContext of one (segment, batch) of the fused forward: own key tiles first, then the
    remote ones last to first, each remote chunk entered after its flag."""

    def __init__(self, traits, geo, Q, KL, VL, O, LSE, KB, VB, flag_base, epoch, *args):
        super().__init__(traits, Q, KL, VL, O, LSE, Q, Q, Q, *args)
        self.geo = geo
        self.KB, self.VB = KB, VB
        self.flag_base, self.epoch = flag_base, epoch

    def _grid_batch_index(self):
        z = fx.Index(gpu.block_idx.z) - fx.Index(1)
        # segment 0 = q1 (more work, dispatched first), 1 = q0
        self.segment = z // fx.Index(self.geo.batch)
        return z % fx.Index(self.geo.batch)

    def _dense_geometry(self):
        g = self.geo
        is_q1 = ArithValue(self.segment == fx.Index(0))

        def sel(a, b):
            return fx.Index(is_q1.select(fx.Index(a), fx.Index(b)))

        self.rem_tok = sel(g.rem, g.low)
        self.rem_tiles = self.rem_tok // fx.Index(self.traits.BLOCK_N)
        self.q_row0 = sel(g.c, 0)
        self.bank_tok0 = sel(0, g.rem - g.low)
        kv_len = self.rem_tok + sel(2 * g.c, g.c)
        # lse rows: [q0 | q1] x batch x c
        self.q_tok_base = (sel(1, 0) * fx.Index(g.batch) + self.batch_idx) * fx.Index(g.c)
        self.kv_tok_base = fx.Index(0)
        self.q_tok_end = self.q_tok_base + fx.Index(g.c)
        self.kv_tok_end = kv_len
        self.seqlen_q_v = self.seq_len_v
        self.seqlen_kv_v = kv_len
        self.seqlen_kv_i32 = fx.Int32(kv_len)

    @staticmethod
    def _div(tensor, base_i64, nrec_bytes, nelem):
        shifted = fx.inttoptr(fx.get_iter(tensor).type, base_i64)
        buf_ptr = fx.rocdl.make_buffer_ptr(shifted, num_records_bytes=fx.Int64(nrec_bytes))
        return fx.logical_divide(fx.make_view(buf_ptr, fx.make_layout(fx.Int32(nelem), fx.Int32(1))), fx.make_layout(1, 1))

    def _setup(self, shared_storage):
        super()._setup(shared_storage)
        traits, g = self.traits, self.geo
        bf16 = fx.Index(traits.BF16_BYTES)
        self.own_tiles = self.split_t_end - self.rem_tiles

        q_per = self.seqlen_q_v * self.stride_q_n_v
        q_off = (self.q_row0 * fx.Index(g.batch) + self.batch_idx) * fx.Index(traits.NUM_HEADS_Q * traits.HEAD_DIM)
        q_off_b = fx.Int64(q_off * bf16)
        self.q_div = self._div(self.Q, fx.Int64(i64_base(self.Q)) + q_off_b, q_per * bf16, q_per)
        self.o_div = self._div(self.O, fx.Int64(i64_base(self.O)) + q_off_b, q_per * bf16, q_per)

        tok_b = self.stride_kv_n_v * bf16
        batch_b = self.batch_idx * fx.Index(traits.NUM_HEADS_KV * traits.HEAD_DIM) * bf16
        rem_off = fx.Int64(self.bank_tok0 * tok_b + batch_b)
        self.kv_srcs = tuple(
            (fx.Int64(i64_base(bank)) + rem_off, fx.Int64(i64_base(local)) + fx.Int64(batch_b))
            for bank, local in ((self.KB, self.K), (self.VB, self.V))
        )
        self.rem_nrec = fx.Int64(self.rem_tok * tok_b)
        self.own_nrec = fx.Int64((self.seqlen_kv_v - self.rem_tok) * tok_b)
        self.kv_nelem = self.seqlen_kv_v * self.stride_kv_n_v

    def tile_start(self, tile_idx):
        t = fx.Index(tile_idx)
        own = ArithValue(t < self.own_tiles)
        phys = fx.Index(own.select(self.rem_tiles + t, self.rem_tiles + self.own_tiles - fx.Index(1) - t))
        return phys * fx.Index(self.traits.BLOCK_N)

    def causal_mask_prologue_if_needed(self, v_s, tile_idx=None, kv_end_pos=None, *args, kv_end_tile=None, **kw):
        # logical tile t + 1 is not the physical successor of t once the order turns into the remote keys
        if tile_idx is None:
            tile_idx = fx.Index(0)
        if kv_end_pos is None:
            kv_end_pos = self.tile_start(tile_idx) + fx.Index(self.traits.BLOCK_N)
        return super().causal_mask_prologue_if_needed(v_s, tile_idx, kv_end_pos, *args, **kw)

    def _source(self, tile_idx, which):
        tok = self.tile_start(tile_idx)
        remote = ArithValue(tok < self.rem_tok)
        rem_base, own_base = self.kv_srcs[which]
        base = fx.Int64(remote.select(rem_base, own_base))
        nrec = fx.Int64(remote.select(self.rem_nrec, self.own_nrec))
        rel = fx.Index(remote.select(tok, tok - self.rem_tok))
        return self._div(self.K, base, nrec, self.kv_nelem), rel

    def _wait_chunk(self, tile_idx):
        """Poll the flag of the remote chunk tile ``tile_idx`` enters (its last tile), if it enters one."""
        g = self.geo
        block_n = self.traits.BLOCK_N
        tok = self.tile_start(tile_idx)
        last = ArithValue((tok // fx.Index(block_n) + fx.Index(1)) % fx.Index(g.c // block_n) == fx.Index(0))
        need = ArithValue(tok < self.rem_tok) & last
        k = (tok + self.bank_tok0) // fx.Index(g.c)
        n1 = g.n - 1 - g.r
        second = fx.Index(_FLAG_SECOND * g.n + g.r + 1) + k
        first_hi = fx.Index(_FLAG_FIRST * g.n + g.r + 1 - n1) + k
        first_lo = k - fx.Index(2 * n1) + fx.Index(_FLAG_FIRST * g.n)
        idx = ArithValue(k < fx.Index(2 * n1)).select(first_hi, first_lo)
        idx = fx.Index(ArithValue(k < fx.Index(n1)).select(second, idx))
        _wait_flag(need, self.flag_base, fx.Int32(idx), self.epoch)

    def load_k_tile(self, tile_idx, buf_id):
        if const_expr(not _DBG_NOWAIT):
            self._wait_chunk(tile_idx)
        div, rel = self._source(tile_idx, 0)
        self._load_kv(rel, buf_id, self.k_dma_m0, div, self.NUM_DMA_K)

    def load_v_tile(self, tile_idx, buf_id):
        div, rel = self._source(tile_idx, 1)
        self._load_kv(rel, buf_id, self.v_dma_m0, div, self.NUM_DMA_V)


_DBG_NOWAIT = os.environ.get("AGKV_FUSED_DBG_NOWAIT") == "1"
_DBG_NOFENCE = os.environ.get("AGKV_FUSED_DBG_NOFENCE") == "1"
_DBG_NOPUSH = os.environ.get("AGKV_FUSED_DBG_NOPUSH") == "1"
_DBG_NOATTN = os.environ.get("AGKV_FUSED_DBG_NOATTN") == "1"
_DBG_TRACE = os.environ.get("AGKV_FUSED_DBG_TRACE") == "1"
TRACE_BUF = None
_TRACE_DUMMY = {}


def _hwreg(reg):
    from flydsl._mlir.dialects import llvm

    return fx.arith.ArithValue(
        llvm.inline_asm(fx.T.i32, [], f"s_getreg_b32 $0, hwreg({reg})", "=s", has_side_effects=True)
    )

_COMPILE_HINTS = {
    "fast_fp_math": True,
    "unsafe_fp_math": True,
    "llvm_options": {
        "amdgpu-sched-strategy": "max-memory-clause",
        "enable-post-misched": True,
        "lsr-drop-solution": True,
    },
}


@functools.lru_cache(maxsize=32)
def _build(num_heads, num_kv_heads, head_dim, n, r, c, batch, num_units, num_save_units, push_blocks, block_m, merge):
    traits = _make_dualwave_swp_traits(
        num_heads,
        num_kv_heads,
        head_dim,
        causal=True,
        dtype_str="bf16",
        waves_per_eu=2,
        dualwave_swp_enable_stagger=False,
        cross_seqlen=True,
        emit_lse=True,
        block_m=block_m,
        gqa_merge=merge,
        sbhd=True,
    )
    assert BLOCK_THREADS % traits.BLOCK_SIZE == 0, "the push / save units are sized for 512-thread blocks"
    geo = _Geometry(n, r, c, batch)
    grid_x, grid_y = num_heads // traits.Q_HEADS_PER_WG, c // traits.BLOCK_M
    grid = min(push_blocks, grid_x * grid_y, max(num_units, 1))
    lds_dtype = dtype_to_elem_type(traits.DTYPE_STR)
    cache_tag = traits.cache_tag + (n, r, c, batch, num_units, num_save_units, grid, _DBG_NOWAIT, _DBG_NOFENCE, _DBG_NOPUSH, _DBG_NOATTN, _DBG_TRACE)

    @fx.struct
    class SharedStorage:
        kv: fx.Array[lds_dtype, traits.LDS_KV_TOTAL_SIZE, 16]

    @flyc.kernel(name="agkv_fused_fwd_kernel", known_block_size=[traits.BLOCK_SIZE, 1, 1])
    def kernel(
        Q: fx.Tensor,
        KL: fx.Tensor,
        VL: fx.Tensor,
        KB: fx.Tensor,
        VB: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        LSE: fx.Tensor,
        SK: fx.Tensor,
        SV: fx.Tensor,
        HEAPS: fx.Tensor,
        FLAGS: fx.Tensor,
        UNITS: fx.Tensor,
        COUNTERS: fx.Tensor,
        SAVE_UNITS: fx.Tensor,
        TRACE: fx.Tensor,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        bank_off: fx.Int64,
        epoch: fx.Int64,
    ):
        z = fx.Index(gpu.block_idx.z)
        flag_base = ld(i64_base(FLAGS), fx.Int32(r), dtype=fx.T.i64)
        epoch_v = cast(epoch, fx.T.i64)
        if const_expr(_DBG_TRACE):
            lin = fx.Int32(
                (z * fx.Index(grid_y) + fx.Index(gpu.block_idx.y)) * fx.Index(grid_x) + fx.Index(gpu.block_idx.x)
            )
            t_start = read_clock()
            hw = fx.arith.extsi(fx.T.i64, _hwreg(4))
            xcc = fx.arith.extsi(fx.T.i64, _hwreg(20))

        def _comm():
            block = fx.Index(gpu.block_idx.y) * fx.Index(grid_x) + fx.Index(gpu.block_idx.x)
            _comm_role(
                cast(fx.thread_idx.x, fx.T.i32),
                fx.Int32(block),
                [i64_base(KL), i64_base(VL)],
                i64_base(HEAPS),
                i64_base(FLAGS),
                i64_base(UNITS),
                i64_base(COUNTERS),
                cast(bank_off, fx.T.i64),
                epoch_v,
                num_units,
                [i64_base(t) for t in (KB, VB, KL, VL)],
                [i64_base(SK), i64_base(SV)],
                i64_base(SAVE_UNITS),
                flag_base,
                num_save_units,
                grid,
                traits.BLOCK_SIZE,
            )

        def _attention():
            if const_expr(not _DBG_NOFENCE):
                memory_fence(order="acquire", scope="sys")
            ctx = _AgkvFwdContext(
                traits,
                geo,
                Q,
                KL,
                VL,
                O,
                LSE,
                KB,
                VB,
                flag_base,
                epoch_v,
                c,
                geo.rem + 2 * c,
                stride_q_n,
                stride_kv_n,
                head_dim,
                0,
            )
            ctx._setup(SharedStorage)
            emit_dualwave_swp_body(ctx, traits)

        @flyc.jit
        def _comm_if():
            if z == fx.Index(0):
                _comm()

        @flyc.jit
        def _attention_if():
            if const_expr(not _DBG_NOATTN):
                if z != fx.Index(0):
                    _attention()

        _comm_if()
        _attention_if()

        if const_expr(_DBG_TRACE):
            fx.rocdl.s_waitcnt(0)
            fx.gpu.barrier()
            t_end = read_clock()

            @flyc.jit
            def _trace_if():
                if cast(fx.thread_idx.x, fx.T.i32) == fx.Int32(0):
                    tb = i64_base(TRACE)
                    st(tb, lin * fx.Int32(4), t_start)
                    st(tb, lin * fx.Int32(4) + fx.Int32(1), t_end)
                    st(tb, lin * fx.Int32(4) + fx.Int32(2), hw)
                    st(tb, lin * fx.Int32(4) + fx.Int32(3), xcc)

            _trace_if()

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        KL: fx.Tensor,
        VL: fx.Tensor,
        KB: fx.Tensor,
        VB: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        LSE: fx.Tensor,
        SK: fx.Tensor,
        SV: fx.Tensor,
        HEAPS: fx.Tensor,
        FLAGS: fx.Tensor,
        UNITS: fx.Tensor,
        COUNTERS: fx.Tensor,
        SAVE_UNITS: fx.Tensor,
        TRACE: fx.Tensor,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        bank_off: fx.Int64,
        epoch: fx.Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = cache_tag
        kernel(
            Q,
            KL,
            VL,
            KB,
            VB,
            O,
            LSE,
            SK,
            SV,
            HEAPS,
            FLAGS,
            UNITS,
            COUNTERS,
            SAVE_UNITS,
            TRACE,
            stride_q_n,
            stride_kv_n,
            bank_off,
            epoch,
            value_attrs={
                "rocdl.waves_per_eu": traits.WAVES_PER_EU,
                "rocdl.flat_work_group_size": f"{traits.BLOCK_SIZE},{traits.BLOCK_SIZE}",
                "llvm.passthrough": [
                    ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                    ["no-nans-fp-math", "true"],
                    ["unsafe-fp-math", "true"],
                ],
            },
        ).launch(grid=(grid_x, grid_y, 1 + 2 * batch), block=(traits.BLOCK_SIZE, 1, 1), stream=stream)

    return launch, [None]


_DBG_TILE = os.environ.get("AGKV_FUSED_TILE", "128")


def _tile(c):
    """(BLOCK_M, GQA merge) of the attention blocks."""
    return {"128": (128, None), "64": (64, None), "128nm": (128, False)}[_DBG_TILE]


def forward(
    q, k, v, kb, vb, out, lse, sk, sv, heaps, flags, units, num_units, counters, save_units, num_save_units, n, r,
    bank_off, epoch, push_blocks, stream,
):
    """Launch the fused forward of rank ``r`` of ``n``.

    ``q`` / ``out`` [2c, B, Hq, D], ``k`` / ``v`` [2c, B, Hkv, D] (sbhd, contiguous, bf16); ``kb`` / ``vb``
    this rank's forward bank of K / V at ``bank_off`` (remote keys in ``_remote_layout`` order); ``lse``
    fp32 [2, B * c, Hq] (q0, q1); ``sk`` / ``sv`` [rem + 2c, B, Hkv, D], filled by ``save_units``
    (``num_save_units`` = 0: nothing saved); ``units`` the ``mega`` forward push units."""
    S, B, Hq, D = q.shape
    Hkv = k.shape[2]
    c = S // 2
    block_m, merge = _tile(c)
    launch, compiled = _build(Hq, Hkv, D, n, r, c, B, num_units, num_save_units, push_blocks, block_m, merge)
    trace = TRACE_BUF
    if trace is None:
        trace = _TRACE_DUMMY.get(q.device)
        if trace is None:
            trace = _TRACE_DUMMY[q.device] = torch.zeros(4, dtype=torch.int64, device=q.device)
    args = (
        q, k, v, kb, vb, out, lse, sk, sv, heaps, flags, units, counters, save_units, trace,
        B * Hq * D, B * Hkv * D, bank_off, epoch, stream,
    )
    if compiled[0] is None:
        # compiling runs the launch once
        with CompilationContext.compile_hints(_COMPILE_HINTS):
            compiled[0] = flyc.compile(launch, *args)
    else:
        compiled[0](*args)
