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
streams. Grid ``(Hq / Q_HEADS_PER_WG, c / BLOCK_M, 1 + items x B)``:

``z = 0``
    the first ``push_blocks`` blocks (lowest block ids, so dispatched before any attention block) store
    this rank's K / V halves into the peers' forward banks (``agkv_comm_kernel._push_loop``, the ``mega``
    units and flags), then copy what the backward needs, ``[ remote | own first | own second ]``, into
    ordinary memory: the own chunks at once, each remote chunk once its flag is up. The other blocks of
    the slice exit.
``z = 1 + item x B + b``
    one part of the keys of q1 (chunk 2n-1-r) or q0 (chunk r) of batch ``b`` (``Plan``): the own part
    (q1: own first + own second chunk, q0: own chunk; causal), or a group of remote chunks of the forward
    bank (visible to every query, so unmasked). Items run own parts first, then the remote groups in the
    order their chunks arrive (first halves, then second halves).

An attention block is the DUALWAVE_SWP flash-attention body over its part's key tiles. Before the first K
tile of each remote chunk every wave polls that chunk's flag (sender, half) until it reaches ``epoch``.
A segment of one part stores the final ``out`` / ``lse``; otherwise each part stores its normalized
output and lse in a workspace slot, counts itself done, and the last part of a (segment, batch, q block,
head group) merges the slots by their lse into ``out`` / ``lse``.

The banks are ordinary (cached) memory. Before an XCD first reads a bank in a call, one of its blocks runs a
system-scope acquire (the first remote part there, ``_xcd_acquire``; the comm blocks their own), so no L2
keeps lines of a bank a peer rewrote for this call: after that a line is only cached by a read that
observed its chunk's flag.
"""

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.math as fmath
import torch
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import vector as _vector
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.utils.arith import ArithValue

from primus_turbo.flydsl.attention.agkv_comm_kernel import (
    PUSH_BATCH,
    UNIT_BYTES,
    WAIT_TIMEOUT,
    _push_loop,
    _unit_field,
)
from primus_turbo.flydsl.attention.flash_attn_fwd_stock import emit_dualwave_swp_body
from primus_turbo.flydsl.deep_ep.common import _asm, _uniform_i64, ld_lds, st_lds
from primus_turbo.flydsl.fused_collective_gemm.common import BLOCK_THREADS, VEC_I32, copy_rows, i64_base
from primus_turbo.flydsl.utils.attn_helper_stock import (
    DualwaveKernelContext,
    _make_dualwave_swp_traits,
    dtype_to_elem_type,
)
from primus_turbo.flydsl.utils.buffer_ops import buffer_load, buffer_store, create_buffer_resource_from_addr
from primus_turbo.flydsl.utils.prims import (
    _readfirstlane_i32,
    atomic_add,
    cast,
    ld,
    memory_fence,
    read_clock,
    spin_timed_out,
    st,
)

# i64 fields of a save unit, at the push unit's positions (fields 5 / 6 unused, flag -1: none)
S_SRC, S_SRC_OFF, S_DST, S_DST_OFF, S_BYTES = range(5)
S_FLAG = 7
# save unit sources: bank K, bank V, local K, local V; destinations: saved K, saved V
SAVE_SRC_BANK_K, SAVE_SRC_BANK_V, SAVE_SRC_K, SAVE_SRC_V = range(4)
# this rank's flag array, ``agkv_comm`` order: [first halves | second halves], indexed by sender
_FLAG_FIRST, _FLAG_SECOND = 0, 1

# i64 fields of a work item: segment (0: q1, 1: q0), remote, first key token in its source (the bank or
# the local K / V), key tiles (remote), workspace slot, parts of its segment, first bank chunk (remote)
I_SEG, I_REMOTE, I_SRC_TOK, I_NTILES, I_SLOT, I_NPARTS, I_CHUNK = range(7)
ITEM_FIELDS = 8
# delta of a remote part: every query sees every key
_NO_MASK_DELTA = 1 << 24
_BLOCK_N = 64


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


def _groups(lo, hi, size):
    """Chunks ``[lo, hi)`` in groups of ``size``, aligned to ``hi``, last group first."""
    out, end = [], hi
    while end > lo:
        out.append((max(lo, end - size), end))
        end = max(lo, end - size)
    return out


class Plan:
    """Work items of rank ``r``: each segment's own part, then its remote chunks in groups of ``group``.

    q0's remote chunks (first halves of the lower ranks) are the bank's last ``r``, a suffix of q1's first
    half chunks; both are grouped from the bank's end, so a q0 group covers the chunks of the q1 group it
    runs beside."""

    def __init__(self, n, r, c, group):
        self.n, self.r, self.c, self.group = n, r, c, group
        rem_chunks, n1 = 2 * n - 2 - r, n - 1 - r
        first, second = _groups(n1, rem_chunks, group), _groups(0, n1, group)
        low = _groups(rem_chunks - r, rem_chunks, group)
        self.p1, self.p0 = 1 + len(first) + len(second), 1 + len(low)
        tiles = c // _BLOCK_N
        items = [(0, 0, 0, 0, 0, self.p1, -1), (1, 0, 0, 0, 0, self.p0, -1)]
        slots = [1, 1]

        def remote(seg, lo, hi):
            items.append((seg, 1, lo * c, (hi - lo) * tiles, slots[seg], (self.p1, self.p0)[seg], lo))
            slots[seg] += 1

        for i, (lo, hi) in enumerate(first):
            remote(0, lo, hi)
            if i < len(low):
                remote(1, *low[i])
        for lo, hi in second:
            remote(0, lo, hi)
        self.items = tuple(items)
        # workspace slots: q1's, then q0's (none when its own part is all of it)
        self.slots0 = self.p0 if self.p0 > 1 else 0
        self.slots = self.p1 + self.slots0


def choose_group(n, r, c, blocks_per_item, max_blocks, max_slots):
    """Fewest remote chunks per part that keeps the launch within ``max_blocks`` attention blocks and the
    workspace within ``max_slots`` slots."""
    rem_chunks = 2 * n - 2 - r
    for group in range(1, rem_chunks + 1):
        plan = Plan(n, r, c, group)
        if len(plan.items) * blocks_per_item <= max_blocks and plan.slots <= max_slots:
            return group
    return rem_chunks


class _Geometry:
    def __init__(self, n, r, c, batch, plan):
        self.n, self.r, self.c, self.batch = n, r, c, batch
        self.p1, self.p0 = plan.p1, plan.p0


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
        copy_rows(
            thread_index + fx.Int32(k * block_threads),
            UNIT_BYTES // 16,
            coords,
            src_res,
            dst_res,
            0,
            0,
            PUSH_BATCH,
        )


@ASTRewriter.transform
def _save_loop(
    thread_index, block_index, srcs, dsts, units, flag_base, epoch, num_units, grid, block_threads
):
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
                grid,
                None,
                0,
                block_threads,
            )
        if const_expr(num_save_units > 0 and not _DBG_NOPUSH):
            _save_loop(
                thread_index,
                block_index,
                save_srcs,
                save_dsts,
                save_units,
                flag_base,
                epoch,
                num_save_units,
                grid,
                block_threads,
            )


@ASTRewriter.transform
def _xcd_acquire(need, xcd_epochs, epoch):
    """System-scope acquire unless a block on this XCD already ran one in this call.

    It only has to drop the L2 lines a bank kept from earlier calls: in this call a bank line is cached
    only by a read that observed its chunk's flag."""
    if need:
        xcc = fx.Int32(_hwreg(20) & fx.Int32(0xF))
        if ld(xcd_epochs, xcc, scope="sys", dtype=fx.T.i64) != epoch:
            memory_fence(order="acquire", scope="sys")
            st(xcd_epochs, xcc, epoch, scope="sys")


@ASTRewriter.transform
def _finish(thread_index, nparts, counters, item_id, lds_base, combine):
    """Count this part done; the last part of its item runs ``combine``."""
    if nparts > fx.Int32(1):
        fx.rocdl.s_waitcnt(0)
        fx.gpu.barrier()
        if thread_index == fx.Int32(0):
            # the slot stores (retired by every thread) leave this XCD's L2 before the count
            memory_fence(order="release", scope="agent")
            done = atomic_add(counters, item_id, fx.Int32(1), scope="agent")
            last = done == nparts - fx.Int32(1)
            st(
                lds_base,
                fx.Int32(0),
                fx.Int32(ArithValue(last).select(fx.Int32(1), fx.Int32(0))),
                space="lds",
            )
            if last:
                st(counters, item_id, fx.Int32(0))
        fx.gpu.barrier()
        if ld(lds_base, fx.Int32(0), space="lds") != fx.Int32(0):
            memory_fence(order="acquire", scope="agent")
            combine()


class _AgkvFwdContext(DualwaveKernelContext):
    """DualwaveKernelContext of one work item (``Plan``) of one batch: its key tiles in order, a remote
    chunk's first tile after its flag, and its output in a workspace slot unless it is its segment's only
    part."""

    def __init__(
        self,
        traits,
        geo,
        Q,
        KL,
        VL,
        O,
        LSE,
        KB,
        VB,
        WS,
        WSL,
        items,
        flag_base,
        epoch,
        *args,
        zid=None,
        bases=None,
    ):
        super().__init__(traits, Q, KL, VL, O, LSE, Q, Q, Q, *args)
        self.geo = geo
        self.KB, self.VB, self.WS, self.WSL = KB, VB, WS, WSL
        self.items = items
        self.flag_base, self.epoch = flag_base, epoch
        # item x batch + batch index of a claimed work item (persistent launch); else the grid's z - 1
        self.zid = zid
        # 64-bit tensor bases by name, read where used (persistent launch); else from the tensors
        self.bases = bases

    def _base(self, name, tensor):
        return self.bases[name] if self.bases is not None else i64_base(tensor)

    def _item_field(self, field):
        value = ld(self.items, self.item * fx.Int32(ITEM_FIELDS) + fx.Int32(field), dtype=fx.T.i64)
        return fx.Index(cast(value, fx.T.i32))

    def _grid_batch_index(self):
        z = self.zid if self.zid is not None else fx.Int32(fx.Index(gpu.block_idx.z)) - fx.Int32(1)
        self.item = z // fx.Int32(self.geo.batch)
        self.seg = self._item_field(I_SEG)
        self.remote = ArithValue(self._item_field(I_REMOTE) != fx.Index(0))
        self.src_tok = self._item_field(I_SRC_TOK)
        self.ntiles = self._item_field(I_NTILES)
        self.slot = self._item_field(I_SLOT)
        self.nparts = self._item_field(I_NPARTS)
        self.chunk0 = self._item_field(I_CHUNK)
        return fx.Index(z % fx.Int32(self.geo.batch))

    def _sel(self, a, b):
        return fx.Index(self.is_q1.select(fx.Index(a), fx.Index(b)))

    def _dense_geometry(self):
        g = self.geo
        self.is_q1 = ArithValue(self.seg == fx.Index(0))
        self.final = ArithValue(self.nparts == fx.Index(1))
        self.own_len = self._sel(2 * g.c, g.c)
        kv_len = fx.Index(self.remote.select(self.ntiles * fx.Index(_BLOCK_N), self.own_len))
        # lse rows: final [q0 | q1] x batch x c, a slot batch x c
        final_base = (self._sel(1, 0) * fx.Index(g.batch) + self.batch_idx) * fx.Index(g.c)
        self.q_tok_base = fx.Index(self.final.select(final_base, self.batch_idx * fx.Index(g.c)))
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
        return fx.logical_divide(
            fx.make_view(buf_ptr, fx.make_layout(fx.Int32(nelem), fx.Int32(1))), fx.make_layout(1, 1)
        )

    def _setup(self, shared_storage, block_ids=None, lds=None, tid=None):
        super()._setup(shared_storage, block_ids, lds, tid)
        traits, g = self.traits, self.geo
        bf16 = fx.Index(traits.BF16_BYTES)
        hqd = traits.NUM_HEADS_Q * traits.HEAD_DIM
        # the own part is causal from its own chunk on (delta c for q1, 0 for q0), a remote part unmasked
        self.delta_i32 = fx.Int32(
            self.remote.select(fx.Int32(_NO_MASK_DELTA), fx.Int32(self.own_len - fx.Index(g.c)))
        )
        self.split_t_end = fx.Index(self.remote.select(self.ntiles, self.split_t_end))
        self.causal_end_raw_i32 = fx.Int32(self.q_start + traits.BLOCK_M) + self.delta_i32
        self.active = ArithValue(self.causal_end_raw_i32 > fx.Int32(0))

        q_per = self.seqlen_q_v * self.stride_q_n_v
        q_off_b = fx.Int64((self._sel(g.c, 0) * fx.Index(g.batch) + self.batch_idx) * fx.Index(hqd) * bf16)
        self.q_div = self._div(self.Q, fx.Int64(self._base("Q", self.Q)) + q_off_b, q_per * bf16, q_per)
        self.slot_g = fx.Index(self.is_q1.select(self.slot, self.slot + fx.Index(g.p1)))
        ws_off = (self.slot_g * fx.Index(g.c * g.batch) + self.batch_idx) * fx.Index(hqd) * bf16
        o_base = self.final.select(
            fx.Int64(self._base("O", self.O)) + q_off_b,
            fx.Int64(self._base("WS", self.WS)) + fx.Int64(ws_off),
        )
        self.o_div = self._div(self.O, fx.Int64(o_base), q_per * bf16, q_per)
        wsl = fx.Int64(self._base("WSL", self.WSL)) + fx.Int64(
            self.slot_g * fx.Index(g.batch * g.c * traits.NUM_HEADS_Q * 4)
        )
        lse_base = fx.Int64(self.final.select(fx.Int64(self._base("LSE", self.DebugCounts)), wsl))
        self.lse_rsrc = create_buffer_resource_from_addr(lse_base, num_records_bytes=0xFFFFFFFF)

        tok_b = self.stride_kv_n_v * bf16
        kv_off = fx.Int64(
            self.src_tok * tok_b + self.batch_idx * fx.Index(traits.NUM_HEADS_KV * traits.HEAD_DIM) * bf16
        )
        nrec = self.seqlen_kv_v * tok_b
        nelem = self.seqlen_kv_v * self.stride_kv_n_v
        self.k_div, self.v_div = (
            self._div(
                self.K,
                fx.Int64(self.remote.select(fx.Int64(self._base(*bank)), fx.Int64(self._base(*local))))
                + kv_off,
                nrec,
                nelem,
            )
            for bank, local in ((("KB", self.KB), ("KL", self.K)), (("VB", self.VB), ("VL", self.V)))
        )

    def _wait_chunk(self, tile_idx):
        """Poll the flag of the remote chunk whose first tile is ``tile_idx``, if it is one."""
        g = self.geo
        tok = fx.Index(tile_idx) * fx.Index(_BLOCK_N)
        need = self.remote & ArithValue(tok % fx.Index(g.c) == fx.Index(0))
        k = self.chunk0 + tok // fx.Index(g.c)
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
        super().load_k_tile(tile_idx, buf_id)

    def combine(self, thread_index, ws, wsl, out, lse):
        """``out`` / ``lse`` of this block's rows and heads from its segment's slots, weighted by their lse."""
        traits, g = self.traits, self.geo
        f32, f32_vec, bf16_vec = fx.T.f32, fx.T.vec(8, fx.T.f32), fx.T.vec(8, fx.T.bf16)
        hq, d, b_n, c = traits.NUM_HEADS_Q, traits.HEAD_DIM, g.batch, g.c
        hqd, qh, vpr = hq * d, traits.Q_HEADS_PER_WG, d // 8
        slot_elems, slot_lse = c * b_n * hqd, b_n * c * hq
        nparts = fx.Int32(self.nparts)
        slot0 = fx.Int64(self._sel(0, g.p1))
        o_res = create_buffer_resource_from_addr(
            ws + slot0 * fx.Int64(slot_elems * 2), num_records_bytes=nparts * fx.Int32(slot_elems * 2)
        )
        l_res = create_buffer_resource_from_addr(
            wsl + slot0 * fx.Int64(slot_lse * 4), num_records_bytes=nparts * fx.Int32(slot_lse * 4)
        )
        out_res = create_buffer_resource_from_addr(out, num_records_bytes=2 * c * b_n * hqd * 2)
        lse_res = create_buffer_resource_from_addr(lse, num_records_bytes=2 * b_n * c * hq * 4)
        q_row0, seg_lse = fx.Int32(self._sel(c, 0)), fx.Int32(self._sel(1, 0))
        head0 = fx.Int32(self.h_kv_idx * traits.GQA_GROUP_SIZE + self.group_id * qh)
        q_start, b = fx.Int32(self.q_start), fx.Int32(self.batch_idx)
        neg_inf = fx.Float32(float("-inf"))
        for k in range_constexpr(traits.BLOCK_M * qh * vpr // traits.BLOCK_SIZE):
            v = thread_index + fx.Int32(k * traits.BLOCK_SIZE)
            row, j, d8 = v // fx.Int32(qh * vpr), (v // fx.Int32(vpr)) % fx.Int32(qh), v % fx.Int32(vpr)
            head, qr = head0 + j, q_start + row
            oi = (qr * fx.Int32(b_n) + b) * fx.Int32(hqd) + head * fx.Int32(d) + d8 * fx.Int32(8)
            li = (b * fx.Int32(c) + qr) * fx.Int32(hq) + head
            valid = [fx.Int32(s) < nparts for s in range(g.p1)]
            lses = [
                ArithValue(valid[s]).select(
                    buffer_load(l_res, li + fx.Int32(s * slot_lse), vec_width=1, dtype=f32, mask=valid[s]),
                    neg_inf,
                )
                for s in range(g.p1)
            ]
            m = lses[0]
            for x in lses[1:]:
                m = fx.arith.maximumf(m, x)
            weights = [fmath.exp(fx.arith.subf(x, m)) for x in lses]
            total = weights[0]
            for w in weights[1:]:
                total = fx.arith.addf(total, w)
            acc = None
            for s in range(g.p1):
                o = fx.arith.extf(
                    f32_vec,
                    buffer_load(
                        o_res, oi + fx.Int32(s * slot_elems), vec_width=8, dtype=fx.T.bf16, mask=valid[s]
                    ),
                )
                term = fx.arith.mulf(o, _vector.broadcast(f32_vec, fx.arith.divf(weights[s], total)))
                acc = term if acc is None else fx.arith.addf(acc, term)
            out_i = (
                ((q_row0 + qr) * fx.Int32(b_n) + b) * fx.Int32(hqd) + head * fx.Int32(d) + d8 * fx.Int32(8)
            )
            buffer_store(fx.arith.trunc_f(bf16_vec, acc), out_res, out_i)
            lse_i = ((seg_lse * fx.Int32(b_n) + b) * fx.Int32(c) + qr) * fx.Int32(hq) + head
            buffer_store(fx.arith.addf(m, fmath.log(total)), lse_res, lse_i, mask=d8 == fx.Int32(0))


_DBG_NOWAIT = os.environ.get("AGKV_FUSED_DBG_NOWAIT") == "1"
_DBG_NOFENCE = os.environ.get("AGKV_FUSED_DBG_NOFENCE") == "1"
_DBG_NOPUSH = os.environ.get("AGKV_FUSED_DBG_NOPUSH") == "1"
_DBG_NOATTN = os.environ.get("AGKV_FUSED_DBG_NOATTN") == "1"
_DBG_TRACE = os.environ.get("AGKV_FUSED_DBG_TRACE") == "1"
_DBG_TILE = os.environ.get("AGKV_FUSED_TILE", "128")
MAX_BLOCKS = int(os.environ.get("AGKV_FUSED_MAX_BLOCKS", "2048"))
MAX_SLOTS = int(os.environ.get("AGKV_FUSED_MAX_SLOTS", "8"))
TRACE_BUF = None
_TRACE_DUMMY = {}
_XCD_EPOCHS = {}
# one persistent launch: blocks claim push / attention / save work instead of one block per grid slot
PERSISTENT = os.environ.get("AGKV_FUSED_PERSISTENT", "1") == "1"
PERSISTENT_BLOCKS = int(os.environ.get("AGKV_FUSED_BLOCKS", "0"))
PERSISTENT_PUSHERS = int(os.environ.get("AGKV_FUSED_PUSHERS", "0"))
_STATES = {}
_SAVE_ROWS = {}
_SCHEDS = {}


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


def _tile(c):
    """(BLOCK_M, GQA merge) of the attention blocks."""
    return {"128": (128, None), "64": (64, None), "128nm": (128, False)}[_DBG_TILE]


def _traits(num_heads, num_kv_heads, head_dim, block_m, merge):
    return _make_dualwave_swp_traits(
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


@functools.lru_cache(maxsize=32)
def _build(
    num_heads,
    num_kv_heads,
    head_dim,
    n,
    r,
    c,
    batch,
    num_units,
    num_save_units,
    push_blocks,
    block_m,
    merge,
    group,
):
    traits = _traits(num_heads, num_kv_heads, head_dim, block_m, merge)
    assert BLOCK_THREADS % traits.BLOCK_SIZE == 0, "the push / save units are sized for 512-thread blocks"
    assert traits.BLOCK_M * traits.Q_HEADS_PER_WG * (head_dim // 8) % traits.BLOCK_SIZE == 0
    plan = Plan(n, r, c, group)
    geo = _Geometry(n, r, c, batch, plan)
    num_items = len(plan.items)
    grid_x, grid_y = num_heads // traits.Q_HEADS_PER_WG, c // traits.BLOCK_M
    grid = min(push_blocks, grid_x * grid_y, max(num_units, 1))
    lds_dtype = dtype_to_elem_type(traits.DTYPE_STR)
    cache_tag = traits.cache_tag + (
        n,
        r,
        c,
        batch,
        num_units,
        num_save_units,
        grid,
        group,
        _DBG_NOWAIT,
        _DBG_NOFENCE,
        _DBG_NOPUSH,
        _DBG_NOATTN,
        _DBG_TRACE,
    )

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
        ITEMS: fx.Tensor,
        WS: fx.Tensor,
        WSL: fx.Tensor,
        DONE: fx.Tensor,
        XCDS: fx.Tensor,
        TRACE: fx.Tensor,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        bank_off: fx.Int64,
        epoch: fx.Int64,
    ):
        z = fx.Index(gpu.block_idx.z)
        flag_base = ld(i64_base(FLAGS), fx.Int32(r), dtype=fx.T.i64)
        epoch_v = cast(epoch, fx.T.i64)
        thread_index = cast(fx.thread_idx.x, fx.T.i32)
        if const_expr(_DBG_TRACE):
            lin = fx.Int32(
                (z * fx.Index(grid_y) + fx.Index(gpu.block_idx.y)) * fx.Index(grid_x)
                + fx.Index(gpu.block_idx.x)
            )
            t_start = read_clock()
            hw = fx.arith.extsi(fx.T.i64, _hwreg(4))
            xcc = fx.arith.extsi(fx.T.i64, _hwreg(20))

        def _comm():
            block = fx.Index(gpu.block_idx.y) * fx.Index(grid_x) + fx.Index(gpu.block_idx.x)
            _comm_role(
                thread_index,
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
                WS,
                WSL,
                i64_base(ITEMS),
                flag_base,
                epoch_v,
                c,
                c,
                stride_q_n,
                stride_kv_n,
                head_dim,
                0,
            )
            ctx._setup(SharedStorage)
            if const_expr(not _DBG_NOFENCE):
                _xcd_acquire(ctx.remote, i64_base(XCDS), epoch_v)
            emit_dualwave_swp_body(ctx, traits)
            item_id = (
                (fx.Int32(ctx.seg) * fx.Int32(batch) + fx.Int32(ctx.batch_idx)) * fx.Int32(grid_y)
                + fx.Int32(fx.Index(gpu.block_idx.y))
            ) * fx.Int32(grid_x) + fx.Int32(fx.Index(gpu.block_idx.x))
            _finish(
                thread_index,
                fx.Int32(ctx.nparts),
                i64_base(DONE),
                item_id,
                ctx.lds_kv_base_idx,
                lambda: ctx.combine(
                    thread_index,
                    fx.Int64(i64_base(WS)),
                    fx.Int64(i64_base(WSL)),
                    fx.Int64(i64_base(O)),
                    fx.Int64(i64_base(LSE)),
                ),
            )

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
                if thread_index == fx.Int32(0):
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
        ITEMS: fx.Tensor,
        WS: fx.Tensor,
        WSL: fx.Tensor,
        DONE: fx.Tensor,
        XCDS: fx.Tensor,
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
            ITEMS,
            WS,
            WSL,
            DONE,
            XCDS,
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
        ).launch(grid=(grid_x, grid_y, 1 + num_items * batch), block=(traits.BLOCK_SIZE, 1, 1), stream=stream)

    return launch, [None]


# 64-bit values the persistent loop reads from LDS where it uses them: held in registers across the
# attention body they spill inside its main loop
_STASH = (
    "Q",
    "KL",
    "VL",
    "KB",
    "VB",
    "O",
    "LSE",
    "SK",
    "SV",
    "SAVE_UNITS",
    "ITEMS",
    "SCHED",
    "STATE",
    "WS",
    "WSL",
    "DONE",
    "XCDS",
    "FLAG_BASE",
    "EPOCH",
)


@ASTRewriter.transform
def _stash_bases(thread_index, stash, values):
    if thread_index == fx.Int32(0):
        for i in range_constexpr(len(values)):
            st(stash, fx.Int32(i), values[i], scope="workgroup", space=3)
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()


def _stashed(stash, i):
    return _uniform_i64(ld(stash, fx.Int32(i), scope="workgroup", space=3, dtype=fx.T.i64))


def _opaque_tid(thread_index):
    """``thread_index`` through an opaque move: inside the claim loop LICM would hoist every unit body's
    lane offsets out of their branches, and all of them would live across the loop and spill."""
    return fx.arith.ArithValue(
        llvm.inline_asm(
            fx.T.i32,
            [fx.as_ir_value(cast(thread_index, fx.T.i32))],
            "v_mov_b32 $0, $1",
            "=v,v",
            has_side_effects=True,
        )
    )


# persistent launch state (int64, zero between launches): blocks out, then the claims of each XCD's queue
ST_EXIT, ST_CLAIM = 0, 1
NUM_XCDS = 8
STATE_LEN = ST_CLAIM + NUM_XCDS
# the schedule opens with (offset, length) of each XCD's queue
SCHED_HEAD = 2 * NUM_XCDS


@ASTRewriter.transform
def _claim(thread_index, state, slot, sched, num_sched):
    """The block's next schedule position: from its own XCD's queue, else the next XCD's with work left.

    Entry ``e`` sits in the queue of XCD ``e % NUM_XCDS``, the XCD the grid launch dispatches it to, so
    the q blocks of a head group share that XCD's L2 for their K/V as they do there."""
    if thread_index == fx.Int32(0):
        xcc = fx.Int32(_hwreg(20) & fx.Int32(0xF))
        k = fx.Int32(0)
        pos = fx.Int32(num_sched)
        while k < fx.Int32(NUM_XCDS):
            q = (xcc + k) % fx.Int32(NUM_XCDS)
            claimed = cast(atomic_add(state, fx.Int32(ST_CLAIM) + q, fx.Int64(1), scope="agent"), fx.T.i32)
            hit = claimed < cast(ld(sched, fx.Int32(2) * q + fx.Int32(1), dtype=fx.T.i64), fx.T.i32)
            pos = hit.select(cast(ld(sched, fx.Int32(2) * q, dtype=fx.T.i64), fx.T.i32) + claimed, pos)
            k = hit.select(fx.Int32(NUM_XCDS), k + fx.Int32(1))
        st_lds(slot, 0, pos)
    # s_barrier does not wait for LDS traffic: drain the store, then the reads
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    unit = _readfirstlane_i32(ld_lds(slot, 0))
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    return unit


@ASTRewriter.transform
def _exit(thread_index, state, grid):
    """The last block out resets the state: every block's claims returned before it counted itself."""
    fx.gpu.barrier()
    if thread_index == fx.Int32(0):
        arrived = atomic_add(state, fx.Int32(ST_EXIT), fx.Int64(1), scope="agent")
        if arrived == fx.Int64(grid - 1):
            for i in range_constexpr(STATE_LEN):
                st(state, fx.Int32(i), fx.Int64(0))


@ASTRewriter.transform
def _pushers(
    thread_index, block, push_blocks, srcs, heaps, flags, units, counters, bank_off, epoch, num_units, bt
):
    if block < fx.Int32(push_blocks):
        _push_loop(
            thread_index,
            block,
            srcs,
            heaps,
            flags,
            units,
            counters,
            bank_off,
            epoch,
            num_units,
            push_blocks,
            None,
            0,
            bt,
        )


@ASTRewriter.transform
def _save_one(thread_index, unit, srcs, dsts, units, flag_base, epoch, xcds, block_threads):
    """Save unit ``unit``, behind its chunk's flag when it reads the bank."""
    flag = _unit_field(units, unit, S_FLAG)
    if flag >= fx.Int32(0):
        if thread_index == fx.Int32(0):
            _spin_flag(flag_base, flag, epoch)
        fx.gpu.barrier()
    _xcd_acquire(flag >= fx.Int32(0), xcds, epoch)
    _save_unit(thread_index, unit, srcs, dsts, units, block_threads)


@ASTRewriter.transform
def _work_loop(thread_index, state, slot, sched, num_sched, attend, save):
    """Claim schedule entries until none are left: ``e >= 0`` attention work item ``e``, else save unit
    ``-1 - e``."""
    unit = _claim(thread_index, state, slot, sched, num_sched)
    while unit < fx.Int32(num_sched):
        entry = _readfirstlane_i32(cast(ld(sched, unit + fx.Int32(SCHED_HEAD), dtype=fx.T.i64), fx.T.i32))
        if entry >= fx.Int32(0):
            attend(entry)
        else:
            save(fx.Int32(-1) - entry)
        # the next item's LDS writes wait for this one's last reads
        fx.gpu.barrier()
        unit = _claim(thread_index, state, slot, sched, num_sched)


@functools.lru_cache(maxsize=32)
def _build_persistent(
    num_heads,
    num_kv_heads,
    head_dim,
    n,
    r,
    c,
    batch,
    num_units,
    num_save_units,
    push_blocks,
    block_m,
    merge,
    group,
    num_sched,
    grid,
):
    traits = _traits(num_heads, num_kv_heads, head_dim, block_m, merge)
    assert BLOCK_THREADS % traits.BLOCK_SIZE == 0, "the push / save units are sized for 512-thread blocks"
    assert traits.BLOCK_M * traits.Q_HEADS_PER_WG * (head_dim // 8) % traits.BLOCK_SIZE == 0
    plan = Plan(n, r, c, group)
    geo = _Geometry(n, r, c, batch, plan)
    grid_x, grid_y = num_heads // traits.Q_HEADS_PER_WG, c // traits.BLOCK_M
    push_blocks = min(push_blocks, grid, max(num_units, 1)) if num_units else 0
    lds_dtype = dtype_to_elem_type(traits.DTYPE_STR)
    cache_tag = traits.cache_tag + (
        "persistent",
        n,
        r,
        c,
        batch,
        num_units,
        num_save_units,
        push_blocks,
        group,
        num_sched,
        grid,
        _DBG_NOWAIT,
        _DBG_NOFENCE,
    )

    @fx.struct
    class SharedStorage:
        kv: fx.Array[lds_dtype, traits.LDS_KV_TOTAL_SIZE, 16]
        claim: fx.Array[fx.Int32, 4, 16]
        stash: fx.Array[fx.Int32, 2 * len(_STASH), 16]

    @flyc.kernel(name="agkv_fused_fwd_persistent_kernel", known_block_size=[traits.BLOCK_SIZE, 1, 1])
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
        ITEMS: fx.Tensor,
        SCHED: fx.Tensor,
        STATE: fx.Tensor,
        WS: fx.Tensor,
        WSL: fx.Tensor,
        DONE: fx.Tensor,
        XCDS: fx.Tensor,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        bank_off: fx.Int64,
        epoch: fx.Int64,
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        claim_slot = fx.Int32(fx.ptrtoint(lds.claim.ptr))
        stash = fx.Int32(fx.ptrtoint(lds.stash.ptr))
        thread_index = cast(fx.thread_idx.x, fx.T.i32)
        block = cast(fx.block_idx.x, fx.T.i32)
        tensors = dict(
            Q=Q,
            KL=KL,
            VL=VL,
            KB=KB,
            VB=VB,
            O=O,
            LSE=LSE,
            SK=SK,
            SV=SV,
            SAVE_UNITS=SAVE_UNITS,
            ITEMS=ITEMS,
            SCHED=SCHED,
            STATE=STATE,
            WS=WS,
            WSL=WSL,
            DONE=DONE,
            XCDS=XCDS,
        )
        values = {name: i64_base(t) for name, t in tensors.items()}
        values["FLAG_BASE"] = ld(i64_base(FLAGS), fx.Int32(r), dtype=fx.T.i64)
        values["EPOCH"] = cast(epoch, fx.T.i64)
        _stash_bases(thread_index, stash, [values[name] for name in _STASH])

        # the lowest block ids (dispatched first) push this rank's halves, then join the workers
        _pushers(
            thread_index,
            block,
            push_blocks,
            [i64_base(KL), i64_base(VL)],
            i64_base(HEAPS),
            i64_base(FLAGS),
            i64_base(UNITS),
            i64_base(COUNTERS),
            cast(bank_off, fx.T.i64),
            cast(epoch, fx.T.i64),
            num_units,
            traits.BLOCK_SIZE,
        )

        def base(name):
            return _stashed(stash, _STASH.index(name))

        def attend(entry):
            per = fx.Int32(grid_x * grid_y)
            zid, rest = entry // per, entry % per
            y, x = rest // fx.Int32(grid_x), rest % fx.Int32(grid_x)
            epoch_v = base("EPOCH")
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
                WS,
                WSL,
                base("ITEMS"),
                base("FLAG_BASE"),
                epoch_v,
                c,
                c,
                stride_q_n,
                stride_kv_n,
                head_dim,
                0,
                zid=zid,
                bases={name: base(name) for name in ("Q", "KL", "VL", "KB", "VB", "O", "LSE", "WS", "WSL")},
            )
            ctx._setup(None, block_ids=(x, y), lds=lds, tid=_opaque_tid(thread_index))
            if const_expr(not _DBG_NOFENCE):
                _xcd_acquire(ctx.remote, base("XCDS"), epoch_v)
            emit_dualwave_swp_body(ctx, traits)
            item_id = (
                (fx.Int32(ctx.seg) * fx.Int32(batch) + fx.Int32(ctx.batch_idx)) * fx.Int32(grid_y) + y
            ) * fx.Int32(grid_x) + x
            _finish(
                _opaque_tid(thread_index),
                fx.Int32(ctx.nparts),
                base("DONE"),
                item_id,
                ctx.lds_kv_base_idx,
                lambda: ctx.combine(
                    _opaque_tid(thread_index), base("WS"), base("WSL"), base("O"), base("LSE")
                ),
            )

        def save(unit):
            _save_one(
                _opaque_tid(thread_index),
                unit,
                [base(name) for name in ("KB", "VB", "KL", "VL")],
                [base("SK"), base("SV")],
                base("SAVE_UNITS"),
                base("FLAG_BASE"),
                base("EPOCH"),
                base("XCDS"),
                traits.BLOCK_SIZE,
            )

        _work_loop(thread_index, base("STATE"), claim_slot, base("SCHED"), num_sched, attend, save)
        _exit(thread_index, base("STATE"), grid)

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
        ITEMS: fx.Tensor,
        SCHED: fx.Tensor,
        STATE: fx.Tensor,
        WS: fx.Tensor,
        WSL: fx.Tensor,
        DONE: fx.Tensor,
        XCDS: fx.Tensor,
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
            ITEMS,
            SCHED,
            STATE,
            WS,
            WSL,
            DONE,
            XCDS,
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
        ).launch(grid=(grid, 1, 1), block=(traits.BLOCK_SIZE, 1, 1), stream=stream)

    return launch, [None]


def _resident_blocks(traits, device):
    """Work-groups of this kernel one GPU holds at once (waves per EU and LDS bound)."""
    props = torch.cuda.get_device_properties(device)
    waves = traits.WAVES_PER_EU * 4 * 64 // traits.BLOCK_SIZE
    lds = 160 * 1024 // (traits.LDS_KV_TOTAL_SIZE * 2 + 256)
    return props.multi_processor_count * max(1, min(waves, lds))


@functools.lru_cache(maxsize=64)
def _schedule(plan, batch, grid_x, grid_y, c, tok, save_rows, device):
    """Claim order of the persistent launch: the work items in ``plan`` order (each over batch, q block,
    head group), the own-chunk save units after the own parts and each remote chunk's save units after
    the first q1 group holding it (its flag is up by then), split into the XCD queues of ``_claim``.
    Attention entries are ``>= 0``, save unit ``u`` is ``-1 - u``."""
    per_item = batch * grid_y * grid_x
    own = [u for u, row in enumerate(save_rows) if row[S_FLAG] < 0]
    by_chunk = {}
    for u, row in enumerate(save_rows):
        if row[S_FLAG] >= 0:
            by_chunk.setdefault(row[S_SRC_OFF] // (c * tok), []).append(u)
    sched = []
    for i, item in enumerate(plan.items):
        sched += range(i * per_item, (i + 1) * per_item)
        if i == 1:
            sched += [-1 - u for u in own]
        if item[I_SEG] == 0 and item[I_REMOTE]:
            for k in range(item[I_CHUNK], item[I_CHUNK] + item[I_NTILES] * _BLOCK_N // c):
                sched += [-1 - u for u in by_chunk.pop(k, [])]
    for units in by_chunk.values():
        sched += [-1 - u for u in units]
    queues = [[] for _ in range(NUM_XCDS)]
    for e in sched:
        queues[(e if e >= 0 else -1 - e) % NUM_XCDS].append(e)
    head, flat = [], []
    for queue in queues:
        head += [len(flat), len(queue)]
        flat += queue
    return torch.tensor(head + flat, dtype=torch.int64, device=device), len(flat)


@functools.lru_cache(maxsize=64)
def _launch_plan(num_heads, num_kv_heads, head_dim, n, r, c, batch, device):
    """(group, plan, item table, done counters) of a shape; the counters return to zero after every launch."""
    block_m, merge = _tile(c)
    traits = _traits(num_heads, num_kv_heads, head_dim, block_m, merge)
    grid_x, grid_y = num_heads // traits.Q_HEADS_PER_WG, c // traits.BLOCK_M
    group = choose_group(n, r, c, grid_x * grid_y * batch, MAX_BLOCKS, MAX_SLOTS)
    plan = Plan(n, r, c, group)
    rows = [list(item) + [0] * (ITEM_FIELDS - len(item)) for item in plan.items]
    items = torch.tensor(rows, dtype=torch.int64, device=device)
    done = torch.zeros(2 * batch * grid_y * grid_x, dtype=torch.int32, device=device)
    return group, plan, items, done


def forward(
    q,
    k,
    v,
    kb,
    vb,
    out,
    lse,
    sk,
    sv,
    heaps,
    flags,
    units,
    num_units,
    counters,
    save_units,
    num_save_units,
    n,
    r,
    bank_off,
    epoch,
    push_blocks,
    stream,
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
    group, plan, items, done = _launch_plan(Hq, Hkv, D, n, r, c, B, q.device)
    assert plan.slots * c * B * Hq * D * 2 < 1 << 31, "workspace offsets are 32-bit"
    ws = torch.empty((plan.slots, c, B, Hq, D), dtype=q.dtype, device=q.device)
    wsl = torch.empty((plan.slots, B * c, Hq), dtype=torch.float32, device=q.device)
    # epoch of each XCD's last acquire: epochs count the calls of one exchange, so one array per exchange
    xcds = _XCD_EPOCHS.get(flags.data_ptr())
    if xcds is None:
        xcds = _XCD_EPOCHS[flags.data_ptr()] = torch.zeros(16, dtype=torch.int64, device=q.device)
    if PERSISTENT:
        traits = _traits(Hq, Hkv, D, block_m, merge)
        grid_x, grid_y = Hq // traits.Q_HEADS_PER_WG, c // traits.BLOCK_M
        tok = k[0].numel() * k.element_size()
        key = (save_units.data_ptr(), num_save_units)
        save_rows = _SAVE_ROWS.get(key)
        if save_rows is None:
            # the save tables are cached by the caller: read each back once
            rows = save_units[:num_save_units].tolist() if num_save_units else []
            save_rows = _SAVE_ROWS[key] = tuple(map(tuple, rows))
        grid = PERSISTENT_BLOCKS or _resident_blocks(traits, q.device)
        sched_key = (key, id(plan), B, grid_x, grid_y, c, tok, q.device)
        if sched_key not in _SCHEDS:
            _SCHEDS[sched_key] = _schedule(plan, B, grid_x, grid_y, c, tok, save_rows, q.device)
        sched, num_sched = _SCHEDS[sched_key]
        push_blocks = PERSISTENT_PUSHERS or push_blocks
        if os.environ.get("AGKV_FUSED_SHOW") == "1" and r == 0:
            print(
                f"persistent fwd: grid {grid} pushers {push_blocks} sched {num_sched} block {traits.BLOCK_SIZE}"
            )
        state = _STATES.get(flags.data_ptr())
        if state is None:
            state = _STATES[flags.data_ptr()] = torch.zeros(STATE_LEN, dtype=torch.int64, device=q.device)
        launch, compiled = _build_persistent(
            Hq,
            Hkv,
            D,
            n,
            r,
            c,
            B,
            num_units,
            num_save_units,
            push_blocks,
            block_m,
            merge,
            group,
            num_sched,
            grid,
        )
        args = (
            q,
            k,
            v,
            kb,
            vb,
            out,
            lse,
            sk,
            sv,
            heaps,
            flags,
            units,
            counters,
            save_units,
            items,
            sched,
            state,
            ws,
            wsl,
            done,
            xcds,
            B * Hq * D,
            B * Hkv * D,
            bank_off,
            epoch,
            stream,
        )
        if compiled[0] is None:
            with CompilationContext.compile_hints(_COMPILE_HINTS):
                compiled[0] = flyc.compile(launch, *args)
        else:
            compiled[0](*args)
        return
    launch, compiled = _build(
        Hq, Hkv, D, n, r, c, B, num_units, num_save_units, push_blocks, block_m, merge, group
    )
    trace = TRACE_BUF
    if trace is None:
        trace = _TRACE_DUMMY.get(q.device)
        if trace is None:
            trace = _TRACE_DUMMY[q.device] = torch.zeros(4, dtype=torch.int64, device=q.device)
    args = (
        q,
        k,
        v,
        kb,
        vb,
        out,
        lse,
        sk,
        sv,
        heaps,
        flags,
        units,
        counters,
        save_units,
        items,
        ws,
        wsl,
        done,
        xcds,
        trace,
        B * Hq * D,
        B * Hkv * D,
        bank_off,
        epoch,
        stream,
    )
    if compiled[0] is None:
        # compiling runs the launch once
        with CompilationContext.compile_hints(_COMPILE_HINTS):
            compiled[0] = flyc.compile(launch, *args)
    else:
        compiled[0](*args)
