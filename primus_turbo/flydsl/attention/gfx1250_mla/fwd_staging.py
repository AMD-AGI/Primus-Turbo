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
# Adapted from the gfx1250 FMHA forward prefill of aiter (https://github.com/ROCm/aiter, commit 6963ae9d),
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc., under the MIT license.

"""Operand staging of the gfx1250 MLA forward: global -> LDS -> WMMA fragments, and O back out.

Q, K and V are copied by the Tensor Data Mover into row-major LDS rows whose padding keeps the
fragment reads bank-conflict free; V is read with the transposing ``ds_load_tr16_b128``. O goes
from the accumulators through padded LDS to global memory with ``global_store_async_from_lds``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx

# UNSTABLE(gfx1250): the TDM atom and tensor_wait, ds_load_tr16_b128, the async LDS->global store,
# sched_barrier and s_wait_dscnt have no stable API in FlyDSL 0.3.4.1; LDS loads/stores are raw LLVM
# ops on pointers built from i32 byte addresses.
from flydsl._mlir.dialects import llvm
from flydsl.expr import rocdl
from flydsl.expr.rocdl import tdm_ops

from primus_turbo.flydsl.utils.buffer_ops import get_element_ptr

D_QK = 192
D_V = 128
N_BLOCK = 64  # kv rows per tile
NUM_WAVES = 8
WAVE_SIZE = 32
WMMA_M = 16
WMMA_K = 32
Q_TILES = 2  # 16-row WMMA tiles per wave
ROWS_PER_WAVE = WMMA_M * Q_TILES
BLOCK_M = ROWS_PER_WAVE * NUM_WAVES  # q rows per workgroup

_CHUNK_ELEMS = 8  # bf16 elements per 16-byte access
_CHUNK_BYTES = 16
# LDS row padding (elements) that keeps the 16-byte fragment accesses conflict free.
Q_ROW_ELEMS = D_QK + 8
K_ROW_ELEMS = D_QK + 8
V_ROW_ELEMS = D_V + 16
O_ROW_ELEMS = D_V + 8
Q_LDS_BYTES = BLOCK_M * Q_ROW_ELEMS * 2
K_LDS_BYTES = N_BLOCK * K_ROW_ELEMS * 2
V_LDS_BYTES = N_BLOCK * V_ROW_ELEMS * 2
O_LDS_BYTES = BLOCK_M * O_ROW_ELEMS * 2


def pow2_segments(width):
    """[(col0, width)] covering a row with power-of-two widths, largest first: a TDM pad interval
    is a power of two, so a 192-wide row takes two copies (128 + 64)."""
    segs, c0 = [], 0
    while c0 < width:
        w = 1 << ((width - c0).bit_length() - 1)
        segs.append((c0, w))
        c0 += w
    return segs


def lds_ptr(addr):
    """Raw ``!llvm.ptr<3>`` for an LDS byte address."""
    pt = fx.PointerType.get(fx.Int32.ir_type, address_space=fx.AddressSpace.Shared, alignment=4)
    return fx.to_llvm_ptr(fx.inttoptr(pt, addr))


def global_ptr(addr):
    """Raw ``!llvm.ptr<1>`` for a global byte address."""
    pt = fx.PointerType.get(fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4)
    return fx.to_llvm_ptr(fx.inttoptr(pt, addr))


def _lds_bf16_ptr_type():
    return fx.PointerType.get(elem_ty=fx.BFloat16.ir_type, address_space=fx.AddressSpace.Shared, alignment=16)


def issue_q_copy(*, q, stride_q_seq, stride_q_head, q_start, q_len, head, block_x, warp_idx, lds_base):
    """TDM-copy this wave's 32 query rows into its own LDS rows and return their base; rows past
    ``q_len`` are zero-filled. The view is (seq, head, dim) with one head: a 2-D view changes the
    generated code."""
    seq0 = block_x * fx.Int32(BLOCK_M) + warp_idx * fx.Int32(ROWS_PER_WAVE)
    n_valid = fx.max(q_len - seq0, fx.Int32(0))
    off = fx.Int64(q_start + seq0) * fx.Int64(stride_q_seq) + fx.Int64(head) * fx.Int64(stride_q_head)
    base = fx.get_iter(q)
    region = lds_base + warp_idx * fx.Int32(ROWS_PER_WAVE * Q_ROW_ELEMS * 2)
    lds_ty = _lds_bf16_ptr_type()
    for c0, w in pow2_segments(D_QK):
        g_view = fx.Tensor(
            fx.make_view(
                fx.add_offset(base, off + fx.Int64(c0)), fx.make_layout((ROWS_PER_WAVE, 1, w), (w, w, 1))
            )
        )
        atom = fx.rocdl.make_tdm_atom(
            g_view,
            [n_valid, None, None],
            strides=[stride_q_seq, stride_q_head, None],
            num_warps=1,
            pad_interval=w,
            pad_amount=Q_ROW_ELEMS - w,
        )
        l_view = fx.Tensor(
            fx.make_view(
                fx.inttoptr(lds_ty, region + fx.Int32(c0 * 2)),
                fx.make_layout((ROWS_PER_WAVE, 1, w), (Q_ROW_ELEMS, Q_ROW_ELEMS, 1)),
            )
        )
        fx.copy_atom_call(atom, g_view, l_view)
    return region


def read_q_fragments(region, lane_idx):
    """Wait for the Q copy and read it as WMMA B operands: [tile][k-step] v16 bf16."""
    tdm_ops.tensor_wait(0)
    v8 = fx.Vector.make_type(_CHUNK_ELEMS, fx.BFloat16)
    lane_base = (
        region
        + (lane_idx % WMMA_M) * fx.Int32(Q_ROW_ELEMS * 2)
        + (lane_idx // WMMA_M) * fx.Int32(_CHUNK_BYTES)
    )
    base = lds_ptr(lane_base)
    frags = [[] for _ in range(Q_TILES)]
    for qt in range(Q_TILES):
        for kt in range(D_QK // WMMA_K):
            imm_lo = qt * WMMA_M * Q_ROW_ELEMS * 2 + kt * WMMA_K * 2
            p_lo = base if imm_lo == 0 else get_element_ptr(base, static_byte_offset=imm_lo)
            p_hi = get_element_ptr(base, static_byte_offset=imm_lo + WMMA_M * 2)
            lo = fx.Vector(llvm.load(v8, p_lo))
            hi = fx.Vector(llvm.load(v8, p_hi))
            frags[qt].append(lo.shuffle(hi, list(range(16))))
    return frags


def kv_copies(*, lds_base, src, stride_seq, stride_head, head, row0, valid, width, row_elems):
    """TDM copies (atom, global view, LDS view) of one N_BLOCK x width K or V tile, split over
    all waves. ``valid`` rows are read; the rest of the tile is zero-filled by the hardware."""
    off = fx.Int64(row0) * fx.Int64(stride_seq) + fx.Int64(head) * fx.Int64(stride_head)
    base = fx.get_iter(src)
    lds_ty = _lds_bf16_ptr_type()
    copies = []
    for c0, w in pow2_segments(width):
        g_view = fx.Tensor(
            fx.make_view(fx.add_offset(base, off + fx.Int64(c0)), fx.make_layout((N_BLOCK, w), (w, 1)))
        )
        atom = fx.rocdl.make_tdm_atom(
            g_view,
            [valid, None],
            strides=[stride_seq, None],
            num_warps=NUM_WAVES,
            pad_interval=w,
            pad_amount=row_elems - w,
        )
        l_view = fx.Tensor(
            fx.make_view(
                fx.inttoptr(lds_ty, lds_base + fx.Int32(c0 * 2)), fx.make_layout((N_BLOCK, w), (row_elems, 1))
            )
        )
        copies.append((atom, g_view, l_view))
    return copies


def k_read_base(lds_base, lane_idx):
    """Per-lane LDS pointer every K fragment read of a tile is an immediate offset from."""
    return lds_ptr(
        lds_base
        + (lane_idx % WMMA_M) * fx.Int32(K_ROW_ELEMS * 2)
        + (lane_idx // WMMA_M) * fx.Int32(_CHUNK_BYTES)
    )


def load_k(base):
    """All K fragment halves of a tile as v8 bf16, listed (kv tile, k-step, half).

    Issued k-step major so the in-order LDS counter releases one k-step of the QK GEMM at a time.
    """
    v8 = fx.Vector.make_type(_CHUNK_ELEMS, fx.BFloat16)
    nkv, nks = N_BLOCK // WMMA_M, D_QK // WMMA_K
    out = [None] * (nkv * nks * 2)
    for ks in range(nks):
        for kv in range(nkv):
            for half in range(2):
                imm = kv * WMMA_M * K_ROW_ELEMS * 2 + (ks * WMMA_K + half * WMMA_M) * 2
                p = base if imm == 0 else get_element_ptr(base, static_byte_offset=imm)
                out[(kv * nks + ks) * 2 + half] = fx.Vector(llvm.load(v8, p))
    return out


def v_read_base(lds_base, lane_idx):
    """Per-lane LDS pointer of the transposing V reads (rows l//16*8 + l%8, columns (l//8)%2*8)."""
    lane_kv = (lane_idx // WMMA_M) * fx.Int32(8) + lane_idx % fx.Int32(8)
    lane_d = ((lane_idx // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)
    return lds_ptr(lds_base + lane_kv * fx.Int32(V_ROW_ELEMS * 2) + lane_d * fx.Int32(2))


def load_v(base):
    """All V^T fragment halves of a tile as v8 bf16, listed (d tile, kv step, half)."""
    v8 = fx.Vector.make_type(_CHUNK_ELEMS, fx.BFloat16)
    out = []
    for dt in range(D_V // WMMA_M):
        for kt in range(N_BLOCK // WMMA_K):
            for half in range(2):
                imm = (kt * WMMA_K + half * WMMA_M) * V_ROW_ELEMS * 2 + dt * WMMA_M * 2
                p = base if imm == 0 else get_element_ptr(base, static_byte_offset=imm)
                out.append(fx.Vector(rocdl.ds_load_tr16_b128(v8, p)))
    return out


@flyc.jit
def _store_rows_if(any_valid, addrs):
    if any_valid:
        for dst, src in addrs:
            rocdl.global_store_async_from_lds_b128(dst, src, 0)


class OutputWriter:
    """Writes a wave's 32 x D_V output rows: accumulators -> padded LDS -> global.

    ``stage`` converts one 16-row tile to bf16; ``flush`` stores both tiles to LDS and issues
    the asynchronous LDS->global copies. Rows at or past ``q_len`` are not written.
    """

    def __init__(self, *, o, stride_o_seq, stride_o_head, q_start, q_len, head, block_x, warp_idx, lane_idx):
        self.o, self.stride_o_seq, self.stride_o_head = o, stride_o_seq, stride_o_head
        self.q_start, self.q_len, self.head = q_start, q_len, head
        self.block_x, self.warp_idx, self.lane_idx = block_x, warp_idx, lane_idx
        self._staged = []

    def stage(self, lds_base, frags, qt):
        region = lds_base + self.warp_idx * fx.Int32(ROWS_PER_WAVE * O_ROW_ELEMS * 2)
        tile = region + fx.Int32(qt * WMMA_M * O_ROW_ELEMS * 2)
        base = lds_ptr(
            tile
            + (self.lane_idx % WMMA_M) * fx.Int32(O_ROW_ELEMS * 2)
            + (self.lane_idx // WMMA_M) * fx.Int32(_CHUNK_BYTES)
        )
        stores = []
        for dt in range(D_V // WMMA_M):
            val = frags[dt].to(fx.BFloat16)
            imm = dt * WMMA_M * 2
            stores.append((val, base if imm == 0 else get_element_ptr(base, static_byte_offset=imm)))
        self._staged.append(stores)
        if qt == 0:
            self._region = region
            self._row0 = self.block_x * fx.Int32(BLOCK_M) + self.warp_idx * fx.Int32(ROWS_PER_WAVE)

    def flush(self):
        rocdl.sched_barrier(0)
        for stores in self._staged:
            for val, p in stores:
                llvm.store(fx.as_ir_value(val), p, alignment=_CHUNK_BYTES)
        addrs, valid_rows = self._row_addresses()
        rocdl.sched_barrier(0)
        rocdl.s_wait_dscnt(0)
        # No wait on the asynchronous stores: they drain before the workgroup retires.
        _store_rows_if(valid_rows > fx.Int32(0), addrs)
        self._staged = []

    def _row_addresses(self):
        # Lane l of round r copies 16-byte chunk r * 32 + l of the wave's rows. Rows past q_len
        # are redirected to the last valid row (an identical rewrite), so every copy issues.
        chunks_per_row = D_V // _CHUNK_ELEMS
        o_base = fx.Int64(fx.ptrtoint(fx.get_iter(self.o)))
        valid_rows = self.q_len - self._row0
        last_valid = fx.min(fx.max(valid_rows - fx.Int32(1), fx.Int32(0)), fx.Int32(ROWS_PER_WAVE - 1))
        addrs = []
        for r in range(ROWS_PER_WAVE * chunks_per_row // WAVE_SIZE):
            c = fx.Int32(r * WAVE_SIZE) + self.lane_idx
            row = c // fx.Int32(chunks_per_row)
            chunk = c % fx.Int32(chunks_per_row)
            row = fx.min(row, last_valid)
            col = chunk * fx.Int32(_CHUNK_ELEMS)
            src = self._region + row * fx.Int32(O_ROW_ELEMS * 2) + chunk * fx.Int32(_CHUNK_BYTES)
            seq = self.q_start + (self._row0 + row)
            off = (
                fx.Int64(seq) * fx.Int64(self.stride_o_seq)
                + fx.Int64(self.head) * fx.Int64(self.stride_o_head)
                + fx.Int64(col)
            )
            addrs.append((global_ptr(o_base + off * fx.Int64(2)), lds_ptr(src)))
        return addrs, valid_rows
