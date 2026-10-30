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

# UNSTABLE(gfx1250): the TDM atom, tensor_load_to_lds with hand-built descriptors, tensor_wait,
# ds_load_tr16_b128, the async LDS->global store, sched_barrier and s_wait_dscnt have no stable API in
# FlyDSL 0.3.4.1; LDS loads/stores are raw LLVM ops on pointers built from i32 byte addresses.
from flydsl._mlir.dialects import llvm
from flydsl.expr import rocdl

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


def _tdm_cfg(w, row_elems):
    """TDM group 1 word 0 for 2-byte elements: pad (row_elems - w) elements after every w."""
    interval_dw, amount_dw = w // 2, (row_elems - w) // 2
    return _i32_word((1 << 16) | (1 << 20) | ((interval_dw.bit_length() - 2) << 22) | ((amount_dw - 1) << 25))


def _tdm_rows(lds, addr, rem, w, rows, stride, cfg):
    """Copy ``rows`` rows of ``w`` elements from global ``addr`` (row stride ``stride`` elements) to LDS
    ``lds``; rows at or past ``rem`` are zero-filled."""
    g1_2 = fx.Int32(0x7FFF) | (rem << fx.Int32(16))  # tensor_dim0 = 0x7FFFFFFF (no clamp)
    g1_3 = rem.shrui(fx.Int32(16))
    z4 = fx.as_ir_value(fx.Vector.filled(4, 0, fx.Int32))
    z8 = fx.as_ir_value(fx.Vector.filled(8, 0, fx.Int32))
    lo = fx.Int32(addr)
    hi = fx.Int32(addr >> fx.Int64(32)) | fx.Int32(_i32_word(1 << 31))
    g0 = fx.Vector.from_elements([fx.Int32(1), lds, lo, hi], fx.Int32)
    g1 = fx.Vector.from_elements(
        [
            fx.Int32(cfg),
            fx.Int32(-65536),  # tensor_dim0 [15:0]
            g1_2,
            fx.Int32(w << 16) | g1_3,  # tile_dim0 | tensor_dim1 [31:16]
            fx.Int32(rows),  # tile_dim1
            fx.Int32(stride),  # row stride (elements)
            fx.Int32(0),
            fx.Int32(0),
        ],
        fx.Int32,
    )
    rocdl.tensor_load_to_lds(fx.as_ir_value(g0), fx.as_ir_value(g1), z4, z4, z8, cache_policy=0)


def issue_q_copy(
    *, q_addr, stride_q_seq, stride_q_head, q_start, q_len, head, block_x, warp_idx, region, tile_stride
):
    """TDM-copy wave ``warp_idx``'s two 16-row q tiles (block rows qt * tile_stride + warp_idx * 32;
    q at global byte address ``q_addr``) to its 32 LDS rows at ``region``; rows past ``q_len`` are
    zero-filled."""
    for qt in range(Q_TILES):
        seq0 = block_x * fx.Int32(BLOCK_M) + fx.Int32(qt * tile_stride) + warp_idx * fx.Int32(ROWS_PER_WAVE)
        n_valid = fx.max(q_len - seq0, fx.Int32(0))
        base = q_addr + (
            fx.Int64(q_start + seq0) * fx.Int64(stride_q_seq) + fx.Int64(head) * fx.Int64(stride_q_head)
        ) * fx.Int64(2)
        for c0, w in pow2_segments(D_QK):
            _tdm_rows(
                region + fx.Int32((qt * WMMA_M * Q_ROW_ELEMS + c0) * 2),
                base + fx.Int64(c0 * 2),
                n_valid,
                w,
                WMMA_M,
                stride_q_seq,
                _tdm_cfg(w, Q_ROW_ELEMS),
            )


def read_q_fragments(region, lane_idx):
    """Read this wave's Q (copied and waited for) as WMMA B operands: [tile][k-step] v16 bf16."""
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


ROWS_PER_COPY = N_BLOCK // (NUM_WAVES // 2)  # K / V tile rows each low wave (0-3) copies


def _i32_word(x):
    return x - (1 << 32) if x >= 1 << 31 else x


class KVTileLoader:
    """A low wave's share of the TDM copies of an N_BLOCK-row K / V tile.

    Each low wave (0-3) copies its ROWS_PER_COPY rows of K columns [0, 128), K columns [128, 192)
    and V, one op each (a TDM pad interval is a power of two). The descriptors are built here, once:
    per tile only the global address and the valid-row count change, a few SALU ops instead of the
    copy atom's whole descriptor (with a v_med3 + readfirstlane for the row clamp) on every issue.
    Rows past the workgroup's last key are zero-filled by the hardware, as with the atom.
    """

    def __init__(
        self,
        *,
        k,
        v,
        stride_k_seq,
        stride_k_head,
        stride_v_seq,
        stride_v_head,
        head,
        kv_start,
        kv_len_wg,
        warp_idx,
        v_lds_off,
    ):
        r0 = warp_idx * fx.Int32(ROWS_PER_COPY)
        self.rem0 = kv_len_wg - r0
        self.copies = []
        for src, sseq, shead, width, row_elems, lds_off in (
            (k, stride_k_seq, stride_k_head, D_QK, K_ROW_ELEMS, 0),
            (v, stride_v_seq, stride_v_head, D_V, V_ROW_ELEMS, v_lds_off),
        ):
            base = src + (
                fx.Int64(kv_start + r0) * fx.Int64(sseq) + fx.Int64(head) * fx.Int64(shead)
            ) * fx.Int64(2)
            row_bytes = fx.Int64(sseq) * fx.Int64(2)
            lds = r0 * fx.Int32(row_elems * 2) + fx.Int32(lds_off)
            for c0, w in pow2_segments(width):
                # group 1 word 0: 2-byte elements, pad (row_elems - w) elements after every w
                interval_dw, amount_dw = w // 2, (row_elems - w) // 2
                cfg = (1 << 16) | (1 << 20) | ((interval_dw.bit_length() - 2) << 22) | ((amount_dw - 1) << 25)
                self.copies.append(
                    dict(
                        base=base + fx.Int64(c0 * 2),
                        row_bytes=row_bytes,
                        stride=sseq,
                        lds=lds + fx.Int32(c0 * 2),
                        cfg=_i32_word(cfg),
                        w=w,
                    )
                )

    def issue(self, slot_base, row):
        """Copy this wave's rows of the tile at kv row ``row`` into the K / V slot at ``slot_base``."""
        # tensor_dim1 = this wave's rows left; no upper clamp (an smin / smax pair becomes v_med3 +
        # readfirstlane): tile_dim1 stops the copy at the slice end either way.
        rem = fx.max(self.rem0 - row, fx.Int32(0))
        g1_2 = fx.Int32(0x7FFF) | (rem << fx.Int32(16))  # tensor_dim0 = 0x7FFFFFFF (no clamp)
        g1_3 = rem.shrui(fx.Int32(16))
        z4 = fx.as_ir_value(fx.Vector.filled(4, 0, fx.Int32))
        z8 = fx.as_ir_value(fx.Vector.filled(8, 0, fx.Int32))
        for c in self.copies:
            addr = c["base"] + fx.Int64(row) * c["row_bytes"]
            lo = fx.Int32(addr)
            hi = fx.Int32(addr >> fx.Int64(32)) | fx.Int32(_i32_word(1 << 31))
            g0 = fx.Vector.from_elements([fx.Int32(1), slot_base + c["lds"], lo, hi], fx.Int32)
            g1 = fx.Vector.from_elements(
                [
                    fx.Int32(c["cfg"]),
                    fx.Int32(-65536),  # tensor_dim0 [15:0]
                    g1_2,
                    fx.Int32(c["w"] << 16) | g1_3,  # tile_dim0 | tensor_dim1 [31:16]
                    fx.Int32(ROWS_PER_COPY),  # tile_dim1
                    fx.Int32(c["stride"]),  # row stride (elements)
                    fx.Int32(0),
                    fx.Int32(0),
                ],
                fx.Int32,
            )
            rocdl.tensor_load_to_lds(fx.as_ir_value(g0), fx.as_ir_value(g1), z4, z4, z8, cache_policy=0)


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

    def __init__(
        self,
        *,
        o_addr,
        stride_o_seq,
        stride_o_head,
        q_start,
        q_len,
        head,
        block_x,
        warp_idx,
        lane_idx,
        tile_stride,
    ):
        self.o_addr, self.stride_o_seq, self.stride_o_head = o_addr, stride_o_seq, stride_o_head
        self.q_start, self.q_len, self.head = q_start, q_len, head
        self.block_x, self.warp_idx, self.lane_idx = block_x, warp_idx, lane_idx
        self.tile_stride = tile_stride
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
        # Lane l of round r copies 16-byte chunk r * 32 + l of the wave's rows: rounds [0, 8) are
        # q tile 0 (block rows warp_idx * 32 + [0, 16)), rounds [8, 16) q tile 1 (tile_stride rows
        # further). seq_len_q is a multiple of 64, so a q tile is valid or past q_len as a whole; the
        # rows of an invalid q tile 1 are redirected to q tile 0's (an identical rewrite), so every
        # copy issues.
        chunks_per_row = D_V // _CHUNK_ELEMS
        rows_per_round = WAVE_SIZE // chunks_per_row
        o_base = self.o_addr
        valid_rows = self.q_len - self._row0
        lane_row = self.lane_idx // fx.Int32(chunks_per_row)
        chunk = self.lane_idx % fx.Int32(chunks_per_row)
        row_bytes = fx.Int64(self.stride_o_seq) * fx.Int64(2)
        g = []
        for qt in range(Q_TILES):
            seq = self.q_start + self._row0 + fx.Int32(qt * self.tile_stride) + lane_row
            off = (
                fx.Int64(seq) * fx.Int64(self.stride_o_seq)
                + fx.Int64(self.head) * fx.Int64(self.stride_o_head)
                + fx.Int64(chunk * fx.Int32(_CHUNK_ELEMS))
            )
            g.append(o_base + off * fx.Int64(2))
        l0 = self._region + lane_row * fx.Int32(O_ROW_ELEMS * 2) + chunk * fx.Int32(_CHUNK_BYTES)
        tile1_valid = valid_rows > fx.Int32(self.tile_stride)
        g1 = tile1_valid.select(g[1], g[0])
        l1 = tile1_valid.select(l0 + fx.Int32(WMMA_M * O_ROW_ELEMS * 2), l0)
        addrs = []
        rounds_per_tile = WMMA_M // rows_per_round
        for r in range(ROWS_PER_WAVE * chunks_per_row // WAVE_SIZE):
            qt, rr = divmod(r, rounds_per_tile)
            gb, lb = (g[0], l0) if qt == 0 else (g1, l1)
            dst = gb + row_bytes * fx.Int64(rr * rows_per_round)
            src = lb + fx.Int32(rr * rows_per_round * O_ROW_ELEMS * 2)
            addrs.append((global_ptr(dst), lds_ptr(src)))
        return addrs, valid_rows
