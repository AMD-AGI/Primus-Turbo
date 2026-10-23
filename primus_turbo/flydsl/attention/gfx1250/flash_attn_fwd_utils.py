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

# Adapted from aiter (https://github.com/ROCm/aiter, commit 6963ae9d), where it is
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc., under the MIT license.

"""bf16 Q/K/V/O LDS staging managers of the gfx1250 flash-attention forward.

Vendored from aiter's gfx1250 FlyDSL prefill forward (github.com/ROCm/aiter, commit
6963ae9d) and tuned for Primus-Turbo. Used by ``flash_attn_fwd_kernel``.

Each manager owns the ``global -> LDS -> VGPR (WMMA fragment)`` path of one operand (Q, K
or V), or the way back for O: the LDS layout, the copy schedule and the fragment access.
They are self-contained -- the caller passes the configuration it already maintains (head
dim, gqa_ratio, kv block width, number of waves) to the constructor, and the runtime
``warp_idx`` / ``lane_idx`` to the member functions that need them. The WMMA/tiling facts
intrinsic to the managed layouts live below as private module constants.

The ``16b`` in the class names is the element width: every layout here assumes a
``b128 == 8`` element chunk (``_BF16_BYTES``); an 8-bit (fp8) operand would need its own
manager family.

Contents:
  - ``QManager16bV2`` -- Q loader (per-wave TDM into private padded LDS, no ring).
  - ``KManager16bV2`` -- K loader (TDM into row-major padded LDS, hardware OOB).
  - ``VManager16bV2`` -- V loader (TDM into padded LDS, transpose ``ds_load_tr16_b128``).
  - ``OManager16bV3`` -- O writer (padded LDS ``ds_store`` -> async ``global_store_async_from_lds_b128``).

All memory ops are plain flydsl/rocdl intrinsics with SSA-visible operands, never opaque
inline asm: the forward runs in gfx1250 expert scheduling mode 2, where LLVM inserts the
dependency waits itself and so must see every dependency, including the LDS read after a
TDM write (see ``flash_attn_fwd_kernel.ENABLE_SCHED_MODE2``).

Target: gfx1250 (MI455X), wave32; 2, 4 or 8 waves per manager.
"""

from .flydsl_version import require_flydsl

require_flydsl()

import flydsl.compiler as flyc
import flydsl.expr as fx

# UNSTABLE(gfx1250): raw llvm.load/store -- LDS ds_load/ds_store with folded immediates; fx.ptr_load
# does not accept an !llvm.ptr<3> on 0.3.4.1.
from flydsl._mlir.dialects import llvm as llvm_dialect

# UNSTABLE(gfx1250): sched_barrier, s_wait_dscnt, ds_load_tr16_b128 and the async LDS->global store
# are ODS builders outside rocdl.__all__; fx.rocdl.s_waitcnt raises on gfx1250 (split counters).
from flydsl.expr import rocdl

# UNSTABLE(gfx1250): tdm_ops (tensor_wait) is outside rocdl.__all__ -- no stable TDM wait yet.
from flydsl.expr.rocdl import tdm_ops

from . import buffer_ops
from .flash_attn_utils import WAVE_SIZE, create_llvm_ptr

# ============================================================================
# Manager-intrinsic tiling constants (private -- not the caller's config).
#
# These are fixed by the WMMA instruction + the layouts implemented here, so
# they are NOT parameters. Anything the caller genuinely chooses (hdim, gqa_ratio,
# kv block width, wave count) arrives through a constructor argument instead.
# ============================================================================

# v_wmma_f32_16x16x32_bf16 shape.
_WMMA_M = 16
_WMMA_K = 32
_ELEM = fx.BFloat16
_BF16_BYTES = 2
_CHUNK_ELEMS = 8  # b128 = 8 bf16
_CHUNK_BYTES = _CHUNK_ELEMS * _BF16_BYTES  # 16


# Waves per workgroup the managers support: 8 (forward DEFAULT tiling), and 4 / 2 (the
# SMALL_GRID tiling's K/V loads / its Q and O rows).
_NUM_WAVES_CHOICES = (2, 4, 8)

# KV sequence block choices (rows of one K/V tile).
_N_BLOCK_CHOICES = (32, 64, 128, 256)


# ============================================================================
# K/V loaders -- TDM (Tensor DMA) global->LDS + row-major PADDED LDS.
#
# The whole n_block x hdim tile is copied by ONE TDM atom whose descriptor carries
# base/extent/stride as state, so there is NO per-lane address VALU (saves address VGPRs)
# and the per-dim extent gives HARDWARE OOB (zero-fill) -- no software kv_valid `.select`
# clamps. The LDS is plain row-major with per-row padding (TDM pad_interval/pad_amount, in
# bf16 elements) so the WMMA ds_load fetch stays bank-conflict-free: K pads 8 elems (4 DW /
# 16 B), V pads 16 elems (8 DW / 32 B). Element (row, col) lives at ``row*ROW_ELEMS + col``
# (elements). The read collapses to ONE per-lane base + compile-time immediates.
# ============================================================================

_K_PAD_ELEMS = 8  # 4 DW = 16 B per K row
_V_PAD_ELEMS = 16  # 8 DW = 32 B per V row
_Q_PAD_ELEMS = 8  # 4 DW = 16 B per Q row (matches K)
_O_PAD_ELEMS = 8  # 4 DW = 16 B per O row (conflict-free ds_store_b128)


def _check_num_waves(num_waves):
    if num_waves not in _NUM_WAVES_CHOICES:
        raise NotImplementedError(f"the forward managers support {_NUM_WAVES_CHOICES} waves; got {num_waves}")


def _pow2_segments(width):
    """Split ``width`` (elements) into power-of-two column segments (largest first). The TDM
    pad_interval must be a power of two, so a non-pow2 row would be copied as multiple
    segments, each with a pow2 pad_interval. 128 -> [(0,128)].
    """
    segs, c0, rem = [], 0, width
    while rem > 0:
        w = 1 << (rem.bit_length() - 1)  # largest power of two <= rem
        segs.append((c0, w))
        c0 += w
        rem -= w
    return segs


def _tdm_load_views(
    *,
    ptr_x,
    stride_seq,
    stride_head,
    head,
    row0,
    valid,
    n_rows,
    hdim,
    pad_elems,
    lds_base,
    num_warps,
):
    """Build a LIST of ``(atom, g_view, lds_view)`` TDM global->LDS copies for one
    ``[n_rows, hdim]`` tile into a row-major padded (``hdim + pad_elems`` element row stride) LDS
    block -- PURE (no memory op), issue each with ``fx.copy_atom_call(*view)`` then drain with
    ``tensor_wait(0)``. hdim is split into power-of-two column segments (pad_interval must be pow2):
    segment ``(c0, w)`` copies global cols ``[c0, c0+w)`` -> LDS cols ``[c0, c0+w)`` with
    ``pad_interval=w``, ``pad_amount=(hdim+pad_elems - w)`` so the LDS row still advances by the
    padded stride. Src base = ``ptr_x[row0, head]``; per-row extent ``valid`` = HW OOB zero-fill;
    the ``num_warps`` waves split the tile. Strides in ELEMENTS.
    """
    row_elems = hdim + pad_elems
    off = fx.Int64(row0) * fx.Int64(stride_seq) + fx.Int64(head) * fx.Int64(stride_head)
    base_iter = fx.get_iter(ptr_x)
    lds_ptr_ty = fx.PointerType.get(
        elem_ty=_ELEM.ir_type,
        address_space=fx.AddressSpace.Shared,
        alignment=16,
    )
    views = []
    for c0, w in _pow2_segments(hdim):
        gbase = fx.add_offset(base_iter, off + fx.Int64(c0))
        g_view = fx.Tensor(fx.make_view(gbase, fx.make_layout((n_rows, w), (w, 1))))
        # UNSTABLE(gfx1250): make_tdm_atom lives in rocdl.cdna5, which rocdl.__all__ does not export.
        atom = fx.rocdl.make_tdm_atom(
            g_view,
            [valid, None],
            strides=[stride_seq, None],
            num_warps=num_warps,
            pad_interval=w,
            pad_amount=row_elems - w,
        )
        lds_iter = fx.inttoptr(lds_ptr_ty, lds_base + c0 * _BF16_BYTES)
        lds_view = fx.Tensor(fx.make_view(lds_iter, fx.make_layout((n_rows, w), (row_elems, 1))))
        views.append((atom, g_view, lds_view))
    return views


class QManager16bV2:
    """Q loader (per-warp TDM + row-major padded LDS). No ring buffer: each wave TDM-copies
    ALL of its Q rows -- a ``(WMMA_M * q_tiles_per_wave) x qk_hdim`` tile -- into its own private
    LDS region in one shot (``num_warps=1``, so the wave copies the whole tile; the regions are
    disjoint so no cross-wave sync), then reads them into WMMA B-fragments (scale folded).

    A 3-D ``[n_seq, gqa_ratio, hdim]`` descriptor carries the GQA row-packing (packed row
    ``pr`` -> seq ``pr//gqa``, head ``kv_head*gqa + pr%gqa``); it degenerates to ``[rows, 1, hdim]``
    at gqa==1. LDS is plain row-major with ``hdim + _Q_PAD_ELEMS`` element row stride (matches K).
    ``num_waves`` is the number of waves with distinct q rows.
    """

    def __init__(self, *, qk_hdim, gqa_ratio, num_waves, q_tiles_per_wave):
        if qk_hdim % _WMMA_K != 0:
            raise ValueError(f"qk_hdim must be a multiple of {_WMMA_K}; got {qk_hdim}")
        _check_num_waves(num_waves)
        self.qk_hdim = qk_hdim  # compile-time
        self.gqa_ratio = gqa_ratio  # compile-time
        self.num_waves = num_waves
        self.q_tiles_per_wave = q_tiles_per_wave
        self.k_tiles = qk_hdim // _WMMA_K
        self.rows_per_warp = _WMMA_M * q_tiles_per_wave  # this wave's Q rows (32)
        self.block_m = self.rows_per_warp * num_waves
        self.row_elems = qk_hdim + _Q_PAD_ELEMS  # padded LDS row stride (elems)
        self.row_bytes = self.row_elems * _BF16_BYTES

    def get_lds_size_in_byte(self):
        return self.block_m * self.row_bytes

    def load_q_to_vgpr_part1(
        self,
        *,
        ptr_Q,
        stride_q_seq,
        stride_q_head,
        q_start,
        q_len,
        kv_head,
        block_x,
        warp_idx,
        lane_idx,
        ptr_lds,
    ):
        """Issue this wave's per-warp TDM copy of its ``rows_per_warp x qk_hdim`` Q tile into its
        private LDS region at ``ptr_lds + warp_idx*rows_per_warp*row_bytes``. ``stride_q_seq``/
        ``stride_q_head`` are in ELEMENTS (host convention). Drain + read in ``load_q_to_vgpr_part2``.
        """
        gqa = self.gqa_ratio
        # A wave holds rows_per_warp contiguous packed rows (seq outer, head inner).
        # When gqa <= rows_per_warp it spans n_seq=rows_per_warp/gqa whole head-groups
        # starting at head 0; when gqa > rows_per_warp it holds a single seq's partial
        # head-slice of n_head=rows_per_warp heads at offset packed_row0%gqa. Requires no
        # seq-straddle within the wave (gqa | rows_per_warp or rows_per_warp | gqa).
        assert gqa % self.rows_per_warp == 0 or self.rows_per_warp % gqa == 0, (
            f"gqa_ratio={gqa} must divide or be a multiple of rows_per_warp={self.rows_per_warp}"
        )
        n_head = min(gqa, self.rows_per_warp)
        n_seq = self.rows_per_warp // n_head
        packed_row0 = block_x * self.block_m + warp_idx * self.rows_per_warp
        seq0 = packed_row0 // gqa
        if gqa > self.rows_per_warp:
            head0 = kv_head * gqa + packed_row0 % gqa
        else:
            head0 = kv_head * gqa
        rem = q_len - seq0
        n_seq_valid = fx.max(rem, fx.Int32(0))

        off = fx.Int64(q_start + seq0) * fx.Int64(stride_q_seq) + fx.Int64(head0) * fx.Int64(stride_q_head)
        base_iter = fx.get_iter(ptr_Q)
        warp_region = ptr_lds + warp_idx * (self.rows_per_warp * self.row_bytes)
        lds_ptr_ty = fx.PointerType.get(
            elem_ty=_ELEM.ir_type,
            address_space=fx.AddressSpace.Shared,
            alignment=16,
        )
        # hdim split into pow2 column segments (TDM pad_interval must be pow2): one copy for 128.
        # Each segment (c0, w): pad_interval=w, pad_amount=row_elems-w so the padded LDS row
        # stride is preserved.
        for c0, w in _pow2_segments(self.qk_hdim):
            gbase = fx.add_offset(base_iter, off + fx.Int64(c0))
            g_view = fx.Tensor(fx.make_view(gbase, fx.make_layout((n_seq, n_head, w), (n_head * w, w, 1))))
            atom = fx.rocdl.make_tdm_atom(
                g_view,
                [n_seq_valid, None, None],
                strides=[stride_q_seq, stride_q_head, None],
                num_warps=1,
                pad_interval=w,
                pad_amount=self.row_elems - w,
            )
            lds_iter = fx.inttoptr(lds_ptr_ty, warp_region + c0 * _BF16_BYTES)
            lds_view = fx.Tensor(
                fx.make_view(
                    lds_iter,
                    fx.make_layout((n_seq, n_head, w), (n_head * self.row_elems, self.row_elems, 1)),
                )
            )
            fx.copy_atom_call(atom, g_view, lds_view)
        self._warp_region = warp_region
        self._lane_idx = lane_idx

    def load_q_to_vgpr_part2(self, *, scale):
        """Drain this wave's Q TDM (``tensor_wait(0)``) and read its ``rows_per_warp x qk_hdim``
        tile into WMMA B-fragments (``scale`` folded). Returns a length-R list; entry ``qt`` is
        that q-tile's list of ``k_tiles`` v16-bf16 fragments.

        Read collapses to 1 per-lane base + compile-time immediates (like K): lane ``l`` reads row
        ``l%16``, d-byte ``(l//16)*16``; fragment (qt, tile) = base + ``qt*16*row_bytes +
        tile*32*2`` (lo) and ``+ 16*2`` more (hi 8-col half)."""
        tdm_ops.tensor_wait(0)
        v8_ty = fx.Vector.make_type(_CHUNK_ELEMS, _ELEM)
        scale_bf16 = scale.to(_ELEM)
        lane = self._lane_idx
        lane_base = (
            self._warp_region
            + (lane % _WMMA_M) * self.row_bytes
            + (lane // _WMMA_M) * (_CHUNK_ELEMS * _BF16_BYTES)
        )
        base = create_llvm_ptr(lane_base, address_space=3)
        q_frags_list = [[] for _ in range(self.q_tiles_per_wave)]
        for qt in range(self.q_tiles_per_wave):
            for tile in range(self.k_tiles):
                imm_lo = qt * _WMMA_M * self.row_bytes + tile * _WMMA_K * _BF16_BYTES
                imm_hi = imm_lo + _WMMA_M * _BF16_BYTES
                p_lo = base if imm_lo == 0 else buffer_ops.get_element_ptr(base, static_byte_offset=imm_lo)
                p_hi = buffer_ops.get_element_ptr(base, static_byte_offset=imm_hi)
                lo = fx.Vector(llvm_dialect.load(v8_ty, p_lo))
                hi = fx.Vector(llvm_dialect.load(v8_ty, p_hi))
                q_frags_list[qt].append(lo.shuffle(hi, list(range(16))) * scale_bf16)
        return q_frags_list


class KManager16bV2:
    """K loader (TDM + row-major padded LDS). B-fragments in ``_qk_gemm`` order
    ``[(kv, dt, half)...]``, natural ``ds_load_b128``."""

    def __init__(self, *, qk_hdim, n_block, num_waves):
        if qk_hdim % _WMMA_K != 0:
            raise ValueError(f"qk_hdim must be a multiple of {_WMMA_K}; got {qk_hdim}")
        if n_block not in _N_BLOCK_CHOICES:
            raise ValueError(f"n_block must be one of {_N_BLOCK_CHOICES}; got {n_block}")
        _check_num_waves(num_waves)
        self.qk_hdim = qk_hdim
        self.n_block = n_block
        self.num_waves = num_waves
        self.row_elems = qk_hdim + _K_PAD_ELEMS  # padded LDS row stride (elems)
        self.row_bytes = self.row_elems * _BF16_BYTES

    def get_lds_size_in_byte(self):
        return self.n_block * self.row_bytes

    def load_views(self, *, ptr_lds, ptr_K, stride_k_seq, stride_k_head, kv_head, kv_row0, kv_valid):
        """Return a LIST of ``(atom, g_view, lds_view)`` TDM copies for this block's K tile into the
        padded LDS at ``ptr_lds`` (fx.Int32 byte base) -- one per pow2 hdim segment. Pure
        (hoistable); issue each with ``fx.copy_atom_call(*view)``, drain with
        ``tdm_ops.tensor_wait(0)``."""
        return _tdm_load_views(
            ptr_x=ptr_K,
            stride_seq=stride_k_seq,
            stride_head=stride_k_head,
            head=kv_head,
            row0=kv_row0,
            valid=kv_valid,
            n_rows=self.n_block,
            hdim=self.qk_hdim,
            pad_elems=_K_PAD_ELEMS,
            lds_base=ptr_lds,
            num_warps=self.num_waves,
        )

    def ds_load_ptrs(self, *, ptr_lds, lane_idx):
        """The **1** per-lane ds_load base pointer (a list of 1) that
        ``load_k_to_reg`` reaches every ``ds_load_b128`` from by a compile-time immediate.
        Lane ``l`` fetches at row ``l%16``, d-byte ``(l//16)*16`` of the block; every
        fragment ``(kv, dt, half)`` is that base + a lane-independent immediate."""
        lane_base = (
            ptr_lds
            + (lane_idx % _WMMA_M) * self.row_bytes
            + (lane_idx // _WMMA_M) * (_CHUNK_ELEMS * _BF16_BYTES)
        )
        return [create_llvm_ptr(lane_base, address_space=3)]

    def load_k_to_reg(self, base_ptrs):
        """Burst all K ``ds_load_b128`` from the row-major padded block, in ``_qk_gemm``
        order ``[(kv, dt, half)...]``, off the single base of ``ds_load_ptrs``. Fragment
        (kv, dt, half) = base + ``kv*16*row_bytes + (dt*32 + half*16)*2`` bytes."""
        v8_ty = fx.Vector.make_type(8, _ELEM)
        NKV = self.n_block // _WMMA_M
        NDT = self.qk_hdim // _WMMA_K
        base = base_ptrs[0]
        out = []
        for kv in range(NKV):
            for dt in range(NDT):
                for half in range(2):
                    imm = kv * _WMMA_M * self.row_bytes + (dt * _WMMA_K + half * _WMMA_M) * _BF16_BYTES
                    p = base
                    if imm:
                        p = buffer_ops.get_element_ptr(base, static_byte_offset=imm)
                    out.append(fx.Vector(llvm_dialect.load(v8_ty, p)))
        return out


class VManager16bV2:
    """V loader (TDM + row-major padded LDS, 32 B/row pad). A-fragments in ``_pv_gemm``
    order ``[(dt, kt, half)...]``, transpose
    ``ds_load_tr16_b128`` crossbar ``V[kv+(l//16)*8+l%8, d+((l//8)%2)*8]``)."""

    def __init__(self, *, v_hdim, n_block, num_waves):
        if v_hdim % _WMMA_M != 0:
            raise ValueError(f"v_hdim must be a multiple of {_WMMA_M}; got {v_hdim}")
        if n_block % _WMMA_K != 0:
            raise ValueError(f"n_block must be a multiple of {_WMMA_K}; got {n_block}")
        _check_num_waves(num_waves)
        self.v_hdim = v_hdim
        self.n_block = n_block
        self.num_waves = num_waves
        self.row_elems = v_hdim + _V_PAD_ELEMS
        self.row_bytes = self.row_elems * _BF16_BYTES

    def get_lds_size_in_byte(self):
        return self.n_block * self.row_bytes

    def load_views(self, *, ptr_lds, ptr_V, stride_v_seq, stride_v_head, kv_head, kv_row0, kv_valid):
        """Return a LIST of ``(atom, g_view, lds_view)`` TDM copies for this block's V tile (v_hdim=128
        is pow2 -> one copy). Pure; issue each with ``fx.copy_atom_call(*view)``, drain with
        ``tdm_ops.tensor_wait(0)``."""
        return _tdm_load_views(
            ptr_x=ptr_V,
            stride_seq=stride_v_seq,
            stride_head=stride_v_head,
            head=kv_head,
            row0=kv_row0,
            valid=kv_valid,
            n_rows=self.n_block,
            hdim=self.v_hdim,
            pad_elems=_V_PAD_ELEMS,
            lds_base=ptr_lds,
            num_warps=self.num_waves,
        )

    def ds_load_ptrs(self, *, ptr_lds, lane_idx):
        """The **1** per-lane transpose-load base pointer (a list of 1).
        Crossbar fetch at ``kv = (l//16)*8 + l%8``, ``d = ((l//8)%2)*8`` of the block;
        every fragment ``(dt, kt, half)`` is that base + a lane-independent immediate.
        """
        lane_kv = (lane_idx // _WMMA_M) * 8 + lane_idx % 8
        lane_d = ((lane_idx // 8) % 2) * 8
        lane_base = ptr_lds + lane_kv * self.row_bytes + lane_d * _BF16_BYTES
        return [create_llvm_ptr(lane_base, address_space=3)]

    def load_v_to_reg(self, base_ptrs, d_tiles):
        """Burst the V ``ds_load_tr16_b128`` of the first ``d_tiles`` 16-wide d-tiles off the
        single base of ``ds_load_ptrs``, in ``_pv_gemm`` order ``[(dt, kt, half)...]``: all
        ``v_hdim // 16`` of them, or a d-split wave's part (the caller offsets the base to it).
        Fragment (dt, kt, half) = base + ``(kt*32 + half*16)*row_bytes + dt*16*2`` bytes.
        """
        v8_ty = fx.Vector.make_type(8, _ELEM)
        nkt = self.n_block // _WMMA_K
        base = base_ptrs[0]
        out = []
        for dt in range(d_tiles):
            for kt in range(nkt):
                for half in range(2):
                    imm = (kt * _WMMA_K + half * _WMMA_M) * self.row_bytes + dt * _WMMA_M * _BF16_BYTES
                    p = base
                    if imm:
                        p = buffer_ops.get_element_ptr(base, static_byte_offset=imm)
                    out.append(fx.Vector(rocdl.ds_load_tr16_b128(v8_ty, p)))
        return out


# ============================================================================
# O writer (VGPR WMMA accumulator -> padded LDS -> async LDS->global store)
# ============================================================================


class OManager16bV3:
    """O writer: conflict-free PADDED LDS ds_store + per-b128 ``global_store_async_from_lds_b128``
    (LDS->VRAM), NO TDM. The LDS rows are padded (bank-conflict-free ``ds_store_b128``, 4 DW pad),
    which a TDM store would not honour (it ignores the padding), so every b128 is addressed here;
    the async store moves LDS->global directly, with no VGPR re-read.

    Accumulator (d-tile k, lane l = ``O[q=l%16, d=16k+(l//16)*8+{0..7}]``) -> row-major PADDED LDS
    ``[16, v_hdim]`` (row stride v_hdim+_O_PAD_ELEMS). Then the 16 x (v_hdim/8) b128 chunks are stored
    LDS->global over ``n_rounds`` waves of 32 lanes: round r lane l -> chunk c=r*32+l, row=c//cpr,
    d_chunk=c%cpr (cpr = v_hdim/8) -> coalesced (consecutive lanes = consecutive global).
    Rows at seq>=q_len are masked off because async stores have no bounds check.
    ``v_hdim`` is the head-dim columns this wave writes (all, or a d-split wave's part, placed by
    ``o_base_elems``); ``num_waves`` is the number of waves with distinct q rows.
    """

    def __init__(self, *, v_hdim, gqa_ratio, num_waves, q_tiles_per_wave):
        if v_hdim % _WMMA_M != 0:
            raise ValueError(f"v_hdim must be a multiple of {_WMMA_M}; got {v_hdim}")
        _check_num_waves(num_waves)
        self.v_hdim = v_hdim
        self.gqa_ratio = gqa_ratio
        self.num_waves = num_waves
        self.q_tiles_per_wave = q_tiles_per_wave
        self.d_tiles = v_hdim // _WMMA_M
        self.rows_per_warp = _WMMA_M * q_tiles_per_wave
        self.block_m = self.rows_per_warp * num_waves
        self.row_elems = v_hdim + _O_PAD_ELEMS  # PADDED (conflict-free ds_store)
        self.row_bytes = self.row_elems * _BF16_BYTES
        self.chunks_per_row = v_hdim // _CHUNK_ELEMS  # b128 chunks per row
        self._pending = []  # per-qtile (tile_lds, base_row)

    def get_lds_size_in_byte(self):
        return self.num_waves * self.rows_per_warp * self.row_bytes

    def store_o_to_vram(
        self,
        *,
        ptr_O,
        o_base_elems,
        stride_o_seq,
        stride_o_head,
        q_start,
        q_len,
        kv_head,
        block_x,
        warp_idx,
        lane_idx,
        ptr_lds,
        o_frags,
        qtile=0,
    ):
        """Called once per q-tile: cvt accumulator->bf16 + compute LDS write pointers (VALU), STASH;
        the LAST call flushes the whole WARP: (a) ds_stores back-to-back, (b) ALL rows_per_warp rows'
        global addresses together (all live -> distinct regs, so no later store's addr VALU reuses an
        earlier still-in-flight store's source reg -- a write-after-read stall when split per-tile), (c) one
        s_wait_dscnt(0) then ONE back-to-back async burst. The warp's q_tiles_per_wave 16-row tiles are
        CONTIGUOUS in LDS (tile qt at warp_region + qt*16*row_bytes) so they form one 32-row block.
        """
        if len(o_frags) != self.d_tiles:
            raise ValueError(f"expected {self.d_tiles} O frags; got {len(o_frags)}")
        warp_region = ptr_lds + warp_idx * (self.rows_per_warp * self.row_bytes)
        tile_lds = warp_region + qtile * _WMMA_M * self.row_bytes

        # (1) Per call: cvt fp32->bf16 + compute LDS write pointers (no issue yet). Stash.
        lane_base = (
            tile_lds
            + (lane_idx % _WMMA_M) * self.row_bytes
            + (lane_idx // _WMMA_M) * (_CHUNK_ELEMS * _BF16_BYTES)
        )
        base_ptr = create_llvm_ptr(lane_base, address_space=3)
        ds_ops = []
        for k in range(self.d_tiles):
            bf = o_frags[k].to(_ELEM)  # cvt
            imm = k * _WMMA_M * _BF16_BYTES
            p = base_ptr if imm == 0 else buffer_ops.get_element_ptr(base_ptr, static_byte_offset=imm)
            ds_ops.append((bf, p))
        self._pending.append(ds_ops)
        if qtile == 0:
            self._warp_region = warp_region
            self._warp_base = block_x * self.block_m + warp_idx * self.rows_per_warp
            self._cfg = {
                "ptr_O": ptr_O,
                "o_base_elems": o_base_elems,
                "stride_o_seq": stride_o_seq,
                "stride_o_head": stride_o_head,
                "q_start": q_start,
                "q_len": q_len,
                "kv_head": kv_head,
                "lane_idx": lane_idx,
            }

        if qtile != self.q_tiles_per_wave - 1:
            return

        # (2) LAST call -- flush the whole warp.
        rocdl.sched_barrier(0)
        for ds_ops in self._pending:
            for bf, p in ds_ops:
                llvm_dialect.store(bf.ir_value(), p, alignment=_CHUNK_BYTES)
        addrs, valid_rows = self._warp_addrs(self._warp_region, self._warp_base, **self._cfg)
        rocdl.sched_barrier(0)  # address VALU above, store burst below -- no interleave
        rocdl.s_wait_dscnt(0)  # all ds_stores landed (every async row reads a full padded row)

        @flyc.jit
        def _burst():
            if valid_rows > fx.Int32(0):
                for gdst, lsrc in addrs:
                    rocdl.global_store_async_from_lds_b128(gdst, lsrc, 0)

        _burst()
        # No s_wait_asynccnt: HW drains the async stores' LDS reads at workgroup retire.
        self._pending = []

    def _warp_addrs(
        self,
        warp_region,
        warp_base,
        *,
        ptr_O,
        o_base_elems,
        stride_o_seq,
        stride_o_head,
        q_start,
        q_len,
        kv_head,
        lane_idx,
    ):
        """Pure ALU: (gdst, lsrc) for every round of the warp's ``rows_per_warp x v_hdim`` block,
        row-coalesced. OOB rows (seq>=q_len) clamp to the warp's last valid row -- a redundant,
        idempotent write (a real lane of THIS warp stores the same bytes; warps own disjoint rows so
        no cross-warp race) -- so every store issues UNCONDITIONALLY (no per-store branch) and the
        burst stays back-to-back. Returns (addrs, valid_rows); valid_rows<=0 => whole warp OOB.
        """
        gqa = self.gqa_ratio
        cpr = self.chunks_per_row
        rpw = self.rows_per_warp
        n_rounds = (rpw * cpr) // WAVE_SIZE
        ptr_O_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(ptr_O)))
        # packed rows [warp_base, warp_base+rpw) valid iff pr//gqa < q_len iff pr < q_len*gqa.
        valid_rows = q_len * gqa - warp_base
        last_valid = fx.min(fx.max(valid_rows - 1, fx.Int32(0)), fx.Int32(rpw - 1))
        addrs = []
        for r in range(n_rounds):
            c = fx.Int32(r * WAVE_SIZE) + lane_idx
            row = c // cpr
            d_chunk = c % cpr
            srow = fx.min(row, last_valid)  # OOB rows -> warp's last valid row (idempotent redirect)
            d = d_chunk * _CHUNK_ELEMS
            lds_src = warp_region + srow * self.row_bytes + d_chunk * _CHUNK_BYTES
            pr = warp_base + srow
            seq_g = pr // gqa if gqa > 1 else pr
            head = kv_head * gqa + pr % gqa if gqa > 1 else kv_head
            token = q_start + seq_g
            off64 = (
                fx.Int64(o_base_elems)
                + fx.Int64(token) * fx.Int64(stride_o_seq)
                + fx.Int64(head) * fx.Int64(stride_o_head)
                + fx.Int64(d)
            )
            gdst = create_llvm_ptr(ptr_O_i64 + off64 * fx.Int64(_BF16_BYTES), address_space=1)
            lsrc = create_llvm_ptr(lds_src, address_space=3)
            addrs.append((gdst, lsrc))
        return addrs, valid_rows
