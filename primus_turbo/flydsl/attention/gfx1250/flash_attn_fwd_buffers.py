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

"""16-bit (bf16/fp16) Q/K/V/O LDS staging managers for the gfx1250 flash-attention forward.

Vendored from aiter's gfx1250 FlyDSL prefill forward (github.com/ROCm/aiter, commit
6963ae9d) and tuned for Primus-Turbo. Used by ``flash_attn_fwd_kernel`` and
``flash_attn_fwd_small_grid_kernel``.

Each manager owns the ``global -> LDS (async) -> VGPR (WMMA fragment)`` path for
one 16-bit-element operand (Q, K or V): the LDS swizzle, the async copy schedule
and the fragment read. They are self-contained -- the only things a caller passes
in are the *configuration* it already maintains (hdim, gqa_ratio, kv block width,
number of waves) via the constructor, and the runtime ``warp_idx`` / ``lane_idx``
into the specific member functions that need them. Nothing here reads a shared
module-level tiling constant from the kernel; the WMMA/tiling facts intrinsic to
the managed layout live below as private module constants.

The ``16b`` suffix names the element width (16-bit): every swizzle here assumes a
``b128 == 8`` element chunk (``_BF16_BYTES``). An 8-bit (fp8) variant would need
its own manager family (different chunk arithmetic), hence the explicit width tag.

Contents:
  - ``QManager16bV2`` -- Q loader (per-warp TDM into private padded LDS, no ring).
  - ``KManager16bV2`` -- K loader (per-warp TDM into row-major padded LDS, HW OOB).
  - ``VManager16bV2`` -- V loader (per-warp TDM into padded LDS, transpose ``ds_load_tr16_b128``).
  - ``OManager16bV1`` -- O writer (WMMA accumulator -> swizzled LDS -> coalesced buffer_store).
  - ``OManager16bV3`` -- O writer (padded LDS ``ds_store`` -> async ``global_store_from_lds_b128``).

Target: gfx1250 (MI400 / mi450), wave32; 8 waves per threadgroup by default (the
``num_waves`` constructor argument also accepts 2 and 4 for the small-grid kernel).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx

# UNSTABLE(gfx1250): raw llvm.load/store -- LDS ds_load/ds_store with folded immediates; fx.ptr_load
# does not accept an !llvm.ptr<3> on 0.3.4.1.
from flydsl._mlir.dialects import llvm as llvm_dialect

# UNSTABLE(gfx1250): sched_barrier, s_wait_dscnt/asynccnt, ds_load_tr16_b128 and the async-LDS ODS
# ops are outside rocdl.__all__; fx.rocdl.s_waitcnt raises on gfx1250 (split counters).
from flydsl.expr import rocdl

# UNSTABLE(gfx1250): tdm_ops (tensor_wait) is outside rocdl.__all__ -- no stable TDM wait yet.
from flydsl.expr.rocdl import tdm_ops

from . import buffer_ops

# Re-export only: flash_attn_fwd_kernel / flash_attn_fwd_small_grid_kernel import ``_ir`` from here.
from .common import create_llvm_ptr, imax, imin

# ============================================================================
# Manager-intrinsic tiling constants (private -- not the caller's config).
#
# These are fixed by the WMMA instruction + the swizzles implemented here, so
# they are NOT parameters. Anything the caller genuinely chooses (hdim, gqa_ratio,
# kv block width, wave count) arrives through a constructor argument instead.
# ============================================================================

# v_wmma_f32_16x16x32_bf16/f16 shape.
_WMMA_M = 16
_WMMA_K = 32
_BF16_BYTES = 2
_CHUNK_ELEMS = 8  # b128 = 8 bf16
_CHUNK_BYTES = _CHUNK_ELEMS * _BF16_BYTES  # 16

# gfx1250 is wave32; every swizzle/reshape here assumes 32 lanes per wave. FlyDSL
# only exposes the wave size compiler-side (GPUTarget.warp_size, from the arch),
# not as a trace-time Python int, so it is a named constant (== the kernels'
# WAVE_SIZE). An fp8/CDNA (wave64) port would revisit the whole file, not just this.
_WAVE_LANES = 32

# Default 8-wave ("m16x8") threadgroup; override via the ``num_waves`` ctor arg.
_DEFAULT_NUM_WAVES = 8

# KV sequence block choices (columns of one QK GEMM tile).
_N_BLOCK_CHOICES = (32, 64, 128, 256)
_DEFAULT_N_BLOCK = 64

# O epilogue software pipeline (OManager16b, see its docstring). 64-col transpose
# units keep each coalesced global store a full 128B row; _O_INFLIGHT_UNITS units
# stay resident in an LDS ring and overlap.
_O_COLS_PER_TILE = 64
_O_INFLIGHT_UNITS = 2
_O_DSCNT_MAX = 63  # s_wait_dscnt SIMM16[5:0]

# LDS budget the O ring fits inside (the caller's non-current K|V slot):
# 8 waves * 2 units * 2KB = 32KB.
_O_LDS_BUDGET_BYTES = 32 * 1024


# O staging (OManager16b): no padding -- an XOR swizzle on the 8-bf16 chunk index
# makes both the b128 store and b128 read bank-conflict-free (same idea as the
# K/QManager swizzle). MI400 LDS = 64 banks x 4 B, so an 8-bf16 (16 B) chunk spans
# one 4-bank group and bank_group(q, slot) = (q*G + slot) % 16 with G = v_hdim/8
# chunks per row. The store touches a fixed chunk-column across all 16 q-rows, so
# the swizzle must spread the slot over all 16 q; the read touches one full row
# (all G chunks) so any within-row bijection is already conflict-free. The swizzle
# ``slot = chunk ^ (q >> shift)``, shift = max(0, 4 - log2(G)), satisfies both:
# for v_hdim=128 (G=16) it is the full ``chunk ^ q``; the shift folds q's low bits
# already carried by the ``q*G`` term when G < 16. Verified conflict-free for
# v_hdim in {64,128,256}.


# ---- gfx1250 Expert Scheduling Mode 2 --------------------------------------
# DEP_MODE=2 turns the HW VA_VDST/VM_VSRC issue interlocks OFF. It is enabled via
# the `amdgpu-expert-scheduling-mode` LLVM hint the kernel passes at jit time (see
# _ensure_*_kernel). With the hint on, LLVM emits the DEP_MODE=2 setreg AND inserts
# ALL the dependency covers itself (post-RA depctr waits) for the plain intrinsic
# memory ops below: the SSA-visible RAW/WAR hazards AND the LDS RAW between the async
# global->LDS load and the ds_load that reads it back. So the kernel emits nothing
# but ordinary flydsl intrinsics in BOTH modes; mode 2 is codegen-identical to mode 0
# plus the one SCHED_MODE setreg, and validated NaN-free (seq 512..16384, causal +
# non-causal) at perf parity with mode 0.
#
# Do NOT hand-write these memory ops as opaque inline asm with manual depctr covers:
# the opacity hides the async-store -> ds-load LDS RAW from LLVM (no SSA edge, LDS
# unmodeled), which mis-schedules under DEP_MODE=2 and produces silent NaN at scale.
# Plain intrinsics expose the dependency so LLVM orders and covers it.
#
# This flag is imported by the kernel modules, which set the LLVM hint from it. It does
# not change codegen in this file -- it only drives the hint. False -> mode 0.
ENABLE_SCHED_MODE2 = True


# ===========================================================================
# Memory ops are emitted inline via the plain flydsl/rocdl intrinsics
# (``create_llvm_ptr`` + ``llvm_dialect.load``/``store`` /
# ``rocdl.ds_load_tr16_b128`` / ``buffer_ops.buffer_store``). There are no
# wrapper helpers: under mode 2 the ``amdgpu-expert-scheduling-mode`` LLVM hint
# makes LLVM insert all DEP_MODE=2 depctr covers itself for these SSA-visible ops,
# so the same code is correct in both modes. NOTE: LDS reads (``ds_load``) MUST be
# these plain intrinsics -- never opaque inline asm -- or LLVM cannot see the RAW
# against the opaque async global->LDS store and mis-orders it under DEP_MODE=2
# (silent NaN at long sequence lengths).
# ===========================================================================


# ============================================================================
# Q loader (global -> LDS async -> VGPR WMMA fragments)
# ============================================================================


# ============================================================================
# K loader (global -> LDS async -> VGPR WMMA B-fragments)
# ============================================================================


# ============================================================================
# V loader (global -> LDS async -> VGPR WMMA A-fragments via transpose load)
# ============================================================================


# ============================================================================
# V2 K/V loaders -- TDM (Tensor DMA) global->LDS + row-major PADDED LDS.
#
# Compared with async loads into swizzled LDS: the whole n_block x hdim tile
# is copied by ONE TDM atom whose descriptor carries base/extent/stride as state, so
# there is NO per-lane address VALU (saves address VGPRs) and the per-dim extent gives
# HARDWARE OOB (zero-fill) -- no software kv_valid `.select` clamps. The LDS is plain
# row-major with per-row padding (TDM pad_interval/pad_amount, in bf16 elements) so the
# WMMA ds_load fetch stays bank-conflict-free: K pads 8 elems (4 DW / 16 B), V pads 16
# elems (8 DW / 32 B). Element (row, col) lives at ``row*ROW_ELEMS + col`` (elements).
# The read collapses to ONE per-lane base + compile-time immediates (leaner than a swizzled layout's 2).
# ============================================================================

_K_PAD_ELEMS = 8  # 4 DW = 16 B per K row
_V_PAD_ELEMS = 16  # 8 DW = 32 B per V row
_Q_PAD_ELEMS = 8  # 4 DW = 16 B per Q row (matches K)
_O_PAD_ELEMS = 8  # 4 DW = 16 B per O row (conflict-free ds_store_b128)


def _pow2_segments(width):
    """Split ``width`` (elements) into power-of-two column segments (largest first). The TDM
    pad_interval must be a power of two, so a non-pow2 row (192) is copied as multiple segments,
    each with a pow2 pad_interval. 128 -> [(0,128)]; 192 -> [(0,128),(128,64)]; 256 -> [(0,256)].
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
    elem_dtype,
    num_warps=_DEFAULT_NUM_WAVES,
):
    """Build a LIST of ``(atom, g_view, lds_view)`` TDM global->LDS copies for one
    ``[n_rows, hdim]`` tile into a row-major padded (``hdim + pad_elems`` element row stride) LDS
    block -- PURE (no memory op), issue each with ``fx.copy_atom_call(*view)`` then drain with
    ``tensor_wait(0)``. hdim is split into power-of-two column segments (pad_interval must be pow2):
    segment ``(c0, w)`` copies global cols ``[c0, c0+w)`` -> LDS cols ``[c0, c0+w)`` with
    ``pad_interval=w``, ``pad_amount=(hdim+pad_elems - w)`` so the LDS row still advances by the
    padded stride. One segment for pow2 hdim (128/256), two for 192. Src base = ``ptr_x[row0, head]``;
    per-row extent ``valid`` = HW OOB zero-fill; all waves split the tile. Strides in ELEMENTS.
    """
    row_elems = hdim + pad_elems
    off = fx.Int64(row0) * fx.Int64(stride_seq) + fx.Int64(head) * fx.Int64(stride_head)
    base_iter = fx.get_iter(ptr_x)
    lds_ptr_ty = fx.PointerType.get(
        elem_ty=elem_dtype.ir_type,
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
    """

    def __init__(
        self,
        *,
        qk_hdim,
        gqa_ratio,
        num_waves=_DEFAULT_NUM_WAVES,
        lds_tiles=None,
        q_tiles_per_wave=1,
        elem_dtype=fx.BFloat16,
    ):
        self.elem_dtype = elem_dtype
        if qk_hdim % _WMMA_K != 0:
            raise ValueError(f"qk_hdim must be a multiple of {_WMMA_K}; got {qk_hdim}")
        if num_waves not in (2, 4, _DEFAULT_NUM_WAVES):  # 2/4 waves: small-grid kernel
            raise NotImplementedError(f"V2 TDM loader supports 2, 4 or 8 waves; got {num_waves}")
        self.qk_hdim = qk_hdim  # compile-time
        self.gqa_ratio = gqa_ratio  # compile-time
        self.num_waves = num_waves
        self.q_tiles_per_wave = q_tiles_per_wave
        self.k_tiles = qk_hdim // _WMMA_K
        self.rows_per_warp = _WMMA_M * q_tiles_per_wave  # this wave's Q rows (32)
        self.block_m = self.rows_per_warp * num_waves  # 256
        self.row_elems = qk_hdim + _Q_PAD_ELEMS  # padded LDS row stride (elems)
        self.row_bytes = self.row_elems * _BF16_BYTES
        # lds_tiles is accepted for signature compatibility; there is no ring.

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
        n_seq_valid = imax(rem, fx.Int32(0))

        off = fx.Int64(q_start + seq0) * fx.Int64(stride_q_seq) + fx.Int64(head0) * fx.Int64(stride_q_head)
        base_iter = fx.get_iter(ptr_Q)
        warp_region = ptr_lds + warp_idx * (self.rows_per_warp * self.row_bytes)
        lds_ptr_ty = fx.PointerType.get(
            elem_ty=self.elem_dtype.ir_type,
            address_space=fx.AddressSpace.Shared,
            alignment=16,
        )
        # hdim split into pow2 column segments (TDM pad_interval must be pow2): one copy for 128/256,
        # two for 192. Each segment (c0, w): pad_interval=w, pad_amount=row_elems-w so the padded LDS
        # row stride is preserved.
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
        v8_ty = fx.Vector.make_type(_CHUNK_ELEMS, self.elem_dtype)
        scale_bf16 = scale.to(self.elem_dtype)
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

    def __init__(
        self,
        *,
        qk_hdim,
        n_block=_DEFAULT_N_BLOCK,
        num_waves=_DEFAULT_NUM_WAVES,
        elem_dtype=fx.BFloat16,
    ):
        self.elem_dtype = elem_dtype
        if qk_hdim % _WMMA_K != 0:
            raise ValueError(f"qk_hdim must be a multiple of {_WMMA_K}; got {qk_hdim}")
        if n_block not in _N_BLOCK_CHOICES:
            raise ValueError(f"n_block must be one of {_N_BLOCK_CHOICES}; got {n_block}")
        if num_waves not in (2, 4, _DEFAULT_NUM_WAVES):  # 2/4 waves: small-grid kernel
            raise NotImplementedError(f"V2 TDM loader supports 2, 4 or 8 waves; got {num_waves}")
        self.qk_hdim = qk_hdim
        self.n_block = n_block
        self.num_waves = num_waves
        self.row_elems = qk_hdim + _K_PAD_ELEMS  # padded LDS row stride (elems)
        self.row_bytes = self.row_elems * _BF16_BYTES

    def get_lds_size_in_byte(self):
        return self.n_block * self.row_bytes

    def load_views(self, *, ptr_lds, ptr_K, stride_k_seq, stride_k_head, kv_head, kv_row0, kv_valid):
        """Return a LIST of ``(atom, g_view, lds_view)`` TDM copies for this block's K tile into the
        padded LDS at ``ptr_lds`` (fx.Int32 byte base) -- one per pow2 hdim segment (1 for 128/256, 2
        for 192). Pure (hoistable); issue each with ``fx.copy_atom_call(*view)``, drain with
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
            elem_dtype=self.elem_dtype,
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

    def load_k_to_reg(self, base_ptrs, lds_imm_offset=0):
        """Burst all K ``ds_load_b128`` from the row-major padded block, in ``_qk_gemm``
        order ``[(kv, dt, half)...]``, off the single base of ``ds_load_ptrs``. Fragment
        (kv, dt, half) = base + ``kv*16*row_bytes + (dt*32 + half*16)*2`` bytes."""
        v8_ty = fx.Vector.make_type(8, self.elem_dtype)
        NKV = self.n_block // _WMMA_M
        NDT = self.qk_hdim // _WMMA_K
        base = base_ptrs[0]
        out = []
        for kv in range(NKV):
            for dt in range(NDT):
                for half in range(2):
                    imm = (
                        kv * _WMMA_M * self.row_bytes
                        + (dt * _WMMA_K + half * _WMMA_M) * _BF16_BYTES
                        + lds_imm_offset
                    )
                    p = base
                    if imm:
                        p = buffer_ops.get_element_ptr(base, static_byte_offset=imm)
                    out.append(fx.Vector(llvm_dialect.load(v8_ty, p)))
        return out


class VManager16bV2:
    """V loader (TDM + row-major padded LDS, 32 B/row pad). A-fragments in ``_pv_gemm``
    order ``[(dt, kt, half)...]``, transpose
    ``ds_load_tr16_b128`` crossbar ``V[kv+(l//16)*8+l%8, d+((l//8)%2)*8]``)."""

    def __init__(
        self,
        *,
        v_hdim,
        n_block=_DEFAULT_N_BLOCK,
        num_waves=_DEFAULT_NUM_WAVES,
        elem_dtype=fx.BFloat16,
    ):
        self.elem_dtype = elem_dtype
        if v_hdim % _WMMA_M != 0:
            raise ValueError(f"v_hdim must be a multiple of {_WMMA_M}; got {v_hdim}")
        if n_block % _WMMA_K != 0:
            raise ValueError(f"n_block must be a multiple of {_WMMA_K}; got {n_block}")
        if num_waves not in (2, 4, _DEFAULT_NUM_WAVES):  # 2/4 waves: small-grid kernel
            raise NotImplementedError(f"V2 TDM loader supports 2, 4 or 8 waves; got {num_waves}")
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
            elem_dtype=self.elem_dtype,
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

    def load_v_to_reg(self, base_ptrs, lds_imm_offset=0):
        """Burst all V ``ds_load_tr16_b128`` from the row-major padded block, in
        ``_pv_gemm`` order ``[(dt, kt, half)...]``, off the single base of ``ds_load_ptrs``.
        Fragment (dt, kt, half) = base + ``(kt*32 + half*16)*row_bytes + dt*16*2`` bytes.
        """
        v8_ty = fx.Vector.make_type(8, self.elem_dtype)
        d_tiles = self.v_hdim // _WMMA_M
        nkt = self.n_block // _WMMA_K
        base = base_ptrs[0]
        out = []
        for dt in range(d_tiles):
            for kt in range(nkt):
                for half in range(2):
                    imm = (
                        (kt * _WMMA_K + half * _WMMA_M) * self.row_bytes
                        + dt * _WMMA_M * _BF16_BYTES
                        + lds_imm_offset
                    )
                    p = base
                    if imm:
                        p = buffer_ops.get_element_ptr(base, static_byte_offset=imm)
                    out.append(fx.Vector(rocdl.ds_load_tr16_b128(v8_ty, p)))
        return out


# ============================================================================
# O writer (VGPR WMMA accumulator -> LDS reshape -> coalesced global store)
# ============================================================================


class OManager16bV1:
    """Owns the O epilogue: fp32 WMMA accumulator -> bf16 -> global VRAM.

    PV leaves each wave's 16 x v_hdim tile in the accumulator layout: for d-tile
    ``k`` lane ``l`` holds ``O[q = l%16, d = (l//16)*8 + 16*k + {0..7}]``. Adjacent
    lanes hold different q rows, so O is transposed through per-warp staging LDS --
    store accumulator-indexed, re-read giving each lane 8 contiguous d of one q,
    then ``buffer_store`` coalesced.

    The 16 x v_hdim warp tile splits into ``n_units = v_hdim / cols_per_tile``
    self-contained transpose units. Staging LDS is an ``inflight_units``-deep ring
    of ``cols_per_tile``-wide slots, each 16(q) x cols_per_tile bf16, row-major with
    the 8-bf16 chunk XOR-swizzled (``slot = chunk ^ (q >> shift)``) -> no padding,
    b128 store and read bank-conflict-free. The units software-pipeline: unit u+1's
    ds_stores are issued ahead of unit u's ds_load re-read + coalesced buffer_store,
    hiding the LDS round-trip. ``cols_per_tile=64`` keeps each global store a full
    128B row. The caller points ``ptr_lds`` at the non-current K|V slot (holding a
    dead prefetch no wave reads), so no threadgroup barrier is needed.

    Ordering uses ``s_wait_dscnt``. DSCNT is a single in-order 6-bit counter shared
    by ds_store and ds_load, so every DS op's global issue index is tracked at
    compile time and a dependency is awaited via ``clamp(issued - 1 - gidx, 0, 63)``.

    Constructor config: ``v_hdim``, ``gqa_ratio``, ``num_waves``, ``cols_per_tile``,
    ``inflight_units``, ``lds_budget_bytes``.
    """

    def __init__(
        self,
        *,
        v_hdim,
        gqa_ratio,
        num_waves=_DEFAULT_NUM_WAVES,
        q_tiles_per_wave=1,
        cols_per_tile=_O_COLS_PER_TILE,
        inflight_units=_O_INFLIGHT_UNITS,
        lds_budget_bytes=_O_LDS_BUDGET_BYTES,
        elem_dtype=fx.BFloat16,
    ):
        self.elem_dtype = elem_dtype
        if v_hdim % _WMMA_M != 0:
            raise ValueError(f"v_hdim must be a multiple of {_WMMA_M}; got {v_hdim}")
        self.v_hdim = v_hdim
        self.gqa_ratio = gqa_ratio
        self.num_waves = num_waves
        # Each wave owns q_tiles_per_wave adjacent 16-row O tiles (contiguous); the
        # R tiles serialize through the SAME per-warp ring (caller loops qtile).
        self.q_tiles_per_wave = q_tiles_per_wave
        self.block_m = _WMMA_M * q_tiles_per_wave * num_waves  # Q/O rows per threadgroup
        self.d_tiles = v_hdim // _WMMA_M  # WMMA output tiles == frags/lane

        cpt = min(v_hdim, cols_per_tile)
        cpt = (cpt // _WMMA_M) * _WMMA_M  # 16-col aligned
        if cpt < _WMMA_M:
            raise ValueError(f"cols_per_tile={cols_per_tile} too small")
        if v_hdim % cpt != 0:
            raise NotImplementedError(f"v_hdim={v_hdim} not a multiple of cols_per_tile={cpt}")
        self.cols_per_tile = cpt
        self.n_units = v_hdim // cpt
        self.tiles_per_unit = cpt // _WMMA_M  # ds_store ops per unit == ds_load ops
        self.inflight_units = min(inflight_units, self.n_units)

        # LDS geometry, per unit slot; the ring holds inflight_units slots.
        self._row_bytes = cpt * _BF16_BYTES
        self._slot_stride = _WMMA_M * self._row_bytes
        self._warp_stride = self.inflight_units * self._slot_stride
        self._chunks_per_row = cpt // _CHUNK_ELEMS  # G, b128 chunks per slot row
        # XOR swizzle: slot = chunk ^ (q // _sw_div), shift = max(0, 4 - log2(G)).
        shift = max(0, 4 - int(self._chunks_per_row).bit_length() + 1)
        self._sw_div = 1 << shift

        total = num_waves * self._warp_stride
        if total > lds_budget_bytes:
            raise ValueError(
                f"O ring {total}B (waves={num_waves} x inflight={self.inflight_units} x "
                f"slot={self._slot_stride}B) exceeds budget {lds_budget_bytes}B"
            )

    def get_lds_size_in_byte(self):
        """LDS bytes the caller must reserve for O staging (all waves, whole ring)."""
        return self.num_waves * self._warp_stride

    def _lds_byte(self, slot_idx, q_row, d_col):
        """Swizzled LDS byte offset (within a warp region) of ring-slot ``slot_idx``'s
        O(q_row, d_col) (``d_col`` is slot-local, in [0, cols_per_tile))."""
        chunk = d_col // _CHUNK_ELEMS
        sw = chunk ^ (q_row // self._sw_div)
        return slot_idx * self._slot_stride + q_row * self._row_bytes + sw * _CHUNK_BYTES

    def store_o_to_vram(
        self,
        *,
        ptr_O,  # fx.Pointer to the O tensor (buffer resource built internally)
        o_base_elems,  # fx.Int32: element offset of this (batch, ...) origin (0 for thd)
        stride_o_seq,  # elements per token step
        stride_o_head,  # elements per q-head step
        q_start,  # fx.Int32: first token of this request (varlen) / 0 (batch)
        q_len,  # fx.Int32: valid query rows; rows with seq >= q_len are masked off
        kv_head,  # fx.Int32: this workgroup's kv head
        block_x,  # fx.Int32: this workgroup's grid-x tile index
        warp_idx,
        lane_idx,
        ptr_lds,  # fx.Int32: base byte addr of the caller's O staging allocation
        o_frags,  # list[d_tiles] of v8 f32 (pre-normalized) WMMA accumulators
        qtile=0,  # which of this wave's q_tiles_per_wave tiles this call stores
    ):
        """Reshape this warp's 16 x v_hdim fp32 accumulator to bf16 and store it.

        ``o_frags[k]`` is this lane's v8 fp32 for d-tile ``k``, already normalized
        (O / row-sum). Rows with seq >= q_len are dropped via the buffer_store mask,
        which redirects to offset ``0x7FFFFFFF`` -- so the internally-built buffer
        resource is bounded to the real ``num_records`` (below 0x7FFFFFFF) so the drop
        lands OOB (dropped) rather than faulting. ``stride_o_*`` in ELEMENTS.
        """
        if len(o_frags) != self.d_tiles:
            raise ValueError(f"expected {self.d_tiles} O frags (v_hdim//{_WMMA_M}); got {len(o_frags)}")
        # Bound records to the last valid token so mask-dropped rows (redirected to byte
        # 0x7FFFFFFF) land OOB instead of faulting; every valid write is far below that.
        # i64: an i32 product can overflow negative -> descriptor sign-extends to a huge
        # bound, defeating the OOB drop.
        o_num_records_bytes = fx.Int64(q_start + q_len) * fx.Int64(stride_o_seq) * fx.Int64(_BF16_BYTES)
        o_rsrc = buffer_ops.create_buffer_resource(ptr_O, num_records_bytes=o_num_records_bytes.ir_value())
        lds_warp = ptr_lds + warp_idx * self._warp_stride
        q_st = lane_idx % _WMMA_M
        d_half = (lane_idx // _WMMA_M) * _CHUNK_ELEMS  # 0 or 8
        v8_ty = fx.Vector.make_type(_CHUNK_ELEMS, self.elem_dtype)
        base_row = block_x * self.block_m + warp_idx * (self.q_tiles_per_wave * _WMMA_M) + qtile * _WMMA_M
        G = self._chunks_per_row
        TPU = self.tiles_per_unit  # ds_store ops per unit == ds_load rounds per unit
        NSL = self.inflight_units  # ring depth (units resident at once)

        # DSCNT is one in-order 6-bit counter for both ds_store and ds_load: op X has
        # retired <=> DSCNT <= issued-1-X. Track each DS op's global index and await a
        # dependency ``dep`` via s_wait_dscnt(clamp(issued-1-dep, 0, 63)).
        issued = 0  # DS ops emitted so far (global issue index)
        unit_last_store = {}  # unit -> gidx of its final ds_store (all TPU stores done)
        unit_loads = {}  # unit -> list of gidx of its TPU ds_loads

        def emit_write(u):
            """Issue unit ``u``'s TPU ds_stores (accumulator -> ring slot u%NSL)."""
            nonlocal issued
            slot = u % NSL
            prev = u - NSL  # last occupant of this slot
            if prev >= 0:
                # WAR: prev unit's re-reads must retire before we overwrite the slot.
                dep = unit_loads[prev][-1]
                rocdl.s_wait_dscnt(max(0, min(_O_DSCNT_MAX, issued - 1 - dep)))
            last = None
            for kk in range(TPU):
                k = u * TPU + kk
                d_col = d_half + kk * _WMMA_M  # slot-local column
                bf = o_frags[k].to(self.elem_dtype)
                addr = lds_warp + self._lds_byte(slot, q_st, d_col)
                lds_ptr = create_llvm_ptr(addr, address_space=3)
                llvm_dialect.store(bf.ir_value(), lds_ptr, alignment=_CHUNK_BYTES)
                last = issued
                issued += 1
            unit_last_store[u] = last

        def emit_read(u):
            """Re-read unit ``u`` coalesced (ring slot u%NSL) and buffer_store to VRAM."""
            nonlocal issued
            slot = u % NSL
            # RAW: every re-read pulls d-columns spanning all TPU stores of the unit,
            # so wait for the unit's final store before the first load.
            dep = unit_last_store[u]
            rocdl.s_wait_dscnt(max(0, min(_O_DSCNT_MAX, issued - 1 - dep)))
            loaded = []
            gidxs = []
            for r in range(TPU):
                f = fx.Int32(r * _WAVE_LANES) + lane_idx  # 32 b128 chunks per round
                q_out = f // G  # q-row within this warp [0,16)
                d_local = (f % G) * _CHUNK_ELEMS
                addr = lds_warp + self._lds_byte(slot, q_out, d_local)
                lds_ptr = create_llvm_ptr(addr, address_space=3)
                data = fx.Vector(llvm_dialect.load(v8_ty, lds_ptr))
                loaded.append((data, q_out, d_local))
                gidxs.append(issued)
                issued += 1
            unit_loads[u] = gidxs
            d_base = fx.Int32(u * self.cols_per_tile)  # global d origin of this unit
            for r in range(TPU):
                data, q_out, d_local = loaded[r]
                # RAW: this load's data must be in VGPR before storing it to VRAM.
                rocdl.s_wait_dscnt(max(0, min(_O_DSCNT_MAX, issued - 1 - gidxs[r])))
                pr = base_row + q_out  # global packed row
                q_head = kv_head * self.gqa_ratio + pr % self.gqa_ratio
                seq = pr // self.gqa_ratio
                valid = seq < q_len
                token = q_start + seq
                off_elems = o_base_elems + token * stride_o_seq + q_head * stride_o_head + d_base + d_local
                off_bytes = off_elems * _BF16_BYTES
                off_masked = valid.select(off_bytes, fx.Int32(0x7FFFFFFF))
                buffer_ops.buffer_store(data, o_rsrc, off_masked, mask=None, offset_is_bytes=True)

        # Two-stage pipeline: prime unit 0, then overlap unit u+1's write with unit
        # u's read. n_units == 1 collapses to a single write+read with one RAW wait.
        emit_write(0)
        for u in range(self.n_units):
            if u + 1 < self.n_units:
                emit_write(u + 1)
            emit_read(u)


class OManager16bV3:
    """O writer: conflict-free PADDED LDS ds_store + per-b128 ``global_store_async_from_lds_b128``
    (LDS->VRAM), NO TDM. Keeps V2's padded LDS (bank-conflict-free ``ds_store_b128``, 4DW pad) but
    avoids the TDM store's "padding ignored" limitation by addressing each b128 ourselves; and skips
    V1's VGPR re-read (the async store goes LDS->global directly).

    Accumulator (d-tile k, lane l = ``O[q=l%16, d=16k+(l//16)*8+{0..7}]``) -> row-major PADDED LDS
    ``[16, v_hdim]`` (row stride v_hdim+_O_PAD_ELEMS). Then the 16 x (v_hdim/8) b128 chunks are stored
    LDS->global over ``n_rounds`` waves of 32 lanes: round r lane l -> chunk c=r*32+l, row=c//cpr,
    d_chunk=c%cpr (cpr = v_hdim/8) -> coalesced (consecutive lanes = consecutive global).
    Rows at seq>=q_len are masked off because async stores have no bounds check.
    """

    def __init__(
        self,
        *,
        v_hdim,
        gqa_ratio,
        num_waves=_DEFAULT_NUM_WAVES,
        q_tiles_per_wave=1,
        elem_dtype=fx.BFloat16,
    ):
        self.elem_dtype = elem_dtype
        if v_hdim % _WMMA_M != 0:
            raise ValueError(f"v_hdim must be a multiple of {_WMMA_M}; got {v_hdim}")
        if num_waves not in (2, 4, _DEFAULT_NUM_WAVES):  # 2/4 waves: small-grid kernel
            raise NotImplementedError("V3 assumes 8 waves")
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
        earlier still-in-flight store's source reg -- a ~161-cyc WAR halt when split per-tile), (c) one
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
            bf = o_frags[k].to(self.elem_dtype)  # cvt
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
        n_rounds = (rpw * cpr) // _WAVE_LANES
        ptr_O_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(ptr_O)))
        # packed rows [warp_base, warp_base+rpw) valid iff pr//gqa < q_len iff pr < q_len*gqa.
        valid_rows = q_len * gqa - warp_base
        last_valid = imin(imax(valid_rows - 1, fx.Int32(0)), fx.Int32(rpw - 1))
        addrs = []
        for r in range(n_rounds):
            c = fx.Int32(r * _WAVE_LANES) + lane_idx
            row = c // cpr
            d_chunk = c % cpr
            srow = imin(row, last_valid)  # OOB rows -> warp's last valid row (idempotent redirect)
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
