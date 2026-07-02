###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""4-wave MXFP4 dense GEMM (per-32-K E8M0 block scaling) for AMD CDNA4 (gfx950).

NT only: A [M, K] fp4 (packed 2/byte), B [N, K] fp4, C = a @ b^T (bf16).

Ported from the FlyDSL standalone 4-wave production kernel. The 4-wave (2x2 wave)
topology gives 1 wave/SIMD so the full 256-AGPR file holds one wave's N-sliced
accumulator (acc_left + acc_right) cleanly, freeing arch-VGPR for operand prefetch.

This file carries the production code path. The standalone source's tuning/debug
knobs are hardcoded to their production value; per-shape launch swizzle,
workload-depth and epilogue variant are chosen by the timed autotune:

  * whole-loop bare-asm compute (the entire K-loop is one inline-asm hw-loop),
  * 2 LDS ping-pong buffers (unroll-2), NEXT-K in-place operand refill,
  * blocked-diagonal MFMA emission (4 A-rows x 8 N-cols) with the K-sub innermost,
  * GAVOID g2s placement (loads only in refill-free MFMA slots),
  * VGPR-direct scales (buffer_load lane-contiguous straight to VGPR; no LDS/ds_read
    for scales), interleaved into the MFMA stream,
  * pinned operand/scale VGPRs (PINBASE=8) to dodge the LLVM RA cascade crash,
  * per-phase s_barrier with vmcnt(10) lgkmcnt(9) drain,
  * agpr-alloc=256, waves_per_eu=1, BLOCK_M=BLOCK_N=BLOCK_K=256, packed E8M0 scales.

Scale tensors are passed pre-shuffled into the lane-contiguous layout produced
directly by the fused MXFP4 quant kernel (VGPR-direct scale read; no host repack).
"""

import torch

# isort: off
from primus_turbo.flydsl.utils.gemm_helper import (
    G2SLoader,
    ceildiv,
    make_fp8_buffer_tensor,
    wait_barrier,
)
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

# isort: on


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


# ── Device-side scale + fragment loaders / geometry ──────────────────────────


class ScaleS2RPacked:
    """Packed-scale buffer resource (one dword per (region, k) holding n_tiles E8M0,
    byte t = tile t). The production kernel uses only ``.rsrc`` (sized to cover the
    lane-contiguous scale tensor); the VGPR-direct loads address it from the asm."""

    def __init__(self, sp_tensor, dim, K, n_tiles):
        group_span = 16 * n_tiles
        nbytes = (dim // group_span) * (K // 128) * 64 * 4  # int32 records, 1/lane
        self.rsrc = buffer_ops.create_buffer_resource(sp_tensor, max_size=False, num_records_bytes=nbytes)


def _swz_fwd(c):
    """LDS bank-swizzle, logical chunk -> physical slot. Modular-add rotate within
    each 1024B block: phase = c//8, rotate the low 3 bits (128B bank period) by phase
    so the 8 same-bank rows of one ds_read_b128 land on all 8 bank-groups. Bijection."""
    ph = c // 8
    return ph * 8 + (c % 8 + ph) % 8


def _swz_inv(c):
    """Inverse of _swz_fwd (physical slot -> logical chunk). G2S lane L writes the
    contiguous physical slot L, so it must fetch the gmem of logical chunk _swz_inv(L)
    for S2R's _swz_fwd read to land on it."""
    ph = c // 8
    return ph * 8 + (c % 8 + 8 - ph) % 8


def fp4_g2s_offsets(lane_id, wave_id, K, n_steps, bytes_per_row, swizzle=False):
    """Per-lane gmem byte offsets for fp4 G2S. Lane L (wave w) copies 16 bytes from
    gmem [row*(K/2)+chunk*16] into its contiguous LDS slot (w*1024 + L*16), which
    algebraically equals row*bpr + chunk*16 -> S2R reads it back at identity
    row*bpr + g*16. ``swizzle`` applies the inverse bank-swizzle so the write lands
    where S2RLoaderFp4's _swz_fwd read expects."""
    n_waves = fx.block_dim.x // 64
    lpr = bytes_per_row // 16  # lanes per row
    rows_per_step = 64 // lpr
    offs = []
    for r in range_constexpr(n_steps):
        cib = _swz_inv(lane_id) if swizzle else lane_id  # physical slot -> logical chunk
        row = cib // lpr + wave_id * rows_per_step + r * (n_waves * rows_per_step)
        chunk = cib % lpr
        offs.append(row * (K // 2) + chunk * 16)
    return offs


def grouped_xcd_pid(pid, c_m, c_n, BLOCK_M, BLOCK_N, group_m=4, num_xcds=8, group_n=0):
    """Map block_idx -> (block_m, block_n) with XCD-aware remap + GROUP_M tiling for
    L2 locality. group_n>0 enables a 2D super-block (N-band) swizzle on top (locks an
    A-slab AND a B-slab into L2). Pure index math; exact bijection when
    total_tiles % num_xcds == 0, falls back safely otherwise."""
    num_pid_m = ceildiv(c_m, BLOCK_M)
    num_pid_n = ceildiv(c_n, BLOCK_N)
    total = num_pid_m * num_pid_n
    pids_per_xcd = (total + num_xcds - 1) // num_xcds
    pid_r = (pid % num_xcds) * pids_per_xcd + pid // num_xcds
    pid_r = arith.select(pid_r < total, pid_r, pid)

    if group_n and group_n > 0:
        band_tiles = num_pid_m * group_n  # tiles in one full band
        n_full_bands = num_pid_n // group_n
        full_region = n_full_bands * band_tiles  # pids covered by full bands
        in_full = pid_r < full_region
        band_id = pid_r // band_tiles
        local_f = pid_r % band_tiles
        nbase_f = band_id * group_n
        bw_f = fx.Int32(group_n)
        rem = num_pid_n - n_full_bands * group_n
        local_r = pid_r - full_region
        nbase_r = n_full_bands * group_n
        bw_r = arith.select(rem < fx.Int32(1), fx.Int32(1), rem)  # avoid /0 in dead branch
        local = arith.select(in_full, local_f, local_r)
        nbase = arith.select(in_full, nbase_f, nbase_r)
        bw = arith.select(in_full, bw_f, bw_r)
        num_in_group = group_m * bw
        group_id = local // num_in_group
        first_m = group_id * group_m
        gsz = num_pid_m - first_m
        gsz = arith.select(gsz < fx.Int32(group_m), gsz, fx.Int32(group_m))
        inner = local % num_in_group
        block_m = first_m + inner % gsz
        block_n = nbase + inner // gsz
        return block_m, block_n

    num_in_group = group_m * num_pid_n
    group_id = pid_r // num_in_group
    first_m = group_id * group_m
    gsz = num_pid_m - first_m
    gsz = arith.select(gsz < fx.Int32(group_m), gsz, fx.Int32(group_m))
    inner = pid_r % num_in_group
    block_m = first_m + inner % gsz
    block_n = inner // gsz
    return block_m, block_n


class S2RLoaderFp4:
    """LDS->reg fp4 fragment loader (identity LDS, bytes_per_row K-iter rows).

    A K-iter spans n_sub == BLOCK_K/128 128-K sub-blocks; each sub-block is one
    16x16x128 MFMA. The production whole-loop reads via ``base_addr`` (one address
    reg per region + a ds_read offset immediate)."""

    def __init__(self, wave_idx, n_tiles, row_stride, swizzle=False):
        self.lane16 = fx.thread_idx.x % 16
        self.g = (fx.thread_idx.x % 64) // 16
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles
        self.row_stride = row_stride
        self.swizzle = swizzle

    def base_addr(self, lds_src, s=0):
        """Single base LDS address (tile 0, sub-block s). Per-tile fragments are at
        base + i*tile_stride (tile_stride = 16*row_stride bytes) -> the asm uses ONE
        address reg per region + a ds_read offset immediate."""
        row0 = self.wave_idx * (self.n_tiles * 16) + self.lane16
        off_nat = row0 * self.row_stride + s * 64 + self.g * 16
        if const_expr(self.swizzle):  # tile i = base + i*tile_stride stays swz-correct
            cib = (off_nat % 1024) // 16  # (tile_stride is a 1024-multiple -> %1024 const)
            off = (off_nat // 1024) * 1024 + _swz_fwd(cib) * 16
        else:
            off = off_nat
        i8_iter = fx.recast_iter(fx.Uint8, fx.add_offset(lds_src.ptr, fx.make_int_tuple(off)))
        return fx.ptrtoint(i8_iter)

    @property
    def tile_stride(self):
        return 16 * self.row_stride


class StoreCPlain:
    """Plain FP32 accumulator -> BF16 store (no scaling; scales folded in MMA).

    Uses ``fx.copy`` over a single divided output view + OOB-index redirect for the
    column-edge mask (``arith.select(col_valid, c_index, oob)``), which the backend
    lowers WITHOUT per-store EXEC-mask save/restore -> a tight store epilogue (the
    masked ``buffer_store`` path emits ~1.8k extra scalar ops, a fixed per-WG cost).
    M, N are multiples of 256 here so every tile is in-bounds; the redirect is just a
    cheap safety net. C is a flat 1D out_ty buffer (num_records bounds the SRD).

    ``out_ty`` is bf16 or fp16 (both 2 bytes). The narrow ``store`` path uses a generic
    f32->out_ty ``.to()`` cast so it serves either; the wide ``store_tacc_wide`` fast
    path is bf16-only (``cvt_pk_bf16_f32``), so fp16 output forces the narrow path."""

    def __init__(self, C, c_rows, c_cols, c_idx_fn, n_tiles_a, n_tiles_b, out_ty=None):
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.out_ty = out_ty if out_ty is not None else fx.BFloat16
        c_nbytes = c_rows * c_cols * 2  # bf16/fp16 both 2 bytes/elem
        self._C = C  # raw C handle for the wide (dwordx4) TACCW store's buffer resource
        gC = rocdl.make_buffer_tensor(C, max_size=False, num_records_bytes=c_nbytes)
        self.c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        self.out_atom_1 = fx.make_copy_atom(rocdl.BufferCopy16b(), self.out_ty)
        self.reg_out_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), self.out_ty)

    def _store_scalar(self, value, c_index):
        fx.memref_store_vec(Vec.filled(1, value, self.out_ty), self.reg_out_1)
        fx.copy(self.out_atom_1, self.reg_out_1, fx.slice(self.c_div, (None, fx.Int32(c_index))))

    def store(self, c_frag, base_row, base_col):
        # MN-ALIGNED fast path: the host wrapper guarantees M % 256 == 0 and
        # N % 256 == 0, so every BLOCK_M x BLOCK_N output tile is fully in-bounds.
        # We therefore drop the per-store column-edge redirect entirely (no
        # `col < c_cols` v_cmp + `arith.select` v_cndmask), trimming the store
        # epilogue's fixed per-WG cost -- a measurable win on small-K / tall-M
        # shapes where the epilogue is a larger fraction of runtime (the OOB SRD
        # num_records still bounds any pathological index as a safety net).
        for ti in range_constexpr(self.n_tiles_a):
            row = base_row + ti * 16 + (self.lane_id // 16) * 4
            for tj in range_constexpr(self.n_tiles_b):
                col = base_col + tj * 16 + self.lane_id % 16
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                for i in range_constexpr(4):
                    val = vec_f32[i].to(self.out_ty)
                    self._store_scalar(val, (row + i) * self.c_cols + col)

    @staticmethod
    def _permlane16_swap(a_i32, b_i32):
        """v_permlane16_swap_b32 a, b -- swap 16-lane row-groups between two regs
        (both read+written, in place). Returns (a', b'). Row-group map:
            a'[rg0]=a[rg0] a'[rg1]=b[rg0] a'[rg2]=a[rg2] a'[rg3]=b[rg2]
            b'[rg0]=a[rg1] b'[rg1]=b[rg1] b'[rg2]=a[rg3] b'[rg3]=b[rg3]"""
        st = "!llvm.struct<(i32, i32)>"
        # The result is consumed by a buffer_store (VMEM), not a VALU op, so the
        # permlane16_swap->read VALU hazard s_nop is unnecessary here (saves ~64 exposed
        # nops on the store-bound epilogue).
        r = _llvm.inline_asm(
            ir.Type.parse(st),
            [_raw(a_i32), _raw(b_i32)],
            "v_permlane16_swap_b32 $0, $1",
            "=v,=v,0,1",
            has_side_effects=False,
        )
        i32t = ir.IntegerType.get_signless(32)
        return _llvm.extractvalue(i32t, r, [0]), _llvm.extractvalue(i32t, r, [1])

    def store_tacc_wide(self, c_frag, base_row, base_col):
        """TACC + AITER permlane16_swap WIDE store (autotune-selected TACCW variant). Combines TWO
        adjacent N sub-blocks (tj, tj+1) into a 16-row x 32-col region written with
        ONE ``buffer_store_dwordx4`` (8 bf16 = 16B) per lane.

        With acc = Cᵀ a lane's 4 f32 are 4 CONSECUTIVE columns. Pack them to 2 bf16
        dwords (cvt_pk), then 2x ``v_permlane16_swap`` reshuffle the 4 row-groups so
        each lane ends up holding 8 CONTIGUOUS columns (AITER's exact recipe):
            rg0 -> tj   cols 0..7    rg1 -> tj+1 cols 0..7
            rg2 -> tj   cols 8..15   rg3 -> tj+1 cols 8..15
        Addressing: row = base_row + ti*16 + lane%16 (lane->row, scattered by row);
        col = base_col + tj*16 + coloff(rg), coloff=[0,16,8,24] -> the 4 row-groups
        of a row write 4 contiguous 16B blocks = 64B-coalesced burst. 16 wide stores
        per half (vs 256 narrow shorts). No LDS, no barrier. Host guarantees
        M%256==N%256==0 so all tiles are in-bounds (SRD num_records is the safety net)."""
        nta, ntb = self.n_tiles_a, self.n_tiles_b
        assert ntb % 2 == 0, "store_tacc_wide pairs N sub-blocks (ntb must be even)"
        c_nbytes = self.c_rows * self.c_cols * 2
        rsrc = buffer_ops.create_buffer_resource(self._C, max_size=False, num_records_bytes=c_nbytes)
        rg = self.lane_id // 16
        col_off = (rg % 2) * 16 + (rg // 2) * 8  # rg0->0 rg1->16 rg2->8 rg3->24
        # Hoisted per-lane base byte offset: (base_row*c_cols + base_col + lane%16*c_cols + col_off)*2.
        # Per-store delta (ti*16*c_cols + tj*16)*2 is a compile-time constant -> one v_add per store
        # instead of recomputing r*c_cols (matches AITER's voffset+imm addressing).
        lane_base = (
            base_row * self.c_cols + base_col + (self.lane_id % 16) * self.c_cols + col_off
        ) * fx.Int32(2)

        def _off(ti, tj):
            return lane_base + fx.Int32((ti * 16 * self.c_cols + tj * 16) * 2)

        def _xpose(ti, tj):
            A = Vec(c_frag[self.c_idx_fn(ti, tj)])
            B = Vec(c_frag[self.c_idx_fn(ti, tj + 1)])
            d_a0 = rocdl.cvt_pk_bf16_f32(A[0], A[1])
            d_a1 = rocdl.cvt_pk_bf16_f32(A[2], A[3])
            d_b0 = rocdl.cvt_pk_bf16_f32(B[0], B[1])
            d_b1 = rocdl.cvt_pk_bf16_f32(B[2], B[3])
            v16, v18 = self._permlane16_swap(d_a0, d_b0)
            v17, v19 = self._permlane16_swap(d_a1, d_b1)
            return Vec.from_elements([v16, v17, v18, v19], fx.Int32).bitcast(fx.BFloat16)

        # Software-pipeline cvt+permlane (phase1) away from the stores (phase2) so the
        # permlane16_swap->store RAW hazard (~80 exposed s_nop) is filled by independent
        # permlane work of later tiles instead of stalls.
        slots = [(ti, 2 * p) for ti in range_constexpr(nta) for p in range_constexpr(ntb // 2)]
        vecs = [_xpose(ti, tj) for (ti, tj) in slots]
        for vec_bf, (ti, tj) in zip(vecs, slots):
            buffer_ops.buffer_store(vec_bf, rsrc, _off(ti, tj), offset_is_bytes=True)


# ── Scaled MFMA whole-loop emitter ───────────────────────────────────────────


class MfmaScaleFp4:
    """16x16x128 f8f6f4 MFMA in fp4 mode (cbsz=4/blgp=4) with packed per-block E8M0
    scales (one packed-i32 scale operand per region, opsel selects the per-XDL byte).

    Only the production whole-loop path is provided: the entire K-loop is one
    inline-asm hardware loop (no per-iter FlyDSL boundary), 2 LDS buffers ping-pong
    (unroll-2), with NEXT-K in-place operand refill and VGPR-direct scales."""

    def __init__(self, n_tiles_a, n_tiles_b, packed=False, wlv=10, elgk=9, coop=False, tacc=False):
        self.res_ty = Vec.make_type(4, fx.Float32)
        self.zero_value = Vec.filled(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.packed = packed
        # tacc: swap MMA operands (src0<->src1, scales, op_sel) so the native
        # accumulator holds Cᵀ -> per sub-block a lane's 4 f32 are 4 CONSECUTIVE
        # columns, enabling a wide buffer_store_dwordx2/4 epilogue (AITER's method).
        self.tacc = tacc
        # phase-barrier in-flight memory depth (autotuned per shape): deeper (e.g.
        # 16/15) hides more steady-state g2s latency -> faster high-K; shallower (10/9)
        # is better on low/mid-trip-K where a deep pipeline can't fill. ELGK<=15 (4-bit
        # lgkmcnt HW field).
        self.wlv = wlv
        self.elgk = elgk
        self.coop = coop

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def call_mxfp4_wholeloop(
        self,
        a_base,
        bl_base,
        br_base,
        ts_a,
        ts_b,
        abase,
        blbase,
        brbase,
        gl_a,
        gl_b,
        rsrc_a,
        rsrc_b,
        kstep,
        scv,
        cL,
        cR,
        n_sub,
        nsa,
        nsb,
        nval,
        soff0,
        soff0_bl,
        soff0_br,
        sc_rb,
        sc_gb,
        sc_rsa,
        sc_rsb,
        sc_voff,
        sc_soff0,
        sca_rb=None,
        sca_gb=None,
        sca_voff=None,
        ki=None,
        sc_buf_stride=0,
        _cache={},  # noqa: B006 -- deliberate cross-call asm compile cache
    ):
        """WHOLE-LOOP bare-asm: the ENTIRE K-loop is ONE inline-asm hw-loop. 2 LDS
        buffers (buf0/buf1) ping-pong, unroll-2. Each phase: ds_read operands (single
        reg buffer) + 128 MFMA (L+R, n_sub) packed scale + G2S buffer_load_lds refill
        (NEXT-K in-place after each operand's last use) + s_barrier. Scales are
        loaded VGPR-direct (buffer_load lane-contiguous straight to pinned VGPR).
        Returns (accL, accR)."""
        assert self.packed
        nta, ntb = self.n_tiles_a, self.n_tiles_b
        nq = nta * ntb
        NT = 2 * nq
        na, nb = nta * n_sub, ntb * n_sub
        ntmp = na + 2 * nb
        _NWc = 4  # n_waves (4-wave kernel)
        nbuf = len(a_base)  # A pool size (= 2, unroll-2 ping-pong)
        nbuf_b = len(bl_base)  # B pool size (= 2)
        _nscbuf = nbuf_b  # scale LDS pool (unused under VGPR-direct, but operand slots are reserved)
        NSET = 1
        _WLV = self.wlv  # vmcnt kept in flight at the phase barrier (deep g2s pipeline)
        _ELGK = self.elgk  # lgkmcnt left at the phase barrier (late refills stay in flight)
        # Cooperative LDS scale staging (vs default per-wave VGPR-direct). The default
        # path has each wave buffer_load its OWN A/B scale groups straight to VGPR, but
        # waves sharing wave_m re-read the SAME A scales (and wave_n -> same B): a 2x
        # redundant HBM scale fetch that exposes scale traffic on low/mid-K, fat-N.
        # COOP makes the 4 waves cooperatively load the 4 UNIQUE scale groups ONCE into
        # SC_lds (wave w -> group w, no predication), s_barrier, then each wave ds_reads
        # the A/B groups it needs (by wave_m/wave_n). All per-wave addr/rsrc/soffset
        # selection is done host-side via arith.select(wave_id) and passed through the
        # EXISTING reserved sc_rb/sc_gb/i_scrsa/i_sca0 operands -> no asm cselect / no new
        # operands. Mirrors AITER's full-160KB-LDS scale staging.
        _COOP = self.coop
        _TACC = self.tacc  # transposed accumulator: swap MMA operands -> acc = Cᵀ
        _PINBASE = 8
        key = (
            nta,
            ntb,
            n_sub,
            nsa,
            nsb,
            ts_a,
            ts_b,
            nbuf,
            nbuf_b,
            (ki is None) or (ki >= 2),
            (ki is not None) and bool(ki & 1),
            self.wlv,
            self.elgk,
            self.coop,
            _TACC,
            ki,
        )
        if key not in _cache:
            o_acc = list(range(NT))
            t_a = NT
            t_bl = t_a + na
            t_br = t_bl + nb  # ds_read temp outputs
            nsct = 4 * n_sub  # scale temps: A-g0, A-g1, BL, BR x n_sub
            t_sc = t_br + nb  # scale temp base
            _scextra = nsct  # VGPR-direct 2nd scale set (ping-pong)
            set_sz = ntmp + nsct + _scextra
            ntmp2 = NSET * set_sz
            o_cnt = NT + ntmp2  # =&s loop counter
            o_sa = o_cnt + 1
            o_sbl = o_sa + 1
            o_sbr = o_sbl + 1  # advancing gmem soffsets A/BL/BR
            o_ta = o_sbr + 1
            o_tbl = o_ta + 1
            o_tbr = o_tbl + 1  # buf1 (=+kstep) scratch soffsets
            o_sca = [o_tbr + 1 + g for g in range(4)]  # 4 scale soffsets (A-g0, A-g1, BL, BR)
            o_sct = o_sca[3] + 1  # scale scratch soffset
            nout = o_sct + 1
            # scale temp accessors (group: 0=A-g0, 1=A-g1, 2=BL, 3=BR; slot=grp*n_sub+s)
            # _scb[0] = ping-pong scale-set base (0 or nsct), set per phase.
            _scb = [0]

            def sa_t(s, g):
                return t_sc + _scb[0] + g * n_sub + s

            def sbl_t(s):
                return t_sc + _scb[0] + 2 * n_sub + s

            def sbr_t(s):
                return t_sc + _scb[0] + 3 * n_sub + s

            # inputs (after outputs):
            i = nout
            i_ab = [[i + b * n_sub + s for s in range(n_sub)] for b in range(nbuf)]
            i += nbuf * n_sub  # A ds_read base
            i_blb = [[i + b * n_sub + s for s in range(n_sub)] for b in range(nbuf_b)]
            i += nbuf_b * n_sub
            i_brb = [[i + b * n_sub + s for s in range(n_sub)] for b in range(nbuf_b)]
            i += nbuf_b * n_sub
            i_g_ab = [i + b for b in range(nbuf)]
            i += nbuf  # g2s A LDS dest base (sgpr)
            i_g_blb = [i + b for b in range(nbuf_b)]
            i += nbuf_b
            i_g_brb = [i + b for b in range(nbuf_b)]
            i += nbuf_b
            i_gla = [i + s for s in range(nsa)]
            i += nsa  # gmem voffsets A
            i_glb = [i + s for s in range(nsb)]
            i += nsb  # gmem voffsets B
            i_rsa = i
            i += 1
            i_rsb = i
            i += 1  # rsrc
            i_kstep = i
            i += 1
            i += 1  # (legacy const scale dummy, reserved operand slot)
            i_nval = i
            i += 1
            i_sa0 = i
            i += 1
            i_sbl0 = i
            i += 1
            i_sbr0 = i
            i += 1  # soffset inits A/BL/BR (region base k=0)
            i_scrb = [i + b for b in range(_nscbuf)]
            i += _nscbuf  # scale LDS read base (A,B for buf0; coop ds_read source)
            i_scgb = [i + b for b in range(_nscbuf)]
            i += _nscbuf  # scale LDS g2s dest base (per-buf, coop g2s dest)
            i_scrsa = i
            i += 1
            i_scrsb = i
            i += 1  # scale rsrc (A_scale, B_scale)
            i_scvoff = i
            i += 1  # scale per-lane gmem voffset
            i_sca0 = [i + g for g in range(4)]
            i += 4  # scale soffset inits (A-g0, A-g1, BL, BR)

            def emit_ds(buf, off=0):
                # operands only; scales are VGPR-direct (emit_sc_vgpr in the loop).
                r = []
                for ii in range(nta):
                    for s in range(n_sub):
                        r.append(
                            f"ds_read_b128 ${t_a + ii * n_sub + s + off}, ${i_ab[buf][s]} offset:{ii * ts_a}"
                        )
                for ji in range(ntb):
                    for s in range(n_sub):
                        r.append(
                            f"ds_read_b128 ${t_bl + ji * n_sub + s + off}, ${i_blb[buf][s]} offset:{ji * ts_b}"
                        )
                for ji in range(ntb):
                    for s in range(n_sub):
                        r.append(
                            f"ds_read_b128 ${t_br + ji * n_sub + s + off}, ${i_brb[buf][s]} offset:{ji * ts_b}"
                        )
                return r

            def emit_g2s(buf, sa_op, sbl_op, sbr_op):
                r = []
                for st in range(nsa):
                    r.append(
                        f"s_add_u32 m0, ${i_g_ab[buf]}, {st * _NWc * 1024}\n"
                        f"buffer_load_dwordx4 ${i_gla[st]}, ${i_rsa}, ${sa_op} offen lds"
                    )
                for st in range(nsb):
                    r.append(
                        f"s_add_u32 m0, ${i_g_blb[buf]}, {st * _NWc * 1024}\n"
                        f"buffer_load_dwordx4 ${i_glb[st]}, ${i_rsb}, ${sbl_op} offen lds"
                    )
                for st in range(nsb):
                    r.append(
                        f"s_add_u32 m0, ${i_g_brb[buf]}, {st * _NWc * 1024}\n"
                        f"buffer_load_dwordx4 ${i_glb[st]}, ${i_rsb}, ${sbr_op} offen lds"
                    )
                return r

            def ds_line(buf, tt):
                # per-temp ds_read for the in-place refill stream. Scale temps
                # (tt >= t_sc) are VGPR-direct -> no LDS read (empty line).
                if tt < t_bl:
                    rel = tt - t_a
                    ii = rel // n_sub
                    s = rel % n_sub
                    return f"ds_read_b128 ${tt}, ${i_ab[buf][s]} offset:{ii * ts_a}"
                if tt < t_br:
                    rel = tt - t_bl
                    ji = rel // n_sub
                    s = rel % n_sub
                    return f"ds_read_b128 ${tt}, ${i_blb[buf][s]} offset:{ji * ts_b}"
                if tt < t_sc:
                    rel = tt - t_br
                    ji = rel // n_sub
                    s = rel % n_sub
                    return f"ds_read_b128 ${tt}, ${i_brb[buf][s]} offset:{ji * ts_b}"
                return ""

            def emit_inplace(nxt_buf, g2sl):
                # NEXT-K in-place refill, blocked-diagonal (4 A-rows x 8 N-cols)
                # with the K-sub (s) INNERMOST so each acc's n_sub MFMA are consecutive.
                # Both operands free progressively (ds_read of nxt_buf overlaps the MFMA
                # tail); scales are VGPR-direct so their refill is a no-op. GAVOID: g2s
                # only at MFMA slots with no ds_read refill.
                # 4x8: wider block separates each acc's two K-sub MFMA by more
                # independent-acc MFMA -> avoids the accumulator RAW stall.
                bm, bn = 4, 8
                ncol = 2 * ntb
                nib = nta // bm
                ncb = ncol // bn
                cells = []
                for D in range(nib + ncb - 1):
                    for iib in range(nib):
                        cb = D - iib
                        if 0 <= cb < ncb:
                            for di in range(bm):
                                for dj in range(bn):
                                    for s in range(n_sub):
                                        ii = iib * bm + di
                                        col = cb * bn + dj
                                        cells.append((ii, col // ntb, col % ntb, s))
                mlist = []
                for ii, sl, ji, s in cells:
                    tb = t_bl if sl == 0 else t_br
                    sbfn = sbl_t if sl == 0 else sbr_t
                    q = sl * nq + ii * ntb + ji
                    oa, ob = ii % 4, ji
                    at = t_a + ii * n_sub + s
                    bt = tb + ji * n_sub + s
                    sat = sa_t(s, ii // 4)
                    sbt = sbfn(s)
                    if _TACC:  # acc = Cᵀ: src0<->src1, scales, op_sel[0]<->op_sel[1]
                        osel = f"op_sel:[{ob & 1},{oa & 1},0] op_sel_hi:[{(ob >> 1) & 1},{(oa >> 1) & 1},0]"
                        mline = (
                            f"v_mfma_scale_f32_16x16x128_f8f6f4 ${q}, ${bt}, ${at}, ${q}, "
                            f"${sbt}, ${sat} {osel} cbsz:4 blgp:4"
                        )
                    else:
                        osel = f"op_sel:[{oa & 1},{ob & 1},0] op_sel_hi:[{(oa >> 1) & 1},{(ob >> 1) & 1},0]"
                        mline = (
                            f"v_mfma_scale_f32_16x16x128_f8f6f4 ${q}, ${at}, ${bt}, ${q}, "
                            f"${sat}, ${sbt} {osel} cbsz:4 blgp:4"
                        )
                    mlist.append((mline, at, bt, sat, sbt))
                last = {}
                for mi, (_ml, at, bt, sat, sbt) in enumerate(mlist):
                    last[at] = mi
                    last[bt] = mi
                    last[sat] = mi
                    last[sbt] = mi
                mid = set(t for t in last if t_a <= t < t_sc)  # operands (scales VGPR-direct)
                _gset = {}
                if g2sl:
                    _rfslot = set()
                    _rf = set()
                    for mi, (ml, at, bt, sat, sbt) in enumerate(mlist):
                        for rt in (at, bt, sat, sbt):
                            if rt in mid and last[rt] == mi and rt not in _rf:
                                _rfslot.add(mi)
                                _rf.add(rt)
                    _free = [mi for mi in range(len(mlist)) if mi not in _rfslot]
                    _fgap = max(len(_free) // max(len(g2sl), 1), 1)
                    for _k, _fi in enumerate(_free):
                        if (_k % _fgap == 0) and len(_gset) < len(g2sl):
                            _gset[_fi] = len(_gset)
                out = []
                gi = 0
                refilled = set()
                for mi, (ml, at, bt, sat, sbt) in enumerate(mlist):
                    out.append(ml)
                    for rt in (at, bt, sat, sbt):
                        if rt in mid and last[rt] == mi and rt not in refilled:
                            out.append(ds_line(nxt_buf, rt))
                            refilled.add(rt)
                    if g2sl and mi in _gset and gi < len(g2sl):
                        out.append(g2sl[gi])
                        gi += 1
                while gi < len(g2sl):
                    out.append(g2sl[gi])
                    gi += 1
                # end drain: refill the still-pending operand temps (used till end).
                for tt in range(t_a, NT + set_sz):
                    if tt not in refilled:
                        out.append(ds_line(nxt_buf, tt))
                return out

            # ── phase boundary drains ─────────────────────────────────────────
            _ipend = f"s_waitcnt vmcnt({_WLV}) lgkmcnt({_ELGK})\ns_barrier"
            _ipenda = f"s_waitcnt vmcnt({_WLV}) lgkmcnt({_ELGK})\ns_barrier"

            # ── VGPR-direct scale prefetch (lane-contiguous buffer_load to VGPR) ──
            # emit_sc_vgpr(tb) loads this kk's 2*n_sub*2 scale dwords DIRECT to the
            # pinned scale set (A -> v[p:..], B -> v[p+scw:..]); no LDS / ds_read. NO
            # vmcnt here (in flight, drained by the phase barrier) -> overlaps mfma.
            _pbsc = _PINBASE  # scale VGPRs pinned first (PINSC), at PINBASE
            _scw = 2 * n_sub  # scale dwords per operand (2 region groups x n_sub subs)
            _scwx = {1: "", 2: "x2", 4: "x4"}.get(_scw, f"x{_scw}")  # buffer_load width suffix

            def emit_sc_vgpr(tb):
                p = _pbsc + tb
                return [
                    f"buffer_load_dword{_scwx} v[{p}:{p + _scw - 1}], ${i_scvoff}, ${i_scrsa}, ${o_sca[0]} offen",
                    f"buffer_load_dword{_scwx} v[{p + _scw}:{p + 2 * _scw - 1}], ${i_scvoff}, ${i_scrsb}, ${o_sca[2]} offen",
                ]

            _scvstep = 64 * (2 * n_sub) * 4  # lane-contig kk stride in bytes

            def _scv_adv():
                return [
                    f"s_add_u32 ${o_sca[0]}, ${o_sca[0]}, {_scvstep}",
                    f"s_add_u32 ${o_sca[2]}, ${o_sca[2]}, {_scvstep}",
                ]

            # ── Cooperative LDS scale staging (_COOP) ─────────────────────────
            # 2-deep software pipeline mirroring the operands: the 4 waves each
            # buffer_load ONE of the 4 unique scale groups straight to its SC_lds
            # slot (no per-wave redundancy), s_barrier, then each wave ds_reads the
            # A group (slot=wave_m) + B group (slot=2+wave_n) it needs into the SAME
            # pinned scale VGPRs the MFMA reads. g2s is 2-ahead of consume, ds_read
            # 1-ahead (VGPR set ping-pong set0/set1, LDS buf ping-pong 0/1). All
            # per-wave rsrc/soffset/addr selection is host-side (arith.select) so the
            # asm carries no cselect. buf1 LDS is reached via ds_read offset: imm.
            def emit_sc_coop_g2s(buf):
                # one wave -> one group (4 dwords/lane) into SC_lds[buf] slot=wave_id.
                return [
                    f"s_add_u32 m0, ${i_scgb[buf]}, 0\n"
                    f"buffer_load_dwordx4 ${i_scvoff}, ${i_scrsa}, ${o_sca[0]} offen lds"
                ]

            def emit_sc_coop_ds(tb, buf):
                # ds_read this wave's A (slot wave_m) + B (slot 2+wave_n) groups from
                # SC_lds[buf] into the pinned scale set at PINBASE+tb.
                p = _pbsc + tb
                off = buf * sc_buf_stride
                _o = f" offset:{off}" if off else ""
                return [
                    f"ds_read_b128 v[{p}:{p + _scw - 1}], ${i_scrb[0]}{_o}",
                    f"ds_read_b128 v[{p + _scw}:{p + 2 * _scw - 1}], ${i_scrb[1]}{_o}",
                ]

            # coop phase barrier: full drain so the 1-ahead scale ds_read (lgkm, shares
            # the counter with operand ds_reads) is guaranteed complete before the next
            # phase's first MFMA.
            _ipend_coop = "s_waitcnt vmcnt(0) lgkmcnt(0)\ns_barrier"

            # ── prologue (before the hw-loop label) ───────────────────────────
            L = [
                f"s_mov_b32 ${o_cnt}, 0",
                f"s_mov_b32 ${o_sa}, ${i_sa0}",
                f"s_mov_b32 ${o_sbl}, ${i_sbl0}",
                f"s_mov_b32 ${o_sbr}, ${i_sbr0}",
            ]
            for g in range(4):
                L.append(f"s_mov_b32 ${o_sca[g]}, ${i_sca0[g]}")
            # in-place double-buffer prologue: read buf0 (k=0) into set0 before the loop.
            # The operand ds_read (lgkmcnt) and the VGPR-direct scale buffer_load (vmcnt)
            # write disjoint VGPRs and are mutually independent, so issue BOTH then drain
            # with a SINGLE combined wait -- their latencies overlap instead of serializing
            # (the loop's first MFMA needs operands AND scales anyway). Removes one prologue
            # wait edge from the fixed (K-independent) launch cost, a larger relative win on
            # low-trip K (e.g. K=4096/16-trip) where the prologue is a bigger fraction.
            L += emit_ds(0, 0)
            if _COOP:
                # 2-deep prime: g2s P0->LDS0 (+P1->LDS1 when a loop follows), barrier
                # (cross-wave LDS visible), then ds_read LDS0->set0 for the P0 consume.
                L += emit_sc_coop_g2s(0)
                L += _scv_adv()
                if (ki is None) or (ki >= 2):
                    L += emit_sc_coop_g2s(1)
                    L += _scv_adv()
                L.append("s_waitcnt vmcnt(0)")
                L.append("s_barrier")
                L += emit_sc_coop_ds(0, 0)
                L.append("s_waitcnt lgkmcnt(0)")
            else:
                # VGPR-direct scale prologue: set A = phase-A iter0 scales.
                L += emit_sc_vgpr(0) + _scv_adv()
                L.append("s_waitcnt vmcnt(0) lgkmcnt(0)")
            # K%256 (odd KI) support: the unroll-2 do-while processes K in PAIRS of
            # 256-blocks. ``i_nval`` is the floor-even count ((KI//2)*2); an odd
            # trailing 256-block is handled by a single MFMA-only phase-A tail after
            # the loop. KI==1 (K=256) has zero full pairs, so the do-while (which runs
            # >=1 trip) is omitted entirely and the prologue feeds the tail directly.
            _has_loop = (ki is None) or (ki >= 2)
            _has_tail = (ki is not None) and bool(ki & 1)

            if _has_loop:
                L.append("1:")

                # ── unroll-2 body (phase A even-k, phase B odd-k) ─────────────────
                # SCV_ILV: the scale buffer_load is interleaved INTO the mfma stream
                # (prepended to g2sl) so it overlaps the MFMA and stops competing with g2s
                # for the boundary VMEM slot; _scv_adv (o_sca advance) is emitted AFTER
                # emit_inplace so the interleaved loads read the un-advanced offset.
                _g2sA = emit_g2s(0, o_sa, o_sbl, o_sbr)
                _g2sB = emit_g2s(1, o_ta, o_tbl, o_tbr)
                # phase A: consume set0; g2s P+2 -> LDS0, ds_read P+1 (LDS1) -> set1.
                _scb[0] = 0
                if _COOP:
                    _scA = emit_sc_coop_g2s(0) + emit_sc_coop_ds(nsct, 1)
                else:
                    _scA = emit_sc_vgpr(nsct)
                L += emit_inplace(1, _scA + _g2sA)
                L += _scv_adv()
                L.append(_ipenda)
                L.append(f"s_add_u32 ${o_ta}, ${o_sa}, ${i_kstep}")
                L.append(f"s_add_u32 ${o_tbl}, ${o_sbl}, ${i_kstep}")
                L.append(f"s_add_u32 ${o_tbr}, ${o_sbr}, ${i_kstep}")
                # phase B: consume set1; g2s P+2 -> LDS1, ds_read P+1 (LDS0) -> set0.
                _scb[0] = nsct
                if _COOP:
                    _scB = emit_sc_coop_g2s(1) + emit_sc_coop_ds(0, 0)
                else:
                    _scB = emit_sc_vgpr(0)
                L += emit_inplace(0, _scB + _g2sB)
                L += _scv_adv()
                L.append(_ipend)
                for _so in (o_sa, o_sbl, o_sbr):
                    L.append(f"s_add_u32 ${_so}, ${_so}, ${i_kstep}")
                    L.append(f"s_add_u32 ${_so}, ${_so}, ${i_kstep}")
                L.append(f"s_add_u32 ${o_cnt}, ${o_cnt}, 2")
                L.append(f"s_cmp_lt_u32 ${o_cnt}, ${i_nval}")
                L.append("s_cbranch_scc1 1b")

            if _has_tail:
                # ── odd-KI trailing phase-A (MFMA-only) ──────────────────────────
                # Operands for the last even-k are already in the ds_read temps (refilled
                # by the loop's last phase-B emit_inplace(nxt_buf=0); or by the prologue
                # emit_ds when KI==1) and the set0 scales are loaded (last phase-B
                # emit_sc_vgpr(0); or prologue emit_sc_vgpr(0)). No g2s / ds / scale
                # reload -- just drain everything and run the 128 scaled MFMAs.
                L.append("s_waitcnt vmcnt(0) lgkmcnt(0)")
                _scb[0] = 0
                _bm, _bn = 4, 8  # match loop-body block (see emit_inplace)
                _ncol = 2 * ntb
                _nib = nta // _bm
                _ncb = _ncol // _bn
                for _D in range(_nib + _ncb - 1):
                    for _iib in range(_nib):
                        _cb = _D - _iib
                        if 0 <= _cb < _ncb:
                            for _di in range(_bm):
                                for _dj in range(_bn):
                                    for _s in range(n_sub):
                                        _ii = _iib * _bm + _di
                                        _col = _cb * _bn + _dj
                                        _sl = _col // ntb
                                        _ji = _col % ntb
                                        _tb = t_bl if _sl == 0 else t_br
                                        _sbfn = sbl_t if _sl == 0 else sbr_t
                                        _q = _sl * nq + _ii * ntb + _ji
                                        _oa, _ob = _ii % 4, _ji
                                        _at = t_a + _ii * n_sub + _s
                                        _bt = _tb + _ji * n_sub + _s
                                        _sat = sa_t(_s, _ii // 4)
                                        _sbt = _sbfn(_s)
                                        if _TACC:  # acc = Cᵀ (swap operands/scales/op_sel)
                                            _osel = (
                                                f"op_sel:[{_ob & 1},{_oa & 1},0] "
                                                f"op_sel_hi:[{(_ob >> 1) & 1},{(_oa >> 1) & 1},0]"
                                            )
                                            L.append(
                                                f"v_mfma_scale_f32_16x16x128_f8f6f4 ${_q}, ${_bt}, "
                                                f"${_at}, ${_q}, ${_sbt}, ${_sat} {_osel} cbsz:4 blgp:4"
                                            )
                                        else:
                                            _osel = (
                                                f"op_sel:[{_oa & 1},{_ob & 1},0] "
                                                f"op_sel_hi:[{(_oa >> 1) & 1},{(_ob >> 1) & 1},0]"
                                            )
                                            L.append(
                                                f"v_mfma_scale_f32_16x16x128_f8f6f4 ${_q}, ${_at}, "
                                                f"${_bt}, ${_q}, ${_sat}, ${_sbt} {_osel} cbsz:4 blgp:4"
                                            )

            # ── register pinning (PIN + PINSC): scales LOW (PINBASE), frags after ──
            # Bypasses the LLVM RA "Cannot decrease cascade number" crash and aligns
            # the scale literals to the PINBASE base that emit_sc_vgpr writes.
            _vtmp = ["=&v"] * ntmp2
            bv = _PINBASE
            for s in range(NSET):
                order = list(range(ntmp))
                _nsc2 = nsct * 2  # 2 ping-pong scale sets (VGPR-direct)
                for j in range(_nsc2):
                    _vtmp[s * set_sz + ntmp + j] = f"=&{{v{bv}}}"
                    bv += 1
                for j in order:  # frags: vector<4xi32> = 4 VGPR
                    _vtmp[s * set_sz + j] = f"=&{{v[{bv}:{bv + 3}]}}"
                    bv += 4
            cons = ",".join(
                ["=a"] * NT
                + _vtmp
                + ["=&s"] * 12  # accs, temps(ops+scale), cnt+3soff+3tmp+4scsoff+1sctmp
                + ["v"] * ((nbuf + 2 * nbuf_b) * n_sub)  # a(nbuf)/bl/br(nbuf_b) ds_read bases
                + ["s"] * (nbuf + 2 * nbuf_b)  # g2s dest bases
                + ["v"] * (nsa + nsb)  # voffsets
                + ["s", "s", "s", "v", "s"]  # rsrc_a, rsrc_b, kstep, scv, nval
                + ["s", "s", "s"]  # operand soffset inits A/BL/BR
                + ["v"] * _nscbuf  # scale LDS read base (reserved)
                + ["s"] * _nscbuf  # scale LDS g2s dest base (reserved)
                + ["s", "s"]  # scale rsrc A, B
                + ["v"]  # scale voffset
                + ["s", "s", "s", "s"]  # scale soffset inits (A-g0, A-g1, BL, BR)
                + [str(q) for q in o_acc]
            )  # tied accs
            st = (
                "!llvm.struct<("
                + ", ".join(
                    ["vector<4xf32>"] * NT
                    + (["vector<4xi32>"] * ntmp + ["i32"] * nsct + ["i32"] * _scextra) * NSET
                    + ["i32"] * 12
                )
                + ")>"
            )
            _cache[key] = ("\n".join(L), cons, st)
        asm, cons, st = _cache[key]
        ins = []
        for b in range_constexpr(nbuf):  # A pool
            for s in range_constexpr(n_sub):
                ins.append(_raw(a_base[b][s]))
        for fr in (bl_base, br_base):  # B pool
            for b in range_constexpr(nbuf_b):
                for s in range_constexpr(n_sub):
                    ins.append(_raw(fr[b][s]))
        for b in range_constexpr(nbuf):  # g2s A dest
            ins.append(_raw(abase[b]))
        for fr in (blbase, brbase):  # g2s B dest
            for b in range_constexpr(nbuf_b):
                ins.append(_raw(fr[b]))
        for v in gl_a:
            ins.append(_raw(v))
        for v in gl_b:
            ins.append(_raw(v))
        ins.append(_raw(rsrc_a))
        ins.append(_raw(rsrc_b))
        ins.append(_raw(kstep))
        ins.append(_raw(scv))
        ins.append(_raw(nval))
        ins.append(_raw(soff0))
        ins.append(_raw(soff0_bl))
        ins.append(_raw(soff0_br))
        for b in range_constexpr(_nscbuf):
            ins.append(_raw(sc_rb[b]))  # scale LDS read base (reserved)
        for b in range_constexpr(_nscbuf):
            ins.append(_raw(sc_gb[b]))  # scale LDS g2s dest base (reserved)
        ins.append(_raw(sc_rsa))
        ins.append(_raw(sc_rsb))  # scale rsrc
        ins.append(_raw(sc_voff))  # scale voffset
        for g in range_constexpr(4):
            ins.append(_raw(sc_soff0[g]))  # scale soffset inits
        for q in range_constexpr(nq):
            ins.append(_raw(cL[q]))
        for q in range_constexpr(nq):
            ins.append(_raw(cR[q]))
        r = _llvm.inline_asm(ir.Type.parse(st), ins, asm, cons, has_side_effects=True)
        o = [Vec(_llvm.extractvalue(ir.Type.parse("vector<4xf32>"), r, [q])) for q in range_constexpr(nq * 2)]
        return o[:nq], o[nq:]


# ── Compile factory (NT, BLOCK_M=BLOCK_N=BLOCK_K=256) ─────────────────────────


def _build_mxfp4_gemm_kernel(
    *,
    K: int,
    group_m: int = 4,
    num_xcds: int = 8,
    group_n: int = 0,
    wlv: int = 10,
    elgk: int = 9,
    coop: bool = False,
    ksplit: int = 1,
    taccw: bool = False,
    out_fp16: bool = False,
):
    BLOCK_M = 256
    BLOCK_N = 256
    BLOCK_K = 256
    # bf16/fp16 output: only the f32->out_ty cast in the store differs. fp16 uses the
    # narrow scalar store (generic ``.to``); the wide TACCW store is bf16-only, so the
    # caller forces taccw=False for fp16.
    _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
    # const_expr() resolves compile-time branches from LOCALS/params, not reliably from
    # module globals -> alias the per-shape epilogue selection to locals before traced use.
    _l_taccw = taccw  # autotune-selected per shape (never-regress epilogue variant axis)
    _l_tacc = _l_taccw  # TACCW needs the acc=Cᵀ MMA operand swap
    swizzle = True
    assert BLOCK_K % 128 == 0 and K % BLOCK_K == 0
    # ── Split-K: each WG computes a K/ksplit slice into workspace[split], host reduces.
    # ksplit>1 fills the CUs on FEW-TILE large-K shapes where the one-WG-per-tile grid
    # leaves most CUs idle and each WG otherwise runs the WHOLE K-loop serially.
    # ONLY the loop trip count + per-split K-start bases change; row/scale STRIDES stay
    # full-K (rows are full length in gmem). The asm K-loop is self-contained (driven by
    # ``ki`` + initial soffsets + CONSTANT in-loop steps), so split-K touches NO asm.
    assert K % ksplit == 0, f"K={K} not divisible by ksplit={ksplit}"
    K_loop = K // ksplit
    assert K_loop % BLOCK_K == 0, f"K/ksplit={K_loop} not a multiple of {BLOCK_K}"

    const_expr(True)
    NBB = const_expr(2)  # B/SC pool (unroll-2)
    NABUF = const_expr(2)  # A pool (unroll-2)
    OCC = const_expr(1)  # 1 wave/SIMD -> full 256-AGPR file for the accumulator

    KI = K_loop // BLOCK_K  # loop trip count = per-split 256-K blocks (full K when ksplit=1)
    N_SUB = BLOCK_K // 128
    BPR = BLOCK_K // 2  # packed-fp4 bytes per K-iter row in LDS
    KSTEP = BPR
    K2 = K // 2  # packed-fp4 gmem row stride (bytes) -- FULL K (rows span all K)
    # Per-split K-start byte shifts (split s in 0..ksplit-1):
    #   A/B operands: s * (K_loop/2) bytes into the packed-fp4 row.
    #   scale soffset: s * KI * _scvstep bytes (_scvstep = 64*(2*N_SUB)*4 per 256-K block,
    #   the in-asm o_sca advance; KI blocks per split -> contiguous within the full region).
    _AB_SPLIT_STEP = K_loop // 2
    _SC_SPLIT_STEP = KI * (64 * (2 * N_SUB) * 4)

    N_TILES_A = BLOCK_M // 32  # 8: wave_m covers 128 M-rows
    LDS_BN_HALF = BLOCK_N // 2  # 128: slice width
    N_TILES_BH = LDS_BN_HALF // 32  # 4: wave_n covers 64 N-cols/slice

    LDS_ROW_STRIDE = BPR
    a_lds_size = BLOCK_M * LDS_ROW_STRIDE  # 256 rows
    bh_lds_size = LDS_BN_HALF * LDS_ROW_STRIDE  # 128 rows per B half

    _ROWS_PER_STEP = 64 // (BPR // 16) * (256 // 64)  # n_waves = 256//64 = 4
    N_LDS_STEPS_A = BLOCK_M // _ROWS_PER_STEP
    N_LDS_STEPS_BH = LDS_BN_HALF // _ROWS_PER_STEP

    _PRELL = const_expr(2)  # operand buffers prefilled (k=0..PRELL-1)
    _NSCBUF = const_expr(2)
    K128 = const_expr(K // 128)
    _SCBUF = 4 * 4 * (BLOCK_K // 128) * 64  # n_waves * groups * n_sub * 64 dwords
    _SCW = const_expr(4 * N_SUB * 64)  # dwords per wave-region per scale buffer

    _anns = {f"A_lds{i}": fx.Array[fx.Float8E4M3FN, a_lds_size, 16] for i in range_constexpr(NABUF)}
    for _b in range_constexpr(NBB):
        _anns[f"BL_lds{_b}"] = fx.Array[fx.Float8E4M3FN, bh_lds_size, 16]
    for _b in range_constexpr(NBB):
        _anns[f"BR_lds{_b}"] = fx.Array[fx.Float8E4M3FN, bh_lds_size, 16]
    for _b in range_constexpr(_NSCBUF):
        _anns[f"SC_lds{_b}"] = fx.Array[fx.Int32, _SCBUF, 16]
    SharedStorageFp4_4w = fx.struct(type("SharedStorageFp4_4w", (), {"__annotations__": _anns}))

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kernel_gemm_4w(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type
        lds = fx.SharedAllocator().allocate(SharedStorageFp4_4w).peek()
        A_buf = [getattr(lds, f"A_lds{i}") for i in range_constexpr(NABUF)]
        BL_buf = [getattr(lds, f"BL_lds{i}") for i in range_constexpr(NBB)]
        BR_buf = [getattr(lds, f"BR_lds{i}") for i in range_constexpr(NBB)]

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 2
        wave_n = wave_id % 2
        # ── Tile-INDEPENDENT setup (hoisted out of the per-tile body; every value
        # below depends only on the fixed LDS buffers / wave id / whole-tensor
        # resources, NOT block_m/n) ──
        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        mfma = MfmaScaleFp4(N_TILES_A, N_TILES_BH, packed=True, wlv=wlv, elgk=elgk, coop=coop, tacc=_l_tacc)

        gl_off_a = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_A, BPR, swizzle=swizzle)
        gl_off_b = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_BH, BPR, swizzle=swizzle)
        rsrc_a = buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=c_m * K2)
        rsrc_b = buffer_ops.create_buffer_resource(B_T, max_size=False, num_records_bytes=c_n * K2)
        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        bl_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_BH, F8_IR_t, wave_id)
        br_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_BH, F8_IR_t, wave_id)

        a_s2r = S2RLoaderFp4(wave_m, N_TILES_A, LDS_ROW_STRIDE, swizzle=swizzle)
        b_s2r = S2RLoaderFp4(wave_n, N_TILES_BH, LDS_ROW_STRIDE, swizzle=swizzle)

        # A scale: 8 M-tiles span 2 x 64-row groups (4 tiles each). Packed -> ONE i32
        # per group. group-span-ceil the dim so the floor in the loader's record count
        # covers the edge scale group.
        _qm = ((c_m + 63) // 64) * 64
        _qn = ((c_n + 63) // 64) * 64
        sa_s2r = ScaleS2RPacked(A_scale, _qm, K, 4)
        sb_s2r = ScaleS2RPacked(B_scale, _qn, K, 4)
        # split-K writes partials to workspace C[ksplit*M, N] (row band split*M); the
        # StoreCPlain c_rows only bounds the SRD (not used in the index), so widen it.
        _c_store_rows = c_m if const_expr(ksplit == 1) else c_m * fx.Int32(ksplit)
        store_c = StoreCPlain(C, _c_store_rows, c_n, mfma.idx, N_TILES_A, N_TILES_BH, _out_ty)

        wave_m_off = wave_m * (N_TILES_A * 16)  # 0 or 128
        wave_n_off = wave_n * (N_TILES_BH * 16)  # 0 or 64
        SC_buf = [getattr(lds, f"SC_lds{b}") for b in range_constexpr(_NSCBUF)]

        # LDS read/g2s-dest bases + scale resources: all derived from the fixed LDS
        # buffers / wave id, so identical for every output tile -> compute once.
        a_base6 = [
            [a_s2r.base_addr(A_buf[b], s) for s in range_constexpr(N_SUB)] for b in range_constexpr(NABUF)
        ]
        bl_base6 = [
            [b_s2r.base_addr(BL_buf[b], s) for s in range_constexpr(N_SUB)] for b in range_constexpr(NBB)
        ]
        br_base6 = [
            [b_s2r.base_addr(BR_buf[b], s) for s in range_constexpr(N_SUB)] for b in range_constexpr(NBB)
        ]

        def _gbase(buf):
            v = fx.Int32(fx.ptrtoint(buf.ptr)) + fx.Int32(wave_id) * fx.Int32(1024)
            return rocdl.readfirstlane(T.i32, v)

        abase6 = [_gbase(A_buf[b]) for b in range_constexpr(NABUF)]
        blbase6 = [_gbase(BL_buf[b]) for b in range_constexpr(NBB)]
        brbase6 = [_gbase(BR_buf[b]) for b in range_constexpr(NBB)]
        gl_a6 = [fx.Int32(gl_off_a[st]) for st in range_constexpr(N_LDS_STEPS_A)]
        gl_b6 = [fx.Int32(gl_off_b[st]) for st in range_constexpr(N_LDS_STEPS_BH)]
        scv6 = fx.Int32(0x7F7F7F7F)
        _scrb_lane = lane_id
        sc_rb6 = [
            fx.ptrtoint(
                fx.add_offset(
                    SC_buf[b].ptr, fx.make_int_tuple(fx.Int32(wave_id) * fx.Int32(_SCW) + _scrb_lane)
                )
            )
            for b in range_constexpr(_NSCBUF)
        ]
        sc_gb6 = [
            rocdl.readfirstlane(
                T.i32,
                fx.Int32(
                    fx.ptrtoint(
                        fx.add_offset(SC_buf[b].ptr, fx.make_int_tuple(fx.Int32(wave_id) * fx.Int32(_SCW)))
                    )
                ),
            )
            for b in range_constexpr(_NSCBUF)
        ]
        _scrsa_v = sa_s2r.rsrc
        _scrsb_v = sb_s2r.rsrc
        sc_voff6 = lane_id * fx.Int32(8 * N_SUB)

        # ── Cooperative scale staging addresses (COOP autotune axis) ────────────
        # The 4 waves cooperatively load the 4 UNIQUE scale groups ONCE into SC_lds
        # (wave w -> slot w, 1KB each = 4 dwords/lane x 64 lanes), then each wave
        # ds_reads the A group it needs (slot=wave_m) and the B group (slot=2+wave_n).
        # All per-wave selection is hoisted here (host) so the asm carries no cselect:
        #   - coop g2s dest      -> reuse sc_gb6[buf]  (SC_buf[buf] + wave_id*1KB)
        #   - coop ds_read A/B   -> reuse sc_rb6[0/1]  (SC_buf[0] + slot*1KB + lane*16);
        #                           buf1 is reached in-asm via a ds_read offset: immediate
        #                           (the SC_lds buf stride is a compile-time constant).
        #   - coop g2s rsrc      -> select(wave_id<2, A_scale, B_scale) -> passed as sc_rsa
        # _SCSLOT = bytes per LDS scale slot (one group = 4 dwords/lane * 64 lanes * 4B).
        _SCSLOT = const_expr(4 * 64 * 4)  # 1024 B
        if const_expr(coop):
            # scalar wave_id so the cond is SCC (not VCC) -> arith.select on the two
            # SGPR rsrc descriptors lowers to s_cselect -> result stays in SGPR (a
            # buffer rsrc MUST be scalar; a VGPR rsrc is an invalid buffer operand).
            _wid_s = rocdl.readfirstlane(T.i32, wave_id)
            _w_lt2 = _wid_s < fx.Int32(2)
            coop_rsa = arith.select(_w_lt2, _scrsa_v, _scrsb_v)
            sc_gb6 = [
                rocdl.readfirstlane(
                    T.i32,
                    fx.Int32(
                        fx.ptrtoint(
                            fx.add_offset(
                                SC_buf[b].ptr,
                                fx.make_int_tuple(fx.Int32(wave_id) * fx.Int32(_SCSLOT // 4)),
                            )
                        )
                    ),
                )
                for b in range_constexpr(_NSCBUF)
            ]
            sc_rb6 = [
                fx.ptrtoint(
                    fx.add_offset(
                        SC_buf[0].ptr,
                        fx.make_int_tuple(_slot * fx.Int32(_SCSLOT // 4) + lane_id * fx.Int32(4)),
                    )
                )
                for _slot in (wave_m, fx.Int32(2) + wave_n)
            ]
        else:
            coop_rsa = _scrsa_v

        def _scsoff(base, extra):
            grp = (base + fx.Int32(extra)) // fx.Int32(64)
            return rocdl.readfirstlane(
                T.i32, (grp * fx.Int32(K128) + fx.Int32(_PRELL * N_SUB)) * fx.Int32(256)
            )

        # ── Per-tile closures (block_m/block_n -> offsets; fill; compute; store) ──
        def _offs(_pid):
            bm, bn = grouped_xcd_pid(
                _pid, c_m, c_n, BLOCK_M, BLOCK_N, group_m=group_m, num_xcds=num_xcds, group_n=group_n
            )
            a_off = bm * BLOCK_M * K2
            bl_off = bn * BLOCK_N * K2
            br_off = (bn * BLOCK_N + LDS_BN_HALF) * K2
            sa_b = fx.Int32(bm * BLOCK_M + wave_m_off)
            sbl_b = fx.Int32(bn * BLOCK_N + wave_n_off)
            sbr_b = fx.Int32(bn * BLOCK_N + LDS_BN_HALF + wave_n_off)
            return (bm, bn, a_off, bl_off, br_off, sa_b, sbl_b, sbr_b)

        def _fill(o):
            # prefill ALL _PRELL operand buffers (k=0..PRELL-1) for A/BL/BR. The
            # buffer_load_lds of different k buffers are mutually independent.
            _, _, a_off, bl_off, br_off, _, _, _ = o
            for _pp in range_constexpr(0, _PRELL):
                if const_expr(KI > _pp):
                    a_g2s.load(A_buf[_pp], a_off + _pp * KSTEP)
            for _pp in range_constexpr(0, _PRELL):
                if const_expr(KI > _pp):
                    bl_g2s.load(BL_buf[_pp], bl_off + _pp * KSTEP)
                    br_g2s.load(BR_buf[_pp], br_off + _pp * KSTEP)

        def _drain():
            # scale stores skipped (VGPR-direct); drain lgkmcnt + barrier so operand
            # g2s land + are cross-wave visible before the asm reads them.
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string="s_waitcnt lgkmcnt(0)",
                constraints="",
                has_side_effects=True,
            )
            wait_barrier(0)

        def _compute(o, _split=None):
            _, _, a_off, bl_off, br_off, sa_b, sbl_b, sbr_b = o
            accL = [mfma.zero_value] * (N_TILES_A * N_TILES_BH)
            accR = [mfma.zero_value] * (N_TILES_A * N_TILES_BH)
            soff6_a = rocdl.readfirstlane(T.i32, a_off + fx.Int32(_PRELL * KSTEP))
            soff6_bl = rocdl.readfirstlane(T.i32, bl_off + fx.Int32(_PRELL * KSTEP))
            soff6_br = rocdl.readfirstlane(T.i32, br_off + fx.Int32(_PRELL * KSTEP))
            # VGPR-direct scale soffsets: A-group0 (_soa) and B (_sob) per the wave's
            # region-group id; A-group1 (+64 rows) and BR keep the packed-group soffset.
            _sc1 = _scsoff(sa_b, 64)
            _sc3 = _scsoff(sbr_b, 0)
            _wia = sa_b // fx.Int32(128)
            _wib = (sbl_b // fx.Int32(256)) * fx.Int32(2) + (sbl_b % fx.Int32(256)) // fx.Int32(64)
            _soa = rocdl.readfirstlane(T.i32, _wia * fx.Int32(K128) * fx.Int32(512))
            _sob = rocdl.readfirstlane(T.i32, _wib * fx.Int32(K128) * fx.Int32(512))
            sc_soff06 = [_soa, _sc1, _sob, _sc3]
            _sc_rsa_arg = _scrsa_v
            if const_expr(coop):
                # per-wave group soffset: waves 0/1 -> A region (2*bm + wave_id),
                # waves 2/3 -> B region (2*bn + (wave_id-2)). The g2s reads ONE group
                # at this soffset; the in-asm coop path uses sc_soff0[0] only.
                bm_t, bn_t = o[0], o[1]
                _coop_reg = arith.select(
                    wave_id < fx.Int32(2),
                    fx.Int32(2) * bm_t + wave_id,
                    fx.Int32(2) * bn_t + (wave_id - fx.Int32(2)),
                )
                _coop_soff = rocdl.readfirstlane(T.i32, _coop_reg * fx.Int32(K128) * fx.Int32(512))
                sc_soff06 = [_coop_soff, _sc1, _sob, _sc3]
                _sc_rsa_arg = coop_rsa
            if const_expr(ksplit > 1):
                # shift every scale soffset to this split's K-start (region is full-K
                # contiguous; per-256-K advance is _scvstep, KI blocks per split).
                _scsh = rocdl.readfirstlane(T.i32, _split * fx.Int32(_SC_SPLIT_STEP))
                sc_soff06 = [rocdl.readfirstlane(T.i32, _x + _scsh) for _x in sc_soff06]
            return mfma.call_mxfp4_wholeloop(
                a_base6,
                bl_base6,
                br_base6,
                a_s2r.tile_stride,
                b_s2r.tile_stride,
                abase6,
                blbase6,
                brbase6,
                gl_a6,
                gl_b6,
                rsrc_a,
                rsrc_b,
                fx.Int32(KSTEP),
                scv6,
                accL,
                accR,
                N_SUB,
                N_LDS_STEPS_A,
                N_LDS_STEPS_BH,
                fx.Int32((KI // 2) * 2),
                soff6_a,
                soff6_bl,
                soff6_br,
                sc_rb6,
                sc_gb6,
                _sc_rsa_arg,
                _scrsb_v,
                sc_voff6,
                sc_soff06,
                ki=KI,
                sc_buf_stride=(_SCBUF * 4),
            )

        def _store(o, accL, accR, _split=None):
            bm, bn = o[0], o[1]
            base_row = bm * BLOCK_M + wave_m_off
            if const_expr(ksplit > 1):
                base_row = base_row + _split * c_m  # write to workspace row band split*M
            base_col_l = bn * BLOCK_N + wave_n_off
            base_col_r = bn * BLOCK_N + LDS_BN_HALF + wave_n_off
            if const_expr(_l_taccw):
                store_c.store_tacc_wide(accL, base_row, base_col_l)
                store_c.store_tacc_wide(accR, base_row, base_col_r)
                return
            store_c.store(accL, base_row, base_col_l)
            store_c.store(accR, base_row, base_col_r)

        def _split_shift(o, _split):
            # add this split's K-start to the A/B operand gmem offsets (row stride full-K).
            bm, bn, a_off, bl_off, br_off, sa_b, sbl_b, sbr_b = o
            _sh = _split * fx.Int32(_AB_SPLIT_STEP)
            return (bm, bn, a_off + _sh, bl_off + _sh, br_off + _sh, sa_b, sbl_b, sbr_b)

        if const_expr(ksplit > 1):
            # split-K: grid = total_tiles*ksplit; bid -> (tile, split). Each WG computes a
            # K/ksplit partial of its tile into workspace row band split*M; host reduces.
            _bid = fx.block_idx.x
            _ntile = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
            _tile = _bid % _ntile
            _split = _bid // _ntile
            o = _split_shift(_offs(_tile), _split)
            _fill(o)
            _drain()
            accL, accR = _compute(o, _split)
            _store(o, accL, accR, _split)
        else:
            o = _offs(fx.block_idx.x)
            _fill(o)
            _drain()
            accL, accR = _compute(o)
            _store(o, accL, accR)

    # agpr-alloc=256 lets the backend place the 256-f32 accumulator in AGPR;
    # waves_per_eu=1 -> the full 512-VGPR file is one wave's (no spill).
    _pt = {"passthrough": [["amdgpu-agpr-alloc", "256"]]}
    gemm_value_attrs = {"rocdl.flat_work_group_size": "256,256", "rocdl.waves_per_eu": OCC, **_pt}

    # Return the BARE kernel (NOT a launch): the fused factory (``_compile_mxfp4_fused``)
    # issues the A/B scale preshuffle kernels + this GEMM from a single @flyc.jit host stub
    # (one Python dispatch, no separate preshuffle launch / CPU sync -- mirrors the mxfp8
    # backend). BLOCK_M/BLOCK_N/ksplit/value_attrs are returned so the stub can size the
    # grid + attrs.
    return kernel_gemm_4w, BLOCK_M, BLOCK_N, ksplit, gemm_value_attrs


# ── Primus-Turbo host wrapper ────────────────────────────────────────────────

_MXFP4_LAUNCH_CACHE: dict = {}  # (K, gm, xcd, gn, wlv, elgk, coop, ksplit, taccw, out_fp16) -> fused launch
_MXFP4_AT_CACHE: dict = {}  # (M, N, K, gm, xcd, gn, wlv, elgk, taccw, coop) -> [raw, compiled_or_None]
_MXFP4_CFG_CACHE: dict = {}  # (M, N, K) -> (gm, gn, xcd, wlv, elgk, taccw, coop)

# Autotune candidates for the L2 (group_m / group_n / num_xcds) workgroup swizzle.
# The swizzle is a pure WG->tile bijection (correctness-invariant), but it drives
# L2 residency and the partial-wave tail, so the per-shape optimum is not captured
# by a static heuristic. We pick the best by a quick timed sweep on the real
# operands, cached per (M, N, K).
_MXFP4_AUTOTUNE_CANDIDATES = [
    (4, 0, 8),
    (8, 0, 8),
    (16, 0, 8),
    (8, 0, 4),
    (16, 0, 4),
    (4, 4, 8),
    (8, 6, 8),
    (4, 8, 8),
    # Wider N-bands (group_n 16/32) + no-M-group (group_m 1) cover wide-N / small-M
    # shapes the group_n<=8 set leaves on the table. Swizzle is correctness-invariant
    # and autotune takes the global min, so extra candidates never regress.
    (4, 16, 8),
    (4, 32, 8),
    (8, 32, 8),
    (1, 8, 8),
]


def _mxfp4_nt_config(M, N, K):
    """Per-shape (group_m, group_n, num_xcds) for the BN256 path (2D N-band L2
    swizzle). Mirrors the standalone production recommend_config BN256 branch:
    wide-N (nb>=96) bands nb//8; big-K down-projections band on top (K>=28672 ->
    16 narrow aligned bands; K>=11008 -> width-4 bands); else 1D GROUP_M swizzle."""
    nb = N // 256
    num_xcds = 8
    if nb >= 96:
        group_n = nb // 8
    elif K >= 28672:
        group_n = 2
        num_xcds = 16
    elif K >= 11008:
        group_n = 4
    else:
        group_n = 0
    group_m = 4
    return group_m, group_n, num_xcds


def _autotune_mxfp4_config(M, N, K, args):
    """Pick (group_m, group_n, num_xcds) for this (M, N, K) by a quick timed sweep
    over ``_MXFP4_AUTOTUNE_CANDIDATES`` on the real operands; cached per shape.

    The swizzle only remaps which workgroup computes which output tile, so every
    candidate is bit-identical -- we are purely chasing L2 residency / tail balance.
    Skipped (falls back to the static heuristic) during CUDA-graph capture (cannot
    time inside capture). Compiled winners are stashed in _MXFP4_AT_CACHE so the
    subsequent real launch reuses them with no recompile."""
    key = (M, N, K)
    cached = _MXFP4_CFG_CACHE.get(key)
    if cached is not None:
        return cached

    if torch.cuda.is_current_stream_capturing():
        cfg = (
            *_mxfp4_nt_config(M, N, K),
            10,
            9,
            False,  # taccw: off outside the timed autotune
            False,  # coop
        )
        _MXFP4_CFG_CACHE[key] = cfg
        return cfg

    # Compile every candidate up front, then warm ALL of them so each is timed against
    # the same (warm) L2 state -- otherwise the first candidate eats the cold-cache cost
    # and later ones look artificially faster (ordering bias picked a poor config).
    # In-flight memory depth axis (phase-barrier vmcnt/lgkmcnt). A deeper pipeline
    # (16/15) hides more steady-state g2s latency and wins on high-trip-K shapes, but
    # adds prologue/ramp cost that hurts low/mid-K -> only OFFER it when K is large; the
    # autotune still takes the global min so it never regresses a shape (low/mid-K keep
    # 10/9). ELGK=15 is the lgkmcnt 4-bit HW max.
    _try_deepwl = K >= 8192
    _wl_opts = ((10, 9), (16, 15)) if _try_deepwl else ((10, 9),)
    compiled_cands = []
    for _wlv, _elgk in _wl_opts:
        for gm, gn, xcd in _MXFP4_AUTOTUNE_CANDIDATES:
            try:
                at_key = (M, N, K, gm, xcd, gn, _wlv, _elgk, False, False)
                entry = _MXFP4_AT_CACHE.get(at_key)
                if entry is None:
                    raw = _get_mxfp4_fused_launch(K, gm, xcd, gn, _wlv, _elgk, coop=False)
                    entry = [raw, flyc.compile(raw, *args)]
                    _MXFP4_AT_CACHE[at_key] = entry
                compiled_cands.append(((gm, gn, xcd, _wlv, _elgk), entry[1]))
            except Exception:  # noqa: BLE001 -- a bad config must not break the GEMM
                continue
    for _ in range(5):
        for _, compiled in compiled_cands:
            compiled(*args)
    torch.cuda.synchronize()

    # Robust timing: min over REPS independent measurements (each ITERS launches).
    # Candidates are timed ROUND-ROBIN (all cfgs once per rep, not all reps of one
    # cfg back-to-back): the top swizzles often sit within a few % of each other, and
    # measuring them sequentially lets slow GPU-clock drift across the ~ms sweep bias
    # whichever cfg happened to be timed in a hotter/cooler window. Interleaving keeps
    # every candidate's reps spread across the same clock states, so the per-cfg min is
    # an apples-to-apples floor -- the previous sequential layout mis-ranked low-K
    # shapes where near-tied swizzles are separated only by clock drift.
    ITERS, REPS = 20, 8
    cand_t = {cfg: float("inf") for cfg, _ in compiled_cands}
    for _ in range(REPS):
        for cfg, compiled in compiled_cands:
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(ITERS):
                compiled(*args)
            e1.record()
            torch.cuda.synchronize()
            cand_t[cfg] = min(cand_t[cfg], e0.elapsed_time(e1))
    # Noise-robust selection for the in-flight-depth (wlv/elgk) axis: its win is small
    # (~1%) and near this system's run-to-run timing noise (~1-3%), so a raw global-min
    # can flip to a deep-pipeline cfg that only *looks* faster in a hot sample and then
    # REGRESS the shape in steady state. We handicap every non-default (wlv,elgk) cfg by
    # _WL_MARGIN, so the deep pipeline is only chosen when it beats the safe 10/9 default
    # by more than the noise floor -- the axis can then only help, never regress. (The
    # swizzle axis is a bit-identical schedule with larger, robust gaps, so it keeps the
    # plain min.)
    _WL_MARGIN = 1.02
    best, best_t = None, float("inf")
    for cfg, t in cand_t.items():
        _teff = t * _WL_MARGIN if cfg[3:5] != (10, 9) else t
        if _teff < best_t:
            best_t, best = _teff, cfg
    if best is None:
        best = (*_mxfp4_nt_config(M, N, K), 10, 9)
    # ── Epilogue/scale variant axis: (COOP scale-load, TACCW wide store) ──
    # Two correctness-preserving MECHANISM swaps, timed together as a never-regress
    # axis on top of the swizzle/wl winner:
    #   COOP  -- the 4 waves cooperatively load the 4 UNIQUE scale groups ONCE into
    #            LDS instead of every wave re-loading them (~4x less scale HBM); wins
    #            when scale traffic is exposed (low/mid-K, fat-N).
    #   TACCW -- acc=Cᵀ + AITER permlane16_swap dwordx4 WIDE epilogue store; wins on
    #            epilogue-exposed fat/low-K shapes but adds permlane VALU on high-K.
    # Each helps some shapes and is ~noise on others (and they STACK on most Llama
    # shapes), so we compile the {TACCW, COOP, TACCW+COOP} twins of the winner, time
    # them round-robin vs the re-measured plain winner in one thermal window, and keep
    # the fastest that beats the plain winner by > the noise floor -> the axis can only
    # help, never regress.
    best = (*best, False, False)  # append (taccw, coop) = (False, False)
    _try_var = best_t < float("inf")
    if _try_var:
        gm0, gn0, xcd0, w0, e0 = best[:5]
        try:
            df_compiled = _MXFP4_AT_CACHE[(M, N, K, gm0, xcd0, gn0, w0, e0, False, False)][1]
            variants = []  # (taccw, coop, compiled)
            for _cp, _tw in ((False, True), (True, False), (True, True)):
                vkey = (M, N, K, gm0, xcd0, gn0, w0, e0, _tw, _cp)
                ventry = _MXFP4_AT_CACHE.get(vkey)
                if ventry is None:
                    vraw = _get_mxfp4_fused_launch(K, gm0, xcd0, gn0, w0, e0, taccw=_tw, coop=_cp)
                    ventry = [vraw, flyc.compile(vraw, *args)]
                    _MXFP4_AT_CACHE[vkey] = ventry
                variants.append((_tw, _cp, ventry[1]))
            for _ in range(5):  # warm every twin + the winner into the same L2/clock state
                df_compiled(*args)
                for _, _, _vc in variants:
                    _vc(*args)
            torch.cuda.synchronize()

            def _time(fn):
                _q0 = torch.cuda.Event(enable_timing=True)
                _q1 = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                _q0.record()
                for _ in range(ITERS):
                    fn(*args)
                _q1.record()
                torch.cuda.synchronize()
                return _q0.elapsed_time(_q1)

            df_t = float("inf")
            vt = [float("inf")] * len(variants)
            _VREPS = 16  # extra reps: the twin wins are small, so we need a tight min
            for _ in range(_VREPS):  # round-robin so every cand shares the same thermal window
                df_t = min(df_t, _time(df_compiled))
                for _i, (_, _, _vc) in enumerate(variants):
                    vt[_i] = min(vt[_i], _time(_vc))
            # Keep the fastest twin, but only if it beats the plain winner by > the noise
            # floor; twins then compete on raw time among themselves. Unlike the wl-depth
            # axis (a pipeline-depth change that can look fast in a hot sample yet REGRESS
            # in steady state -> needs a wide 1% guard), COOP/TACCW are stable mechanism
            # swaps whose small wins are real and STACK across shapes, so a tight 0.5%
            # guard is safe (worst case: pick a noise-equal twin -> no regression).
            _bi, _bt = -1, df_t * 0.995
            for _i in range(len(variants)):
                if vt[_i] < _bt:
                    _bt, _bi = vt[_i], _i
            if _bi >= 0:
                _tw, _cp, _ = variants[_bi]
                best = (gm0, gn0, xcd0, w0, e0, _tw, _cp)
        except Exception:  # noqa: BLE001 -- a bad twin must not break the GEMM
            pass
    _MXFP4_CFG_CACHE[key] = best
    return best


_MXFP4_KSPLIT_CACHE: dict = {}  # (M, N, K) -> chosen ksplit (timed, never regresses vs 1)


def _ksplit_candidates(M, N, K):
    """Split-K candidates for the timed ksplit autotune. Only FEW-TILE large-K shapes
    (the one-WG-per-tile grid leaves CUs idle) are worth splitting; the sweep always
    includes ksplit=1 and takes the global min, so a bad split can never regress a shape.
    ksplit must divide K//256 (whole 256-K blocks per split) and K//ksplit % 256 == 0."""
    tiles = ceildiv(M, 256) * ceildiv(N, 256)
    ncu = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    if tiles >= ncu // 2 or K < 2048:
        return [1]  # already enough WGs to fill the CUs (or K too small to split)
    kb = K // 256
    cands = [1]
    for s in (2, 3, 4, 6, 8, 12, 16):
        if kb % s == 0 and s <= kb and tiles * s <= ncu * 2:
            cands.append(s)
    return cands


def _compile_mxfp4_fused(K, gm, xcd, gn, wlv=10, elgk=9, ksplit=1, taccw=False, coop=False, out_fp16=False):
    """Turbo/mxfp8-style fused @flyc.jit stub: ONE host dispatch enqueues the A scale
    preshuffle, the B scale preshuffle, then the NT GEMM on the same stream (no separate
    preshuffle launch, no CPU sync). The preshuffle kernels repack raw E8M0 (int32-viewed)
    into the caller-owned a_sp/b_sp packed-int32 workspace; the GEMM reads it in stream
    order. ksplit>1 writes K/ksplit partials into a [ksplit*M, N] workspace C (the host sums
    the row bands outside the stub)."""
    K128 = K // 128
    pre_a = _build_mxfp4_preshuffle_kernel(0)
    pre_b = _build_mxfp4_preshuffle_kernel(1)
    gemm_kern, BM, BN, _ks, gemm_value_attrs = _build_mxfp4_gemm_kernel(
        K=K,
        group_m=gm,
        num_xcds=xcd,
        group_n=gn,
        wlv=wlv,
        elgk=elgk,
        coop=coop,
        ksplit=ksplit,
        taccw=taccw,
        out_fp16=out_fp16,
    )
    _PGRID = _MXFP4_PRESHUF_NG * _MXFP4_PRESHUF_BLK  # threads-per-block * fan-out

    @flyc.jit
    def launch_mxfp4_fused(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_raw: fx.Tensor,
        B_raw: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        # 1) A + 2) B scale preshuffle (raw E8M0 -> packed int32 in A_scale/B_scale ws).
        # grid = ceildiv(dim*K128 / NG, BLK); dim*K128 is divisible by NG so this equals
        # ceildiv(dim*K128, NG*BLK).
        pre_a(A_raw, A_scale, c_m, fx.Int32(K128)).launch(
            grid=(ceildiv(c_m * fx.Int32(K128), _PGRID), 1, 1),
            block=(_MXFP4_PRESHUF_BLK, 1, 1),
            stream=stream,
        )
        pre_b(B_raw, B_scale, c_n, fx.Int32(K128)).launch(
            grid=(ceildiv(c_n * fx.Int32(K128), _PGRID), 1, 1),
            block=(_MXFP4_PRESHUF_BLK, 1, 1),
            stream=stream,
        )
        # 3) NT GEMM (reads the just-written A_scale/B_scale ws; same stream => ordered).
        grid_x = ceildiv(c_m, BM) * ceildiv(c_n, BN)
        if const_expr(ksplit > 1):
            grid_x = grid_x * fx.Int32(ksplit)  # split-K: one WG per (tile, split)
        gemm_kern(A, B_T, C, A_scale, B_scale, c_m, c_n, value_attrs=gemm_value_attrs).launch(
            grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch_mxfp4_fused


def _get_mxfp4_fused_launch(
    K, gm, xcd, gn, wlv=10, elgk=9, ksplit=1, taccw=False, coop=False, out_fp16=False
):
    # wlv/elgk pick the phase-barrier in-flight memory depth (autotuned per shape).
    # ksplit>1 compiles a K/ksplit slice kernel for few-tile large-K shapes.
    # coop/taccw are the never-regress scale-load / wide-store autotune axes.
    # out_fp16 selects the store cast dtype (fp16 vs bf16); fp16 implies taccw=False.
    lk = (K, gm, xcd, gn, wlv, elgk, coop, ksplit, taccw, out_fp16)
    launch = _MXFP4_LAUNCH_CACHE.get(lk)
    if launch is None:
        launch = _compile_mxfp4_fused(
            K, gm, xcd, gn, wlv=wlv, elgk=elgk, ksplit=ksplit, taccw=taccw, coop=coop, out_fp16=out_fp16
        )
        _MXFP4_LAUNCH_CACHE[lk] = launch
    return launch


# ── Scale preshuffle (separate FlyDSL kernel; mirrors the mxfp8 gemm's decoupling) ──
# The quant emits plain canonical E8M0 scales [DIM, K/32]; this kernel repacks them into
# the lane-contiguous packed int32 layout the GEMM's ScaleS2RPacked consumes, run once on
# the same stream right before the GEMM (so quant stays generic -- no fused scale write).
#
# Gather form: one thread per output int32 dword. The packed layout is
#   dword = ((wi*KK + kk)*64 + lane)*nd + last,  byte t of that dword = row (row/16)%4 == t
# with n_sub=2, nd=4, KK=K/256, lane=g_byte*16 + r (g_byte=kb%4, r=row%16), k128=kk*2+s,
# last=r_region*2 + s. Decoding a dword index -> (wi, kk, lane, last) and inverting the A/B
# group map gives (grp, r, kb) shared by the dword's 4 source rows grp*64 + t*16 + r; each
# source E8M0 byte is read from the int32-viewed raw scale [DIM, K128] and packed by byte.
# This is the exact forward map of the deleted C++ compute_preshuffle_scale_index_mxfp4.

_MXFP4_PRESHUF_BLK = 256
_MXFP4_PRESHUF_NG = 4  # g_byte fan-out folded into one preshuffle thread (grid is 1/NG the dwords)
_MXFP4_SCALE_WS: dict = {}  # (M, N, K, device) -> (a_sp, b_sp) packed int32 workspace


def _mxfp4_grp_from(wi, r_region, mode):
    # Inverse of compute_preshuffle_scale_index_mxfp4's group map. Plain Python helper
    # (NOT inside the @flyc.kernel body) so the mode branch is a trace-time Python if.
    if mode == 0:  # A: grp = 2*wi + r_region
        return 2 * wi + r_region
    # B: stride-2 groups with block interleave (g0 = 4*(wi//2)+(wi%2); grp = g0 + 2*r_region)
    return 4 * (wi // 2) + (wi % 2) + 2 * r_region


def _build_mxfp4_preshuffle_kernel(mode):
    # mode: 0 = A operand, 1 = B operand (matches the GEMM's ScaleS2RPacked A/B layout).
    # Returns the BARE @flyc.kernel issued by the fused GEMM stub (_compile_mxfp4_fused).
    # Each thread emits all NG=4 g_byte output
    # dwords for one (wi, kk, r, last), loading the NG-shared 4 source int32 ONCE instead of
    # NG times (a naive one-dword-per-thread kernel has 4x cross-thread read amplification:
    # the four g_byte threads reload the same 4 rows). Grid is 1/NG the size; HBM read ~4x lower.
    n_sub = 2
    nd = 4
    NG = _MXFP4_PRESHUF_NG

    @flyc.kernel(known_block_size=[_MXFP4_PRESHUF_BLK, 1, 1])
    def kern(raw: fx.Tensor, out: fx.Tensor, dim: fx.Int32, K128: fx.Int32):
        KK = K128 // n_sub  # K/256
        total = dim * K128  # output int32 dwords
        total4 = total // NG  # threads (one per g_byte group)
        gid4 = fx.block_idx.x * _MXFP4_PRESHUF_BLK + fx.thread_idx.x
        rin = buffer_ops.create_buffer_resource(raw, max_size=False, num_records_bytes=dim * K128 * 4)
        rout = buffer_ops.create_buffer_resource(out, max_size=False, num_records_bytes=dim * K128 * 4)

        last = gid4 % nd
        e1 = gid4 // nd
        r = e1 % 16
        e2 = e1 // 16
        kk = e2 % KK
        wi = e2 // KK

        r_region = last // n_sub
        s = last % n_sub
        k128 = kk * n_sub + s
        grp = _mxfp4_grp_from(wi, r_region, mode)
        # base output dword (g_byte=0); the g-th g_byte lands at base + g*64
        base = ((wi * KK + kk) * 64 + r) * nd + last

        dws = [fx.Int32(0)] * nd
        for t in range_constexpr(nd):
            row = grp * 64 + t * 16 + r
            dws[t] = fx.Int32(
                buffer_ops.buffer_load(
                    rin, row * K128 + k128, vec_width=1, dtype=T.i32, mask=(gid4 < total4) & (row < dim)
                )
            )
        for g in range_constexpr(NG):
            sh = fx.Int32(g * 8)
            packed = fx.Int32(0)
            for t in range_constexpr(nd):
                b = (dws[t] >> sh) & fx.Int32(0xFF)
                packed = packed | (b << fx.Int32(t * 8))
            buffer_ops.buffer_store(packed, rout, base + g * 64, mask=gid4 < total4)

    return kern


def _get_mxfp4_scale_ws(M, N, K, device):
    """Caller-owned packed-scale workspace (a_sp/b_sp), cached per (M, N, K, device). Sized
    to the ScaleS2RPacked extent (dim * K/128 int32); the preshuffle writes it and the GEMM
    reads it in stream order, so same-shape reuse on one stream is safe."""
    K128 = K // 128
    key = (M, N, K, device)
    e = _MXFP4_SCALE_WS.get(key)
    if e is None:
        a_sp = torch.empty(M * K128, dtype=torch.int32, device=device)
        b_sp = torch.empty(N * K128, dtype=torch.int32, device=device)
        _MXFP4_SCALE_WS[key] = e = (a_sp, b_sp)
    return e


def gemm_mxfp4_flydsl_kernel(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
    trans_c: bool = False,
) -> torch.Tensor:
    """MXFP4 (per-32-K E8M0 block-scaled) dense GEMM, gfx950 (4-wave whole-loop).

    NT only (trans_a=False, trans_b=True): A [M, K] fp4, B [N, K] fp4, C = a @ b^T.
    ``a``/``b`` are fp4 (float4_e2m1fn_x2, 2 values/byte) so the K byte-stride is K//2.

    ``a_scale``/``b_scale`` are CANONICAL E8M0 block scales ([M, K/32] / [N, K/32],
    row-major, as emitted by the generic quant). This wrapper repacks them into the
    lane-contiguous packed int32 layout the whole-loop reads (VGPR-direct scale path)
    via a separate FlyDSL preshuffle kernel launched on the same stream right before
    the GEMM -- so the quant stays generic (no fused scale write). Mirrors the mxfp8
    GEMM's quant/preshuffle decoupling; ``a`` is always the A operand, ``b`` the B.

    Constraints: K % 256 == 0; M % 256 == 0; N % 256 == 0.

    The whole-loop bare-asm body is an unroll-2 ping-pong that processes K in PAIRS
    of BLOCK_K=256. An odd number of 256-K blocks (K % 512 == 256) is handled by a
    single MFMA-only phase-A tail emitted after the hardware loop (see
    ``call_mxfp4_wholeloop``'s ``ki``/tail path); the loop runs ``KI//2`` full pairs
    and the trailing block is accumulated by the tail. K=256 (KI==1) omits the loop
    entirely and runs only the tail. No host-side K padding is required.
    """
    assert a.dim() == 2 and b.dim() == 2, "a, b must be 2D"
    assert out_dtype in (torch.bfloat16, torch.float16), "mxfp4 FlyDSL store emits bf16/fp16"
    out_fp16 = out_dtype == torch.float16
    if not ((not trans_a) and trans_b):
        raise NotImplementedError(
            "mxfp4 FlyDSL GEMM is NT only (trans_a=False, trans_b=True); "
            f"got trans_a={trans_a}, trans_b={trans_b}."
        )

    M, Kb_a = a.shape
    N, Kb_b = b.shape
    K = Kb_a * 2  # packed 2 fp4 / byte
    assert Kb_a == Kb_b, f"K mismatch: a {a.shape}, b {b.shape}"
    # K % 256: the unroll-2 whole-loop runs KI//2 pairs + an MFMA-only tail for the
    # odd trailing 256-block (see the docstring / call_mxfp4_wholeloop).
    assert K % 256 == 0, f"K must be a multiple of 256, got {K}"
    assert M % 256 == 0, f"M must be a multiple of 256, got {M}"
    assert N % 256 == 0, f"N must be a multiple of 256, got {N}"

    stream = torch.cuda.current_stream()
    # Fused (turbo/mxfp8-style) path: a single @flyc.jit stub enqueues the A + B scale
    # preshuffle kernels and then the GEMM on this stream -- one host dispatch, no separate
    # preshuffle launch / CPU sync. The preshuffle repacks the canonical E8M0 scales into
    # the caller-owned packed-int32 workspace (a_sp/b_sp); the quant stays generic. The
    # workspace is cached per shape (stable across CUDA-graph replays); the timed autotune
    # launches include the (fixed) preshuffle, so the gemm-config ranking is preserved.
    _capturing = torch.cuda.is_current_stream_capturing()
    a_sp, b_sp = _get_mxfp4_scale_ws(M, N, K, a.device)
    a_raw = a_scale.contiguous().view(torch.int32).reshape(-1)
    b_raw = b_scale.contiguous().view(torch.int32).reshape(-1)
    out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    # FLAT 1D int8 views: the prologue operand G2S uses the NON-rebased FlyDSL
    # ``G2SLoader`` (flat byte offsets via ``fx.slice``), so the source tensor MUST be
    # a 1D layout; a 2D [M, K//2] layout makes the k=0/1 prologue loads address wrong
    # (the in-loop asm refill is SRD+byte-offset and shape-agnostic). C is flat too
    # (StoreCPlain re-bases per row band from C's base index + the passed c_n).
    a8 = a.contiguous().view(torch.int8).reshape(-1)
    b8 = b.contiguous().view(torch.int8).reshape(-1)
    out_flat = out.view(-1)

    # Fused stub args: (A, B_T, C, A_raw, B_raw, A_scale_ws, B_scale_ws, c_m, c_n, stream).
    fused_args = (a8, b8, out_flat, a_raw, b_raw, a_sp, b_sp, M, N, stream)

    def _exec_plain():
        # default one-WG-per-tile path (autotuned swizzle / pipe depth / scale-load / wide store).
        gm, gn, xcd, _wlv, _elgk, _taccw, _coop = _autotune_mxfp4_config(M, N, K, fused_args)
        # fp16 has no wide (TACCW) store path -> force the narrow scalar store.
        _tw = _taccw and not out_fp16
        launch = _get_mxfp4_fused_launch(
            K, gm, xcd, gn, _wlv, _elgk, taccw=_tw, coop=_coop, out_fp16=out_fp16
        )
        at_key = (M, N, K, gm, xcd, gn, _wlv, _elgk, _tw, _coop, out_fp16)
        entry = _MXFP4_AT_CACHE.get(at_key)
        if entry is None:
            entry = [launch, None]
            _MXFP4_AT_CACHE[at_key] = entry
        raw, compiled = entry
        if _capturing:
            raw(*fused_args)
        else:
            if compiled is None:
                compiled = flyc.compile(raw, *fused_args)
                entry[1] = compiled
            compiled(*fused_args)
        return out

    def _exec_split(ksplit):
        # split-K: grid x ksplit fills the CUs on few-tile large-K shapes. Each split writes
        # its K/ksplit partial into a bf16 workspace[ksplit*M, N]; host sums the ksplit row
        # bands (BW-bound bf16 reduce, faster than an atomic-fused reduce for these shapes).
        # The fused stub still preshuffles A/B into a_sp/b_sp before the split GEMM.
        gm, gn, xcd = _mxfp4_nt_config(M, N, K)
        ws = torch.empty((ksplit * M, N), dtype=out_dtype, device=a.device)
        cbuf = ws.view(-1)
        sk_args = (a8, b8, cbuf, a_raw, b_raw, a_sp, b_sp, M, N, stream)
        launch = _get_mxfp4_fused_launch(K, gm, xcd, gn, 10, 9, ksplit=ksplit, out_fp16=out_fp16)
        sk_key = (M, N, K, gm, xcd, gn, 10, 9, ksplit, out_fp16)
        entry = _MXFP4_AT_CACHE.get(sk_key)
        if entry is None:
            entry = [launch, None]
            _MXFP4_AT_CACHE[sk_key] = entry
        raw, compiled = entry
        if _capturing:
            raw(*sk_args)
        else:
            if compiled is None:
                compiled = flyc.compile(raw, *sk_args)
                entry[1] = compiled
            compiled(*sk_args)
        return ws.view(ksplit, M, N).sum(dim=0)

    # ── Choose ksplit: cached timed pick > timed autotune.
    # The autotune times {plain, split+reduce} end-to-end on the real operands and takes the
    # global min, so split-K is used ONLY where it actually wins and never regresses a shape.
    # Skipped during graph capture (uses the cached pick).
    ks = _MXFP4_KSPLIT_CACHE.get((M, N, K))
    if ks is None:
        cands = _ksplit_candidates(M, N, K)
        if _capturing or len(cands) == 1:
            ks = 1  # cannot time inside capture / nothing to try
            if not _capturing:
                _MXFP4_KSPLIT_CACHE[(M, N, K)] = ks
        else:

            def _bench(fn):
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                best = float("inf")
                for _ in range(5):
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    for _ in range(20):
                        fn()
                    e1.record()
                    torch.cuda.synchronize()
                    best = min(best, e0.elapsed_time(e1))
                return best

            times = {}
            for s in cands:
                try:
                    fn = _exec_plain if s == 1 else (lambda s=s: _exec_split(s))
                    times[s] = _bench(fn)
                except Exception:  # noqa: BLE001 -- a bad variant must not break the GEMM
                    continue
            ks = min(times, key=times.get) if times else 1
            _MXFP4_KSPLIT_CACHE[(M, N, K)] = ks

    out2 = _exec_split(ks) if ks > 1 else _exec_plain()
    return out2.t().contiguous() if trans_c else out2
