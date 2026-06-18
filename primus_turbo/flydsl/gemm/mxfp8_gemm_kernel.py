###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""8-wave MXFP8 matmul (per-1x32 E8M0 block scaling) for AMD CDNA4 (gfx950).

Derived from ``kernels/fp8_gemm_8wave.py`` (tensorwise FP8). The structural
difference vs the tensorwise kernel:

  * tensorwise applies a single per-row (A) / per-col (B) FP32 scale in the
    epilogue, with the MFMA run un-scaled (identity scale operand).
  * mxfp8 carries a per-32-element-K-block E8M0 scale that MUST be fed to the
    ``v_mfma_scale_f32_16x16x128_f8f6f4`` instruction per K-iteration. The
    epilogue therefore becomes a plain FP32->BF16 store (all scaling already
    folded into the accumulator by the MMA).

Scale operand semantics (gfx950): the MMA takes one i32 scale per operand,
holding 4 packed E8M0 bytes -- one byte per 32-K block. A single
16x16x128 MFMA spans K=128 == 4 micro-blocks, so exactly one i32 scale per
(row/col tile, K-iteration).

Scale tensor layout expected by this kernel (passed pre-packed from host):
  A_scale: int32 [M, K // 128]   (each i32 == 4 consecutive E8M0 bytes of a row)
  B_scale: int32 [N, K // 128]
i.e. the raw uint8 E8M0 [DIM, K//32] viewed little-endian as int32.
"""

import torch

# isort: off
# Shared fp8 GEMM primitives from primus_turbo/flydsl/utils/gemm_helper.py (the
# tensorwise FlyDSL backend, #356). Must be importable as module globals
# (@flyc.kernel needs them as globals, not closure cells). NT only (compute-only
# PR), so the NN/TN transpose loaders (S2RLoaderTr / swizzle_nn) are not imported.
from primus_turbo.flydsl.utils.gemm_helper import (
    G2SLoader,
    S2RLoader,
    as_i8_flat,
    block_mn,
    ceildiv,
    compute_global_swizzle,
    get_compiled,
    make_fp8_buffer_tensor,
    make_row_band_resource,
    make_value_attrs,
    wait_barrier,
    xcd_remap_pid,
)

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

# isort: on


def preshuffle_scale(e8m0_u8, K, n_tiles):
    """Host-side E8M0 scale pre-shuffle for the mxfp8 8-wave kernel (broadcast layout,
    scale_pack=1). Pairs with ``ScaleS2R``.

    ``n_tiles`` = the per-wave sub-tile fan-out the kernel loads together in one
    vectorized dword{n_tiles} (A: BLOCK_M//64).

    Input : uint8 [DIM, K//32] row-major E8M0 (DIM multiple of 16*n_tiles).
    Output: int32 [DIM//(16*n_tiles), K//128, 64, n_tiles] where each i32 broadcasts
        one E8M0 byte to all 4 byte-positions (opsel==0 reads byte 0; broadcast =>
        byte-position safe). This is the no-host-repack-needed compute path; the
        byte-packed (scale_pack>1) layout is emitted by the quant and lands in the
        quant PR.
    """
    DIM, Kb = e8m0_u8.shape
    assert Kb == K // 32 and K % 128 == 0
    assert DIM % (16 * n_tiles) == 0, f"DIM={DIM} must be multiple of {16 * n_tiles}"
    K128 = K // 128
    G = DIM // (16 * n_tiles)
    s = e8m0_u8.reshape(DIM, K128, 4)  # [DIM, k, g]
    s = s.reshape(G, n_tiles, 16, K128, 4)  # [grp, s, r, k, g]
    s = s.permute(0, 3, 4, 2, 1).contiguous()  # [grp, k, g, r, s]
    s = s.reshape(G, K128, 64, n_tiles).to(torch.int32)  # lane == g*16 + r
    sp = s | (s << 8) | (s << 16) | (s << 24)
    return sp.contiguous()


def preshuffle_scale_b_comb(e8m0_u8, K):
    """Combined-B E8M0 pre-shuffle (broadcast layout, scale_pack=1). Pairs with
    ``ScaleBComb``. Packs BOTH N-regions' (b0,b1) 4 sub-tiles for a wave into one
    dword{4}, so the kernel issues a single dwordx4 for all B scales per K-iter.

    Handles N % 64 == 0 (not just N % 256): the input is zero-padded up to the next
    256-multiple so the buffer holds cdiv(N,256)*4 groups (matches ``ScaleBComb``'s
    num_records sizing). Columns >= N are dropped by the StoreC col clamp and their
    data reads clamp to 0, so the zero-padded (finite, 2^-127) scale is harmless.

    A wave's 4 B sub-tiles sit at cols c+{0,16,128,144}, c = block_n*256 + wave_n*32.
    Output int32 [cdiv(N,256)*4, K//128, 64, 4]:
        SP[grp, k, lane, s] = broadcast( scale[c + OFF[s] + lane%16, 4k + lane//16] )
    grp = block_n*4 + wave_n;  OFF = [0,16,128,144].
    """
    N, Kb = e8m0_u8.shape
    assert Kb == K // 32 and K % 128 == 0 and N % 64 == 0
    K128 = K // 128
    Npad = ((N + 255) // 256) * 256
    if Npad != N:
        pad = torch.zeros((Npad - N, Kb), dtype=e8m0_u8.dtype, device=e8m0_u8.device)
        e8m0_u8 = torch.cat([e8m0_u8, pad], dim=0)
    OFF = [0, 16, 128, 144]
    dev = e8m0_u8.device
    s = e8m0_u8.reshape(Npad // 256, 256, K128, 4)  # [nblk, col256, k, g]
    # col256 = wn*32 + OFF[si] + r  (bijection of 0..255)
    wn = torch.arange(4, device=dev).view(4, 1, 1)
    r = torch.arange(16, device=dev).view(1, 1, 16)
    off = torch.tensor(OFF, device=dev).view(1, 4, 1)
    colidx = (wn * 32 + off + r).reshape(-1)  # [wn,si,r] flattened
    g = s[:, colidx, :, :].reshape(Npad // 256, 4, 4, 16, K128, 4)  # [nblk, wn, si, r, k, g]
    g = g.permute(0, 1, 4, 5, 3, 2).contiguous()  # [nblk, wn, k, g, r, si]
    g = g.reshape(Npad // 64, K128, 64, 4).to(torch.int32)  # grp=nblk*4+wn, lane=g*16+r
    sp = g | (g << 8) | (g << 16) | (g << 24)
    return sp.contiguous()


def _asm_mma_scale_do(a, b, c, sa, sb, opsel):
    """Inline-asm scaled MFMA v_mfma_scale_f32_16x16x128_f8f6f4. =&v early-clobber
    forces dst disjoint from srcA/srcB; opaque to the backend so it co-schedules with
    the asm ds_read_b64_tr_b8 loads. opsel (0..3) picks the packed E8M0 byte via
    op_sel (low bit) / op_sel_hi (high bit)."""
    v4f32 = ir.VectorType.get([4], ir.F32Type.get())
    lo = opsel & 1
    hi = (opsel >> 1) & 1
    osel = f"op_sel:[{lo},{lo},0] op_sel_hi:[{hi},{hi},0]"
    cons = "=&v,v,v,0,v,v"  # VGPR early-clobber accumulator
    op = _llvm.InlineAsmOp(
        res=v4f32,
        operands_=[_raw(a), _raw(b), _raw(c), _raw(sa), _raw(sb)],
        asm_string=f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $0, $4, $5 {osel}",
        constraints=cons,
        has_side_effects=False,
    )
    return Vec(op.result)


class ScaleBComb:
    """Combined B scale loader (pairs with ``preshuffle_scale_b_comb``).

    One dwordx4 per lane returns [s0,s1,s2,s3]; (s0,s1)=b0 sub-tiles, (s2,s3)=b1.
    """

    def __init__(self, sp_tensor, dim, K, pack=1):
        self.K128 = K // (128 * pack)  # number of K-groups (pack K-iters per i32)
        self.lane = fx.thread_idx.x % 64
        # grp = (col//256)*4 + wn is block-strided, so the buffer holds cdiv(dim,256)*4
        # groups (matches the C++ preshuffle B sizing). A partial last 256-block reads
        # only its valid wn groups; OOB-col reads clamp to 0 and StoreC drops them.
        # dim%256==0 -> cdiv(dim,256)*4 == dim//64 (no change for aligned shapes).
        nbytes = ((dim + 255) // 256) * 4 * self.K128 * 64 * 4 * 4  # int32 records
        self.rsrc = buffer_ops.create_buffer_resource(sp_tensor, max_size=False, num_records_bytes=nbytes)

    def load(self, base, k):
        """base: sb_base0 (b0 region col base). Returns 4 i32 (b0:0,1  b1:2,3)."""
        grp = (base // 256) * 4 + (base % 256) // 32
        idx = ((grp * self.K128 + k) * 64 + self.lane) * 4
        v = Vec(buffer_ops.buffer_load(self.rsrc, idx, vec_width=4, dtype=T.i32))
        return [v[i].ir_value() for i in range_constexpr(4)]


class MfmaScale16x16x128:
    """16x16x128 f8f6f4 MFMA with per-block E8M0 scale operands.

    Mirrors ``Mfma16x16x128`` but routes through the raw rocdl intrinsic so
    the (scale_a, scale_b) i32 operands can be supplied per call.
    """

    def __init__(self, n_tiles_a, n_tiles_b, asm_mma=False):
        self.res_ty = Vec.make_type(4, fx.Float32)
        self.zero_value = Vec.filled(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        # asm_mma routes through the inline-asm scaled MFMA (see _asm_mma_scale_do).
        # opsel picks which of the i32 scale operand's 4 E8M0 bytes the MMA reads
        # (default 0); see preshuffle_scale / preshuffle_scale_pack for the layout.
        self.opsel = 0
        self.asm_mma = asm_mma

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _do_mma(self, a, b, c, sa, sb):
        # operand order: a, b, c, cbsz, blgp, opsel_a, scale_a, opsel_b, scale_b
        if self.asm_mma:  # inline-asm scaled MFMA (co-schedules with asm tr8 loads)
            return _asm_mma_scale_do(a, b, c, sa, sb, self.opsel)
        return rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            self.res_ty,
            [a, b, c, 0, 0, self.opsel, sa, self.opsel, sb],
        )

    def call(self, a, b, c, sa, sb):
        assert len(a) == self.n_tiles_a
        assert len(b) == self.n_tiles_b
        assert len(c) == self.n_tiles_a * self.n_tiles_b
        assert len(sa) == self.n_tiles_a
        assert len(sb) == self.n_tiles_b

        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                c[self.idx(i, j)] = self._do_mma(a[i], b[j], c[self.idx(i, j)], sa[i], sb[j])
        return c


class ScaleS2R:
    """Per-lane E8M0 scale loader for v_mfma_scale_f32_16x16x128 (preshuffled).

    The 16x16x128 MFMA distributes K=128 so lane ``(g, r)`` with
    ``g = lane//16`` (0..3) and ``r = lane%16`` holds the A/B data for matrix
    row/col ``r`` and the 32-K micro-block ``g``. With opsel==0 the hardware
    samples byte 0 of each lane's scale operand, so lane ``(g, r)`` just needs
    ``scale[r, 4k+g]`` in a register.

    To make that a single fully-coalesced dword load with no per-lane ALU, the
    host pre-shuffles the raw E8M0 [DIM, K//32] into

        SP[rt, k, lane] = broadcast_u8_to_u32( scale[rt*16 + lane%16, 4k + lane//16] )

    laid out int32 [DIM//16, K//128, 64]. For row-tile ``rt`` and K-iter ``k``
    the 64 lanes of a wave read 64 contiguous dwords. See ``preshuffle_scale``.
    """

    def __init__(self, sp_tensor, dim, K, n_tiles, pack=1):
        self.K128 = K // (128 * pack)  # number of K-groups (pack K-iters per i32)
        self.n_tiles = n_tiles
        self.group_span = 16 * n_tiles
        self.lane = fx.thread_idx.x % 64  # == (lane//16)*16 + lane%16
        nbytes = (dim // self.group_span) * self.K128 * 64 * n_tiles * 4  # int32 records
        self.rsrc = buffer_ops.create_buffer_resource(sp_tensor, max_size=False, num_records_bytes=nbytes)

    def load(self, base, k):
        """base: runtime global row/col base for this (region, wave). Returns n_tiles i32.

        One vectorized dword{n_tiles} load: the n_tiles sub-tile scales for this
        wave at (group, k) are contiguous per lane (see ``preshuffle_scale``).
        """
        grp = base // self.group_span
        idx = ((grp * self.K128 + k) * 64 + self.lane) * self.n_tiles
        v = Vec(buffer_ops.buffer_load(self.rsrc, idx, vec_width=self.n_tiles, dtype=T.i32))
        return [v[i].ir_value() for i in range_constexpr(self.n_tiles)]


class StoreCPlain:
    """Plain FP32 accumulator -> BF16 store (no scaling; scales folded in MMA).

    int64-safe: the output buffer is re-based per workgroup at its row band
    (byte base = base_row * c_cols * 2 computed in the 64-bit ``index`` type),
    so a single 32-bit buffer offset only spans that band — handling outputs
    whose flat M*N exceeds 2^31 / 4GB (mirrors Triton's ptr + offset.to(int64)).
    C is passed as a 2D [M, N] tensor so its shape packs within int32.
    """

    def __init__(self, C, c_rows, c_cols, c_idx_fn, n_tiles_a, n_tiles_b):
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.c_base = buffer_ops.extract_base_index(C)  # index = byte base address

    def store(self, c_frag, base_row, base_col):
        rsrc = make_row_band_resource(self.c_base, base_row, self.c_rows, self.c_cols, 2)
        for ti in range_constexpr(self.n_tiles_a):
            row_local = ti * 16 + (self.lane_id // 16) * 4  # relative to base_row
            for tj in range_constexpr(self.n_tiles_b):
                col = base_col + tj * 16 + self.lane_id % 16
                col_valid = col < self.c_cols
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                for i in range_constexpr(4):
                    val = vec_f32[i].to(fx.BFloat16)
                    # byte offset within band (<= BLOCK_M*c_cols*2, fits i32)
                    off = ((row_local + i) * self.c_cols + col) * 2
                    buffer_ops.buffer_store(
                        val,
                        rsrc,
                        off,
                        mask=col_valid,
                        offset_is_bytes=True,
                    )


def _compile_mxfp8_nt(
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    GROUP_M: int = 4,
    group_n: int = 0,  # 0 = 1D GROUP_M swizzle; >0 = 2D N-band (big-N L2 reuse)
    num_xcd: int = 8,
    waves_per_eu: int = 2,
    scale_pack: int = 1,  # 1 = broadcast scale (1-deep prefetch); >1 = opsel byte-pack
):
    BLOCK_K = 128
    assert GROUP_M >= 1

    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert K % BLOCK_K == 0

    K_ITERS = K // BLOCK_K

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    assert N_ACCUMS > 0

    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2

    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)

    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K

    # scale-tile fanout per MFMA wrapper call (A sub-tiles / B sub-tiles per wave).
    SA_TILES = N_TILES_A

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_mxfp8_nt(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type

        n_blocks = ceildiv(c_n, BLOCK_N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0 = lds.A_lds_cur_0
        a_cur1 = lds.A_lds_cur_1
        a_next0 = lds.A_lds_next_0
        a_next1 = lds.A_lds_next_1
        b_cur0 = lds.B_lds_cur_0
        b_cur1 = lds.B_lds_cur_1
        b_next0 = lds.B_lds_next_0
        b_next1 = lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        # 1D GROUP_M super-row swizzle (group_n=0) or 2D N-band (group_n>0, big-N L2
        # reuse: cuts the B re-stream). XCD-aware remap. See block_mn / xcd_remap_pid.
        num_pid_m = ceildiv(c_m, BLOCK_M)
        pid = xcd_remap_pid(fx.block_idx.x, num_pid_m * n_blocks, num_xcd)
        block_m, block_n = block_mn(pid, num_pid_m, n_blocks, GROUP_M, group_n)

        A0_gl_offset = (block_m * BLOCK_M) * K
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
        B0_gl_offset = (block_n * BLOCK_N) * K
        B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * K

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
        gl_off_b = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)

        mfma = MfmaScale16x16x128(N_TILES_A, N_TILES_B)

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)

        sa_s2r = ScaleS2R(A_scale, c_m, K, SA_TILES, pack=scale_pack)
        sb_s2r = ScaleBComb(B_scale, c_n, K, pack=scale_pack)  # one dwordx4 = b0+b1 scales
        store_c = StoreCPlain(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

        # Global row/col bases for the two M / N regions (region1 = +LDS half).
        wave_m_offset = wave_m * (N_TILES_A * 16)
        wave_n_offset = wave_n * (N_TILES_B * 16)
        sa_base0 = fx.Int32(block_m * BLOCK_M + wave_m_offset)
        sa_base1 = sa_base0 + fx.Int32(LDS_BLOCK_M)
        sb_base0 = fx.Int32(block_n * BLOCK_N + wave_n_offset)

        # 2x2 config of accumulators
        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

        if wave_m == 1:
            rocdl.s_barrier()

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
        b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        # scale_pack==1: 1-deep broadcast prefetch (2-deep spills: V=256 maxed). Pre-
        # load k=0, prefetch k+1, distributed across barrier sections.
        # scale_pack>1: opsel byte-pack -> load one i32 per `scale_pack` K-iters at the
        # loop top (held across them), pick this iter's byte via mfma.opsel.
        if const_expr(scale_pack == 1):
            sa0 = sa_s2r.load(sa_base0, 0)
            sa1 = sa_s2r.load(sa_base1, 0)
            sb_all = sb_s2r.load(sb_base0, 0)
            sb0, sb1 = sb_all[0:2], sb_all[2:4]

        for k in range_constexpr(K_ITERS - 2):
            if const_expr(scale_pack > 1):
                mfma.opsel = k % scale_pack
                if const_expr(k % scale_pack == 0):
                    sa0 = sa_s2r.load(sa_base0, k // scale_pack)
                    sa1 = sa_s2r.load(sa_base1, k // scale_pack)
                    sb_all = sb_s2r.load(sb_base0, k // scale_pack)
                    sb0, sb1 = sb_all[0:2], sb_all[2:4]
            else:
                sa0n = sa_s2r.load(sa_base0, k + 1)

            b0_frag = b_s2r.load(b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K)
            if const_expr(scale_pack == 1):
                sb_alln = sb_s2r.load(sb_base0, k + 1)  # one dwordx4 = both B regions
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
            if const_expr(scale_pack == 1):
                sa1n = sa_s2r.load(sa_base1, k + 1)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            rocdl.s_setprio(1)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1
            if const_expr(scale_pack == 1):
                sa0, sa1 = sa0n, sa1n
                sb_all = sb_alln
                sb0, sb1 = sb_all[0:2], sb_all[2:4]

        # Step k = K_ITERS - 2 (sa*/sb* hold scales[K_ITERS-2]; prefetch last iter)
        k = K_ITERS - 2
        if const_expr(scale_pack > 1):
            mfma.opsel = (K_ITERS - 2) % scale_pack
            if const_expr((K_ITERS - 2) % scale_pack == 0):
                sa0 = sa_s2r.load(sa_base0, (K_ITERS - 2) // scale_pack)
                sa1 = sa_s2r.load(sa_base1, (K_ITERS - 2) // scale_pack)
                sb_all = sb_s2r.load(sb_base0, (K_ITERS - 2) // scale_pack)
                sb0, sb1 = sb_all[0:2], sb_all[2:4]
        else:
            sa0n = sa_s2r.load(sa_base0, K_ITERS - 1)
            sa1n = sa_s2r.load(sa_base1, K_ITERS - 1)
            sb_alln = sb_s2r.load(sb_base0, K_ITERS - 1)

        b0_frag = b_s2r.load(b_cur0)
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        a_g2s.load(a_next1, A1_gl_offset + (K_ITERS - 1) * BLOCK_K)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b0_frag = b_s2r.load(b_next0)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1
        if const_expr(scale_pack == 1):
            sa0, sa1 = sa0n, sa1n
            sb_all = sb_alln
            sb0, sb1 = sb_all[0:2], sb_all[2:4]

        # Step k = K_ITERS - 1 (sa*/sb* already hold scales[K_ITERS-1])
        k = K_ITERS - 1
        if const_expr(scale_pack > 1):
            mfma.opsel = (K_ITERS - 1) % scale_pack
            if const_expr((K_ITERS - 1) % scale_pack == 0):
                sa0 = sa_s2r.load(sa_base0, (K_ITERS - 1) // scale_pack)
                sa1 = sa_s2r.load(sa_base1, (K_ITERS - 1) // scale_pack)
                sb_all = sb_s2r.load(sb_base0, (K_ITERS - 1) // scale_pack)
                sb0, sb1 = sb_all[0:2], sb_all[2:4]
        a0_frag = a_s2r.load(a_cur0)
        wait_barrier(0)

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        # Store back to gmem (no scaling)
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset

        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_mxfp8_nt(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_mxfp8_nt(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, 0, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_mxfp8_nt


# ── Primus-Turbo host wrapper ────────────────────────────────────────────────

_BLOCK_M = 256
_BLOCK_N = 256

_MXFP8_LAUNCH_CACHE: dict = {}  # (K, BLOCK_M, GROUP_M, num_xcd, layout, group_n) -> launch_mxfp8_nt
_COMPILED_MXFP8_CACHE: dict = {}  # (id(launch), shapes/dtypes/ints) -> compiled

# Per-shape NT autotune candidates (BLOCK_M, GROUP_M, num_xcd); BLOCK_N fixed 256.
# BLOCK_M=128 doubles the tiles (fills the CUs on skinny/small shapes), 256 wins big
# square / B-streaming; GROUP_M is the per-XCD L2-reuse super-block depth.
# BLOCK_M fixed at 256 (n_tiles_a = 256//64 = 4): the A-scale preshuffle layout is
# bm-dependent, and scales are now emitted preshuffled by the quant (no host repack),
# so the gemm cannot re-pack per candidate -> BLOCK_M must be constant. autotune sweeps
# only GROUP_M / num_xcd / group_n (none of which change the scale layout).
_MXFP8_NT_CANDIDATES = [
    (256, 4, 8),
    (256, 8, 8),
]


_MXFP8_AUTOTUNE_CACHE: dict = {}  # (M,N,K,out_dtype,layout) -> (BLOCK_M, GROUP_M, num_xcd, group_n)


def peek_mxfp8_cfg(M, N, K, out_dtype, layout="nt"):
    """Cached (BLOCK_M, GROUP_M, num_xcd, group_n) for this shape, or None if not
    yet benched. Lets the quant layer fuse the A-scale preshuffle (fanout
    n_tiles = BLOCK_M//64) only once the BLOCK_M has been picked by autotune."""
    return _MXFP8_AUTOTUNE_CACHE.get((M, N, K, out_dtype, layout))


# Layout -> compile factory. NT only (compute-only PR: NN/TN dropped). The A-scale
# (ScaleS2R) / B-scale (ScaleBComb) loaders are layout-invariant; the data path
# (G2S offsets / swizzle / MMA form) lives in the compile fn.
_LAYOUT_COMPILE = {
    "nt": _compile_mxfp8_nt,
}


def _mx_pack(K):
    """opsel scale byte-pack factor: pack consecutive K-iters' E8M0 into one i32
    (read via opsel) -> pack-fold fewer scale VMEM loads. 4 if K_ITERS % 4 == 0,
    else 2 if even, else 1 (off). Requires the matching packed host preshuffle."""
    ki = K // 128
    return 4 if ki % 4 == 0 else (2 if ki % 2 == 0 else 1)


def _mx_nt_gn_cands(N):
    """NT 2D N-band candidate widths for the autotune stage-2 sweep. The seed band
    (gn=0, NT's 1D swizzle) is measured separately as the baseline, so the final
    pick can never regress. Only offer a band when there are >= 2*gn 256-col
    N-blocks (else the band can't create the cross-tile B reuse it exists for).
    Winners are shape-dependent (NT: 7B GateUp gn16, 70B QKV gn8/16), so the
    per-shape bench picks rather than a single heuristic. Set env MX_DISABLE_NT_GN
    to force the seed band (NT -> 1D swizzle)."""
    import os

    if os.environ.get("MX_DISABLE_NT_GN"):
        return []
    n_blocks = (N + _BLOCK_N - 1) // _BLOCK_N
    # gn=32 was probed and dropped: its only win (NT 7B_GateUp +1.7% over gn16) is
    # coupled to tile (256,4), but stage-1 picks the tile at the seed band and lands
    # on (256,8) there, so the gn=32 win isn't reliably reachable — not worth the
    # extra autotune compile on every N>=16384 shape. {4,8,16} captures the robust
    # wins. (A fuller tile x gn cross-sweep could reach it but costs far more.)
    return [g for g in (4, 8, 16) if n_blocks >= 2 * g]


def _get_mxfp8_launch(K, bm, gm, xcd, layout="nt", N=0, gn=0):
    # gn is the autotuned 2D N-band width (0 = 1D swizzle). NT only (compute-only PR).
    lk = (K, bm, gm, xcd, layout, gn)
    launch = _MXFP8_LAUNCH_CACHE.get(lk)
    if launch is None:
        # NT plain-load path. scale_pack=1: opsel byte-pack needs quant-emitted packed
        # scales (-> quant PR); the compute-only host preshuffle is broadcast (pack=1).
        launch = _LAYOUT_COMPILE[layout](
            K=K,
            BLOCK_M=bm,
            BLOCK_N=_BLOCK_N,
            GROUP_M=gm,
            group_n=gn,
            num_xcd=xcd,
            scale_pack=1,
        )
        _MXFP8_LAUNCH_CACHE[lk] = launch
    return launch


def _autotune_mxfp8(a8, b8, out_view, a_sp, b_sp, M, N, K, out_dtype, layout="nt"):
    """First-call micro-bench of the candidates for (M,N,K,layout); cache the
    fastest cfg by shape. Returns (BLOCK_M, GROUP_M, num_xcd, group_n). Scales are
    pre-shuffled by the quant (BLOCK_M fixed at 256), so the same a_sp/b_sp feed
    every candidate -- autotune sweeps only GROUP_M / num_xcd / group_n. NN/TN fold
    the fixed scale-delivery combo into _get_mxfp8_launch.

    Two-stage for NT: stage 1 picks (BM,GM,XCD) at group_n=0 (1D swizzle); stage 2
    fixes that tile and sweeps the 2D N-band width group_n. gn=0 is measured in
    stage 1, so the staged pick can never regress vs the old (gn-less) NT path —
    it only captures the big-/mid-N L2-reuse win on shapes where a band helps."""
    key = (M, N, K, out_dtype, layout)
    cached = _MXFP8_AUTOTUNE_CACHE.get(key)
    if cached is not None:
        return cached
    cands = _MXFP8_NT_CANDIDATES
    stream = torch.cuda.current_stream()

    def _time_cfg(bm, gm, xcd, gn):
        try:
            launch = _get_mxfp8_launch(K, bm, gm, xcd, layout, N, gn)
            args = (a8, b8, out_view, a_sp, b_sp, M, N, stream)
            c = get_compiled(_COMPILED_MXFP8_CACHE, launch, args)
            c(*args)
            torch.cuda.synchronize()
            if not torch.isfinite(out_view.reshape(-1)[:1024].float()).all().item():
                return float("inf")
            for _ in range(2):
                c(*args)
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(20):
                c(*args)
            e1.record()
            torch.cuda.synchronize()
            return e0.elapsed_time(e1) * 1000.0 / 20
        except Exception:
            return float("inf")

    # Stage 1: best (BLOCK_M, GROUP_M, num_xcd) at the SEED band width gn=0 (NT's
    # native 1D swizzle); stage 2 sweeps the 2D N-band width on top.
    seed_gn = 0
    best_us = float("inf")
    best = None
    for bm, gm, xcd in cands:
        us = _time_cfg(bm, gm, xcd, seed_gn)
        if us < best_us:
            best_us = us
            best = (bm, gm, xcd, seed_gn)
    if best is None:
        best = (_BLOCK_M, 4, 8, seed_gn)  # fall back to the always-valid 256/g4/x8
    # Stage 2: fix the winning tile, sweep the 2D N-band width via robust min-of-4
    # timing, re-measuring the seed the same way; adopt a band only past a 1.5%
    # margin over the re-measured seed -> no regression by construction. gn=0 stays
    # a candidate so an over-applied band can be dropped.
    gn_cands = _mx_nt_gn_cands(N)
    if gn_cands:
        bm, gm, xcd, _ = best

        def _robust(gn):
            return min(_time_cfg(bm, gm, xcd, gn) for _ in range(4))

        seed_us = _robust(seed_gn)  # re-measured seed baseline (same estimator as the bands)
        bgn, bus = seed_gn, seed_us
        for gn in sorted(set([0] + gn_cands) - {seed_gn}):
            us = _robust(gn)
            if us < bus and us < seed_us * 0.985:
                bgn, bus = gn, us
        best = (bm, gm, xcd, bgn)
    _MXFP8_AUTOTUNE_CACHE[key] = best
    return best


def gemm_mxfp8_flydsl_kernel(
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
    """MXFP8 (per-1x32 E8M0 block-scaled) dense GEMM, gfx950.

    ``trans_c=True`` returns ``out.t().contiguous()`` (mirrors the tensorwise
    FlyDSL wrapper).

    Computes the per-32-K-block E8M0-scaled product with the scale folded into
    the MFMA (``v_mfma_scale_f32_16x16x128_f8f6f4``). NT only (compute-only PR;
    NN/TN dropped):
      - NT (F, T): A [M,K], B [N,K] (B^T storage), C = a @ b^T.
      - any other (trans_a, trans_b): unsupported (raises).

    COMPUTE-ONLY: the E8M0 scales arrive RAW (no quant preshuffle) as ``a_scale``
    [free=M, K//32] and ``b_scale`` [free=N, K//32] uint8/e8m0, and are host-preshuffled
    here (``preshuffle_scale`` / ``preshuffle_scale_b_comb``, broadcast layout =>
    scale_pack=1). BLOCK_M is fixed at 256 so the A-scale fanout n_tiles = 4 is constant;
    autotune sweeps only GROUP_M / num_xcd / group_n. The quant-emitted preshuffle (drops
    the host repack + enables opsel byte-pack) lands in the separate quant PR.

    Args:
      a, b:     float8_e4m3fn, shapes per the layout above.
      a_scale:  raw A-operand E8M0 scale [free=M, K//32] (uint8/e8m0).
      b_scale:  raw B-operand E8M0 scale [free=N, K//32] (uint8/e8m0).
      out_dtype: bf16 (the kernel epilogue stores bf16).

    Constraints: K % 128 == 0 and K >= 256; M % 64 == 0; N % 64 == 0.
    """
    assert a.dim() == 2 and b.dim() == 2, "a, b must be 2D"
    assert out_dtype == torch.bfloat16, "mxfp8 FlyDSL store emits bf16 only"

    if (not trans_a) and trans_b:
        layout = "nt"
        M, K = a.shape
        N, Kb = b.shape
    else:
        raise NotImplementedError(
            "mxfp8 FlyDSL GEMM is NT only (trans_a=False, trans_b=True); "
            f"got trans_a={trans_a}, trans_b={trans_b}."
        )
    assert K == Kb, f"K mismatch: a {a.shape}, b {b.shape} (layout {layout})"
    assert K % 128 == 0 and K >= 256, f"K must be a multiple of 128 and >= 256, got {K}"
    # M (BLOCK_M=256) / N (BLOCK_N=256) need only be 64-multiples: partial output tiles
    # are handled in-kernel by buffer/SRD clamping (data + pre-shuffled scale reads bound
    # by num_records, StoreC bounds rows/cols), so non-256 shapes run without host repack.
    assert M % 64 == 0, f"M must be a multiple of 64 (A-scale preshuffle), got {M}"
    assert N % 64 == 0, f"N must be a multiple of 64 (combined-B scale preshuffle), got {N}"

    # C is passed as 2D [M, N] (NOT flat): FlyDSL packs each shape dim as int32, so a 1D
    # [M*N] view overflows when M*N > 2^31; StoreCPlain addresses C via its i64 per-tile
    # re-basing, so the 2D shape is only metadata.
    # COMPUTE-ONLY: scales arrive as RAW E8M0 [free, K//32] (free = M for A, N for B),
    # host-preshuffled here into the kernel's broadcast layout (scale_pack=1). The fused
    # quant-emitted preshuffle (no host repack) is deferred to the quant PR.
    n_tiles_a = _BLOCK_M // 64
    a_e8 = a_scale.contiguous().view(torch.uint8)
    b_e8 = b_scale.contiguous().view(torch.uint8)
    a_sp = preshuffle_scale(a_e8, K, n_tiles_a).view(torch.int32).reshape(-1)
    b_sp = preshuffle_scale_b_comb(b_e8, K).view(torch.int32).reshape(-1)
    out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    a8 = as_i8_flat(a)
    b8 = as_i8_flat(b)
    # Per-shape cfg: first call benches GROUP_M/num_xcd/group_n (BLOCK_M fixed 256),
    # caches the winner by (M,N,K,layout); the same pre-shuffled a_sp feeds all candidates.
    bm, gm, xcd, gn = _autotune_mxfp8(a8, b8, out, a_sp, b_sp, M, N, K, out_dtype, layout)
    launch = _get_mxfp8_launch(K, bm, gm, xcd, layout, N, gn)
    args = (a8, b8, out, a_sp, b_sp, M, N, torch.cuda.current_stream())
    get_compiled(_COMPILED_MXFP8_CACHE, launch, args)(*args)
    return out.t().contiguous() if trans_c else out
