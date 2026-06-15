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
# Shared fp8 GEMM primitives are vendored in primus_turbo/flydsl/utils/gemm_helper.py
# (landed with the tensorwise FlyDSL backend, #356). They must be importable as
# module globals (@flyc.kernel needs them as globals, not closure cells).
#   * G2SLoader carries the per-wave LDS ``chunk_stride``, so the TN bank-spread
#     writer is just ``G2SLoader(..., chunk_stride=_LDS_CS)`` (no separate subclass).
#   * S2RLoaderTr is the unified wave-coop ds_read_b64_tr_b8 transpose loader
#     serving NN-B / TN-A / TN-B (operand picked by ``tile_stride``); mxfp8 reuses
#     it unchanged since the scale operands are layout-invariant (a property of the
#     MFMA, not the data load), so ScaleS2R/ScaleBComb stay as-is.
from primus_turbo.flydsl.utils.gemm_helper import (
    G2SLoader,
    S2RLoader,
    S2RLoaderTr,
    as_i8_flat,
    block_mn,
    ceildiv,
    compute_global_swizzle,
    compute_global_swizzle_nn,
    get_compiled,
    make_fp8_buffer_tensor,
    make_row_band_resource,
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


def preshuffle_scale(e8m0_u8, K, n_tiles):
    """Host-side E8M0 scale pre-shuffle for the mxfp8 8-wave kernel.

    ``n_tiles`` = the per-wave sub-tile fan-out the kernel loads together in one
    vectorized dword{n_tiles} (A: BLOCK_M//64, B: BLOCK_N//128).

    Input : uint8 [DIM, K//32] row-major E8M0 (DIM multiple of 16*n_tiles).
    Output: int32 [DIM//(16*n_tiles), K//128, 64, n_tiles] where
        SP[grp, k, lane, s] = broadcast( scale[grp*16*n_tiles + s*16 + lane%16,
                                               4k + lane//16] )
    so a wave reads its ``n_tiles`` sub-tile scales for (grp, k) as one coalesced
    vector load of ``n_tiles`` contiguous dwords per lane, directly usable as the
    MFMA scale operand (opsel==0 reads byte 0; broadcast => byte-position safe).
    """
    DIM, Kb = e8m0_u8.shape
    assert Kb == K // 32 and K % 128 == 0
    assert DIM % (16 * n_tiles) == 0, f"DIM={DIM} must be multiple of {16 * n_tiles}"
    K128 = K // 128
    G = DIM // (16 * n_tiles)
    s = e8m0_u8.reshape(G, n_tiles, 16, K128, 4)  # [grp, s, r, k, g]
    s = s.permute(0, 3, 4, 2, 1).reshape(G, K128, 64, n_tiles)  # [grp,k,g,r,s] -> one copy
    # broadcast each E8M0 byte into all 4 bytes of the dword (s*0x01010101)
    return s.to(torch.int32).mul_(0x01010101)


_B_COMB_COLIDX_CACHE: dict = {}  # device -> [256] gather index (col256 = wn*32 + OFF[si] + r)


def _b_comb_colidx(device):
    """Cached column-gather index for preshuffle_scale_b_comb. Constant (shape-
    independent: always the 256-col bijection), so build once per device on the
    GPU instead of materializing torch.arange on CPU + syncing every call."""
    ci = _B_COMB_COLIDX_CACHE.get(device)
    if ci is None:
        wn = torch.arange(4, device=device).view(4, 1, 1)
        r = torch.arange(16, device=device).view(1, 1, 16)
        off = torch.tensor([0, 16, 128, 144], device=device).view(1, 4, 1)
        ci = (wn * 32 + off + r).reshape(-1)  # [wn, si, r] flattened
        _B_COMB_COLIDX_CACHE[device] = ci
    return ci


def preshuffle_scale_b_comb(e8m0_u8, K):
    """Combined-B E8M0 pre-shuffle: pack BOTH N-regions' (b0,b1) 4 sub-tiles for a
    wave into one dword{4}, so the kernel issues a single dwordx4 for all B scales
    per K-iter (vs two loads). Requires N % 256 == 0.

    A wave's 4 B sub-tiles sit at cols c+{0,16,128,144} (b0: 0,16; b1: 128,144),
    c = block_n*256 + wave_n*32. Output int32 [N//64, K//128, 64, 4]:
        SP[grp, k, lane, s] = broadcast( scale[c + OFF[s] + lane%16, 4k + lane//16] )
    grp = block_n*4 + wave_n;  OFF = [0,16,128,144].
    """
    N, Kb = e8m0_u8.shape
    assert Kb == K // 32 and K % 128 == 0 and N % 256 == 0
    K128 = K // 128
    colidx = _b_comb_colidx(e8m0_u8.device)  # cached GPU index (no per-call CPU arange/sync)
    s = e8m0_u8.reshape(N // 256, 256, K128, 4)  # [nblk, col256, k, g]
    g = s[:, colidx, :, :].reshape(N // 256, 4, 4, 16, K128, 4)  # [nblk, wn, si, r, k, g]
    g = g.permute(0, 1, 4, 5, 3, 2).reshape(N // 64, K128, 64, 4)  # grp=nblk*4+wn, lane=g*16+r
    # broadcast E8M0 byte into the dword (see preshuffle_scale).
    return g.to(torch.int32).mul_(0x01010101)


def preshuffle_scale_pack(e8m0_u8, K, n_tiles, pack):
    """Byte-PACKED A-scale preshuffle: identical lane layout to ``preshuffle_scale``
    but instead of broadcasting one E8M0 byte into all 4 dword bytes, pack ``pack``
    consecutive K-iters' scales into bytes 0..pack-1 of the i32. The kernel then
    loads one i32 per ``pack`` K-iters and the MMA at K-iter k uses opsel=k%pack to
    select byte (k%pack) -> pack-fold fewer scale VMEM transactions. Output int32
    [DIM//(16*n_tiles), K//(128*pack), 64, n_tiles]. pack in {1,2,4}; pack=1 ==
    broadcast (preshuffle_scale). Requires (K//128) % pack == 0."""
    DIM, Kb = e8m0_u8.shape
    assert Kb == K // 32 and K % 128 == 0
    K128 = K // 128
    assert K128 % pack == 0, f"K_ITERS={K128} must be a multiple of pack={pack}"
    G = DIM // (16 * n_tiles)
    s = e8m0_u8.reshape(G, n_tiles, 16, K128, 4).permute(0, 3, 4, 2, 1)  # [grp,k,g,r,s]
    s = s.reshape(G, K128 // pack, pack, 64, n_tiles)  # [grp, kg, j, lane, s]
    out = torch.zeros(G, K128 // pack, 64, n_tiles, dtype=torch.int32, device=e8m0_u8.device)
    for j in range(pack):  # byte j = K-iter kg*pack + j (read via opsel=j)
        out |= s[:, :, j].to(torch.int32) << (8 * j)
    return out


def preshuffle_scale_b_comb_pack(e8m0_u8, K, pack):
    """Byte-PACKED combined-B scale preshuffle (see preshuffle_scale_pack +
    preshuffle_scale_b_comb). Output int32 [N//64, K//(128*pack), 64, 4]."""
    N, Kb = e8m0_u8.shape
    assert Kb == K // 32 and K % 128 == 0 and N % 256 == 0
    K128 = K // 128
    assert K128 % pack == 0, f"K_ITERS={K128} must be a multiple of pack={pack}"
    colidx = _b_comb_colidx(e8m0_u8.device)
    s = e8m0_u8.reshape(N // 256, 256, K128, 4)
    g = s[:, colidx, :, :].reshape(N // 256, 4, 4, 16, K128, 4)  # [nblk,wn,si,r,k,g]
    g = g.permute(0, 1, 4, 5, 3, 2).reshape(N // 64, K128, 64, 4)  # [grp,k,lane,s]
    g = g.reshape(N // 64, K128 // pack, pack, 64, 4)  # [grp, kg, j, lane, s]
    out = torch.zeros(N // 64, K128 // pack, 64, 4, dtype=torch.int32, device=e8m0_u8.device)
    for j in range(pack):
        out |= g[:, :, j].to(torch.int32) << (8 * j)
    return out


class ScaleBComb:
    """Combined B scale loader (pairs with ``preshuffle_scale_b_comb``).

    One dwordx4 per lane returns [s0,s1,s2,s3]; (s0,s1)=b0 sub-tiles, (s2,s3)=b1.
    """

    def __init__(self, sp_tensor, dim, K, pack=1):
        self.K128 = K // (128 * pack)  # number of K-groups (pack K-iters per i32)
        self.lane = fx.thread_idx.x % 64
        nbytes = (dim // 64) * self.K128 * 64 * 4 * 4  # int32 records
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


def _mx_value_attrs(waves_per_eu, agpr_alloc=0):
    """Kernel value_attrs. ``agpr_alloc`` > 0 adds an ``amdgpu-agpr-alloc 0,N``
    passthrough so the inline-asm ds_read_b64_tr_b8 ``=v`` constraint has an AGPR
    budget (compiler-decide AGPR=0 + inline asm -> nan) and the asm tr8 register
    pressure can spill to AGPR instead of scratch.
    """
    d = {"rocdl.waves_per_eu": waves_per_eu, "rocdl.flat_work_group_size": "512,512"}
    pt = []
    if agpr_alloc > 0:
        pt.append(["amdgpu-agpr-alloc", f"0,{agpr_alloc}"])
    if pt:
        d["passthrough"] = pt
    return d


def _compile_mxfp8_nt(
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    GROUP_M: int = 4,
    group_n: int = 0,  # 0 = 1D GROUP_M swizzle; >0 = 2D N-band (big-N L2 reuse)
    num_xcd: int = 8,
    waves_per_eu: int = 2,
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

        sa_s2r = ScaleS2R(A_scale, c_m, K, SA_TILES)
        sb_s2r = ScaleBComb(B_scale, c_n, K)  # one dwordx4 = b0+b1 scales
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

        # 1-deep scale prefetch (2-deep spills: V=256 maxed, register pressure
        # dominated the latency-hiding benefit). Pre-load k=0, prefetch k+1, scale
        # loads distributed across barrier sections.
        sa0 = sa_s2r.load(sa_base0, 0)
        sa1 = sa_s2r.load(sa_base1, 0)
        sb_all = sb_s2r.load(sb_base0, 0)
        sb0, sb1 = sb_all[0:2], sb_all[2:4]

        for k in range_constexpr(K_ITERS - 2):
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
            sb_alln = sb_s2r.load(sb_base0, k + 1)  # one dwordx4 = both B regions
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
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
            sa0, sa1 = sa0n, sa1n
            sb_all = sb_alln
            sb0, sb1 = sb_all[0:2], sb_all[2:4]

        # Step k = K_ITERS - 2 (sa*/sb* hold scales[K_ITERS-2]; prefetch last iter)
        k = K_ITERS - 2
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
        sa0, sa1 = sa0n, sa1n
        sb_all = sb_alln
        sb0, sb1 = sb_all[0:2], sb_all[2:4]

        # Step k = K_ITERS - 1 (sa*/sb* already hold scales[K_ITERS-1])
        k = K_ITERS - 1
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
            value_attrs=_mx_value_attrs(waves_per_eu),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_mxfp8_nt


def _compile_mxfp8_nn(
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    GROUP_M: int = 4,
    group_n: int = 0,  # 0 = 1D GROUP_M swizzle; >0 = 2D N-band (big-N L2 reuse)
    num_xcd: int = 8,
    waves_per_eu: int = 2,
    b_inline_asm: bool = False,
    agpr_alloc: int = 0,
    vmcnt_hint: int = 2,
    scale_pack: int = 1,
):
    """MXFP8 NN-layout dense GEMM: A [M, K] (K-contig), B [K, N] (N-contig),
    C = A @ B with per-1x32-K-block E8M0 scaling folded into the MFMA.

    Structural identity vs the NT kernel ``_compile_mxfp8_nt``:
      * A path (G2S / S2R / ScaleS2R / A-scale prefetch) is IDENTICAL — A is
        [M, K] in both NT and NN.
      * B path takes the tensorwise NT->NN delta: B is [K, N] row-major, so the
        global offset drops the ``* K`` row stride, the K-iter step advances
        K-rows by ``BLOCK_K * c_n`` elements, the load uses ``compute_global_
        swizzle_nn`` + the ``ds_read_b64_tr_b8`` transpose loader S2RLoaderTr.
      * Scale operands are UNCHANGED: the MFMA distributes the B-scale operand by
        lane independent of how B was loaded into registers (S2RLoaderTr yields
        the same mfma B-operand byte layout as the plain NT S2RLoader), so the
        B-scale is still logically [N, K//32] consumed by ScaleBComb.

    S2RLoaderTr is used with ``inline_asm=False`` (rocdl ds_read_tr8_b64 intrinsic
    + compiler-auto vmcnt drains): the inline-asm path needs nonzero AGPR, and an
    AGPR-resident scaled MMA is a known dead end for mxfp8 (acc-clobber). The
    accumulator therefore stays in VGPR, exactly as in the NT kernel.
    """
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
    def kernel_mxfp8_nn(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        # Force thread_idx.x materialization before S2RLoaderTr lazy-evaluates it
        # inside range_constexpr loops (else flydsl emits an out-of-order
        # ds_read_tr8_b64 schedule -> garbage). Same workaround as tensorwise NN.
        _ = str(fx.thread_idx.x)
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
        # 1D GROUP_M super-row swizzle (group_n=0) or 2D N-band (group_n>0): NN's B is
        # [K,N] N-contig like TN, so the band reuse applies. See block_mn.
        num_pid_m = ceildiv(c_m, BLOCK_M)
        pid = xcd_remap_pid(fx.block_idx.x, num_pid_m * n_blocks, num_xcd)
        block_m, block_n = block_mn(pid, num_pid_m, n_blocks, GROUP_M, group_n)

        # A: same as NT ([M, K] K-contig).
        A0_gl_offset = (block_m * BLOCK_M) * K
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
        # B: NN-specific. B is [K, N] row-major; offset is the N-col base (no
        # * K row stride). The K-iter advances K-rows by BLOCK_K, = BLOCK_K * c_n
        # elements (applied at each load below).
        B0_gl_offset = block_n * BLOCK_N + 0
        B1_gl_offset = block_n * BLOCK_N + LDS_BLOCK_N

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
        gl_off_b = compute_global_swizzle_nn(lane_id, wave_id, c_n, N_LDS_ROUNDS)

        mfma = MfmaScale16x16x128(N_TILES_A, N_TILES_B)

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoaderTr(wave_n, N_TILES_B, 32, inline_asm=b_inline_asm, vmcnt_hint=vmcnt_hint)

        sa_s2r = ScaleS2R(A_scale, c_m, K, SA_TILES, pack=scale_pack)
        sb_s2r = ScaleBComb(B_scale, c_n, K, pack=scale_pack)
        store_c = StoreCPlain(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

        wave_m_offset = wave_m * (N_TILES_A * 16)
        wave_n_offset = wave_n * (N_TILES_B * 16)
        sa_base0 = fx.Int32(block_m * BLOCK_M + wave_m_offset)
        sa_base1 = sa_base0 + fx.Int32(LDS_BLOCK_M)
        sb_base0 = fx.Int32(block_n * BLOCK_N + wave_n_offset)

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K * c_n)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K * c_n)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

        if wave_m == 1:
            rocdl.s_barrier()

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K * c_n)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
        b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K * c_n)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        # 0-deep scale: opsel byte-pack loads one i32 per `scale_pack` K-iters (held
        # across them); for scale_pack==1, load this iter's scale at the top. No k+1
        # prefetch -> one scale set live, so the tr8-tight BLOCK_M=256 fits (a 2nd
        # live set spills to scratch); load latency hides under the per-MFMA barriers.
        for k in range_constexpr(K_ITERS - 2):
            if const_expr(scale_pack > 1):
                # byte-packed scale: load one i32 per `scale_pack` K-iters (held
                # across them), pick the byte for this iter via the MMA opsel.
                mfma.opsel = k % scale_pack
                if const_expr(k % scale_pack == 0):
                    sa0 = sa_s2r.load(sa_base0, k // scale_pack)
                    sa1 = sa_s2r.load(sa_base1, k // scale_pack)
                    sb_all = sb_s2r.load(sb_base0, k // scale_pack)
                    sb0, sb1 = sb_all[0:2], sb_all[2:4]
            else:
                sa0 = sa_s2r.load(sa_base0, k)
                sa1 = sa_s2r.load(sa_base1, k)
                sb_all = sb_s2r.load(sb_base0, k)
                sb0, sb1 = sb_all[0:2], sb_all[2:4]

            b0_frag = b_s2r.load(b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K * c_n)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K * c_n)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            rocdl.s_setprio(1)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 2
        k = K_ITERS - 2
        if const_expr(scale_pack > 1):
            mfma.opsel = (K_ITERS - 2) % scale_pack
            if const_expr((K_ITERS - 2) % scale_pack == 0):
                sa0 = sa_s2r.load(sa_base0, (K_ITERS - 2) // scale_pack)
                sa1 = sa_s2r.load(sa_base1, (K_ITERS - 2) // scale_pack)
                sb_all = sb_s2r.load(sb_base0, (K_ITERS - 2) // scale_pack)
                sb0, sb1 = sb_all[0:2], sb_all[2:4]
        else:
            sa0 = sa_s2r.load(sa_base0, K_ITERS - 2)
            sa1 = sa_s2r.load(sa_base1, K_ITERS - 2)
            sb_all = sb_s2r.load(sb_base0, K_ITERS - 2)
            sb0, sb1 = sb_all[0:2], sb_all[2:4]

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

        # Step k = K_ITERS - 1
        k = K_ITERS - 1
        if const_expr(scale_pack > 1):
            mfma.opsel = (K_ITERS - 1) % scale_pack
            if const_expr((K_ITERS - 1) % scale_pack == 0):
                sa0 = sa_s2r.load(sa_base0, (K_ITERS - 1) // scale_pack)
                sa1 = sa_s2r.load(sa_base1, (K_ITERS - 1) // scale_pack)
                sb_all = sb_s2r.load(sb_base0, (K_ITERS - 1) // scale_pack)
                sb0, sb1 = sb_all[0:2], sb_all[2:4]
        else:
            sa0 = sa_s2r.load(sa_base0, K_ITERS - 1)
            sa1 = sa_s2r.load(sa_base1, K_ITERS - 1)
            sb_all = sb_s2r.load(sb_base0, K_ITERS - 1)
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

        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset

        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_mxfp8_nn(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_mxfp8_nn(
            A,
            B,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs=_mx_value_attrs(waves_per_eu, agpr_alloc),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_mxfp8_nn


def _compile_mxfp8_tn(
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    GROUP_M: int = 4,
    group_n: int = 0,  # 0 = 1D GROUP_M swizzle; >0 = 2D N-band (big-N L2 reuse)
    num_xcd: int = 8,
    waves_per_eu: int = 2,
    b_inline_asm: bool = False,
    a_inline_asm: bool = False,
    agpr_alloc: int = 0,
    vmcnt_hint: int = 2,
    scale_pack: int = 1,
    asm_mma: bool = False,
):
    """MXFP8 TN-layout dense GEMM: A [K, M] (M-contig), B [K, N] (N-contig),
    C = A^T @ B with per-1x32-K-block E8M0 scaling folded into the MFMA.

    Both operands are K-row-strided, so BOTH use the unified wave-coop
    ds_read_b64_tr_b8 transpose loader S2RLoaderTr (A with tile_stride=LDS_BLOCK_M//2,
    B with tile_stride=32) over a bank-spread (_LDS_CS=1056) LDS, mirroring the
    tensorwise TN kernel's data path.

    CRITICAL mxfp8 deviation from tensorwise TN: tensorwise TN routes the MFMA
    through inline-asm with an AGPR-resident accumulator (``=a,v,v,0``) for
    spill-free occupancy. For mxfp8 the scaled MFMA in AGPR is a proven dead end
    (acc-clobber garbage / INVALID_ISA), so this kernel keeps the rocdl
    *intrinsic* scaled MFMA (MfmaScale16x16x128, VGPR accumulator) — identical to
    the NT/NN kernels — and keeps the full 7-barrier schedule for determinism.
    Consequently the transpose loaders run inline_asm=False (intrinsic
    ds_read_tr8_b64 + compiler-auto vmcnt drains), so no AGPR is required.

    Scale operands are UNCHANGED from NT/NN: the MFMA distributes each operand's
    scale by lane independent of how the operand was loaded (tr8 yields the same
    mfma A/B operand byte layout as a plain load), so A-scale stays logically
    [M, K//32] (ScaleS2R, free=M) and B-scale [N, K//32] (ScaleBComb, free=N).
    """
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

    # TN A path uses tr8 (S2RLoaderTr K_log in [0,128)) => force >=2 G2S rounds
    # (16K LDS slot) even when BM=128 would naturally give 1.
    N_LDS_STEPS_A = max(LDS_BLOCK_M // 64, 2)
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)

    # Bank-spread LDS chunk stride (gfx950 transpose-read bank-conflict fix).
    # Load-bearing constant shared by the G2S writer (G2SLoader chunk_stride) and
    # the tr8 reader (S2RLoaderTr). See tensorwise TN.
    _LDS_CS = 1056
    a_lds_size = max(LDS_BLOCK_M * BLOCK_K, 2 * 8 * 1024) // 1024 * _LDS_CS
    b_lds_size = (LDS_BLOCK_N * BLOCK_K) // 1024 * _LDS_CS

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
    def kernel_mxfp8_tn(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        _ = str(fx.thread_idx.x)
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
        num_pid_m = ceildiv(c_m, BLOCK_M)
        pid = xcd_remap_pid(fx.block_idx.x, num_pid_m * n_blocks, num_xcd)
        # 1D GROUP_M super-row swizzle (group_n=0) or 2D N-band (group_n>0, big-N
        # L2 reuse: cuts the B re-stream). See block_mn.
        block_m, block_n = block_mn(pid, num_pid_m, n_blocks, GROUP_M, group_n)

        # A: TN [K, M] M-contig -> offset is the M-col base (no * K row stride);
        # K-iter advances K-rows by BLOCK_K * c_m elements.
        A0_gl_offset = block_m * BLOCK_M + 0
        A1_gl_offset = block_m * BLOCK_M + LDS_BLOCK_M
        # B: [K, N] N-contig (same as NN); K-iter advances by BLOCK_K * c_n.
        B0_gl_offset = block_n * BLOCK_N + 0
        B1_gl_offset = block_n * BLOCK_N + LDS_BLOCK_N

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        # Both A and B use the NN-style K-strided global swizzle.
        gl_off_a = compute_global_swizzle_nn(lane_id, wave_id, c_m, N_LDS_ROUNDS)
        gl_off_b = compute_global_swizzle_nn(lane_id, wave_id, c_n, N_LDS_ROUNDS)

        mfma = MfmaScale16x16x128(N_TILES_A, N_TILES_B, asm_mma=asm_mma)

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id, chunk_stride=_LDS_CS)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id, chunk_stride=_LDS_CS)
        # Unified S2RLoaderTr: A operand tile_stride = LDS_BLOCK_M // 2, B = 32
        # (same values main's tensorwise TN uses for the identical data path).
        a_s2r = S2RLoaderTr(
            wave_m,
            N_TILES_A,
            LDS_BLOCK_M // 2,
            inline_asm=a_inline_asm,
            vmcnt_hint=vmcnt_hint,
            chunk_stride=_LDS_CS,
        )
        b_s2r = S2RLoaderTr(
            wave_n, N_TILES_B, 32, inline_asm=b_inline_asm, vmcnt_hint=vmcnt_hint, chunk_stride=_LDS_CS
        )

        sa_s2r = ScaleS2R(A_scale, c_m, K, SA_TILES, pack=scale_pack)
        sb_s2r = ScaleBComb(B_scale, c_n, K, pack=scale_pack)
        store_c = StoreCPlain(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

        wave_m_offset = wave_m * (N_TILES_A * 16)
        wave_n_offset = wave_n * (N_TILES_B * 16)
        sa_base0 = fx.Int32(block_m * BLOCK_M + wave_m_offset)
        sa_base1 = sa_base0 + fx.Int32(LDS_BLOCK_M)
        sb_base0 = fx.Int32(block_n * BLOCK_N + wave_n_offset)

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K * c_n)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K * c_m)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K * c_n)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K * c_m)

        if wave_m == 1:
            rocdl.s_barrier()

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K * c_n)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K * c_m)
        b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K * c_n)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        # 0-deep scale (see NN kernel for the rationale). TN is the most VGPR-tight
        # layout — tr8 on BOTH operands (a_s2r and b_s2r are S2RLoaderTr).
        for k in range_constexpr(K_ITERS - 2):
            if const_expr(scale_pack > 1):
                mfma.opsel = k % scale_pack
                if const_expr(k % scale_pack == 0):
                    sa0 = sa_s2r.load(sa_base0, k // scale_pack)
                    sa1 = sa_s2r.load(sa_base1, k // scale_pack)
                    sb_all = sb_s2r.load(sb_base0, k // scale_pack)
                    sb0, sb1 = sb_all[0:2], sb_all[2:4]
            else:
                sa0 = sa_s2r.load(sa_base0, k)
                sa1 = sa_s2r.load(sa_base1, k)
                sb_all = sb_s2r.load(sb_base0, k)
                sb0, sb1 = sb_all[0:2], sb_all[2:4]

            b0_frag = b_s2r.load(b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K * c_m)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K * c_n)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K * c_m)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K * c_n)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            rocdl.s_setprio(1)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 2
        k = K_ITERS - 2
        if const_expr(scale_pack > 1):
            mfma.opsel = (K_ITERS - 2) % scale_pack
            if const_expr((K_ITERS - 2) % scale_pack == 0):
                sa0 = sa_s2r.load(sa_base0, (K_ITERS - 2) // scale_pack)
                sa1 = sa_s2r.load(sa_base1, (K_ITERS - 2) // scale_pack)
                sb_all = sb_s2r.load(sb_base0, (K_ITERS - 2) // scale_pack)
                sb0, sb1 = sb_all[0:2], sb_all[2:4]
        else:
            sa0 = sa_s2r.load(sa_base0, K_ITERS - 2)
            sa1 = sa_s2r.load(sa_base1, K_ITERS - 2)
            sb_all = sb_s2r.load(sb_base0, K_ITERS - 2)
            sb0, sb1 = sb_all[0:2], sb_all[2:4]

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
        a_g2s.load(a_next1, A1_gl_offset + (K_ITERS - 1) * BLOCK_K * c_m)
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

        # Step k = K_ITERS - 1
        k = K_ITERS - 1
        if const_expr(scale_pack > 1):
            mfma.opsel = (K_ITERS - 1) % scale_pack
            if const_expr((K_ITERS - 1) % scale_pack == 0):
                sa0 = sa_s2r.load(sa_base0, (K_ITERS - 1) // scale_pack)
                sa1 = sa_s2r.load(sa_base1, (K_ITERS - 1) // scale_pack)
                sb_all = sb_s2r.load(sb_base0, (K_ITERS - 1) // scale_pack)
                sb0, sb1 = sb_all[0:2], sb_all[2:4]
        else:
            sa0 = sa_s2r.load(sa_base0, K_ITERS - 1)
            sa1 = sa_s2r.load(sa_base1, K_ITERS - 1)
            sb_all = sb_s2r.load(sb_base0, K_ITERS - 1)
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

        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset

        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_mxfp8_tn(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_mxfp8_tn(
            A,
            B,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs=_mx_value_attrs(waves_per_eu, agpr_alloc),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_mxfp8_tn


# ── Primus-Turbo host wrapper ────────────────────────────────────────────────

_BLOCK_M = 256
_BLOCK_N = 256

_MXFP8_LAUNCH_CACHE: dict = {}  # (K, BLOCK_M, GROUP_M, num_xcd, layout, group_n) -> launch_mxfp8_nt
_COMPILED_MXFP8_CACHE: dict = {}  # (id(launch), shapes/dtypes/ints) -> compiled

# Per-shape NT autotune candidates (BLOCK_M, GROUP_M, num_xcd); BLOCK_N fixed 256.
# BLOCK_M=128 doubles the tiles (fills the CUs on skinny/small shapes), 256 wins big
# square / B-streaming; GROUP_M is the per-XCD L2-reuse super-block depth.
_MXFP8_NT_CANDIDATES = [
    (256, 4, 8),
    (256, 8, 8),
    (128, 4, 8),
    (128, 16, 8),
]
# The NN/TN scale-delivery combo (asm tr8 + agpr + opsel byte-pack) is set
# per-layout in _get_mxfp8_launch; candidates vary only the tile.
_MXFP8_NN_TN_CANDIDATES = [
    (256, 4, 8),
    (256, 8, 8),
    (128, 8, 8),
    (128, 16, 8),
]
# AGPR budget for the asm tr8 + scale-prefetch spill; 32-128 within bench noise.
_MX_NN_TN_AGPR = 64
# asm-tr8 vmcnt_hint; matches tensorwise vmcnt(3). Bit-exact across 3/4/5 (pure schedule).
_MX_VMCNT = 3


_MXFP8_AUTOTUNE_CACHE: dict = {}  # (M,N,K,out_dtype,layout) -> (BLOCK_M, GROUP_M, num_xcd, group_n)


# Layout -> compile factory. All three share the (BLOCK_M, GROUP_M, num_xcd)
# candidate space and the same A-scale (ScaleS2R) / B-scale (ScaleBComb) loaders;
# only the data path (G2S offsets / swizzle / transpose loaders / MMA form)
# differs inside the compile fn. See module docstring / the compile fns.
_LAYOUT_COMPILE = {
    "nt": _compile_mxfp8_nt,
    "nn": _compile_mxfp8_nn,
    "tn": _compile_mxfp8_tn,
}


def _mx_pack(K):
    """opsel scale byte-pack factor: pack consecutive K-iters' E8M0 into one i32
    (read via opsel) -> pack-fold fewer scale VMEM loads. 4 if K_ITERS % 4 == 0,
    else 2 if even, else 1 (off). Requires the matching packed host preshuffle."""
    ki = K // 128
    return 4 if ki % 4 == 0 else (2 if ki % 2 == 0 else 1)


def _mx_tn_gn(N):
    """TN 2D N-band width: only very-large N (>=16384) benefits from band blocking
    (cuts the B re-stream); moderate N is neutral/slightly worse, so 0 there.
    Measured: 70B_GateUp N=28672 0.77->0.81; N<=11008 neutral."""
    return 8 if N >= 16384 else 0


def _mx_nt_gn_cands(N):
    """NT/TN 2D N-band candidate widths for the autotune stage-2 sweep. The seed
    band (NT gn=0, TN _mx_tn_gn(N)) is measured separately as the baseline, so the
    final pick can never regress. Only offer a band when there are >= 2*gn 256-col
    N-blocks (else the band can't create the cross-tile B reuse it exists for).
    Winners are shape- AND layout-dependent (NT: 7B GateUp gn16, 70B QKV gn8/16;
    TN: 70B QKV gn16 +6.4%, 70B GateUp gn4 +3.1%), so the per-shape bench picks
    rather than a single heuristic. Set env MX_DISABLE_NT_GN to force the seed band
    (A/B baseline: NT -> 1D swizzle, TN -> the shipping heuristic)."""
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
    # gn is the autotuned 2D N-band width (0 = 1D swizzle) for all three layouts.
    lk = (K, bm, gm, xcd, layout, gn)
    launch = _MXFP8_LAUNCH_CACHE.get(lk)
    if launch is None:
        if layout == "nt":  # plain-load path, no scale-pack / asm knobs
            launch = _LAYOUT_COMPILE[layout](
                K=K, BLOCK_M=bm, BLOCK_N=_BLOCK_N, GROUP_M=gm, group_n=gn, num_xcd=xcd
            )
        else:  # nn/tn: opsel scale byte-pack + asm B tr8 + agpr
            kwargs = dict(
                K=K,
                BLOCK_M=bm,
                BLOCK_N=_BLOCK_N,
                GROUP_M=gm,
                group_n=gn,
                num_xcd=xcd,
                scale_pack=_mx_pack(K),
                b_inline_asm=True,
                agpr_alloc=_MX_NN_TN_AGPR,
                vmcnt_hint=_MX_VMCNT,
            )
            if layout == "tn":
                # BLOCK_M=128 has the VGPR room for the FULL asm pipeline (asm A tr8 +
                # asm scaled MFMA); BLOCK_M=256 spills, so A stays intrinsic tr8 (B-asm
                # only). a_inline_asm/asm_mma are TN-only kwargs (NN compile has none).
                full_asm = bm == 128
                kwargs.update(a_inline_asm=full_asm, asm_mma=full_asm)
            launch = _LAYOUT_COMPILE[layout](**kwargs)
        _MXFP8_LAUNCH_CACHE[lk] = launch
    return launch


def _autotune_mxfp8(a8, b8, out_view, a_sc_u8, b_sp, M, N, K, out_dtype, layout="nt"):
    """First-call micro-bench of the candidates for (M,N,K,layout); cache the
    fastest cfg by shape. Returns (BLOCK_M, GROUP_M, num_xcd, group_n). Each
    candidate is compiled, finite-checked, then timed (2 warmup + 20 iter); the
    A-scale preshuffle fanout (n_tiles_a = BLOCK_M//64) differs per candidate so
    a_sp is rebuilt for each. NN/TN candidates fold the fixed scale-delivery combo
    into _get_mxfp8_launch, so only the tile (BM/GM/XCD) is swept.

    Two-stage for NT: stage 1 picks (BM,GM,XCD) at group_n=0 (1D swizzle); stage 2
    fixes that tile and sweeps the 2D N-band width group_n. gn=0 is measured in
    stage 1, so the staged pick can never regress vs the old (gn-less) NT path —
    it only captures the big-/mid-N L2-reuse win on shapes where a band helps."""
    key = (M, N, K, out_dtype, layout)
    cached = _MXFP8_AUTOTUNE_CACHE.get(key)
    if cached is not None:
        return cached
    cands = _MXFP8_NT_CANDIDATES if layout == "nt" else _MXFP8_NN_TN_CANDIDATES
    stream = torch.cuda.current_stream()
    pack = _mx_pack(K) if layout in ("nn", "tn") else 1  # NN/TN use opsel scale byte-pack

    def _time_cfg(bm, gm, xcd, gn):
        try:
            if pack > 1:
                a_sp = preshuffle_scale_pack(a_sc_u8, K, bm // 64, pack).reshape(-1)
            else:
                a_sp = preshuffle_scale(a_sc_u8, K, bm // 64).reshape(-1)
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

    # Stage 1: best (BLOCK_M, GROUP_M, num_xcd) at the layout's SEED band width.
    # NT seeds gn=0 (its native 1D swizzle); TN seeds the shipping _mx_tn_gn(N)
    # heuristic so the tile is chosen exactly as it was before this autotune — TN's
    # gn is tile-coupled, so re-selecting the tile at gn=0 could land on a tile the
    # band can't reuse. NN has no band (seed 0).
    seed_gn = _mx_tn_gn(N) if layout == "tn" else 0
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
    the MFMA (``v_mfma_scale_f32_16x16x128_f8f6f4``). Layout dispatch by
    ``(trans_a, trans_b)`` mirrors the tensorwise wrapper:
      - NT (F, T): A [M,K], B [N,K] (B^T storage), C = a @ b^T.
      - NN (F, F): A [M,K], B [K,N],                C = a @ b.
      - TN (T, F): A [K,M], B [K,N],                C = a^T @ b.
      - TT (T, T): unsupported (raises).

    The E8M0 scales are ALWAYS passed in the operand's natural [free, K//32]
    layout — ``a_scale`` is [M, K//32] (free = M) and ``b_scale`` is [N, K//32]
    (free = N) for every layout, because the MFMA distributes each operand's
    scale by lane independent of how the data tile was loaded (the transpose
    loaders reproduce the plain-load mfma operand byte layout). So the A-scale
    (ScaleS2R) and B-scale (ScaleBComb) loaders + preshuffles are layout-invariant.

    Args:
      a, b:     float8_e4m3fn, shapes per the layout above.
      a_scale:  E8M0 [M, K // 32] (raw bytes, free = M).
      b_scale:  E8M0 [N, K // 32] (raw bytes, free = N).
      out_dtype: bf16 (the kernel epilogue stores bf16).

    Constraints: K % 128 == 0 and K >= 256; M % 64 == 0; N % 256 == 0.
    """
    assert a.dim() == 2 and b.dim() == 2, "a, b must be 2D"
    assert out_dtype == torch.bfloat16, "mxfp8 FlyDSL store emits bf16 only"

    if (not trans_a) and trans_b:
        layout = "nt"
        M, K = a.shape
        N, Kb = b.shape
    elif (not trans_a) and (not trans_b):
        layout = "nn"
        M, K = a.shape
        Kb, N = b.shape
    elif trans_a and (not trans_b):
        layout = "tn"
        K, M = a.shape
        Kb, N = b.shape
    else:
        raise NotImplementedError(
            f"mxfp8 FlyDSL GEMM does not support the TT layout (trans_a={trans_a}, trans_b={trans_b})."
        )
    assert K == Kb, f"K mismatch: a {a.shape}, b {b.shape} (layout {layout})"
    assert K % 128 == 0 and K >= 256, f"K must be a multiple of 128 and >= 256, got {K}"
    assert M % 64 == 0, f"M must be a multiple of 64 (A-scale preshuffle), got {M}"
    assert N % 256 == 0, f"N must be a multiple of 256 (combined-B scale preshuffle), got {N}"

    # E8M0 raw bytes -> uint8 [free, K//32]; host pre-shuffle to packed int32.
    # NN uses opsel scale byte-pack (_mx_pack); NT/TN use the broadcast preshuffle.
    pack = _mx_pack(K) if layout in ("nn", "tn") else 1
    a_sc_u8 = a_scale.contiguous().view(torch.uint8).reshape(M, K // 32)
    b_sc_u8 = b_scale.contiguous().view(torch.uint8).reshape(N, K // 32)
    if pack > 1:
        b_sp = preshuffle_scale_b_comb_pack(b_sc_u8, K, pack).reshape(-1)  # BLOCK_M-independent
    else:
        b_sp = preshuffle_scale_b_comb(b_sc_u8, K).reshape(-1)

    out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    a8 = as_i8_flat(a)
    b8 = as_i8_flat(b)
    # Pass C as 2D [M, N] (NOT flat): FlyDSL packs each shape dim as int32, so a 1D
    # [M*N] view overflows when M*N > 2^31. StoreCPlain addresses C via its i64 base
    # (extract_base_index) + per-tile re-basing, so the 2D shape is only metadata.
    out_view = out

    # Per-shape cfg: first call benches the candidates (rebuilding a_sp per
    # BLOCK_M), caches the winner by (M,N,K,layout). The A-scale preshuffle fanout
    # (n_tiles_a = BLOCK_M//64) depends on the chosen BLOCK_M, so a_sp is built
    # AFTER the cfg is known.
    bm, gm, xcd, gn = _autotune_mxfp8(a8, b8, out_view, a_sc_u8, b_sp, M, N, K, out_dtype, layout)
    if pack > 1:
        a_sp = preshuffle_scale_pack(a_sc_u8, K, bm // 64, pack).reshape(-1)
    else:
        a_sp = preshuffle_scale(a_sc_u8, K, bm // 64).reshape(-1)
    launch = _get_mxfp8_launch(K, bm, gm, xcd, layout, N, gn)
    args = (a8, b8, out_view, a_sp, b_sp, M, N, torch.cuda.current_stream())
    get_compiled(_COMPILED_MXFP8_CACHE, launch, args)(*args)
    return out.t().contiguous() if trans_c else out
