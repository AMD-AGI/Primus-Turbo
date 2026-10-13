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

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MXFP4/MXFP6/MXFP8 A x MXFP4 B preshuffle GEMM (gfx950): per-32 E8M0 scales folded into
a scaled 16x16x128 fx.gemm; A streams global->LDS via double-buffered async DMA. Layout
matches the host preshuffle (shuffle_weight_w4(.,16) + shuffle_scale_w4).

FLUX copy of FlyDSL v0.2.4's kernels/gemm/mxfp4_preshuffle.py, used for the A6W4 forward
(FLUX_FP4_FPROP_CAST=a6w4) through `gemm_a6w4`. Installed as
primus_turbo/flydsl/gemm/gemm_a6w4_kernel.py."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, const_expr, gpu, math, range_constexpr, rocdl
from flydsl.expr.arith import _to_raw
from flydsl.expr.typing import (
    BFloat16,
    Constexpr,
    Float4E2M1FN,
    Float6E2M3FN,
    Float8E4M3FN,
    Float16,
    Float32,
    Int8,
    Int32,
    T,
)
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.quantization.mxfp4_quant_kernel_flux import (
    _DG_KB_TWO_LOG2E,
    _DG_KK,
)
from primus_turbo.flydsl.utils.gemm_epilogue_helper_main import (
    A6W4_COL_SCALE_BIAS,
    MXFP4ColQuantStore,
    MXFP6RowQuantStore,
)
from primus_turbo.flydsl.utils.gemm_helper import block_mn, run_compiled, xcd_remap_pid

_A_ELEM = {"fp4": Float4E2M1FN, "fp6": Float6E2M3FN, "fp8": Float8E4M3FN}
MX_BLOCK = 32  # values per E8M0 scale, row-wise and col-wise alike
# f8f6f4 operand format codes (the MFMA's cbsz / blgp fields).
_F8F6F4_FMT = {"fp4": 4, "fp6": 2, "fp8": 0}


def _mfma_scale_agpr(src0, src1, acc, sc0, sc1, os0, os1, fmt0, fmt1, agpr=True):
    """One v_mfma_scale_f32_16x16x128_f8f6f4 with D = C tied, in the AGPR or the arch-VGPR file.
    os0/os1 pick the E8M0 byte of sc0/sc1, fmt0/fmt1 are src0/src1's format codes. Fragments
    past the 256-dword AGPR file take agpr=False and live in the arch VGPRs."""
    v4f32 = ir.VectorType.get([4], ir.F32Type.get())
    osel = f"op_sel:[{os0 & 1},{os1 & 1},0] op_sel_hi:[{(os0 >> 1) & 1},{(os1 >> 1) & 1},0]"
    return _llvm.inline_asm(
        v4f32,
        [_to_raw(src0), _to_raw(src1), _to_raw(acc), _to_raw(sc0), _to_raw(sc1)],
        f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $0, $4, $5 {osel} cbsz:{fmt0} blgp:{fmt1}",
        ("=a,v,v,0,v,v" if agpr else "=v,v,v,0,v,v"),
        has_side_effects=False,
    )


def _scale_mma_atoms(a_dtype, b_first=False):
    """16 scaled-MFMA atoms keyed (opsel of A, opsel of B); A elem is fp4/fp6/fp8, B always fp4.
    b_first: B is the MFMA's first operand, so the accumulators hold C^T fragments."""
    elem_a = _A_ELEM[a_dtype]
    if b_first:
        return {
            (osa, osb): fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, Float4E2M1FN, elem_a, opsel_a=osb, opsel_b=osa)
            )
            for osa in range(4)
            for osb in range(4)
        }
    return {
        (osa, osb): fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, elem_a, Float4E2M1FN, opsel_a=osa, opsel_b=osb)
        )
        for osa in range(4)
        for osb in range(4)
    }


_PERMLANE_PAIR_TY = "!llvm.struct<(i32, i32)>"


def _permlane_swap(op, a_i32, b_i32):
    """One v_permlane*_swap_b32 as an intrinsic rather than inline asm.

    Both forms emit the same instruction, but the intrinsic is visible to the hazard
    recognizer, so it pads only the VALU writes and reads that actually need it. The
    asm form had to carry an unconditional `s_nop 1` on each side, which back-to-back
    swaps -- what the merge below is made of -- pay for every time.
    """
    r = op(ir.Type.parse(_PERMLANE_PAIR_TY), _to_raw(a_i32), _to_raw(b_i32), False, True)
    i32 = ir.IntegerType.get_signless(32)
    return _llvm.extractvalue(i32, r, [0]), _llvm.extractvalue(i32, r, [1])


def _permlane16_swap(a_i32, b_i32):
    """a's odd 16-lane rows <-> b's even rows."""
    return _permlane_swap(rocdl.permlane16_swap, a_i32, b_i32)


def _permlane32_swap(a_i32, b_i32):
    """a's lanes 32-63 <-> b's lanes 0-31."""
    return _permlane_swap(rocdl.permlane32_swap, a_i32, b_i32)


def _bq_view(arg_bq_addr, row_elems, KH4, k_tiles, k_halves):
    """Preshuffled B view for one N-row tile; index [l//16, l%16, kt, half, None] -> i32[4]."""
    col_base = rocdl.readfirstlane(T.i32, row_elems * KH4)
    i32_ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=16)
    off_i64 = fx.Int64(col_base)
    base_iter = fx.inttoptr(i32_ptr_ty, arg_bq_addr + off_i64 * fx.Int64(4))
    shape = (4, 16, k_tiles, k_halves, 4)
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout(shape, (64, 4, k_halves * 256, 256, 1))))
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


@flyc.jit
def launch_gemm(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_scale_a: fx.Pointer,
    arg_scale_b: fx.Pointer,
    i32_m: fx.Int32,
    i32_n: fx.Int32,
    stream: fx.Stream,
    N: Constexpr[int],
    K: Constexpr[int],
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    a_dtype: Constexpr[str],
    out_dtype: Constexpr[str],
    batch: Constexpr[int],
    a_row_stride: Constexpr[int],
    a_batch_stride: Constexpr[int],
    sca_row_stride: Constexpr[int],
    sca_batch_stride: Constexpr[int],
    c_row_stride: Constexpr[int],
    c_batch_stride: Constexpr[int],
    waves_per_eu: Constexpr[int],
    group_m: Constexpr[int],
    num_xcd: Constexpr[int],
    pad16: Constexpr[int],
    b_prefetch: Constexpr[int],
    sa_lds: Constexpr[int],
    epi: Constexpr[int],
    arg_bias: fx.Pointer,
    has_bias: Constexpr[bool],
    a_stages: Constexpr[int],
    a_pipe: Constexpr[int],
    fi: Constexpr[int],
    store_mode: Constexpr[int],
    vs_ahead: Constexpr[int],
    a_pack: Constexpr[int],
    gelu: Constexpr[int],
    keep_preact: Constexpr[int],
    act_rounded: Constexpr[int],
    arg_rq: fx.Pointer,
    arg_rs: fx.Pointer,
    arg_cq: fx.Pointer,
    arg_cs: fx.Pointer,
    rq: Constexpr[int],
):
    """Direct @flyc.jit launcher. Operands are fx.Pointer (pass ptr_arg(t): raw data_ptr, no
    per-launch DLPack). Compile once with flyc.compile, then cf(*runtime). a_dtype fp4/fp6/fp8
    A x preshuffled MXFP4 B, e8m0 scales. batch>1 = strided-batched over grid.z. The
    a_/sca_/c_ row/batch strides make A/scale_a/C addressing caller-controlled; each <0 keeps
    the contiguous [B,M,*] bmn default, all set = the [M,B,*] mbn layout. waves_per_eu<=0 = unset.
    group_m<=0 = native 2D grid (bid_x, bid_y); group_m>0 = flat 1D grid banded by block_mn.
    num_xcd>1 (group_m>0 only) remaps the flat id XCD-major first, so each XCD walks one
    contiguous run of the band order instead of every num_xcd-th tile.
    pad16==0 = dense A-tile LDS layout (byte-identical to the pre-R6 kernel); pad16>0 = D.1 LDS
    row padding (goal.md D.1), see A_LDS_B/CHUNK_ROWS below.
    b_prefetch selects the K loop: 0 = one K-tile per iteration, B loaded in the iteration that
    consumes it; 1/2 = two K-tiles per iteration with B loaded one half-iteration ahead (U2,
    see the loop below), 1 = next B/scales issued ahead of the A DMA and every half drained
    with vmcnt(0), 2 = issued behind the A DMA so each half drains only the DMA. 1/2 need
    tile_k=128 and an even K-tile count; otherwise the b_prefetch=0 loop is used.
    sa_lds=1 (U2 loop, 4 A-scale words per chunk, i.e. tile_m=128): wave w DMAs the chunk's
    A-scale word w into a double-buffered LDS slot and every wave reads all four back, instead
    of every wave loading the same four words; ignored where those conditions do not hold.
    epi selects the C store (same output bytes): 0 = one buffer_store_short per element;
    1 = row-merged, a permlane16_swap per n-fragment pair puts one row's 32 columns in each
    32-lane half, same store count, each store two 64 B row segments instead of four 32 B
    ones; 2 = MFMA operands swapped so a lane holds 4 consecutive columns of one row,
    cvt_pk + permlane16_swap pack 8 of them into one buffer_store_dwordx4 (16 stores per wave
    instead of 128), bias (has_bias) loaded in the prologue as one 8 B load per n-fragment.
    a_stages=3 (sa_lds U2 loop, chunk count a multiple of 3) = a three-slot A LDS ring: K-tile
    t's half issues the DMA of A(t+2), so every A tile has two halves to land instead of one;
    2 (or those conditions unmet) = the double-buffered U2 loop.
    a_pipe>0 (a_stages=3 ring only): each half reads m-row mi's A fragment a_pipe m-rows ahead
    of mi's MFMAs instead of reading all A fragments before the first MFMA; 0 = the latter.
    tile_m=256 keeps the 1x4 wave split (wave w owns the 256 x tile_n/4 C block w, so no B
    fragment is loaded by two waves), keeps the accumulators in AGPRs through inline-asm MFMAs
    (one wave per SIMD, so pass waves_per_eu=1) and, in the a_stages=3 ring with a_pipe=0, runs
    each half's MFMAs as one group per m-row with the half's loads issued in the gaps between
    the groups, reads each m-row's A fragment vs_ahead groups ahead, and (epi=2) runs the
    last chunk peeled with m-row g's C stores in the gap after group g+1 of its last half.
    fi (that tile_m=256 ring only) orders the pieces of m-row group g, each between
    sched_barrier(0) fences, with the MFMAs in the same order: 0 = the A read of m-row
    g+vs_ahead, the group's MFMAs, then the gap's load item; 1 = MFMA 0, A read, MFMA 1, load
    item, the other MFMAs; 2 = MFMA 0, load item, MFMA 1, A read, the other MFMAs; 3 = A read,
    MFMA 0, load item, the other MFMAs.
    store_mode (epi=2): 0 = each m-row stored as 16 rows x 64 B per n-fragment pair, the two
    halves of every 128 B line in separate stores; 1/2/3 = lanes j and j^8 of each 16-lane row
    trade halves (DPP row_ror:8) so two stores write 8 whole 128 B lines each, cached (1), nt (2)
    or sc0|nt (3). Same bytes. At tile_n=384 a wave owns 1.5 lines of each C row, so c_split
    deals it n-fragments 0..3 = its aligned line, 4..5 = the leftover half line (see c_line).
    a_pack (a_dtype=fp6 only) picks A's packing: 0 = the FP8-padded row-major form, 1 =
    quant_act_a6w4's dense preshuffle. Same MFMA operands either way.
    rq>0 (epi=2, tile_m=256, store_mode=0) also writes the next GEMM's A6W4 operands from
    the same accumulators, so the quantiser that would have read C back has nothing left to
    do: 1 = the row half only (arg_rq/arg_rs, the (aq, sa) pair of
    flydsl_quant_a6w4_act_dual); 2 = both halves, adding the col half (arg_cq/arg_cs, its
    (co, cs) pair). See store_mquad below for how a micro-block reaches one lane.
    """

    # Nested here, not beside its scalar original: the JIT cache key is this function's own
    # source plus the globals it names, so a body kept outside would never be keyed and an
    # edit to it would silently reuse the last compile.
    def _splat(v, c):
        """``c`` broadcast over ``v``'s shape, so a math builder sees a vector exemplar in args[0]."""
        return Vec.filled(v.numel, c, Float32)

    def _exp2_native(f):
        """One `v_exp_f32` per element: the hardware exp2 `__ocml_exp_f32` itself issues.
        `math.exp2` instead lowers to a denormal-range fixup around it, measured at two
        `v_cmp_f32`, two `v_cndmask_b32` and one `v_ldexp_f32` per element, all of them dead
        here: an `e` small enough to be denormal rounds `2*e*r - 1` to exactly -1, so the GELU
        returns +-0 whichever lowering runs.
        Checked rather than assumed: over all 2**32 f32 inputs the two give bit-identical GELU
        results (_probe_geluquiv.py `nat` arm, max f32 ulp 0)."""
        return Vec.from_elements(
            [
                _llvm.call_intrinsic(T.f32, "llvm.amdgcn.exp2.f32", [_to_raw(f[i])], [], [])
                for i in range(f.numel)
            ],
            Float32,
        )

    def _bf16_round_bias_vec(x, bias):
        """``x.to(out_elem).to(Float32) + bias``, two elements per convert instead of one.

        The round trip through bf16 is what makes the fused epilogue see the same input the
        separate pointwise pass would have read back from C, so it cannot be skipped. Written
        as `.to().to()` it costs one `v_cvt_pk_bf16_f32` whose upper half is thrown away plus
        one `v_lshlrev_b32` per element; `v_cvt_pk_bf16_f32` already rounds two elements at
        once, so feeding it a pair keeps the same two bf16 values and widening them back is a
        shift for the low one and a mask for the high one, with the bias then half a
        `v_pk_add_f32` each: 1.5 v_* per element.

        `v_dot2_f32_bf16` (`s0.lo*s1.lo + s0.hi*s1.hi + s2` over a bf16 pair with an f32
        addend) does the widen and the add of one half in a single instruction with a
        multiplier of (1, 0) or (0, 1), for 1.0 instead, and it does measure -0.5 v_* per
        element and -1.2% on fc1_double. It is not this add, though: its f32 accumulate is
        not the `v_add_f32` one. Both products here are exact (a bf16 times one is itself,
        times zero is a zero whose sign the three-term sum keeps exactly as `widen + bias`
        does), so only the final rounding is left to differ, and it does -- 12982 of 1e8 `aq`
        bytes and 2720 of 5e7 `co` bytes, the one-ulp input difference amplified by the GELU's
        own exp2. The lever needs an add whose rounding is pinned, not a dot product.

        Pinning the VOP3P encoding is a prerequisite either way: `llvm.amdgcn.fdot2.f32.bf16`
        selects as the VOP2 `v_dot2c_f32_bf16`, whose src2 is tied to the destination, and an
        n-column's bias is live across all 16 m-rows, so every use pays a `v_mov_b32` to copy
        it (-576 shift/mask/pk_add for +360 v_mov). Inline asm with the multiplier in an SGPR
        (one scalar source is all VOP3P allows, and the bias is already in a VGPR) gets the
        three-operand form with no copies and no extra VGPR."""
        bits = []
        for i in range_constexpr(x.numel // 2):
            w = rocdl.cvt_pk_bf16_f32(x[2 * i], x[2 * i + 1])
            bits.append(w << 16)
            bits.append(w & 0xFFFF0000)
        return Vec.from_elements(bits, Int32).bitcast(Float32) + bias

    def _bf16_round_vec(x):
        """``x.to(out_elem).to(Float32)``, the rounding C's own store applies."""
        bits = []
        for i in range_constexpr(x.numel // 2):
            w = rocdl.cvt_pk_bf16_f32(x[2 * i], x[2 * i + 1])
            bits.append(w << 16)
            bits.append(w & 0xFFFF0000)
        return Vec.from_elements(bits, Int32).bitcast(Float32)

    def _clamp01_vec(v):
        """``min(v, 1.0)`` for a non-negative ``v``, at half an instruction per element.

        gfx950's packed f32 ALU is V_PK_FMA/MUL/ADD/MOV_F32 only -- there is no packed min or
        max -- so `arith.minnumf` scalarises to one `v_min_f32` per element. Every VOP3P does
        carry the clamp bit, which is `min(max(., 0), 1)` on each half, and a multiply by 1.0
        is exact, so one `v_pk_mul_f32 .. clamp` is that same min on a pair. It has to be asm:
        `SIFoldOperands` folds a clamp idiom only into the instruction that *defines* the
        value, and this one's definer (the `exp2`) also feeds the `e + 1` beside it."""
        out = []
        for i in range_constexpr(v.numel // 2):
            pair = Vec.from_elements([v[2 * i], v[2 * i + 1]], Float32)
            clamped = Vec(
                _llvm.InlineAsmOp(
                    res=Vec.make_type(2, Float32),
                    operands_=[arith.as_ir_value(pair)],
                    asm_string="v_pk_mul_f32 $0, $1, 1.0 op_sel_hi:[1,0] clamp",
                    constraints="=v,v",
                    has_side_effects=False,
                ).result,
                (2,),
                Float32,
            )
            out.append(clamped[0])
            out.append(clamped[1])
        return Vec.from_elements(out, Float32)

    def _gelu1_torch_bf16_vec(x):
        """`_gelu1_torch_bf16` over a whole register fragment, bit-identical to calling it per
        element.

        It folds kBeta into the exponent's log2(e), so one `exp2` replaces the ocml tanh: gone
        are the polynomial branch, |v|, copysign, and the software range reduction around
        `__ocml_exp_f32` (`v_rndne_f32`, `v_cvt_i32_f32`, `v_ldexp_f32` and the `a - n` subtract,
        one of each per element). `tanh(u) = 1 - 2/(E+1)` with `E = exp(2u)`, written here with
        `e = exp(2u)` on both sides of a branch: each side is the one that does not cancel, so
        the reciprocal's rounding stays below the result's last bit instead of being amplified.
        Dropping the select and keeping only the `e >= 1` side -- correct for every u, and the
        side whose arithmetic the reference performs bit for bit -- is 4% faster and fails the
        gate: it perturbs x < 0 by up to one bf16 ulp of 0.17, and micro-blocks whose amax is
        small enough to resolve that do occur (127 of 1e8 `aq` bytes and 37 of 5e7 `co` bytes).

        The two are not the same function on all of f32 -- sweeping every one of the 2**32 bit
        patterns (_probe_geluquiv.py) the bf16 results differ on 3501 of them, always by one
        bf16 ulp, never on inf/nan, never for |x| >= 8, and never where |gelu(x)| > 2.79. On the
        scored distribution (sums of two bf16, 1.07e9 samples) the rate is 5.7e-8/element with
        |gelu(x)| <= 0.286, three orders of magnitude under a 32-element micro-block's amax, so
        both round to the same FP6/FP4 code and the quantised packs come out byte-identical."""
        a = math.fma(_splat(x, _DG_KK), (x * x) * x, x) * _DG_KB_TWO_LOG2E
        e = _exp2_native(a)
        d = e + _splat(x, 1.0)
        r = Vec.from_elements([Float32(rocdl.rcp(T.f32, _to_raw(d[i]))) for i in range(x.numel)], Float32)
        q = _clamp01_vec(e) * r
        t = math.fma(q, _splat(x, -2.0), _splat(x, 1.0))
        # Keep the tanh torch computes, not the cheaper sigmoid form: the gate is byte
        # identity and the two part by a bf16 ulp often enough to move `aq` codes.
        return x * math.fma(t, math.copysign(_splat(x, 0.5), x), _splat(x, 0.5))

    BM, BN, BK = tile_m, tile_n, tile_k
    if const_expr(out_dtype == "bf16"):
        out_elem = BFloat16
    else:
        out_elem = Float16

    # Row sizes + read_a fragment layout (i32 units): fp6/fp8 read two b128 halves -> i32[A_NDW], fp4 one -> i32[4].
    if const_expr(a_dtype == "fp4"):  # 2 codes/byte
        a_row_bytes, A_ROW_B = K // 2, BK // 2
        A_GK_I32, A_KH_I32, A_HI_OFF, A_NDW = 4, 16, 0, 4
    else:
        a_row_bytes, A_ROW_B = K, BK
        if const_expr(a_dtype == "fp8"):
            A_GK_I32, A_KH_I32, A_HI_OFF, A_NDW = 4, 32, 16, 8
        else:  # fp6
            A_GK_I32, A_KH_I32, A_HI_OFF, A_NDW = 8, 32, 4, 6

    # a_pack only moves where A sits in DRAM and LDS: both forms feed the MFMAs the same
    # bytes, so the product is bit-for-bit identical either way.
    a_planes = a_pack > 0 and a_dtype == "fp6"
    if const_expr(a_planes):
        a_row_bytes, A_ROW_B = K * 3 // 4, BK * 3 // 4

    # D.1 per-shape LDS row padding (R6, goal.md D.1): the R3 swizzle leaves a residual 2x LDS
    # bank conflict (SQ_LDS_IDX_ACTIVE/inst 8.00 vs a 4-cycle floor); inserting pad16*16 bytes of
    # unused LDS after every CHUNK_ROWS=1024//A_ROW_B rows (the DMA buffer_load_dwordx4...lds
    # cooperative load's per-wave contiguous-byte unit -- the only independently placeable
    # padding granularity, since each lane's LDS destination within one coop load is hardware-
    # fixed at M0+lane*16) eliminates it (measured: SQ_LDS_BANK_CONFLICT 66,060,288 -> 0 at
    # pad16=8, bitwise-identical output -- padding only moves WHERE A lives in LDS, not what data
    # flows). A_REAL_B (unpadded data volume) drives n_coop below and must NOT include padding
    # bytes, or the DMA would walk past the real A row. pad16==0 reduces this to the exact
    # pre-R6 dense formula (BM % CHUNK_ROWS == 0 for every tile_m used here, so the two are
    # numerically identical, not just approximately).
    N_PLANE_KH = 6  # 16 B units per 128-K tile row: 4 low halves + 2 paired high halves
    N_RBLK = BM // 64  # 64-row preshuffle blocks per A tile (one per wave at tile_m=256)
    A_RBLK_B = 64 * A_ROW_B  # bytes of one 64-row preshuffle block, gmem and LDS alike
    A_RBLK_I32 = A_RBLK_B // 4
    MI_PER_RBLK = 64 // 16  # read_a m-chunks (16 rows each) per 64-row block
    A_COOP_STEP = (256 // BM) * 1024  # gmem == LDS byte step between a wave's coop loads
    if const_expr(a_planes):
        CHUNK_ROWS, PAD_B = 0, 0
        CHUNK_B = 1024  # one wave's 64 lanes x 16 B of a cooperative load, contiguous in LDS
        A_REAL_B = A_LDS_B = BM * A_ROW_B
        swz_lds = False
        k_blk16 = 0
    else:
        CHUNK_ROWS = 1024 // A_ROW_B
        PAD_B = pad16 * 16
        CHUNK_B = CHUNK_ROWS * A_ROW_B + PAD_B
        assert BM % CHUNK_ROWS == 0, "D.1 padding requires CHUNK_ROWS | tile_m"
        assert 16 % CHUNK_ROWS == 0, "D.1 strength reduction requires CHUNK_ROWS | 16 (one mi-step)"
        A_REAL_B = BM * A_ROW_B  # real (unpadded) A-tile bytes -- drives n_coop, independent of pad16
        A_LDS_B = (BM // CHUNK_ROWS) * CHUNK_B  # padded LDS bytes per A-tile (== A_REAL_B at pad16=0)
        swz_lds = True
        k_blk16 = A_ROW_B // 16
        if const_expr(pad16 > 0):
            assert k_blk16 == CHUNK_ROWS, "D.1 swizzle-key reuse requires k_blk16 == CHUNK_ROWS"
    CHUNK_I32 = CHUNK_B // 4
    A_ROW_I32 = A_ROW_B // 4
    PLANE_I32 = 256  # a_planes: 16 B x 64 rows per plane, in i32 units
    K_HALF = K // 2
    KH4 = K_HALF // 4
    K_TILES = K // BK
    GY = N // BN  # N-tile count for the group_m raster band (R4); N and BN are both Constexpr
    k_halves = BK // 128  # 16x16x128 MFMA k-steps per K-tile
    # e8m0 scales are 256-K granular, B 128-K: tiles_per_chunk K-tiles share a word (hi/lo 16b = 128-K half).
    tiles_per_chunk = 256 // BK  # 1 for tile_k=256, 2 for tile_k=128
    # tile_m=256: all BM rows x BN/4 columns per wave as at tile_m=128, accumulators in AGPRs.
    bm256 = BM == 256
    m_chunks = BM // 16
    num_acc_n = BN // 4 // 16  # 16-col n-subblocks per wave
    # Past 128 accumulator dwords the arch-VGPR file cannot hold them and the operands
    # both, so the MFMAs must be the asm form whose D = C is pinned to an AGPR.
    agpr_acc = (BM // 16) * num_acc_n * 4 > 128
    n_agpr_ni = min(num_acc_n, 256 // ((BM // 16) * 4)) if agpr_acc else num_acc_n
    n_sa_words = BM // 32  # A-scale words per 256-K chunk for the whole workgroup
    n_sa_dma = max(1, n_sa_words // 4)  # sa_lds A-scale DMAs per wave per chunk
    _scale_chunk_dw = (K // 32 // 4 // 2) * 64  # e8m0 stride (dwords), per shuffle_scale_w4
    _scale_k0_dw = 64
    n_coop = A_REAL_B // 256 // 16  # 16B cooperative loads per thread (unpadded data volume)
    n_pairs = max(1, num_acc_n // 2)
    m_pairs = max(1, m_chunks // 2)
    c_line = store_mode > 0 and epi == 2 and num_acc_n in (4, 6)
    c_split = c_line and num_acc_n == 6
    # The row pack needs epi=2's C^T fragments, four m-rows at a time; asserted below.
    if const_expr(rq):
        assert epi == 2 and not c_line, "rq needs the epi=2 store_mode=0 C store"
        assert m_chunks % 4 == 0 and num_acc_n % 2 == 0, "rq quantises four m-rows at a time"
        assert BN % 128 == 0 and N % BN == 0, "rq needs 32-column micro-blocks inside the tile"
    # U2 K loop (b_prefetch 1/2, described at the loop below): two K-tiles per 256-K scale chunk.
    u2 = b_prefetch > 0 and tiles_per_chunk == 2 and K_TILES % 2 == 0
    # sa_lds maps A-scale word w (and w + 4, .. at tile_m=256) of a chunk to wave w and every
    # wave reads all n_sa_words words back, so it needs a multiple of 4 words.
    sa_lds_on = sa_lds > 0 and u2 and m_pairs == n_sa_words and n_sa_words % 4 == 0
    # a_stages=3 (sa_lds U2 loop, chunk count a multiple of 3): three A LDS slots, each its own
    # LDS symbol; the loop runs three chunks per iteration so every half addresses its slots
    # through a fixed symbol (see the loop below).
    a3 = a_stages == 3 and sa_lds_on and (K_TILES // 2) % 3 == 0
    n_sc_slots = 3 if a3 else 2
    SC_LDS_B = n_sc_slots * n_sa_words * 256  # slots x n_sa_words words x 64 lanes x 4 B
    # tile_m=256 ring: the half's loads go between one-m-row MFMA groups, in vs_np parts of B
    vsplit = bm256 and a3 and a_pipe == 0
    vs_np = m_chunks // 2
    vs_b = [list(range(p * num_acc_n // vs_np, (p + 1) * num_acc_n // vs_np)) for p in range(vs_np)]
    vs_c = [list(range(p * n_coop // vs_np, (p + 1) * n_coop // vs_np)) for p in range(vs_np)]

    # Scheduler counts per loop iter: MFMAs, A LDS reads/thread (fp6/fp8 2 per (mi,kh)), gmem loads.
    sched_mfma_total = k_halves * m_chunks * num_acc_n
    n_mfma_hints = 0 if agpr_acc else sched_mfma_total
    if const_expr(a_dtype == "fp4"):
        a_ds_per = 1
    else:
        a_ds_per = 2
    sched_num_ds_load = m_chunks * k_halves * a_ds_per
    sched_num_gmem = n_coop + num_acc_n * k_halves + m_pairs + n_pairs

    if const_expr(rq >= 2):
        assert bm256 and a3, "the col pack's band is the 1x4 wave split's accumulator quad"
        assert BM % 64 == 0 and (BN // 4) % 32 == 0, "the col pack needs whole micro-blocks"
    CQ_LDS_B = 4 * 4 * MXFP4ColQuantStore.lds_words(64, BN // 4) if rq >= 2 else 0

    if const_expr(a3 and rq >= 2):

        @fx.struct
        class SharedA:
            a0: fx.Array[Int8, A_LDS_B, 16]
            a1: fx.Array[Int8, A_LDS_B, 16]
            a2: fx.Array[Int8, A_LDS_B, 16]
            sc: fx.Array[Int8, SC_LDS_B, 16]
            cq: fx.Array[Int8, CQ_LDS_B, 16]

    elif const_expr(a3):

        @fx.struct
        class SharedA:
            a0: fx.Array[Int8, A_LDS_B, 16]
            a1: fx.Array[Int8, A_LDS_B, 16]
            a2: fx.Array[Int8, A_LDS_B, 16]
            sc: fx.Array[Int8, SC_LDS_B, 16]

    elif const_expr(sa_lds_on):

        @fx.struct
        class SharedA:
            a0: fx.Array[Int8, A_LDS_B, 16]
            a1: fx.Array[Int8, A_LDS_B, 16]
            sc: fx.Array[Int8, SC_LDS_B, 16]

    else:

        @fx.struct
        class SharedA:
            a0: fx.Array[Int8, A_LDS_B, 16]
            a1: fx.Array[Int8, A_LDS_B, 16]

    @flyc.kernel
    def kernel_gemm(
        arg_c: fx.Int64,
        arg_a: fx.Int64,
        arg_b: fx.Int64,
        arg_scale_a: fx.Int64,
        arg_scale_b: fx.Int64,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        arg_bias: fx.Int64,
        arg_rq: fx.Int64,
        arg_rs: fx.Int64,
        arg_cq: fx.Int64,
        arg_cs: fx.Int64,
    ):
        scale_atoms = _scale_mma_atoms(a_dtype, b_first=epi == 2)

        tid = fx.Int32(fx.thread_idx.x)
        bid_x, bid_y, bid_z = fx.block_idx
        wave = rocdl.readfirstlane(T.i32, tid // 64)
        if const_expr(a_planes):
            wave_lds_b = rocdl.readfirstlane(T.i32, (wave % N_RBLK) * A_RBLK_B + (wave // N_RBLK) * 1024)
        else:
            wave_lds_b = rocdl.readfirstlane(T.i32, wave * CHUNK_B)

        def wave_col0():
            # First C column of this wave within the tile.
            return wave * (BN // 4)

        if const_expr(c_split):
            n_line_c = (wave % 2) * 32
            n_half_c = 64 - n_line_c * 2

        def n_col(ni):
            if const_expr(c_split):
                return n_line_c + ni * 16 if ni < 4 else n_half_c + (ni - 4) * 16
            return ni * 16

        lane = tid % 64
        lane_div_16 = lane // 16
        lane_mod_16 = lane % 16
        # Per-shape group_m L2 band raster (R4, goal.md SS C): group_m==0 keeps today's native
        # 2D grid, (pid_m, pid_n) = (bid_x, bid_y), byte-for-byte unchanged. group_m>0 decodes a
        # flat 1D grid id (launched as (gx*gy, 1, batch) below) through block_mn's 1D GROUP_M
        # band (GN=0) -- the same bijection gemm_fp8_kernel.py's dense NT/NN GEMMs already use
        # for L2 reuse. block_mn's arith.select(rem_m < GM, rem_m, GM) is the gsize =
        # min(gx-first_m, gm) clamp that keeps the M=M_ALIGN tail band correct for any group_m.
        # num_xcd>1: the dispatcher deals workgroups round-robin over the XCDs (each with its own
        # L2), so xcd_remap_pid (a bijection over the gx*gy launched ids) hands each XCD one
        # contiguous run of the band order before block_mn decodes it.
        if const_expr(group_m > 0):
            num_pid_m = (i32_m + (BM - 1)) // BM
            if const_expr(num_xcd > 1):
                pid_flat = xcd_remap_pid(bid_x, num_pid_m * GY, num_xcd)
            else:
                pid_flat = bid_x
            pid_m, pid_n = block_mn(pid_flat, num_pid_m, GY, group_m, 0)
        else:
            pid_m, pid_n = bid_x, bid_y
        bx_m = pid_m * BM
        by_n = pid_n * BN

        # Strided-batched: shift each base to batch bid_z (A/scale_a via explicit strides or
        # the contiguous default; B/scale_b stay batch-contiguous). batch==1 emits no batch math.
        if const_expr(batch > 1):
            a_rstride = fx.Int32(a_row_bytes if a_row_stride < 0 else a_row_stride)
            sca_rstride = fx.Int32(_scale_chunk_dw if sca_row_stride < 0 else sca_row_stride)
            bz = fx.Int64(bid_z)
            if const_expr(a_batch_stride < 0):
                arg_a = arg_a + bz * (fx.Int64(i32_m) * fx.Int64(a_row_bytes))
            else:
                arg_a = arg_a + bz * fx.Int64(a_batch_stride)
            arg_b = arg_b + bz * fx.Int64(N * (K // 2))
            if const_expr(sca_batch_stride < 0):
                sc_bstride = fx.Int64((i32_m + 31) // 32) * fx.Int64(_scale_chunk_dw) * fx.Int64(4)
                arg_scale_a = arg_scale_a + bz * sc_bstride
            else:
                arg_scale_a = arg_scale_a + bz * fx.Int64(sca_batch_stride)
            arg_scale_b = arg_scale_b + bz * fx.Int64((N // 32) * _scale_chunk_dw * 4)
        else:
            a_rstride = fx.Int32(a_row_bytes)
            sca_rstride = fx.Int32(_scale_chunk_dw)

        _i8g = fx.PointerType.get(T.i8, address_space=fx.AddressSpace.Global, alignment=16)
        if const_expr(batch > 1 and a_row_stride >= 0):
            a_nrec = fx.Int64(i32_m - fx.Int32(1)) * fx.Int64(a_rstride) + fx.Int64(a_row_bytes)
        else:
            a_nrec = fx.Int64(i32_m) * fx.Int64(a_row_bytes)
        a_flat_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(
                    fx.make_view(
                        fx.inttoptr(_i8g, arg_a),
                        fx.make_layout(65536 * a_row_bytes, 1),
                    )
                ),
                max_size=False,
                num_records_bytes=a_nrec,
            ),
            fx.make_layout(1, 1),
        )
        if const_expr(a_planes):
            a_coop_base = (bx_m + (wave % N_RBLK) * 64) * a_rstride + (wave // N_RBLK) * 1024 + lane * 16
        lds = fx.SharedAllocator().allocate(SharedA).peek()
        # A-LDS modeled as i32 (16B = 4 i32): fx.copy is dtype-agnostic, only the MMA cares.
        sA0_i32 = fx.recast_iter(Int32, lds.a0.ptr)
        lds_db = fx.Int32(fx.ptrtoint(lds.a1.ptr)) - fx.Int32(
            fx.ptrtoint(lds.a0.ptr)
        )  # ping/pong byte stride
        lds_db_i32 = lds_db // 4
        lds_copy = fx.make_copy_atom(fx.UniversalCopy128b(), Int32)
        lds_copy64 = fx.make_copy_atom(fx.UniversalCopy64b(), Int32)
        dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        _i8s = fx.PointerType.get(Int8.ir_type, fx.AddressSpace.Shared, 512)
        sA0_i8 = fx.recast_iter(_i8s, lds.a0.ptr)
        if const_expr(a3):
            # Each slot keeps its own symbol, so an access through it carries that symbol's alias
            # scope: the waitcnt pass then lets the next slot's DMA stay in flight across a slot's
            # ds_reads instead of draining it (which it does for any DMA into the same symbol).
            a_slots = (lds.a0.ptr, lds.a1.ptr, lds.a2.ptr)
            slot_i32 = [fx.recast_iter(Int32, p) for p in a_slots]
            slot_i8 = [fx.recast_iter(_i8s, p) for p in a_slots]

        def _iter_of(parity):  # parity in {0,1} (runtime) -> i32 LDS iterator
            return fx.add_offset(sA0_i32, parity * lds_db_i32)

        def _lds_view(base_iter, off_i32):
            return fx.make_view(fx.add_offset(base_iter, off_i32), fx.make_layout(4, 1))

        # Async A: gmem->LDS DMA (buffer_load_lds); issued after B/scale loads to overlap the MFMAs.
        # D.1 (pad16>0): the per-wave LDS destination base/stride use the padded CHUNK_B instead
        # of the dense 1024B (64 lanes * 16B); `wave` is a readfirstlane'd per-wave scalar, so
        # this multiply runs on SALU, parallel to the VALU work -- source (gmem) addressing below
        # is completely unaffected by padding, since only the LDS destination placement changes.
        #
        # R8 (goal.md A.3 WAR-drain fix): the K-tile index used to land here as a per-lane VALU
        # add (`gmem_byte = ... + kt*A_ROW_B + ...`), so the per-lane voffset value was a *fresh*
        # SSA value every loop iteration even though `(bx_m+row)*a_rstride+col` on its own does
        # not depend on kt -- LLVM cannot treat "invariant + loop-varying-scalar" as invariant,
        # so it recomputed (and re-lived) those voffset VGPRs every iteration, and the register
        # allocator was then free to recycle the same physical VGPRs for the scale-load
        # destinations later in the same iteration -- the exact WAR hazard goal.md A.3 measured
        # (4 `s_waitcnt vmcnt(N)` drains between the A-DMA issue and the first MFMA, draining
        # 9-of-14 in-flight VMEM ops just to make those registers safe to reuse).
        #
        # kt rides the MUBUF soffset SGPR, keeping the per-lane address a true loop invariant.
        # soffset escapes raw-buffer bounds checking, so the row term stays the ragged-M guard.
        def dma_a_to_lds(kt, parity, slot_i8=None, irange=None):
            # slot_i8 (a3): the destination slot's own LDS symbol; otherwise buffer `parity` off a0.
            if const_expr(irange is not None and len(irange) == 0):
                return
            if const_expr(slot_i8 is not None):
                lds_ptr0 = fx.add_offset(slot_i8, wave_lds_b)
            else:
                base_off = rocdl.readfirstlane(T.i32, parity * lds_db) + wave_lds_b
                lds_ptr0 = fx.add_offset(sA0_i8, base_off)
            kt_soff = fx.Int32(kt) * fx.Int32(A_ROW_B * (64 if a_planes else 1))
            for i in range_constexpr(n_coop):
                if const_expr(irange is not None and i not in irange):
                    continue
                imm = 0
                if const_expr(a_planes):
                    win, imm = divmod(i * A_COOP_STEP, 4096)
                    assert imm <= 4095, "MUBUF inst_offset must fit 12 bits"
                    lds_off, gmem_byte = win * 4096, a_coop_base + win * 4096
                else:
                    lds_off = i * 4 * CHUNK_B
                    lin = (i * 256 + tid) * 16
                    row = lin // A_ROW_B
                    col = lin % A_ROW_B
                    if const_expr(swz_lds):
                        col = col ^ ((row % k_blk16) * 16)
                    gmem_byte = (bx_m + row) * a_rstride + col
                lds_ptr = lds_ptr0 if const_expr(lds_off == 0) else fx.add_offset(lds_ptr0, fx.Int32(lds_off))
                dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
                src = fx.slice(a_flat_div, (None, gmem_byte))
                fx.copy(dma_atom, src, dst, soffset=kt_soff, imm_offset=imm)

        def _read16(base_iter, off_i32):
            # ds_read_b128 straight into an i32[4] register fragment.
            t = fx.make_rmem_tensor(4, Int32)
            fx.copy(lds_copy, _lds_view(base_iter, off_i32), t)
            return t

        def _read8(base_iter, off_i32):
            # ds_read_b64 into an i32[2] register fragment.
            t = fx.make_rmem_tensor(2, Int32)
            fx.copy(lds_copy64, fx.make_view(fx.add_offset(base_iter, off_i32), fx.make_layout(2, 1)), t)
            return t

        if const_expr(a_planes):
            lane_lo_off = lane_div_16 * PLANE_I32 + lane_mod_16 * 4
            lane_hi_off = (4 + lane_div_16 // 2) * PLANE_I32 + lane_mod_16 * 4 + (lane_div_16 % 2) * 2
            mi_step = 64
        elif const_expr(pad16 > 0):
            lane_chunk_off = (lane_mod_16 // CHUNK_ROWS) * CHUNK_I32 + (lane_mod_16 % CHUNK_ROWS) * A_ROW_I32
            swz_key = lane_mod_16 % CHUNK_ROWS
            mi_step = (16 // CHUNK_ROWS) * CHUNK_I32  # i32 units; Python int: trace-time constant
        else:
            lane_chunk_off = lane_mod_16 * A_ROW_I32
            swz_key = lane_mod_16 % k_blk16
            mi_step = (16 // CHUNK_ROWS) * CHUNK_I32

        def _a_off(mi, kh, hi_half):
            if const_expr(a_planes):
                base = (
                    (lane_hi_off if hi_half else lane_lo_off)
                    + (mi // MI_PER_RBLK) * A_RBLK_I32
                    + (mi % MI_PER_RBLK) * mi_step
                )
                return base + kh * (N_PLANE_KH * PLANE_I32)
            row_base = lane_chunk_off + mi * mi_step
            blk = kh * (A_KH_I32 // 4) + lane_div_16 * (A_GK_I32 // 4) + hi_half * (A_HI_OFF // 4)
            if const_expr(swz_lds):
                return row_base + (blk ^ swz_key) * 4
            return row_base + kh * A_KH_I32 + lane_div_16 * A_GK_I32 + hi_half * A_HI_OFF

        def read_a(parity, base_iter=None, mis=None):
            # base_iter (a3): the slot's own LDS symbol; otherwise buffer `parity` off a0.
            # mis: read only these m-rows (the others' entries are None).
            if const_expr(base_iter is None):
                base_iter = _iter_of(parity)
            mi_list = list(range(m_chunks)) if mis is None else list(mis)
            av = []
            for mi in range_constexpr(m_chunks):
                if const_expr(mi not in mi_list):
                    av.extend([None] * k_halves)
                    continue
                for kh in range_constexpr(k_halves):
                    lo16 = _read16(base_iter, _a_off(mi, kh, 0))
                    if const_expr(a_dtype == "fp4"):
                        av.append(lo16)
                        continue
                    lo = Vec(fx.memref_load_vec(lo16))
                    if const_expr(A_NDW == 6):
                        hi = Vec(fx.memref_load_vec(_read8(base_iter, _a_off(mi, kh, 1))))
                    else:
                        hi = Vec(fx.memref_load_vec(_read16(base_iter, _a_off(mi, kh, 1))))
                    t = fx.make_rmem_tensor(A_NDW, Int32)
                    t.store(lo.shuffle(hi, list(range(A_NDW))))
                    av.append(t)
            return av

        n_col_base = by_n + wave_col0()
        bq_views = [
            _bq_view(arg_b, n_col_base + n_col(ni), KH4, K_TILES, k_halves)
            for ni in range_constexpr(num_acc_n)
        ]
        b_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)
        bs_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        # e8m0 scale buffers bounded to real size (OOB rows read 0); scale_a to the last 32-row chunk.
        _i32g = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
        _sc_layout = fx.make_layout(1 << 28, 1)
        _a_sc_chunks = (i32_m + 31) // 32
        if const_expr(batch > 1 and sca_row_stride >= 0):
            a_sc_nrec = (
                fx.Int64(_a_sc_chunks - 1) * fx.Int64(sca_rstride) + fx.Int64(_scale_chunk_dw)
            ) * fx.Int64(4)
        else:
            a_sc_nrec = fx.Int64(_a_sc_chunks) * fx.Int64(_scale_chunk_dw) * fx.Int64(4)
        b_sc_nrec = fx.Int64((N // 32) * _scale_chunk_dw * 4)
        sa_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.inttoptr(_i32g, arg_scale_a), _sc_layout)),
                max_size=False,
                num_records_bytes=a_sc_nrec,
            ),
            fx.make_layout(1, 1),
        )
        sb_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.inttoptr(_i32g, arg_scale_b), _sc_layout)),
                max_size=False,
                num_records_bytes=b_sc_nrec,
            ),
            fx.make_layout(1, 1),
        )
        a_sc_base = [
            rocdl.readfirstlane(T.i32, (bx_m // 32 + mp) * sca_rstride) for mp in range_constexpr(m_pairs)
        ]
        nsb = by_n // 32 + wave * (BN // 128)
        b_sc_base = [
            rocdl.readfirstlane(T.i32, (nsb + n_col(2 * np) // 32) * _scale_chunk_dw)
            for np in range_constexpr(n_pairs)
        ]
        sc_lane = lane_div_16 * 16 + lane_mod_16

        if const_expr(sa_lds_on):
            # LDS slot s holds a chunk's m_pairs A-scale words, word mp at s*m_pairs*256 + mp*256 B.
            sc_dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS32b(), 32)
            lds_copy32 = fx.make_copy_atom(fx.UniversalCopy32b(), Int32)
            sa_flat_i8 = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(
                    fx.Tensor(fx.make_view(fx.inttoptr(_i8g, arg_scale_a), fx.make_layout(1 << 30, 1))),
                    max_size=False,
                    num_records_bytes=a_sc_nrec,
                ),
                fx.make_layout(1, 1),
            )
            sSC_i8 = fx.recast_iter(_i8s, lds.sc.ptr)
            sSC_i32 = fx.recast_iter(Int32, lds.sc.ptr)
            a_sc_base_w = (bx_m // 32 + wave) * sca_rstride
            sa_dma_lds_b = [
                rocdl.readfirstlane(T.i32, (wave if r == 0 else wave + 4 * r) * 256)
                for r in range_constexpr(n_sa_dma)
            ]
            sa_dma_soff_b = [
                rocdl.readfirstlane(
                    T.i32, (a_sc_base_w if r == 0 else a_sc_base_w + (4 * r) * sca_rstride) * 4
                )
                for r in range_constexpr(n_sa_dma)
            ]

        def dma_sa_to_lds(chunk_kt, slot):
            # Wave w DMAs words w, w + 4, .. of the chunk (64 lanes x 4 B); drained like the A DMA
            koff_b = rocdl.readfirstlane(T.i32, chunk_kt * (_scale_k0_dw * 4))
            if const_expr(isinstance(slot, int)):
                slot_b = slot * (n_sa_words * 256)
            else:
                slot_b = rocdl.readfirstlane(T.i32, slot * (n_sa_words * 256))
            for r in range_constexpr(n_sa_dma):
                lds_ptr = fx.add_offset(sSC_i8, sa_dma_lds_b[r] + slot_b)
                soff = sa_dma_soff_b[r] + koff_b
                src = fx.slice(sa_flat_i8, (None, sc_lane * 4))
                fx.copy(sc_dma_atom, src, fx.make_view(lds_ptr, fx.make_layout(1, 1)), soffset=soff)

        def read_sa(slot):
            sa = []
            for mp in range_constexpr(m_pairs):
                t = fx.make_rmem_tensor(1, Int32)
                off = slot * (m_pairs * 64) + mp * 64 + sc_lane
                fx.copy(lds_copy32, fx.make_view(fx.add_offset(sSC_i32, off), fx.make_layout(1, 1)), t)
                sa.append(Vec(fx.memref_load_vec(t))[0])
            return sa

        n_acc = m_chunks * num_acc_n

        def load_b(kt, nis=None):
            # buffer_load_dwordx4 straight into i32[4] register fragments; nis: only these n-frags.
            ops = []
            for ni in range_constexpr(num_acc_n):
                if const_expr(nis is not None and ni not in nis):
                    continue
                for kh in range_constexpr(k_halves):
                    bf = fx.make_rmem_tensor(4, Int32)
                    fx.copy_atom_call(b_copy, bq_views[ni][lane_div_16, lane_mod_16, kt, kh, None], bf)
                    ops.append(bf)
            return ops

        def load_sc(chunk_kt):
            # (sa, sb) e8m0 words per m-/n-pair for one 256-K chunk (uniform base -> SGPR soffset).
            koff = rocdl.readfirstlane(T.i32, chunk_kt * _scale_k0_dw)
            sa = (
                []
                if sa_lds_on
                else [
                    Vec(
                        fly.copy_atom_call_ssa(
                            [T.vec(1, T.i32)],
                            bs_copy,
                            sa_flat[
                                None,
                                a_sc_base[mp] + koff + sc_lane,
                            ],
                        )
                    )[0]
                    for mp in range_constexpr(m_pairs)
                ]
            )
            sb = [
                Vec(
                    fly.copy_atom_call_ssa(
                        [T.vec(1, T.i32)],
                        bs_copy,
                        sb_flat[
                            None,
                            b_sc_base[np] + koff + sc_lane,
                        ],
                    )
                )[0]
                for np in range_constexpr(n_pairs)
            ]
            return sa, sb

        def compute(accs, av, bv, sa_v, sb_v, scale_shift=None, mis=None, nis=None):
            # mis: run only these m-rows' MFMAs; nis (tile_m=256 only): only these n-fragments.
            os_half = 0
            if const_expr(isinstance(scale_shift, int)):
                os_half = scale_shift // 8
                assert os_half + 2 * k_halves <= 4, "opsel has 4 e8m0 byte slots per scale word"
            elif const_expr(scale_shift is not None):
                sa_v = [v.shrui(scale_shift) for v in sa_v]
                sb_v = [v.shrui(scale_shift) for v in sb_v]
            # kh OUTERMOST: consecutive MFMAs hit distinct accumulators (dense issue). Each
            # scaled MFMA = fx.gemm over rank-1 i32[4] A/B frags, e8m0 word on scale_a=/scale_b=.
            mi_set = list(range(m_chunks)) if mis is None else list(mis)
            if const_expr(agpr_acc):
                # The same MFMAs in the same order, as inline asm whose D = C is tied to an AGPR.
                fmt_a = _F8F6F4_FMT[a_dtype]
                a_vals = {}
                for mi in mi_set:
                    for kh in range_constexpr(k_halves):
                        a_vals[mi * k_halves + kh] = Vec(
                            fx.memref_load_vec(av[mi * k_halves + kh])
                        ).ir_value()
                b_vals = [Vec(fx.memref_load_vec(t)).ir_value() for t in bv]
                ni_set = list(range(num_acc_n)) if nis is None else list(nis)
                for kh in range_constexpr(k_halves):
                    for ni in ni_set:
                        np_i, in_b = ni // 2, ni % 2
                        for mi in mi_set:
                            mp_i, im = mi // 2, mi % 2
                            idx = mi * num_acc_n + ni
                            osa, osb = os_half + kh * 2 + im, os_half + kh * 2 + in_b
                            a_f, b_f = a_vals[mi * k_halves + kh], b_vals[ni * k_halves + kh]
                            in_agpr = ni < n_agpr_ni
                            if const_expr(epi == 2):
                                accs[idx] = _mfma_scale_agpr(
                                    b_f,
                                    a_f,
                                    accs[idx],
                                    sb_v[np_i],
                                    sa_v[mp_i],
                                    osb,
                                    osa,
                                    _F8F6F4_FMT["fp4"],
                                    fmt_a,
                                    in_agpr,
                                )
                            else:
                                accs[idx] = _mfma_scale_agpr(
                                    a_f,
                                    b_f,
                                    accs[idx],
                                    sa_v[mp_i],
                                    sb_v[np_i],
                                    osa,
                                    osb,
                                    fmt_a,
                                    _F8F6F4_FMT["fp4"],
                                    in_agpr,
                                )
                return accs
            c_frags = [None] * n_acc
            for mi in mi_set:
                for ni in range_constexpr(num_acc_n):
                    c_frags[mi * num_acc_n + ni] = fx.make_rmem_tensor(4, Float32)
                    c_frags[mi * num_acc_n + ni].store(Vec(accs[mi * num_acc_n + ni]))
            for kh in range_constexpr(k_halves):
                for ni in range_constexpr(num_acc_n):
                    np_i, in_b = ni // 2, ni % 2
                    for mi in mi_set:
                        mp_i, im = mi // 2, mi % 2
                        cf = c_frags[mi * num_acc_n + ni]
                        atom = scale_atoms[(os_half + kh * 2 + im, os_half + kh * 2 + in_b)]
                        if const_expr(epi == 2):
                            fx.gemm(
                                atom,
                                cf,
                                bv[ni * k_halves + kh],
                                av[mi * k_halves + kh],
                                cf,
                                scale_a=sb_v[np_i],
                                scale_b=sa_v[mp_i],
                            )
                        else:
                            fx.gemm(
                                atom,
                                cf,
                                av[mi * k_halves + kh],
                                bv[ni * k_halves + kh],
                                cf,
                                scale_a=sa_v[mp_i],
                                scale_b=sb_v[np_i],
                            )
            for mi in mi_set:
                for ni in range_constexpr(num_acc_n):
                    accs[mi * num_acc_n + ni] = c_frags[mi * num_acc_n + ni].load().ir_value()
            return accs

        def compute_a_pipe(accs, base_iter, bv, sa_v, sb_v, scale_shift=None):
            # a_pipe: m-row mi's A fragment is read a_pipe groups ahead of mi's MFMAs; the
            # sched_barrier(0) fences keep the groups in order, so the backend's lgkmcnt waits
            # before each group are partial instead of one full drain before the first MFMA.
            if const_expr(scale_shift is not None and not isinstance(scale_shift, int)):
                sa_v = [v.shrui(scale_shift) for v in sa_v]
                sb_v = [v.shrui(scale_shift) for v in sb_v]
                scale_shift = None
            frags = {}
            for mi in range_constexpr(min(a_pipe, m_chunks)):
                frags[mi] = read_a(None, base_iter, [mi])
            for mi in range_constexpr(m_chunks):
                if const_expr(mi + a_pipe < m_chunks):
                    frags[mi + a_pipe] = read_a(None, base_iter, [mi + a_pipe])
                rocdl.sched_barrier(0)
                accs = compute(accs, frags[mi], bv, sa_v, sb_v, scale_shift, [mi])
                rocdl.sched_barrier(0)
            return accs

        def hot_loop_scheduler():
            # Interleave the MFMAs with the tile's vmem + A-LDS loads: preload all hints, then issue MFMAs 1-by-1.
            rocdl.sched_vmem(sched_num_gmem)
            rocdl.sched_dsrd(sched_num_ds_load)
            for _ in range_constexpr(n_mfma_hints):
                rocdl.sched_mfma(1)
            rocdl.sched_barrier(0)

        accs_init = [Vec.filled(4, 0.0, Float32).ir_value() for _ in range_constexpr(n_acc)]

        # R7 (goal.md SS C "R6", scale-only half-pipeline): carry the 6 e8m0 scale words
        # (m_pairs sa + n_pairs sb, plain Int32 SSA scalars -- NumericMeta implements
        # __extract_to_ir_values__/__construct_from_ir_values__ natively, no make_rmem_tensor
        # round-trip needed) one K-tile ahead as `range(..., init=[...])` loop state alongside
        # `accs`, mirroring the existing double-buffered A-DMA prefetch-ahead-of-use shape. B
        # stays exactly as before (`load_b(kt)`, synchronous, same iteration): the full B+scale
        # pipeline (carrying 4 i32[4] B fragments too, via make_rmem_tensor/.store()/.load() per
        # fragment) was measured bitwise-correct and VGPR-safe (216->237, <256) but the
        # pack/unpack round-trip added ~20 VALU/iter that outweighed the latency-hiding gain on
        # 5/6 shapes (regressed up to +8%); only single_l2 (K=15360) netted positive. Scale words
        # are cheap scalars with no memref round-trip (216->221 VGPR only), so this isolates the
        # latency-hiding benefit from the B-pipeline's overhead. rocprofv3 confirms the mechanism
        # on both evidence shapes: SQ_WAIT_ANY -25%(fc2)/-28%(l2), and unlike the full pipeline
        # SQ_WAVE_CYCLES actually drops here too (fc2 -2.9%, l2 -4.3%) instead of staying flat,
        # because there isn't enough new VALU overhead to reabsorb the freed cycles. Card-6 ABBA
        # (N=30, trimmed) measured net non-negative on all 6 shapes: double_fc2 -1.36% (passes
        # the strict mean>1%+stdev<half-mean trust gate), single_l2 -0.85% (stdev 0.04, very
        # tight but just under the 1% bar), double_fc1 -0.77% (28/30 blocks negative), the
        # remaining three (double_qkv +0.22%, double_proj +0.19%, single_l1 +0.01%) flat at
        # noise level -- no shape regresses. (R8 then moved the A-DMA's kt*A_ROW_B into the
        # buffer descriptor base, see dma_a_to_lds.)
        #
        # U2 (campaign 2 R2, b_prefetch 1/2): the loop runs K_TILES/2 iterations, one per 256-K
        # scale chunk, as two halves. Half h reads A(kt0+h) from LDS buffer h, DMAs A(kt0+h+1)
        # into the other buffer and runs its MFMAs on B(kt0+h), which the previous half loaded:
        # half 0 consumes the B carried across the iteration boundary, half 1 the B half 0
        # loaded, half 1 loads the B the next iteration's half 0 consumes. That is the B
        # carry of R7's note without its cost: one carried B set per two K-tiles and no phi
        # copies. The chunk's scale words load once (half 0) instead of once per K-tile; half 0
        # uses their low 16 bits, half 1 the high 16 bits. b_prefetch=2 issues the next B and
        # scales behind the A DMA, so the end-of-half wait drains exactly the DMA: vmcnt(<loads
        # issued after it>). Two orderings are load-bearing (each measured >10% when broken):
        # the half's ds_reads must precede its DMA in program order (an LDS DMA ahead of a
        # ds_read makes the backend wait vmcnt(0) before the read), and the sched_barrier(0)
        # right after the DMA keeps the scheduler from hoisting the B/scale loads above it
        # (the backend then strengthens the partial wait to vmcnt(0)).
        # sa_lds: the next chunk's A-scale DMA joins half 0's A DMA group, so half 0's partial
        # wait drains it too; the chunk's words are read back at the top of the next iteration,
        # two barriers after the slot they overwrite was last read.
        N_CH = K_TILES // 2
        n_bfr = num_acc_n * k_halves
        n_sa_c = 0 if sa_lds_on else m_pairs  # carried A-scale words
        n_sc = n_sa_c + n_pairs

        def sched_half(n_vmem):
            # n_vmem = B/scale loads of the half. Hint counts must equal the region's opcode
            # counts, or the backend silently drops the whole group.
            if const_expr(b_prefetch == 2):
                # ds_reads and the DMA sit above the fence; this region is B/scales + MFMAs.
                rocdl.sched_vmem(n_vmem)
            else:
                rocdl.sched_vmem(n_coop + n_vmem)
                rocdl.sched_dsrd(sched_num_ds_load)
            for _ in range_constexpr(n_mfma_hints):
                rocdl.sched_mfma(1)
            rocdl.sched_barrier(0)

        def wait_vmcnt(n):
            # s_waitcnt vmcnt(n) expcnt(7) lgkmcnt(0); vmcnt splits into bits [3:0] and [15:14].
            rocdl.s_waitcnt((n & 0xF) | (7 << 4) | (((n >> 4) & 0x3) << 14))

        def drain_half(n_after_dma):
            if const_expr(b_prefetch == 2):
                wait_vmcnt(n_after_dma)
            else:
                rocdl.s_waitcnt(0)

        def sched_mfma_region():
            for _ in range_constexpr(n_mfma_hints):
                rocdl.sched_mfma(1)
            rocdl.sched_barrier(0)

        def clamp_kt(kt):
            # min(kt, K_TILES - 1) for kt <= K_TILES + 1 (the redundant tail prefetches).
            return kt - (kt // K_TILES) * (kt - (K_TILES - 1))

        def _frag(v):
            t = fx.make_rmem_tensor(4, Int32)
            t.store(Vec(v))
            return t

        sc_hi = 16 if const_expr(k_halves == 1) else fx.Int32(16)

        # epi=2 ring: the last chunk runs peeled after the loop with the C stores in its gaps.
        epi_ov = vsplit and epi == 2 and N_CH // 3 >= 1 and not rq

        def vs_chunk(jj, c, accs, sb_v, bv0, store_fn=None):
            # One chunk (two halves) of the tile_m=256 ring, see the K loop below. store_fn (the
            # peeled last chunk): half 0 issues only B(t0+1) -- the tail scale loads, A DMA and
            # A-scale DMA would be redundant, and the end-of-half wait counts the B loads only --
            # and half 1 issues no loads; store_fn(accs, mi) stores m-row mi's C in the gap after
            # the next m-row's group, once mi's accumulators are final.
            last = store_fn is not None
            ch = jj * 3 + c
            t0 = ch * 2
            nch = ch + 1
            ch_pf = nch - nch // N_CH  # clamp the last chunk prefetch to N_CH-1
            sc_slot = c  # compile-time: one A-scale slot per in-iteration chunk (n_sc_slots==3)
            sa_v = read_sa(sc_slot)
            la = min(vs_ahead, m_chunks)

            def a_first(sl):
                # The first la m-rows' A fragments at the half's top, the rest group by group.
                fr = {}
                for r in range_constexpr(la):
                    fr[r] = read_a(None, sl, [r])
                return fr

            def a_group(sl, fr, g):
                if const_expr(g + la < m_chunks):
                    fr[g + la] = read_a(None, sl, [g + la])
                return fr[g]

            def fi_group(accs, sl, fr, g, bv, shift, item):
                # fi>0: m-row g's MFMAs with the A read of row g+la and the gap's load item
                # between them, each piece fenced.
                av = fr[g]
                if const_expr(fi == 1):
                    seq = [("m", [0]), ("r", None), ("m", [1]), ("i", None), ("m", list(range(2, num_acc_n)))]
                elif const_expr(fi == 2):
                    seq = [("m", [0]), ("i", None), ("m", [1]), ("r", None), ("m", list(range(2, num_acc_n)))]
                else:
                    seq = [("r", None), ("m", [0]), ("i", None), ("m", list(range(1, num_acc_n)))]
                for kind, nis in seq:
                    rocdl.sched_barrier(0)
                    if const_expr(kind == "m"):
                        accs = compute(accs, av, bv, list(sa_v), list(sb_v), shift, [g], nis)
                    elif const_expr(kind == "r"):
                        if const_expr(g + la < m_chunks):
                            fr[g + la] = read_a(None, sl, [g + la])
                    else:
                        item()
                rocdl.sched_barrier(0)
                return accs

            sl0 = slot_i32[(2 * c) % 3]
            fr = a_first(sl0)
            kt_a = clamp_kt(t0 + 2)
            bparts = [load_b(t0 + 1, vs_b[0])]
            if const_expr(last):
                sb_next = sb_v
            else:
                sa_next, sb_next = load_sc(ch_pf)
            for g in range_constexpr(m_chunks):

                def item0(g=g):
                    if const_expr(g + 1 < vs_np):
                        bparts.append(load_b(t0 + 1, vs_b[g + 1]))
                    elif const_expr(g + 1 < 2 * vs_np and not last):
                        dma_a_to_lds(kt_a, None, slot_i8[(2 * c + 2) % 3], vs_c[g + 1 - vs_np])
                        if const_expr(g + 2 == 2 * vs_np):
                            dma_sa_to_lds(ch_pf, (c + 1) % 3)

                if const_expr(fi != 0):
                    accs = fi_group(accs, sl0, fr, g, bv0, None, item0)
                    continue
                rocdl.sched_barrier(0)
                av = a_group(sl0, fr, g)
                accs = compute(accs, av, bv0, list(sa_v), list(sb_v), None, [g])
                rocdl.sched_barrier(0)
                item0()
            bv1 = sum(bparts, [])
            if const_expr(last):
                wait_vmcnt(n_bfr)
            else:
                wait_vmcnt(n_bfr + n_sc + n_coop + n_sa_dma)
            rocdl.s_barrier()
            sl1 = slot_i32[(2 * c + 1) % 3]
            fr = a_first(sl1)
            kt_b = clamp_kt(t0 + 2)
            kt_a = clamp_kt(t0 + 3)
            bparts = [] if last else [load_b(kt_b, vs_b[0])]
            for g in range_constexpr(m_chunks):

                def item1(g=g):
                    if const_expr(last):
                        pass
                    elif const_expr(g + 1 < vs_np):
                        bparts.append(load_b(kt_b, vs_b[g + 1]))
                    elif const_expr(g + 1 < 2 * vs_np):
                        dma_a_to_lds(kt_a, None, slot_i8[(2 * c + 3) % 3], vs_c[g + 1 - vs_np])

                if const_expr(fi != 0):
                    accs = fi_group(accs, sl1, fr, g, bv1, sc_hi, item1)
                    if const_expr(last and g > 0):
                        store_fn(accs, g - 1)
                    continue
                rocdl.sched_barrier(0)
                av = a_group(sl1, fr, g)
                accs = compute(accs, av, bv1, list(sa_v), list(sb_v), sc_hi, [g])
                rocdl.sched_barrier(0)
                if const_expr(last):
                    if const_expr(g > 0):
                        store_fn(accs, g - 1)
                else:
                    item1()
            if const_expr(last):
                rocdl.sched_barrier(0)
                store_fn(accs, m_chunks - 1)
                return accs, sb_v, []
            bv0 = sum(bparts, [])
            wait_vmcnt(n_bfr + n_coop)
            rocdl.s_barrier()
            return accs, sb_next, bv0

        if const_expr(a3):
            dma_a_to_lds(fx.Int32(0), None, slot_i8[0])
        else:
            dma_a_to_lds(fx.Int32(0), fx.Int32(0))
        if const_expr(sa_lds_on):
            dma_sa_to_lds(fx.Int32(0), 0)
        if const_expr(has_bias and epi == 2):
            # epi=2 bias: a lane's 4 consecutive columns per n-fragment as one 8 B load, issued
            # here so the prologue drain covers it instead of a wait after the K loop.
            bias_ptr_ty = fx.PointerType.get(
                out_elem.ir_type, address_space=fx.AddressSpace.Global, alignment=8
            )
            bias_flat4 = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(
                    fx.Tensor(fx.make_view(fx.inttoptr(bias_ptr_ty, arg_bias), fx.make_layout(1 << 28, 1))),
                    max_size=False,
                    num_records_bytes=fx.Int64(N) * fx.Int64(2),
                ),
                fx.make_layout(4, 1),
            )
            bias_copy64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), out_elem)
            col_t4 = (by_n + wave_col0() + lane_div_16 * 4) // 4
            bias_t = []
            for ni in range_constexpr(num_acc_n):
                bt = fx.make_rmem_tensor(4, out_elem)
                fx.copy(bias_copy64, bias_flat4[None, col_t4 + n_col(ni) // 4], bt)
                bias_t.append(bt.load())
        sa_pf0, sb_pf0 = load_sc(fx.Int32(0))
        b_pf0 = load_b(fx.Int32(0)) if u2 else []
        if const_expr(a3):
            # A(1) stays in flight into the loop: the first half's drain retires it. A bare
            # s_barrier: gpu.barrier()'s workgroup release fence would drain every LDS DMA.
            rocdl.sched_barrier(0)
            dma_a_to_lds(fx.Int32(1), None, slot_i8[1])
            wait_vmcnt(n_coop)
            rocdl.s_barrier()
        else:
            rocdl.s_waitcnt(0)
            gpu.barrier()
        loop_init = accs_init + list(sa_pf0) + list(sb_pf0) + [b.load().ir_value() for b in b_pf0]
        if const_expr(a3):
            # epi_ov: the last iteration runs peeled after the loop, with the C stores in it.
            n_iters = N_CH // 3 - (1 if epi_ov else 0)
        elif const_expr(u2):
            n_iters = N_CH
        else:
            n_iters = K_TILES
        for iv, state in range(fx.Index(0), fx.Index(n_iters), fx.Index(1), init=loop_init):
            state = list(state)
            accs = state[:n_acc]
            sa_v = state[n_acc : n_acc + n_sa_c]
            sb_v = state[n_acc + n_sa_c : n_acc + n_sc]
            if const_expr(a3):
                # a_stages=3: three chunks (six K-tiles) per iteration, so K-tile t's slot t%3 is a
                # fixed symbol in every half. Half t reads A(t), issues B(t+1) (and in half 0 the
                # next chunk's scales) and only then the DMA of A(t+2) into slot (t+2)%3, so the
                # waits on B(t+1) in half t+1 never retire that DMA (vmcnt is in order). Its drain
                # moves one half later: the end of half t retires A(t+1), issued in half t-1,
                # with vmcnt(<loads issued after it>): 4 B + 2 scales + 4 DMA + 1 A-scale DMA = 11
                # after half 0, 4 B + 4 DMA = 8 after half 1 (A(t+2) and the A-scale DMA of half 0);
                # 8 + 4 + 8 + 2 = 22 and 8 + 8 = 16 at tile_m=256.
                # Slot (t+2)%3 was last read by K-tile t-1, one barrier earlier. The halves end
                # with a bare s_barrier: the waits above already order LDS, and gpu.barrier()'s
                # workgroup release fence would also drain the A(t+2) DMA (vmcnt(0)).
                jj = fx.Int32(iv)
                bv0 = [_frag(v) for v in state[n_acc + n_sc :]]
                for c in range_constexpr(3):
                    if const_expr(vsplit):
                        # tile_m=256 (one wave per SIMD): m-row g's MFMAs run as group g between
                        # sched_barrier(0) fences and the half's loads go into the gaps in their
                        # usual order: B part 0 (and in half 0 the scales) before group 0, then one
                        # B part per gap, then the A-DMA parts, the A-scale DMA after the last one.
                        # Issued in one burst they stall the single wave behind a full VMEM queue.
                        # Every load is still issued after the previous half's DMA, so the same
                        # vmcnt counts retire the same DMAs; MFMA order per accumulator is unchanged.
                        accs, sb_v, bv0 = vs_chunk(jj, c, accs, sb_v, bv0)
                        continue
                    ch = jj * 3 + c
                    t0 = ch * 2
                    nch = ch + 1
                    ch_pf = nch - nch // N_CH  # clamp the last chunk prefetch to N_CH-1
                    sc_slot = c  # compile-time, see vs_chunk
                    sa_v = read_sa(sc_slot)
                    # half 0: A(t0) in slot (2c)%3, carried B(t0), chunk ch's low scale half.
                    if const_expr(a_pipe == 0):
                        av = read_a(None, slot_i32[(2 * c) % 3])
                    bv1 = load_b(t0 + 1)
                    sa_next, sb_next = load_sc(ch_pf)
                    rocdl.sched_barrier(0)
                    dma_a_to_lds(clamp_kt(t0 + 2), None, slot_i8[(2 * c + 2) % 3])
                    dma_sa_to_lds(ch_pf, (c + 1) % 3)
                    rocdl.sched_barrier(0)
                    if const_expr(a_pipe > 0):
                        accs = compute_a_pipe(accs, slot_i32[(2 * c) % 3], bv0, list(sa_v), list(sb_v), None)
                    else:
                        accs = compute(accs, av, bv0, list(sa_v), list(sb_v), None)
                        sched_mfma_region()
                    wait_vmcnt(n_bfr + n_sc + n_coop + n_sa_dma)
                    rocdl.s_barrier()
                    # half 1: A(t0+1) in slot (2c+1)%3, B(t0+1), chunk ch's high scale half.
                    if const_expr(a_pipe == 0):
                        av = read_a(None, slot_i32[(2 * c + 1) % 3])
                    bv2 = load_b(clamp_kt(t0 + 2))
                    rocdl.sched_barrier(0)
                    dma_a_to_lds(clamp_kt(t0 + 3), None, slot_i8[(2 * c + 3) % 3])
                    rocdl.sched_barrier(0)
                    if const_expr(a_pipe > 0):
                        accs = compute_a_pipe(
                            accs, slot_i32[(2 * c + 1) % 3], bv1, list(sa_v), list(sb_v), sc_hi
                        )
                    else:
                        accs = compute(accs, av, bv1, list(sa_v), list(sb_v), sc_hi)
                        sched_mfma_region()
                    wait_vmcnt(n_bfr + n_coop)
                    rocdl.s_barrier()
                    bv0 = bv2
                    sb_v = sb_next
                carry = accs + list(sb_v) + [b.load().ir_value() for b in bv0]
            elif const_expr(u2):
                j = fx.Int32(iv)
                kt1 = j * 2 + 1
                nj = j + 1
                ch_pf = nj - nj // N_CH  # clamp the last chunk prefetch to N_CH-1
                nkt = kt1 + 1
                kt_pf = nkt - nkt // K_TILES  # clamp the last tile prefetch to K_TILES-1
                if const_expr(sa_lds_on):
                    sc_slot = j % 2
                    sa_v = read_sa(sc_slot)
                    if const_expr(b_prefetch != 2):
                        # Keeps these reads out of half 0's sched_dsrd count.
                        rocdl.sched_barrier(0)
                # half 0: A(kt0) in LDS buffer 0, carried B(kt0), chunk j's low scale half.
                av = read_a(fx.Int32(0))
                bv0 = [_frag(v) for v in state[n_acc + n_sc :]]
                if const_expr(b_prefetch == 2):
                    dma_a_to_lds(kt1, fx.Int32(1))
                    if const_expr(sa_lds_on):
                        dma_sa_to_lds(ch_pf, fx.Int32(1) - sc_slot)
                    rocdl.sched_barrier(0)
                    bv1 = load_b(kt1)
                    sa_next, sb_next = load_sc(ch_pf)
                else:
                    bv1 = load_b(kt1)
                    sa_next, sb_next = load_sc(ch_pf)
                    dma_a_to_lds(kt1, fx.Int32(1))
                    if const_expr(sa_lds_on):
                        dma_sa_to_lds(ch_pf, fx.Int32(1) - sc_slot)
                accs = compute(accs, av, bv0, list(sa_v), list(sb_v), None)
                sched_half(n_bfr + n_sc + (n_sa_dma if sa_lds_on and b_prefetch != 2 else 0))
                drain_half(n_bfr + n_sc)
                gpu.barrier()
                # half 1: A(kt1) in LDS buffer 1, B(kt1), chunk j's high scale half.
                av = read_a(fx.Int32(1))
                if const_expr(b_prefetch == 2):
                    dma_a_to_lds(kt_pf, fx.Int32(0))
                    rocdl.sched_barrier(0)
                    bv2 = load_b(kt_pf)
                else:
                    bv2 = load_b(kt_pf)
                    dma_a_to_lds(kt_pf, fx.Int32(0))
                accs = compute(accs, av, bv1, list(sa_v), list(sb_v), sc_hi)
                sched_half(n_bfr)
                drain_half(n_bfr)
                gpu.barrier()
                carry = accs + list(sa_next) + list(sb_next) + [b.load().ir_value() for b in bv2]
            else:
                kt = fx.Int32(iv)
                cur = kt % 2
                nxt = (kt + 1) % 2
                nkt = kt + 1
                pf_kt = nkt - nkt // K_TILES  # clamp last-iter prefetch to K_TILES-1
                scale_shift = None if tiles_per_chunk == 1 else (kt % tiles_per_chunk) * 16
                chunk_pf = pf_kt if tiles_per_chunk == 1 else pf_kt // tiles_per_chunk
                av = read_a(cur)
                bv = load_b(kt)
                sa_next, sb_next = load_sc(chunk_pf)  # prefetch kt+1's scale chunk ahead of compute
                dma_a_to_lds(pf_kt, nxt)  # A DMA after B/scale loads -> overlaps the MFMAs
                accs = compute(accs, av, bv, list(sa_v), list(sb_v), scale_shift)
                hot_loop_scheduler()
                rocdl.s_waitcnt(0)  # drain the A DMA before the barrier
                gpu.barrier()
                carry = accs + list(sa_next) + list(sb_next)
            results = yield carry
        accs = list(results)[:n_acc]

        # Epilogue via fx.copy: a lane owns 4 rows per (mi,ni) accm (row m*16+(l//16)*4+ii, col
        # base+l%16), c_stride apart; c_flat bounds ragged-M OOB and honors c_row/batch_stride.
        c_stride = N if c_row_stride < 0 else c_row_stride
        if const_expr(c_row_stride < 0):
            c_nrec = fx.Int64(i32_m) * fx.Int64(N) * fx.Int64(2)
        else:
            c_nrec = (fx.Int64(i32_m - fx.Int32(1)) * fx.Int64(c_stride) + fx.Int64(N)) * fx.Int64(2)
        c_addr = arg_c
        if const_expr(batch > 1):
            c_bstride = (
                fx.Int64(i32_m) * fx.Int64(N) * fx.Int64(2)
                if c_batch_stride < 0
                else fx.Int64(c_batch_stride)
            )
            c_addr = c_addr + fx.Int64(bid_z) * c_bstride
        c_ptr_ty = fx.PointerType.get(out_elem.ir_type, address_space=fx.AddressSpace.Global, alignment=2)
        c_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.inttoptr(c_ptr_ty, c_addr), fx.make_layout(1 << 28, 1))),
                max_size=False,
                num_records_bytes=c_nrec,
            ),
            fx.make_layout(1, 1),
        )
        c_copy = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), out_elem)
        c_rstride = fx.Int32(c_stride)
        col_w = by_n + wave_col0() + lane_mod_16
        bx_w = bx_m  # this wave's first C row
        # Bias joins the bf16-rounded product in fp32 and rounds again: the same bytes as a
        # separate `out + bias`, which is what the caller would otherwise launch.
        if const_expr(has_bias and epi == 2):
            bias_f = [Vec(bias_t[ni]).to(Float32) for ni in range_constexpr(num_acc_n)]
        elif const_expr(has_bias):
            bias_flat = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(
                    fx.Tensor(fx.make_view(fx.inttoptr(c_ptr_ty, arg_bias), fx.make_layout(1 << 28, 1))),
                    max_size=False,
                    num_records_bytes=fx.Int64(N) * fx.Int64(2),
                ),
                fx.make_layout(1, 1),
            )
            bias_f = []
            for ni in range_constexpr(num_acc_n):
                bf = fx.make_rmem_tensor(1, out_elem)
                fx.copy(c_copy, bias_flat[None, col_w + ni * 16], bf)
                b = bf.load().to(Float32)[0]
                bias_f.append(Vec.from_elements([b, b, b, b], Float32))
        if const_expr(epi == 2):
            # C^T fragments: lane l holds row l%16, columns (l//16)*4..+3 of each n-fragment. Per
            # n-fragment pair the two permlane16_swaps leave 8 consecutive columns in each lane,
            # at col_off 0/16/8/24 for lane group 0/1/2/3: one 16 B store per lane.
            c_ptr_ty16 = fx.PointerType.get(
                out_elem.ir_type, address_space=fx.AddressSpace.Global, alignment=16
            )
            c_flat8 = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(
                    fx.Tensor(fx.make_view(fx.inttoptr(c_ptr_ty16, c_addr), fx.make_layout(1 << 28, 1))),
                    max_size=False,
                    num_records_bytes=c_nrec,
                ),
                fx.make_layout(8, 1),
            )
            c_cpol = 2 if rq >= 2 else {0: 0, 1: 0, 2: 2, 3: 3}[store_mode]
            c_copy128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(c_cpol), out_elem)
            col_l = by_n + wave_col0() + (lane_div_16 % 2) * 16 + (lane_div_16 // 2) * 8
            pair_col = [n_col(2 * p) for p in range_constexpr(n_pairs)]
            c_full = c_line
            n_full_pairs = 2 if c_full else 0
            p_order = list(range(n_full_pairs, n_pairs)) + list(range(n_full_pairs))
            if const_expr(c_full):
                c_copy128_fl = fx.make_copy_atom(fx.rocdl.BufferCopy128b(c_cpol), out_elem)

            def pair_words(accs_s, mi, p, act=True):
                # Pair p as four bf16 words: col_off 0/16/8/24 for lane group 0/1/2/3.
                # act=False stops before the GELU, which is the pre-activation backward saves.
                va = Vec(accs_s[mi * num_acc_n + 2 * p])
                vb = Vec(accs_s[mi * num_acc_n + 2 * p + 1])
                if const_expr(has_bias):
                    va = _bf16_round_bias_vec(va, bias_f[2 * p])
                    vb = _bf16_round_bias_vec(vb, bias_f[2 * p + 1])
                if const_expr(gelu and act):
                    if const_expr(act_rounded):
                        va = _bf16_round_vec(va)
                        vb = _bf16_round_vec(vb)
                    va = _gelu1_torch_bf16_vec(va)
                    vb = _gelu1_torch_bf16_vec(vb)
                d_a0 = rocdl.cvt_pk_bf16_f32(va[0], va[1])
                d_a1 = rocdl.cvt_pk_bf16_f32(va[2], va[3])
                d_b0 = rocdl.cvt_pk_bf16_f32(vb[0], vb[1])
                d_b1 = rocdl.cvt_pk_bf16_f32(vb[2], vb[3])
                w0, w2 = _permlane16_swap(d_a0, d_b0)
                w1, w3 = _permlane16_swap(d_a1, d_b1)
                return [w0, w1, w2, w3]

            def store_pair_c(words, mi, p):
                t8 = fx.make_rmem_tensor(8, out_elem)
                t8.store(Vec.from_elements(words, Int32).bitcast(out_elem))
                off = (bx_w + mi * 16 + lane_mod_16) * c_rstride + col_l + pair_col[p]
                fx.copy(c_copy128, t8, c_flat8[None, off // 8])

            def store_mrow(accs_s, mi, fence=False):
                # fence: a scheduling fence after each store (the last m-row of the peeled chunk).
                pair_w = []
                for p in p_order:
                    words = pair_words(accs_s, mi, p)
                    if const_expr(p < n_full_pairs):
                        pair_w.append(words)
                        continue
                    store_pair_c(words, mi, p)
                    if const_expr(fence):
                        rocdl.sched_barrier(0)
                if const_expr(c_full):
                    # Lane j of a 16-lane row holds C row j. Store 0 (rows 0-7): lanes 0-7 keep
                    # their pair 0, lanes 8-15 take lane j-8's pair 1; store 1 (rows 8-15): lanes
                    # 0-7 take lane j+8's pair 0, lanes 8-15 keep their pair 1.
                    x0, x1 = pair_w
                    lo8 = [rocdl.update_dpp(T.i32, x0[d], x1[d], 0x128, 0xF, 0xC, False) for d in range(4)]
                    hi8 = [rocdl.update_dpp(T.i32, x1[d], x0[d], 0x128, 0xF, 0x3, False) for d in range(4)]
                    row_f = bx_w + mi * 16 + lane_mod_16 % 8
                    col_f = col_l + pair_col[0] + (lane_mod_16 // 8) * 32
                    for h, w4 in ((0, lo8), (8, hi8)):
                        t8 = fx.make_rmem_tensor(8, out_elem)
                        t8.store(Vec.from_elements(w4, Int32).bitcast(out_elem))
                        off = (row_f + h) * c_rstride + col_f
                        fx.copy(c_copy128_fl, t8, c_flat8[None, off // 8])
                        if const_expr(fence):
                            rocdl.sched_barrier(0)

            if const_expr(rq):
                # The A6W4 row operand. A lane holds only 8 of a micro-block's 32 columns, so
                # permlane32 then permlane16 join four m-rows' pieces into four whole blocks.
                OW_ROW = N // 4
                SC_BAND = N
                # Lane group g holds m-row m0 + (0, 2, 1, 3)[g], i.e. 2 * (g & 1) + (g >> 1);
                # every per-lane term below follows from that and is loop invariant.
                g_row = ((lane_div_16 & 1) * 2 + (lane_div_16 // 2)) * 16 + lane_mod_16
                rq_out_lane = g_row * OW_ROW
                rq_sc_lane = (lane_div_16 & 1) * SC_BAND + lane_mod_16 * 4 + lane_div_16 // 2
                rq_store = MXFP6RowQuantStore(
                    arg_rq,
                    arg_rs,
                    fx.Int64(i32_m) * fx.Int64(N),
                    fx.Int64(i32_m) * fx.Int64(N // 32),
                    0,  # the row half's e8m0 rounding bias, per flydsl_quant_a6w4_act_dual
                )

                def rq_base(m0, p):
                    gcmb = rocdl.readfirstlane(T.i32, (by_n + wave_col0() + pair_col[p]) // MX_BLOCK)
                    ob = (bx_w + m0 * 16) * OW_ROW + gcmb * 8
                    sb = (bx_w // 32 + m0 // 2) * SC_BAND + (gcmb // 8) * 256
                    sb = sb + (gcmb % 4) * 64 + ((gcmb // 4) % 2) * 2
                    return ob, rocdl.readfirstlane(T.i32, sb)

                def rq_merge(ws):
                    pq = []
                    for half in range_constexpr(2):
                        v = [None] * 8
                        for d in range_constexpr(4):
                            v[d], v[4 + d] = _permlane32_swap(ws[2 * half][d], ws[2 * half + 1][d])
                        pq.append(v)
                    out = [None] * 16
                    for d in range_constexpr(8):
                        out[d], out[8 + d] = _permlane16_swap(pq[0][d], pq[1][d])
                    return out

                if const_expr(rq >= 2):
                    # The col operand's micro-block is 32 *rows*, landing in 32 lanes of two
                    # m-fragments, so permlanes cannot assemble it: it goes through LDS.
                    cq_store = MXFP4ColQuantStore(
                        arg_cq,
                        arg_cs,
                        fx.Int64(N) * (fx.Int64(i32_m) // 2),
                        fx.Int64(N) * (fx.Int64(i32_m) // 32),
                        A6W4_COL_SCALE_BIAS,
                        fx.recast_iter(Int32, lds.cq.ptr),
                        wave,
                        lane,
                        i32_m // 8,  # i32 words per column of the feature-major operand
                        i32_m // 32,  # scale bytes per column
                        64,
                        BN // 4,
                    )
                    cq_col0 = rocdl.readfirstlane(T.i32, by_n + wave_col0())

                def store_mquad(accs_s, m0):
                    pre = None
                    for p in p_order:
                        ws = [pair_words(accs_s, m0 + s, p) for s in range_constexpr(4)]
                        # Backward wants the GELU's input, so only the operands activate.
                        wc = (
                            [pair_words(accs_s, m0 + s, p, act=False) for s in range_constexpr(4)]
                            if const_expr(gelu and keep_preact)
                            else ws
                        )
                        for s in range_constexpr(4):
                            store_pair_c(wc[s], m0 + s, p)
                        if const_expr(rq >= 2):
                            cq_store.stage(pair_col[p] // 32, ws)
                            if const_expr(p == p_order[-1]):
                                pre = cq_store.prefetch()
                        ob, sb = rq_base(m0, p)
                        rq_store.store_block(rq_merge(ws), ob + rq_out_lane, sb + rq_sc_lane)
                    return pre

                for mq in range_constexpr(m_chunks // 4):
                    cq_pre = store_mquad(accs, 4 * mq)
                    if const_expr(rq >= 2):
                        cq_store.convert(bx_w // 32 + mq * 2, cq_col0, cq_pre)
            elif const_expr(epi_ov):
                # The peeled last iteration: m-row mi's accumulators are final after its group in
                # the last half, so its stores go into the next group's gap (one group later:
                # the hazard recognizer does not see the inline-asm MFMAs, so a read of an
                # accumulator right behind the MFMA that writes it would get no wait states).
                st = list(results)
                sb_l = st[n_acc + n_sa_c : n_acc + n_sc]
                bv_l = [_frag(v) for v in st[n_acc + n_sc :]]
                jj_l = fx.Int32(N_CH // 3 - 1)

                def store_fn(accs_s, mi):
                    store_mrow(accs_s, mi, fence=mi == m_chunks - 1)

                for c in range_constexpr(3):
                    accs, sb_l, bv_l = vs_chunk(jj_l, c, accs, sb_l, bv_l, store_fn if c == 2 else None)
            else:
                for mi in range_constexpr(m_chunks):
                    store_mrow(accs, mi)
        elif const_expr(epi == 1):
            # Lane group g holds rows 4g..4g+3, two per dword (h). Swapping fragment 2p's odd groups
            # with fragment 2p+1's even groups leaves each 32-lane half two rows of 32 consecutive
            # columns: s0 rows 2h+e (lanes 0-31) / 8+2h+e (lanes 32-63), s1 the same plus 4.
            col_rm = by_n + wave_col0() + lane % 32
            row_rm = bx_w + (lane // 32) * 8
            for mi in range_constexpr(m_chunks):
                for p in range_constexpr(num_acc_n // 2):
                    va = Vec(accs[mi * num_acc_n + 2 * p]).to(out_elem)
                    vb = Vec(accs[mi * num_acc_n + 2 * p + 1]).to(out_elem)
                    if const_expr(has_bias):
                        va = (va.to(Float32) + bias_f[2 * p]).to(out_elem)
                        vb = (vb.to(Float32) + bias_f[2 * p + 1]).to(out_elem)
                    wa = va.bitcast(Int32)
                    wb = vb.bitcast(Int32)
                    for h in range_constexpr(2):
                        s0, s1 = _permlane16_swap(wa[h], wb[h])
                        for w, r0 in ((s0, 2 * h), (s1, 4 + 2 * h)):
                            pair = Vec.from_elements([fx.Int32(w)], Int32).bitcast(out_elem)
                            for e in range_constexpr(2):
                                cf = fx.make_rmem_tensor(1, out_elem)
                                cf.store(Vec.from_elements([pair[e]], out_elem))
                                off = (row_rm + mi * 16 + r0 + e) * c_rstride + col_rm + p * 32
                                fx.copy(c_copy, cf, c_flat[None, off])
        else:
            for mi in range_constexpr(m_chunks):
                row_m = bx_w + mi * 16 + lane_div_16 * 4
                for ni in range_constexpr(num_acc_n):
                    col = col_w + ni * 16
                    acc = Vec(accs[mi * num_acc_n + ni]).to(out_elem)
                    if const_expr(has_bias):
                        acc = (acc.to(Float32) + bias_f[ni]).to(out_elem)
                    for ii in range_constexpr(4):
                        cf = fx.make_rmem_tensor(1, out_elem)
                        cf.store(Vec.from_elements([acc[ii]], out_elem))
                        off = (row_m + ii) * c_rstride + col
                        fx.copy(c_copy, cf, c_flat[None, off])

    c_addr = fx.Int64(fx.ptrtoint(arg_c))
    a_addr = fx.Int64(fx.ptrtoint(arg_a))
    b_addr = fx.Int64(fx.ptrtoint(arg_b))
    sa_addr = fx.Int64(fx.ptrtoint(arg_scale_a))
    sb_addr = fx.Int64(fx.ptrtoint(arg_scale_b))
    bias_addr = fx.Int64(fx.ptrtoint(arg_bias))
    rq_addr = fx.Int64(fx.ptrtoint(arg_rq))
    rs_addr = fx.Int64(fx.ptrtoint(arg_rs))
    cq_addr = fx.Int64(fx.ptrtoint(arg_cq))
    cs_addr = fx.Int64(fx.ptrtoint(arg_cs))
    if const_expr(waves_per_eu > 0):
        wpe = waves_per_eu
    else:
        wpe = None
    gx = (i32_m + (BM - 1)) // BM
    gy = i32_n // BN
    # group_m>0: flatten to a 1D grid so kernel_gemm's block_mn decode (above) sees every
    # workgroup id in band order; group_m==0: today's native 2D grid, unchanged.
    if const_expr(group_m > 0):
        grid = (gx * gy, 1, batch)
    else:
        grid = (gx, gy, batch)
    kernel_gemm(
        c_addr,
        a_addr,
        b_addr,
        sa_addr,
        sb_addr,
        i32_m,
        i32_n,
        bias_addr,
        rq_addr,
        rs_addr,
        cq_addr,
        cs_addr,
        value_attrs={"rocdl.waves_per_eu": wpe},
    ).launch(grid=grid, block=(256, 1, 1), stream=stream)


# The caller pads M to a multiple of this and guarantees nothing more, so every tile_m
# chosen below must divide it.
M_ALIGN = 256
# A's packing contract with quant_act_a6w4: 24 dense code bytes per 32 E2M3 values.
A_BYTES_PER_32 = 24
# (tile_m, tile_n, tile_k, waves_per_eu, group_m, pad16) per (N, K), from the FLUX-shape sweep on
# MI355X. group_m (R4, goal.md SS C/B.4): per-shape L2 band raster width for the flat-grid
# block_mn decode above; 0 = native 2D grid. Measured on a knob-parameterised probe copy
# (p4_ab.py / kvar.py) before landing here: single_l1 (N=21504) and single_l2 (K=15360) are
# the two shapes where banding measured positive, each at its own interior optimum (gm=1 and
# gm=gx both measure at or above the native grid, so gm=32 / gm=2 are not a dispatch-order
# artifact); the other four shapes measured neutral-to-worse at every gm tried, so they keep
# gm=0.
# pad16 (R6, goal.md D.1): per-shape LDS row padding, in units of 16 bytes, after every
# CHUNK_ROWS=8 A rows (see launch_gemm). Correctness is identical regardless of pad16 (padding
# only moves WHERE A lives in LDS, verified bitwise-identical output at pad16 in {0,8} on all six
# shapes, at both native M and M=M_ALIGN=256); the per-shape value below is purely a timing
# choice, picked by isolated (no concurrent GPU activity -- concurrent probes on other cards
# measurably confounded the smaller shapes), interleaved A/B/A/B median comparison, N>=34 blocks
# per shape, re-run twice: single_l2/double_fc2/double_proj measured a reproducible, directionally
# -100%-consistent win at pad16=8 (-2.2 to -3.0%, -0.86 to -1.1%, -1.3 to -1.6% respectively,
# excluding box-noise-contaminated samples per kb/flydsl/pitfalls.md's canary rule); single_l1
# measured a reproducible, 0-faster-out-of-34-blocks regression (+4.2 to +4.4%) that persists even
# with the strength-reduced read_a (ISA-confirmed near-identical to the un-reduced form, so the
# regression is not from extra per-mi address VALU, likely an occupancy/launch-pattern effect at
# single_l1's much larger grid -- next lever, not re-tried blindly this round); double_fc1 and
# double_qkv showed a small (+1.4%, ~0%) non-win once isolated from cross-probe interference (an
# earlier concurrent-cards measurement wrongly suggested a double_fc1 win -- see this round's
# ROUND_REPORT), so fc1 keeps pad16=0 rather than ship an unproven per-shape change.
# R9 (this round, goal.md "double_proj / double_qkv per-shape tile/launch" directive): swept
# tile_m/n/k in {(128,256,128) [current], (256,128,128), (256,256,128), (128,128,256)} x
# waves_per_eu{1,2} x group_m{0,1,2,3,4,6,8} x pad16{0,8} (pad16=0 only at tile_k=256, which
# trips the k_blk16==CHUNK_ROWS assert at pad16>0) for double_proj and double_qkv specifically,
# on top of R6/R7/R8's since-landed scale-prefetch + A-DMA SRD-base LICM. Every alternate tile
# shape measured worse than today's (128,256,128) for both shapes (256,128,128 and 256,256,128
# both regress sharply, 256,256,128 at waves_per_eu=2 catastrophically so -- n_acc doubles to 64,
# ~9x slower, consistent with an occupancy collapse at that accumulator count; 128,128,256 is a
# smaller but still consistent regression on both shapes, consistent with the campaign's existing
# tile_k=256 dead end now reconfirmed on these two shapes too) and no group_m banding beat 0 for
# either shape at any pad16/wpe (double_proj was already known-optimal; double_qkv's own gm scan
# at pad16=8 confirms the same gm=0 optimum). The one real move: holding tile/wpe/gm fixed at
# today's values, double_qkv's pad16 flips from a measured non-win in R6 (pre-R7/R8 codebase) to
# a reproducible win now -- the R7 scale-prefetch and R8 SRD-base LICM changes evidently shifted
# the balance between pad16's own extra chunk-address VALU cost and the LDS-bank-conflict cycles
# it removes (R6's own pitfalls note already flagged this exact trade-off as shape- and
# codebase-state-dependent, not a one-time verdict). Confirmed via: a direct min-of-30 (bench.py's
# own timed_min semantics) A/B on two separate cards (-8.2%, -11.3%); a 38/38-block, mean>1%+
# std<half-mean-significant isolated card-6 ABBA at N_REP=80 (-1.87% median, -1.90% mean, std
# 0.14, trim10% -1.89%) that isolates pure GPU pipe time from host dispatch, confirming the
# effect is real (not purely host-side) even though its magnitude under that burst-timing method
# is smaller than under the ruler's own per-call-sync timing; and the official card-7 bench.sh
# itself end to end (next section) -- all four measurements agree on direction and on
# significance, only disagreeing on exact magnitude depending on how much of the per-call host
# round-trip (torch.cuda.Event + synchronize every rep, exactly what bench.py's timed_min does)
# each methodology includes. ISA: scratch=0, spill=0, agpr=0, vgpr 219->219 (unchanged), sgpr
# 55->55 (unchanged) at both pad16 values for this shape; LDS 32768->36864B (+12.5%, the expected
# pad16=8 cost), still far under the 160 KiB/CU budget so occupancy is unaffected. Correctness
# (bitwise-identical regardless of pad16, since padding only moves WHERE A lives in LDS) was
# already established for all six shapes at pad16 in {0,8} in R3/R6; reconfirmed this round
# end-to-end for double_qkv specifically against the current (post-R7/R8) codebase.
# b_prefetch (7th field, campaign 2 R2): the U2 loop flips two of the verdicts above, so the
# loop and its configs were re-measured together (in-process A/B vs the round-0 kernel,
# bitwise identical): pad16=8 now wins on single_l1 and double_fc1, single_l1 is faster with
# waves_per_eu unset than at 2, and single_l2 (K=15360) prefers b_prefetch=1.
# sa_lds (8th field, campaign 2 R4): A-scale words DMA'd once per workgroup and shared through
# LDS (see launch_gemm), on top of the fp6 ds_read_b64 hi-half read it was measured with.
# epi (9th field, campaign 2 R5): C store per shape, picked by in-process A/B against epi=0
# (bitwise): the dwordx4 store with the prologue bias load (2) on five shapes (-0.7..-2.4%),
# the row-merged store (1) on double_fc1, where 2 measured +0.1..+0.6% and 1 -1.1..-1.7%.
# xcd (10th field, campaign 2 R5): num_xcd for the XCD-major remap above, with group_m re-swept
# on top of the new epilogues (in-process A/B, bitwise): xcd=8 with gm=5/6/5 on double_qkv/
# double_proj/double_fc1 (-3.7%, -1.2..-2.4%, -4.1..-4.6%); without the remap no gm beat the
# native grid there, and the remap with the native M-fastest order is +6..+18% on every M=8192
# shape. double_fc2 and both single shapes measured flat-to-worse with the remap at every gm.
# a_stages (11th field, campaign 2 R7): the three-slot A ring (see launch_gemm), in-process A/B
# against a_stages=2 (two replicates each, bitwise): double_qkv -0.3/-0.6%, double_proj
# -1.7/-2.8%, double_fc2 -1.3/-0.5%; single_l1 +1.4%, single_l2 +0.1..+0.3%, and double_fc1
# (epi=1) needs 264 VGPR + 8 AGPR with it (+11.5%), so those three keep 2.
# a_pipe (12th field, campaign 2 R7): with the ring, single_l1 reads each A fragment 4 m-rows
# ahead of its MFMAs. 2 ahead measured -0.5..-0.8% against a_stages=2 over three replicates (1
# ahead the same), where the ring alone is +1.3..+1.4%; 4 ahead then -0.6..-0.8% against 2 over
# four replicates (3 and 6 ahead in between; group_m 16..64 all slower than 32 on the ring). On
# the other shapes a_pipe measured +0.4..+0.6% (double_fc2), +1.8..+2.2% (double_proj), flat
# (double_qkv), +0.4..+0.8% (single_l2, ring) and +0.3..+1.9% (double_fc1, ring), so they keep 0.
# single_l2 (campaign 3 R2): tile_m=256 (AGPR accumulators, 1 wave/SIMD) on the ring with the
# half's loads spread between one-m-row MFMA groups, group_m=4 (1024-row bands; band heights
# that do not divide its 64 M-tiles measured +6%; gm2/8/16 +2.0/+1.2/+0.3%, xcd remap +3.2%).
# In-process against the 128x256 config, bitwise: -6.4% in short runs; in 12 s windows at the
# 1400 W cap -6.5% at the ruler's cadence (both ~1990 MHz) and -5.0% back-to-back, where it
# holds 1764 MHz against 1622. R4: the 1x4 wave split instead of 2x2 (the 2x2 split loaded every
# B fragment in two waves; skipping that half of the B loads measured -7..-9%), A fragments two
# groups ahead and the last chunk's C stores in its gaps: -2.7..-3.4% short runs, bitwise; 12 s
# windows -2.0..-2.8% at the ruler's cadence and -3.3..-3.5% back-to-back (1674 vs 1761 MHz).
# fi (13th field, campaign 3 R6): the order inside the tile_m=256 ring's m-row groups (see
# launch_gemm; only that ring reads it). 2 puts the gap's load item after the group's first MFMA
# and the A read after its second, so those issue under MFMAs instead of ahead of them. single_l2
# against fi=0, bitwise: -3.8% cycles in a one-CU thread trace, -0.9% through the unmodified bench
# (7 of 7 blocks, champion re-measured in each); 12 s windows at the 1400 W cap -0.5% at the
# ruler's cadence and -1.5% back-to-back, ~45 MHz lower (1 and 3 in between). single_l1 moves to
# that ring with fi=2 and group_m=12: -3.0..-3.3% against its 128-row row through the same bench
# (7 of 7 blocks, bitwise); 12 s windows -2.9% at the ruler's cadence (both 1742 MHz) and -2.2%
# back-to-back (1661 vs 1640 MHz). There fi=0 measured -1.7..-2.2%, fi 1/3 +0.1/+0.4% against 2,
# group_m 4/8/10/14/16/20 +0.3/+1.4/+4.7/+2.8/+0.9/+1.4%; xcd remap and waves_per_eu 0/2 slower.
# double_qkv (campaign 3 R7) moves to that ring with fi=2 and group_m=16 (12/14/18/22/24
# +2.8/+0.1/+3.2/+2.7/+2.0% against 16; 20 -0.4..-0.6% at the ruler's cadence but +1.6..+2.3%
# back-to-back; fi=1 +0.04%). Against its 128-row row, bitwise: -1.1% in short runs; 12 s windows
# at the 1400 W cap -1.0% at the ruler's cadence (2226 vs 2169 MHz) and -2.0% back-to-back (1881
# vs 1789 MHz); -1.4% through the unmodified bench on card 7 (9 of 13 paired blocks), where it is
# bimodal in every arm (~146 vs ~153 us). double_fc1 on the same ring (group_m 16; 12/14/18/20/24
# +0.5..+1.9%) measured -1.6..-2.2% on card 6 and -2.1/-1.7% in 12 s windows (2191 vs 2087 MHz,
# 1753 vs 1607 MHz) but +0.9% on card 7 (5 of 13 paired blocks) with the cached half-line stores;
# it moves to that ring with the full-line nt stores (store_mode below).
# store_mode (14th field, campaign 3 R8): epi=2's C stores as 8 full 128 B lines each instead of
# 16 half lines (see launch_gemm), nt (2) or sc0|nt (3). Through the unmodified bench on card 7
# against the cached halves (7 paired blocks, bitwise), median of the per-block minima:
# double_qkv -4.3% with 3 / -3.0% with 2 (7 / 6 of 7 blocks), double_proj -2.6% / -1.9% (7 / 7),
# double_fc2 +0.3% / -0.7% (3 / 5), single_l1 -1.3% with either (7 / 7). single_l2 measured
# +0.8 / +1.0% (1 / 3 of 7) and flat in 12 s windows at the 1400 W cap (27..42 MHz more clock for
# the same time), so it keeps 0. In short runs the full-line layout alone (cached, or sc0 only)
# is +0.0..+0.4%: the gain needs nt on the full lines. double_fc1 (R9) on the double_qkv ring
# with nt (2), against its 128-row epi=1 row, bitwise: -5.0% through the unmodified bench on
# card 7 (median of 6 paired blocks, 6 of 6 faster), -4.2..-5.4% on cards 0/1/4/6; 12 s windows
# at the 1400 W cap -4.8% at the ruler's cadence (2150 vs 2096 MHz) and -1.3% back-to-back (1715
# vs 1626 MHz); 3 (sc0|nt) measured +0.1% against 2 in short runs.
_A6W4_CONFIGS = {
    (9216, 3072): (256, 384, 128, 1, 0, 8, 2, 1, 2, 1, 3, 0, 2, 0, 1),  # double_qkv
    (3072, 3072): (256, 384, 128, 1, 0, 8, 2, 1, 2, 1, 3, 0, 2, 0, 1),  # double_proj
    (12288, 3072): (256, 384, 128, 1, 0, 8, 2, 1, 2, 1, 3, 0, 2, 0, 1),  # double_fc1
    (3072, 12288): (256, 384, 128, 1, 0, 8, 2, 1, 2, 1, 3, 0, 2, 3, 1),  # double_fc2
    (21504, 3072): (256, 384, 128, 1, 0, 8, 2, 1, 2, 1, 3, 0, 2, 2, 1),  # single_l1
    (3072, 15360): (256, 384, 128, 1, 2, 8, 2, 1, 2, 2, 3, 0, 2, 3, 1),  # single_l2
}
_A6W4_DEFAULT = (128, 256, 128, 0, 0, 0, 2, 0, 0, 1, 2, 0, 0, 0, 2)

# One flyc.compile'd launch_gemm artifact per (N, K). Every Constexpr argument launch_gemm takes
# (tile_m/n/k, a_dtype, out_dtype, batch, the five stride flags, waves_per_eu, group_m, num_xcd,
# pad16, b_prefetch, sa_lds, epi, a_stages, a_pipe, fi, store_mode, vs_ahead) is a pure function
# of (n, k) through _A6W4_CONFIGS/_A6W4_DEFAULT above, while i32_m, i32_n, stream and the five operand
# pointers are all runtime args whose FlyDSL types (Int32 / Stream / Pointer) are
# annotated-runtime-type -> type-only in the compile cache key (kb/flydsl/compilation_pipeline.md
# S:_arg_cache_sig), so a single artifact is correct for every M at a given (N, K). Verified by
# compiling once at M=2048 and calling the same object at M=256/2048/8192/16384: SNR 55.588/
# 55.593/55.601/55.597, all finite (campaign probe p7_mindep.py); group_m's own grid/band math is
# rebuilt from i32_m on every call (both the host gx and the device num_pid_m), so this M-independence
# holds with group_m>0 too. Calling launch_gemm directly instead re-pays its @flyc.jit dispatch on
# every call (signature bind over 24 args + cache-key build + globals-drift check), measured at
# ~42.5 us/call here -- flat across all six shapes and dwarfing torch.empty's 1.1 us, so it is JIT
# dispatch, not allocation. gemm_mxfp4_kernel.py (the production MXFP4 GEMM) and gemm_fp8_kernel.py
# both already compile-cache for this reason; this is the same pattern applied to the one launcher
# in this file that still used the raw @flyc.jit closure on every call.
_A6W4_LAUNCH_CACHE: dict = {}


def _rq_level(m, n, k, want, a_pack=1):
    """The fused-operand level this shape's config can actually host, at most ``want``.

    The col pack's staging band is LDS the A ring does not give back, and the FP8-padded A
    packing (``a_pack=0``, a K-byte row) already asks for a third more of it than the dense
    one, so the two together run the tile past the 160 KB the hardware has. Dropping to the
    row half frees the band."""
    cfg = _A6W4_CONFIGS.get((n, k), _A6W4_DEFAULT)
    tm, tn = cfg[0], cfg[1]
    a_stg, epi, store_mode = cfg[10], cfg[8], cfg[13]
    rq = int(want)
    if rq >= 2 and not a_pack:
        rq = 1
    if rq and not (epi == 2 and store_mode == 0 and tm % 64 == 0 and tn % 128 == 0):
        rq = 0
    if rq and not (n % tn == 0 and m % 64 == 0):
        rq = 0
    if rq >= 2 and not (tm == 256 and a_stg == 3 and (tn // 4) % 32 == 0):
        rq = 1
    return rq


def gelu_dual_eligible(m, n, k, a_pack=1):
    """Whether `gemm_a6w4(..., gelu=True)` would attach both fused operand halves here."""
    return _rq_level(m, n, k, 2, a_pack) >= 2


def gemm_a6w4(
    aq, wq, sa, sb, n, k, bias=None, gelu=False, row_quant=None, keep_preact=False, act_rounded=False
):
    """C[M, N] bf16 = A6 @ W4^T (+ bias) for the A6W4 forward.

    aq    uint8 [M, 3K/4]   dense preshuffled FP6 (E2M3), A_BYTES_PER_32 bytes per 32 values
    wq    uint8 [N, K/2]    E2M1, low nibble = even element, preshuffled (shuffle_weight_w4)
    sa    uint8 [M, K/32]   E8M0, shuffle_scale_w4 layout
    sb    uint8 [N, K/32]   E8M0, shuffle_scale_w4 layout
    bias  bf16 [N] or None  added to the bf16 product, rounding like a separate `c + bias`
    M is a multiple of M_ALIGN; N and K are the FLUX shapes (N % 256 == 0, K % 256 == 0).
    An aq whose row is K bytes long is read as the older FP8-padded packing instead.

    keep_preact leaves C holding the bias-added accumulator rather than the GELU of it,
    which is the tensor the MLP's backward saves; the operands still carry the activation.

    act_rounded says where the activation starts, which is the only thing the two callers
    disagree on. A consumer that read C back saw the rounded value, so it wants 1; one that
    was handed the GEMM's bias-free output and added the bias itself, in f32 ahead of its
    own GELU, saw the unrounded sum and wants 0. Both are byte-exact against their own
    caller and neither is against the other's. Only meaningful with `gelu` and a level-2
    `row_quant`.

    row_quant additionally writes C's own A6W4 operands -- what
    flydsl_quant_a6w4_act_dual would have produced by reading C back -- while the
    accumulators are still in registers: 1 = the row half (aq, sa), attached to C as
    `c.a6w4_row`; 2 = both halves (aq, sa, co, cs), attached as `c.a6w4_dual`. It is a
    hint either way: shapes whose config cannot host the fused store (see rq in
    launch_gemm) just ignore it, so the caller must check the attribute rather than assume
    it. None (the default) takes level 2 under `gelu` -- the fused-MLP path, where the
    quantiser is always the next thing to run -- and nothing otherwise.

    The row half alone is a net loss, and not for the reason its own timing suggests.
    Measured on 8192x12288x3072, the fused epilogue costs +39 us against a 33.5 us budget
    (what the standalone quantiser gives back by dropping to its col-only kernel), so the
    two kernels in isolation come to only +5.5 us; run as the chain the bench scores, the
    same pair costs +17.5 us (257.3 -> 274.8), and 16384x12288x3072 goes +43.4 us. The gap
    is traffic, not VALU: the row pack adds 103 MB (206 MB at the larger shape) of writes on
    top of C, which pushes C out of the last-level cache before the col pass reads it back.
    The col kernel's own numbers show the same effect directly -- 6.56 TB/s when C is 201 MB
    and fits, 4.80 TB/s when it is 402 MB and does not. The row pack moves no bytes on its
    own (the chain writes 558 MB either way), so it cannot pay for the cache it costs; the
    byte saving is the 201 MB re-read of C, which is why level 2 exists.

    Level 2 is a win, and most of it is not the bytes either. The chain goes 257.3 -> 249.8
    us at 8192 just by fusing the col pack, which is about what the removed re-read is worth;
    the other two thirds come from C's store turning non-temporal, which it may only do once
    nothing reads C back (249.8 -> 232.2, and 490.4 -> 471.7 at 16384). Both packs cost
    +9.8 v_* per output element over the plain GELU epilogue and the whole fused kernel still
    fits in 508 VGPRs with accum_offset 252, no spill, and 132096 B of LDS.

    Correctness is not the open question -- all four packs are byte-identical to the
    standalone quantiser's, which is the gate _bench_gelu_epi.py applies.
    """
    import torch

    m = aq.shape[0]
    a_pack = 0 if aq.shape[-1] == k else 1  # a K-byte row is the older FP8-padded packing
    cfg = _A6W4_CONFIGS.get((n, k), _A6W4_DEFAULT)
    tm, tn, tk, wpe, gm, pad16, bpf, sa_lds, epi, xcd, a_stg, a_pipe, fi, store_mode, ahead = cfg
    rq = _rq_level(m, n, k, 2 if row_quant is None and gelu else int(row_quant or 0), a_pack)
    c = torch.empty(m, n, dtype=torch.bfloat16, device=aq.device)
    if rq:
        # An FP6 row padded to FP8's length, scales in shuffle_scale_w4 order.
        c_rq = torch.empty(m, n, dtype=torch.uint8, device=aq.device)
        c_rs = torch.empty(m * n // MX_BLOCK, dtype=torch.uint8, device=aq.device)
    if rq >= 2:
        c_cq = torch.empty(n, m // 2, dtype=torch.float4_e2m1fn_x2, device=aq.device)
        c_cs = torch.empty(n, m // MX_BLOCK, dtype=torch.float8_e8m0fnu, device=aq.device)
    ptr = lambda t: flyc.from_c_void_p(fx.Uint8, t.data_ptr())
    if bias is not None:
        bias = bias.to(torch.bfloat16).contiguous()
    args = (
        ptr(c),
        ptr(aq),
        ptr(wq),
        ptr(sa),
        ptr(sb),
        m,
        n,
        torch.cuda.current_stream(),
        n,
        k,
        tm,
        tn,
        tk,
        "fp6",
        "bf16",
        1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        wpe,
        gm,
        xcd,
        pad16,
        bpf,
        sa_lds,
        epi,
        ptr(c if bias is None else bias),
        bias is not None,
        a_stg,
        a_pipe,
        fi,
        store_mode,
        ahead,
        a_pack,
        int(gelu),
        int(keep_preact),
        int(act_rounded),
        ptr(c_rq if rq else c),
        ptr(c_rs if rq else c),
        ptr(c_cq if rq >= 2 else c),
        ptr(c_cs if rq >= 2 else c),
        rq,
    )
    if torch.cuda.is_current_stream_capturing():
        # A flyc.compile'd object regresses under CUDA-graph capture (no recompile mid-capture);
        # fall back to the raw @flyc.jit closure, same rule gemm_mxfp4_kernel.py follows.
        launch_gemm(*args)
    else:
        run_compiled(
            _A6W4_LAUNCH_CACHE,
            (n, k, bias is not None, a_pack, bool(gelu), rq, bool(keep_preact), bool(act_rounded)),
            launch_gemm,
            *args,
        )
    if rq >= 2:
        c.a6w4_dual = (c_rq, c_rs, c_cq, c_cs)
    elif rq:
        c.a6w4_row = (c_rq, c_rs)
    return c
