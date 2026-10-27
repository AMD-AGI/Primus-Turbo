/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// hk_attn/frag.h -- register-only work on accumulators: C -> operand pack, causal masks, row /
// column max and sum, exp2 and the fused softmax argument. Lane maps: layout.h. Accumulator c of
// lane (r, h) holds C[8h + i][r], i = 0..7 (column r, rows 8h..8h+7).
#pragma once
#include "types.h"

namespace hk_attn {

// ---- C -> operand pack (4 x v_cvt_pk_bf16_f32 per accumulator, no data movement)
// --------------------- c0 = rows [0, 16), c1 = rows [16, 32) of a 32 x 16 fp32 matrix X (rows =
// the next GEMM's k):
//   result element e of lane (r, h) = bf16(X[op_k(l, e)][r]):  e 0..7 = c0[e], e 8..15 = c1[e - 8].
// As a B operand that is B = X; as an A operand A = X^T. RNE rounding (fptrunc).
HKA_DEV opnd_t pack_k32(acc_t c0, acc_t c1) {
    return cat_u32(cvt_pk_bf16(c0), cvt_pk_bf16(c1));
}
// fwd: S^T accumulators of kv sub-tiles 2t (kv 32t + [0,16)) and 2t + 1 (kv 32t + [16,32)), lane =
// q
// -> the B operand P^T [kv][q] of O^T += V^T P^T (FlyDSL _pv_gemm: p[2kt].shuffle(p[2kt + 1])).
HKA_DEV opnd_t pack_pt_fwd(acc_t st_kv_lo, acc_t st_kv_hi) {
    return pack_k32(st_kv_lo, st_kv_hi);
}
// bwd: S (or dS) accumulators of q sub-tiles (q 0..15 and 16..31 of a 32-query step), lane = kv
// -> the B operand P [q][kv] of dV^T += dO^T P (dS -> dK^T += Q^T dS); ASM bwd / REF k_dkdv pack.
HKA_DEV opnd_t pack_p_bwd(acc_t s_q_lo, acc_t s_q_hi) {
    return pack_k32(s_q_lo, s_q_hi);
}

// ---- masks (per element: v_cmp_{lt,gt}_i32 vs the lane threshold + v_cndmask_b32; branch-free)
// ------- element i masked iff i < thr (mask_lt) / i > thr (mask_gt); masked elements become `neg`.
// Which value to mask, and with what:
//   fwd: mask S^T with -inf (the default) BEFORE the max; softmax_arg(-inf, c, nm) = -inf -> exp2 0
//        only while nm is finite, i.e. the running max is seeded finite (FlyDSL BIG_NEG -1e30).
//   bwd: mask the exp2 ARGUMENT t = softmax_arg(s, c, NL) AFTER the fma, with the finite MASK_NEG
//        (REF k_dkdv96ds K1_NEG, FlyDSL fused NEG). Masking S first breaks a query row with no
//        visible key (LSE = -inf, NL = +inf; possible when Sq > Skv): fma(-inf, c, +inf) = NaN and
//        fma(MASK_NEG, c, +inf) = +inf poison dV / dK, while mask-after-fma gives exp2(MASK_NEG) =
//        0.
inline constexpr float MASK_NEG = -3.0e38f;
HKA_DEV acc_t          mask_lt(acc_t c, int thr, float neg) {
    static_for<8>([&](auto i) { c[(int) i] = ((int) i < thr) ? neg : c[(int) i]; });
    return c;
}
HKA_DEV acc_t mask_gt(acc_t c, int thr, float neg) {
    static_for<8>([&](auto i) { c[(int) i] = ((int) i > thr) ? neg : c[(int) i]; });
    return c;
}
// tile-local column n > row m + off -> neg (off wave-uniform). Bottom-right causal on S = Q K^T
// (rows q from m0, cols kv from n0): off = m0 + cshift - n0, cshift = Skv - Sq. In the bwd apply it
// to the post-fma exp2 argument with neg = MASK_NEG (see above), not to S.
HKA_DEV acc_t mask_col_gt_row(acc_t c, int lane, int off, float neg = -__builtin_inff()) {
    return mask_lt(c, lay::thr_cgr(lane, off), neg);
}
// tile-local row m > column n + off -> neg. Bottom-right causal on S^T = K Q^T (rows kv from m0,
// cols q from n0): off = n0 + cshift - m0.
HKA_DEV acc_t mask_row_gt_col(acc_t c, int lane, int off, float neg = -__builtin_inff()) {
    return mask_gt(c, lay::thr_rgc(lane, off), neg);
}

// ---- reductions
// ------------------------------------------------------------------------------------- Max uses
// fmaxf (v_max_num_f32 / v_max3_num_f32; gfx1250 has no IEEE mode, so no canonicalize); inputs are
// never NaN (masked entries are -inf or a finite big negative). Sums are fp32 adds in a fixed
// balanced tree: over the flattened index j = t * 8 + i of NT tiles, reduce(j0, j1) = op(reduce(j0,
// mid), reduce(mid, j1)), mid = j0 + (j1 - j0) / 2; the tree's pairing fixes the bits (no
// reassociation without fast-math; t_reduce uses inputs whose sum depends on the order).
struct op_max {
    HKA_DEV float operator()(float a, float b) const { return __builtin_fmaxf(a, b); }
};
struct op_add {
    HKA_DEV float operator()(float a, float b) const { return a + b; }
};
template <int J0, int J1, int NT, typename OP>
HKA_DEV float tree_reduce(const acc_t (&c)[NT], OP op) {
    if constexpr (J1 - J0 == 1) {
        return c[J0 / 8][J0 % 8];
    } else {
        constexpr int MID = J0 + (J1 - J0) / 2;
        return op(tree_reduce<J0, MID>(c, op), tree_reduce<MID, J1>(c, op));
    }
}
// in-lane: over the 8 * NT values of this lane (rows 8h + i of every tile, column r)
template <int NT> HKA_DEV float lane_max(const acc_t (&c)[NT]) {
    return tree_reduce<0, 8 * NT>(c, op_max{});
}
template <int NT> HKA_DEV float lane_sum(const acc_t (&c)[NT]) {
    return tree_reduce<0, 8 * NT>(c, op_add{});
}
// v_permlanex16_b32 with the identity selects (0x76543210, 0xFEDCBA98): the value of lane l ^ 16
HKA_DEV float xhalf(float v) {
    const int x = __builtin_bit_cast(int, v);
    return __builtin_bit_cast(
        float, __builtin_amdgcn_permlanex16(x, x, 0x76543210, (int) 0xFEDCBA98u, false, false));
}
// column reductions over all 16 * NT rows of column r (the fwd softmax over kv on S^T tiles):
// in-lane tree, then one permlanex16 with the other half; lanes r and r + 16 get identical bits.
template <int NT> HKA_DEV float col_max(const acc_t (&c)[NT]) {
    const float v = lane_max(c);
    return __builtin_fmaxf(v, xhalf(v));
}
template <int NT> HKA_DEV float col_sum(const acc_t (&c)[NT]) {
    const float v = lane_sum(c);
    return v + xhalf(v);
}
// DPP16 row_xmask:M (v_*_dpp; bank/row masks 0xF): the value of lane l ^ M, M < 16 (within a half)
template <int M> HKA_DEV float xor16(float v) {
    static_assert(M > 0 && M < 16, "row_xmask stays inside a 16-lane row");
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v),
                                                                 0x160 | M, 0xF, 0xF, false));
}
// row reductions of one tile over its 16 columns (the 16 lanes of a half): butterfly xmask 1, 2, 4,
// 8 (folded into v_max_num_f32_dpp / v_add_f32_dpp); element i of every lane of half h = the
// reduction of row 8h + i, identical bits in all 16 lanes (the sum pairs lanes n, n ^ 1, then the
// pairs n, n ^ 2, ...).
HKA_DEV acc_t row_max(acc_t c) {
    static_for<8>([&](auto i) {
        float v    = c[(int) i];
        v          = __builtin_fmaxf(v, xor16<1>(v));
        v          = __builtin_fmaxf(v, xor16<2>(v));
        v          = __builtin_fmaxf(v, xor16<4>(v));
        c[(int) i] = __builtin_fmaxf(v, xor16<8>(v));
    });
    return c;
}
HKA_DEV acc_t row_sum(acc_t c) {
    static_for<8>([&](auto i) {
        float v    = c[(int) i];
        v          = v + xor16<1>(v);
        v          = v + xor16<2>(v);
        v          = v + xor16<4>(v);
        c[(int) i] = v + xor16<8>(v);
    });
    return c;
}

// ---- exp2 and the softmax argument
// ------------------------------------------------------------------- v_exp_f32 (llvm.amdgcn.exp2:
// no denormal range reduction, FlyDSL rocdl.exp2); exp2(-inf) = 0.
HKA_DEV float exp2_hw(float x) {
    return __builtin_amdgcn_exp2f(x);
}
// -(m * c) (v_mul_f32 with a neg modifier): the per-column bias of softmax_arg, c = scale * log2(e)
HKA_DEV float neg_scaled(float m, float c) {
    return -(m * c);
}
// s * c - m * c as ONE fused op (v_fma_f32), nm = neg_scaled(m, c): p = exp2_hw(softmax_arg(...))
HKA_DEV float softmax_arg(float s, float c, float nm) {
    return __builtin_fmaf(s, c, nm);
}
// one accumulator: p[i] = exp2(fma(s[i], c, nm)) as 4 v_pk_fma_f32 + 8 v_exp_f32 (same fp32 FMA per
// element as softmax_arg, two per instruction; c / nm uniform per lane)
HKA_DEV acc_t exp2_tile(acc_t s, float c, float nm) {
    const f32x2 c2 = {c, c}, n2 = {nm, nm};
    acc_t       p;
    static_for<4>([&](auto j) {
        const f32x2 a = __builtin_elementwise_fma(f32x2{s[2 * j], s[2 * j + 1]}, c2, n2);
        p[2 * j]      = exp2_hw(a[0]);
        p[2 * j + 1]  = exp2_hw(a[1]);
    });
    return p;
}

} // namespace hk_attn
