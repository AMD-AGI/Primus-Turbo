/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// hk_attn/ext_fwd.h -- primitives the forward v0 (k_attn_fwd_v0) needs beyond prims.h
// (fwd_v0_design section 13 "Extra"). Each lowers to the instruction(s) in its comment
// (hip/tests/t_ext_fwd.hip, driver hk/lab/test_ext_fwd.py).
//
//   tdm_g1_words(w)        group 1 of a TDM op from 8 precomputed words (fwd/idx.h f0::G1: q_g1,
//   kv_g1,
//                          dummy_g1), so the kernel issues exactly the words its index header
//                          encodes
//   tdm_g1_dummy(w0)       group 1 of the 16-B dummy op: a REAL 8 x 1 element load (counted by
//   TENSORCNT,
//                          unlike a NULL descriptor, whose accounting is unverified), so every wave
//                          issues the same number of ops and every s_wait_tensorcnt count is a
//                          constant
//   ballot(p)              v_cmp -> SGPR lane mask (llvm.amdgcn.ballot.i32); a wave-uniform value
//   any_set(m)             uniform test of a ballot mask (s_cmp + s_cbranch_scc: the lazy-rescale
//   branch) log2_hw(x)             v_log_f32 (llvm.amdgcn.log: no denormal handling; LSE)
//   rcp_ieee(x)            IEEE 1 / x (v_div_scale / v_rcp / fma / v_div_fixup: correctly rounded)
//   pk_fma / pk_mul / pk_add   v_pk_{fma,mul,add}_f32 on f32x2 (a broadcast operand -> op_sel_hi, a
//                          negated one -> neg_lo / neg_hi)
//   max3(a, b, c)          v_max3_num_f32 (nested fmaxf; no NaN ever reaches it)
//   acc_pair(c, j)         elements 2j, 2j + 1 of an accumulator as an f32x2 (adjacent VGPRs: no
//   move)
#pragma once
#include "prims.h"

namespace hk_attn {

// ---- TDM
// ------------------------------------------------------------------------------------------
HKA_DEV u32x8 tdm_g1_words(const uint32_t (&w)[8]) {
    return u32x8{w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]};
}
// 16-B dummy: tile = tensor = 8 x 1 elements of 2 B at the op's global address (row 0 of a tensor
// that exists), any w0 (its pad interval, >= 256 B, is never reached by a 16-B row)
HKA_DEV u32x8 tdm_g1_dummy(uint32_t w0) {
    return u32x8{w0, 8u << 16, 1u << 16, 8u << 16, 1u, 0u, 0u, 0u};
}
inline constexpr int TDM_DUMMY_BYTES = 16;

// ---- wave-uniform control
// -------------------------------------------------------------------------
HKA_DEV uint32_t ballot(bool p) {
    return __builtin_amdgcn_ballot_w32(p);
}
HKA_DEV bool any_set(uint32_t m) {
    return m != 0u;
}

// ---- transcendental
// ---------------------------------------------------------------------------------
HKA_DEV float log2_hw(float x) {
    return __builtin_amdgcn_logf(x);
}
HKA_DEV float rcp_ieee(float x) {
    return 1.0f / x;
}

// ---- packed fp32 and max3
// ------------------------------------------------------------------------------
HKA_DEV f32x2 pk_fma(f32x2 a, f32x2 b, f32x2 c) {
    return __builtin_elementwise_fma(a, b, c);
}
HKA_DEV f32x2 pk_mul(f32x2 a, f32x2 b) {
    return a * b;
}
HKA_DEV f32x2 pk_add(f32x2 a, f32x2 b) {
    return a + b;
}
HKA_DEV float max3(float a, float b, float c) {
    return __builtin_fmaxf(__builtin_fmaxf(a, b), c);
}
template <int J> HKA_DEV f32x2 acc_pair(acc_t c) {
    static_assert(J >= 0 && J < 4, "accumulator pair index");
    return f32x2{c[2 * J], c[2 * J + 1]};
}

} // namespace hk_attn
