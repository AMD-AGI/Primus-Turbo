/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// hk_attn/types.h -- register types and compile-time helpers of the gfx1250 attention primitives
// (HIP, amdclang++ 24, wave32). Lane maps: layout.h.
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <utility>

#include "layout.h"

namespace hk_attn {

typedef __bf16                                   bf16;
typedef __bf16                                   bf16x2 __attribute__((ext_vector_type(2)));
typedef __bf16                                   bf16x8 __attribute__((ext_vector_type(8)));
typedef __bf16                                   bf16x16 __attribute__((ext_vector_type(16)));
typedef float                                    f32x2 __attribute__((ext_vector_type(2)));
typedef float                                    f32x4 __attribute__((ext_vector_type(4)));
typedef float                                    f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t                                 u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t                                 u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t                                 u32x8 __attribute__((ext_vector_type(8)));
typedef int32_t                                  i32x4 __attribute__((ext_vector_type(4)));
typedef int32_t                                  i32x8 __attribute__((ext_vector_type(8)));
typedef __amdgpu_buffer_rsrc_t                   rsrc_t;
typedef __attribute__((address_space(3))) unsigned char lds_u8;
typedef __attribute__((address_space(3))) u32x4  lds_u32x4;
typedef __attribute__((address_space(3))) bf16x8 lds_bf16x8;

// A or B operand of v_wmma_f32_16x16x32_bf16 (8 VGPRs; layout.h op_k) and its C / D accumulator
// (8 VGPRs; layout.h c_m / c_n)
typedef bf16x16 opnd_t;
typedef f32x8   acc_t;

#define HKA_DEV __device__ __forceinline__

// ---- compile-time loops: f(std::integral_constant<int, I>{}) for I in [0, N)
template <typename F, int... I>
HKA_DEV void static_for_impl(F &&f, std::integer_sequence<int, I...>) {
    (f(std::integral_constant<int, I>{}), ...);
}
template <int N, typename F> HKA_DEV void static_for(F &&f) {
    static_for_impl(static_cast<F &&>(f), std::make_integer_sequence<int, N>{});
}

// ---- vectors (register renames only; no instruction when the halves are adjacent VGPRs) ----------
HKA_DEV bf16x16 cat(bf16x8 lo, bf16x8 hi) {
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}
HKA_DEV bf16x16 cat_u32(u32x4 lo, u32x4 hi) {
    return __builtin_bit_cast(bf16x16, __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7));
}
HKA_DEV u32x4 lo_u32(acc_t c) {
    const u32x8 w = __builtin_bit_cast(u32x8, c);
    return __builtin_shufflevector(w, w, 0, 1, 2, 3);
}
HKA_DEV u32x4 hi_u32(acc_t c) {
    const u32x8 w = __builtin_bit_cast(u32x8, c);
    return __builtin_shufflevector(w, w, 4, 5, 6, 7);
}
HKA_DEV acc_t zero_acc() {
    return acc_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
}

// ---- fp32 -> bf16 (fptrunc, round to nearest even = v_cvt_pk_bf16_f32, one per element pair) ------
// Packed per PAIR and kept in the integer domain until an operand is formed (cat_u32): a vector
// fptrunc of 8 followed by the operand concat is canonicalised by InstCombine into fptrunc(concat),
// which moves the conversions from the softmax that produces them to the head of the WMMA run that
// consumes them (REF q1/prims.h cvt_bf16x8_u32). Dword j = bf16(c[2j]) | bf16(c[2j + 1]) << 16.
HKA_DEV uint32_t cvt_pk_bf16(float lo, float hi) {
    return __builtin_bit_cast(uint32_t, __builtin_convertvector(f32x2{lo, hi}, bf16x2));
}
HKA_DEV u32x4 cvt_pk_bf16(acc_t c) {
    return u32x4{cvt_pk_bf16(c[0], c[1]), cvt_pk_bf16(c[2], c[3]), cvt_pk_bf16(c[4], c[5]),
                 cvt_pk_bf16(c[6], c[7])};
}

// ---- identity -------------------------------------------------------------------------------------
// lane index (v_and_b32 of the 1-D workitem id; blocks are 1-D)
HKA_DEV int lane_id() {
    return (int) __builtin_amdgcn_workitem_id_x() & (lay::WAVE - 1);
}
// wave index in the workgroup as a uniform SGPR (v_readfirstlane_b32 + s_lshr_b32; an unsigned
// shift, so LLVM knows it is non-negative when it appears in a buffer offset)
HKA_DEV int wave_id() {
    return (int) ((uint32_t) __builtin_amdgcn_readfirstlane((int) __builtin_amdgcn_workitem_id_x()) >> 5);
}
// v_readfirstlane_b32: declares x wave-uniform (it must be: lanes other than the first are dropped)
HKA_DEV int uniform(int x) {
    return __builtin_amdgcn_readfirstlane(x);
}
// s_ballot (v_cmp + an SGPR mask): true in every lane iff p holds in any active lane
HKA_DEV bool wave_any(bool p) {
    return __builtin_amdgcn_ballot_w32(p) != 0u;
}

} // namespace hk_attn
