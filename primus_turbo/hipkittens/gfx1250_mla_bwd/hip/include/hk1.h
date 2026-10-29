/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// hk1.h -- the raw gfx1250 builtins the bwd_hk_k1 kernels use (HIP, amdclang++ 23, wave32).
//
// Each wrapper lowers to exactly one instruction (or to the fence/barrier sequence named), with the
// operand order of the LLVM intrinsic FlyDSL's k_dkdv64 calls (its LLVM IR dump), so a port that
// feeds the same values in the same lanes produces the same bits. No HipKittens header is included:
// HK's cdna5 attention-side primitives are the defective ones (MFMA mma_AB/AtB, gfx950 transposed
// loads, permlane32 reductions, wait_barrier without s_wait_dscnt); a checked hk_ext replaces them
// later.
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <utility>

namespace hk1 {

typedef __bf16                                   bf16;
typedef __bf16                                   bf16x8 __attribute__((ext_vector_type(8)));
typedef __bf16                                   bf16x16 __attribute__((ext_vector_type(16)));
typedef float                                    f32x4 __attribute__((ext_vector_type(4)));
typedef float                                    f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t                                 u32x4 __attribute__((ext_vector_type(4)));
typedef int32_t                                  i32x4 __attribute__((ext_vector_type(4)));
typedef int32_t                                  i32x8 __attribute__((ext_vector_type(8)));
typedef int32_t                                  i32x16 __attribute__((ext_vector_type(16)));
typedef __attribute__((address_space(3))) bf16x8 lds_bf16x8;
typedef __attribute__((address_space(3))) unsigned char lds_u8;

#define HK1_INLINE __device__ __forceinline__

// ---- compile-time loops: f(std::integral_constant<int, I>{}) for I in [0, N)
// -------------------------
template <int N, typename F, int... I>
HK1_INLINE void static_for_impl(F &&f, std::integer_sequence<int, I...>) {
    (f(std::integral_constant<int, I>{}), ...);
}
template <int N, typename F> HK1_INLINE void static_for(F &&f) {
    static_for_impl<N>(static_cast<F &&>(f), std::make_integer_sequence<int, N>{});
}

// ---- vectors
// ------------------------------------------------------------------------------------------
HK1_INLINE bf16x16 cat(bf16x8 a, bf16x8 b) {
    return __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}
// fptrunc f32 -> bf16 (v_cvt_pk_bf16_f32, round to nearest even), FlyDSL's .to(BFloat16)
HK1_INLINE bf16 to_bf16(float x) {
    return (bf16) x;
}
HK1_INLINE bf16x8 to_bf16x8(f32x8 v) {
    return bf16x8{to_bf16(v[0]), to_bf16(v[1]), to_bf16(v[2]), to_bf16(v[3]),
                  to_bf16(v[4]), to_bf16(v[5]), to_bf16(v[6]), to_bf16(v[7])};
}

// ---- WMMA 16x16x32 bf16 -> f32 (llvm.amdgcn.wmma.f32.16x16x32.bf16, neg 0, c_mod 0)
// --------------------
template <bool REUSE_A, bool REUSE_B> HK1_INLINE f32x8 wmma(bf16x16 a, bf16x16 b, f32x8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x32_bf16(false, a, false, b, (short) 0, c, REUSE_A,
                                                   REUSE_B);
}

// ---- LDS (addresses are byte offsets in the workgroup's LDS)
// -----------------------------------------
HK1_INLINE uint32_t lds_addr(const void *p) {
    return (uint32_t) (uintptr_t) p;
}
HK1_INLINE bf16x8 lds_load_b128(uint32_t a) {
    return *(const lds_bf16x8 *) (uintptr_t) a;
}
HK1_INLINE void lds_store_b128(uint32_t a, bf16x8 v) {
    *(lds_bf16x8 *) (uintptr_t) a = v;
}
// ds_load_tr16_b128 (llvm.amdgcn.ds.load.tr16.b128.v8bf16)
HK1_INLINE bf16x8 lds_tr16(uint32_t a) {
    return __builtin_amdgcn_ds_load_tr16_b128_v8bf16((lds_bf16x8 *) (uintptr_t) a);
}
// base pointer + compile-time byte offset: a GEP, so the constant stays in the DS offset field
// instead of being re-associated into one loop-invariant VGPR per offset (plain i32 sums do that
// under LICM)
HK1_INLINE const lds_u8 *lds_ptr(uint32_t a) {
    return (const lds_u8 *) (uintptr_t) a;
}
HK1_INLINE bf16x8 lds_load_b128(const lds_u8 *p, int off) {
    return *(const lds_bf16x8 *) (p + off);
}
HK1_INLINE bf16x8 lds_tr16(const lds_u8 *p, int off) {
    return __builtin_amdgcn_ds_load_tr16_b128_v8bf16((lds_bf16x8 *) (p + off));
}

// ---- buffer resources and raw buffer ops (llvm.amdgcn.make.buffer.rsrc / raw.ptr.buffer.*)
// ----------- flags 159744 = FlyDSL's make_buffer_tensor word-3 flags on gfx1250 (k_dkdv64_0
// 21_llvm_ir.ll)
inline constexpr int              RSRC_FLAGS = 159744;
HK1_INLINE __amdgpu_buffer_rsrc_t rsrc(const void *base, int nbytes) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void *>(base), (short) 0, (long) nbytes,
                                             RSRC_FLAGS);
}
HK1_INLINE bf16x8 buf_load_b128(__amdgpu_buffer_rsrc_t r, int voff, int soff) {
    return __builtin_bit_cast(bf16x8, __builtin_amdgcn_raw_buffer_load_b128(r, voff, soff, 0));
}
HK1_INLINE f32x4 buf_load_f32x4(__amdgpu_buffer_rsrc_t r, int voff, int soff) {
    return __builtin_bit_cast(f32x4, __builtin_amdgcn_raw_buffer_load_b128(r, voff, soff, 0));
}
HK1_INLINE void buf_store_b128(bf16x8 v, __amdgpu_buffer_rsrc_t r, int voff, int soff) {
    __builtin_amdgcn_raw_buffer_store_b128(__builtin_bit_cast(u32x4, v), r, voff, soff, 0);
}

// ---- Tensor Data Mover (2-D tensor_load_to_lds, clang-23 6-operand form)
// ------------------------------
HK1_INLINE void tdm_load(i32x4 g0, i32x8 g1) {
    const i32x4 z4 = {0, 0, 0, 0};
    const i32x8 z8 = {0, 0, 0, 0, 0, 0, 0, 0};
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, z4, z4, z8, 0);
}
template <int N> HK1_INLINE void tensor_wait() {
    __builtin_amdgcn_s_wait_tensorcnt((short) N);
}

// ---- scheduling / synchronisation
// ---------------------------------------------------------------------
template <int MASK> HK1_INLINE void sched_barrier() {
    __builtin_amdgcn_sched_barrier(MASK);
}
template <int MASK, int N, int ID> HK1_INLINE void sched_group_barrier() {
    __builtin_amdgcn_sched_group_barrier(MASK, N, ID);
}
// FlyDSL _lds_barrier(): workgroup release fence on LDS only (s_wait_dscnt 0: this wave's DS ops
// retired; in-flight global loads are NOT waited for), s_barrier_signal -1 / s_barrier_wait -1,
// acquire fence. Memory orders as their C11 values (hk_ext/sync.h does the same): a fence is not an
// atomic operation, but the HK-line source lint rejects the macro spelling.
inline constexpr int MO_ACQUIRE = 2, MO_RELEASE = 3;
HK1_INLINE void      lds_barrier() {
    __builtin_amdgcn_fence(MO_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
    __builtin_amdgcn_fence(MO_ACQUIRE, "workgroup", "local");
}
// split form of lds_barrier() (HipKittens cdna5 barrier.cuh arrive()/wait(), rung 08 split barrier): arrive =
// release fence (s_wait_dscnt 0: this wave's DS ops retired) + s_barrier_signal -1; wait = s_barrier_wait -1 +
// acquire fence. Work placed between the two (pinned with sched_barrier<0>) overlaps the other wave's arrival.
HK1_INLINE void lds_arrive() {
    __builtin_amdgcn_fence(MO_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier_signal(-1);
}
HK1_INLINE void lds_wait() {
    __builtin_amdgcn_s_barrier_wait(-1);
    __builtin_amdgcn_fence(MO_ACQUIRE, "workgroup", "local");
}
// lock_simd: SCHED_MODE.DISABLE_XDL_ARB_STALL (hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1) = 1), FlyDSL
// rocdl.disable_xdl_arb_stall() == llvm.amdgcn.s.setreg(i32 154, i32 1); the one s_setreg the
// HK-line lint and isa_static S2 allow.
HK1_INLINE void lock_simd() {
    __builtin_amdgcn_s_setreg(154, 1);
}
// wave index in the workgroup as a uniform SGPR (FlyDSL: llvm.amdgcn.wave.id)
HK1_INLINE int wave_id(int tid) {
    return __builtin_amdgcn_readfirstlane(tid) >> 5;
}
HK1_INLINE int uniform(int x) {
    return __builtin_amdgcn_readfirstlane(x);
}
HK1_INLINE float exp2_raw(float x) {
    return __builtin_amdgcn_exp2f(x);
} // v_exp_f32, llvm.amdgcn.exp2

} // namespace hk1
