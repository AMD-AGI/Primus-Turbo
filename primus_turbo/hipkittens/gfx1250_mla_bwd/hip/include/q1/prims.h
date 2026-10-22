/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// q1/prims.h -- the gfx1250 primitives the bwd_hk_q1 kernels use, as raw clang-23 builtins.
//
// Local on purpose: hk_ext has no emulator-checked
// release yet, so this arm vendors the few wrappers it needs. Each one lowers to exactly the LLVM
// intrinsic the FlyDSL champion's k_dqg96 / k_dqg emit (their LLVM IR
// dumps), so the backend sees the same operations:
//   wmma_bf16     llvm.amdgcn.wmma.f32.16x16x32.bf16 (neg/c_mod/reuse all 0)
//   tr16          llvm.amdgcn.ds.load.tr16.b128.v8bf16
//   lds_ld/st     plain addrspace(3) <8 x bfloat> load / store (ds_load_b128 / ds_store_b128)
//   buf_*         llvm.amdgcn.make.buffer.rsrc + raw.ptr.buffer.load/store (i32 / i128)
//   tdm_load      llvm.amdgcn.tensor.load.to.lds, 6-operand clang-23 form, groups 2..4 zero
//   wait_*        llvm.amdgcn.s.wait.dscnt (via an asm label: clang 23 has no builtin for it; the
//                 legacy __builtin_amdgcn_s_waitcnt cannot be selected on gfx1250) /
//                 s.wait.tensorcnt
//   wg_sync       FlyDSL _wg_sync: dscnt 0, then __syncthreads() (release fence (wg),
//   s_barrier_signal/-wait -1,
//                 acquire fence (wg))
//   lock_simd     HipKittens kittens::sched::lock_simd (vendored, byte-identical)
// No inline asm, no named barriers, no atomics, no clusters.
#pragma once
#ifdef Q1_HOST_EMU
// CPU functional emulation only (one std::thread per lane; the emulator is not shipped); never part
// of a kernel build
#include "prims_host.h"
#else
#include "cdna5/ops/warp/sched/sched.cuh" // HK be1c9184, vendored byte-identical (lock_simd)
#include <hip/hip_runtime.h>

namespace q1 {

typedef __bf16                                   bf16x8 __attribute__((ext_vector_type(8)));
typedef __bf16                                   bf16x16 __attribute__((ext_vector_type(16)));
typedef float                                    f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned int                             u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int                             u32x8 __attribute__((ext_vector_type(8)));
typedef int                                      i32x4 __attribute__((ext_vector_type(4)));
typedef int                                      i32x8 __attribute__((ext_vector_type(8)));
typedef __attribute__((address_space(3))) char   lds_char;
typedef __attribute__((address_space(3))) bf16x8 lds_bf16x8;
typedef __amdgpu_buffer_rsrc_t                   rsrc_t;

#define Q1_DEV __device__ __attribute__((always_inline)) inline

// ---- compile-time loops: f(ic<i>{}) for i in [B, E)
// -------------------------------------------------
template <int N> struct ic {
    static constexpr int value = N;
};
template <int B, int E, class F> Q1_DEV void sfor(F &&f) {
    if constexpr (B < E) {
        f(ic<B>{});
        sfor<B + 1, E>(static_cast<F &&>(f));
    }
}

// ---- operands
// ---------------------------------------------------------------------------------------
Q1_DEV bf16x16 cat16(bf16x8 a, bf16x8 b) {
    return __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}
Q1_DEV f32x8 wmma_bf16(bf16x16 a, bf16x16 b, f32x8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x32_bf16(false, a, false, b, (short) 0, c, false, false);
}
Q1_DEV f32x8 zero8() {
    return f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
}

// ---- identity, transcendental
// --------------------------------------------------------------------
Q1_DEV int thread_id_x() {
    return (int) __builtin_amdgcn_workitem_id_x();
}
Q1_DEV int block_id_x() {
    return (int) __builtin_amdgcn_workgroup_id_x();
}
Q1_DEV int block_id_y() {
    return (int) __builtin_amdgcn_workgroup_id_y();
}
Q1_DEV int block_id_z() {
    return (int) __builtin_amdgcn_workgroup_id_z();
}
Q1_DEV int readfirstlane(int x) {
    return __builtin_amdgcn_readfirstlane(x);
}
Q1_DEV float exp2_hw(float x) {
    return __builtin_amdgcn_exp2f(x);
} // v_exp_f32 (FlyDSL rocdl.exp2)
// fp32 -> bf16 x8 (RNE): the dQ epilogue's C fragment pack (fptrunc <8 x float>, 4 x
// v_cvt_pk_bf16_f32)
Q1_DEV bf16x8 cvt_bf16x8(f32x8 x) {
    return __builtin_convertvector(x, bf16x8);
}

// fp32 -> bf16 (RNE, fptrunc = v_cvt_pk_bf16_f32) of 8 values, returned as 4 packed dwords. Packed
// per PAIR (fptrunc <2 x float> -> one cvt_pk) and kept in the integer domain until the WMMA
// operand is formed (cat16_u32): a vector fptrunc of 8 followed by the operand concat is
// canonicalised by InstCombine into fptrunc(concat), which moves all 8 cvt_pk of a dS pack from the
// softmax chunk that computes it to the head of the dQ chunk that consumes it (in front of its
// first WMMA). FlyDSL converts per element at the softmax site; this keeps the conversion there.
typedef float  f32x2 __attribute__((ext_vector_type(2)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
Q1_DEV u32x4   cvt_bf16x8_u32(f32x8 x) {
    u32x4 w;
    w[0] = __builtin_bit_cast(unsigned, __builtin_convertvector(f32x2{x[0], x[1]}, bf16x2));
    w[1] = __builtin_bit_cast(unsigned, __builtin_convertvector(f32x2{x[2], x[3]}, bf16x2));
    w[2] = __builtin_bit_cast(unsigned, __builtin_convertvector(f32x2{x[4], x[5]}, bf16x2));
    w[3] = __builtin_bit_cast(unsigned, __builtin_convertvector(f32x2{x[6], x[7]}, bf16x2));
    return w;
}
Q1_DEV bf16x16 cat16_u32(u32x4 a, u32x4 b) {
    return __builtin_bit_cast(bf16x16, __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7));
}

// ---- LDS
// -------------------------------------------------------------------------------------------- The
// single static allocation (the 3-stage K/V ring, later the dQ epilogue images) is declared by the
// kernel TU; these take byte addresses relative to its base, which is LDS address 0.
Q1_DEV bf16x8 lds_ld(lds_char *base, int off) {
    return *(lds_bf16x8 *) (base + off);
}
Q1_DEV void lds_st(lds_char *base, int off, bf16x8 v) {
    *(lds_bf16x8 *) (base + off) = v;
}
Q1_DEV bf16x8 tr16(lds_char *base, int off) {
    return __builtin_amdgcn_ds_load_tr16_b128_v8bf16((lds_bf16x8 *) (base + off));
}
Q1_DEV unsigned lds_addr(lds_char *base) {
    return (unsigned) (__SIZE_TYPE__) base;
}

// ---- buffer (V#) loads / stores
// ----------------------------------------------------------------------
Q1_DEV rsrc_t make_rsrc(const void *p, long long nbytes, int flags) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void *>(p), (short) 0, nbytes, flags);
}
Q1_DEV bf16x8 buf_ld128(rsrc_t r, int voff) {
    return __builtin_bit_cast(bf16x8, __builtin_amdgcn_raw_buffer_load_b128(r, voff, 0, 0));
}
Q1_DEV float buf_ld32f(rsrc_t r, int voff) {
    return __builtin_bit_cast(float, __builtin_amdgcn_raw_buffer_load_b32(r, voff, 0, 0));
}
Q1_DEV void buf_st128(rsrc_t r, int voff, bf16x8 v) {
    __builtin_amdgcn_raw_buffer_store_b128(__builtin_bit_cast(u32x4, v), r, voff, 0, 0);
}

// ---- TDM
// ----------------------------------------------------------------------------------------------
Q1_DEV void tdm_load(u32x4 g0, u32x8 g1) {
    i32x4 z4 = {0, 0, 0, 0};
    i32x8 z8 = {0, 0, 0, 0, 0, 0, 0, 0};
    __builtin_amdgcn_tensor_load_to_lds(__builtin_bit_cast(i32x4, g0),
                                        __builtin_bit_cast(i32x8, g1), z4, z4, z8, 0);
}

// ---- waits, barrier, scheduling
// --------------------------------------------------------------------
extern "C" __device__ void q1_llvm_s_wait_dscnt(short) __asm("llvm.amdgcn.s.wait.dscnt");
#define Q1_WAIT_DSCNT(n) q1_llvm_s_wait_dscnt((short) (n))
#define Q1_WAIT_TENSORCNT(n) __builtin_amdgcn_s_wait_tensorcnt((short) (n))
#define Q1_SCHED_BARRIER() __builtin_amdgcn_sched_barrier(0)
// sched_group_barrier masks (llvm AMDGPU IGroupLP): 0x008 MFMA/WMMA, 0x002 VALU, 0x400 TRANS, 0x004
// SALU
#define Q1_SGB(mask, n) __builtin_amdgcn_sched_group_barrier((mask), (n), 0)

// FlyDSL kernels._wg_sync (DQ_BARRIER_FENCE = True): every LDS read of this wave retired (dscnt 0),
// then workgroup release fence + barrier + workgroup acquire fence = HIP's __syncthreads()
// (__work_group_barrier: fence(release, "workgroup"); s_barrier; fence(acquire, "workgroup");
// s_barrier is s_barrier_signal -1 + s_barrier_wait -1 on gfx12, nothing between them), the IR
// FlyDSL emits; sched_barrier(0) on both sides.
Q1_DEV void wg_sync() {
    Q1_SCHED_BARRIER();
    Q1_WAIT_DSCNT(0);
    __syncthreads();
    Q1_SCHED_BARRIER();
}

// SCHED_MODE.DISABLE_XDL_ARB_STALL (lock_simd): HipKittens' kittens::sched::lock_simd (vendored
// byte-identical under include/cdna5/, HK be1c9184) = s_setreg(26 | 2 << 6, 1) = FlyDSL's
// rocdl.disable_xdl_arb_stall (154).
Q1_DEV void lock_simd() {
    kittens::sched::lock_simd();
}

} // namespace q1
#endif // Q1_HOST_EMU
