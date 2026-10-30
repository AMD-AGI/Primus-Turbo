/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// dq_prims.h -- the gfx1250 primitives of k_dqds_c (wave32) as raw clang builtins, namespace q1.
// Each lowers to the LLVM intrinsic the FlyDSL MLA backward emits for the same operation:
//   wmma_bf16     llvm.amdgcn.wmma.f32.16x16x32.bf16 (neg / c_mod / reuse all 0)
//   tr16          llvm.amdgcn.ds.load.tr16.b128.v8bf16
//   lds_ld / st   plain addrspace(3) <8 x bfloat> load / store (ds_load_b128 / ds_store_b128)
//   buf_st128     llvm.amdgcn.make.buffer.rsrc + raw.ptr.buffer.store (i128)
//   tdm_load      llvm.amdgcn.tensor.load.to.lds, groups 2..4 zero
//   Q1_WAIT_*     llvm.amdgcn.s.wait.dscnt (through an asm label: there is no builtin for it, and
//                 the legacy __builtin_amdgcn_s_waitcnt cannot be selected on gfx1250) /
//                 llvm.amdgcn.s.wait.tensorcnt
//   wg_sync       dscnt 0, then __syncthreads() (release fence, s_barrier_signal -1 /
//                 s_barrier_wait -1, acquire fence)
//   lock_simd     SCHED_MODE.DISABLE_XDL_ARB_STALL
#pragma once
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
Q1_DEV bf16x16 cat16(bf16x8 a, bf16x8 b) {
    return __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}
Q1_DEV f32x8 wmma_bf16(bf16x16 a, bf16x16 b, f32x8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x32_bf16(false, a, false, b, (short) 0, c, false, false);
}
Q1_DEV f32x8 zero8() {
    return f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
}

// ---- identity, conversion
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
// fp32 -> bf16 x8 (RNE): the dQ epilogue's C fragment pack (fptrunc <8 x float>, 4 x
// v_cvt_pk_bf16_f32)
Q1_DEV bf16x8 cvt_bf16x8(f32x8 x) {
    return __builtin_convertvector(x, bf16x8);
}

// ---- LDS: the kernel's single static allocation (q1_smem: the K and dS rings, then the per-wave
// dQ images) starts at LDS address 0; these take byte offsets from its base
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

// ---- buffer (V#) stores
Q1_DEV rsrc_t make_rsrc(const void *p, long long nbytes, int flags) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void *>(p), (short) 0, nbytes, flags);
}
Q1_DEV void buf_st128(rsrc_t r, int voff, bf16x8 v) {
    __builtin_amdgcn_raw_buffer_store_b128(__builtin_bit_cast(u32x4, v), r, voff, 0, 0);
}

// ---- TDM
Q1_DEV void tdm_load(u32x4 g0, u32x8 g1) {
    i32x4 z4 = {0, 0, 0, 0};
    i32x8 z8 = {0, 0, 0, 0, 0, 0, 0, 0};
    __builtin_amdgcn_tensor_load_to_lds(__builtin_bit_cast(i32x4, g0),
                                        __builtin_bit_cast(i32x8, g1), z4, z4, z8, 0);
}

// ---- waits, barrier, scheduling
extern "C" __device__ void q1_llvm_s_wait_dscnt(short) __asm("llvm.amdgcn.s.wait.dscnt");
#define Q1_WAIT_DSCNT(n) q1_llvm_s_wait_dscnt((short) (n))
#define Q1_WAIT_TENSORCNT(n) __builtin_amdgcn_s_wait_tensorcnt((short) (n))
#define Q1_SCHED_BARRIER() __builtin_amdgcn_sched_barrier(0)
// sched_group_barrier masks (llvm AMDGPU IGroupLP): 0x008 MFMA/WMMA, 0x002 VALU, 0x400 TRANS, 0x004
// SALU, 0x100 DS read
#define Q1_SGB(mask, n) __builtin_amdgcn_sched_group_barrier((mask), (n), 0)

// workgroup barrier: this wave's LDS reads retired (dscnt 0), then __syncthreads() = workgroup
// release fence, s_barrier_signal -1 / s_barrier_wait -1, workgroup acquire fence; sched_barrier(0)
// on both sides
Q1_DEV void wg_sync() {
    Q1_SCHED_BARRIER();
    Q1_WAIT_DSCNT(0);
    __syncthreads();
    Q1_SCHED_BARRIER();
}

// SCHED_MODE.DISABLE_XDL_ARB_STALL = 1: s_setreg of hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1) (encoded
// 154), as FlyDSL's rocdl.disable_xdl_arb_stall
Q1_DEV void lock_simd() {
    __builtin_amdgcn_s_setreg(154, 1);
}

} // namespace q1
