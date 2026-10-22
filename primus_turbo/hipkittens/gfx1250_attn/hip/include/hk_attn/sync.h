/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// hk_attn/sync.h -- waits, workgroup barriers, fences and scheduling controls of the attention
// kernels. No inline asm: every wait is an LLVM intrinsic (visible to the waitcnt pass and to
// expert-mode dependency tracking); clang 24 has no builtin for s_wait_{ds,load,store,km}cnt, so
// they are reached through asm labels (REF q1/prims.h).
#pragma once
#include "cdna5/ops/warp/sched/sched.cuh" // HK be1c9184 (== 8cde3dd1 for include/cdna5), vendored
#include "tdm.h"
#include "types.h"

extern "C" __device__ void hka_llvm_s_wait_dscnt(short) __asm("llvm.amdgcn.s.wait.dscnt");
extern "C" __device__ void hka_llvm_s_wait_loadcnt(short) __asm("llvm.amdgcn.s.wait.loadcnt");
extern "C" __device__ void hka_llvm_s_wait_storecnt(short) __asm("llvm.amdgcn.s.wait.storecnt");
extern "C" __device__ void hka_llvm_s_wait_kmcnt(short) __asm("llvm.amdgcn.s.wait.kmcnt");

namespace hk_attn {

// ---- counter waits (one s_wait_* each; N = outstanding ops of this wave left in flight) --------------
template <int N> HKA_DEV void wait_dscnt() {
    hka_llvm_s_wait_dscnt((short) N);
} // LDS (ds_*) ops
template <int N> HKA_DEV void wait_loadcnt() {
    hka_llvm_s_wait_loadcnt((short) N);
} // VMEM loads (buffer_load)
template <int N> HKA_DEV void wait_storecnt() {
    hka_llvm_s_wait_storecnt((short) N);
} // VMEM stores and no-return atomics
template <int N> HKA_DEV void wait_kmcnt() {
    hka_llvm_s_wait_kmcnt((short) N);
} // scalar memory (s_load), messages
template <int N> HKA_DEV void wait_asynccnt() {
    __builtin_amdgcn_s_wait_asynccnt((short) N);
} // async LDS<->global copies
// wait_tensorcnt<N>: tdm.h

// ---- scheduling (compile-time only; no instruction) ------------------------------------------------------
// llvm.amdgcn.sched_group_barrier / sched_barrier masks (AMDGPU IGroupLP SchedGroupMask)
namespace sg {
inline constexpr int NONE = 0x000, ALU = 0x001, VALU = 0x002, SALU = 0x004, WMMA = 0x008,
                     VMEM = 0x010, VMEM_READ = 0x020, VMEM_WRITE = 0x040, DS = 0x080,
                     DS_READ = 0x100, DS_WRITE = 0x200, TRANS = 0x400;
} // namespace sg
// sched_barrier(MASK): nothing crosses it except the instruction classes in MASK (0 = nothing)
template <int MASK = 0> HKA_DEV void sched_barrier() {
    __builtin_amdgcn_sched_barrier(MASK);
}
// sched_group_barrier(MASK, N, ID): the next N instructions of class MASK form a group of pipeline ID
template <int MASK, int N, int ID = 0> HKA_DEV void sched_group_barrier() {
    __builtin_amdgcn_sched_group_barrier(MASK, N, ID);
}
// IR-level compiler fence (fence syncscope("singlethread") seq_cst): no instruction; LLVM may not
// move memory accesses across it (sched_barrier covers the machine scheduler, this the IR passes)
HKA_DEV void compiler_fence() {
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
}
// both: no instruction; no memory access crosses it at IR level and nothing crosses it in the
// machine schedulers (e.g. between the protocol stores and the uncovered loads of split_signal_dscnt)
HKA_DEV void order_fence() {
    compiler_fence();
    sched_barrier<0>();
}

// ---- fences ------------------------------------------------------------------------------------------------
// C11 memory orders as values (a fence is not an atomic operation)
inline constexpr int MO_ACQUIRE = 2, MO_RELEASE = 3;
// workgroup release fence restricted to LDS (MMRA "local"): `s_wait_dscnt 0` only -- in-flight
// buffer loads / stores / fp32 atomics are NOT drained (the plain workgroup fence emits
// s_wait_storecnt_dscnt 0, which would wait for the dQ atomics). Verified in the disassembly.
HKA_DEV void lds_release() {
    __builtin_amdgcn_fence(MO_RELEASE, "workgroup", "local");
}
// workgroup acquire fence on LDS (explicit opt-in; no split-barrier primitive uses it): emits
// `s_wait_dscnt 0` whenever this wave has DS ops in flight (the fence must complete its paired load,
// so LLVM drains every one), e.g. the loads issued between a split signal and its wait. Verified in
// the disassembly (t_sync before the review fix; probe hk/lab/prims_rev/r_wait.hip).
HKA_DEV void lds_acquire() {
    __builtin_amdgcn_fence(MO_ACQUIRE, "workgroup", "local");
}

// ---- workgroup barrier (gfx12 split barrier, barrier id -1 = the workgroup barrier) --------------------
HKA_DEV void barrier_signal() {
    __builtin_amdgcn_s_barrier_signal(-1);
} // s_barrier_signal -1
HKA_DEV void barrier_wait() {
    __builtin_amdgcn_s_barrier_wait((short) -1);
} // s_barrier_wait 0xffff
// split barrier, LDS protocol: every LDS access of this wave before the signal is complete and
// visible to every wave after its wait. Signal = s_wait_dscnt 0 + s_barrier_signal -1 (nothing else:
// global loads / stores / atomics and TDM stay in flight). Between signal and wait any work that
// does not touch the LDS the barrier protects may run; a DS load issued in that window must not
// read bytes another wave writes after its wait (nothing orders the two).
HKA_DEV void split_signal_lds() {
    sched_barrier<0>();
    lds_release();
    barrier_signal();
    sched_barrier<0>();
}
// Wait = s_barrier_wait 0xffff and nothing else: DS loads issued between signal and wait stay in
// flight (FlyDSL _split_barrier_wait). No acquire fence: the signal side already drained the
// protocol's DS ops, and the acquire would add `s_wait_dscnt 0`; the intrinsic itself clobbers
// memory at IR level, the compiler fence states that intent without an instruction.
HKA_DEV void split_wait_lds() {
    sched_barrier<0>();
    barrier_wait();
    compiler_fence();
    sched_barrier<0>();
}
// signal with N younger DS ops left in flight (FlyDSL fused bwd W4_R1_LOADS): the caller guarantees
// that the N newest DS ops of this wave are loads not covered by the protocol (DS ops of a wave
// complete in order, so dscnt N retires every older one) and that none of them reads bytes another
// wave writes after the barrier. s_wait_dscnt N + s_barrier_signal -1.
// Compiler transformations that shrink the uncovered loads below N (or reorder them) break that
// guarantee silently:
//   * the machine scheduler moves a protocol store behind a load it provably does not alias;
//   * IR passes sink a protocol store past the loads;
//   * a load of bytes this wave just stored is forwarded (no instruction; t_sync reads another slot);
//   * a load whose result is unused is deleted, a partly used b128 is narrowed (t_sync's first
//     version became ds_load_b32), adjacent b32 / b64 loads merge into one ds_load_2addr_*.
// Hence: order_fence() between the last protocol DS op and the uncovered loads; uncovered loads
// are ds_load_b128 / ds_load_tr16_b128 whose results are fully used after the signal; and every
// kernel's ISA passes test_prims.py's structural check (isa_ds_signal: no DS store / atomic in
// flight at any s_barrier_signal, over every CFG path).
template <int N> HKA_DEV void split_signal_dscnt() {
    sched_barrier<0>();
    compiler_fence();
    wait_dscnt<N>();
    barrier_signal();
    compiler_fence();
    sched_barrier<0>();
}
// TDM ring variant: retire all but this wave's TW newest TDM ops, then the LDS signal (a wave's own
// TDM writes become visible to the workgroup through the barrier that follows its tensor wait)
template <int TW> HKA_DEV void split_signal_tdm_lds() {
    sched_barrier<0>();
    wait_tensorcnt<TW>();
    lds_release();
    barrier_signal();
    sched_barrier<0>();
}
// full LDS barrier = split_signal_lds + split_wait_lds
HKA_DEV void lds_barrier() {
    split_signal_lds();
    split_wait_lds();
}

// ---- wave scheduling state ------------------------------------------------------------------------------
// SCHED_MODE.DISABLE_XDL_ARB_STALL = 1 (s_setreg hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1): this
// wave issues WMMAs back to back without yielding the pipe to co-resident waves. Kernel-lifetime;
// HK kittens::sched::lock_simd == FlyDSL rocdl.disable_xdl_arb_stall (s_setreg 154).
HKA_DEV void lock_simd() {
    kittens::sched::lock_simd();
}
// s_setprio P (0..3): issue priority of this wave among the waves of its SIMD
template <int P> HKA_DEV void set_prio() {
    kittens::sched::set_priority<P>();
}
// SCHED_MODE[1:0] (hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2)): 0 = default (hardware VA_VDST / VM_VSRC
// interlocks: a DS / VMEM issued after a WMMA waits for it), 2 = expert (no interlocks; the
// compiler must insert s_wait_alu depctr_* itself).
// Expert mode is a BUILD property: `-mllvm -amdgpu-expert-scheduling-mode` makes LLVM emit
// `s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2` in the kernel prologue AND the
// dependency waits (FlyDSL fwd's "amdgpu-expert-scheduling-mode" llvm option). Switching to expert
// in a kernel built without it would run code with no dependency waits, so sched_mode_expert exists
// only when the build defines HK_ATTN_EXPERT_SCHED_BUILD next to that flag. sched_mode_default is
// always safe (the extra waits of an expert build are then merely redundant).
HKA_DEV void sched_mode_default() {
    __builtin_amdgcn_s_setreg(26 | (0 << 6) | (1 << 11), 0u);
}
#if defined(HK_ATTN_EXPERT_SCHED_BUILD)
HKA_DEV void sched_mode_expert() {
    __builtin_amdgcn_s_setreg(26 | (0 << 6) | (1 << 11), 2u);
}
#endif

} // namespace hk_attn
