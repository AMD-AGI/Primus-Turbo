/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// hk_attn/gmem.h -- every global memory access of the attention kernels: buffer resources (V#) with
// TRUE byte extents and the raw buffer loads / stores / fp32 atomics on them. No flat or global_*
// pointer access to a data tensor exists in these kernels (TDM, tdm.h, is the other global path).
//
// Range check (raw buffer, stride 0; FlyDSL / REF, proven): an access entirely at or above
// num_records is dropped -- a load returns 0, a store / atomic does nothing, never a fault. An
// access that STRADDLES num_records (e.g. a b128 whose last dwords are out) is UNVERIFIED (per-dword
// or whole-access; t_gmem INFO probes buf_partial_oob_*): kernels put every extent on an access
// boundary and never rely on it. The offset checked is voffset + the instruction offset; whether
// soffset is included is UNVERIFIED on gfx1250 (INFO buf_soffset_range_check), so an offset that must
// be range-checked goes in voffset.
#pragma once
#include "types.h"

namespace hk_attn {

// make.buffer.rsrc flags operand. On gfx1250 this LLVM keeps only flags[3:0] (-> V# word-3 bits
// 31:28) and discards the rest (probe hk/lab/prims_rev/r_rsrc3.hip: each flag bit 4..31 alone
// yields the same SALU as 0), so FlyDSL's / REF hk1.h's 0x27000 contributes nothing: the proven V#
// has those bits 0 (default range-check mode). 0 = byte-identical code today, and stays that proven
// V# if a later LLVM maps more flag bits.
inline constexpr int RSRC_FLAGS = 0;
static_assert((RSRC_FLAGS & 0xF) == 0, "V# word-3 bits 31:28 must stay 0 (the proven descriptor)");

// buffer resource over [base, base + nbytes). nbytes is an int64 byte count; the backend packs all
// of it into the 45-bit num_records field (word 1 bits 31:25 = nbytes[6:0], words 2..3 = nbytes >>
// 7: exact to the byte, verified from the SALU in the disassembly). base and nbytes uniform (SGPRs).
HKA_DEV rsrc_t make_rsrc(const void *base, int64_t nbytes) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void *>(base), (short) 0, (long) nbytes,
                                             RSRC_FLAGS);
}

// ---- loads (buffer_load_b{32,64,128} vD, vOFF, s[rsrc], sOFF offen [offset:IMM]) ---------------------
HKA_DEV uint32_t buf_load_b32(rsrc_t r, int voff, int soff) {
    return __builtin_amdgcn_raw_buffer_load_b32(r, voff, soff, 0);
}
HKA_DEV u32x2 buf_load_b64(rsrc_t r, int voff, int soff) {
    return __builtin_amdgcn_raw_buffer_load_b64(r, voff, soff, 0);
}
HKA_DEV u32x4 buf_load_b128(rsrc_t r, int voff, int soff) {
    return __builtin_amdgcn_raw_buffer_load_b128(r, voff, soff, 0);
}
HKA_DEV float buf_load_f32(rsrc_t r, int voff, int soff) {
    return __builtin_bit_cast(float, buf_load_b32(r, voff, soff));
}
HKA_DEV f32x4 buf_load_f32x4(rsrc_t r, int voff, int soff) {
    return __builtin_bit_cast(f32x4, buf_load_b128(r, voff, soff));
}
HKA_DEV bf16x8 buf_load_bf16x8(rsrc_t r, int voff, int soff) {
    return __builtin_bit_cast(bf16x8, buf_load_b128(r, voff, soff));
}

// ---- stores (buffer_store_b{32,64,128}) ------------------------------------------------------------------
HKA_DEV void buf_store_b32(rsrc_t r, int voff, int soff, uint32_t v) {
    __builtin_amdgcn_raw_buffer_store_b32(v, r, voff, soff, 0);
}
HKA_DEV void buf_store_b64(rsrc_t r, int voff, int soff, u32x2 v) {
    __builtin_amdgcn_raw_buffer_store_b64(v, r, voff, soff, 0);
}
HKA_DEV void buf_store_b128(rsrc_t r, int voff, int soff, u32x4 v) {
    __builtin_amdgcn_raw_buffer_store_b128(v, r, voff, soff, 0);
}
HKA_DEV void buf_store_f32(rsrc_t r, int voff, int soff, float v) {
    buf_store_b32(r, voff, soff, __builtin_bit_cast(uint32_t, v));
}

// ---- fp32 atomics (buffer_atomic_add_f32, no return) ----------------------------------------------------
// Cache policy: FlyDSL's fused bwd dQ atomic (atomicrmw fadd monotonic syncscope("agent")) lowers to
// `global_atomic_add_f32 ... scope:SCOPE_DEV`; the buffer form gets the same scope through the aux
// operand: CPOL bits 4:3 = scope (0 CU, 8 SE, 16 DEV, 24 SYS), no TH_ATOMIC_RETURN (result unused).
inline constexpr int CPOL_SCOPE_DEV = 16;
HKA_DEV void buf_atomic_add_f32(rsrc_t r, int voff, int soff, float v) {
    (void) __builtin_amdgcn_raw_ptr_buffer_atomic_fadd_f32(v, r, voff, soff, CPOL_SCOPE_DEV);
}
// voffset no-wrap hint: declares 0 <= v (< 2^31). The backend folds a constant into the buffer
// instruction's offset field only when it can prove `v + C` does not wrap (nuw); for a sum of a lane
// term and a uniform base it cannot without this (each element then costs a v_add and a live VGPR;
// verified in the disassembly). Never pass a negative voffset through it (UB).
HKA_DEV int voff_nonneg(int v) {
    __builtin_assume(v >= 0);
    return v;
}
// dQ-style flush of one accumulator in C-fragment order: element i -> byte voff + i * SLOT_B. With
// voff = 4 * lane + (per-wave 128-B line) + (per-step tile base) every one of the 8 atomics is one
// aligned, contiguous 128-B line (layout.h cfrag_lane), and i * SLOT_B lands in the instruction
// offset field (`offset:512` ...; precondition voff >= 0, see voff_nonneg). Keep the per-step base
// in voff, not soff: with a loop-invariant voff LICM hoists the 8 sums out of the loop into 8 live
// VGPRs.
template <int SLOT_B> HKA_DEV void atomic_add_c(rsrc_t r, int voff, int soff, acc_t c) {
    static_assert(SLOT_B >= 0 && 7 * SLOT_B < (1 << 23), "element offsets fit the offset field");
    const int v = voff_nonneg(voff);
    static_for<8>([&](auto i) { buf_atomic_add_f32(r, v + (int) i * SLOT_B, soff, c[(int) i]); });
}

// ---- row vectors (LSE, delta, NL) ---------------------------------------------------------------------------
// accumulator-row order load: lane (r, h) gets v[i] = X[row0 + 8h + i] (fp32) for the vector X at
// byte voff_row0 (+ soff): 2 x buffer_load_b128 at voff_row0 + 32h and + 16 (bwd: LSE / delta of
// the query rows of an S = Q K^T accumulator).
HKA_DEV acc_t load_rows8_f32(rsrc_t r, int voff_row0, int soff, int lane) {
    const int   vo = voff_row0 + lay::rows8_lane(lane);
    const f32x4 a = buf_load_f32x4(r, vo, soff), b = buf_load_f32x4(r, vo + 16, soff);
    return __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7);
}
// column store: lanes 0..15 store v (the value of column r) to byte voff_col0 + 4r (+ soff); lanes
// 16..31 (which hold the same column after a col_* reduction) store nothing (exec-masked branch;
// fwd: LSE of the queries of an S^T accumulator).
HKA_DEV void store_cols16_f32(rsrc_t r, int voff_col0, int soff, float v, int lane) {
    if (lane < 16) buf_store_f32(r, voff_col0 + lay::cols16_lane(lane), soff, v);
}
// C^T as bf16 rows straight from registers: lane (r, h) stores bf16(c[0..7]) (RNE) to byte voff +
// (soff) where the caller's voff = (row n0 + r) * pitch + 2 * (m0 + 8h), i.e. layout.h ct_lane with
// the global row pitch: the fwd O^T accumulator (lane = q, rows = d) -> 16 B of O row q.
HKA_DEV void buf_store_ct_bf16(rsrc_t r, int voff, int soff, acc_t c) {
    buf_store_b128(r, voff, soff, cvt_pk_bf16(c));
}

} // namespace hk_attn
