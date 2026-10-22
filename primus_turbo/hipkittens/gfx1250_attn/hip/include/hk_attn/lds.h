/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// hk_attn/lds.h -- LDS accesses of the attention kernels: raw ds_load_b128 / ds_store_b128 /
// ds_load_tr16_b128, the two WMMA operand loaders and the accumulator (C^T) store.
//
// Addressing: a tile is a row-major bf16 image at an LDS byte address with row pitch PITCH bytes
// (any multiple of 16 works; the lane maps are pitch-independent).
// Pitch choice (bank model: 64 banks x 4 B, a b128 burst served 16 lanes at a time; NOT measured
// here, hk/mb mb_tr16pitch measures it): a rows load / C^T store of 16 lanes touches 16 rows x one
// 16-B chunk, conflict-free when the pitch is an odd multiple of 16 B (272 = 256 + 16, 80 = 64 +
// 16); a tr16 load touches 8 rows x two adjacent 16-B chunks, conflict-free when the pitch is an
// odd multiple of 32 B (288 = 256 + 32, 96 = 64 + 32; at 272 rows r and r + 1 collide, 2-way). So
// tiles read with load_rows use 272 / 80 and tiles read with load_tr (V, dO, K for dQ, dS^T) 288 /
// 96 (FlyDSL fwd: K pad 16 B, V pad 32 B). No pitch is conflict-free for both (dS^T: stored rows,
// read tr16). A loader is split into a per-lane base, computed once (lane part of layout.h), and a
// compile-time immediate per fragment, so every load of a burst is `ds_load_* vD, vBASE offset:IMM`
// (a GEP on the base pointer keeps the constant in the 16-bit DS offset field instead of a
// re-associated VGPR). Every lane address is 16-B aligned when the tile base is. All loads are
// plain addrspace(3) loads or the tr16 intrinsic, i.e. visible to LLVM's waitcnt insertion and to
// expert-mode dependency tracking.
#pragma once
#include "types.h"

// 16-B alignment precondition on a fragment coordinate: a compile error when the argument is a
// constant (every intended call: literals, static_for indices) that is not a multiple of 8
// elements; runtime arguments are not diagnosed. A misaligned b128 / tr16 address silently breaks
// the lane map.
#define HKA_MUL8(x, msg) __attribute__((diagnose_if(((x) & 7) != 0, msg, "error")))

namespace hk_attn {

// ---- raw accesses
// ------------------------------------------------------------------------------------- LDS byte
// address of a generic pointer to a __shared__ object (addrspacecast + ptrtoint, folded)
HKA_DEV uint32_t lds_addr(const void *p) {
    return (uint32_t) (uintptr_t) (const lds_u8 *) p;
}
HKA_DEV lds_u8 *lds_ptr(uint32_t a) {
    return (lds_u8 *) (uintptr_t) a;
}
// ds_load_b128 / ds_store_b128 at base + off
HKA_DEV bf16x8 ds_load_b128(const lds_u8 *base, int off) {
    return *(const lds_bf16x8 *) (base + off);
}
HKA_DEV u32x4 ds_load_b128_u32(const lds_u8 *base, int off) {
    return *(const lds_u32x4 *) (base + off);
}
HKA_DEV void ds_store_b128(lds_u8 *base, int off, bf16x8 v) {
    *(lds_bf16x8 *) (base + off) = v;
}
HKA_DEV void ds_store_b128(lds_u8 *base, int off, u32x4 v) {
    *(lds_u32x4 *) (base + off) = v;
}
// ds_load_tr16_b128 (llvm.amdgcn.ds.load.tr16.b128.v8bf16): every aligned group of 8 lanes
// transposes the 8x8 bf16 block formed by the 16 bytes each of its lanes addresses: element j of
// lane 8g + c = element c of the bytes addressed by lane 8g + j.
HKA_DEV bf16x8 ds_load_tr16_b128(const lds_u8 *base, int off) {
    return __builtin_amdgcn_ds_load_tr16_b128_v8bf16((lds_bf16x8 *) (base + off));
}

// ---- operand loader 1: "rows" (2 x ds_load_b128)
// -------------------------------------------------------- The tile's rows are the operand's lane
// dimension: an A operand from A [M][K] row-major, or a B operand from B^T [N][K] row-major (Q / K
// / dO / V in the S and dP GEMMs).
//   lane (r, h): element e = T[r0 + r][k0 + op_k(l, e)]
//   loads: row r0 + r, bytes 2*k0 + 16h + [0, 16) -> elements 0..7, + 32 -> elements 8..15
// rows_base once per tile (or per ring stage; tile 16-B aligned), then load_rows(base, r0, k0) per
// fragment with r0 / k0 compile-time (k0 % 8 == 0, diagnosed).
template <int PITCH> HKA_DEV lds_u8 *rows_base(uint32_t tile, int lane) {
    static_assert(PITCH % 16 == 0, "16-B aligned rows");
    return lds_ptr(tile + (uint32_t) lay::rows_lane(lane, PITCH));
}
template <int PITCH>
HKA_DEV opnd_t load_rows(const lds_u8 *base, int r0, int k0)
    HKA_MUL8(k0, "load_rows: k0 % 8 != 0 (16-B aligned rows chunk)") {
    const int off = lay::rows_imm(PITCH, r0, k0, 0);
    return cat(ds_load_b128(base, off), ds_load_b128(base, off + 32));
}

// ---- operand loader 2: "tr" (2 x ds_load_tr16_b128)
// ------------------------------------------------------ The tile's rows are the contraction index
// k: an A operand from A^T [K][M] row-major (V^T from V, dO^T from dO, Q^T from Q, dS from dS^T),
// or a B operand from B [K][N] row-major (K in dQ = dS K).
//   lane (r, h): element e = T[k0 + op_k(l, e)][r0 + r]
//   lane l addresses row k0 + 8h + l % 8 (+ 16 for elements 8..15), bytes 2*r0 + 16*((l / 8) % 2)
// tr_base once per tile (16-B aligned), load_tr(base, k0, r0) per fragment with k0 / r0
// compile-time (r0 % 8 == 0, diagnosed).
template <int PITCH> HKA_DEV lds_u8 *tr_base(uint32_t tile, int lane) {
    static_assert(PITCH % 16 == 0, "16-B aligned rows");
    return lds_ptr(tile + (uint32_t) lay::tr_lane(lane, PITCH));
}
template <int PITCH>
HKA_DEV opnd_t load_tr(const lds_u8 *base, int k0, int r0)
    HKA_MUL8(r0, "load_tr: r0 % 8 != 0 (16-B aligned tr16 chunk)") {
    return cat(ds_load_tr16_b128(base, lay::tr_imm(PITCH, k0, r0, 0)),
               ds_load_tr16_b128(base, lay::tr_imm(PITCH, k0, r0, 1)));
}

// ---- accumulator store: C^T as bf16 rows (1 x v_cvt_pk_bf16_f32 x4 + ds_store_b128)
// ---------------------- lane (r, h) writes bf16(c[0..7]) to row n0 + r, columns m0 + 8h + [0, 8):
// the tile receives T[n0 + n][m0 + m] = bf16(C[m][n]) (RNE). base = rows_base<PITCH>(tile, lane)
// (same lane offset); m0 % 8 == 0 (diagnosed). Use: the bwd dS^T exchange (C of dS = Q
// K^T-oriented, lane = kv -> T = dS^T [kv][q], read back as the dQ A operand with load_tr), and the
// fwd O epilogue (O^T accumulator -> O rows).
template <int PITCH>
HKA_DEV void store_ct(lds_u8 *base, int n0, int m0, acc_t c)
    HKA_MUL8(m0, "store_ct: m0 % 8 != 0 (16-B aligned C^T chunk)") {
    ds_store_b128(base, lay::ct_imm(PITCH, n0, m0), cvt_pk_bf16(c));
}

} // namespace hk_attn
