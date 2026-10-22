/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// hk_attn/layout.h -- every lane map of the gfx1250 attention primitives, as constexpr functions of
// scalars (host-compilable: no HIP header, no builtin). The device wrappers (lds.h, frag.h, gmem.h)
// compute their addresses with exactly these; hk/lab/test_prims.py mirrors them in Python and checks
// them bit-exactly on the GPU.
//
// Names: wave32, lane l in [0, 32), r = l % 16 (the operand / accumulator "lane row"), h = l / 16
// (the lane half). A 16-bit element e of an operand register set is in [0, 16); an fp32 element i of
// an accumulator is in [0, 8).
//
// v_wmma_f32_16x16x32_bf16 D[16x16] = A[16x32] . B[32x16] + C (all lanes, EXEC all ones):
//   A operand (8 VGPRs, bf16x16):  lane (r, h) element e holds A[m = r][k = op_k(l, e)]
//   B operand (8 VGPRs, bf16x16):  lane (r, h) element e holds B[k = op_k(l, e)][n = r]
//   C / D     (8 VGPRs, f32x8):    lane (r, h) element i holds C[m = c_m(l, i)][n = r]
//   op_k(l, e) = (e / 8) * 16 + h * 8 + e % 8:
//     e 0..7  (VGPRs 0..3) -> k = 8h + [0, 8),   e 8..15 (VGPRs 4..7) -> k = 16 + 8h + [0, 8)
//   c_m(l, i) = 8h + i: lanes 0..15 hold rows 0..7, lanes 16..31 rows 8..15, VGPR i = row i (+8h).
// So A and B have the same register layout with the lane indexing the non-contracted dimension, and
// two C tiles stacked along M (rows 0..15 = tile 0, rows 16..31 = tile 1) ARE a B operand (see
// pack_k32 below): no data movement, one v_cvt_pk_bf16_f32 per element pair.
#pragma once
#include <stdint.h>
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#define HKA_HD static inline constexpr __host__ __device__

namespace hk_attn {
namespace lay {

inline constexpr int WAVE = 32;

// ---- operand / accumulator element maps -------------------------------------------------------------
HKA_HD int lane_r(int lane) {
    return lane % 16;
}
HKA_HD int lane_h(int lane) {
    return lane / 16;
}
// k of element e (0..15) of an A or B operand in lane `lane`
HKA_HD int op_k(int lane, int e) {
    return (e / 8) * 16 + (lane / 16) * 8 + e % 8;
}
// row m of accumulator element i (0..7) in lane `lane`; its column is lane_r(lane)
HKA_HD int c_m(int lane, int i) {
    return (lane / 16) * 8 + i;
}
HKA_HD int c_n(int lane) {
    return lane % 16;
}

// ---- C -> operand pack (frag.h pack_k32) ---------------------------------------------------------------
// Two accumulators c0 (rows 0..15) and c1 (rows 16..31) of a 32x16 matrix X (rows = the next GEMM's
// contraction index k, columns = lane_r) give the operand whose element e of lane l is
//   e < 8: c0[e] = X[8h + e][r],   e >= 8: c1[e - 8] = X[16 + 8h + e - 8][r]   == X[op_k(l, e)][r].
// As a B operand that is B = X ([k][n]); as an A operand it is A = X^T ([m = r][k]).
HKA_HD int pack_src_tile(int e) {
    return e / 8;
} // which accumulator element e comes from
HKA_HD int pack_src_elem(int e) {
    return e % 8;
} // and which of its fp32 elements

// ---- LDS operand loaders (lds.h) -----------------------------------------------------------------------
// "rows" loader, 2 x ds_load_b128: the LDS tile is row-major bf16 with row pitch `pitch` bytes and its
// rows are the operand's lane dimension (M of an A operand, N of a B operand stored as B^T).
// lane (r, h) reads row r0 + r: bytes 2*k0 + 16h + [0, 16) (elements 0..7) and + 32 (elements 8..15):
//   element e = T[r0 + r][k0 + op_k(l, e)]
HKA_HD int rows_lane(int lane, int pitch) {
    return (lane % 16) * pitch + (lane / 16) * 16;
}
HKA_HD int rows_imm(int pitch, int r0, int k0, int half) {
    return r0 * pitch + 2 * k0 + 32 * half;
}
// "tr" loader, 2 x ds_load_tr16_b128: the LDS tile's rows are the contraction index k (A stored as A^T
// = [K][M], or B stored as [K][N]). Lane l ADDRESSES row k0 + 8h + l % 8 (+16 for the second load),
// bytes 2*r0 + 16*((l / 8) % 2) + [0, 16). ds_load_tr16_b128 transposes each aligned group of 8 lanes:
// for g = l / 8 and c = l % 8, result element j of lane 8g + c = element c of the 16 bytes addressed
// by lane 8g + j. Net result:
//   element e = T[k0 + op_k(l, e)][r0 + r]
HKA_HD int tr_lane(int lane, int pitch) {
    return ((lane / 16) * 8 + lane % 8) * pitch + ((lane / 8) % 2) * 16;
}
HKA_HD int tr_imm(int pitch, int k0, int r0, int half) {
    return (k0 + 16 * half) * pitch + 2 * r0;
}
// C^T store (lds.h store_ct / gmem.h buf_store_ct): one b128 per lane; lane (r, h) writes the 8 bf16
// of its accumulator to row n0 + r, columns m0 + 8h + [0, 8) of a row-major tile of pitch `pitch`,
// i.e. the tile receives C^T: T[n0 + n][m0 + m] = bf16(C[m][n]). Same lane offset as rows_lane.
HKA_HD int ct_lane(int lane, int pitch) {
    return rows_lane(lane, pitch);
}
HKA_HD int ct_imm(int pitch, int n0, int m0) {
    return n0 * pitch + 2 * m0;
}

// ---- causal masks on an accumulator (frag.h) ----------------------------------------------------------
// tile-local row m = 8h + i, column n = r; `off` is wave-uniform.
// mask_col_gt_row: masked iff n > m + off  <=>  i < n - 8h - off   (threshold thr_cgr, test i < thr)
// mask_row_gt_col: masked iff m > n + off  <=>  i > n - 8h + off   (threshold thr_rgc, test i > thr)
// Bottom-right causal (query q sees kv <= q + cshift, cshift = Skv - Sq):
//   S   = Q K^T (rows q from m0, cols kv from n0): mask_col_gt_row with off = m0 + cshift - n0
//   S^T = K Q^T (rows kv from m0, cols q from n0): mask_row_gt_col with off = n0 + cshift - m0
HKA_HD int thr_cgr(int lane, int off) {
    return lane % 16 - 8 * (lane / 16) - off;
}
HKA_HD int thr_rgc(int lane, int off) {
    return lane % 16 - 8 * (lane / 16) + off;
}

// ---- row vectors (gmem.h) ------------------------------------------------------------------------------
// load_rows8: lane (r, h) loads the 8 fp32 of rows 8h + [0, 8) (accumulator row order) from byte
// voff + 32h (two b128). store_cols16: lanes 0..15 store the fp32 of column r at byte voff + 4r.
HKA_HD int rows8_lane(int lane) {
    return (lane / 16) * 32;
}
HKA_HD int cols16_lane(int lane) {
    return (lane % 16) * 4;
}

// ---- dQ-style fp32 atomics in C-fragment order (gmem.h atomic_add_c) ------------------------------------
// element i of lane l goes to byte voff + i * slot_b with voff = 4l + (per-wave line): every atomic
// instruction covers one aligned, contiguous 128-B line (32 lanes x 4 B).
HKA_HD int cfrag_lane(int lane) {
    return lane * 4;
}

// ---- TDM descriptor group 1 word 0 (tdm.h) ---------------------------------------------------------------
// bits 17:16 data_size = log2(bytes per element), 20 pad_enable, 24:22 pad_interval = log2(interval
// dwords) - 1, 31:25 pad_amount = pad dwords - 1. The interval is the tile row (tile_cols * elem_b
// bytes, a power of two, 8..1024 B) and the pad is pitch - row bytes (a multiple of 4 B, <= 512 B).
HKA_HD int ilog2(int x) {
    return x <= 1 ? 0 : 1 + ilog2(x / 2);
}
HKA_HD bool tdm_tile_ok(int tile_cols, int pitch_b, int elem_b) {
    const int row_b = tile_cols * elem_b, pad_b = pitch_b - row_b;
    const bool pow2 = row_b >= 8 && row_b <= 1024 && (row_b & (row_b - 1)) == 0;
    const bool esz  = elem_b == 1 || elem_b == 2 || elem_b == 4 || elem_b == 8;
    return esz && pow2 && pad_b >= 0 && pad_b % 4 == 0 && pad_b <= 512;
}
HKA_HD uint32_t tdm_w0(int tile_cols, int pitch_b, int elem_b) {
    const int      row_b = tile_cols * elem_b, pad_b = pitch_b - row_b;
    const uint32_t ds    = (uint32_t) ilog2(elem_b) << 16;
    return pad_b == 0 ? ds
                      : ds | (1u << 20) | ((uint32_t) (ilog2(row_b / 4) - 1) << 22) |
                            ((uint32_t) (pad_b / 4 - 1) << 25);
}

} // namespace lay
} // namespace hk_attn
