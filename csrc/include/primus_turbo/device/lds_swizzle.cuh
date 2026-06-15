// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// LDS swizzle helpers for CDNA4 (gfx950 / MI355X).
//
// ───────────────────────────────────────────────────────────────────
//  Why this header exists
// ───────────────────────────────────────────────────────────────────
//
// CDNA4 LDS has **64 banks** of 4 B each (twice CDNA3's 32 banks).  A
// ``ds_read_b128`` issued by a wave64 reads 16 B per lane = 4 banks per
// lane, total 256 B = 64 banks - exactly one full bank cycle when the
// access pattern is conflict-free.  Any layout that DG/CUTLASS designed
// for 32-bank SM100 LDS will alias half its threads onto the same bank
// pair on AMD and degrade to 16-32% utilization.
//
// Concretely, for a row-major A tile staged as
//
//     SMEM_A[stage][m][k]      // FP8, 1 B per element
//
// the natural ``ds_read_b128`` lane mapping ``m = lane / 8, k = lane % 8
// * 16`` issues 8 reads per row across lanes; rows that align to 128 B
// will collide on lanes that share the same ``m`` bit.  The mitigation
// is the same as on NVIDIA's swizzled LDS - XOR the column index with a
// row-derived nibble - but the nibble width must double to cover the
// extra bank bits.
//
// ───────────────────────────────────────────────────────────────────
//  Design (scaffold only - concrete tile body is co-designed with the
//  A/B loader that has not landed yet; see MegaKernel/TODO.md Section
//  B item 5 and Section A item 7)
// ───────────────────────────────────────────────────────────────────
//
// We expose three primitives:
//
//   1.  ``swizzle_offset_128b_64bank<RowBytes>(m, k_byte)`` - returns
//       the in-LDS byte offset for a (m, k_byte) tile element under a
//       128 B swizzle tuned for 64 banks.  Encodes the standard
//       ``k_byte ^= (m & 7u) * 16`` XOR but pads to a 16 B chunk so
//       ``ds_read_b128`` lands on disjoint bank groups for every
//       (m, k) pair in a wave64.
//
//   2.  ``a_tile_smem_addr<Stage, LoadBlockM, BlockK>(stage_buf, m,
//       k_byte)`` - per-stage A tile addressing wrapper around (1).
//       Takes the byte offset of the swizzle and folds in the per-stage
//       base offset.
//
//   3.  ``b_tile_smem_addr<Stage, LoadBlockN, BlockK>(stage_buf, n,
//       k_byte)`` - same for B.  B is FP4 unpacked-in-LDS so the
//       element stride is identical to A (1 B / element); only the row
//       count differs.
//
// All three are *constexpr* index helpers - they perform no LDS access
// themselves.  The loader pairs them with ``buffer_load_lds`` to write
// remote A/B tiles into the swizzled slots, and the MMA-issue role
// pairs them with ``ds_read_pinned`` (from ``device/memory.cuh``) to
// pull MFMA operands back into VGPRs without conflict.
//
// The current bodies below are the *scaffold*: the XOR transform is the
// correct shape but the (RowBytes, BankWidth) tuple is hard-coded for
// the BLOCK_K=128, FP8/FP4u staging used by mega-MoE.  Generalizing to
// arbitrary tile shapes is deferred until the loader body lands and we
// can measure ``ds_bank_conflict`` counters with rocprofv3.

#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

namespace primus_turbo::device {

// ════════════════════════════════════════════════════════════════
//  64-bank swizzle index transform
// ════════════════════════════════════════════════════════════════

// Per-row XOR shift expressed as a byte offset.  For BLOCK_K=128 with
// 16 B (uint4) ``ds_read_b128`` chunks the natural lane layout is
//     lane -> (m = lane >> 3, k_chunk = lane & 7)
// On a 64-bank LDS this lane layout already hits 8 distinct bank groups
// per row, but two rows of the same ``m & 7`` collide on the same group
// pair when ``ds_read_b128`` ships 4 banks per lane.  Rotating ``k_chunk``
// by ``(m & 7) << 1`` (4 chunks of 16 B = 64 B = one rotation cycle) keeps
// every wave64 access on a unique 16 B chunk per bank group.
//
// This is the 64-bank analogue of CUTLASS's ``Sw<3,4,3>`` swizzle for
// SM90/SM100 LDS: same shape, doubled row stride to match the doubled
// bank count.
template <uint32_t RowBytes>
__host__ __device__ __forceinline__ constexpr uint32_t swizzle_offset_128b_64bank(uint32_t m,
                                                                                  uint32_t k_byte) {
    // The swizzle MUST be invariant inside one 16 B ``ds_read_b128``
    // chunk - otherwise the four banks read by a single lane would no
    // longer be contiguous.  Quantize ``k_byte`` to 16 B, XOR, then add
    // the intra-chunk offset back unmodified.
    constexpr uint32_t kChunkBytes  = 16u; // ds_read_b128 transaction width
    constexpr uint32_t kRotateMaskM = 7u;  // (m & 7) selects rotation
    static_assert(RowBytes % kChunkBytes == 0u, "RowBytes must be 16 B aligned");

    const uint32_t row_base      = m * RowBytes;
    const uint32_t k_chunk       = k_byte / kChunkBytes;
    const uint32_t k_intra       = k_byte % kChunkBytes;
    const uint32_t rotated_chunk = k_chunk ^ (m & kRotateMaskM);
    return row_base + rotated_chunk * kChunkBytes + k_intra;
}

// ════════════════════════════════════════════════════════════════
//  Per-tile addressing wrappers
// ════════════════════════════════════════════════════════════════

// A tile address - LOAD_BLOCK_M rows of BLOCK_K bytes each, per stage.
// ``stage_buf_lds_byte`` is the per-stage base offset within the LDS
// carve-out (already includes the stage rotation).
template <uint32_t LoadBlockM, uint32_t BlockK>
__host__ __device__ __forceinline__ constexpr uint32_t
a_tile_smem_byte_offset(uint32_t stage_buf_lds_byte, uint32_t m, uint32_t k_byte) {
    // The (m & 7) rotation has an 8-row period.  For LoadBlockM > 64
    // rows beyond row 63 alias the same XOR pattern as their
    // (row - 64) counterpart - harmless for the sequential-issue
    // loader pattern used by the compute role (each ds_read serves one
    // row at a time, so aliasing only matters within a single wave64
    // transaction).  Capped at 128 to match BLOCK_M / BLOCK_N today;
    // raising further requires a wider rotation mask to keep bank
    // conflicts off the wave64 issue.
    static_assert(LoadBlockM <= 128u,
                  "LoadBlockM > 128 needs a wider (m & N) rotation - see lds_swizzle.cuh");
    return stage_buf_lds_byte + swizzle_offset_128b_64bank<BlockK>(m, k_byte);
}

// B tile address - LOAD_BLOCK_N rows of BLOCK_K bytes each, per stage.
// B is FP4 stored unpacked (1 B / element) in LDS so its stride matches
// A.  Same swizzle applies.
template <uint32_t LoadBlockN, uint32_t BlockK>
__host__ __device__ __forceinline__ constexpr uint32_t
b_tile_smem_byte_offset(uint32_t stage_buf_lds_byte, uint32_t n, uint32_t k_byte) {
    static_assert(LoadBlockN <= 128u,
                  "LoadBlockN > 128 needs a wider (n & N) rotation - see lds_swizzle.cuh");
    return stage_buf_lds_byte + swizzle_offset_128b_64bank<BlockK>(n, k_byte);
}

// Row-major (un-swizzled) B tile address.  Used for the M3 FP4-B tile whose
// row is only BLOCK_K/2 = 64 bytes = 4 ds_read_b128 chunks.  The 64-bank
// swizzle's XOR rotation mask is fixed at 7 (assumes >=8 chunks / >=128 B
// rows); with only 4 chunks the rotated chunk index spills past the row
// (k_chunk ^ (m&7) can reach 7 > 3), corrupting cross-row LDS addressing and
// the R145 linear-write cancellation.  A plain row-major layout is correct
// (bank-conflict avoidance is a perf-only concern; the FP4 B row is half the
// width so conflict pressure is already lower).
template <uint32_t LoadBlockN, uint32_t BlockK>
__host__ __device__ __forceinline__ constexpr uint32_t
b_tile_smem_byte_offset_rowmajor(uint32_t stage_buf_lds_byte, uint32_t n, uint32_t k_byte) {
    static_assert(LoadBlockN <= 128u, "LoadBlockN > 128 unsupported");
    return stage_buf_lds_byte + n * BlockK + k_byte;
}

} // namespace primus_turbo::device
