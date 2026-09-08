/***************************************************************************************************
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

#pragma once

#include "primus_turbo/common.h"
#include <hip/hip_runtime.h>
#include <optional>

namespace primus_turbo {

template <typename T>
void compute_scale_from_amax(const T *amax, const T q_max, T *scale, T *scale_inv, const int64_t n,
                             hipStream_t stream, const float eps = 1e-12);

// Whole-tensor abs-amax -> scale / scale_inv for tensorwise quant: two launches
// (nontemporal stream + finalise) vs the generic reduce_row chain's four, output
// bit-identical. `workspace` needs `tensorwise_amax_workspace_elems()` floats.
int64_t tensorwise_amax_workspace_elems();

template <typename FType>
void quantize_tensorwise_amax_scale_impl(const FType *x, const int64_t n, const float q_max,
                                         float *amax, float *scale, float *scale_inv,
                                         float *workspace, hipStream_t stream);

// *************** Quantize ***************
template <typename FType, typename QType, typename ComputeType = float>
void quantize_tensorwise_impl(const FType *x, const float *scale, QType *y, const int64_t n,
                              hipStream_t stream);

// Tensorwise FP8 quant that K-pads the innermost dim K -> Kp=ceil128(K), columns
// [K, Kp) zeroed; real columns [0, K) byte-identical to quantize_tensorwise_impl.
// Optionally also pads the penultimate dim N -> np_pen (n_pen real rows).
template <typename FType, typename QType, typename ComputeType = float>
void quantize_tensorwise_pad_impl(const FType *x, const float *scale, QType *y, const int64_t rows,
                                  const int64_t K, const int64_t Kp, hipStream_t stream,
                                  const int64_t n_pen = 0, const int64_t np_pen = 0);

// Segment-padded group offsets (each segment rounded up to block_size), on-device.
template <typename IndexType>
void compute_padded_group_offs(const IndexType *group_lens_ptr, IndexType *padded_lens_ptr,
                               IndexType *padded_offs_ptr, const int64_t group_num,
                               const IndexType block_size, hipStream_t stream);

// Fused single-pass row + segment-padded col blockwise FP8 quant (grouped fwd/bwd).
template <typename FType, typename QType>
void quantize_blockwise_segment_m_row_col_impl(const FType *x, QType *y_row, QType *y_col_padded,
                                               float *scales_row, float *scales_col_padded,
                                               const int64_t *group_offs,
                                               const int64_t *padded_group_offs, const int64_t M_in,
                                               const int64_t N, const int64_t M_padded_max,
                                               const int num_groups, const float fp8_max,
                                               hipStream_t stream);

// Blockwise FP8 weight quant: [B, M, N] (or [M, N]), one scalar scale per [128,128] tile.
template <typename FType, typename QType>
void quantize_blockwise_for_weight_impl(const FType *w, QType *w_fp8, float *w_scales_inv,
                                        const int64_t B, const int64_t M, const int64_t N,
                                        const float fp8_max, hipStream_t stream);

template <typename FType, typename QType, typename ComputeType = float,
          bool PreComputeScale = false>
void quantize_rowwise_row_major_impl(const FType *x, float *scale, float *scale_inv, QType *y,
                                     const int64_t outer_len, const int64_t inner_len,
                                     hipStream_t stream);

template <typename FType, typename QType, typename ComputeType = float>
void quantize_rowwise_col_major_impl(const FType *x, float *scale, float *scale_inv, QType *y,
                                     const int64_t batch, const int64_t m, const int64_t n,
                                     hipStream_t stream);

namespace detail {

enum class QuantizeMode { ROWWISE, COLWISE };

// MX format: each scale covers 32 elements
constexpr int MXFP4_BLOCK_SIZE = 32;
constexpr int MXFP8_BLOCK_SIZE = 32;

// Padding alignment expected for the public ``padding_align_size`` op argument.
// Must stay in sync with ``MXFP4_PADDING_ALIGN_SIZE`` / ``MXFP8_PADDING_ALIGN_SIZE``
// declared in ``primus_turbo/pytorch/core/low_precision.py``.
constexpr int MXFP4_K_DIM_PADDING_ALIGN_SIZE = 128;
constexpr int MXFP8_K_DIM_PADDING_ALIGN_SIZE = 128;

constexpr int MXFP8_GROUP_M_PADDING_ALIGN_SIZE = 32;

struct ScalingRecipe {
    bool use_2d_block = false;
    bool use_sr       = false;
    bool use_rht      = false;

    bool shuffle_scale = false;
    bool shuffle_out   = false;
};

constexpr int FP32_MANTISSA_BITS     = 23;
constexpr int FP32_EXPONENT_BITS     = 8;
constexpr int FP32_EXPONENT_EXP_BIAS = 127;

constexpr int FP4_MANTISSA_BITS   = 1;
constexpr int FP4_EXPONENT_BITS   = 2;
constexpr int FP4_TARGET_MAX_POW2 = 2;

constexpr int   FP8E5M2_MANTISSA_BITS   = 2;
constexpr int   FP8E5M2_EXPONENT_BITS   = 5;
constexpr float FP8E5M2_MAX             = 57344.0;
constexpr int   FP8E5M2_TARGET_MAX_POW2 = 15;

constexpr int FP8E4M3_MANTISSA_BITS = 3;
constexpr int FP8E4M3_EXPONENT_BITS = 4;
// NOTE: The max value of fp8 e4m3 ocp is 448.
constexpr float FP8E4M3_MAX             = 448.0;
constexpr int   FP8E4M3_TARGET_MAX_POW2 = 8;
// NOTE: The max value of fp8 e4m3 fnuz is 240.
constexpr float FP8E4M3_FNUZ_MAX             = 240.0;
constexpr int   FP8E4M3_FNUZ_TARGET_MAX_POW2 = 7;

constexpr int E8M0_EXPONENT_BIAS = 127;

} // namespace detail

// ---------------------------------------------------------------------------
// MXFP6 (E2M3) quantize + pack into AITER's mxfp6_c0c1_256_padk2 blob layout.
//
// Unlike the other formats there is no strided output tensor here: the A6W6 assembly
// consumes an opaque re-tiled blob whose logical shape is the caller's to remember.
// The mandatory 32-point Hadamard along the contraction axis is folded in.
// ---------------------------------------------------------------------------

// Trailing K-tiles present in every packed blob. Their contents are never read, but the
// space is mandatory: the assembly derives its row-tile stride from k/128 + 2.
constexpr int MXFP6_GUARD_K_TILES = 2;

enum class MXFP6Direction {
    Row,  // contract along the last axis
    Col,  // contract along the first axis, i.e. pack the rows of x.T
    Dual, // both, from a single read of the input
};

// Writes (row_packed, row_scale) and/or (col_packed, col_scale) for a [M, N] input,
// according to `direction`. Pointers for a direction that is not requested are unused
// and may be null. Sizes must come from mxfp6_pack_sizes on the Python side.
template <typename DType>
void quantize_mxfp6_impl(const DType *input, uint8_t *row_packed, uint8_t *row_scale,
                         uint8_t *col_packed, uint8_t *col_scale, const int M, const int N,
                         const MXFP6Direction direction, hipStream_t stream);

// Elementwise epilogue folded into the packer's LDS staging read, so the tensor it
// applies to never reaches HBM. The result is rounded back to DType before staging, which
// makes the packed blob bit-identical to packing the epilogue's materialised output.
enum class MXFP6Prologue {
    Identity,         // stage the input unchanged
    BiasGelu,         // gelu_tanh(input + bias)
    BiasGeluBackward, // d/dx gelu_tanh(input + bias) * aux, where aux is the incoming grad
    // The QK-norm + RoPE backward, for the QKV projection's dgrad. Unlike the three above
    // this one does not act elementwise on `input`: it *computes* the tensor to be packed
    // out of nine operands, carried in MXFP6QkNormRopeArgs. `input` is the forward's
    // mixed_qkv, which the prologue needs as the norm's saved input and which happens to
    // share the packer's own [M, N] indexing.
    //
    // It requires TILE_N == head_dim, because the norm's row reduction runs over head_dim
    // and a block-local reduction is the only kind this kernel can do without cross-lane
    // traffic. That is a static_assert, not a hope.
    QkNormRopeBackward,
};

// Operands for MXFP6Prologue::QkNormRopeBackward.
//
// Passed as a struct rather than as arguments because there are nine of them and the
// kernel already takes eleven. Every pointer is required; there is no null-means-skip
// convention here, unlike `bias` above.
//
// Shapes, in the packer's own 2D view of a [S, B, H, 3D] tensor -- M = S*B rows and
// N = H*3D columns, so column n selects head h = n / (3D) and element d = n % D, with
// (n % 3D) / D choosing q, k or v:
//
//   dq, dk, dv   [M, H*D]   the FMHA gradients, one per slice, contiguous per row
//   cos, sin     [M, D]     Flux's tables are per (position, batch), which is the packer's
//                           row exactly -- see the note in RESULTS_prologue_reduce.md on
//                           why this rules out any reuse
//   wq, wk       [D]        the norm weights
//   rstd_q,k     [M*H]      one per normalised vector, indexed m * H + h
//   dw_q, dw_k   [rows, H, D]  fp32 partials, rows = mxfp6_col_sum_rows(M). Summed over
//                           both leading axes by the caller. Two axes rather than one
//                           because dw is a reduction over rows *and* heads, and a block
//                           owns one head.
template <typename DType> struct MXFP6QkNormRopeArgs {
    const DType *dq, *dk, *dv;
    const DType *cos, *sin;
    const DType *wq, *wk;
    const float *rstd_q, *rstd_k;
    float       *dw_q, *dw_k;
    int32_t      num_heads;
    int32_t      head_dim;
};

// M-rows per row of the bias-gradient partial buffer, i.e. the packer's M-tile height.
// Declared here rather than left as a kernel-private tuning constant because the host and
// Python both have to size that buffer; the kernel static_asserts they agree.
// Overridable only so a tile sweep can be driven from the command line; 64 is the
// shipped value. Changing it in a real build also changes the partial buffer's geometry,
// which Python sizes from mxfp6_col_sum_rows(), so the two must move together.
#ifndef MXFP6_TILE_M
#define MXFP6_COL_SUM_TILE_M_DEFAULT 64
#else
#define MXFP6_COL_SUM_TILE_M_DEFAULT MXFP6_TILE_M
#endif

constexpr int MXFP6_COL_SUM_TILE_M = MXFP6_COL_SUM_TILE_M_DEFAULT;

// Rows of the partial buffer for an [M, N] input. One per M-tile of the launch grid, which
// covers M padded to whole 256-row tiles.
constexpr int mxfp6_col_sum_rows(const int M) {
    const int m_padded = ((M + 255) / 256) * 256;
    return (m_padded + MXFP6_COL_SUM_TILE_M - 1) / MXFP6_COL_SUM_TILE_M;
}

// As quantize_mxfp6_impl with direction Dual, but applies `prologue` to the input while
// staging it. `bias` is broadcast along N and may be null (no bias term). `aux` is only
// read by BiasGeluBackward and may be null otherwise. Both must have the input's dtype,
// and `aux` its shape. Only Dual is provided: the fused path always needs both directions,
// and instantiating the prologue against the direction flags too would quadruple the
// template expansion for no caller.
//
// `col_sum` is optional and may be null. When given it receives per-column sums of the
// staged (post-prologue, pre-Hadamard) values as a
// [mxfp6_col_sum_rows(M), N] fp32 buffer, one row per M-tile, to be finished with a sum
// over that axis. This is how a bias gradient survives the fusion: the tensor it would
// otherwise be reduced from never reaches HBM, and the tile is already in LDS here, so the
// column sums cost a second pass over shared memory rather than a pass over HBM.
template <typename DType>
void quantize_mxfp6_fused_impl(const DType *input, const DType *aux, const DType *bias,
                               uint8_t *row_packed, uint8_t *row_scale, uint8_t *col_packed,
                               uint8_t *col_scale, float *col_sum, const int M, const int N,
                               const MXFP6Prologue prologue, hipStream_t stream);

// As above for MXFP6Prologue::QkNormRopeBackward, which needs its own entry point because
// its operands do not fit the (input, aux, bias) shape and because it runs at a different
// tile width: head_dim, not the shipped TILE_N. `input` is mixed_qkv.
//
// Kept separate rather than folded into quantize_mxfp6_fused_impl with a defaulted struct
// so that the two tile widths stay two instantiations and no existing caller's generated
// code moves. The wider width also measured faster than the shipped one on these shapes, so
// this is not a concession -- see packer/RESULTS_tile_ab.md.
// Constraints, all checked at the entry point rather than assumed:
//   * head_dim must be 128. The tile width *is* head_dim, because the norm reduces over
//     head_dim and the kernel only reduces inside a block, so each supported head_dim is a
//     separate instantiation and 128 is the only one built.
//   * num_heads must be even. The kernel reads its (head, slice) off blockIdx.x, which
//     requires the grid to be exactly num_heads * 3 tiles wide; the grid rounds N up to 256,
//     so N = num_heads * 3 * 128 has to be a multiple of 256.
//   * the rotary embedding must be interleaved, not half-split. The backward pairs index 2i
//     with 2i+1, which keeps a pair inside one thread's vector; the half-split convention
//     pairs d with d + head_dim/2, in a different thread. This one the caller must uphold --
//     it is not visible in any argument.
template <typename DType>
void quantize_mxfp6_qk_norm_rope_bwd_impl(const DType *input,
                                          const MXFP6QkNormRopeArgs<DType> &args,
                                          uint8_t *row_packed, uint8_t *row_scale,
                                          uint8_t *col_packed, uint8_t *col_scale, float *col_sum,
                                          const int M, const int N, hipStream_t stream);

template <typename DType>
void quantize_mxfp4_dual_impl(const DType *input, dtype::float4x2_e2m1 *rowwise_output,
                              uint8_t *rowwise_scale, dtype::float4x2_e2m1 *colwise_output,
                              uint8_t *colwise_scale, int G, int M, int N, int M_pad, int N_pad,
                              int rowwise_scale_stride, int colwise_scale_stride,
                              int rowwise_scale_N, int rowwise_scale_M_pad, int rowwise_scale_N_pad,
                              int colwise_scale_M, int colwise_scale_N, int colwise_scale_M_pad,
                              int colwise_scale_N_pad, detail::ScalingRecipe rowwise_recipe,
                              detail::ScalingRecipe colwise_recipe, hipStream_t stream);

template <typename DType>
void quantize_mxfp4_impl(const DType *input, dtype::float4x2_e2m1 *output, uint8_t *scale,
                         detail::QuantizeMode mode, int G, int M, int N, int M_pad, int N_pad,
                         int scale_stride, int scale_N, int scale_M_pad, int scale_N_pad,
                         detail::ScalingRecipe recipe, hipStream_t stream);

template <typename IType, typename OType>
void quantize_mxfp8_dual_impl(const IType *input, OType *rowwise_output, uint8_t *rowwise_scale,
                              OType *colwise_output, uint8_t *colwise_scale, int G, int M, int N,
                              int M_pad, int N_pad, int rowwise_scale_stride,
                              int colwise_scale_stride, int rowwise_scale_N,
                              int rowwise_scale_M_pad, int rowwise_scale_N_pad, int colwise_scale_M,
                              int colwise_scale_N, int colwise_scale_M_pad, int colwise_scale_N_pad,
                              detail::ScalingRecipe rowwise_recipe,
                              detail::ScalingRecipe colwise_recipe, hipStream_t stream);

template <typename IType, typename OType>
void quantize_mxfp8_impl(const IType *input, OType *output, uint8_t *scale,
                         detail::QuantizeMode mode, int G, int M, int N, int M_pad, int N_pad,
                         int scale_stride, int scale_N, int scale_M_pad, int scale_N_pad,
                         detail::ScalingRecipe recipe, hipStream_t stream);

template <typename IType, typename OType>
void grouped_quantize_mxfp8_dual_impl(
    const IType *input, OType *rowwise_output, uint8_t *rowwise_scale, OType *colwise_output,
    uint8_t *colwise_scale, const int64_t *group_offs, const int64_t *group_offs_padded_colwise,
    const int64_t *group_offs_padded_rowwise, int G, int total_M, int N, int N_pad,
    int rowwise_scale_stride, int colwise_scale_stride, int rowwise_scale_N,
    int rowwise_scale_M_pad, int rowwise_scale_N_pad, int colwise_scale_M, int colwise_scale_N,
    int colwise_scale_M_pad, int colwise_scale_N_pad, detail::ScalingRecipe rowwise_recipe,
    detail::ScalingRecipe colwise_recipe, hipStream_t stream);

template <typename IType, typename OType>
void grouped_quantize_mxfp8_impl(const IType *input, OType *output, uint8_t *scale,
                                 const int64_t *group_offs, const int64_t *group_offs_padded,
                                 detail::QuantizeMode mode, int G, int total_M, int N, int N_pad,
                                 int scale_stride, int scale_N, int scale_M_pad, int scale_N_pad,
                                 detail::ScalingRecipe recipe, hipStream_t stream);

template <typename DType>
void grouped_quantize_mxfp4_dual_impl(const DType *input, dtype::float4x2_e2m1 *rowwise_output,
                                      uint8_t *rowwise_scale, dtype::float4x2_e2m1 *colwise_output,
                                      uint8_t *colwise_scale, const int64_t *group_offs,
                                      const int64_t *group_offs_padded_colwise, int G, int total_M,
                                      int N, int M_pad_col, int N_pad, int rowwise_scale_stride,
                                      int colwise_scale_stride, int rowwise_scale_N,
                                      int colwise_scale_N, detail::ScalingRecipe rowwise_recipe,
                                      detail::ScalingRecipe colwise_recipe, hipStream_t stream);

// Single-direction (rowwise OR colwise) grouped MXFP4 quant.
template <typename DType>
void grouped_quantize_mxfp4_impl(const DType *input, dtype::float4x2_e2m1 *output, uint8_t *scale,
                                 const int64_t       *group_offs,
                                 const int64_t       *group_offs_padded_colwise,
                                 detail::QuantizeMode mode, int G, int total_M, int N,
                                 int M_pad_col, int N_pad, int scale_stride, int scale_N,
                                 detail::ScalingRecipe recipe, hipStream_t stream);

// *************** Grouped Padded Layout ***************
//
// Computes per-group padded lengths/offsets (rounded up to ``align``) on
// the GPU; results live in device memory so no D2H sync is required.
void compute_padded_layout_gpu(const int64_t *group_lens, int64_t *group_lens_padded,
                               int64_t *group_offs_padded, int G, int64_t align,
                               hipStream_t stream);

// *************** DeQuantize ***************
template <typename FType, typename QType, typename ComputeType = float>
void dequantize_tensorwise_impl(const QType *x, const float *scale_inv, FType *y, const int64_t n,
                                hipStream_t stream);

// Rowwise dequantize when the per-row dim is the innermost (last) dim.
// scale_inv has shape [outer_len] (one scalar per row).
template <typename FType, typename QType, typename ComputeType = float>
void dequantize_rowwise_row_major_impl(const QType *x, const float *scale_inv, FType *y,
                                       const int64_t outer_len, const int64_t inner_len,
                                       hipStream_t stream);

// Rowwise dequantize when the per-row dim is a middle dim.
// Input is viewed as [B, M, N], scale_inv has shape [B, N] (broadcast across M).
template <typename FType, typename QType, typename ComputeType = float>
void dequantize_rowwise_col_major_impl(const QType *x, const float *scale_inv, FType *y,
                                       const int64_t batch, const int64_t m, const int64_t n,
                                       hipStream_t stream);

// *************** MX Block-scaled DeQuantize ***************
template <typename OType, typename QType>
void dequantize_mxfp8_impl(const QType *x, OType *y, const int64_t stride_x_row,
                           const int64_t stride_x_col, const int64_t stride_y_row,
                           const int64_t stride_y_col, const int n_rows, const int n_cols,
                           const uint8_t *scale_inv, const int64_t stride_scale_row,
                           const int64_t stride_scale_col, const int scale_n_rows,
                           const int scale_n_cols, const int block_size, const bool use_rowwise,
                           hipStream_t stream);

template <typename OType, typename QType>
void grouped_dequantize_mxfp8_impl(const QType *x, OType *y, const int64_t stride_x_row,
                                   const int64_t stride_y_row, const int total_M, const int n_rows,
                                   const int n_cols, const uint8_t *scale_inv,
                                   const int64_t stride_scale_row, const int64_t stride_scale_col,
                                   const int scale_n_rows, const int scale_n_cols,
                                   const int64_t *group_offs, const int64_t *group_offs_padded,
                                   int G, int block_size, bool use_rowwise, hipStream_t stream);

template <typename OType>
void dequantize_mxfp4_impl(const uint8_t *x, OType *y, const int64_t stride_x_row,
                           const int64_t stride_x_col, const int64_t stride_y_row,
                           const int64_t stride_y_col, const int n_rows, const int n_cols,
                           const uint8_t *scale_inv, const int64_t stride_scale_row,
                           const int64_t stride_scale_col, const int scale_n_rows,
                           const int scale_n_cols, const int block_size, const bool use_rowwise,
                           hipStream_t stream);

} // namespace primus_turbo
