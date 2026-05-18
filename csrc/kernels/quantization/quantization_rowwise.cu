// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// Rowwise FP8 quantize / dequantize (row-major & col-major layouts).
//
// The quant op (QuantOpBase) and the amax -> scale helpers live in
// primus_turbo/device/quant_utils.cuh so they can be shared with the
// tensorwise kernels.

#include "primus_turbo/common.h"
#include "primus_turbo/device/quant_utils.cuh"
#include "primus_turbo/device/reduce.cuh"
#include "primus_turbo/memory_pack.h"
#include "primus_turbo/quantization.h"

namespace primus_turbo {

using namespace primus_turbo::dtype;

// ---------------------------------------------------------------------------
// Pack-size helper (shared by row-major & col-major dispatchers)
// ---------------------------------------------------------------------------
template <typename T>
static int32_t get_quantize_rowwise_pack_size(const int32_t pack_size, const int64_t inner_len) {
    PRIMUS_TURBO_CHECK(pack_size == 8 || pack_size == 4 || pack_size == 2 || pack_size == 1);
    PRIMUS_TURBO_CHECK(inner_len > 0);

    int32_t u = 1;
    if (pack_size == 8) {
        u = valid_pack<T, 8>();
    } else if (pack_size == 4) {
        u = valid_pack<T, 4>();
    } else if (pack_size == 2) {
        u = valid_pack<T, 2>();
    } else {
        u = 1;
    }

    while (u > 1 && (inner_len % u) != 0) {
        u >>= 1;
    }
    return u;
}

// ===========================================================================
// Quantize Rowwise: Row-Major
// ===========================================================================
template <int BLOCK_SIZE, int UNROLL, bool PreComputeScale, typename FType, typename QType,
          typename ComputeType = float>
__launch_bounds__(BLOCK_SIZE) __global__
    void quantize_rowwise_row_major_two_scan_kernel(const FType *__restrict__ input_ptr,
                                                    float *__restrict__ scale_ptr,
                                                    float *__restrict__ scale_inv_ptr,
                                                    QType *__restrict__ output_ptr,
                                                    const int64_t inner_len) {
    const int64_t bid     = blockIdx.x;
    const int32_t warp_id = threadIdx.x / BLOCK_SIZE;
    const int32_t lane_id = threadIdx.x % BLOCK_SIZE;

    const ComputeType CLIP_MIN = static_cast<ComputeType>(std::numeric_limits<QType>::lowest());
    const ComputeType CLIP_MAX = static_cast<ComputeType>(std::numeric_limits<QType>::max());
    const ComputeType EPS      = 1e-12;

    const int32_t start_offset = warp_id * BLOCK_SIZE * UNROLL + lane_id * UNROLL;

    input_ptr += bid * inner_len;
    output_ptr += bid * inner_len;

    FType ld_regs[UNROLL];
#pragma unroll
    for (int32_t i = 0; i < UNROLL; ++i) {
        ld_regs[i] = static_cast<FType>(0.0f);
    }

    // scale & scale_inv
    ComputeType scale;
    ComputeType scale_inv;
    if (PreComputeScale == true) {
        scale     = static_cast<ComputeType>(scale_ptr[bid]);
        scale_inv = static_cast<ComputeType>(scale_inv_ptr[bid]);
    } else {
        // amax
        ComputeType amax_regs[UNROLL];
#pragma unroll
        for (int32_t i = 0; i < UNROLL; ++i) {
            amax_regs[i] = AbsMaxOp<ComputeType>::init();
        }

        for (int64_t offset = start_offset; offset < inner_len; offset += (BLOCK_SIZE * UNROLL)) {
            load_data<FType, UNROLL>(input_ptr + offset, ld_regs);
#pragma unroll
            for (int32_t i = 0; i < UNROLL; ++i) {
                amax_regs[i] =
                    AbsMaxOp<ComputeType>::op(amax_regs[i], static_cast<ComputeType>(ld_regs[i]));
            }
        }

        ComputeType amax = AbsMaxOp<ComputeType>::init();
#pragma unroll
        for (int32_t i = 0; i < UNROLL; ++i) {
            amax = AbsMaxOp<ComputeType>::op(amax, amax_regs[i]);
        }
        amax = BlockReduce<AbsMaxOp, ComputeType>(amax);

        // scale
        scale     = compute_scale_from_amax_device_kernel<ComputeType>(amax, CLIP_MAX, EPS);
        scale_inv = 1.0f / scale;
    }

    // quantize
    QType st_regs[UNROLL];
    for (int64_t offset = start_offset; offset < inner_len; offset += (BLOCK_SIZE * UNROLL)) {
        load_data<FType, UNROLL>(input_ptr + offset, ld_regs);
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            st_regs[i] = static_cast<QType>(
                QuantOpBase<ComputeType>::quant(ld_regs[i], scale, CLIP_MIN, CLIP_MAX));
        }
        store_data<QType, UNROLL>(output_ptr + offset, st_regs);
    }

    if (PreComputeScale == false && threadIdx.x == 0) {
        scale_ptr[bid]     = static_cast<float>(scale);
        scale_inv_ptr[bid] = static_cast<float>(scale_inv);
    }
}

// Rowwise
template <typename FType, typename QType, typename ComputeType, bool PreComputeScale>
void quantize_rowwise_row_major_impl(const FType *x, float *scale, float *scale_inv, QType *y,
                                     const int64_t outer_len, const int64_t inner_len,
                                     hipStream_t stream) {

    const int32_t BLOCK_SIZE = 512;
    const int32_t GRID_SIZE  = outer_len;
    int32_t       pack_size  = std::min(get_pack_size<FType>(x), get_pack_size<QType>(y));
    pack_size                = get_quantize_rowwise_pack_size<FType>(pack_size, inner_len);

    switch (pack_size) {
    case 8: {
        const int32_t UNROLL = valid_pack<FType, 8>();
        quantize_rowwise_row_major_two_scan_kernel<BLOCK_SIZE, UNROLL, PreComputeScale, FType,
                                                   QType, ComputeType>
            <<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(x, scale, scale_inv, y, inner_len);
        break;
    }
    case 4: {
        const int32_t UNROLL = valid_pack<FType, 4>();
        quantize_rowwise_row_major_two_scan_kernel<BLOCK_SIZE, UNROLL, PreComputeScale, FType,
                                                   QType, ComputeType>
            <<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(x, scale, scale_inv, y, inner_len);
        break;
    }
    case 2: {
        const int32_t UNROLL = valid_pack<FType, 2>();
        quantize_rowwise_row_major_two_scan_kernel<BLOCK_SIZE, UNROLL, PreComputeScale, FType,
                                                   QType, ComputeType>
            <<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(x, scale, scale_inv, y, inner_len);
        break;
    }
    case 1: {
        const int32_t UNROLL = 1;
        quantize_rowwise_row_major_two_scan_kernel<BLOCK_SIZE, UNROLL, PreComputeScale, FType,
                                                   QType, ComputeType>
            <<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(x, scale, scale_inv, y, inner_len);
        break;
    }
    default:
        PRIMUS_TURBO_ERROR("Error Pack Size");
        break;
    }
}

// ===========================================================================
// Quantize Rowwise: Col-Major
// ===========================================================================
template <int BLOCK_SIZE, int UNROLL_M, int UNROLL_N, typename FType, typename QType,
          typename ComputeType = float>
__launch_bounds__(BLOCK_SIZE) __global__
    void quantize_rowwise_col_major_kernel(const FType *__restrict__ input_ptr,
                                           const float *__restrict__ scale_ptr,
                                           QType *__restrict__ output_ptr, const int64_t m,
                                           const int64_t n) {
    const ComputeType CLIP_MIN = static_cast<ComputeType>(std::numeric_limits<QType>::lowest());
    const ComputeType CLIP_MAX = static_cast<ComputeType>(std::numeric_limits<QType>::max());

    const int32_t tid   = threadIdx.x;
    const int32_t bid_x = blockIdx.x;
    const int32_t bid_y = blockIdx.y;
    const int32_t bid_z = blockIdx.z;

    const int64_t offset_m     = bid_y * UNROLL_M;
    const int64_t offset_n     = bid_x * BLOCK_SIZE * UNROLL_N + tid * UNROLL_N;
    const int64_t offset_input = bid_z * m * n + offset_m * n + offset_n;
    const int64_t offset_scale = bid_z * n + offset_n;

    if (offset_n >= n)
        return;

    input_ptr += offset_input;
    scale_ptr += offset_scale;
    output_ptr += offset_input;

    FType ld_regs[UNROLL_N];
    QType st_regs[UNROLL_N];
    float scale_regs[UNROLL_N];

    if constexpr (UNROLL_N == 8) {
        load_data<float, 4>(scale_ptr + 0, scale_regs + 0);
        load_data<float, 4>(scale_ptr + 4, scale_regs + 4);
    } else {
        load_data<float, UNROLL_N>(scale_ptr, scale_regs);
    }

    const int32_t m_remaining = static_cast<int32_t>(m - offset_m);
    const int32_t m_valid     = m_remaining > UNROLL_M ? UNROLL_M : m_remaining;
    for (int mi = 0; mi < m_valid; ++mi) {
        load_data<FType, UNROLL_N>(input_ptr + mi * n, ld_regs);
#pragma unroll
        for (int i = 0; i < UNROLL_N; ++i) {
            st_regs[i] = static_cast<QType>(
                QuantOpBase<ComputeType>::quant(ld_regs[i], scale_regs[i], CLIP_MIN, CLIP_MAX));
        }
        store_data<QType, UNROLL_N>(output_ptr + mi * n, st_regs);
    }
}

template <typename FType, typename QType, typename ComputeType>
void quantize_rowwise_col_major_impl(const FType *x, float *scale, float *scale_inv, QType *y,
                                     const int64_t batch, const int64_t m, const int64_t n,
                                     hipStream_t stream) {
    const int32_t UNROLL_M = 32;

    int32_t pack_size        = std::min(get_pack_size<FType>(x), get_pack_size<QType>(y));
    pack_size                = get_quantize_rowwise_pack_size<FType>(pack_size, n);
    const int32_t BLOCK_SIZE = 512;

    switch (pack_size) {
    case 8: {
        const int32_t UNROLL_N = valid_pack<FType, 8>();
        const dim3 GRID_SIZE(DIVUP<int64_t>(n, BLOCK_SIZE * UNROLL_N), DIVUP<int64_t>(m, UNROLL_M),
                             batch);
        quantize_rowwise_col_major_kernel<BLOCK_SIZE, UNROLL_M, UNROLL_N, FType, QType, float>
            <<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(x, scale, y, m, n);
        break;
    }
    case 4: {
        const int32_t UNROLL_N = valid_pack<FType, 4>();
        const dim3 GRID_SIZE(DIVUP<int64_t>(n, BLOCK_SIZE * UNROLL_N), DIVUP<int64_t>(m, UNROLL_M),
                             batch);
        quantize_rowwise_col_major_kernel<BLOCK_SIZE, UNROLL_M, UNROLL_N, FType, QType, float>
            <<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(x, scale, y, m, n);
        break;
    }
    case 2: {
        const int32_t UNROLL_N = valid_pack<FType, 2>();
        const dim3 GRID_SIZE(DIVUP<int64_t>(n, BLOCK_SIZE * UNROLL_N), DIVUP<int64_t>(m, UNROLL_M),
                             batch);
        quantize_rowwise_col_major_kernel<BLOCK_SIZE, UNROLL_M, UNROLL_N, FType, QType, float>
            <<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(x, scale, y, m, n);
        break;
    }
    case 1: {
        const int32_t UNROLL_N = 1;
        const dim3 GRID_SIZE(DIVUP<int64_t>(n, BLOCK_SIZE * UNROLL_N), DIVUP<int64_t>(m, UNROLL_M),
                             batch);
        quantize_rowwise_col_major_kernel<BLOCK_SIZE, UNROLL_M, UNROLL_N, FType, QType, float>
            <<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(x, scale, y, m, n);
        break;
    }
    default:
        PRIMUS_TURBO_ERROR("Error Pack Size");
        break;
    }
}

// ===========================================================================
// Dequantize Rowwise: Row-Major
// ===========================================================================
// Row-major dequant: scale_inv has one element per row of the flattened
// [outer_len, inner_len] view.
template <int BLOCK_SIZE, int UNROLL, typename FType, typename QType, typename ComputeType = float>
__launch_bounds__(BLOCK_SIZE) __global__
    void dequantize_rowwise_row_major_kernel(const QType *__restrict__ input_ptr,
                                             const float *__restrict__ scale_inv_ptr,
                                             FType *__restrict__ output_ptr,
                                             const int64_t inner_len) {
    const int64_t bid = blockIdx.x;
    const int32_t tid = threadIdx.x;

    const ComputeType scale_inv = static_cast<ComputeType>(scale_inv_ptr[bid]);

    input_ptr += bid * inner_len;
    output_ptr += bid * inner_len;

    QType ld_regs[UNROLL];
    FType st_regs[UNROLL];

    const int64_t start_offset = static_cast<int64_t>(tid) * UNROLL;
    const int64_t stride       = static_cast<int64_t>(BLOCK_SIZE) * UNROLL;

    for (int64_t offset = start_offset; offset < inner_len; offset += stride) {
        load_data<QType, UNROLL>(input_ptr + offset, ld_regs);
#pragma unroll
        for (int32_t i = 0; i < UNROLL; ++i) {
            st_regs[i] = static_cast<FType>(static_cast<ComputeType>(ld_regs[i]) * scale_inv);
        }
        store_data<FType, UNROLL>(output_ptr + offset, st_regs);
    }
}

template <typename FType, typename QType, typename ComputeType>
void dequantize_rowwise_row_major_impl(const QType *x, const float *scale_inv, FType *y,
                                       const int64_t outer_len, const int64_t inner_len,
                                       hipStream_t stream) {
    const int32_t BLOCK_SIZE = 512;
    const int32_t GRID_SIZE  = outer_len;
    int32_t       pack_size  = std::min(get_pack_size<QType>(x), get_pack_size<FType>(y));
    pack_size                = get_quantize_rowwise_pack_size<FType>(pack_size, inner_len);

    switch (pack_size) {
    case 8: {
        const int32_t UNROLL = valid_pack<FType, 8>();
        dequantize_rowwise_row_major_kernel<BLOCK_SIZE, UNROLL, FType, QType, ComputeType>
            <<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(x, scale_inv, y, inner_len);
        break;
    }
    case 4: {
        const int32_t UNROLL = valid_pack<FType, 4>();
        dequantize_rowwise_row_major_kernel<BLOCK_SIZE, UNROLL, FType, QType, ComputeType>
            <<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(x, scale_inv, y, inner_len);
        break;
    }
    case 2: {
        const int32_t UNROLL = valid_pack<FType, 2>();
        dequantize_rowwise_row_major_kernel<BLOCK_SIZE, UNROLL, FType, QType, ComputeType>
            <<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(x, scale_inv, y, inner_len);
        break;
    }
    case 1: {
        const int32_t UNROLL = 1;
        dequantize_rowwise_row_major_kernel<BLOCK_SIZE, UNROLL, FType, QType, ComputeType>
            <<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(x, scale_inv, y, inner_len);
        break;
    }
    default:
        PRIMUS_TURBO_ERROR("Error Pack Size");
        break;
    }
}

// ===========================================================================
// Dequantize Rowwise: Col-Major
// ===========================================================================
// Col-major dequant: input is viewed as [B, M, N] and scale_inv as [B, N]
// (broadcast across the M dim).
template <int BLOCK_SIZE, int UNROLL_M, int UNROLL_N, typename FType, typename QType,
          typename ComputeType = float>
__launch_bounds__(BLOCK_SIZE) __global__
    void dequantize_rowwise_col_major_kernel(const QType *__restrict__ input_ptr,
                                             const float *__restrict__ scale_inv_ptr,
                                             FType *__restrict__ output_ptr, const int64_t m,
                                             const int64_t n) {
    const int32_t tid   = threadIdx.x;
    const int32_t bid_x = blockIdx.x;
    const int32_t bid_y = blockIdx.y;
    const int32_t bid_z = blockIdx.z;

    const int64_t offset_m         = bid_y * UNROLL_M;
    const int64_t offset_n         = bid_x * BLOCK_SIZE * UNROLL_N + tid * UNROLL_N;
    const int64_t offset_input     = bid_z * m * n + offset_m * n + offset_n;
    const int64_t offset_scale_inv = bid_z * n + offset_n;

    if (offset_n >= n)
        return;

    input_ptr += offset_input;
    scale_inv_ptr += offset_scale_inv;
    output_ptr += offset_input;

    QType ld_regs[UNROLL_N];
    FType st_regs[UNROLL_N];
    float scale_inv_regs[UNROLL_N];

    if constexpr (UNROLL_N == 8) {
        load_data<float, 4>(scale_inv_ptr + 0, scale_inv_regs + 0);
        load_data<float, 4>(scale_inv_ptr + 4, scale_inv_regs + 4);
    } else {
        load_data<float, UNROLL_N>(scale_inv_ptr, scale_inv_regs);
    }

    const int32_t m_remaining = static_cast<int32_t>(m - offset_m);
    const int32_t m_valid     = m_remaining > UNROLL_M ? UNROLL_M : m_remaining;
    for (int mi = 0; mi < m_valid; ++mi) {
        load_data<QType, UNROLL_N>(input_ptr + mi * n, ld_regs);
#pragma unroll
        for (int i = 0; i < UNROLL_N; ++i) {
            st_regs[i] = static_cast<FType>(static_cast<ComputeType>(ld_regs[i]) *
                                            static_cast<ComputeType>(scale_inv_regs[i]));
        }
        store_data<FType, UNROLL_N>(output_ptr + mi * n, st_regs);
    }
}

template <typename FType, typename QType, typename ComputeType>
void dequantize_rowwise_col_major_impl(const QType *x, const float *scale_inv, FType *y,
                                       const int64_t batch, const int64_t m, const int64_t n,
                                       hipStream_t stream) {
    const int32_t UNROLL_M = 32;

    int32_t pack_size        = std::min(get_pack_size<QType>(x), get_pack_size<FType>(y));
    pack_size                = get_quantize_rowwise_pack_size<FType>(pack_size, n);
    const int32_t BLOCK_SIZE = 512;

    switch (pack_size) {
    case 8: {
        const int32_t UNROLL_N = valid_pack<FType, 8>();
        const dim3 GRID_SIZE(DIVUP<int64_t>(n, BLOCK_SIZE * UNROLL_N), DIVUP<int64_t>(m, UNROLL_M),
                             batch);
        dequantize_rowwise_col_major_kernel<BLOCK_SIZE, UNROLL_M, UNROLL_N, FType, QType,
                                            ComputeType>
            <<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(x, scale_inv, y, m, n);
        break;
    }
    case 4: {
        const int32_t UNROLL_N = valid_pack<FType, 4>();
        const dim3 GRID_SIZE(DIVUP<int64_t>(n, BLOCK_SIZE * UNROLL_N), DIVUP<int64_t>(m, UNROLL_M),
                             batch);
        dequantize_rowwise_col_major_kernel<BLOCK_SIZE, UNROLL_M, UNROLL_N, FType, QType,
                                            ComputeType>
            <<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(x, scale_inv, y, m, n);
        break;
    }
    case 2: {
        const int32_t UNROLL_N = valid_pack<FType, 2>();
        const dim3 GRID_SIZE(DIVUP<int64_t>(n, BLOCK_SIZE * UNROLL_N), DIVUP<int64_t>(m, UNROLL_M),
                             batch);
        dequantize_rowwise_col_major_kernel<BLOCK_SIZE, UNROLL_M, UNROLL_N, FType, QType,
                                            ComputeType>
            <<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(x, scale_inv, y, m, n);
        break;
    }
    case 1: {
        const int32_t UNROLL_N = 1;
        const dim3 GRID_SIZE(DIVUP<int64_t>(n, BLOCK_SIZE * UNROLL_N), DIVUP<int64_t>(m, UNROLL_M),
                             batch);
        dequantize_rowwise_col_major_kernel<BLOCK_SIZE, UNROLL_M, UNROLL_N, FType, QType,
                                            ComputeType>
            <<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(x, scale_inv, y, m, n);
        break;
    }
    default:
        PRIMUS_TURBO_ERROR("Error Pack Size");
        break;
    }
}

// ===========================================================================
// Explicit instantiations
// ===========================================================================
#define DECL_QUANT_ROWWISE_ROW_MAJOR_INSTANCE(FType, QType)                                        \
    template void quantize_rowwise_row_major_impl<FType, QType, float, true>(                      \
        const FType *x, float *scale, float *scale_inv, QType *y, const int64_t outer_len,         \
        const int64_t inner_len, hipStream_t stream);                                              \
    template void quantize_rowwise_row_major_impl<FType, QType, float, false>(                     \
        const FType *x, float *scale, float *scale_inv, QType *y, const int64_t outer_len,         \
        const int64_t inner_len, hipStream_t stream);

DECL_QUANT_ROWWISE_ROW_MAJOR_INSTANCE(dtype::float16, dtype::float8_e4m3)
DECL_QUANT_ROWWISE_ROW_MAJOR_INSTANCE(dtype::float16, dtype::float8_e5m2)
DECL_QUANT_ROWWISE_ROW_MAJOR_INSTANCE(dtype::bfloat16, dtype::float8_e4m3)
DECL_QUANT_ROWWISE_ROW_MAJOR_INSTANCE(dtype::bfloat16, dtype::float8_e5m2)
DECL_QUANT_ROWWISE_ROW_MAJOR_INSTANCE(dtype::float32, dtype::float8_e4m3)
DECL_QUANT_ROWWISE_ROW_MAJOR_INSTANCE(dtype::float32, dtype::float8_e5m2)

#undef DECL_QUANT_ROWWISE_ROW_MAJOR_INSTANCE

#define DECL_QUANT_ROWWISE_COL_MAJOR_INSTANCE(FType, QType)                                        \
    template void quantize_rowwise_col_major_impl<FType, QType, float>(                            \
        const FType *x, float *scale, float *scale_inv, QType *y, const int64_t batch,             \
        const int64_t m, const int64_t n, hipStream_t stream);

// F16/BF16/F32 -> FP8 (E4M3/E5M2)
DECL_QUANT_ROWWISE_COL_MAJOR_INSTANCE(dtype::float16, dtype::float8_e4m3)
DECL_QUANT_ROWWISE_COL_MAJOR_INSTANCE(dtype::float16, dtype::float8_e5m2)
DECL_QUANT_ROWWISE_COL_MAJOR_INSTANCE(dtype::bfloat16, dtype::float8_e4m3)
DECL_QUANT_ROWWISE_COL_MAJOR_INSTANCE(dtype::bfloat16, dtype::float8_e5m2)
DECL_QUANT_ROWWISE_COL_MAJOR_INSTANCE(dtype::float32, dtype::float8_e4m3)
DECL_QUANT_ROWWISE_COL_MAJOR_INSTANCE(dtype::float32, dtype::float8_e5m2)

#undef DECL_QUANT_ROWWISE_COL_MAJOR_INSTANCE

#define DECL_DEQUANT_ROWWISE_INSTANCE(FType, QType)                                                \
    template void dequantize_rowwise_row_major_impl<FType, QType, float>(                          \
        const QType *x, const float *scale_inv, FType *y, const int64_t outer_len,                 \
        const int64_t inner_len, hipStream_t stream);                                              \
    template void dequantize_rowwise_col_major_impl<FType, QType, float>(                          \
        const QType *x, const float *scale_inv, FType *y, const int64_t batch, const int64_t m,    \
        const int64_t n, hipStream_t stream);

DECL_DEQUANT_ROWWISE_INSTANCE(dtype::float16, dtype::float8_e4m3)
DECL_DEQUANT_ROWWISE_INSTANCE(dtype::float16, dtype::float8_e5m2)
DECL_DEQUANT_ROWWISE_INSTANCE(dtype::bfloat16, dtype::float8_e4m3)
DECL_DEQUANT_ROWWISE_INSTANCE(dtype::bfloat16, dtype::float8_e5m2)
DECL_DEQUANT_ROWWISE_INSTANCE(dtype::float32, dtype::float8_e4m3)
DECL_DEQUANT_ROWWISE_INSTANCE(dtype::float32, dtype::float8_e5m2)

#undef DECL_DEQUANT_ROWWISE_INSTANCE

} // namespace primus_turbo
