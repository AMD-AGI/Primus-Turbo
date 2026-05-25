// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/common.h"
#include "primus_turbo/device/reduce.cuh"
#include "primus_turbo/elementwise/binary_kernel_template.cuh"
#include "primus_turbo/elementwise/unary_kernel_template.cuh"
#include "primus_turbo/memory_pack.h"
#include "primus_turbo/quantization.h"

namespace primus_turbo {

using namespace primus_turbo::dtype;

template <typename ComputeType = float> struct QuantOpBase {
    static PRIMUS_TURBO_HOST_DEVICE ComputeType quant(const ComputeType x, const ComputeType scale,
                                                      const ComputeType clip_min,
                                                      const ComputeType clip_max) {
        const ComputeType v = x * scale;
        return fmax(fmin(v, clip_max), clip_min);
    }
};

template <typename ComputeType = float> struct QuantOp : QuantOpBase<ComputeType> {
    ComputeType clip_min;
    ComputeType clip_max;

    PRIMUS_TURBO_HOST_DEVICE ComputeType operator()(const ComputeType x,
                                                    const ComputeType scale) const {
        return QuantOpBase<ComputeType>::quant(x, scale, clip_min, clip_max);
    }
};

template <typename ComputeType = float>
struct QuantTensorwiseScalePtrOp : QuantOpBase<ComputeType> {
    const ComputeType *scale_ptr;
    ComputeType        clip_min;
    ComputeType        clip_max;

    PRIMUS_TURBO_HOST_DEVICE ComputeType operator()(ComputeType x) const {
        const ComputeType scale = scale_ptr[0];
        return QuantOpBase<ComputeType>::quant(x, scale, clip_min, clip_max);
    }
};

template <typename ComputeType = float> struct DeQuantTensorwiseScaleInvPtrOp {
    const ComputeType *scale_inv_ptr;

    PRIMUS_TURBO_HOST_DEVICE ComputeType operator()(ComputeType x) const {
        const ComputeType scale_inv = scale_inv_ptr[0];
        return x * scale_inv;
    }
};

template <typename T = float>
PRIMUS_TURBO_DEVICE T compute_scale_from_amax_device_kernel(const T amax, const T q_max,
                                                            const float eps) {
    float amax_t = fmax(static_cast<float>(amax), eps);
    return static_cast<T>(static_cast<float>(q_max) / amax_t);
}

template <typename T>
__global__ void compute_scale_from_amax_kernel(const T *amax_ptr, const T q_max, T *scale_ptr,
                                               T *scale_inv_ptr, const int64_t n, const float eps) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < n) {
        float amax         = static_cast<float>(amax_ptr[tid]);
        amax               = fmax(amax, eps);
        float scale        = static_cast<float>(q_max) / amax;
        float scale_inv    = 1.0f / scale;
        scale_ptr[tid]     = static_cast<T>(scale);
        scale_inv_ptr[tid] = static_cast<T>(scale_inv);
    }
}

template <typename T>
void compute_scale_from_amax(const T *amax, const T q_max, T *scale, T *scale_inv, const int64_t n,
                             hipStream_t stream, const float eps) {
    const int64_t BLOCK_SIZE = 512;
    const int64_t GRID_SIZE  = DIVUP<int64_t>(n, BLOCK_SIZE);
    compute_scale_from_amax_kernel<T>
        <<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(amax, q_max, scale, scale_inv, n, eps);
}

template <typename FType, typename QType, typename ComputeType>
void quantize_tensorwise_impl(const FType *x, const float *scale, QType *y, const int64_t n,
                              hipStream_t stream) {
    QuantTensorwiseScalePtrOp<ComputeType> op{
        {},
        reinterpret_cast<const ComputeType *>(scale),
        static_cast<ComputeType>(std::numeric_limits<QType>::lowest()),
        static_cast<ComputeType>(std::numeric_limits<QType>::max())};

    const int32_t BLOCK_SIZE = 512;

    int32_t pack_size = std::min(get_pack_size<FType>(x), get_pack_size<QType>(y));
    switch (pack_size) {
    case 8: {
        const int32_t       UNROLL = valid_pack<FType, 8>();
        PackedEltwiseConfig pack_cfg(n, UNROLL, BLOCK_SIZE);
        unary_kernel<BLOCK_SIZE, UNROLL, FType, QType, QuantTensorwiseScalePtrOp<ComputeType>>
            <<<pack_cfg.nBlock, BLOCK_SIZE, 0, stream>>>(x, y, op, pack_cfg);
        break;
    }
    case 4: {
        const int32_t       UNROLL = valid_pack<FType, 4>();
        PackedEltwiseConfig pack_cfg(n, UNROLL, BLOCK_SIZE);
        unary_kernel<BLOCK_SIZE, UNROLL, FType, QType, QuantTensorwiseScalePtrOp<ComputeType>>
            <<<pack_cfg.nBlock, BLOCK_SIZE, 0, stream>>>(x, y, op, pack_cfg);
        break;
    }
    case 2: {
        const int32_t       UNROLL = valid_pack<FType, 2>();
        PackedEltwiseConfig pack_cfg(n, UNROLL, BLOCK_SIZE);
        unary_kernel<BLOCK_SIZE, UNROLL, FType, QType, QuantTensorwiseScalePtrOp<ComputeType>>
            <<<pack_cfg.nBlock, BLOCK_SIZE, 0, stream>>>(x, y, op, pack_cfg);
        break;
    }
    case 1: {
        PackedEltwiseConfig pack_cfg(n, 1, BLOCK_SIZE);
        unary_kernel<BLOCK_SIZE, 1, FType, QType, QuantTensorwiseScalePtrOp<ComputeType>>
            <<<pack_cfg.nBlock, BLOCK_SIZE, 0, stream>>>(x, y, op, pack_cfg);
        break;
    }
    default:
        PRIMUS_TURBO_ERROR("Error Pack Size");
        break;
    }
}

template <typename FType, typename QType, typename ComputeType>
void dequantize_tensorwise_impl(const QType *x, const float *scale_inv, FType *y, const int64_t n,
                                hipStream_t stream) {
    DeQuantTensorwiseScaleInvPtrOp<ComputeType> op{
        reinterpret_cast<const ComputeType *>(scale_inv),
    };

    const int32_t BLOCK_SIZE = 512;
    int32_t       pack_size  = std::min(get_pack_size<QType>(x), get_pack_size<FType>(y));
    switch (pack_size) {
    case 8: {
        const int32_t       UNROLL = valid_pack<FType, 8>();
        PackedEltwiseConfig pack_cfg(n, UNROLL, BLOCK_SIZE);
        unary_kernel<BLOCK_SIZE, UNROLL, QType, FType, DeQuantTensorwiseScaleInvPtrOp<ComputeType>>
            <<<pack_cfg.nBlock, BLOCK_SIZE, 0, stream>>>(x, y, op, pack_cfg);
        break;
    }
    case 4: {
        const int32_t       UNROLL = valid_pack<FType, 4>();
        PackedEltwiseConfig pack_cfg(n, UNROLL, BLOCK_SIZE);
        unary_kernel<BLOCK_SIZE, UNROLL, QType, FType, DeQuantTensorwiseScaleInvPtrOp<ComputeType>>
            <<<pack_cfg.nBlock, BLOCK_SIZE, 0, stream>>>(x, y, op, pack_cfg);
        break;
    }
    case 2: {
        const int32_t       UNROLL = valid_pack<FType, 2>();
        PackedEltwiseConfig pack_cfg(n, UNROLL, BLOCK_SIZE);
        unary_kernel<BLOCK_SIZE, UNROLL, QType, FType, DeQuantTensorwiseScaleInvPtrOp<ComputeType>>
            <<<pack_cfg.nBlock, BLOCK_SIZE, 0, stream>>>(x, y, op, pack_cfg);
        break;
    }
    case 1: {
        PackedEltwiseConfig pack_cfg(n, 1, BLOCK_SIZE);
        unary_kernel<BLOCK_SIZE, 1, QType, FType, DeQuantTensorwiseScaleInvPtrOp<ComputeType>>
            <<<pack_cfg.nBlock, BLOCK_SIZE, 0, stream>>>(x, y, op, pack_cfg);
        break;
    }
    default:
        PRIMUS_TURBO_ERROR("Error Pack Size");
        break;
    }
}

// **** Explicit Instantiation ****
template void compute_scale_from_amax<float>(const float *amax, float q_max, float *scale,
                                             float *scale_inv, const int64_t n, hipStream_t stream,
                                             const float eps);

#define DECL_QUANT_ADN_DEQUANT_TENSORWISE_INSTANCE(FType, QType)                                   \
    template void quantize_tensorwise_impl<FType, QType>(                                          \
        const FType *x, const float *scale, QType *y, const int64_t n, hipStream_t stream);        \
    template void dequantize_tensorwise_impl<FType, QType>(                                        \
        const QType *x, const float *scale_inv, FType *y, const int64_t n, hipStream_t stream);

DECL_QUANT_ADN_DEQUANT_TENSORWISE_INSTANCE(dtype::float16, dtype::float8_e4m3)
DECL_QUANT_ADN_DEQUANT_TENSORWISE_INSTANCE(dtype::float16, dtype::float8_e5m2)
DECL_QUANT_ADN_DEQUANT_TENSORWISE_INSTANCE(dtype::bfloat16, dtype::float8_e4m3)
DECL_QUANT_ADN_DEQUANT_TENSORWISE_INSTANCE(dtype::bfloat16, dtype::float8_e5m2)
DECL_QUANT_ADN_DEQUANT_TENSORWISE_INSTANCE(dtype::float32, dtype::float8_e4m3)
DECL_QUANT_ADN_DEQUANT_TENSORWISE_INSTANCE(dtype::float32, dtype::float8_e5m2)

#undef DECL_QUANT_ADN_DEQUANT_TENSORWISE_INSTANCE

// ******************************************************************
// ******************************************************************
// ******************************************************************

template <typename T>
int32_t get_quantize_rowwise_pack_size(const int32_t pack_size, const int64_t inner_len) {
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

#define DECL_QUANT_ADN_DEQUANT_ROWWISE_ROW_MAJOR_INSTANCE(FType, QType)                            \
    template void quantize_rowwise_row_major_impl<FType, QType, float, true>(                      \
        const FType *x, float *scale, float *scale_inv, QType *y, const int64_t outer_len,         \
        const int64_t inner_len, hipStream_t stream);                                              \
    template void quantize_rowwise_row_major_impl<FType, QType, float, false>(                     \
        const FType *x, float *scale, float *scale_inv, QType *y, const int64_t outer_len,         \
        const int64_t inner_len, hipStream_t stream);

DECL_QUANT_ADN_DEQUANT_ROWWISE_ROW_MAJOR_INSTANCE(dtype::float16, dtype::float8_e4m3)
DECL_QUANT_ADN_DEQUANT_ROWWISE_ROW_MAJOR_INSTANCE(dtype::float16, dtype::float8_e5m2)
DECL_QUANT_ADN_DEQUANT_ROWWISE_ROW_MAJOR_INSTANCE(dtype::bfloat16, dtype::float8_e4m3)
DECL_QUANT_ADN_DEQUANT_ROWWISE_ROW_MAJOR_INSTANCE(dtype::bfloat16, dtype::float8_e5m2)
DECL_QUANT_ADN_DEQUANT_ROWWISE_ROW_MAJOR_INSTANCE(dtype::float32, dtype::float8_e4m3)
DECL_QUANT_ADN_DEQUANT_ROWWISE_ROW_MAJOR_INSTANCE(dtype::float32, dtype::float8_e5m2)

#undef DECL_QUANT_ADN_DEQUANT_ROWWISE_INSTANCE

#define DECL_QUANT_AND_DEQUANT_ROWWISE_COL_MAJOR_INSTANCE(FType, QType)                            \
    template void quantize_rowwise_col_major_impl<FType, QType, float>(                            \
        const FType *x, float *scale, float *scale_inv, QType *y, const int64_t batch,             \
        const int64_t m, const int64_t n, hipStream_t stream);

// F16/BF16/F32 -> FP8 (E4M3/E5M2)
DECL_QUANT_AND_DEQUANT_ROWWISE_COL_MAJOR_INSTANCE(dtype::float16, dtype::float8_e4m3)
DECL_QUANT_AND_DEQUANT_ROWWISE_COL_MAJOR_INSTANCE(dtype::float16, dtype::float8_e5m2)
DECL_QUANT_AND_DEQUANT_ROWWISE_COL_MAJOR_INSTANCE(dtype::bfloat16, dtype::float8_e4m3)
DECL_QUANT_AND_DEQUANT_ROWWISE_COL_MAJOR_INSTANCE(dtype::bfloat16, dtype::float8_e5m2)
DECL_QUANT_AND_DEQUANT_ROWWISE_COL_MAJOR_INSTANCE(dtype::float32, dtype::float8_e4m3)
DECL_QUANT_AND_DEQUANT_ROWWISE_COL_MAJOR_INSTANCE(dtype::float32, dtype::float8_e5m2)

#undef DECL_QUANT_AND_DEQUANT_ROWWISE_COL_MAJOR_INSTANCE

// Blockwise FP8 quant (BLOCK=128, axis-wise scaling).
// 256 threads/block; vec-8 packed bf16 loads + FP8 stores; sub-warp xor reduce for amax.
//   axis=1 tile = 16 rows × 128 cols, 16 threads per row.
//   axis=0 tile = 128 rows × 128 cols via 32KB LDS cache, 2 threads per col.
template <typename FType, typename QType>
__launch_bounds__(256) __global__
void quant_fp8_blockwise_axis1_tile_kernel(const FType *__restrict__ x_ptr,
                                            QType *__restrict__ x_fp8_ptr,
                                            float *__restrict__ x_scales_inv_ptr,
                                            const int64_t M, const int64_t N,
                                            const float fp8_max) {
    constexpr int BLOCK_M    = 16;
    constexpr int BLOCK_N    = 128;
    constexpr int PACK       = 8;
    constexpr int THREADS_PER_ROW = BLOCK_N / PACK;  // 16
    const int64_t row_blk = blockIdx.x;
    const int64_t col_blk = blockIdx.y;
    const int     tid     = threadIdx.x;
    const int     row_in_blk = tid / THREADS_PER_ROW;
    const int     pack_idx   = tid % THREADS_PER_ROW;

    const int64_t row = row_blk * BLOCK_M + row_in_blk;
    if (row >= M) return;
    const int64_t col_start = col_blk * BLOCK_N;
    const int64_t base      = row * N + col_start;
    const int     col_off   = pack_idx * PACK;
    const bool    in_range  = (col_start + col_off + PACK <= N);

    FType vals_f[PACK];
    if (in_range) {
        load_data<FType, PACK>(x_ptr + base + col_off, vals_f);
    } else {
        #pragma unroll
        for (int i = 0; i < PACK; ++i) vals_f[i] = static_cast<FType>(0.f);
        for (int i = 0; i < PACK; ++i) {
            if (col_start + col_off + i < N) vals_f[i] = x_ptr[base + col_off + i];
        }
    }

    float amax = 0.f;
    float vals[PACK];
    #pragma unroll
    for (int i = 0; i < PACK; ++i) {
        vals[i] = static_cast<float>(vals_f[i]);
        amax    = fmaxf(amax, fabsf(vals[i]));
    }
    #pragma unroll
    for (int offset = THREADS_PER_ROW >> 1; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, THREADS_PER_ROW));
    }

    // Clamp eps matches Triton reference (tl.maximum(amax, 1e-4)) so tiny amax
    // doesn't blow up the scale and overflow FP8 on FNUZ.
    const float scale     = static_cast<float>(fp8_max) / fmaxf(amax, 1e-4f);
    const float scale_inv = 1.0f / scale;
    const float clip_lo   = -static_cast<float>(fp8_max);
    const float clip_hi   =  static_cast<float>(fp8_max);

    QType out[PACK];
    #pragma unroll
    for (int i = 0; i < PACK; ++i) {
        out[i] = static_cast<QType>(fmaxf(fminf(vals[i] * scale, clip_hi), clip_lo));
    }
    if (in_range) {
        store_data<QType, PACK>(x_fp8_ptr + base + col_off, out);
    } else {
        for (int i = 0; i < PACK; ++i) {
            if (col_start + col_off + i < N) x_fp8_ptr[base + col_off + i] = out[i];
        }
    }
    if (pack_idx == 0) {
        const int64_t scale_n_blocks = (N + BLOCK_N - 1) / BLOCK_N;
        x_scales_inv_ptr[row * scale_n_blocks + col_blk] = scale_inv;
    }
}

template <typename FType, typename QType>
__launch_bounds__(256) __global__
void quant_fp8_blockwise_axis0_tile_kernel(const FType *__restrict__ x_ptr,
                                            QType *__restrict__ x_fp8_ptr,
                                            float *__restrict__ x_scales_inv_ptr,
                                            const int64_t M, const int64_t N,
                                            const float fp8_max) {
    constexpr int BLOCK_SIZE = 128;
    constexpr int PACK       = 8;
    constexpr int THREADS_PER_ROW = BLOCK_SIZE / PACK;   // 16
    constexpr int ROWS_PER_ROUND  = 256 / THREADS_PER_ROW;  // 16
    constexpr int ROWS_PER_THREAD = BLOCK_SIZE / 2;      // 64 (2 threads per col)
    const int64_t row_blk = blockIdx.x;
    const int64_t col_blk = blockIdx.y;
    const int     tid     = threadIdx.x;

    const int64_t row_start = row_blk * BLOCK_SIZE;
    const int64_t col_start = col_blk * BLOCK_SIZE;

    __shared__ FType s_tile[BLOCK_SIZE][BLOCK_SIZE];     // 32KB

    const int pack_idx      = tid % THREADS_PER_ROW;
    const int load_row_base = tid / THREADS_PER_ROW;
    #pragma unroll
    for (int r = 0; r < BLOCK_SIZE / ROWS_PER_ROUND; ++r) {
        const int local_m = load_row_base + r * ROWS_PER_ROUND;
        const int local_n = pack_idx * PACK;
        const int64_t gm  = row_start + local_m;
        const int64_t gn  = col_start + local_n;
        if (gm < M && gn + PACK <= N) {
            load_data<FType, PACK>(x_ptr + gm * N + gn, &s_tile[local_m][local_n]);
        } else {
            FType pad[PACK];
            #pragma unroll
            for (int i = 0; i < PACK; ++i) {
                const int64_t cn = gn + i;
                pad[i] = (gm < M && cn < N) ? x_ptr[gm * N + cn] : static_cast<FType>(0.f);
            }
            #pragma unroll
            for (int i = 0; i < PACK; ++i) s_tile[local_m][local_n + i] = pad[i];
        }
    }
    __syncthreads();

    const int col_in_tile = tid % BLOCK_SIZE;
    const int half_row    = tid / BLOCK_SIZE;
    const int row_lo      = half_row * ROWS_PER_THREAD;
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < ROWS_PER_THREAD; ++i) {
        amax = fmaxf(amax, fabsf(static_cast<float>(s_tile[row_lo + i][col_in_tile])));
    }
    __shared__ float s_amax_partial[BLOCK_SIZE][2];
    s_amax_partial[col_in_tile][half_row] = amax;
    __syncthreads();

    __shared__ float s_scale[BLOCK_SIZE];
    if (half_row == 0) {
        amax = fmaxf(s_amax_partial[col_in_tile][0], s_amax_partial[col_in_tile][1]);
        const float scale = static_cast<float>(fp8_max) / fmaxf(amax, 1e-4f);
        s_scale[col_in_tile] = scale;
        const int64_t gn = col_start + col_in_tile;
        if (gn < N) x_scales_inv_ptr[row_blk * N + gn] = 1.0f / scale;
    }
    __syncthreads();

    const float clip_lo = -static_cast<float>(fp8_max);
    const float clip_hi =  static_cast<float>(fp8_max);
    #pragma unroll
    for (int r = 0; r < BLOCK_SIZE / ROWS_PER_ROUND; ++r) {
        const int local_m = load_row_base + r * ROWS_PER_ROUND;
        const int local_n = pack_idx * PACK;
        const int64_t gm  = row_start + local_m;
        const int64_t gn  = col_start + local_n;

        QType out[PACK];
        #pragma unroll
        for (int i = 0; i < PACK; ++i) {
            const float v  = static_cast<float>(s_tile[local_m][local_n + i]);
            const float sc = s_scale[local_n + i];
            out[i] = static_cast<QType>(fmaxf(fminf(v * sc, clip_hi), clip_lo));
        }
        if (gm < M && gn + PACK <= N) {
            store_data<QType, PACK>(x_fp8_ptr + gm * N + gn, out);
        } else {
            #pragma unroll
            for (int i = 0; i < PACK; ++i) {
                const int64_t cn = gn + i;
                if (gm < M && cn < N) x_fp8_ptr[gm * N + cn] = out[i];
            }
        }
    }
}

template <typename FType, typename QType, typename ComputeType>
void quantize_blockwise_impl(const FType *x, QType *y, float *scale_inv,
                             const int64_t M, const int64_t N, const int axis,
                             const float fp8_max, hipStream_t stream) {
    constexpr int BLOCK_SIZE = 128;
    const int64_t m_blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int64_t n_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(m_blocks, n_blocks);
    if (axis == 1) {
        constexpr int BLOCK_M_AXIS1 = 16;
        const int64_t m_blocks_a1 = (M + BLOCK_M_AXIS1 - 1) / BLOCK_M_AXIS1;
        dim3 grid_a1(m_blocks_a1, n_blocks);
        quant_fp8_blockwise_axis1_tile_kernel<FType, QType>
            <<<grid_a1, dim3(256), 0, stream>>>(x, y, scale_inv, M, N, fp8_max);
    } else {
        quant_fp8_blockwise_axis0_tile_kernel<FType, QType>
            <<<grid, dim3(256), 0, stream>>>(x, y, scale_inv, M, N, fp8_max);
    }
}

#define DECL_QUANT_BLOCKWISE_INSTANCE(FType, QType)                                     \
    template void quantize_blockwise_impl<FType, QType, float>(                         \
        const FType *x, QType *y, float *scale_inv, const int64_t M, const int64_t N,   \
        const int axis, const float fp8_max, hipStream_t stream);
DECL_QUANT_BLOCKWISE_INSTANCE(dtype::bfloat16, dtype::float8_e4m3)
DECL_QUANT_BLOCKWISE_INSTANCE(dtype::bfloat16, dtype::float8_e5m2)
DECL_QUANT_BLOCKWISE_INSTANCE(dtype::float16, dtype::float8_e4m3)
DECL_QUANT_BLOCKWISE_INSTANCE(dtype::float16, dtype::float8_e5m2)
#undef DECL_QUANT_BLOCKWISE_INSTANCE


// Single-pass fused row + segment-padded col blockwise FP8 quant.
// One bf16 read of x [M_in, N]; vals cached in vregs (64 fp32/thread).
// Row amax: 16-thread sub-warp xor reduce → write row FP8 + scale per round.
// Col amax: per-thread col_amax[8] over 8 rounds → LDS reduce across 16 row-slots
// → broadcast scale; col-padded FP8 written from regs. ~9KB LDS, 3-4 blocks/CU.
template <typename FType, typename QType>
__launch_bounds__(256) __global__
void quant_fp8_blockwise_segment_m_row_col_kernel(
    const FType *__restrict__ x_ptr,
    QType *__restrict__ x_fp8_row_ptr,
    QType *__restrict__ x_fp8_col_padded_ptr,
    float *__restrict__ x_scales_row_ptr,
    float *__restrict__ x_scales_col_padded_ptr,
    const int64_t *__restrict__ group_offs_ptr,
    const int64_t *__restrict__ padded_group_offs_ptr,
    const int64_t M_in, const int64_t N, const int num_groups,
    const float fp8_max) {
    constexpr int BLOCK_SIZE = 128;
    constexpr int PACK       = 8;
    constexpr int THREADS_PER_ROW = BLOCK_SIZE / PACK;            // 16
    constexpr int ROWS_PER_ROUND  = 256 / THREADS_PER_ROW;        // 16
    constexpr int ROUNDS          = BLOCK_SIZE / ROWS_PER_ROUND;  // 8

    const int64_t pid_m   = blockIdx.x;
    const int64_t col_blk = blockIdx.y;
    const int     tid     = threadIdx.x;
    const int     pack_idx      = tid % THREADS_PER_ROW;
    const int     load_row_base = tid / THREADS_PER_ROW;

    const int64_t M_padded = padded_group_offs_ptr[num_groups];
    const int64_t pad_block_start = pid_m * BLOCK_SIZE;
    if (pad_block_start >= M_padded) return;

    int group_id = 0;
    #pragma unroll 1
    for (int g = 0; g < num_groups; ++g) {
        if (pad_block_start >= padded_group_offs_ptr[g] &&
            pad_block_start <  padded_group_offs_ptr[g + 1]) group_id = g;
    }
    const int64_t orig_start = group_offs_ptr[group_id];
    const int64_t orig_end   = group_offs_ptr[group_id + 1];
    const int64_t pad_start  = padded_group_offs_ptr[group_id];

    const int64_t col_start      = col_blk * BLOCK_SIZE;
    const float   clip_lo = -static_cast<float>(fp8_max);
    const float   clip_hi =  static_cast<float>(fp8_max);

    float vals[ROUNDS][PACK];
    float col_amax_local[PACK];
    #pragma unroll
    for (int i = 0; i < PACK; ++i) col_amax_local[i] = 0.f;

    __shared__ bool s_valid_row[BLOCK_SIZE];

    #pragma unroll
    for (int r = 0; r < ROUNDS; ++r) {
        const int local_m = load_row_base + r * ROWS_PER_ROUND;
        const int local_n = pack_idx * PACK;
        const int64_t in_m    = orig_start + (pad_block_start + local_m - pad_start);
        const bool    valid_in = (in_m >= orig_start) && (in_m < orig_end) && (in_m < M_in);
        if (pack_idx == 0) s_valid_row[local_m] = valid_in;

        const int64_t gn = col_start + local_n;
        FType buf[PACK];
        if (valid_in && gn + PACK <= N) {
            load_data<FType, PACK>(x_ptr + in_m * N + gn, buf);
        } else {
            #pragma unroll
            for (int i = 0; i < PACK; ++i) {
                const int64_t cn = gn + i;
                buf[i] = (valid_in && cn < N) ? x_ptr[in_m * N + cn] : static_cast<FType>(0.f);
            }
        }

        float row_amax = 0.f;
        #pragma unroll
        for (int i = 0; i < PACK; ++i) {
            const float v  = static_cast<float>(buf[i]);
            const float av = fabsf(v);
            vals[r][i] = v;
            row_amax = fmaxf(row_amax, av);
            if (valid_in) col_amax_local[i] = fmaxf(col_amax_local[i], av);
        }
        #pragma unroll
        for (int offset = THREADS_PER_ROW >> 1; offset > 0; offset >>= 1) {
            row_amax = fmaxf(row_amax, __shfl_xor(row_amax, offset, THREADS_PER_ROW));
        }
        const float row_scale = static_cast<float>(fp8_max) / fmaxf(row_amax, 1e-4f);

        if (valid_in) {
            QType out[PACK];
            #pragma unroll
            for (int i = 0; i < PACK; ++i) {
                out[i] = static_cast<QType>(fmaxf(fminf(vals[r][i] * row_scale, clip_hi), clip_lo));
            }
            if (gn + PACK <= N) {
                store_data<QType, PACK>(x_fp8_row_ptr + in_m * N + gn, out);
            } else {
                #pragma unroll
                for (int i = 0; i < PACK; ++i) {
                    const int64_t cn = gn + i;
                    if (cn < N) x_fp8_row_ptr[in_m * N + cn] = out[i];
                }
            }
            if (pack_idx == 0) {
                // Pshuffled [N_blocks, M_in]: matches persistent fwd GEMM scale order.
                x_scales_row_ptr[col_blk * M_in + in_m] = 1.0f / row_scale;
            }
        }
    }

    __shared__ float s_col_partial[ROWS_PER_ROUND][BLOCK_SIZE];   // 8KB
    #pragma unroll
    for (int i = 0; i < PACK; ++i) {
        s_col_partial[load_row_base][pack_idx * PACK + i] = col_amax_local[i];
    }
    __syncthreads();

    __shared__ float s_col_scale[BLOCK_SIZE];
    if (tid < BLOCK_SIZE) {
        float col_amax = 0.f;
        #pragma unroll
        for (int rs = 0; rs < ROWS_PER_ROUND; ++rs) {
            col_amax = fmaxf(col_amax, s_col_partial[rs][tid]);
        }
        const float col_scale = static_cast<float>(fp8_max) / fmaxf(col_amax, 1e-4f);
        s_col_scale[tid] = col_scale;
        const int64_t gn = col_start + tid;
        if (gn < N) x_scales_col_padded_ptr[pid_m * N + gn] = 1.0f / col_scale;
    }
    __syncthreads();

    #pragma unroll
    for (int r = 0; r < ROUNDS; ++r) {
        const int local_m = load_row_base + r * ROWS_PER_ROUND;
        const int local_n = pack_idx * PACK;
        const int64_t out_m = pad_block_start + local_m;
        const int64_t gn    = col_start + local_n;
        const bool    valid = s_valid_row[local_m];

        QType out[PACK];
        #pragma unroll
        for (int i = 0; i < PACK; ++i) {
            const float v = valid ? vals[r][i] : 0.f;
            out[i] = static_cast<QType>(
                fmaxf(fminf(v * s_col_scale[local_n + i], clip_hi), clip_lo));
        }
        if (out_m < M_padded && gn + PACK <= N) {
            store_data<QType, PACK>(x_fp8_col_padded_ptr + out_m * N + gn, out);
        } else if (out_m < M_padded) {
            #pragma unroll
            for (int i = 0; i < PACK; ++i) {
                const int64_t cn = gn + i;
                if (cn < N) x_fp8_col_padded_ptr[out_m * N + cn] = out[i];
            }
        }
    }
}

template <typename FType, typename QType>
void quantize_blockwise_segment_m_row_col_impl(
    const FType *x, QType *y_row, QType *y_col_padded,
    float *scales_row, float *scales_col_padded,
    const int64_t *group_offs, const int64_t *padded_group_offs,
    const int64_t M_in, const int64_t N, const int64_t M_padded_max,
    const int num_groups, const float fp8_max, hipStream_t stream) {
    constexpr int BLOCK_SIZE = 128;
    const int64_t m_blocks = (M_padded_max + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int64_t n_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(m_blocks, n_blocks);
    quant_fp8_blockwise_segment_m_row_col_kernel<FType, QType>
        <<<grid, dim3(256), 0, stream>>>(
            x, y_row, y_col_padded, scales_row, scales_col_padded,
            group_offs, padded_group_offs, M_in, N, num_groups, fp8_max);
}

#define DECL_QUANT_BLOCKWISE_SEGM_INSTANCE(FType, QType)                          \
    template void quantize_blockwise_segment_m_row_col_impl<FType, QType>(        \
        const FType *x, QType *y_row, QType *y_col_padded, float *scales_row,           \
        float *scales_col_padded, const int64_t *group_offs,                            \
        const int64_t *padded_group_offs, const int64_t M_in, const int64_t N,          \
        const int64_t M_padded_max, const int num_groups, const float fp8_max,          \
        hipStream_t stream);
DECL_QUANT_BLOCKWISE_SEGM_INSTANCE(dtype::bfloat16, dtype::float8_e4m3)
DECL_QUANT_BLOCKWISE_SEGM_INSTANCE(dtype::bfloat16, dtype::float8_e5m2)
DECL_QUANT_BLOCKWISE_SEGM_INSTANCE(dtype::float16, dtype::float8_e4m3)
DECL_QUANT_BLOCKWISE_SEGM_INSTANCE(dtype::float16, dtype::float8_e5m2)
#undef DECL_QUANT_BLOCKWISE_SEGM_INSTANCE

// Weight blockwise FP8 quant: 3D weight [B, M, N], single scalar scale per
// [128, 128] tile. 256 threads/block; vec-8 packed loads; BlockReduce<AbsMax>.
template <typename FType, typename QType>
__launch_bounds__(256) __global__
void quant_fp8_blockwise_for_weight_kernel(const FType *__restrict__ w_ptr,
                                           QType *__restrict__ w_fp8_ptr,
                                           float *__restrict__ w_scales_inv_ptr,
                                           const int64_t M, const int64_t N,
                                           const float fp8_max) {
    constexpr int BLOCK_SIZE = 128;
    constexpr int PACK       = 8;
    constexpr int THREADS_PER_ROW = BLOCK_SIZE / PACK;            // 16
    constexpr int ROWS_PER_ROUND  = 256 / THREADS_PER_ROW;        // 16
    constexpr int ROUNDS          = BLOCK_SIZE / ROWS_PER_ROUND;  // 8

    const int64_t bid     = blockIdx.x;
    const int64_t row_blk = blockIdx.y;
    const int64_t col_blk = blockIdx.z;
    const int     tid     = threadIdx.x;
    const int     pack_idx      = tid % THREADS_PER_ROW;
    const int     load_row_base = tid / THREADS_PER_ROW;

    const int64_t row_start = row_blk * BLOCK_SIZE;
    const int64_t col_start = col_blk * BLOCK_SIZE;
    const int64_t batch_off = bid * M * N;

    float vals[ROUNDS][PACK];
    float amax = 0.f;

    #pragma unroll
    for (int r = 0; r < ROUNDS; ++r) {
        const int local_m = load_row_base + r * ROWS_PER_ROUND;
        const int local_n = pack_idx * PACK;
        const int64_t gm  = row_start + local_m;
        const int64_t gn  = col_start + local_n;
        FType buf[PACK];
        if (gm < M && gn + PACK <= N) {
            load_data<FType, PACK>(w_ptr + batch_off + gm * N + gn, buf);
        } else {
            #pragma unroll
            for (int i = 0; i < PACK; ++i) {
                const int64_t cn = gn + i;
                buf[i] = (gm < M && cn < N) ? w_ptr[batch_off + gm * N + cn]
                                            : static_cast<FType>(0.f);
            }
        }
        #pragma unroll
        for (int i = 0; i < PACK; ++i) {
            const float v = static_cast<float>(buf[i]);
            vals[r][i] = v;
            amax = fmaxf(amax, fabsf(v));
        }
    }

    amax = BlockReduce<AbsMaxOp, float>(amax);
    // Clamp eps matches Triton reference (tl.maximum(w_tile_max, 1e-4)).
    const float scale   = static_cast<float>(fp8_max) / fmaxf(amax, 1e-4f);
    const float clip_lo = -static_cast<float>(fp8_max);
    const float clip_hi =  static_cast<float>(fp8_max);

    if (tid == 0) {
        const int64_t sn = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const int64_t sm = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
        w_scales_inv_ptr[bid * sm * sn + row_blk * sn + col_blk] = 1.0f / scale;
    }

    #pragma unroll
    for (int r = 0; r < ROUNDS; ++r) {
        const int local_m = load_row_base + r * ROWS_PER_ROUND;
        const int local_n = pack_idx * PACK;
        const int64_t gm  = row_start + local_m;
        const int64_t gn  = col_start + local_n;
        QType out[PACK];
        #pragma unroll
        for (int i = 0; i < PACK; ++i) {
            out[i] = static_cast<QType>(fmaxf(fminf(vals[r][i] * scale, clip_hi), clip_lo));
        }
        if (gm < M && gn + PACK <= N) {
            store_data<QType, PACK>(w_fp8_ptr + batch_off + gm * N + gn, out);
        } else if (gm < M) {
            #pragma unroll
            for (int i = 0; i < PACK; ++i) {
                const int64_t cn = gn + i;
                if (cn < N) w_fp8_ptr[batch_off + gm * N + cn] = out[i];
            }
        }
    }
}

template <typename FType, typename QType>
void quantize_blockwise_for_weight_impl(const FType *w, QType *w_fp8, float *w_scales_inv,
                                         const int64_t B, const int64_t M, const int64_t N,
                                         const float fp8_max, hipStream_t stream) {
    constexpr int BLOCK_SIZE = 128;
    const int64_t m_blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int64_t n_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(B, m_blocks, n_blocks);
    quant_fp8_blockwise_for_weight_kernel<FType, QType>
        <<<grid, dim3(256), 0, stream>>>(w, w_fp8, w_scales_inv, M, N, fp8_max);
}

#define DECL_QUANT_BLOCKWISE_FOR_WEIGHT_INSTANCE(FType, QType)                          \
    template void quantize_blockwise_for_weight_impl<FType, QType>(                     \
        const FType *w, QType *w_fp8, float *w_scales_inv,                              \
        const int64_t B, const int64_t M, const int64_t N,                              \
        const float fp8_max, hipStream_t stream);
DECL_QUANT_BLOCKWISE_FOR_WEIGHT_INSTANCE(dtype::bfloat16, dtype::float8_e4m3)
DECL_QUANT_BLOCKWISE_FOR_WEIGHT_INSTANCE(dtype::bfloat16, dtype::float8_e5m2)
DECL_QUANT_BLOCKWISE_FOR_WEIGHT_INSTANCE(dtype::float16, dtype::float8_e4m3)
DECL_QUANT_BLOCKWISE_FOR_WEIGHT_INSTANCE(dtype::float16, dtype::float8_e5m2)
#undef DECL_QUANT_BLOCKWISE_FOR_WEIGHT_INSTANCE

} // namespace primus_turbo
