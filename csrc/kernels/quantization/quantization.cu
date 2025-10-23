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

template <typename FType, typename QType, typename ComputeType, bool PreComputeScale>
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
    template void quantize_rowwise_col_major_impl<FType, QType, float, true>(                      \
        const FType *x, float *scale, float *scale_inv, QType *y, const int64_t batch,             \
        const int64_t m, const int64_t n, hipStream_t stream);                                     \
    template void quantize_rowwise_col_major_impl<FType, QType, float, false>(                     \
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

} // namespace primus_turbo
