// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/elementwise/unary_kernel_template.cuh"
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

template <typename FType, typename QType>
void quantize_tensorwise_impl(const FType *x, const float *scale, QType *y, const int64_t n,
                              hipStream_t stream) {
    using ComputeType = float;
    ComputeType qmax  = static_cast<ComputeType>(std::numeric_limits<QType>::max());
    ComputeType qmin  = static_cast<ComputeType>(std::numeric_limits<QType>::lowest());

    QuantTensorwiseScalePtrOp<ComputeType> op{
        {},
        reinterpret_cast<const ComputeType *>(scale),
        static_cast<ComputeType>(std::numeric_limits<QType>::lowest()),
        static_cast<ComputeType>(std::numeric_limits<QType>::max())};
    // printf("quantize_tensorwise_impl: qmax=%f, qmin=%f\n", qmax, qmin);

    const int32_t BLOCK_SIZE = 512;
    const int32_t GRID_SIZE  = DIVUP<int32_t>(n, BLOCK_SIZE);

    unary_kernel<FType, QType, QuantTensorwiseScalePtrOp<ComputeType>>
        <<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(x, y, n, op);
}

#define DECL_QUANT_TENSORWISE_INSTANCE(FType, QType)                                               \
    template void quantize_tensorwise_impl<FType, QType>(                                          \
        const FType *x, const float *scale, QType *y, const int64_t n, hipStream_t stream);

DECL_QUANT_TENSORWISE_INSTANCE(dtype::float16, dtype::float8_e4m3)
DECL_QUANT_TENSORWISE_INSTANCE(dtype::float16, dtype::float8_e5m2)
DECL_QUANT_TENSORWISE_INSTANCE(dtype::bfloat16, dtype::float8_e4m3)
DECL_QUANT_TENSORWISE_INSTANCE(dtype::bfloat16, dtype::float8_e5m2)
DECL_QUANT_TENSORWISE_INSTANCE(dtype::float32, dtype::float8_e4m3)
DECL_QUANT_TENSORWISE_INSTANCE(dtype::float32, dtype::float8_e5m2)

#undef DECL_QUANT_TENSORWISE_INSTANCE

} // namespace primus_turbo
