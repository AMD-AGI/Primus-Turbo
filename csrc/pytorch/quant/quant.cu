// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "../extensions.h"
#include "../utils.h"

#include "../../kernels/pointwise/vectorized_pointwise.h"

namespace primus_turbo::pytorch {

using namespace primus_turbo::dtype;

struct QuantizeParam {
    const float32 *scale;
};

__device__ inline float32 quantize_func(float32 value, const QuantizeParam &param) {
    return value * (*(param.scale));
}

struct DequantizeParam {
    const float32 *scale_inv;
};

__device__ inline float32 dequantize_func(float32 value, const DequantizeParam &param) {
    return value * (*(param.scale_inv));
}

inline bool is_torch_fp8(const at::ScalarType dtype) {
    return dtype == at::kFloat8_e4m3fn || dtype == at::kFloat8_e4m3fnuz ||
           dtype == at::kFloat8_e5m2 || dtype == at::kFloat8_e5m2fnuz;
}

at::Tensor fp8_quantize(const at::Tensor input, const at::Tensor scale,
                        const at::ScalarType dest_dtype) {
    PRIMUS_TURBO_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf ||
                       input.scalar_type() == at::kFloat);
    PRIMUS_TURBO_CHECK(scale.scalar_type() == at::kFloat);
    PRIMUS_TURBO_CHECK(scale.ndimension() == 1);
    PRIMUS_TURBO_CHECK(is_torch_fp8(dest_dtype));

    auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    auto device = input.device();

    at::Tensor output = torch::empty_like(input, torch::dtype(dest_dtype).device(device));

    TORCH_TYPE_SWITCH_INPUT(
        input.scalar_type(), IType,
        TORCH_TYPE_SWITCH_FP8ONLY(
            output.scalar_type(), OType, constexpr int nvec = 32 / sizeof(IType); QuantizeParam p;
            p.scale = reinterpret_cast<const float32 *>(scale.data_ptr());
            VectorizedUnaryKernelLauncher<nvec, QuantizeParam, quantize_func>(
                reinterpret_cast<const IType *>(input.data_ptr()),
                reinterpret_cast<OType *>(output.data_ptr()), input.numel(), p,
                stream);); // NOLINT(*)
    );

    return output;
}

at::Tensor fp8_dequantize(const at::Tensor input, const at::Tensor scale_inv,
                          const at::ScalarType dest_dtype) {
    PRIMUS_TURBO_CHECK(is_torch_fp8(input.scalar_type()));
    PRIMUS_TURBO_CHECK(scale_inv.scalar_type() == at::kFloat);
    PRIMUS_TURBO_CHECK(scale_inv.ndimension() == 1);
    PRIMUS_TURBO_CHECK(dest_dtype == at::kBFloat16 || dest_dtype == at::kHalf ||
                       dest_dtype == at::kFloat);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();

    at::Tensor output = torch::empty_like(input, torch::dtype(dest_dtype).device(device));

    TORCH_TYPE_SWITCH_FP8ONLY(
        input.scalar_type(), IType,
        TORCH_TYPE_SWITCH_INPUT(
            output.scalar_type(), OType, constexpr int nvec = 32 / sizeof(OType); DequantizeParam p;
            p.scale_inv = reinterpret_cast<const float32 *>(scale_inv.data_ptr());
            VectorizedUnaryKernelLauncher<nvec, DequantizeParam, dequantize_func>(
                reinterpret_cast<const IType *>(input.data_ptr()),
                reinterpret_cast<OType *>(output.data_ptr()), input.numel(), p,
                stream);); // NOLINT(*)
    );

    return output;
}

} // namespace primus_turbo::pytorch
