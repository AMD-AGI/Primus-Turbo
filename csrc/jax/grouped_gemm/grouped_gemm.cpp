// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/grouped_gemm.h"
#include "../extensions.h"
#include "primus_turbo/arch.h"
#include "primus_turbo/quantization.h"
#include "primus_turbo/reduce.h"

namespace primus_turbo::jax {

using namespace primus_turbo::dtype;

// Helper function to get FP8 max value
inline float get_float8_max(ffi::DataType dtype) {
    if (dtype == ffi::F8E4M3FN) {
        return 448.0f;
    } else if (dtype == ffi::F8E4M3FNUZ) {
        return 240.0f;
    } else if (dtype == ffi::F8E5M2) {
        return 57344.0f;
    } else if (dtype == ffi::F8E5M2FNUZ) {
        return 57344.0f;
    } else {
        return 1.0f;
    }
}

// Get the number of compute units for grouped GEMM
inline uint32_t get_grouped_gemm_num_cu(cudaStream_t stream, int64_t num_cu) {
    int device_id = 0;
    hipStreamGetDevice(stream, &device_id);
    int32_t cus        = get_multi_processor_count(device_id);
    int32_t num_cu_val = num_cu;
    return num_cu_val <= 0 ? uint32_t(cus) : uint32_t(std::min(num_cu_val, cus));
}

// Helper function to create CK grouped GEMM parameters
template <typename AType, typename BType, typename CType>
inline CKGroupedGemmParams<AType, BType, CType>
make_ck_grouped_gemm_params(void *args_ptr, ffi::AnyBuffer a, ffi::AnyBuffer b,
                            ffi::Result<ffi::AnyBuffer> c, ffi::AnyBuffer group_lens,
                            ffi::AnyBuffer group_offs, bool transA, bool transB, int32_t group_num,
                            int32_t m, int32_t n, int32_t k, cudaStream_t stream, uint32_t num_cu) {
    CKGroupedGemmParams<AType, BType, CType> params;
    params.args_ptr       = args_ptr;
    params.a_ptr          = reinterpret_cast<const AType *>(a.untyped_data());
    params.b_ptr          = reinterpret_cast<const BType *>(b.untyped_data());
    params.c_ptr          = reinterpret_cast<CType *>(c->untyped_data());
    params.group_lens_ptr = reinterpret_cast<const int64_t *>(group_lens.untyped_data());
    params.group_offs_ptr = reinterpret_cast<const int64_t *>(group_offs.untyped_data());
    params.transA         = transA;
    params.transB         = transB;
    params.group_num      = group_num;
    params.m              = m;
    params.n              = n;
    params.k              = k;
    params.stream         = stream;
    params.num_cu         = num_cu;
    return params;
}

// Helper function to create CK grouped GEMM FP8 parameters
template <typename AType, typename BType, typename CType, typename ACCType>
inline CKGroupedGemmFP8Params<AType, BType, CType, ACCType> make_ck_grouped_gemm_fp8_params(
    void *args_ptr, ffi::AnyBuffer a, ffi::AnyBuffer b, ffi::Result<ffi::AnyBuffer> c,
    ffi::AnyBuffer a_scales, ffi::AnyBuffer b_scales, ffi::AnyBuffer group_lens,
    ffi::AnyBuffer group_offs, bool transA, bool transB, int32_t group_num, int32_t m, int32_t n,
    int32_t k, cudaStream_t stream, uint32_t num_cu) {
    CKGroupedGemmFP8Params<AType, BType, CType, ACCType> params;
    params.args_ptr       = args_ptr;
    params.a_ptr          = reinterpret_cast<const AType *>(a.untyped_data());
    params.b_ptr          = reinterpret_cast<const BType *>(b.untyped_data());
    params.c_ptr          = reinterpret_cast<CType *>(c->untyped_data());
    params.aq_ptr         = reinterpret_cast<const ACCType *>(a_scales.untyped_data());
    params.bq_ptr         = reinterpret_cast<const ACCType *>(b_scales.untyped_data());
    params.group_lens_ptr = reinterpret_cast<const int64_t *>(group_lens.untyped_data());
    params.group_offs_ptr = reinterpret_cast<const int64_t *>(group_offs.untyped_data());
    params.transA         = transA;
    params.transB         = transB;
    params.group_num      = group_num;
    params.m              = m;
    params.n              = n;
    params.k              = k;
    params.stream         = stream;
    params.num_cu         = num_cu;
    return params;
}

// Version that accepts raw pointers (for fused implementation)
template <typename AType, typename BType, typename CType, typename ACCType>
inline CKGroupedGemmFP8Params<AType, BType, CType, ACCType>
make_ck_grouped_gemm_fp8_params_direct(void *args_ptr, void *a_ptr, void *b_ptr, void *c_ptr,
                                       const ACCType *a_scales_ptr, const ACCType *b_scales_ptr,
                                       const int64_t *group_lens_ptr, const int64_t *group_offs_ptr,
                                       bool transA, bool transB, int32_t group_num, int32_t m,
                                       int32_t n, int32_t k, cudaStream_t stream, uint32_t num_cu) {
    CKGroupedGemmFP8Params<AType, BType, CType, ACCType> params;
    params.args_ptr       = args_ptr;
    params.a_ptr          = reinterpret_cast<const AType *>(a_ptr);
    params.b_ptr          = reinterpret_cast<const BType *>(b_ptr);
    params.c_ptr          = reinterpret_cast<CType *>(c_ptr);
    params.aq_ptr         = a_scales_ptr;
    params.bq_ptr         = b_scales_ptr;
    params.group_lens_ptr = group_lens_ptr;
    params.group_offs_ptr = group_offs_ptr;
    params.transA         = transA;
    params.transB         = transB;
    params.group_num      = group_num;
    params.m              = m;
    params.n              = n;
    params.k              = k;
    params.stream         = stream;
    params.num_cu         = num_cu;
    return params;
}

// Grouped GEMM FFI Handler
ffi::Error GroupedGemmFFI(cudaStream_t stream, ffi::AnyBuffer a, ffi::AnyBuffer b,
                          ffi::AnyBuffer group_lens, ffi::AnyBuffer group_offs,
                          ffi::AnyBuffer workspace, ffi::Result<ffi::AnyBuffer> c, bool transA,
                          bool transB, int64_t num_cu) {
    // Check input types
    if (a.element_type() != b.element_type()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "a and b dtype mismatch");
    }

    if (group_lens.element_type() != ffi::S64 || group_offs.element_type() != ffi::S64) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "group_lens and group_offs must be int64");
    }

    // Get dimensions
    const int32_t group_num = static_cast<int32_t>(b.dimensions()[0]); // group_num == batch size
    const int32_t m         = transA ? a.dimensions()[1] : a.dimensions()[0];
    const int32_t k         = transA ? a.dimensions()[0] : a.dimensions()[1];
    const int32_t n         = transB ? b.dimensions()[1] : b.dimensions()[2];

    // Use provided workspace buffer
    void *args_ptr = workspace.untyped_data();

    // Get num_cu
    uint32_t num_cu_val = get_grouped_gemm_num_cu(stream, num_cu);

    // Call implementation based on dtype
    if (a.element_type() == ffi::F16) {
        using DataType = ck_tile::half_t;
        auto params    = make_ck_grouped_gemm_params<DataType, DataType, DataType>(
            args_ptr, a, b, c, group_lens, group_offs, transA, transB, group_num, m, n, k, stream,
            num_cu_val);
        ck_grouped_gemm<DataType, DataType, DataType>(params);
    } else if (a.element_type() == ffi::BF16) {
        using DataType = ck_tile::bf16_t;
        auto params    = make_ck_grouped_gemm_params<DataType, DataType, DataType>(
            args_ptr, a, b, c, group_lens, group_offs, transA, transB, group_num, m, n, k, stream,
            num_cu_val);
        ck_grouped_gemm<DataType, DataType, DataType>(params);
    } else {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "GroupedGemm only supports float16 and bfloat16");
    }

    return ffi::Error::Success();
}

// Grouped GEMM Variable K FFI Handler
ffi::Error GroupedGemmVariableKFFI(cudaStream_t stream, ffi::AnyBuffer a, ffi::AnyBuffer b,
                                   ffi::AnyBuffer group_lens, ffi::AnyBuffer group_offs,
                                   ffi::AnyBuffer workspace, ffi::Result<ffi::AnyBuffer> c,
                                   bool transA, bool transB, int64_t num_cu) {
    // Check input types
    if (a.element_type() != b.element_type()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "a and b dtype mismatch");
    }

    if (group_lens.element_type() != ffi::S64 || group_offs.element_type() != ffi::S64) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "group_lens and group_offs must be int64");
    }

    // Only support transA=True, transB=False
    if (!transA || transB) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "grouped_gemm_variable_k only supports transA=True, transB=False");
    }

    // Get dimensions
    // For variable_k with transA=True, transB=False:
    // a: [k, m] (will be transposed), b: [k, n]
    // PyTorch logic: m = transA ? a.size(1) : a.size(0)
    const int32_t group_num = static_cast<int32_t>(group_lens.element_count());
    const int32_t m         = a.dimensions()[1]; // transA=True, so m is a.dim[1]
    const int32_t k         = a.dimensions()[0]; // transA=True, so k is a.dim[0]
    const int32_t n         = b.dimensions()[1]; // transB=False, so n is b.dim[1]

    // Use provided workspace buffer
    void *args_ptr = workspace.untyped_data();

    // Get num_cu
    uint32_t num_cu_val = get_grouped_gemm_num_cu(stream, num_cu);

    // Call implementation based on dtype
    if (a.element_type() == ffi::F16) {
        using DataType = ck_tile::half_t;
        auto params    = make_ck_grouped_gemm_params<DataType, DataType, DataType>(
            args_ptr, a, b, c, group_lens, group_offs, transA, transB, group_num, m, n, k, stream,
            num_cu_val);
        ck_grouped_gemm_variable_k<DataType, DataType, DataType>(params);
    } else if (a.element_type() == ffi::BF16) {
        using DataType = ck_tile::bf16_t;
        auto params    = make_ck_grouped_gemm_params<DataType, DataType, DataType>(
            args_ptr, a, b, c, group_lens, group_offs, transA, transB, group_num, m, n, k, stream,
            num_cu_val);
        ck_grouped_gemm_variable_k<DataType, DataType, DataType>(params);
    } else {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "GroupedGemm only supports float16 and bfloat16");
    }

    return ffi::Error::Success();
}

// Compute group offsets FFI Handler
ffi::Error ComputeGroupOffsFFI(cudaStream_t stream, ffi::AnyBuffer group_lens,
                               ffi::Result<ffi::AnyBuffer> group_offs) {
    const int64_t group_num = group_lens.element_count();

    // Only support int64
    if (group_lens.element_type() != ffi::S64) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "compute_group_offs only supports int64");
    }

    compute_group_offs<int64_t>(group_lens.typed_data<int64_t>(), group_offs->typed_data<int64_t>(),
                                group_num, stream);

    return ffi::Error::Success();
}

// Grouped GEMM FP8 FFI Handler
ffi::Error GroupedGemmFP8FFI(cudaStream_t stream, ffi::AnyBuffer a, ffi::AnyBuffer b,
                             ffi::AnyBuffer a_scales, ffi::AnyBuffer b_scales,
                             ffi::AnyBuffer group_lens, ffi::AnyBuffer group_offs,
                             ffi::AnyBuffer workspace, ffi::Result<ffi::AnyBuffer> c, bool transA,
                             bool transB, int64_t num_cu, std::string_view granularity,
                             std::string_view out_dtype_str) {
    // Check input types
    if (a.element_type() != b.element_type()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "a and b dtype mismatch");
    }

    if (group_lens.element_type() != ffi::S64 || group_offs.element_type() != ffi::S64) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "group_lens and group_offs must be int64");
    }

    // Get dimensions
    const int32_t group_num = static_cast<int32_t>(b.dimensions()[0]);
    const int32_t m         = transA ? a.dimensions()[1] : a.dimensions()[0];
    const int32_t k         = transA ? a.dimensions()[0] : a.dimensions()[1];
    const int32_t n         = transB ? b.dimensions()[1] : b.dimensions()[2];

    // Use provided workspace buffer
    void *args_ptr = workspace.untyped_data();

    // Get num_cu
    uint32_t num_cu_val = get_grouped_gemm_num_cu(stream, num_cu);

    // Call implementation based on dtype
    if (a.element_type() == ffi::F8E4M3FNUZ || a.element_type() == ffi::F8E4M3FN) {
        using AType             = ck_tile::fp8_t;
        using BType             = ck_tile::fp8_t;
        ffi::DataType out_dtype = c->element_type();

        if (out_dtype == ffi::F16) {
            using CType = ck_tile::half_t;
            auto params = make_ck_grouped_gemm_fp8_params<AType, BType, CType, float>(
                args_ptr, a, b, c, a_scales, b_scales, group_lens, group_offs, transA, transB,
                group_num, m, n, k, stream, num_cu_val);
            if (granularity == "TENSORWISE")
                ck_grouped_gemm_fp8<AType, BType, CType, float, ck_tile::QuantType::TensorQuant>(
                    params);
            else
                ck_grouped_gemm_fp8<AType, BType, CType, float, ck_tile::QuantType::RowColQuant>(
                    params);
        } else if (out_dtype == ffi::BF16) {
            using CType = ck_tile::bf16_t;
            auto params = make_ck_grouped_gemm_fp8_params<AType, BType, CType, float>(
                args_ptr, a, b, c, a_scales, b_scales, group_lens, group_offs, transA, transB,
                group_num, m, n, k, stream, num_cu_val);
            if (granularity == "TENSORWISE")
                ck_grouped_gemm_fp8<AType, BType, CType, float, ck_tile::QuantType::TensorQuant>(
                    params);
            else
                ck_grouped_gemm_fp8<AType, BType, CType, float, ck_tile::QuantType::RowColQuant>(
                    params);
        } else {
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                              "GroupedGemmFP8 output must be float16 or bfloat16");
        }
    } else if (a.element_type() == ffi::F8E5M2FNUZ || a.element_type() == ffi::F8E5M2) {
        using AType             = ck_tile::bf8_t;
        using BType             = ck_tile::bf8_t;
        ffi::DataType out_dtype = c->element_type();

        if (out_dtype == ffi::F16) {
            using CType = ck_tile::half_t;
            auto params = make_ck_grouped_gemm_fp8_params<AType, BType, CType, float>(
                args_ptr, a, b, c, a_scales, b_scales, group_lens, group_offs, transA, transB,
                group_num, m, n, k, stream, num_cu_val);
            if (granularity == "TENSORWISE")
                ck_grouped_gemm_fp8<AType, BType, CType, float, ck_tile::QuantType::TensorQuant>(
                    params);
            else
                ck_grouped_gemm_fp8<AType, BType, CType, float, ck_tile::QuantType::RowColQuant>(
                    params);
        } else if (out_dtype == ffi::BF16) {
            using CType = ck_tile::bf16_t;
            auto params = make_ck_grouped_gemm_fp8_params<AType, BType, CType, float>(
                args_ptr, a, b, c, a_scales, b_scales, group_lens, group_offs, transA, transB,
                group_num, m, n, k, stream, num_cu_val);
            if (granularity == "TENSORWISE")
                ck_grouped_gemm_fp8<AType, BType, CType, float, ck_tile::QuantType::TensorQuant>(
                    params);
            else
                ck_grouped_gemm_fp8<AType, BType, CType, float, ck_tile::QuantType::RowColQuant>(
                    params);
        } else {
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                              "GroupedGemmFP8 output must be float16 or bfloat16");
        }
    } else {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "GroupedGemmFP8 only supports fp8 e4m3 and e5m2");
    }

    return ffi::Error::Success();
}

// Grouped GEMM FP8 Variable K FFI Handler
ffi::Error GroupedGemmFP8VariableKFFI(cudaStream_t stream, ffi::AnyBuffer a, ffi::AnyBuffer b,
                                      ffi::AnyBuffer a_scales, ffi::AnyBuffer b_scales,
                                      ffi::AnyBuffer group_lens, ffi::AnyBuffer group_offs,
                                      ffi::AnyBuffer workspace, ffi::Result<ffi::AnyBuffer> c,
                                      bool transA, bool transB, int64_t num_cu,
                                      std::string_view granularity,
                                      std::string_view out_dtype_str) {
    // Check input types
    if (a.element_type() != b.element_type()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "a and b dtype mismatch");
    }

    if (group_lens.element_type() != ffi::S64 || group_offs.element_type() != ffi::S64) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "group_lens and group_offs must be int64");
    }

    // Only support transA=True, transB=False
    if (!transA || transB) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "grouped_gemm_fp8_variable_k only supports transA=True, transB=False");
    }

    // Get dimensions
    const int32_t group_num = static_cast<int32_t>(group_lens.element_count());
    const int32_t m         = a.dimensions()[1]; // transA=True
    const int32_t k         = a.dimensions()[0];
    const int32_t n         = b.dimensions()[1]; // transB=False

    // Use provided workspace buffer
    void *args_ptr = workspace.untyped_data();

    // Get num_cu
    uint32_t num_cu_val = get_grouped_gemm_num_cu(stream, num_cu);

    // Scales are passed in already expanded from Python layer
    // No need for hipMalloc/hipMemcpy/hipFree here

    // Call implementation based on dtype
    if (a.element_type() == ffi::F8E4M3FNUZ || a.element_type() == ffi::F8E4M3FN) {
        using AType             = ck_tile::fp8_t;
        using BType             = ck_tile::fp8_t;
        ffi::DataType out_dtype = c->element_type();

        if (out_dtype == ffi::F16) {
            using CType = ck_tile::half_t;
            auto params = make_ck_grouped_gemm_fp8_params<AType, BType, CType, float>(
                args_ptr, a, b, c, a_scales, b_scales, group_lens, group_offs, transA, transB,
                group_num, m, n, k, stream, num_cu_val);
            if (granularity == "TENSORWISE")
                ck_grouped_gemm_fp8_variable_k<AType, BType, CType, float,
                                               ck_tile::QuantType::TensorQuant>(params);
            else
                ck_grouped_gemm_fp8_variable_k<AType, BType, CType, float,
                                               ck_tile::QuantType::RowColQuant>(params);
        } else if (out_dtype == ffi::BF16) {
            using CType = ck_tile::bf16_t;
            auto params = make_ck_grouped_gemm_fp8_params<AType, BType, CType, float>(
                args_ptr, a, b, c, a_scales, b_scales, group_lens, group_offs, transA, transB,
                group_num, m, n, k, stream, num_cu_val);
            if (granularity == "TENSORWISE")
                ck_grouped_gemm_fp8_variable_k<AType, BType, CType, float,
                                               ck_tile::QuantType::TensorQuant>(params);
            else
                ck_grouped_gemm_fp8_variable_k<AType, BType, CType, float,
                                               ck_tile::QuantType::RowColQuant>(params);
        } else {
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                              "GroupedGemmFP8 output must be float16 or bfloat16");
        }
    } else if (a.element_type() == ffi::F8E5M2FNUZ || a.element_type() == ffi::F8E5M2) {
        using AType             = ck_tile::bf8_t;
        using BType             = ck_tile::bf8_t;
        ffi::DataType out_dtype = c->element_type();

        if (out_dtype == ffi::F16) {
            using CType = ck_tile::half_t;
            auto params = make_ck_grouped_gemm_fp8_params<AType, BType, CType, float>(
                args_ptr, a, b, c, a_scales, b_scales, group_lens, group_offs, transA, transB,
                group_num, m, n, k, stream, num_cu_val);
            if (granularity == "TENSORWISE")
                ck_grouped_gemm_fp8_variable_k<AType, BType, CType, float,
                                               ck_tile::QuantType::TensorQuant>(params);
            else
                ck_grouped_gemm_fp8_variable_k<AType, BType, CType, float,
                                               ck_tile::QuantType::RowColQuant>(params);
        } else if (out_dtype == ffi::BF16) {
            using CType = ck_tile::bf16_t;
            auto params = make_ck_grouped_gemm_fp8_params<AType, BType, CType, float>(
                args_ptr, a, b, c, a_scales, b_scales, group_lens, group_offs, transA, transB,
                group_num, m, n, k, stream, num_cu_val);
            if (granularity == "TENSORWISE")
                ck_grouped_gemm_fp8_variable_k<AType, BType, CType, float,
                                               ck_tile::QuantType::TensorQuant>(params);
            else
                ck_grouped_gemm_fp8_variable_k<AType, BType, CType, float,
                                               ck_tile::QuantType::RowColQuant>(params);
        } else {
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                              "GroupedGemmFP8 output must be float16 or bfloat16");
        }
    } else {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "GroupedGemmFP8 only supports fp8 e4m3 and e5m2");
    }

    return ffi::Error::Success();
}

// Register FFI handlers
XLA_FFI_DEFINE_HANDLER_SYMBOL(GroupedGemmHandler, GroupedGemmFFI,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
                                  .Arg<ffi::AnyBuffer>()                    // a
                                  .Arg<ffi::AnyBuffer>()                    // b
                                  .Arg<ffi::AnyBuffer>()                    // group_lens
                                  .Arg<ffi::AnyBuffer>()                    // group_offs
                                  .Arg<ffi::AnyBuffer>()                    // workspace
                                  .Ret<ffi::AnyBuffer>()                    // c
                                  .Attr<bool>("transA")                     // transA
                                  .Attr<bool>("transB")                     // transB
                                  .Attr<int64_t>("num_cu")                  // num_cu
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(ComputeGroupOffsHandler, ComputeGroupOffsFFI,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
                                  .Arg<ffi::AnyBuffer>()                    // group_lens
                                  .Ret<ffi::AnyBuffer>()                    // group_offs
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(GroupedGemmVariableKHandler, GroupedGemmVariableKFFI,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
                                  .Arg<ffi::AnyBuffer>()                    // a
                                  .Arg<ffi::AnyBuffer>()                    // b
                                  .Arg<ffi::AnyBuffer>()                    // group_lens
                                  .Arg<ffi::AnyBuffer>()                    // group_offs
                                  .Arg<ffi::AnyBuffer>()                    // workspace
                                  .Ret<ffi::AnyBuffer>()                    // c
                                  .Attr<bool>("transA")                     // transA
                                  .Attr<bool>("transB")                     // transB
                                  .Attr<int64_t>("num_cu")                  // num_cu
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(GroupedGemmFP8Handler, GroupedGemmFP8FFI,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
                                  .Arg<ffi::AnyBuffer>()                    // a
                                  .Arg<ffi::AnyBuffer>()                    // b
                                  .Arg<ffi::AnyBuffer>()                    // a_scales
                                  .Arg<ffi::AnyBuffer>()                    // b_scales
                                  .Arg<ffi::AnyBuffer>()                    // group_lens
                                  .Arg<ffi::AnyBuffer>()                    // group_offs
                                  .Arg<ffi::AnyBuffer>()                    // workspace
                                  .Ret<ffi::AnyBuffer>()                    // c
                                  .Attr<bool>("transA")                     // transA
                                  .Attr<bool>("transB")                     // transB
                                  .Attr<int64_t>("num_cu")                  // num_cu
                                  .Attr<std::string_view>("granularity")    // granularity
                                  .Attr<std::string_view>("out_dtype_str")  // out_dtype_str
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(GroupedGemmFP8VariableKHandler, GroupedGemmFP8VariableKFFI,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
                                  .Arg<ffi::AnyBuffer>()                    // a
                                  .Arg<ffi::AnyBuffer>()                    // b
                                  .Arg<ffi::AnyBuffer>()                    // a_scales
                                  .Arg<ffi::AnyBuffer>()                    // b_scales
                                  .Arg<ffi::AnyBuffer>()                    // group_lens
                                  .Arg<ffi::AnyBuffer>()                    // group_offs
                                  .Arg<ffi::AnyBuffer>()                    // workspace
                                  .Ret<ffi::AnyBuffer>()                    // c
                                  .Attr<bool>("transA")                     // transA
                                  .Attr<bool>("transB")                     // transB
                                  .Attr<int64_t>("num_cu")                  // num_cu
                                  .Attr<std::string_view>("granularity")    // granularity
                                  .Attr<std::string_view>("out_dtype_str")  // out_dtype_str
);

// ============================================================================
// Fused Grouped GEMM FP8 (quantize + gemm in one FFI call)
// ============================================================================

// Fused TENSORWISE version: accepts BF16/FP16 inputs, quantizes them, then runs grouped_gemm
// This reduces FFI overhead from 3 calls (quant_a + quant_b + gemm) to 1 call
ffi::Error GroupedGemmFP8FusedTensorwiseFFI(cudaStream_t stream, ffi::AnyBuffer a_fp16,
                                            ffi::AnyBuffer b_fp16, ffi::AnyBuffer group_lens,
                                            ffi::AnyBuffer group_offs, ffi::AnyBuffer workspace,
                                            ffi::Result<ffi::AnyBuffer> c, bool transA, bool transB,
                                            int64_t num_cu, std::string_view fp8_dtype_str,
                                            std::string_view out_dtype_str) {

    // Get dimensions
    // b_fp16 shape: (B, N, K) if transB=True, or (B, K, N) if transB=False
    const int32_t group_num = static_cast<int32_t>(b_fp16.dimensions()[0]);
    const int32_t k =
        static_cast<int32_t>(transA ? a_fp16.dimensions()[0] : a_fp16.dimensions()[1]);
    // When transB=True: b is (B, N, K), so n = b.dimensions()[1] (N)
    // When transB=False: b is (B, K, N), so n = b.dimensions()[2] (N)
    const int32_t n =
        static_cast<int32_t>(transB ? b_fp16.dimensions()[1] : b_fp16.dimensions()[2]);
    int64_t m_total = transA ? a_fp16.dimensions()[1] : a_fp16.dimensions()[0];
    // For regular grouped_gemm, m is per-group (all groups have same m)
    const int32_t m = static_cast<int32_t>(m_total / group_num);

    int64_t a_size = 1;
    for (auto dim : a_fp16.dimensions())
        a_size *= dim;
    int64_t b_size = 1;
    for (auto dim : b_fp16.dimensions())
        b_size *= dim;

    // Workspace layout:
    // [quant_a_ws][quant_b_ws][a_fp8][b_fp8][a_scale][b_scale][gemm_args]
    char   *ws_base = reinterpret_cast<char *>(workspace.untyped_data());
    int64_t offset  = 0;

    // Quantization workspaces (use correct size calculation)
    int64_t       reduce_ws_a_size = get_reduce_row_workspace_sizes<float>(1, a_size);
    int64_t       reduce_ws_b_size = get_reduce_row_workspace_sizes<float>(1, b_size);
    const int64_t quant_ws_a_size  = 256 + ((reduce_ws_a_size + 255) / 256) * 256 + 256 + 256;
    const int64_t quant_ws_b_size  = 256 + ((reduce_ws_b_size + 255) / 256) * 256 + 256 + 256;
    void         *quant_ws_a       = ws_base + offset;
    offset += quant_ws_a_size;
    void *quant_ws_b = ws_base + offset;
    offset += quant_ws_b_size;

    // FP8 buffers (aligned)
    const int64_t a_fp8_size = ((a_size + 255) / 256) * 256;
    const int64_t b_fp8_size = ((b_size + 255) / 256) * 256;
    void         *a_fp8_ptr  = ws_base + offset;
    offset += a_fp8_size;
    void *b_fp8_ptr = ws_base + offset;
    offset += b_fp8_size;

    // Scale buffers (256 bytes each, aligned)
    float *a_scale_inv_ptr = reinterpret_cast<float *>(ws_base + offset);
    offset += 256;
    float *b_scale_inv_ptr = reinterpret_cast<float *>(ws_base + offset);
    offset += 256;

    // GEMM args workspace
    void *gemm_args_ptr = ws_base + offset;

    auto          input_dtype = a_fp16.element_type();
    ffi::DataType fp8_dtype;
    if (fp8_dtype_str == "e4m3" || fp8_dtype_str == "float8_e4m3fn") {
        fp8_dtype = ffi::F8E4M3FN;
    } else {
        fp8_dtype = ffi::F8E5M2;
    }
    float fp8_max = (fp8_dtype == ffi::F8E4M3FN) ? 448.0f : 57344.0f;

    // Step 1: Quantize a (TENSORWISE)
    {
        int64_t reduce_ws_a_size = get_reduce_row_workspace_sizes<float>(1, a_size);
        char   *ws_char          = reinterpret_cast<char *>(quant_ws_a);
        float  *amax_ptr         = reinterpret_cast<float *>(ws_char);
        void   *reduce_ws_ptr    = ws_char + 256;
        float  *scale_ptr =
            reinterpret_cast<float *>(ws_char + 256 + ((reduce_ws_a_size + 255) / 256) * 256);

        // Reduce to get amax
        if (input_dtype == ffi::F32) {
            reduce_row<float, float, float>(PrimusTurboReduceOp::REDUCE_ABS_MAX,
                                            const_cast<float *>(a_fp16.typed_data<float>()),
                                            amax_ptr, 1, a_size, reduce_ws_a_size, reduce_ws_ptr,
                                            stream);
        } else if (input_dtype == ffi::F16) {
            reduce_row<float16, float, float>(PrimusTurboReduceOp::REDUCE_ABS_MAX,
                                              const_cast<float16 *>(a_fp16.typed_data<float16>()),
                                              amax_ptr, 1, a_size, reduce_ws_a_size, reduce_ws_ptr,
                                              stream);
        } else if (input_dtype == ffi::BF16) {
            reduce_row<bfloat16, float, float>(
                PrimusTurboReduceOp::REDUCE_ABS_MAX,
                const_cast<bfloat16 *>(a_fp16.typed_data<bfloat16>()), amax_ptr, 1, a_size,
                reduce_ws_a_size, reduce_ws_ptr, stream);
        }

        // Compute scale and scale_inv
        compute_scale_from_amax<float>(amax_ptr, fp8_max, scale_ptr, a_scale_inv_ptr, 1, stream);

        // Quantize
        if (fp8_dtype == ffi::F8E4M3FN) {
            if (input_dtype == ffi::F16) {
                quantize_tensorwise_impl<float16, float8_e4m3_t, float>(
                    a_fp16.typed_data<float16>(), scale_ptr,
                    reinterpret_cast<float8_e4m3_t *>(a_fp8_ptr), a_size, stream);
            } else if (input_dtype == ffi::BF16) {
                quantize_tensorwise_impl<bfloat16, float8_e4m3_t, float>(
                    a_fp16.typed_data<bfloat16>(), scale_ptr,
                    reinterpret_cast<float8_e4m3_t *>(a_fp8_ptr), a_size, stream);
            }
        } else { // E5M2
            if (input_dtype == ffi::F16) {
                quantize_tensorwise_impl<float16, float8_e5m2_t, float>(
                    a_fp16.typed_data<float16>(), scale_ptr,
                    reinterpret_cast<float8_e5m2_t *>(a_fp8_ptr), a_size, stream);
            } else if (input_dtype == ffi::BF16) {
                quantize_tensorwise_impl<bfloat16, float8_e5m2_t, float>(
                    a_fp16.typed_data<bfloat16>(), scale_ptr,
                    reinterpret_cast<float8_e5m2_t *>(a_fp8_ptr), a_size, stream);
            }
        }
    }

    // Step 2: Quantize b (TENSORWISE)
    {
        int64_t reduce_ws_b_size = get_reduce_row_workspace_sizes<float>(1, b_size);
        char   *ws_char          = reinterpret_cast<char *>(quant_ws_b);
        float  *amax_ptr         = reinterpret_cast<float *>(ws_char);
        void   *reduce_ws_ptr    = ws_char + 256;
        float  *scale_ptr =
            reinterpret_cast<float *>(ws_char + 256 + ((reduce_ws_b_size + 255) / 256) * 256);

        // Reduce to get amax
        if (input_dtype == ffi::F32) {
            reduce_row<float, float, float>(PrimusTurboReduceOp::REDUCE_ABS_MAX,
                                            const_cast<float *>(b_fp16.typed_data<float>()),
                                            amax_ptr, 1, b_size, reduce_ws_b_size, reduce_ws_ptr,
                                            stream);
        } else if (input_dtype == ffi::F16) {
            reduce_row<float16, float, float>(PrimusTurboReduceOp::REDUCE_ABS_MAX,
                                              const_cast<float16 *>(b_fp16.typed_data<float16>()),
                                              amax_ptr, 1, b_size, reduce_ws_b_size, reduce_ws_ptr,
                                              stream);
        } else if (input_dtype == ffi::BF16) {
            reduce_row<bfloat16, float, float>(
                PrimusTurboReduceOp::REDUCE_ABS_MAX,
                const_cast<bfloat16 *>(b_fp16.typed_data<bfloat16>()), amax_ptr, 1, b_size,
                reduce_ws_b_size, reduce_ws_ptr, stream);
        }

        // Compute scale and scale_inv
        compute_scale_from_amax<float>(amax_ptr, fp8_max, scale_ptr, b_scale_inv_ptr, 1, stream);

        // Quantize
        if (fp8_dtype == ffi::F8E4M3FN) {
            if (input_dtype == ffi::F16) {
                quantize_tensorwise_impl<float16, float8_e4m3_t, float>(
                    b_fp16.typed_data<float16>(), scale_ptr,
                    reinterpret_cast<float8_e4m3_t *>(b_fp8_ptr), b_size, stream);
            } else if (input_dtype == ffi::BF16) {
                quantize_tensorwise_impl<bfloat16, float8_e4m3_t, float>(
                    b_fp16.typed_data<bfloat16>(), scale_ptr,
                    reinterpret_cast<float8_e4m3_t *>(b_fp8_ptr), b_size, stream);
            }
        } else { // E5M2
            if (input_dtype == ffi::F16) {
                quantize_tensorwise_impl<float16, float8_e5m2_t, float>(
                    b_fp16.typed_data<float16>(), scale_ptr,
                    reinterpret_cast<float8_e5m2_t *>(b_fp8_ptr), b_size, stream);
            } else if (input_dtype == ffi::BF16) {
                quantize_tensorwise_impl<bfloat16, float8_e5m2_t, float>(
                    b_fp16.typed_data<bfloat16>(), scale_ptr,
                    reinterpret_cast<float8_e5m2_t *>(b_fp8_ptr), b_size, stream);
            }
        }
    }

    // Synchronize to ensure quantization kernels complete before GEMM
    hipStreamSynchronize(stream);

    // Step 3: Call grouped_gemm_fp8 kernel
    // Get num_cu
    uint32_t num_cu_val = get_grouped_gemm_num_cu(stream, num_cu);

    // Determine FP8 data type for CK kernel
    if (fp8_dtype == ffi::F8E4M3FN) {
        using AType = ck_tile::fp8_t;
        using BType = ck_tile::fp8_t;

        ffi::DataType out_dtype_enum = (out_dtype_str == "float16") ? ffi::F16 : ffi::BF16;

        if (out_dtype_enum == ffi::F16) {
            using CType = ck_tile::half_t;
            auto params = make_ck_grouped_gemm_fp8_params_direct<AType, BType, CType, float>(
                gemm_args_ptr, a_fp8_ptr, b_fp8_ptr, c->untyped_data(), a_scale_inv_ptr,
                b_scale_inv_ptr, group_lens.typed_data<int64_t>(), group_offs.typed_data<int64_t>(),
                transA, transB, group_num, m, n, k, stream, num_cu_val);
            ck_grouped_gemm_fp8<AType, BType, CType, float, ck_tile::QuantType::TensorQuant>(
                params);
        } else {
            using CType = ck_tile::bf16_t;
            auto params = make_ck_grouped_gemm_fp8_params_direct<AType, BType, CType, float>(
                gemm_args_ptr, a_fp8_ptr, b_fp8_ptr, c->untyped_data(), a_scale_inv_ptr,
                b_scale_inv_ptr, group_lens.typed_data<int64_t>(), group_offs.typed_data<int64_t>(),
                transA, transB, group_num, m, n, k, stream, num_cu_val);
            ck_grouped_gemm_fp8<AType, BType, CType, float, ck_tile::QuantType::TensorQuant>(
                params);
        }
    } else { // E5M2
        using AType = ck_tile::bf8_t;
        using BType = ck_tile::bf8_t;

        ffi::DataType out_dtype_enum = (out_dtype_str == "float16") ? ffi::F16 : ffi::BF16;

        if (out_dtype_enum == ffi::F16) {
            using CType = ck_tile::half_t;
            auto params = make_ck_grouped_gemm_fp8_params_direct<AType, BType, CType, float>(
                gemm_args_ptr, a_fp8_ptr, b_fp8_ptr, c->untyped_data(), a_scale_inv_ptr,
                b_scale_inv_ptr, group_lens.typed_data<int64_t>(), group_offs.typed_data<int64_t>(),
                transA, transB, group_num, m, n, k, stream, num_cu_val);
            ck_grouped_gemm_fp8<AType, BType, CType, float, ck_tile::QuantType::TensorQuant>(
                params);
        } else {
            using CType = ck_tile::bf16_t;
            auto params = make_ck_grouped_gemm_fp8_params_direct<AType, BType, CType, float>(
                gemm_args_ptr, a_fp8_ptr, b_fp8_ptr, c->untyped_data(), a_scale_inv_ptr,
                b_scale_inv_ptr, group_lens.typed_data<int64_t>(), group_offs.typed_data<int64_t>(),
                transA, transB, group_num, m, n, k, stream, num_cu_val);
            ck_grouped_gemm_fp8<AType, BType, CType, float, ck_tile::QuantType::TensorQuant>(
                params);
        }
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GroupedGemmFP8FusedTensorwiseHandler,
                              GroupedGemmFP8FusedTensorwiseFFI,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
                                  .Arg<ffi::AnyBuffer>()                    // a_fp16
                                  .Arg<ffi::AnyBuffer>()                    // b_fp16
                                  .Arg<ffi::AnyBuffer>()                    // group_lens
                                  .Arg<ffi::AnyBuffer>()                    // group_offs
                                  .Arg<ffi::AnyBuffer>()                    // workspace
                                  .Ret<ffi::AnyBuffer>()                    // c
                                  .Attr<bool>("transA")                     // transA
                                  .Attr<bool>("transB")                     // transB
                                  .Attr<int64_t>("num_cu")                  // num_cu
                                  .Attr<std::string_view>("fp8_dtype_str")  // fp8_dtype_str
                                  .Attr<std::string_view>("out_dtype_str")  // out_dtype_str
);

} // namespace primus_turbo::jax
