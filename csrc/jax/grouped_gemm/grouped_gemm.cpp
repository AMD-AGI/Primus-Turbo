// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/grouped_gemm.h"
#include "../extensions.h"
#include "primus_turbo/arch.h"

namespace primus_turbo::jax {

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

// Grouped GEMM FFI Handler
ffi::Error GroupedGemmFFI(cudaStream_t stream, ffi::AnyBuffer a, ffi::AnyBuffer b,
                          ffi::AnyBuffer group_lens, ffi::AnyBuffer group_offs,
                          ffi::Result<ffi::AnyBuffer> c, bool transA, bool transB, int64_t num_cu) {
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

    // Allocate args workspace on GPU
    const int64_t args_sizes = get_ck_grouped_gemm_args_sizes(group_num);
    void         *args_ptr   = nullptr;
    hipMalloc(&args_ptr, args_sizes);

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
        hipFree(args_ptr);
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "GroupedGemm only supports float16 and bfloat16");
    }

    hipFree(args_ptr);
    return ffi::Error::Success();
}

// Grouped GEMM Variable K FFI Handler
ffi::Error GroupedGemmVariableKFFI(cudaStream_t stream, ffi::AnyBuffer a, ffi::AnyBuffer b,
                                   ffi::AnyBuffer group_lens, ffi::AnyBuffer group_offs,
                                   ffi::Result<ffi::AnyBuffer> c, bool transA, bool transB,
                                   int64_t num_cu) {
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

    // Allocate args workspace on GPU
    const int64_t args_sizes = get_ck_grouped_gemm_args_sizes(group_num);
    void         *args_ptr   = nullptr;
    hipMalloc(&args_ptr, args_sizes);

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
        hipFree(args_ptr);
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "GroupedGemm only supports float16 and bfloat16");
    }

    hipFree(args_ptr);
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
                             ffi::Result<ffi::AnyBuffer> c, bool transA, bool transB,
                             int64_t num_cu, std::string_view granularity) {
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

    // Allocate args workspace on GPU
    const int64_t args_sizes = get_ck_grouped_gemm_fp8_args_sizes(group_num);
    void         *args_ptr   = nullptr;
    hipMalloc(&args_ptr, args_sizes);

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
            hipFree(args_ptr);
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
            hipFree(args_ptr);
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                              "GroupedGemmFP8 output must be float16 or bfloat16");
        }
    } else {
        hipFree(args_ptr);
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "GroupedGemmFP8 only supports fp8 e4m3 and e5m2");
    }

    hipFree(args_ptr);
    return ffi::Error::Success();
}

// Grouped GEMM FP8 Variable K FFI Handler
ffi::Error GroupedGemmFP8VariableKFFI(cudaStream_t stream, ffi::AnyBuffer a, ffi::AnyBuffer b,
                                      ffi::AnyBuffer a_scales, ffi::AnyBuffer b_scales,
                                      ffi::AnyBuffer group_lens, ffi::AnyBuffer group_offs,
                                      ffi::Result<ffi::AnyBuffer> c, bool transA, bool transB,
                                      int64_t num_cu, std::string_view granularity) {
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

    // Allocate args workspace on GPU
    const int64_t args_sizes = get_ck_grouped_gemm_fp8_args_sizes(group_num);
    void         *args_ptr   = nullptr;
    hipMalloc(&args_ptr, args_sizes);

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
            hipFree(args_ptr);
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
            hipFree(args_ptr);
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                              "GroupedGemmFP8 output must be float16 or bfloat16");
        }
    } else {
        hipFree(args_ptr);
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "GroupedGemmFP8 only supports fp8 e4m3 and e5m2");
    }

    hipFree(args_ptr);
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
                                  .Ret<ffi::AnyBuffer>()                    // c
                                  .Attr<bool>("transA")                     // transA
                                  .Attr<bool>("transB")                     // transB
                                  .Attr<int64_t>("num_cu")                  // num_cu
                                  .Attr<std::string_view>("granularity")    // granularity
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
                                  .Ret<ffi::AnyBuffer>()                    // c
                                  .Attr<bool>("transA")                     // transA
                                  .Attr<bool>("transB")                     // transB
                                  .Attr<int64_t>("num_cu")                  // num_cu
                                  .Attr<std::string_view>("granularity")    // granularity
);

} // namespace primus_turbo::jax
