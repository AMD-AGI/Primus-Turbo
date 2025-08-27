// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/grouped_gemm.h"
#include "../extensions.h"
#include "../type_traits.h"
#include "primus_turbo/arch.h"

namespace primus_turbo::pytorch {

at::Tensor grouped_gemm_compute_offs(at::Tensor &group_lens) {
    // Check input tensor type
    PRIMUS_TURBO_CHECK(group_lens.scalar_type() == at::kLong,
                       "group_lens must be of type Long (int64_t)");

    // Create output tensor with one more element than input
    at::Tensor group_offs = at::empty({group_lens.numel() + 1}, group_lens.options());

    // Get current CUDA stream
    auto stream = at::cuda::getCurrentCUDAStream();

    // Call the CUDA implementation to compute group offsets
    compute_group_offs<int64_t>(reinterpret_cast<const int64_t *>(group_lens.data_ptr()),
                                reinterpret_cast<int64_t *>(group_offs.data_ptr()),
                                group_lens.numel(), stream);

    return group_offs;
}

uint32_t get_grouped_gemm_num_cu(c10::optional<int64_t> num_cu) {
    auto    stream     = at::cuda::getCurrentCUDAStream();
    int32_t cus        = get_multi_processor_count(stream.device_index());
    int32_t num_cu_val = num_cu.has_value() ? num_cu.value() : -1;
    return num_cu_val <= 0 ? uint32_t(cus) : uint32_t(std::min(num_cu_val, cus));
}

at::Tensor grouped_gemm(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                        at::Tensor &group_offs, const bool transA, const bool transB,
                        c10::optional<int64_t> num_cu) {
    // TODO:
    auto out_dtype = a.scalar_type();

    // Check
    PRIMUS_TURBO_CHECK(is_16bit_floating_point_dtype(a.scalar_type()));
    PRIMUS_TURBO_CHECK(is_16bit_floating_point_dtype(b.scalar_type()));
    PRIMUS_TURBO_CHECK(group_lens.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(group_offs.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(a.scalar_type() == b.scalar_type(), "a and b dtype mismatch");

    // Alloc args workspace
    const int64_t args_sizes = get_ck_grouped_gemm_args_sizes(group_lens.numel());
    at::Tensor    args_tensor =
        at::empty({args_sizes}, at::TensorOptions().dtype(at::kByte).device(group_lens.device()));

    // Determine output tensor size based on transA and transB
    const int32_t bs = b.size(0);
    const int32_t m  = transA ? a.size(1) : a.size(0);
    const int32_t n  = transB ? b.size(1) : b.size(2);
    const int32_t k  = transA ? a.size(0) : a.size(1);
    at::Tensor    c  = at::empty({m, n}, at::dtype(out_dtype).device(at::kCUDA));

    auto stream = at::cuda::getCurrentCUDAStream();
    if (a.dtype() == at::kHalf) {
        using AType = typename TorchToCKTileType<at::kHalf>::type;
        using BType = AType;
        using CType = AType;
        ck_grouped_gemm<AType, BType, CType>(
            args_tensor.data_ptr(), reinterpret_cast<const AType *>(a.data_ptr()),
            reinterpret_cast<const BType *>(b.data_ptr()), reinterpret_cast<CType *>(c.data_ptr()),
            reinterpret_cast<const int64_t *>(group_lens.data_ptr()),
            reinterpret_cast<const int64_t *>(group_offs.data_ptr()), transA, transB, bs, m, n, k,
            stream, get_grouped_gemm_num_cu(num_cu));
    } else if (a.dtype() == at::kBFloat16) {
        using AType = typename TorchToCKTileType<at::kBFloat16>::type;
        using BType = AType;
        using CType = AType;
        ck_grouped_gemm<AType, BType, CType>(
            args_tensor.data_ptr(), reinterpret_cast<const AType *>(a.data_ptr()),
            reinterpret_cast<const BType *>(b.data_ptr()), reinterpret_cast<CType *>(c.data_ptr()),
            reinterpret_cast<const int64_t *>(group_lens.data_ptr()),
            reinterpret_cast<const int64_t *>(group_offs.data_ptr()), transA, transB, bs, m, n, k,
            stream, get_grouped_gemm_num_cu(num_cu));
    } else {
        PRIMUS_TURBO_CHECK(false, "GroupedGemm only support float16 and bfloat16");
    }
    return c;
}

at::Tensor grouped_gemm_fp8(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                            at::Tensor &group_offs, const bool transA, const bool transB,
                            at::ScalarType out_dtype, c10::optional<int64_t> num_cu) {

    // Check
    PRIMUS_TURBO_CHECK(is_8bit_floating_point_dtype(a.scalar_type()));
    PRIMUS_TURBO_CHECK(is_8bit_floating_point_dtype(b.scalar_type()));
    PRIMUS_TURBO_CHECK(group_lens.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(group_offs.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(a.scalar_type() == b.scalar_type(), "a and b dtype mismatch");
    PRIMUS_TURBO_CHECK(out_dtype == at::kBFloat16 || out_dtype == at::kHalf,
                       "out_dtype must be kBFloat16 or kHalf");

    // Alloc args workspace
    const int64_t args_sizes = get_ck_grouped_gemm_args_sizes(group_lens.numel());
    at::Tensor    args_tensor =
        at::empty({args_sizes}, at::TensorOptions().dtype(at::kByte).device(group_lens.device()));

    // Determine output tensor size based on transA and transB
    const int32_t bs     = b.size(0);
    const int32_t m      = transA ? a.size(1) : a.size(0);
    const int32_t n      = transB ? b.size(1) : b.size(2);
    const int32_t k      = transA ? a.size(0) : a.size(1);
    at::Tensor    c      = at::empty({m, n}, at::dtype(out_dtype).device(at::kCUDA));
    auto          stream = at::cuda::getCurrentCUDAStream();

    if (a.dtype() == at::kFloat8_e4m3fnuz || a.dtype() == at::kFloat8_e4m3fn) {
        using AType = ck_tile::fp8_t;
        using BType = AType;

        if (out_dtype == at::kBFloat16) {
            using CType = ck_tile::bfloat16_t;
            ck_grouped_gemm<AType, BType, CType>(
                args_tensor.data_ptr(), reinterpret_cast<const AType *>(a.data_ptr()),
                reinterpret_cast<const BType *>(b.data_ptr()),
                reinterpret_cast<CType *>(c.data_ptr()),
                reinterpret_cast<const int64_t *>(group_lens.data_ptr()),
                reinterpret_cast<const int64_t *>(group_offs.data_ptr()), transA, transB, bs, m, n,
                k, stream, get_grouped_gemm_num_cu(num_cu));
        } else if (out_dtype == at::kHalf) {
            using CType = ck_tile::half_t;
            ck_grouped_gemm<AType, BType, CType>(
                args_tensor.data_ptr(), reinterpret_cast<const AType *>(a.data_ptr()),
                reinterpret_cast<const BType *>(b.data_ptr()),
                reinterpret_cast<CType *>(c.data_ptr()),
                reinterpret_cast<const int64_t *>(group_lens.data_ptr()),
                reinterpret_cast<const int64_t *>(group_offs.data_ptr()), transA, transB, bs, m, n,
                k, stream, get_grouped_gemm_num_cu(num_cu));
        } else {
            PRIMUS_TURBO_CHECK(false, "Unsupported out_dtype for fp8 e4m3");
        }
    } else if (a.dtype() == at::kFloat8_e5m2fnuz || a.dtype() == at::kFloat8_e5m2) {
        using AType = ck_tile::bf8_t;
        using BType = AType;

        if (out_dtype == at::kBFloat16) {
            using CType = ck_tile::bfloat16_t;
            ck_grouped_gemm<AType, BType, CType>(
                args_tensor.data_ptr(), reinterpret_cast<const AType *>(a.data_ptr()),
                reinterpret_cast<const BType *>(b.data_ptr()),
                reinterpret_cast<CType *>(c.data_ptr()),
                reinterpret_cast<const int64_t *>(group_lens.data_ptr()),
                reinterpret_cast<const int64_t *>(group_offs.data_ptr()), transA, transB, bs, m, n,
                k, stream, get_grouped_gemm_num_cu(num_cu));
        } else if (out_dtype == at::kHalf) {
            using CType = ck_tile::half_t;
            ck_grouped_gemm<AType, BType, CType>(
                args_tensor.data_ptr(), reinterpret_cast<const AType *>(a.data_ptr()),
                reinterpret_cast<const BType *>(b.data_ptr()),
                reinterpret_cast<CType *>(c.data_ptr()),
                reinterpret_cast<const int64_t *>(group_lens.data_ptr()),
                reinterpret_cast<const int64_t *>(group_offs.data_ptr()), transA, transB, bs, m, n,
                k, stream, get_grouped_gemm_num_cu(num_cu));
        } else {
            PRIMUS_TURBO_CHECK(false, "Unsupported out_dtype for fp8 e5m2");
        }
    } else {
        PRIMUS_TURBO_CHECK(false, "GroupedGemmFp8 only support fp8/bf8");
    }
    return c;
}

at::Tensor grouped_gemm_variable_k(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                                   at::Tensor &group_offs, const bool transA, const bool transB,
                                   c10::optional<int64_t> num_cu) {
    // TODO: output datatype
    auto out_dtype = a.scalar_type();

    // Check
    PRIMUS_TURBO_CHECK(is_16bit_floating_point_dtype(a.scalar_type()));
    PRIMUS_TURBO_CHECK(is_16bit_floating_point_dtype(b.scalar_type()));
    PRIMUS_TURBO_CHECK(group_lens.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(group_offs.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(a.scalar_type() == b.scalar_type(), "a and b dtype mismatch");

    // Alloc args workspace
    const int64_t args_sizes = get_ck_grouped_gemm_args_sizes(group_lens.numel());
    at::Tensor    args_tensor =
        at::empty({args_sizes}, at::TensorOptions().dtype(at::kByte).device(group_lens.device()));

    // Determine output tensor size based on transA and transB
    const int32_t bs = group_lens.numel();
    const int64_t m  = transA ? a.size(1) : a.size(0);
    const int64_t n  = transB ? b.size(0) : b.size(1);
    const int32_t k  = transA ? a.size(0) : a.size(1);
    at::Tensor    c  = at::empty({bs, m, n}, at::dtype(out_dtype).device(at::kCUDA));

    auto stream = at::cuda::getCurrentCUDAStream();
    if (a.dtype() == at::kHalf) {
        using AType = typename TorchToCKTileType<at::kHalf>::type;
        using BType = AType;
        using CType = AType;
        ck_grouped_gemm_variable_k<AType, BType, CType>(
            args_tensor.data_ptr(), reinterpret_cast<const AType *>(a.data_ptr()),
            reinterpret_cast<const BType *>(b.data_ptr()), reinterpret_cast<CType *>(c.data_ptr()),
            reinterpret_cast<const int64_t *>(group_lens.data_ptr()),
            reinterpret_cast<const int64_t *>(group_offs.data_ptr()), transA, transB, bs, m, n, k,
            stream, get_grouped_gemm_num_cu(num_cu));
    } else if (a.dtype() == at::kBFloat16) {
        using AType = typename TorchToCKTileType<at::kBFloat16>::type;
        using BType = AType;
        using CType = AType;
        ck_grouped_gemm_variable_k<AType, BType, CType>(
            args_tensor.data_ptr(), reinterpret_cast<const AType *>(a.data_ptr()),
            reinterpret_cast<const BType *>(b.data_ptr()), reinterpret_cast<CType *>(c.data_ptr()),
            reinterpret_cast<const int64_t *>(group_lens.data_ptr()),
            reinterpret_cast<const int64_t *>(group_offs.data_ptr()), transA, transB, bs, m, n, k,
            stream, get_grouped_gemm_num_cu(num_cu));
    } else {
        PRIMUS_TURBO_CHECK(false, "GroupedGemm only support float16 and bfloat16");
    }

    return c;
}

at::Tensor grouped_gemm_fp8_variable_k(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                                       at::Tensor &group_offs, const bool transA, const bool transB,
                                       at::ScalarType out_dtype, c10::optional<int64_t> num_cu) {
    // Check
    PRIMUS_TURBO_CHECK(is_8bit_floating_point_dtype(a.scalar_type()));
    PRIMUS_TURBO_CHECK(is_8bit_floating_point_dtype(b.scalar_type()));
    PRIMUS_TURBO_CHECK(group_lens.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(group_offs.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(a.scalar_type() == b.scalar_type(), "a and b dtype mismatch");
    PRIMUS_TURBO_CHECK(out_dtype == at::kBFloat16 || out_dtype == at::kHalf,
                       "out_dtype must be kBFloat16 or kHalf");

    // Alloc args workspace
    const int64_t args_sizes = get_ck_grouped_gemm_args_sizes(group_lens.numel());
    at::Tensor    args_tensor =
        at::empty({args_sizes}, at::TensorOptions().dtype(at::kByte).device(group_lens.device()));

    // Determine output tensor size based on transA and transB
    const int32_t bs = group_lens.numel();
    const int64_t m  = transA ? a.size(1) : a.size(0);
    const int64_t n  = transB ? b.size(0) : b.size(1);
    const int32_t k  = transA ? a.size(0) : a.size(1);
    at::Tensor    c  = at::empty({bs, m, n}, at::dtype(out_dtype).device(at::kCUDA));

    auto stream = at::cuda::getCurrentCUDAStream();

    if (a.dtype() == at::kFloat8_e4m3fnuz || a.dtype() == at::kFloat8_e4m3fn) {
        using AType = ck_tile::fp8_t;
        using BType = AType;
        if (out_dtype == at::kBFloat16) {
            using CType = ck_tile::bfloat16_t;
            ck_grouped_gemm_variable_k<AType, BType, CType>(
                args_tensor.data_ptr(), reinterpret_cast<const AType *>(a.data_ptr()),
                reinterpret_cast<const BType *>(b.data_ptr()),
                reinterpret_cast<CType *>(c.data_ptr()),
                reinterpret_cast<const int64_t *>(group_lens.data_ptr()),
                reinterpret_cast<const int64_t *>(group_offs.data_ptr()), transA, transB, bs, m, n,
                k, stream, get_grouped_gemm_num_cu(num_cu));
        }
    } else if (a.dtype() == at::kFloat8_e5m2fnuz || a.dtype() == at::kFloat8_e5m2) {
        using AType = ck_tile::bf8_t;
        using BType = AType;
        if (out_dtype == at::kBFloat16) {
            using CType = ck_tile::bfloat16_t;
            ck_grouped_gemm_variable_k<AType, BType, CType>(
                args_tensor.data_ptr(), reinterpret_cast<const AType *>(a.data_ptr()),
                reinterpret_cast<const BType *>(b.data_ptr()),
                reinterpret_cast<CType *>(c.data_ptr()),
                reinterpret_cast<const int64_t *>(group_lens.data_ptr()),
                reinterpret_cast<const int64_t *>(group_offs.data_ptr()), transA, transB, bs, m, n,
                k, stream, get_grouped_gemm_num_cu(num_cu));
        }
    } else {
        PRIMUS_TURBO_CHECK(false, "GroupedGemmFp8 only support fp8");
    }

    return c;
}

} // namespace primus_turbo::pytorch
