// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "pytorch/extensions.h"
#include "pytorch/type_traits.h"

#include "primus_turbo/arch.h"
#include "primus_turbo/grouped_gemm.h"

namespace primus_turbo::pytorch {

template <typename AType, typename BType, typename CType>
inline CKGroupedGemmParams<AType, BType, CType>
make_ck_groued_gemm_params(void *args_ptr, const at::Tensor &a, const at::Tensor &b, at::Tensor &c,
                           const at::Tensor &group_lens, const at::Tensor &group_offs, bool transA,
                           bool transB, int32_t group_num, int32_t m, int32_t n, int32_t k,
                           hipStream_t stream, uint32_t num_cu) {
    CKGroupedGemmParams<AType, BType, CType> params;
    params.args_ptr       = args_ptr;
    params.a_ptr          = reinterpret_cast<const AType *>(a.data_ptr());
    params.b_ptr          = reinterpret_cast<const BType *>(b.data_ptr());
    params.c_ptr          = reinterpret_cast<CType *>(c.data_ptr());
    params.group_lens_ptr = reinterpret_cast<const int64_t *>(group_lens.data_ptr());
    params.group_offs_ptr = reinterpret_cast<const int64_t *>(group_offs.data_ptr());
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

template <typename AType, typename BType, typename CType, typename ACCType>
inline CKGroupedGemmFP8Params<AType, BType, CType, ACCType> make_ck_groued_gemm_fp8_params(
    void *args_ptr, const at::Tensor &a, const at::Tensor &b, at::Tensor &c,
    const at::Tensor &a_scales, const at::Tensor &b_scales, const at::Tensor &group_lens,
    const at::Tensor &group_offs, bool transA, bool transB, int32_t group_num, int32_t m, int32_t n,
    int32_t k, hipStream_t stream, uint32_t num_cu) {
    CKGroupedGemmFP8Params<AType, BType, CType, ACCType> params;
    params.args_ptr       = args_ptr;
    params.a_ptr          = reinterpret_cast<const AType *>(a.data_ptr());
    params.b_ptr          = reinterpret_cast<const BType *>(b.data_ptr());
    params.c_ptr          = reinterpret_cast<CType *>(c.data_ptr());
    params.aq_ptr         = reinterpret_cast<const ACCType *>(a_scales.data_ptr());
    params.bq_ptr         = reinterpret_cast<const ACCType *>(b_scales.data_ptr());
    params.group_lens_ptr = reinterpret_cast<const int64_t *>(group_lens.data_ptr());
    params.group_offs_ptr = reinterpret_cast<const int64_t *>(group_offs.data_ptr());
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

at::Tensor ck_grouped_gemm(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
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
        auto params = make_ck_groued_gemm_params<AType, BType, CType>(
            args_tensor.data_ptr(), a, b, c, group_lens, group_offs, transA, transB, bs, m, n, k,
            stream, get_grouped_gemm_num_cu(num_cu));
        primus_turbo::ck_grouped_gemm<AType, BType, CType>(params);
    } else if (a.dtype() == at::kBFloat16) {
        using AType = typename TorchToCKTileType<at::kBFloat16>::type;
        using BType = AType;
        using CType = AType;
        auto params = make_ck_groued_gemm_params<AType, BType, CType>(
            args_tensor.data_ptr(), a, b, c, group_lens, group_offs, transA, transB, bs, m, n, k,
            stream, get_grouped_gemm_num_cu(num_cu));
        primus_turbo::ck_grouped_gemm<AType, BType, CType>(params);
    } else {
        PRIMUS_TURBO_CHECK(false, "GroupedGemm only support float16 and bfloat16");
    }
    return c;
}

at::Tensor ck_grouped_gemm_fp8(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                               at::Tensor &b_scales, at::Tensor &group_lens, at::Tensor &group_offs,
                               const bool transA, const bool transB, at::ScalarType out_dtype,
                               const std::string &granularity, c10::optional<int64_t> num_cu) {

    // Check
    PRIMUS_TURBO_CHECK(is_8bit_floating_point_dtype(a.scalar_type()));
    PRIMUS_TURBO_CHECK(is_8bit_floating_point_dtype(b.scalar_type()));
    PRIMUS_TURBO_CHECK(group_lens.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(group_offs.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(a.scalar_type() == b.scalar_type(), "a and b dtype mismatch");
    PRIMUS_TURBO_CHECK(out_dtype == at::kBFloat16 || out_dtype == at::kHalf,
                       "out_dtype must be kBFloat16 or kHalf");
    PRIMUS_TURBO_CHECK(granularity == "TENSORWISE" || granularity == "ROWWISE" ||
                           granularity == "BLOCKWISE",
                       "granularity must be 'TENSORWISE', 'ROWWISE', or 'BLOCKWISE'");

    // Determine output tensor size based on transA and transB
    const int64_t bs = b.size(0);
    const int64_t m  = transA ? a.size(1) : a.size(0);
    const int64_t n  = transB ? b.size(1) : b.size(2);
    const int64_t k  = transA ? a.size(0) : a.size(1);

    // For BLOCKWISE (ABQuantGrouped), check alignment requirements
    if (granularity == "BLOCKWISE" && !transA) {
        PRIMUS_TURBO_CHECK(k % 128 == 0,
                           "For BLOCKWISE granularity, k must be a multiple of 128, got k=", k);
        PRIMUS_TURBO_CHECK(n % 128 == 0,
                           "For BLOCKWISE granularity, n must be a multiple of 128, got n=", n);
        PRIMUS_TURBO_CHECK(k >= 128, "For BLOCKWISE granularity, k must be at least 128");
    }

    // Alloc args workspace
    const int64_t args_sizes = get_ck_grouped_gemm_fp8_args_sizes(group_lens.numel());
    at::Tensor    args_tensor =
        at::empty({args_sizes}, at::TensorOptions().dtype(at::kByte).device(group_lens.device()));

    // Handle scale tensor transpose for BLOCKWISE
    if (granularity == "BLOCKWISE") {
        a_scales = transA ? a_scales.transpose(-1, -2) : a_scales;
        b_scales = !transB ? b_scales.transpose(-1, -2) : b_scales;
    }

    at::Tensor aq_tensor = a_scales.contiguous();
    at::Tensor bq_tensor = b_scales.contiguous();

    at::Tensor c      = at::empty({m, n}, at::dtype(out_dtype).device(at::kCUDA));
    auto       stream = at::cuda::getCurrentCUDAStream();

    if (a.dtype() == at::kFloat8_e4m3fnuz || a.dtype() == at::kFloat8_e4m3fn) {
        using AType = typename TorchToCKTileType<at::kFloat8_e4m3fnuz>::type;
        using BType = AType;

        if (out_dtype == at::kBFloat16) {
            using CType = typename TorchToCKTileType<at::kBFloat16>::type;
            auto params = make_ck_groued_gemm_fp8_params<AType, BType, CType, float>(
                args_tensor.data_ptr(), a, b, c, aq_tensor, bq_tensor, group_lens, group_offs,
                transA, transB, bs, m, n, k, stream, get_grouped_gemm_num_cu(num_cu));
            if (granularity == "TENSORWISE")
                primus_turbo::ck_grouped_gemm_fp8<AType, BType, CType, float,
                                                  ck_tile::QuantType::TensorQuant>(params);
            else if (granularity == "ROWWISE")
                primus_turbo::ck_grouped_gemm_fp8<AType, BType, CType, float,
                                                  ck_tile::QuantType::RowColQuant>(params);
            else // BLOCKWISE
                primus_turbo::ck_grouped_gemm_fp8<AType, BType, CType, float,
                                                  ck_tile::QuantType::ABQuantGrouped>(params);
        } else if (out_dtype == at::kHalf) {
            using CType = typename TorchToCKTileType<at::kHalf>::type;
            auto params = make_ck_groued_gemm_fp8_params<AType, BType, CType, float>(
                args_tensor.data_ptr(), a, b, c, aq_tensor, bq_tensor, group_lens, group_offs,
                transA, transB, bs, m, n, k, stream, get_grouped_gemm_num_cu(num_cu));
            if (granularity == "TENSORWISE")
                primus_turbo::ck_grouped_gemm_fp8<AType, BType, CType, float,
                                                  ck_tile::QuantType::TensorQuant>(params);
            else if (granularity == "ROWWISE")
                primus_turbo::ck_grouped_gemm_fp8<AType, BType, CType, float,
                                                  ck_tile::QuantType::RowColQuant>(params);
            else // BLOCKWISE
                primus_turbo::ck_grouped_gemm_fp8<AType, BType, CType, float,
                                                  ck_tile::QuantType::ABQuantGrouped>(params);
        } else {
            PRIMUS_TURBO_CHECK(false, "Unsupported out_dtype for fp8 e4m3");
        }
    } else if (a.dtype() == at::kFloat8_e5m2fnuz || a.dtype() == at::kFloat8_e5m2) {
        using AType = typename TorchToCKTileType<at::kFloat8_e5m2fnuz>::type;
        using BType = AType;

        if (out_dtype == at::kBFloat16) {
            using CType = typename TorchToCKTileType<at::kBFloat16>::type;
            auto params = make_ck_groued_gemm_fp8_params<AType, BType, CType, float>(
                args_tensor.data_ptr(), a, b, c, aq_tensor, bq_tensor, group_lens, group_offs,
                transA, transB, bs, m, n, k, stream, get_grouped_gemm_num_cu(num_cu));
            if (granularity == "TENSORWISE")
                primus_turbo::ck_grouped_gemm_fp8<AType, BType, CType, float,
                                                  ck_tile::QuantType::TensorQuant>(params);
            else if (granularity == "ROWWISE")
                primus_turbo::ck_grouped_gemm_fp8<AType, BType, CType, float,
                                                  ck_tile::QuantType::RowColQuant>(params);
            else // BLOCKWISE
                primus_turbo::ck_grouped_gemm_fp8<AType, BType, CType, float,
                                                  ck_tile::QuantType::ABQuantGrouped>(params);
        } else if (out_dtype == at::kHalf) {
            using CType = typename TorchToCKTileType<at::kHalf>::type;
            auto params = make_ck_groued_gemm_fp8_params<AType, BType, CType, float>(
                args_tensor.data_ptr(), a, b, c, aq_tensor, bq_tensor, group_lens, group_offs,
                transA, transB, bs, m, n, k, stream, get_grouped_gemm_num_cu(num_cu));
            if (granularity == "TENSORWISE")
                primus_turbo::ck_grouped_gemm_fp8<AType, BType, CType, float,
                                                  ck_tile::QuantType::TensorQuant>(params);
            else if (granularity == "ROWWISE")
                primus_turbo::ck_grouped_gemm_fp8<AType, BType, CType, float,
                                                  ck_tile::QuantType::RowColQuant>(params);
            else // BLOCKWISE
                primus_turbo::ck_grouped_gemm_fp8<AType, BType, CType, float,
                                                  ck_tile::QuantType::ABQuantGrouped>(params);
        } else {
            PRIMUS_TURBO_CHECK(false, "Unsupported out_dtype for fp8 e5m2");
        }
    } else {
        PRIMUS_TURBO_CHECK(false, "GroupedGemmFp8 only support fp8/bf8");
    }

    return c;
}

at::Tensor ck_grouped_gemm_variable_k(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                                      at::Tensor &group_offs, const bool transA, const bool transB,
                                      c10::optional<int64_t> num_cu) {
    // TODO: output datatype
    auto out_dtype = a.scalar_type();
    printf("test2\r\n");
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
    const int64_t bs = group_lens.numel();
    const int64_t m  = transA ? a.size(1) : a.size(0);
    const int64_t n  = transB ? b.size(0) : b.size(1);
    const int64_t k  = transA ? a.size(0) : a.size(1);
    at::Tensor    c  = at::empty({bs, m, n}, at::dtype(out_dtype).device(at::kCUDA));

    auto stream = at::cuda::getCurrentCUDAStream();
    if (a.dtype() == at::kHalf) {
        using AType = typename TorchToCKTileType<at::kHalf>::type;
        using BType = AType;
        using CType = AType;
        auto params = make_ck_groued_gemm_params<AType, BType, CType>(
            args_tensor.data_ptr(), a, b, c, group_lens, group_offs, transA, transB, bs, m, n, k,
            stream, get_grouped_gemm_num_cu(num_cu));
        primus_turbo::ck_grouped_gemm_variable_k<AType, BType, CType>(params);
    } else if (a.dtype() == at::kBFloat16) {
        using AType = typename TorchToCKTileType<at::kBFloat16>::type;
        using BType = AType;
        using CType = AType;
        auto params = make_ck_groued_gemm_params<AType, BType, CType>(
            args_tensor.data_ptr(), a, b, c, group_lens, group_offs, transA, transB, bs, m, n, k,
            stream, get_grouped_gemm_num_cu(num_cu));
        primus_turbo::ck_grouped_gemm_variable_k<AType, BType, CType>(params);
    } else {
        PRIMUS_TURBO_CHECK(false, "GroupedGemm only support float16 and bfloat16");
    }

    return c;
}

at::Tensor ck_grouped_gemm_fp8_variable_k(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                                          at::Tensor &b_scales, at::Tensor &group_lens,
                                          at::Tensor &group_offs, const bool transA,
                                          const bool transB, at::ScalarType out_dtype,
                                          const std::string     &granularity,
                                          c10::optional<int64_t> num_cu) {
    // Check
    PRIMUS_TURBO_CHECK(is_8bit_floating_point_dtype(a.scalar_type()));
    PRIMUS_TURBO_CHECK(is_8bit_floating_point_dtype(b.scalar_type()));
    PRIMUS_TURBO_CHECK(group_lens.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(group_offs.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(a.scalar_type() == b.scalar_type(), "a and b dtype mismatch");
    PRIMUS_TURBO_CHECK(out_dtype == at::kBFloat16 || out_dtype == at::kHalf,
                       "out_dtype must be kBFloat16 or kHalf");
    PRIMUS_TURBO_CHECK(granularity == "TENSORWISE" || granularity == "ROWWISE" ||
                           granularity == "BLOCKWISE",
                       "granularity must be 'TENSORWISE', 'ROWWISE', or 'BLOCKWISE'");

    // Determine output tensor size based on transA and transB
    const int64_t bs = group_lens.numel();
    const int64_t m  = transA ? a.size(1) : a.size(0);
    const int64_t n  = transB ? b.size(0) : b.size(1);
    const int64_t k  = transA ? a.size(0) : a.size(1);

    // For BLOCKWISE (ABQuantGrouped), check alignment requirements
    if (granularity == "BLOCKWISE") {
        PRIMUS_TURBO_CHECK(m % 128 == 0,
                           "For BLOCKWISE granularity, m must be a multiple of 128, got m=", m);
        PRIMUS_TURBO_CHECK(n % 128 == 0,
                           "For BLOCKWISE granularity, n must be a multiple of 128, got n=", n);
        PRIMUS_TURBO_CHECK(k >= 128, "For BLOCKWISE granularity, k must be at least 128");
    }

    // Alloc args workspace
    const int64_t args_sizes = get_ck_grouped_gemm_fp8_args_sizes(group_lens.numel());
    at::Tensor    args_tensor =
        at::empty({args_sizes}, at::TensorOptions().dtype(at::kByte).device(group_lens.device()));

    at::Tensor c = at::empty({bs, m, n}, at::dtype(out_dtype).device(at::kCUDA));

    // Handle scale tensor transpose for BLOCKWISE
    if (granularity == "BLOCKWISE") {
        a_scales = transA ? a_scales.transpose(-1, -2) : a_scales;
        b_scales = !transB ? b_scales.transpose(-1, -2) : b_scales;
    }

    at::Tensor aq_tensor = a_scales.contiguous();
    at::Tensor bq_tensor = b_scales.contiguous();

    auto stream = at::cuda::getCurrentCUDAStream();

    if (a.dtype() == at::kFloat8_e4m3fnuz || a.dtype() == at::kFloat8_e4m3fn) {
        using AType = typename TorchToCKTileType<at::kFloat8_e4m3fnuz>::type;
        using BType = AType;
        if (out_dtype == at::kBFloat16) {
            using CType = typename TorchToCKTileType<at::kBFloat16>::type;
            auto params = make_ck_groued_gemm_fp8_params<AType, BType, CType, float>(
                args_tensor.data_ptr(), a, b, c, aq_tensor, bq_tensor, group_lens, group_offs,
                transA, transB, bs, m, n, k, stream, get_grouped_gemm_num_cu(num_cu));
            if (granularity == "TENSORWISE")
                primus_turbo::ck_grouped_gemm_fp8_variable_k<AType, BType, CType, float,
                                                             ck_tile::QuantType::TensorQuant>(
                    params);
            else if (granularity == "ROWWISE")
                primus_turbo::ck_grouped_gemm_fp8_variable_k<AType, BType, CType, float,
                                                             ck_tile::QuantType::RowColQuant>(
                    params);
            else // BLOCKWISE
                primus_turbo::ck_grouped_gemm_fp8_variable_k<AType, BType, CType, float,
                                                             ck_tile::QuantType::ABQuantGrouped>(
                    params);
        } else if (out_dtype == at::kHalf) {
            using CType = typename TorchToCKTileType<at::kHalf>::type;
            auto params = make_ck_groued_gemm_fp8_params<AType, BType, CType, float>(
                args_tensor.data_ptr(), a, b, c, aq_tensor, bq_tensor, group_lens, group_offs,
                transA, transB, bs, m, n, k, stream, get_grouped_gemm_num_cu(num_cu));
            if (granularity == "TENSORWISE")
                primus_turbo::ck_grouped_gemm_fp8_variable_k<AType, BType, CType, float,
                                                             ck_tile::QuantType::TensorQuant>(
                    params);
            else if (granularity == "ROWWISE")
                primus_turbo::ck_grouped_gemm_fp8_variable_k<AType, BType, CType, float,
                                                             ck_tile::QuantType::RowColQuant>(
                    params);
            else // BLOCKWISE
                primus_turbo::ck_grouped_gemm_fp8_variable_k<AType, BType, CType, float,
                                                             ck_tile::QuantType::ABQuantGrouped>(
                    params);
        } else {
            PRIMUS_TURBO_CHECK(false, "GroupedGemmFp8: out dtype only support fp16 and bf16");
        }
    } else if (a.dtype() == at::kFloat8_e5m2fnuz || a.dtype() == at::kFloat8_e5m2) {
        using AType = typename TorchToCKTileType<at::kFloat8_e5m2fnuz>::type;
        using BType = AType;
        if (out_dtype == at::kBFloat16) {
            using CType = typename TorchToCKTileType<at::kBFloat16>::type;
            auto params = make_ck_groued_gemm_fp8_params<AType, BType, CType, float>(
                args_tensor.data_ptr(), a, b, c, aq_tensor, bq_tensor, group_lens, group_offs,
                transA, transB, bs, m, n, k, stream, get_grouped_gemm_num_cu(num_cu));
            if (granularity == "TENSORWISE")
                primus_turbo::ck_grouped_gemm_fp8_variable_k<AType, BType, CType, float,
                                                             ck_tile::QuantType::TensorQuant>(
                    params);
            else if (granularity == "ROWWISE")
                primus_turbo::ck_grouped_gemm_fp8_variable_k<AType, BType, CType, float,
                                                             ck_tile::QuantType::RowColQuant>(
                    params);
            else // BLOCKWISE
                primus_turbo::ck_grouped_gemm_fp8_variable_k<AType, BType, CType, float,
                                                             ck_tile::QuantType::ABQuantGrouped>(
                    params);
        } else if (out_dtype == at::kHalf) {
            using CType = typename TorchToCKTileType<at::kHalf>::type;
            auto params = make_ck_groued_gemm_fp8_params<AType, BType, CType, float>(
                args_tensor.data_ptr(), a, b, c, aq_tensor, bq_tensor, group_lens, group_offs,
                transA, transB, bs, m, n, k, stream, get_grouped_gemm_num_cu(num_cu));
            if (granularity == "TENSORWISE")
                primus_turbo::ck_grouped_gemm_fp8_variable_k<AType, BType, CType, float,
                                                             ck_tile::QuantType::TensorQuant>(
                    params);
            else if (granularity == "ROWWISE")
                primus_turbo::ck_grouped_gemm_fp8_variable_k<AType, BType, CType, float,
                                                             ck_tile::QuantType::RowColQuant>(
                    params);
            else // BLOCKWISE
                primus_turbo::ck_grouped_gemm_fp8_variable_k<AType, BType, CType, float,
                                                             ck_tile::QuantType::ABQuantGrouped>(
                    params);
        } else {
            PRIMUS_TURBO_CHECK(false, "GroupedGemmFp8: out dtype only support fp16 and bf16");
        }
    } else {
        PRIMUS_TURBO_CHECK(false, "GroupedGemmFp8 input dtype only support fp8/bf8");
    }

    return c;
}

} // namespace primus_turbo::pytorch
