// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "ck_tile/host/hip_check_error.hpp"

#include "ck_grouped_gemm_kernel_instance_factory.h"
#include "ck_grouped_gemm_kernel_template.h"
#include "primus_turbo/grouped_gemm.h"

namespace primus_turbo {

std::int64_t get_ck_grouped_gemm_args_sizes(const int group_num) {
    return group_num * sizeof(ck_tile::GemmTransKernelArg<>);
}

std::int64_t get_ck_grouped_gemm_fp8_args_sizes(const int group_num) {
    return group_num * sizeof(ck_tile::QuantGemmTransKernelArg);
}

template <typename ADataType, typename BDataType, typename CDataType>
__global__ void
compute_grouped_gemm_args(ck_tile::GemmTransKernelArg<> *args_ptr, const ADataType *a_ptr,
                          const BDataType *b_ptr, CDataType *c_ptr, const int64_t *group_lens_ptr,
                          const int64_t *group_offs_ptr, const ck_tile::index_t group_num,
                          const ck_tile::index_t n, const ck_tile::index_t k,
                          const ck_tile::index_t strideA, const ck_tile::index_t strideB,
                          const ck_tile::index_t strideC, const ck_tile::index_t k_batch) {
    const int64_t group_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_id >= group_num)
        return;

    const_cast<std::array<const void *, 1> &>(args_ptr[group_id].group_karg.as_ptr)[0] =
        static_cast<const void *>(a_ptr + group_offs_ptr[group_id] * k);
    const_cast<std::array<const void *, 1> &>(args_ptr[group_id].group_karg.bs_ptr)[0] =
        static_cast<const void *>(b_ptr + group_id * n * k);
    args_ptr[group_id].group_karg.e_ptr        = c_ptr + group_offs_ptr[group_id] * n;
    args_ptr[group_id].group_karg.M            = group_lens_ptr[group_id];
    args_ptr[group_id].group_karg.N            = n;
    args_ptr[group_id].group_karg.K            = k;
    args_ptr[group_id].group_karg.stride_As[0] = strideA;
    args_ptr[group_id].group_karg.stride_Bs[0] = strideB;
    args_ptr[group_id].group_karg.stride_E     = strideC;
    args_ptr[group_id].group_karg.k_batch      = k_batch;
}

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType = float,
          ck_tile::QuantType QuantMode = ck_tile::QuantType::TensorQuant>
__global__ void compute_grouped_gemm_fp8_args(
    ck_tile::QuantGemmTransKernelArg *args_ptr, const ADataType *a_ptr, const BDataType *b_ptr,
    CDataType *c_ptr, const AccDataType *aq_ptr, const AccDataType *bq_ptr,
    const int64_t *group_lens_ptr, const int64_t *group_offs_ptr, const ck_tile::index_t group_num,
    const ck_tile::index_t n, const ck_tile::index_t k, const ck_tile::index_t strideA,
    const ck_tile::index_t strideB, const ck_tile::index_t strideC, const ck_tile::index_t strideAQ,
    const ck_tile::index_t strideBQ, const ck_tile::index_t k_batch) {
    const int64_t group_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_id >= group_num)
        return;
    if (QuantMode == ck_tile::QuantType::TensorQuant) {
        args_ptr[group_id].group_karg.aq_ptr = aq_ptr;
        args_ptr[group_id].group_karg.bq_ptr = bq_ptr;
    } else if (QuantMode == ck_tile::QuantType::RowColQuant) {
        args_ptr[group_id].group_karg.aq_ptr = aq_ptr + group_offs_ptr[group_id];
        args_ptr[group_id].group_karg.bq_ptr = bq_ptr + group_id * n;
    }
    args_ptr[group_id].group_karg.a_ptr     = a_ptr + group_offs_ptr[group_id] * k;
    args_ptr[group_id].group_karg.b_ptr     = b_ptr + group_id * n * k;
    args_ptr[group_id].group_karg.c_ptr     = c_ptr + group_offs_ptr[group_id] * n;
    args_ptr[group_id].group_karg.M         = group_lens_ptr[group_id];
    args_ptr[group_id].group_karg.N         = n;
    args_ptr[group_id].group_karg.K         = k;
    args_ptr[group_id].group_karg.stride_A  = strideA;
    args_ptr[group_id].group_karg.stride_B  = strideB;
    args_ptr[group_id].group_karg.stride_AQ = strideAQ;
    args_ptr[group_id].group_karg.stride_BQ = strideBQ;
    args_ptr[group_id].group_karg.stride_C  = strideC;
    args_ptr[group_id].group_karg.k_batch   = k_batch;
}

template <typename IndexType>
__global__ void compute_group_offs_device(const IndexType       *group_lens_ptr,
                                          IndexType             *group_offs_ptr,
                                          const ck_tile::index_t group_num) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        group_offs_ptr[0] = 0;
    }

    if (idx < group_num) {
        // Compute cumulative sum for group offsets
        IndexType cumsum = 0;
        for (ck_tile::index_t i = 0; i < idx; i++) {
            cumsum += group_lens_ptr[i];
        }
        group_offs_ptr[idx + 1] = cumsum + group_lens_ptr[idx];
    }
}

template <typename IndexType>
void compute_group_offs(const IndexType *group_lens_ptr, IndexType *group_offs_ptr,
                        const int64_t group_num, hipStream_t stream) {
    const ck_tile::index_t total_elements    = group_num;
    const int              threads_per_block = 256;
    const int              blocks =
        static_cast<int>((total_elements + threads_per_block - 1) / threads_per_block);

    compute_group_offs_device<IndexType>
        <<<blocks, threads_per_block, 0, stream>>>(group_lens_ptr, group_offs_ptr, group_num);
}

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType>
void ck_grouped_gemm(const CKGroupedGemmParams<ADataType, BDataType, CDataType> &params) {
    const ck_tile::index_t k_batch = 1;
    const bool             splitk  = k_batch > 1;

    const ck_tile::index_t strideA = params.transA ? params.m : params.k;
    const ck_tile::index_t strideB = params.transB ? params.k : params.n;
    const ck_tile::index_t strideC = params.n;

    // Setting args
    {
        const int threads = std::min(MAX_THREADS_PER_BLOCK, params.group_num);
        const int blocks  = (params.group_num + threads - 1) / threads;
        compute_grouped_gemm_args<ADataType, BDataType, CDataType>
            <<<blocks, threads, 0, params.stream>>>(
                reinterpret_cast<ck_tile::GemmTransKernelArg<> *>(params.args_ptr), params.a_ptr,
                params.b_ptr, params.c_ptr, params.group_lens_ptr, params.group_offs_ptr,
                params.group_num, params.n, params.k, strideA, strideB, strideC, k_batch);
    }

    const auto stream_cfg = ck_tile::stream_config{params.stream};
    std::unique_ptr<CKGroupedGemmRunnerInterFace> runner;
    using CLayout = RowMajor;
    if (!params.transA && !params.transB) { // NN
        using ALayout = RowMajor;
        using BLayout = RowMajor;
        runner = get_ck_grouped_gemm_instance<ADataType, BDataType, CDataType, AccDataType, ALayout,
                                              BLayout, CLayout, ck_tile::QuantType::TensorQuant>(
            params.group_num, params.m, params.n, params.k);
    } else if (!params.transA && params.transB) { // NT
        using ALayout = RowMajor;
        using BLayout = ColMajor;
        runner = get_ck_grouped_gemm_instance<ADataType, BDataType, CDataType, AccDataType, ALayout,
                                              BLayout, CLayout, ck_tile::QuantType::TensorQuant>(
            params.group_num, params.m, params.n, params.k);
    } else {
        PRIMUS_TURBO_CHECK(false, "CKGroupedGemm only support NN and NT");
    }
    runner->run(stream_cfg, params.group_num, params.args_ptr, params.num_cu);
}

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType,
          ck_tile::QuantType QuantMode>
void ck_grouped_gemm_fp8(
    const CKGroupedGemmFP8Params<ADataType, BDataType, CDataType, AccDataType> &params) {
    const ck_tile::index_t k_batch = 1;
    const bool             splitk  = k_batch > 1;

    const ck_tile::index_t strideA  = params.transA ? params.m : params.k;
    const ck_tile::index_t strideB  = params.transB ? params.k : params.n;
    const ck_tile::index_t strideC  = params.n;
    const ck_tile::index_t strideAQ = 1;
    const ck_tile::index_t strideBQ = 1;
    // Setting args
    {
        const int threads = std::min(MAX_THREADS_PER_BLOCK, params.group_num);
        const int blocks  = (params.group_num + threads - 1) / threads;
        compute_grouped_gemm_fp8_args<ADataType, BDataType, CDataType, AccDataType, QuantMode>
            <<<blocks, threads, 0, params.stream>>>(
                reinterpret_cast<ck_tile::QuantGemmTransKernelArg *>(params.args_ptr), params.a_ptr,
                params.b_ptr, params.c_ptr, params.aq_ptr, params.bq_ptr, params.group_lens_ptr,
                params.group_offs_ptr, params.group_num, params.n, params.k, strideA, strideB,
                strideC, strideAQ, strideBQ, k_batch);
    }

    const auto stream_cfg = ck_tile::stream_config{params.stream};
    std::unique_ptr<CKGroupedGemmRunnerInterFace> runner;
    using CLayout = RowMajor;
    if (!params.transA && !params.transB) { // NN
        using ALayout = RowMajor;
        using BLayout = RowMajor;
        runner = get_ck_grouped_gemm_instance<ADataType, BDataType, CDataType, AccDataType, ALayout,
                                              BLayout, CLayout, QuantMode>(
            params.group_num, params.m, params.n, params.k);
    } else if (!params.transA && params.transB) { // NT
        using ALayout = RowMajor;
        using BLayout = ColMajor;
        runner = get_ck_grouped_gemm_instance<ADataType, BDataType, CDataType, AccDataType, ALayout,
                                              BLayout, CLayout, QuantMode>(
            params.group_num, params.m, params.n, params.k);
    } else {
        PRIMUS_TURBO_CHECK(false, "CKGroupedGemm only support NN and NT");
    }
    runner->run(stream_cfg, params.group_num, params.args_ptr, params.num_cu);
}

template <typename ADataType, typename BDataType, typename CDataType>
__global__ void compute_grouped_gemm_variable_k_args(
    ck_tile::GemmTransKernelArg<> *args_ptr, const ADataType *a_ptr, const BDataType *b_ptr,
    CDataType *c_ptr, const int64_t *group_lens_ptr, const int64_t *group_offs_ptr,
    const bool transA, const bool transB, const ck_tile::index_t group_num,
    const ck_tile::index_t m, const ck_tile::index_t n, const ck_tile::index_t strideA,
    const ck_tile::index_t strideB, const ck_tile::index_t strideC,
    const ck_tile::index_t k_batch) {
    const int64_t group_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_id >= group_num)
        return;

    const int64_t strideAK = transA ? m : 1;
    const int64_t strideBK = transB ? 1 : n;
    const_cast<std::array<const void *, 1> &>(args_ptr[group_id].group_karg.as_ptr)[0] =
        static_cast<const void *>(a_ptr + group_offs_ptr[group_id] * strideAK);
    const_cast<std::array<const void *, 1> &>(args_ptr[group_id].group_karg.bs_ptr)[0] =
        static_cast<const void *>(b_ptr + group_offs_ptr[group_id] * strideBK);
    args_ptr[group_id].group_karg.e_ptr        = c_ptr + group_id * m * n;
    args_ptr[group_id].group_karg.M            = m;
    args_ptr[group_id].group_karg.N            = n;
    args_ptr[group_id].group_karg.K            = group_lens_ptr[group_id];
    args_ptr[group_id].group_karg.stride_As[0] = strideA;
    args_ptr[group_id].group_karg.stride_Bs[0] = strideB;
    args_ptr[group_id].group_karg.stride_E     = strideC;
    args_ptr[group_id].group_karg.k_batch      = k_batch;
}

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType,
          ck_tile::QuantType QuantMode = ck_tile::QuantType::TensorQuant>
__global__ void compute_grouped_gemm_fp8_variable_k_args(
    ck_tile::QuantGemmTransKernelArg *args_ptr, const ADataType *a_ptr, const BDataType *b_ptr,
    CDataType *c_ptr, const AccDataType *aq_ptr, const AccDataType *bq_ptr,
    const int64_t *group_lens_ptr, const int64_t *group_offs_ptr, const bool transA,
    const bool transB, const ck_tile::index_t group_num, const ck_tile::index_t m,
    const ck_tile::index_t n, const ck_tile::index_t strideA, const ck_tile::index_t strideB,
    const ck_tile::index_t strideC, const ck_tile::index_t strideAQ,
    const ck_tile::index_t strideBQ, const ck_tile::index_t k_batch) {
    const int64_t group_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_id >= group_num)
        return;

    const int64_t strideAK = transA ? m : 1;
    const int64_t strideBK = transB ? 1 : n;

    args_ptr[group_id].group_karg.aq_ptr = aq_ptr;
    args_ptr[group_id].group_karg.bq_ptr = bq_ptr;

    // Set data pointers
    args_ptr[group_id].group_karg.a_ptr     = a_ptr + group_offs_ptr[group_id] * strideAK;
    args_ptr[group_id].group_karg.b_ptr     = b_ptr + group_offs_ptr[group_id] * strideBK;
    args_ptr[group_id].group_karg.c_ptr     = c_ptr + group_id * m * n;
    args_ptr[group_id].group_karg.M         = m;
    args_ptr[group_id].group_karg.N         = n;
    args_ptr[group_id].group_karg.K         = group_lens_ptr[group_id];
    args_ptr[group_id].group_karg.stride_A  = strideA;
    args_ptr[group_id].group_karg.stride_B  = strideB;
    args_ptr[group_id].group_karg.stride_AQ = strideAQ;
    args_ptr[group_id].group_karg.stride_BQ = strideBQ;
    args_ptr[group_id].group_karg.stride_C  = strideC;
    args_ptr[group_id].group_karg.k_batch   = k_batch;
}

/**
 * PostProcess: Set the non-computed parts in C to zero.
 */
template <typename T>
__global__ void
grouped_gemm_variable_k_postprocess(T *c_ptr, const int64_t *group_lens_ptr,
                                    const int64_t *group_offs_ptr, const ck_tile::index_t group_num,
                                    const ck_tile::index_t m, const ck_tile::index_t n) {
    const int group_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_id >= group_num)
        return;

    const int group_len = group_lens_ptr[group_id];
    const int group_off = group_offs_ptr[group_id];
    if (group_len > 0)
        return;

    c_ptr = c_ptr + group_id * m * n;
    memset(c_ptr, 0, sizeof(T) * m * n);
}

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType>
void ck_grouped_gemm_variable_k(
    const CKGroupedGemmParams<ADataType, BDataType, CDataType> &params) {
    const ck_tile::index_t k_batch = 1;
    const bool             splitk  = k_batch > 1;

    const ck_tile::index_t strideA = params.transA ? params.m : params.k;
    const ck_tile::index_t strideB = params.transB ? params.k : params.n;
    const ck_tile::index_t strideC = params.n;

    {
        const int threads = std::min(MAX_THREADS_PER_BLOCK, params.group_num);
        const int grids   = (params.group_num + threads - 1) / threads;
        compute_grouped_gemm_variable_k_args<ADataType, BDataType, CDataType>
            <<<grids, threads, 0, params.stream>>>(
                reinterpret_cast<ck_tile::GemmTransKernelArg<> *>(params.args_ptr), params.a_ptr,
                params.b_ptr, params.c_ptr, params.group_lens_ptr, params.group_offs_ptr,
                params.transA, params.transB, params.group_num, params.m, params.n, strideA,
                strideB, strideC, k_batch);
    }

    const auto stream_cfg = ck_tile::stream_config{params.stream};
    using CLayout         = RowMajor;
    std::unique_ptr<CKGroupedGemmRunnerInterFace> runner;
    if (params.transA && !params.transB) { // TN
        using ALayout = ColMajor;
        using BLayout = RowMajor;
        runner = get_ck_grouped_gemm_instance<ADataType, BDataType, CDataType, AccDataType, ALayout,
                                              BLayout, CLayout, ck_tile::QuantType::TensorQuant>(
            params.group_num, params.m, params.n, params.k);
    } else {
        PRIMUS_TURBO_CHECK(false, "CKGroupedGemm-VariableK only support TN");
    }
    runner->run(stream_cfg, params.group_num, params.args_ptr, params.num_cu);

    // Postprocess
    {
        const int threads = std::min(MAX_THREADS_PER_BLOCK, params.group_num);
        const int grids   = (params.group_num + threads - 1) / threads;
        grouped_gemm_variable_k_postprocess<CDataType><<<grids, threads, 0, params.stream>>>(
            params.c_ptr, params.group_lens_ptr, params.group_offs_ptr, params.group_num, params.m,
            params.n);
    }
}

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType,
          ck_tile::QuantType QuantMode>
void ck_grouped_gemm_fp8_variable_k(
    const CKGroupedGemmFP8Params<ADataType, BDataType, CDataType, AccDataType> &params) {
    const ck_tile::index_t k_batch = 1;
    const bool             splitk  = k_batch > 1;

    const ck_tile::index_t strideA  = params.transA ? params.m : params.k;
    const ck_tile::index_t strideB  = params.transB ? params.k : params.n;
    const ck_tile::index_t strideC  = params.n;
    const ck_tile::index_t strideAQ = 1;
    const ck_tile::index_t strideBQ = 1;

    {
        const int threads = std::min(MAX_THREADS_PER_BLOCK, params.group_num);
        const int grids   = (params.group_num + threads - 1) / threads;
        compute_grouped_gemm_fp8_variable_k_args<ADataType, BDataType, CDataType, AccDataType,
                                                 QuantMode><<<grids, threads, 0, params.stream>>>(
            reinterpret_cast<ck_tile::QuantGemmTransKernelArg *>(params.args_ptr), params.a_ptr,
            params.b_ptr, params.c_ptr, params.aq_ptr, params.bq_ptr, params.group_lens_ptr,
            params.group_offs_ptr, params.transA, params.transB, params.group_num, params.m,
            params.n, strideA, strideB, strideC, strideAQ, strideBQ, k_batch);
    }

    const auto stream_cfg = ck_tile::stream_config{params.stream};
    using CLayout         = RowMajor;
    std::unique_ptr<CKGroupedGemmRunnerInterFace> runner;
    if (params.transA && !params.transB) { // TN
        using ALayout = ColMajor;
        using BLayout = RowMajor;
        runner = get_ck_grouped_gemm_instance<ADataType, BDataType, CDataType, AccDataType, ALayout,
                                              BLayout, CLayout, QuantMode>(
            params.group_num, params.m, params.n, params.k);
    } else {
        PRIMUS_TURBO_CHECK(false, "CKGroupedGemm-VariableK only support TN");
    }
    runner->run(stream_cfg, params.group_num, params.args_ptr, params.num_cu);

    // Postprocess
    {
        const int threads = std::min(MAX_THREADS_PER_BLOCK, params.group_num);
        const int grids   = (params.group_num + threads - 1) / threads;
        grouped_gemm_variable_k_postprocess<CDataType><<<grids, threads, 0, params.stream>>>(
            params.c_ptr, params.group_lens_ptr, params.group_offs_ptr, params.group_num, params.m,
            params.n);
    }
}

// ck_grouped_gemm explicit instantiation.
// fp16 * fp16 -> fp16
template void ck_grouped_gemm<ck_tile::half_t, ck_tile::half_t, ck_tile::half_t>(
    const CKGroupedGemmParams<ck_tile::half_t, ck_tile::half_t, ck_tile::half_t> &params);

// bf16 * bf16 -> bf16
template void ck_grouped_gemm<ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t>(
    const CKGroupedGemmParams<ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t>
        &params);

// fp8 * fp8 -> fp16
template void ck_grouped_gemm_fp8<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, float,
                                  ck_tile::QuantType::TensorQuant>(
    const CKGroupedGemmFP8Params<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, float> &params);
template void ck_grouped_gemm_fp8<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, float,
                                  ck_tile::QuantType::RowColQuant>(
    const CKGroupedGemmFP8Params<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, float> &params);
// bf8 * bf8 -> fp16
template void ck_grouped_gemm_fp8<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float,
                                  ck_tile::QuantType::TensorQuant>(
    const CKGroupedGemmFP8Params<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float> &params);
template void ck_grouped_gemm_fp8<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float,
                                  ck_tile::QuantType::RowColQuant>(
    const CKGroupedGemmFP8Params<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float> &params);
// fp8 * fp8 -> bf16
template void ck_grouped_gemm_fp8<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, float,
                                  ck_tile::QuantType::TensorQuant>(
    const CKGroupedGemmFP8Params<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, float>
        &params);
template void ck_grouped_gemm_fp8<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, float,
                                  ck_tile::QuantType::RowColQuant>(
    const CKGroupedGemmFP8Params<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, float>
        &params);
// bf8 * bf8 -> bf16
template void ck_grouped_gemm_fp8<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, float,
                                  ck_tile::QuantType::TensorQuant>(
    const CKGroupedGemmFP8Params<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, float>
        &params);
template void ck_grouped_gemm_fp8<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, float,
                                  ck_tile::QuantType::RowColQuant>(
    const CKGroupedGemmFP8Params<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, float>
        &params);
// ck_grouped_gemm_variable_k explicit instantiation.
// fp16 * fp16 -> fp16
template void ck_grouped_gemm_variable_k<ck_tile::half_t, ck_tile::half_t, ck_tile::half_t>(
    const CKGroupedGemmParams<ck_tile::half_t, ck_tile::half_t, ck_tile::half_t> &params);
// bf16 * bf16 -> bf16
template void
ck_grouped_gemm_variable_k<ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t>(
    const CKGroupedGemmParams<ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t>
        &params);

// fp8 * fp8 -> fp16
template void ck_grouped_gemm_fp8_variable_k<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, float,
                                             ck_tile::QuantType::TensorQuant>(
    const CKGroupedGemmFP8Params<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, float> &params);
template void ck_grouped_gemm_fp8_variable_k<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, float,
                                             ck_tile::QuantType::RowColQuant>(
    const CKGroupedGemmFP8Params<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, float> &params);
// bf8 * bf8 -> fp16
template void ck_grouped_gemm_fp8_variable_k<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float,
                                             ck_tile::QuantType::TensorQuant>(
    const CKGroupedGemmFP8Params<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float> &params);
template void ck_grouped_gemm_fp8_variable_k<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float,
                                             ck_tile::QuantType::RowColQuant>(
    const CKGroupedGemmFP8Params<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float> &params);
// fp8 * fp8 -> bf16
template void ck_grouped_gemm_fp8_variable_k<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t,
                                             float, ck_tile::QuantType::TensorQuant>(
    const CKGroupedGemmFP8Params<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, float>
        &params);
template void ck_grouped_gemm_fp8_variable_k<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t,
                                             float, ck_tile::QuantType::RowColQuant>(
    const CKGroupedGemmFP8Params<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, float>
        &params);
// bf8 * bf8 -> bf16
template void ck_grouped_gemm_fp8_variable_k<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t,
                                             float, ck_tile::QuantType::TensorQuant>(
    const CKGroupedGemmFP8Params<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, float>
        &params);
template void ck_grouped_gemm_fp8_variable_k<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t,
                                             float, ck_tile::QuantType::RowColQuant>(
    const CKGroupedGemmFP8Params<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, float>
        &params);
template void compute_group_offs<int64_t>(const int64_t *group_lens_ptr, int64_t *group_offs_ptr,
                                          const int64_t group_num, hipStream_t stream);
} // namespace primus_turbo
