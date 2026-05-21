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

// Round v up to next power of 2 (v >= 1).
inline static int next_pow2(int64_t v) {
    int p = 1;
    while (static_cast<int64_t>(p) < v)
        p <<= 1;
    return p;
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

template <int BLOCK_SIZE, int TILE_INNER, int ELEMS_PER_THREAD, typename FType, typename QType,
          typename ComputeType = float>
__launch_bounds__(BLOCK_SIZE) __global__
    void dequantize_rowwise_col_major_kernel(const QType *__restrict__ input,
                                             const float *__restrict__ scale_inv,
                                             FType *__restrict__ output, const int64_t m,
                                             const int64_t n) {
    const int threads_per_row = blockDim.x;
    const int rows_per_block  = blockDim.y;
    const int thread_idx_n    = threadIdx.x;
    const int thread_idx_m    = threadIdx.y;

    // ---- Batch offset ---------------------------------------------------
    // gridDim.z indexes the batch. Each batch has its own contiguous (M, N)
    // tile of input/output and its own length-N row of scale_inv.
    const int64_t batch_idx = static_cast<int64_t>(blockIdx.z);
    input += batch_idx * m * n;
    output += batch_idx * m * n;
    scale_inv += batch_idx * n;

    const int64_t col = static_cast<int64_t>(blockIdx.x) * threads_per_row * ELEMS_PER_THREAD +
                        static_cast<int64_t>(thread_idx_n) * ELEMS_PER_THREAD;
    const int64_t row_block = static_cast<int64_t>(blockIdx.y) * rows_per_block * TILE_INNER;

    if (col >= n)
        return;

    // Load ELEMS_PER_THREAD scales for this column strip ONCE; reused
    // TILE_INNER times by the M-sweep below.
    float scale_regs[ELEMS_PER_THREAD];
    primus_turbo::load_data<float, ELEMS_PER_THREAD>(scale_inv + col, scale_regs);

    QType load_regs[ELEMS_PER_THREAD];
    FType store_regs[ELEMS_PER_THREAD];

#pragma unroll
    for (int m_iter = 0; m_iter < TILE_INNER; ++m_iter) {
        const int64_t row =
            row_block + static_cast<int64_t>(m_iter) * rows_per_block + thread_idx_m;
        if (row >= m)
            return;

        const int64_t addr = row * n + col;
        primus_turbo::load_data<QType, ELEMS_PER_THREAD>(input + addr, load_regs);
#pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            store_regs[i] =
                static_cast<FType>(static_cast<ComputeType>(load_regs[i]) * scale_regs[i]);
        }
        primus_turbo::store_data<FType, ELEMS_PER_THREAD>(output + addr, store_regs);
    }
}

template <typename FType, typename QType, typename ComputeType>
inline void dequantize_rowwise_col_major_impl(const QType *x, const float *scale_inv, FType *y,
                                              const int64_t batch, const int64_t m, const int64_t n,
                                              hipStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int TILE_INNER = 32;

    int pack_size = static_cast<int>(
        std::min(primus_turbo::get_pack_size<QType>(x), primus_turbo::get_pack_size<FType>(y)));
    // Cap at 4: avoids ELEMS_PER_THREAD * sizeof(FType) > 16 (e.g. float + 8 elems
    // = 32 bytes exceeds load/store_data's 16-byte limit) and keeps the dispatch
    // switch below in sync (no case 8).
    if (pack_size > 4)
        pack_size = 4;
    while (pack_size > 1 && (n % pack_size) != 0)
        pack_size >>= 1;
    const int elems_per_thread = pack_size;

    int threads_per_row;
    if (n >= static_cast<int64_t>(BLOCK_SIZE) * elems_per_thread) {
        threads_per_row = BLOCK_SIZE;
    } else {
        const int64_t threads_needed = (n + elems_per_thread - 1) / elems_per_thread;
        threads_per_row              = next_pow2(threads_needed);
        if (threads_per_row > BLOCK_SIZE)
            threads_per_row = BLOCK_SIZE;
        if (threads_per_row < 1)
            threads_per_row = 1;
    }
    const int rows_per_block = BLOCK_SIZE / threads_per_row;

    const int64_t grid_x = (n + static_cast<int64_t>(threads_per_row) * elems_per_thread - 1) /
                           (static_cast<int64_t>(threads_per_row) * elems_per_thread);
    const int64_t other_dims = grid_x * batch;

    auto launch = [&](auto elems_tag) {
        constexpr int elems_per_thread_value = decltype(elems_tag)::value;
        const int64_t grid_y = (m + static_cast<int64_t>(rows_per_block) * TILE_INNER - 1) /
                               (static_cast<int64_t>(rows_per_block) * TILE_INNER);
        const dim3 block(static_cast<unsigned>(threads_per_row),
                         static_cast<unsigned>(rows_per_block), 1u);
        const dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y),
                        static_cast<unsigned>(batch));
        dequantize_rowwise_col_major_kernel<BLOCK_SIZE, TILE_INNER, elems_per_thread_value, FType,
                                            QType, ComputeType>
            <<<grid, block, 0, stream>>>(x, scale_inv, y, m, n);
    };

    switch (elems_per_thread) {
    case 4:
        launch(std::integral_constant<int, 4>{});
        break;
    case 2:
        launch(std::integral_constant<int, 2>{});
        break;
    default:
        launch(std::integral_constant<int, 1>{});
        break;
    }
}

template <int BLOCK_SIZE, int TILE_OUTER, int ELEMS_PER_THREAD, typename FType, typename QType,
          typename ComputeType = float>
__launch_bounds__(BLOCK_SIZE) __global__
    void dequantize_rowwise_row_major_kernel(const QType *__restrict__ input,
                                             const float *__restrict__ scale_inv,
                                             FType *__restrict__ output, const int64_t outer_len,
                                             const int64_t inner_len) {
    const int threads_per_row  = blockDim.x;
    const int rows_per_block   = blockDim.y;
    const int thread_idx_inner = threadIdx.x;
    const int thread_idx_outer = threadIdx.y;

    const int64_t inner_col =
        static_cast<int64_t>(blockIdx.x) * threads_per_row * ELEMS_PER_THREAD +
        static_cast<int64_t>(thread_idx_inner) * ELEMS_PER_THREAD;
    const int64_t outer_block = static_cast<int64_t>(blockIdx.y) * rows_per_block * TILE_OUTER;

    if (inner_col >= inner_len)
        return;

    QType load_regs[ELEMS_PER_THREAD];
    FType store_regs[ELEMS_PER_THREAD];

#pragma unroll
    for (int outer_iter = 0; outer_iter < TILE_OUTER; ++outer_iter) {
        const int64_t outer_row =
            outer_block + static_cast<int64_t>(outer_iter) * rows_per_block + thread_idx_outer;
        if (outer_row >= outer_len)
            return;

        const ComputeType scale_inv_val = static_cast<ComputeType>(scale_inv[outer_row]);

        const int64_t addr = outer_row * inner_len + inner_col;
        primus_turbo::load_data<QType, ELEMS_PER_THREAD>(input + addr, load_regs);
#pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            store_regs[i] =
                static_cast<FType>(static_cast<ComputeType>(load_regs[i]) * scale_inv_val);
        }
        primus_turbo::store_data<FType, ELEMS_PER_THREAD>(output + addr, store_regs);
    }
}

template <typename FType, typename QType, typename ComputeType>
void dequantize_rowwise_row_major_impl(const QType *x, const float *scale_inv, FType *y,
                                       const int64_t outer_len, const int64_t inner_len,
                                       hipStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int TILE_OUTER = 32;

    int pack_size = static_cast<int>(
        std::min(primus_turbo::get_pack_size<QType>(x), primus_turbo::get_pack_size<FType>(y)));
    // Cap at 4: avoids ELEMS_PER_THREAD * sizeof(FType) > 16 (e.g. float + 8 elems
    // = 32 bytes exceeds load/store_data's 16-byte limit) and keeps the dispatch
    // switch below in sync (no case 8).
    if (pack_size > 4)
        pack_size = 4;
    while (pack_size > 1 && (inner_len % pack_size) != 0)
        pack_size >>= 1;
    const int elems_per_thread = pack_size;

    int threads_per_row;
    if (inner_len >= static_cast<int64_t>(BLOCK_SIZE) * elems_per_thread) {
        threads_per_row = BLOCK_SIZE;
    } else {
        const int64_t threads_needed = (inner_len + elems_per_thread - 1) / elems_per_thread;
        threads_per_row              = next_pow2(threads_needed);
        if (threads_per_row > BLOCK_SIZE)
            threads_per_row = BLOCK_SIZE;
        if (threads_per_row < 1)
            threads_per_row = 1;
    }
    const int rows_per_block = BLOCK_SIZE / threads_per_row;

    const int64_t grid_x =
        (inner_len + static_cast<int64_t>(threads_per_row) * elems_per_thread - 1) /
        (static_cast<int64_t>(threads_per_row) * elems_per_thread);

    auto launch = [&](auto elems_tag) {
        constexpr int elems_per_thread_value = decltype(elems_tag)::value;
        const int64_t grid_y = (outer_len + static_cast<int64_t>(rows_per_block) * TILE_OUTER - 1) /
                               (static_cast<int64_t>(rows_per_block) * TILE_OUTER);
        const dim3 block(static_cast<unsigned>(threads_per_row),
                         static_cast<unsigned>(rows_per_block), 1u);
        const dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y), 1u);
        dequantize_rowwise_row_major_kernel<BLOCK_SIZE, TILE_OUTER, elems_per_thread_value, FType,
                                            QType, ComputeType>
            <<<grid, block, 0, stream>>>(x, scale_inv, y, outer_len, inner_len);
    };

    switch (elems_per_thread) {
    case 4:
        launch(std::integral_constant<int, 4>{});
        break;
    case 2:
        launch(std::integral_constant<int, 2>{});
        break;
    default:
        launch(std::integral_constant<int, 1>{});
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
