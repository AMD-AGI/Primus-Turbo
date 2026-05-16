// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/arch.h"
#include "primus_turbo/grouped_gemm.h"
#include "turbo/turbo_grouped_gemm_mxfp8_kernel.h"
#include "turbo/turbo_grouped_gemm_mxfp8_wgrad_kernel.h"
#include <hip/hip_runtime.h>

namespace primus_turbo {

// ── Preshuffle (grouped-GEMM path) ──
//
// Bit-for-bit equivalent to ``turbo::preshuffle_scale_16x4_kernel`` but
// stages each 16-row tile through LDS in ``CHUNK_COLS``-wide column
// slabs so all loads/stores are fully coalesced.  Used for both fwd /
// dgrad (A+B scales) and wgrad (LHS+RHS scales).
template <int BLOCK_THREADS, int CHUNK_COLS>
__device__ __forceinline__ void preshuffle_one_tile(const uint8_t *__restrict__ in,
                                                    uint32_t *__restrict__ out, const int cols) {
    static_assert(CHUNK_COLS % 4 == 0, "CHUNK_COLS must be /4 aligned");
    static_assert(BLOCK_THREADS >= CHUNK_COLS, "BLOCK_THREADS must be >= CHUNK_COLS");

    __shared__ uint8_t s_tile[16 * CHUNK_COLS];

    const int tid = threadIdx.x;

    for (int col_start = 0; col_start < cols; col_start += CHUNK_COLS) {
        const int chunk_cols = min(CHUNK_COLS, cols - col_start);

#pragma unroll
        for (int row = 0; row < 16; ++row) {
            if (tid < chunk_cols) {
                s_tile[row * CHUNK_COLS + tid] = in[row * cols + col_start + tid];
            }
        }
        __syncthreads();

        const int chunk_blocks = chunk_cols / 4;
        const int total_out    = chunk_blocks * 64;
        const int out_base     = (col_start / 4) * 64;
        for (int idx = tid; idx < total_out; idx += BLOCK_THREADS) {
            const int     col_block = idx >> 6;
            const int     sub       = idx & 63;
            const int     row       = sub & 15;
            const int     col       = sub >> 4;
            const uint8_t v         = s_tile[row * CHUNK_COLS + col_block * 4 + col];
            out[out_base + idx]     = static_cast<uint32_t>(v);
        }
        __syncthreads();
    }
}

template <int BLOCK_THREADS, int CHUNK_COLS>
__global__ __launch_bounds__(BLOCK_THREADS, 4) void preshuffle_scale_16x4_dual_kernel(
    const uint8_t *__restrict__ in0, uint32_t *__restrict__ out0, const int rows0,
    const uint8_t *__restrict__ in1, uint32_t *__restrict__ out1, const int cols) {

    const int      blocks0 = rows0 / 16;
    const int      bid     = blockIdx.x;
    const uint8_t *in;
    uint32_t      *out;
    if (bid < blocks0) {
        in  = in0 + (size_t) bid * 16 * cols;
        out = out0 + (size_t) bid * 16 * cols;
    } else {
        const int sb = bid - blocks0;
        in           = in1 + (size_t) sb * 16 * cols;
        out          = out1 + (size_t) sb * 16 * cols;
    }
    preshuffle_one_tile<BLOCK_THREADS, CHUNK_COLS>(in, out, cols);
}

static constexpr int PRESHUFFLE_BLOCK_THREADS = 256;
static constexpr int PRESHUFFLE_CHUNK_COLS    = 256;

// rows0 / rows1 are multiples of 16 and cols a multiple of 4 by upstream
// checks (per-group M_g % 128, N % 16, K % 128).
static inline void preshuffle_dual_launch(const uint8_t *in0, uint32_t *out0, int rows0,
                                          const uint8_t *in1, uint32_t *out1, int rows1, int cols,
                                          hipStream_t stream) {
    const int grid = (rows0 + rows1) / 16;
    preshuffle_scale_16x4_dual_kernel<PRESHUFFLE_BLOCK_THREADS, PRESHUFFLE_CHUNK_COLS>
        <<<grid, PRESHUFFLE_BLOCK_THREADS, 0, stream>>>(in0, out0, rows0, in1, out1, cols);
}

// ── Workspace size ──
//
// Layout: [ A_scale_preshuf : total_M  * scale_cols * uint32 ]
//         [ B_scale_preshuf : groups*N * scale_cols * uint32 ]
size_t turbo_grouped_gemm_mxfp8_workspace_size(int32_t total_m, int32_t group_num, int32_t n,
                                               int32_t k) {
    constexpr int32_t MX_BLOCK_SIZE = 32;
    const int32_t     scale_cols    = (k + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;
    const size_t      a_scale_bytes = (size_t) total_m * scale_cols * sizeof(uint32_t);
    const size_t b_scale_bytes = (size_t) group_num * (size_t) n * scale_cols * sizeof(uint32_t);
    return a_scale_bytes + b_scale_bytes;
}

// ── Public API ──

template <typename AType, typename BType, typename CType>
void turbo_grouped_gemm_mxfp8_impl(const TurboGroupedGemmMXFP8Params<AType, BType, CType> &params) {
    constexpr int32_t MX_BLOCK_SIZE = 32;
    const int32_t     total_m       = params.total_m;
    const int32_t     group_num     = params.group_num;
    const int32_t     n             = params.n;
    const int32_t     k             = params.k;
    const int32_t     scale_cols    = (k + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;
    const size_t      a_scale_bytes = (size_t) total_m * scale_cols * sizeof(uint32_t);

    auto *a_scale_preshuf = reinterpret_cast<uint32_t *>(params.workspace);
    auto *b_scale_preshuf =
        reinterpret_cast<uint32_t *>(reinterpret_cast<uint8_t *>(params.workspace) + a_scale_bytes);

    // E8M0 raw bytes → uint32_t (zero-extend) + preshuffle (A & B fused)
    auto *a_scale_raw = reinterpret_cast<const uint8_t *>(params.a_scale_ptr);
    auto *b_scale_raw = reinterpret_cast<const uint8_t *>(params.b_scale_ptr);
    preshuffle_dual_launch(a_scale_raw, a_scale_preshuf, total_m, b_scale_raw, b_scale_preshuf,
                           group_num * n, scale_cols, params.stream);

    const int32_t grid_m = params.grid_x;
    const int32_t grid_n = (n + 255) / 256;
    dim3          grid(256);
    dim3          block(256);
    turbo::turbo_grouped_gemm_mxfp8_256x256x128_16x16x128_4wave_persistent_kernel<AType, BType,
                                                                                  CType>
        <<<grid, block, 0, params.stream>>>(params.a_ptr, params.b_ptr, a_scale_preshuf,
                                            b_scale_preshuf, params.c_ptr, params.group_lens_ptr,
                                            params.a_group_offs_ptr, params.c_group_offs_ptr,
                                            params.c_padding_align_mask, group_num, (uint32_t) n,
                                            (uint32_t) k, grid_m, grid_n);
}

// ── Explicit instantiations ──

#define INSTANTIATE_TURBO_GROUPED_GEMM(A, B, C)                                                    \
    template void turbo_grouped_gemm_mxfp8_impl<A, B, C>(                                          \
        const TurboGroupedGemmMXFP8Params<A, B, C> &);

INSTANTIATE_TURBO_GROUPED_GEMM(dtype::float8_e4m3, dtype::float8_e4m3, dtype::float16)
INSTANTIATE_TURBO_GROUPED_GEMM(dtype::float8_e4m3, dtype::float8_e4m3, dtype::bfloat16)
INSTANTIATE_TURBO_GROUPED_GEMM(dtype::float8_e5m2, dtype::float8_e5m2, dtype::float16)
INSTANTIATE_TURBO_GROUPED_GEMM(dtype::float8_e5m2, dtype::float8_e5m2, dtype::bfloat16)
INSTANTIATE_TURBO_GROUPED_GEMM(dtype::float8_e4m3, dtype::float8_e5m2, dtype::float16)
INSTANTIATE_TURBO_GROUPED_GEMM(dtype::float8_e4m3, dtype::float8_e5m2, dtype::bfloat16)
INSTANTIATE_TURBO_GROUPED_GEMM(dtype::float8_e5m2, dtype::float8_e4m3, dtype::float16)
INSTANTIATE_TURBO_GROUPED_GEMM(dtype::float8_e5m2, dtype::float8_e4m3, dtype::bfloat16)

#undef INSTANTIATE_TURBO_GROUPED_GEMM

// ── Wgrad workspace size ──
//
// Layout: [ LHS_scale_preshuf : N * scale_cols * uint32 ]
//         [ RHS_scale_preshuf : K * scale_cols * uint32 ]
size_t turbo_grouped_gemm_mxfp8_wgrad_workspace_size(int32_t total_m, int32_t n, int32_t k) {
    constexpr int32_t MX_BLOCK_SIZE   = 32;
    const int32_t     scale_cols      = (total_m + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;
    const size_t      lhs_scale_bytes = (size_t) n * scale_cols * sizeof(uint32_t);
    const size_t      rhs_scale_bytes = (size_t) k * scale_cols * sizeof(uint32_t);
    return lhs_scale_bytes + rhs_scale_bytes;
}

template <typename AType, typename BType, typename CType>
void turbo_grouped_gemm_mxfp8_wgrad_impl(
    const TurboGroupedGemmMXFP8WgradParams<AType, BType, CType> &params) {
    constexpr int32_t MX_BLOCK_SIZE   = 32;
    const int32_t     total_m         = params.total_m;
    const int32_t     group_num       = params.group_num;
    const int32_t     n               = params.n;
    const int32_t     k               = params.k;
    const int32_t     scale_cols      = (total_m + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;
    const size_t      lhs_scale_bytes = (size_t) n * scale_cols * sizeof(uint32_t);

    auto *lhs_scale_preshuf = reinterpret_cast<uint32_t *>(params.workspace);
    auto *rhs_scale_preshuf = reinterpret_cast<uint32_t *>(
        reinterpret_cast<uint8_t *>(params.workspace) + lhs_scale_bytes);

    // E8M0 raw bytes → uint32_t (zero-extend) + preshuffle (LHS & RHS fused)
    auto *lhs_scale_raw = reinterpret_cast<const uint8_t *>(params.lhs_scale_ptr);
    auto *rhs_scale_raw = reinterpret_cast<const uint8_t *>(params.rhs_scale_ptr);
    preshuffle_dual_launch(lhs_scale_raw, lhs_scale_preshuf, n, rhs_scale_raw, rhs_scale_preshuf, k,
                           scale_cols, params.stream);

    const int32_t grid_n = (n + 255) / 256;
    const int32_t grid_k = (k + 255) / 256;
    dim3          grid(256);
    dim3          block(256);
    turbo::turbo_grouped_gemm_mxfp8_wgrad_256x256x128_16x16x128_4wave_persistent_kernel<
        AType, BType, CType><<<grid, block, 0, params.stream>>>(
        params.lhs_ptr, params.rhs_ptr, lhs_scale_preshuf, rhs_scale_preshuf, params.db_ptr,
        params.group_lens_ptr, params.a_group_offs_ptr, group_num, (uint32_t) total_m, (uint32_t) n,
        (uint32_t) k, grid_n, grid_k);
}

#define INSTANTIATE_TURBO_GROUPED_GEMM_WGRAD(A, B, C)                                              \
    template void turbo_grouped_gemm_mxfp8_wgrad_impl<A, B, C>(                                    \
        const TurboGroupedGemmMXFP8WgradParams<A, B, C> &);

INSTANTIATE_TURBO_GROUPED_GEMM_WGRAD(dtype::float8_e4m3, dtype::float8_e4m3, dtype::float16)
INSTANTIATE_TURBO_GROUPED_GEMM_WGRAD(dtype::float8_e4m3, dtype::float8_e4m3, dtype::bfloat16)
INSTANTIATE_TURBO_GROUPED_GEMM_WGRAD(dtype::float8_e5m2, dtype::float8_e5m2, dtype::float16)
INSTANTIATE_TURBO_GROUPED_GEMM_WGRAD(dtype::float8_e5m2, dtype::float8_e5m2, dtype::bfloat16)
INSTANTIATE_TURBO_GROUPED_GEMM_WGRAD(dtype::float8_e5m2, dtype::float8_e4m3, dtype::float16)
INSTANTIATE_TURBO_GROUPED_GEMM_WGRAD(dtype::float8_e5m2, dtype::float8_e4m3, dtype::bfloat16)
INSTANTIATE_TURBO_GROUPED_GEMM_WGRAD(dtype::float8_e4m3, dtype::float8_e5m2, dtype::float16)
INSTANTIATE_TURBO_GROUPED_GEMM_WGRAD(dtype::float8_e4m3, dtype::float8_e5m2, dtype::bfloat16)

#undef INSTANTIATE_TURBO_GROUPED_GEMM_WGRAD

} // namespace primus_turbo
