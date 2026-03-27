// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

/*
 * MXFP8 Quantization Kernel (CUDA/HIP)
 * =========================================
 *
 * This kernel performs fused casting to MXFP8 format with optional transpose,
 * supporting both rowwise and colwise quantization.
 *
 * Block/Tile Structure:
 *   - Block size: 64x64 (BLOCK_M x BLOCK_N)
 *   - MXFP8 tile: 32x32 elements per quantization block
 *   - Thread block: 256 threads (4 warps of 64 threads each)
 *
 * Memory Layout:
 *   - Input: bfloat16 or half matrix (M x N)
 *   - Rowwise output: FP8 (M x N) + E8M0 scales (M x N/32)
 *   - Colwise output: FP8 (N x M) + E8M0 scales (N x M/32)
 */

#include "primus_turbo/common.h"
#include "primus_turbo/device/reduce.cuh"
#include "primus_turbo/device/shuffle.cuh"
#include "primus_turbo/device/utils.cuh"
#include "primus_turbo/memory_pack.h"
#include "primus_turbo/quantization.h"

namespace primus_turbo {

using namespace primus_turbo::dtype;
using namespace primus_turbo::detail;

// ============================================================================
// CONSTANTS - Block and Tile Dimensions
// ============================================================================

// Hardware architecture parameters
constexpr int WARP_SIZE         = 64;  // AMD wavefront size
constexpr int THREADS_PER_BLOCK = 256; // 4 warps per block
constexpr int WARPS_PER_BLOCK   = THREADS_PER_BLOCK / WARP_SIZE;

// Tile dimensions for main kernel loop
constexpr int BLOCK_M = 64; // Rows per thread block
constexpr int BLOCK_N = 64; // cols per thread block

// Derived tile counts
constexpr int NUM_CHUNKS_M = BLOCK_M / MXFP8_BLOCK_SIZE; // 2 chunks in M
constexpr int NUM_CHUNKS_N = BLOCK_N / MXFP8_BLOCK_SIZE; // 2 chunks in N

// Thread work distribution within 32-element rows
constexpr int ELEMS_PER_THREAD = 4; // Elements per thread
constexpr int THREADS_PER_ROW =
    MXFP8_BLOCK_SIZE / ELEMS_PER_THREAD; // Threads cooperating on one row

// Shared memory optimization
constexpr int SMEM_PADDING = 2; // Padding to avoid bank conflicts

// ============================================================================
// QUANTIZATION - E8M0 Scale Computation and FP8 Conversion
// ============================================================================

/*
 * E8M0 Scale Computation
 * ----------------------
 * Computes the E8M0 format scale factor for MXFP8 quantization.
 * E8M0 = 8-bit exponent only (no mantissa), representing powers of 2.
 *
 */
template <typename DType>
__device__ __forceinline__ void compute_tile_scale(float r_amax, float &r_scale_native,
                                                   uint8_t &r_scale_e8m0) {
    using namespace primus_turbo::detail;

    constexpr int hp_mbits    = FP32_MANTISSA_BITS;
    constexpr int hp_ebits    = FP32_EXPONENT_BITS;
    constexpr int hp_exp_bias = FP32_EXPONENT_EXP_BIAS;

    constexpr int mbits =
        std::is_same_v<DType, dtype::float8_e5m2> ? FP8E5M2_MANTISSA_BITS : FP8E4M3_MANTISSA_BITS;
    constexpr int target_max_pow2 = std::is_same_v<DType, dtype::float8_e5m2>
                                        ? FP8E5M2_TARGET_MAX_POW2
                                        : FP8E4M3_TARGET_MAX_POW2;

    constexpr int e8m0_exponent_bias = E8M0_EXPONENT_BIAS;

    uint32_t amax_bits = float_as_uint(r_amax);

    // round even (adaptive)
    int val_to_add     = 1 << (hp_mbits - mbits - 1);
    int hp_exp_mask    = (1 << (hp_ebits + 1)) - 1;
    int extracted_pow2 = (((amax_bits + val_to_add) >> hp_mbits) & hp_exp_mask) - hp_exp_bias;
    extracted_pow2     = extracted_pow2 - target_max_pow2;

    // Clamp to exponents that can be represented in e8m0.
    // Add 1 to upper bound to preserve NaN encoding behavior.
    int scale_e8m0_unbiased = extracted_pow2;
    scale_e8m0_unbiased =
        scale_e8m0_unbiased > -e8m0_exponent_bias ? scale_e8m0_unbiased : -e8m0_exponent_bias;
    scale_e8m0_unbiased   = scale_e8m0_unbiased < (e8m0_exponent_bias + 1) ? scale_e8m0_unbiased
                                                                           : (e8m0_exponent_bias + 1);
    int scale_e8m0_biased = scale_e8m0_unbiased + e8m0_exponent_bias;

    // Store scale
    r_scale_e8m0   = (uint8_t) scale_e8m0_biased;
    r_scale_native = uint_as_float((uint32_t) scale_e8m0_biased << hp_mbits);
}

/*
 * FP32 to FP8 Conversion
 * ----------------------
 * Converts 4 FP32 values to 4 FP8 values using AMD hardware instruction.
 *
 * v_cvt_scalef32_pk_fp8_f32:
 *   - Converts 4 FP32 inputs to 4 FP8 outputs (packed in 32 bits)
 *   - Scaled conversion uses fval / scale (overflow -> NaN if |fval/scale| > fp8 max; see CK
 * mxf8_utils)
 *   - FP8 format: E4M3 (1 sign bit + 4 exponent bits + 3 mantissa bits)
 *
 * v_cvt_scalef32_pk_bf8_f32:
 *   - Converts 4 FP32 inputs to 4 BF8 outputs (packed in 32 bits)
 *   - Applies scaling during conversion
 *   - BF8 format: E5M2 (1 sign bit + 5 exponent bits + 2 mantissa bits)
 *
 * Reference: AMD CDNA4 ISA, v_cvt_scalef32_pk_fp8_f32, v_cvt_scalef32_pk_bf8_f32 (page 380)
 */
template <typename DType>
__device__ __forceinline__ uint32_t cvt_f32x4_to_fp8x4(float v0, float v1, float v2, float v3,
                                                       float scale) {
#if defined(__gfx950__)
    uint32_t result = 0;
    if constexpr (std::is_same_v<DType, dtype::float8_e4m3>) {
        // If fval / scale > max fp8, returns Nan, Do soft clamping to avoid NaN.
        const float lim = FP8E4M3_MAX * scale;
        const float v0c = fminf(fmaxf(v0, -lim), lim);
        const float v1c = fminf(fmaxf(v1, -lim), lim);
        const float v2c = fminf(fmaxf(v2, -lim), lim);
        const float v3c = fminf(fmaxf(v3, -lim), lim);

        uint16_t tmp0 = 0;
        asm volatile("v_cvt_scalef32_pk_fp8_f32 %0, %1, %2, %3"
                     : "+v"(tmp0)
                     : "v"(v0c), "v"(v1c), "v"(scale));

        uint16_t tmp1 = 0;
        asm volatile("v_cvt_scalef32_pk_fp8_f32 %0, %1, %2, %3"
                     : "+v"(tmp1)
                     : "v"(v2c), "v"(v3c), "v"(scale));

        result = tmp1;
        result = (result << 16) | tmp0;
    } else if constexpr (std::is_same_v<DType, dtype::float8_e5m2>) {
        uint16_t tmp0 = 0;
        // Convert first pair (v0, v1) to 16-bit packed BF8 (E5M2)
        asm volatile("v_cvt_scalef32_pk_bf8_f32 %0, %1, %2, %3"
                     : "+v"(tmp0)
                     : "v"(v0), "v"(v1), "v"(scale));

        // Convert second pair (v2, v3) to 16-bit packed BF8 (E5M2)
        uint16_t tmp1 = 0;
        asm volatile("v_cvt_scalef32_pk_bf8_f32 %0, %1, %2, %3"
                     : "+v"(tmp1)
                     : "v"(v2), "v"(v3), "v"(scale));

        // Combine into 32-bit result: [v0, v1] in low 16 bits, [v2, v3] in high 16 bits
        result = tmp1;
        result = (result << 16) | tmp0;
    }

    return result;
#else
    __builtin_trap();
    return 0;
#endif
}

/*
 * MXFP8 Single-Direction Quantization Kernel
 * ----------------------------------------------
 * Supports rowwise (horizontal) or colwise (vertical) quantization,
 * selected at compile-time via the MODE template parameter.
 *
 * Template Parameters (compile-time):
 *   IType:          Data type of input (float16 or bfloat16)
 *   OType:          Data type of output (float8_e5m2 or float8_e4m3)
 *   MODE:           QuantizeMode::ROWWISE or QuantizeMode::COLWISE
 *   USE_2D_BLOCK:   Use 2D block (tile-level) r_amax reduction for scale computation
 *
 * Rowwise mode:  reads from registers horizontally, stores FP8 in row-major layout.
 * Colwise mode:  reads from shared memory (transposed), stores FP8 in col-major layout.
 */
template <typename IType, typename OType, QuantizeMode MODE, bool USE_2D_BLOCK = false>
__global__ __launch_bounds__(THREADS_PER_BLOCK, 4) void quantize_mxfp8_kernel(
    const IType *__restrict__ input, OType *__restrict__ out_fp8, uint8_t *__restrict__ out_scale,
    const int M, const int N, const int M_pad, const int N_pad, const int scale_stride,
    const int scale_N, const int scale_M_pad, const int scale_N_pad, const bool shuffle_out,
    const bool shuffle_scale) {
    // ========================================================================
    // Thread and Block Identification
    // ========================================================================
    constexpr bool kIsHalf    = std::is_same_v<IType, dtype::float16>;
    constexpr bool kIsRowwise = (MODE == QuantizeMode::ROWWISE);

    const int tid           = threadIdx.x;
    const int warp_id       = tid / WARP_SIZE;
    const int lane_id       = tid % WARP_SIZE;
    const int row_in_warp   = lane_id / THREADS_PER_ROW;
    const int thread_in_row = lane_id % THREADS_PER_ROW;

    const int block_m = blockIdx.x;
    const int block_n = blockIdx.y;
    const int base_m  = block_m * BLOCK_M;
    const int base_n  = block_n * BLOCK_N;

    // FP8: 1 byte per element. Output stride is the padded dimension.
    // Rowwise: output is [M, N_pad], stride = N_pad
    // Colwise: output is [N, M_pad], stride = M_pad
    const int output_packed_stride = kIsRowwise ? N_pad : M_pad;

    constexpr int ROWS_PER_PASS   = WARP_SIZE / THREADS_PER_ROW;
    constexpr int PASSES_PER_TILE = MXFP8_BLOCK_SIZE / ROWS_PER_PASS;
    constexpr int TOTAL_CHUNKS    = NUM_CHUNKS_M * NUM_CHUNKS_N;

    // Shared memory for colwise transposed reads (minimized for rowwise mode)
    constexpr int       s_tile_DEPTH = kIsRowwise ? 1 : (MXFP8_BLOCK_SIZE + SMEM_PADDING);
    __shared__ uint16_t s_tile[WARPS_PER_BLOCK][MXFP8_BLOCK_SIZE][s_tile_DEPTH];

    // ========================================================================
    // Main Loop - Each Warp Processes One 32x32 Chunk Independently
    // ========================================================================
    for (int round = 0; round < TOTAL_CHUNKS; round += WARPS_PER_BLOCK) {
        const int chunk_index = round + warp_id;
        if (chunk_index >= TOTAL_CHUNKS)
            break;

        const int chunk_m = chunk_index / NUM_CHUNKS_N;
        const int chunk_n = chunk_index % NUM_CHUNKS_N;
        const int tile_m  = base_m + chunk_m * MXFP8_BLOCK_SIZE;
        const int tile_n  = base_n + chunk_n * MXFP8_BLOCK_SIZE;

        // ================================================================
        // Load Tile: Global → registers (+ shared memory for colwise)
        // ================================================================
        uint64_t r_tile[PASSES_PER_TILE];

        {
            const auto *input_as_uint16 = reinterpret_cast<const uint16_t *>(input);
            const int   col_base        = thread_in_row * ELEMS_PER_THREAD;
            const int   global_col      = tile_n + col_base;

#pragma unroll
            for (int pass = 0; pass < PASSES_PER_TILE; pass++) {
                const int local_row  = pass * ROWS_PER_PASS + row_in_warp;
                const int global_row = tile_m + local_row;

                uint64_t packed = 0;
                if (global_row < M) {
                    if (global_col + ELEMS_PER_THREAD - 1 < N) {
                        packed = *reinterpret_cast<const uint64_t *>(
                            &input_as_uint16[global_row * N + global_col]);
                    } else {
                        uint16_t elem0 =
                            (global_col < N) ? input_as_uint16[global_row * N + global_col] : 0;
                        uint16_t elem1 = (global_col + 1 < N)
                                             ? input_as_uint16[global_row * N + global_col + 1]
                                             : 0;
                        uint16_t elem2 = (global_col + 2 < N)
                                             ? input_as_uint16[global_row * N + global_col + 2]
                                             : 0;
                        uint16_t elem3 = (global_col + 3 < N)
                                             ? input_as_uint16[global_row * N + global_col + 3]
                                             : 0;
                        packed         = (uint64_t) elem0 | ((uint64_t) elem1 << 16) |
                                 ((uint64_t) elem2 << 32) | ((uint64_t) elem3 << 48);
                    }
                }

                // Write data to shared memory for transpose
                if constexpr (!kIsRowwise) {
                    *reinterpret_cast<uint32_t *>(&s_tile[warp_id][local_row][col_base]) =
                        (uint32_t) packed;
                    *reinterpret_cast<uint32_t *>(&s_tile[warp_id][local_row][col_base + 2]) =
                        (uint32_t) (packed >> 32);
                }

                r_tile[pass] = packed;
            }
        }

        // Synchronize threads to wait for all threads to write to shared memory
        if constexpr (!kIsRowwise) {
            __syncthreads();
        }

        // ================================================================
        // Step 1: Unpack values + Compute absolute max
        // ================================================================
        float r_vals[PASSES_PER_TILE][ELEMS_PER_THREAD];
        float r_amax[PASSES_PER_TILE];

        {
#pragma unroll
            for (int pass = 0; pass < PASSES_PER_TILE; pass++) {
                r_vals[pass][0] = r_vals[pass][1] = r_vals[pass][2] = r_vals[pass][3] = 0.f;
                r_amax[pass]                                                          = 0.f;

                if constexpr (kIsRowwise) {
                    // Rowwise: each pass processes one row, read from registers
                    const int global_row = tile_m + pass * ROWS_PER_PASS + row_in_warp;

                    if (global_row < M) {
                        packed_uint16x4_to_floatx4<kIsHalf>(r_tile[pass], r_vals[pass][0],
                                                            r_vals[pass][1], r_vals[pass][2],
                                                            r_vals[pass][3]);

                        float local_amax =
                            fmaxf(fmaxf(fabsf(r_vals[pass][0]), fabsf(r_vals[pass][1])),
                                  fmaxf(fabsf(r_vals[pass][2]), fabsf(r_vals[pass][3])));
                        r_amax[pass] = warp_reduce_max_8_dpp(local_amax);
                    }
                } else {
                    // Colwise: each pass processes one col, read from shared memory (transposed)
                    const int row_base   = thread_in_row * ELEMS_PER_THREAD;
                    const int local_col  = pass * ROWS_PER_PASS + row_in_warp;
                    const int global_col = tile_n + local_col;

                    if (global_col < N) {
                        r_vals[pass][0] =
                            uint16_to_float<kIsHalf>(s_tile[warp_id][row_base][local_col]);
                        r_vals[pass][1] =
                            uint16_to_float<kIsHalf>(s_tile[warp_id][row_base + 1][local_col]);
                        r_vals[pass][2] =
                            uint16_to_float<kIsHalf>(s_tile[warp_id][row_base + 2][local_col]);
                        r_vals[pass][3] =
                            uint16_to_float<kIsHalf>(s_tile[warp_id][row_base + 3][local_col]);

                        float local_amax =
                            fmaxf(fmaxf(fabsf(r_vals[pass][0]), fabsf(r_vals[pass][1])),
                                  fmaxf(fabsf(r_vals[pass][2]), fabsf(r_vals[pass][3])));
                        r_amax[pass] = warp_reduce_max_8_dpp(local_amax);
                    }
                }
            }
        }

        // ================================================================
        // Step 2: Compute scale — per-group or per-tile (2D Block)
        // ================================================================
        float   r_scale_native[PASSES_PER_TILE];
        uint8_t r_scale_e8m0[PASSES_PER_TILE];

        if constexpr (USE_2D_BLOCK) {
            float tile_amax = 0.f;
#pragma unroll
            for (int pass = 0; pass < PASSES_PER_TILE; pass++)
                tile_amax = fmaxf(tile_amax, r_amax[pass]);
            tile_amax = warp_reduce_max_64_dpp(tile_amax);
            float   tile_scale_native;
            uint8_t tile_scale_e8m0;
            compute_tile_scale<OType>(tile_amax, tile_scale_native, tile_scale_e8m0);
#pragma unroll
            for (int pass = 0; pass < PASSES_PER_TILE; pass++) {
                r_scale_native[pass] = tile_scale_native;
                r_scale_e8m0[pass]   = tile_scale_e8m0;
            }
        } else {
#pragma unroll
            for (int pass = 0; pass < PASSES_PER_TILE; pass++)
                compute_tile_scale<OType>(r_amax[pass], r_scale_native[pass], r_scale_e8m0[pass]);
        }

        // ================================================================
        // Step 3: Quantize + Store FP8 and Scale
        // ================================================================
        {
#pragma unroll
            for (int pass = 0; pass < PASSES_PER_TILE; pass++) {
                uint32_t fp8x4 =
                    cvt_f32x4_to_fp8x4<OType>(r_vals[pass][0], r_vals[pass][1], r_vals[pass][2],
                                              r_vals[pass][3], r_scale_native[pass]);

                if constexpr (kIsRowwise) {
                    // ---- Rowwise: iterate rows, store FP8 in row-major ----
                    const int col_base   = thread_in_row * ELEMS_PER_THREAD;
                    const int global_col = tile_n + col_base;
                    const int local_row  = pass * ROWS_PER_PASS + row_in_warp;
                    const int global_row = tile_m + local_row;

                    if (global_row < M) {
                        if (global_col < N_pad) {
                            if (shuffle_out) {
                                int shuffled_index = compute_shuffled_index<OType>(
                                    global_row, global_col, output_packed_stride);
                                *reinterpret_cast<uint32_t *>(out_fp8 + shuffled_index) = fp8x4;
                            } else {
                                *reinterpret_cast<uint32_t *>(
                                    out_fp8 + global_row * output_packed_stride + global_col) =
                                    fp8x4;
                            }
                        }

                        if (thread_in_row == 0) {
                            int scale_col = block_n * NUM_CHUNKS_N + chunk_n;
                            if (shuffle_scale) {
                                if (global_row < scale_M_pad && scale_col < scale_N_pad) {
                                    int scale_index = compute_shuffle_scale_index(
                                        global_row, scale_col, scale_N_pad);
                                    out_scale[scale_index] = (scale_col < scale_N)
                                                                 ? r_scale_e8m0[pass]
                                                                 : E8M0_EXPONENT_BIAS;
                                }
                            } else {
                                if (scale_col < scale_N) {
                                    out_scale[global_row * scale_stride + scale_col] =
                                        r_scale_e8m0[pass];
                                }
                            }
                        }
                    }

                    if (shuffle_scale && thread_in_row == 0 && global_row >= M &&
                        global_row < scale_M_pad) {
                        int scale_col = block_n * NUM_CHUNKS_N + chunk_n;
                        if (scale_col < scale_N_pad) {
                            int scale_index =
                                compute_shuffle_scale_index(global_row, scale_col, scale_N_pad);
                            out_scale[scale_index] = E8M0_EXPONENT_BIAS;
                        }
                    }

                } else {
                    // ---- Colwise: iterate cols, store FP8 in col-major ----
                    const int row_base        = thread_in_row * ELEMS_PER_THREAD;
                    const int global_row_base = tile_m + row_base;
                    const int local_col       = pass * ROWS_PER_PASS + row_in_warp;
                    const int global_col      = tile_n + local_col;

                    if (global_col < N) {
                        if (global_row_base < M_pad) {
                            if (shuffle_out) {
                                int shuffled_index = compute_shuffled_index<OType>(
                                    global_col, global_row_base, output_packed_stride);
                                *reinterpret_cast<uint32_t *>(out_fp8 + shuffled_index) = fp8x4;
                            } else {
                                *reinterpret_cast<uint32_t *>(
                                    out_fp8 + global_col * output_packed_stride + global_row_base) =
                                    fp8x4;
                            }
                        }

                        if (thread_in_row == 0) {
                            int scale_col = block_m * NUM_CHUNKS_M + chunk_m;
                            if (shuffle_scale) {
                                if (global_col < scale_M_pad && scale_col < scale_N_pad) {
                                    int scale_index = compute_shuffle_scale_index(
                                        global_col, scale_col, scale_N_pad);
                                    out_scale[scale_index] = (scale_col < scale_N)
                                                                 ? r_scale_e8m0[pass]
                                                                 : E8M0_EXPONENT_BIAS;
                                }
                            } else {
                                if (scale_col < scale_N) {
                                    out_scale[global_col * scale_stride + scale_col] =
                                        r_scale_e8m0[pass];
                                }
                            }
                        }
                    }

                    if (shuffle_scale && thread_in_row == 0 && global_col >= N &&
                        global_col < scale_M_pad) {
                        int scale_col = block_m * NUM_CHUNKS_M + chunk_m;
                        if (scale_col < scale_N_pad) {
                            int scale_index =
                                compute_shuffle_scale_index(global_col, scale_col, scale_N_pad);
                            out_scale[scale_index] = E8M0_EXPONENT_BIAS;
                        }
                    }
                }
            }
        }
    }
}

/*
 * MXFP8 Quantization Kernel with dual mode
 * ----------------------------------------------
 * Template Parameters (compile-time):
 *   IType:                                       Data type of input
 *   OType:                                       Data type of output
 *   ROWWISE_USE_2D_BLOCK / COLWISE_USE_2D_BLOCK: Use 2D block for r_amax reduction
 */
template <typename IType, typename OType, bool ROWWISE_USE_2D_BLOCK = false,
          bool COLWISE_USE_2D_BLOCK = false>
__global__ __launch_bounds__(THREADS_PER_BLOCK, 4) void quantize_mxfp8_dual_kernel(
    const IType *__restrict__ input, OType *__restrict__ rowwise_fp8,
    uint8_t *__restrict__ rowwise_scale, OType *__restrict__ colwise_fp8,
    uint8_t *__restrict__ colwise_scale, const int M, const int N, const int M_pad, const int N_pad,
    const int rowwise_scale_stride, const int colwise_scale_stride, const int rowwise_scale_N,
    const int rowwise_scale_M_pad, const int rowwise_scale_N_pad, const int colwise_scale_M,
    const int colwise_scale_N, const int colwise_scale_M_pad, const int colwise_scale_N_pad,
    const bool shuffle_rowwise, const bool shuffle_colwise, const bool shuffle_rowwise_scale,
    const bool shuffle_colwise_scale) {
    // ========================================================================
    // Thread and Block Identification
    // ========================================================================
    constexpr bool kIshalf = std::is_same_v<IType, dtype::float16>;

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // Within each warp: 8 rows, each processed by 8 threads
    const int row_in_warp   = lane_id / THREADS_PER_ROW;
    const int thread_in_row = lane_id % THREADS_PER_ROW;

    // Block indices in the grid
    const int block_m = blockIdx.x;
    const int block_n = blockIdx.y;

    // Base coordinates for this block's tile
    const int base_m = block_m * BLOCK_M;
    const int base_n = block_n * BLOCK_N;

    // FP8: 1 byte per element. Output stride is the padded dimension.
    // Rowwise: output is [M, N_pad], stride = N_pad
    // Colwise: output is [N, M_pad], stride = M_pad
    const int K_packed = N_pad;
    const int M_packed = M_pad;

    constexpr int ROWS_PER_PASS   = WARP_SIZE / THREADS_PER_ROW;
    constexpr int PASSES_PER_TILE = MXFP8_BLOCK_SIZE / ROWS_PER_PASS;
    constexpr int TOTAL_CHUNKS    = NUM_CHUNKS_M * NUM_CHUNKS_N;

    // ========================================================================
    // Shared Memory - Per-Warp 32x32 Tiles
    // ========================================================================
    __shared__ uint16_t s_tile[WARPS_PER_BLOCK][MXFP8_BLOCK_SIZE][MXFP8_BLOCK_SIZE + SMEM_PADDING];

    // ========================================================================
    // Main Loop - Each Warp Processes One 32x32 Chunk Independently
    // ========================================================================
    // 4 warps process 4 chunks in parallel.
    for (int round = 0; round < TOTAL_CHUNKS; round += WARPS_PER_BLOCK) {
        const int chunk_idx = round + warp_id;
        if (chunk_idx >= TOTAL_CHUNKS)
            break;

        const int chunk_m = chunk_idx / NUM_CHUNKS_N;
        const int chunk_n = chunk_idx % NUM_CHUNKS_N;
        const int tile_m  = base_m + chunk_m * MXFP8_BLOCK_SIZE;
        const int tile_n  = base_n + chunk_n * MXFP8_BLOCK_SIZE;

        // ================================================================
        // Load Tile: Global → smem + packed regs
        // ================================================================
        uint64_t r_tile[PASSES_PER_TILE];

        {
            const auto *input_u16  = reinterpret_cast<const uint16_t *>(input);
            const int   col_base   = thread_in_row * ELEMS_PER_THREAD;
            const int   global_col = tile_n + col_base;

#pragma unroll
            for (int pass = 0; pass < PASSES_PER_TILE; pass++) {
                const int local_row  = pass * ROWS_PER_PASS + row_in_warp;
                const int global_row = tile_m + local_row;

                uint64_t packed = 0;
                if (global_row < M) {
                    if (global_col + ELEMS_PER_THREAD - 1 < N) {
                        packed = *reinterpret_cast<const uint64_t *>(
                            &input_u16[global_row * N + global_col]);
                    } else {
                        uint16_t s0 = (global_col < N) ? input_u16[global_row * N + global_col] : 0;
                        uint16_t s1 =
                            (global_col + 1 < N) ? input_u16[global_row * N + global_col + 1] : 0;
                        uint16_t s2 =
                            (global_col + 2 < N) ? input_u16[global_row * N + global_col + 2] : 0;
                        uint16_t s3 =
                            (global_col + 3 < N) ? input_u16[global_row * N + global_col + 3] : 0;
                        packed = (uint64_t) s0 | ((uint64_t) s1 << 16) | ((uint64_t) s2 << 32) |
                                 ((uint64_t) s3 << 48);
                    }
                }

                *reinterpret_cast<uint32_t *>(&s_tile[warp_id][local_row][col_base]) =
                    (uint32_t) packed;
                *reinterpret_cast<uint32_t *>(&s_tile[warp_id][local_row][col_base + 2]) =
                    (uint32_t) (packed >> 32);

                r_tile[pass] = packed;
            }
        }

        // ================================================================
        // Rowwise Quantization (Horizontal Processing)
        // Step 1: Unpack values + compute per-row r_amax
        // ================================================================
        float r_rowwise_vals[PASSES_PER_TILE][ELEMS_PER_THREAD];
        float r_rowwise_amax[PASSES_PER_TILE];

        {
// Repeat PASSES_PER_TILE times for each warp
#pragma unroll
            for (int pass = 0; pass < PASSES_PER_TILE; pass++) {
                const int global_row = tile_m + pass * ROWS_PER_PASS + row_in_warp;

                r_rowwise_vals[pass][0] = r_rowwise_vals[pass][1] = r_rowwise_vals[pass][2] =
                    r_rowwise_vals[pass][3]                       = 0.f;
                r_rowwise_amax[pass]                              = 0.f;

                if (global_row < M) {
                    packed_uint16x4_to_floatx4<kIshalf>(
                        r_tile[pass], r_rowwise_vals[pass][0], r_rowwise_vals[pass][1],
                        r_rowwise_vals[pass][2], r_rowwise_vals[pass][3]);

                    float local_amax = fmaxf(
                        fmaxf(fabsf(r_rowwise_vals[pass][0]), fabsf(r_rowwise_vals[pass][1])),
                        fmaxf(fabsf(r_rowwise_vals[pass][2]), fabsf(r_rowwise_vals[pass][3])));
                    r_rowwise_amax[pass] = warp_reduce_max_8_dpp(local_amax);
                }
            }
        }

        // ================================================================
        // Rowwise Quantization (Horizontal Processing)
        // Step 2: Compute scale — per-row or per-tile (2D Block)
        // ================================================================
        float   r_rowwise_scale_native[PASSES_PER_TILE];
        uint8_t r_rowwise_scale_e8m0[PASSES_PER_TILE];

        if constexpr (ROWWISE_USE_2D_BLOCK) {
            float tile_amax = 0.f;
#pragma unroll
            for (int p = 0; p < PASSES_PER_TILE; p++)
                tile_amax = fmaxf(tile_amax, r_rowwise_amax[p]);
            tile_amax = warp_reduce_max_64_dpp(tile_amax);
            float   r_scale_native;
            uint8_t r_scale_e8m0;
            compute_tile_scale<OType>(tile_amax, r_scale_native, r_scale_e8m0);
#pragma unroll
            for (int p = 0; p < PASSES_PER_TILE; p++) {
                r_rowwise_scale_native[p] = r_scale_native;
                r_rowwise_scale_e8m0[p]   = r_scale_e8m0;
            }
        } else {
#pragma unroll
            for (int p = 0; p < PASSES_PER_TILE; p++)
                compute_tile_scale<OType>(r_rowwise_amax[p], r_rowwise_scale_native[p],
                                          r_rowwise_scale_e8m0[p]);
        }

        // ================================================================
        // Rowwise Quantization (Horizontal Processing)
        // Step 3: Quantize from regs + Store FP8 / Scale
        // ================================================================
        {
            const int col_base   = thread_in_row * ELEMS_PER_THREAD;
            const int global_col = tile_n + col_base;

// Repeat PASSES_PER_TILE times for each warp
#pragma unroll
            for (int pass = 0; pass < PASSES_PER_TILE; pass++) {
                const int local_row  = pass * ROWS_PER_PASS + row_in_warp;
                const int global_row = tile_m + local_row;

                if (global_row < M) {
                    uint32_t fp8x4;
                    fp8x4 = cvt_f32x4_to_fp8x4<OType>(
                        r_rowwise_vals[pass][0], r_rowwise_vals[pass][1], r_rowwise_vals[pass][2],
                        r_rowwise_vals[pass][3], r_rowwise_scale_native[pass]);

                    if (global_col < N_pad) {
                        if (shuffle_rowwise) {
                            int shuffled_idx =
                                compute_shuffled_index<OType>(global_row, global_col, K_packed);
                            *reinterpret_cast<uint32_t *>(rowwise_fp8 + shuffled_idx) = fp8x4;
                        } else {
                            *reinterpret_cast<uint32_t *>(rowwise_fp8 + global_row * K_packed +
                                                          global_col) = fp8x4;
                        }
                    }

                    if (thread_in_row == 0) {
                        int scale_col = block_n * NUM_CHUNKS_N + chunk_n;
                        if (shuffle_rowwise_scale) {
                            if (scale_col < rowwise_scale_N && global_row < rowwise_scale_M_pad &&
                                scale_col < rowwise_scale_N_pad) {
                                int idx = compute_shuffle_scale_index(global_row, scale_col,
                                                                      rowwise_scale_N_pad);
                                rowwise_scale[idx] = r_rowwise_scale_e8m0[pass];
                            }
                        } else {
                            if (scale_col < rowwise_scale_N) {
                                rowwise_scale[global_row * rowwise_scale_stride + scale_col] =
                                    r_rowwise_scale_e8m0[pass];
                            }
                        }
                    }
                }
            }
        }

        // Colwise quantization read val from smem. Need  wait smem write to finish.
        __syncthreads();

        // ================================================================
        // Colwise Quantization (Vertical Processing)
        // Step 1: Read smem (transposed) + compute per-col r_amax
        // ================================================================
        float r_colwise_vals[PASSES_PER_TILE][ELEMS_PER_THREAD];
        float r_colwise_amax[PASSES_PER_TILE];

        {
            const int row_base = thread_in_row * ELEMS_PER_THREAD;

// Repeat PASSES_PER_TILE times for each warp
#pragma unroll
            for (int pass = 0; pass < PASSES_PER_TILE; pass++) {
                const int local_col  = pass * ROWS_PER_PASS + row_in_warp;
                const int global_col = tile_n + local_col;

                r_colwise_vals[pass][0] = r_colwise_vals[pass][1] = r_colwise_vals[pass][2] =
                    r_colwise_vals[pass][3]                       = 0.f;
                r_colwise_amax[pass]                              = 0.f;

                if (global_col < N) {
                    r_colwise_vals[pass][0] =
                        uint16_to_float<kIshalf>(s_tile[warp_id][row_base][local_col]);
                    r_colwise_vals[pass][1] =
                        uint16_to_float<kIshalf>(s_tile[warp_id][row_base + 1][local_col]);
                    r_colwise_vals[pass][2] =
                        uint16_to_float<kIshalf>(s_tile[warp_id][row_base + 2][local_col]);
                    r_colwise_vals[pass][3] =
                        uint16_to_float<kIshalf>(s_tile[warp_id][row_base + 3][local_col]);

                    float local_amax = fmaxf(
                        fmaxf(fabsf(r_colwise_vals[pass][0]), fabsf(r_colwise_vals[pass][1])),
                        fmaxf(fabsf(r_colwise_vals[pass][2]), fabsf(r_colwise_vals[pass][3])));
                    r_colwise_amax[pass] = warp_reduce_max_8_dpp(local_amax);
                }
            }
        }

        // ================================================================
        // Colwise Quantization (Vertical Processing)
        // Step 2: Compute scale — per-col or per-tile (2D Block)
        // ================================================================
        float   r_colwise_scale_native[PASSES_PER_TILE];
        uint8_t r_colwise_scale_e8m0[PASSES_PER_TILE];

        if constexpr (COLWISE_USE_2D_BLOCK) {
            float tile_amax = 0.f;
#pragma unroll
            for (int p = 0; p < PASSES_PER_TILE; p++)
                tile_amax = fmaxf(tile_amax, r_colwise_amax[p]);
            tile_amax = warp_reduce_max_64_dpp(tile_amax);
            float   r_scale_native;
            uint8_t r_scale_e8m0;
            compute_tile_scale<OType>(tile_amax, r_scale_native, r_scale_e8m0);
#pragma unroll
            for (int p = 0; p < PASSES_PER_TILE; p++) {
                r_colwise_scale_native[p] = r_scale_native;
                r_colwise_scale_e8m0[p]   = r_scale_e8m0;
            }
        } else {
#pragma unroll
            for (int p = 0; p < PASSES_PER_TILE; p++)
                compute_tile_scale<OType>(r_colwise_amax[p], r_colwise_scale_native[p],
                                          r_colwise_scale_e8m0[p]);
        }

        // ================================================================
        // Colwise Quantization (Vertical Processing)
        // Step 3: Quantize from regs + Store FP8 / Scale
        // ================================================================
        {
            const int row_base        = thread_in_row * ELEMS_PER_THREAD;
            const int global_row_base = tile_m + row_base;

// Repeat PASSES_PER_TILE times for each warp
#pragma unroll
            for (int pass = 0; pass < PASSES_PER_TILE; pass++) {
                const int local_col  = pass * ROWS_PER_PASS + row_in_warp;
                const int global_col = tile_n + local_col;

                if (global_col < N) {
                    // Convert packed FP32 to FP8
                    uint32_t fp8x4 = cvt_f32x4_to_fp8x4<OType>(
                        r_colwise_vals[pass][0], r_colwise_vals[pass][1], r_colwise_vals[pass][2],
                        r_colwise_vals[pass][3], r_colwise_scale_native[pass]);

                    if (global_row_base < M_pad) {
                        if (shuffle_colwise) {
                            int shuffled_idx = compute_shuffled_index<OType>(
                                global_col, global_row_base, M_packed);
                            *reinterpret_cast<uint32_t *>(colwise_fp8 + shuffled_idx) = fp8x4;
                        } else {
                            *reinterpret_cast<uint32_t *>(colwise_fp8 + global_col * M_packed +
                                                          global_row_base) = fp8x4;
                        }
                    }

                    if (thread_in_row == 0) {
                        int scale_col = block_m * NUM_CHUNKS_M + chunk_m;
                        if (shuffle_colwise_scale) {
                            if (scale_col < colwise_scale_N && global_col < colwise_scale_M_pad &&
                                scale_col < colwise_scale_N_pad) {
                                int idx = compute_shuffle_scale_index(global_col, scale_col,
                                                                      colwise_scale_N_pad);
                                colwise_scale[idx] = r_colwise_scale_e8m0[pass];
                            }
                        } else {
                            if (scale_col < colwise_scale_N) {
                                colwise_scale[global_col * colwise_scale_stride + scale_col] =
                                    r_colwise_scale_e8m0[pass];
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename IType, typename OType>
void quantize_mxfp8_dual_impl(const IType *input, OType *rowwise_output, uint8_t *rowwise_scale,
                              OType *colwise_output, uint8_t *colwise_scale, int M, int N,
                              int M_pad, int N_pad, int rowwise_scale_stride,
                              int colwise_scale_stride, int rowwise_scale_N,
                              int rowwise_scale_M_pad, int rowwise_scale_N_pad, int colwise_scale_M,
                              int colwise_scale_N, int colwise_scale_M_pad, int colwise_scale_N_pad,
                              MXScalingRecipe rowwise_recipe, MXScalingRecipe colwise_recipe,
                              hipStream_t stream) {
    dim3 grid((M_pad + BLOCK_M - 1) / BLOCK_M, (N_pad + BLOCK_N - 1) / BLOCK_N);
    dim3 block(THREADS_PER_BLOCK);

    PRIMUS_TURBO_CHECK(rowwise_recipe.use_rht == false, "MXFP8 not support RHT");
    PRIMUS_TURBO_CHECK(colwise_recipe.use_rht == false, "MXFP8 not support RHT");
    PRIMUS_TURBO_CHECK(rowwise_recipe.use_sr == false, "MXFP8 not support SR");
    PRIMUS_TURBO_CHECK(colwise_recipe.use_sr == false, "MXFP8 not support SR");

#define QUANTIZE_MXFP8_DUAL                                                                        \
    input, rowwise_output, rowwise_scale, colwise_output, colwise_scale, M, N, M_pad, N_pad,       \
        rowwise_scale_stride, colwise_scale_stride, rowwise_scale_N, rowwise_scale_M_pad,          \
        rowwise_scale_N_pad, colwise_scale_M, colwise_scale_N, colwise_scale_M_pad,                \
        colwise_scale_N_pad, rowwise_recipe.shuffle_out, colwise_recipe.shuffle_out,               \
        rowwise_recipe.shuffle_scale, colwise_recipe.shuffle_scale

#define QUANTIZE_MXFP8_DUAL_LAUNCH_KERNEL(ROWWISE_USE_2D_BLOCK, COLWISE_USE_2D_BLOCK)              \
    quantize_mxfp8_dual_kernel<IType, OType, ROWWISE_USE_2D_BLOCK, COLWISE_USE_2D_BLOCK>           \
        <<<grid, block, 0, stream>>>(QUANTIZE_MXFP8_DUAL)

#define DISPATCH_QUANTIZE_MXFP8_DUAL_WITH_2D()                                                     \
    if (rowwise_recipe.use_2d_block) {                                                             \
        if (colwise_recipe.use_2d_block) {                                                         \
            QUANTIZE_MXFP8_DUAL_LAUNCH_KERNEL(true, true);                                         \
        } else {                                                                                   \
            QUANTIZE_MXFP8_DUAL_LAUNCH_KERNEL(true, false);                                        \
        }                                                                                          \
    } else {                                                                                       \
        if (colwise_recipe.use_2d_block) {                                                         \
            QUANTIZE_MXFP8_DUAL_LAUNCH_KERNEL(false, true);                                        \
        } else {                                                                                   \
            QUANTIZE_MXFP8_DUAL_LAUNCH_KERNEL(false, false);                                       \
        }                                                                                          \
    }
    // launch kernel
    DISPATCH_QUANTIZE_MXFP8_DUAL_WITH_2D()

#undef DISPATCH_QUANTIZE_MXFP8_DUAL_WITH_2D
#undef QUANTIZE_MXFP8_DUAL_LAUNCH_KERNEL
#undef QUANTIZE_MXFP8_DUAL
}

template void quantize_mxfp8_dual_impl<dtype::float16, dtype::float8_e5m2>(
    const dtype::float16 *x, dtype::float8_e5m2 *rowwise_output, uint8_t *rowwise_scale,
    dtype::float8_e5m2 *colwise_output, uint8_t *colwise_scale, int M, int N, int M_pad, int N_pad,
    int rowwise_scale_stride, int colwise_scale_stride, int rowwise_scale_N,
    int rowwise_scale_M_pad, int rowwise_scale_N_pad, int colwise_scale_M, int colwise_scale_N,
    int colwise_scale_M_pad, int colwise_scale_N_pad, MXScalingRecipe rowwise_recipe,
    MXScalingRecipe colwise_recipe, hipStream_t stream);
template void quantize_mxfp8_dual_impl<dtype::bfloat16, dtype::float8_e5m2>(
    const dtype::bfloat16 *x, dtype::float8_e5m2 *rowwise_output, uint8_t *rowwise_scale,
    dtype::float8_e5m2 *colwise_output, uint8_t *colwise_scale, int M, int N, int M_pad, int N_pad,
    int rowwise_scale_stride, int colwise_scale_stride, int rowwise_scale_N,
    int rowwise_scale_M_pad, int rowwise_scale_N_pad, int colwise_scale_M, int colwise_scale_N,
    int colwise_scale_M_pad, int colwise_scale_N_pad, MXScalingRecipe rowwise_recipe,
    MXScalingRecipe colwise_recipe, hipStream_t stream);
template void quantize_mxfp8_dual_impl<dtype::float16, dtype::float8_e4m3>(
    const dtype::float16 *x, dtype::float8_e4m3 *rowwise_output, uint8_t *rowwise_scale,
    dtype::float8_e4m3 *colwise_output, uint8_t *colwise_scale, int M, int N, int M_pad, int N_pad,
    int rowwise_scale_stride, int colwise_scale_stride, int rowwise_scale_N,
    int rowwise_scale_M_pad, int rowwise_scale_N_pad, int colwise_scale_M, int colwise_scale_N,
    int colwise_scale_M_pad, int colwise_scale_N_pad, MXScalingRecipe rowwise_recipe,
    MXScalingRecipe colwise_recipe, hipStream_t stream);
template void quantize_mxfp8_dual_impl<dtype::bfloat16, dtype::float8_e4m3>(
    const dtype::bfloat16 *x, dtype::float8_e4m3 *rowwise_output, uint8_t *rowwise_scale,
    dtype::float8_e4m3 *colwise_output, uint8_t *colwise_scale, int M, int N, int M_pad, int N_pad,
    int rowwise_scale_stride, int colwise_scale_stride, int rowwise_scale_N,
    int rowwise_scale_M_pad, int rowwise_scale_N_pad, int colwise_scale_M, int colwise_scale_N,
    int colwise_scale_M_pad, int colwise_scale_N_pad, MXScalingRecipe rowwise_recipe,
    MXScalingRecipe colwise_recipe, hipStream_t stream);

template <typename IType, typename OType>
void quantize_mxfp8_impl(const IType *input, OType *output, uint8_t *scale, QuantizeMode mode,
                         int M, int N, int M_pad, int N_pad, int scale_stride, int scale_N,
                         int scale_M_pad, int scale_N_pad, MXScalingRecipe recipe,
                         hipStream_t stream) {
    dim3 grid((M_pad + BLOCK_M - 1) / BLOCK_M, (N_pad + BLOCK_N - 1) / BLOCK_N);
    dim3 block(THREADS_PER_BLOCK);

    PRIMUS_TURBO_CHECK(recipe.use_rht == false, "MXFP8 not support RHT");
    PRIMUS_TURBO_CHECK(recipe.use_sr == false, "MXFP8 not support SR");

#define QUANTIZE_MXFP8_KERNEL_ARGS                                                                 \
    input, output, scale, M, N, M_pad, N_pad, scale_stride, scale_N, scale_M_pad, scale_N_pad,     \
        recipe.shuffle_out, recipe.shuffle_scale

#define QUANTIZE_MXFP8_LAUNCH_KERNEL(USE_2D_BLOCK)                                                 \
    if (mode == QuantizeMode::ROWWISE) {                                                           \
        quantize_mxfp8_kernel<IType, OType, QuantizeMode::ROWWISE, USE_2D_BLOCK>                   \
            <<<grid, block, 0, stream>>>(QUANTIZE_MXFP8_KERNEL_ARGS);                              \
    } else {                                                                                       \
        quantize_mxfp8_kernel<IType, OType, QuantizeMode::COLWISE, USE_2D_BLOCK>                   \
            <<<grid, block, 0, stream>>>(QUANTIZE_MXFP8_KERNEL_ARGS);                              \
    }

#define DISPATCH_QUANTIZE_MXFP8_WITH_2D()                                                          \
    if (recipe.use_2d_block) {                                                                     \
        QUANTIZE_MXFP8_LAUNCH_KERNEL(true);                                                        \
    } else {                                                                                       \
        QUANTIZE_MXFP8_LAUNCH_KERNEL(false);                                                       \
    }

    // launch kernel
    DISPATCH_QUANTIZE_MXFP8_WITH_2D()

#undef DISPATCH_QUANTIZE_MXFP8_WITH_2D
#undef QUANTIZE_MXFP8_LAUNCH_KERNEL
#undef QUANTIZE_MXFP8_KERNEL_ARGS
}

template void quantize_mxfp8_impl<dtype::float16, dtype::float8_e5m2>(
    const dtype::float16 *x, dtype::float8_e5m2 *output, uint8_t *scale, QuantizeMode mode, int M,
    int N, int M_pad, int N_pad, int scale_stride, int scale_N, int scale_M_pad, int scale_N_pad,
    MXScalingRecipe recipe, hipStream_t stream);
template void quantize_mxfp8_impl<dtype::bfloat16, dtype::float8_e5m2>(
    const dtype::bfloat16 *x, dtype::float8_e5m2 *output, uint8_t *scale, QuantizeMode mode, int M,
    int N, int M_pad, int N_pad, int scale_stride, int scale_N, int scale_M_pad, int scale_N_pad,
    MXScalingRecipe recipe, hipStream_t stream);
template void quantize_mxfp8_impl<dtype::float16, dtype::float8_e4m3>(
    const dtype::float16 *x, dtype::float8_e4m3 *output, uint8_t *scale, QuantizeMode mode, int M,
    int N, int M_pad, int N_pad, int scale_stride, int scale_N, int scale_M_pad, int scale_N_pad,
    MXScalingRecipe recipe, hipStream_t stream);
template void quantize_mxfp8_impl<dtype::bfloat16, dtype::float8_e4m3>(
    const dtype::bfloat16 *x, dtype::float8_e4m3 *output, uint8_t *scale, QuantizeMode mode, int M,
    int N, int M_pad, int N_pad, int scale_stride, int scale_N, int scale_M_pad, int scale_N_pad,
    MXScalingRecipe recipe, hipStream_t stream);

} // namespace primus_turbo
