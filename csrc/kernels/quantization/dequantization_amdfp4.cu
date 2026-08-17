// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
#include "primus_turbo/arch.h"
#include "primus_turbo/common.h"
#include "primus_turbo/device/utils.cuh"
#include "primus_turbo/quantization.h"

namespace primus_turbo {

using namespace primus_turbo::dtype;
using namespace primus_turbo::detail;

namespace {

// Each block processes one BLOCK_M x BLOCK_N tile; THREADS_PER_BLOCK threads are
// laid out as blockDim (BLOCK_M, THREADS_PER_BLOCK / BLOCK_M). BLOCK_M == 64 so a
// full 64-lane wavefront covers one tile row (128B coalesced transposed writes /
// 64B coalesced reads in the colwise transpose).
constexpr int THREADS_PER_BLOCK = 512; // 8 warps per block
constexpr int BLOCK_M           = 64;  // rows per block (tile height)
constexpr int BLOCK_N           = 64;  // cols per block (tile width)

// Decoder half of the ScaleCodec pair used by the quantizer: maps a block scale
// byte back to FP32. ``ONE_CODE`` is the encoding of 1.0, used as the neutral
// scale for tile elements that fall outside the scale array.
template <ScaleType SCALE_TYPE> struct ScaleDecoder;

// AMDFP4: exponent bias 15, so 1.0 is exponent field 15 with a zero mantissa.
template <> struct ScaleDecoder<ScaleType::E5M3> {
    static constexpr uint8_t ONE_CODE = E5M3_EXPONENT_BIAS << E5M3_MANTISSA_BITS;

    __device__ __forceinline__ static float decode(uint8_t code) { return e5m3_to_scale(code); }
};

// NVFP4: same layout with exponent bias 7.
template <> struct ScaleDecoder<ScaleType::E4M3> {
    static constexpr uint8_t ONE_CODE = E4M3_EXPONENT_BIAS << E4M3_MANTISSA_BITS;

    __device__ __forceinline__ static float decode(uint8_t code) { return e4m3_to_scale(code); }
};

// Convert two packed FP4 (E2M1) values held in the low byte of ``fp4`` to a
// pair of scaled outputs ``out[0]``, ``out[1]`` (low nibble -> out[0]).
//
// Only AMDFP4 producers -- gfx1250 -- can have written this data, so this
// mirrors the quantizer and stays CDNA5-only rather than carrying an untested
// fallback for archs that cannot generate the format in the first place.
template <typename OType>
__device__ __forceinline__ void cvt_fp4x2_scaled(uint32_t fp4, float scale, OType *out) {
#if defined(__gfx1250__)
    // The scale is applied in registers rather than by the converter: the
    // hardware operand is exponent-only and would drop the mantissa bits an
    // E5M3/E4M3 block scale carries. The PK8 converter also selects its scale
    // cross-lane (lanes 16..31 read lanes 0..15), so running it as a pure
    // FP4->F32 unpack (scale byte = 0x7f == 2^0, opsel 0) is what keeps each
    // thread independent. Only the low 2 nibbles of ``fp4`` are meaningful.
    typedef float     float32x8_t __attribute__((ext_vector_type(8)));
    const float32x8_t r = __builtin_amdgcn_cvt_scale_pk8_f32_fp4(fp4, 0x7f7f7f7fu, 0);
    out[0]              = static_cast<OType>(r[0] * scale);
    out[1]              = static_cast<OType>(r[1] * scale);
#else
    __builtin_trap();
#endif
}

// ---------------------------------------------------------------------------
// AMDFP4 / NVFP4 de-quantization kernel
// ---------------------------------------------------------------------------
template <typename OType, ScaleType SCALE_TYPE, bool USE_ROWWISE>
__global__ void dequantize_amdfp4_kernel(const uint8_t *__restrict__ x, OType *__restrict__ y,
                                         const int64_t stride_x_row, const int64_t stride_y_row,
                                         const int64_t stride_y_col, const int n_rows,
                                         const int n_cols, const uint8_t *__restrict__ scale_inv,
                                         const int64_t stride_scale_row,
                                         const int64_t stride_scale_col, const int scale_n_rows,
                                         const int scale_n_cols, const int block_size) {
    using Decoder = ScaleDecoder<SCALE_TYPE>;

    constexpr int VEC            = 16 / static_cast<int>(sizeof(OType)); // cols per chunk
    constexpr int PACKED         = VEC / 2;                              // packed bytes per chunk
    constexpr int CHUNKS_PER_ROW = BLOCK_N / VEC;                        // VEC-chunks per tile row
    constexpr int TOTAL_CHUNKS   = BLOCK_M * CHUNKS_PER_ROW;
    constexpr int COL_STEP       = THREADS_PER_BLOCK / BLOCK_M; // == blockDim.y

    // Both modes share the same launch: a BLOCK_M x BLOCK_N tile per block,
    // blockDim (BLOCK_M, COL_STEP). Threads walk the tile in VEC-column chunks
    // via a flattened tid. A chunk starts on a VEC boundary and VEC divides the
    // 16-element block, so a chunk never straddles two block scales.
    const int nthreads = blockDim.x * blockDim.y;
    const int tid      = threadIdx.y * blockDim.x + threadIdx.x;
    const int c0       = blockIdx.x * BLOCK_N;
    const int r0       = blockIdx.y * BLOCK_M;

    if constexpr (USE_ROWWISE) {
        // Read VEC contiguous columns + FP4 unpack -> contiguous store.
        for (int ci = tid; ci < TOTAL_CHUNKS; ci += nthreads) {
            const int row = r0 + ci / CHUNKS_PER_ROW;
            const int col = c0 + (ci % CHUNKS_PER_ROW) * VEC;
            if (row < n_rows && col < n_cols) {
                const int col_block  = col / block_size;
                uint8_t   scale_code = Decoder::ONE_CODE;
                if (row < scale_n_rows && col_block < scale_n_cols) {
                    scale_code = scale_inv[static_cast<int64_t>(row) * stride_scale_row +
                                           static_cast<int64_t>(col_block) * stride_scale_col];
                }
                const float scale = Decoder::decode(scale_code);
                uint8_t     x_reg[PACKED];
                OType       y_reg[VEC];
                load_data<uint8_t, PACKED>(
                    x + static_cast<int64_t>(row) * stride_x_row + (col >> 1), x_reg);
#pragma unroll
                for (int p = 0; p < PACKED; ++p) {
                    // low nibble -> even column, high nibble -> odd column
                    cvt_fp4x2_scaled<OType>(static_cast<uint32_t>(x_reg[p]), scale, &y_reg[2 * p]);
                }
                store_data<OType, VEC>(y + static_cast<int64_t>(row) * stride_y_row + col, y_reg);
            }
        }
    } else {
        __shared__ OType s_tile[BLOCK_M][BLOCK_N + 2]; // +2 (even) keeps pair stores aligned

        // Phase 1: vectorized read + FP4 unpack -> row-major smem.
        for (int ci = tid; ci < TOTAL_CHUNKS; ci += nthreads) {
            const int local_row = ci / CHUNKS_PER_ROW;
            const int local_col = (ci % CHUNKS_PER_ROW) * VEC;
            const int grow      = r0 + local_row;
            const int gcol      = c0 + local_col;
            if (grow < n_rows && gcol < n_cols) {
                const int col_block  = gcol / block_size; // VEC cols share one scale
                uint8_t   scale_code = Decoder::ONE_CODE;
                if (grow < scale_n_rows && col_block < scale_n_cols) {
                    scale_code = scale_inv[static_cast<int64_t>(grow) * stride_scale_row +
                                           static_cast<int64_t>(col_block) * stride_scale_col];
                }
                const float scale = Decoder::decode(scale_code);
                uint8_t     pk[PACKED];
                load_data<uint8_t, PACKED>(
                    x + static_cast<int64_t>(grow) * stride_x_row + (gcol >> 1), pk);
#pragma unroll
                for (int p = 0; p < PACKED; ++p) {
                    // 2 OType -> one aligned 32b/64b store into adjacent smem columns
                    cvt_fp4x2_scaled<OType>(static_cast<uint32_t>(pk[p]), scale,
                                            &s_tile[local_row][local_col + 2 * p]);
                }
            }
        }
        __syncthreads();

        // Phase 2: coalesced transposed write (threadIdx.x -> output row).
        const int tx   = threadIdx.x;
        const int ty   = threadIdx.y;
        const int orow = r0 + tx;
#pragma unroll
        for (int j = 0; j < BLOCK_N; j += COL_STEP) {
            const int ocol = c0 + ty + j;
            if (orow < n_rows && ocol < n_cols) {
                y[static_cast<int64_t>(orow) * stride_y_col +
                  static_cast<int64_t>(ocol) * stride_y_row] = s_tile[tx][ty + j];
            }
        }
    }
}

} // namespace

template <typename OType>
void dequantize_amdfp4_impl(const uint8_t *x, OType *y, const int64_t stride_x_row,
                            const int64_t stride_x_col, const int64_t stride_y_row,
                            const int64_t stride_y_col, const int n_rows, const int n_cols,
                            const uint8_t *scale_inv, const int64_t stride_scale_row,
                            const int64_t stride_scale_col, const int scale_n_rows,
                            const int scale_n_cols, const int block_size,
                            const ScaleType scale_type, const bool use_rowwise,
                            hipStream_t stream) {
    PRIMUS_TURBO_CHECK(is_gfx1250(), "AMDFP4/NVFP4 de-quantization requires gfx1250 (CDNA5)");
    (void) stride_x_col; // packed input is contiguous along columns (stride == 1)
    if (n_rows == 0 || n_cols == 0)
        return;

    // One kernel, one launch geometry for both modes: a BLOCK_M x BLOCK_N tile
    // per block, blockDim (BLOCK_M, THREADS_PER_BLOCK / BLOCK_M).
    const dim3 grid((n_cols + BLOCK_N - 1) / BLOCK_N, (n_rows + BLOCK_M - 1) / BLOCK_M);
    const dim3 block(BLOCK_M, THREADS_PER_BLOCK / BLOCK_M);

#define DEQUANTIZE_AMDFP4_LAUNCH_KERNEL(SCALE_TYPE)                                                \
    if (use_rowwise) {                                                                             \
        dequantize_amdfp4_kernel<OType, SCALE_TYPE, true><<<grid, block, 0, stream>>>(             \
            x, y, stride_x_row, stride_y_row, stride_y_col, n_rows, n_cols, scale_inv,             \
            stride_scale_row, stride_scale_col, scale_n_rows, scale_n_cols, block_size);           \
    } else {                                                                                       \
        dequantize_amdfp4_kernel<OType, SCALE_TYPE, false><<<grid, block, 0, stream>>>(            \
            x, y, stride_x_row, stride_y_row, stride_y_col, n_rows, n_cols, scale_inv,             \
            stride_scale_row, stride_scale_col, scale_n_rows, scale_n_cols, block_size);           \
    }

    switch (scale_type) {
    case ScaleType::E5M3:
        DEQUANTIZE_AMDFP4_LAUNCH_KERNEL(ScaleType::E5M3);
        break;
    case ScaleType::E4M3:
        DEQUANTIZE_AMDFP4_LAUNCH_KERNEL(ScaleType::E4M3);
        break;
    default:
        PRIMUS_TURBO_ERROR("16-element block FP4 dequant supports E5M3 (AMDFP4) or E4M3 (NVFP4) "
                           "scales; E8M0 is MXFP4");
    }

#undef DEQUANTIZE_AMDFP4_LAUNCH_KERNEL
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------
#define DECL_DEQUANT_AMDFP4_INSTANCE(OType)                                                        \
    template void dequantize_amdfp4_impl<OType>(                                                   \
        const uint8_t *x, OType *y, const int64_t stride_x_row, const int64_t stride_x_col,        \
        const int64_t stride_y_row, const int64_t stride_y_col, const int n_rows,                  \
        const int n_cols, const uint8_t *scale_inv, const int64_t stride_scale_row,                \
        const int64_t stride_scale_col, const int scale_n_rows, const int scale_n_cols,            \
        const int block_size, const ScaleType scale_type, const bool use_rowwise,                  \
        hipStream_t stream);

DECL_DEQUANT_AMDFP4_INSTANCE(dtype::float16)
DECL_DEQUANT_AMDFP4_INSTANCE(dtype::bfloat16)
DECL_DEQUANT_AMDFP4_INSTANCE(dtype::float32)

#undef DECL_DEQUANT_AMDFP4_INSTANCE

} // namespace primus_turbo
