/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * See LICENSE for license information.
 **************************************************************************************************/

#include "primus_turbo/deoscillation.h"
#include "primus_turbo/device/utils.cuh"
#include "primus_turbo/quantization.h"

namespace primus_turbo {
namespace {

// A wave owns a complete 32x32 tile. No other block can read a weight that
// this wave snaps, so QDQ and closure may safely share a kernel launch.
template <bool Seed, bool Close, bool Count, bool Packed>
__global__ void
weight_deosc_qdq_kernel(float *__restrict__ master, dtype::bfloat16 *__restrict__ previous,
                        dtype::bfloat16 *__restrict__ previous_qdq, float *__restrict__ dist,
                        float *__restrict__ dist_qdq, unsigned long long *__restrict__ reset_count,
                        int64_t numel, int64_t rows, int64_t cols, int64_t start,
                        int64_t first_tile_row, int64_t tile_row_count, int rounding_bias,
                        float ratio_threshold, float eps, bool grouped) {
#if defined(__gfx950__)
    constexpr int kWave     = 64;
    const int     lane      = threadIdx.x % kWave;
    const int64_t tile      = static_cast<int64_t>(blockIdx.x) * 4 + threadIdx.x / kWave;
    const int64_t tile_cols = (cols + 31) / 32;
    if (tile >= tile_row_count * tile_cols)
        return; // Partial final blocks use only wave-local count reduction.
    const int64_t tile_row             = first_tile_row + tile / tile_cols;
    const int64_t tile_rows_per_matrix = (rows + 31) / 32;
    const int64_t matrix               = tile_row / tile_rows_per_matrix;
    const int64_t row_base             = (tile_row % tile_rows_per_matrix) * 32;
    const int64_t col_base             = (tile % tile_cols) * 32;

    float    values[16];
    int64_t  offsets[16];
    uint32_t amax = 0;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        const int     i       = pair * 2;
        const int     element = lane * 2 + pair * 128;
        const int64_t row     = row_base + element / 32;
        const int64_t col     = col_base + element % 32;
        const int64_t offset  = (matrix * rows + row) * cols + col - start;
        if constexpr (Packed) {
            // Host dispatch guarantees even columns/shard bounds and aligned
            // pointers: a pair is either wholly present or wholly absent.
            const bool   valid = row < rows && col < cols && offset >= 0 && offset < numel;
            const float2 input = valid ? *reinterpret_cast<const float2 *>(master + offset)
                                       : make_float2(0.0f, 0.0f);
            offsets[i]         = valid ? offset : -1;
            offsets[i + 1]     = valid ? offset + 1 : -1;
            values[i]          = static_cast<float>(dtype::bfloat16(input.x));
            values[i + 1]      = static_cast<float>(dtype::bfloat16(input.y));
        } else {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const bool valid =
                    row < rows && col + j < cols && offset + j >= 0 && offset + j < numel;
                offsets[i + j] = valid ? offset + j : -1;
                values[i + j] =
                    valid ? static_cast<float>(dtype::bfloat16(master[offset + j])) : 0.0f;
            }
        }
        // Cast before amax and quantization, preserving the forward BF16 grid.
        amax = max(amax, float_as_uint(values[i]) & 0x7fffffffU);
        amax = max(amax, float_as_uint(values[i + 1]) & 0x7fffffffU);
    }
#pragma unroll
    for (int delta = 32; delta > 0; delta >>= 1)
        amax = max(amax, __shfl_xor(amax, delta, kWave));

    if (amax > 0x7f800000U) {
        // Dense HIP fmax ignores NaNs; batched FlyDSL maximumf propagates
        // them. Preserve the existing backend selected for this matrix.
        // Keep this rare path off the finite-input fast path.
        const int64_t matrix_begin = matrix * rows * cols;
        const bool    partial_matrix =
            start > matrix_begin || start + numel < matrix_begin + rows * cols;
        const bool flydsl = grouped && cols % 64 == 0 && (rows % 64 == 0 || partial_matrix);
        if (!flydsl) {
            float finite_max = 0.0f;
#pragma unroll
            for (int i = 0; i < 16; ++i)
                finite_max = fmaxf(finite_max, fabsf(values[i]));
#pragma unroll
            for (int delta = 32; delta > 0; delta >>= 1)
                finite_max = fmaxf(finite_max, __shfl_xor(finite_max, delta, kWave));
            amax = float_as_uint(finite_max);
        }
    }

    // Same exponent rounding and E8M0 decode as quantization_mxfp4.cu.
    const int      exponent = static_cast<int>(((amax + rounding_bias) >> 23) & 511) - 2;
    const uint32_t biased   = static_cast<uint32_t>(max(0, min(255, exponent)));
    const float    scale    = uint_as_float(biased << 23);
    unsigned int   count    = 0;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        const int i          = pair * 2;
        uint32_t  packed_fp4 = 0;
        asm volatile("v_cvt_scalef32_pk_fp4_f32 %0, %1, %2, %3"
                     : "+v"(packed_fp4)
                     : "v"(values[i]), "v"(values[i + 1]), "v"(scale));
        uint32_t packed_bf16;
        asm volatile("v_cvt_scalef32_pk_bf16_fp4 %0, %1, %2 op_sel:[0,0]"
                     : "=v"(packed_bf16)
                     : "v"(packed_fp4), "v"(scale));
        if constexpr (Packed) {
            const int64_t offset = offsets[i];
            if (offset < 0)
                continue;
            float2   distance     = make_float2(0.0f, 0.0f);
            float2   distance_qdq = make_float2(0.0f, 0.0f);
            uint32_t next_prev =
                (float_as_uint(values[i]) >> 16) | (float_as_uint(values[i + 1]) & 0xffff0000U);
            if constexpr (!Seed) {
                const uint32_t prev   = *reinterpret_cast<const uint32_t *>(previous + offset);
                const uint32_t prev_q = *reinterpret_cast<const uint32_t *>(previous_qdq + offset);
                distance              = *reinterpret_cast<const float2 *>(dist + offset);
                distance_qdq          = *reinterpret_cast<const float2 *>(dist_qdq + offset);
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const dtype::bfloat16 current(values[i + j]);
                    const dtype::bfloat16 qdq(
                        uint_as_float(((packed_bf16 >> (16 * j)) & 0xffffU) << 16));
                    const dtype::bfloat16 old(uint_as_float(((prev >> (16 * j)) & 0xffffU) << 16));
                    const dtype::bfloat16 old_q(
                        uint_as_float(((prev_q >> (16 * j)) & 0xffffU) << 16));
                    const dtype::bfloat16 delta   = current - old;
                    const dtype::bfloat16 delta_q = qdq - old_q;
                    float                &d       = j == 0 ? distance.x : distance.y;
                    float                &dq      = j == 0 ? distance_qdq.x : distance_qdq.y;
                    d += fabsf(static_cast<float>(delta));
                    dq += fabsf(static_cast<float>(delta_q));
                    if constexpr (Close) {
                        const float denominator = d < eps ? eps : d;
                        if (d > 0.0f && dq / denominator >= ratio_threshold) {
                            master[offset + j]  = static_cast<float>(qdq);
                            const uint32_t mask = 0xffffU << (16 * j);
                            next_prev           = (next_prev & ~mask) | (packed_bf16 & mask);
                            ++count;
                        }
                    }
                }
            }
            *reinterpret_cast<uint32_t *>(previous + offset)     = next_prev;
            *reinterpret_cast<uint32_t *>(previous_qdq + offset) = packed_bf16;
            *reinterpret_cast<float2 *>(dist + offset) = Close ? make_float2(0.0f, 0.0f) : distance;
            *reinterpret_cast<float2 *>(dist_qdq + offset) =
                Close ? make_float2(0.0f, 0.0f) : distance_qdq;
            continue;
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int64_t offset = offsets[i + j];
            if (offset < 0)
                continue;
            const dtype::bfloat16 current(values[i + j]);
            const dtype::bfloat16 qdq(uint_as_float(((packed_bf16 >> (16 * j)) & 0xffffU) << 16));
            float                 distance     = 0.0f;
            float                 distance_qdq = 0.0f;
            bool                  reset        = false;
            if constexpr (!Seed) {
                // Match the intermediate BF16 subtraction in torch.
                const dtype::bfloat16 delta   = current - previous[offset];
                const dtype::bfloat16 delta_q = qdq - previous_qdq[offset];
                distance                      = dist[offset] + fabsf(static_cast<float>(delta));
                distance_qdq = dist_qdq[offset] + fabsf(static_cast<float>(delta_q));
                if constexpr (Close) {
                    const float denominator = distance < eps ? eps : distance;
                    reset = distance > 0.0f && distance_qdq / denominator >= ratio_threshold;
                    if (reset) {
                        master[offset] = static_cast<float>(qdq);
                        ++count;
                    }
                }
            }
            previous[offset]     = reset ? qdq : current;
            previous_qdq[offset] = qdq;
            dist[offset]         = Close ? 0.0f : distance;
            dist_qdq[offset]     = Close ? 0.0f : distance_qdq;
        }
    }
    if constexpr (Count) {
#pragma unroll
        for (int delta = 32; delta > 0; delta >>= 1)
            count += __shfl_down(count, delta, kWave);
        // Only full blocks enter the barrier: some waves of the final block
        // may have returned above. Combining the four waves reduces pressure
        // on the shared diagnostic counter without a scratch allocation.
        if (static_cast<int64_t>(blockIdx.x) * 4 + 3 < tile_row_count * tile_cols) {
            __shared__ unsigned int wave_counts[4];
            if (lane == 0)
                wave_counts[threadIdx.x / kWave] = count;
            __syncthreads();
            if (threadIdx.x == 0) {
                const unsigned int block_count =
                    wave_counts[0] + wave_counts[1] + wave_counts[2] + wave_counts[3];
                if (block_count)
                    atomicAdd(reset_count, static_cast<unsigned long long>(block_count));
            }
        } else if (lane == 0 && count) {
            atomicAdd(reset_count, static_cast<unsigned long long>(count));
        }
    }
#else
    __builtin_trap(); // The PyTorch entry point rejects unsupported architectures.
#endif
}

} // namespace

void weight_deosc_qdq(float *master, dtype::bfloat16 *previous, dtype::bfloat16 *previous_qdq,
                      float *dist, float *dist_qdq, int64_t *reset_count, int64_t numel,
                      int64_t rows, int64_t cols, int64_t start, int scale_rounding_mode, bool seed,
                      bool close, float ratio_threshold, float eps, bool grouped,
                      hipStream_t stream) {
    if (numel == 0)
        return;
    const int64_t matrix_size          = rows * cols;
    const int64_t tile_rows_per_matrix = (rows + 31) / 32;
    const int64_t first =
        (start / matrix_size) * tile_rows_per_matrix + ((start % matrix_size) / cols) / 32;
    const int64_t last_element = start + numel - 1;
    const int64_t last         = (last_element / matrix_size) * tile_rows_per_matrix +
                         ((last_element % matrix_size) / cols) / 32;
    const int64_t tile_rows = last - first + 1;
    const dim3    grid(DIVUP<int64_t>(tile_rows * DIVUP<int64_t>(cols, 32), 4));
    const int     bias = detail::mxfp4_scale_rounding_bias(scale_rounding_mode);
#define LAUNCH_PACKED(SEED, CLOSE, COUNT, PACKED)                                                  \
    weight_deosc_qdq_kernel<SEED, CLOSE, COUNT, PACKED><<<grid, 256, 0, stream>>>(                 \
        master, previous, previous_qdq, dist, dist_qdq,                                            \
        reinterpret_cast<unsigned long long *>(reset_count), numel, rows, cols, start, first,      \
        tile_rows, bias, ratio_threshold, eps, grouped)
    const bool packed = start % 2 == 0 && cols % 2 == 0 && numel % 2 == 0 &&
                        reinterpret_cast<uintptr_t>(master) % 8 == 0 &&
                        reinterpret_cast<uintptr_t>(previous) % 4 == 0 &&
                        reinterpret_cast<uintptr_t>(previous_qdq) % 4 == 0 &&
                        reinterpret_cast<uintptr_t>(dist) % 8 == 0 &&
                        reinterpret_cast<uintptr_t>(dist_qdq) % 8 == 0;
#define LAUNCH(SEED, CLOSE, COUNT)                                                                 \
    if (packed) {                                                                                  \
        LAUNCH_PACKED(SEED, CLOSE, COUNT, true);                                                   \
    } else {                                                                                       \
        LAUNCH_PACKED(SEED, CLOSE, COUNT, false);                                                  \
    }
    if (seed) {
        LAUNCH(true, false, false);
    } else if (!close) {
        LAUNCH(false, false, false);
    } else if (reset_count) {
        LAUNCH(false, true, true);
    } else {
        LAUNCH(false, true, false);
    }
#undef LAUNCH
#undef LAUNCH_PACKED
    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

} // namespace primus_turbo
