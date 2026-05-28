// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include <algorithm>

#include "primus_turbo/device/reduce.cuh"
#include "primus_turbo/device/utils.cuh"
#include "primus_turbo/normalization.h"
#include "primus_turbo/normalization_tunables.h"

namespace primus_turbo {

using namespace primus_turbo::dtype;

// Stage 0: each CTA grid-strides over a subset of rows, accumulating its share of
// dgamma in registers and writing one [n_parts, cols] partial slab at the end.
//
// Per row, this kernel reads {x, dy} and the saved rs (no mean-square recompute),
// then writes dx. Block size matches forward (full warps).
//
//   y_i    = rs * x_i
//   dy_g_i = gamma_i * dz_i
//   mdyy   = (1/N) * sum_j(dy_g_j * y_j)
//   dx_i   = rs * (dy_g_i - mdyy * y_i)
//   dg_i   = sum_rows( dz_i * y_i )
//
// Pass 1 loads x and dz once, computes y_regs and dy_g_regs (fp32), accumulates
// dot and dg in registers. After pass 1 the raw bf16 x_regs / dy_regs / gamma_regs
// are dead — only the fp32 y_regs and dy_g_regs live across the BlockReduce
// barrier. This is the TE pass-1-dg-accumulation pattern; it keeps the
// per-thread live state at the reduce barrier minimal (2 fp32 arrays of
// LDGS*UNROLL each + the dg accumulator), which the compiler can satisfy
// without spilling at high LDGS*UNROLL geometries.
template <typename T, int LDGS, int UNROLL>
__global__ void rmsnorm_bwd_stage0_kernel(const T *__restrict__ input, const T *__restrict__ gamma,
                                          const T *__restrict__ grad_out,
                                          const float *__restrict__ rs_in, T *__restrict__ grad_in,
                                          float *__restrict__ dgamma_part, const int64_t inner_len,
                                          const int64_t outer_len) {
    const int     tid          = threadIdx.x;
    const int     blocksize    = blockDim.x;
    const int64_t start_offset = (int64_t) tid * UNROLL;
    const int64_t stride       = (int64_t) blocksize * UNROLL;

    // Load gamma once per CTA lifetime. Kept in bf16/fp16 to save register
    // pressure (each gamma fma-mixes into fp32 in pass 1).
    T gamma_regs[LDGS][UNROLL];
#pragma unroll
    for (int i = 0; i < LDGS; ++i) {
        const int64_t offset = start_offset + (int64_t) i * stride;
        if (offset < inner_len) {
            load_data<T, UNROLL>(gamma + offset, gamma_regs[i]);
        } else {
#pragma unroll
            for (int j = 0; j < UNROLL; ++j)
                gamma_regs[i][j] = T(0);
        }
    }

    // Per-thread fp32 dgamma accumulator across all rows this CTA processes.
    float dg_accum[LDGS][UNROLL];
#pragma unroll
    for (int i = 0; i < LDGS; ++i) {
#pragma unroll
        for (int j = 0; j < UNROLL; ++j)
            dg_accum[i][j] = 0.0f;
    }

    const float inv_n = 1.0f / static_cast<float>(inner_len);

    for (int64_t row = blockIdx.x; row < outer_len; row += gridDim.x) {
        const T    *input_ptr    = input + row * inner_len;
        const T    *grad_out_ptr = grad_out + row * inner_len;
        T          *grad_in_ptr  = grad_in + row * inner_len;
        const float rs           = rs_in[row];

        // Pass 1: load x and dz, materialize y = rs*x and dy_g = gamma*dz in
        // fp32, accumulate local_dot = sum(dy_g * y) and dg += dz * y.
        // x and dz are scope-local to this loop iteration and DO NOT live
        // across the BlockReduce barrier — y_regs and dy_g_regs do.
        float y_regs[LDGS][UNROLL];
        float dy_g_regs[LDGS][UNROLL];
        float local_dot = 0.0f;
        {
            T x_regs[LDGS][UNROLL];
            T dy_regs[LDGS][UNROLL];
#pragma unroll
            for (int i = 0; i < LDGS; ++i) {
                const int64_t offset = start_offset + (int64_t) i * stride;
                if (offset < inner_len) {
                    load_data<T, UNROLL>(input_ptr + offset, x_regs[i]);
                    load_data<T, UNROLL>(grad_out_ptr + offset, dy_regs[i]);
                } else {
#pragma unroll
                    for (int j = 0; j < UNROLL; ++j) {
                        x_regs[i][j]  = T(0);
                        dy_regs[i][j] = T(0);
                    }
                }
            }

#pragma unroll
            for (int i = 0; i < LDGS; ++i) {
#pragma unroll
                for (int j = 0; j < UNROLL; ++j) {
                    const float xv  = static_cast<float>(x_regs[i][j]);
                    const float dzv = static_cast<float>(dy_regs[i][j]);
                    const float gv  = static_cast<float>(gamma_regs[i][j]);
                    const float yv  = rs * xv;
                    const float dyg = gv * dzv;
                    y_regs[i][j]    = yv;
                    dy_g_regs[i][j] = dyg;
                    local_dot += dyg * yv;
                    dg_accum[i][j] += dzv * yv;
                }
            }
        }

        // See fwd kernel: BlockReduce's trailing sync guards the write, not the
        // post-sync read of smem[0]; without the leading barrier a fast warp
        // could overwrite it on the next row.
        __syncthreads();
        const float dot  = BlockReduce<SumOp, float>(local_dot);
        const float mdyy = dot * inv_n;

        // Pass 2: dx = rs * (dy_g - mdyy * y). Touches NO raw bf16 inputs —
        // the bf16 x_regs / dy_regs are out of scope, gamma_regs is unused.
        // The compiler is free to drop those vgprs before this point.
        T dx_regs[UNROLL];
#pragma unroll
        for (int i = 0; i < LDGS; ++i) {
            const int64_t offset = start_offset + (int64_t) i * stride;
            if (offset < inner_len) {
#pragma unroll
                for (int j = 0; j < UNROLL; ++j) {
                    const float dx = rs * (dy_g_regs[i][j] - mdyy * y_regs[i][j]);
                    dx_regs[j]     = static_cast<T>(dx);
                }
                store_data<T, UNROLL>(grad_in_ptr + offset, dx_regs);
            }
        }
    }

    // Write this CTA's partial dgamma into dgamma_part[blockIdx.x, :]. Each
    // thread's UNROLL fp32 lanes are contiguous in dgamma_part; split into
    // 16-byte (= 4 fp32) chunks so the compiler emits global_store_dwordx4.
    float *dg_part_row = dgamma_part + (int64_t) blockIdx.x * inner_len;
#pragma unroll
    for (int i = 0; i < LDGS; ++i) {
        const int64_t offset = start_offset + (int64_t) i * stride;
        if (offset < inner_len) {
            constexpr int CHUNK = 4; // 16 bytes / sizeof(float)
            static_assert(UNROLL % CHUNK == 0 || UNROLL < CHUNK,
                          "UNROLL must be a multiple of CHUNK or smaller than CHUNK");
            if constexpr (UNROLL >= CHUNK) {
#pragma unroll
                for (int j = 0; j < UNROLL; j += CHUNK) {
                    store_data<float, CHUNK>(dg_part_row + offset + j, &dg_accum[i][j]);
                }
            } else {
#pragma unroll
                for (int j = 0; j < UNROLL; ++j)
                    dg_part_row[offset + j] = dg_accum[i][j];
            }
        }
    }
}

// Stage 1: dgamma_part [n_parts, cols] -> dgamma [cols] in T.
//
// Two variants, dispatched by cols:
//
// SMALL (cols < RMSNORM_FINALIZE_COALESCED_THRESHOLD = 2048): warp-per-col
// with 4-way in-lane unroll. Reads are stride-`cols` apart (poorly coalesced)
// but dgamma_part fits in L2 at small cols so the cache hides it. Launches
// cols/4 CTAs → fully saturates 304 CUs even at cols=128.
//
// LARGE (cols >= threshold): TE-style col-tiled layout. CTA = 4 wavefronts ×
// 64 lanes = 256 threads owning 64 consecutive cols. Per warp w in CTA: cover
// a balanced 1/4 slice of n_parts rows, lanes 0..63 read 64 consecutive fp32
// per iter → single coalesced cache line. 4 register accumulators per lane
// (4-way unroll along rows) breaks the serial-add dep chain and preserves
// fp32 precision. Cross-warp reduce via smem[4][64].
//
// The split tracks "cols where dgamma_part stops fitting in L2", which is
// arch-specific (L2 size × num_cus). gfx942 measured at 2048; other archs
// fall back to the gfx942 default until tuned. See
// primus_turbo/normalization_tunables.h and benchmark/ops/tune_rmsnorm.py.

// SMALL-cols variant (warp per col).
template <typename T>
__global__ void rmsnorm_bwd_finalize_kernel(const float *__restrict__ dgamma_part,
                                            T *__restrict__ dgamma_out, const int64_t cols,
                                            const int64_t n_parts) {
    const int     warps_per_block = blockDim.x / THREADS_PER_WARP;
    const int     warp_id         = threadIdx.x / THREADS_PER_WARP;
    const int     lane            = threadIdx.x % THREADS_PER_WARP;
    const int64_t col             = (int64_t) blockIdx.x * warps_per_block + warp_id;
    if (col >= cols)
        return;

    float         a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const int64_t step4 = (int64_t) 4 * THREADS_PER_WARP;
    int64_t       r     = lane;
    for (; r + step4 <= n_parts; r += step4) {
        a0 += dgamma_part[(r + 0 * THREADS_PER_WARP) * cols + col];
        a1 += dgamma_part[(r + 1 * THREADS_PER_WARP) * cols + col];
        a2 += dgamma_part[(r + 2 * THREADS_PER_WARP) * cols + col];
        a3 += dgamma_part[(r + 3 * THREADS_PER_WARP) * cols + col];
    }
    for (; r < n_parts; r += THREADS_PER_WARP) {
        a0 += dgamma_part[r * cols + col];
    }

    float acc = (a0 + a1) + (a2 + a3);
    acc       = WarpReduce<SumOp, float>(acc);

    if (lane == 0) {
        dgamma_out[col] = static_cast<T>(acc);
    }
}

// LARGE-cols variant (col-tiled, coalesced).
template <typename T, int W>
__global__ void rmsnorm_bwd_finalize_coalesced_kernel(const float *__restrict__ dgamma_part,
                                                      T *__restrict__ dgamma_out,
                                                      const int64_t cols, const int64_t n_parts) {
    const int     warp_id = threadIdx.x / THREADS_PER_WARP;
    const int     lane    = threadIdx.x % THREADS_PER_WARP;
    const int64_t col     = (int64_t) blockIdx.x * THREADS_PER_WARP + lane;

    // Balanced split of n_parts across W warps.
    const int64_t rem     = n_parts % W;
    const int64_t base    = (n_parts / W) * warp_id + (warp_id < rem ? warp_id : rem);
    const int64_t my_rows = (n_parts / W) + (warp_id < rem ? 1 : 0);
    const int64_t end     = base + my_rows;

    float      a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const bool active = (col < cols);

    if (active) {
        int64_t r = base;
        for (; r + 4 <= end; r += 4) {
            a0 += dgamma_part[(r + 0) * cols + col];
            a1 += dgamma_part[(r + 1) * cols + col];
            a2 += dgamma_part[(r + 2) * cols + col];
            a3 += dgamma_part[(r + 3) * cols + col];
        }
        for (; r < end; ++r) {
            a0 += dgamma_part[r * cols + col];
        }
    }

    const float warp_sum = (a0 + a1) + (a2 + a3);

    __shared__ float smem[W][THREADS_PER_WARP];
    smem[warp_id][lane] = warp_sum;
    __syncthreads();

    if (warp_id == 0 && active) {
        float acc = 0.0f;
#pragma unroll
        for (int w = 0; w < W; ++w) {
            acc += smem[w][lane];
        }
        dgamma_out[col] = static_cast<T>(acc);
    }
}

// Warp-per-row backward stage 0: each warp processes one row at a time,
// packs WARPS_PER_BLOCK rows per CTA. Each warp accumulates its own dgamma
// partial in registers; at end, all warps in the CTA write into a shared
// dgamma_part slot indexed by blockIdx.x.
//
// To avoid smem inter-warp contention on the reduce, we keep the per-warp
// dg_accum private and at the end have each warp write to its own row of
// dgamma_part — i.e. dgamma_part has WARPS_PER_BLOCK * gridDim.x rows.
template <typename T, int UNROLL, int WARPS_PER_BLOCK>
__global__ void rmsnorm_bwd_stage0_warp_per_row_kernel(
    const T *__restrict__ input, const T *__restrict__ gamma, const T *__restrict__ grad_out,
    const float *__restrict__ rs_in, T *__restrict__ grad_in, float *__restrict__ dgamma_part,
    const int64_t inner_len, const int64_t outer_len) {
    const int     warp_id      = threadIdx.x / THREADS_PER_WARP;
    const int     lane         = threadIdx.x % THREADS_PER_WARP;
    const int64_t start_offset = (int64_t) lane * UNROLL;

    T gamma_regs[UNROLL];
    if (start_offset < inner_len) {
        load_data<T, UNROLL>(gamma + start_offset, gamma_regs);
    } else {
#pragma unroll
        for (int j = 0; j < UNROLL; ++j)
            gamma_regs[j] = T(0);
    }

    float dg_accum[UNROLL];
#pragma unroll
    for (int j = 0; j < UNROLL; ++j)
        dg_accum[j] = 0.0f;

    const float inv_n = 1.0f / static_cast<float>(inner_len);

    const int64_t global_warp_id = (int64_t) blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int64_t total_warps    = (int64_t) gridDim.x * WARPS_PER_BLOCK;

    for (int64_t row = global_warp_id; row < outer_len; row += total_warps) {
        const T    *input_ptr    = input + row * inner_len;
        const T    *grad_out_ptr = grad_out + row * inner_len;
        T          *grad_in_ptr  = grad_in + row * inner_len;
        const float rs           = rs_in[row];

        // Pass 1: materialize y = rs*x and dy_g = gamma*dz in fp32, accumulate
        // dot = sum(dy_g * y) and dg += dz * y. Raw bf16 x/dy die before pass 2.
        float y_regs[UNROLL];
        float dy_g_regs[UNROLL];
        float local_dot = 0.0f;
        if (start_offset < inner_len) {
            T x_regs[UNROLL];
            T dy_regs[UNROLL];
            load_data<T, UNROLL>(input_ptr + start_offset, x_regs);
            load_data<T, UNROLL>(grad_out_ptr + start_offset, dy_regs);
#pragma unroll
            for (int j = 0; j < UNROLL; ++j) {
                const float xv  = static_cast<float>(x_regs[j]);
                const float dzv = static_cast<float>(dy_regs[j]);
                const float gv  = static_cast<float>(gamma_regs[j]);
                const float yv  = rs * xv;
                const float dyg = gv * dzv;
                y_regs[j]       = yv;
                dy_g_regs[j]    = dyg;
                local_dot += dyg * yv;
                dg_accum[j] += dzv * yv;
            }
        } else {
#pragma unroll
            for (int j = 0; j < UNROLL; ++j) {
                y_regs[j]    = 0.0f;
                dy_g_regs[j] = 0.0f;
            }
        }

        local_dot        = WarpReduce<SumOp, float>(local_dot);
        const float mdyy = local_dot * inv_n;

        // Pass 2: dx = rs * (dy_g - mdyy * y). No raw bf16 inputs touched.
        if (start_offset < inner_len) {
            T dx_regs[UNROLL];
#pragma unroll
            for (int j = 0; j < UNROLL; ++j) {
                const float dx = rs * (dy_g_regs[j] - mdyy * y_regs[j]);
                dx_regs[j]     = static_cast<T>(dx);
            }
            store_data<T, UNROLL>(grad_in_ptr + start_offset, dx_regs);
        }
    }

    // Each warp in the CTA writes its own slab into dgamma_part —
    // dgamma_part has shape [gridDim.x * WARPS_PER_BLOCK, cols]. Same chunked
    // store as the block-per-row variant.
    const int64_t out_row = (int64_t) blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (start_offset < inner_len) {
        float        *dst   = dgamma_part + out_row * inner_len + start_offset;
        constexpr int CHUNK = 4;
        if constexpr (UNROLL >= CHUNK) {
#pragma unroll
            for (int j = 0; j < UNROLL; j += CHUNK) {
                store_data<float, CHUNK>(dst + j, &dg_accum[j]);
            }
        } else {
#pragma unroll
            for (int j = 0; j < UNROLL; ++j)
                dst[j] = dg_accum[j];
        }
    }
}

// ---- launch helpers ----

// O2: query real (register/smem-aware) occupancy for the specific kernel
// template we're about to launch. Mirrors TE's
// `cudaOccupancyMaxActiveBlocksPerMultiprocessor` flow at
// `rmsnorm_bwd_semi_cuda_kernel.cu:22-37`. Caller still passes a hint
// `target_ctas` (sized by host's closed-form upper bound, used for
// `dgamma_part` allocation); we clamp grid to the real occupancy so we don't
// inflate `n_parts` past what actually runs concurrently.
template <typename Kernel> static int64_t real_target_ctas(Kernel kernel, int block, int64_t hint) {
    int ctas_per_sm = 0;
    PRIMUS_TURBO_CHECK_HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &ctas_per_sm, reinterpret_cast<const void *>(kernel), block, 0));
    int dev = 0;
    PRIMUS_TURBO_CHECK_HIP(hipGetDevice(&dev));
    int num_cus = 0;
    PRIMUS_TURBO_CHECK_HIP(
        hipDeviceGetAttribute(&num_cus, hipDeviceAttributeMultiprocessorCount, dev));
    int64_t real = (int64_t) num_cus * ctas_per_sm;
    return real < hint ? real : hint;
}

template <typename T, int UNROLL>
static int64_t launch_bwd_stage0(const T *input, const T *gamma, const T *grad_out, const float *rs,
                                 T *grad_in, float *dgamma_part, const int64_t inner_len,
                                 const int64_t outer_len, const int block,
                                 const int64_t target_ctas_hint, hipStream_t stream) {
    const int64_t span = (int64_t) block * UNROLL;
    // The LDGS template chain tops out at LDGS=8 — past that each thread no
    // longer covers the full row. Catch it loudly instead of silently writing
    // garbage. Currently no LLM has inner_len anywhere near this (max is
    // ~32K with block=1024 UNROLL=8 → cap is 64K), so this is a safety net
    // for unusual inputs, not a hot path.
    PRIMUS_TURBO_CHECK(inner_len <= 8 * span,
                       "rmsnorm bwd: inner_len exceeds LDGS=8 capacity for this dtype + block; "
                       "the kernel's LDGS dispatch chain caps at 8 — extend it or pick a larger "
                       "block.");
    int64_t grid64;
    if (inner_len <= span) {
        grid64 =
            real_target_ctas(&rmsnorm_bwd_stage0_kernel<T, 1, UNROLL>, block, target_ctas_hint);
        grid64 = std::min<int64_t>(outer_len, grid64);
        rmsnorm_bwd_stage0_kernel<T, 1, UNROLL><<<static_cast<int>(grid64), block, 0, stream>>>(
            input, gamma, grad_out, rs, grad_in, dgamma_part, inner_len, outer_len);
    } else if (inner_len <= 2 * span) {
        grid64 =
            real_target_ctas(&rmsnorm_bwd_stage0_kernel<T, 2, UNROLL>, block, target_ctas_hint);
        grid64 = std::min<int64_t>(outer_len, grid64);
        rmsnorm_bwd_stage0_kernel<T, 2, UNROLL><<<static_cast<int>(grid64), block, 0, stream>>>(
            input, gamma, grad_out, rs, grad_in, dgamma_part, inner_len, outer_len);
    } else if (inner_len <= 4 * span) {
        grid64 =
            real_target_ctas(&rmsnorm_bwd_stage0_kernel<T, 4, UNROLL>, block, target_ctas_hint);
        grid64 = std::min<int64_t>(outer_len, grid64);
        rmsnorm_bwd_stage0_kernel<T, 4, UNROLL><<<static_cast<int>(grid64), block, 0, stream>>>(
            input, gamma, grad_out, rs, grad_in, dgamma_part, inner_len, outer_len);
    } else {
        grid64 =
            real_target_ctas(&rmsnorm_bwd_stage0_kernel<T, 8, UNROLL>, block, target_ctas_hint);
        grid64 = std::min<int64_t>(outer_len, grid64);
        rmsnorm_bwd_stage0_kernel<T, 8, UNROLL><<<static_cast<int>(grid64), block, 0, stream>>>(
            input, gamma, grad_out, rs, grad_in, dgamma_part, inner_len, outer_len);
    }
    return grid64;
}

// Helper templated on UNROLL — does the actual launch dispatch. Hoisting this
// out of rmsnorm_bwd_stage0_impl lets the public entry pick UNROLL based on
// shape (per-shape dispatch).
template <typename T, int UNROLL>
static int64_t rmsnorm_bwd_stage0_dispatch(const T *input, const T *gamma, const T *grad_out,
                                           const float *rs, T *grad_in, float *dgamma_part,
                                           const int64_t inner_len, const int64_t outer_len,
                                           const int64_t target_ctas, hipStream_t stream) {
    const bool    aligned   = (inner_len % UNROLL == 0);
    const int64_t warp_span = (int64_t) THREADS_PER_WARP * UNROLL;

    // Fast path: cols fit in one warp's vector tile. Pack RMSNORM_WARPS_PER_BLOCK
    // rows per CTA, one row per warp. dgamma_part has
    // gridDim.x * RMSNORM_WARPS_PER_BLOCK rows.
    if (aligned && inner_len <= warp_span) {
        constexpr int W     = RMSNORM_WARPS_PER_BLOCK;
        const int     block = W * THREADS_PER_WARP;

        // ===========================================================
        // BUFFER-OVERRUN INVARIANT — DO NOT VIOLATE
        // ===========================================================
        // `dgamma_part` was sized by the host using the closed-form bound
        // `parts_alloc ≤ num_cus * rmsnorm_effective_ctas_per_cu(block)`,
        // rounded up to a multiple of W. This launcher writes exactly
        // `grid64 * W` rows. Therefore:
        //   target_ctas (host)  MUST equal the closed-form value.
        //   grid64               MUST be bounded by ceil(target_ctas / W).
        //
        // If you ever change this function to use the runtime
        // `hipOccupancyMaxActiveBlocksPerMultiprocessor` ceiling (as
        // `launch_bwd_stage0` does for block-per-row), you MUST also bump
        // the host's `parts_alloc` to use the same runtime bound. The
        // runtime ceiling at block=256 is 8 CTAs/CU on CDNA3 while the
        // closed-form is 4 → using runtime here without bumping parts_alloc
        // overruns the buffer (caught experimentally in iter-2: cols=128
        // bwd jumped 42→82µs as the kernel scribbled past the slab).
        //
        // We don't recompute the closed-form locally because the runtime
        // cost of hipDeviceGetAttribute (~20 µs on this path) dominates the
        // warp-per-row bwd at cols=128 (~25 µs total). Trust the host's
        // target_ctas and protect the invariant via this comment + the
        // static_assert in normalization.h that ties RMSNORM_CTAS_PER_CU to
        // RMSNORM_WARPS_PER_BLOCK.
        const int64_t blocks_needed = (outer_len + W - 1) / W;
        const int64_t max_blocks    = (target_ctas + W - 1) / W;
        const int64_t grid64        = std::min(blocks_needed, max_blocks);
        const int     grid          = static_cast<int>(grid64);
        rmsnorm_bwd_stage0_warp_per_row_kernel<T, UNROLL, W><<<grid, block, 0, stream>>>(
            input, gamma, grad_out, rs, grad_in, dgamma_part, inner_len, outer_len);
        return grid64 * W;
    }

    // Unaligned path uses UNROLL=1; block must still fit the row.
    const int block = rmsnorm_pick_blocksize(inner_len, aligned ? UNROLL : 1);

    int64_t grid64;
    if (aligned) {
        grid64 = launch_bwd_stage0<T, UNROLL>(input, gamma, grad_out, rs, grad_in, dgamma_part,
                                              inner_len, outer_len, block, target_ctas, stream);
    } else {
        grid64 = launch_bwd_stage0<T, 1>(input, gamma, grad_out, rs, grad_in, dgamma_part,
                                         inner_len, outer_len, block, target_ctas, stream);
    }
    return grid64;
}

// Iter-3: per-shape UNROLL dispatch. At medium row widths use HALF the
// natural uint4 LDG width — turns LDGS=1 into LDGS=2 → 2× in-flight HBM
// transactions per thread → better latency hiding on bandwidth-bound bwd.
//
// Range expressed in BYTES PER ROW so dispatch is dtype-agnostic. gfx942
// measured: 8192-16384 bytes (= bf16 cols 4096-8192 or fp32 cols 2048-4096).
// Outside this range:
//   - lower row bytes: warp-per-row regime / small block-reduce — smaller
//     UNROLL halves active lanes → loss.
//   - higher row bytes: LDGS=2 instruction count dominates over latency-
//     hiding gain → loss.
//
// Iter-5: at large outer_len the dispatch flips. block=1024 caps real
// occupancy at 2 CTAs/CU = 608 concurrent CTAs on MI325X; once outer_len
// exceeds ~25 rows/CTA at that occupancy the smaller block=512 +
// UNROLL_FULL path (4 CTAs/CU = 1216 concurrent CTAs) finishes stage 0
// faster despite the shorter LDG burst — stage 0 dominates at high
// outer_len and extra concurrency wins. gfx942 measured the crossover
// between outer_len=16384 (iter-3 still ahead) and 32768 (UNROLL_FULL
// ahead by ~10 µs at C=4096 bf16/fp16); threshold defaults to 16384.
//
// The range and threshold track per-CU register budget and HBM
// bandwidth-to-compute ratio — all arch-specific. See
// primus_turbo/normalization_tunables.h.
template <typename T>
static bool rmsnorm_bwd_use_half_unroll(int64_t inner_len, int64_t outer_len) {
    constexpr int UNROLL_FULL = sizeof(uint4) / sizeof(T);
    if (UNROLL_FULL < 2)
        return false;
    const auto    tun           = rmsnorm_tunables();
    const int64_t bytes_per_row = inner_len * (int64_t) sizeof(T);
    if (bytes_per_row < tun.bwd_half_unroll_lo_bytes ||
        bytes_per_row > tun.bwd_half_unroll_hi_bytes)
        return false;
    if (tun.bwd_half_unroll_outer_len_max > 0 && outer_len > tun.bwd_half_unroll_outer_len_max)
        return false;
    return true;
}

template <typename T>
int64_t rmsnorm_bwd_stage0_impl(const T *input, const T *gamma, const T *grad_out, const float *rs,
                                T *grad_in, float *dgamma_part, const int64_t inner_len,
                                const int64_t outer_len, const int64_t target_ctas,
                                hipStream_t stream) {
    constexpr int UNROLL_FULL = sizeof(uint4) / sizeof(T);
    constexpr int UNROLL_HALF = UNROLL_FULL >= 2 ? UNROLL_FULL / 2 : 1;

    if (rmsnorm_bwd_use_half_unroll<T>(inner_len, outer_len)) {
        return rmsnorm_bwd_stage0_dispatch<T, UNROLL_HALF>(input, gamma, grad_out, rs, grad_in,
                                                           dgamma_part, inner_len, outer_len,
                                                           target_ctas, stream);
    }
    return rmsnorm_bwd_stage0_dispatch<T, UNROLL_FULL>(input, gamma, grad_out, rs, grad_in,
                                                       dgamma_part, inner_len, outer_len,
                                                       target_ctas, stream);
}

template <typename T>
void rmsnorm_bwd_finalize_impl(const float *dgamma_part, T *dgamma, const int64_t cols,
                               const int64_t n_parts, hipStream_t stream) {
    const int threshold = rmsnorm_tunables().finalize_coalesced_threshold;
    if (cols < threshold) {
        // SMALL-cols: warp per column, pack RMSNORM_WARPS_PER_BLOCK cols per CTA.
        constexpr int W     = RMSNORM_WARPS_PER_BLOCK;
        const int     block = W * THREADS_PER_WARP;
        const int     grid  = static_cast<int>((cols + W - 1) / W);
        rmsnorm_bwd_finalize_kernel<T>
            <<<grid, block, 0, stream>>>(dgamma_part, dgamma, cols, n_parts);
    } else {
        // LARGE-cols: TE-style col-tiled, 64 cols per CTA, 4 warps cooperate
        // across rows. Coalesced reads recover the L2/HBM bandwidth that the
        // warp-per-col layout wastes at cols ≥ 2K (dgamma_part > L2).
        constexpr int W     = 4;
        const int     block = W * THREADS_PER_WARP;
        const int     grid  = static_cast<int>((cols + THREADS_PER_WARP - 1) / THREADS_PER_WARP);
        rmsnorm_bwd_finalize_coalesced_kernel<T, W>
            <<<grid, block, 0, stream>>>(dgamma_part, dgamma, cols, n_parts);
    }
}

template int64_t rmsnorm_bwd_stage0_impl<float>(const float *, const float *, const float *,
                                                const float *, float *, float *, const int64_t,
                                                const int64_t, const int64_t, hipStream_t);
template int64_t rmsnorm_bwd_stage0_impl<float16>(const float16 *, const float16 *, const float16 *,
                                                  const float *, float16 *, float *, const int64_t,
                                                  const int64_t, const int64_t, hipStream_t);
template int64_t rmsnorm_bwd_stage0_impl<bfloat16>(const bfloat16 *, const bfloat16 *,
                                                   const bfloat16 *, const float *, bfloat16 *,
                                                   float *, const int64_t, const int64_t,
                                                   const int64_t, hipStream_t);

template void rmsnorm_bwd_finalize_impl<float>(const float *, float *, const int64_t, const int64_t,
                                               hipStream_t);
template void rmsnorm_bwd_finalize_impl<float16>(const float *, float16 *, const int64_t,
                                                 const int64_t, hipStream_t);
template void rmsnorm_bwd_finalize_impl<bfloat16>(const float *, bfloat16 *, const int64_t,
                                                  const int64_t, hipStream_t);

} // namespace primus_turbo
