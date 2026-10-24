/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k_delta -- the backward's row constants, all [B, H, S] fp32: delta = rowsum(dO * O), and what k_dkdv96ds reads,
// NL = -LOG2E * lse and ND = -scale * delta. Bitwise equal to FlyDSL k_delta_bshd (same per-row arithmetic).
// dO / O: [B, S, H, 128] bf16 contiguous; lse [B, H, S] fp32 contiguous (natural log).
// grid (S / 64, B * H), block 256 = 16 rows x 16 lanes: workgroup (sblk, bh) owns queries [64 sblk, 64 sblk + 64)
// of batch-major head bh (FlyDSL's mapping with 64 queries instead of 32), so each output is one 256-B run. Pass p
// (0..3) loads query 64 sblk + 16 p + (tid >> 4); lane l = tid & 15 loads elements [8 l, 8 l + 8) of its dO and O
// rows (16 B each). All 8 loads are issued before any arithmetic.
// Arithmetic, as k_delta_bshd: two chains per lane, e0 += dO[2c] O[2c] and e1 += dO[2c+1] O[2c+1] (c = 0..3, from
// 0), acc = e0 + e1, then acc += shfl_xor(acc, 1 / 2 / 4 / 8). FlyDSL issues separate multiplies and adds; a fused
// multiply-add would round differently once a bf16 x bf16 product underflows into the fp32 denormal range, so fp
// contraction is off here. fp32 denormals are preserved, as in FlyDSL (.amdhsa_float_denorm_mode_32 3).
// Bounds: impl launches it only with S % 64 == 0 (the grid covers [0, S) exactly), and impl._check keeps the dO / O
// byte extent B*S*H*256 below 2^31.
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {
constexpr int kQPW = 64;                 // queries per workgroup
constexpr int kLanes = 16;               // lanes per row: 16 x 8 bf16 = D_V 128
constexpr int kRows = 256 / kLanes;      // rows per pass
constexpr int kPasses = kQPW / kRows;
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float bf_lo(unsigned w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(unsigned w) { return __uint_as_float(w & 0xFFFF0000u); }
}  // namespace

extern "C" __global__ __launch_bounds__(256) void k_delta(const u32x4* __restrict__ DO, const u32x4* __restrict__ O,
                                                          float* __restrict__ DEL, const float* __restrict__ LSE,
                                                          float* __restrict__ NL, float* __restrict__ ND,
                                                          float scale, int S, int H) {
#pragma clang fp contract(off)
  __shared__ float sums[kQPW];
  const int t = threadIdx.x, lr = t & (kLanes - 1), rg = t / kLanes;
  const int s0 = blockIdx.x * kQPW, bh = blockIdx.y, b = bh / H, h = bh - b * H;
  u32x4 vd[kPasses], vo[kPasses];
#pragma unroll
  for (int p = 0; p < kPasses; ++p) {
    const long long e = (((long long)b * S + s0 + p * kRows + rg) * H + h) * kLanes + lr;
    vd[p] = DO[e];
    vo[p] = O[e];
  }
#pragma unroll
  for (int p = 0; p < kPasses; ++p) {
    float e0 = 0.0f, e1 = 0.0f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const unsigned wd = vd[p][c], wo = vo[p][c];
      e0 = e0 + bf_lo(wd) * bf_lo(wo);
      e1 = e1 + bf_hi(wd) * bf_hi(wo);
    }
    float acc = e0 + e1;
    acc = acc + __shfl_xor(acc, 1, 32);
    acc = acc + __shfl_xor(acc, 2, 32);
    acc = acc + __shfl_xor(acc, 4, 32);
    acc = acc + __shfl_xor(acc, 8, 32);
    if (lr == 0) sums[p * kRows + rg] = acc;
  }
  __syncthreads();
  const float nsc = 0.0f - scale;
  const float nlog2e = (float)(-1.4426950408889634);  // FlyDSL's fx.Float32(-LOG2E)
  if (t < kQPW) {
    const long long o = (long long)bh * S + s0 + t;
    const float v = sums[t];
    DEL[o] = v;
    ND[o] = v * nsc;
    NL[o] = LSE[o] * nlog2e;
  }
}
