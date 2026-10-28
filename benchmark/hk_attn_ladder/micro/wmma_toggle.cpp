// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

// wmma_toggle: does the ORDER of WMMA operands change energy (and so time under the power cap)?
//
//   wmma_toggle [seconds per variant (default 3)] [rounds (default 3)]
//
// Pure-register v_wmma_f32_16x16x32_bf16 loop, one wave per SIMD (4-wave workgroups, 2 per CU x 256 CUs), randn
// operands, 8 independent accumulators (no RAW stall between consecutive WMMAs). Variants, same instruction count:
//   share_a : 8 consecutive WMMAs share A (A[j], B[0..7])        -- like dkdv's dV run (A = dO^T over kh)
//   share_b : 8 consecutive WMMAs share B (A[0..7], B[j])        -- like dkdv's dK run / dq's run
//   none    : every WMMA has a new A and a new B (A[i], B[i+j])  -- like dkdv's S / dP pairs
//   zeros   : share-nothing order on all-zero operands            -- the toggling floor
// Each variant runs back to back for the given seconds (the clock settles under the cap), rounds rotate the order.
// Reports ms per launch and the implied clock (WMMA count x 8 clk / time). Memory: one load per operand at the
// start, one store per accumulator at the end; nothing else touches memory.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));

#define CHECK(x)                                                                          \
  do {                                                                                    \
    hipError_t e = (x);                                                                   \
    if (e != hipSuccess) {                                                                \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e));    \
      exit(1);                                                                            \
    }                                                                                     \
  } while (0)

constexpr int NOP = 8;     // operand fragments per side per wave
constexpr int NACC = 8;    // accumulators
constexpr int WAVES = 4;   // per workgroup
constexpr int ITERS = 2048;  // loop trips per launch (64 WMMAs each)

template <int MODE>  // 0 share_a, 1 share_b, 2 none
__global__ __launch_bounds__(32 * WAVES, 1) void k_toggle(const bf16x16* __restrict__ src, f32x8* __restrict__ dst) {
  const int lane = threadIdx.x % 32, wv = threadIdx.x / 32;
  const int gw = blockIdx.x * WAVES + wv;
  bf16x16 a[NOP], b[NOP];
#pragma unroll
  for (int i = 0; i < NOP; ++i) {
    a[i] = src[((gw * 2 * NOP) + i) * 32 + lane];
    b[i] = src[((gw * 2 * NOP) + NOP + i) * 32 + lane];
  }
  f32x8 acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
  for (int it = 0; it < ITERS; ++it) {
#pragma unroll
    for (int j = 0; j < NOP; ++j) {
#pragma unroll
      for (int i = 0; i < NACC; ++i) {
        const bf16x16& A = MODE == 0 ? a[j] : (MODE == 1 ? a[i] : a[i]);
        const bf16x16& B = MODE == 0 ? b[i] : (MODE == 1 ? b[j] : b[(i + j) % NOP]);
        acc[i] = __builtin_amdgcn_wmma_f32_16x16x32_bf16(false, A, false, B, (short)0, acc[i], false, false);
      }
    }
    // keep the operands live and opaque per trip (no hoisting / folding across trips)
#pragma unroll
    for (int i = 0; i < NOP; ++i) asm volatile("" : "+v"(a[i]), "+v"(b[i]));
  }
#pragma unroll
  for (int i = 0; i < NACC; ++i) dst[(gw * NACC + i) * 32 + lane] = acc[i];
}

int main(int argc, char** argv) {
  const double secs = argc > 1 ? atof(argv[1]) : 3.0;
  const int rounds = argc > 2 ? atoi(argv[2]) : 3;
  hipDeviceProp_t p;
  CHECK(hipGetDeviceProperties(&p, 0));
  const int grid = p.multiProcessorCount;  // one 4-wave workgroup per CU: one wave per SIMD
  const size_t nfrag = (size_t)grid * WAVES * 2 * NOP * 32;
  std::vector<__bf16> h(nfrag * 16);
  std::mt19937 rng(0);
  std::normal_distribution<float> nd(0.f, 1.f);
  for (auto& x : h) x = (__bf16)nd(rng);
  bf16x16 *src, *zsrc;
  f32x8* dst;
  CHECK(hipMalloc(&src, nfrag * sizeof(bf16x16)));
  CHECK(hipMalloc(&zsrc, nfrag * sizeof(bf16x16)));
  CHECK(hipMalloc(&dst, (size_t)grid * WAVES * NACC * 32 * sizeof(f32x8)));
  CHECK(hipMemcpy(src, h.data(), nfrag * sizeof(bf16x16), hipMemcpyHostToDevice));
  CHECK(hipMemset(zsrc, 0, nfrag * sizeof(bf16x16)));
  const char* names[4] = {"share_a", "share_b", "none", "zeros"};
  auto launch = [&](int v) {
    if (v == 0) k_toggle<0><<<grid, 32 * WAVES>>>(src, dst);
    if (v == 1) k_toggle<1><<<grid, 32 * WAVES>>>(src, dst);
    if (v == 2) k_toggle<2><<<grid, 32 * WAVES>>>(src, dst);
    if (v == 3) k_toggle<2><<<grid, 32 * WAVES>>>(zsrc, dst);
  };
  const double wmma_per_wave = (double)ITERS * NOP * NACC;
  printf("grid %d x %d waves (1 wave / SIMD), %.0f WMMA per wave per launch, %.1f s per cell\n", grid, WAVES,
         wmma_per_wave, secs);
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  std::vector<std::vector<double>> ms(4);
  for (int r = 0; r < rounds; ++r) {
    for (int k = 0; k < 4; ++k) {
      const int v = (k + r) % 4;
      // settle, then time the second half of the cell
      auto t0 = std::chrono::steady_clock::now();
      int n = 0;
      while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < secs / 2) {
        launch(v);
        if (++n % 16 == 0) CHECK(hipDeviceSynchronize());
      }
      CHECK(hipDeviceSynchronize());
      const int reps = 64;
      CHECK(hipEventRecord(e0));
      for (int i = 0; i < reps; ++i) launch(v);
      CHECK(hipEventRecord(e1));
      CHECK(hipEventSynchronize(e1));
      float t;
      CHECK(hipEventElapsedTime(&t, e0, e1));
      ms[v].push_back(t / reps);
      printf("round %d %-8s %.4f ms/launch  implied %.0f MHz\n", r, names[v], t / reps,
             wmma_per_wave * 8.0 / (t / reps * 1e-3) / 1e6);
      fflush(stdout);
    }
  }
  for (int v = 0; v < 4; ++v) {
    std::vector<double> x = ms[v];
    std::sort(x.begin(), x.end());
    const double med = x[x.size() / 2];
    printf("%-8s median %.4f ms  implied %.0f MHz\n", names[v], med, wmma_per_wave * 8.0 / (med * 1e-3) / 1e6);
  }
  std::vector<double> none = ms[2];
  std::sort(none.begin(), none.end());
  for (int v = 0; v < 4; ++v) {
    std::vector<double> x = ms[v];
    std::sort(x.begin(), x.end());
    printf("speed %-8s / none = %.3fx\n", names[v], none[none.size() / 2] / x[x.size() / 2]);
  }
  return 0;
}
