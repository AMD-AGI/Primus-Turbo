// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

// lds_lat: shader cycles from issue to dscnt-retire of ds_load_b128 / ds_load_tr16_b128 (one wave per SIMD, every
// wave of the CU loading at the same time; expert scheduling mode). B loads back to back, then s_wait_dscnt 0: the
// latency of a burst of B (B = 1: a single load). Median wave.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x)                                                                       \
  do {                                                                                 \
    hipError_t e = (x);                                                                \
    if (e != hipSuccess) {                                                             \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e)); \
      exit(1);                                                                         \
    }                                                                                  \
  } while (0)
constexpr int WAVES = 4, ITERS = 64;

template <bool TR, int OFF>
__device__ __attribute__((always_inline)) void ld(unsigned a) {
  if constexpr (TR)
    asm volatile("ds_load_tr16_b128 v[100:103], %0 offset:%1" ::"v"(a), "i"(OFF) : "v100", "v101", "v102", "v103");
  else
    asm volatile("ds_load_b128 v[100:103], %0 offset:%1" ::"v"(a), "i"(OFF) : "v100", "v101", "v102", "v103");
}

template <int B, bool TR>
__global__ __launch_bounds__(32 * WAVES, 1) void k(unsigned long long* cyc) {
  __shared__ __attribute__((aligned(16))) unsigned char lds[WAVES * 8192];
  const int lane = threadIdx.x % 32, w = threadIdx.x / 32;
  for (int i = threadIdx.x; i < WAVES * 8192 / 4; i += 32 * WAVES) ((int*)lds)[i] = i;
  __syncthreads();
  const unsigned a = (unsigned)(uintptr_t)(lds + w * 8192 + lane * 16);
  unsigned long long tot = 0;
  for (int it = 0; it < ITERS; ++it) {
    unsigned long long t0, t1;
    asm volatile("s_wait_dscnt 0x0\n s_get_shader_cycles_u64 %0" : "=s"(t0)::"memory");
    ld<TR, 0>(a);
    if constexpr (B > 1) ld<TR, 512>(a);
    if constexpr (B > 2) { ld<TR, 1024>(a); ld<TR, 1536>(a); }
    if constexpr (B > 4) { ld<TR, 2048>(a); ld<TR, 2560>(a); ld<TR, 3072>(a); ld<TR, 3584>(a); }
    if constexpr (B > 8) {
      ld<TR, 4096>(a); ld<TR, 4608>(a); ld<TR, 5120>(a); ld<TR, 5632>(a);
      ld<TR, 6144>(a); ld<TR, 6656>(a); ld<TR, 7168>(a); ld<TR, 7680>(a);
    }
    asm volatile("s_wait_dscnt 0x0\n s_get_shader_cycles_u64 %0" : "=s"(t1)::"memory");
    tot += t1 - t0;
  }
  if (lane == 0) cyc[blockIdx.x * WAVES + w] = tot;
}

template <int B, bool TR>
double run(unsigned long long* cyc, int grid) {
  for (int r = 0; r < 3; ++r) k<B, TR><<<grid, 32 * WAVES>>>(cyc);
  CHECK(hipDeviceSynchronize());
  std::vector<unsigned long long> h(grid * WAVES);
  CHECK(hipMemcpy(h.data(), cyc, h.size() * 8, hipMemcpyDeviceToHost));
  std::sort(h.begin(), h.end());
  return (double)h[h.size() / 2] / ITERS;
}

int main() {
  hipDeviceProp_t p;
  CHECK(hipGetDeviceProperties(&p, 0));
  const int grid = p.multiProcessorCount;
  unsigned long long* cyc;
  CHECK(hipMalloc(&cyc, (size_t)grid * WAVES * 8));
  printf("burst  ds_load_b128  ds_load_tr16_b128   (cycles issue -> dscnt 0, median wave)\n");
  printf("1      %8.1f %12.1f\n", run<1, false>(cyc, grid), run<1, true>(cyc, grid));
  printf("2      %8.1f %12.1f\n", run<2, false>(cyc, grid), run<2, true>(cyc, grid));
  printf("4      %8.1f %12.1f\n", run<4, false>(cyc, grid), run<4, true>(cyc, grid));
  printf("8      %8.1f %12.1f\n", run<8, false>(cyc, grid), run<8, true>(cyc, grid));
  printf("16     %8.1f %12.1f\n", run<16, false>(cyc, grid), run<16, true>(cyc, grid));
  return 0;
}
