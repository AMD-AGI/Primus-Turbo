// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

// wmma_chain: issue cost of v_wmma_f32_16x16x32_bf16 when each WMMA accumulates into the C of the WMMA issued
// D slots earlier (D independent accumulators used round robin), one wave per SIMD, expert scheduling mode as the
// MLA kernels (-mllvm -amdgpu-expert-scheduling-mode). Shader cycles per WMMA from s_get_shader_cycles in the kernel
// (clock independent), median over waves.
//
//   wmma_chain            (prints D = 1, 2, 3, 4, 6, 8)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));

#define CHECK(x)                                                                         \
  do {                                                                                   \
    hipError_t e = (x);                                                                  \
    if (e != hipSuccess) {                                                               \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e));   \
      exit(1);                                                                           \
    }                                                                                    \
  } while (0)

constexpr int WAVES = 4;
constexpr int ITERS = 512;
constexpr int PER = 48;  // WMMAs per trip (a multiple of every D)

template <int D>
__global__ __launch_bounds__(32 * WAVES, 1) void k_chain(const bf16x16* __restrict__ src, f32x8* __restrict__ dst,
                                                         unsigned long long* __restrict__ cyc) {
  const int lane = threadIdx.x % 32, gw = blockIdx.x * WAVES + threadIdx.x / 32;
  bf16x16 a[4], b[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    a[i] = src[(gw * 8 + i) * 32 + lane];
    b[i] = src[(gw * 8 + 4 + i) * 32 + lane];
  }
  f32x8 acc[D];
#pragma unroll
  for (int i = 0; i < D; ++i) acc[i] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
  const unsigned long long t0 = __builtin_readcyclecounter();
  for (int it = 0; it < ITERS; ++it) {
#pragma unroll
    for (int j = 0; j < PER; ++j)
      acc[j % D] = __builtin_amdgcn_wmma_f32_16x16x32_bf16(false, a[j % 4], false, b[(j / 4) % 4], (short)0,
                                                          acc[j % D], false, false);
#pragma unroll
    for (int i = 0; i < 4; ++i) asm volatile("" : "+v"(a[i]), "+v"(b[i]));
  }
  const unsigned long long t1 = __builtin_readcyclecounter();
  f32x8 s = acc[0];
#pragma unroll
  for (int i = 1; i < D; ++i) s += acc[i];
  dst[gw * 32 + lane] = s;
  if (lane == 0) cyc[gw] = t1 - t0;
}

template <int D>
double run(const bf16x16* src, f32x8* dst, unsigned long long* cyc, int grid) {
  for (int rep = 0; rep < 3; ++rep) k_chain<D><<<grid, 32 * WAVES>>>(src, dst, cyc);
  CHECK(hipDeviceSynchronize());
  std::vector<unsigned long long> h(grid * WAVES);
  CHECK(hipMemcpy(h.data(), cyc, h.size() * 8, hipMemcpyDeviceToHost));
  std::sort(h.begin(), h.end());
  return (double)h[h.size() / 2] / ((double)ITERS * PER);
}

int main() {
  hipDeviceProp_t p;
  CHECK(hipGetDeviceProperties(&p, 0));
  const int grid = p.multiProcessorCount;  // one 4-wave workgroup per CU: one wave per SIMD
  const size_t nfrag = (size_t)grid * WAVES * 8 * 32;
  std::vector<__bf16> h(nfrag * 16);
  for (size_t i = 0; i < h.size(); ++i) h[i] = (__bf16)((float)((i * 2654435761u) % 2001) / 1000.f - 1.f);
  bf16x16* src;
  f32x8* dst;
  unsigned long long* cyc;
  CHECK(hipMalloc(&src, nfrag * sizeof(bf16x16)));
  CHECK(hipMalloc(&dst, (size_t)grid * WAVES * 32 * sizeof(f32x8)));
  CHECK(hipMalloc(&cyc, (size_t)grid * WAVES * 8));
  CHECK(hipMemcpy(src, h.data(), nfrag * sizeof(bf16x16), hipMemcpyHostToDevice));
  printf("D (WMMAs between a WMMA and the one that writes its C) -> shader cycles per WMMA (median wave)\n");
  printf("D=1 %.2f\n", run<1>(src, dst, cyc, grid));
  printf("D=2 %.2f\n", run<2>(src, dst, cyc, grid));
  printf("D=3 %.2f\n", run<3>(src, dst, cyc, grid));
  printf("D=4 %.2f\n", run<4>(src, dst, cyc, grid));
  printf("D=6 %.2f\n", run<6>(src, dst, cyc, grid));
  printf("D=8 %.2f\n", run<8>(src, dst, cyc, grid));
  return 0;
}
