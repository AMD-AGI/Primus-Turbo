#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef unsigned long long u64;
struct Rec { u64 cyc, rt; unsigned h1, h2; };
#define KEEP(x) asm volatile("" ::"v"(x))

extern "C" __global__ __launch_bounds__(512) void k_base(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_fma1(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_fma_f32 %10, %10, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_fma_f32 %11, %11, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_fma_f32 %12, %12, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_fma_f32 %13, %13, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_fma_f32 %14, %14, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_fma_f32 %15, %15, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_fma_f32 %16, %16, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_fma_f32 %17, %17, %18, %19"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_fma2(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_fma_f32 %10, %10, %18, %19\n\tv_fma_f32 %11, %11, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_fma_f32 %12, %12, %18, %19\n\tv_fma_f32 %13, %13, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_fma_f32 %14, %14, %18, %19\n\tv_fma_f32 %15, %15, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_fma_f32 %16, %16, %18, %19\n\tv_fma_f32 %17, %17, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_fma_f32 %10, %10, %18, %19\n\tv_fma_f32 %11, %11, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_fma_f32 %12, %12, %18, %19\n\tv_fma_f32 %13, %13, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_fma_f32 %14, %14, %18, %19\n\tv_fma_f32 %15, %15, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_fma_f32 %16, %16, %18, %19\n\tv_fma_f32 %17, %17, %18, %19"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_fma4(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_fma_f32 %10, %10, %18, %19\n\tv_fma_f32 %11, %11, %18, %19\n\tv_fma_f32 %12, %12, %18, %19\n\tv_fma_f32 %13, %13, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_fma_f32 %14, %14, %18, %19\n\tv_fma_f32 %15, %15, %18, %19\n\tv_fma_f32 %16, %16, %18, %19\n\tv_fma_f32 %17, %17, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_fma_f32 %10, %10, %18, %19\n\tv_fma_f32 %11, %11, %18, %19\n\tv_fma_f32 %12, %12, %18, %19\n\tv_fma_f32 %13, %13, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_fma_f32 %14, %14, %18, %19\n\tv_fma_f32 %15, %15, %18, %19\n\tv_fma_f32 %16, %16, %18, %19\n\tv_fma_f32 %17, %17, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_fma_f32 %10, %10, %18, %19\n\tv_fma_f32 %11, %11, %18, %19\n\tv_fma_f32 %12, %12, %18, %19\n\tv_fma_f32 %13, %13, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_fma_f32 %14, %14, %18, %19\n\tv_fma_f32 %15, %15, %18, %19\n\tv_fma_f32 %16, %16, %18, %19\n\tv_fma_f32 %17, %17, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_fma_f32 %10, %10, %18, %19\n\tv_fma_f32 %11, %11, %18, %19\n\tv_fma_f32 %12, %12, %18, %19\n\tv_fma_f32 %13, %13, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_fma_f32 %14, %14, %18, %19\n\tv_fma_f32 %15, %15, %18, %19\n\tv_fma_f32 %16, %16, %18, %19\n\tv_fma_f32 %17, %17, %18, %19"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_fma8(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_fma_f32 %10, %10, %18, %19\n\tv_fma_f32 %11, %11, %18, %19\n\tv_fma_f32 %12, %12, %18, %19\n\tv_fma_f32 %13, %13, %18, %19\n\tv_fma_f32 %14, %14, %18, %19\n\tv_fma_f32 %15, %15, %18, %19\n\tv_fma_f32 %16, %16, %18, %19\n\tv_fma_f32 %17, %17, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_fma_f32 %10, %10, %18, %19\n\tv_fma_f32 %11, %11, %18, %19\n\tv_fma_f32 %12, %12, %18, %19\n\tv_fma_f32 %13, %13, %18, %19\n\tv_fma_f32 %14, %14, %18, %19\n\tv_fma_f32 %15, %15, %18, %19\n\tv_fma_f32 %16, %16, %18, %19\n\tv_fma_f32 %17, %17, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_fma_f32 %10, %10, %18, %19\n\tv_fma_f32 %11, %11, %18, %19\n\tv_fma_f32 %12, %12, %18, %19\n\tv_fma_f32 %13, %13, %18, %19\n\tv_fma_f32 %14, %14, %18, %19\n\tv_fma_f32 %15, %15, %18, %19\n\tv_fma_f32 %16, %16, %18, %19\n\tv_fma_f32 %17, %17, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_fma_f32 %10, %10, %18, %19\n\tv_fma_f32 %11, %11, %18, %19\n\tv_fma_f32 %12, %12, %18, %19\n\tv_fma_f32 %13, %13, %18, %19\n\tv_fma_f32 %14, %14, %18, %19\n\tv_fma_f32 %15, %15, %18, %19\n\tv_fma_f32 %16, %16, %18, %19\n\tv_fma_f32 %17, %17, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_fma_f32 %10, %10, %18, %19\n\tv_fma_f32 %11, %11, %18, %19\n\tv_fma_f32 %12, %12, %18, %19\n\tv_fma_f32 %13, %13, %18, %19\n\tv_fma_f32 %14, %14, %18, %19\n\tv_fma_f32 %15, %15, %18, %19\n\tv_fma_f32 %16, %16, %18, %19\n\tv_fma_f32 %17, %17, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_fma_f32 %10, %10, %18, %19\n\tv_fma_f32 %11, %11, %18, %19\n\tv_fma_f32 %12, %12, %18, %19\n\tv_fma_f32 %13, %13, %18, %19\n\tv_fma_f32 %14, %14, %18, %19\n\tv_fma_f32 %15, %15, %18, %19\n\tv_fma_f32 %16, %16, %18, %19\n\tv_fma_f32 %17, %17, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_fma_f32 %10, %10, %18, %19\n\tv_fma_f32 %11, %11, %18, %19\n\tv_fma_f32 %12, %12, %18, %19\n\tv_fma_f32 %13, %13, %18, %19\n\tv_fma_f32 %14, %14, %18, %19\n\tv_fma_f32 %15, %15, %18, %19\n\tv_fma_f32 %16, %16, %18, %19\n\tv_fma_f32 %17, %17, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_fma_f32 %10, %10, %18, %19\n\tv_fma_f32 %11, %11, %18, %19\n\tv_fma_f32 %12, %12, %18, %19\n\tv_fma_f32 %13, %13, %18, %19\n\tv_fma_f32 %14, %14, %18, %19\n\tv_fma_f32 %15, %15, %18, %19\n\tv_fma_f32 %16, %16, %18, %19\n\tv_fma_f32 %17, %17, %18, %19"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_add1(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_add_f32 %10, %10, %19\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_add_f32 %11, %11, %19\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_add_f32 %12, %12, %19\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_add_f32 %13, %13, %19\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_add_f32 %14, %14, %19\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_add_f32 %15, %15, %19\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_add_f32 %16, %16, %19\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_add_f32 %17, %17, %19"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_add2(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_add_f32 %10, %10, %19\n\tv_add_f32 %11, %11, %19\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_add_f32 %12, %12, %19\n\tv_add_f32 %13, %13, %19\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_add_f32 %14, %14, %19\n\tv_add_f32 %15, %15, %19\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_add_f32 %16, %16, %19\n\tv_add_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_add_f32 %10, %10, %19\n\tv_add_f32 %11, %11, %19\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_add_f32 %12, %12, %19\n\tv_add_f32 %13, %13, %19\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_add_f32 %14, %14, %19\n\tv_add_f32 %15, %15, %19\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_add_f32 %16, %16, %19\n\tv_add_f32 %17, %17, %19"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_add4(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_add_f32 %10, %10, %19\n\tv_add_f32 %11, %11, %19\n\tv_add_f32 %12, %12, %19\n\tv_add_f32 %13, %13, %19\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_add_f32 %14, %14, %19\n\tv_add_f32 %15, %15, %19\n\tv_add_f32 %16, %16, %19\n\tv_add_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_add_f32 %10, %10, %19\n\tv_add_f32 %11, %11, %19\n\tv_add_f32 %12, %12, %19\n\tv_add_f32 %13, %13, %19\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_add_f32 %14, %14, %19\n\tv_add_f32 %15, %15, %19\n\tv_add_f32 %16, %16, %19\n\tv_add_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_add_f32 %10, %10, %19\n\tv_add_f32 %11, %11, %19\n\tv_add_f32 %12, %12, %19\n\tv_add_f32 %13, %13, %19\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_add_f32 %14, %14, %19\n\tv_add_f32 %15, %15, %19\n\tv_add_f32 %16, %16, %19\n\tv_add_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_add_f32 %10, %10, %19\n\tv_add_f32 %11, %11, %19\n\tv_add_f32 %12, %12, %19\n\tv_add_f32 %13, %13, %19\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_add_f32 %14, %14, %19\n\tv_add_f32 %15, %15, %19\n\tv_add_f32 %16, %16, %19\n\tv_add_f32 %17, %17, %19"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_add8(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_add_f32 %10, %10, %19\n\tv_add_f32 %11, %11, %19\n\tv_add_f32 %12, %12, %19\n\tv_add_f32 %13, %13, %19\n\tv_add_f32 %14, %14, %19\n\tv_add_f32 %15, %15, %19\n\tv_add_f32 %16, %16, %19\n\tv_add_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_add_f32 %10, %10, %19\n\tv_add_f32 %11, %11, %19\n\tv_add_f32 %12, %12, %19\n\tv_add_f32 %13, %13, %19\n\tv_add_f32 %14, %14, %19\n\tv_add_f32 %15, %15, %19\n\tv_add_f32 %16, %16, %19\n\tv_add_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_add_f32 %10, %10, %19\n\tv_add_f32 %11, %11, %19\n\tv_add_f32 %12, %12, %19\n\tv_add_f32 %13, %13, %19\n\tv_add_f32 %14, %14, %19\n\tv_add_f32 %15, %15, %19\n\tv_add_f32 %16, %16, %19\n\tv_add_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_add_f32 %10, %10, %19\n\tv_add_f32 %11, %11, %19\n\tv_add_f32 %12, %12, %19\n\tv_add_f32 %13, %13, %19\n\tv_add_f32 %14, %14, %19\n\tv_add_f32 %15, %15, %19\n\tv_add_f32 %16, %16, %19\n\tv_add_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_add_f32 %10, %10, %19\n\tv_add_f32 %11, %11, %19\n\tv_add_f32 %12, %12, %19\n\tv_add_f32 %13, %13, %19\n\tv_add_f32 %14, %14, %19\n\tv_add_f32 %15, %15, %19\n\tv_add_f32 %16, %16, %19\n\tv_add_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_add_f32 %10, %10, %19\n\tv_add_f32 %11, %11, %19\n\tv_add_f32 %12, %12, %19\n\tv_add_f32 %13, %13, %19\n\tv_add_f32 %14, %14, %19\n\tv_add_f32 %15, %15, %19\n\tv_add_f32 %16, %16, %19\n\tv_add_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_add_f32 %10, %10, %19\n\tv_add_f32 %11, %11, %19\n\tv_add_f32 %12, %12, %19\n\tv_add_f32 %13, %13, %19\n\tv_add_f32 %14, %14, %19\n\tv_add_f32 %15, %15, %19\n\tv_add_f32 %16, %16, %19\n\tv_add_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_add_f32 %10, %10, %19\n\tv_add_f32 %11, %11, %19\n\tv_add_f32 %12, %12, %19\n\tv_add_f32 %13, %13, %19\n\tv_add_f32 %14, %14, %19\n\tv_add_f32 %15, %15, %19\n\tv_add_f32 %16, %16, %19\n\tv_add_f32 %17, %17, %19"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_mul1(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_mul_f32 %10, %10, %18\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_mul_f32 %11, %11, %18\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_mul_f32 %12, %12, %18\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_mul_f32 %13, %13, %18\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_mul_f32 %14, %14, %18\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_mul_f32 %15, %15, %18\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_mul_f32 %16, %16, %18\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_mul_f32 %17, %17, %18"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_mul2(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_mul_f32 %10, %10, %18\n\tv_mul_f32 %11, %11, %18\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_mul_f32 %12, %12, %18\n\tv_mul_f32 %13, %13, %18\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_mul_f32 %14, %14, %18\n\tv_mul_f32 %15, %15, %18\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_mul_f32 %16, %16, %18\n\tv_mul_f32 %17, %17, %18\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_mul_f32 %10, %10, %18\n\tv_mul_f32 %11, %11, %18\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_mul_f32 %12, %12, %18\n\tv_mul_f32 %13, %13, %18\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_mul_f32 %14, %14, %18\n\tv_mul_f32 %15, %15, %18\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_mul_f32 %16, %16, %18\n\tv_mul_f32 %17, %17, %18"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_mul4(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_mul_f32 %10, %10, %18\n\tv_mul_f32 %11, %11, %18\n\tv_mul_f32 %12, %12, %18\n\tv_mul_f32 %13, %13, %18\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_mul_f32 %14, %14, %18\n\tv_mul_f32 %15, %15, %18\n\tv_mul_f32 %16, %16, %18\n\tv_mul_f32 %17, %17, %18\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_mul_f32 %10, %10, %18\n\tv_mul_f32 %11, %11, %18\n\tv_mul_f32 %12, %12, %18\n\tv_mul_f32 %13, %13, %18\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_mul_f32 %14, %14, %18\n\tv_mul_f32 %15, %15, %18\n\tv_mul_f32 %16, %16, %18\n\tv_mul_f32 %17, %17, %18\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_mul_f32 %10, %10, %18\n\tv_mul_f32 %11, %11, %18\n\tv_mul_f32 %12, %12, %18\n\tv_mul_f32 %13, %13, %18\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_mul_f32 %14, %14, %18\n\tv_mul_f32 %15, %15, %18\n\tv_mul_f32 %16, %16, %18\n\tv_mul_f32 %17, %17, %18\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_mul_f32 %10, %10, %18\n\tv_mul_f32 %11, %11, %18\n\tv_mul_f32 %12, %12, %18\n\tv_mul_f32 %13, %13, %18\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_mul_f32 %14, %14, %18\n\tv_mul_f32 %15, %15, %18\n\tv_mul_f32 %16, %16, %18\n\tv_mul_f32 %17, %17, %18"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_mul8(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_mul_f32 %10, %10, %18\n\tv_mul_f32 %11, %11, %18\n\tv_mul_f32 %12, %12, %18\n\tv_mul_f32 %13, %13, %18\n\tv_mul_f32 %14, %14, %18\n\tv_mul_f32 %15, %15, %18\n\tv_mul_f32 %16, %16, %18\n\tv_mul_f32 %17, %17, %18\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_mul_f32 %10, %10, %18\n\tv_mul_f32 %11, %11, %18\n\tv_mul_f32 %12, %12, %18\n\tv_mul_f32 %13, %13, %18\n\tv_mul_f32 %14, %14, %18\n\tv_mul_f32 %15, %15, %18\n\tv_mul_f32 %16, %16, %18\n\tv_mul_f32 %17, %17, %18\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_mul_f32 %10, %10, %18\n\tv_mul_f32 %11, %11, %18\n\tv_mul_f32 %12, %12, %18\n\tv_mul_f32 %13, %13, %18\n\tv_mul_f32 %14, %14, %18\n\tv_mul_f32 %15, %15, %18\n\tv_mul_f32 %16, %16, %18\n\tv_mul_f32 %17, %17, %18\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_mul_f32 %10, %10, %18\n\tv_mul_f32 %11, %11, %18\n\tv_mul_f32 %12, %12, %18\n\tv_mul_f32 %13, %13, %18\n\tv_mul_f32 %14, %14, %18\n\tv_mul_f32 %15, %15, %18\n\tv_mul_f32 %16, %16, %18\n\tv_mul_f32 %17, %17, %18\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_mul_f32 %10, %10, %18\n\tv_mul_f32 %11, %11, %18\n\tv_mul_f32 %12, %12, %18\n\tv_mul_f32 %13, %13, %18\n\tv_mul_f32 %14, %14, %18\n\tv_mul_f32 %15, %15, %18\n\tv_mul_f32 %16, %16, %18\n\tv_mul_f32 %17, %17, %18\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_mul_f32 %10, %10, %18\n\tv_mul_f32 %11, %11, %18\n\tv_mul_f32 %12, %12, %18\n\tv_mul_f32 %13, %13, %18\n\tv_mul_f32 %14, %14, %18\n\tv_mul_f32 %15, %15, %18\n\tv_mul_f32 %16, %16, %18\n\tv_mul_f32 %17, %17, %18\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_mul_f32 %10, %10, %18\n\tv_mul_f32 %11, %11, %18\n\tv_mul_f32 %12, %12, %18\n\tv_mul_f32 %13, %13, %18\n\tv_mul_f32 %14, %14, %18\n\tv_mul_f32 %15, %15, %18\n\tv_mul_f32 %16, %16, %18\n\tv_mul_f32 %17, %17, %18\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_mul_f32 %10, %10, %18\n\tv_mul_f32 %11, %11, %18\n\tv_mul_f32 %12, %12, %18\n\tv_mul_f32 %13, %13, %18\n\tv_mul_f32 %14, %14, %18\n\tv_mul_f32 %15, %15, %18\n\tv_mul_f32 %16, %16, %18\n\tv_mul_f32 %17, %17, %18"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_max1(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_max_num_f32 %10, %10, %19\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_max_num_f32 %11, %11, %19\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_max_num_f32 %12, %12, %19\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_max_num_f32 %13, %13, %19\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_max_num_f32 %14, %14, %19\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_max_num_f32 %15, %15, %19\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_max_num_f32 %16, %16, %19\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_max_num_f32 %17, %17, %19"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_max2(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_max_num_f32 %10, %10, %19\n\tv_max_num_f32 %11, %11, %19\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_max_num_f32 %12, %12, %19\n\tv_max_num_f32 %13, %13, %19\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_max_num_f32 %14, %14, %19\n\tv_max_num_f32 %15, %15, %19\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_max_num_f32 %16, %16, %19\n\tv_max_num_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_max_num_f32 %10, %10, %19\n\tv_max_num_f32 %11, %11, %19\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_max_num_f32 %12, %12, %19\n\tv_max_num_f32 %13, %13, %19\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_max_num_f32 %14, %14, %19\n\tv_max_num_f32 %15, %15, %19\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_max_num_f32 %16, %16, %19\n\tv_max_num_f32 %17, %17, %19"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_max4(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_max_num_f32 %10, %10, %19\n\tv_max_num_f32 %11, %11, %19\n\tv_max_num_f32 %12, %12, %19\n\tv_max_num_f32 %13, %13, %19\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_max_num_f32 %14, %14, %19\n\tv_max_num_f32 %15, %15, %19\n\tv_max_num_f32 %16, %16, %19\n\tv_max_num_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_max_num_f32 %10, %10, %19\n\tv_max_num_f32 %11, %11, %19\n\tv_max_num_f32 %12, %12, %19\n\tv_max_num_f32 %13, %13, %19\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_max_num_f32 %14, %14, %19\n\tv_max_num_f32 %15, %15, %19\n\tv_max_num_f32 %16, %16, %19\n\tv_max_num_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_max_num_f32 %10, %10, %19\n\tv_max_num_f32 %11, %11, %19\n\tv_max_num_f32 %12, %12, %19\n\tv_max_num_f32 %13, %13, %19\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_max_num_f32 %14, %14, %19\n\tv_max_num_f32 %15, %15, %19\n\tv_max_num_f32 %16, %16, %19\n\tv_max_num_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_max_num_f32 %10, %10, %19\n\tv_max_num_f32 %11, %11, %19\n\tv_max_num_f32 %12, %12, %19\n\tv_max_num_f32 %13, %13, %19\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_max_num_f32 %14, %14, %19\n\tv_max_num_f32 %15, %15, %19\n\tv_max_num_f32 %16, %16, %19\n\tv_max_num_f32 %17, %17, %19"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_max8(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_max_num_f32 %10, %10, %19\n\tv_max_num_f32 %11, %11, %19\n\tv_max_num_f32 %12, %12, %19\n\tv_max_num_f32 %13, %13, %19\n\tv_max_num_f32 %14, %14, %19\n\tv_max_num_f32 %15, %15, %19\n\tv_max_num_f32 %16, %16, %19\n\tv_max_num_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_max_num_f32 %10, %10, %19\n\tv_max_num_f32 %11, %11, %19\n\tv_max_num_f32 %12, %12, %19\n\tv_max_num_f32 %13, %13, %19\n\tv_max_num_f32 %14, %14, %19\n\tv_max_num_f32 %15, %15, %19\n\tv_max_num_f32 %16, %16, %19\n\tv_max_num_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_max_num_f32 %10, %10, %19\n\tv_max_num_f32 %11, %11, %19\n\tv_max_num_f32 %12, %12, %19\n\tv_max_num_f32 %13, %13, %19\n\tv_max_num_f32 %14, %14, %19\n\tv_max_num_f32 %15, %15, %19\n\tv_max_num_f32 %16, %16, %19\n\tv_max_num_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_max_num_f32 %10, %10, %19\n\tv_max_num_f32 %11, %11, %19\n\tv_max_num_f32 %12, %12, %19\n\tv_max_num_f32 %13, %13, %19\n\tv_max_num_f32 %14, %14, %19\n\tv_max_num_f32 %15, %15, %19\n\tv_max_num_f32 %16, %16, %19\n\tv_max_num_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_max_num_f32 %10, %10, %19\n\tv_max_num_f32 %11, %11, %19\n\tv_max_num_f32 %12, %12, %19\n\tv_max_num_f32 %13, %13, %19\n\tv_max_num_f32 %14, %14, %19\n\tv_max_num_f32 %15, %15, %19\n\tv_max_num_f32 %16, %16, %19\n\tv_max_num_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_max_num_f32 %10, %10, %19\n\tv_max_num_f32 %11, %11, %19\n\tv_max_num_f32 %12, %12, %19\n\tv_max_num_f32 %13, %13, %19\n\tv_max_num_f32 %14, %14, %19\n\tv_max_num_f32 %15, %15, %19\n\tv_max_num_f32 %16, %16, %19\n\tv_max_num_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_max_num_f32 %10, %10, %19\n\tv_max_num_f32 %11, %11, %19\n\tv_max_num_f32 %12, %12, %19\n\tv_max_num_f32 %13, %13, %19\n\tv_max_num_f32 %14, %14, %19\n\tv_max_num_f32 %15, %15, %19\n\tv_max_num_f32 %16, %16, %19\n\tv_max_num_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_max_num_f32 %10, %10, %19\n\tv_max_num_f32 %11, %11, %19\n\tv_max_num_f32 %12, %12, %19\n\tv_max_num_f32 %13, %13, %19\n\tv_max_num_f32 %14, %14, %19\n\tv_max_num_f32 %15, %15, %19\n\tv_max_num_f32 %16, %16, %19\n\tv_max_num_f32 %17, %17, %19"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_exp1(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_exp_f32 %10, %10\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_exp_f32 %11, %11\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_exp_f32 %12, %12\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_exp_f32 %13, %13\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_exp_f32 %14, %14\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_exp_f32 %15, %15\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_exp_f32 %16, %16\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_exp_f32 %17, %17"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_exp2(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_exp_f32 %10, %10\n\tv_exp_f32 %11, %11\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_exp_f32 %12, %12\n\tv_exp_f32 %13, %13\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_exp_f32 %14, %14\n\tv_exp_f32 %15, %15\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_exp_f32 %16, %16\n\tv_exp_f32 %17, %17\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_exp_f32 %10, %10\n\tv_exp_f32 %11, %11\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_exp_f32 %12, %12\n\tv_exp_f32 %13, %13\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_exp_f32 %14, %14\n\tv_exp_f32 %15, %15\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_exp_f32 %16, %16\n\tv_exp_f32 %17, %17"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_exp4(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_exp_f32 %10, %10\n\tv_exp_f32 %11, %11\n\tv_exp_f32 %12, %12\n\tv_exp_f32 %13, %13\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_exp_f32 %14, %14\n\tv_exp_f32 %15, %15\n\tv_exp_f32 %16, %16\n\tv_exp_f32 %17, %17\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_exp_f32 %10, %10\n\tv_exp_f32 %11, %11\n\tv_exp_f32 %12, %12\n\tv_exp_f32 %13, %13\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_exp_f32 %14, %14\n\tv_exp_f32 %15, %15\n\tv_exp_f32 %16, %16\n\tv_exp_f32 %17, %17\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_exp_f32 %10, %10\n\tv_exp_f32 %11, %11\n\tv_exp_f32 %12, %12\n\tv_exp_f32 %13, %13\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_exp_f32 %14, %14\n\tv_exp_f32 %15, %15\n\tv_exp_f32 %16, %16\n\tv_exp_f32 %17, %17\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_exp_f32 %10, %10\n\tv_exp_f32 %11, %11\n\tv_exp_f32 %12, %12\n\tv_exp_f32 %13, %13\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_exp_f32 %14, %14\n\tv_exp_f32 %15, %15\n\tv_exp_f32 %16, %16\n\tv_exp_f32 %17, %17"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_exp8(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_exp_f32 %10, %10\n\tv_exp_f32 %11, %11\n\tv_exp_f32 %12, %12\n\tv_exp_f32 %13, %13\n\tv_exp_f32 %14, %14\n\tv_exp_f32 %15, %15\n\tv_exp_f32 %16, %16\n\tv_exp_f32 %17, %17\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_exp_f32 %10, %10\n\tv_exp_f32 %11, %11\n\tv_exp_f32 %12, %12\n\tv_exp_f32 %13, %13\n\tv_exp_f32 %14, %14\n\tv_exp_f32 %15, %15\n\tv_exp_f32 %16, %16\n\tv_exp_f32 %17, %17\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_exp_f32 %10, %10\n\tv_exp_f32 %11, %11\n\tv_exp_f32 %12, %12\n\tv_exp_f32 %13, %13\n\tv_exp_f32 %14, %14\n\tv_exp_f32 %15, %15\n\tv_exp_f32 %16, %16\n\tv_exp_f32 %17, %17\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_exp_f32 %10, %10\n\tv_exp_f32 %11, %11\n\tv_exp_f32 %12, %12\n\tv_exp_f32 %13, %13\n\tv_exp_f32 %14, %14\n\tv_exp_f32 %15, %15\n\tv_exp_f32 %16, %16\n\tv_exp_f32 %17, %17\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_exp_f32 %10, %10\n\tv_exp_f32 %11, %11\n\tv_exp_f32 %12, %12\n\tv_exp_f32 %13, %13\n\tv_exp_f32 %14, %14\n\tv_exp_f32 %15, %15\n\tv_exp_f32 %16, %16\n\tv_exp_f32 %17, %17\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_exp_f32 %10, %10\n\tv_exp_f32 %11, %11\n\tv_exp_f32 %12, %12\n\tv_exp_f32 %13, %13\n\tv_exp_f32 %14, %14\n\tv_exp_f32 %15, %15\n\tv_exp_f32 %16, %16\n\tv_exp_f32 %17, %17\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_exp_f32 %10, %10\n\tv_exp_f32 %11, %11\n\tv_exp_f32 %12, %12\n\tv_exp_f32 %13, %13\n\tv_exp_f32 %14, %14\n\tv_exp_f32 %15, %15\n\tv_exp_f32 %16, %16\n\tv_exp_f32 %17, %17\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_exp_f32 %10, %10\n\tv_exp_f32 %11, %11\n\tv_exp_f32 %12, %12\n\tv_exp_f32 %13, %13\n\tv_exp_f32 %14, %14\n\tv_exp_f32 %15, %15\n\tv_exp_f32 %16, %16\n\tv_exp_f32 %17, %17"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_cvt1(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_cvt_pk_bf16_f32 %10, %10, %19\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_cvt_pk_bf16_f32 %11, %11, %19\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_cvt_pk_bf16_f32 %12, %12, %19\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_cvt_pk_bf16_f32 %13, %13, %19\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_cvt_pk_bf16_f32 %14, %14, %19\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_cvt_pk_bf16_f32 %15, %15, %19\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_cvt_pk_bf16_f32 %16, %16, %19\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_cvt_pk_bf16_f32 %17, %17, %19"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_cvt2(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_cvt_pk_bf16_f32 %10, %10, %19\n\tv_cvt_pk_bf16_f32 %11, %11, %19\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_cvt_pk_bf16_f32 %12, %12, %19\n\tv_cvt_pk_bf16_f32 %13, %13, %19\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_cvt_pk_bf16_f32 %14, %14, %19\n\tv_cvt_pk_bf16_f32 %15, %15, %19\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_cvt_pk_bf16_f32 %16, %16, %19\n\tv_cvt_pk_bf16_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_cvt_pk_bf16_f32 %10, %10, %19\n\tv_cvt_pk_bf16_f32 %11, %11, %19\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_cvt_pk_bf16_f32 %12, %12, %19\n\tv_cvt_pk_bf16_f32 %13, %13, %19\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_cvt_pk_bf16_f32 %14, %14, %19\n\tv_cvt_pk_bf16_f32 %15, %15, %19\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_cvt_pk_bf16_f32 %16, %16, %19\n\tv_cvt_pk_bf16_f32 %17, %17, %19"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_cvt4(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_cvt_pk_bf16_f32 %10, %10, %19\n\tv_cvt_pk_bf16_f32 %11, %11, %19\n\tv_cvt_pk_bf16_f32 %12, %12, %19\n\tv_cvt_pk_bf16_f32 %13, %13, %19\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_cvt_pk_bf16_f32 %14, %14, %19\n\tv_cvt_pk_bf16_f32 %15, %15, %19\n\tv_cvt_pk_bf16_f32 %16, %16, %19\n\tv_cvt_pk_bf16_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_cvt_pk_bf16_f32 %10, %10, %19\n\tv_cvt_pk_bf16_f32 %11, %11, %19\n\tv_cvt_pk_bf16_f32 %12, %12, %19\n\tv_cvt_pk_bf16_f32 %13, %13, %19\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_cvt_pk_bf16_f32 %14, %14, %19\n\tv_cvt_pk_bf16_f32 %15, %15, %19\n\tv_cvt_pk_bf16_f32 %16, %16, %19\n\tv_cvt_pk_bf16_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_cvt_pk_bf16_f32 %10, %10, %19\n\tv_cvt_pk_bf16_f32 %11, %11, %19\n\tv_cvt_pk_bf16_f32 %12, %12, %19\n\tv_cvt_pk_bf16_f32 %13, %13, %19\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_cvt_pk_bf16_f32 %14, %14, %19\n\tv_cvt_pk_bf16_f32 %15, %15, %19\n\tv_cvt_pk_bf16_f32 %16, %16, %19\n\tv_cvt_pk_bf16_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_cvt_pk_bf16_f32 %10, %10, %19\n\tv_cvt_pk_bf16_f32 %11, %11, %19\n\tv_cvt_pk_bf16_f32 %12, %12, %19\n\tv_cvt_pk_bf16_f32 %13, %13, %19\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_cvt_pk_bf16_f32 %14, %14, %19\n\tv_cvt_pk_bf16_f32 %15, %15, %19\n\tv_cvt_pk_bf16_f32 %16, %16, %19\n\tv_cvt_pk_bf16_f32 %17, %17, %19"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_cvt8(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_cvt_pk_bf16_f32 %10, %10, %19\n\tv_cvt_pk_bf16_f32 %11, %11, %19\n\tv_cvt_pk_bf16_f32 %12, %12, %19\n\tv_cvt_pk_bf16_f32 %13, %13, %19\n\tv_cvt_pk_bf16_f32 %14, %14, %19\n\tv_cvt_pk_bf16_f32 %15, %15, %19\n\tv_cvt_pk_bf16_f32 %16, %16, %19\n\tv_cvt_pk_bf16_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_cvt_pk_bf16_f32 %10, %10, %19\n\tv_cvt_pk_bf16_f32 %11, %11, %19\n\tv_cvt_pk_bf16_f32 %12, %12, %19\n\tv_cvt_pk_bf16_f32 %13, %13, %19\n\tv_cvt_pk_bf16_f32 %14, %14, %19\n\tv_cvt_pk_bf16_f32 %15, %15, %19\n\tv_cvt_pk_bf16_f32 %16, %16, %19\n\tv_cvt_pk_bf16_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_cvt_pk_bf16_f32 %10, %10, %19\n\tv_cvt_pk_bf16_f32 %11, %11, %19\n\tv_cvt_pk_bf16_f32 %12, %12, %19\n\tv_cvt_pk_bf16_f32 %13, %13, %19\n\tv_cvt_pk_bf16_f32 %14, %14, %19\n\tv_cvt_pk_bf16_f32 %15, %15, %19\n\tv_cvt_pk_bf16_f32 %16, %16, %19\n\tv_cvt_pk_bf16_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_cvt_pk_bf16_f32 %10, %10, %19\n\tv_cvt_pk_bf16_f32 %11, %11, %19\n\tv_cvt_pk_bf16_f32 %12, %12, %19\n\tv_cvt_pk_bf16_f32 %13, %13, %19\n\tv_cvt_pk_bf16_f32 %14, %14, %19\n\tv_cvt_pk_bf16_f32 %15, %15, %19\n\tv_cvt_pk_bf16_f32 %16, %16, %19\n\tv_cvt_pk_bf16_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_cvt_pk_bf16_f32 %10, %10, %19\n\tv_cvt_pk_bf16_f32 %11, %11, %19\n\tv_cvt_pk_bf16_f32 %12, %12, %19\n\tv_cvt_pk_bf16_f32 %13, %13, %19\n\tv_cvt_pk_bf16_f32 %14, %14, %19\n\tv_cvt_pk_bf16_f32 %15, %15, %19\n\tv_cvt_pk_bf16_f32 %16, %16, %19\n\tv_cvt_pk_bf16_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_cvt_pk_bf16_f32 %10, %10, %19\n\tv_cvt_pk_bf16_f32 %11, %11, %19\n\tv_cvt_pk_bf16_f32 %12, %12, %19\n\tv_cvt_pk_bf16_f32 %13, %13, %19\n\tv_cvt_pk_bf16_f32 %14, %14, %19\n\tv_cvt_pk_bf16_f32 %15, %15, %19\n\tv_cvt_pk_bf16_f32 %16, %16, %19\n\tv_cvt_pk_bf16_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_cvt_pk_bf16_f32 %10, %10, %19\n\tv_cvt_pk_bf16_f32 %11, %11, %19\n\tv_cvt_pk_bf16_f32 %12, %12, %19\n\tv_cvt_pk_bf16_f32 %13, %13, %19\n\tv_cvt_pk_bf16_f32 %14, %14, %19\n\tv_cvt_pk_bf16_f32 %15, %15, %19\n\tv_cvt_pk_bf16_f32 %16, %16, %19\n\tv_cvt_pk_bf16_f32 %17, %17, %19\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_cvt_pk_bf16_f32 %10, %10, %19\n\tv_cvt_pk_bf16_f32 %11, %11, %19\n\tv_cvt_pk_bf16_f32 %12, %12, %19\n\tv_cvt_pk_bf16_f32 %13, %13, %19\n\tv_cvt_pk_bf16_f32 %14, %14, %19\n\tv_cvt_pk_bf16_f32 %15, %15, %19\n\tv_cvt_pk_bf16_f32 %16, %16, %19\n\tv_cvt_pk_bf16_f32 %17, %17, %19"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_pkfma1(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  f32x2 y[8]; f32x2 m = {0.999f, 0.998f}, d = {1e-4f * threadIdx.x, 2e-4f}; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = f32x2{1.0f + k, 1e-3f * threadIdx.x}; }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_pk_fma_f32 %10, %10, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_pk_fma_f32 %11, %11, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_pk_fma_f32 %12, %12, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_pk_fma_f32 %13, %13, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_pk_fma_f32 %14, %14, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_pk_fma_f32 %15, %15, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_pk_fma_f32 %16, %16, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_pk_fma_f32 %17, %17, %18, %19"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k][0];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_pkfma2(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  f32x2 y[8]; f32x2 m = {0.999f, 0.998f}, d = {1e-4f * threadIdx.x, 2e-4f}; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = f32x2{1.0f + k, 1e-3f * threadIdx.x}; }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_pk_fma_f32 %10, %10, %18, %19\n\tv_pk_fma_f32 %11, %11, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_pk_fma_f32 %12, %12, %18, %19\n\tv_pk_fma_f32 %13, %13, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_pk_fma_f32 %14, %14, %18, %19\n\tv_pk_fma_f32 %15, %15, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_pk_fma_f32 %16, %16, %18, %19\n\tv_pk_fma_f32 %17, %17, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_pk_fma_f32 %10, %10, %18, %19\n\tv_pk_fma_f32 %11, %11, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_pk_fma_f32 %12, %12, %18, %19\n\tv_pk_fma_f32 %13, %13, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_pk_fma_f32 %14, %14, %18, %19\n\tv_pk_fma_f32 %15, %15, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_pk_fma_f32 %16, %16, %18, %19\n\tv_pk_fma_f32 %17, %17, %18, %19"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k][0];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_pkfma4(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  f32x2 y[8]; f32x2 m = {0.999f, 0.998f}, d = {1e-4f * threadIdx.x, 2e-4f}; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = f32x2{1.0f + k, 1e-3f * threadIdx.x}; }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_pk_fma_f32 %10, %10, %18, %19\n\tv_pk_fma_f32 %11, %11, %18, %19\n\tv_pk_fma_f32 %12, %12, %18, %19\n\tv_pk_fma_f32 %13, %13, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_pk_fma_f32 %14, %14, %18, %19\n\tv_pk_fma_f32 %15, %15, %18, %19\n\tv_pk_fma_f32 %16, %16, %18, %19\n\tv_pk_fma_f32 %17, %17, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_pk_fma_f32 %10, %10, %18, %19\n\tv_pk_fma_f32 %11, %11, %18, %19\n\tv_pk_fma_f32 %12, %12, %18, %19\n\tv_pk_fma_f32 %13, %13, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_pk_fma_f32 %14, %14, %18, %19\n\tv_pk_fma_f32 %15, %15, %18, %19\n\tv_pk_fma_f32 %16, %16, %18, %19\n\tv_pk_fma_f32 %17, %17, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_pk_fma_f32 %10, %10, %18, %19\n\tv_pk_fma_f32 %11, %11, %18, %19\n\tv_pk_fma_f32 %12, %12, %18, %19\n\tv_pk_fma_f32 %13, %13, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_pk_fma_f32 %14, %14, %18, %19\n\tv_pk_fma_f32 %15, %15, %18, %19\n\tv_pk_fma_f32 %16, %16, %18, %19\n\tv_pk_fma_f32 %17, %17, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_pk_fma_f32 %10, %10, %18, %19\n\tv_pk_fma_f32 %11, %11, %18, %19\n\tv_pk_fma_f32 %12, %12, %18, %19\n\tv_pk_fma_f32 %13, %13, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_pk_fma_f32 %14, %14, %18, %19\n\tv_pk_fma_f32 %15, %15, %18, %19\n\tv_pk_fma_f32 %16, %16, %18, %19\n\tv_pk_fma_f32 %17, %17, %18, %19"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k][0];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_pkfma8(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  f32x2 y[8]; f32x2 m = {0.999f, 0.998f}, d = {1e-4f * threadIdx.x, 2e-4f}; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = f32x2{1.0f + k, 1e-3f * threadIdx.x}; }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_pk_fma_f32 %10, %10, %18, %19\n\tv_pk_fma_f32 %11, %11, %18, %19\n\tv_pk_fma_f32 %12, %12, %18, %19\n\tv_pk_fma_f32 %13, %13, %18, %19\n\tv_pk_fma_f32 %14, %14, %18, %19\n\tv_pk_fma_f32 %15, %15, %18, %19\n\tv_pk_fma_f32 %16, %16, %18, %19\n\tv_pk_fma_f32 %17, %17, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_pk_fma_f32 %10, %10, %18, %19\n\tv_pk_fma_f32 %11, %11, %18, %19\n\tv_pk_fma_f32 %12, %12, %18, %19\n\tv_pk_fma_f32 %13, %13, %18, %19\n\tv_pk_fma_f32 %14, %14, %18, %19\n\tv_pk_fma_f32 %15, %15, %18, %19\n\tv_pk_fma_f32 %16, %16, %18, %19\n\tv_pk_fma_f32 %17, %17, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_pk_fma_f32 %10, %10, %18, %19\n\tv_pk_fma_f32 %11, %11, %18, %19\n\tv_pk_fma_f32 %12, %12, %18, %19\n\tv_pk_fma_f32 %13, %13, %18, %19\n\tv_pk_fma_f32 %14, %14, %18, %19\n\tv_pk_fma_f32 %15, %15, %18, %19\n\tv_pk_fma_f32 %16, %16, %18, %19\n\tv_pk_fma_f32 %17, %17, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_pk_fma_f32 %10, %10, %18, %19\n\tv_pk_fma_f32 %11, %11, %18, %19\n\tv_pk_fma_f32 %12, %12, %18, %19\n\tv_pk_fma_f32 %13, %13, %18, %19\n\tv_pk_fma_f32 %14, %14, %18, %19\n\tv_pk_fma_f32 %15, %15, %18, %19\n\tv_pk_fma_f32 %16, %16, %18, %19\n\tv_pk_fma_f32 %17, %17, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_pk_fma_f32 %10, %10, %18, %19\n\tv_pk_fma_f32 %11, %11, %18, %19\n\tv_pk_fma_f32 %12, %12, %18, %19\n\tv_pk_fma_f32 %13, %13, %18, %19\n\tv_pk_fma_f32 %14, %14, %18, %19\n\tv_pk_fma_f32 %15, %15, %18, %19\n\tv_pk_fma_f32 %16, %16, %18, %19\n\tv_pk_fma_f32 %17, %17, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_pk_fma_f32 %10, %10, %18, %19\n\tv_pk_fma_f32 %11, %11, %18, %19\n\tv_pk_fma_f32 %12, %12, %18, %19\n\tv_pk_fma_f32 %13, %13, %18, %19\n\tv_pk_fma_f32 %14, %14, %18, %19\n\tv_pk_fma_f32 %15, %15, %18, %19\n\tv_pk_fma_f32 %16, %16, %18, %19\n\tv_pk_fma_f32 %17, %17, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_pk_fma_f32 %10, %10, %18, %19\n\tv_pk_fma_f32 %11, %11, %18, %19\n\tv_pk_fma_f32 %12, %12, %18, %19\n\tv_pk_fma_f32 %13, %13, %18, %19\n\tv_pk_fma_f32 %14, %14, %18, %19\n\tv_pk_fma_f32 %15, %15, %18, %19\n\tv_pk_fma_f32 %16, %16, %18, %19\n\tv_pk_fma_f32 %17, %17, %18, %19\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_pk_fma_f32 %10, %10, %18, %19\n\tv_pk_fma_f32 %11, %11, %18, %19\n\tv_pk_fma_f32 %12, %12, %18, %19\n\tv_pk_fma_f32 %13, %13, %18, %19\n\tv_pk_fma_f32 %14, %14, %18, %19\n\tv_pk_fma_f32 %15, %15, %18, %19\n\tv_pk_fma_f32 %16, %16, %18, %19\n\tv_pk_fma_f32 %17, %17, %18, %19"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k][0];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_pkmul1(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  f32x2 y[8]; f32x2 m = {0.999f, 0.998f}, d = {1e-4f * threadIdx.x, 2e-4f}; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = f32x2{1.0f + k, 1e-3f * threadIdx.x}; }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_pk_mul_f32 %10, %10, %18\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_pk_mul_f32 %11, %11, %18\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_pk_mul_f32 %12, %12, %18\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_pk_mul_f32 %13, %13, %18\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_pk_mul_f32 %14, %14, %18\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_pk_mul_f32 %15, %15, %18\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_pk_mul_f32 %16, %16, %18\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_pk_mul_f32 %17, %17, %18"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k][0];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_pkmul2(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  f32x2 y[8]; f32x2 m = {0.999f, 0.998f}, d = {1e-4f * threadIdx.x, 2e-4f}; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = f32x2{1.0f + k, 1e-3f * threadIdx.x}; }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_pk_mul_f32 %10, %10, %18\n\tv_pk_mul_f32 %11, %11, %18\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_pk_mul_f32 %12, %12, %18\n\tv_pk_mul_f32 %13, %13, %18\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_pk_mul_f32 %14, %14, %18\n\tv_pk_mul_f32 %15, %15, %18\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_pk_mul_f32 %16, %16, %18\n\tv_pk_mul_f32 %17, %17, %18\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_pk_mul_f32 %10, %10, %18\n\tv_pk_mul_f32 %11, %11, %18\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_pk_mul_f32 %12, %12, %18\n\tv_pk_mul_f32 %13, %13, %18\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_pk_mul_f32 %14, %14, %18\n\tv_pk_mul_f32 %15, %15, %18\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_pk_mul_f32 %16, %16, %18\n\tv_pk_mul_f32 %17, %17, %18"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k][0];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_pkmul4(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  f32x2 y[8]; f32x2 m = {0.999f, 0.998f}, d = {1e-4f * threadIdx.x, 2e-4f}; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = f32x2{1.0f + k, 1e-3f * threadIdx.x}; }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_pk_mul_f32 %10, %10, %18\n\tv_pk_mul_f32 %11, %11, %18\n\tv_pk_mul_f32 %12, %12, %18\n\tv_pk_mul_f32 %13, %13, %18\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_pk_mul_f32 %14, %14, %18\n\tv_pk_mul_f32 %15, %15, %18\n\tv_pk_mul_f32 %16, %16, %18\n\tv_pk_mul_f32 %17, %17, %18\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_pk_mul_f32 %10, %10, %18\n\tv_pk_mul_f32 %11, %11, %18\n\tv_pk_mul_f32 %12, %12, %18\n\tv_pk_mul_f32 %13, %13, %18\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_pk_mul_f32 %14, %14, %18\n\tv_pk_mul_f32 %15, %15, %18\n\tv_pk_mul_f32 %16, %16, %18\n\tv_pk_mul_f32 %17, %17, %18\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_pk_mul_f32 %10, %10, %18\n\tv_pk_mul_f32 %11, %11, %18\n\tv_pk_mul_f32 %12, %12, %18\n\tv_pk_mul_f32 %13, %13, %18\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_pk_mul_f32 %14, %14, %18\n\tv_pk_mul_f32 %15, %15, %18\n\tv_pk_mul_f32 %16, %16, %18\n\tv_pk_mul_f32 %17, %17, %18\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_pk_mul_f32 %10, %10, %18\n\tv_pk_mul_f32 %11, %11, %18\n\tv_pk_mul_f32 %12, %12, %18\n\tv_pk_mul_f32 %13, %13, %18\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_pk_mul_f32 %14, %14, %18\n\tv_pk_mul_f32 %15, %15, %18\n\tv_pk_mul_f32 %16, %16, %18\n\tv_pk_mul_f32 %17, %17, %18"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k][0];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_pkmul8(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  f32x2 y[8]; f32x2 m = {0.999f, 0.998f}, d = {1e-4f * threadIdx.x, 2e-4f}; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = f32x2{1.0f + k, 1e-3f * threadIdx.x}; }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_pk_mul_f32 %10, %10, %18\n\tv_pk_mul_f32 %11, %11, %18\n\tv_pk_mul_f32 %12, %12, %18\n\tv_pk_mul_f32 %13, %13, %18\n\tv_pk_mul_f32 %14, %14, %18\n\tv_pk_mul_f32 %15, %15, %18\n\tv_pk_mul_f32 %16, %16, %18\n\tv_pk_mul_f32 %17, %17, %18\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_pk_mul_f32 %10, %10, %18\n\tv_pk_mul_f32 %11, %11, %18\n\tv_pk_mul_f32 %12, %12, %18\n\tv_pk_mul_f32 %13, %13, %18\n\tv_pk_mul_f32 %14, %14, %18\n\tv_pk_mul_f32 %15, %15, %18\n\tv_pk_mul_f32 %16, %16, %18\n\tv_pk_mul_f32 %17, %17, %18\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_pk_mul_f32 %10, %10, %18\n\tv_pk_mul_f32 %11, %11, %18\n\tv_pk_mul_f32 %12, %12, %18\n\tv_pk_mul_f32 %13, %13, %18\n\tv_pk_mul_f32 %14, %14, %18\n\tv_pk_mul_f32 %15, %15, %18\n\tv_pk_mul_f32 %16, %16, %18\n\tv_pk_mul_f32 %17, %17, %18\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_pk_mul_f32 %10, %10, %18\n\tv_pk_mul_f32 %11, %11, %18\n\tv_pk_mul_f32 %12, %12, %18\n\tv_pk_mul_f32 %13, %13, %18\n\tv_pk_mul_f32 %14, %14, %18\n\tv_pk_mul_f32 %15, %15, %18\n\tv_pk_mul_f32 %16, %16, %18\n\tv_pk_mul_f32 %17, %17, %18\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_pk_mul_f32 %10, %10, %18\n\tv_pk_mul_f32 %11, %11, %18\n\tv_pk_mul_f32 %12, %12, %18\n\tv_pk_mul_f32 %13, %13, %18\n\tv_pk_mul_f32 %14, %14, %18\n\tv_pk_mul_f32 %15, %15, %18\n\tv_pk_mul_f32 %16, %16, %18\n\tv_pk_mul_f32 %17, %17, %18\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_pk_mul_f32 %10, %10, %18\n\tv_pk_mul_f32 %11, %11, %18\n\tv_pk_mul_f32 %12, %12, %18\n\tv_pk_mul_f32 %13, %13, %18\n\tv_pk_mul_f32 %14, %14, %18\n\tv_pk_mul_f32 %15, %15, %18\n\tv_pk_mul_f32 %16, %16, %18\n\tv_pk_mul_f32 %17, %17, %18\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_pk_mul_f32 %10, %10, %18\n\tv_pk_mul_f32 %11, %11, %18\n\tv_pk_mul_f32 %12, %12, %18\n\tv_pk_mul_f32 %13, %13, %18\n\tv_pk_mul_f32 %14, %14, %18\n\tv_pk_mul_f32 %15, %15, %18\n\tv_pk_mul_f32 %16, %16, %18\n\tv_pk_mul_f32 %17, %17, %18\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_pk_mul_f32 %10, %10, %18\n\tv_pk_mul_f32 %11, %11, %18\n\tv_pk_mul_f32 %12, %12, %18\n\tv_pk_mul_f32 %13, %13, %18\n\tv_pk_mul_f32 %14, %14, %18\n\tv_pk_mul_f32 %15, %15, %18\n\tv_pk_mul_f32 %16, %16, %18\n\tv_pk_mul_f32 %17, %17, %18"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k][0];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_nop1(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_nop"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_nop2(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_nop\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_nop\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_nop\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_nop\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_nop\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_nop\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_nop\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_nop\n\tv_nop"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_nop4(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_nop8(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop\n\tv_nop"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_salu1(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\ts_add_co_i32 %20, %20, 1"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_salu2(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_salu4(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}

extern "C" __global__ __launch_bounds__(512) void k_salu8(float* out, const bf16x16* in, int iters, Rec* rec) {
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  float y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x; int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) { y[k] = -1.0f - 1e-3f * (threadIdx.x + k); }
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {
    asm volatile("v_wmma_f32_16x16x32_bf16 %0, %8, %9, %0\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %1, %8, %9, %1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %2, %8, %9, %2\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %3, %8, %9, %3\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %4, %8, %9, %4\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %5, %8, %9, %5\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %6, %8, %9, %6\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\tv_wmma_f32_16x16x32_bf16 %7, %8, %9, %7\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1\n\ts_add_co_i32 %20, %20, 1"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{t1 - t0, r1 - r0, h1, 0};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIPERR %s line %d\n", hipGetErrorString(e_), __LINE__); return 3; } } while (0)
typedef void (*KFn)(float*, const bf16x16*, int, Rec*);
static u64 med(std::vector<u64> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
int main(int argc, char** argv) {
  bool toy = argc > 1 && !strcmp(argv[1], "toy");
  struct C { const char* n; KFn f; } cs[] = { {"base", k_base}, {"fma1", k_fma1}, {"fma2", k_fma2}, {"fma4", k_fma4}, {"fma8", k_fma8}, {"add1", k_add1}, {"add2", k_add2}, {"add4", k_add4}, {"add8", k_add8}, {"mul1", k_mul1}, {"mul2", k_mul2}, {"mul4", k_mul4}, {"mul8", k_mul8}, {"max1", k_max1}, {"max2", k_max2}, {"max4", k_max4}, {"max8", k_max8}, {"exp1", k_exp1}, {"exp2", k_exp2}, {"exp4", k_exp4}, {"exp8", k_exp8}, {"cvt1", k_cvt1}, {"cvt2", k_cvt2}, {"cvt4", k_cvt4}, {"cvt8", k_cvt8}, {"pkfma1", k_pkfma1}, {"pkfma2", k_pkfma2}, {"pkfma4", k_pkfma4}, {"pkfma8", k_pkfma8}, {"pkmul1", k_pkmul1}, {"pkmul2", k_pkmul2}, {"pkmul4", k_pkmul4}, {"pkmul8", k_pkmul8}, {"nop1", k_nop1}, {"nop2", k_nop2}, {"nop4", k_nop4}, {"nop8", k_nop8}, {"salu1", k_salu1}, {"salu2", k_salu2}, {"salu4", k_salu4}, {"salu8", k_salu8} };
  const size_t LDS = 200 * 1024;
  std::vector<__bf16> hin(1024 * 16); for (size_t i = 0; i < hin.size(); i++) hin[i] = (__bf16)((i * 37 % 101) / 101.f - 0.5f);
  bf16x16* din; CK(hipMalloc(&din, hin.size() * 2)); CK(hipMemcpy(din, hin.data(), hin.size() * 2, hipMemcpyHostToDevice));
  float* dout; CK(hipMalloc(&dout, 256 * 512 * 4)); Rec* drec; CK(hipMalloc(&drec, 256 * 16 * sizeof(Rec)));
  hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
  for (auto& c : cs) {
    CK(hipFuncSetAttribute((const void*)c.f, hipFuncAttributeMaxDynamicSharedMemorySize, (int)LDS));
    for (int wps = 1; wps <= 2; wps++) {
      if (toy && wps == 2) continue;
      int grid = toy ? 1 : 256, block = 128 * wps, nw = grid * block / 32, iters = toy ? 16 : 8192 / wps;
      std::vector<u64> cy, rt; std::vector<float> ev; std::vector<Rec> h(nw);
      for (int rep = 0; rep < 5; rep++) {
        CK(hipMemset(drec, 0, nw * sizeof(Rec)));
        CK(hipEventRecord(e0)); hipLaunchKernelGGL(c.f, dim3(grid), dim3(block), LDS, 0, dout, din, iters, drec);
        CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1)); CK(hipGetLastError());
        float ms; CK(hipEventElapsedTime(&ms, e0, e1));
        CK(hipMemcpy(h.data(), drec, nw * sizeof(Rec), hipMemcpyDeviceToHost));
        std::vector<u64> a, b; int miss = 0; for (auto& r : h) { if (!r.cyc) miss++; a.push_back(r.cyc); b.push_back(r.rt); }
        if (miss) printf("BAD %s missing=%d\n", c.n, miss);
        if (rep >= 1) { cy.push_back(med(a)); rt.push_back(med(b)); ev.push_back(ms); }
      }
      std::sort(ev.begin(), ev.end());
      u64 C = med(cy), R = med(rt); double nwmma = (double)iters * 8;
      // cycles per WMMA per SIMD: wps waves share a SIMD and each wave's span covers all of them (checked: p50 span)
      printf("MB2 %s wps=%d ev_ms=%.4f cyc=%llu f_rt=%.0f cyc_per_wmma_simd=%.3f\n", c.n, wps, ev[ev.size() / 2], C, (double)C / R * 100, (double)C / (nwmma * wps));
      fflush(stdout);
    }
  }
  printf("MBDONE\n"); return 0;
}
