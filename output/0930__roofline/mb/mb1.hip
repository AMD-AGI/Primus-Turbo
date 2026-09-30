// Roofline microbenchmarks, batch 1: register-only compute (WMMA, trans, VALU, co-issue).
// The only memory traffic is reading `in` (fixed small buffer, index < blockDim+128) and one store per lane
// plus one timing record per wave. Every index is in range by construction.
//
// Per wave, lane 0 writes {shader cycles, realtime ticks, HW_ID1, HW_ID2} measured around the timed loop.
// Usage: mb1 [toy]   (toy = 1 WG, few iterations: counter sanity only)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef unsigned long long u64;

#define WMMA(c) c = __builtin_amdgcn_wmma_f32_16x16x32_bf16(false, a, false, b, (short)0, c, false, false)
#define KEEP(x) asm volatile("" ::"v"(x))

struct Rec { u64 cyc, rt; unsigned h1, h2; };

__device__ __forceinline__ void rec_begin(u64& c0, u64& r0) {
  asm volatile("" ::: "memory");
  c0 = __builtin_readcyclecounter();
  r0 = __builtin_readsteadycounter();
}
__device__ __forceinline__ void rec_end(Rec* rec, u64 c0, u64 r0) {
  u64 c1 = __builtin_readcyclecounter();
  u64 r1 = __builtin_readsteadycounter();
  unsigned h1, h2;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID2)" : "=s"(h2));
  if ((threadIdx.x & 31) == 0) {
    unsigned w = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    rec[w] = Rec{c1 - c0, r1 - r0, h1, h2};
  }
}
// Pins one WG per CU: dynamic LDS of 200 KB is requested by the host, touched once here.
__device__ __forceinline__ void touch_lds() {
  extern __shared__ float lds[];
  if (threadIdx.x == 0) lds[0] = 0.f;
}

// ---- M2: WMMA throughput with K independent accumulators (K=1 is the dependent-chain latency) ----
template <int K>
__device__ __forceinline__ void wmma_k_body(float* out, const bf16x16* in, int iters, Rec* rec) {
  touch_lds();
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c[K];
#pragma unroll
  for (int k = 0; k < K; k++) c[k] = f32x8{};
  u64 c0, r0; rec_begin(c0, r0);
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int u = 0; u < 16 / (K < 16 ? K : 16); u++)
#pragma unroll
      for (int k = 0; k < K; k++) WMMA(c[k]);
  }
#pragma unroll
  for (int k = 0; k < K; k++) KEEP(c[k]);
  rec_end(rec, c0, r0);
  f32x8 s = c[0];
#pragma unroll
  for (int k = 1; k < K; k++) s += c[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s[0] + s[7];
}

// ---- M3a: v_exp_f32 throughput, 8 chains, 16 exps per chain-iteration ----
__global__ __launch_bounds__(512) void exp8(float* out, const bf16x16* in, int iters, Rec* rec) {
  touch_lds();
  float x[8];
#pragma unroll
  for (int k = 0; k < 8; k++) x[k] = -1.0f - 0.001f * (threadIdx.x + k);
  u64 c0, r0; rec_begin(c0, r0);
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int u = 0; u < 2; u++)
#pragma unroll
      for (int k = 0; k < 8; k++) x[k] = -__builtin_amdgcn_exp2f(x[k]);  // stays in (-1,0): finite
  }
#pragma unroll
  for (int k = 0; k < 8; k++) KEEP(x[k]);
  rec_end(rec, c0, r0);
  float s = 0; for (int k = 0; k < 8; k++) s += x[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// ---- M3b: VALU fp32 FMA (v_fma_f32) and packed (v_pk_fma_f32), 8 chains, 16 ops per iteration ----
__global__ __launch_bounds__(512) void fma8(float* out, const bf16x16* in, int iters, Rec* rec) {
  touch_lds();
  float x[8]; float m = 0.999f, d = 1e-4f * threadIdx.x;
#pragma unroll
  for (int k = 0; k < 8; k++) x[k] = 1.0f + k;
  u64 c0, r0; rec_begin(c0, r0);
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int u = 0; u < 2; u++)
#pragma unroll
      for (int k = 0; k < 8; k++) asm volatile("v_fma_f32 %0, %0, %1, %2" : "+v"(x[k]) : "v"(m), "v"(d));
  }
#pragma unroll
  for (int k = 0; k < 8; k++) KEEP(x[k]);
  rec_end(rec, c0, r0);
  float s = 0; for (int k = 0; k < 8; k++) s += x[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
__global__ __launch_bounds__(512) void pkfma8(float* out, const bf16x16* in, int iters, Rec* rec) {
  touch_lds();
  f32x2 x[8]; f32x2 m = {0.999f, 0.998f}, d = {1e-4f * threadIdx.x, 2e-4f};
#pragma unroll
  for (int k = 0; k < 8; k++) x[k] = f32x2{1.0f + k, 2.0f + k};
  u64 c0, r0; rec_begin(c0, r0);
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int u = 0; u < 2; u++)
#pragma unroll
      for (int k = 0; k < 8; k++) x[k] = __builtin_elementwise_fma(x[k], m, d);
  }
#pragma unroll
  for (int k = 0; k < 8; k++) KEEP(x[k]);
  rec_end(rec, c0, r0);
  f32x2 s = {}; for (int k = 0; k < 8; k++) s += x[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s[0] + s[1];
}

// ---- M3c: in-wave co-issue. 8 WMMA chains; after every WMMA, E v_exp_f32 and V v_fma_f32 (independent) ----
// sched_group_barrier forces the interleave: 1 WMMA, then E trans, then V VALU.
template <int E, int V>
__device__ __forceinline__ void mix_body(float* out, const bf16x16* in, int iters, Rec* rec) {
  touch_lds();
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c[8];
#pragma unroll
  for (int k = 0; k < 8; k++) c[k] = f32x8{};
  float x[8], y[8]; float m = 0.999f, d = 1e-4f * threadIdx.x;
#pragma unroll
  for (int k = 0; k < 8; k++) { x[k] = -1.0f - 0.001f * (threadIdx.x + k); y[k] = 1.0f + k + 1e-3f * threadIdx.x; }
  u64 c0, r0; rec_begin(c0, r0);
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int u = 0; u < 2; u++) {
#pragma unroll
      for (int k = 0; k < 8; k++) {
        WMMA(c[k]);
#pragma unroll
        for (int e = 0; e < E; e++) x[(k + e) & 7] = -__builtin_amdgcn_exp2f(x[(k + e) & 7]);
#pragma unroll
        for (int v = 0; v < V; v++) y[(k + v) & 7] = __builtin_fmaf(y[(k + v) & 7], m, d);
      }
#pragma unroll
      for (int k = 0; k < 8; k++) {
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
        if (E) __builtin_amdgcn_sched_group_barrier(0x400, E, 0);
        if (V) __builtin_amdgcn_sched_group_barrier(0x002, V, 0);
      }
    }
  }
#pragma unroll
  for (int k = 0; k < 8; k++) { KEEP(c[k]); KEEP(x[k]); KEEP(y[k]); }
  rec_end(rec, c0, r0);
  f32x8 s = c[0];
#pragma unroll
  for (int k = 1; k < 8; k++) s += c[k];
  float t = 0; for (int k = 0; k < 8; k++) t += x[k] + y[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s[0] + t;
}

// ---- M3d: cross-wave co-issue. Waves with (wave_in_wg / 4) odd run exp8's body, even ones WMMA (8 chains). ----
// With 8 waves/WG and 4 SIMDs, waves w and w+4 are expected on the same SIMD (checked from HW_ID).
__global__ __launch_bounds__(512) void roles(float* out, const bf16x16* in, int iters, Rec* rec) {
  touch_lds();
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  int role = ((threadIdx.x >> 5) >> 2) & 1;
  f32x8 c[8];
#pragma unroll
  for (int k = 0; k < 8; k++) c[k] = f32x8{};
  float x[8];
#pragma unroll
  for (int k = 0; k < 8; k++) x[k] = -1.0f - 0.001f * (threadIdx.x + k);
  u64 c0, r0; rec_begin(c0, r0);
  if (role == 0) {
    for (int i = 0; i < iters; i++)
#pragma unroll
      for (int u = 0; u < 2; u++)
#pragma unroll
        for (int k = 0; k < 8; k++) WMMA(c[k]);
  } else {
    for (int i = 0; i < iters; i++)
#pragma unroll
      for (int u = 0; u < 2; u++)
#pragma unroll
        for (int k = 0; k < 8; k++) x[k] = -__builtin_amdgcn_exp2f(x[k]);
  }
#pragma unroll
  for (int k = 0; k < 8; k++) { KEEP(c[k]); KEEP(x[k]); }
  rec_end(rec, c0, r0);
  f32x8 s = c[0];
#pragma unroll
  for (int k = 1; k < 8; k++) s += c[k];
  float t = 0; for (int k = 0; k < 8; k++) t += x[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s[0] + t;
}


#define KW(K) extern "C" __global__ __launch_bounds__(512) void wmma_k##K(float* o, const bf16x16* i, int n, Rec* r) { wmma_k_body<K>(o, i, n, r); }
KW(1) KW(2) KW(4) KW(8) KW(16)
#define KM(E, V) extern "C" __global__ __launch_bounds__(512) void mix_e##E##v##V(float* o, const bf16x16* i, int n, Rec* r) { mix_body<E, V>(o, i, n, r); }
KM(1, 0) KM(2, 0) KM(4, 0) KM(0, 2) KM(0, 4) KM(0, 8) KM(1, 3) KM(1, 6)

// ---------------------------------------------------------------- host
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIPERR %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(3); } } while (0)
typedef void (*KFn)(float*, const bf16x16*, int, Rec*);

struct Cfg { const char* name; KFn fn; int ops_per_iter; const char* unit; };

static u64 med(std::vector<u64> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
  bool toy = argc > 1 && !strcmp(argv[1], "toy");
  bool zero = argc > 2 && !strcmp(argv[2], "zero");
  int dev; CK(hipGetDevice(&dev));
  hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, dev));
  printf("DEV %s cus=%d clk_khz=%d lds_per_block_max=%zu\n", p.gcnArchName, p.multiProcessorCount, p.clockRate, (size_t)p.sharedMemPerBlock);
  const int NCU = p.multiProcessorCount;
  const size_t LDS = 200 * 1024;

  std::vector<Cfg> cfgs = {
      {"wmma_k1", wmma_k1, 16, "wmma"},  {"wmma_k2", wmma_k2, 16, "wmma"},  {"wmma_k4", wmma_k4, 16, "wmma"},
      {"wmma_k8", wmma_k8, 16, "wmma"},  {"wmma_k16", wmma_k16, 16, "wmma"},
      {"exp8", exp8, 16, "exp"},           {"fma8", fma8, 16, "fma"},           {"pkfma8", pkfma8, 16, "pkfma"},
      {"mix_e1", mix_e1v0, 16, "wmma"},   {"mix_e2", mix_e2v0, 16, "wmma"},   {"mix_e4", mix_e4v0, 16, "wmma"},
      {"mix_v2", mix_e0v2, 16, "wmma"},   {"mix_v4", mix_e0v4, 16, "wmma"},   {"mix_v8", mix_e0v8, 16, "wmma"},
      {"mix_e1v3", mix_e1v3, 16, "wmma"}, {"mix_e1v6", mix_e1v6, 16, "wmma"},
      {"roles", roles, 16, "mixed"},
  };
  for (auto& c : cfgs) CK(hipFuncSetAttribute((const void*)c.fn, hipFuncAttributeMaxDynamicSharedMemorySize, (int)LDS));

  // inputs: 1024 x bf16x16, random in [-1,1) (or zeros)
  std::vector<__bf16> hin(1024 * 16);
  srand(1);
  for (auto& v : hin) v = (__bf16)(zero ? 0.f : (rand() / (float)RAND_MAX * 2.f - 1.f));
  bf16x16* din; CK(hipMalloc(&din, hin.size() * sizeof(__bf16)));
  CK(hipMemcpy(din, hin.data(), hin.size() * sizeof(__bf16), hipMemcpyHostToDevice));
  const int MAXT = 512;
  float* dout; CK(hipMalloc(&dout, (size_t)NCU * MAXT * sizeof(float)));
  Rec* drec; CK(hipMalloc(&drec, (size_t)NCU * (MAXT / 32) * sizeof(Rec)));
  hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));

  int wps_list_full[] = {1, 2, 4};
  for (auto& c : cfgs) {
    for (int wps : wps_list_full) {
      if (toy && wps != 1) continue;
      if (!strcmp(c.name, "roles") && wps != 2) continue;
      if (!strncmp(c.name, "mix", 3) && wps == 4) continue;
      int grid = toy ? 1 : NCU, block = 128 * wps, nw = grid * block / 32;
      // ~4e6 cycles per wave at the expected rate for 1 wave/SIMD; toy uses 64 iterations
      int iters = toy ? 64 : (c.unit[0] == 'w' || c.unit[0] == 'm' ? 16384 : 65536) / wps;
      if (!strcmp(c.name, "wmma_k1")) iters = toy ? 64 : 4096 / wps;  // latency-bound chain is slower
      std::vector<u64> cyc_reps, rt_reps; std::vector<float> ms_reps;
      std::vector<Rec> h(nw);
      for (int rep = 0; rep < 7; rep++) {
        CK(hipMemset(drec, 0, nw * sizeof(Rec)));
        CK(hipEventRecord(e0));
        hipLaunchKernelGGL(c.fn, dim3(grid), dim3(block), LDS, 0, dout, din, iters, drec);
        CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1));
        CK(hipGetLastError());
        float ms; CK(hipEventElapsedTime(&ms, e0, e1));
        CK(hipMemcpy(h.data(), drec, nw * sizeof(Rec), hipMemcpyDeviceToHost));
        std::vector<u64> cy, rt;
        int zeros = 0;
        for (auto& r : h) { if (!r.cyc) zeros++; cy.push_back(r.cyc); rt.push_back(r.rt); }
        if (zeros) { printf("BAD %s wps=%d rep=%d missing_records=%d\n", c.name, wps, rep, zeros); }
        if (rep >= 2) { cyc_reps.push_back(med(cy)); rt_reps.push_back(med(rt)); ms_reps.push_back(ms); }
        if (rep == 6) {
          // placement: distinct (HW_ID1 without WAVE_ID bits) per SIMD slot; spread of per-wave cycles
          std::vector<unsigned> keys; for (auto& r : h) keys.push_back((r.h1 & ~0x1fu) ^ (r.h2 << 1));
          std::sort(keys.begin(), keys.end()); int distinct = std::unique(keys.begin(), keys.end()) - keys.begin();
          std::sort(cy.begin(), cy.end());
          printf("PLACE %s wps=%d waves=%d distinct_hwid=%d cyc_min=%llu cyc_p50=%llu cyc_max=%llu h1[0]=0x%x h1[1]=0x%x h1[4]=0x%x h2[0]=0x%x\n",
                 c.name, wps, nw, distinct, cy.front(), cy[cy.size() / 2], cy.back(), h[0].h1, h[nw > 1 ? 1 : 0].h1, h[nw > 4 ? 4 : 0].h1, h[0].h2);
        }
      }
      u64 cyc = med(cyc_reps), rt = med(rt_reps);
      std::sort(ms_reps.begin(), ms_reps.end()); float ms = ms_reps[ms_reps.size() / 2];
      double ops_per_wave = (double)iters * c.ops_per_iter * (strcmp(c.name, "wmma_k1") && strncmp(c.name, "wmma_k", 6) ? 1 :
                             (!strcmp(c.name, "wmma_k1") ? 1 : 1));
      // wmma_k<K>: ops per iteration = (16/min(K,16)) * K = 16 for all K  -> ops_per_iter=16
      double cyc_per_op_per_simd = (double)cyc / (ops_per_wave * wps);  // wps waves share one SIMD
      printf("MB %s wps=%d grid=%d iters=%d ev_ms=%.4f cyc_med=%llu rt_med=%llu cyc_per_rt=%.3f cyc_per_op_simd=%.4f\n",
             c.name, wps, grid, iters, ms, cyc, rt, (double)cyc / rt, cyc_per_op_per_simd);
      fflush(stdout);
    }
  }
  printf("MBDONE\n");
  return 0;
}
