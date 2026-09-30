// Roofline microbenchmarks, batch 6: global memory. Plain HIP grid-stride kernels; every index is < n by the loop
// condition. read: sum of 16-B loads; write: 16-B stores; copy. Working sets from 4 MB (L2-resident) to 4 GB (HBM).
// Usage: mb6 [toy]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef unsigned long long u64;

__global__ __launch_bounds__(256) void rd(const i32x4* __restrict__ a, size_t n, int passes, int* out, u64* rec) {
  u64 c0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  i32x4 s = {};
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (int p = 0; p < passes; p++) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 3 * stride < n; i += 4 * stride) {  // 4 independent 16-B loads in flight per thread
      i32x4 x0 = a[i], x1 = a[i + stride], x2 = a[i + 2 * stride], x3 = a[i + 3 * stride];
      s += x0 ^ x1 ^ x2 ^ x3;
    }
    for (; i < n; i += stride) s += a[i];
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = s[0] ^ s[1] ^ s[2] ^ s[3];
  if (threadIdx.x == 0) { rec[2 * blockIdx.x] = __builtin_readcyclecounter() - c0; rec[2 * blockIdx.x + 1] = __builtin_readsteadycounter() - r0; }
}
__global__ __launch_bounds__(256) void wr(i32x4* __restrict__ a, size_t n, int passes, int* out, u64* rec) {
  u64 c0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  size_t stride = (size_t)gridDim.x * blockDim.x;
  i32x4 v = {(int)threadIdx.x, 1, 2, 3};
  for (int p = 0; p < passes; p++)
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) a[i] = v;
  if (threadIdx.x == 0) { rec[2 * blockIdx.x] = __builtin_readcyclecounter() - c0; rec[2 * blockIdx.x + 1] = __builtin_readsteadycounter() - r0; }
}
__global__ __launch_bounds__(256) void cp(const i32x4* __restrict__ a, i32x4* __restrict__ b, size_t n, int passes, int* out, u64* rec) {
  u64 c0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (int p = 0; p < passes; p++)
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) b[i] = a[i];
  if (threadIdx.x == 0) { rec[2 * blockIdx.x] = __builtin_readcyclecounter() - c0; rec[2 * blockIdx.x + 1] = __builtin_readsteadycounter() - r0; }
}

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIPERR %s line %d\n", hipGetErrorString(e_), __LINE__); return 3; } } while (0)

int main(int argc, char** argv) {
  bool toy = argc > 1 && !strcmp(argv[1], "toy");
  const size_t MAXB = toy ? (64ull << 20) : (4ull << 30);
  i32x4 *a, *b; CK(hipMalloc(&a, MAXB)); CK(hipMalloc(&b, MAXB));
  CK(hipMemset(a, 1, MAXB)); CK(hipMemset(b, 0, MAXB));
  const int GRID = toy ? 64 : 4096, BLK = 256;
  int* out; CK(hipMalloc(&out, (size_t)GRID * BLK * 4));
  u64* drec; CK(hipMalloc(&drec, (size_t)GRID * 2 * 8));
  std::vector<u64> h(GRID * 2);
  hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
  std::vector<size_t> sizes = toy ? std::vector<size_t>{4ull << 20, 64ull << 20}
                                  : std::vector<size_t>{4ull << 20, 16ull << 20, 64ull << 20, 134ull << 20, 256ull << 20, 1ull << 30, 4ull << 30};
  for (int kind = 0; kind < 3; kind++)
    for (size_t bytes : sizes) {
      if (kind == 2 && bytes > MAXB) continue;
      size_t n = bytes / 16;
      // enough passes for >= ~2 ms of traffic at ~10 TB/s
      int passes = (int)std::max<size_t>(1, (size_t)(toy ? 1 : 20e9) / bytes);
      std::vector<float> ev; double fr = 0;
      for (int rep = 0; rep < 5; rep++) {
        CK(hipEventRecord(e0));
        if (kind == 0) hipLaunchKernelGGL(rd, dim3(GRID), dim3(BLK), 0, 0, a, n, passes, out, drec);
        if (kind == 1) hipLaunchKernelGGL(wr, dim3(GRID), dim3(BLK), 0, 0, b, n, passes, out, drec);
        if (kind == 2) hipLaunchKernelGGL(cp, dim3(GRID), dim3(BLK), 0, 0, a, b, n, passes, out, drec);
        CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1)); CK(hipGetLastError());
        float ms; CK(hipEventElapsedTime(&ms, e0, e1));
        CK(hipMemcpy(h.data(), drec, h.size() * 8, hipMemcpyDeviceToHost));
        u64 mc = 0, mr = 0; for (int g = 0; g < GRID; g++) if (h[2 * g] > mc) { mc = h[2 * g]; mr = h[2 * g + 1]; }
        if (rep >= 1) { ev.push_back(ms); fr += (double)mc / mr * 100; }
      }
      std::sort(ev.begin(), ev.end());
      double ms = ev[ev.size() / 2], moved = (double)bytes * passes * (kind == 2 ? 2 : 1);
      printf("MB6 %s bytes=%zu passes=%d ev_ms=%.4f TBps=%.3f f_rt=%.0f B_per_clk_CU=%.1f\n", kind == 0 ? "read" : kind == 1 ? "write" : "copy", bytes, passes,
             ms, moved / ms / 1e9, fr / ev.size(), moved / (ms * 1e-3 * fr / ev.size() * 1e6) / 256);
      fflush(stdout);
    }
  printf("MBDONE\n");
  return 0;
}
