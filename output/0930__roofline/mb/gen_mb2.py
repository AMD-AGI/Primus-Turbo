# Generates mb2.hip: exact in-wave sequences "WMMA, then N x OP" via inline asm (8 independent accumulators,
# 8 independent OP chains). No memory access inside the loop. Hazards: WMMA accumulators are reused only after
# 8 WMMAs; OP chains touch registers disjoint from WMMA operands, so no WMMA/VALU dependency exists.
OPS = {
  # name: (operand kind, asm template using {y} {m} {d}, init)
  "fma":   ("f", "v_fma_f32 {y}, {y}, {m}, {d}"),
  "add":   ("f", "v_add_f32 {y}, {y}, {d}"),
  "mul":   ("f", "v_mul_f32 {y}, {y}, {m}"),
  "max":   ("f", "v_max_num_f32 {y}, {y}, {d}"),
  "exp":   ("f", "v_exp_f32 {y}, {y}"),
  "cvt":   ("f", "v_cvt_pk_bf16_f32 {y}, {y}, {d}"),
  "pkfma": ("p", "v_pk_fma_f32 {y}, {y}, {m}, {d}"),
  "pkmul": ("p", "v_pk_mul_f32 {y}, {y}, {m}"),
  "nop":   ("n", "v_nop"),
  "salu":  ("s", "s_add_co_i32 {s}, {s}, 1"),
}
NS = [1, 2, 4, 8]
hdr = r'''#include <hip/hip_runtime.h>
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
'''
kern = []
names = []
def body(opname, n):
    kind, tpl = OPS[opname] if opname else ("x", "")
    lines = []
    for k in range(8):
        lines.append(f"v_wmma_f32_16x16x32_bf16 %{k}, %8, %9, %{k}")
        for j in range(n):
            y = f"%{10 + (k * n + j) % 8}"
            if kind == "f": lines.append(tpl.format(y=y, m="%18", d="%19"))
            elif kind == "p": lines.append(tpl.format(y=y, m="%18", d="%19"))
            elif kind == "n": lines.append(tpl)
            elif kind == "s": lines.append(tpl.format(s="%20"))
    return "\\n\\t".join(lines)
cfgs = [("base", None, 0)] + [(f"{o}{n}", o, n) for o in OPS for n in NS]
for nm, op, n in cfgs:
    kind = OPS[op][0] if op else "f"
    yt = "f32x2" if kind == "p" else "float"
    init = "y[k] = f32x2{1.0f + k, 1e-3f * threadIdx.x};" if kind == "p" else "y[k] = -1.0f - 1e-3f * (threadIdx.x + k);"
    mt = "f32x2 m = {0.999f, 0.998f}, d = {1e-4f * threadIdx.x, 2e-4f};" if kind == "p" else "float m = 0.999f, d = 1e-4f * threadIdx.x;"
    k = f'''
extern "C" __global__ __launch_bounds__(512) void k_{nm}(float* out, const bf16x16* in, int iters, Rec* rec) {{
  extern __shared__ float lds[]; if (threadIdx.x == 0) lds[0] = 0.f;
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {{}}, c1 = {{}}, c2 = {{}}, c3 = {{}}, c4 = {{}}, c5 = {{}}, c6 = {{}}, c7 = {{}};
  {yt} y[8]; {mt} int s = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int k = 0; k < 8; k++) {{ {init} }}
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {{
    asm volatile("{body(op, n)}"
      : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7), "+v"(a), "+v"(b),
        "+v"(y[0]), "+v"(y[1]), "+v"(y[2]), "+v"(y[3]), "+v"(y[4]), "+v"(y[5]), "+v"(y[6]), "+v"(y[7]),
        "+v"(m), "+v"(d), "+s"(s));
  }}
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if ((threadIdx.x & 31) == 0) rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = Rec{{t1 - t0, r1 - r0, h1, 0}};
  f32x8 sum = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = s;
  for (int k = 0; k < 8; k++) t += {"y[k][0]" if kind == "p" else "y[k]"};
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum[0] + t;
}}'''
    kern.append(k); names.append((nm, n))
host = r'''
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIPERR %s line %d\n", hipGetErrorString(e_), __LINE__); return 3; } } while (0)
typedef void (*KFn)(float*, const bf16x16*, int, Rec*);
static u64 med(std::vector<u64> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
int main(int argc, char** argv) {
  bool toy = argc > 1 && !strcmp(argv[1], "toy");
  struct C { const char* n; KFn f; } cs[] = { %CS% };
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
'''
cs = ", ".join(f'{{"{nm}", k_{nm}}}' for nm, _ in names)
open("mb2.hip", "w").write(hdr + "\n".join(kern) + host.replace("%CS%", cs))
print(len(names), "kernels")
