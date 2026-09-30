# mb3.hip: LDS bandwidth. Dynamic LDS = 256 KB (1 WG/CU). Per wave: base address vaddr (lane*16 + per-wave offset),
# masked to stay < 64 KB, plus a segment offset (0 or 64 KB); instruction offsets j*2048 (< 32 KB).
# Max byte touched < 64K + 64K + 32K + 16 = 160 KB < 256 KB allocated. Loads pipelined with s_wait_dscnt lag 8.
hdr = r'''#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef unsigned long long u64;
struct Rec { u64 cyc, rt; unsigned h1, h2; };
'''
def ld_block(op, nload, lag=8):
    # 16 destination sets d0..d15 (operands %0..%15), address %16. Each group of 8 loads then wait dscnt<=lag.
    L=[]
    for j in range(nload):
        L.append(f"{op} %{j%16}, %16 offset:{(j%16)*2048}")
        if j%8==7: L.append(f"s_wait_dscnt {lag}")
    return L
kerns=[]; names=[]
def emit(name, lines, with_wmma):
    body="\\n\\t".join(lines)
    dsts=", ".join(f'"=v"(d[{i}])' for i in range(16))
    if with_wmma:
        cons=f'"+v"(d[0]),"+v"(d[1]),"+v"(d[2]),"+v"(d[3]),"+v"(d[4]),"+v"(d[5]),"+v"(d[6]),"+v"(d[7]),"+v"(d[8]),"+v"(d[9]),"+v"(d[10]),"+v"(d[11]),"+v"(d[12]),"+v"(d[13]),"+v"(d[14]),"+v"(d[15]),"+v"(addr),"+v"(c0),"+v"(c1),"+v"(c2),"+v"(c3),"+v"(c4),"+v"(c5),"+v"(c6),"+v"(c7),"+v"(a),"+v"(b)'
    else:
        cons=f'"+v"(d[0]),"+v"(d[1]),"+v"(d[2]),"+v"(d[3]),"+v"(d[4]),"+v"(d[5]),"+v"(d[6]),"+v"(d[7]),"+v"(d[8]),"+v"(d[9]),"+v"(d[10]),"+v"(d[11]),"+v"(d[12]),"+v"(d[13]),"+v"(d[14]),"+v"(d[15]),"+v"(addr)'
    k=f'''
extern "C" __global__ __launch_bounds__(512) void k_{name}(float* out, const bf16x16* in, int iters, Rec* rec, int seg) {{
  extern __shared__ i32x4 lds4[];
  for (int i = threadIdx.x; i < 256 * 1024 / 16; i += blockDim.x) lds4[i] = i32x4{{i, i + 1, i + 2, i + 3}};
  __syncthreads();
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {{}}, c1 = {{}}, c2 = {{}}, c3 = {{}}, c4 = {{}}, c5 = {{}}, c6 = {{}}, c7 = {{}};
  i32x4 d[16]; for (int i = 0; i < 16; i++) d[i] = i32x4{{}};
  unsigned w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  // seg: 0 = every wave reads segment 0; 1 = waves alternate segment 0/1 by SIMD-pair parity
  unsigned addr = ((lane * 16 + w * 4096) & 0xFFFF) + (seg ? ((w & 1) << 16) : 0u);
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {{
    asm volatile("{body}" : {cons} :: "memory");
  }}
  asm volatile("s_wait_dscnt 0" ::: "memory");
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if (lane == 0) rec[blockIdx.x * (blockDim.x >> 5) + w] = Rec{{t1 - t0, r1 - r0, h1, 0}};
  i32x4 s = {{}}; for (int i = 0; i < 16; i++) s += d[i];
  f32x8 cs = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
  out[blockIdx.x * blockDim.x + threadIdx.x] = s[0] + s[3] + cs[0];
}}'''
    kerns.append(k)
# pure LDS: 16 loads per iteration
for op,nm in (("ds_load_b128","b128"),("ds_load_tr16_b128","tr16")):
    emit(f"ld_{nm}", ld_block(op,16), False); names.append((f"ld_{nm}", 16, 0))
# WMMA + L loads per WMMA (L=1,2): 8 WMMA per iteration, loads interleaved, wait lag 8 every 8 loads
for L in (1,2):
    for op,nm in (("ds_load_b128","b128"),("ds_load_tr16_b128","tr16")):
        lines=[]; j=0
        for k in range(8):
            lines.append(f"v_wmma_f32_16x16x32_bf16 %{17+k}, %25, %26, %{17+k}")
            for _ in range(L):
                lines.append(f"{op} %{j%16}, %16 offset:{(j%16)*2048}"); j+=1
                if j%8==0: lines.append("s_wait_dscnt 8")
        emit(f"wl{L}_{nm}", lines, True); names.append((f"wl{L}_{nm}", 8*L, 8))
host = r'''
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIPERR %s line %d\n", hipGetErrorString(e_), __LINE__); return 3; } } while (0)
typedef void (*KFn)(float*, const bf16x16*, int, Rec*, int);
static u64 med(std::vector<u64> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
int main(int argc, char** argv) {
  bool toy = argc > 1 && !strcmp(argv[1], "toy");
  struct C { const char* n; KFn f; int loads, wmmas; } cs[] = { %CS% };
  const size_t LDS = 256 * 1024;
  std::vector<__bf16> hin(1024 * 16); for (size_t i = 0; i < hin.size(); i++) hin[i] = (__bf16)((i * 37 % 101) / 101.f - 0.5f);
  bf16x16* din; CK(hipMalloc(&din, hin.size() * 2)); CK(hipMemcpy(din, hin.data(), hin.size() * 2, hipMemcpyHostToDevice));
  float* dout; CK(hipMalloc(&dout, 256 * 512 * 4)); Rec* drec; CK(hipMalloc(&drec, 256 * 16 * sizeof(Rec)));
  hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
  for (auto& c : cs) {
    CK(hipFuncSetAttribute((const void*)c.f, hipFuncAttributeMaxDynamicSharedMemorySize, (int)LDS));
    for (int seg = 0; seg <= 1; seg++)
    for (int wps = 1; wps <= 4; wps *= 2) {
      if (toy && (wps > 1 || seg)) continue;
      int grid = toy ? 1 : 256, block = 128 * wps, nw = grid * block / 32, iters = toy ? 16 : 16384 / wps;
      std::vector<u64> cy, rt; std::vector<float> ev; std::vector<Rec> h(nw);
      for (int rep = 0; rep < 5; rep++) {
        CK(hipMemset(drec, 0, nw * sizeof(Rec)));
        CK(hipEventRecord(e0)); hipLaunchKernelGGL(c.f, dim3(grid), dim3(block), LDS, 0, dout, din, iters, drec, seg);
        CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1)); CK(hipGetLastError());
        float ms; CK(hipEventElapsedTime(&ms, e0, e1));
        CK(hipMemcpy(h.data(), drec, nw * sizeof(Rec), hipMemcpyDeviceToHost));
        std::vector<u64> a, b; int miss = 0; for (auto& r : h) { if (!r.cyc) miss++; a.push_back(r.cyc); b.push_back(r.rt); }
        if (miss) printf("BAD %s missing=%d\n", c.n, miss);
        if (rep >= 1) { cy.push_back(*std::max_element(a.begin(), a.end())); rt.push_back(*std::max_element(b.begin(), b.end())); ev.push_back(ms); }
      }
      std::sort(ev.begin(), ev.end());
      u64 C = med(cy), R = med(rt);
      // per CU: 4*wps waves, each iters*loads loads of 512 B; C = slowest wave's span (all waves co-resident)
      double bytes_cu = 4.0 * wps * iters * c.loads * 512.0;
      printf("MB3 %s seg=%d wps=%d ev_ms=%.4f cyc_max=%llu f_rt=%.0f B_per_clk_CU=%.1f cyc_per_wmma_simd=%.3f\n", c.n, seg, wps, ev[ev.size() / 2], C,
             (double)C / R * 100, bytes_cu / C, c.wmmas ? (double)C / ((double)iters * c.wmmas * wps) : 0.0);
      fflush(stdout);
    }
  }
  printf("MBDONE\n"); return 0;
}
'''
cs=", ".join(f'{{"{n}", k_{n}, {l}, {w}}}' for n,l,w in names)
open("mb3.hip","w").write(hdr+"\n".join(kerns)+host.replace("%CS%",cs))
print(names)
