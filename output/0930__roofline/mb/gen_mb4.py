# mb4.hip: WMMA + LDS-load interleave with deeper dscnt lag. LDS destinations are explicit VGPRs v[96:255]
# (40 sets of 4, declared as clobbers); WMMA operands are compiler-allocated. A destination set is reused only
# after 40 loads while the lag keeps <= lag (< 40) loads outstanding, and DS returns complete in order.
NSET=40
hdr = open('gen_mb3.py').read().split("hdr = r'''")[1].split("'''")[0]
clob = ", ".join(f'"v{r}"' for r in range(96, 96 + 4*NSET))
kerns=[]; names=[]
def dst(j): r=96+4*(j%NSET); return f"v[{r}:{r+3}]"
def emit(name, lines, loads, wmmas):
    body="\\n\\t".join(lines)
    k=f'''
extern "C" __global__ __launch_bounds__(256) void k_{name}(float* out, const bf16x16* in, int iters, Rec* rec, int seg) {{
  extern __shared__ i32x4 lds4[];
  for (int i = threadIdx.x; i < 256 * 1024 / 16; i += blockDim.x) lds4[i] = i32x4{{i, i + 1, i + 2, i + 3}};
  __syncthreads();
  bf16x16 a = in[threadIdx.x], b = in[threadIdx.x + 128];
  f32x8 c0 = {{}}, c1 = {{}}, c2 = {{}}, c3 = {{}}, c4 = {{}}, c5 = {{}}, c6 = {{}}, c7 = {{}};
  unsigned w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned addr = ((lane * 16 + w * 4096) & 0xFFFF) + (seg ? ((w & 1) << 16) : 0u);
  asm volatile("" ::: "memory");
  u64 t0 = __builtin_readcyclecounter(), r0 = __builtin_readsteadycounter();
  for (int i = 0; i < iters; i++) {{
    asm volatile("{body}" : "+v"(c0),"+v"(c1),"+v"(c2),"+v"(c3),"+v"(c4),"+v"(c5),"+v"(c6),"+v"(c7),"+v"(a),"+v"(b),"+v"(addr) :: "memory", {clob});
  }}
  asm volatile("s_wait_dscnt 0" ::: "memory");
  u64 t1 = __builtin_readcyclecounter(), r1 = __builtin_readsteadycounter();
  unsigned h1; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_HW_ID1)" : "=s"(h1));
  if (lane == 0) rec[blockIdx.x * (blockDim.x >> 5) + w] = Rec{{t1 - t0, r1 - r0, h1, 0}};
  f32x8 cs = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
  out[blockIdx.x * blockDim.x + threadIdx.x] = cs[0] + addr;
}}'''
    kerns.append(k); names.append((name, loads, wmmas))
# 16 WMMA per iteration (operands %0..%7 accumulators, %8 a, %9 b, %10 addr)
for op,nm in (("ds_load_b128","b128"),("ds_load_tr16_b128","tr16")):
  for L in (1,2):
    for lag in (8,16,24,32):
        if L==2 and lag==8: continue
        lines=[]; j=0
        for k in range(16):
            lines.append(f"v_wmma_f32_16x16x32_bf16 %{k%8}, %8, %9, %{k%8}")
            for _ in range(L):
                lines.append(f"{op} {dst(j)}, %10 offset:{(j%16)*2048}"); j+=1
                if j%8==0: lines.append(f"s_wait_dscnt {lag}")
        emit(f"i{L}_{nm}_lag{lag}", lines, 16*L, 16)
  # grouped: 8 WMMA then 8 loads, lag 16
  lines=[]; j=0
  for g in range(2):
    for k in range(8): lines.append(f"v_wmma_f32_16x16x32_bf16 %{k}, %8, %9, %{k}")
    for _ in range(8): lines.append(f"{op} {dst(j)}, %10 offset:{(j%16)*2048}"); j+=1
    lines.append("s_wait_dscnt 16")
  emit(f"g1_{nm}_lag16", lines, 16, 16)
  # pure loads, lag 32
  lines=[]; j=0
  for _ in range(16):
    lines.append(f"{op} {dst(j)}, %10 offset:{(j%16)*2048}"); j+=1
    if j%8==0: lines.append("s_wait_dscnt 32")
  emit(f"p_{nm}_lag32", lines, 16, 0)
host = open('gen_mb3.py').read().split("host = r'''")[1].split("'''")[0]
host = host.replace('printf("MB3 ', 'printf("MB4 ').replace("wps <= 4", "wps <= 2")
cs=", ".join(f'{{"{n}", k_{n}, {l}, {w}}}' for n,l,w in names)
open("mb4.hip","w").write(hdr+"\n".join(kerns)+host.replace("%CS%",cs))
print(len(names), [n for n,_,_ in names])
