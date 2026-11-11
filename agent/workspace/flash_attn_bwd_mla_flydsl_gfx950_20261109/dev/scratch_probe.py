"""Does one earlier dispatch with a larger private segment slow the r8 dkdv kernel?
Usage: python scratch_probe.py <floats>[,<floats>...]  (0 = no pre-dispatch). Runs, in order, one
throwaway HIP kernel per entry whose per-lane scratch holds <floats> fp32, then times r8 bwd at
B1 S16K with the bench sampler."""

import statistics
import sys

sys.path.insert(0, "/perf_apps/xiaoming/Primus-Turbo/agent/workspace/flash_attn_bwd_mla_flydsl_gfx950_20261109")
import quick_test_bench as qb  # noqa: E402
import torch  # noqa: E402
from torch.utils.cpp_extension import load_inline  # noqa: E402

SIZES = (40, 61, 64, 75, 96, 127, 128, 255)
SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
template <int SZ>
__global__ void scratch_k(float* out, int idx) {
  volatile float buf[SZ];
  for (int i = 0; i < SZ; ++i) buf[i] = out[i & 3] + i;
  out[4 + threadIdx.x] = buf[(idx + threadIdx.x) % SZ];
}
template <int SZ>
int run(torch::Tensor out, int idx) {
  hipLaunchKernelGGL(scratch_k<SZ>, dim3(1024), dim3(256), 0, 0, out.data_ptr<float>(), idx);
  hipFuncAttributes a;
  hipFuncGetAttributes(&a, (const void*)scratch_k<SZ>);
  return (int)a.localSizeBytes;
}
int run_sz(torch::Tensor out, int64_t sz, int64_t idx) {
  switch (sz) {
""" + "".join(f"    case {s}: return run<{s}>(out, idx);\n" for s in SIZES) + r"""
  }
  return -1;
}
"""

pre = [int(x) for x in sys.argv[1].split(",") if int(x)]
local = []
if pre:
    ext = load_inline("scratch_probe_ext2", cpp_sources="int run_sz(torch::Tensor out, int64_t sz, int64_t idx);",
                      cuda_sources=SRC, functions=["run_sz"], verbose=False)
    out = torch.zeros(260, device="cuda", dtype=torch.float32)
    for n in pre:
        local.append(ext.run_sz(out, n, 7))
        torch.cuda.synchronize()

S, B = 16384, 1
g = torch.Generator(device="cuda").manual_seed(0)
q, k = (torch.randn((S, B, 128, 192), device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(2))
v = torch.randn((S, B, 128, 128), device="cuda", dtype=torch.bfloat16, generator=g)
do = torch.randn((S, B, 128, 128), device="cuda", dtype=torch.bfloat16, generator=g)
o, lse = qb.mla_fwd_sbhd(q, k, v, causal=True, return_lse=True)
sampler = qb.timed(lambda: qb.mla_bwd_sbhd(do, q, k, v, o, lse), 5, 3.0)
xs = [sampler() for _ in range(4)]
print(f"pre floats {pre} localSizeBytes {local}: r8 B1 S16K {statistics.median(xs):8.3f} ms {['%.2f' % x for x in xs]}",
      flush=True)
