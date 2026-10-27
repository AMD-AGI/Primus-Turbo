"""gemm_tile_ab.py [nt|nn]: one GPU, the L2 fwd (nt: N=7168, K=2048) or L1 dgrad (nn: N=7168, K=4096, B [G, K, N])
pool GEMM of the EP2 t8192 random step (128 experts, 271 pool blocks of 256 rows), timed for
  gg        the standalone FlyDSL grouped GEMM (gemm_only of the stage bench)
  mega22    Mega's nt_gemm_tile launcher, 2x2 waves, GROUP_M 16
  mega42    4x2 waves (gemm_combine's warp grid), GROUP_M 16
  mega42rm  4x2 waves, row-major tile order (GROUP_M 1, gemm_combine's order)
  mega22rm  2x2 waves, row-major
interleaved over REPS rounds; prints the median ms per variant."""
import statistics
import sys

import torch

from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import (
    grouped_gemm_bf16_nn_flydsl_kernel,
    grouped_gemm_bf16_nt_flydsl_kernel,
)
from primus_turbo.flydsl.mega.bf16_gfx1250.gemm_nt_tile import mega_grouped_gemm_nt_gfx1250

mode = sys.argv[1] if len(sys.argv) > 1 else "nt"
REPS = 7
G, N = 128, 7168
K = 2048 if mode == "nt" else 4096
blocks = [2] * G
for e in range(0, G, 9)[:15]:
    blocks[e] = 3
num_blocks = sum(blocks)
rows = num_blocks * 256
dev = torch.device("cuda")
torch.manual_seed(0)
a = (torch.randn(rows, K, device=dev) / 8).to(torch.bfloat16)
b = (torch.randn(G, N, K, device=dev) / 8).to(torch.bfloat16)
if mode == "nn":
    b = b.transpose(1, 2).contiguous()  # [G, K, N]
tile_to_expert = torch.repeat_interleave(torch.arange(G, dtype=torch.int32), torch.tensor(blocks)).to(dev)
num_tile_blocks = torch.tensor([num_blocks], dtype=torch.int32, device=dev)
offs = torch.zeros(G + 1, dtype=torch.int64, device=dev)
offs[1:] = torch.tensor(blocks, device=dev).cumsum(0) * 256
out = torch.empty(rows, N, dtype=torch.bfloat16, device=dev)
b_nn = mode == "nn"
gg = grouped_gemm_bf16_nn_flydsl_kernel if b_nn else grouped_gemm_bf16_nt_flydsl_kernel


def mega(m_warp, n_warp, group_m):
    return lambda: mega_grouped_gemm_nt_gfx1250(
        a, b, out, tile_to_expert, num_tile_blocks, m_warp=m_warp, n_warp=n_warp, group_m=group_m, b_nn=b_nn
    )


variants = {
    "gg": lambda: gg(a, b, offs),
    "mega22": mega(2, 2, 16),
    "mega42": mega(4, 2, 16),
    "mega42rm": mega(4, 2, 1),
    "mega22rm": mega(2, 2, 1),
}
ref = gg(a, b, offs).float()
for name, fn in variants.items():
    if name != "gg":
        out.zero_()
        fn()
        err = (out.float() - ref).abs().max().item()
        print(f"{name}: max abs diff vs gg {err:.3e}")


def timed(fn, iters=20):
    for _ in range(3):
        fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


ms = {n: [] for n in variants}
for _ in range(REPS):
    for n, fn in variants.items():
        ms[n].append(timed(fn))
flops = 2.0 * rows * N * K
base = statistics.median(ms["gg"])
print(f"{mode} G={G} N={N} K={K} pool blocks {num_blocks}")
for n, v in ms.items():
    m = statistics.median(v)
    print(f"  {n:9s} {m:.3f} ms  {flops / m / 1e9:7.1f} TF  vs gg {100 * (m / base - 1):+.1f}%")
