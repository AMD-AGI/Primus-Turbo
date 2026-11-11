"""Where the MLA backward goes wrong: SNR per 16-column d-tile and per 64-row block.
Usage: python mla_diag.py [B] [S] [H] [block_q]"""

import math
import sys

import torch

sys.path.insert(0, "/perf_apps/xiaoming/megaattention_overlap_bench")
from flydsl_mla import mla_fwd_sbhd  # noqa: E402
from flydsl_mla.ops_bwd import mla_bwd_sbhd  # noqa: E402

argv = sys.argv[1:]
B, S, H = (int(x) for x in argv[0:3]) if len(argv) >= 3 else (1, 1024, 8)
BQ = int(argv[3]) if len(argv) > 3 else 32
DQK, DV = 192, 128


def snr(x, r):
    x, r = x.double(), r.double()
    return 10 * math.log10(r.pow(2).sum().item() / max((x - r).pow(2).sum().item(), 1e-300))


g = torch.Generator(device="cuda").manual_seed(0)
q, k, v = (torch.randn((S, B, H, d), device="cuda", dtype=torch.bfloat16, generator=g) for d in (DQK, DQK, DV))
dout = torch.randn((S, B, H, DV), device="cuda", dtype=torch.bfloat16, generator=g)
ref_in = [t.permute(1, 2, 0, 3).float().requires_grad_() for t in (q, k, v)]
with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
    o_ref = torch.nn.functional.scaled_dot_product_attention(*ref_in, is_causal=True, scale=DQK**-0.5)
ref = [t.permute(2, 0, 1, 3).contiguous() for t in torch.autograd.grad(o_ref, ref_in, dout.permute(1, 2, 0, 3).float())]
out, lse = mla_fwd_sbhd(q, k, v, causal=True, return_lse=True)
print("fwd O snr", f"{snr(out, o_ref.detach().permute(2, 0, 1, 3)):.2f}")
dq, dk, dv = mla_bwd_sbhd(dout, q, k, v, out, lse, block_q=BQ)
torch.cuda.synchronize()
for name, x, r in (("dq", dq, ref[0]), ("dk", dk, ref[1]), ("dv", dv, ref[2])):
    d = x.shape[-1]
    cols = " ".join(f"{snr(x[..., c:c + 16], r[..., c:c + 16]):5.1f}" for c in range(0, d, 16))
    print(f"{name} per d-tile: {cols}")
    rows = " ".join(f"{snr(x[s:s + 64], r[s:s + 64]):5.1f}" for s in range(0, min(S, 1024), 64))
    print(f"{name} per 64 rows: {rows}")
    ratio = (x.double().norm() / r.double().norm()).item()
    cos = torch.nn.functional.cosine_similarity(x.double().flatten(), r.double().flatten(), dim=0).item()
    print(f"{name} norm ratio {ratio:.4f} cos {cos:.6f}")
