"""dQ image count check: force K = 1/2/4 at a small shape by lowering ops_bwd._A16_ADDS_PER_IMG, then
compare SNR vs fp32 math SDPA and dK/dV bitwise against K = 1. Usage: python imgs_check.py [B] [S] [H]"""

import math
import sys

import torch

sys.path.insert(0, "/perf_apps/xiaoming/megaattention_overlap_bench")
from flydsl_mla import mla_fwd_sbhd  # noqa: E402
from flydsl_mla import ops_bwd  # noqa: E402

B, S, H = (int(x) for x in sys.argv[1:4]) if len(sys.argv) > 3 else (1, 2048, 8)
DQK, DV = 192, 128


def snr_db(x, ref):
    x, ref = x.double(), ref.double()
    return 10 * math.log10(ref.pow(2).sum().item() / (x - ref).pow(2).sum().clamp_min(1e-300).item())


g = torch.Generator(device="cuda").manual_seed(0)
q, k = (torch.randn((S, B, H, DQK), device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(2))
v, do = (torch.randn((S, B, H, DV), device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(2))
ref_in = [t.permute(1, 2, 0, 3).float().requires_grad_() for t in (q, k, v)]
with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
    o_ref = torch.nn.functional.scaled_dot_product_attention(*ref_in, is_causal=True, scale=DQK**-0.5)
ref = [t.permute(2, 0, 1, 3) for t in torch.autograd.grad(o_ref, ref_in, do.permute(1, 2, 0, 3).float())]
o, lse = mla_fwd_sbhd(q, k, v, causal=True, return_lse=True)

bands = S // 128
base = None
for k_img in (1, 2, 4):
    ops_bwd._A16_ADDS_PER_IMG = -(-bands // k_img)
    assert ops_bwd._a16_imgs_for(S, 128) == k_img
    dq, dk, dv = ops_bwd.mla_bwd_sbhd(do, q, k, v, o, lse)
    torch.cuda.synchronize()
    if base is None:
        base = (dk.clone(), dv.clone())
    same = torch.equal(dk, base[0]) and torch.equal(dv, base[1])
    print(
        f"K={k_img} (adds/img {ops_bwd._A16_ADDS_PER_IMG}): dq {snr_db(dq, ref[0]):.2f} dk {snr_db(dk, ref[1]):.2f}"
        f" dv {snr_db(dv, ref[2]):.2f}  dkdv_equal_K1={same}  finite={bool(torch.isfinite(dq).all())}"
    )
