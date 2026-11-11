"""FlyDSL MLA 192/128 backward vs an fp32 math-SDPA reference (SNR in dB), sbhd causal.
The reference's own bf16 rounding is the floor; aiter a16 is printed alongside when asked.
Usage: python mla_acc.py [B] [S] [H] [seed] [aiter(0/1)] [block_q] [block_kv]"""

import math
import os
import sys

import torch

sys.path.insert(0, "/perf_apps/xiaoming/megaattention_overlap_bench")
from flydsl_mla import mla_fwd_sbhd  # noqa: E402
from flydsl_mla.ops_bwd import mla_bwd_sbhd  # noqa: E402

argv = sys.argv[1:]
B, S, H = (int(x) for x in argv[0:3]) if len(argv) >= 3 else (1, 4096, 128)
SEED = int(argv[3]) if len(argv) > 3 else 0
AITER = len(argv) > 4 and argv[4] == "1"
BQ = int(argv[5]) if len(argv) > 5 else 32
BKV = int(argv[6]) if len(argv) > 6 else 128
DQK, DV = 192, 128


def snr_db(x, ref):
    x, ref = x.double(), ref.double()
    return 10 * math.log10(ref.pow(2).sum().item() / (x - ref).pow(2).sum().clamp_min(1e-300).item())


def main():
    g = torch.Generator(device="cuda").manual_seed(SEED)
    q, k, v = (torch.randn((S, B, H, d), device="cuda", dtype=torch.bfloat16, generator=g) for d in (DQK, DQK, DV))
    dout = torch.randn((S, B, H, DV), device="cuda", dtype=torch.bfloat16, generator=g)

    ref_in = [t.permute(1, 2, 0, 3).float().requires_grad_() for t in (q, k, v)]
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        o_ref = torch.nn.functional.scaled_dot_product_attention(*ref_in, is_causal=True, scale=DQK**-0.5)
    ref = torch.autograd.grad(o_ref, ref_in, dout.permute(1, 2, 0, 3).float())
    ref = [t.permute(2, 0, 1, 3).contiguous() for t in ref]
    del o_ref, ref_in
    torch.cuda.empty_cache()
    print(f"B{B} S{S} H{H} seed{SEED} d{DQK}/{DV} block_q{BQ} block_kv{BKV}")
    print("  floor (fp32 ref rounded to bf16): " + "  ".join(f"d{n}={snr_db(r.bfloat16(), r):.2f}" for n, r in zip("qkv", ref)))

    out, lse = mla_fwd_sbhd(q, k, v, causal=True, return_lse=True)
    g1 = mla_bwd_sbhd(dout, q, k, v, out, lse, block_q=BQ, block_kv=BKV)
    g2 = mla_bwd_sbhd(dout, q, k, v, out, lse, block_q=BQ, block_kv=BKV)
    torch.cuda.synchronize()
    snrs = "  ".join(f"d{n}={snr_db(x, r):.2f}" for n, x, r in zip("qkv", g1, ref))
    rep = "  ".join(f"d{n}_bitwise_repeat={torch.equal(a, b)}" for n, a, b in zip("qkv", g1, g2))
    fin = all(torch.isfinite(x).all().item() for x in g1)
    print(f"  flydsl: {snrs}   {rep}   finite={fin}")

    if AITER:
        import primus_turbo.pytorch as turbo
        from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager

        GlobalBackendManager.set_attn_backend(BackendType.AITER)
        os.environ["PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32"] = "0"
        qa, ka, va = (t.permute(1, 0, 2, 3).detach().requires_grad_() for t in (q, k, v))
        oa = turbo.ops.flash_attn_func(qa, ka, va, softmax_scale=DQK**-0.5, causal=True)
        ga = torch.autograd.grad(oa, (qa, ka, va), dout.permute(1, 0, 2, 3))
        print("  aiter a16: " + "  ".join(f"d{n}={snr_db(x.permute(1, 0, 2, 3), r):.2f}" for n, x, r in zip("qkv", ga, ref)))


if __name__ == "__main__":
    main()
