###############################################################################
# End-to-end autograd test for hybrid HIP MX-FP8 grouped GEMM.
# Validates fwd, dgrad, wgrad correctness against bf16 reference + prequant path.
###############################################################################

from __future__ import annotations

import argparse
import math

import torch
import torch.utils.benchmark as tbench

from primus_turbo.hip.grouped_gemm_mxfp8.autograd import (
    grouped_gemm_mxfp8_hip,
    prequant_mxfp8_weights_hip,
)


def snr(ref, out):
    ref = ref.float(); out = out.float()
    n = (ref - out).pow(2).mean().item()
    s = ref.pow(2).mean().item()
    return 10 * math.log10(s / max(n, 1e-30)) if n else float("inf")


def bf16_reference_fwd(a_bf, b_bf, group_offs):
    """Per-expert bf16 reference: out[m_start:m_end] = a[m_start:m_end] @ b[g]^T"""
    g = b_bf.shape[0]
    out = torch.zeros(a_bf.shape[0], b_bf.shape[1], device=a_bf.device, dtype=torch.bfloat16)
    for gi in range(g):
        s, e = int(group_offs[gi]), int(group_offs[gi + 1])
        out[s:e] = (a_bf[s:e].float() @ b_bf[gi].float().T).to(torch.bfloat16)
    return out


def bf16_reference_bwd(a_bf, b_bf, grad_out, group_offs):
    g = b_bf.shape[0]
    grad_a = torch.zeros_like(a_bf)
    grad_b = torch.zeros_like(b_bf)
    for gi in range(g):
        s, e = int(group_offs[gi]), int(group_offs[gi + 1])
        a_g = a_bf[s:e].float()
        b_g = b_bf[gi].float()
        dy_g = grad_out[s:e].float()
        grad_a[s:e] = (dy_g @ b_g).to(torch.bfloat16)      # [M_g, N] @ [N, K] -> [M_g, K]
        grad_b[gi]  = (dy_g.T @ a_g).to(torch.bfloat16)    # [N, M_g] @ [M_g, K] -> [N, K]
    return grad_a, grad_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=65536)
    ap.add_argument("--k", type=int, default=2880)
    ap.add_argument("--n", type=int, default=5760)
    ap.add_argument("--g", type=int, default=32)
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    if args.small:
        m, k, n, g = 1024, 512, 512, 4
    else:
        m, k, n, g = args.m, args.k, args.n, args.g

    torch.manual_seed(0)
    device = "cuda"
    assert m % g == 0
    m_per = m // g
    print(f"Shape: M={m}, K={k}, N={n}, G={g}, balanced (M_g={m_per})")

    a_bf = torch.randn(m, k, device=device, dtype=torch.bfloat16, requires_grad=True)
    b_bf = torch.randn(g, n, k, device=device, dtype=torch.bfloat16, requires_grad=True)
    group_offs = torch.arange(0, m + 1, m_per, dtype=torch.int64, device=device)
    group_lens = torch.full((g,), m_per, dtype=torch.int64, device=device)

    # Forward + backward
    out = grouped_gemm_mxfp8_hip(a_bf, b_bf, group_lens, group_offs)
    grad_out = torch.randn_like(out)
    grad_a, grad_b = torch.autograd.grad(out, [a_bf, b_bf], grad_out, retain_graph=False)

    # Reference
    a_det = a_bf.detach()
    b_det = b_bf.detach()
    out_ref = bf16_reference_fwd(a_det, b_det, group_offs)
    grad_a_ref, grad_b_ref = bf16_reference_bwd(a_det, b_det, grad_out, group_offs)

    print("\n─── Correctness vs bf16 reference ───")
    print(f"  out    SNR: {snr(out_ref,    out):.2f} dB")
    print(f"  grad_a SNR: {snr(grad_a_ref, grad_a):.2f} dB")
    print(f"  grad_b SNR: {snr(grad_b_ref, grad_b):.2f} dB")

    all_ok = min(snr(out_ref, out), snr(grad_a_ref, grad_a), snr(grad_b_ref, grad_b)) >= 25.0
    print(f"  Gate (>=25 dB on all): {'PASS' if all_ok else 'FAIL'}")

    if not all_ok:
        return

    # Bench: step (fwd + bwd) without prequant + with prequant
    print("\n─── Bench: full step (fwd + bwd) ───")
    # Autograd routes wgrad through HIP by default; expose a knob to see
    # the Triton-wgrad variant too for direct comparison.
    import os
    def step_no_prequant():
        a2 = a_bf.detach().requires_grad_(True)
        b2 = b_bf.detach().requires_grad_(True)
        o = grouped_gemm_mxfp8_hip(a2, b2, group_lens, group_offs)
        torch.autograd.grad(o, [a2, b2], grad_out)

    def step_with_prequant(prequant):
        a2 = a_bf.detach().requires_grad_(True)
        o = grouped_gemm_mxfp8_hip(a2, prequant, group_lens, group_offs)
        torch.autograd.grad(o, [a2], grad_out)

    # Warm
    for _ in range(3): step_no_prequant()
    torch.cuda.synchronize()
    t1 = tbench.Timer(stmt="f()", globals={"f": step_no_prequant}).timeit(args.iters)
    print(f"  no prequant : {t1.mean*1e3:7.3f} ms")

    prequant = prequant_mxfp8_weights_hip(b_bf.detach())
    for _ in range(3): step_with_prequant(prequant)
    torch.cuda.synchronize()
    t2 = tbench.Timer(stmt="f(p)", globals={"f": step_with_prequant, "p": prequant}).timeit(args.iters)
    print(f"  prequant (b already quantized): {t2.mean*1e3:7.3f} ms")


if __name__ == "__main__":
    main()
