"""Standalone correctness + perf harness for csa_pool_sparse_fwd kernel.

Sparse-branch-only forward: for each query row m, gather top-K pool rows given
by topk[b,m,:] (head-independent) and run softmax over them. Produces
(out_sparse, lse_sparse). Empty rows -> lse=-inf, out=0.
"""
import math
import sys
import torch

from primus_turbo.flydsl.attention.kernels.csa_pool_sparse_fwd_kernel import (
    build_csa_pool_sparse_fwd_module,
)


def ref_sparse(q, pool, topk, scale):
    # q [B,H,S,D], pool [B,P,D], topk [B,S,K]
    B, H, S, D = q.shape
    P = pool.shape[1]
    K = topk.shape[2]
    bidx = torch.arange(B, device=q.device).view(B, 1, 1)
    tk = topk.clone()
    valid = (tk >= 0) & (tk < P)
    tk_safe = torch.where(valid, tk, torch.zeros_like(tk))
    gathered = pool[bidx, tk_safe]  # [B,S,K,D]
    gf = gathered.float()
    qf = q.float()
    # scores [B,H,S,K]
    scores = torch.einsum("bhsd,bskd->bhsk", qf, gf) * scale
    mask = valid.unsqueeze(1)  # [B,1,S,K]
    scores = scores.masked_fill(~mask, float("-inf"))
    m = scores.max(dim=-1).values  # [B,H,S]
    all_masked = ~torch.isfinite(m)
    m_safe = torch.where(all_masked, torch.zeros_like(m), m)
    p = torch.exp(scores - m_safe.unsqueeze(-1))
    p = torch.where(mask, p, torch.zeros_like(p))
    l = p.sum(dim=-1)  # [B,H,S]
    out = torch.einsum("bhsk,bskd->bhsd", p, gf)  # [B,H,S,D]
    l_safe = torch.where(l == 0, torch.ones_like(l), l)
    out = out / l_safe.unsqueeze(-1)
    out = torch.where((l == 0).unsqueeze(-1), torch.zeros_like(out), out)
    lse = m_safe + torch.log(l_safe)
    lse = torch.where(l == 0, torch.full_like(lse, -1.0e30), lse)
    return out, lse


def snr(ref, got):
    ref = ref.float()
    got = got.float()
    noise = (ref - got).pow(2).mean()
    sig = ref.pow(2).mean()
    if noise == 0:
        return 99.0
    return 10 * math.log10((sig / noise).item())


def run(B, H, S, D, K, P, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    scale = D**-0.5
    q = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
    pool = torch.randn(B, P, D, device=dev, dtype=torch.bfloat16)
    topk = torch.randint(0, P, (B, S, K), device=dev, dtype=torch.int32)
    # sprinkle some -1 pads
    padmask = torch.rand(B, S, K, device=dev) < 0.1
    topk = torch.where(padmask, torch.full_like(topk, -1), topk)

    o = torch.empty(B, H, S, D, device=dev, dtype=torch.bfloat16)
    lse = torch.zeros(B, H, S, device=dev, dtype=torch.float32)

    launch = build_csa_pool_sparse_fwd_module(num_heads=H, head_dim=D, dtype_str="bf16")
    launch(q.view(-1), pool.view(-1), topk.view(-1), o.view(-1), lse.view(-1),
           B, S, int(K), int(P))
    torch.cuda.synchronize()

    o_ref, lse_ref = ref_sparse(q, pool, topk, scale)
    # only compare finite-lse rows for lse
    fin = torch.isfinite(lse_ref) & (lse_ref > -1e29)
    o_snr = snr(o_ref, o)
    lse_snr = snr(lse_ref[fin], lse[fin]) if fin.any() else 99.0
    print(f"B{B} H{H} S{S} D{D} K{K} P{P}: out SNR {o_snr:.1f} dB, lse SNR {lse_snr:.1f} dB")

    # timing
    def timed():
        launch(q.view(-1), pool.view(-1), topk.view(-1), o.view(-1), lse.view(-1),
               B, S, int(K), int(P))
    for _ in range(5):
        timed()
    torch.cuda.synchronize()
    import time
    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        timed()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    print(f"    kernel time: {ms:.3f} ms")
    return o_snr, lse_snr, ms


if __name__ == "__main__":
    cases = [
        (1, 8, 512, 512, 17, 64),
        (1, 16, 512, 512, 48, 128),
        (1, 64, 2048, 512, 512, 512),
    ]
    for c in cases:
        run(*c)
