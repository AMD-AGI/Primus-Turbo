"""Flydsl sparse-MLA bwd TF across the 6 bwd groups: {flash,pro} x cr{0,4,128} @seq=4096.
Mirrors bench_triton.py's cr->P/K/topk mapping. Set PRIMUS_DSA_BWD_FLYDSL_DQ=1
PRIMUS_DSA_INTERM_FLYDSL=1 for the all-flydsl path."""
import argparse, math, torch
from primus_turbo.flydsl.attention.kernels.sparse_mla_v2.dsa_fwd import sparse_mla_fwd_v4_flydsl
from primus_turbo.flydsl.attention.kernels.sparse_mla_v2.dsa_bwd import sparse_mla_bwd_v4_flydsl

_ROPE_DIM, _HEAD_DIM, _SWA_WINDOW = 64, 512, 128
_VARIANTS = {"flash": dict(H=64, index_topk=512), "pro": dict(H=128, index_topk=1024)}

def _build(cr, H, S, D, K, P, W, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev, dt = "cuda", torch.bfloat16
    latent = torch.randn(S, D, generator=g, device=dev, dtype=dt)
    q512 = torch.randn(S, H, D, generator=g, device=dev, dtype=dt)
    q_g = torch.cat([q512, torch.zeros(S, H, _ROPE_DIM, device=dev, dtype=dt)], -1).contiguous()
    sink = torch.randn(H, generator=g, device=dev, dtype=torch.float32) * 0.1
    do = torch.randn(S, H, D, generator=g, device=dev, dtype=dt)
    ti = torch.arange(S, device=dev).view(S, 1)
    win = ti - W + 1 + torch.arange(W, device=dev).view(1, W)
    win = torch.where(win >= 0, win, torch.full_like(win, -1))
    if cr == 0:
        kv512 = latent.unsqueeze(1); topk = win
    else:
        pool = torch.randn(P, D, generator=g, device=dev, dtype=dt)
        kv512 = torch.cat([latent, pool], 0).unsqueeze(1)
        if cr == 4:
            pool_topk = S + torch.randint(0, P, (S, K), generator=g, device=dev)
        else:
            ps = torch.arange(P, device=dev).view(1, P)
            pool_topk = torch.where(((ps + 1) * cr - 1) <= ti, S + ps, torch.full_like(ps.expand(S, P), -1))
        topk = torch.cat([win, pool_topk], 1)
    tk = topk.shape[1]; pad = ((tk + 63) // 64) * 64 - tk
    if pad > 0:
        topk = torch.cat([topk, torch.full((S, pad), -1, device=dev, dtype=topk.dtype)], 1)
    kv_g = torch.cat([kv512, torch.zeros(kv512.shape[0], 1, _ROPE_DIM, device=dev, dtype=dt)], -1).contiguous()
    return q_g, kv_g, topk.to(torch.int32).contiguous(), sink, do

def _time(fn, warmup, iters):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    ts = []
    for s, e in ev:
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2]

def bench(variant, cr, S, warmup, iters):
    cfg = _VARIANTS[variant]; H, D = cfg["H"], _HEAD_DIM; scale = 1.0/math.sqrt(D)
    if cr == 4:
        P = max(S//4, 1); K = min(cfg["index_topk"], P); topk_eff = _SWA_WINDOW + K
    elif cr == 0:
        P, K, topk_eff = 0, 0, _SWA_WINDOW
    else:
        P = max(S//cr, 1); K, topk_eff = 0, _SWA_WINDOW + P
    gq, gkv, gtopk, sink, do = _build(cr, H, S, D, K, P, _SWA_WINDOW)
    out, lse = sparse_mla_fwd_v4_flydsl(gq, gkv, gtopk, attn_sink=sink, kv_lora_rank=D, scale=scale)
    def bwd(): return sparse_mla_bwd_v4_flydsl(gq, gkv, out, do, gtopk, lse, attn_sink=sink, kv_lora_rank=D, scale=scale)
    fwd_flop = 2.0 * S * H * topk_eff * (D + D)
    bwd_med = _time(bwd, warmup, iters)
    tf = 2.5 * fwd_flop / (bwd_med*1e-3) / 1e12
    print(f"  {variant:5s} cr={cr:<3d} topk={topk_eff:<5d} | bwd {bwd_med:6.2f}ms {tf:7.1f}TF", flush=True)
    return tf

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=8); ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    print(f"=== flydsl bwd | S={a.seq} bf16 sink swa=128 | 6 bwd groups ===", flush=True)
    tfs = []
    for v in ("flash", "pro"):
        for cr in (0, 4, 128):
            tfs.append(bench(v, cr, a.seq, a.warmup, a.iters))
    print(f"  bwd mean={sum(tfs)/len(tfs):.1f}TF  min={min(tfs):.1f}TF  (target 600)", flush=True)
