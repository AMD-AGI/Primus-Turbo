"""Full CSA forward = local SWA + sparse MFMA + host merge, vs eager reference.

Production MQA layout: k_local/v_local [B,1,S,D]. sink [H] optional (lse-only).
merge: lse = ln(exp(lse_l)+exp(lse_s)+exp(sink)); out = out_l*exp(lse_l-lse)+out_s*exp(lse_s-lse).
"""
import math, time, sys
import torch

from primus_turbo.flydsl.attention.kernels.csa_pool_sparse_fwd_kernel import build_csa_pool_sparse_fwd_module
from primus_turbo.flydsl.attention.kernels.sla_fwd_kernel import build_swa_fwd_module
from primus_turbo.pytorch.ops.attention.deepseek_attention_reference import eager_csa_attention


def snr(ref, got):
    ref = ref.float(); got = got.float()
    noise = (ref - got).pow(2).mean(); sig = ref.pow(2).mean()
    if noise == 0: return 99.0
    return 10 * math.log10((sig / noise).item())


_swa_cache = {}
def get_swa(H, D, W):
    key = (H, D, W)
    if key not in _swa_cache:
        _swa_cache[key] = build_swa_fwd_module(num_heads=H, head_dim=D, swa_window=int(W),
            dtype_str="bf16", layout_bhld=True, mqa_kv=True, block_m=128, block_n=32)
    return _swa_cache[key]

_sp_cache = {}
def get_sparse(H, D):
    if (H, D) not in _sp_cache:
        _sp_cache[(H, D)] = build_csa_pool_sparse_fwd_module(num_heads=H, head_dim=D, dtype_str="bf16")
    return _sp_cache[(H, D)]


def full_fwd(q, k_local, v_local, pool, topk, sink, W, scale):
    B, H, S, D = q.shape
    K = topk.shape[2]; P = pool.shape[1]
    dev = q.device
    # local SWA
    o_local = torch.empty_like(q)
    lse_local = torch.zeros(B, H, S, device=dev, dtype=torch.float32)
    swa = get_swa(H, D, W)
    swa(q.contiguous().view(-1), k_local.contiguous().view(-1), v_local.contiguous().view(-1),
        o_local.view(-1), lse_local.view(-1), B, S)
    # sparse
    o_sparse = torch.empty_like(q)
    lse_sparse = torch.zeros(B, H, S, device=dev, dtype=torch.float32)
    sp = get_sparse(H, D)
    sp(q.contiguous().view(-1), pool.contiguous().view(-1), topk.to(torch.int32).contiguous().view(-1),
       o_sparse.view(-1), lse_sparse.view(-1), B, S, int(K), int(P))
    # merge
    parts = [lse_local, lse_sparse]
    if sink is not None:
        parts.append(sink.float().view(1, H, 1).expand(B, H, S))
    lse_stack = torch.stack(parts, 0)  # [n, B,H,S]
    lse = torch.logsumexp(lse_stack, dim=0)
    wl = torch.exp(lse_local - lse).unsqueeze(-1)
    ws = torch.exp(lse_sparse - lse).unsqueeze(-1)
    out = o_local.float() * wl + o_sparse.float() * ws
    return out.to(q.dtype), lse


def run(B, H, S, D, K, P, W=128, sink_on=False, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"; scale = D**-0.5
    q = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
    k_local = torch.randn(B, 1, S, D, device=dev, dtype=torch.bfloat16)
    v_local = torch.randn(B, 1, S, D, device=dev, dtype=torch.bfloat16)
    pool = torch.randn(B, P, D, device=dev, dtype=torch.bfloat16)
    topk = torch.randint(0, P, (B, S, K), device=dev, dtype=torch.int64)
    padmask = torch.rand(B, S, K, device=dev) < 0.1
    topk = torch.where(padmask, torch.full_like(topk, -1), topk)
    sink = torch.randn(H, device=dev, dtype=torch.bfloat16) if sink_on else None

    out, lse = full_fwd(q, k_local, v_local, pool, topk, sink, W, scale)
    torch.cuda.synchronize()

    # reference
    kexp = k_local.expand(B, H, S, D); vexp = v_local.expand(B, H, S, D)
    bidx = torch.arange(B, device=dev).view(B, 1, 1)
    tk = topk.clone(); valid = (tk >= 0) & (tk < P)
    tk_safe = torch.where(valid, tk, torch.zeros_like(tk))
    gathered = pool[bidx, tk_safe]
    sparse_mask = torch.where(valid, torch.zeros(B, S, K, device=dev), torch.full((B, S, K), float("-inf"), device=dev)).to(torch.bfloat16)
    out_ref = eager_csa_attention(q, kexp, vexp, gathered, sink=sink, swa_window=W,
        sparse_mask=sparse_mask, attn_dropout=0.0, training=True, scale=scale)
    s = snr(out_ref, out)
    print(f"B{B} H{H} S{S} D{D} K{K} P{P} sink{sink_on}: out SNR {s:.1f} dB")

    def timed():
        full_fwd(q, k_local, v_local, pool, topk, sink, W, scale)
    for _ in range(5): timed()
    torch.cuda.synchronize()
    it = 20; t0 = time.perf_counter()
    for _ in range(it): timed()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / it * 1e3
    print(f"    full fwd time: {ms:.3f} ms  (Triton fwd ~0.70)")
    return s, ms


if __name__ == "__main__":
    run(1, 64, 2048, 512, 512, 512, sink_on=False)
    run(1, 64, 2048, 512, 512, 512, sink_on=True)
    run(1, 128, 2048, 512, 1024, 1024, sink_on=True)
