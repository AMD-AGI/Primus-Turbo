"""Backward correctness+perf harness: flydsl dq (M=16 candidate) vs triton_v2/gluon.

Compares dq/dkv/dsink SNR of the flydsl bwd against triton_v2 (reference) and
reports bwd TFLOP/s. flydsl dq M=16 kernel gated via PRIMUS_DSA_FLYDSL_BWD_DQ_M16.
"""
import math, time, os, sys
import torch

os.environ.setdefault("PRIMUS_DSA_FLYDSL_FWD_TR16", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/apps/tas/yaoc/agent_work/mi355x/flydsl-dpskv4-attn/Primus")
_ROPE = 64

from primus_turbo.flydsl.attention.kernels.sparse_mla_v2.dsa_fwd import sparse_mla_fwd_v4_flydsl
from primus_turbo.flydsl.attention.kernels.sparse_mla_v2.dsa_bwd import sparse_mla_bwd_v4_flydsl
from primus_turbo.flydsl.attention.kernels.sparse_mla_v2.adapter import _build_csa_topk, _pad_topk_64
from primus_turbo.triton.attention.deepseek.sparse_mla_v2.dsa_fwd import sparse_mla_fwd_v4_triton
from primus_turbo.triton.attention.deepseek.sparse_mla_v2.dsa_bwd import sparse_mla_bwd_v4_triton


def snr(ref, got):
    if ref is None or got is None:
        return float("nan")
    ref = ref.float(); got = got.float()
    n = (ref - got).pow(2).mean(); s = ref.pow(2).mean()
    return 99.0 if n == 0 else 10 * math.log10((s / n).item())


def make(B, H, S, D, K, P, W, sink_on, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    q_bh = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
    latent = torch.randn(B, S, D, device=dev, dtype=torch.bfloat16)
    pool = torch.randn(B, P, D, device=dev, dtype=torch.bfloat16)
    topk_idxs = torch.randint(0, P, (B, S, K), device=dev, dtype=torch.int64)
    topk_idxs = torch.where(torch.rand(B, S, K, device=dev) < 0.1, torch.full_like(topk_idxs, -1), topk_idxs)
    sink = torch.randn(H, device=dev, dtype=torch.float32) if sink_on else None
    z_q = torch.zeros(B * S, H, _ROPE, device=dev, dtype=torch.bfloat16)
    q_g = torch.cat([q_bh.permute(0, 2, 1, 3).reshape(B * S, H, D), z_q], dim=-1).contiguous()
    kv512 = torch.cat([latent, pool], dim=1).reshape(B * (S + P), 1, D)
    z_kv = torch.zeros(B * (S + P), 1, _ROPE, device=dev, dtype=torch.bfloat16)
    kv_g = torch.cat([kv512, z_kv], dim=-1).contiguous()
    topk_g = _pad_topk_64(_build_csa_topk(topk_idxs, S, P, W))
    return q_g, kv_g, topk_g, sink


def timed(fn, iters=20, warmup=8):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def run(B, H, S, D, K, P, W, sink_on):
    q, kv, topk, sink = make(B, H, S, D, K, P, W, sink_on)
    scale = 1.0 / math.sqrt(D + _ROPE)
    valid = (topk >= 0).float().sum(dim=1).mean().item()
    bwd_flops = 2.0 * 2.0 * (B * S) * H * valid * D * 2.5
    print(f"\n== B{B} H{H} S{S} K{K} P{P} W{W} sink{int(sink_on)}  valid_k~{valid:.0f} ==")

    # fwd (flydsl tr16) gives o, lse for both bwds
    o, lse = sparse_mla_fwd_v4_flydsl(q, kv, topk, attn_sink=sink, kv_lora_rank=D, scale=scale)
    do = torch.randn_like(o)

    # reference bwd = triton_v2
    dq_r, dkv_r, dsink_r = sparse_mla_bwd_v4_triton(q, kv, o, do, topk, lse, attn_sink=sink, kv_lora_rank=D, scale=scale)
    ms_r = timed(lambda: sparse_mla_bwd_v4_triton(q, kv, o, do, topk, lse, attn_sink=sink, kv_lora_rank=D, scale=scale))
    print(f"  triton_v2  : {ms_r:7.3f} ms  {bwd_flops/(ms_r*1e-3)/1e12:6.1f} TFLOP/s  (reference)")

    # flydsl bwd
    try:
        dq_f, dkv_f, dsink_f = sparse_mla_bwd_v4_flydsl(q, kv, o, do, topk, lse, attn_sink=sink, kv_lora_rank=D, scale=scale)
        s_dq = snr(dq_r[:, :, :D], dq_f[:, :, :D])
        s_dkv = snr(dkv_r, dkv_f)
        s_dsink = snr(dsink_r, dsink_f) if dsink_r is not None else float("nan")
        ms_f = timed(lambda: sparse_mla_bwd_v4_flydsl(q, kv, o, do, topk, lse, attn_sink=sink, kv_lora_rank=D, scale=scale))
        print(f"  flydsl     : {ms_f:7.3f} ms  {bwd_flops/(ms_f*1e-3)/1e12:6.1f} TFLOP/s  "
              f"dq {s_dq:.0f}dB dkv {s_dkv:.0f}dB dsink {s_dsink:.0f}dB   speedup vs prev: {ms_r/ms_f:.2f}x-of-triton")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  flydsl FAILED: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    print("device:", torch.cuda.get_device_properties(0).gcnArchName)
    print("BWD_DQ_M16 =", os.environ.get("PRIMUS_DSA_FLYDSL_BWD_DQ_M16", "0"))
    run(1, 128, 4096, 512, 512, 4096, 512, sink_on=True)
    run(1, 64, 4096, 512, 512, 4096, 512, sink_on=True)
    run(1, 128, 4096, 512, 2048, 4096, 512, sink_on=True)
