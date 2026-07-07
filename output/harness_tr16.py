"""Correctness+perf harness for the new M=16 ds_read_tr PV kernel (path B).

Reference = the production M=32 flydsl kernel (known good). We compare the new
tr16 kernel's (o, lse) against it via SNR, and report TFLOP/s. Uses the real
adapter's structured [window++pool] topk (same as bench_h2h.py).
"""
import math, time, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primus_turbo.flydsl.attention.kernels.sparse_mla_v2.dsa_fwd import sparse_mla_fwd_v4_flydsl
from primus_turbo.flydsl.attention.kernels.sparse_mla_v2.adapter import _build_csa_topk, _pad_topk_64

_ROPE = 64


def snr(ref, got):
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


def timed(fn, iters=50):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def _call(fn_name, q, kv, topk, sink, D, scale):
    """fn_name in {'m32','tr16'} selects the kernel via env before calling."""
    return sparse_mla_fwd_v4_flydsl(q, kv, topk, attn_sink=sink, kv_lora_rank=D, scale=scale)


def run(B, H, S, D, K, P, W, sink_on):
    q_g, kv_g, topk_g, sink = make(B, H, S, D, K, P, W, sink_on)
    scale = 1.0 / math.sqrt(D + _ROPE)
    valid = (topk_g >= 0).float().sum(dim=1).mean().item()
    flops = 2.0 * 2.0 * (B * S) * H * valid * D
    print(f"\n== B{B} H{H} S{S} K{K} P{P} W{W} sink{int(sink_on)}  valid_k~{valid:.0f} ==")

    # reference = M=32 (force tr16 OFF)
    os.environ["PRIMUS_DSA_FLYDSL_FWD_TR16"] = "0"
    o_ref, lse_ref = sparse_mla_fwd_v4_flydsl(q_g, kv_g, topk_g, attn_sink=sink, kv_lora_rank=D, scale=scale)
    ms_ref = timed(lambda: sparse_mla_fwd_v4_flydsl(q_g, kv_g, topk_g, attn_sink=sink, kv_lora_rank=D, scale=scale))
    print(f"  M32 (ref): {ms_ref:.3f} ms  {flops/(ms_ref*1e-3)/1e12:.1f} TFLOP/s")

    # new tr16 kernel
    os.environ["PRIMUS_DSA_FLYDSL_FWD_TR16"] = "1"
    try:
        o_n, lse_n = sparse_mla_fwd_v4_flydsl(q_g, kv_g, topk_g, attn_sink=sink, kv_lora_rank=D, scale=scale)
        so = snr(o_ref, o_n); sl = snr(lse_ref, lse_n)
        ms_n = timed(lambda: sparse_mla_fwd_v4_flydsl(q_g, kv_g, topk_g, attn_sink=sink, kv_lora_rank=D, scale=scale))
        print(f"  tr16 (new): {ms_n:.3f} ms  {flops/(ms_n*1e-3)/1e12:.1f} TFLOP/s   out SNR {so:.1f} dB  lse SNR {sl:.1f} dB")
        print(f"  ==> tr16/M32 speedup: {ms_ref/ms_n:.3f}x")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  tr16 FAILED: {type(e).__name__}: {str(e)[:300]}")
    finally:
        os.environ["PRIMUS_DSA_FLYDSL_FWD_TR16"] = "0"


if __name__ == "__main__":
    print("device:", torch.cuda.get_device_properties(0).gcnArchName)
    run(1, 128, 4096, 512, 512, 4096, 512, sink_on=True)
    run(1, 64, 4096, 512, 512, 4096, 512, sink_on=True)
    run(1, 128, 4096, 512, 2048, 4096, 512, sink_on=True)
