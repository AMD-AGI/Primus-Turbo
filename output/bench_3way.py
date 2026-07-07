"""3-way fwd+bwd comparison: flydsl(tr16) vs triton_v2 vs gluon_v2, identical inputs.

Uses the real adapter structured [window++pool] topk. flydsl fwd uses the tr16
kernel (PRIMUS_DSA_FLYDSL_FWD_TR16=1). All backends share the same fwd/bwd kernel-
pair signature: fwd(q,kv,topk,attn_sink,kv_lora_rank,scale)->(o,lse);
bwd(q,kv,o,do,topk,lse,attn_sink,kv_lora_rank,scale)->(dq,dkv,dsink...).
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

try:
    from primus.backends.megatron.core.transformer.v4_attention_kernels._gluon_v2.dsa_fwd_v4_gluon import sparse_mla_fwd_v4_gluon_v2
    from primus.backends.megatron.core.transformer.v4_attention_kernels._gluon_v2.dsa_bwd_v4_gluon import sparse_mla_bwd_v4_gluon_v2
    _HAS_GLUON = True
except Exception as e:
    print(f"gluon import failed: {type(e).__name__}: {str(e)[:200]}")
    _HAS_GLUON = False


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


def timed(fn, iters=30, warmup=10):
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
    fwd_flops = 2.0 * 2.0 * (B * S) * H * valid * D
    bwd_flops = fwd_flops * 2.5  # standard bwd = 2.5x fwd
    print(f"\n{'='*78}\n== B{B} H{H} S{S} K{K} P{P} W{W} sink{int(sink_on)}  valid_k~{valid:.0f} ==")

    fwds = {
        "flydsl(tr16)": lambda: sparse_mla_fwd_v4_flydsl(q, kv, topk, attn_sink=sink, kv_lora_rank=D, scale=scale),
        "triton_v2":    lambda: sparse_mla_fwd_v4_triton(q, kv, topk, attn_sink=sink, kv_lora_rank=D, scale=scale),
    }
    if _HAS_GLUON:
        fwds["gluon_v2"] = lambda: sparse_mla_fwd_v4_gluon_v2(q, kv, topk, attn_sink=sink, kv_lora_rank=D, scale=scale)

    # reference fwd (gluon if present else triton) for SNR
    ref_name = "gluon_v2" if _HAS_GLUON else "triton_v2"
    o_ref, lse_ref = fwds[ref_name]()

    print(f"  {'--- FORWARD ---':<20}")
    fwd_results = {}
    for name, fn in fwds.items():
        try:
            o, lse = fn()
            s = snr(o_ref, o)
            ms = timed(fn)
            tf = fwd_flops / (ms * 1e-3) / 1e12
            fwd_results[name] = (ms, tf)
            print(f"  {name:<14} {ms:7.3f} ms  {tf:6.1f} TFLOP/s   out SNR vs {ref_name}: {s:5.1f} dB")
        except Exception as e:
            print(f"  {name:<14} FWD FAILED: {type(e).__name__}: {str(e)[:150]}")

    # ---- backward ----
    print(f"  {'--- BACKWARD ---':<20}")
    do = torch.randn_like(o_ref)
    bwds = {
        "flydsl":     lambda o, lse: sparse_mla_bwd_v4_flydsl(q, kv, o, do, topk, lse, attn_sink=sink, kv_lora_rank=D, scale=scale),
        "triton_v2":  lambda o, lse: sparse_mla_bwd_v4_triton(q, kv, o, do, topk, lse, attn_sink=sink, kv_lora_rank=D, scale=scale),
    }
    if _HAS_GLUON:
        bwds["gluon_v2"] = lambda o, lse: sparse_mla_bwd_v4_gluon_v2(q, kv, o, do, topk, lse, attn_sink=sink, kv_lora_rank=D, scale=scale)

    # each backend needs its own (o, lse) from its fwd
    fwd_map = {"flydsl": "flydsl(tr16)", "triton_v2": "triton_v2", "gluon_v2": "gluon_v2"}
    for name, fn in bwds.items():
        try:
            o_b, lse_b = fwds[fwd_map[name]]()
            _ = fn(o_b, lse_b)  # warm/correctness
            ms = timed(lambda: fn(o_b, lse_b))
            tf = bwd_flops / (ms * 1e-3) / 1e12
            print(f"  {name:<14} {ms:7.3f} ms  {tf:6.1f} TFLOP/s")
        except Exception as e:
            print(f"  {name:<14} BWD FAILED: {type(e).__name__}: {str(e)[:150]}")


if __name__ == "__main__":
    print("device:", torch.cuda.get_device_properties(0).gcnArchName)
    print("flydsl fwd = tr16 (PRIMUS_DSA_FLYDSL_FWD_TR16=%s)" % os.environ.get("PRIMUS_DSA_FLYDSL_FWD_TR16"))
    run(1, 128, 4096, 512, 512, 4096, 512, sink_on=True)
    run(1, 64, 4096, 512, 512, 4096, 512, sink_on=True)
    run(1, 128, 4096, 512, 2048, 4096, 512, sink_on=True)
