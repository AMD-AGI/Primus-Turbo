"""One GQA attention block (fwd + bwd) as Primus' ulysses_fused_a2a patch runs it, profiled per kernel class.

    input RMSNorm -> ulysses_qkv_gemm_a2a -> q/k RMSNorm (qk-norm models) -> RoPE over the full sequence
    -> flash_attn_func (causal) -> ulysses_a2a_o_gemm

``--mode 1`` fused, ``-1`` the patch's serial ablation (hipBLASLt + pack + RCCL). RoPE: TE's fused kernel
when the model config sets apply_rope_fusion, else Megatron's unfused torch path (MiniMax-M2.5, partial
rotary). Prints the layer's fwd + bwd time (median, max over ranks) and, on rank 0, GPU time per kernel
class and the top kernels outside the fused ops.

    torchrun --nproc_per_node 4 bench_gqa_layer.py --model glm-4.7 --seq 32768 --mode 1
"""

import argparse
import collections
import os
import statistics
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

REPO = os.environ.get(
    "TURBO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")),
)
sys.path.insert(0, REPO)

from primus_turbo.pytorch.ops import flash_attn_func
from primus_turbo.pytorch.ops.attention.ulysses_fused_a2a import (
    ulysses_a2a_o_gemm,
    ulysses_qkv_gemm_a2a,
)
from primus_turbo.pytorch.ops.attention.usp.attention_a2a_helper import (
    get_attention_cp_a2a_helper,
)

# attention-side settings of the Primus model configs (examples/megatron/configs, primus/configs/models)
LAYERS = {
    "qwen3-30b-a3b": dict(
        hidden=2048, hq=32, hkv=4, d=128, qk_norm=True, rope_fusion=True, rot=1.0, base=1e6
    ),
    "qwen3-235b-a22b": dict(
        hidden=4096, hq=64, hkv=4, d=128, qk_norm=True, rope_fusion=True, rot=1.0, base=1e6
    ),
    "glm-4.7": dict(hidden=5120, hq=96, hkv=8, d=128, qk_norm=False, rope_fusion=True, rot=1.0, base=1e4),
    "minimax-m2.5": dict(
        hidden=3072, hq=48, hkv=8, d=128, qk_norm=True, rope_fusion=False, rot=0.5, base=5e6
    ),
}


def _a2a(t, group):
    from torch.distributed.nn.functional import all_to_all_single

    return all_to_all_single(torch.empty_like(t), t.contiguous(), group=group)


def serial_qkv(x, w_qkv, hq, hkv, d, group, qkv_format, weight_layout):
    s_local, b = x.shape[:2]
    helper = get_attention_cp_a2a_helper(b, s_local * group.size(), hq, hkv, d, d, 0, group.size())
    qkv = F.linear(x, w_qkv).view(s_local, b, hkv, hq // hkv + 2, d)
    q = qkv[..., : hq // hkv, :].reshape(s_local, b, hq, d)
    k, v = qkv[..., hq // hkv, :], qkv[..., hq // hkv + 1, :]
    return helper.splits_qkv_after_a2a(_a2a(helper.combine_qkv_before_a2a(q, k, v), group))


def serial_o(o, w_o, residual, group, qkv_format):
    seq, b, heads, d = o.shape
    n = group.size()
    helper = get_attention_cp_a2a_helper(b, seq, heads * n, heads * n, d, d, 0, n)
    og = helper.reshape_o_after_a2a(_a2a(helper.reshape_o_before_a2a(o), group))
    return F.linear(og.reshape(*og.shape[:2], -1), w_o)


def rope_unfused(t, freqs):
    """Megatron's _apply_rotary_pos_emb_bshd (non-interleaved, mscale 1), partial rotary."""
    rot = freqs.shape[-1]
    t, t_pass = t[..., :rot], t[..., rot:]
    cos, sin = torch.cos(freqs).to(t.dtype), torch.sin(freqs).to(t.dtype)
    x1, x2 = torch.chunk(t, 2, dim=-1)
    t = t * cos + torch.cat((-x2, x1), dim=-1) * sin
    return torch.cat((t, t_pass), dim=-1)


def classify(name):
    low = name.lower()
    for key, cls in (
        ("ulysses_gemm_a2a", "fused GemmA2A"),
        ("ulysses_a2a_gemm", "fused A2AGemm"),
        ("nccl", "RCCL"),
        ("rccl", "RCCL"),
        ("fmha", "attention"),
        ("attn", "attention"),
        ("flash", "attention"),
        ("rope", "RoPE"),
        ("rotary", "RoPE"),
        ("rmsnorm", "norm"),
        ("rms_norm", "norm"),
        ("layernorm", "norm"),
        ("layer_norm", "norm"),
        ("cijk", "hipBLASLt GEMM"),
        ("gemm", "hipBLASLt GEMM"),
        ("memcpy", "copy"),
        ("copy", "copy"),
        ("cat", "copy"),
        ("elementwise", "elementwise"),
        ("reduce", "elementwise"),
    ):
        if key in low:
            return cls
    return "other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(LAYERS))
    p.add_argument("--seq", type=int, default=32768)
    p.add_argument("--mode", type=int, default=1, choices=[1, -1])
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--trace", default=None, help="write a chrome trace of one profiled iteration (rank 0)")
    args = p.parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    n, rank = group.size(), group.rank()
    cfg = LAYERS[args.model]
    hidden, hq, hkv, d = cfg["hidden"], cfg["hq"], cfg["hkv"], cfg["d"]
    s_local, b = args.seq // n, 1
    torch.manual_seed(0)
    dev = "cuda"
    x0 = torch.randn(s_local, b, hidden, device=dev, dtype=torch.bfloat16)
    ln_w = torch.nn.Parameter(torch.ones(hidden, device=dev, dtype=torch.bfloat16))
    w_qkv = torch.nn.Parameter((torch.randn((hq + 2 * hkv) * d, hidden, device=dev) * 0.02).bfloat16())
    w_o = torch.nn.Parameter((torch.randn(hidden, hq * d, device=dev) * 0.02).bfloat16())
    q_norm = k_norm = None
    if cfg["qk_norm"]:
        import transformer_engine.pytorch as te

        q_norm = te.RMSNorm(d, eps=1e-6, params_dtype=torch.bfloat16)
        k_norm = te.RMSNorm(d, eps=1e-6, params_dtype=torch.bfloat16)
    rot = int(d * cfg["rot"])
    inv = 1.0 / (cfg["base"] ** (torch.arange(0, rot, 2, device=dev, dtype=torch.float32) / rot))
    freqs = torch.outer(torch.arange(args.seq, device=dev, dtype=torch.float32), inv)
    freqs = torch.cat((freqs, freqs), dim=-1)[:, None, None, :]
    if cfg["rope_fusion"]:
        from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb

        rope = lambda t: apply_rotary_pos_emb(t, freqs, tensor_format="sbhd", fused=True)
    else:
        rope = lambda t: rope_unfused(t, freqs)
    qkv_op, o_op = (ulysses_qkv_gemm_a2a, ulysses_a2a_o_gemm) if args.mode == 1 else (serial_qkv, serial_o)

    def layer():
        x = x0.detach().requires_grad_(True)
        h = F.rms_norm(x, (hidden,), ln_w, 1e-6)
        q, k, v = qkv_op(h, w_qkv, hq, hkv, d, group, "sbhd", "interleaved")
        if q_norm is not None:
            q, k = q_norm(q), k_norm(k)
        q, k = rope(q), rope(k)
        o = flash_attn_func(
            q.permute(1, 0, 2, 3), k.permute(1, 0, 2, 3), v.permute(1, 0, 2, 3), dropout_p=0.0, causal=True
        )
        o = o.permute(1, 0, 2, 3).contiguous()
        y = o_op(o, w_o, None, group, "sbhd")
        y.backward(dy)

    dy = torch.randn(s_local, b, hidden, device=dev, dtype=torch.bfloat16)
    for _ in range(3):
        layer()
    times = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        dist.barrier()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(2_000_000)
        s.record()
        layer()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1e3)
    t = torch.tensor([statistics.median(times)], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)

    prof_iters = 3
    torch.cuda.synchronize()
    dist.barrier()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(prof_iters):
            layer()
        torch.cuda.synchronize()
    if rank == 0:
        if args.trace:
            prof.export_chrome_trace(args.trace)
        per_cls = collections.defaultdict(float)
        per_kernel = collections.defaultdict(lambda: [0.0, 0])
        for ev in prof.events():
            if ev.device_type != torch.autograd.DeviceType.CUDA:
                continue
            us = ev.device_time / prof_iters if hasattr(ev, "device_time") else ev.cuda_time / prof_iters
            per_cls[classify(ev.name)] += us
            per_kernel[ev.name][0] += us
            per_kernel[ev.name][1] += 1
        total = sum(per_cls.values())
        mode = "fused" if args.mode == 1 else "serial"
        print(
            f"\n== {args.model} seq={args.seq} n={n} {mode}: layer fwd+bwd {t.item():.0f} us (median, max over ranks)"
        )
        print(f"   GPU kernel time per iteration {total:.0f} us (rank 0, sum over kernels):")
        for cls, us in sorted(per_cls.items(), key=lambda kv: -kv[1]):
            print(f"     {cls:<16}{us:>9.0f} us {us / total:>6.1%}")
        print("   top kernels outside the fused ops / attention:")
        rest = [
            (k, v)
            for k, v in per_kernel.items()
            if classify(k) not in ("fused GemmA2A", "fused A2AGemm", "attention")
        ]
        for k, (us, cnt) in sorted(rest, key=lambda kv: -kv[1][0])[:16]:
            print(f"     {us:>8.1f} us  x{cnt // prof_iters:<3} {k[:110]}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
