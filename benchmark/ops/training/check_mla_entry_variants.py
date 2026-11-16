"""q / k / v of the explicit and laga entry baselines against the latent path, same weights."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from bench_ulysses_mla_layer import MLABlock
from bench_ulysses_mla_latent import MODELS

from primus_turbo.pytorch.ops.attention.ulysses_mla_latent import ulysses_mla_latent_up_proj

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dist.init_process_group("nccl", timeout=timedelta(minutes=5))
group = dist.group.WORLD
n = group.size()
for model in ("deepseek-v3", "kimi-k2"):
    cfg = MODELS[model]
    seq = 2048
    torch.manual_seed(0)
    ref = MLABlock(cfg, seq, group, "unfused")
    q_lat = torch.randn(seq // n, 1, cfg["q_lora"], device="cuda").bfloat16().requires_grad_()
    kv_lat = torch.randn(seq // n, 1, cfg["kv_lora"] + cfg["rope"], device="cuda").bfloat16().requires_grad_()
    args = (cfg["heads"], group, cfg["nope"], cfg["rope"], cfg["v"], "sbhd")
    want = ulysses_mla_latent_up_proj(q_lat, kv_lat, ref.w_q, ref.w_kv, *args, fused=False)
    grads = [torch.randn_like(t) for t in want]
    torch.autograd.backward(want, grads)
    # weights are replicated over the group: compare their gradients summed over it
    for w in (ref.w_q, ref.w_kv):
        dist.all_reduce(w.grad, group=group)
    want_g = [t.grad.clone() for t in (q_lat, kv_lat, ref.w_q, ref.w_kv)]
    for variant in ("explicit", "laga"):
        for t in (q_lat, kv_lat, ref.w_q, ref.w_kv):
            t.grad = None
        blk = MLABlock(cfg, seq, group, variant)
        blk.w_q, blk.w_kv = ref.w_q, ref.w_kv
        got = blk.entry_per_head(q_lat, kv_lat)
        torch.autograd.backward(got, grads)
        for w in (ref.w_q, ref.w_kv):
            dist.all_reduce(w.grad, group=group)
        got_g = [t.grad for t in (q_lat, kv_lat, ref.w_q, ref.w_kv)]
        errs = [(a.float() - b.float()).abs().max().item() for a, b in zip(got, want)]
        gerrs = [(a.float() - b.float()).abs().max().item() / b.float().abs().max().item() for a, b in zip(got_g, want_g)]
        if dist.get_rank() == 0:
            print(f"{model} n={n} {variant}: fwd max|d| q/k/v {errs}  bwd rel q_lat/kv_lat/w_q/w_kv "
                  f"{[f'{e:.1e}' for e in gerrs]}")
dist.destroy_process_group()
