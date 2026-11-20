###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-stage timing of a Ulysses MLA attention block, unfused vs fused, forward and backward.

The block is the part of an MLA layer that Ulysses touches, from the normalized latents to the
o_proj output plus residual:

    entry   latent up-projection of the local heads over the full sequence
            (unfused: RCCL all-gather + hipBLASLt; fused: AgGemm, backward GemmRS + wgrad)
    attn    causal attention of the local heads (q / k 192, v 128), ``flash_attn_func``
    exit    o all-to-all + o_proj + residual
            (unfused: pack + RCCL all-to-all + unpack + hipBLASLt; fused: A2AGemm, backward GemmA2A + dW_o)

``--variants`` adds two entry baselines that move per-head tensors instead of the latents (both
with the unfused exit):

    explicit    Megatron-Core's MLA path: up-project every head of the local tokens, then one RCCL
                all-to-all of q, k (k_pe broadcast to every head) and v
    laga        LAGA (MLSys 2026): RCCL all-gather of [c_kv | k_pe] + local-head kv up-projection,
                q up-projected for every head and all-to-all'd

The exit forward and the entry backward follow attention, so without ``--sync-stages`` they also
absorb the time a rank waits for the slowest rank's attention (milliseconds at 128K).

The q / kv down-projections, their norms and RoPE are left out: Ulysses does not change them.
Stage times are CUDA-event gaps in issue order, the median over iterations of each rank, then
the max over ranks.

    torchrun --nproc_per_node 8 benchmark/ops/training/bench_ulysses_mla_layer.py --model deepseek-v3 --seq 16384 32768
"""

import argparse
import os
import statistics
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from bench_ulysses_a2a import REC, _barrier, mark
from bench_ulysses_mla_latent import MODELS
from torch.distributed.nn.functional import all_gather, all_to_all_single

from primus_turbo.pytorch.ops import flash_attn_func
from primus_turbo.pytorch.ops.attention.ulysses_fused_a2a import ulysses_a2a_o_gemm
from primus_turbo.pytorch.ops.attention.ulysses_mla_latent import ulysses_mla_latent_up_proj
from primus_turbo.pytorch.ops.attention.usp.attention_a2a_helper import (
    get_attention_cp_a2a_helper,
)

HIDDEN = 7168
STAGES = ("entry", "attn", "exit")
VARIANTS = ("explicit", "laga", "unfused", "fused")


def seq_to_heads(tensors, group):
    """``[s/n, 1, h, d_i]`` tensors of this rank's tokens -> ``[s, 1, h/n, d_i]`` of its heads, in
    one RCCL all-to-all."""
    n = group.size()
    parts = [t.view(t.shape[0], n, t.shape[2] // n, t.shape[3]).movedim(1, 0).reshape(n, -1) for t in tensors]
    flat = torch.cat(parts, dim=1).contiguous()
    flat = all_to_all_single(torch.empty_like(flat), flat, group=group)
    outs = flat.split([p.shape[1] for p in parts], dim=1)
    return [o.reshape(t.shape[0] * n, 1, t.shape[2] // n, t.shape[3]) for o, t in zip(outs, tensors)]


def _sync_ranks(group, phase):
    """Line the ranks up after attention and restart the clock (``_sync`` is left out of every
    stage), so the exit forward and the entry backward do not absorb the ranks' attention skew."""
    torch.cuda.synchronize()
    dist.barrier(group)
    REC.rec(phase, "_sync")


class _SyncBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, *tensors):
        ctx.group = group
        return tuple(t.view_as(t) for t in tensors)

    @staticmethod
    def backward(ctx, *grads):
        _sync_ranks(ctx.group, "bwd")
        return (None, *grads)


class MLABlock:
    def __init__(self, cfg, seq, group, variant, sync_stages=False):
        n = group.size()
        fused = variant == "fused"
        self.cfg, self.group, self.variant, self.fused, self.seq = cfg, group, variant, fused, seq
        self.sync_stages = sync_stages
        h, cq, ckv, dn, dr, dv = (cfg[k] for k in ("heads", "q_lora", "kv_lora", "nope", "rope", "v"))

        def weight(*shape):
            return (torch.randn(*shape, device="cuda") * 0.02).bfloat16().requires_grad_()

        self.w_q = weight(h * (dn + dr), cq)
        self.w_kv = weight(h * (dn + dv), ckv)
        self.w_o = weight(HIDDEN, h * dv)
        self.params = [self.w_q, self.w_kv, self.w_o]
        # sbhd at b = 1, so the bshd helper's [b, s, h, d] views are the same memory
        self.helper = get_attention_cp_a2a_helper(1, seq, h, h, dv, dv, 1, n)

    def exit_unfused(self, o, x):
        h, dv, n = self.cfg["heads"], self.cfg["v"], self.group.size()
        o = self.helper.reshape_o_before_a2a(o.view(1, self.seq, h // n, dv))
        o = all_to_all_single(torch.empty_like(o), o.contiguous(), group=self.group)
        o = self.helper.reshape_o_after_a2a(o)
        return F.linear(o.reshape(x.shape[0], 1, h * dv), self.w_o) + x

    def entry_per_head(self, q_lat, kv_lat):
        h, ckv, dn, dr, dv = (self.cfg[k] for k in ("heads", "kv_lora", "nope", "rope", "v"))
        n, rank = self.group.size(), self.group.rank()
        t = q_lat.shape[0]
        q = F.linear(q_lat, self.w_q).view(t, 1, h, dn + dr)
        c_kv, k_pe = kv_lat.split([ckv, dr], dim=-1)
        if self.variant == "explicit":
            kv = F.linear(c_kv, self.w_kv).view(t, 1, h, dn + dv)
            k_nope, v = kv.split([dn, dv], dim=-1)
            k = torch.cat((k_nope, k_pe.unsqueeze(2).expand(t, 1, h, dr)), dim=-1)
            return seq_to_heads([q, k, v], self.group)
        (q,) = seq_to_heads([q], self.group)
        kv_full = torch.cat(all_gather(kv_lat.contiguous(), group=self.group), dim=0)
        c_kv, k_pe = kv_full.split([ckv, dr], dim=-1)
        hl = h // n
        w_kv = self.w_kv.view(h, dn + dv, ckv)[rank * hl : (rank + 1) * hl].reshape(hl * (dn + dv), ckv)
        kv = F.linear(c_kv, w_kv).view(t * n, 1, hl, dn + dv)
        k_nope, v = kv.split([dn, dv], dim=-1)
        k = torch.cat((k_nope, k_pe.unsqueeze(2).expand(t * n, 1, hl, dr)), dim=-1)
        return q, k, v

    def __call__(self, q_lat, kv_lat, x):
        cfg = self.cfg
        q_lat, kv_lat = mark("start", "entry", q_lat, kv_lat)
        if self.variant in ("explicit", "laga"):
            q, k, v = self.entry_per_head(q_lat, kv_lat)
        else:
            q, k, v = ulysses_mla_latent_up_proj(
                q_lat, kv_lat, self.w_q, self.w_kv, cfg["heads"], self.group,
                cfg["nope"], cfg["rope"], cfg["v"], "sbhd", fused=self.fused,
            )  # fmt: skip
        if self.sync_stages:
            q, k, v = _SyncBackward.apply(self.group, q, k, v)
        q, k, v = mark("entry", "attn", q, k, v)
        o = flash_attn_func(*(t.permute(1, 0, 2, 3) for t in (q, k, v)), causal=True).permute(1, 0, 2, 3)
        o = mark("attn", "exit", o)
        if self.sync_stages:
            _sync_ranks(self.group, "fwd")
        if self.fused:
            y = ulysses_a2a_o_gemm(o, self.w_o, x, self.group, "sbhd")
        else:
            y = self.exit_unfused(o, x)
        return mark("exit", None, y)


def run(cfg, seq, group, variant, warmup, iters, sync_stages=False):
    n = group.size()
    block = MLABlock(cfg, seq, group, variant, sync_stages)
    s_local = seq // n
    q_lat0 = torch.randn(s_local, 1, cfg["q_lora"], device="cuda").bfloat16()
    kv_lat0 = torch.randn(s_local, 1, cfg["kv_lora"] + cfg["rope"], device="cuda").bfloat16()
    x0 = torch.randn(s_local, 1, HIDDEN, device="cuda").bfloat16()
    dy = torch.randn_like(x0)
    per_iter = {"fwd": [], "bwd": []}
    for it in range(warmup + iters):
        REC.reset(enabled=it >= warmup)
        for p in block.params:
            p.grad = None
        q_lat, kv_lat, x = (t.clone().requires_grad_() for t in (q_lat0, kv_lat0, x0))
        _barrier()
        torch.cuda._sleep(2_000_000)
        y = block(q_lat, kv_lat, x)
        REC.rec("bwd", "start")
        y.backward(dy)
        torch.cuda.synchronize()
        if REC.enabled:
            for phase in per_iter:
                per_iter[phase].append(REC.durations_us(phase))
    local = [statistics.median(d[st] for d in per_iter[ph]) for ph in ("fwd", "bwd") for st in STAGES]
    local += [statistics.median(sum(d[st] for st in STAGES) for d in per_iter[ph]) for ph in ("fwd", "bwd")]
    t = torch.tensor(local, device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    vals = t.tolist()
    del block
    torch.cuda.empty_cache()
    out = {(ph, st): vals[i * 3 + j] for i, ph in enumerate(("fwd", "bwd")) for j, st in enumerate(STAGES)}
    out["fwd", "total"], out["bwd", "total"] = vals[6], vals[7]
    return out


def gemm_only(cfg, seq, n, warmup, iters):
    """hipBLASLt time of the GEMMs each stage of the fused path runs, with no communication: the
    floor a fused kernel approaches when its exchange is fully hidden."""
    h, cq, ckv, dn, dr, dv = (cfg[k] for k in ("heads", "q_lora", "kv_lora", "nope", "rope", "v"))
    hl, t = h // n, seq // n

    def rnd(*shape):
        return torch.randn(*shape, device="cuda").bfloat16()

    lq, lc, wq, wkv = rnd(seq, cq), rnd(seq, ckv), rnd(hl * (dn + dr), cq), rnd(hl * (dn + dv), ckv)
    dq, dkv = rnd(seq, hl * (dn + dr)), rnd(seq, hl * (dn + dv))
    o, wo, dy = rnd(t, h * dv), rnd(HIDDEN, h * dv), rnd(t, HIDDEN)
    work = {
        ("fwd", "entry"): lambda: (lq @ wq.t(), lc @ wkv.t()),
        ("bwd", "entry"): lambda: (dq @ wq, dkv @ wkv, dq.t() @ lq, dkv.t() @ lc),
        ("fwd", "exit"): lambda: o @ wo.t(),
        ("bwd", "exit"): lambda: (dy @ wo, dy.t() @ o),
    }
    out = {}
    for key, fn in work.items():
        times = []
        for it in range(warmup + iters):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda._sleep(1_000_000)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            if it >= warmup:
                times.append(start.elapsed_time(end) * 1e3)
        out[key] = statistics.median(times)
    t = torch.tensor([out[k] for k in work], device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return dict(zip(work, t.tolist()))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", nargs="+", default=["deepseek-v3"], choices=sorted(MODELS))
    parser.add_argument("--seq", nargs="+", type=int, default=[32768])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--variants", nargs="+", default=["unfused", "fused"], choices=VARIANTS)
    parser.add_argument(
        "--gemm-only", action="store_true", help="also time the stages' GEMMs without communication"
    )
    parser.add_argument(
        "--sync-stages", action="store_true",
        help="barrier after attention (forward and backward) so the next stage starts with the ranks lined up",
    )  # fmt: skip
    args = parser.parse_args()
    variants = [v for v in VARIANTS if v in args.variants]
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    group = dist.group.WORLD
    n, rank = group.size(), group.rank()
    for model in args.model:
        for seq in args.seq:
            res = {
                v: run(MODELS[model], seq, group, v, args.warmup, args.iters, args.sync_stages)
                for v in variants
            }
            floor = gemm_only(MODELS[model], seq, n, args.warmup, args.iters) if args.gemm_only else None
            if rank != 0:
                continue
            if floor is not None:
                print(
                    f"\n== {model} gemm-only seq={seq} ulysses={n}  (us; fused / gemm-only = exchange left exposed)"
                )
                for key, g in floor.items():
                    line = f"  {' '.join(key):<14}{'gemm':>6}{g:>9.0f}"
                    for v in variants:
                        line += f"{v:>10}{res[v][key]:>8.0f} ({res[v][key] / g:.2f})"
                    print(line)
            for r in res.values():
                r["fwd+bwd", ""] = r["fwd", "total"] + r["bwd", "total"]
            rows = [(ph, st) for ph in ("fwd", "bwd") for st in (*STAGES, "total")] + [("fwd+bwd", "")]
            print(f"\n== {model} MLA block seq={seq} ulysses={n}  (us, median per rank, max over ranks)")
            if variants == ["unfused", "fused"]:
                print(f"  {'stage':<14}{'unfused':>10}{'fused':>10}{'saved':>10}{'speedup':>9}")
                for row in rows:
                    b, f = res["unfused"][row], res["fused"][row]
                    print(f"  {' '.join(row).strip():<14}{b:>10.0f}{f:>10.0f}{b - f:>10.0f}{b / f:>8.2f}x")
                continue
            others = [v for v in variants if v != "fused"] if "fused" in variants else []
            head = "".join(f"{v:>10}" for v in variants) + "".join(f"{'vs ' + v:>14}" for v in others)
            print(f"  {'stage':<14}{head}")
            for row in rows:
                line = "".join(f"{res[v][row]:>10.0f}" for v in variants)
                line += "".join(f"{res[v][row] / res['fused'][row]:>13.2f}x" for v in others)
                print(f"  {' '.join(row).strip():<14}{line}")
    _barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
