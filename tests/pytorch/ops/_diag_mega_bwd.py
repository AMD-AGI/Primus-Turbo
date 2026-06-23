"""Standalone backward validation for mega_moe_fused (in-kernel d_topk_w + prologue weight_recv_buf).

Compares dx / dW1 / dW2 / d_topk_w vs a torch autograd reference (dW all-reduced
across ranks). Self-contained so it survives concurrent edits to the shared EP test.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import math

import torch
import torch.distributed as dist

from primus_turbo.flydsl.mega.mega_moe_epilogue import ACTIVATION_CLAMP
from primus_turbo.pytorch.ops.moe.mega_moe_fused import mega_moe_fused


def _gate3(g, r):
    g, r = g.float().flatten(), r.float().flatten()
    cos = float(torch.dot(g, r) / (g.norm() * r.norm() + 1e-12))
    rel = float((g - r).norm() / (r.norm() + 1e-12))
    return cos, rel, (cos >= 0.99 and rel <= 0.05)


def _ref(x, topk_idx, topk_w, W1g, W2g, I, clamp):
    T, H = x.shape
    xf = x.float()
    y = torch.zeros((T, H), dtype=torch.float32, device=x.device)
    valid = topk_idx >= 0
    pairs = torch.nonzero(valid, as_tuple=False)
    tok, kidx = pairs[:, 0], pairs[:, 1]
    expert = topk_idx[tok, kidx].to(torch.int64)
    weight = topk_w[tok, kidx].float()
    x_pairs = xf[tok]
    out = torch.zeros((x_pairs.size(0), H), dtype=torch.float32, device=x.device)
    for e in torch.unique(expert).tolist():
        m = expert == e
        acc1 = x_pairs[m] @ W1g[e].float().T
        gate = acc1[:, :I].clamp(-clamp, clamp)
        up = acc1[:, I:].clamp(-clamp, clamp)
        a = (gate * torch.sigmoid(gate)) * up
        out = out.index_copy(0, torch.nonzero(m, as_tuple=False).flatten(), a @ W2g[e].float().T)
    return y.index_add(0, tok, out * weight.unsqueeze(1))


def _run(rank, world, args):
    H, I, E, K, T = args
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:8542", world_size=world, rank=rank)
    torch.set_default_device("cuda")
    group = dist.new_group(list(range(world)))
    epr = E // world
    clamp = ACTIVATION_CLAMP

    g = torch.Generator(device="cuda").manual_seed(1234)
    W1g = torch.randn((E, 2 * I, H), generator=g, dtype=torch.bfloat16) * (2.0 / math.sqrt(H))
    W2g = torch.randn((E, H, I), generator=g, dtype=torch.bfloat16) * (2.0 / math.sqrt(I))
    W1 = W1g[rank * epr : (rank + 1) * epr].contiguous()
    W2 = W2g[rank * epr : (rank + 1) * epr].contiguous()

    torch.manual_seed(7 + rank)
    x = torch.randn((T, H), dtype=torch.float32).bfloat16()
    scores = torch.rand(T, E, generator=g).abs() + 1
    topk_w, topk_idx = torch.topk(scores.softmax(-1), K, dim=-1)
    topk_idx = topk_idx.to(torch.int64)
    topk_w = topk_w.to(torch.float32)
    torch.manual_seed(123 + rank)
    dy = torch.randn((T, H), dtype=torch.float32)

    with torch.no_grad():
        mega_moe_fused(group, x, topk_idx, topk_w, W1, W2)
    torch.cuda.synchronize()
    group.barrier()

    x_m = x.clone().detach().requires_grad_(True)
    w1_m = W1.clone().detach().requires_grad_(True)
    w2_m = W2.clone().detach().requires_grad_(True)
    tw_m = topk_w.clone().detach().requires_grad_(True)
    y_m = mega_moe_fused(group, x_m, topk_idx, tw_m, w1_m, w2_m)
    y_m.backward(dy.to(torch.bfloat16))

    x_r = x.clone().detach().requires_grad_(True)
    W1g_r = W1g.clone().detach().requires_grad_(True)
    W2g_r = W2g.clone().detach().requires_grad_(True)
    tw_r = topk_w.clone().detach().requires_grad_(True)
    y_r = _ref(x_r, topk_idx, tw_r, W1g_r, W2g_r, I, clamp)
    y_r.backward(dy)
    dW1g = W1g_r.grad.clone()
    dist.all_reduce(dW1g, group=group)
    dW2g = W2g_r.grad.clone()
    dist.all_reduce(dW2g, group=group)

    res = {
        "dx": _gate3(x_m.grad, x_r.grad),
        "dW1": _gate3(w1_m.grad, dW1g[rank * epr : (rank + 1) * epr]),
        "dW2": _gate3(w2_m.grad, dW2g[rank * epr : (rank + 1) * epr]),
        "d_topk_w": _gate3(tw_m.grad, tw_r.grad),
    }
    gathered = [None] * world
    dist.all_gather_object(gathered, (rank, res), group=group)
    if rank == 0:
        print("BWD gate-3 (cos / rel / ok):")
        for r, rr in sorted(gathered, key=lambda t: t[0]):
            print("  rank=%d  " % r + "  ".join(f"{n}: {v[0]:.5f}/{v[1]:.4f}/{v[2]}" for n, v in rr.items()))
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.multiprocessing.spawn(_run, args=(8, (7168, 2048, 32, 8, 2048)), nprocs=8)
