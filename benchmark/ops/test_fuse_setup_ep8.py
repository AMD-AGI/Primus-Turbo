#!/usr/bin/env python3
"""Quick EP8 smoke test for PT_DISPATCH_FUSE_SETUP=1 (x-quant in SETUP role)."""
import os
import torch
import torch.distributed as dist

from primus_turbo.flydsl.mega.fp8 import (
    dispatch_prologue,
    get_symm_buffer_for_mega_moe,
    preshuffle_b_scale,
)
from primus_turbo.flydsl.mega.fp8.dispatch_grouped_gemm_mxfp8_kernel import (
    _BSP_CACHE,
    _FUSED_COMPILED,
    dispatch_grouped_gemm_mxfp8,
)
from primus_turbo.pytorch.kernels.mega_moe.mega_moe_forward_fp8_impl import _w1_fp8_cached


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)

    T, H, I, E, K = 512, 7168, 2048, 256, 8
    BM, BN = 256, 256
    epr = E // world

    _FUSED_COMPILED.clear()

    torch.manual_seed(7 + rank)
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(100 + rank)
    scores = torch.rand(T, E, generator=g, device="cuda").abs() + 1
    topk_w, topk_idx = torch.topk(scores.softmax(-1), K, dim=-1)
    topk_w, topk_idx = topk_w.float(), topk_idx.long()

    W1g = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16) * 0.01
    W1 = W1g[rank * epr : (rank + 1) * epr].contiguous()
    w1q, w1s = _w1_fp8_cached(W1)
    G, N, Kw = w1q.shape
    _bk = (w1s.data_ptr(), G, N, Kw, 1)
    _BSP_CACHE[_bk] = preshuffle_b_scale(w1s, G, N, Kw, pack=1)

    symm = get_symm_buffer_for_mega_moe(
        dist.group.WORLD,
        num_experts=E,
        num_max_tokens_per_rank=T,
        num_topk=K,
        hidden=H,
        intermediate_hidden=I,
        block_m=BM,
        block_n=BN,
        use_mxfp8=True,
    )
    sl = symm.make_sym_layout()
    handle = tuple(
        dispatch_prologue(
            topk_idx,
            topk_w,
            sym_layout=sl,
            num_tokens=T,
            num_topk=K,
            num_experts=E,
            world_size=world,
            rank=rank,
            experts_per_rank=epr,
            block_m=BM,
            num_max_pool_tokens=symm.num_max_pool_tokens,
        )
    )

    if rank == 0:
        print("launch fused setup (x-quant, bsp cached)", flush=True)

    out = dispatch_grouped_gemm_mxfp8(
        x, w1q, w1s, handle, sl, symm,
        num_dispatch_cu=16, num_preshuffle_cu=16, BM=BM, BN=BN,
    )
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print("ok", out.shape, float(out.abs().max()), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
