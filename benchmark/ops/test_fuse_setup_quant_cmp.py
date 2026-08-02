#!/usr/bin/env python3
"""Compare fused SETUP x-quant scratch vs host quantize_rowwise_mxfp8_flydsl."""
import os
import torch
import torch.distributed as dist

from primus_turbo.flydsl.mega.fp8 import quantize_rowwise_mxfp8_flydsl
from primus_turbo.flydsl.mega.fp8.dispatch_grouped_gemm_mxfp8_kernel import (
    _XQ_SCRATCH,
    _XS_SCRATCH,
    dispatch_grouped_gemm_mxfp8,
)
from primus_turbo.flydsl.mega.fp8 import dispatch_prologue, get_symm_buffer_for_mega_moe, preshuffle_b_scale
from primus_turbo.flydsl.mega.fp8.dispatch_grouped_gemm_mxfp8_kernel import _BSP_CACHE, _FUSED_COMPILED
from primus_turbo.pytorch.kernels.mega_moe.mega_moe_forward_fp8_impl import _w1_fp8_cached


def main():
    port = os.environ.get("MASTER_PORT", "8610")
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", world_size=1, rank=0)
    torch.cuda.set_device(0)

    T, H, I, E, K = 64, 7168, 2048, 8, 8
    BM, BN = 256, 256
    setup_cu = int(os.environ.get("PT_DISPATCH_SETUP_CU", "1"))
    os.environ["PT_DISPATCH_FUSE_SETUP"] = "1"
    _XQ_SCRATCH.clear()
    _XS_SCRATCH.clear()
    _FUSED_COMPILED.clear()

    torch.manual_seed(0)
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    xq_ref, xs_ref = quantize_rowwise_mxfp8_flydsl(x)
    xs_ref = xs_ref.view(torch.uint8)

    g = torch.Generator(device="cuda").manual_seed(100)
    scores = torch.rand(T, E, generator=g, device="cuda").abs() + 1
    topk_w, topk_idx = torch.topk(scores.softmax(-1), K, dim=-1)
    topk_w, topk_idx = topk_w.float(), topk_idx.long()
    W1 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16) * 0.01
    w1q, w1s = _w1_fp8_cached(W1)
    _bk = (w1s.data_ptr(), w1q.shape[0], w1q.shape[1], w1q.shape[2], 1)
    _BSP_CACHE[_bk] = preshuffle_b_scale(w1s, *w1q.shape, pack=1)

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
            topk_idx, topk_w, sym_layout=sl, num_tokens=T, num_topk=K, num_experts=E,
            world_size=1, rank=0, experts_per_rank=E, block_m=BM,
            num_max_pool_tokens=symm.num_max_pool_tokens,
        )
    )

    print(f"SETUP_CU={setup_cu}", flush=True)
    dispatch_grouped_gemm_mxfp8(
        x, w1q, w1s, handle, sl, symm,
        num_dispatch_cu=4, num_preshuffle_cu=4, BM=BM, BN=BN,
    )
    torch.cuda.synchronize()

    sk = (T, H, x.device)
    xq_k = _XQ_SCRATCH[sk]
    xs_k = _XS_SCRATCH[sk]
    xq_match = torch.equal(xq_k, xq_ref)
    xs_match = torch.equal(xs_k, xs_ref)
    xq_close = (xq_k.float() - xq_ref.float()).abs().max().item()
    xs_diff = (xs_k.int() - xs_ref.int()).abs().max().item()
    print(f"xq exact match: {xq_match} max_diff={xq_close}")
    print(f"xs exact match: {xs_match} max_diff={xs_diff}")
    if not xq_match:
        bad = (xq_k != xq_ref).nonzero(as_tuple=False)
        print("first xq mismatch:", bad[:5].tolist())
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
