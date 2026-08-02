#!/usr/bin/env python3
import os
import torch
import torch.distributed as dist
from primus_turbo.flydsl.mega.fp8 import quantize_rowwise_mxfp8_flydsl
from primus_turbo.flydsl.mega.fp8.dispatch_grouped_gemm_mxfp8_kernel import (
    _XQ_SCRATCH, _XS_SCRATCH, _FUSED_COMPILED, dispatch_grouped_gemm_mxfp8,
)
from primus_turbo.flydsl.mega.fp8 import dispatch_prologue, get_symm_buffer_for_mega_moe, preshuffle_b_scale
from primus_turbo.flydsl.mega.fp8.dispatch_grouped_gemm_mxfp8_kernel import _BSP_CACHE
from primus_turbo.pytorch.kernels.mega_moe.mega_moe_forward_fp8_impl import _w1_fp8_cached

port = os.environ.get("MASTER_PORT", "8660")
dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", world_size=1, rank=0)
torch.cuda.set_device(0)
T, H, I, E, K = 512, 7168, 2048, 8, 8
BM, BN = 256, 256
_XQ_SCRATCH.clear(); _XS_SCRATCH.clear(); _FUSED_COMPILED.clear()
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
    dist.group.WORLD, num_experts=E, num_max_tokens_per_rank=T, num_topk=K,
    hidden=H, intermediate_hidden=I, block_m=BM, block_n=BN, use_mxfp8=True,
)
sl = symm.make_sym_layout()
handle = tuple(dispatch_prologue(
    topk_idx, topk_w, sym_layout=sl, num_tokens=T, num_topk=K, num_experts=E,
    world_size=1, rank=0, experts_per_rank=E, block_m=BM,
    num_max_pool_tokens=symm.num_max_pool_tokens,
))
print("launch T=512", flush=True)
dispatch_grouped_gemm_mxfp8(x, w1q, w1s, handle, sl, symm, num_dispatch_cu=4, num_preshuffle_cu=4, BM=BM, BN=BN)
torch.cuda.synchronize()
sk = (T, H, x.device)
print("xq match", torch.equal(_XQ_SCRATCH[sk], xq_ref))
print("xs match", torch.equal(_XS_SCRATCH[sk], xs_ref))
dist.destroy_process_group()
