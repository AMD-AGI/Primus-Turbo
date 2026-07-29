#!/usr/bin/env python3
"""Compare rowcol_dual_quant rowwise half vs quantize_rowwise_mxfp8_flydsl on real grad_l1."""
import argparse
import datetime
import math
import os

import torch
import torch.distributed as dist

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.flydsl.mega.fp8 import (
    dispatch_grouped_gemm_mxfp8,
    dispatch_prologue,
    get_symm_buffer_for_mega_moe,
    quantize_grouped_weight_mxfp8,
)
from primus_turbo.flydsl.mega import swiglu_backward_flydsl_kernel
from primus_turbo.flydsl.mega.fp8.quant_colwise_trans_flydsl import (
    colwise_grouped_meta,
    rowcol_dual_quant_mxfp8_grouped_flydsl,
)
from primus_turbo.flydsl.mega.fp8.quant_flydsl import quantize_rowwise_mxfp8_flydsl
from primus_turbo.pytorch.kernels.mega_moe.mega_moe_backward_fp8_impl import (
    _DW_FP8_FORMAT,
    _dispatch_l2_dgrad_mxfp8_flydsl_kernel,
)


def worker(local_rank, world, args):
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8492"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://{ip}:{port}", world_size=world, rank=local_rank,
        timeout=datetime.timedelta(seconds=180),
    )
    torch.set_default_device("cuda")
    group = dist.new_group(list(range(world)))
    rank = dist.get_rank()
    T, H, I, E, K = args.num_tokens, 7168, 2048, 256, 8
    epr = E // world
    BM, BN = 256, 256

    g = torch.Generator(device="cuda").manual_seed(100 + rank)
    scores = torch.rand(T, E, generator=g, device="cuda").abs() + 1
    topk_w, topk_idx = torch.topk(scores.softmax(-1), K, dim=-1)
    topk_idx, topk_w = topk_idx.to(torch.int64), topk_w.to(torch.float32)

    g2 = torch.Generator(device="cuda").manual_seed(1234)
    W1 = torch.randn((epr, 2 * I, H), generator=g2, device="cuda", dtype=torch.bfloat16) * (2.0 / math.sqrt(H))
    W2 = torch.randn((epr, H, I), generator=g2, device="cuda", dtype=torch.bfloat16) * (2.0 / math.sqrt(I))
    x = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)
    dy = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)

    symm = get_symm_buffer_for_mega_moe(
        group, num_experts=E, num_max_tokens_per_rank=T, num_topk=K, hidden=H,
        intermediate_hidden=I, block_m=BM, block_n=BN, use_mxfp8=True,
    )
    sl = symm.make_sym_layout()
    handle = tuple(dispatch_prologue(
        topk_idx, topk_w, sym_layout=sl, num_tokens=T, num_topk=K, num_experts=E,
        world_size=world, rank=symm.rank, experts_per_rank=epr, block_m=BM,
        num_max_pool_tokens=symm.num_max_pool_tokens,
    ))
    w1q, w1s = quantize_grouped_weight_mxfp8(W1)
    torch.cuda.synchronize()
    group.barrier()
    l1 = dispatch_grouped_gemm_mxfp8(x, None, w1q, w1s, handle, sl, symm, BM=BM, BN=BN)
    dw = symm.weight_recv_buf.clone()
    grad_swiglu, _ = _dispatch_l2_dgrad_mxfp8_flydsl_kernel(dy, W2, group, handle, BM, BN)
    grad_l1, _ = swiglu_backward_flydsl_kernel(
        grad_swiglu, l1, symm.meta_scalars[1:2], scale=dw, return_gate=True,
    )

    meta = colwise_grouped_meta(handle[9], handle[10])
    m_used = int(handle[10][-1].item())
    lens, offs = handle[9], handle[10]
    real_mask = torch.zeros(m_used, dtype=torch.bool, device="cuda")
    for g in range(lens.numel()):
        s0 = int(offs[g].item())
        lg = int(lens[g].item())
        if lg > 0:
            real_mask[s0 : s0 + lg] = True
    n_real = int(real_mask.sum().item())

    gl1_q_row, gl1_a_sp, _, _ = rowcol_dual_quant_mxfp8_grouped_flydsl(grad_l1, _DW_FP8_FORMAT, meta=meta)
    # Match production backward: standalone quant on the full symm pool buffer.
    q_sep, a_sp_sep = quantize_rowwise_mxfp8_flydsl(grad_l1.contiguous(), preshuffle=True)

    q_d = gl1_q_row[:m_used].view(torch.int8)[real_mask]
    q_s = q_sep[:m_used].view(torch.int8)[real_mask]
    q_diff = int((q_d != q_s).sum().item())
    q_ok = float(q_diff == 0)
    sp_ok = float(torch.equal(gl1_a_sp, a_sp_sep)) if gl1_a_sp.shape == a_sp_sep.shape else 0.0
    row_bad = int((gl1_q_row[:m_used].view(torch.int8)[real_mask].reshape(n_real, -1)
                   != q_sep[:m_used].view(torch.int8)[real_mask].reshape(n_real, -1)).any(dim=1).sum().item())
    sp_shape_ok = float(gl1_a_sp.shape == a_sp_sep.shape)
    t = torch.tensor([q_ok, sp_ok, sp_shape_ok, float(q_diff), float(row_bad), float(n_real)], device="cuda")
    dist.all_reduce(t[:3], op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(t[3:], op=dist.ReduceOp.MAX, group=group)
    if rank == 0:
        print(f"m_used={m_used} n_real={int(t[5])}  grad_l1={tuple(grad_l1.shape)}")
        print(f"q_byte_equal(real rows)={bool(t[0])}  a_sp_equal={bool(t[1])}  a_sp_shape_match={bool(t[2])}")
        print(f"a_sp shapes dual={tuple(gl1_a_sp.shape)} sep={tuple(a_sp_sep.shape)}")
        print(f"mismatched_q_bytes={int(t[3])}  real_rows_with_q_mismatch={int(t[4])}/{int(t[5])}")

    symm.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--num-tokens", type=int, default=8192)
    args = ap.parse_args()
    if args.num_processes == 1:
        worker(0, 1, args)
    else:
        torch.multiprocessing.spawn(worker, args=(args.num_processes, args), nprocs=args.num_processes)
