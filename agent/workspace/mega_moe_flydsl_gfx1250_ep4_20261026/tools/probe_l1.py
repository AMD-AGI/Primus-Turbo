"""L1 forward dispatch_gemm at EP>1: fused time vs comm-only and GEMM-only, per comm CU count.

torchrun --nproc-per-node N probe_l1.py T   (DSV3 per-rank shape, random routing; rank 0 prints)
"""
import os, sys, torch, torch.distributed as dist

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank))
import primus_turbo.pytorch  # noqa
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import (
    grouped_gemm_bf16_nt_flydsl_kernel as gg_nt,
)
from primus_turbo.flydsl.mega.bf16.dispatch_prologue_kernel import run_dispatch_prologue
from primus_turbo.flydsl.mega.bf16.symm_buffer import get_symm_buffer_for_mega_moe
from primus_turbo.flydsl.mega.bf16_gfx1250.dispatch_gemm_kernel import dispatch_gemm_nt_gfx1250, dispatch_only_gfx1250

T, H, I, EPR, K = int(sys.argv[1]), 7168, 2048, 32, 8
E = EPR * world
g = dist.group.WORLD
gen = torch.Generator(device="cuda").manual_seed(7 + rank)
w1 = (torch.randn(EPR, 2 * I, H, device="cuda", generator=gen) * 2 / H**0.5).to(torch.bfloat16)
x = torch.randn(T, H, device="cuda", generator=gen).to(torch.bfloat16)
tw, idx = torch.randn(T, E, device="cuda", generator=gen).softmax(-1).topk(K, dim=-1)
h, dw = run_dispatch_prologue(x, w1, g, idx, tw)
symm = get_symm_buffer_for_mega_moe()
pre64 = h.num_tokens_per_expert_prefix.to(torch.int64)
pool = h.pool_row_to_route.numel()
xp = symm.dispatch_token_buffer[h.pool_row_to_recv_token[:pool].long().clamp(max=symm.dispatch_token_buffer.shape[0] - 1)]


def timeit(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(n):
        fn()
    en.record()
    torch.cuda.synchronize()
    t = torch.tensor([st.elapsed_time(en) / n], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item() * 1e3


out = [f"EP{world} T={T}: pool rows {pool}, rows routed here {int(h.num_tokens_per_expert.sum())}"]
out.append(f"gg-pool (GEMM alone, 256 CUs)          {timeit(lambda: gg_nt(xp, w1, pre64)):7.0f} us")
for cu in (32, 64, 128, 255):
    out.append(f"dispatch_only {cu:3d} CUs                  {timeit(lambda: dispatch_only_gfx1250(x, h, num_dispatch_cu=cu)):7.0f} us")
for cu in (32, 64, 128):
    out.append(f"dispatch_gemm {cu:3d} comm CUs             {timeit(lambda: dispatch_gemm_nt_gfx1250(x, w1, h, num_dispatch_cu=cu)):7.0f} us")
if rank == 0:
    print("\n".join(out), flush=True)
dist.destroy_process_group()
