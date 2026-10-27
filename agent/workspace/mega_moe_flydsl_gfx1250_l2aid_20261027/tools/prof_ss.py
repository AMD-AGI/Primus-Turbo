"""prof_ss.py T: steady-state kernel trace of fused_mega_moe fwd+bwd (8 back-to-back iterations, one sync), DSV3
shapes, under `rocprofv3 --kernel-trace`."""
import os, sys, torch, torch.distributed as dist

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank))
g = dist.group.WORLD
import primus_turbo.pytorch  # noqa
from primus_turbo.pytorch.ops.moe.fused_mega_moe import fused_mega_moe

T, H, I, EPR, K = int(sys.argv[1]), 7168, 2048, 32, 8
E = EPR * world
gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
randn = lambda *s, scale=1.0: (torch.randn(*s, device="cuda", generator=gen) * scale).to(torch.bfloat16)
w1 = randn(EPR, 2 * I, H, scale=2 / H**0.5).requires_grad_()
w2 = randn(EPR, H, I, scale=2 / I**0.5).requires_grad_()
x = randn(T, H).requires_grad_()
tw, idx = torch.randn(T, E, device="cuda", generator=gen).softmax(-1).topk(K, dim=-1)
tw = tw.detach().requires_grad_()
dy = randn(T, H)


def step():
    for t in (x, tw, w1, w2):
        t.grad = None
    fused_mega_moe(g, x, idx, tw, w1, w2).backward(dy)


for _ in range(3):
    step()
torch.cuda.synchronize(); dist.barrier()
for _ in range(8):
    step()
torch.cuda.synchronize(); dist.barrier()
dist.destroy_process_group()
