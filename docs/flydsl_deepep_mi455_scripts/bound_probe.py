# torchrun --nproc_per_node=2 bound_probe.py ; env S, W = combine send / recv tokens
# Every token goes to every rank, one channel: the combine ring bound is hit exactly.
import os
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

os.environ["PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM"] = "1"

from primus_turbo.pytorch.deep_ep.flydsl_intranode import FlyDSLIntranodeRuntime

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")
S, W = int(os.environ["S"]), int(os.environ["W"])
T, H, E = int(os.environ.get("T", 200)), 1024, 64

rt = FlyDSLIntranodeRuntime(dist.group.WORLD, int(1e9), torch.cuda.current_stream())
x = torch.randn((T, H), dtype=torch.bfloat16, device="cuda")
per_rank = E // world
topk = torch.tensor([r * per_rank + j for r in range(world) for j in range(4)], device="cuda").repeat(T, 1)
w = torch.rand(topk.shape, device="cuda")
dispatch_cfg = SimpleNamespace(num_sms=2, num_max_nvl_chunked_send_tokens=8, num_max_nvl_chunked_recv_tokens=256)
combine_cfg = SimpleNamespace(num_sms=2, num_max_nvl_chunked_send_tokens=S, num_max_nvl_chunked_recv_tokens=W)

ntr, _, nte, itr, _ = rt.get_dispatch_layout(topk, E, None, False, False)
recv_x, _, _, rtw, _, rpm, _, rcpm, src, sh, _ = rt.intranode_dispatch(
    x, None, topk, w, ntr, itr, nte, 0, None, None, 1, 0, dispatch_cfg, None, False, False
)
torch.cuda.synchronize()
out, _, _ = rt.intranode_combine(recv_x, rtw, None, None, src, rpm, rcpm, sh, combine_cfg, None, False, False)
torch.cuda.synchronize()
ok = torch.allclose(out.float(), x.float() * world, rtol=2e-2, atol=2e-2)
print(f"[rank {rank}] S={S} W={W} combine {'ok' if ok else 'MISMATCH'}", flush=True)
dist.barrier()
rt.destroy()
dist.destroy_process_group()
sys.exit(0 if ok else 1)
