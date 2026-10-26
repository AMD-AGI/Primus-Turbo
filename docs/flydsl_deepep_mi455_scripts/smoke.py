# torchrun --nproc_per_node=2 smoke.py
import os
import sys
import time

import torch
import torch.distributed as dist

os.environ["PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM"] = "1"
from types import SimpleNamespace

from primus_turbo.pytorch.deep_ep.flydsl_intranode import FlyDSLIntranodeRuntime

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")


def log(*a):
    print(f"[rank {rank}]", *a, flush=True)


rt = FlyDSLIntranodeRuntime(dist.group.WORLD, int(1e9), torch.cuda.current_stream())
T = int(os.environ.get("T", 1024))
H = int(os.environ.get("H", 7168))
K, E = 8, 256
torch.manual_seed(rank)
x = torch.ones((T, H), dtype=torch.bfloat16, device="cuda") * (rank + 1)
topk = torch.topk(torch.rand(T, E, device="cuda"), K, dim=-1)[1]
w = torch.rand(T, K, device="cuda")
cfg = SimpleNamespace(num_sms=8, num_max_nvl_chunked_send_tokens=8, num_max_nvl_chunked_recv_tokens=256)

t0 = time.time()
ntr, _, nte, itr, _ = rt.get_dispatch_layout(topk, E, None, False, False)
torch.cuda.synchronize()
log("layout ok", ntr.tolist(), f"{time.time() - t0:.1f}s")

t0 = time.time()
recv_x, _, _, rtw, _, rpm, _, rcpm, src, sh, _ = rt.intranode_dispatch(
    x, None, topk, w, ntr, itr, nte, 0, None, None, 1, 0, cfg, None, False, False
)
torch.cuda.synchronize()
log("dispatch done", tuple(recv_x.shape), f"{time.time() - t0:.1f}s")
start = 0
ok = True
for i in range(world):  # rows from rank i carry the value i + 1
    end = rpm[i][rank].item()
    good = bool((recv_x[start:end].float() == i + 1).all())
    ok &= good
    log(f"  rows from rank {i}: [{start}, {end}) ok={good}")
    start = end

t0 = time.time()
out, _, _ = rt.intranode_combine(recv_x, rtw, None, None, src, rpm, rcpm, sh, cfg, None, False, False)
torch.cuda.synchronize()
log("combine done", f"{time.time() - t0:.1f}s")
good = torch.allclose(out.float(), x.float() * itr.sum(1, keepdim=True))
ok &= good
log("combine ok", good)
dist.barrier()
log("SMOKE", "PASS" if ok else "FAIL")
rt.destroy()
dist.destroy_process_group()
sys.exit(0 if ok else 1)
