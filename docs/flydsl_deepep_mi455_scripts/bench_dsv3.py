# torchrun --nproc_per_node=2 bench_dsv3.py ; the DSv3 EP2 proxy's MoE shape (8192 tokens, hidden 7168,
# topk 8 of 64 experts). env CHEAP_FENCE=1 swaps gfx1250's fence for the gfx9 one (timing only).
import dataclasses
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

os.environ["PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM"] = "1"

import primus_turbo.flydsl.deep_ep.arch as arch

if os.environ.get("CHEAP_FENCE") == "1":
    arch._TRAITS["gfx1250"] = dataclasses.replace(arch._TRAITS["gfx1250"], cheap_fence=True)
    arch.get_arch_traits.cache_clear()

from primus_turbo.pytorch.deep_ep.flydsl_intranode import FlyDSLIntranodeRuntime

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")
T, H, K, E = 8192, 7168, 8, 64
NUM_SMS = int(os.environ.get("NUM_SMS", 32))

rt = FlyDSLIntranodeRuntime(dist.group.WORLD, int(2e9), torch.cuda.current_stream())
torch.manual_seed(rank)
x = torch.randn((T, H), dtype=torch.bfloat16, device="cuda")
topk = torch.stack([torch.randperm(E, device="cuda")[:K] for _ in range(T)])
w = torch.rand(T, K, device="cuda")
dcfg = SimpleNamespace(num_sms=NUM_SMS, num_max_nvl_chunked_send_tokens=24, num_max_nvl_chunked_recv_tokens=256)
ccfg = SimpleNamespace(num_sms=NUM_SMS, num_max_nvl_chunked_send_tokens=10, num_max_nvl_chunked_recv_tokens=256)


def timed(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


ntr, _, nte, itr, _ = rt.get_dispatch_layout(topk, E, None, False, False)
state = {}


def dispatch():
    state["d"] = rt.intranode_dispatch(x, None, topk, w, ntr, itr, nte, 0, None, None, 1, 0, dcfg, None, False, False)


dispatch()
recv_x, _, _, rtw, _, rpm, _, rcpm, src, sh, _ = state["d"]


def combine():
    rt.intranode_combine(recv_x, rtw, None, None, src, rpm, rcpm, sh, ccfg, None, False, False)


t_d, t_c = timed(dispatch), timed(combine)
remote_rows = int(itr[:, 1 - rank].sum())
# Megatron's all-to-all dispatcher sends one row per (token, expert) pair, half of them to the peer
a2a_in = torch.empty((T * K, H), dtype=torch.bfloat16, device="cuda")
a2a_out = torch.empty_like(a2a_in)
t_a2a = timed(lambda: dist.all_to_all_single(a2a_out, a2a_in))
if rank == 0:
    mb = recv_x.numel() * 2 / 1e6
    print(
        f"cheap_fence={os.environ.get('CHEAP_FENCE', '0')} num_sms={NUM_SMS}: dispatch {t_d:.3f} ms, combine {t_c:.3f} ms "
        f"(recv {recv_x.size(0)} rows = {mb:.0f} MB, {remote_rows} rows to the peer); "
        f"rccl all_to_all_single of {T * K} rows {t_a2a:.3f} ms",
        flush=True,
    )
rt.destroy()
dist.destroy_process_group()
