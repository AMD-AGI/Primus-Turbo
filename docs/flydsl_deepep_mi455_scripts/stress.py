# torchrun --nproc_per_node=2 stress.py
# Small rings / chunks so heads and tails wrap and back-pressure on every channel.
import os
import sys
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

os.environ["PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM"] = "1"

import dataclasses

import primus_turbo.flydsl.deep_ep.arch as arch

if os.environ.get("CHEAP_FENCE") == "1":
    arch._TRAITS["gfx1250"] = dataclasses.replace(arch._TRAITS["gfx1250"], cheap_fence=True)
    arch.get_arch_traits.cache_clear()

import dataclasses

import primus_turbo.flydsl.deep_ep.arch as arch

if os.environ.get("CHEAP_FENCE") == "1":
    arch._TRAITS["gfx1250"] = dataclasses.replace(arch._TRAITS["gfx1250"], cheap_fence=True)
    arch.get_arch_traits.cache_clear()

from primus_turbo.flydsl.deep_ep.intranode_kernel import combine_min_recv_buffer_tokens
from primus_turbo.pytorch.deep_ep.flydsl_intranode import FlyDSLIntranodeRuntime

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")
REPS = int(os.environ.get("REPS", 5))


def log(*a):
    if rank == 0:
        print(*a, flush=True)


rt = FlyDSLIntranodeRuntime(dist.group.WORLD, int(1e9), torch.cuda.current_stream())
K, E = 8, 64

# (num_sms, send_tokens, recv_tokens, T, H)
CASES = [
    (2, 1, 2, 37, 256),
    (2, 2, 4, 513, 7168),
    (8, 1, 2, 1000, 2048),
    (8, 4, 8, 4096, 7168),
    (8, 8, 16, 2049, 4096),
    (24, 6, 256, 4096, 7168),
    (24, 3, 7, 777, 1024),
    (8, 8, 256, 1, 7168),
    # 5760-byte rows: the last wave32 step of a copy / reduction runs past the row end
    (8, 4, 8, 1000, 2880),
    (24, 6, 256, 4096, 2880),
]

if os.environ.get("CASE"):
    CASES = [CASES[int(c)] for c in os.environ["CASE"].split(",")]
all_ok = True
for num_sms, send, recv, T, H in CASES:
    cfg = SimpleNamespace(num_sms=num_sms, num_max_nvl_chunked_send_tokens=send, num_max_nvl_chunked_recv_tokens=recv)
    # the combine ring at its deadlock bound, the dispatch one as small as asked
    combine_recv = max(recv, combine_min_recv_buffer_tokens(send))
    combine_cfg = SimpleNamespace(
        num_sms=num_sms, num_max_nvl_chunked_send_tokens=send, num_max_nvl_chunked_recv_tokens=combine_recv
    )
    case_ok = True
    t0 = time.time()
    for rep in range(REPS):
        if os.environ.get("VERBOSE"):
            log(f"  rep {rep}")
        g = torch.Generator(device="cuda").manual_seed(1000 * rep + rank)
        x = torch.randn((T, H), dtype=torch.bfloat16, device="cuda", generator=g)
        topk = torch.topk(torch.rand(T, E, device="cuda", generator=g), K, dim=-1)[1]
        topk[torch.rand(T, K, device="cuda", generator=g) < 0.3] = -1
        w = torch.rand(T, K, device="cuda", generator=g)
        xs = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(xs, x)

        ntr, _, nte, itr, _ = rt.get_dispatch_layout(topk, E, None, False, False)
        recv_x, _, _, rtw, _, rpm, _, rcpm, src, sh, _ = rt.intranode_dispatch(
            x, None, topk, w, ntr, itr, nte, 0, None, None, 1, 0, cfg, None, False, False
        )
        itrs = [torch.empty_like(itr) for _ in range(world)]
        dist.all_gather(itrs, itr)
        torch.cuda.synchronize()
        if os.environ.get("VERBOSE"):
            log("  dispatch done")

        start = 0
        for i in range(world):
            end = rpm[i][rank].item()
            want = itrs[i][:, rank].nonzero().flatten()
            got = src[start:end].long()
            ok = torch.equal(torch.sort(got)[0], want) and torch.equal(recv_x[start:end], xs[i][got])
            if not ok:
                print(f"[rank {rank}] DISPATCH MISMATCH case={(num_sms, send, recv, T, H)} rep={rep} from={i}", flush=True)
            case_ok &= ok
            start = end
        case_ok &= start == recv_x.size(0)

        out, _, _ = rt.intranode_combine(recv_x, rtw, None, None, src, rpm, rcpm, sh, combine_cfg, None, False, False)
        torch.cuda.synchronize()
        ref = x.float() * itr.sum(1, keepdim=True)
        ok = torch.allclose(out.float(), ref, rtol=2e-2, atol=2e-2)
        if not ok:
            print(f"[rank {rank}] COMBINE MISMATCH case={(num_sms, send, recv, T, H)} rep={rep} "
                  f"maxdiff={(out.float() - ref).abs().max().item()}", flush=True)
        case_ok &= ok

    flag = torch.tensor([int(case_ok)], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    log(f"sms={num_sms:2d} send={send} recv={recv:3d}/{combine_recv:3d} T={T:4d} H={H:4d} x{REPS}: "
        f"{'ok' if flag.item() else 'FAIL'} {time.time() - t0:.1f}s")
    all_ok &= bool(flag.item())

dist.barrier()
log("STRESS", "PASS" if all_ok else "FAIL")
rt.destroy()
dist.destroy_process_group()
sys.exit(0 if all_ok else 1)
