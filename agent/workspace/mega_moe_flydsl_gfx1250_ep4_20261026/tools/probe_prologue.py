"""Phase times of the dispatch prologue (block 0's view), from prologue_timed.py's stamps.

FLYDSL_RUNTIME_ENABLE_CACHE=0 torchrun --nproc-per-node N probe_prologue.py T
"""
import os, sys, statistics, torch, torch.distributed as dist

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank))
sys.path.insert(0, os.path.dirname(__file__))
import primus_turbo.pytorch  # noqa
import prologue_timed as pt

T, H, I, EPR, K = int(sys.argv[1]), 7168, 2048, 32, 8
E = EPR * world
gen = torch.Generator(device="cuda").manual_seed(7 + rank)
w1 = torch.empty(EPR, 2 * I, H, device="cuda", dtype=torch.bfloat16)
x = torch.empty(T, H, device="cuda", dtype=torch.bfloat16)
tw, idx = torch.randn(T, E, device="cuda", generator=gen).softmax(-1).topk(K, dim=-1)
STAMPS = {0: "start", 1: "A histogram", 2: "grid_sync A", 3: "xgmi B1", 4: "push counts", 5: "xgmi B3",
          12: "C1 pool bases (sys loads)", 13: "C2 segment starts", 14: "C3 send tasks", 15: "C4 send offsets (serial)",
          6: "C5 expert tables", 7: "grid_sync C", 8: "D scatter", 9: "grid_sync D", 10: "reset", 11: "xgmi E"}
order = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 6, 7, 8, 9, 10, 11]
runs = []
for it in range(25):
    dist.barrier(); torch.cuda.synchronize()
    pt.DBG.zero_()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    pt.run_dispatch_prologue(x, w1, dist.group.WORLD, idx, tw)
    en.record(); torch.cuda.synchronize()
    d = pt.DBG.cpu().tolist()
    if it >= 5:
        runs.append(([(d[order[i]] - d[order[i - 1]]) / 100.0 for i in range(1, len(order))],
                     (d[11] - d[0]) / 100.0, st.elapsed_time(en) * 1e3))
lines = [f"rank {rank} EP{world} T={T}: kernel (block 0 start->end) median {statistics.median(r[1] for r in runs):.1f} us,"
         f" host-timed prologue call median {statistics.median(r[2] for r in runs):.1f} us"]
for i in range(1, len(order)):
    lines.append(f"  {STAMPS[order[i]]:28s} {statistics.median(r[0][i - 1] for r in runs):8.1f} us")
for r in range(world):
    if r == rank:
        print("\n".join(lines), flush=True)
    dist.barrier()
dist.destroy_process_group()
