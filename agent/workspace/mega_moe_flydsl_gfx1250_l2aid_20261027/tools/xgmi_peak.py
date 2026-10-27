"""xgmi_peak.py: what the 4-GPU XGMI fabric delivers, as the ceiling for the Mega dispatch / combine pushes.

HIP_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 xgmi_peak.py   (HSA_ENABLE_SDMA=0: blit-kernel copies)
  A. RCCL all_to_all_single (the unfused base path's all-to-all): per-rank egress to peers, GB/s.
  B. rank 0 alone, device-to-device peer copies, 1 GiB each, concurrent on separate streams:
     one link one way / both ways, one GPU to its 3 peers, 3 peers to one GPU, all 12 pairs at once.
GB/s = 1e9 bytes/s; egress counts only bytes leaving a GPU.
"""
import os, time, torch, torch.distributed as dist

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank))
dev = torch.device("cuda", rank)
sdma = os.environ.get("HSA_ENABLE_SDMA", "1")


def p0(*a):
    if rank == 0:
        print(*a, flush=True)


p0(f"HSA_ENABLE_SDMA={sdma}")
p0("A. RCCL all_to_all_single, per-rank egress (bytes to the 3 peers / time), max-over-ranks time")
for mb in (32, 128, 320, 1024):
    n = mb * (1 << 20) // world * world
    src = torch.empty(n, dtype=torch.uint8, device=dev)
    dst = torch.empty_like(src)
    for _ in range(5):
        dist.all_to_all_single(dst, src)
    torch.cuda.synchronize(); dist.barrier()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    iters = 20
    a.record()
    for _ in range(iters):
        dist.all_to_all_single(dst, src)
    b.record(); torch.cuda.synchronize()
    t = torch.tensor([a.elapsed_time(b) / iters], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    egress = n * (world - 1) / world
    p0(f"  {mb:5d} MiB/rank: {t.item():8.3f} ms  egress {egress / t.item() / 1e6:8.1f} GB/s per rank")
    del src, dst
torch.cuda.synchronize(); dist.barrier()

if rank == 0:
    print("B. peer copies issued by one process (others idle), 1 GiB per copy, concurrent streams", flush=True)
    G = torch.cuda.device_count()
    size = 1 << 30
    src = [torch.empty(size, dtype=torch.uint8, device=f"cuda:{i}") for i in range(G)]
    dst = {(i, j): torch.empty(size, dtype=torch.uint8, device=f"cuda:{j}")
           for i in range(G) for j in range(G) if i != j}
    streams = {k: torch.cuda.Stream(device=f"cuda:{k[1]}") for k in dst}

    def run(pairs, reps=8):
        def once():
            for k in pairs:
                with torch.cuda.stream(streams[k]):
                    dst[k].copy_(src[k[0]], non_blocking=True)
        once()
        for i in range(G):
            torch.cuda.synchronize(i)
        t0 = time.perf_counter()
        for _ in range(reps):
            once()
        for i in range(G):
            torch.cuda.synchronize(i)
        return (time.perf_counter() - t0) / reps

    cases = [
        ("one link, one way (0->1)", [(0, 1)]),
        ("one link, both ways (0<->1)", [(0, 1), (1, 0)]),
        ("GPU 0 -> its 3 peers", [(0, 1), (0, 2), (0, 3)]),
        ("3 peers -> GPU 0", [(1, 0), (2, 0), (3, 0)]),
        ("all 12 pairs (all-to-all)", list(dst)),
    ]
    for name, pairs in cases:
        t = run(pairs)
        per_link = size / t / 1e9
        out0 = sum(1 for k in pairs if k[0] == 0)
        in0 = sum(1 for k in pairs if k[1] == 0)
        print(f"  {name:<30} {t * 1e3:8.3f} ms  per copy {per_link:7.1f} GB/s  GPU0 egress {out0 * per_link:7.1f}"
              f"  GPU0 ingress {in0 * per_link:7.1f}  total {len(pairs) * per_link:7.1f} GB/s", flush=True)
dist.barrier()
dist.destroy_process_group()
