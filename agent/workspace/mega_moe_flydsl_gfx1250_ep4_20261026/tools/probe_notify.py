"""Host view of the prologue's rows notify: when the host learns the rows vs when the kernel is done.

torchrun --nproc-per-node N probe_notify.py T
"""
import os, sys, time, statistics, torch, torch.distributed as dist

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank))
import primus_turbo.pytorch  # noqa
import primus_turbo.flydsl.mega.bf16.dispatch_prologue_kernel as dp

T, H, I, EPR, K = int(sys.argv[1]), 7168, 2048, 32, 8
E = EPR * world
gen = torch.Generator(device="cuda").manual_seed(7 + rank)
w1 = torch.empty(EPR, 2 * I, H, device="cuda", dtype=torch.bfloat16)
x = torch.empty(T, H, device="cuda", dtype=torch.bfloat16)
tw, idx = torch.randn(T, E, device="cuda", generator=gen).softmax(-1).topk(K, dim=-1)

log = {}
_orig_wait = dp._wait_host_rows


def _wait(host_rows, prefix):
    t_in = time.perf_counter()
    spins = 0
    while (rows := host_rows.cell.value) < 0:
        spins += 1
        if host_rows.done.query():
            rows = host_rows.cell.value
            log["fallback"] = rows < 0
            if rows < 0:
                rows = int(prefix[-1].item())
            break
    else:
        log["fallback"] = False
    log["t_rows"] = time.perf_counter()
    log["done_at_rows"] = host_rows.done.query()
    log["spins"] = spins
    log["t_wait_in"] = t_in
    log["rows_ok"] = rows == int(prefix[-1].item())
    return rows


dp._wait_host_rows = _wait
runs = []
for it in range(30):
    dist.barrier(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    dp.run_dispatch_prologue(x, w1, dist.group.WORLD, idx, tw)
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    if it >= 5:
        runs.append((log["fallback"], log["done_at_rows"], log["rows_ok"], log["spins"],
                     (log["t_wait_in"] - t0) * 1e6, (log["t_rows"] - t0) * 1e6, (t1 - t0) * 1e6, (t2 - t0) * 1e6))
fb = sum(r[0] for r in runs); dn = sum(r[1] for r in runs); ok = sum(r[2] for r in runs)
med = lambda i: statistics.median(r[i] for r in runs)
msg = (f"rank {rank}: fallback {fb}/{len(runs)}, kernel already done when rows seen {dn}/{len(runs)}, rows match {ok}/{len(runs)}, "
       f"spins med {med(3):.0f}; host us from call: wait entered {med(4):.0f}, rows seen {med(5):.0f}, call returned {med(6):.0f}, "
       f"GPU idle (sync) {med(7):.0f}")
for r in range(world):
    if r == rank:
        print(msg, flush=True)
    dist.barrier()
dist.destroy_process_group()
