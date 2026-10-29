###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Shared pieces of the DeepEP roofline benchmarks: process setup, DeepSeek-V3 proxy routing, timing, results.
#
# Timing (every benchmark): before each iteration a FLUSH_MB write evicts L2; an optional untimed `pre`
# (a GPU barrier) runs; then CUDA events bracket the call. Per iteration the slowest rank counts; a round
# reports the median of ITERS iterations; ROUNDS rounds give mean (min-max), in microseconds.
import json
import os
import statistics

import torch
import torch.distributed as dist

RANK = int(os.environ["RANK"])
WORLD = int(os.environ["WORLD_SIZE"])
LOCAL = int(os.environ["LOCAL_RANK"])
ITERS = int(os.environ.get("ITERS", 30))
ROUNDS = int(os.environ.get("ROUNDS", 3))
FLUSH_MB = int(os.environ.get("FLUSH_MB", 512))


def init():
    torch.cuda.set_device(LOCAL)
    dist.init_process_group("nccl", device_id=torch.device("cuda", LOCAL))
    return dist.group.WORLD


_flush = None


def _flush_l2():
    global _flush
    if _flush is None:
        _flush = torch.empty(FLUSH_MB << 20, dtype=torch.uint8, device="cuda")
    _flush.zero_()


def bench(fn, pre=None, iters=None, rounds=None, warmup=3):
    """Times fn(); returns {"us", "min", "max", "rounds", "rank_us"} (rank_us: each rank's mean of medians)."""
    iters = iters or ITERS
    rounds = rounds or ROUNDS
    for _ in range(warmup):
        if pre is not None:
            pre()
        fn()
    torch.cuda.synchronize()
    round_meds, rank_meds = [], [[] for _ in range(WORLD)]
    for _ in range(rounds):
        dist.barrier()
        evs = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)
        ]
        for s, e in evs:
            _flush_l2()
            if pre is not None:
                pre()
            s.record()
            fn()
            e.record()
        torch.cuda.synchronize()
        mine = [s.elapsed_time(e) * 1e3 for s, e in evs]
        allr = [None] * WORLD
        dist.all_gather_object(allr, mine)
        round_meds.append(statistics.median(max(r[i] for r in allr) for i in range(iters)))
        for k in range(WORLD):
            rank_meds[k].append(statistics.median(allr[k]))
    return {
        "us": statistics.mean(round_meds),
        "min": min(round_meds),
        "max": max(round_meds),
        "rounds": round_meds,
        "rank_us": [statistics.mean(v) for v in rank_meds],
    }


def routing(num_tokens, num_experts, topk, kind, seed, num_groups=8, group_topk=4, scaling=2.5):
    """(topk_idx int64 [T, K], topk_weights float32 [T, K]).

    "dsv3": what the MI455X DeepSeek-V3 proxy (moe_router_force_load_balancing) routes: Megatron's
    RandomSTE logits (standard normal, a seed per rank), sigmoid scores, group-limited top-k (groups
    ranked by the sum of their top topk // group_topk scores, top group_topk groups kept), weights
    normalized and scaled by moe_router_topk_scaling_factor. The expert bias starts at zero.
    "uniform": topk distinct experts chosen uniformly at random."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    if kind == "uniform":
        idx = torch.rand(num_tokens, num_experts, device="cuda", generator=g).topk(topk, dim=1).indices
        w = torch.rand(num_tokens, topk, device="cuda", generator=g) + 0.1
        return idx.long(), (w / w.sum(1, keepdim=True)).float()
    assert kind == "dsv3", kind
    logits = torch.randn(num_tokens, num_experts, device="cuda", generator=g, dtype=torch.float32)
    scores = torch.sigmoid(logits)
    group_scores = scores.view(num_tokens, num_groups, -1).topk(topk // group_topk, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_groups, num_experts // num_groups)
        .reshape(num_tokens, -1)
    )
    probs, idx = scores.masked_fill(~score_mask.bool(), float("-inf")).topk(topk, dim=-1)
    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-20) * scaling
    return idx.long(), probs.float()


def gather_matrix(v):
    """Per-rank [WORLD] counts -> [sender, receiver] matrix on every rank (CPU)."""
    out = [torch.empty_like(v) for _ in range(WORLD)]
    dist.all_gather(out, v)
    return torch.stack(out).cpu()


def xgmi_rows(m):
    """m[s][r] rows from s to r. (most rows a rank receives from peers, most rows a rank sends to peers)."""
    recv = [int(sum(m[s][r] for s in range(WORLD) if s != r)) for r in range(WORLD)]
    send = [int(sum(m[s][r] for r in range(WORLD) if r != s)) for s in range(WORLD)]
    return max(recv), max(send)


class Results:
    def __init__(self, meta):
        self.data = dict(meta, world=WORLD, iters=ITERS, rounds=ROUNDS, flush_mb=FLUSH_MB, rows=[])

    def add(self, **row):
        self.data["rows"].append(row)
        if RANK == 0:
            extra = " ".join(
                f"{k}={v}" for k, v in row.items() if k not in ("us", "min", "max", "rounds", "rank_us")
            )
            if "us" in row:
                print(f"  {row['us']:10.1f} us ({row['min']:.1f}-{row['max']:.1f})  {extra}", flush=True)
            else:
                print(f"  {extra}", flush=True)

    def save(self):
        path = os.environ.get("OUT_JSON")
        if RANK == 0 and path:
            with open(path, "w") as f:
                json.dump(self.data, f, indent=1)


def finish(results):
    torch.cuda.synchronize()
    dist.barrier()
    results.save()
    if os.environ.get("EXIT_MODE") == "normal":
        # a profiler writes its output at exit; the teardown crash below hits a few percent of these runs
        dist.destroy_process_group()
        return
    # destroy_process_group trips the HIP VMM import use-after-free of this ROCm SDK; results are saved.
    # NCCL_CUMEM_ENABLE=0 would avoid it but changes RCCL's transport, so the benchmarks exit instead.
    os._exit(0)
