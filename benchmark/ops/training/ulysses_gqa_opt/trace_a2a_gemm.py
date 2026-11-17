"""CU-time breakdown of one A2AGemm launch from its in-kernel trace (100 MHz realtime stamps).

torchrun --nproc_per_node 4 trace_a2a_gemm.py --model glm-4.7 --seq 32768
"""

import os
import sys

_REPO = os.environ.get(
    "TURBO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")),
)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "benchmark", "ops", "training"))

import argparse
import os

import torch
import torch.distributed as dist
from bench_ulysses_overlap import MODELS, build_ops, timeit

import primus_turbo.pytorch.ops.attention.ulysses_fused_a2a as U

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--seq", type=int, default=32768)
p.add_argument("--ops", nargs="+", default=["fwd_o", "bwd_qkv"], choices=["fwd_o", "bwd_qkv"])
p.add_argument("--BM", type=int, default=None)
p.add_argument("--num_push", type=int, default=None)
p.add_argument("--wgrad_at", type=int, default=None)
a = p.parse_args()
OVERRIDE = {
    k: v for k, v in (("BM", a.BM), ("num_push", a.num_push), ("wgrad_at", a.wgrad_at)) if v is not None
}
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dist.init_process_group("nccl")
g = dist.group.WORLD
n, rank = g.size(), g.rank()
ops = build_ops(MODELS[a.model], a.seq, 1, g)
orig = U.ulysses_a2a_gemm
last = {}


def traced(*args, **kw):
    buf = torch.zeros(1 << 22, dtype=torch.int64, device="cuda")
    last["buf"] = buf
    captured = {}
    import primus_turbo.flydsl.ulysses.a2a_gemm_kernel as K

    real = K.run_launcher

    def spy(launch, *la, **lk):
        captured["args"] = la
        return real(launch, *la, **lk)

    K.run_launcher = spy
    try:
        r = orig(*args, trace=buf, **{**kw, **OVERRIDE})
    finally:
        K.run_launcher = real
    last["args"] = captured["args"]
    return r


U.ulysses_a2a_gemm = traced
num_cus = torch.cuda.get_device_properties(0).multi_processor_count
NAMES = {"fwd_o": "fwd o_proj (A2AGemm)", "bwd_qkv": "bwd QKV (A2AGemm+wgrad)"}


def busy_split(intervals, grid, end):
    """(idle CU-us after the last moment every CU was busy, idle CU-us before it): the tail and the gaps."""
    ev = sorted([(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals])
    busy, t_prev, last_full = 0, 0.0, 0.0
    segs = []
    for t, d in ev:
        segs.append((t_prev, t, busy))
        busy += d
        t_prev = t
    segs.append((t_prev, end, busy))
    for s, e, b in segs:
        if b >= grid:
            last_full = e
    tail = sum((e - max(s, last_full)) * (grid - b) for s, e, b in segs if e > last_full)
    gaps = sum((e - s) * max(grid - b, 0) for s, e, b in segs if e <= last_full)
    gaps += sum((last_full - s) * max(grid - b, 0) for s, e, b in segs if s < last_full < e)
    return tail, gaps, last_full


for name in (NAMES[o] for o in a.ops):
    fused = ops[name][5]
    for _ in range(5):
        fused()
    us = timeit(fused, g, warmup=3, iters=10)
    torch.cuda.synchronize()
    la = last["args"]
    # positional launcher args after the 12 tensors: M K N BM BN rank num_ranks seq_local is_bshd has_res save_o num_push push_split
    M, Kd, N, BM, BN = la[12:17]
    num_push, push_split = la[23], la[24]
    m_tiles, n_tiles = -(-M // BM), -(-N // BN)
    num_tiles = m_tiles * n_tiles
    BMw = la[35]
    num_wtiles = Kd // BMw * (N // BN) if BMw else 0
    grid = min(num_tiles + num_wtiles + num_push, num_cus)
    stagger = m_tiles // n if num_tiles + num_wtiles <= grid - min(num_push, grid) else 0
    push_split = push_split if stagger else 1
    units = n * m_tiles * push_split
    t = last["buf"].cpu()
    tiles = t[: 3 * num_tiles].view(num_tiles, 3).double()
    pu = t[3 * num_tiles : 3 * num_tiles + 2 * units].view(units, 2).double()
    wstart = 3 * num_tiles + 2 * units
    w = t[wstart : wstart + 2 * num_wtiles].view(-1, 2).double()
    w = w[(w[:, 0] > 0)]
    pu = pu[pu[:, 0] > 0]
    stamps = torch.cat([tiles.flatten(), pu.flatten(), w.flatten()])
    t0 = stamps[stamps > 0].min()
    f = lambda x, t0=t0: (x - t0) / 100.0  # us
    end = max(f(tiles[:, 2]).max().item(), f(w[:, 1]).max().item() if len(w) else 0)
    push_end = f(pu[:, 1]).max().item()
    land_last = f(tiles[:, 1]).max().item()
    # the landed stamp is taken when the last source segment's flag is seen inside the K walk, so
    # end - start mixes compute and waits; tiles whose rows had all landed at claim time are pure compute
    dur = tiles[:, 2] - tiles[:, 0]
    pre = (tiles[:, 1] - tiles[:, 0]) < 100
    pure = dur[pre].median().item() / 100 if pre.any() else float("nan")
    stall = float(pre.sum().item())
    gemm_cu = dur.sum().item() / 100
    push_cu = (pu[:, 1] - pu[:, 0]).sum().item() / 100
    w_cu = (w[:, 1] - w[:, 0]).sum().item() / 100 if len(w) else 0.0

    w_dur = (w[:, 1] - w[:, 0]) / 100 if len(w) else torch.zeros(1, dtype=torch.float64)
    w_late = (f(w[:, 0]) > push_end) if len(w) else torch.zeros(1, dtype=torch.bool)
    w_pure = w_dur[w_late].median().item() if w_late.any() else float("nan")
    ivs = [(f(s).item(), f(e).item()) for s, e in zip(tiles[:, 0], tiles[:, 2])]
    ivs += [(f(s).item(), f(e).item()) for s, e in zip(pu[:, 0], pu[:, 1])]
    ivs += [(f(s).item(), f(e).item()) for s, e in zip(w[:, 0], w[:, 1])] if len(w) else []
    tail_idle, gap_idle, last_full = busy_split(ivs, grid, end)
    hip_g = timeit(ops[name][0], g)
    hip_w = timeit(ops[name][1], g) if ops[name][1] else 0.0
    d_start, d_end = f(tiles[:, 0]).min().item(), f(tiles[:, 2]).max().item()
    w_start = f(w[:, 0]).min().item() if len(w) else 0.0
    w_end = f(w[:, 1]).max().item() if len(w) else 0.0
    res = torch.tensor(
        [
            us,
            end,
            push_end,
            land_last,
            stall,
            gemm_cu,
            push_cu,
            w_cu,
            pure,
            w_pure,
            tail_idle,
            gap_idle,
            last_full,
        ]
        + [d_start, d_end, w_start, w_end],
        dtype=torch.float64,
        device="cuda",
    )
    allr = [torch.zeros_like(res) for _ in range(n)]
    dist.all_gather(allr, res, group=g)
    if rank == 0:
        hip_tile = hip_g * num_cus / num_tiles
        print(
            f"\n== {a.model} n{n} {a.seq // 1024}K {name}: M={M} K={Kd} N={N} BM={BM} tiles={num_tiles} wgrad_tiles={len(w)} push={num_push} split={push_split} stagger={stagger} grid={grid}"
        )
        print(
            f"   hipBLASLt gemm {hip_g:.0f} us  wgrad {hip_w:.0f} us -> {num_cus}-CU work {(hip_g + hip_w) * num_cus:.0f} CU-us, {hip_tile:.1f} CU-us per {BM}x{BN} tile"
        )
        print(
            f"   {'rank':>4} {'timed':>7} {'kernel':>7} {'pushEnd':>8} {'lastLand':>9} {'tail':>6} | CU-us: {'tiles':>8} {'wgrad':>7} {'push':>7} {'idle':>7} | {'landed':>6} {'pureTile':>8} {'vsHip':>6} | tile eff"
        )
        for r, v in enumerate(allr):
            us_, end_, pe, ll, nlanded, gc, pc, wc, pure_ = v.tolist()[:9]
            idle = grid * end_ - gc - pc - wc
            eff = hip_g * num_cus / max(gc, 1)
            print(
                f"   {r:>4} {us_:7.0f} {end_:7.0f} {pe:8.0f} {ll:9.0f} {end_ - ll:6.0f} |        {gc:8.0f} {wc:7.0f} {pc:7.0f} {idle:7.0f} | {nlanded:6.0f} {pure_:8.1f} {pure_ / hip_tile:6.2f} | {eff:5.2f}"
            )
        # kernel span x all CUs = hipBLASLt work + losses, in us of the whole GPU (CU-us / num_cus)
        hip_wtile = hip_w * num_cus / max(len(w), 1)
        print(
            f"   breakdown (us = CU-us / {num_cus}; tile/wgrad cfg = (pure - hipBLASLt) x count, wait = rest of the "
            f"tile time; pure wgrad = median of wgrad tiles started after the last push, hip {hip_wtile:.1f} CU-us/tile)"
        )
        print(
            f"   {'rank':>4} {'kernel':>7} = {'hipWork':>7} + {'tileCfg':>7} {'tileWait':>8} {'wgCfg':>6} {'wgWait':>6}"
            f" {'push':>6} {'tail':>6} {'gaps':>6} {'noCU':>5} | lastFull {'wgPure':>7}"
        )
        for r, v in enumerate(allr):
            us_, end_, pe, ll, nlanded, gc, pc, wc, pure_, wp, ti, gi, lf = v.tolist()[:13]
            pure_ = pure_ if pure_ == pure_ else gc / num_tiles
            wp_ = wp if wp == wp else (wc / max(len(w), 1))
            parts = [
                (hip_g + hip_w) * num_cus,
                (pure_ - hip_tile) * num_tiles,
                gc - pure_ * num_tiles,
                (wp_ - hip_wtile) * len(w) if len(w) else 0.0,
                wc - wp_ * len(w) if len(w) else 0.0,
                pc,
                ti,
                gi,
                (num_cus - grid) * end_,
            ]
            print(
                f"   {r:>4} {end_:7.0f} = "
                + " ".join(f"{x / num_cus:>{wd}.0f}" for x, wd in zip(parts, (7, 7, 8, 6, 6, 6, 6, 6, 5)))
                + f" | {lf:8.0f} {wp:7.1f}  (sum {sum(parts) / num_cus:.0f})"
            )
        print(
            f"   timeline (us from the first stamp): {'rank':>4} {'gemm tiles':>15} {'wgrad tiles':>15} {'pushEnd':>8}"
        )
        for r, v in enumerate(allr):
            ds, de, ws, we = v.tolist()[13:17]
            print(f"   {'':>35} {r:>4} {ds:6.0f} - {de:6.0f} {ws:6.0f} - {we:6.0f} {v.tolist()[2]:8.0f}")
dist.barrier()
dist.destroy_process_group()
