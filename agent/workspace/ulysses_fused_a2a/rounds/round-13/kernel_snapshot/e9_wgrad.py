"""bwd QKV A2AGemm with in-kernel wgrad: sweep configs, optional trace.

args: cfg ...   cfg = off | w,<wgrad_at>,<BMw>[,<BM>[,<push>[,<stagger>[,<push_split>]]]]
env: SEQS=16384,32768  TRACE=1 (trace the last cfg)
"""
import os, sys, torch, torch.distributed as dist
sys.path.insert(0, "/home/xiaompen/Turbo-dev/benchmark/ops/training")
import bench_ulysses_overlap as B
from primus_turbo.flydsl.ulysses.a2a_gemm_kernel import a2a_gemm_bank_bytes, ulysses_a2a_gemm
from primus_turbo.flydsl.ulysses.workspace import REGION_QKV_A2A_GEMM, get_ulysses_workspace
torch.cuda.set_device(int(os.environ["LOCAL_RANK"])); dist.init_process_group("nccl")
g = dist.group.WORLD; rank, n = dist.get_rank(), dist.get_world_size()
hq, hkv, d, hidden = 32, 4, 128, 2048
cfgs = sys.argv[1:]
seqs = [int(s) for s in os.environ.get("SEQS", "16384,32768").split(",")]
for seq in seqs:
    M = seq // n; K = (hq + 2 * hkv) * d
    torch.manual_seed(rank)
    parts = [torch.randn(seq, h // n, d, device="cuda").bfloat16() for h in (hq, hkv, hkv)]
    w = (torch.randn(K, hidden, device="cuda") * 0.02).bfloat16(); x = torch.randn(M, hidden, device="cuda").bfloat16()
    dx = torch.empty(M, hidden, device="cuda", dtype=torch.bfloat16); og = torch.empty(M, K, device="cuda", dtype=torch.bfloat16)
    dw = torch.empty(K, hidden, device="cuda", dtype=torch.bfloat16)
    ws = get_ulysses_workspace(g, a2a_gemm_bank_bytes(M, K))

    def run(cfg, trace=None):
        if cfg == "off":
            ulysses_a2a_gemm(parts, w, None, dx, og, ws, seq_local=M, qkv_format="sbhd", flag_region=REGION_QKV_A2A_GEMM, layout="nn", groups=1)
            torch.mm(og.t(), x, out=dw)
            return
        if cfg == "dgrad":
            ulysses_a2a_gemm(parts, w, None, dx, None, ws, seq_local=M, qkv_format="sbhd", flag_region=REGION_QKV_A2A_GEMM, layout="nn", groups=1)
            return
        f = cfg.split(",")[1:]
        at, bmw = int(f[0]), int(f[1])
        kw = {}
        if len(f) > 2 and f[2] != "0": kw["BM"] = int(f[2])
        if len(f) > 3 and f[3] != "0": kw["num_push"] = int(f[3])
        if len(f) > 4: kw["stagger"] = int(f[4])
        if len(f) > 5: kw["push_split"] = int(f[5])
        ulysses_a2a_gemm(parts, w, None, dx, None, ws, seq_local=M, qkv_format="sbhd", flag_region=REGION_QKV_A2A_GEMM,
                         layout="nn", groups=1, wgrad=(x, dw), BMw=bmw, wgrad_at=at, trace=trace, **kw)

    run("off"); torch.cuda.synchronize(); ref_dx, ref_dw = dx.clone(), dw.clone()
    res = []
    for cfg in cfgs:
        run(cfg); torch.cuda.synchronize()
        e1 = ((dx - ref_dx).float().norm() / ref_dx.float().norm()).item()
        e2 = ((dw - ref_dw).float().norm() / ref_dw.float().norm()).item()
        res.append((B.timeit(lambda: run(cfg), g, iters=30), max(e1, e2)))
    if rank == 0:
        print(f"seq={seq} " + "  ".join(f"{c}={t:.0f}(e{e:.0e})" for c, (t, e) in zip(cfgs, res)), flush=True)

    if os.environ.get("TRACE") and cfgs[-1].startswith("w"):
        cfg = cfgs[-1]; f = cfg.split(",")[1:]; bmw = int(f[1])
        BM = int(f[2]) if len(f) > 2 and f[2] != "0" else None
        num_push = int(f[3]) if len(f) > 3 and f[3] != "0" else 16
        nwt = K // bmw * (hidden // 256)
        if BM is None:
            BM = 256 if (M + 255) // 256 * (hidden // 256) + nwt + num_push <= 256 else 128
        mt = M // BM; tiles = mt * (hidden // 256)
        stagger = int(f[4]) > 0 if len(f) > 4 and f[4] != "-1" else (tiles + nwt) <= 256 - num_push
        S = (int(f[5]) if len(f) > 5 else 8) if stagger else 1; units = n * mt * S
        sp = 0
        tr = torch.zeros(3 * tiles + 2 * units + 2 * nwt + 3 * sp * (hidden // 256) * n, dtype=torch.int64, device="cuda")
        for it in range(4):
            torch.cuda.synchronize(); dist.barrier(); torch.cuda.synchronize(); torch.cuda._sleep(2_000_000)
            tr.zero_(); run(cfg, trace=tr)
        torch.cuda.synchronize()
        t = tr.cpu().double() / 100.0
        t[t == 0] = float("nan")
        T = t[: 3 * tiles].view(tiles, 3); U = t[3 * tiles: 3 * tiles + 2 * units].view(units, 2)
        Wt = t[3 * tiles + 2 * units: 3 * tiles + 2 * units + 2 * nwt].view(nwt, 2); Ps = t[3 * tiles + 2 * units + 2 * nwt:].view(-1, 3)
        t0 = min(T[:, 0].nan_to_num(1e30).min(), U[:, 0].nan_to_num(1e30).min(), Wt[:, 0].nan_to_num(1e30).min())
        T -= t0; U -= t0; Wt -= t0; Ps -= t0
        pct = lambda v: " ".join(f"{v[int(q*(len(v)-1))]:.0f}" for q in (0, .25, .5, .75, .9, 1))
        s = lambda v: v[~v.isnan()].sort().values
        lines = [f"rank{rank}: end={max(T[:,2].nan_to_num(0).max(), Wt[:,1].nan_to_num(0).max()):.0f} push_end={U[:,1].max():.0f}",
                 f"  dgrad start {pct(s(T[:,0]))} | ready {pct(s(T[:,1]))} | end {pct(s(T[:,2]))} | dur(after ready) {pct(s(T[:,2]-T[:,1]))}",
                 f"  wgrad start {pct(s(Wt[:,0]))} | end {pct(s(Wt[:,1]))} | dur {pct(s(Wt[:,1]-Wt[:,0]))}",
                 f"  push unit end {pct(s(U[:,1]))}"]
        sp = 0
        if sp:
            ns = sp * (hidden // 256); Ts = T[tiles - ns:]; Tw = T[:tiles - ns]
            lines += [f"  whole start {pct(s(Tw[:,0]))} | end {pct(s(Tw[:,2]))}",
                      f"  split start {pct(s(Ts[:,0]))} | end {pct(s(Ts[:,2]))} | dur {pct(s(Ts[:,2]-Ts[:,0]))}",
                      f"  part start {pct(s(Ps[:,0]))} | mma end {pct(s(Ps[:,1]))} | mma dur {pct(s(Ps[:,1]-Ps[:,0]))} | count dur {pct(s(Ps[:,2]-Ps[:,1]))} | reduce dur {pct(s(Ts[:,2] - Ps.view(-1, n, 3)[:, :, 2].nan_to_num(-1e30).max(1).values))}"]
        out = [None] * n; dist.all_gather_object(out, "\n".join(lines))
        if rank == 0: print("\n".join(out), flush=True)
dist.destroy_process_group()
