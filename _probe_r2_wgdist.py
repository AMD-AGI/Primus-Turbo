"""Optimize r1: wgrad tile-swizzle + split-K-policy cost across ALL FOUR bench distributions.

Run 1 of this probe settled the r1 scan's xcd axis: every xcd>1 row is +4% on balanced and
-50%/-63%/-62% on moderate/heavy/extreme (KB pitfalls/06 records -22%/-43% for the same
kernel family; this regime is worse), so xcd>1 can never be adopted and decoupling it from the
split-K switch buys nothing.  The open axis is the (group_m, group_n) band at xcd=1, where
(2,1,2) came out +1.3% balanced AND +1.1/+2.8/+2.9% on the three guardrails.

`mode`:
  prod      -- production
  nosplit   -- _wgrad_split_geom disabled: no window, no policy, grid = TOTAL (upper bound on
               everything the split-K path costs, and the value it returns on skew)
  polyfast  -- the O(G) per-workgroup policy scan replaced by ONE group_offs load that yields
               the same (lo, n, s, code) = (TOTAL, 0, 1, 0) at runtime without being a
               compile-time constant, so the per-tile slice arithmetic stays live.  Behaviour
               is IDENTICAL to production wherever the real policy returns s == 1 (moderate,
               extreme), which prices the scan alone.

usage: _probe_r2_wgdist.py <OUT_N>
"""

import sys

import torch

import flydsl.expr as fx

import primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel as MK
from primus_turbo.flydsl.grouped_gemm.gemm_fp8_grouped_kernel import _wgrad_split_ws
from primus_turbo.flydsl.utils.gemm_helper import _readfirstlane_i32

DEV = "cuda"
G = 32
PER = 4096
MTOT = G * PER
H = 2944
UNIT = 512
F8 = torch.float8_e4m3fn
PACK = 4
NCU = torch.cuda.get_device_properties(0).multi_processor_count

# (group_m, num_xcd, group_n, mode)
CFGS = [
    (4, 1, 0, "prod"),  # production
    (2, 1, 2, "prod"),
    (4, 1, 2, "prod"),
    (2, 1, 4, "prod"),
    (4, 1, 0, "polyfast"),
    (2, 1, 2, "polyfast"),
    (4, 1, 0, "nosplit"),
]


def _lens(weights):
    nu = MTOT // UNIT
    tot = sum(weights)
    u = [max(0, round(nu * w / tot)) for w in weights]
    u[u.index(max(u))] += nu - sum(u)
    return [x * UNIT for x in u]


DISTS = {
    "balanced": [PER] * G,
    "moderate": _lens([1.0 / (i + 1) ** 1.1 for i in range(G)]),
    "heavy": _lens([1.0 / (i + 1) ** 2.2 for i in range(G)]),
    "extreme": [MTOT // 4] * 4 + [0] * (G - 4),
}


def offs_of(lens):
    o = torch.zeros(G + 1, dtype=torch.int64, device=DEV)
    o[1:] = torch.tensor(lens, dtype=torch.int64, device=DEV).cumsum(0)
    return o


def timed(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(3):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) / iters)
    return best


def main():
    OUT_N = int(sys.argv[1])
    OUT_M = H
    torch.manual_seed(0)
    lhs = (torch.randn(OUT_M, MTOT, device=DEV) * 0.5).to(F8)
    rhs = (torch.randn(OUT_N, MTOT, device=DEV) * 0.5).to(F8)
    a_raw = torch.full((OUT_M * MTOT // 32,), 127, dtype=torch.uint8, device=DEV).view(torch.int32)
    b_raw = torch.full((OUT_N * MTOT // 32,), 127, dtype=torch.uint8, device=DEV).view(torch.int32)
    a8, b8 = lhs.view(torch.int8), rhs.view(torch.int8)
    out = torch.empty((G, OUT_M, OUT_N), dtype=torch.bfloat16, device=DEV)
    stream = torch.cuda.current_stream()
    K128 = MTOT // 128
    a_sp, b_sp = MK._get_grouped_wgrad_workspace(OUT_M, OUT_N, K128, G, PACK, DEV, stream)
    a_ngrp = (OUT_M + 63) // 64
    b_ngrp = ((OUT_N + 255) // 256) * 4
    n_ck = K128 // MK._PRESHUF_KT + G
    a_blocks = a_ngrp * n_ck
    pre_grid = a_blocks + b_ngrp * n_ck
    flop = 2.0 * MTOT * OUT_M * OUT_N
    ws = _wgrad_split_ws(OUT_M, OUT_N, G, DEV, torch.bfloat16, BLOCK_M=256, BLOCK_N=256)
    n_bm, n_bn = (OUT_M + 255) // 256, (OUT_N + 255) // 256
    print(f"OUT_M={OUT_M} OUT_N={OUT_N} blocks={n_bm}x{n_bn} tpg={n_bm*n_bn} total={G*n_bm*n_bn}", flush=True)

    _geom, _pol = MK._wgrad_split_geom, MK._wgrad_split_policy

    def _pol_fast(go_div, *a, **kw):
        z = _readfirstlane_i32(MK._load_go(go_div, 0))  # == 0, but opaque to the folder
        tot = a[2] if len(a) > 2 else kw["TOTAL"]
        return (
            _readfirstlane_i32(z + fx.Int32(tot)),
            _readfirstlane_i32(z),
            _readfirstlane_i32(z + fx.Int32(1)),
            _readfirstlane_i32(z),
        )

    def build(gm, xcd, gn, mode, ps):
        if mode == "nosplit":
            MK._wgrad_split_geom = lambda *a: (1, 1, 1, 0, 0)
        elif mode == "polyfast":
            MK._wgrad_split_policy = _pol_fast
        try:
            return MK._compile_grouped_mxfp8_wgrad_fused(
                OUT_M, OUT_N, G, 256, 256, gm, xcd, gn, 0, 0, False, pack=PACK, preshuffle=ps
            )
        finally:
            MK._wgrad_split_geom, MK._wgrad_split_policy = _geom, _pol

    launches = {}
    for cfg in CFGS:
        launches[cfg] = build(*cfg, False)
        print(f"  compiled {cfg}", flush=True)
    base_ps = build(*CFGS[0], True)

    res = {}
    for dist, lens in DISTS.items():
        go = offs_of(lens).view(torch.int32)
        args = (a8, b8, out, a_raw, b_raw, a_sp, b_sp, go, ws, MTOT, K128, n_ck, a_blocks, pre_grid, stream)
        base_ps(*args)  # per-dist scale preshuffle (each group packs from its own start)
        torch.cuda.synchronize()
        ref = out.clone()
        for cfg in list(CFGS) + list(reversed(CFGS)):  # palindrome: equal mean position per arm
            t = timed(lambda L=launches[cfg]: L(*args))
            res[(dist, cfg)] = min(t, res.get((dist, cfg), 1e9))
            res[(dist, cfg, "nd")] = int((out != ref).sum().item())
        b = res[(dist, CFGS[0])]
        for cfg in CFGS:
            t = res[(dist, cfg)]
            print(
                f"  {dist:9s} gm={cfg[0]:2d} gn={cfg[2]} xcd={cfg[1]} {cfg[3]:9s} "
                f"ms={t:.4f} TF={flop/(t*1e9):7.1f} rel={b/t:.4f} ndiff={res[(dist,cfg,'nd')]}",
                flush=True,
            )
    print("  --- summary rel-to-production (>1 = faster) ---", flush=True)
    for cfg in CFGS:
        rs = " ".join(f"{d}={res[(d,CFGS[0])]/res[(d,cfg)]:.4f}" for d in DISTS)
        print(f"  gm={cfg[0]:2d} gn={cfg[2]} xcd={cfg[1]} {cfg[3]:9s}  {rs}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
