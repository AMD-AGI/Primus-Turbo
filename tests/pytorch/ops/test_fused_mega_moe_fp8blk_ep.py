###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""fused_mega_moe_fp8blk_ep (EP>1 FP8-block MegaMoE) vs EP=1 fused_mega_moe_fp8blk, on one GPU.

The EP group is emulated in one process (EmulatedEPComm(R)): the T tokens are split into R rank shards of T/R tokens,
the E experts into R shards of E/R, and the all-to-alls become device copies with the same block semantics. Compares
y, dx, d topk_w, dW1, dW2 bit for bit against EP=1 on the same inputs; where bits differ, reports the error of both vs
an fp64 (unquantized) reference.

  python test_fused_mega_moe_fp8blk_ep.py dry                  # new kernels, tiny shape (first run on a node)
  python test_fused_mega_moe_fp8blk_ep.py full [--real] [--small] [--R 1,4,8] [--dispatch fp8,bf16] [--ref64]
  python test_fused_mega_moe_fp8blk_ep.py graph                # CUDA-graph capture of fwd+bwd (R=4) vs eager
  python test_fused_mega_moe_fp8blk_ep.py time [--R 4,8]       # per-stage kernel time, EP=1 vs emulated EP
"""

import argparse
import os
import sys
import time

import torch

import primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8blk as FM
import primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8blk_ep as EP
from primus_turbo.triton.quantization import quant_blk_ep_gfx1250 as QE
from primus_turbo.triton.quantization.quant_blk_gfx1250 import build_routing_blk, dequant, quant_dual_blk

dev = "cuda"
_BITS = {torch.bfloat16: torch.int16, torch.float32: torch.int32, torch.float8_e4m3fn: torch.uint8,
         torch.uint8: torch.uint8, torch.int32: torch.int32}


def bits_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(_BITS[a.dtype]), b.view(_BITS[b.dtype]))


def nbits_diff(a, b):
    return int((a.view(_BITS[a.dtype]) != b.view(_BITS[b.dtype])).sum().item())


def rel(a, ref):
    a, ref = a.double(), ref.double()
    return ((a - ref).norm() / ref.norm().clamp_min(1e-300)).item(), (a - ref).abs().max().item()


# ------------------------------------------------------------------------------------------------- inputs
def make_inputs(T, H, I, E, K, routing="uniform", seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(T, H, device=dev, generator=g).to(torch.bfloat16)
    w1 = (torch.randn(E, 2 * I, H, device=dev, generator=g) * H ** -0.5).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, device=dev, generator=g) * I ** -0.5).to(torch.bfloat16)
    score = torch.randn(T, E, device=dev, generator=g)
    if routing == "skew":  # rank 0's experts hot, last expert never routed (empty, 256 padding rows)
        score = score + torch.linspace(3.0, -3.0, E, device=dev)[None, :]
        score[:, E - 1] = -float("inf")
    top, idx = score.topk(K, dim=1)
    tw = torch.softmax(top, dim=1).float()
    dy = torch.randn(T, H, device=dev, generator=g).to(torch.bfloat16)
    return x, idx, tw, w1, w2, dy


def _param(w, accumulate):
    p = w.detach().clone().requires_grad_(True)
    if accumulate:
        p.main_grad = torch.zeros_like(p)
    return p


def run_ep1(x, idx, tw, w1, w2, dy, accumulate):
    xx = x.clone().requires_grad_(True)
    tt = tw.clone().requires_grad_(True)
    a1, a2 = _param(w1, accumulate), _param(w2, accumulate)
    c1, c2 = FM.FP8BlockWeightCache(a1), FM.FP8BlockWeightCache(a2)
    y = FM.fused_mega_moe_fp8blk(xx, idx, tt, a1, a2, c1, c2, accumulate_wgrad=accumulate)
    y.backward(dy)
    torch.cuda.synchronize()
    dw1 = a1.main_grad if accumulate else a1.grad
    dw2 = a2.main_grad if accumulate else a2.grad
    return dict(y=y.detach(), dx=xx.grad, dtw=tt.grad, dw1=dw1, dw2=dw2)


def ep_setup(x, idx, tw, w1, w2, R, accumulate):
    T, E = x.shape[0], w1.shape[0]
    Tr, El = T // R, E // R
    assert T % R == 0 and E % R == 0
    xs = [x[r * Tr:(r + 1) * Tr].clone().requires_grad_(True) for r in range(R)]
    idxs = [idx[r * Tr:(r + 1) * Tr].contiguous() for r in range(R)]
    tws = [tw[r * Tr:(r + 1) * Tr].clone().requires_grad_(True) for r in range(R)]
    mg1 = torch.zeros_like(w1) if accumulate else None
    mg2 = torch.zeros_like(w2) if accumulate else None
    w1s, w2s = [], []
    for r in range(R):
        a1 = w1[r * El:(r + 1) * El].detach().clone().requires_grad_(True)
        a2 = w2[r * El:(r + 1) * El].detach().clone().requires_grad_(True)
        if accumulate:  # each rank's main_grad is its slice of one [E, ...] buffer: directly comparable to EP=1
            a1.main_grad = mg1[r * El:(r + 1) * El]
            a2.main_grad = mg2[r * El:(r + 1) * El]
        w1s.append(a1), w2s.append(a2)
    c1s = [FM.FP8BlockWeightCache(a) for a in w1s]
    c2s = [FM.FP8BlockWeightCache(a) for a in w2s]
    return xs, idxs, tws, w1s, w2s, c1s, c2s, mg1, mg2


def run_ep(x, idx, tw, w1, w2, dy, R, accumulate, dispatch):
    xs, idxs, tws, w1s, w2s, c1s, c2s, mg1, mg2 = ep_setup(x, idx, tw, w1, w2, R, accumulate)
    comm = EP.EmulatedEPComm(R)
    ys = EP.fused_mega_moe_fp8blk_ep_multi(xs, idxs, tws, w1s, w2s, c1s, c2s, comm, accumulate_wgrad=accumulate,
                                           dispatch=dispatch)
    torch.autograd.backward(ys, list(dy.chunk(R)))
    torch.cuda.synchronize()
    cat = lambda ts: torch.cat([t.detach() for t in ts])
    return dict(y=cat(ys), dx=cat([a.grad for a in xs]), dtw=cat([a.grad for a in tws]),
                dw1=mg1 if accumulate else cat([a.grad for a in w1s]),
                dw2=mg2 if accumulate else cat([a.grad for a in w2s]))


def ref64(x, idx, tw, w1, w2, dy):
    """unquantized math in fp64 (clamped SiLU-SwiGLU, +-10 on both halves, the kernels' clamp-gradient convention)."""
    T, H = x.shape
    E, I2, _ = w1.shape
    I = I2 // 2
    with torch.enable_grad():
        x64 = x.double().requires_grad_(True)
        t64 = tw.double().requires_grad_(True)
        a1 = w1.double().requires_grad_(True)
        a2 = w2.double().requires_grad_(True)
        y = torch.zeros(T, H, device=dev, dtype=torch.float64)
        for e in range(E):
            t, k = (idx == e).nonzero(as_tuple=True)
            if t.numel() == 0:
                continue
            h = x64[t] @ a1[e].t()
            g = h[:, :I].clamp(-10.0, 10.0)
            u = h[:, I:].clamp(-10.0, 10.0)
            a = g * torch.sigmoid(g) * u * t64[t, k][:, None]
            y = y.index_add(0, t, a @ a2[e].t())
        y.backward(dy.double())
    return dict(y=y.detach(), dx=x64.grad, dtw=t64.grad, dw1=a1.grad, dw2=a2.grad)


# ------------------------------------------------------------------------------------------------- dry: kernels
def src_layout_ref(idx, tw, R, E_l, cap, ccap):
    """torch reference of src_layout_ep (the first, torch-only version of the op's source layout)."""
    T, K = idx.shape
    ar = torch.arange(R, device=dev)
    dst = torch.div(idx, E_l, rounding_mode="floor")
    sends = torch.zeros((R, T), dtype=torch.int32, device=dev).scatter_(0, dst.t(), 1)
    slot = torch.cumsum(sends, 1, dtype=torch.int32) - sends
    slot = torch.where(sends > 0, slot, -1).t().to(torch.int32).contiguous()
    cnt = sends.sum(1, dtype=torch.int32)
    oh = (ar[:, None] == dst.reshape(1, -1)).to(torch.int32)
    rpos = (torch.cumsum(oh, 1, dtype=torch.int32) - oh).gather(0, dst.reshape(1, -1)).view(-1)
    cinv = (dst.reshape(-1) * ccap + rpos).to(torch.int32)
    to_d = dst[:, None, :] == ar[None, :, None]
    mi = torch.where(to_d, (idx - dst * E_l)[:, None, :], -1).to(torch.int32)
    mw = torch.where(to_d, tw.float()[:, None, :], 0.0)
    mr = torch.where(to_d, rpos.view(T, 1, K), -1).to(torch.int32)
    meta = torch.cat((mi, mw.view(torch.int32), mr), dim=2).view(T * R, 3 * K)
    flat = torch.where(slot >= 0, ar[None, :] * cap + slot, R * cap).view(-1)
    buf = torch.full((R * cap + 1, 3 * K), -1, dtype=torch.int32, device=dev)
    buf.index_copy_(0, flat, meta)
    return slot, cinv, buf[: R * cap].view(R, cap, 3 * K), cnt


def dry():
    ok = True
    torch.manual_seed(0)
    T, H, R, cap = 300, 1024, 4, 300
    x = (torch.randn(T, H, device=dev) * torch.logspace(-3, 3, T, device=dev)[:, None]).to(torch.bfloat16)
    # source quant into send slots == the row half of quant_dual_blk (pool = identity rows)
    sends = (torch.rand(T, R, device=dev) < 0.5).to(torch.int32)
    slot = torch.where(sends > 0, torch.cumsum(sends, 0, dtype=torch.int32) - sends, -1).to(torch.int32).contiguous()
    q, s = QE.quant_rows_send(x, slot, cap)
    Pp = -(-T // 128) * 128
    rq, rs, _, _ = quant_dual_blk(x, Pp, None)
    for d in range(R):
        tt = (slot[:, d] >= 0).nonzero().view(-1)
        sl = slot[tt, d].long()
        a, b = q[d, sl], rq[tt]
        sb = rs[tt].view(-1, H // 32)[:, ::4].contiguous()
        e1 = bits_equal(a, b) and bits_equal(s[d, sl], sb)
        ok &= e1
    print(f"quant_rows_send == quant_dual_blk rows: {ok}", flush=True)
    qb, _ = QE.quant_rows_send(x, slot, cap, quant=False)
    d0 = (slot[:, 0] >= 0).nonzero().view(-1)
    e2 = bits_equal(qb[0, slot[d0, 0].long()], x[d0])
    print(f"quant_rows_send bf16 copy: {e2}", flush=True)
    ok &= e2
    # masked routing == build_routing_blk when every route is local
    E, K, Tn = 8, 3, 700
    idx = torch.stack([torch.randperm(E, device=dev)[:K] for _ in range(Tn)])
    tw = torch.rand(Tn, K, device=dev)
    P = Tn * K + E * 256
    a = build_routing_blk(idx, tw, E, P)
    b = QE.build_routing_blk_masked(idx.to(torch.int32), tw, E, P)
    e3 = all(bits_equal(u.contiguous(), v.contiguous()) for u, v in zip(a[:4], b[:4]))
    print(f"build_routing_blk_masked (all local) == build_routing_blk: {e3}", flush=True)
    ok &= e3
    # with dropped routes: same as routing the kept routes only (compacted rows keep order)
    keep = torch.rand(Tn, K, device=dev) < 0.6
    li = torch.where(keep, idx, -1).to(torch.int32)
    rowmap, rw, inv, poffs, preal, _ = QE.build_routing_blk_masked(li, tw, E, P)
    torch.cuda.synchronize()
    ok4 = bool((inv[~keep.view(-1)] == -1).all())
    for e in range(E):
        rr = ((li.view(-1) == e).nonzero().view(-1))
        seg = rowmap[poffs[e]:poffs[e] + rr.numel()].long()
        ok4 &= bool(torch.equal(seg, rr // K)) and bool(torch.equal(inv[rr].long(), poffs[e] + torch.arange(rr.numel(), device=dev)))
        ok4 &= bool((rowmap[poffs[e] + rr.numel():poffs[e + 1]] == -1).all())
    print(f"build_routing_blk_masked with dropped routes: {ok4}", flush=True)
    ok &= ok4
    # gather_dual_fp8: row copy exact, 128x1 = quant_dual_blk of the dequantized rows
    N = 600
    xr = (torch.randn(N, H, device=dev) * torch.logspace(-2, 2, N, device=dev)[:, None]).to(torch.bfloat16)
    sl_all = torch.zeros(N, 1, dtype=torch.int32, device=dev)
    sl_all[:, 0] = torch.arange(N, device=dev, dtype=torch.int32)
    xq, xs = QE.quant_rows_send(xr, sl_all, N)
    xq, xs = xq[0], xs[0]
    rmap = torch.full((1024,), -1, dtype=torch.int32, device=dev)
    perm = torch.randperm(N, device=dev)[:700 if N >= 700 else N].to(torch.int32)
    rmap[:perm.numel()] = perm
    rmap = rmap[torch.randperm(1024, device=dev)].contiguous()
    for ct in (False, True):
        gq, gs, gT, gTs = QE.gather_dual_fp8(xq, xs, 1024, rmap, compact_t=ct)
        deq = dequant(xq, xs.repeat_interleave(4, 1)).to(torch.float32)
        # reference: quant_dual_blk on the dequantized rows (fp32 input path) for the 128x1 half
        ref = quant_dual_blk(deq.contiguous(), 1024, rmap, compact_t=ct)
        v = rmap >= 0
        e5 = bits_equal(gq[v], xq[rmap[v].long()]) and bits_equal(gq[~v], torch.zeros_like(gq[~v]))
        e6 = bits_equal(gT, ref[2]) and bits_equal(gTs, ref[3])
        e7 = bits_equal(gs[v], xs[rmap[v].long()].repeat_interleave(4, 1))
        print(f"gather_dual_fp8 ct={ct}: rows {e5} row-scales {e7} 128x1 == quant_dual_blk(dequant) {e6}", flush=True)
        ok &= e5 and e6 and e7
    # src_layout_ep == the torch reference, routing over E = R * El global experts
    for (Rr, El, Tn, K2) in ((8, 4, 700, 6), (4, 8, 300, 6), (8, 32, 1000, 6), (4, 2, 257, 2)):
        Eg = Rr * El
        gi = torch.randn(Tn, Eg, device=dev).topk(K2, dim=1)[1]
        gw = torch.rand(Tn, K2, device=dev)
        cap, ccap = Tn, Tn * min(K2, El)
        slot, cinv, meta, cnt = QE.src_layout_ep(gi, gw, Rr, El, cap, ccap)
        rs, rci, rm, rc = src_layout_ref(gi, gw, Rr, El, cap, ccap)
        e9 = bits_equal(slot, rs) and bits_equal(cinv, rci) and bits_equal(cnt, rc)
        live = torch.arange(cap, device=dev)[None, :] < cnt[:, None]
        e9 &= bits_equal(meta[live], rm[live])
        print(f"src_layout_ep R={Rr} E_l={El} T={Tn} K={K2} == torch reference: {e9}", flush=True)
        ok &= e9
    # pack_pool_rows
    Pn, nsl = 1024, 3000
    src = torch.randn(Pn, H, device=dev).to(torch.bfloat16)
    ext = torch.randn(Pn, device=dev)
    p2s = torch.full((Pn,), -1, dtype=torch.int32, device=dev)
    pr = 700
    sl = torch.randperm(nsl, device=dev)[:pr].to(torch.int32)
    p2s[:pr] = sl
    p2s[torch.randperm(pr, device=dev)[:100]] = -1  # padding rows inside the live pool
    pre = torch.tensor([pr], dtype=torch.int32, device=dev)
    out, eo = QE.pack_pool_rows(src, p2s, pre, nsl, ext)
    v = p2s[:pr] >= 0
    e10 = bits_equal(out[p2s[:pr][v].long()], src[:pr][v]) and bits_equal(eo[p2s[:pr][v].long()], ext[:pr][v])
    print(f"pack_pool_rows: {e10}", flush=True)
    ok &= e10
    # gather_rows
    src = torch.randn(500, H, device=dev).to(torch.bfloat16)
    m = torch.randint(-1, 500, (900,), device=dev, dtype=torch.int32)
    out = torch.zeros(900, H, device=dev, dtype=torch.bfloat16)
    QE.gather_rows(src, m, out)
    e8 = bits_equal(out[m >= 0], src[m[m >= 0].long()]) and bool((out[m < 0] == 0).all())
    print(f"gather_rows: {e8}", flush=True)
    ok &= e8
    # whole op, tiny: R=4 vs EP=1
    ok &= full_case(512, 1024, 256, 16, 4, [4], ["fp8", "bf16"], "uniform", True, False)
    print("RESULT", "PASS" if ok else "FAIL", flush=True)
    return ok


# ------------------------------------------------------------------------------------------------- full
NAMES = ("y", "dx", "dtw", "dw1", "dw2")


def full_case(T, H, I, E, K, Rs, dispatches, routing, accumulate, do_ref64, seed=0, noref=False):
    t0 = time.time()
    free, tot = torch.cuda.mem_get_info()
    print(f"[mem] free {free/2**30:.0f} / {tot/2**30:.0f} GiB", flush=True)
    x, idx, tw, w1, w2, dy = make_inputs(T, H, I, E, K, routing, seed)
    base = run_ep1(x, idx, tw, w1, w2, dy, accumulate)
    head = f"T{T} H{H} I{I} E{E} K{K} {routing} acc={int(accumulate)}"
    ok = True
    r64 = None
    for R in Rs:
        for disp in dispatches:
            got = run_ep(x, idx, tw, w1, w2, dy, R, accumulate, disp)
            parts = []
            for n in NAMES:
                eq = bits_equal(got[n], base[n])
                fin = bool(torch.isfinite(got[n].float()).all())
                parts.append(f"{n} {'BITEQ' if eq else 'diff %d' % nbits_diff(got[n], base[n])}")
                if not fin:
                    parts[-1] += " NONFINITE"
                    ok = False
                must = disp == "bf16" or n in ("y", "dx", "dtw")
                if must and not eq:
                    ok = False
            print(f"[{head}] R={R} dispatch={disp}: " + ", ".join(parts), flush=True)
            if noref:
                for n in NAMES:
                    if not bits_equal(got[n], base[n]):
                        ee, me = rel(got[n], base[n])
                        print(f"    {n}: |EP - EP1|/|EP1| {ee:.3e} (max abs {me:.3e}); fp64 reference skipped", flush=True)
            elif (do_ref64 or any(not bits_equal(got[n], base[n]) for n in NAMES)) and r64 is None:
                r64 = ref64(x, idx, tw, w1, w2, dy)
            if r64 is not None:
                for n in NAMES:
                    if bits_equal(got[n], base[n]) and not do_ref64:
                        continue
                    e1, m1 = rel(base[n], r64[n])
                    e2, m2 = rel(got[n], r64[n])
                    ee, me = rel(got[n], base[n])
                    print(f"    {n}: rel-err vs fp64  EP=1 {e1:.4e} (max {m1:.3e})  EP {e2:.4e} (max {m2:.3e})  "
                          f"|EP - EP1|/|EP1| {ee:.3e}", flush=True)
            del got
            torch.cuda.empty_cache()
    print(f"[{head}] {'PASS' if ok else 'FAIL'} ({time.time() - t0:.1f} s)", flush=True)
    return ok


# ------------------------------------------------------------------------------------------------- graph
def graph_case(T=1024, H=1024, I=512, E=32, K=6, R=4, dispatch="fp8"):
    x, idx, tw, w1, w2, dy = make_inputs(T, H, I, E, K, "skew", 1)
    xs, idxs, tws, w1s, w2s, c1s, c2s, mg1, mg2 = ep_setup(x, idx, tw, w1, w2, R, True)
    comm = EP.EmulatedEPComm(R)
    dys = [d.contiguous() for d in dy.chunk(R)]

    def step():
        ys = EP.fused_mega_moe_fp8blk_ep_multi(xs, idxs, tws, w1s, w2s, c1s, c2s, comm, accumulate_wgrad=True,
                                               dispatch=dispatch)
        gr = torch.autograd.grad(ys, xs + tws, dys)
        return [y.detach() for y in ys], gr

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):  # warm-up: compiles, weight caches
            mg1.zero_(), mg2.zero_()
            ys_e, gr_e = step()
        ys_e = [y.clone() for y in ys_e]
        gr_e = [g.clone() for g in gr_e]
        m1e, m2e = mg1.clone(), mg2.clone()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    mg1.zero_(), mg2.zero_()
    with torch.cuda.graph(g):
        ys_g, gr_g = step()
    ok = True
    for it in range(2):
        mg1.zero_(), mg2.zero_()
        g.replay()
        torch.cuda.synchronize()
        e = all(bits_equal(a, b) for a, b in zip(ys_g, ys_e)) and all(bits_equal(a, b) for a, b in zip(gr_g, gr_e))
        e = e and bits_equal(mg1, m1e) and bits_equal(mg2, m2e)
        print(f"graph replay {it}: y/dx/dtw/dW1/dW2 bit-equal to eager: {e}", flush=True)
        ok &= e
    print("RESULT", "PASS" if ok else "FAIL", flush=True)
    return ok


# ------------------------------------------------------------------------------------------------- time
_CATS = [
    ("src quant+pack (dispatch send)", ("_quant_rows_send",)),
    ("emulated a2a (device copies)", ("copy", "Copy", "Memcpy", "elementwise_kernel_manual_unroll")),
    ("routing / layout tables", ("_route", "cumsum", "scan", "index", "scatter", "gather_kernel", "where", "Fill",
                                 "fill", "arange", "reduce", "compare", "unrolled_elementwise", "argsort", "sort",
                                 "radix", "Radix")),
    ("pool operands 1x128+128x1", ("_gather_dual", "_dual_1x128")),
    ("grouped GEMM fc1/fc2/dgrads", ("gg", "flygg", "kernel_gg", "ggk", "nt_")),
    ("SwiGLU fwd/bwd + quant", ("_swiglu",)),
    ("wgrad (vk/vkp)", ("vk",)),
    ("combine pack (pool->slots)", ("_gather_rows",)),
    ("combine sum (combine_rows)", ("_combine",)),
]


def _cat(name):
    for c, keys in _CATS:
        if any(k in name for k in keys):
            return c
    return "other"


def time_case(T, H, I, E, K, Rs, dispatch, iters=5, ep1_per_rank=False):
    """ep1_per_rank: EP=1 runs one rank's load (T/R tokens, E/R experts) instead of the whole group's (for E where
    the EP=1 routing kernels do not compile, e.g. E = 256)."""
    from torch.profiler import ProfilerActivity, profile

    x, idx, tw, w1, w2, dy = make_inputs(T, H, I, E, K, "uniform", 0)
    runs = [("EP=1", None)] + [(f"EP R={R}", R) for R in Rs]
    for label, R in runs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m0 = torch.cuda.memory_allocated()
        if R is None:
            if ep1_per_rank:
                Rm = max(Rs)
                x1, idx1, tw1, w11, w21, dy1 = make_inputs(T // Rm, H, I, E // Rm, K, "uniform", 0)
                label = f"EP=1 one rank's load (T={T // Rm}, E={E // Rm})"
            else:
                x1, idx1, tw1, w11, w21, dy1 = x, idx, tw, w1, w2, dy
            xx = x1.clone().requires_grad_(True)
            tt = tw1.clone().requires_grad_(True)
            a1, a2 = _param(w11, True), _param(w21, True)
            c1, c2 = FM.FP8BlockWeightCache(a1), FM.FP8BlockWeightCache(a2)
            fwd = lambda: FM.fused_mega_moe_fp8blk(xx, idx1, tt, a1, a2, c1, c2, accumulate_wgrad=True)
            bwd = lambda y: y.backward(dy1)
            leaves = [xx, tt, a1, a2]
        else:
            xs, idxs, tws, w1s, w2s, c1s, c2s, mg1, mg2 = ep_setup(x, idx, tw, w1, w2, R, True)
            comm = EP.EmulatedEPComm(R)
            dys = list(dy.chunk(R))
            fwd = lambda: EP.fused_mega_moe_fp8blk_ep_multi(xs, idxs, tws, w1s, w2s, c1s, c2s, comm,
                                                            accumulate_wgrad=True, dispatch=dispatch)
            bwd = lambda ys: torch.autograd.backward(ys, dys)
            leaves = xs + tws + w1s + w2s
        def clear():  # fresh .grad every iteration (no dummy-wgrad accumulation adds, on both sides)
            for t in leaves:
                t.grad = None

        for _ in range(2):
            clear()
            bwd(fwd())
        torch.cuda.synchronize()
        tf, tb = [], []
        for _ in range(iters):
            clear()
            e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
            e0.record()
            y = fwd()
            e1.record()
            bwd(y)
            e2.record()
            torch.cuda.synchronize()
            tf.append(e0.elapsed_time(e1))
            tb.append(e1.elapsed_time(e2))
        tf.sort(), tb.sort()
        clear()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            bwd(fwd())
            torch.cuda.synchronize()
        cats = {}
        for ev in prof.key_averages():
            dt = getattr(ev, "device_time_total", None)
            if dt is None:
                dt = ev.cuda_time_total
            if dt <= 0 or ev.key.startswith("ProfilerStep"):
                continue
            c = _cat(ev.key)
            cats.setdefault(c, [0.0, set()])
            cats[c][0] += dt
            cats[c][1].add(ev.key[:40])
        tot = sum(v[0] for v in cats.values())
        print(f"[time {label} dispatch={dispatch if R else '-'} T{T} H{H} I{I} E{E} K{K}] wall fwd {tf[len(tf)//2]:.3f} ms "
              f"bwd {tb[len(tb)//2]:.3f} ms; kernel sum {tot/1e3:.3f} ms; peak mem above inputs "
              f"{(torch.cuda.max_memory_allocated() - m0)/2**30:.1f} GiB", flush=True)
        for c, (v, names) in sorted(cats.items(), key=lambda kv: -kv[1][0]):
            print(f"    {c:34s} {v/1e3:8.3f} ms  {sorted(names)[:4]}", flush=True)
        if os.environ.get("EP_TIME_TOPK"):
            evs = sorted(prof.key_averages(), key=lambda e: -getattr(e, "device_time_total", 0))
            for ev in evs[: int(os.environ["EP_TIME_TOPK"])]:
                print(f"      {getattr(ev, 'device_time_total', 0)/1e3:8.3f} ms x{ev.count:4d} {ev.key[:110]}", flush=True)
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dry", "full", "graph", "time"])
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--R", default="1,4,8")
    ap.add_argument("--dispatch", default="fp8,bf16")
    ap.add_argument("--routing", default="uniform,skew")
    ap.add_argument("--acc", default="1")
    ap.add_argument("--ref64", action="store_true")
    ap.add_argument("--noref", action="store_true", help="no fp64 reference (large shapes)")
    ap.add_argument("--ep1-per-rank", action="store_true", help="time: EP=1 at one rank's load")
    ap.add_argument("--shape", action="append", default=[], help="T,H,I,E,K (repeatable)")
    a = ap.parse_args()
    print(f"env WGRAD={os.environ.get('PRIMUS_TURBO_MOE_FP8BLK_WGRAD', 'vk')} GG={os.environ.get('PRIMUS_TURBO_MOE_FP8BLK_GG', 'tile')} "
          f"SCHED2={os.environ.get('PRIMUS_TURBO_MOE_FP8BLK_VKP_SCHED2', '0')}", flush=True)
    if a.mode == "dry":
        sys.exit(0 if dry() else 1)
    if a.mode == "graph":
        sys.exit(0 if graph_case(dispatch=a.dispatch.split(",")[0]) else 1)
    Rs = [int(r) for r in a.R.split(",")]
    shapes = []
    if a.small:
        shapes += [(1024, 1024, 512, 32, 6), (512, 1024, 256, 32, 4), (2048, 2048, 256, 32, 6)]
    if a.real:
        shapes += [(4096, 4096, 2048, 32, 6)]
    shapes += [tuple(int(v) for v in sh.split(",")) for sh in a.shape]
    if a.mode == "time":
        for s in shapes:
            time_case(*s, [r for r in Rs if r > 1], a.dispatch.split(",")[0], ep1_per_rank=a.ep1_per_rank)
        return
    ok = True
    for s in shapes:
        for routing in a.routing.split(","):
            for acc in a.acc.split(","):
                ok &= full_case(*s, Rs, a.dispatch.split(","), routing, acc == "1", a.ref64, noref=a.noref)
    print("RESULT", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
