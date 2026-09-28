"""Kernel-only timing at prod, palindromic, one process. usage: ktime.py <arms csv> <iters> <json>

arms:  C    champion k_dkdv                      (job_context/op/current, flydsl 0.3.2)
       P1   k_dkdv_f5, F5_ATOM=False probe        (lab3/f5p1; dQ WMMAs folded into dV acc 0)
       P3k  k_dkdv_f5, real atomics, kernel only  (lab3/f5; dq_acc.zero_() outside the window)
       P3t  zero_ + k_dkdv_f5 + cvt_dq            (lab3/f5; everything FUSED5 adds besides k_delta)
       DQ   champion k_dq                         (what FUSED5 deletes)
Harness method copied from benchmark.py: 3 s continuous warmup per arm, L2 flush (256 MB
zero_) outside every event window, median, palindromic order.
"""
import json, sys, time
from pathlib import Path
OP = Path("/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/"
          "gfx1250-flydsl-attn-bwd-20260917-115934/job_context/op")
LAB = Path("/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab3")
sys.path.insert(0, str(OP / "ut")); sys.path.insert(0, str(OP))
import torch
from common import load_impl, make_inputs
from refcache_util import cached_forward

arms, iters, out = sys.argv[1].split(","), int(sys.argv[2]), sys.argv[3]
mods = {}
for tag, p in (("champ", OP / "current"), ("f5p1", LAB / "f5p1"), ("f5", LAB / "f5")):
    fn = load_impl(p)
    mods[tag] = fn.__globals__
    mods[tag]["_check_env_once"]()
C, P, F = mods["champ"], mods["f5p1"], mods["f5"]
print("flydsl", C["_k"].flyc.__name__ if hasattr(C["_k"], "flyc") else "", flush=True)

q, k, v, do = make_inputs("prod", seed=0)
o, lse = cached_forward("prod", q, k, v, causal=True)
lse = lse.contiguous().float()
b, sq, hq, d = q.shape; skv, hkv = k.shape[1], k.shape[2]; g = hq // hkv
scale = float(d ** -0.5); st = torch.cuda.current_stream()
Kc = C["_k"]
n_rows = b * sq * hq
delta = torch.empty((b, hq, sq), device=q.device, dtype=torch.float32)
C["_launch"]("delta", Kc.launch_delta, (do, o, delta, sq, hq, n_rows, n_rows // Kc.ROWS_DELTA, st))
torch.cuda.synchronize()

bufs = {a: (torch.empty_like(k), torch.empty_like(k)) for a in ("C", "P1", "P3k", "P3t")}
dqa = {a: torch.zeros((b, sq, hq, d), device=q.device, dtype=torch.float32) for a in ("P1", "P3k", "P3t")}
dq_o = torch.empty_like(q); dq_c = torch.empty_like(q)
tail = (sq, skv, hq, hkv, g, sq // 16, skv - sq, 1, skv // Kc.BLOCK_KV, hkv, b, st)
n_vec = (b * sq * hq * d) // Kc.RED_VEC; rblk = (n_vec + Kc.RED_THREADS - 1) // Kc.RED_THREADS

def run_C():
    dv_, dk_ = bufs["C"]
    C["_launch"]("dkdv", Kc.launch_dkdv, (q, k, v, do, lse, delta, dv_, dk_, scale) + tail)
def run_P1():
    dv_, dk_ = bufs["P1"]
    P["_launch"]("dkdv_f5", P["_k"].launch_dkdv_f5, (q, k, v, do, lse, delta, dv_, dk_, dqa["P1"], scale) + tail)
def run_P3k():
    dv_, dk_ = bufs["P3k"]
    F["_launch"]("dkdv_f5", F["_k"].launch_dkdv_f5, (q, k, v, do, lse, delta, dv_, dk_, dqa["P3k"], scale) + tail)
def run_P3t():
    dv_, dk_ = bufs["P3t"]
    dqa["P3t"].zero_()
    F["_launch"]("dkdv_f5", F["_k"].launch_dkdv_f5, (q, k, v, do, lse, delta, dv_, dk_, dqa["P3t"], scale) + tail)
    F["_launch"]("cvt_dq", F["_k"].launch_redsp_q, (dqa["P3t"], dq_o, n_vec, 1, rblk, st))
def run_DQ():
    C["_launch"]("dq", Kc.launch_dq, (q, k, v, do, lse, delta, dq_c, scale, sq, skv, hq, hkv, g,
                  skv // Kc.KV_STEP, skv - sq, 1, sq // Kc.BLOCK_Q, hq, b, st))
fns = {"C": run_C, "P1": run_P1, "P3k": run_P3k, "P3t": run_P3t, "DQ": run_DQ}
pre = {"P3k": lambda: dqa["P3k"].zero_()}

flush = torch.empty(256 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
for a in arms:
    fns[a]()
torch.cuda.synchronize()
for a in arms:
    t_end = time.perf_counter() + 3.0
    while time.perf_counter() < t_end:
        if a in pre: pre[a]()
        fns[a]()
    torch.cuda.synchronize()

times = {a: [] for a in arms}
e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
for i in range(iters):
    for a in (arms if i % 2 == 0 else arms[::-1]):
        if a in pre: pre[a]()
        flush.zero_()
        e0.record(); fns[a](); e1.record(); e1.synchronize()
        times[a].append(e0.elapsed_time(e1))

med = lambda xs: sorted(xs)[len(xs) // 2]
rows = {a: {"median_ms": med(times[a]), "min_ms": min(times[a]), "max_ms": max(times[a]), "n": iters}
        for a in arms}
# witnesses: the subject ran and wrote what it should
if "C" in arms:
    for a in ("P1", "P3k", "P3t"):
        if a in arms:
            rows[a]["dk_bitwise_vs_C"] = bool(torch.equal(bufs["C"][1], bufs[a][1]))
            rows[a]["dv_bitwise_vs_C"] = bool(torch.equal(bufs["C"][0], bufs[a][0]))
if "P3t" in arms and "DQ" in arms:
    r = dq_c.float(); x = dq_o.float()
    rows["P3t"]["dq_sqnr_vs_kdq_db"] = float(10 * torch.log10(r.pow(2).mean() / (r - x).pow(2).mean().clamp_min(1e-30)))
ref = rows["C"]["median_ms"] if "C" in arms else None
for a in arms:
    if ref: rows[a]["ratio_vs_C"] = rows[a]["median_ms"] / ref
    print("KTIME " + a + " " + " ".join(f"{kk}={vv:.4f}" if isinstance(vv, float) else f"{kk}={vv}"
                                        for kk, vv in rows[a].items()), flush=True)
Path(out).write_text(json.dumps({"arms": arms, "rows": rows, "times": times}, indent=1))
