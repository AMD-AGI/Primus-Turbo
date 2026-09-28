"""Op-level check of the nkfix operand-layout rewrite on the e2e step's GEMM shapes (B0, GPU 0).

For every aten::mm class seen in the Llama-3.1-8B step (MBS 4 x seq 8192 -> M = 32768), in ONE
process, each call run a handful of times (no sustained loop -- LAB-RULES 6):

  orig : torch.mm(a, b) exactly as autograd issues it (layouts reproduced with .t() views)
  nk   : the nkfix rewrite (dgrad: B -> N-major copy; wgrad: A contiguous + B N-major)
  swap : zero-copy (b.t() @ a.t()).t() -- tested because it would avoid the copies

Reports per class and variant: ms (median of 3 after 1 warm-up, CUDA events), TF/s, the Tensile
kernel name (torch.profiler), whether nk is bitwise equal to orig / max bf16 ulp distance, SQNR of
each vs an fp32 reference, determinism of nk over 3 repeats, and NaN-prefill coverage (mm out=).
"""
import json, os, sys, time
import torch

torch.manual_seed(0)
dev = "cuda"
BF = torch.bfloat16
M = 32768
# (name, kind, A shape as passed to mm, B shape as passed to mm)
CLASSES = [
    ("fwd_qo",      "fwd",   (M, 4096),   (4096, 4096)),
    ("fwd_mlp13",   "fwd",   (M, 4096),   (4096, 14336)),
    ("dgrad_qo",    "dgrad", (M, 4096),   (4096, 4096)),
    ("dgrad_kv",    "dgrad", (M, 1024),   (1024, 4096)),
    ("dgrad_mlp13", "dgrad", (M, 14336),  (14336, 4096)),
    ("dgrad_mlp2",  "dgrad", (M, 4096),   (4096, 14336)),
    ("dgrad_head",  "dgrad", (M, 128256), (128256, 4096)),
    ("wgrad_qo",    "wgrad", (4096, M),   (M, 4096)),
    ("wgrad_kv",    "wgrad", (1024, M),   (M, 4096)),
    ("wgrad_mlp13", "wgrad", (14336, M),  (M, 4096)),
    ("wgrad_mlp2",  "wgrad", (4096, M),   (M, 14336)),
    ("wgrad_head",  "wgrad", (128256, M), (M, 4096)),
]
only = os.environ.get("PROBE_ONLY", "")
if only:
    CLASSES = [c for c in CLASSES if c[0] in only.split(",")]


def make(kind, ash, bsh):
    # fwd: A contiguous, B = W.t() view.  dgrad: A, B contiguous.  wgrad: A = g.t() view, B contig.
    if kind == "fwd":
        a = torch.randn(ash, device=dev, dtype=BF)
        b = torch.randn(bsh[::-1], device=dev, dtype=BF).t()
    elif kind == "dgrad":
        a = torch.randn(ash, device=dev, dtype=BF)
        b = torch.randn(bsh, device=dev, dtype=BF) * 0.02
    else:
        a = torch.randn(ash[::-1], device=dev, dtype=BF).t()
        b = torch.randn(bsh, device=dev, dtype=BF)
    return a, b


def v_orig(a, b):
    return torch.mm(a, b)


def v_nk(a, b):
    if not a.is_contiguous() and b.is_contiguous():
        return torch.mm(a.contiguous(), b.t().contiguous().t())
    if b.is_contiguous():
        return torch.mm(a, b.t().contiguous().t())
    return torch.mm(a, b)


def v_swap(a, b):
    return torch.mm(b.t(), a.t()).t()


VARIANTS = {"orig": v_orig, "nk": v_nk, "swap": v_swap}


def timeit(fn, a, b, n=3):
    fn(a, b); torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); fn(a, b); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    ts.sort()
    return ts[len(ts) // 2]


def kernels(fn, a, b):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn(a, b); torch.cuda.synchronize()
    names = []
    for e in p.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            names.append((e.name, round(e.device_time_total / 1000.0, 3)))
    return names


def ulp_int(x):
    i = x.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(i < 0, -32768 - i, i)       # monotone map of bf16 bits


def sqnr(ref, x):
    x = x.float()
    n = (ref - x).pow(2).sum().item()
    s = ref.pow(2).sum().item()
    return 10 * __import__("math").log10(s / max(n, 1e-30))


res = []
for name, kind, ash, bsh in CLASSES:
    a, b = make(kind, ash, bsh)
    Mm, K = a.shape; N = b.shape[1]
    flops = 2.0 * Mm * N * K
    row = dict(name=name, kind=kind, M=Mm, N=N, K=K,
               A="c" if a.is_contiguous() else "v", B="c" if b.is_contiguous() else "v")
    outs = {}
    for vn, fn in VARIANTS.items():
        if kind == "fwd" and vn != "orig":
            continue
        ms = timeit(fn, a, b)
        row[vn + "_ms"] = round(ms, 3)
        row[vn + "_tfs"] = round(flops / ms / 1e9, 1)
        row[vn + "_kern"] = kernels(fn, a, b)
        outs[vn] = fn(a, b)
    torch.cuda.synchronize()
    if kind != "fwd":
        o, k = outs["orig"], outs["nk"]
        row["nk_bitwise_eq_orig"] = bool(torch.equal(o, k))
        row["nk_max_ulp_vs_orig"] = int((ulp_int(o) - ulp_int(k)).abs().max().item())
        row["nk_frac_diff_vs_orig"] = float((o != k).float().mean().item())
        row["swap_bitwise_eq_orig"] = bool(torch.equal(o, outs["swap"]))
        row["swap_bitwise_eq_nk"] = bool(torch.equal(k, outs["swap"]))
        rep = [v_nk(a, b) for _ in range(3)]
        row["nk_deterministic_x3"] = all(torch.equal(k, r) for r in rep)
        del rep
        ref = a.float() @ b.float()
        for vn in ("orig", "nk", "swap"):
            row[vn + "_sqnr_db"] = round(sqnr(ref, outs[vn]), 2)
        del ref
        # NaN-prefill coverage of the rewritten call (mm.out writes every element?)
        if not a.is_contiguous() and b.is_contiguous():
            a2, b2 = a.contiguous(), b.t().contiguous().t()
        else:
            a2, b2 = a, b.t().contiguous().t()
        buf = torch.full((Mm, N), float("nan"), device=dev, dtype=BF)
        torch.mm(a2, b2, out=buf)
        row["nk_out_prefill_nonfinite"] = int((~torch.isfinite(buf)).sum().item())
        row["nk_out_prefill_eq"] = bool(torch.equal(buf, k))
        del buf, a2, b2
        row["nk_speedup"] = round(row["orig_ms"] / row["nk_ms"], 2)
    for x in list(outs):
        del outs[x]
    del a, b
    torch.cuda.empty_cache()
    res.append(row)
    kn = {v: (row.get(v + "_kern") or [("-", 0)]) for v in VARIANTS}
    short = lambda ks: ";".join("%s(%.2f)" % (n[:90], t) for n, t in ks)
    print("%-12s %s%s M=%-6d N=%-6d K=%-6d orig %8.3f ms %7.1f TF/s | nk %s ms %s TF/s x%s | "
          "swap %s ms | eq=%s ulp=%s det=%s nanfill=%s sqnr o/n/s=%s/%s/%s"
          % (name, row["A"], row["B"], Mm, N, K, row["orig_ms"], row["orig_tfs"],
             row.get("nk_ms"), row.get("nk_tfs"), row.get("nk_speedup"), row.get("swap_ms"),
             row.get("nk_bitwise_eq_orig"), row.get("nk_max_ulp_vs_orig"),
             row.get("nk_deterministic_x3"), row.get("nk_out_prefill_nonfinite"),
             row.get("orig_sqnr_db"), row.get("nk_sqnr_db"), row.get("swap_sqnr_db")), flush=True)
    for v in VARIANTS:
        if row.get(v + "_kern"):
            print("      %-4s %s" % (v, short(row[v + "_kern"])), flush=True)
    time.sleep(0.5)

out = sys.argv[1] if len(sys.argv) > 1 else "opprobe.json"
with open(out, "w") as f:
    json.dump(res, f, indent=1)
print("wrote", out)
