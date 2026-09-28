"""Second op-level check: the chunked rewrite actually shipped in nkfix_b0.py.

  1. per class: nkfix_b0._dgrad/_wgrad (chunked where the copy exceeds NKFIX_CHUNK_BYTES) vs the
     unchunked rewrite (bitwise?) vs orig (ulp statistics: |nk-orig| in units of the bf16 ulp of
     max(|nk|,|orig|); fraction within 1 / 2 ulp; max relative to rms), timing, kernels.
  2. the dispatch mode end to end on a Linear under autograd (backward runs in the autograd
     engine thread): which rules fire, grads finite, grads vs the no-mode grads.
"""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nkfix_b0 as nk

dev = "cuda"; BF = torch.bfloat16; M = 32768
torch.manual_seed(1)
CLASSES = [
    ("dgrad_qo", "dgrad", (M, 4096), (4096, 4096)),
    ("dgrad_kv", "dgrad", (M, 1024), (1024, 4096)),
    ("dgrad_mlp13", "dgrad", (M, 14336), (14336, 4096)),
    ("dgrad_mlp2", "dgrad", (M, 4096), (4096, 14336)),
    ("dgrad_head", "dgrad", (M, 128256), (128256, 4096)),
    ("wgrad_qo", "wgrad", (4096, M), (M, 4096)),
    ("wgrad_kv", "wgrad", (1024, M), (M, 4096)),
    ("wgrad_mlp13", "wgrad", (14336, M), (M, 4096)),
    ("wgrad_mlp2", "wgrad", (4096, M), (M, 14336)),
    ("wgrad_head", "wgrad", (128256, M), (M, 4096)),
]


def make(kind, ash, bsh):
    if kind == "dgrad":
        return torch.randn(ash, device=dev, dtype=BF), torch.randn(bsh, device=dev, dtype=BF) * 0.02
    return torch.randn(ash[::-1], device=dev, dtype=BF).t(), torch.randn(bsh, device=dev, dtype=BF)


def t_ms(fn, n=3):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(n):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    return sorted(ts)[n // 2]


def ulp_stats(o, k):
    o = o.float(); k = k.float(); d = (o - k).abs()
    m = torch.maximum(o.abs(), k.abs()).clamp_min(2.0 ** -126)
    u = torch.exp2(torch.floor(torch.log2(m)) - 7)          # bf16: 8 significant bits
    r = d / u
    q = torch.quantile(r.flatten()[:: max(1, r.numel() // 4_000_000)],
                       torch.tensor([0.5, 0.99, 0.9999], device=dev))
    return dict(within1=float((r <= 1).float().mean()), within2=float((r <= 2).float().mean()),
                p50=float(q[0]), p99=float(q[1]), p9999=float(q[2]), max=float(r.max()),
                maxdiff_over_rms=float(d.max() / o.pow(2).mean().sqrt()))


res = []
for name, kind, ash, bsh in CLASSES:
    a, b = make(kind, ash, bsh)
    o = torch.mm(a, b)
    if kind == "dgrad":
        full = lambda: torch.mm(a, b.t().contiguous().t()); ch = lambda: nk._dgrad(a, b)
    else:
        full = lambda: torch.mm(a.contiguous(), b.t().contiguous().t()); ch = lambda: nk._wgrad(a, b)
    f, c = full(), ch()
    row = dict(name=name, chunk_bytes=nk._CHUNK, chunked_eq_full=bool(torch.equal(f, c)),
               chunked_nonfinite=int((~torch.isfinite(c)).sum()),
               full_ms=round(t_ms(full), 3), chunked_ms=round(t_ms(ch), 3))
    row.update({"ulp_" + k: v for k, v in ulp_stats(o, c).items()})
    del f, c, o, a, b; torch.cuda.empty_cache()
    print(json.dumps(row), flush=True); res.append(row); time.sleep(0.3)

# --- 2. the mode under autograd --------------------------------------------------------------
lin = torch.nn.Linear(4096, 14336, bias=False, device=dev, dtype=BF)
x = torch.randn(4, 8192, 4096, device=dev, dtype=BF, requires_grad=True)


def grads():
    lin.weight.grad = None; x.grad = None
    y = lin(x); (y.float().pow(2).mean()).backward()
    return x.grad.clone(), lin.weight.grad.clone()


gx0, gw0 = grads()
nk.install()
gx1, gw1 = grads()
torch.cuda.synchronize()
auto = dict(stats=dict(nk.stats), gx_eq=bool(torch.equal(gx0, gx1)), gw_eq=bool(torch.equal(gw0, gw1)),
            gx_finite=bool(torch.isfinite(gx1).all()), gw_finite=bool(torch.isfinite(gw1).all()),
            gx_ulp=ulp_stats(gx0, gx1), gw_ulp=ulp_stats(gw0, gw1))
print(json.dumps(auto), flush=True)
with open(sys.argv[1] if len(sys.argv) > 1 else "opprobe2.json", "w") as fh:
    json.dump(dict(classes=res, autograd=auto), fh, indent=1)
