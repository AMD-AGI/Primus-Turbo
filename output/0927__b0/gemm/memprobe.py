"""Transient memory of one rewritten call vs the untouched mm (max_memory_allocated delta), per class,
with and without the non-finite check. One process, each call run once (no loop)."""
import os, sys, json
os.environ.setdefault("NKFIX_CHECK", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import nkfix_b0 as nk

dev = "cuda"; BF = torch.bfloat16; M = 32768; MiB = 1 << 20
CL = [("dgrad_mlp13", "dgrad", (M, 14336), (14336, 4096)), ("dgrad_head", "dgrad", (M, 128256), (128256, 4096)),
      ("wgrad_mlp13", "wgrad", (14336, M), (M, 4096)), ("wgrad_mlp2", "wgrad", (4096, M), (M, 14336)),
      ("wgrad_head", "wgrad", (128256, M), (M, 4096))]


def peak(fn):
    torch.cuda.synchronize(); base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    out = fn(); torch.cuda.synchronize()
    p = torch.cuda.max_memory_allocated() - base
    o = out.numel() * out.element_size(); del out
    return round(p / MiB, 1), round(o / MiB, 1)


nk._rewrite(torch.randn(4096, 4096, device=dev, dtype=BF), torch.randn(4096, 4096, device=dev, dtype=BF), True)
res = []
for name, kind, ash, bsh in CL:
    if kind == "dgrad":
        a = torch.randn(ash, device=dev, dtype=BF); b = torch.randn(bsh, device=dev, dtype=BF)
    else:
        a = torch.randn(ash[::-1], device=dev, dtype=BF).t(); b = torch.randn(bsh, device=dev, dtype=BF)
    copy_a = kind == "wgrad"
    r = dict(name=name)
    r["orig_peak_MiB"], r["out_MiB"] = peak(lambda: torch.mm(a, b))
    r["nk_peak_MiB"], _ = peak(lambda: nk._rewrite(a, b, copy_a))
    def chk():
        o = nk._rewrite(a, b, copy_a); nk._check(a, b, o, kind); return o
    r["nk_check2_peak_MiB"], _ = peak(chk)
    print(json.dumps(r), flush=True); res.append(r)
    del a, b; torch.cuda.empty_cache()
print("scratch_bytes", nk.stats.get("scratch_bytes"))
json.dump(res, open(sys.argv[1], "w"), indent=1)
