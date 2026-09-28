"""nkfix rewrite with the Triton transpose vs with torch's copy: bitwise equal? ms per call class."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import nkfix_b0 as nk
from transpose_triton import transpose_into
dev = "cuda"; BF = torch.bfloat16; M = 32768
CL = [("dgrad_qo", "dgrad", (M, 4096), (4096, 4096)), ("dgrad_kv", "dgrad", (M, 1024), (1024, 4096)),
      ("dgrad_mlp13", "dgrad", (M, 14336), (14336, 4096)), ("dgrad_mlp2", "dgrad", (M, 4096), (4096, 14336)),
      ("dgrad_head", "dgrad", (M, 128256), (128256, 4096)), ("wgrad_qo", "wgrad", (4096, M), (M, 4096)),
      ("wgrad_kv", "wgrad", (1024, M), (M, 4096)), ("wgrad_mlp13", "wgrad", (14336, M), (M, 4096)),
      ("wgrad_mlp2", "wgrad", (4096, M), (M, 14336)), ("wgrad_head", "wgrad", (128256, M), (M, 4096))]
def t_ms(fn, n=3):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(n):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    return sorted(ts)[n // 2]
res = []
for name, kind, ash, bsh in CL:
    if kind == "dgrad":
        a = torch.randn(ash, device=dev, dtype=BF); b = torch.randn(bsh, device=dev, dtype=BF) * 0.02
    else:
        a = torch.randn(ash[::-1], device=dev, dtype=BF).t(); b = torch.randn(bsh, device=dev, dtype=BF)
    ca = kind == "wgrad"
    nk._tt = transpose_into; o_tr = nk._rewrite(a, b, ca); ms_tr = t_ms(lambda: nk._rewrite(a, b, ca))
    nk._tt = None; o_to = nk._rewrite(a, b, ca); ms_to = t_ms(lambda: nk._rewrite(a, b, ca))
    ms_orig = t_ms(lambda: torch.mm(a, b), n=1)
    r = dict(name=name, bit_equal=bool(torch.equal(o_tr, o_to)), triton_ms=round(ms_tr, 3),
             torchcopy_ms=round(ms_to, 3), orig_ms=round(ms_orig, 3))
    print(json.dumps(r), flush=True); res.append(r)
    del a, b, o_tr, o_to; torch.cuda.empty_cache(); time.sleep(0.3)
json.dump(res, open(sys.argv[1], "w"), indent=1)
