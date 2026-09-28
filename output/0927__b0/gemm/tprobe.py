"""Transpose probe: Triton tiled transpose vs torch copy -- bit-exactness first, then time.

  tprobe.py toy  : small odd shapes only (first launch of the new kernel, its own process)
  tprobe.py prod : the nkfix copy shapes of the 32-layer step, ms + GB/s for both
"""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transpose_triton import transpose_into

dev = "cuda"; BF = torch.bfloat16
mode = sys.argv[1]
M = 32768
if mode == "toy":
    # (R, C, base-is-transposed?) -- odd sizes exercise every mask edge
    cases = [(1, 1, False), (7, 5, False), (65, 130, True), (257, 63, False), (300, 1000, True)]
else:
    cases = [(4096, 4096, False), (4096, 14336, False), (14336, 4096, False), (1024, 4096, False),
             (M, 4096, True), (M, 1024, True), (M, 14336, True), (M, 4096, False), (M, 14336, False),
             (M, 8192, True)]
res = []
for R, C, tr in cases:
    # x: logical (R, C); tr=True -> x is the transpose view of a contiguous (C, R) base
    base = torch.randn((C, R) if tr else (R, C), device=dev, dtype=BF)
    x = base.t() if tr else base
    ref = x.t().contiguous()
    y = torch.full((C, R), float("nan"), device=dev, dtype=BF)
    transpose_into(y, x)
    torch.cuda.synchronize()
    row = dict(R=R, C=C, x_transposed_view=tr, bit_exact=bool(torch.equal(y, ref)),
               nonfinite=int((~torch.isfinite(y)).sum()))
    if mode == "prod":
        def t(fn, n=5):
            fn(); torch.cuda.synchronize(); ts = []
            for _ in range(n):
                e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
                e0.record(); fn(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
            return sorted(ts)[n // 2]
        yy = torch.empty((C, R), device=dev, dtype=BF)
        row["torch_ms"] = round(t(lambda: yy.copy_(x.t())), 3)
        row["triton_ms"] = round(t(lambda: transpose_into(yy, x)), 3)
        gb = 2 * R * C * 2 / 1e9
        row["torch_GBs"] = round(gb / row["torch_ms"] * 1e3, 1)
        row["triton_GBs"] = round(gb / row["triton_ms"] * 1e3, 1)
    print(json.dumps(row), flush=True); res.append(row)
    time.sleep(0.2)
json.dump(res, open(sys.argv[2], "w"), indent=1)
print("ALL_BIT_EXACT" if all(r["bit_exact"] and r["nonfinite"] == 0 for r in res) else "MISMATCH")
