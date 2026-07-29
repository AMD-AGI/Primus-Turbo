"""Diagnostic driver for rocprofv3 PMC: run ONLY the campaign `fwd down balanced` GEMM main
kernel, mx (preshuffle=False) or tw, a few times. Regime classification for the weakest
campaign configs. Read-only w.r.t. production code.
"""
import sys

import torch

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.flydsl.grouped_gemm.gemm_fp8_grouped_kernel import (
    grouped_gemm_fp8_tensorwise_flydsl_kernel as fly_tw,
)
from primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel import grouped_gemm_mxfp8_flydsl_kernel as fly_mx
from primus_turbo.pytorch.core.low_precision import float8_e4m3

DEV = "cuda"
G, M, U = 32, 131072, 512
WHICH = sys.argv[1] if len(sys.argv) > 1 else "mx"
PROJ = sys.argv[2] if len(sys.argv) > 2 else "down"
DIST = sys.argv[3] if len(sys.argv) > 3 else "balanced"
A, B = {"gate_up": (2944, 5760), "down": (2944, 2944)}[PROJ]
ONE = torch.ones(1, dtype=torch.float32, device=DEV)


def f8(*s):
    t = torch.empty(s, dtype=float8_e4m3, device=DEV)
    t.view(torch.uint8).random_(0, 64)
    return t


def _alloc(w):
    NU = M // U
    s = sum(w)
    raw = [max(1, round(NU * wi / s)) for wi in w]
    raw[raw.index(max(raw))] += NU - sum(raw)
    return raw


units = {
    "balanced": [M // U // G] * G,
    "heavy": _alloc([1.0 / (i + 1) ** 2.2 for i in range(G)]),
}[DIST]
o = [0]
for u in units:
    o.append(o[-1] + u * U)
o = torch.tensor(o, dtype=torch.int64, device=DEV)

a, w = f8(M, A), f8(G, B, A)
a_s = torch.full((M, A // 32), 127, dtype=torch.uint8, device=DEV)
w_s = torch.full((G, B, A // 32), 127, dtype=torch.uint8, device=DEV)

if WHICH == "mx":
    fly_mx(a, a_s, w, w_s, o, B, A, out_dtype=torch.bfloat16, num_cu=-1)
    torch.cuda.synchronize()
    fn = lambda: fly_mx(a, a_s, w, w_s, o, B, A, out_dtype=torch.bfloat16, num_cu=-1, preshuffle=False)
else:
    fn = lambda: fly_tw(a, w, ONE, ONE, o, trans_b=True, out_dtype=torch.bfloat16, num_cu=-1)

for _ in range(20):
    fn()
torch.cuda.synchronize()
for _ in range(5):
    fn()
torch.cuda.synchronize()
print(f"done {WHICH} {PROJ} {DIST}", flush=True)
