"""Same-process A/B of one FlyDSL forward impl against the aiter ASM forward.

One shape per process (h31). Prints one RESULT JSON line; the ratio fly/asm is the
drift-immune figure to compare across processes with different flydsl versions.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--impl", required=True)
ap.add_argument("--shape", default="prod", choices=["fast", "proxy", "prod"])
ap.add_argument("--iters", type=int, default=51)
ap.add_argument("--warmup-s", type=float, default=3.0)
ap.add_argument("--json")
a = ap.parse_args()

spec = importlib.util.spec_from_file_location("cand_impl", os.path.join(a.impl, "impl.py"))
impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impl)          # puts the impl's flydsl on sys.path first
import flydsl
import torch

sys.path.insert(0, "/home/lihuzhan/code/aiter-src")
from aiter.ops.mha import fmha_fwd_with_sink_asm

SHAPES = {"fast": (1, 1024, 8, 2), "proxy": (1, 4096, 32, 8), "prod": (4, 8192, 32, 8)}
b, s, hq, hkv = SHAPES[a.shape]
d = 128
flop = 2.0 * b * hq * d * 2 * (s * (s + 1) / 2)
torch.manual_seed(0)
q = torch.randn(b, s, hq, d, device="cuda", dtype=torch.bfloat16)
k = torch.randn(b, s, hkv, d, device="cuda", dtype=torch.bfloat16)
v = torch.randn(b, s, hkv, d, device="cuda", dtype=torch.bfloat16)
scale = d ** -0.5


def asm():
    return fmha_fwd_with_sink_asm(q, k, v, scale, True, True)


def fly():
    return impl.attn_fwd(q, k, v, scale, True)


def sclk():
    try:
        out = subprocess.run(["/opt/venv/bin/rocm-smi", "--showclocks"], capture_output=True,
                             text=True, timeout=10).stdout
        for line in out.splitlines():
            if "sclk" in line:
                return int(line.split("(")[1].split("Mhz")[0])
    except Exception:
        return None


def sqnr(x, ref):
    x, ref = x.float(), ref.float()
    return float(10 * torch.log10(ref.pow(2).sum() / (x - ref).pow(2).sum().clamp_min(1e-30)))


o_a = asm()[0]
o_f, lse_f = fly()
torch.cuda.synchronize()
agree_db = sqnr(o_f, o_a)
assert agree_db > 40, f"fly vs asm output SQNR {agree_db:.1f} dB -- wrong kernel"

flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")


def timed(fn, n):
    ts = []
    for _ in range(n):
        flush.zero_()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1))
    return ts


t0 = time.time()
while time.time() - t0 < a.warmup_s:
    asm(); fly()
torch.cuda.synchronize()
clk0 = sclk()
timed(asm, 5)                           # burn-in arm, discarded
ta, tf = [], []
half = a.iters // 2 + 1
ta += timed(asm, half); tf += timed(fly, half)
tf += timed(fly, half); ta += timed(asm, half)
clk1 = sclk()
med = lambda xs: sorted(xs)[len(xs) // 2]
r = {
    "shape": a.shape, "impl": os.path.abspath(a.impl), "flydsl": flydsl.__version__,
    "asm_ms": med(ta), "fly_ms": med(tf),
    "asm_tflops": flop / med(ta) / 1e9, "fly_tflops": flop / med(tf) / 1e9,
    "ratio_fly_over_asm_tflops": med(ta) / med(tf),
    "agree_db": agree_db, "sclk": [clk0, clk1], "n": len(ta),
}
print("RESULT", json.dumps(r))
if a.json:
    with open(a.json, "a") as f:
        f.write(json.dumps(r) + "\n")
