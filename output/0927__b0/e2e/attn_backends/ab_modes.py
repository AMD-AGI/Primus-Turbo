"""Same-process asm vs fly at prod, under three launch patterns, to explain why opcheck's fwd
numbers depend on how they are timed (block of one kernel vs interleaved fwd/bwd).

  block   20 back-to-back calls of one kernel, one event pair each; blocks A B B A ...
  single  one call, then synchronize (GPU idles for the host gap), alternating arms
  train   fwd then bwd of the same arm, as a training step issues them; alternating arms
Prints per-pattern medians and fly/asm ratios. One shape (prod), P2 process only.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/lihuzhan/.local/flydsl0341")
os.environ["E2E_ATTN"] = "asm"
import torch  # noqa: E402
from e2e_attn import arms  # noqa: E402

torch.manual_seed(0)
b, s, hq, hkv, d = 4, 8192, 32, 8, 128
q = torch.randn(b, s, hq, d, device="cuda", dtype=torch.bfloat16)
k = torch.randn(b, s, hkv, d, device="cuda", dtype=torch.bfloat16)
v = torch.randn(b, s, hkv, d, device="cuda", dtype=torch.bfloat16)
do = torch.randn(b, s, hq, d, device="cuda", dtype=torch.bfloat16)
sc = d ** -0.5
F = {a: arms.get_fwd(a) for a in ("asm", "fly")}
B = {a: arms.get_bwd(a) for a in ("asm", "fly")}
O = {a: F[a](q, k, v, sc) for a in ("asm", "fly")}
for a in ("asm", "fly"):
    B[a](do, q, k, v, *O[a], sc)
torch.cuda.synchronize()
print(arms.check_versions(), flush=True)


def ev():
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def med(x):
    x = sorted(x)
    return x[len(x) // 2]


def sclk():
    try:
        return [l for l in open("/sys/class/drm/card0/device/pp_dpm_sclk") if "*" in l][0].split()[1]
    except Exception:
        return "?"


R = {}
N = int(os.environ.get("AB_N", "101"))
# block
for i, a in enumerate(["asm", "fly", "fly", "asm"] * 3):
    for _ in range(20):
        e0 = ev(); F[a](q, k, v, sc); e1 = ev()
        e1.synchronize()
        R.setdefault(("block_fwd", a), []).append(e0.elapsed_time(e1))
print("after block sclk", sclk(), flush=True)
# single
for i in range(N):
    for a in (("asm", "fly") if i % 2 == 0 else ("fly", "asm")):
        e0 = ev(); F[a](q, k, v, sc); e1 = ev()
        torch.cuda.synchronize()
        R.setdefault(("single_fwd", a), []).append(e0.elapsed_time(e1))
print("after single sclk", sclk(), flush=True)
# train pattern: fwd then bwd back to back, no sync inside
for i in range(N):
    for a in (("asm", "fly") if i % 2 == 0 else ("fly", "asm")):
        e0 = ev(); o, lse = F[a](q, k, v, sc); e1 = ev(); B[a](do, q, k, v, o, lse, sc); e2 = ev()
        torch.cuda.synchronize()
        R.setdefault(("train_fwd", a), []).append(e0.elapsed_time(e1))
        R.setdefault(("train_bwd", a), []).append(e1.elapsed_time(e2))
print("after train sclk", sclk(), flush=True)
for pat in ("block_fwd", "single_fwd", "train_fwd", "train_bwd"):
    ma, mf = med(R[(pat, "asm")]), med(R[(pat, "fly")])
    print(f"{pat:11s} asm {ma:7.4f}  fly {mf:7.4f}  fly/asm {mf / ma:6.3f}  (n={len(R[(pat, 'asm')])})",
          flush=True)
