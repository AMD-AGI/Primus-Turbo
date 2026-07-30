"""wall / sclk / power triple for one bench shape, palindrome-ordered mx/tw/tw/mx.

The grouped mxfp8 NT kernel runs at ~99% of board TBP, so wall time alone mixes
"spent fewer cycles" with "was allowed a higher clock". Each arm drives the GPU
continuously while rocm-smi is sampled, so sclk/power are load-steady; the
palindrome order cancels any drift between the first and last arm.
"""
import statistics
import subprocess
import sys
import threading
import time

import torch

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.flydsl.grouped_gemm.gemm_fp8_grouped_kernel import (
    grouped_gemm_fp8_tensorwise_flydsl_kernel as fly_tw,
)
from primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel import (
    grouped_gemm_mxfp8_flydsl_kernel as fly_mx,
)
from primus_turbo.pytorch.core.low_precision import float8_e4m3

DEV = "cuda"
G, M = 32, 131072
A, B = 2944, 2944
DRIVE_S = float(sys.argv[1]) if len(sys.argv) > 1 else 14.0
ONE = torch.ones(1, dtype=torch.float32, device=DEV)


def f8(*s):
    t = torch.empty(s, dtype=float8_e4m3, device=DEV)
    t.view(torch.uint8).random_(0, 64)
    return t


def smi():
    out = subprocess.run(
        ["rocm-smi", "-d", "2", "--showgpuclocks", "--showpower", "--csv"],
        capture_output=True, text=True,
    ).stdout
    for ln in out.splitlines():
        if ln.startswith("card"):
            f = ln.split(",")
            return float(f[1].split("(")[1].split("Mhz")[0]), float(f[2])
    return None


def drive(fn, secs):
    """Run fn back-to-back for secs while sampling rocm-smi; return (ms, sclk, W)."""
    stop, samp = threading.Event(), []

    def poll():
        time.sleep(2.0)  # let DVFS settle into the load state
        while not stop.is_set():
            s = smi()
            if s:
                samp.append(s)
            time.sleep(0.4)

    th = threading.Thread(target=poll, daemon=True)
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    th.start()
    t_end, ts = time.time() + secs, []
    while time.time() < t_end:
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(8):
            fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / 8)
    stop.set()
    th.join()
    return (
        statistics.median(ts),
        statistics.mean(s[0] for s in samp),
        statistics.mean(s[1] for s in samp),
    )


o = torch.tensor([i * (M // G) for i in range(G + 1)], dtype=torch.int64, device=DEV)
a, w = f8(M, A), f8(G, B, A)
a_s, w_s = (torch.full(s, 127, dtype=torch.uint8, device=DEV) for s in ((M, A // 32), (G, B, A // 32)))
mx_full = lambda: fly_mx(a, a_s, w, w_s, o, B, A, out_dtype=torch.bfloat16, num_cu=-1)
arms = {
    "mx": lambda: fly_mx(a, a_s, w, w_s, o, B, A, out_dtype=torch.bfloat16, num_cu=-1, preshuffle=False),
    "tw": lambda: fly_tw(a, w, ONE, ONE, o, trans_b=True, out_dtype=torch.bfloat16, num_cu=-1),
}
mx_full()
torch.cuda.synchronize()
print(f"idle: sclk {smi()[0]:.0f} MHz  power {smi()[1]:.0f} W")
flop = 2.0 * M * B * A
res = {}
for k in ("mx", "tw", "tw", "mx"):
    ms, sclk, pw = drive(arms[k], DRIVE_S)
    res.setdefault(k, []).append((ms, sclk, pw))
    print(f"{k:3s} wall {ms:.4f} ms | sclk {sclk:7.1f} MHz | power {pw:6.1f} W | "
          f"{flop / ms * 1e-9:7.1f} TFLOPS | {pw * ms * 1e-3 / flop * 1e12:.4f} pJ/FLOP",
          flush=True)
mw, ms_ = statistics.mean(x[0] for x in res["mx"]), statistics.mean(x[1] for x in res["mx"])
tw_, ts_ = statistics.mean(x[0] for x in res["tw"]), statistics.mean(x[1] for x in res["tw"])
print(f"\nwall tw/mx = {tw_ / mw:.4f} | sclk mx/tw = {ms_ / ts_:.4f} | "
      f"cycle-normalized tw/mx = {(tw_ * ts_) / (mw * ms_):.4f}")
