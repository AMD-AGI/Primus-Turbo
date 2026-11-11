"""Sustained-loop clock / power / PPT-throttle probe for one MLA forward kernel on HIP device 0.

HIP device 0 is amd-smi GPU 3 on smci355-ccs-aus-n14-17 (PCI bus 0x75); pass --smi-gpu otherwise.
Throttle = growth of PPT_ACCUMULATED over growth of ACCUMULATION_COUNTER during the loop.

Usage: python power_probe.py --kernel DIR|aiter [--shape b1_s16k] [--secs 10] [--smi-gpu 3]
"""

import argparse
import json
import re
import statistics
import subprocess
import threading
import time

import torch

import ab_bench as ab


def smi_json(gpu, flags):
    out = subprocess.run(["amd-smi", "metric", "-g", str(gpu), *flags, "--json"], capture_output=True, text=True)
    j = json.loads(out.stdout)
    j = j["gpu_data"] if isinstance(j, dict) and "gpu_data" in j else j
    return j[0] if isinstance(j, list) else j


def throttle_counters(gpu):
    out = subprocess.run(["amd-smi", "metric", "-g", str(gpu), "-v"], capture_output=True, text=True).stdout
    acc = int(re.search(r"ACCUMULATION_COUNTER:\s*(\d+)", out).group(1))
    ppt = int(re.search(r"PPT_ACCUMULATED:\s*(\d+)", out).group(1))
    return acc, ppt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--shape", default="b1_s16k")
    ap.add_argument("--secs", type=float, default=10.0)
    ap.add_argument("--smi-gpu", type=int, default=3)
    args = ap.parse_args()

    label, B, S, H = next(s for s in ab.FULL if s[0] == args.shape)
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(S, B, H, d, generator=g, device="cuda", dtype=torch.bfloat16) for d in (ab.DQK, ab.DQK, ab.DV))
    if args.kernel == "aiter":
        ab.GlobalBackendManager.set_attn_backend(ab.BackendType.AITER, ab.PrecisionType.BF16_FP16_FP32)
        leaves = [t.permute(1, 0, 2, 3).detach().requires_grad_() for t in (q, k, v)]
        fn = lambda: ab.pt.ops.flash_attn_func(*leaves, softmax_scale=ab.DQK**-0.5, causal=True)  # noqa: E731
    else:
        m = ab.load_pkg(args.kernel, "mla_probe")
        fn = lambda: m.mla_fwd_sbhd(q, k, v, causal=True, return_lse=True)  # noqa: E731

    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                j = smi_json(args.smi_gpu, ["-c", "-p"])
                samples.append((float(j["clock"]["gfx_0"]["clk"]["value"]), float(j["power"]["socket_power"]["value"])))
            except Exception:  # noqa: BLE001
                pass

    th = threading.Thread(target=sampler)
    acc0, ppt0 = throttle_counters(args.smi_gpu)
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    n, t_end = 0, time.time() + args.secs
    st.record()
    th.start()
    while time.time() < t_end:
        for _ in range(10):
            fn()
        n += 10
        torch.cuda.synchronize()
    en.record()
    torch.cuda.synchronize()
    stop.set()
    th.join()
    acc1, ppt1 = throttle_counters(args.smi_gpu)
    ms = st.elapsed_time(en) / n
    tflops = B * S * (S + 1) * H * (ab.DQK + ab.DV) / ms / 1e9
    warm = samples[2:] or samples
    clk = statistics.median(s[0] for s in warm)
    pw = statistics.median(s[1] for s in warm)
    thr = (ppt1 - ppt0) / max(1, acc1 - acc0)
    print(
        f"{args.kernel} {label}: {ms:.4f} ms ({tflops:.1f} TFLOPS) | clk {clk:.0f} MHz (min {min(s[0] for s in warm):.0f}"
        f" max {max(s[0] for s in warm):.0f}) | power {pw:.0f} W | PPT throttle {thr:.2f} | TFLOPS/MHz {tflops / clk:.4f}"
        f" | {len(warm)} samples, {n} calls",
        flush=True,
    )


if __name__ == "__main__":
    main()
