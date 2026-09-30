#!/usr/bin/env python3
"""lab-kdq: per-kernel (dq only) and full-op timing, same process, ONE shape.

usage: kbench.py SHAPE MODE ITERS LABEL=/abs/impl ...
  MODE blk   : per-kernel blocked ruler (lead 4 + block 9, palindromic rounds over arms,
               256 MB L2 flush before every timed call, as benchmark.py) for the dq launch
               alone; then the same for the full op.
  MODE gb    : GEMM-lowered clock: before EVERY timed call, NG bf16 GEMMs
               (32768x4096 @ 4096x14336, ~3.8 TFLOP each, the e2e MLP up-proj) then the
               timed call immediately after; sclk read after each call. Bounded: ITERS calls
               per arm, a few seconds of GEMM in total -- not a burn loop.
Prints KB lines: KB shape mode what arm median_ms min max n sclk_med
"""
import os, sys, time, statistics
from pathlib import Path
HERE = Path.cwd()
sys.path.insert(0, str(HERE / "ut")); sys.path.insert(0, str(HERE))
import torch
import torch.nn.functional as F
from common import SHAPES, load_impl, make_inputs
from refcache_util import cached_forward

shape, mode, iters = sys.argv[1], sys.argv[2], int(sys.argv[3])
arms = [(a.split("=", 1)[0], Path(a.split("=", 1)[1]).resolve()) for a in sys.argv[4:]]
NG = int(os.environ.get("KB_NG", "4"))
PHYS = None


def sclk():
    try:
        with open("/sys/class/drm/card1/device/pp_dpm_sclk") as f:
            for line in f:
                if line.rstrip().endswith("*"):
                    return int(line.split(":")[1].strip().split("Mhz")[0])
    except Exception:
        pass
    return -1


if mode == "gb":            # hipBLASLt init before any arm's _env.py
    xg = torch.randn(32768, 4096, device="cuda", dtype=torch.bfloat16)
    wg = torch.randn(14336, 4096, device="cuda", dtype=torch.bfloat16) * 0.02
    for _ in range(2):
        F.linear(xg, wg)
    torch.cuda.synchronize()

b, sq, skv, hq, hkv, d = SHAPES[shape]
q, k, v, do = make_inputs(shape, seed=0)
o, lse = cached_forward(shape, q, k, v, causal=True)
lse = lse.contiguous().float()
torch.cuda.synchronize()
scale = d ** -0.5
g = hq // hkv
flush = torch.empty(256 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)

full, dqk = {}, {}
for label, path in arms:
    fn = load_impl(path)
    mod = sys.modules["op_impl_" + str(abs(hash(str(path))))]
    K = mod._k
    fn(do, q, k, v, o, lse, causal=True)          # compile + first launch of every kernel
    torch.cuda.synchronize()
    delta = torch.empty((b, hq, sq), device="cuda", dtype=torch.float32)
    n_rows = b * sq * hq
    mod._launch("delta", K.launch_delta, (do, o, delta, sq, hq, n_rows, n_rows // K.ROWS_DELTA,
                                          torch.cuda.current_stream()))
    dq_o = torch.empty_like(q)
    grouped = hasattr(K, "launch_dqg") and hq % K.DQ_NW == 0 and sq % K.DQ_BQW == 0
    nsp_q = 1
    while ((sq + K.BLOCK_Q - 1) // K.BLOCK_Q) * hq * b * nsp_q < 2048 and nsp_q < 8:
        nsp_q *= 2
    assert nsp_q == 1, "per-kernel timing is for the non-split path (proxy/prod)"

    def mk(mod=mod, K=K, delta=delta, dq_o=dq_o, grouped=grouped):
        st = torch.cuda.current_stream()
        if grouped:
            a = (q, k, v, do, o, lse, delta, dq_o, float(scale), sq, skv, hq, hkv, g,
                 skv // K.KV_STEP, skv - sq, 1, sq // K.DQ_BQW, hq // K.DQ_NW, b, st)
            return lambda: mod._launch("dqg", K.launch_dqg, a)
        a = (q, k, v, do, lse, delta, dq_o, float(scale), sq, skv, hq, hkv, g,
             skv // K.KV_STEP, skv - sq, 1, sq // K.BLOCK_Q, hq, b, st)
        return lambda: mod._launch("dq", K.launch_dq, a)
    dqk[label] = mk()
    dqk[label]()
    full[label] = (lambda fn=fn: fn(do, q, k, v, o, lse, causal=True))
    torch.cuda.synchronize()
    print(f"# arm {label} {path} grouped={grouped}", flush=True)

ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)


import threading
_SAMP = []            # (perf_counter, sclk) from a 0.5 ms sysfs poller while a run is active
_SAMP_ON = [False]


def _sampler():
    while True:
        if _SAMP_ON[0]:
            _SAMP.append((time.perf_counter(), sclk()))
            if len(_SAMP) > 400000:
                del _SAMP[:200000]
        time.sleep(0.0005)


threading.Thread(target=_sampler, daemon=True).start()


def timed(f, pre):
    """returns (kernel ms, sclk median inside the call's host window, sclk right after pre, pre ms)"""
    tp = time.perf_counter()
    pre()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    c_pre = sclk()
    ev0.record(); f(); ev1.record(); ev1.synchronize()
    t1 = time.perf_counter()
    win = [c for (t, c) in _SAMP[-4000:] if t0 <= t <= t1]
    return ev0.elapsed_time(ev1), (statistics.median(win) if win else sclk()), c_pre, (t0 - tp) * 1e3


def run(what, table, pre, lead, block):
    labels = [a for a, _ in arms]
    for lb in labels:                       # warm, continuous
        t_end = time.perf_counter() + 2.0
        while time.perf_counter() < t_end:
            table[lb]()
        torch.cuda.synchronize()
    res = {lb: [] for lb in labels}
    _SAMP_ON[0] = True
    rounds = -(-iters // block)
    for r in range(rounds):
        for lb in (labels if r % 2 == 0 else labels[::-1]):
            for _ in range(lead):
                timed(table[lb], pre)
            for _ in range(block):
                res[lb].append(timed(table[lb], pre))
    _SAMP_ON[0] = False
    for lb in labels:
        ts = sorted(x[0] for x in res[lb]); ck = [x[1] for x in res[lb]]
        cp = [x[2] for x in res[lb]]; pm = [x[3] for x in res[lb]]
        print(f"KB shape={shape} mode={mode} what={what} arm={lb} median_ms={statistics.median(ts):.4f} "
              f"min={ts[0]:.4f} max={ts[-1]:.4f} n={len(ts)} sclk_med={statistics.median(ck)} "
              f"sclk_pre={statistics.median(cp)} pre_ms={statistics.median(pm):.2f} "
              f"order={','.join(labels)}", flush=True)


if mode == "blk":
    pre = lambda: flush.zero_()
    run("dq", dqk, pre, 4, 9)
    run("full", full, pre, 4, 9)
elif mode == "gb":
    def pre():
        for _ in range(NG):
            F.linear(xg, wg)
    run("dq", dqk, pre, 1, 5)
    run("full", full, pre, 1, 5)
print("KBDONE", flush=True)
