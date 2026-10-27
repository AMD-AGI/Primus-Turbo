###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Measure a ladder and print its table. Runs on the GPU it measures (one process, one device).

    python3 -m hkl.ladder llama/ladder_fwd.json                         # the main line + baselines
    python3 -m hkl.ladder llama/ladder_bwd.json --side -r 12 --json bwd.json --md bwd.md
    python3 -m hkl.ladder mla/ladder_bwd.json --rung hk_v0 r1 r7 --shape b=1 sq=4096 skv=4096 hq=16 hkv=16

Protocol (the HK GEMM ladder's, plus what the a08-1 attention work learned on this power-capped card):
  * every arm is checked once against the fp32 reference at the timing shape before any timing starts
    (--verify none skips it); a failure aborts the campaign there;
  * each arm is pre-warmed for --warmup-s seconds (clocks settle under the power cap);
  * R rounds; each round runs every arm once, in an order rotated by one per round (cyclic shift), so a
    step's delta is formed between neighbouring cells of the same round and drift cancels;
  * a cell = W unmeasured calls (block alternation: never one call per arm) then I measured calls, each
    timed by its own event pair with a --flush-mb fill enqueued before it (no call sees the previous
    one's L2);
  * the top rung is also loaded a second time as a separate arm ("<id>#null", its own module instance):
    the spread between the two bounds how small a delta can be and still mean something;
  * sclk: a host thread samples the GPU's hwmon freq1_input every ~2 ms; each cell reports its median
    (on this card time follows clock: cycles are stable, sustained clocks are not).
Results: per arm the median over rounds of the cell medians (ms), TFLOP/s on the useful FLOPs, the step
over the previous rung of the main line (per-round ratio, median and 95% CI), sclk. --json keeps every
sample; --md writes the table.
"""

import argparse
import glob
import json
import math
import os
import statistics
import sys
import threading
import time

import torch

from . import ref as R
from . import runner

T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 12: 2.18,
       15: 2.13, 19: 2.09, 20: 2.09, 25: 2.06, 29: 2.05}  # fmt: skip


def t95(dof):
    keys = sorted(k for k in T95 if k <= dof)
    return T95[keys[-1]] if keys else 12.71


class Sclk:
    """Background sampler of the device's current shader clock (MHz) from hwmon; inert if not found."""

    def __init__(self, dev_index=0):
        self.path = os.environ.get("HKL_HWMON_FREQ") or self._find(dev_index)
        self.samples, self._stop = [], threading.Event()
        self._t = None

    @staticmethod
    def _find(i):
        p = torch.cuda.get_device_properties(i)
        bus = getattr(p, "pci_bus_id", None)
        dom = getattr(p, "pci_domain_id", 0) or 0
        dv = getattr(p, "pci_device_id", 0) or 0
        if bus is None:
            return None
        hits = glob.glob(f"/sys/bus/pci/devices/{dom:04x}:{bus:02x}:{dv:02x}.0/hwmon/hwmon*/freq1_input")
        return hits[0] if hits else None

    def start(self):
        if self.path:
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                with open(self.path) as f:
                    self.samples.append((time.monotonic(), int(f.read()) / 1e6))
            except OSError:
                return
            time.sleep(0.002)

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join()

    def median(self, t0, t1):
        v = [m for t, m in self.samples if t0 <= t <= t1]
        return statistics.median(v) if v else None


def make_call(arm, direction, inp, sh):
    if direction == "fwd":
        return lambda: arm.fwd(inp["q"], inp["k"], inp["v"], causal=sh.causal)
    return lambda: arm.bwd(inp["do"], inp["q"], inp["k"], inp["v"], inp["o"], inp["lse"], causal=sh.causal)


def verify(arms, calls, direction, refs, min_snr):
    for arm in arms:
        out = calls[arm.id]()
        torch.cuda.synchronize()
        names = ("o", "lse") if direction == "fwd" else ("dq", "dk", "dv")
        snr = {n: R.snr_db(x, refs[n]) for n, x in zip(names, out)}
        s = " ".join(f"{k} {v:.1f}" for k, v in snr.items())
        print(f"  verify {arm.id:<18} {s}", file=sys.stderr, flush=True)
        if any(not (v >= min_snr) for v in snr.values()):
            sys.exit(f"{arm.id} failed verification ({s} dB < {min_snr}); campaign aborted before timing")


def cell(call, flush, warm, iters):
    for _ in range(warm):
        flush.fill_(0)
        call()
    beg = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        flush.fill_(0)
        beg[i].record()
        call()
        end[i].record()
    torch.cuda.synchronize()
    return [beg[i].elapsed_time(end[i]) for i in range(iters)]


def summarize(xs):
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, t95(len(xs) - 1) * statistics.stdev(xs) / math.sqrt(len(xs))


def parse_shape(items, prob):
    over = {}
    for it in items or []:
        k, v = it.split("=")
        over[k] = (v.lower() in ("1", "true")) if k == "causal" else int(v)
    return R.Shape.from_problem(prob, **over)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ladder")
    ap.add_argument("--rung", nargs="*", help="rungs in ladder order (default: the main line, side rungs excluded)")
    ap.add_argument("--side", action="store_true", help="also time the side rungs (negative results etc.)")
    ap.add_argument("--baseline", nargs="*", help="baseline ids (default: all)")
    ap.add_argument("--shape", nargs="*", help="overrides of the ladder's problem, e.g. b=1 sq=4096")
    ap.add_argument("-r", "--rounds", type=int, default=10)
    ap.add_argument("-i", "--iters", type=int, default=20)
    ap.add_argument("-w", "--warm", type=int, default=3, help="unmeasured calls per cell")
    ap.add_argument("--warmup-s", type=float, default=2.0, help="pre-warm seconds per arm")
    ap.add_argument("--flush-mb", type=int, default=512)
    ap.add_argument("--no-null", action="store_true", help="skip the null-control arm")
    ap.add_argument("--verify", choices=("ref", "none"), default="ref")
    ap.add_argument("--min-snr", type=float, default=40.0)
    ap.add_argument("--json")
    ap.add_argument("--md")
    a = ap.parse_args(argv)

    spec = runner.load_ladder(a.ladder)
    direction, prob = spec["direction"], spec["problem"]
    main_line = [r["id"] for r in spec["rungs"] if not r.get("side")]
    rungs = a.rung or (main_line + ([r["id"] for r in spec["rungs"] if r.get("side")] if a.side else []))
    baselines = a.baseline if a.baseline is not None else runner.baseline_ids(a.ladder)
    sh = parse_shape(a.shape, prob)
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"device {props.gcnArchName} ({props.multi_processor_count} CUs)  torch {torch.__version__}  "
          f"ladder {spec['name']}  shape {sh}", flush=True)  # fmt: skip

    arms = [runner.load_baseline(a.ladder, b) for b in baselines] + [runner.load(a.ladder, r) for r in rungs]
    if not a.no_null and rungs:
        null = runner.load(a.ladder, rungs[-1])
        null.id = f"{rungs[-1]}#null"
        arms.append(null)

    inp = R.make_inputs(sh, dev, seed=0)
    refs = None
    if direction == "bwd" or a.verify == "ref":
        t0 = time.time()
        o_ref, lse_ref = R.ref_fwd(sh, inp["q"], inp["k"], inp["v"])
        inp["o"], inp["lse"] = o_ref.to(torch.bfloat16), lse_ref
        refs = {"o": o_ref, "lse": lse_ref}
        if direction == "bwd" and a.verify == "ref":
            dq, dk, dv = R.ref_bwd(sh, inp["q"], inp["k"], inp["v"], inp["do"])
            refs = {"dq": dq, "dk": dk, "dv": dv}
        print(f"reference {time.time() - t0:.1f} s", file=sys.stderr, flush=True)
    calls = {arm.id: make_call(arm, direction, inp, sh) for arm in arms}
    if a.verify == "ref":
        verify(arms, calls, direction, refs, a.min_snr)
    del refs
    torch.cuda.empty_cache()

    flush = torch.empty(a.flush_mb * (1 << 20) // 4, dtype=torch.float32, device=dev)
    for arm in arms:  # pre-warm; synchronize every call: launches are asynchronous, and an unsynchronized
        t_end = time.time() + a.warmup_s  # loop queues thousands of calls in seconds (minutes of GPU work)
        while time.time() < t_end:
            calls[arm.id]()
            torch.cuda.synchronize()

    sclk = Sclk().start()
    ids = [arm.id for arm in arms]
    samples = {i: [] for i in ids}  # per round: cell median ms
    clocks = {i: [] for i in ids}
    for rnd in range(a.rounds):
        order = ids[rnd % len(ids):] + ids[:rnd % len(ids)]
        for i in order:
            t0 = time.monotonic()
            ms = cell(calls[i], flush, a.warm, a.iters)
            t1 = time.monotonic()
            samples[i].append(statistics.median(ms))
            clocks[i].append(sclk.median(t0, t1))
        print(f"  round {rnd + 1}/{a.rounds}: " + " ".join(f"{i}={samples[i][-1]:.4f}" for i in ids), flush=True)
    sclk.stop()

    flops = R.useful_flops(sh, direction)
    rows = []
    by_id = spec["_by_id"]
    prev = None
    for i in ids:
        m = statistics.median(samples[i])
        mean, ci = summarize(samples[i])
        c = [x for x in clocks[i] if x is not None]
        row = {"arm": i, "ms": m, "mean": mean, "ci95": ci, "tflops": flops / (m * 1e-3) / 1e12,
               "sclk": statistics.median(c) if c else None, "rounds": samples[i]}  # fmt: skip
        base_id = i.split("#")[0]
        if base_id in by_id:
            r = by_id[base_id]
            row["feature"] = r.get("feature", "")
            row["ref_ms_a08"] = r.get("ref_ms")
            if not i.endswith("#null") and base_id in main_line and prev is not None:
                ratios = [p / c for p, c in zip(samples[prev], samples[i])]
                sm, sci = summarize(ratios)
                row["step"] = {"over": prev, "median": statistics.median(ratios), "mean": sm, "ci95": sci}
            if not i.endswith("#null") and base_id in main_line:
                prev = i
        rows.append(row)
    if not a.no_null and rungs:
        top, nul = rungs[-1], f"{rungs[-1]}#null"
        ratios = [p / c for p, c in zip(samples[top], samples[nul])]
        nm, nci = summarize(ratios)
        null_ctl = {"median": statistics.median(ratios), "mean": nm, "ci95": nci}
    else:
        null_ctl = None

    hdr = f"{'arm':<20} {'ms':>8} {'TFLOP/s':>8} {'step':>16} {'sclk':>6} {'a08 ms':>7}  feature"
    lines = [hdr, "-" * len(hdr)]
    for row in rows:
        st = row.get("step")
        s = f"{st['median']:.3f}x±{st['ci95']:.3f}" if st else ""
        sc = f"{row['sclk']:.0f}" if row["sclk"] else "-"
        ra = f"{row['ref_ms_a08']:.3f}" if row.get("ref_ms_a08") else ""
        lines.append(f"{row['arm']:<20} {row['ms']:8.4f} {row['tflops']:8.1f} {s:>16} {sc:>6} {ra:>7}  "
                     f"{row.get('feature', '')[:70]}")  # fmt: skip
    if null_ctl:
        lines.append(f"null control {rungs[-1]} vs #null: {null_ctl['median']:.4f}x ± {null_ctl['ci95']:.4f}")
    print("\n".join(lines))
    meta = {"ladder": spec["name"], "shape": repr(sh), "device": props.gcnArchName, "cus": props.multi_processor_count,
            "torch": torch.__version__, "rounds": a.rounds, "iters": a.iters, "warm": a.warm, "flush_mb": a.flush_mb,
            "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"), "host": os.uname().nodename,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "useful_flops": flops}  # fmt: skip
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"meta": meta, "rows": rows, "null_control": null_ctl}, f, indent=1)
    if a.md:
        md = [f"{spec.get('title', spec['name'])}, {sh}, {meta['host']} GPU {meta['hip_visible_devices']}, {meta['time']}",
              "", "| Rung | ms | TFLOP/s | Step | sclk MHz | Feature |", "| --- | ---: | ---: | ---: | ---: | --- |"]  # fmt: skip
        for row in rows:
            st = row.get("step")
            step = f"{st['median']:.3f}x" if st else ""
            sc = f"{row['sclk']:.0f}" if row["sclk"] else "-"
            md.append(f"| {row['arm']} | {row['ms']:.4f} | {row['tflops']:.0f} | {step} | {sc} | "
                      f"{row.get('feature', '')} |")  # fmt: skip
        with open(a.md, "w") as f:
            f.write("\n".join(md) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
