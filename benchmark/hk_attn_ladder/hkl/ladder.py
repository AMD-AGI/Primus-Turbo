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
  * R rounds; each round runs every arm once, rotated by one per round and in reverse order every other
    round, so an arm's predecessor alternates (a cyclic shift alone keeps every arm behind the same arm,
    and its power state carries over: the first e15-02 run measured a 3% null-control bias that way);
  * a cell = --settle-s seconds of synchronized calls (the shader clock settles to THIS arm's power draw;
    a cell of ~50 ms of work inherits the previous arm's clock), then I measured calls, each timed by its
    own event pair with a --flush-mb fill enqueued before it (no call sees the previous one's L2);
  * the top rung is also loaded a second time as a separate arm ("<id>#null", its own module instance):
    the spread between the two bounds how small a delta can be and still mean something;
  * sclk: a host thread samples the GPU's hwmon freq1_input every ~2 ms; each cell reports its median
    (on this card time follows clock: cycles are stable, sustained clocks are not).
Results: per arm the median over rounds of the cell medians (ms), TFLOP/s on the useful FLOPs, the step
over the rung's parent when it was timed, else over the previous rung of the main line (per-round ratio, median and 95% CI) in time and in cycles
(ms x the cell's median sclk), sclk. --json keeps every
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
    """Background sampler of the device's current shader clock (MHz) and, when its hwmon has power1_input,
    socket power (W); inert if not found."""

    def __init__(self, dev_index=0):
        self.path = os.environ.get("HKL_HWMON_FREQ") or self._find(dev_index)
        pw = os.environ.get("HKL_HWMON_POWER") or (os.path.join(os.path.dirname(self.path), "power1_input")
                                                   if self.path else None)  # fmt: skip
        self.ppath = pw if pw and os.path.exists(pw) else None
        self.samples, self.psamples, self._stop = [], [], threading.Event()
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
        fails = 0  # hwmon reads fail transiently (firmware busy); give up only on a long run of failures
        while not self._stop.is_set() and fails < 1000:
            for path, out in ((self.path, self.samples), (self.ppath, self.psamples)):
                if not path:
                    continue
                try:
                    with open(path) as f:
                        out.append((time.monotonic(), int(f.read()) / 1e6))
                    fails = 0
                except (OSError, ValueError):
                    fails += 1
            time.sleep(0.002)

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join()

    def median(self, t0, t1):
        v = [m for t, m in self.samples if t0 <= t <= t1]
        return statistics.median(v) if v else None

    def power(self, t0, t1):
        v = [w for t, w in self.psamples if t0 <= t <= t1]
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


def settle(call, flush, seconds, minimum=2):
    """Synchronized calls for `seconds` (at least `minimum`): the clock settles to this arm's own power draw."""
    t_end, n = time.time() + seconds, 0
    while n < minimum or time.time() < t_end:
        flush.fill_(0)
        call()
        torch.cuda.synchronize()
        n += 1


def cell(call, flush, iters, mode="cold", gap_s=0.02):
    """Per-call ms of one cell. cold: a flush before every call, each call its own event pair; hot: the iters
    calls back to back (no flush) inside one event pair, their mean (the Primus-Turbo README method); burst:
    each call alone after gap_s seconds of idle GPU (then a flush), its own event pair -- the clock has not
    been pulled down by this arm's own power draw, as for a kernel inside a training step."""
    if mode == "hot":
        beg, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        beg.record()
        for _ in range(iters):
            call()
        end.record()
        torch.cuda.synchronize()
        return [beg.elapsed_time(end) / iters]
    beg = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if mode == "burst":
            torch.cuda.synchronize()
            time.sleep(gap_s)
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
    ap.add_argument("--settle-s", type=float, default=1.0, help="synchronized unmeasured calls per cell (seconds)")
    ap.add_argument("--warmup-s", type=float, default=2.0, help="pre-warm seconds per arm")
    ap.add_argument("--flush-mb", type=int, default=512)
    ap.add_argument("--no-null", action="store_true", help="skip the null-control arm")
    ap.add_argument("--verify", choices=("ref", "none"), default="ref")
    ap.add_argument("--min-snr", type=float, default=40.0)
    ap.add_argument("--data", choices=("randn", "zeros"), default=os.environ.get("HKL_DATA", "randn"),
                    help="input data; zeros (no verification) shows the clock without the power cap's pull")
    ap.add_argument("--mode", choices=("cold", "hot", "burst"), default="cold",
                    help="cold: flush before every call (default); hot: back-to-back calls, one event pair; burst: "
                         "every call after --gap-ms of idle (no settle; sclk / cycles then mostly idle samples)")
    ap.add_argument("--gap-ms", type=float, default=20.0, help="burst mode: idle time before every call")
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

    if a.data == "zeros":
        a.verify = "none"  # an all-zero reference has no SNR
    inp = R.make_inputs(sh, dev, seed=0, data=a.data)
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
    cycles = {i: [] for i in ids}  # per round: cell median ms x sclk (Mcycles)
    watts = {i: [] for i in ids}  # per round: cell median socket power (W)
    for rnd in range(a.rounds):
        order = ids[rnd % len(ids):] + ids[:rnd % len(ids)]
        if rnd % 2:
            order = order[::-1]
        for i in order:
            if a.mode == "burst":
                torch.cuda.synchronize()
                time.sleep(0.3)  # the previous arm's draw fades; no settle: the point is an unsettled clock
            else:
                settle(calls[i], flush, a.settle_s)
            t0 = time.monotonic()
            ms = cell(calls[i], flush, a.iters, a.mode, a.gap_ms / 1e3)
            t1 = time.monotonic()
            samples[i].append(statistics.median(ms))
            clocks[i].append(sclk.median(t0, t1))
            cycles[i].append(samples[i][-1] * clocks[i][-1] / 1e3 if clocks[i][-1] else None)
            watts[i].append(sclk.power(t0, t1))
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
        cy = [x for x in cycles[i] if x is not None]
        w = [x for x in watts[i] if x is not None]
        mj = [x * ms for x, ms in zip(watts[i], samples[i]) if x is not None]  # W x ms = mJ per call
        row = {"arm": i, "ms": m, "mean": mean, "ci95": ci, "tflops": flops / (m * 1e-3) / 1e12,
               "sclk": statistics.median(c) if c else None, "mcyc": statistics.median(cy) if cy else None,
               "watts": statistics.median(w) if w else None, "mj": statistics.median(mj) if mj else None,
               "rounds": samples[i], "sclk_rounds": clocks[i], "watt_rounds": watts[i]}  # fmt: skip
        base_id = i.split("#")[0]
        if base_id in by_id:
            r = by_id[base_id]
            row["feature"] = r.get("feature", "")
            row["ref_ms_a08"] = r.get("ref_ms")
            # step over the rung's parent when it was timed, else over the previous main-line rung
            over = r.get("parent") if r.get("parent") in samples else (prev if base_id in main_line else None)
            if not i.endswith("#null") and over is not None:
                ratios = [p / c for p, c in zip(samples[over], samples[i])]
                sm, sci = summarize(ratios)
                row["step"] = {"over": over, "median": statistics.median(ratios), "mean": sm, "ci95": sci}
                cr = [p / c for p, c in zip(cycles[over], cycles[i]) if p and c]
                if cr:
                    row["step_cyc"] = {"median": statistics.median(cr), "ci95": summarize(cr)[1]}
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

    hdr = f"{'arm':<20} {'ms':>8} {'TFLOP/s':>8} {'over':>8} {'step':>16} {'step cyc':>9} {'sclk':>6} {'Mcyc':>7} {'W':>6} {'mJ':>6} {'a08 ms':>7}  feature"
    lines = [hdr, "-" * len(hdr)]
    for row in rows:
        st = row.get("step")
        s = f"{st['median']:.3f}x±{st['ci95']:.3f}" if st else ""
        sc = f"{row['sclk']:.0f}" if row["sclk"] else "-"
        ra = f"{row['ref_ms_a08']:.3f}" if row.get("ref_ms_a08") else ""
        sy = f"{row['step_cyc']['median']:.3f}x" if row.get("step_cyc") else ""
        mc = f"{row['mcyc']:.3f}" if row.get("mcyc") else "-"
        ov = st["over"] if st else ""
        wt = f"{row['watts']:.0f}" if row.get("watts") else "-"
        mj = f"{row['mj']:.2f}" if row.get("mj") else "-"
        lines.append(f"{row['arm']:<20} {row['ms']:8.4f} {row['tflops']:8.1f} {ov:>8} {s:>16} {sy:>9} {sc:>6} {mc:>7} {wt:>6} {mj:>6} {ra:>7}  "
                     f"{row.get('feature', '')[:60]}")  # fmt: skip
    if null_ctl:
        lines.append(f"null control {rungs[-1]} vs #null: {null_ctl['median']:.4f}x ± {null_ctl['ci95']:.4f}")
    print("\n".join(lines))
    meta = {"ladder": spec["name"], "shape": repr(sh), "device": props.gcnArchName, "cus": props.multi_processor_count,
            "torch": torch.__version__, "rounds": a.rounds, "iters": a.iters, "settle_s": a.settle_s, "flush_mb": a.flush_mb,
            "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"), "host": os.uname().nodename,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "useful_flops": flops, "data": a.data,
            "mode": a.mode, "gap_ms": a.gap_ms if a.mode == "burst" else None}  # fmt: skip
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"meta": meta, "rows": rows, "null_control": null_ctl}, f, indent=1)
    if a.md:
        md = [f"{spec.get('title', spec['name'])}, {sh}, {a.data} inputs, {a.mode} timing, {meta['host']} GPU {meta['hip_visible_devices']}, {meta['time']}",
              "", "| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |",
              "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]  # fmt: skip
        for row in rows:
            st = row.get("step")
            step = f"{st['median']:.3f}x" if st else ""
            sc = f"{row['sclk']:.0f}" if row["sclk"] else "-"
            sy = f"{row['step_cyc']['median']:.3f}x" if row.get("step_cyc") else ""
            mc = f"{row['mcyc']:.3f}" if row.get("mcyc") else "-"
            ov = st["over"] if st else ""
            wt = f"{row['watts']:.0f}" if row.get("watts") else "-"
            mj = f"{row['mj']:.2f}" if row.get("mj") else "-"
            md.append(f"| {row['arm']} | {row['ms']:.4f} | {row['tflops']:.0f} | {ov} | {step} | {sy} | {sc} | {mc} | {wt} | {mj} | "
                      f"{row.get('feature', '')} |")  # fmt: skip
        with open(a.md, "w") as f:
            f.write("\n".join(md) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
