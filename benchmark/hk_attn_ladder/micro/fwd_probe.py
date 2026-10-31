###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Phase breakdown of the MLA forward from an HKGEN_PROBE build (mla/ladder_fwd2.json, launch opt probe: true).

    python3 -m micro.fwd_probe mla/ladder_fwd2.json f0p [--shape b=2 sq=4096] [--warm-s 2] [--calls 5] [--json f]

Runs the rung for --warm-s seconds (synchronized calls: the clock settles under the power cap), then --calls more,
and decodes the probe records of each of those calls (k_mla_fwd.hip, HKGEN_PROBE: per wave and block the shader-cycle
stamps loop top / after B_0 / after INIT / after the TILE_C loop / after each TILE_Dk / loop bottom, the summed
TILE_C, TILE_D and FINAL region cycles, and the realtime at the loop top and bottom). Reports, averaged over waves and
calls: cycles per workgroup by phase, per TILE_C tile (region vs the shell between regions), per TILE_Dk, per block
(B_0 wait, INIT, epilogue, FINAL), and the realtime spread of the workgroups' end times by XCD (li % 8)."""

import argparse
import json
import statistics
import sys
import time

import torch

from hkl import ref as R
from hkl import runner

PR_MAXB, WORDS = 64, 16
STATE = {}


def parse_shape(items, prob):
    over = {}
    for it in items or []:
        k, v = it.split("=")
        over[k] = int(v)
    return R.Shape.from_problem(prob, **over)


def decode(buf, nwg):
    """[(li, wave, [block records])]: a record is the 16 words (unsigned) of one block that ran."""
    a = buf.view(nwg, 4, PR_MAXB, WORDS).cpu().numpy().astype("int64") & 0xFFFFFFFF
    out = []
    for li in range(nwg):
        for w in range(4):
            recs = [list(map(int, a[li, w, s])) for s in range(PR_MAXB) if a[li, w, s, 15] != 0]
            out.append((li, w, recs))
    return out


def d32(b, a):
    return (b - a) & 0xFFFFFFFF


def phases(recs):
    """Per-wave totals over its blocks (cycles), plus counts."""
    t = dict(b0=0, init=0, cloop=0, creg=0, ncc=0, diag=0, dreg=0, epi=0, fin=0, blocks=0, tiles=0,
             d=[0, 0, 0, 0], dn=[0, 0, 0, 0], span=0)  # fmt: skip
    for r in recs:
        t["blocks"] += 1
        t["tiles"] += r[15]
        t["b0"] += d32(r[1], r[0])
        t["init"] += d32(r[2], r[1])
        t["cloop"] += d32(r[3], r[2])
        t["creg"] += r[4]
        t["ncc"] += r[5]
        prev = r[3]
        for k in range(4):
            if r[6 + k]:
                t["d"][k] += d32(r[6 + k], prev)
                t["dn"][k] += 1
                prev = r[6 + k]
        t["diag"] += d32(prev, r[3])
        t["dreg"] += r[10]
        t["epi"] += d32(r[11], prev)
        t["fin"] += r[12]
        t["span"] += d32(r[11], r[0])
    return t


def report(calls_dec, nwg):
    per = {}  # key -> list over (call, wg, wave)
    ends = []  # (call, li, end realtime - first start of the call)
    for dec in calls_dec:
        t0 = min(r[0][13] for _, _, r in dec if r)
        for li, w, recs in dec:
            if not recs:
                continue
            p = phases(recs)
            for k, v in p.items():
                if isinstance(v, list):
                    for i, x in enumerate(v):
                        per.setdefault(f"{k}{i}", []).append(x)
                else:
                    per.setdefault(k, []).append(v)
            if w == 0:
                ends.append((li, d32(recs[0][13], t0), d32(recs[-1][14], t0)))
    m = {k: statistics.mean(v) for k, v in per.items()}
    # per wave id: the TILE_C loop, its regions, word 12 (FINAL regions, or with HKGEN_PROBE 2 the TILE_C barrier waits)
    byw = {}
    for dec in calls_dec:
        for li, w, recs in dec:
            if recs:
                p_ = phases(recs)
                byw.setdefault(w, []).append((p_["cloop"], p_["creg"], p_["fin"], p_["ncc"]))
    for w in sorted(byw):
        cl, cr, fi, nc = (statistics.mean(x[i] for x in byw[w]) for i in range(4))
        print(f"  wave {w}: per TILE_C tile: loop {cl / nc:.0f}, region {cr / nc:.0f}, shell {(cl - cr) / nc:.0f}, "
              f"word12/tile {fi / nc:.0f}")
    tot = m["span"]
    print(f"per wave (mean over waves and calls): {m['blocks']:.1f} blocks, {m['tiles']:.1f} tiles "
          f"({m['ncc']:.1f} TILE_C), {tot / 1e3:.1f} kcyc from the first block's top to the last block's bottom")
    rows = [("B_0 wait (loop top -> after B_0)", m["b0"]), ("INIT (after B_0 -> INIT end)", m["init"]),
            ("TILE_C loop", m["cloop"]), ("  of which TILE_C regions", m["creg"]),
            ("  of which shell between regions", m["cloop"] - m["creg"]), ("TILE_D tiles", m["diag"]),
            ("  of which TILE_D regions", m["dreg"]), ("epilogue (after the last tile -> loop bottom)", m["epi"]),
            ("  of which FINAL regions", m["fin"])]  # fmt: skip
    for name, v in rows:
        print(f"  {name:<48} {v / 1e3:9.1f} kcyc  {100 * v / tot:5.1f}%")
    print(f"per TILE_C tile: iteration {m['cloop'] / m['ncc']:.0f} cyc = region {m['creg'] / m['ncc']:.0f} + "
          f"shell {(m['cloop'] - m['creg']) / m['ncc']:.0f}")
    for k in range(4):
        if m[f"dn{k}"]:
            print(f"per TILE_D{k} tile: {m[f'd{k}'] / m[f'dn{k}']:.0f} cyc ({m[f'dn{k}']:.1f} per wave)")
    print(f"per block: B_0 {m['b0'] / m['blocks']:.0f}, INIT {m['init'] / m['blocks']:.0f}, epilogue "
          f"{m['epi'] / m['blocks']:.0f} (FINAL {m['fin'] / m['blocks']:.0f}) cyc")
    # per workgroup (wave 0): start / end on the realtime counter, span in shader cycles -> effective clock
    wg = []
    for ci, dec in enumerate(calls_dec):
        t0 = min(r[0][13] for _, _, r in dec if r)
        for li, w, recs in dec:
            if w or not recs:
                continue
            st, en = d32(recs[0][13], t0), d32(recs[-1][14], t0)
            cyc = sum(d32(r[11], r[0]) for r in recs)
            wg.append((ci, li, st, en, cyc))
    if wg:
        sts = [x[2] for x in wg]
        durs = [x[3] - x[2] for x in wg]
        cyc = [x[4] for x in wg]
        mhz = [c / (d * 10.0) for c, d in zip(cyc, durs) if d]  # cycles per 10 ns tick -> MHz = c / (d * 0.01 us)
        print(f"workgroup starts (ticks after the first): median {statistics.median(sts):.0f}, max {max(sts)}")
        print(f"workgroup durations: min {min(durs)} median {statistics.median(durs):.0f} max {max(durs)} ticks; "
              f"span cycles: min {min(cyc)} median {statistics.median(cyc):.0f} max {max(cyc)}")
        print(f"effective clock per workgroup (span cycles / duration): min {min(mhz):.0f} median "
              f"{statistics.median(mhz):.0f} max {max(mhz):.0f} MHz")
        q = sorted(range(len(wg)), key=lambda j: durs[j])
        for tag, js in (("fastest 5", q[:5]), ("slowest 5", q[-5:])):
            print(f"  {tag}: " + "; ".join(f"call {wg[j][0]} li {wg[j][1]} (xcd {wg[j][1] % 8}) start {wg[j][2]} dur "
                                          f"{durs[j]} cyc {cyc[j]}" for j in js))
        STATE["wg"] = wg
    if ends:
        by_x = {}
        for li, st, en in ends:
            by_x.setdefault(li % 8, []).append(en)
        allend = [e for _, _, e in ends]
        span = max(allend)
        print(f"workgroup end times (realtime, 10 ns ticks from the call's first start): max {span}, "
              f"mean {statistics.mean(allend):.0f}; busy fraction {statistics.mean(allend) / span:.3f}")
        for x in sorted(by_x):
            v = by_x[x]
            print(f"  XCD {x}: mean end {statistics.mean(v):.0f}  min {min(v)}  max {max(v)}  ({len(v)} WG-calls)")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ladder")
    ap.add_argument("rung")
    ap.add_argument("--shape", nargs="*")
    ap.add_argument("--warm-s", type=float, default=2.0)
    ap.add_argument("--calls", type=int, default=5)
    ap.add_argument("--flush-mb", type=int, default=512)
    ap.add_argument("--data", choices=("randn", "zeros"), default="randn")
    ap.add_argument("--json")
    a = ap.parse_args()
    spec = runner.load_ladder(a.ladder)
    sh = parse_shape(a.shape, spec["problem"])
    arm = runner.load(a.ladder, a.rung)
    mod = arm.module
    dev = torch.device("cuda")
    inp = R.make_inputs(sh, dev, seed=0, data=a.data)
    flush = torch.empty(a.flush_mb * (1 << 20) // 4, dtype=torch.float32, device=dev)
    t_end = time.time() + a.warm_s
    while time.time() < t_end:
        flush.fill_(0)
        arm.fwd(inp["q"], inp["k"], inp["v"], causal=sh.causal)
        torch.cuda.synchronize()
    decs = []
    nwg = None
    for _ in range(a.calls):
        flush.fill_(0)
        arm.fwd(inp["q"], inp["k"], inp["v"], causal=sh.causal)
        torch.cuda.synchronize()
        buf = mod.LAST_PROBE
        nwg = buf.numel() // (4 * PR_MAXB * WORDS)
        decs.append(decode(buf, nwg))
    print(f"{a.rung} {sh} {a.data}: {nwg} workgroups, {a.calls} calls decoded", flush=True)
    m = report(decs, nwg)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"rung": a.rung, "shape": repr(sh), "data": a.data, "mean": m,
                       "wg": STATE.get("wg")}, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
