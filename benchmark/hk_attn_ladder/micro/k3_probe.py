"""Per-phase cycles of the 2-wave dK/dV full loop: a rung built with -DHKL_PROBE=1 (patches/a5_probe.patch) sums
s_get_shader_cycles deltas per phase over the full iterations of workgroup (0, 0, 0) (SALU only inside the loop, the
production asm patch regenerated for its .s), and writes them after the loop to the 64 KB guard in front of the dS
layout: wave w, u32 [w * 16, +6) phase sums, [w * 16 + 6] the full-iteration count.

    python3 -m micro.k3_probe mla/ladder_bwd.json r8a_pr [--calls 30]

Phases of one iteration: readback (hh0 of the next stage, after the ring barrier) + loop back; S/dP + softmax
(sdp3); dO^T tr16 burst issue; dS^T export (DPP regroup + stores); dV run with the Q^T tr16 interleave; dK kh0 +
tensorcnt + split barrier + dK kh1/kh2 window + wait.
"""
import argparse
import sys

import torch

from hkl import ladder as L
from hkl import ref as R
from hkl import runner

PHASES = ("readback + loop", "S/dP + softmax", "dO^T burst issue", "dS^T export", "dV + Q^T tr16",
          "dK + ring barrier")
# --fine (patches/a7_probe8.patch): S/dP + softmax split at the end of hh0's and of hh1's three (hh, kh) pairs
PHASES8 = ("readback + loop", "pack (P / dS)", "dO^T burst issue", "dS^T export", "dV + Q^T tr16",
           "dK + ring barrier", "top + hh0 S/dP + softmax", "hh1 S/dP + softmax")


def _stats(v):
    v = sorted(v)
    return f"mean {sum(v) / len(v):7.0f}  median {v[len(v) // 2]:6d}  p10 {v[len(v) // 10]:6d}  p90 {v[len(v) * 9 // 10]:6d}"


def split_report(g, nwg, waves=2):
    """HKGEN_PROBE=3: workgroups li % 64 < 8 (sample j = li / 64 * 8 + li % 8): the prologue (from INIT's start), the
    loop and the epilogue (from the LOOP marker's end), per wave and per dispatch slot li % 8 (XCD)."""
    for w in range(waves):
        rows = []
        for li in range(nwg):
            if li % 64 >= 8:
                continue
            j = li // 64 * 8 + li % 8
            x = [g[1024 + 4 * waves * j + 4 * w + i] & 0xFFFFFFFF for i in range(4)]
            if x[3]:
                rows.append(dict(li=li, init=x[0] >> 16, tile0=x[0] & 0xFFFF, pro=x[1] >> 16, entry=x[1] & 0xFFFF,
                                 ewait=x[2] >> 16, epi=x[2] & 0xFFFF, loop=x[3]))
        if not rows:
            print(f"wave {w}: nothing recorded")
            continue
        print(f"wave {w}: {len(rows)} workgroups (li % 64 < 8), cycles")
        for lab, k in (("entry -> INIT start", "entry"), ("INIT issued", "init"), ("tile 0 landed (own half)", "tile0"),
                       ("prologue (+ barrier, LOOP inputs)", "pro"), ("epilogue: tensorcnt 0 + barrier", "ewait"),
                       ("epilogue (FINAL issued)", "epi")):
            print(f"   {lab:34s} {_stats([r[k] for r in rows])}")
        sl = [r for r in rows if r["li"] >= 512]
        print("   per li % 8 (XCD), workgroups li >= 512: mean entry / prologue / loop / epilogue (cycles)")
        for k in range(8):
            v = [r for r in sl if r["li"] % 8 == k]
            if v:
                m = {f: sum(r[f] for r in v) / len(v) for f in ("entry", "pro", "loop", "epi")}
                print(f"      {k}: n {len(v):3d}  {m['entry']:6.0f} {m['pro']:7.0f} {m['loop']:9.0f} {m['epi']:7.0f}")


def rt_report(g, nwg, slots, grid=None):
    """HKGEN_PROBE=4: every workgroup's [start, end) (10-ns ticks): the kernel's occupancy over time."""
    rec = [(li, (x >> 14) & 0x3FFFF, x & 0x3FFF) for li, x in ((li, g[1024 + li] & 0xFFFFFFFF) for li in range(nwg)) if x]
    if not rec:
        print("nothing recorded")
        return
    m = 1 << 18
    st = sorted(r[1] for r in rec)
    gaps = [((st[(i + 1) % len(st)] - st[i]) % m, i) for i in range(len(st))]
    _, i0 = max(gaps)
    s0 = st[(i0 + 1) % len(st)]  # the window starts after the largest circular gap
    ev = [(li, (s - s0) % m, d) for li, s, d in rec]
    t_end = max(t + d for _, t, d in ev)
    busy = sum(d for _, _, d in ev)
    sat = sum(1 for _, _, d in ev if d == 0x3FFF)
    print(f"{len(ev)} workgroups; first start .. last end {t_end / 100:.1f} us; sum of spans {busy / 100:.0f} us "
          f"= {busy / (slots * t_end) * 100:.1f}% of {slots} slots x span; durations saturated: {sat}")
    nb = 40
    w = t_end / nb
    occ = [0.0] * nb
    for _, t, d in ev:
        a, b = t, t + d
        for k in range(int(a // w), min(nb, int(b // w) + 1)):
            lo, hi = max(a, k * w), min(b, (k + 1) * w)
            if hi > lo:
                occ[k] += (hi - lo) / w
    print("resident workgroups per 1/40 of the span: " + " ".join(f"{o:.0f}" for o in occ))
    mid = occ[nb // 8: nb * 7 // 8]
    print(f"   middle 3/4 mean {sum(mid) / len(mid):.1f}; first bin {occ[0]:.0f}, last 3 bins "
          + ", ".join(f"{o:.0f}" for o in occ[-3:]))
    fin = sorted(t + d for _, t, d in ev)
    for q in (0.9, 0.99):
        print(f"   {q * 100:.0f}% of the workgroups ended by {fin[int(len(fin) * q) - 1] / 100:.1f} us")
    dur = sorted(d for _, _, d in ev)
    print(f"   workgroup span (us): median {dur[len(dur) // 2] / 100:.1f}, p10 {dur[len(dur) // 10] / 100:.1f}, "
          f"p90 {dur[len(dur) * 9 // 10] / 100:.1f}, max {dur[-1] / 100:.1f}")
    # per XCD (dispatch round-robin: li % 8): workgroups, sum of spans, last end, mean span
    print("   per li % 8 (XCD): n, sum of spans (us), mean span (us), last end (us), busy share of 64 slots x span")
    for k in range(8):
        v = [(t, d) for li, t, d in ev if li % 8 == k]
        if v:
            sb = sum(d for _, d in v)
            print(f"      {k}: {len(v):5d} {sb / 100:9.0f} {sb / len(v) / 100:7.2f} {max(t + d for t, d in v) / 100:8.1f} "
                  f"{sb / (slots / 8 * max(t + d for t, d in v)) * 100:6.1f}%")
    if grid:  # (x, y, z) of the dkdv grid: linear index li = (z * gy + y) * gx + x
        gx, gy = grid[0], grid[1]
        late = sorted(ev, key=lambda e: e[1] + e[2])[-12:]
        print("   the 12 last to end: li (x, y, z) start..end us")
        for li, t, d in late:
            print(f"      {li:6d} ({li % gx:3d}, {li // gx % gy:2d}, {li // (gx * gy)}) {t / 100:7.1f} .. {(t + d) / 100:7.1f}")
        # per (y, z): mean start / span, to see the dispatch order and the batch boundary
        byyz = {}
        for li, t, d in ev:
            byyz.setdefault((li // (gx * gy), li // gx % gy), []).append((t, d))
        print("   per (z, y): mean start / mean span (us), every 6th y")
        for (z, y), v in sorted(byyz.items()):
            if y % 6 == 0 or y == gy - 1:
                print(f"      z {z} y {y:2d}: start {sum(a for a, _ in v) / len(v) / 100:7.1f}  span "
                      f"{sum(b for _, b in v) / len(v) / 100:6.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ladder")
    ap.add_argument("rung")
    ap.add_argument("--calls", type=int, default=30)
    ap.add_argument("--fine", action="store_true", help="8-phase probe (patches/a7_probe8.patch)")
    ap.add_argument("--shape", nargs="*", help="overrides of the ladder's problem, e.g. b=1 sq=384 skv=384 hq=16 hkv=16")
    ap.add_argument("--abc", action="store_true", help="HKGEN skeleton (abcsk_*): slot 0 = the whole loop, n = steps")
    ap.add_argument("--allwg", type=int, default=0, help="HKGEN probe shells: fit every workgroup's whole span "
                    "(word 1024 + linear index: steps << 24 | cycles) over this many workgroups")
    ap.add_argument("--proepi", action="store_true", help="with --allwg, an HKGEN_PROBE=2 build: the words are "
                    "prologue << 16 | epilogue cycles")
    ap.add_argument("--split", action="store_true", help="with --allwg, an HKGEN_PROBE=3 build: workgroups li %% 8 == 0, "
                    "wave w words li + 4 w + 0..3 = (INIT issued | tile 0 landed), (prologue | entry -> INIT), "
                    "(tensorcnt 0 | + barrier), (epilogue | steps)")
    ap.add_argument("--rt", action="store_true", help="with --allwg, an HKGEN_PROBE=4 build: every workgroup's "
                    "(realtime start mod 2^18) << 14 | min(end - start, 0x3FFF) in 10-ns ticks")
    ap.add_argument("--slots", type=int, default=512, help="--rt: workgroups resident at once (occupancy)")
    ap.add_argument("--dq", action="store_true", help="--rt: the words are k_dqds_c's (-DDQ_PROBE=1), its grid")
    ap.add_argument("--waves", type=int, default=2, help="--split: waves per workgroup (4: -DHKGEN_W4)")
    a = ap.parse_args()
    spec = runner.load_ladder(a.ladder)
    sh = L.parse_shape(a.shape, spec["problem"])
    mod = runner.load(a.ladder, a.rung).module
    dev = torch.device("cuda")
    inp = R.make_inputs(sh, dev, seed=0)
    o, lse = R.ref_fwd(sh, inp["q"], inp["k"], inp["v"])
    o = o.to(torch.bfloat16)
    sbhd = mod.is_sbhd(inp["q"], lse)
    f = [mod._fold(t, sbhd) for t in (inp["do"], inp["q"], inp["k"], inp["v"], o)]
    lse_k = (lse.reshape(1, -1, lse.shape[-1]) if sbhd else lse).contiguous().float()
    stream = torch.cuda.current_stream()
    table = mod.kernels()
    plan = mod._plan(*f, lse_k, sh.d ** -0.5, True, stream)
    for _ in range(a.calls):
        for _, fn, args in plan["launches"]:
            if isinstance(fn, str):
                grid, block, values = args
                table[fn].call(grid, block, values, stream)
            elif fn in getattr(mod, "HOST_STEPS", ()):
                fn(*args)
            else:
                mod._fly._launch(fn, args)
    torch.cuda.synchronize()
    g = plan["ds_full"][: mod.DS_GUARD_B // 2].view(torch.int32).cpu().tolist()
    if a.allwg and (a.split or a.rt):  # modes 3 / 4 leave the per-phase words unwritten
        split_report(g, a.allwg, a.waves) if a.split else rt_report(g, a.allwg, a.slots, plan["grids"]["dq" if a.dq else "dkdv"])
        return 0
    if a.allwg and a.proepi:
        rec = [x & 0xFFFFFFFF for x in g[1024: 1024 + a.allwg]]
        pe = sorted((x >> 16, x & 0xFFFF) for x in rec if x)
        if pe:
            pro = sorted(p[0] for p in pe); epi = sorted(p[1] for p in pe)
            print(f"all workgroups ({len(pe)}): prologue mean {sum(pro) / len(pro):.0f} median {pro[len(pro) // 2]} "
                  f"p90 {pro[len(pro) * 9 // 10]}; epilogue mean {sum(epi) / len(epi):.0f} median {epi[len(epi) // 2]} "
                  f"p90 {epi[len(epi) * 9 // 10]} cycles")
    elif a.allwg:
        rec = [x & 0xFFFFFFFF for x in g[1024: 1024 + a.allwg]]
        pts = [(x >> 24, x & 0xFFFFFF) for x in rec if x]
        if pts:
            n_ = len(pts)
            sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
            sxx = sum(p[0] * p[0] for p in pts); sxy = sum(p[0] * p[1] for p in pts)
            b_ = (n_ * sxy - sx * sy) / max(n_ * sxx - sx * sx, 1)
            a_ = (sy - b_ * sx) / n_
            tot = sum(p[1] for p in pts)
            print(f"all workgroups: {n_} recorded, fit span = {a_:.0f} + {b_:.1f} * steps; "
                  f"sum of spans {tot / 1e6:.2f} Mcycles, fixed part {n_ * a_ / max(tot, 1) * 100:.1f}%")
            byn = {}
            for st, cy in pts:
                byn.setdefault(st, []).append(cy)
            for st in sorted(byn)[:: max(1, len(byn) // 8)]:
                v = sorted(byn[st])
                print(f"   steps {st:4d}: n {len(v):4d}  median {v[len(v) // 2]:7d}  per step {v[len(v) // 2] / max(st, 1):6.0f}")
    names, nph = (PHASES8, 8) if a.fine else (PHASES, 6)
    for w in range(a.waves):
        sums = [x & 0xFFFFFFFF for x in g[w * 16: w * 16 + nph]]
        n = g[w * 16 + nph]
        tot = (sums[0] if a.abc else sum(sums)) / max(n, 1)
        print(f"wave {w}: {n} {'steps' if a.abc else 'full iterations'}, {tot:.0f} cycles per iteration")
        if a.abc:
            for i, lab in enumerate(("start -> signal (B + C)", "signal -> wait (barrier)", "wait -> C end",
                                     "C end -> end (A)")):
                print(f"   {lab:26s} {sums[1 + i] / max(n, 1):7.1f}")
            pro, epi = g[w * 16 + 7] & 0xFFFFFFFF, g[w * 16 + 8] & 0xFFFFFFFF
            if pro or epi:  # HKGEN shells since r10j_pr: prologue (INIT .. LOOP marker) / epilogue (LOOP .. FINAL)
                print(f"   prologue {pro} cycles, epilogue {epi} cycles, loop {sums[0] - pro - epi} cycles "
                      f"({(sums[0] - pro - epi) / max(n, 1):.0f} per step incl. headers)")
            continue
        order = (6, 7, 1, 2, 3, 4, 5, 0) if a.fine else range(nph)
        for i in order:
            name, s = names[i], sums[i]
            print(f"   {name:20s} {s / max(n, 1):7.1f}  ({100 * s / max(sum(sums), 1):4.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
