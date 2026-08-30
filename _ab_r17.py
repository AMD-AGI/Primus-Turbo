#!/usr/bin/env python3
"""Is r17 actually faster than best, or was +0.20% just noise?

The campaign's own verdict was "no new best, gain=+0.20%" against a calibration spread of
1.15-1.42%. Four separate rounds (r11 +0.12, r12 +0.30, r15 +0.25, r17 +0.20) all landed in that
same sub-noise band, which is exactly what you would see if the tree really had improved by a few
tenths and the 3-rep harness could not resolve it. This settles it with more samples and a
palindromic block order.

  arm A = 49ee57a0  (the kept r4 commit; best = 1.07433)
  arm B = r17       (the working copy the campaign was iterating on)

Block order ABBA ABBA cancels any monotone drift (thermal, neighbour load). Each block clears the
FlyDSL cache so the kernel is genuinely rebuilt -- otherwise a stale compiled module is timed --
and each cell runs in its own process at min-of-40 after 5 warms, identical to the scorer.

usage: _ab_r17.py [blocks]      (default 2 => ABBA ABBA => 4 samples per arm per cell)
"""
import json, os, shutil, statistics, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TGT = os.path.join(HERE, "primus_turbo/flydsl/attention/flash_attn_bwd.py")
ARMS = {"A": os.path.join(HERE, "_arm_base.py"), "B": os.path.join(HERE, "_arm_r17.py")}
CELLS = ["d64_gptoss_b4", "d64_b2"]
FLOPS = {"d64_gptoss_b4": 5.4976e12, "d64_b2": 2.7488e12}
BASE_MS = {"d64_gptoss_b4": 5.5707, "d64_b2": 2.8557}


def run_cell(cell):
    r = subprocess.run([sys.executable, os.path.join(HERE, "_bench_campaign_d64.py"), f"--cell={cell}"],
                       capture_output=True, text=True, timeout=1800, cwd=HERE)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    if not line:
        raise AssertionError(f"{cell}: no result\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return json.loads(line[-1])[f"{cell}_bwd"]


def main():
    nblk = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    order = ("A", "B", "B", "A") * nblk
    got = {a: {c: [] for c in CELLS} for a in ARMS}
    shutil.copy2(TGT, TGT + ".orig")
    try:
        for i, arm in enumerate(order):
            shutil.copy2(ARMS[arm], TGT)
            subprocess.run("rm -rf /root/.flydsl/cache", shell=True)
            for c in CELLS:
                ms = run_cell(c)
                got[arm][c].append(ms)
                print(f"  blk{i} arm{arm} {c:16s} {ms:.4f} ms  "
                      f"{FLOPS[c] / (ms * 1e-3) / 1e12:.1f} TF/s", flush=True)
    finally:
        shutil.copy2(TGT + ".orig", TGT)
        os.remove(TGT + ".orig")

    print("\n=== VERDICT (median of each arm, own process, min-of-40, ABBA-interleaved) ===")
    out = {}
    for c in CELLS:
        a, b = statistics.median(got["A"][c]), statistics.median(got["B"][c])
        sa = (max(got["A"][c]) - min(got["A"][c])) / a * 100
        sb = (max(got["B"][c]) - min(got["B"][c])) / b * 100
        print(f"{c:16s} A(base) {a:.4f} ms {FLOPS[c]/(a*1e-3)/1e12:7.1f} TF/s  spread {sa:.2f}%")
        print(f"{'':16s} B(r17)  {b:.4f} ms {FLOPS[c]/(b*1e-3)/1e12:7.1f} TF/s  spread {sb:.2f}%")
        print(f"{'':16s} B vs A  {(a/b - 1) * 100:+.2f}%   "
              f"| within-arm spread {max(sa, sb):.2f}%  -> "
              f"{'REAL' if abs(a/b - 1) * 100 > max(sa, sb) else 'INSIDE NOISE'}")
        out[c] = {"A_ms": a, "B_ms": b, "gain_pct": round((a / b - 1) * 100, 3),
                  "A_runs": got["A"][c], "B_runs": got["B"][c]}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
