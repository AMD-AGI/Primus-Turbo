"""Config sweep of the MLA backward with one process per (config, round), rotated across rounds.

Later dispatches in a process run with the scratch of the largest private segment dispatched
before them, and the dkdv kernel's spills run 5-8 % slower under a 16 or 32 KB per-wave scratch
(round 8), so configs timed inside one process are not independent. Here each config runs alone
in a fresh process. Usage: python sweep_proc.py <configs.py> [rounds] where configs.py defines
CONFIGS = [("label", {overrides}), ...]; the first entry is the reference. An override key
"__pkg__" sets that config's MLA_PKG_ROOT (package root holding flydsl_mla/). Shapes: env
SWEEP_SHAPES (default "2x4096,1x16384"). SNR is checked in the first round only."""

import json
import math
import os
import runpy
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
SHAPES = [tuple(int(x) for x in s.split("x")) for s in os.environ.get("SWEEP_SHAPES", "2x4096,1x16384").split(",")]


def child(cfg_path, idx, check):
    import quick_test_bench as qb
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    label, over = runpy.run_path(cfg_path)["CONFIGS"][idx]
    over = {k: v for k, v in over.items() if k != "__pkg__"}
    res = {"label": label, "ms": {}, "snr": [], "ok": True}

    def make(b, s):
        g = torch.Generator(device="cuda").manual_seed(0)
        q, k = (torch.randn((s, b, 128, 192), device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(2))
        v = torch.randn((s, b, 128, 128), device="cuda", dtype=torch.bfloat16, generator=g)
        do = torch.randn((s, b, 128, 128), device="cuda", dtype=torch.bfloat16, generator=g)
        o, lse = qb.mla_fwd_sbhd(q, k, v, causal=True, return_lse=True)
        return q, k, v, do, o, lse

    try:
        for n, sh in enumerate(SHAPES):
            q, k, v, do, o, lse = make(*sh)
            if check and n == 0:
                refs = {(0, h): qb.ref_head(q[:, 0, h], k[:, 0, h], v[:, 0, h], do[:, 0, h]) for h in (0, 127)}
                gr = qb.mla_bwd_sbhd(do, q, k, v, o, lse, **over)
                torch.cuda.synchronize()
                res["snr"] = [float(x) for x in qb.grads_snr(gr, refs, list(refs))]
                res["ok"] = all(bool(torch.isfinite(t).all()) for t in gr) and min(res["snr"]) > 40
                del gr, refs
            sampler = qb.timed(lambda: qb.mla_bwd_sbhd(do, q, k, v, o, lse, **over), 3, 1.5)
            res["ms"]["%dx%d" % sh] = statistics.median(sampler() for _ in range(3))
            del q, k, v, do, o, lse
            torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        res.update(ok=False, err=f"{type(e).__name__}: {str(e)[:200]}")
    print("RESULT " + json.dumps(res), flush=True)


def parent(cfg_path, rounds):
    configs = runpy.run_path(cfg_path)["CONFIGS"]
    idxs = list(range(len(configs)))
    ms = {i: {"%dx%d" % sh: [] for sh in SHAPES} for i in idxs}
    snr, ok = {}, {i: True for i in idxs}
    for r in range(rounds):
        for i in idxs[r % len(idxs) :] + idxs[: r % len(idxs)]:
            if not ok[i]:
                continue
            env = dict(os.environ)
            if "__pkg__" in configs[i][1]:
                env["MLA_PKG_ROOT"] = configs[i][1]["__pkg__"]
            p = subprocess.run([sys.executable, __file__, "--child", cfg_path, str(i), str(int(r == 0))],
                               capture_output=True, text=True, env=env)
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
            res = json.loads(lines[-1][7:]) if lines else {"ok": False, "err": p.stderr[-300:], "ms": {}, "snr": []}
            if r == 0:
                snr[i], ok[i] = res["snr"], res["ok"]
            if not res["ok"]:
                ok[i] = False
                print(f"{configs[i][0]}: FAIL {res.get('err', 'snr')} snr={res['snr']}", flush=True)
                continue
            for sh, m in res["ms"].items():
                ms[i][sh].append(m)
            print(f"round {r} {configs[i][0]}: " + " ".join(f"{sh} {m:.3f}" for sh, m in res["ms"].items()), flush=True)

    keys = ["%dx%d" % sh for sh in SHAPES]
    print(f"\n{'config':<26}" + "".join(f"{'B%sS%s ms' % tuple(k.split('x')):>14}{'d%':>8}" for k in keys)
          + f"{'gm d%':>8}  snr dq/dk/dv")
    for i in idxs:
        if not ok[i] or not ok[0]:
            continue
        cells, rel = "", []
        for k in keys:
            m, m0 = statistics.median(ms[i][k]), statistics.median(ms[0][k])
            rel.append(m0 / m)
            cells += f"{m:>14.3f}{100 * (m0 / m - 1):>+8.2f}"
        gm = math.exp(sum(math.log(x) for x in rel) / len(rel))
        print(f"{configs[i][0]:<26}{cells}{100 * (gm - 1):>+8.2f}  {'/'.join('%.2f' % x for x in snr[i])}")


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        child(sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1")
    else:
        parent(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
