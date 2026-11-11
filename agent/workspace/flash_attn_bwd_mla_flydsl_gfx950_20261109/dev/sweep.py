"""One-knob-at-a-time config sweep of the MLA backward (dkdv builder overrides), rotated timing.

For every config: SNR vs the chunked fp32 reference on (b0, h0) and (b0, h127) at B2 S4K, then bwd ms
at B2 S4K and B1 S16K (CUDA events, rotated over repeats). Usage: python sweep.py <configs.py> [repeats]
where configs.py defines CONFIGS = [("label", {overrides}), ...]; the first entry is the reference."""

import math
import os
import runpy
import statistics
import sys
import traceback

sys.path.insert(0, "/perf_apps/xiaoming/Primus-Turbo/agent/workspace/flash_attn_bwd_mla_flydsl_gfx950_20261109")
import quick_test_bench as qb  # noqa: E402
import torch  # noqa: E402

CONFIGS = runpy.run_path(sys.argv[1])["CONFIGS"]
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SHAPES = [tuple(int(x) for x in s.split("x")) for s in os.environ.get("SWEEP_SHAPES", "2x4096,1x16384").split(",")]
torch.backends.cuda.matmul.allow_tf32 = False


def make(b, s, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k = (torch.randn((s, b, 128, 192), device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(2))
    v = torch.randn((s, b, 128, 128), device="cuda", dtype=torch.bfloat16, generator=g)
    do = torch.randn((s, b, 128, 128), device="cuda", dtype=torch.bfloat16, generator=g)
    o, lse = qb.mla_fwd_sbhd(q, k, v, causal=True, return_lse=True)
    return q, k, v, do, o, lse


def ev_ms(fn, iters):
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


data = {sh: make(*sh) for sh in SHAPES}
q, k, v, do, o, lse = data[SHAPES[0]]
refs = {(0, h): qb.ref_head(q[:, 0, h], k[:, 0, h], v[:, 0, h], do[:, 0, h]) for h in (0, 127)}
slices = list(refs)

ok, snrs = {}, {}
for label, over in CONFIGS:
    try:
        g = qb.mla_bwd_sbhd(do, q, k, v, o, lse, **over)
        torch.cuda.synchronize()
        snrs[label] = qb.grads_snr(g, refs, slices)
        ok[label] = all(bool(torch.isfinite(t).all()) for t in g) and min(snrs[label]) > 40
        for sh in SHAPES:
            qq, kk, vv, dd, oo, ll = data[sh]
            qb.mla_bwd_sbhd(dd, qq, kk, vv, oo, ll, **over)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        ok[label] = False
        print(f"{label}: BUILD/RUN FAIL {type(e).__name__}: {str(e)[:200]}", flush=True)
        traceback.print_exc(limit=2)
    print(f"{label}: ok={ok[label]} snr={['%.2f' % x for x in snrs.get(label, [])]}", flush=True)

live = [(l, o_) for l, o_ in CONFIGS if ok[l]]
ms = {(l, sh): [] for l, _ in live for sh in SHAPES}
for r in range(REPEATS):
    order = live[r % len(live) :] + live[: r % len(live)]
    for label, over in order:
        for sh in SHAPES:
            qq, kk, vv, dd, oo, ll = data[sh]
            fn = lambda: qb.mla_bwd_sbhd(dd, qq, kk, vv, oo, ll, **over)  # noqa: E731
            fn()
            ms[(label, sh)].append(ev_ms(fn, 10 if sh[1] <= 4096 else 3))

ref_label = CONFIGS[0][0]
print(f"\n{'config':<22}" + "".join(f"{'B%dS%d ms' % sh:>14}{'d%':>8}" for sh in SHAPES) + f"{'gm d%':>8}  snr dq/dk/dv")
for label, _ in live:
    cells, rel = "", []
    for sh in SHAPES:
        m = statistics.median(ms[(label, sh)])
        m0 = statistics.median(ms[(ref_label, sh)]) if ok[ref_label] else float("nan")
        rel.append(m0 / m)
        cells += f"{m:>14.3f}{100 * (m0 / m - 1):>+8.2f}"
    gm = math.exp(sum(math.log(x) for x in rel) / len(rel))
    print(f"{label:<22}{cells}{100 * (gm - 1):>+8.2f}  {'/'.join('%.2f' % x for x in snrs[label])}")
