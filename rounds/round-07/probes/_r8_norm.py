#!/usr/bin/env python3
"""Round-8 probe: rank the RMSNorm changes inside the scored proj unit's own composition.

Modes:
  corr   correctness of dual-dx and the two-stage dgamma fold
  ab     palindrome A/B of {dual_dx, dgamma fold, cache policy} in the proj carrier
  sub    subtractive pricing: what a norm that emitted quantised output could save
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _bench_gptoss_stepfuse as BE
from primus_turbo.pytorch.kernels.normalization import rmsnorm_impl as RI

DEV = "cuda"


def _sync_time(fn, warms, reps):
    for _ in range(warms):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


# ---------------------------------------------------------------- control arms
_ORIG_FINALIZE = RI._finalize_dgamma


def _finalize_old(dg_partial, gamma_dtype):
    """Round-7 behaviour: torch reduce above the cap."""
    n_parts, H = dg_partial.shape
    if n_parts > RI._FINALIZE_TRITON_MAX_PARTS:
        return dg_partial.sum(dim=0).to(gamma_dtype)
    return _ORIG_FINALIZE(dg_partial, gamma_dtype)


def set_arm(dual_dx, dg_fold, fwd_ld, fwd_st, bwd_ld, bwd_st, grid_mult=2):
    import primus_turbo.pytorch.ops.normalization as NM

    NM._DUAL_DX = dual_dx
    RI._FWD_LD_CM, RI._FWD_ST_CM = fwd_ld, fwd_st
    RI._BWD_LD_CM, RI._BWD_ST_CM = bwd_ld, bwd_st
    RI._BWD_GRID_MULT = grid_mult
    RI._finalize_dgamma = _ORIG_FINALIZE if dg_fold else _finalize_old
    RI._pick_bwd_config.cache_clear() if hasattr(RI._pick_bwd_config, "cache_clear") else None


def corr():
    torch.manual_seed(3)
    ok = True
    for B, H, dt in [(32768, 2880, torch.bfloat16), (2097152, 64, torch.bfloat16),
                     (4096, 2880, torch.float32), (111, 50, torch.float16),
                     (262144, 64, torch.bfloat16)]:
        x = torch.randn(B, H, device=DEV, dtype=dt, requires_grad=True)
        r = torch.randn(B, H, device=DEV, dtype=dt, requires_grad=True)
        g = torch.randn(H, device=DEV, dtype=dt, requires_grad=True)
        cot = torch.randn(B, H, device=DEV, dtype=dt)

        # dual-dx: x and residual grads must both equal the single-buffer answer.
        set_arm(False, False, "", "", "", "")
        y, _ = BE.rmsnorm_residual(x, r, g)
        gx0, gr0, gg0 = torch.autograd.grad(y, [x, r, g], cot)
        set_arm(True, True, "", "", "", "")
        y1, _ = BE.rmsnorm_residual(x, r, g)
        gx1, gr1, gg1 = torch.autograd.grad(y1, [x, r, g], cot)
        d = [torch.equal(y, y1), torch.equal(gx0, gx1), torch.equal(gr0, gr1),
             torch.equal(gx1, gr1), torch.equal(gg0, gg1)]
        # plain norm exercises the dgamma fold on the tall-partial widths.
        z0 = BE.rmsnorm(x, g)
        pg0, pgg0 = torch.autograd.grad(z0, [x, g], cot)
        set_arm(False, False, "", "", "", "")
        z1 = BE.rmsnorm(x, g)
        pg1, pgg1 = torch.autograd.grad(z1, [x, g], cot)
        dg_rel = ((pgg0.float() - pgg1.float()).norm() / (pgg1.float().norm() + 1e-30)).item()
        d += [torch.equal(z0, z1), torch.equal(pg0, pg1)]
        ok &= all(d)
        print(f"B={B} H={H} {str(dt).split('.')[-1]:9s} eq={d} dgamma_rel={dg_rel:.3e}", flush=True)
    print("CORR", "PASS" if ok else "FAIL", flush=True)


def ab(warms=6, reps=25):
    lv, cots = BE._proj_leaves(), BE._proj_cots()

    def step():
        for v in lv.values():
            v.grad = None
        BE._proj_unit(lv, cots)

    arms = {
        # (dual_dx, dg_fold, fwd_ld, fwd_st, bwd_ld, bwd_st)
        "g2": (True, True, "", ".cs", ".cg", ".cs", 2),
        "g1": (True, True, "", ".cs", ".cg", ".cs", 1),
        "g3": (True, True, "", ".cs", ".cg", ".cs", 3),
        "g4": (True, True, "", ".cs", ".cg", ".cs", 4),
        "g6": (True, True, "", ".cs", ".cg", ".cs", 6),
        "g8": (True, True, "", ".cs", ".cg", ".cs", 8),
        "g12": (True, True, "", ".cs", ".cg", ".cs", 12),
        "g16": (True, True, "", ".cs", ".cg", ".cs", 16),
    }
    names = list(arms)
    # Compile every arm first: a JIT gap between arms re-enters the DVFS ramp.
    for n in names:
        set_arm(*arms[n])
        step()
    torch.cuda.synchronize()

    out = {n: [] for n in names}
    for order in (names, names[::-1]):
        for n in order:
            set_arm(*arms[n])
            step()  # discard arm, post-switch
            out[n].append(_sync_time(step, warms, reps))
    base = sum(out[names[0]]) / 2
    for n in names:
        v = out[n]
        print(f"{n:10s} {v[0]:.4f} {v[1]:.4f}  mean {sum(v)/2:.4f}  ratio {sum(v)/2/base:.4f}", flush=True)


def sub(warms=6, reps=25):
    """Upper bound on 'RMSNorm emits quantised output' = delete the casts it feeds."""
    import primus_turbo.pytorch.kernels.quantization.quantization_impl as QI

    lv, cots = BE._proj_leaves(), BE._proj_cots()

    def step():
        for v in lv.values():
            v.grad = None
        BE._proj_unit(lv, cots)

    set_arm(True, True, "", "", "", "")
    full = _sync_time(step, warms, reps)

    orig = QI.quantize_fp8_tensorwise
    calls = {"n": 0, "shapes": []}

    def counting(x, *a, **kw):
        calls["n"] += 1
        calls["shapes"].append(tuple(x.shape))
        return orig(x, *a, **kw)

    QI.quantize_fp8_tensorwise = counting
    import primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl as GI

    if hasattr(GI, "quantize_fp8_tensorwise"):
        GI.quantize_fp8_tensorwise = counting
    calls["n"] = 0
    step()
    torch.cuda.synchronize()
    QI.quantize_fp8_tensorwise = orig
    if hasattr(GI, "quantize_fp8_tensorwise"):
        GI.quantize_fp8_tensorwise = orig
    print(f"full {full:.4f} ms  quantize calls/unit {calls['n']} {calls['shapes']}", flush=True)


def prof():
    from collections import defaultdict

    lv, cots = BE._proj_leaves(), BE._proj_cots()

    def step():
        for v in lv.values():
            v.grad = None
        BE._proj_unit(lv, cots)

    for _ in range(8):
        step()
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(3):
            step()
        torch.cuda.synchronize()
    agg = defaultdict(lambda: [0.0, 0])
    for e in p.key_averages():
        if e.device_time_total > 0 and e.key not in ("cudaDeviceSynchronize",):
            agg[e.key][0] += e.device_time_total / 3
            agg[e.key][1] += e.count // 3
    tot = 0.0
    for k, (t, n) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        if t < 3:
            continue
        tot += t
        print(f"{t:9.1f} us  x{n:3d}  {k[:88]}", flush=True)
    print(f"{tot:9.1f} us  TOTAL(listed)", flush=True)


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "corr"
    torch.backends.cuda.matmul.allow_tf32 = True
    {"corr": corr, "ab": ab, "sub": sub, "prof": prof}[m]()
