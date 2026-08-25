#!/usr/bin/env python3
# Prototype: the aiter/CK deterministic dQ shape -- a SMALL number of dQ accumulator slots
# walked in groups -- priced against our one-slot-per-band scheme.
#
# CK's deterministic batch path is a persistent kernel: a fixed worker owns a fixed slice of
# dQ, sweeps its own bands into it, and a convert pass folds the few slices at the end
# (fmha_bwd_kernel.hpp:129 GetDqAccSplits; nsplits is 2 for our l70b_b2 shape against our 64).
# We already have that memory shape, reached by host-side sequencing rather than persistence:
# `_band_span_for` walks the band axis in groups and `_reduce_dq_partials(band=...)` folds each
# group into an fp32 carry. It exists for the 16 GiB workspace cap and has never been chosen
# for speed. Lowering the cap is enough to select it, so this measures the scheme with no
# kernel change at all.
#
#   python _probe_bandgroup.py [--cell=<tag>]
#
# Reports, per span: the ms, the speedup against the deployed whole-axis path, and whether dQ
# is still bitwise what the deployed path produces (it must be -- both fold ascending in fp32).
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import primus_turbo.flydsl.attention.flash_attn_bwd as bwd
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_backward_impl,
    flash_attn_sbhd_flydsl_forward_impl,
)

DEV, DT = "cuda", torch.bfloat16
WARMS, REPS = 5, 40

CELLS = {
    "d128_70b_b2": (2, 64, 8, 8192, 128),
    "d128_8b_b2": (2, 32, 8, 8192, 128),
    "d64_gptoss_b4": (4, 64, 8, 8192, 64),
    "d64_b2": (2, 64, 8, 8192, 64),
    "d128_b4": (4, 64, 8, 8192, 128),
    "d128_16k_b1": (1, 64, 8, 16384, 128),
}


def _sbhd(B, S, H, D):
    return torch.randn(S, B, H, D, device=DEV, dtype=DT)


def _make(cell):
    B, Hq, Hkv, S, D = cell
    torch.manual_seed(11)
    q, k, v = _sbhd(B, S, Hq, D), _sbhd(B, S, Hkv, D), _sbhd(B, S, Hkv, D)
    do = torch.randn_like(q)
    o, lse = flash_attn_sbhd_flydsl_forward_impl(q, k, v, causal=True, window_size=(-1, -1), return_lse=True)
    lse_h = lse.view(B, S, Hq).permute(0, 2, 1)
    return lambda: flash_attn_sbhd_flydsl_backward_impl(
        do, q, k, v, o, lse_h, causal=True, window_size=(-1, -1)
    )


def _time(fn):
    for _ in range(WARMS):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(REPS):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))
    return best


def _reset_caches():
    """Plan and module caches key on shape, not on the budget, so they must go between arms."""
    for name in ("_BWD_CACHE", "_DQRED_CACHE"):
        c = getattr(bwd, name, None)
        if isinstance(c, dict):
            c.clear()
    torch.cuda.empty_cache()


def run(tag):
    B, Hq, Hkv, S, D = CELLS[tag]
    n_bands = S // 128  # BLOCK_KV=128 at D=128; D=64 may fuse to 256, reported as measured
    band_bytes = B * S * Hq * D * 2
    out = {"cell": tag, "n_bands_at_bkv128": n_bands, "band_MB": round(band_bytes / 2**20, 1)}

    # One arm per process: the whole-axis workspace for this shape is bands*|dQ| ~ 17 GB, so
    # two arms cannot be resident together.
    span = int(os.environ.get("SPAN", "0"))
    _reset_caches()
    # span < 0 lifts the cap out of the way entirely: the whole-axis arm for shapes
    # whose workspace exceeds the deployed 16 GiB and are therefore forced into groups.
    # span 0 leaves the module's own budget rule in place (the deployed behaviour); a
    # positive span pins the cap to that many bands; a negative one lifts it out of the way.
    bwd._WSQ_BUDGET_BYTES = None if span == 0 else (1 << 42) if span < 0 else (span * band_bytes)
    fn = _make((B, Hq, Hkv, S, D))
    got = [t.clone() for t in fn()[:3]]
    out["span"] = "deployed" if span == 0 else ("uncapped" if span < 0 else span)
    out["ms"] = round(_time(fn), 4)
    ref = os.environ.get("REF")
    if ref and os.path.exists(ref):
        out["bitwise_eq"] = all(
            torch.equal(a.cpu(), b) for a, b in zip(got, torch.load(ref))
        )
    elif ref:
        torch.save([t.cpu() for t in got], ref)
        out["bitwise_eq"] = "saved"
    return out


if __name__ == "__main__":
    want = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--cell=")] or list(CELLS)
    for tag in want:
        print(json.dumps(run(tag)))
