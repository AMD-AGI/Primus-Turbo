#!/usr/bin/env python3
"""Forward config A/B without touching the kernel tree.

Mirrors the deployed `_fwd_module` config (waves_per_eu=2, stagger off, block_m=128) and
overrides one knob per arm, so the arms differ only in what is being priced.

usage: _probe_fwdcfg.py <B> <Hq> <Hkv> <S> <D> <arm> [reps] [W]
arms:  base | nomerge | bm64 | bm256 | wpe3 | wpe4 | isa
"""
import functools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import primus_turbo.pytorch.kernels.attention.attention_flydsl_impl as impl
from primus_turbo.flydsl.attention.flash_attn_fwd import build_flash_attn_dualwave_swp_module
from primus_turbo.flydsl.utils.attn_helper import DualwaveKernelContext

DEV, DT = "cuda", torch.bfloat16

ARMS = {
    "base": {},
    "nomerge": dict(gqa_merge=False),
    "bm64": dict(block_m=64),
    "bm256": dict(block_m=256),
    "wpe3": dict(waves_per_eu=3),
    "wpe4": dict(waves_per_eu=4),
    "nomask": {},
    "setprio0": dict(dualwave_swp_setprio=False),
}


def patch(arm):
    over = ARMS[arm]
    if arm == "nomask":
        # Pricing probe: the causal/pad mask never applies (WRONG O on the diagonal tile).
        DualwaveKernelContext.causal_mask_prologue_if_needed = lambda self, v_s, *a, **k: v_s
        DualwaveKernelContext.causal_mask_split_prologue_if_needed = lambda self, v_s, *a, **k: v_s
        DualwaveKernelContext.seq_pad_mask_if_needed = lambda self, v_s, *a, **k: v_s

    @functools.lru_cache(maxsize=64)
    def _fwd_module(Hq, Hkv, D, causal, cross_seqlen, emit_lse, window_left, sbhd=False, has_sink=False):
        cfg = dict(waves_per_eu=2, dualwave_swp_enable_stagger=False, block_m=128)
        cfg.update(over)
        return build_flash_attn_dualwave_swp_module(
            num_heads=Hq,
            head_dim=D,
            causal=causal,
            dtype_str="bf16",
            num_kv_heads=Hkv,
            varlen=not sbhd,
            cross_seqlen=cross_seqlen,
            emit_lse=emit_lse,
            window_left=window_left,
            sbhd=sbhd,
            has_sink=has_sink,
            **cfg,
        )

    impl._fwd_module = _fwd_module


def main():
    a = sys.argv[1:]
    B, Hq, Hkv, S, D = (int(x) for x in a[:5])
    arm = a[5]
    reps = int(a[6]) if len(a) > 6 else 40
    W = int(a[7]) if len(a) > 7 else -1
    ws = (W, 0) if W >= 0 else (-1, -1)
    patch(arm)
    sh = lambda H: torch.randn(S, B, H, D, device=DEV, dtype=DT)
    q, k, v = sh(Hq), sh(Hkv), sh(Hkv)
    f = lambda: impl.flash_attn_sbhd_flydsl_forward_impl(
        q, k, v, causal=True, window_size=ws, return_lse=True
    )
    ref = f()[0]
    for _ in range(5):
        f()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        f()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))
    d = (ref.float() - f()[0].float()).abs().max().item()
    print(json.dumps({"arm": arm, "B": B, "Hq": Hq, "S": S, "D": D, "W": W, "fwd": round(best, 4), "maxdiff": d}))


if __name__ == "__main__":
    main()
