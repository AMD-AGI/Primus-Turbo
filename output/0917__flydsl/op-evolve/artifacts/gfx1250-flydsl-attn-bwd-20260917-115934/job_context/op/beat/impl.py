"""op.target.beat: aiter's prebuilt gfx1250 ASM backward, behind op.reference.api.

This is the live anchor. It is NOT a precision reference -- nothing in this job is ever
aligned against it; op/eager/ is the only reference.

`op.target.beat` names it exactly: "aiter prebuilt gfx1250 ASM backward via
tools/gfx1250/asm_bwd_launcher.py, dkdv_heads=q plus the host reduction". Both halves of
that are here and both are inside the timed region, because both are what it costs to get
a correct dk/dv out of this kernel:

  * dkdv_heads="q": the grid is (kv_tiles, nhead_q, batch), so under GQA `ratio`
    workgroups would write the same dk/dv tile unsynchronised. Giving each q head its own
    slice is the documented workaround; measured at ratio=4 without it, dk/dv come back at
    about -0.3 dB.
  * the host reduction that sums those `ratio` slices back down to heads_kv.
  * dq_acc.zero_(): 514 buffer_atomic_add_f32 accumulate into it, so it must be zeroed
    every call. The scratch block is allocated once and reused; the zeroing is not
    optional and stays in the timed region.

Loaded BY FILE PATH, never as `primus_turbo.pytorch...`: importing it through the package
pulls `primus_turbo.pytorch.__init__`, whose FlyDSL tree imports `flydsl.expr.buffer_ops`,
removed in 0.3.2. That would take down the whole process.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch

TURBO = Path("/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo")
KERNARGS = TURBO / "primus_turbo" / "pytorch" / "kernels" / "attention" / "_asm_bwd_kernargs.py"

_mod = None
_hip = None
_scratch: dict = {}


def _load():
    global _mod, _hip
    if _mod is None:
        if not KERNARGS.is_file():
            raise FileNotFoundError(
                f"the beat target's launch machinery is missing: {KERNARGS}")
        spec = importlib.util.spec_from_file_location("_asm_bwd_kernargs", KERNARGS)
        _mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mod)
        for stem in ("bwd_hd128_odo_bf16", "bwd_hd128_bf16_causal_br_a32_pssk",
                     "bwd_hd128_dq_convert_bf16"):
            p = _mod.ASM_DIR / f"{stem}.co"
            if not p.is_file():
                raise FileNotFoundError(f"missing prebuilt ASM object {p}")
        _hip = _mod.HipModule()
    return _mod, _hip


def _get_scratch(b, sq, skv, hq, d, device, dtype):
    """Allocate the anchor's scratch once. Allocation is not part of what is measured;
    zeroing dq_acc is, and asm_backward does it on every call."""
    key = (b, sq, skv, hq, d, str(device), str(dtype))
    s = _scratch.get(key)
    if s is None:
        s = {
            "dq_acc": torch.empty((b, hq, sq, d), device=device, dtype=torch.float32),
            "dk": torch.empty((b, skv, hq, d), device=device, dtype=dtype),
            "dv": torch.empty((b, skv, hq, d), device=device, dtype=dtype),
        }
        _scratch[key] = s
    return s


def asm_attn_bwd(do, q, k, v, o, lse, softmax_scale=None, causal=True):
    """Same signature as op.reference.api. Returns (dq, dk, dv) in q/k/v's dtype."""
    mod, hip = _load()
    b, sq, hq, d = q.shape
    skv, hkv = k.shape[1], k.shape[2]
    g = hq // hkv
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    scratch = _get_scratch(b, sq, skv, hq, d, q.device, q.dtype)
    dq, dk_q, dv_q = mod.asm_backward(
        q, k, v, o, do, lse, softmax_scale=float(softmax_scale), hip=hip,
        dkdv_heads="q", causal=bool(causal), scratch=scratch)
    # the host reduction op.target.beat names: heads_q slices -> heads_kv
    dk = dk_q.view(b, skv, hkv, g, d).sum(dim=3).to(k.dtype)
    dv = dv_q.view(b, skv, hkv, g, d).sum(dim=3).to(v.dtype)
    return dq, dk, dv


attn_bwd = asm_attn_bwd   # uniform name every loader uses
