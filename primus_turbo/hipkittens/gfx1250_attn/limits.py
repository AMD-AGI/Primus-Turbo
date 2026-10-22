###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shape rules of the gfx1250 HipKittens attention kernels (pure Python, no torch: the gate imports it).

Every kernel module is a subset of the FlyDSL gfx1250 head-dim-128 kernels' range (the HIPKITTENS
backend only takes calls FlyDSL could take, and runs FlyDSL for the rest), so these rules come on top
of ``primus_turbo.flydsl.attention.gfx1250.interface.unsupported_reason``:

- every module: bf16, head dim 128 for q / k / v, no empty dimension, Hq a multiple of Hkv, bottom-right
  causal only, Sq <= Skv;
- fwd_v0: G = Hq / Hkv a multiple of 4 (a wave's four q tiles are four heads of one kv head), Sq and Skv
  multiples of 64, and the int32 buffer offsets of q / o / lse / k / v below 2^31;
- fwd_pp: the fwd_v0 rules until the module's own gate says otherwise (``unsupported_reason`` of the
  module is asked as well, so these only have to be no wider than the kernel);
- bwd_v0: G == 4 (compiled into k_hk_bwd_g4), Sq a multiple of 64, Skv of 128 (one kv block of 128
  rows per workgroup) and the int32 offsets of the prep / main / dQ-convert kernels below 2^31
  (hk/notes/bwd_v0_design.md 11.3).

``MODULE_RULES`` holds the numbers; ``shape_unsupported_reason`` applies them.
"""

from typing import Optional

HEAD_DIM = 128
INT32_LIMIT = 1 << 31

# module -> rules. g_values: allowed Hq / Hkv (None: any multiple of g_multiple).
MODULE_RULES = {
    "fwd_v0": {"g_multiple": 4, "g_values": None, "sq_multiple": 64, "skv_multiple": 64},
    "fwd_pp": {"g_multiple": 4, "g_values": None, "sq_multiple": 64, "skv_multiple": 64},
    "bwd_v0": {"g_multiple": 4, "g_values": (4,), "sq_multiple": 64, "skv_multiple": 128},
}


def _fwd_extents(b, sq, skv, hq, hkv):
    """(name, bytes) of every int32-addressed extent of the forward kernels."""
    return (
        ("q / o", 2 * b * sq * hq * HEAD_DIM),
        ("lse", 4 * b * hq * sq),
        ("k / v", 2 * b * skv * hkv * HEAD_DIM),
    )


def _bwd_extents(b, sq, skv, hq, hkv):
    """bwd_v0_design.md 11.3: prep / dQ-convert voffsets, the fp32 dQ accumulator, the main kernel's
    per-group q / dO resource, its int32 kv arguments and the last kv head's base."""
    g = hq // hkv
    return (
        ("q / o / do", 2 * b * sq * hq * HEAD_DIM),
        ("dq accumulator", 4 * b * hq * sq * HEAD_DIM),
        ("q / do head group", 2 * g * sq * HEAD_DIM * 2),
        ("k / v", 2 * b * skv * hkv * HEAD_DIM),
        ("kv head base", (127 * hkv + 1) * 2 * HEAD_DIM),
    )


_EXTENTS = {"fwd_v0": _fwd_extents, "fwd_pp": _fwd_extents, "bwd_v0": _bwd_extents}


def shape_unsupported_reason(module, q_shape, k_shape, v_shape, causal) -> Optional[str]:
    """Why ``module``'s kernels cannot take this BSHD problem, or None. Shapes only (dtype, device and
    layout are the caller's); never raises for a malformed shape, it is a reason."""
    rules = MODULE_RULES.get(module)
    if rules is None:
        return f"no shape rules for kernel module {module!r} (known: {sorted(MODULE_RULES)})"
    q_shape, k_shape, v_shape = (tuple(int(x) for x in s) for s in (q_shape, k_shape, v_shape))
    if len(q_shape) != 4 or len(k_shape) != 4 or len(v_shape) != 4:
        return f"q / k / v must be 4-D [B, S, H, D], got {q_shape} / {k_shape} / {v_shape}"
    if 0 in (*q_shape, *k_shape, *v_shape):
        return f"q {q_shape} / k {k_shape} / v {v_shape} must have no empty dimension"
    b, sq, hq, d = q_shape
    bk, skv, hkv, dk = k_shape
    if v_shape != k_shape or bk != b:
        return f"k {k_shape} and v {v_shape} must match, with q's batch {b}"
    if d != HEAD_DIM or dk != HEAD_DIM:
        return f"head_dim must be {HEAD_DIM}, got q {d} / k {dk}"
    if hq % hkv:
        return f"heads_q {hq} must be a multiple of heads_kv {hkv}"
    g = hq // hkv
    if rules["g_values"] is not None and g not in rules["g_values"]:
        return f"{module}: heads_q / heads_kv must be one of {rules['g_values']}, got {hq} / {hkv}"
    if g % rules["g_multiple"]:
        return f"{module}: heads_q / heads_kv must be a multiple of {rules['g_multiple']}, got {hq} / {hkv}"
    if not causal:
        return f"{module}: causal attention only"
    if sq > skv:
        return f"{module}: causal needs seqlen_q <= seqlen_kv, got {sq} > {skv}"
    if sq % rules["sq_multiple"]:
        return f"{module}: seqlen_q must be a multiple of {rules['sq_multiple']}, got {sq}"
    if skv % rules["skv_multiple"]:
        return f"{module}: seqlen_kv must be a multiple of {rules['skv_multiple']}, got {skv}"
    for name, nbytes in _EXTENTS[module](b, sq, skv, hq, hkv):
        if nbytes >= INT32_LIMIT:
            return f"{module}: the {name} extent of {nbytes} bytes overflows the kernels' int32 offsets"
    return None
