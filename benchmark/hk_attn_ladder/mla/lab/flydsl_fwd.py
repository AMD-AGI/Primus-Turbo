###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The FlyDSL MLA forward, the baseline of the MLA forward ladder, with fwd_hk's layout folding (SBHD -> [1, S, B*H,
D]) so it sees exactly the same call.

    o, lse = flash_attn_fwd(q, k, v, softmax_scale=None, causal=True, impl="main", layout="sbhd")

impl "main": this Primus-Turbo checkout's primus_turbo.flydsl.attention.gfx1250_mla.fwd (main, #567).
impl "b12f5953": the persistent forward of dev/lhz/dsv3-attn-hk-b0 (b12f5953, "persistent gfx1250 MLA forward with
dynamic per-XCD block scheduling"), the FlyDSL the HK forward was measured against on a08-1. Its package is vendored
by tools/vendor_flydsl_fwd.sh into build/vendor/flydsl_b12f5953/ (git show, byte for byte) and loaded here under a
private module name; its absolute imports (primus_turbo.flydsl.utils, identical at both commits) resolve to this
checkout.
"""

import importlib.util
import sys
from pathlib import Path

from . import _turbo_path  # noqa: F401

D_V = 128
VENDOR = Path(__file__).resolve().parents[2] / "build" / "vendor" / "flydsl_b12f5953"
_MODS = {}


def _module(impl):
    if impl in _MODS:
        return _MODS[impl]
    if impl == "main":
        from primus_turbo.flydsl.attention.gfx1250_mla import fwd as m
    elif impl == "b12f5953":
        pkg_dir = VENDOR / "primus_turbo" / "flydsl" / "attention" / "gfx1250_mla"
        if not (pkg_dir / "fwd.py").is_file():
            raise RuntimeError(f"{pkg_dir}: run tools/vendor_flydsl_fwd.sh first")
        name = "_hkl_flydsl_mla_b12f5953"
        spec = importlib.util.spec_from_file_location(name, pkg_dir / "__init__.py",
                                                      submodule_search_locations=[str(pkg_dir)])  # fmt: skip
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[name] = pkg
        spec.loader.exec_module(pkg)
        m = importlib.import_module(f"{name}.fwd")
    else:
        raise ValueError(f"unknown FlyDSL forward {impl!r}")
    _MODS[impl] = m
    return m


def flash_attn_fwd(q, k, v, softmax_scale=None, causal=True, impl="main", layout="sbhd"):
    """(o in the problem's layout, lse [B, H, Sq] fp32)."""
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    if layout == "bshd":
        return _module(impl).flash_attn_mla_fwd(q.contiguous(), k.contiguous(), v.contiguous(), scale, causal)
    s, b, h, _ = q.shape
    f = [t.contiguous().view(1, t.shape[0], t.shape[1] * t.shape[2], t.shape[3]) for t in (q, k, v)]
    o, lse = _module(impl).flash_attn_mla_fwd(*f, scale, causal)
    return o.view(s, b, h, D_V), lse.view(b, h, s)
