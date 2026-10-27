###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Load ladder rungs as callables.

    from hkl import runner
    r = runner.load("llama/ladder_fwd.json", "f03")      # build/llama_fwd/f03 must exist (hkl.build)
    o, lse = r.fwd(q, k, v, causal=True)
    r = runner.load("llama/ladder_bwd.json", "p1")
    dq, dk, dv = r.bwd(do, q, k, v, o, lse, causal=True)

A rung's ``launch`` names a module of its family's ``lab/`` package (``llama/lab``, ``mla/lab``) and
optional ``opts`` for that module's ``configure``. The lab modules keep their state (code-object
directory, loaded kernels, configuration) in module globals, so every rung gets its OWN instance of
the package (imported under a private name): rungs that share a module (v0, v0_resc, p1 all use
fwd_v0) can be loaded side by side in one process and timed in rotation.

Nothing here touches a GPU until a rung is called: load() reads the code objects (sha256 checked
against the build's meta.json by the lab module at the first call).
"""

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

from .build import DEFAULT_OUT, ladder_name, load_ladder

__all__ = ["Baseline", "Rung", "baseline_ids", "load", "load_baseline", "rung_ids"]


class Rung:
    def __init__(self, spec, rung, out_root, module):
        self.spec, self.rung, self.module = spec, rung, module
        self.id = rung["id"]
        self.ladder = ladder_name(spec)
        self.dir = Path(out_root) / self.ladder / self.id
        self.kernels = list(rung["kernels"])

    def __repr__(self):
        return f"Rung({self.ladder}:{self.id})"

    def meta(self):
        """The build's meta.json of every kernel of the rung (sha256, resources, ok)."""
        out = {}
        for k in self.kernels:
            with open(self.dir / f"{k}.meta.json") as f:
                out[k] = json.load(f)
        return out

    def fwd(self, q, k, v, softmax_scale=None, causal=True):
        # explicit scale: fwd_pp has no default for it
        scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
        return self.module.flash_attn_fwd(q, k, v, scale, causal)

    def bwd(self, do, q, k, v, o, lse, softmax_scale=None, causal=True):
        scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
        return self.module.flash_attn_bwd(do, q, k, v, o, lse, softmax_scale=scale, causal=causal)

    def plan(self, *args, **kw):
        return self.module.plan(*args, **kw)


_COUNTER = [0]


def _fresh_package(lab_dir):
    """A new, independent instance of the package at lab_dir (private module name)."""
    _COUNTER[0] += 1
    name = f"_hkl_lab_{_COUNTER[0]}"
    init = Path(lab_dir) / "__init__.py"
    spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[str(lab_dir)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[name] = pkg
    spec.loader.exec_module(pkg)
    return name


class Baseline:
    """An external arm (e.g. Primus-Turbo's FlyDSL kernels) with the rung call signature."""

    def __init__(self, spec, b):
        self.spec, self.b = spec, b
        self.id = b["id"]
        self.ladder = ladder_name(spec)
        root = os.environ.get("HKL_TURBO_DIR") or str(Path(__file__).resolve().parents[3])
        if root not in sys.path:
            sys.path.insert(0, root)  # the Primus-Turbo checkout this ladder lives in (or HKL_TURBO_DIR)
        self.module = importlib.import_module(b["module"])
        self._fwd = getattr(self.module, b["fwd"]) if b.get("fwd") else None
        self._bwd = getattr(self.module, b["bwd"]) if b.get("bwd") else None
        self.kwargs = dict(b.get("kwargs", {}))

    def __repr__(self):
        return f"Baseline({self.ladder}:{self.id})"

    def fwd(self, q, k, v, softmax_scale=None, causal=True):
        scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
        return self._fwd(q, k, v, softmax_scale=scale, causal=causal, **self.kwargs)

    def bwd(self, do, q, k, v, o, lse, softmax_scale=None, causal=True):
        scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
        return self._bwd(do, q, k, v, o, lse, softmax_scale=scale, causal=causal, **self.kwargs)


def load_baseline(ladder, bid):
    spec = load_ladder(ladder)
    for b in spec.get("baselines", []):
        if b["id"] == bid:
            return Baseline(spec, b)
    raise KeyError(f"{ladder}: no baseline {bid!r}")


def baseline_ids(ladder):
    return [b["id"] for b in load_ladder(ladder).get("baselines", [])]


def rung_ids(ladder, include_side=True):
    spec = load_ladder(ladder)
    return [r["id"] for r in spec["rungs"] if include_side or not r.get("side")]


def load(ladder, rung_id, out=None):
    """The rung as a Rung (its lab module configured and pointed at the rung's build directory)."""
    spec = load_ladder(ladder)
    rung = spec["_by_id"][rung_id]
    out_root = Path(out or DEFAULT_OUT).resolve()
    d = out_root / ladder_name(spec) / rung_id
    if not d.is_dir():
        raise FileNotFoundError(f"{d}: build the rung first (python3 -m hkl.build {spec['_path']} --rung {rung_id})")
    launch = rung.get("launch", {})
    pkg = _fresh_package(spec["_dir"] / "lab")
    mod = importlib.import_module(f"{pkg}.{launch['module']}")
    if launch.get("opts"):
        mod.configure(**launch["opts"])
    mod.set_hsaco_dir(str(d))
    return Rung(spec, rung, out_root, mod)
