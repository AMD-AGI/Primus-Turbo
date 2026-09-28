"""The attention arms of the B0 e2e bake-off, each as a pair of plain callables.

    fwd(q, k, v, scale)                   -> (o, lse)    q/k/v/o BSHD bf16 contiguous,
                                                         lse [B, Hq, Sq] fp32 natural log
    bwd(do, q, k, v, o, lse, scale)       -> (dq, dk, dv) BSHD, q/k/v dtype

Arms (causal is always True -- the model's mask on the turbo path is plain causal):

    asm   fwd  aiter.ops.mha.fmha_fwd_with_sink_asm (prebuilt gfx1250 ASM)
          bwd  hand-launched aiter ASM backward (_asm_bwd_kernargs.asm_backward), dkdv_heads="q",
               scratch allocated once and reused, then the host GQA sum over the q-head slices --
               exactly what the bwd job's op/beat/impl.py times.
    fly   fwd  FlyDSL fwd round-6 champion   (arms/fwd_r6,        flydsl 0.3.4.1)
          bwd  FlyDSL bwd round-20 champion  (arms/bwd_r20_0341,  r20 kernels, _env pinned 0.3.4.1)

The turbo (stock Primus-Turbo Triton) arm is NOT here: it needs the real primus_turbo package,
which cannot share a process with flydsl 0.3.4.1. See e2e_attn/__init__.py (`turbo_pair`).

Extra FlyDSL variants (e.g. a newer round next to the current one) are registered through
E2E_FLY_TREES='{"flynew": {"fwd": "<dir>", "bwd": "<dir>"}}' -- each tree loads under a
directory-unique module name, so two trees in one process never share JIT'd kernels.
Everything is loaded lazily, on the first call of an arm.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

E2E = Path(__file__).resolve().parents[2]            # .../0927__b0/e2e
ARMS = E2E / "arms"
FLYDSL0341 = "/home/lihuzhan/.local/flydsl0341"
AITER_SRC = "/home/lihuzhan/code/aiter-src"

FLY_TREES = {"fly": {"fwd": str(ARMS / "fwd_r6"), "bwd": str(ARMS / "bwd_r20_0341")}}
if os.environ.get("E2E_FLY_TREES"):
    FLY_TREES.update(json.loads(os.environ["E2E_FLY_TREES"]))

_cache: dict = {}


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_flydsl0341():
    # Primus' base_env.sh puts the image site-packages (flydsl 0.2.4) AHEAD of PYTHONPATH, so
    # the 0.3.4.1 install has to be put first by hand before anything imports flydsl.
    # it may already be on sys.path via PYTHONPATH but BEHIND the image site-packages (base_env.sh
    # reorders), so always move it to the front (smoke s1_p2smoke4L died on exactly this).
    while FLYDSL0341 in sys.path:
        sys.path.remove(FLYDSL0341)
    sys.path.insert(0, FLYDSL0341)
    fl = sys.modules.get("flydsl")
    if fl is not None and FLYDSL0341 not in (fl.__file__ or ""):
        raise RuntimeError(f"flydsl already imported from {fl.__file__}, not {FLYDSL0341}")
    if AITER_SRC not in sys.path:
        sys.path.append(AITER_SRC)


# ----------------------------------------------------------------------------- asm
def asm_fwd():
    if "asm_fwd" not in _cache:
        _ensure_flydsl0341()
        import aiter.ops.mha as mha
        assert mha.__file__.startswith(AITER_SRC), f"aiter resolved to {mha.__file__}"
        f = mha.fmha_fwd_with_sink_asm

        def fwd(q, k, v, scale):
            return f(q, k, v, float(scale), True, True)
        _cache["asm_fwd"] = fwd
    return _cache["asm_fwd"]


class _AsmBwd:
    def __init__(self):
        _ensure_flydsl0341()
        self.mod = _load_by_path("e2e_asm_bwd_kernargs", ARMS / "asm" / "_asm_bwd_kernargs.py")
        for stem in ("bwd_hd128_odo_bf16", "bwd_hd128_bf16_causal_br_a32_pssk",
                     "bwd_hd128_dq_convert_bf16"):
            p = self.mod.ASM_DIR / f"{stem}.co"
            if not p.is_file():
                raise FileNotFoundError(f"missing prebuilt ASM object {p}")
        self.hip = self.mod.HipModule()
        self.scratch = {}

    def get_scratch(self, q):
        b, sq, hq, d = q.shape
        key = (b, sq, hq, d, str(q.device), str(q.dtype))
        s = self.scratch.get(key)
        if s is None:
            # Resident for the life of the process: dq_acc fp32 + per-q-head dk/dv slices
            # (1 GiB at b4 s8192 hq32 d128). Only dk/dv are reduced into fresh tensors below,
            # so handing the same block to every layer is safe.
            s = {"dq_acc": torch.empty((b, hq, sq, d), device=q.device, dtype=torch.float32),
                 "dk": torch.empty((b, sq, hq, d), device=q.device, dtype=q.dtype),
                 "dv": torch.empty((b, sq, hq, d), device=q.device, dtype=q.dtype)}
            self.scratch[key] = s
        return s

    def __call__(self, do, q, k, v, o, lse, scale):
        b, sq, hq, d = q.shape
        skv, hkv = k.shape[1], k.shape[2]
        assert skv == sq, "scratch is keyed on q's shape; sq == skv only"
        g = hq // hkv
        dq, dk_q, dv_q = self.mod.asm_backward(
            q, k, v, o, do, lse, softmax_scale=float(scale), hip=self.hip,
            dkdv_heads="q", causal=True, scratch=self.get_scratch(q))
        with torch.profiler.record_function("e2e::asm_gqa_sum"):
            dk = dk_q.view(b, skv, hkv, g, d).sum(dim=3).to(k.dtype)
            dv = dv_q.view(b, skv, hkv, g, d).sum(dim=3).to(v.dtype)
        return dq, dk, dv


def asm_bwd():
    if "asm_bwd" not in _cache:
        _cache["asm_bwd"] = _AsmBwd()
    return _cache["asm_bwd"]


# ----------------------------------------------------------------------------- fly
def fly_fwd(name="fly"):
    key = f"{name}_fwd"
    if key not in _cache:
        _ensure_flydsl0341()
        d = Path(FLY_TREES[name]["fwd"]).resolve()
        mod = _load_by_path(f"e2e_{name}_fwd_{abs(hash(str(d)))}", d / "impl.py")
        f = mod.attn_fwd

        def fwd(q, k, v, scale):
            return f(q, k, v, float(scale), True)
        _cache[key] = fwd
    return _cache[key]


def fly_bwd(name="fly"):
    key = f"{name}_bwd"
    if key not in _cache:
        _ensure_flydsl0341()
        d = Path(FLY_TREES[name]["bwd"]).resolve()
        mod = _load_by_path(f"e2e_{name}_bwd_{abs(hash(str(d)))}", d / "impl.py")
        f = mod.attn_bwd

        def bwd(do, q, k, v, o, lse, scale):
            return f(do, q, k, v, o, lse, float(scale), True)
        _cache[key] = bwd
    return _cache[key]


def get_fwd(arm):
    return asm_fwd() if arm == "asm" else fly_fwd(arm)


def get_bwd(arm):
    return asm_bwd() if arm == "asm" else fly_bwd(arm)


def known(arm):
    return arm == "asm" or arm in FLY_TREES


def check_versions():
    """One line of evidence: which flydsl / aiter this process really runs."""
    import flydsl
    out = f"flydsl {flydsl.__version__} @ {flydsl.__file__}"
    assert FLYDSL0341 in flydsl.__file__, out
    return out
