"""The attention arms of the e2e runs, each as a pair of plain callables -- A0 copy of 2026-10-02.

    fwd(q, k, v, scale)                   -> (o, lse)    q/k/v/o BSHD bf16 contiguous,
                                                         lse [B, Hq, Sq] fp32 natural log
    bwd(do, q, k, v, o, lse, scale)       -> (dq, dk, dv) BSHD, q/k/v dtype

    asm   fwd  aiter.ops.mha.fmha_fwd_with_sink_asm (prebuilt gfx1250 ASM)
          bwd  hand-launched aiter ASM backward (output/0927__b0/e2e/arms/asm/_asm_bwd_kernargs.py),
               dkdv_heads="q", scratch allocated once and reused, then the host GQA sum.
    <name>     FlyDSL trees named in E2E_FLY_TREES='{"fly": {"fwd": "<dir>", "bwd": "<dir>"}, ...}'.
               There is NO default FlyDSL tree in this copy: a missing E2E_FLY_TREES makes every
               FlyDSL arm "unknown" and the model build fails before the first step (the B0 copy
               silently fell back to fwd r6 + bwd r20).

Changes vs output/0927__b0/e2e/attn_backends/e2e_attn/arms.py:
  * guard_blas(): called after every arm load (and by E2EAttention.__init__). If anything
    re-pointed HIPBLASLT_TENSILE_LIBPATH / TORCH_BLAS_PREFER_HIPBLASLT away from what the
    launcher assigned (E2E_EXPECT_BLAS_LIB, "1"), the value is restored and a
    "!! BLAS-REPOINT" line is logged (the launcher's watchdog stops the run on it).
  * every loaded FlyDSL tree logs the md5 of its .py files (evidence of what really ran).
  * asm paths are absolute (output/0927__b0/e2e/arms/asm), not relative to this copy.
Each tree loads under a directory-unique module name, so two trees in one process never share
JIT'd kernels; everything is loaded lazily, on the first call of an arm.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

B0_E2E = Path("/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/e2e")
ASM_KERNARGS = B0_E2E / "arms" / "asm" / "_asm_bwd_kernargs.py"
FLYDSL0341 = "/home/lihuzhan/.local/flydsl0341"
AITER_SRC = "/home/lihuzhan/code/aiter-src"

FLY_TREES = {}
if os.environ.get("E2E_FLY_TREES"):
    FLY_TREES.update(json.loads(os.environ["E2E_FLY_TREES"]))

EXPECT_BLAS = os.environ.get("E2E_EXPECT_BLAS_LIB", "")
BLAS_EVENTS = []          # (where, seen HIPBLASLT_TENSILE_LIBPATH, seen TORCH_BLAS_PREFER_HIPBLASLT)

_cache: dict = {}


def _log(msg):
    print(f"[e2e_attn] {msg}", file=sys.stderr, flush=True)


def guard_blas(where):
    """Undo any change of the hipBLASLt env made after the launcher assigned it."""
    if not EXPECT_BLAS:
        return
    cur = os.environ.get("HIPBLASLT_TENSILE_LIBPATH")
    pref = os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT")
    if cur != EXPECT_BLAS or pref != "1":
        os.environ["HIPBLASLT_TENSILE_LIBPATH"] = EXPECT_BLAS
        os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
        BLAS_EVENTS.append((where, cur, pref))
        _log(f"!! BLAS-REPOINT by {where}: HIPBLASLT_TENSILE_LIBPATH={cur!r} "
             f"TORCH_BLAS_PREFER_HIPBLASLT={pref!r} -> restored {EXPECT_BLAS!r} / '1'")


def _tree_md5(d):
    out = {}
    for p in sorted(Path(d).rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        out[str(p.relative_to(d))] = hashlib.md5(p.read_bytes()).hexdigest()[:8]
    return out


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_flydsl0341():
    # Primus' base_env.sh puts the image site-packages (flydsl 0.2.4) AHEAD of PYTHONPATH, so
    # the 0.3.4.1 install has to be put first by hand before anything imports flydsl.
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
        guard_blas("import aiter.ops.mha")
        f = mha.fmha_fwd_with_sink_asm

        def fwd(q, k, v, scale):
            return f(q, k, v, float(scale), True, True)
        _cache["asm_fwd"] = fwd
    return _cache["asm_fwd"]


class _AsmBwd:
    def __init__(self):
        _ensure_flydsl0341()
        self.mod = _load_by_path("e2e_asm_bwd_kernargs", ASM_KERNARGS)
        guard_blas(f"load {ASM_KERNARGS}")
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
def _fly_load(name, kind):
    _ensure_flydsl0341()
    d = Path(FLY_TREES[name][kind]).resolve()
    mod = _load_by_path(f"e2e_{name}_{kind}_{abs(hash(str(d)))}", d / "impl.py")
    guard_blas(f"load {name}/{kind} {d}")
    _log(f"tree {name}/{kind} {d} md5 {_tree_md5(d)}")
    return mod


def fly_fwd(name="fly"):
    key = f"{name}_fwd"
    if key not in _cache:
        f = _fly_load(name, "fwd").attn_fwd

        def fwd(q, k, v, scale):
            return f(q, k, v, float(scale), True)
        _cache[key] = fwd
    return _cache[key]


def fly_bwd(name="fly"):
    key = f"{name}_bwd"
    if key not in _cache:
        f = _fly_load(name, "bwd").attn_bwd

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
