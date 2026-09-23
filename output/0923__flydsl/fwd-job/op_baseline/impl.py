"""op.reference.api, backed by the VENDORED gfx1250 FlyDSL forward kernel.

Everything this module imports comes from `flydsl_fwd/` next to this file, never from
the installed `aiter` package -- that is the whole point of the vendoring. If you see
`aiter` anywhere in an import path reached from here, the tree is broken.
"""
import importlib as _il
import importlib.util as _ilu
import pathlib as _pl
import sys as _sys

_HERE = _pl.Path(__file__).resolve().parent


def _sibling(stem):
    """Import a module from THIS directory under a name unique to this directory.

    A plain `import kernels` binds `sys.modules["kernels"]`, so loading a second
    implementation in the same process silently reuses the first one's module -- and with
    it the first one's JIT-compiled kernels. The two arms then differ by under 0.05% and
    produce identical output, which is exactly what a real result looks like. The
    directory is the identity, so the module name has to carry it.
    """
    name = f"{stem}__{abs(hash(str(_HERE)))}"
    spec = _ilu.spec_from_file_location(name, _HERE / f"{stem}.py")
    mod = _ilu.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sibling_pkg(pkg):
    """Same trick, for a PACKAGE directory rather than a single module.

    The forward kernel is five files that import each other, so the unique name has to
    be carried by the package, not by one module: `submodule_search_locations` makes the
    package's own relative imports (`from .kernels_common import ...`) resolve as
    `<unique>.kernels_common`, so a second arm in the same process gets its own copy of
    all five modules and its own JIT cache. Registering the package in `sys.modules`
    before exec_module is required -- the relative imports run during exec and look the
    parent up by name.
    """
    name = f"{pkg}__{abs(hash(str(_HERE)))}"
    if name in _sys.modules:
        return _sys.modules[name]
    d = _HERE / pkg
    spec = _ilu.spec_from_file_location(
        name, d / "__init__.py", submodule_search_locations=[str(d)]
    )
    mod = _ilu.module_from_spec(spec)
    _sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        _sys.modules.pop(name, None)
        raise
    return mod


_env = _sibling("_env")

import math

import torch

_fwd = _sibling_pkg("flydsl_fwd")
# The kernel module, under the package's unique name, so a second arm in the same
# process compiles its own kernels instead of reusing this one's.
_kern = _il.import_module(f"{_fwd.__name__}.fmha_fwd_prefill_a16w16_m32x8")

_ENV_CHECKED = False


def _check_env_once():
    global _ENV_CHECKED
    if not _ENV_CHECKED:
        _env.assert_environment()
        _ENV_CHECKED = True


def flydsl_attn_fwd(q, k, v, softmax_scale=None, causal=True):
    """Grouped-query flash-attention forward on gfx1250, BSHD.

    q  [B, Sq, Hq, D] bf16      k/v  [B, Skv, Hkv, D] bf16
    Returns (o, lse):
      o    [B, Sq, Hq, 128]  q.dtype
      lse  [B, Hq, Sq] fp32, NATURAL log -- the layout the backward job consumes.

    causal is BOTTOM-RIGHT: query i attends keys j <= i + (Skv - Sq). The kernel
    computes this as `causal_off = kv_len - q_len`
    (flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py:793), which is the same convention
    the backward job uses.

    Host entry: `flash_attn_batch_m32x8(q, k, v, softmax_scale=None, causal=False,
    window_size=(-1,-1), out=None, return_lse=False, sink=None, lse=None)`
    (flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py:2093-2104); with return_lse=True it
    returns `(out, lse)` (same file, :2262-2264).
    """
    _check_env_once()
    b, sq, hq, d = q.shape
    hkv = k.shape[2]
    assert d in _kern.SUPPORTED_QK_HDIM, (
        f"qk_hdim must be one of {_kern.SUPPORTED_QK_HDIM}, got {d}")
    assert v.shape[-1] == 128, f"this kernel is v_hdim 128 only, got {v.shape[-1]}"
    assert hq % hkv == 0, f"heads_q {hq} is not a multiple of heads_kv {hkv}"
    for name, t in (("q", q), ("k", k), ("v", v)):
        assert t.is_contiguous(), f"{name} must be contiguous"
        assert t.dtype == torch.bfloat16, f"{name} must be bf16, got {t.dtype}"

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    out, lse = _kern.flash_attn_batch_m32x8(
        q,
        k,
        v,
        softmax_scale=float(softmax_scale),
        causal=bool(causal),
        return_lse=True,
    )
    return out, lse


attn_fwd = flydsl_attn_fwd   # uniform name every loader uses
