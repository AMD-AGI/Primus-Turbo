"""op.reference.api, backed by the three gfx1250 FlyDSL kernels in kernels.py."""
import importlib.util as _ilu
import pathlib as _pl

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
    import sys as _sys
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_env = _sibling("_env")

import math

import torch

_k = _sibling("kernels")

_ENV_CHECKED = False


def _check_env_once():
    global _ENV_CHECKED
    if not _ENV_CHECKED:
        _env.assert_environment()
        _ENV_CHECKED = True


def flydsl_attn_bwd(do, q, k, v, o, lse, softmax_scale=None, causal=True):
    """Grouped-query flash-attention backward on gfx1250.

    q/o/do  [B, Sq, Hq, D] bf16      k/v  [B, Skv, Hkv, D] bf16
    lse     [B, Hq, Sq] fp32, NATURAL log, as aiter's gfx1250 forward emits it.
    Returns (dq, dk, dv) in q/k/v's dtype, laid out like q/k/v.

    causal is BOTTOM-RIGHT: query i attends keys j <= i + (Skv - Sq).
    """
    _check_env_once()
    b, sq, hq, d = q.shape
    skv, hkv = k.shape[1], k.shape[2]
    assert d == _k.D, f"this kernel is head_dim {_k.D} only, got {d}"
    assert hq % hkv == 0, f"heads_q {hq} is not a multiple of heads_kv {hkv}"
    assert sq % 32 == 0, (
        f"seqlen_q must be a multiple of 32 (k_dkdv consumes query tiles in pairs), "
        f"got {sq}")
    assert skv % _k.KV_STEP == 0, (
        f"seqlen_kv must be a multiple of {_k.KV_STEP}, got {skv}")
    n_rows = b * sq * hq
    assert n_rows % _k.ROWS_DELTA == 0, (
        f"batch*seqlen_q*heads_q must be a multiple of {_k.ROWS_DELTA}, got {n_rows}")
    for name, t in (("do", do), ("q", q), ("k", k), ("v", v), ("o", o)):
        assert t.is_contiguous(), f"{name} must be contiguous"
        assert t.dtype == torch.bfloat16, f"{name} must be bf16, got {t.dtype}"
    lse = lse.contiguous().float()

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    g = hq // hkv
    stream = torch.cuda.current_stream()

    delta = torch.empty((b, hq, sq), device=q.device, dtype=torch.float32)
    _k.launch_delta(do, o, delta, sq, hq, n_rows, n_rows // _k.ROWS_DELTA, stream)

    dk32 = torch.empty((b, skv, hkv, d), device=k.device, dtype=torch.float32)
    dv32 = torch.empty_like(dk32)
    _k.launch_dkdv(q, k, v, do, lse, delta, dv32, dk32, float(softmax_scale),
                   sq, skv, hq, hkv, g, sq // 16, skv - sq, int(bool(causal)),
                   skv // 16, hkv, b, stream)

    dq32 = torch.empty((b, sq, hq, d), device=q.device, dtype=torch.float32)
    _k.launch_dq(q, k, v, do, lse, delta, dq32, float(softmax_scale),
                 sq, skv, hq, hkv, g, skv // _k.KV_STEP, skv - sq, int(bool(causal)),
                 sq // 16, hq, b, stream)

    return dq32.to(q.dtype), dk32.to(k.dtype), dv32.to(v.dtype)


attn_bwd = flydsl_attn_bwd   # uniform name every loader uses
