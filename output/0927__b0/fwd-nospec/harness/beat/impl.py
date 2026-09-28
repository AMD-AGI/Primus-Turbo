"""The beat arm: aiter's prebuilt gfx1250 ASM forward behind op.reference.api.

    aiter.ops.mha.fmha_fwd_with_sink_asm(q, k, v, softmax_scale, is_causal, return_lse,
                                         sink=None, out=None) -> (out, lse)
    (/home/lihuzhan/code/aiter-src/aiter/ops/mha.py:589)

Imported from the aiter checkout, never copied, and never edited. See PROVENANCE.md.
"""
import importlib.util as _ilu
import math
import pathlib as _pl
import sys as _sys

_HERE = _pl.Path(__file__).resolve().parent


def _sibling(stem):
    name = f"{stem}__{abs(hash(str(_HERE)))}"
    spec = _ilu.spec_from_file_location(name, _HERE / f"{stem}.py")
    mod = _ilu.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_env = _sibling("_env")

import torch  # noqa: E402
import aiter.ops.mha as _mha  # noqa: E402

assert _mha.__file__.startswith(_env.AITER_SRC), f"aiter resolved to {_mha.__file__}"
_ASM = _mha.fmha_fwd_with_sink_asm


def asm_attn_fwd(q, k, v, softmax_scale=None, causal=True):
    """q [B,Sq,Hq,128] bf16, k/v [B,Skv,Hkv,128] bf16 -> (o [B,Sq,Hq,128], lse [B,Hq,Sq] fp32).

    sink=None. Only validated at seqlen_q == seqlen_kv (the three op.shape entries); the
    ASM kernel's causal alignment for seqlen_q != seqlen_kv is not relied on anywhere.
    """
    for name, t in (("q", q), ("k", k), ("v", v)):
        assert t.is_contiguous(), f"{name} must be contiguous"
        assert t.dtype == torch.bfloat16, f"{name} must be bf16, got {t.dtype}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    return _ASM(q, k, v, float(softmax_scale), bool(causal), True)


attn_fwd = asm_attn_fwd   # uniform name every loader uses
