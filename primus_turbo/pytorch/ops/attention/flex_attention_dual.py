###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Dual mods: one mod, a Triton and a FlyDSL implementation.

The Triton backend inlines ``@triton.jit`` mods; the FlyDSL backend traces plain Python
functions over ``flydsl.expr`` values, with FA4's signatures:

    score_mod(score, b, h, q_idx, kv_idx, seqlen_info, aux_tensors) -> score
    score_mod_bwd(dscore, score, b, h, q_idx, kv_idx, seqlen_info, aux_tensors) -> dscore
    mask_mod(b, h, q_idx, kv_idx, seqlen_info, aux_tensors) -> bool

(see ``primus_turbo.flydsl.flex_attention.mods``). A mod is still passed to the API as
its Triton function; ``dual_mod`` registers the FlyDSL twin for it, which the FlyDSL
backend looks up. The built-in masks and mods have twins already. Registration is lazy
(a factory is allowed) so importing this module does not import FlyDSL.
"""

from typing import Callable

__all__ = ["dual_mod", "flydsl_twin"]

_TWINS: dict = {}


def dual_mod(triton_fn, flydsl_fn=None, *, factory: Callable = None):
    """Register ``flydsl_fn`` (or a zero-argument ``factory`` returning it) as the FlyDSL
    twin of ``triton_fn``; returns ``triton_fn`` so it can be used as a decorator target."""
    if (flydsl_fn is None) == (factory is None):
        raise ValueError("pass exactly one of flydsl_fn and factory")
    _TWINS[triton_fn] = (flydsl_fn, factory)
    return triton_fn


def flydsl_twin(triton_fn):
    """The FlyDSL implementation of ``triton_fn``, or None if it has none."""
    entry = _TWINS.get(triton_fn)
    if entry is None:
        return None
    fn, factory = entry
    if fn is None:
        fn = factory()
        _TWINS[triton_fn] = (fn, None)
    return fn
