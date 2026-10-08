###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""Activation spec for MegaMoE's fused gated-activation kernels.

Every MegaMoE activation kernel (bf16 and mxfp8, forward and backward) computes, on
``gate, up = x.chunk(2, dim=-1)`` (the first ``I`` columns are the gate, not interleaved)::

    g   = gate.clamp(gate_clamp_lo, gate_clamp_hi)
    u   = up.clamp(-up_clamp, up_clamp)
    out = g * sigmoid(alpha * g) * (u + glu_offset)

The spec is pure data. The kernels bake it in as compile-time constants, so each distinct spec is
its own compiled kernel and a bound that is ``None`` costs no instruction at all.
"""

import math
from typing import NamedTuple, Optional, Sequence, Tuple

__all__ = ["ACTIVATION_CLAMP", "GLUActivation", "activation_constexpr"]

# The clamp the kernels have always applied to both halves; the default spec keeps it.
ACTIVATION_CLAMP = 10.0


class GLUActivation(NamedTuple):
    """``g * sigmoid(alpha * g) * (u + glu_offset)`` with ``g``/``u`` the clamped gate/up halves.

    A ``None`` bound is not applied. The defaults are SiLU-SwiGLU with the symmetric +-10 clamp
    MegaMoE has always used on both halves, so ``GLUActivation()`` reproduces the historical kernels
    bit for bit.
    """

    alpha: float = 1.0
    glu_offset: float = 0.0
    gate_clamp_lo: Optional[float] = -ACTIVATION_CLAMP
    gate_clamp_hi: Optional[float] = ACTIVATION_CLAMP
    up_clamp: Optional[float] = ACTIVATION_CLAMP

    @classmethod
    def swigluoai(cls, alpha: float = 1.702, limit: Optional[float] = 7.0, glu_offset: float = 1.0):
        """GPT-OSS-style ``swigluoai`` (MiniMax-M3's ``hidden_act``) on the chunked gate||up layout.

        The gate is clamped from above only and ``up`` on both sides, which is also Megatron's
        ``quick_geglu`` with ``glu_linear_offset`` and ``activation_func_clamp_value``.
        """
        return cls(
            alpha=alpha, glu_offset=glu_offset, gate_clamp_lo=None, gate_clamp_hi=limit, up_clamp=limit
        )


def activation_constexpr(activation: Optional[Sequence[Optional[float]]]) -> Tuple[float, ...]:
    """The kernels' compile-time form: ``(alpha, glu_offset, gate_lo, gate_hi, up_clamp)`` as floats.

    FlyDSL constexprs and cache keys take scalars only, so a missing bound becomes an infinity, which
    the kernels read as "emit no clamp". ``None`` is the default spec. Accepts a
    :class:`GLUActivation` or this function's own output (which is how a spec crosses a
    ``torch.library`` op boundary, whose schema has no room for a NamedTuple).
    """
    if activation is None:
        activation = GLUActivation()
    assert len(activation) == len(GLUActivation._fields), f"bad activation spec {activation!r}"
    act = GLUActivation(*activation)
    lo = -math.inf if act.gate_clamp_lo is None else float(act.gate_clamp_lo)
    hi = math.inf if act.gate_clamp_hi is None else float(act.gate_clamp_hi)
    up = math.inf if act.up_clamp is None else float(act.up_clamp)
    assert lo < hi, f"gate clamp needs lo < hi, got [{lo}, {hi}]"
    assert up > 0, f"up_clamp must be positive (it clamps to [-up_clamp, up_clamp]), got {up}"
    assert math.isfinite(act.alpha), f"alpha must be finite, got {act.alpha}"
    assert math.isfinite(act.glu_offset), f"glu_offset must be finite, got {act.glu_offset}"
    return (float(act.alpha), float(act.glu_offset), lo, hi, up)
