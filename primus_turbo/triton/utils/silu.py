###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""SwiGLU activation math on fp32 register tiles, for fused GLU epilogues.

The sigmoid is spelled as ``exp2`` plus a raw ``v_rcp_f32``: every high-level
form of ``x / (1 + exp(-x))`` -- ``tl.fdiv``, ``tl.sigmoid``, plain ``/`` --
lowers to an IEEE-exact divide costing several times the VALU ops. That matters
where these run, in an epilogue nothing hides: the waves of a workgroup share a
tile, so they enter the epilogue together and the matrix pipe goes idle for its
whole length. Skipping the Newton fixup leaves ``v_rcp_f32`` at ~1 ulp, orders
below bf16's 8-bit mantissa, so stored results are unchanged in practice.

AMDGCN-only: the reciprocal is inline asm.
"""

import triton
import triton.language as tl


@triton.jit
def _sigmoid_rcp(x):
    """``sigmoid(x)`` via exp2 and the raw hardware reciprocal."""
    log2e: tl.constexpr = 1.4426950408889634  # folds exp's log2e into exp2
    d = 1.0 + tl.exp2(-x * log2e)
    return tl.inline_asm_elementwise(
        "v_rcp_f32_e32 $0, $1", "=v,v", [d], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def silu_mul_probs(gate, up, probs_row):
    """``silu(gate) * up * probs`` with ``probs_row`` [BLOCK_M] broadcast over columns."""
    return gate * _sigmoid_rcp(gate) * up * probs_row[:, None]


@triton.jit
def silu_mul_bwd_act(gate, up, dout):
    """Gradients of ``silu(gate) * up``, as ``(dgate, dup, dout_act)``.

    ``dout_act`` is ``dout * silu(gate) * up``: the per-element term a routed MLP
    sums over the hidden dimension to get the gradient wrt the routing
    probabilities. It costs one multiply on top of the gradients, since ``dup``
    is already ``dout * silu(gate)``; it is only that gradient if ``dout`` is
    unscaled by the probabilities, so the caller has to apply those to the halves
    after.
    """
    s = _sigmoid_rcp(gate)
    silu = s * gate
    dup = dout * silu
    # s * (1 + gate * (1 - s)) rewritten as s * (1 + gate - silu): silu is
    # already live, so this drops a multiply per element.
    dgate = dout * up * s * (1.0 + gate - silu)
    return dgate, dup, dup * up
