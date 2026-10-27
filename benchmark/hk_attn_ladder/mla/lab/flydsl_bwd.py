###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The FlyDSL MLA backward (Primus-Turbo primus_turbo.flydsl.attention.gfx1250_mla.bwd), the baseline of the
MLA ladder, with bwd_hk's layout folding (SBHD -> [1, S, B*H, D]) so it sees exactly the same call.

    dq, dk, dv = flash_attn_bwd(do, q, k, v, o, lse, softmax_scale=None, causal=True)
"""

from . import _turbo_path  # noqa: F401
from primus_turbo.flydsl.attention.gfx1250_mla import bwd as _fly  # noqa: E402

from .bwd_hk import _fold, _unfold, is_sbhd  # noqa: E402


def flash_attn_bwd(do, q, k, v, o, lse, softmax_scale=None, causal=True):
    sbhd = is_sbhd(q, lse)
    b = q.shape[1] if sbhd else q.shape[0]
    f = [_fold(t, sbhd) for t in (do, q, k, v, o)]
    lse_k = lse.reshape(1, -1, lse.shape[-1]) if sbhd else lse
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    grads = _fly.flash_attn_mla_bwd(*f, lse_k, scale, causal)
    return tuple(_unfold(t, b, sbhd) for t in grads)
