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

"""Host entry of the gfx1250 FlyDSL MLA forward.

q [B, Sq, Hq, 192], k [B, Skv, Hkv, 192], v [B, Skv, Hkv, 128], bf16 and contiguous, Hq a
multiple of Hkv. Returns o [B, Sq, Hq, 128] in q's dtype and lse [B, Hq, Sq] fp32 in natural
log, the layout the gfx1250 MLA backward consumes. causal is bottom-right: query i attends
keys j <= i + (Skv - Sq).
"""

import math

import torch

from . import fmha_fwd_prefill_a16w16_m32x8 as _kern

D_QK = 192
D_V = 128


def flash_attn_fwd(q, k, v, softmax_scale=None, causal=True):
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.dtype != torch.bfloat16 or not t.is_contiguous() or t.dim() != 4:
            raise ValueError(f"{name} must be a contiguous 4-D bfloat16 tensor")
    if q.shape[-1] != D_QK or k.shape[-1] != D_QK or v.shape[-1] != D_V:
        raise ValueError(
            f"head dims must be qk {D_QK} / v {D_V}, got {q.shape[-1]}/{k.shape[-1]}/{v.shape[-1]}"
        )
    if q.shape[2] % k.shape[2]:
        raise ValueError(f"heads_q {q.shape[2]} is not a multiple of heads_kv {k.shape[2]}")
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D_QK)
    return _kern.flash_attn_batch_m32x8(
        q, k, v, softmax_scale=float(softmax_scale), causal=bool(causal), return_lse=True
    )
