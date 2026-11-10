###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 sparse-MLA kernel-pair bound to the generic FlyDSL sparse-MLA.

Unlike the V4-specific fused single-latent backends
(:mod:`primus_turbo.gluon.attention.deepseek_v4.v2`,
:mod:`primus_turbo.triton.attention.deepseek_v4.v2`,
:mod:`primus_turbo.flydsl.attention.deepseek_v4.v1`), this backend reuses the
generic native-FlyDSL sparse-MLA attention under
``primus_turbo.flydsl.attention``.

Public kernel-pair API (identical to the other sparse-MLA backends):

* ``sparse_mla_fwd_v4_turbo_flydsl(q, kv, topk, attn_sink=None,
  kv_lora_rank=512, scale=None) -> (o, lse)``
* ``sparse_mla_bwd_v4_turbo_flydsl(q, kv, o, do, topk, lse, attn_sink=None,
  kv_lora_rank=512, scale=None) -> (dq, dkv, d_sink)``

The ``flydsl`` pip package is required; the import fails with a clear message
otherwise (handled by :func:`..load_turbo_attention_backends`).
"""

from __future__ import annotations

from primus_turbo.flydsl.attention.sparse_mla_bwd import (
    sparse_mla_bwd_flydsl as _sparse_mla_bwd,
)
from primus_turbo.flydsl.attention.sparse_mla_fwd import (
    sparse_mla_fwd_flydsl as _sparse_mla_fwd,
)

# Turbo-suffixed re-exports so the V4 wrapper / benchmark can bind them without
# clashing with the V4 FlyDSL backend (which has identically-named kernels).
sparse_mla_fwd_v4_turbo_flydsl = _sparse_mla_fwd
sparse_mla_bwd_v4_turbo_flydsl = _sparse_mla_bwd

__all__ = ["sparse_mla_fwd_v4_turbo_flydsl", "sparse_mla_bwd_v4_turbo_flydsl"]
