###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""V4 attention via the Gluon sparse-MLA backend ("gluon_v2").

``gluon_v2`` forward is the Gluon sparse-MLA kernel (gfx950 / CDNA4 hardware-controlled
layouts + async double-buffered pipeline, rope-skip + exp2 + MFMA K=32); the backward is
currently the plain-Triton chunked-gather kernel (shared with ``triton_v2``) and is being
migrated to Gluon. Thin binding of the kernel-agnostic V4 adapters
(:mod:`v4_sparse_mla_adapter`) to the ``sparse_mla_{fwd,bwd}_v4_gluon_v2`` kernel pair
(``primus_turbo.gluon.attention.deepseek_v4.v2``).

The Gluon forward requires a Gluon-capable (recompiled) triton; the kernel raises a clear
build hint otherwise. Numerically equivalent to the eager V4 references (validated in
tests/pytorch/ops/deepseek_v4/test_attention_gluon_v2.py).
"""

from __future__ import annotations

from primus_turbo.gluon.attention.deepseek_v4.v2 import (
    sparse_mla_bwd_v4_gluon_v2,
    sparse_mla_fwd_v4_gluon_v2,
)
from primus_turbo.pytorch.ops.attention.deepseek_v4.v4_sparse_mla_adapter import (
    make_attention,
    make_csa_from_pool,
)

v4_csa_attention_gluon_v2 = make_csa_from_pool(sparse_mla_fwd_v4_gluon_v2, sparse_mla_bwd_v4_gluon_v2)
v4_attention_gluon_v2 = make_attention(sparse_mla_fwd_v4_gluon_v2, sparse_mla_bwd_v4_gluon_v2)

__all__ = [
    "v4_csa_attention_gluon_v2",
    "v4_attention_gluon_v2",
]
