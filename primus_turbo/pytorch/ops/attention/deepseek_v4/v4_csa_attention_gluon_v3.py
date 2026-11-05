###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""V4 attention via the Gluon sparse-MLA backend ("gluon_v3")."""

from __future__ import annotations

from primus_turbo.gluon.attention.deepseek_v4.v3 import (
    sparse_mla_bwd_v4_gluon_v3,
    sparse_mla_fwd_v4_gluon_v3,
)
from primus_turbo.pytorch.ops.attention.deepseek_v4.v4_sparse_mla_adapter import (
    make_attention,
    make_csa_from_pool,
)

v4_csa_attention_gluon_v3 = make_csa_from_pool(sparse_mla_fwd_v4_gluon_v3, sparse_mla_bwd_v4_gluon_v3)
v4_attention_gluon_v3 = make_attention(sparse_mla_fwd_v4_gluon_v3, sparse_mla_bwd_v4_gluon_v3)

__all__ = [
    "v4_csa_attention_gluon_v3",
    "v4_attention_gluon_v3",
]
