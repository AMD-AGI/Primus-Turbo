###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plain-Triton DeepSeek-V4 fused single-latent (K==V) sparse-MLA CSA backend.

Ported from upstream Primus ``_triton_v2`` (the "triton_v2" backend, PR #853).
Same fused single-latent representation and MFMA (``tl.dot``) kernels; wrapped by
:func:`csa_pool_attention_triton_v2` which bridges Primus-Turbo's separate-K/V
CSA-from-pool contract onto the single-latent kernel pair.
"""

from .adapter import csa_pool_attention_triton_v2
from .dsa_bwd import sparse_mla_bwd_v4_triton
from .dsa_fwd import sparse_mla_fwd_v4_triton

__all__ = [
    "csa_pool_attention_triton_v2",
    "sparse_mla_fwd_v4_triton",
    "sparse_mla_bwd_v4_triton",
]
