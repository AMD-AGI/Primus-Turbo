###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 attention Triton kernels (ported from Primus).

Two kernel families covering V4's three per-layer attention types:

* :mod:`hca_attention_fwd` / :mod:`hca_attention_bwd` — dense
  (``compress_ratio == 0``) and HCA (``compress_ratio == 128``) paths.
  HCA reuses the dense kernel via the split-mask branch
  (``hca_local_seqlen > 0``).
* :mod:`csa_attention_fwd` / :mod:`csa_attention_bwd` — CSA
  (``compress_ratio == 4``) fused local-SWA + per-query top-K sparse +
  shared sink path.

These are raw launchers (no autograd); the autograd Functions and the
public functional API live in
:mod:`primus_turbo.pytorch.ops.attention.hca_attention`.
"""

from primus_turbo.triton.attention.deepseek.hca_attention_bwd import (
    _launch_hca_attention_bwd,
)
from primus_turbo.triton.attention.deepseek.hca_attention_fwd import (
    _launch_hca_attention_fwd,
)
from primus_turbo.triton.attention.deepseek.csa_attention_bwd import (
    _launch_csa_attention_pool_bwd,
)
from primus_turbo.triton.attention.deepseek.csa_attention_fwd import (
    _launch_csa_attention_pool_fwd,
)

__all__ = [
    "_launch_hca_attention_fwd",
    "_launch_hca_attention_bwd",
    "_launch_csa_attention_pool_fwd",
    "_launch_csa_attention_pool_bwd",
]
