###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL DeepSeek-V4 attention kernels (ported from Primus).

Exposes the forward / backward launcher entry points consumed by the
``kernels/attention/deepseek_attn_impl.py`` dispatcher:

* :func:`hca_attention_fwd_flydsl_kernel` — dense / SWA forward.
* :func:`csa_pool_attention_fwd_flydsl_kernel` — CSA fused forward with in-kernel
  gather from the compressed pool (local SWA + sparse top-K + optional sink).
* :func:`hca_attention_bwd_flydsl_kernel` — dense / SWA backward (MQA, bf16).

The underlying FlyDSL kernel *builders* (``build_*_module``) live in the
``kernels`` subpackage, ported from the Primus DeepSeek-V4 ``_flydsl/kernels``
suite.
"""

from primus_turbo.flydsl.attention.deepseek_attn_bwd_kernel import (
    hca_attention_bwd_flydsl_kernel,
)
from primus_turbo.flydsl.attention.deepseek_attn_fwd_kernel import (
    csa_pool_attention_fwd_flydsl_kernel,
    hca_attention_fwd_flydsl_kernel,
)

__all__ = [
    "hca_attention_fwd_flydsl_kernel",
    "csa_pool_attention_fwd_flydsl_kernel",
    "hca_attention_bwd_flydsl_kernel",
]
