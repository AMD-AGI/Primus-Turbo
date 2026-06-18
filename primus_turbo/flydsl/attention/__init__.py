###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL DeepSeek-V4 attention kernels (design §10 落地文件清单).

Exposes the forward / backward launcher entry points consumed by the
``kernels/attention/deepseek_attn_impl.py`` dispatcher. The backward launcher
is a deferred stub (design §5.4 / §7.3 round 10) that raises until the FlyDSL
backward is implemented; the forward is the round-1 bring-up baseline.
"""

from primus_turbo.flydsl.attention.deepseek_attn_bwd_kernel import (
    hca_attention_bwd_flydsl_kernel,
)
from primus_turbo.flydsl.attention.deepseek_attn_fwd_kernel import (
    hca_attention_fwd_flydsl_kernel,
)

__all__ = [
    "hca_attention_fwd_flydsl_kernel",
    "hca_attention_bwd_flydsl_kernel",
]
