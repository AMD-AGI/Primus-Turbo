###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL flash/flex attention, structured after flash-attention's ``flash_attn/cute``
(FA4), for MI300 (gfx942) and MI355 (gfx950). See ``flash_fwd.py``."""

import torch

from primus_turbo.flydsl.flex_attention.flash_fwd import FlashAttentionForward
from primus_turbo.flydsl.flex_attention.mods import make_softcap_mods
from primus_turbo.flydsl.flex_attention.torch_utils import (
    BlockSparseTensors,
    dense_to_block_sparse,
    flash_attn_combine,
)


def FlashAttentionBackward(*args, **kwargs):  # noqa: N802 - constructor-style factory
    """The deterministic backward builder for the current device (same arguments for both).
    ``flash_bwd_gfx942.py`` (MI300) and ``flash_bwd_gfx950.py`` (MI355) share the design
    (separate dQ and dK/dV kernels, each owning its output, after FA4's ``flash_bwd.py``) but
    are tuned and evolve separately."""
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    if "gfx95" in arch:
        from primus_turbo.flydsl.flex_attention.flash_bwd_gfx950 import FlashAttentionBackward as impl
    else:
        from primus_turbo.flydsl.flex_attention.flash_bwd_gfx942 import FlashAttentionBackward as impl
    return impl(*args, **kwargs)


__all__ = [
    "BlockSparseTensors",
    "FlashAttentionBackward",
    "FlashAttentionForward",
    "dense_to_block_sparse",
    "flash_attn_combine",
    "make_softcap_mods",
]
