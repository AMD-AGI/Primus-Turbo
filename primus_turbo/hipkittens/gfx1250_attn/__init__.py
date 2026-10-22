###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""gfx1250 (MI455X) flash attention on HipKittens HIP kernels, head dim 128 (Llama-3.1-8B: GQA, causal).

The HIPKITTENS attention backend on gfx1250 (``primus_turbo.pytorch.kernels.attention.
attention_hk_gfx1250_impl``) calls ``impl.hk_attn_fwd`` / ``impl.hk_attn_bwd`` when
``impl.hk_unsupported_reason`` allows, and the FlyDSL gfx1250 kernels otherwise. The kernel modules
(``fwd_pp``, ``fwd_v0``, ``bwd_v0``) launch HIP code objects built at install time into
``bin/*.hsaco`` through ``hiprt`` (ctypes, torch's HIP runtime); ``hsaco`` builds and verifies them
against ``manifest.json``, and ``limits`` holds the shape rules. ``hip/include/cdna5`` vendors a
subset of HipKittens (MIT).

Importing this package imports nothing (it does not assign ``HSACO_DIR`` either: the attention lab
sets that attribute on the package before running this file). ``hk_attn_fwd``, ``hk_attn_bwd`` and
``hk_unsupported_reason`` import ``impl`` (torch) on first access.
"""

__all__ = ["hk_attn_bwd", "hk_attn_fwd", "hk_unsupported_reason"]


def __getattr__(name):
    if name in __all__:
        from . import impl

        return getattr(impl, name)
    raise AttributeError(name)
