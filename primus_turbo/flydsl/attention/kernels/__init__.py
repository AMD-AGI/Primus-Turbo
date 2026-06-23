###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 attention FlyDSL kernel modules (ported from Primus).

These are the FlyDSL kernel *builders* (``build_*_module``) ported verbatim
from the Primus DeepSeek-V4 ``_flydsl/kernels`` suite. Only the imports
were adapted: ``kernels.kernels_common`` -> the vendored ``kernels_common`` here,
and the sibling-by-sys.path imports -> package-relative imports. The launcher
adapters that wrap these builders for the Primus-Turbo dispatcher live one level
up in ``deepseek_attn_fwd_kernel.py`` / ``deepseek_attn_bwd_kernel.py``.
"""
