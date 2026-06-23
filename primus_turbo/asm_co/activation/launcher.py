###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Launch function for the SwiGLU BWD ASM .co kernel.

Enable via ``PRIMUS_ASM_SWIGLU_BWD=1``.  The kernel is only invoked when the
input shape matches the fixed specialization baked into the binary.
"""

import ctypes
import os
import struct

import torch

from primus_turbo.asm_co.hip_utils import asm_co_module_launch
from primus_turbo.asm_co.activation.loader import get_asm_co_swiglu_bwd_func

__all__ = [
    "USE_ASM_SWIGLU_BWD",
    "launch_asm_co_swiglu_bwd",
]

# ── Feature flag ─────────────────────────────────────────────────────────────
USE_ASM_SWIGLU_BWD: bool = os.environ.get("PRIMUS_ASM_SWIGLU_BWD", "0") == "1"
if USE_ASM_SWIGLU_BWD and int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) == 0:
    print("[PRIMUS-ASM] SwiGLU BWD ASM kernel ENABLED (PRIMUS_ASM_SWIGLU_BWD=1)")

# ── Kernel launch parameters ──────────────────────────────────────────────────
_ASM_CO_SWIGLU_BWD_THREADS      = 256
_ASM_CO_SWIGLU_BWD_LDS_BYTES    = 16
_ASM_CO_SWIGLU_BWD_KERNARG_SIZE = 80

# Shape specialization baked into the binary
ASM_CO_SWIGLU_BWD_EXPECTED_TOKENS     = 131072
ASM_CO_SWIGLU_BWD_EXPECTED_HIDDEN     = 4096
ASM_CO_SWIGLU_BWD_EXPECTED_BLOCK_SIZE = 8192


def launch_asm_co_swiglu_bwd(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    probs: torch.Tensor,
    row_mask: torch.Tensor,
    grad_x: torch.Tensor,
    grad_probs: torch.Tensor,
) -> None:
    """Launch the SwiGLU BWD ASM kernel via the HIP module API."""
    func = get_asm_co_swiglu_bwd_func()

    buf = ctypes.create_string_buffer(_ASM_CO_SWIGLU_BWD_KERNARG_SIZE)
    struct.pack_into(
        "<QQQQQQ", buf, 0,
        grad_out.data_ptr(), x.data_ptr(), probs.data_ptr(),
        row_mask.data_ptr(), grad_x.data_ptr(), grad_probs.data_ptr(),
    )
    struct.pack_into(
        "<iii", buf, 48,
        grad_out.stride(0), x.stride(0), grad_x.stride(0),
    )
    struct.pack_into("<I",  buf, 60, 0)
    struct.pack_into("<QQ", buf, 64, 0, 0)

    grid_x = ASM_CO_SWIGLU_BWD_EXPECTED_BLOCK_SIZE
    asm_co_module_launch(
        func, buf, _ASM_CO_SWIGLU_BWD_KERNARG_SIZE,
        grid_x, _ASM_CO_SWIGLU_BWD_THREADS, _ASM_CO_SWIGLU_BWD_LDS_BYTES,
        grad_out.device, "swiglu_bwd",
    )
