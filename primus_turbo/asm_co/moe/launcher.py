###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Launch function for the fused MoE router ASM .co kernel.

Enable via ``PRIMUS_ASM_ROUTER=1``.  The kernel is only invoked when the
input shape matches the fixed specialization baked into the binary.
"""

import ctypes
import os
import struct

import torch

from primus_turbo.asm_co.hip_utils import asm_co_module_launch
from primus_turbo.asm_co.moe.loader import get_asm_co_router_func

__all__ = [
    "USE_ASM_ROUTER",
    "launch_asm_co_router",
]

# ── Feature flag ─────────────────────────────────────────────────────────────
USE_ASM_ROUTER: bool = os.environ.get("PRIMUS_ASM_ROUTER", "0") == "1"
if USE_ASM_ROUTER and int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) == 0:
    print("[PRIMUS-ASM] Fused Router ASM kernel ENABLED (PRIMUS_ASM_ROUTER=1)")

# ── Kernel launch parameters ──────────────────────────────────────────────────
_ASM_CO_ROUTER_THREADS       = 256
_ASM_CO_ROUTER_LDS_BYTES     = 0
_ASM_CO_ROUTER_KERNARG_SIZE  = 328

# Shape specialization baked into the binary
ASM_CO_ROUTER_EXPECTED_S = 32768
ASM_CO_ROUTER_EXPECTED_E = 32
ASM_CO_ROUTER_EXPECTED_G = 1
ASM_CO_ROUTER_EXPECTED_K = 4


def launch_asm_co_router(
    input_logit: torch.Tensor,
    output_scores: torch.Tensor,
    output_topk_idx: torch.Tensor,
    output_raw_topk_logits: torch.Tensor,
    output_probs: torch.Tensor,
    output_routing_map: torch.Tensor,
    scaling_factor: float,
    grid_x: int,
) -> None:
    """Launch the fused router ASM kernel via the HIP module API."""
    func = get_asm_co_router_func()

    buf = ctypes.create_string_buffer(_ASM_CO_ROUTER_KERNARG_SIZE)
    struct.pack_into(
        "<QQQQQQ", buf, 0,
        input_logit.data_ptr(), output_scores.data_ptr(),
        output_topk_idx.data_ptr(), output_raw_topk_logits.data_ptr(),
        output_probs.data_ptr(), output_routing_map.data_ptr(),
    )
    struct.pack_into("<f",  buf, 48, scaling_factor)
    struct.pack_into("<QQ", buf, 56, 0, 0)
    struct.pack_into("<III", buf, 72, grid_x, 1, 1)
    struct.pack_into("<HHH", buf, 84, _ASM_CO_ROUTER_THREADS, 1, 1)
    struct.pack_into("<HHH", buf, 90, 0, 0, 0)
    struct.pack_into("<QQQ", buf, 112, 0, 0, 0)
    struct.pack_into("<H",  buf, 136, 1)

    asm_co_module_launch(
        func, buf, _ASM_CO_ROUTER_KERNARG_SIZE,
        grid_x, _ASM_CO_ROUTER_THREADS, _ASM_CO_ROUTER_LDS_BYTES,
        input_logit.device, "fused_router",
    )
