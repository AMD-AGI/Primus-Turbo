###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import ctypes
import os
import struct
from typing import Tuple

import torch
import triton
from torch.library import triton_op, wrap_triton

from primus_turbo.triton.moe.fused_router_kernel import (
    fused_scaling_group_sum_routing_backward_kernel,
    fused_scaling_group_sum_routing_kernel,
)

# ── ASM .co fused router launcher ───────────────────────────────────────────
_USE_ASM_ROUTER = os.environ.get("PRIMUS_ASM_ROUTER", "0") == "1"
if _USE_ASM_ROUTER and int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) == 0:
    print("[PRIMUS-ASM] Fused Router ASM kernel ENABLED (PRIMUS_ASM_ROUTER=1)")

_ASM_CO_ROUTER_PATH = "/opt/asm_gemm/fused_router_opt.co"
_ASM_CO_ROUTER_KERNEL_NAME = "fused_scaling_group_sum_routing_kernel"
_ASM_CO_ROUTER_THREADS = 256
_ASM_CO_ROUTER_LDS_BYTES = 0
_ASM_CO_ROUTER_KERNARG_SIZE = 328

_ASM_CO_ROUTER_EXPECTED_S = 32768
_ASM_CO_ROUTER_EXPECTED_E = 32
_ASM_CO_ROUTER_EXPECTED_G = 1
_ASM_CO_ROUTER_EXPECTED_K = 4

_HIP_LIB: ctypes.CDLL | None = None


def _get_libhip() -> ctypes.CDLL:
    global _HIP_LIB
    if _HIP_LIB is None:
        for name in ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"):
            try:
                _HIP_LIB = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if _HIP_LIB is None:
            raise OSError("Cannot load libamdhip64.so")
    return _HIP_LIB


_HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
_HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(0x02)
_HIP_LAUNCH_PARAM_END = ctypes.c_void_p(0x03)

_ASM_CO_ROUTER_MODULE: ctypes.c_void_p | None = None
_ASM_CO_ROUTER_FUNC: ctypes.c_void_p | None = None


def _get_asm_co_router_func() -> ctypes.c_void_p:
    global _ASM_CO_ROUTER_MODULE, _ASM_CO_ROUTER_FUNC
    if _ASM_CO_ROUTER_FUNC is None:
        hip = _get_libhip()
        mod = ctypes.c_void_p()
        rc = hip.hipModuleLoad(ctypes.byref(mod), _ASM_CO_ROUTER_PATH.encode())
        if rc != 0:
            hip.hipGetErrorString.restype = ctypes.c_char_p
            raise RuntimeError(
                f"hipModuleLoad({_ASM_CO_ROUTER_PATH}) failed rc={rc}: "
                f"{hip.hipGetErrorString(rc)}"
            )
        func = ctypes.c_void_p()
        rc = hip.hipModuleGetFunction(
            ctypes.byref(func), mod, _ASM_CO_ROUTER_KERNEL_NAME.encode()
        )
        if rc != 0:
            hip.hipGetErrorString.restype = ctypes.c_char_p
            raise RuntimeError(
                f"hipModuleGetFunction({_ASM_CO_ROUTER_KERNEL_NAME}) failed rc={rc}: "
                f"{hip.hipGetErrorString(rc)}"
            )
        _ASM_CO_ROUTER_MODULE = mod
        _ASM_CO_ROUTER_FUNC = func
    return _ASM_CO_ROUTER_FUNC


def _launch_asm_co_router(
    input_logit: torch.Tensor,
    output_scores: torch.Tensor,
    output_topk_idx: torch.Tensor,
    output_raw_topk_logits: torch.Tensor,
    output_probs: torch.Tensor,
    output_routing_map: torch.Tensor,
    scaling_factor: float,
    grid_x: int,
) -> None:
    func = _get_asm_co_router_func()
    hip = _get_libhip()

    buf = ctypes.create_string_buffer(_ASM_CO_ROUTER_KERNARG_SIZE)
    struct.pack_into(
        "<QQQQQQ", buf, 0,
        input_logit.data_ptr(), output_scores.data_ptr(),
        output_topk_idx.data_ptr(), output_raw_topk_logits.data_ptr(),
        output_probs.data_ptr(), output_routing_map.data_ptr(),
    )
    struct.pack_into("<f", buf, 48, scaling_factor)
    struct.pack_into("<QQ", buf, 56, 0, 0)
    struct.pack_into("<III", buf, 72, grid_x, 1, 1)
    struct.pack_into("<HHH", buf, 84, _ASM_CO_ROUTER_THREADS, 1, 1)
    struct.pack_into("<HHH", buf, 90, 0, 0, 0)
    struct.pack_into("<QQQ", buf, 112, 0, 0, 0)
    struct.pack_into("<H", buf, 136, 1)

    arg_size = ctypes.c_size_t(_ASM_CO_ROUTER_KERNARG_SIZE)
    config = (ctypes.c_void_p * 5)(
        _HIP_LAUNCH_PARAM_BUFFER_POINTER,
        ctypes.cast(buf, ctypes.c_void_p),
        _HIP_LAUNCH_PARAM_BUFFER_SIZE,
        ctypes.cast(ctypes.pointer(arg_size), ctypes.c_void_p),
        _HIP_LAUNCH_PARAM_END,
    )

    stream = torch.cuda.current_stream().cuda_stream
    rc = hip.hipModuleLaunchKernel(
        func,
        grid_x, 1, 1,
        _ASM_CO_ROUTER_THREADS, 1, 1,
        _ASM_CO_ROUTER_LDS_BYTES,
        ctypes.c_void_p(stream),
        None,
        config,
    )
    if rc != 0:
        hip.hipGetErrorString.restype = ctypes.c_char_p
        raise RuntimeError(
            f"hipModuleLaunchKernel (fused_router) failed rc={rc}: "
            f"{hip.hipGetErrorString(rc)}"
        )


def fused_moe_router_fwd(
    logits: torch.Tensor,
    s: int,
    e: int,
    groups: int,
    topk: int,
    selected_groups: int,
    score_function: str,
    scaling_factor: float,
):
    return torch.ops.primus_turbo.fused_moe_router_fwd_triton.default(
        logits, s, e, groups, topk, selected_groups, score_function, scaling_factor
    )


def fused_moe_router_bkwd(
    g_probs: torch.Tensor,
    g_scores: torch.Tensor,
    logits: torch.Tensor,
    output_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    raw_topk_logits: torch.Tensor,
    out_scores: torch.Tensor,
    routing_map: torch.Tensor,
    score_function: str,
    scaling_factor: float,
):
    return torch.ops.primus_turbo.fused_moe_router_bkwd_triton.default(
        g_probs,
        g_scores,
        logits,
        output_probs,
        topk_indices,
        raw_topk_logits,
        out_scores,
        routing_map,
        score_function,
        scaling_factor,
    )


@triton_op("primus_turbo::fused_moe_router_fwd_triton", mutates_args={})
def fused_moe_router_fwd_triton(
    logits: torch.Tensor,
    s: int,
    e: int,
    groups: int,
    topk: int,
    selected_groups: int,
    score_function: str,
    scaling_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # todo add warmup
    num_stages = 2
    num_programs = s

    E_ALIGNED = triton.next_power_of_2(e)
    INNER_GROUP_K_ALIGNED = triton.next_power_of_2(topk // selected_groups)

    raw_topk_logits = torch.empty((s, topk), device="cuda", dtype=logits.dtype)
    output_topk_indices = torch.ones((s, topk), device="cuda", dtype=torch.int64)
    output_scores = torch.empty((s, e), device="cuda", dtype=logits.dtype)

    output_probs = torch.zeros((s, e), device="cuda", dtype=logits.dtype)
    output_routing_map = torch.zeros((s, e), device="cuda", dtype=torch.int32)

    if (
        _USE_ASM_ROUTER
        and s == _ASM_CO_ROUTER_EXPECTED_S
        and e == _ASM_CO_ROUTER_EXPECTED_E
        and groups == _ASM_CO_ROUTER_EXPECTED_G
        and topk == _ASM_CO_ROUTER_EXPECTED_K
        and score_function == "sigmoid"
    ):
        _launch_asm_co_router(
            logits, output_scores, output_topk_indices, raw_topk_logits,
            output_probs, output_routing_map, scaling_factor, num_programs,
        )
    else:
        wrap_triton(fused_scaling_group_sum_routing_kernel)[(num_programs,)](
            logits,
            output_scores,
            output_topk_indices,
            raw_topk_logits,
            output_probs,
            output_routing_map,
            s,
            e,
            groups,
            topk,
            selected_groups,
            E_ALIGNED,
            INNER_GROUP_K_ALIGNED,
            num_stages,
            0 if score_function == "sigmoid" else 1,
            scaling_factor,
        )

    return output_scores, output_topk_indices, raw_topk_logits, output_probs, output_routing_map


@triton_op("primus_turbo::fused_moe_router_bkwd_triton", mutates_args={})
def fused_moe_router_bkwd_triton(
    g_probs: torch.Tensor,
    g_scores: torch.Tensor,
    logits: torch.Tensor,
    output_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    raw_topk_logits: torch.Tensor,
    out_scores: torch.Tensor,
    routing_map: torch.Tensor,
    score_function: str,
    scaling_factor: float,
) -> torch.Tensor:
    s, e = out_scores.shape
    k = topk_indices.shape[1]

    num_stages = 2
    num_programs = s

    E_ALIGNED = triton.next_power_of_2(e)
    K_ALIGNED = triton.next_power_of_2(k)

    g_probs = g_probs.contiguous()
    g_scores = g_scores.contiguous()

    output_g_probs = torch.zeros_like(g_probs)
    output_g_scores = torch.empty_like(g_scores)

    wrap_triton(fused_scaling_group_sum_routing_backward_kernel)[(num_programs,)](
        g_probs,
        g_scores,
        logits,
        output_probs,
        topk_indices,
        raw_topk_logits,
        out_scores,
        routing_map,
        output_g_probs,
        output_g_scores,
        s,
        e,
        k,
        K_ALIGNED,
        E_ALIGNED,
        num_stages,
        0 if score_function == "sigmoid" else 1,
        scaling_factor,
    )

    output_g_logits = output_g_probs + output_g_scores

    return output_g_logits
