###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import ctypes
import os
import struct
from typing import Optional

import torch
import triton

from primus_turbo.triton.activation.swiglu_kernel import (
    swiglu_bwd_kernel,
    swiglu_fwd_kernel,
    swiglu_with_mask_bwd_kernel,
    swiglu_with_mask_fwd_kernel,
)

# ── ASM .co SwiGLU BWD launcher ─────────────────────────────────────────────
_USE_ASM_SWIGLU_BWD = os.environ.get("PRIMUS_ASM_SWIGLU_BWD", "0") == "1"
if _USE_ASM_SWIGLU_BWD and int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) == 0:
    print("[PRIMUS-ASM] SwiGLU BWD ASM kernel ENABLED (PRIMUS_ASM_SWIGLU_BWD=1)")

_ASM_CO_SWIGLU_BWD_PATH = "/opt/asm_gemm/swiglu_bwd_opt.co"
_ASM_CO_SWIGLU_BWD_KERNEL_NAME = "swiglu_with_mask_bwd_kernel"
_ASM_CO_SWIGLU_BWD_THREADS = 256
_ASM_CO_SWIGLU_BWD_LDS_BYTES = 16
_ASM_CO_SWIGLU_BWD_KERNARG_SIZE = 80

_ASM_CO_SWIGLU_BWD_EXPECTED_TOKENS = 131072
_ASM_CO_SWIGLU_BWD_EXPECTED_HIDDEN = 4096
_ASM_CO_SWIGLU_BWD_EXPECTED_BLOCK_SIZE = 8192

_HIP_LIB_SWIGLU: ctypes.CDLL | None = None


def _get_libhip_swiglu() -> ctypes.CDLL:
    global _HIP_LIB_SWIGLU
    if _HIP_LIB_SWIGLU is None:
        for name in ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"):
            try:
                _HIP_LIB_SWIGLU = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if _HIP_LIB_SWIGLU is None:
            raise OSError("Cannot load libamdhip64.so")
    return _HIP_LIB_SWIGLU


_HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
_HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(0x02)
_HIP_LAUNCH_PARAM_END = ctypes.c_void_p(0x03)

_ASM_CO_SWIGLU_BWD_MODULE: ctypes.c_void_p | None = None
_ASM_CO_SWIGLU_BWD_FUNC: ctypes.c_void_p | None = None


def _get_asm_co_swiglu_bwd_func() -> ctypes.c_void_p:
    global _ASM_CO_SWIGLU_BWD_MODULE, _ASM_CO_SWIGLU_BWD_FUNC
    if _ASM_CO_SWIGLU_BWD_FUNC is None:
        hip = _get_libhip_swiglu()
        mod = ctypes.c_void_p()
        rc = hip.hipModuleLoad(ctypes.byref(mod), _ASM_CO_SWIGLU_BWD_PATH.encode())
        if rc != 0:
            hip.hipGetErrorString.restype = ctypes.c_char_p
            raise RuntimeError(
                f"hipModuleLoad({_ASM_CO_SWIGLU_BWD_PATH}) failed rc={rc}: "
                f"{hip.hipGetErrorString(rc)}"
            )
        func = ctypes.c_void_p()
        rc = hip.hipModuleGetFunction(
            ctypes.byref(func), mod, _ASM_CO_SWIGLU_BWD_KERNEL_NAME.encode()
        )
        if rc != 0:
            hip.hipGetErrorString.restype = ctypes.c_char_p
            raise RuntimeError(
                f"hipModuleGetFunction({_ASM_CO_SWIGLU_BWD_KERNEL_NAME}) failed rc={rc}: "
                f"{hip.hipGetErrorString(rc)}"
            )
        _ASM_CO_SWIGLU_BWD_MODULE = mod
        _ASM_CO_SWIGLU_BWD_FUNC = func
    return _ASM_CO_SWIGLU_BWD_FUNC


def _launch_asm_co_swiglu_bwd(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    probs: torch.Tensor,
    row_mask: torch.Tensor,
    grad_x: torch.Tensor,
    grad_probs: torch.Tensor,
) -> None:
    func = _get_asm_co_swiglu_bwd_func()
    hip = _get_libhip_swiglu()

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
    struct.pack_into("<I", buf, 60, 0)
    struct.pack_into("<QQ", buf, 64, 0, 0)

    arg_size = ctypes.c_size_t(_ASM_CO_SWIGLU_BWD_KERNARG_SIZE)
    config = (ctypes.c_void_p * 5)(
        _HIP_LAUNCH_PARAM_BUFFER_POINTER,
        ctypes.cast(buf, ctypes.c_void_p),
        _HIP_LAUNCH_PARAM_BUFFER_SIZE,
        ctypes.cast(ctypes.pointer(arg_size), ctypes.c_void_p),
        _HIP_LAUNCH_PARAM_END,
    )

    grid_x = _ASM_CO_SWIGLU_BWD_EXPECTED_BLOCK_SIZE
    stream = torch.cuda.current_stream().cuda_stream
    rc = hip.hipModuleLaunchKernel(
        func,
        grid_x, 1, 1,
        _ASM_CO_SWIGLU_BWD_THREADS, 1, 1,
        _ASM_CO_SWIGLU_BWD_LDS_BYTES,
        ctypes.c_void_p(stream),
        None,
        config,
    )
    if rc != 0:
        hip.hipGetErrorString.restype = ctypes.c_char_p
        raise RuntimeError(
            f"hipModuleLaunchKernel (swiglu_bwd) failed rc={rc}: "
            f"{hip.hipGetErrorString(rc)}"
        )


def swiglu_fwd_with_probs(x: torch.Tensor, probs: torch.Tensor, row_mask: Optional[torch.Tensor] = None):
    num_tokens, double_hidden_size = x.size()

    probs = probs.unsqueeze(-1)

    out = torch.empty(num_tokens, double_hidden_size // 2, dtype=x.dtype, device=x.device)

    if row_mask is None:
        grid = (num_tokens,)
        swiglu_fwd_kernel[grid](
            x,
            probs,
            out,
            num_tokens=num_tokens,
            stride_x_token=x.stride(0),
            stride_probs_token=probs.stride(0),
            stride_out_token=out.stride(0),
            LOAD_WIDTH=triton.next_power_of_2(double_hidden_size // 2),
        )
    else:
        assert row_mask.is_cuda, "row_mask must be a CUDA tensor"

        BLOCK_SIZE = 8192
        grid = (BLOCK_SIZE,)
        swiglu_with_mask_fwd_kernel[grid](
            x,
            probs,
            row_mask,
            out,
            num_tokens=num_tokens,
            stride_x_token=x.stride(0),
            stride_probs_token=probs.stride(0),
            stride_out_token=out.stride(0),
            LOAD_WIDTH=triton.next_power_of_2(double_hidden_size // 2),
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out


def swiglu_bwd_with_probs(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    probs: torch.Tensor,
    row_mask: Optional[torch.Tensor] = None,
):
    num_tokens, hidden_size = grad_out.size()

    grad_x = torch.empty_like(x)
    grad_probs = torch.empty_like(probs)

    if row_mask is None:
        grid = (num_tokens,)
        swiglu_bwd_kernel[grid](
            grad_out,
            x,
            probs,
            grad_x,
            grad_probs,
            num_tokens=num_tokens,
            stride_grad_out_token=grad_out.stride(0),
            stride_x_token=x.stride(0),
            stride_probs_token=probs.stride(0),
            stride_grad_x_token=grad_x.stride(0),
            stride_grad_probs_token=grad_probs.stride(0),
            LOAD_WIDTH=triton.next_power_of_2(hidden_size),
        )
    else:
        assert row_mask.is_cuda, "tokens_per_expert must be a CUDA tensor"

        BLOCK_SIZE = 8192
        grid = (BLOCK_SIZE,)

        if (
            _USE_ASM_SWIGLU_BWD
            and num_tokens == _ASM_CO_SWIGLU_BWD_EXPECTED_TOKENS
            and hidden_size == _ASM_CO_SWIGLU_BWD_EXPECTED_HIDDEN
            and BLOCK_SIZE == _ASM_CO_SWIGLU_BWD_EXPECTED_BLOCK_SIZE
        ):
            _launch_asm_co_swiglu_bwd(
                grad_out, x, probs, row_mask, grad_x, grad_probs,
            )
        else:
            swiglu_with_mask_bwd_kernel[grid](
                grad_out,
                x,
                probs,
                row_mask,
                grad_x,
                grad_probs,
                num_tokens=num_tokens,
                stride_grad_out_token=grad_out.stride(0),
                stride_x_token=x.stride(0),
                stride_probs_token=probs.stride(0),
                stride_grad_x_token=grad_x.stride(0),
                stride_grad_probs_token=grad_probs.stride(0),
                LOAD_WIDTH=triton.next_power_of_2(hidden_size),
                BLOCK_SIZE=BLOCK_SIZE,
            )

    return grad_x, grad_probs
