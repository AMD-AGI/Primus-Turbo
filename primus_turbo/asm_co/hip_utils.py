###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Shared HIP module-API utilities for ASM .co/.hsaco kernel management.

All ASM_CO sub-packages (grouped_gemm, moe, activation) import from here
instead of each carrying a private copy of the libhip loader.
"""

import ctypes

import torch

__all__ = [
    "HIP_LAUNCH_PARAM_BUFFER_POINTER",
    "HIP_LAUNCH_PARAM_BUFFER_SIZE",
    "HIP_LAUNCH_PARAM_END",
    "get_libhip",
    "load_co_func",
    "asm_co_module_launch",
]

# ── HIP launch-config sentinel tokens ────────────────────────────────────────
HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
HIP_LAUNCH_PARAM_BUFFER_SIZE    = ctypes.c_void_p(0x02)
HIP_LAUNCH_PARAM_END            = ctypes.c_void_p(0x03)

_HIP_LIB: ctypes.CDLL | None = None


def get_libhip() -> ctypes.CDLL:
    """Return a cached handle to libamdhip64, loading it on first call."""
    global _HIP_LIB
    if _HIP_LIB is None:
        for name in ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"):
            try:
                _HIP_LIB = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if _HIP_LIB is None:
            raise OSError("Cannot load libamdhip64.so — not running inside a ROCm container?")
    return _HIP_LIB


def load_co_func(co_path: str, kernel_name: str) -> tuple[ctypes.c_void_p, ctypes.c_void_p]:
    """Load a .co/.hsaco module and return (module_handle, function_handle).

    The caller is responsible for caching the returned handles.
    """
    hip = get_libhip()
    mod = ctypes.c_void_p()
    rc = hip.hipModuleLoad(ctypes.byref(mod), co_path.encode())
    if rc != 0:
        hip.hipGetErrorString.restype = ctypes.c_char_p
        raise RuntimeError(
            f"hipModuleLoad({co_path}) failed rc={rc}: {hip.hipGetErrorString(rc)}"
        )
    func = ctypes.c_void_p()
    rc = hip.hipModuleGetFunction(ctypes.byref(func), mod, kernel_name.encode())
    if rc != 0:
        hip.hipGetErrorString.restype = ctypes.c_char_p
        raise RuntimeError(
            f"hipModuleGetFunction({kernel_name}) from {co_path} "
            f"failed rc={rc}: {hip.hipGetErrorString(rc)}"
        )
    return mod, func


def asm_co_module_launch(
    func: ctypes.c_void_p,
    kernarg_buf: ctypes.Array,
    kernarg_size: int,
    grid_x: int,
    threads_x: int,
    lds_bytes: int,
    device: torch.device,
    label: str,
) -> None:
    """Launch a pre-loaded ASM kernel via hipModuleLaunchKernel.

    Args:
        func:          Function handle from :func:`load_co_func`.
        kernarg_buf:   ctypes buffer holding the packed kernel arguments.
        kernarg_size:  Size of *kernarg_buf* in bytes.
        grid_x:        Grid dimension X (number of workgroups).
        threads_x:     Block dimension X (wavefronts × 64 or SIMD width).
        lds_bytes:     Shared-memory (LDS) allocation in bytes.
        device:        CUDA/ROCm device used to obtain the current stream.
        label:         Human-readable name for error messages.
    """
    hip = get_libhip()
    arg_size = ctypes.c_size_t(kernarg_size)
    config = (ctypes.c_void_p * 5)(
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        ctypes.cast(kernarg_buf, ctypes.c_void_p),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        ctypes.cast(ctypes.pointer(arg_size), ctypes.c_void_p),
        HIP_LAUNCH_PARAM_END,
    )
    stream = torch.cuda.current_stream().cuda_stream
    rc = hip.hipModuleLaunchKernel(
        func,
        grid_x, 1, 1,
        threads_x, 1, 1,
        lds_bytes,
        ctypes.c_void_p(stream),
        None,
        config,
    )
    if rc != 0:
        hip.hipGetErrorString.restype = ctypes.c_char_p
        raise RuntimeError(
            f"hipModuleLaunchKernel ({label}) failed rc={rc}: {hip.hipGetErrorString(rc)}"
        )
