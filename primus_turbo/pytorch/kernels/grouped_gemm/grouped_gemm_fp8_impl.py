###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import ctypes
import os
import struct

import torch

from primus_turbo.pytorch.core.backend import (
    BackendEntry,
    BackendType,
    GlobalBackendManager,
    KernelBackend,
    PrecisionType,
    TuneCache,
)
from primus_turbo.pytorch.core.low_precision import (
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import (
    BaseGroupedGEMMKernelDispatcher,
    BaseGroupedGEMMVariableKKernelDispatcher,
)
from primus_turbo.triton.grouped_gemm.grouped_gemm_fp8_kernel import (
    _compute_tile_cumsum_kernel,
    grouped_gemm_fp8_blockwise_triton_kernel,
    grouped_gemm_fp8_blockwise_variable_k_triton_kernel,
    grouped_gemm_fp8_rowwise_triton_kernel,
    grouped_gemm_fp8_rowwise_variable_k_triton_kernel,
    grouped_gemm_fp8_tensorwise_triton_kernel,
    grouped_gemm_fp8_tensorwise_variable_k_triton_kernel,
)

_COMMON_SUPPORTED_DTYPES = (
    (float8_e4m3, float8_e4m3, torch.float16),
    (float8_e4m3, float8_e4m3, torch.bfloat16),
    (float8_e5m2, float8_e5m2, torch.float16),
    (float8_e5m2, float8_e5m2, torch.bfloat16),
)

_HYBRID_SUPPORTED_DTYPES = (
    (float8_e4m3, float8_e5m2, torch.float16),
    (float8_e4m3, float8_e5m2, torch.bfloat16),
    (float8_e5m2, float8_e4m3, torch.float16),
    (float8_e5m2, float8_e4m3, torch.bfloat16),
)


class GroupedGEMMFP8CKBackend(KernelBackend):
    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
        ScalingGranularity.ROWWISE,
        ScalingGranularity.BLOCKWISE,
    }

    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES)

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        granularity: ScalingGranularity,
        num_cu: int | None,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.dim() == 2 and b.dim() == 3
        supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8CKBackend.SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8CKBackend.SUPPORTED_GRANULARITIES
        supported &= not trans_a
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        granularity: ScalingGranularity,
        num_cu: int | None,
        **kwargs,
    ):
        return torch.ops.primus_turbo_cpp_extension.ck_grouped_gemm_fp8(
            a,
            b,
            a_scales,
            b_scales,
            group_lens,
            group_offs,
            trans_a,
            trans_b,
            out_dtype,
            granularity.name,
            num_cu,
        )


class GroupedGEMMFP8VariableKCKBackend(KernelBackend):
    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
        ScalingGranularity.ROWWISE,
        ScalingGranularity.BLOCKWISE,
    }

    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES)

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        trans_c: bool,
        out_dtype: torch.dtype,
        granularity: ScalingGranularity,
        num_cu: int | None,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.dim() == 2 and b.dim() == 2
        supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8VariableKCKBackend.SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8VariableKCKBackend.SUPPORTED_GRANULARITIES
        supported &= trans_a and not trans_b
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        trans_c: bool,
        out_dtype: torch.dtype,
        granularity: ScalingGranularity,
        num_cu: int | None,
        **kwargs,
    ):
        if trans_c:
            lhs, rhs = b, a
            lhs_scales, rhs_scales = b_scales, a_scales
            trans_lhs, trans_rhs = not trans_b, not trans_a
        else:
            lhs, rhs = a, b
            lhs_scales, rhs_scales = a_scales, b_scales
            trans_lhs, trans_rhs = trans_a, trans_b
        return torch.ops.primus_turbo_cpp_extension.ck_grouped_gemm_fp8_variable_k(
            lhs,
            rhs,
            lhs_scales,
            rhs_scales,
            group_lens,
            group_offs,
            trans_lhs,
            trans_rhs,
            out_dtype,
            granularity.name,
            num_cu,
        )


class GroupedGEMMFP8HipblasltBackend(KernelBackend):
    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
    }

    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES + _HYBRID_SUPPORTED_DTYPES)

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        granularity: ScalingGranularity,
        num_cu: int | None,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.dim() == 2 and b.dim() == 3
        supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8HipblasltBackend.SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8HipblasltBackend.SUPPORTED_GRANULARITIES
        supported &= not trans_a
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        granularity: ScalingGranularity,
        num_cu: int | None,
        maybe_pre_sync: bool = False,
    ):
        return torch.ops.primus_turbo_cpp_extension.hipblaslt_grouped_gemm_fp8(
            a,
            b,
            a_scales,
            b_scales,
            group_lens,
            group_offs,
            trans_a,
            trans_b,
            out_dtype,
            granularity.name,
            maybe_pre_sync,
        )


class GroupedGEMMFP8VariableKHipblasltBackend(KernelBackend):
    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
    }

    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES + _HYBRID_SUPPORTED_DTYPES)

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        trans_c: bool,
        out_dtype: torch.dtype,
        granularity: ScalingGranularity,
        num_cu: int | None,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.dim() == 2 and b.dim() == 2
        supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8VariableKHipblasltBackend.SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8VariableKHipblasltBackend.SUPPORTED_GRANULARITIES
        supported &= trans_a and not trans_b
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        trans_c: bool,
        out_dtype: torch.dtype,
        granularity: ScalingGranularity,
        num_cu: int | None,
        maybe_pre_sync: bool = False,
    ):
        if trans_c:
            lhs, rhs = b, a
            lhs_scales, rhs_scales = b_scales, a_scales
            trans_lhs, trans_rhs = not trans_b, not trans_a
        else:
            lhs, rhs = a, b
            lhs_scales, rhs_scales = a_scales, b_scales
            trans_lhs, trans_rhs = trans_a, trans_b
        return torch.ops.primus_turbo_cpp_extension.hipblaslt_grouped_gemm_fp8(
            lhs,
            rhs,
            lhs_scales,
            rhs_scales,
            group_lens,
            group_offs,
            trans_lhs,
            trans_rhs,
            out_dtype,
            granularity.name,
            maybe_pre_sync,
        )


class GroupedGEMMFP8TritonBackend(KernelBackend):
    """Triton persistent-kernel backend for FP8 grouped GEMM (CPU-sync-free).

    Supports:
      - TENSORWISE: per-tensor scaling
      - ROWWISE: per-row/per-col vector scaling
      - BLOCKWISE: block-wise scaling (2D B_scales per group)
    """

    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
        ScalingGranularity.ROWWISE,
        ScalingGranularity.BLOCKWISE,
    }

    # tl.dot is dtype-agnostic and handles mixed e4m3/e5m2 FP8 natively on
    # gfx950 via v_mfma_f32_32x32x64_f8f6f4 with per-operand cbsz/blgp
    # modifiers, matching the hipBLASLt backend's hybrid dtype support.
    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES + _HYBRID_SUPPORTED_DTYPES)

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        granularity: ScalingGranularity,
        num_cu: int | None,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.dim() == 2 and b.dim() == 3
        supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8TritonBackend.SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8TritonBackend.SUPPORTED_GRANULARITIES
        supported &= not trans_a
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        granularity: ScalingGranularity,
        num_cu: int | None,
        **kwargs,
    ):
        if granularity == ScalingGranularity.BLOCKWISE:
            return grouped_gemm_fp8_blockwise_triton_kernel(
                a,
                b,
                a_scales,
                b_scales,
                group_offs,
                trans_b=trans_b,
                out_dtype=out_dtype,
            )
        elif granularity == ScalingGranularity.ROWWISE:
            return grouped_gemm_fp8_rowwise_triton_kernel(
                a,
                b,
                a_scales,
                b_scales,
                group_offs,
                trans_b=trans_b,
                out_dtype=out_dtype,
            )
        return grouped_gemm_fp8_tensorwise_triton_kernel(
            a,
            b,
            a_scales,
            b_scales,
            group_offs,
            trans_b=trans_b,
            out_dtype=out_dtype,
        )


# (K, N) shapes of the gpt-oss MoE FP8 fwd/dgrad call sites that the
# hand-tuned combined_v5.co kernel is specialised for.
# K = a.shape[1], N = b.shape[-2] if trans_b else b.shape[-1]
_ASM_CO_FWD_SITES = {
    (2880, 5760),  # gate_up_fwd
    (2880, 2880),  # down_fwd / down_dgrad
    (5760, 2880),  # gate_up_dgrad
}


# ── ASM .co shared HIP launcher infrastructure ────────────────────────────────
_ASM_CO_THREADS = 512
_ASM_CO_LDS_BYTES = 65536

_HIP_LIB: ctypes.CDLL | None = None

_HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
_HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(0x02)
_HIP_LAUNCH_PARAM_END = ctypes.c_void_p(0x03)


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
            raise OSError("Cannot load libamdhip64.so — not running inside a ROCm container?")
    return _HIP_LIB


def _asm_co_module_launch(
    func: ctypes.c_void_p,
    kernarg_buf: ctypes.Array,
    num_cu: int | None,
    device: torch.device,
    label: str,
    lds_bytes: int = _ASM_CO_LDS_BYTES,
) -> None:
    hip = _get_libhip()
    arg_size = ctypes.c_size_t(96)
    config = (ctypes.c_void_p * 5)(
        _HIP_LAUNCH_PARAM_BUFFER_POINTER,
        ctypes.cast(kernarg_buf, ctypes.c_void_p),
        _HIP_LAUNCH_PARAM_BUFFER_SIZE,
        ctypes.cast(ctypes.pointer(arg_size), ctypes.c_void_p),
        _HIP_LAUNCH_PARAM_END,
    )
    grid_cu = (
        num_cu if num_cu is not None else torch.cuda.get_device_properties(device).multi_processor_count
    )
    stream = torch.cuda.current_stream().cuda_stream
    rc = hip.hipModuleLaunchKernel(
        func,
        grid_cu,
        1,
        1,
        _ASM_CO_THREADS,
        1,
        1,
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


# ── ASM .co fwd/dgrad launcher ──────────────────────────────────────────────
_ASM_CO_FWD_CO_PATH = "/opt/asm_ggemm/combined_v5.co"
_ASM_CO_FWD_KERNEL_NAME = "_grouped_fp8_persistent_gemm_kernel"

_ASM_CO_FWD_MODULE: ctypes.c_void_p | None = None
_ASM_CO_FWD_FUNC: ctypes.c_void_p | None = None


def _get_asm_co_fwd_func() -> ctypes.c_void_p:
    """Load and cache the fwd/dgrad .co function handle."""
    global _ASM_CO_FWD_MODULE, _ASM_CO_FWD_FUNC
    if _ASM_CO_FWD_FUNC is None:
        hip = _get_libhip()
        mod = ctypes.c_void_p()
        rc = hip.hipModuleLoad(ctypes.byref(mod), _ASM_CO_FWD_CO_PATH.encode())
        if rc != 0:
            hip.hipGetErrorString.restype = ctypes.c_char_p
            raise RuntimeError(
                f"hipModuleLoad({_ASM_CO_FWD_CO_PATH}) failed rc={rc}: {hip.hipGetErrorString(rc)}"
            )
        func = ctypes.c_void_p()
        rc = hip.hipModuleGetFunction(ctypes.byref(func), mod, _ASM_CO_FWD_KERNEL_NAME.encode())
        if rc != 0:
            hip.hipGetErrorString.restype = ctypes.c_char_p
            raise RuntimeError(
                f"hipModuleGetFunction({_ASM_CO_FWD_KERNEL_NAME}) failed rc={rc}: "
                f"{hip.hipGetErrorString(rc)}"
            )
        _ASM_CO_FWD_MODULE = mod
        _ASM_CO_FWD_FUNC = func
    return _ASM_CO_FWD_FUNC


def _launch_asm_co_fwd_dgrad(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_offs: torch.Tensor,
    trans_b: bool,
    out_dtype: torch.dtype,
    num_cu: int | None,
) -> torch.Tensor:
    """Launch hand-tuned fwd/dgrad .co kernel via HIP module API.

    Kernarg layout (96 bytes) matches ``asm_co_grouped_gemm_fp8`` in asm_co_grouped_gemm.cpp.
    """
    m = a.shape[0]
    k = a.shape[1]
    e = b.shape[0]
    n = b.shape[1] if trans_b else b.shape[2]
    out = torch.empty((m, n), device=a.device, dtype=out_dtype)

    buf = ctypes.create_string_buffer(96)
    struct.pack_into(
        "<QQQQQQ",
        buf,
        0,
        a.data_ptr(),
        b.data_ptr(),
        out.data_ptr(),
        a_scales.data_ptr(),
        b_scales.data_ptr(),
        group_offs.data_ptr(),
    )
    struct.pack_into(
        "<iiiiiiii",
        buf,
        48,
        e,
        n,
        k,
        k,
        n * k,
        k,
        n,
        1,
    )

    _asm_co_module_launch(
        _get_asm_co_fwd_func(),
        buf,
        num_cu,
        a.device,
        f"fwd/dgrad K={k}, N={n}",
    )
    return out


class GroupedGEMMFP8ASMCOBackend(KernelBackend):
    """Hand-tuned AMDGCN assembly (.co/.hsaco) backend for FP8 grouped GEMM fwd/dgrad.

    Handles two code paths:
      - DGRAD (trans_b=True): uses combined_v5.co with 6-pointer kernarg layout
      - FWD   (trans_b=False): uses per-shape .hsaco files with 7-pointer kernarg
        layout (includes tile_cumsum_ptr)

    Activated when PRIMUS_TURBO_GROUPED_GEMM_BACKEND=ASM_CO. Tensorwise scaling
    only, E=32 experts, gpt-oss MoE shapes on MI355X (gfx950).
    """

    SUPPORTED_GRANULARITIES = {ScalingGranularity.TENSORWISE}
    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES + _HYBRID_SUPPORTED_DTYPES)
    _first_use_dgrad: bool = True
    _first_use_fwd: bool = True

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        granularity: ScalingGranularity,
        num_cu: int | None,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.dim() == 2 and b.dim() == 3
        supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8ASMCOBackend.SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8ASMCOBackend.SUPPORTED_GRANULARITIES
        supported &= not trans_a
        supported &= b.shape[0] == 32  # E
        k = a.shape[1]
        if trans_b:
            n = b.shape[-2]
            supported &= (k, n) in _ASM_CO_FWD_SITES
        else:
            n = b.shape[-1]
            supported &= (k, n) in _ASM_CO_FWD_SITES_NOTRANSB
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        granularity: ScalingGranularity,
        num_cu: int | None,
        **kwargs,
    ):
        if trans_b:
            if GroupedGEMMFP8ASMCOBackend._first_use_dgrad:
                GroupedGEMMFP8ASMCOBackend._first_use_dgrad = False
                print(
                    f"[ASM_CO] dgrad kernel first use — "
                    f"a={tuple(a.shape)} b={tuple(b.shape)} "
                    f"out_dtype={out_dtype} granularity={granularity.name}",
                    flush=True,
                )
            return _launch_asm_co_fwd_dgrad(
                a,
                b,
                a_scales,
                b_scales,
                group_offs,
                trans_b,
                out_dtype,
                num_cu,
            )
        else:
            if GroupedGEMMFP8ASMCOBackend._first_use_fwd:
                GroupedGEMMFP8ASMCOBackend._first_use_fwd = False
                print(
                    f"[ASM_CO] fwd kernel first use — "
                    f"a={tuple(a.shape)} b={tuple(b.shape)} "
                    f"out_dtype={out_dtype} granularity={granularity.name}",
                    flush=True,
                )
            return _launch_asm_co_fwd(
                a,
                b,
                a_scales,
                b_scales,
                group_offs,
                out_dtype,
                num_cu,
            )


# ── ASM .hsaco FWD launcher (uses tile_cumsum, trans_b=False) ────────────────
_ASM_CO_FWD_HSACO_DIR = "/opt/asm_ggemm"
_ASM_CO_FWD_KERNEL_NAME_HSACO = "_grouped_fp8_persistent_gemm_kernel"
_ASM_CO_FWD_LDS_BYTES = 131072

_ASM_CO_FWD_SITES_NOTRANSB = {
    (2880, 5760),  # gate_up_fwd
    (2880, 2880),  # down_fwd
}

_ASM_CO_FWD_HSACO_PATHS: dict[int, str] = {
    5760: os.path.join(_ASM_CO_FWD_HSACO_DIR, "reference_grouped_gemm_fwd_5760.hsaco"),
    2880: os.path.join(_ASM_CO_FWD_HSACO_DIR, "reference_grouped_gemm_fwd_2880.hsaco"),
}

_ASM_CO_FWD_MODULES: dict[int, ctypes.c_void_p] = {}
_ASM_CO_FWD_FUNCS: dict[int, ctypes.c_void_p] = {}


def _get_asm_co_fwd_hsaco_func(n: int) -> ctypes.c_void_p:
    """Load and cache the FWD .hsaco function handle keyed by N dimension."""
    if n not in _ASM_CO_FWD_FUNCS:
        path = _ASM_CO_FWD_HSACO_PATHS[n]
        hip = _get_libhip()
        mod = ctypes.c_void_p()
        rc = hip.hipModuleLoad(ctypes.byref(mod), path.encode())
        if rc != 0:
            hip.hipGetErrorString.restype = ctypes.c_char_p
            raise RuntimeError(
                f"hipModuleLoad({path}) failed rc={rc}: {hip.hipGetErrorString(rc)}"
            )
        func = ctypes.c_void_p()
        rc = hip.hipModuleGetFunction(
            ctypes.byref(func), mod, _ASM_CO_FWD_KERNEL_NAME_HSACO.encode()
        )
        if rc != 0:
            hip.hipGetErrorString.restype = ctypes.c_char_p
            raise RuntimeError(
                f"hipModuleGetFunction({_ASM_CO_FWD_KERNEL_NAME_HSACO}) from {path} "
                f"failed rc={rc}: {hip.hipGetErrorString(rc)}"
            )
        _ASM_CO_FWD_MODULES[n] = mod
        _ASM_CO_FWD_FUNCS[n] = func
    return _ASM_CO_FWD_FUNCS[n]


def _launch_asm_co_fwd(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_offs: torch.Tensor,
    out_dtype: torch.dtype,
    num_cu: int | None,
) -> torch.Tensor:
    """Launch hand-tuned FWD .hsaco kernel (trans_b=False) via HIP module API.

    Kernarg layout (88 bytes, 96-byte buffer):
      7 pointers: A, B, C, A_scale, B_scale, group_offs, tile_cumsum
      8 int32s:   G, N, K, stride_am, stride_bg, stride_bn, stride_cm, stride_cn
    """
    m = a.shape[0]
    k = a.shape[1]
    g = b.shape[0]
    n = b.shape[2]  # trans_b=False → N is last dim
    out = torch.empty((m, n), device=a.device, dtype=out_dtype)

    # Compute tile_cumsum on device (single-program triton kernel, no host sync)
    blk_m = 256  # BLOCK_SIZE_M baked into .hsaco
    blk_n = 256  # BLOCK_SIZE_N baked into .hsaco
    num_pid_n = (n + blk_n - 1) // blk_n
    tile_cumsum = torch.empty(g + 1, device=a.device, dtype=torch.int32)
    _compute_tile_cumsum_kernel[(1,)](
        group_offs,
        tile_cumsum,
        g,
        num_pid_n,
        BLOCK_SIZE_M=blk_m,
    )

    func = _get_asm_co_fwd_hsaco_func(n)

    buf = ctypes.create_string_buffer(96)
    struct.pack_into(
        "<QQQQQQQ",
        buf,
        0,
        a.data_ptr(),
        b.data_ptr(),
        out.data_ptr(),
        a_scales.data_ptr(),
        b_scales.data_ptr(),
        group_offs.data_ptr(),
        tile_cumsum.data_ptr(),
    )
    # stride_bn (=1) and stride_cn (=1) are compile-time specialized by Triton
    # and NOT present in the kernarg buffer.
    struct.pack_into(
        "<iiiiii",
        buf,
        56,
        g,
        n,
        k,
        a.stride(0),       # stride_am = K
        b.stride(0),       # stride_bg = K*N
        out.stride(0),     # stride_cm = N
    )

    _asm_co_module_launch(
        func,
        buf,
        num_cu,
        a.device,
        f"fwd K={k}, N={n}",
        lds_bytes=_ASM_CO_FWD_LDS_BYTES,
    )
    return out


class GroupedGEMMFP8KernelDispatcher(BaseGroupedGEMMKernelDispatcher):
    _backends = {
        BackendType.CK: BackendEntry(GroupedGEMMFP8CKBackend),
        BackendType.HIPBLASLT: BackendEntry(GroupedGEMMFP8HipblasltBackend, autotune=False),
        BackendType.TRITON: BackendEntry(GroupedGEMMFP8TritonBackend),
        BackendType.ASM_CO: BackendEntry(GroupedGEMMFP8ASMCOBackend, autotune=False),
    }
    _cache = TuneCache(1024)

    @classmethod
    def make_key(
        cls,
        a,
        b,
        a_scales,
        b_scales,
        group_lens,
        group_offs,
        trans_a,
        trans_b,
        out_dtype,
        granularity,
        num_cu,
        **kwargs,
    ):
        bs = b.shape[0]
        m = a.shape[1] if trans_a else a.shape[0]
        n = b.shape[-2] if trans_b else b.shape[-1]
        k = a.shape[0] if trans_a else a.shape[1]
        # bs, m, n, k, a.dtype, b.dtype, out_dtype, trans_a, trans_b, trans_c, granularity
        return (bs, m, n, k, a.dtype, b.dtype, out_dtype, trans_a, trans_b, False, granularity)


class GroupedGEMMFP8VariableKTritonBackend(KernelBackend):
    """Triton persistent-kernel backend for FP8 variable-K grouped GEMM (backward).

    Supports:
      - TENSORWISE: per-tensor scaling
      - ROWWISE: per-row/per-col vector scaling
      - BLOCKWISE: 1D+1D block-wise scaling (TN/CRR layout)
    """

    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
        ScalingGranularity.ROWWISE,
        ScalingGranularity.BLOCKWISE,
    }

    # See ``GroupedGEMMFP8TritonBackend.SUPPORTED_DTYPES`` for the rationale
    # behind including the hybrid e4m3/e5m2 pairs in the Triton backend.
    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES + _HYBRID_SUPPORTED_DTYPES)

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        trans_c: bool,
        out_dtype: torch.dtype,
        granularity: ScalingGranularity,
        num_cu: int | None,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.dim() == 2 and b.dim() == 2
        supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8VariableKTritonBackend.SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8VariableKTritonBackend.SUPPORTED_GRANULARITIES
        supported &= trans_a and not trans_b
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        trans_c: bool,
        out_dtype: torch.dtype,
        granularity: ScalingGranularity,
        num_cu: int | None,
        **kwargs,
    ):
        if trans_c:
            lhs, rhs = b, a
            lhs_scales, rhs_scales = b_scales, a_scales
        else:
            lhs, rhs = a, b
            lhs_scales, rhs_scales = a_scales, b_scales

        if granularity == ScalingGranularity.BLOCKWISE:
            return grouped_gemm_fp8_blockwise_variable_k_triton_kernel(
                lhs,
                rhs,
                lhs_scales,
                rhs_scales,
                group_offs,
                out_dtype=out_dtype,
            )
        elif granularity == ScalingGranularity.ROWWISE:
            return grouped_gemm_fp8_rowwise_variable_k_triton_kernel(
                lhs,
                rhs,
                lhs_scales,
                rhs_scales,
                group_offs,
                out_dtype=out_dtype,
            )
        return grouped_gemm_fp8_tensorwise_variable_k_triton_kernel(
            lhs,
            rhs,
            lhs_scales,
            rhs_scales,
            group_offs,
            out_dtype=out_dtype,
        )


# (OUT_M, OUT_N) shapes of the gpt-oss MoE FP8 wgrad call sites that the
# hand-tuned variable_k_wgrad_mega.co kernel is specialised for.
_ASM_CO_WGRAD_SITES = {
    (2880, 5760),  # gate_up_wgrad
    (2880, 2880),  # down_wgrad
}

# ── ASM .co wgrad launcher (variable-K) ─────────────────────────────────────
_ASM_CO_WGRAD_CO_PATH = "/opt/asm_ggemm/variable_k_wgrad_mega.co"
_ASM_CO_WGRAD_KERNEL_NAME = "_grouped_variable_k_gemm_kernel"

_ASM_CO_WGRAD_MODULE: ctypes.c_void_p | None = None
_ASM_CO_WGRAD_FUNC: ctypes.c_void_p | None = None


def _get_asm_co_wgrad_func() -> ctypes.c_void_p:
    """Load and cache the variable-K wgrad .co function handle."""
    global _ASM_CO_WGRAD_MODULE, _ASM_CO_WGRAD_FUNC
    if _ASM_CO_WGRAD_FUNC is None:
        hip = _get_libhip()
        mod = ctypes.c_void_p()
        rc = hip.hipModuleLoad(ctypes.byref(mod), _ASM_CO_WGRAD_CO_PATH.encode())
        if rc != 0:
            hip.hipGetErrorString.restype = ctypes.c_char_p
            raise RuntimeError(
                f"hipModuleLoad({_ASM_CO_WGRAD_CO_PATH}) failed rc={rc}: {hip.hipGetErrorString(rc)}"
            )
        func = ctypes.c_void_p()
        rc = hip.hipModuleGetFunction(ctypes.byref(func), mod, _ASM_CO_WGRAD_KERNEL_NAME.encode())
        if rc != 0:
            hip.hipGetErrorString.restype = ctypes.c_char_p
            raise RuntimeError(
                f"hipModuleGetFunction({_ASM_CO_WGRAD_KERNEL_NAME}) failed rc={rc}: "
                f"{hip.hipGetErrorString(rc)}"
            )
        _ASM_CO_WGRAD_MODULE = mod
        _ASM_CO_WGRAD_FUNC = func
    return _ASM_CO_WGRAD_FUNC


def _launch_asm_co_wgrad_variable_k(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_scale: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    out_dtype: torch.dtype,
    num_cu: int | None,
) -> torch.Tensor:
    """Launch hand-tuned variable-K wgrad .co kernel via HIP module API.

    Kernarg layout (96 bytes) matches ``KernArgs`` in asm_co_grouped_gemm.cpp.
    """
    out_m = lhs.shape[1]
    out_n = rhs.shape[1]
    g = group_lens.shape[0]
    out = torch.empty((g, out_m, out_n), device=lhs.device, dtype=out_dtype)

    func = _get_asm_co_wgrad_func()

    # 96-byte flat kernarg buffer (KernArgs in asm_co_grouped_gemm.cpp)
    buf = ctypes.create_string_buffer(96)
    struct.pack_into(
        "<QQQQQQ",
        buf,
        0,
        lhs.data_ptr(),
        rhs.data_ptr(),
        out.data_ptr(),
        lhs_scale.data_ptr(),
        rhs_scale.data_ptr(),
        group_offs.data_ptr(),
    )
    struct.pack_into(
        "<iiiiiiii",
        buf,
        48,
        g,
        out_m,
        out_n,
        lhs.stride(0),
        rhs.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )

    _asm_co_module_launch(
        func,
        buf,
        num_cu,
        lhs.device,
        f"wgrad OUT_M={out_m}, OUT_N={out_n}",
    )
    return out


class GroupedGEMMFP8VariableKASMCOBackend(KernelBackend):
    """Hand-tuned AMDGCN assembly (.co) backend for FP8 variable-K grouped GEMM (wgrad).

    Activated when PRIMUS_TURBO_GROUPED_GEMM_BACKEND=ASM_CO and the call
    arrives with is_bwd=True (wgrad pass). Tensorwise scaling only,
    E=32 experts, gpt-oss MoE shapes on MI355X (gfx950).
    """

    SUPPORTED_GRANULARITIES = {ScalingGranularity.TENSORWISE}
    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES + _HYBRID_SUPPORTED_DTYPES)
    _first_use: bool = True

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        trans_c: bool,
        out_dtype: torch.dtype,
        granularity: ScalingGranularity,
        num_cu: int | None,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.dim() == 2 and b.dim() == 2
        supported &= (
            a.dtype,
            b.dtype,
            out_dtype,
        ) in GroupedGEMMFP8VariableKASMCOBackend.SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8VariableKASMCOBackend.SUPPORTED_GRANULARITIES
        supported &= trans_a and not trans_b
        out_m = a.shape[1]
        out_n = b.shape[1]
        if trans_c:
            out_m, out_n = out_n, out_m
        supported &= (out_m, out_n) in _ASM_CO_WGRAD_SITES
        supported &= group_lens.shape[0] == 32  # E
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        trans_c: bool,
        out_dtype: torch.dtype,
        granularity: ScalingGranularity,
        num_cu: int | None,
        **kwargs,
    ):
        if trans_c:
            lhs, rhs = b, a
            lhs_scales, rhs_scales = b_scales, a_scales
        else:
            lhs, rhs = a, b
            lhs_scales, rhs_scales = a_scales, b_scales
        if GroupedGEMMFP8VariableKASMCOBackend._first_use:
            GroupedGEMMFP8VariableKASMCOBackend._first_use = False
            print(
                f"[ASM_CO] wgrad kernel first use — "
                f"a={tuple(a.shape)} b={tuple(b.shape)} "
                f"out_dtype={out_dtype} granularity={granularity.name}",
                flush=True,
            )
        return _launch_asm_co_wgrad_variable_k(
            lhs,
            rhs,
            lhs_scales,
            rhs_scales,
            group_lens,
            group_offs,
            out_dtype,
            num_cu,
        )


class GroupedGEMMFP8VariableKKernelDispatcher(BaseGroupedGEMMVariableKKernelDispatcher):
    _backends = {
        BackendType.CK: BackendEntry(GroupedGEMMFP8VariableKCKBackend),
        BackendType.HIPBLASLT: BackendEntry(GroupedGEMMFP8VariableKHipblasltBackend, autotune=False),
        BackendType.TRITON: BackendEntry(GroupedGEMMFP8VariableKTritonBackend),
        BackendType.ASM_CO: BackendEntry(GroupedGEMMFP8VariableKASMCOBackend, autotune=False),
    }
    _cache = TuneCache(1024)

    @classmethod
    def make_key(
        cls,
        a,
        b,
        a_scales,
        b_scales,
        group_lens,
        group_offs,
        trans_a,
        trans_b,
        trans_c,
        out_dtype,
        granularity,
        num_cu,
        **kwargs,
    ):
        bs = group_lens.shape[0]
        m = a.shape[1] if trans_a else a.shape[0]
        n = b.shape[-2] if trans_b else b.shape[-1]
        k = a.shape[0] if trans_a else a.shape[1]
        if trans_c:
            m, n = n, m
        return (bs, m, n, k, a.dtype, b.dtype, out_dtype, trans_a, trans_b, trans_c, granularity)


_torch_custom_op_wrapper = torch.library.custom_op


@_torch_custom_op_wrapper("primus_turbo::grouped_gemm_fp8_impl", mutates_args=(), device_types="cuda")
def grouped_gemm_fp8_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    out_dtype: torch.dtype,
    granularity: int,
    num_cu: int | None,
    default_backend: int,
    is_bwd: bool = False,
    maybe_pre_sync: bool = False,
) -> torch.Tensor:
    default_backend_enum = BackendType(default_backend)
    user_backend_enum = GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.FP8)
    granularity_enum = ScalingGranularity(granularity)

    # ASM_CO mode: both fwd and bwd use ASM .co/.hsaco kernels where shapes match,
    # with graceful fallback to Triton for unsupported shapes.
    if user_backend_enum == BackendType.ASM_CO:
        user_backend_enum = None
        default_backend_enum = BackendType.ASM_CO

    kwargs = dict(
        a=a,
        b=b,
        a_scales=a_scales,
        b_scales=b_scales,
        group_lens=group_lens,
        group_offs=group_offs,
        trans_a=trans_a,
        trans_b=trans_b,
        out_dtype=out_dtype,
        granularity=granularity_enum,
        num_cu=num_cu,
        maybe_pre_sync=maybe_pre_sync,
    )

    return GroupedGEMMFP8KernelDispatcher.dispatch(default_backend_enum, user_backend_enum, **kwargs)


@_torch_custom_op_wrapper(
    "primus_turbo::grouped_gemm_fp8_variable_k_impl", mutates_args=(), device_types="cuda"
)
def grouped_gemm_fp8_variable_k_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    trans_c: bool,
    out_dtype: torch.dtype,
    granularity: int,
    num_cu: int | None,
    default_backend: int,
    is_bwd: bool = False,
    maybe_pre_sync: bool = False,
) -> torch.Tensor:
    default_backend_enum = BackendType(default_backend)
    user_backend_enum = GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.FP8)
    granularity_enum = ScalingGranularity(granularity)

    # ASM_CO hybrid mode: wgrad is always backward, so use ASM as default
    # with graceful fallback to Triton when shapes are unsupported.
    if user_backend_enum == BackendType.ASM_CO:
        if is_bwd:
            user_backend_enum = None
            default_backend_enum = BackendType.ASM_CO
        else:
            user_backend_enum = BackendType.TRITON

    kwargs = dict(
        a=a,
        b=b,
        a_scales=a_scales,
        b_scales=b_scales,
        group_lens=group_lens,
        group_offs=group_offs,
        trans_a=trans_a,
        trans_b=trans_b,
        trans_c=trans_c,
        out_dtype=out_dtype,
        granularity=granularity_enum,
        num_cu=num_cu,
        maybe_pre_sync=maybe_pre_sync,
    )

    return GroupedGEMMFP8VariableKKernelDispatcher.dispatch(default_backend_enum, user_backend_enum, **kwargs)


def grouped_gemm_compute_offs(group_lens: torch.Tensor) -> torch.Tensor:
    group_offs = torch.ops.primus_turbo_cpp_extension.grouped_gemm_compute_offs(group_lens)
    return group_offs


@grouped_gemm_fp8_impl.register_fake
def grouped_gemm_fp8_impl_meta(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    out_dtype: torch.dtype,
    granularity: int,
    num_cu: int | None,
    default_backend: int,
    is_bwd: bool = False,
    maybe_pre_sync: bool = False,
) -> torch.Tensor:
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 3, f"b must be 3D, got {b.shape}"
    assert a.dtype in [float8_e4m3, float8_e5m2], f"a must be fp8, got {a.dtype}"
    assert b.dtype in [float8_e4m3, float8_e5m2], f"b must be fp8, got {b.dtype}"
    assert out_dtype in [
        torch.float16,
        torch.bfloat16,
    ], f"out_dtype must be float16 or bfloat16, got {out_dtype}"
    assert trans_a == False, "Only trans_a=False is supported."

    m = a.shape[1] if trans_a else a.shape[0]
    n = b.shape[-2] if trans_b else b.shape[-1]
    return torch.empty((m, n), device=a.device, dtype=out_dtype)


@grouped_gemm_fp8_variable_k_impl.register_fake
def grouped_gemm_fp8_variable_k_impl_meta(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    trans_c: bool,
    out_dtype: torch.dtype,
    granularity: int,
    num_cu: int | None,
    default_backend: int,
    is_bwd: bool = False,
    maybe_pre_sync: bool = False,
) -> torch.Tensor:
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 2, f"b must be 2D, got {b.shape}"
    assert a.dtype in [float8_e4m3, float8_e5m2], f"a must be fp8, got {a.dtype}"
    assert b.dtype in [float8_e4m3, float8_e5m2], f"b must be fp8, got {b.dtype}"
    assert out_dtype in [
        torch.float16,
        torch.bfloat16,
    ], f"out_dtype must be float16 or bfloat16, got {out_dtype}"
    assert trans_a and not trans_b, "Only trans_a=True and trans_b=False are supported."

    bs = group_lens.shape[0]
    m = a.shape[1] if trans_a else a.shape[0]
    n = b.shape[-2] if trans_b else b.shape[-1]
    if trans_c:
        m, n = n, m
    return torch.empty((bs, m, n), device=a.device, dtype=out_dtype)
