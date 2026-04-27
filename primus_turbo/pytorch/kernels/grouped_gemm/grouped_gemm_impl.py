###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch

from primus_turbo.pytorch.core.backend import (
    BackendEntry,
    BackendType,
    GlobalBackendManager,
    KernelBackend,
    PrecisionType,
    TuneCache,
)
from primus_turbo.pytorch.kernels.gemm.gemm_impl import _load_hipkitten_module
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import (
    BaseGroupedGEMMKernelDispatcher,
    BaseGroupedGEMMVariableKKernelDispatcher,
)
from primus_turbo.triton.grouped_gemm.grouped_gemm_kernel import (
    grouped_gemm_triton_kernel,
    grouped_gemm_variable_k_triton_kernel,
)

_COMMON_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_HIPKITTEN_SUPPORTED_DTYPES = (torch.bfloat16,)


def _hipkitten_grouped_cfg(m: int, n: int, k: int, layout: str) -> tuple[int, int]:
    _, cache = _load_hipkitten_module()
    return cache.get((m, n, k), {}).get(layout, (1, 2))


def _aligned_for_hipkitten(m: int, n: int, k: int) -> bool:
    return m % 256 == 0 and n % 256 == 0 and k % 64 == 0


def _grouped_bf16_supported(m: int, n: int, k: int, layout: str) -> bool:
    if not _aligned_for_hipkitten(m, n, k):
        return False
    # Validated under benchmark strict allclose gate.
    if layout == "rcr":
        return (m, n, k) in {
            (4096, 4096, 7168),  # DeepSeek GateUP
        }
    if layout == "rrr":
        return (m, n, k) in {
            (4096, 2048, 7168),  # DeepSeek Down dA
            (4096, 4096, 7168),  # DeepSeek dA (RRR forward variant)
            (4096, 7168, 4096),  # DeepSeek GateUP dA
        }
    if layout == "crr":
        return (m, n, k) in {
            (4096, 7168, 4096),  # DeepSeek GateUP dB
            (7168, 2048, 4096),  # DeepSeek Down dB
        }
    return False


def _grouped_bf16_supported_with_groups(group_count: int, m: int, n: int, k: int, layout: str) -> bool:
    if group_count >= 32 and m < 4096:
        return False
    return _grouped_bf16_supported(m, n, k, layout)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _round_up(a: int, b: int) -> int:
    return _ceil_div(a, b) * b


def _hipkitten_bf16_padded_shape(m: int, n: int, k: int, layout: str) -> tuple[int, int, int]:
    m_pad = _round_up(m, 256)
    n_pad = _round_up(n, 256)
    k_pad = _round_up(k, 64)
    if layout == "rrr":
        n_pad = max(n_pad, 4096)
    if layout == "crr":
        k_pad = max(k_pad, 4096)
    return m_pad, n_pad, k_pad


def _pad_2d(x: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    if x.shape[0] == rows and x.shape[1] == cols:
        return x
    out = torch.zeros((rows, cols), dtype=x.dtype, device=x.device)
    out[: x.shape[0], : x.shape[1]] = x
    return out


def _group_offsets_cpu(group_offs: torch.Tensor) -> list[int]:
    return [int(x) for x in group_offs.detach().cpu().tolist()]


def _is_balanced_group_lens(group_lens: torch.Tensor) -> bool:
    vals = [int(x) for x in group_lens.detach().cpu().tolist()]
    return len(vals) > 0 and all(v == vals[0] for v in vals)


def _pad_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    if x.shape[0] >= rows:
        return x
    out = torch.zeros((rows, x.shape[1]), dtype=x.dtype, device=x.device)
    out[: x.shape[0]] = x
    return out


class GroupedGEMMCKBackend(KernelBackend):

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        num_cu: int | None,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.dim() == 2 and b.dim() == 3
        supported &= a.dtype in _COMMON_SUPPORTED_DTYPES and b.dtype in _COMMON_SUPPORTED_DTYPES
        supported &= not trans_a
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        num_cu: int | None,
        **kwargs,
    ) -> torch.Tensor:
        return torch.ops.primus_turbo_cpp_extension.ck_grouped_gemm(
            a, b, group_lens, group_offs, trans_a, trans_b, num_cu
        )


class GroupedGEMMVariableKCKBackend(KernelBackend):
    @staticmethod
    def can_handle(
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        trans_c: bool,
        num_cu: int | None,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.dim() == 2 and b.dim() == 2
        supported &= a.dtype in _COMMON_SUPPORTED_DTYPES and b.dtype in _COMMON_SUPPORTED_DTYPES
        supported &= trans_a and not trans_b
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        trans_c: bool,
        num_cu: int | None,
        **kwargs,
    ) -> torch.Tensor:
        if trans_c:
            lhs, rhs = b, a
            trans_lhs, trans_rhs = not trans_b, not trans_a
        else:
            lhs, rhs = a, b
            trans_lhs, trans_rhs = trans_a, trans_b
        return torch.ops.primus_turbo_cpp_extension.ck_grouped_gemm_variable_k(
            lhs, rhs, group_lens, group_offs, trans_lhs, trans_rhs, num_cu
        )


class GroupedGEMMHipblasltBackend(KernelBackend):
    @staticmethod
    def can_handle(
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        num_cu: int | None,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.dim() == 2 and b.dim() == 3
        supported &= a.dtype in _COMMON_SUPPORTED_DTYPES and b.dtype in _COMMON_SUPPORTED_DTYPES
        supported &= not trans_a
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        num_cu: int | None,
        maybe_pre_sync: bool = False,
    ) -> torch.Tensor:
        return torch.ops.primus_turbo_cpp_extension.hipblaslt_grouped_gemm(
            a, b, group_lens, group_offs, trans_a, trans_b, maybe_pre_sync
        )


class GroupedGEMMVariableKHipblasltBackend(KernelBackend):
    @staticmethod
    def can_handle(
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        trans_c: bool,
        num_cu: int | None,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.dim() == 2 and b.dim() == 2
        supported &= a.dtype in _COMMON_SUPPORTED_DTYPES and b.dtype in _COMMON_SUPPORTED_DTYPES
        supported &= trans_a and not trans_b
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        trans_c: bool,
        num_cu: int | None,
        maybe_pre_sync: bool = False,
    ) -> torch.Tensor:
        if trans_c:
            lhs, rhs = b, a
            trans_lhs, trans_rhs = not trans_b, not trans_a
        else:
            lhs, rhs = a, b
            trans_lhs, trans_rhs = trans_a, trans_b

        return torch.ops.primus_turbo_cpp_extension.hipblaslt_grouped_gemm(
            lhs, rhs, group_lens, group_offs, trans_lhs, trans_rhs, maybe_pre_sync
        )


class GroupedGEMMHipKittenBackend(KernelBackend):
    @staticmethod
    def can_handle(
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        num_cu: int | None,
        **kwargs,
    ) -> bool:
        if a.dim() != 2 or b.dim() != 3 or trans_a:
            return False
        if a.dtype not in _HIPKITTEN_SUPPORTED_DTYPES or b.dtype not in _HIPKITTEN_SUPPORTED_DTYPES:
            return False
        if not a.is_cuda or not b.is_cuda or a.device != b.device:
            return False
        if group_lens.numel() != b.shape[0] or a.shape[0] % b.shape[0] != 0:
            return False
        if not _is_balanced_group_lens(group_lens):
            return False
        m = a.shape[0] // b.shape[0]
        n = b.shape[-2] if trans_b else b.shape[-1]
        k = a.shape[1]
        layout = "rcr" if trans_b else "rrr"
        return _grouped_bf16_supported_with_groups(b.shape[0], m, n, k, layout)

    @staticmethod
    def execute(
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        num_cu: int | None,
        **kwargs,
    ) -> torch.Tensor:
        hipkitten, _ = _load_hipkitten_module()
        n = b.shape[-2] if trans_b else b.shape[-1]
        out = torch.zeros((a.shape[0], n), dtype=a.dtype, device=a.device)
        m = int(group_lens[0].item())
        k = a.shape[1]
        layout = "rcr" if trans_b else "rrr"
        if not _grouped_bf16_supported_with_groups(b.shape[0], m, n, k, layout):
            raise ValueError(f"HipKitten grouped GEMM unsupported shape M={m}, N={n}, K={k}")
        if trans_b:
            group_m, num_xcds = _hipkitten_grouped_cfg(m, n, k, "rcr")
            hipkitten.grouped_rcr_balanced(a.contiguous(), b.contiguous(), out, group_m, num_xcds)
        else:
            m_pad, n_pad, k_pad = _hipkitten_bf16_padded_shape(m, n, k, "rrr")
            group_m, num_xcds = _hipkitten_grouped_cfg(m_pad, n_pad, k_pad, "rrr")
            if (m_pad, n_pad, k_pad) == (m, n, k):
                hipkitten.grouped_rrr_balanced(a.contiguous(), b.contiguous(), out, group_m, num_xcds)
            else:
                # HipKittens BF16 kernels can have a one-time per-shape setup miss on the first call.
                # Warm up on scratch so the first real group is not the cold launch.
                warmup_out = torch.zeros((m_pad, n_pad), dtype=a.dtype, device=a.device)
                hipkitten.gemm_rrr(
                    _pad_2d(a[:m].contiguous(), m_pad, k_pad),
                    _pad_2d(b[0].contiguous(), k_pad, n_pad),
                    warmup_out,
                    group_m,
                    num_xcds,
                )
                offs = _group_offsets_cpu(group_offs)
                for group_idx in range(b.shape[0]):
                    start, end = offs[group_idx], offs[group_idx + 1]
                    out_pad = torch.zeros((m_pad, n_pad), dtype=a.dtype, device=a.device)
                    hipkitten.gemm_rrr(
                        _pad_2d(a[start:end].contiguous(), m_pad, k_pad),
                        _pad_2d(b[group_idx].contiguous(), k_pad, n_pad),
                        out_pad,
                        group_m,
                        num_xcds,
                    )
                    out[start:end].copy_(out_pad[:m, :n])
        return out


class GroupedGEMMVariableKHipKittenBackend(KernelBackend):
    @staticmethod
    def can_handle(
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        trans_c: bool,
        num_cu: int | None,
        **kwargs,
    ) -> bool:
        if a.dim() != 2 or b.dim() != 2:
            return False
        if a.dtype not in _HIPKITTEN_SUPPORTED_DTYPES or b.dtype not in _HIPKITTEN_SUPPORTED_DTYPES:
            return False
        if not a.is_cuda or not b.is_cuda or a.device != b.device:
            return False
        if not (trans_a and not trans_b and trans_c):
            return False
        if group_lens.numel() <= 0 or a.shape[0] % group_lens.numel() != 0:
            return False
        if not _is_balanced_group_lens(group_lens):
            return False
        m = a.shape[0] // group_lens.numel()
        n = b.shape[1]
        k = a.shape[1]
        return _grouped_bf16_supported(n, k, m, "crr")

    @staticmethod
    def execute(
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        trans_c: bool,
        num_cu: int | None,
        **kwargs,
    ) -> torch.Tensor:
        hipkitten, _ = _load_hipkitten_module()
        group_num = group_lens.numel()
        n = b.shape[1]
        k = a.shape[1]
        out = torch.zeros((group_num, n, k), dtype=a.dtype, device=a.device)
        m = int(group_lens[0].item())
        if not _grouped_bf16_supported(n, k, m, "crr"):
            raise ValueError(f"HipKitten grouped GEMM dB unsupported shape M={m}, N={n}, K={k}")
        group_m, num_xcds = _hipkitten_grouped_cfg(n, k, m, "crr")
        hipkitten.grouped_crr_balanced(b.contiguous(), a.contiguous(), out, group_m, num_xcds)
        return out


class GroupedGEMMTritonBackend(KernelBackend):
    """Triton persistent-kernel backend for grouped GEMM (CPU-sync-free)."""

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        num_cu: int | None,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.dim() == 2 and b.dim() == 3
        supported &= a.dtype in _COMMON_SUPPORTED_DTYPES and b.dtype in _COMMON_SUPPORTED_DTYPES
        supported &= not trans_a
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        num_cu: int | None,
        **kwargs,
    ) -> torch.Tensor:
        return grouped_gemm_triton_kernel(a, b, group_offs, trans_b=trans_b)


_GROUPED_GEMM_BACKENDS = {
    BackendType.CK: BackendEntry(GroupedGEMMCKBackend),
    BackendType.HIPBLASLT: BackendEntry(GroupedGEMMHipblasltBackend, autotune=False),
    BackendType.HIPKITTEN: BackendEntry(GroupedGEMMHipKittenBackend, autotune=False),
    BackendType.TRITON: BackendEntry(GroupedGEMMTritonBackend),
}


class GroupedGEMMVariableKTritonBackend(KernelBackend):
    """Triton persistent-kernel backend for variable-K grouped GEMM (backward pass)."""

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        trans_c: bool,
        num_cu: int | None,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.dim() == 2 and b.dim() == 2
        supported &= a.dtype in _COMMON_SUPPORTED_DTYPES and b.dtype in _COMMON_SUPPORTED_DTYPES
        supported &= trans_a and not trans_b
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        trans_c: bool,
        num_cu: int | None,
        **kwargs,
    ) -> torch.Tensor:
        if trans_c:
            lhs, rhs = b, a
        else:
            lhs, rhs = a, b
        return grouped_gemm_variable_k_triton_kernel(lhs, rhs, group_offs)


_GROUPED_GEMM_VARIABLE_K_BACKENDS = {
    BackendType.CK: BackendEntry(GroupedGEMMVariableKCKBackend),
    BackendType.HIPBLASLT: BackendEntry(GroupedGEMMVariableKHipblasltBackend, autotune=False),
    BackendType.HIPKITTEN: BackendEntry(GroupedGEMMVariableKHipKittenBackend, autotune=False),
    BackendType.TRITON: BackendEntry(GroupedGEMMVariableKTritonBackend),
}


class GroupedGEMMKernelDispatcher(BaseGroupedGEMMKernelDispatcher):
    _backends = _GROUPED_GEMM_BACKENDS
    _cache = TuneCache(1024)

    @classmethod
    def make_key(cls, a, b, group_lens, group_offs, trans_a, trans_b, num_cu, **kwargs):
        bs = b.shape[0]
        m = a.shape[1] if trans_a else a.shape[0]
        n = b.shape[-2] if trans_b else b.shape[-1]
        k = a.shape[0] if trans_a else a.shape[1]
        # bs, m, n, k, a.dtype, b.dtype, out_dtype, trans_a, trans_b, trans_c
        return (bs, m, n, k, a.dtype, b.dtype, a.dtype, trans_a, trans_b, False)


class GroupedGEMMVariableKKernelDispatcher(BaseGroupedGEMMVariableKKernelDispatcher):
    _backends = _GROUPED_GEMM_VARIABLE_K_BACKENDS
    _cache = TuneCache(1024)

    @classmethod
    def make_key(
        cls, a, b, group_lens, group_offs, trans_a, trans_b, trans_c, num_cu, maybe_pre_sync, **kwargs
    ):
        bs = group_lens.shape[0]
        m = a.shape[1] if trans_a else a.shape[0]
        n = b.shape[-2] if trans_b else b.shape[-1]
        k = a.shape[0] if trans_a else a.shape[1]
        if trans_c:
            m, n = n, m
        return (bs, m, n, k, a.dtype, b.dtype, a.dtype, trans_a, trans_b, trans_c, maybe_pre_sync)


_torch_custom_op_wrapper = torch.library.custom_op


@_torch_custom_op_wrapper("primus_turbo::grouped_gemm_impl", mutates_args=(), device_types="cuda")
def grouped_gemm_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    num_cu: int | None,
    default_backend: int,
    maybe_pre_sync: bool = False,
) -> torch.Tensor:
    default_backend_enum = BackendType(default_backend)
    user_backend_enum = GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.BF16_FP16_FP32)

    kwargs = dict(
        a=a,
        b=b,
        group_lens=group_lens,
        group_offs=group_offs,
        trans_a=trans_a,
        trans_b=trans_b,
        num_cu=num_cu,
        maybe_pre_sync=maybe_pre_sync,
    )

    return GroupedGEMMKernelDispatcher.dispatch(default_backend_enum, user_backend_enum, **kwargs)


@_torch_custom_op_wrapper("primus_turbo::grouped_gemm_variable_k_impl", mutates_args=(), device_types="cuda")
def grouped_gemm_variable_k_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    trans_c: bool,
    num_cu: int | None,
    default_backend: int,
    maybe_pre_sync: bool = False,
) -> torch.Tensor:
    default_backend_enum = BackendType(default_backend)
    user_backend_enum = GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.BF16_FP16_FP32)
    kwargs = dict(
        a=a,
        b=b,
        group_lens=group_lens,
        group_offs=group_offs,
        trans_a=trans_a,
        trans_b=trans_b,
        trans_c=trans_c,
        num_cu=num_cu,
        maybe_pre_sync=maybe_pre_sync,
    )
    return GroupedGEMMVariableKKernelDispatcher.dispatch(default_backend_enum, user_backend_enum, **kwargs)


@grouped_gemm_impl.register_fake
def grouped_gemm_impl_meta(
    a: torch.Tensor,
    b: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    num_cu: int | None,
    default_backend: int,
    maybe_pre_sync: bool = False,
) -> torch.Tensor:
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 3, f"b must be 3D, got {b.shape}"
    assert a.dtype in [torch.float16, torch.bfloat16], f"a must be float16 or bfloat16, got {a.dtype}"
    assert b.dtype in [torch.float16, torch.bfloat16], f"b must be float16 or bfloat16, got {b.dtype}"
    assert trans_a == False, "Only trans_a=False is supported."

    m = a.shape[1] if trans_a else a.shape[0]
    n = b.shape[-2] if trans_b else b.shape[-1]
    return torch.empty((m, n), device=a.device, dtype=a.dtype)


@grouped_gemm_variable_k_impl.register_fake
def grouped_gemm_variable_k_impl_meta(
    a: torch.Tensor,
    b: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    trans_c: bool,
    num_cu: int | None,
    default_backend: int,
    maybe_pre_sync: bool = False,
) -> torch.Tensor:
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 2, f"b must be 2D, got {b.shape}"
    assert a.dtype in [torch.float16, torch.bfloat16], f"a must be float16 or bfloat16, got {a.dtype}"
    assert b.dtype in [torch.float16, torch.bfloat16], f"b must be float16 or bfloat16, got {b.dtype}"
    assert trans_a and not trans_b, "Only trans_a=True and trans_b=False are supported."

    bs = group_lens.shape[0]
    m = a.shape[1] if trans_a else a.shape[0]
    n = b.shape[-2] if trans_b else b.shape[-1]
    if trans_c:
        m, n = n, m
    return torch.empty((bs, m, n), device=a.device, dtype=a.dtype)
