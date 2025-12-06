###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch

from primus_turbo.pytorch.core.backend import BackendType, KernelBackend

_COMMON_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


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
    ) -> bool:
        supported = True
        supported &= a.dim() == 2 and b.dim() == 3
        supported &= a.dtype in _COMMON_SUPPORTED_DTYPES and b.dtype in _COMMON_SUPPORTED_DTYPES
        supported &= trans_a == False
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
    ) -> torch.Tensor:
        return torch.ops.primus_turbo_cpp_extension.grouped_gemm(
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
    ) -> bool:
        # TODO: add more checks
        return True

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
    ) -> torch.Tensor:
        if trans_c:
            lhs, rhs = b, a
            trans_lhs, trans_rhs = not trans_b, not trans_a
        else:
            lhs, rhs = a, b
            trans_lhs, trans_rhs = trans_a, trans_b
        return torch.ops.primus_turbo_cpp_extension.grouped_gemm_variable_k(
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
    ) -> bool:
        # TODO: add more checks
        return True

    @staticmethod
    def execute(
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        num_cu: int | None,
    ) -> torch.Tensor:
        # TODO: sync
        torch.cuda.synchronize()

        return torch.ops.primus_turbo_cpp_extension.hipblaslt_grouped_gemm(
            a, b, group_lens, group_offs, trans_a, trans_b
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
    ) -> bool:
        # TODO: add more checks
        return True

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
    ) -> torch.Tensor:
        if trans_c:
            lhs, rhs = b, a
            trans_lhs, trans_rhs = not trans_b, not trans_a
        else:
            lhs, rhs = a, b
            trans_lhs, trans_rhs = trans_a, trans_b

        # TODO: ???
        lhs_t = lhs.t().contiguous()
        rhs_t = rhs.t().contiguous()

        return torch.ops.primus_turbo_cpp_extension.hipblaslt_grouped_gemm_variable_k(
            lhs_t, rhs_t, group_lens, group_offs, trans_lhs, trans_rhs
        )


_GROUPED_GEMM_BACKENDS = {
    BackendType.CK: GroupedGEMMCKBackend,
    BackendType.HIPBLASLT: GroupedGEMMHipblasltBackend,
}

_GROUPED_GEMM_VARIABLE_K_BACKENDS = {
    BackendType.CK: GroupedGEMMVariableKCKBackend,
    BackendType.HIPBLASLT: GroupedGEMMVariableKHipblasltBackend,
}


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
) -> torch.Tensor:
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 3, f"b must be 3D, got {b.shape}"
    assert a.dtype in [torch.float16, torch.bfloat16], f"a must be float16 or bfloat16, got {a.dtype}"
    assert b.dtype in [torch.float16, torch.bfloat16], f"b must be float16 or bfloat16, got {b.dtype}"
    assert trans_a == False

    default_backend_enum = BackendType(default_backend)
    backend_cls = _GROUPED_GEMM_BACKENDS.get(default_backend_enum, None)

    kwargs = dict(
        a=a,
        b=b,
        group_lens=group_lens,
        group_offs=group_offs,
        trans_a=trans_a,
        trans_b=trans_b,
        num_cu=num_cu,
    )

    return backend_cls.execute(**kwargs)


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
) -> torch.Tensor:
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 2, f"b must be 2D, got {b.shape}"
    assert a.dtype in [torch.float16, torch.bfloat16], f"a must be float16 or bfloat16, got {a.dtype}"
    assert b.dtype in [torch.float16, torch.bfloat16], f"b must be float16 or bfloat16, got {b.dtype}"
    assert trans_a == True and trans_b == False, "Only trans_a=True and trans_b=False are supported."

    default_backend_enum = BackendType(default_backend)
    backend_cls = _GROUPED_GEMM_VARIABLE_K_BACKENDS.get(default_backend_enum, None)

    kwargs = dict(
        a=a,
        b=b,
        group_lens=group_lens,
        group_offs=group_offs,
        trans_a=trans_a,
        trans_b=trans_b,
        trans_c=trans_c,
        num_cu=num_cu,
    )
    return backend_cls.execute(**kwargs)


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
) -> torch.Tensor:
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 3, f"b must be 3D, got {b.shape}"
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
) -> torch.Tensor:
    bs = group_lens.shape[0]
    m = a.shape[1] if trans_a else a.shape[0]
    n = b.shape[-2] if trans_b else b.shape[-1]
    if trans_c:
        m, n = n, m
    return torch.empty((bs, m, n), device=a.device, dtype=a.dtype)
