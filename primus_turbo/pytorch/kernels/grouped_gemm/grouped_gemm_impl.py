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
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    _autotune_pick,
)


# BF16 grouped binding-default is (group_m=4, num_xcds=8), distinct from
# FP8 (which defaults to num_xcds=0/binding-internal). The BF16 grouped
# kernel actually accepts a much wider XCD-swizzle space than FP8 — the
# old per-shape cell table picked num_xcds=32 6× and (gm=16, xcds=4) 3×
# for gpt_oss-class shapes. Include those extreme points so autotune can
# reach the same regime without baking in shape-specific rules.
# Larger group_m (16, 24) help over-saturated grids (B=32 m_total >=65k).
_HK_BF16_RCR_CANDIDATES: tuple[tuple[int, int], ...] = (
    (1, 4), (2, 4), (2, 32), (4, 4), (4, 8), (4, 32),
    (8, 4), (8, 8), (16, 4), (16, 8), (24, 2), (24, 4),
)
_HK_BF16_VARK_CANDIDATES: tuple[tuple[int, int], ...] = (
    (1, 4), (4, 4), (4, 8), (8, 4), (8, 8),
    # Extra group_m values for over-saturated grids (B=32 with M_total
    # >= 65536) where deeper M-side super-blocks improve B reuse. The
    # GateUP B=32 wgrad shape (m_total=65536, n=5760, k=2880) was the
    # most lossy at 0.95× vs Triton with the smaller sweep.
    (16, 4), (16, 8), (24, 4), (24, 8), (32, 4),
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import (
    BaseGroupedGEMMKernelDispatcher,
    BaseGroupedGEMMVariableKKernelDispatcher,
)
from primus_turbo.triton.utils.fp8_transpose import bf16_transpose_3d
from primus_turbo.triton.grouped_gemm.grouped_gemm_kernel import (
    grouped_gemm_triton_kernel,
    grouped_gemm_variable_k_triton_kernel,
)

_COMMON_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_HIPKITTEN_SUPPORTED_DTYPES = (torch.bfloat16,)


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
        # Hard constraints only — dtype, dim, device, B-divisibility.
        # No alignment gate: the HipKittens BF16 grouped main kernel
        # natively handles non-aligned (M, N, K) (main kernel sweeps
        # the BLOCK_SIZE-aligned interior; ``grouped_tail_kernel``
        # handles partial M / N / K cells on the same launch).
        if a.dim() != 2 or b.dim() != 3 or trans_a:
            return False
        if a.dtype not in _HIPKITTEN_SUPPORTED_DTYPES or b.dtype not in _HIPKITTEN_SUPPORTED_DTYPES:
            return False
        if not a.is_cuda or not b.is_cuda or a.device != b.device:
            return False
        if group_lens.numel() != b.shape[0] or a.shape[0] % b.shape[0] != 0:
            return False
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
        **kwargs,
    ) -> torch.Tensor:
        del trans_a, num_cu, kwargs
        bs = b.shape[0]
        m_total = a.shape[0]
        avg_m = max(m_total // bs, 1) if bs > 0 else max(m_total, 1)
        a_in = a if a.is_contiguous() else a.contiguous()
        # dgrad (trans_b=False) layout dispatch by kernel-K (= b.shape[1]):
        #   * b is [G, K, N] in HBM with K = N_fwd, N = K_fwd
        #   * Native RRR direct (kernel reads K-major B): correct via
        #     BF16_RRR_FUSE_PROBE=1, but B's K-stride access in the main
        #     K-loop misses L1/L2 every K-iter on shapes where K is
        #     small (= short K-loop / short compute window per tile to
        #     amortise the strided HBM reads).
        #   * H4 reroute (bf16_transpose_3d → RCR): pays ~30us upfront for
        #     the transpose copy but the resulting RCR call has K-major
        #     inner stride on B → cache-friendly. Wins for short K.
        #
        # Bench-derived crossover (gpt_oss-20B Balanced):
        #   GateUP (kernel-K=5760, kernel-N=2880): RRR direct 1038 > H4 885 → RRR
        #   Down   (kernel-K=2880, kernel-N=2880): RRR direct 528  < H4 696 → H4
        # Threshold 4096 splits the two cleanly. The threshold is on
        # kernel-K only — generic regime split, not a per-(M,N,K) cell.
        _BF16_RRR_DIRECT_MIN_K = 4096
        if trans_b:
            b_in = b if b.is_contiguous() else b.contiguous()
            op = torch.ops.primus_turbo_cpp_extension.hk_grouped_rcr_bf16
            n_out, k = b_in.shape[1], b_in.shape[2]
            key = ("rcr_bf16", m_total, n_out, k)
        elif b.shape[1] >= _BF16_RRR_DIRECT_MIN_K:
            # Native RRR direct: long K → main loop has enough compute
            # cycles to overlap the K-strided B HBM cache-miss latency.
            b_in = b if b.is_contiguous() else b.contiguous()
            op = torch.ops.primus_turbo_cpp_extension.hk_grouped_rrr_bf16
            k, n_out = b_in.shape[1], b_in.shape[2]
            key = ("rrr_bf16", m_total, n_out, k)
        else:
            # H4 reroute: short K → too few compute cycles to hide RRR's
            # cache misses; transpose to RCR layout where K is inner-most.
            b_in = bf16_transpose_3d(b) if b.is_contiguous() \
                else b.transpose(-2, -1).contiguous()
            op = torch.ops.primus_turbo_cpp_extension.hk_grouped_rcr_bf16
            n_out, k = b_in.shape[1], b_in.shape[2]
            key = ("rcr_bf16", m_total, n_out, k)
        # Positional layout: a, b, group_offs, group_m(3), m_per_group(4), num_xcds(5).
        fixed = (a_in, b_in, group_offs, 0, avg_m, 0)
        gm, xcds = _autotune_pick(
            op, fixed, _HK_BF16_RCR_CANDIDATES, (3, 5), key=key,
        )
        return op(a_in, b_in, group_offs, gm, avg_m, xcds)


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
        # wgrad requires trans_a=True, !trans_b. trans_c can be either:
        #   trans_c=True  → c[g] = a^T @ b, shape [G, K_fwd, N_fwd] (forward trans_b=True path)
        #   trans_c=False → c[g] = b^T @ a, shape [G, N_fwd, K_fwd] (forward trans_b=False path)
        # These two are transposes of each other; we handle both by swapping
        # the op-side (grad_out, x) roles in execute() below.
        if not (trans_a and not trans_b):
            return False
        return group_lens.numel() > 0

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
        del trans_a, trans_b, num_cu, kwargs
        # Kernel computes c[g] = op_a[g]^T @ op_b[g] with output shape
        # [G, op_a.shape[1], op_b.shape[1]]. Dispatcher's (a, b) are
        # (x, grad_out). Two output layouts via swapping the op-side roles:
        #   trans_c=True  → c [G, N_fwd, K_fwd] = grad_out^T @ x
        #                   (forward trans_b=True wgrad: grad_b shape matches
        #                    b which is [B, N, K])
        #     op_a=grad_out (PT ``b``)
        #     op_b=x        (PT ``a``)
        #   trans_c=False → c [G, K_fwd, N_fwd] = x^T @ grad_out
        #                   (forward trans_b=False wgrad: grad_b shape matches
        #                    b which is [B, K, N])
        #     op_a=x        (PT ``a``)
        #     op_b=grad_out (PT ``b``)
        x_2d = a if a.is_contiguous() else a.contiguous()
        grad_out_2d = b if b.is_contiguous() else b.contiguous()
        if trans_c:
            op_a, op_b = grad_out_2d, x_2d
        else:
            op_a, op_b = x_2d, grad_out_2d
        m_total = op_a.shape[0]
        n_out = op_a.shape[1]
        k = op_b.shape[1]
        op = torch.ops.primus_turbo_cpp_extension.hk_grouped_var_k_crr_bf16
        # Positional layout: a, b, group_offs, group_m(3), num_xcds(4).
        fixed = (op_a, op_b, group_offs, 0, 0)
        gm, xcds = _autotune_pick(
            op, fixed, _HK_BF16_VARK_CANDIDATES, (3, 4),
            key=("vark_bf16", m_total, n_out, k),
        )
        return op(op_a, op_b, group_offs, gm, xcds)


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
