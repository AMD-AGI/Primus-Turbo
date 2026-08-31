###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch

from primus_turbo.pytorch.core.backend import (
    BackendChoice,
    BackendEntry,
    BackendType,
    GlobalBackendManager,
    KernelBackend,
    PrecisionType,
    TuneCache,
)
from primus_turbo.pytorch.core.utils import build_ck, is_gfx950, is_gfx1250
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_ck_ws_heuristic import (
    approximate_ck_standard_total_tiles,
    compute_ck_variable_k_total_tiles,
    resolve_ck_ws_local_per_xcd,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import (
    BaseGroupedGEMMKernelDispatcher,
    BaseGroupedGEMMVariableKKernelDispatcher,
)
from primus_turbo.triton.grouped_gemm.grouped_gemm_helper import (
    grouped_gemm_output_tail_kernel,
)
from primus_turbo.triton.grouped_gemm.grouped_gemm_kernel import (
    grouped_gemm_triton_kernel,
    grouped_gemm_variable_k_triton_kernel,
)

_COMMON_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

# Supported schedule values exposed at the public op layer.
#   "static"      -- static-stride persistent kernel (default).
#   "work_steal"  -- work-stealing persistent kernel with the kernel-aware
#                    heuristic that picks the WS sub-mode (per-XCD /
#                    global / hierarchical) from tensor metadata.
#                    Triton + CK backends.
_SUPPORTED_SCHEDULES: tuple[str, ...] = ("static", "work_steal")
_WS_SUPPORTED_SCHEDULES: tuple[str, ...] = ("static", "work_steal")
_NON_WS_SUPPORTED_SCHEDULES: tuple[str, ...] = ("static",)

# Per-device cache for the CK work-stealing counter buffer.
#
# Layout: [xcd0..xcd7, global, done] -- 10 int32 slots. The kernel self-resets
# every slot to 0 before exit (last-out CTA pattern, see
# grouped_gemm_kernel_ws.hpp), so the only zero we ever need is the one
# ``torch.zeros`` does at first allocation. Caching one buffer per device makes
# that allocation a one-shot cost per process.
#
# Caveat: the singleton is not stream-safe. Concurrent WS launches on different
# streams of the same device would race on the same slots. Safe under the
# typical single-stream autograd graph. The public ``grouped_gemm()`` op also
# rejects ``schedule="work_steal"`` paired with ``num_cu != None``, so partial-
# grid launches never reach the WS path from the high level.
_CK_WS_COUNTER_NUM_SLOTS = 10
_ck_ws_counters: dict[torch.device, torch.Tensor] = {}


def _get_ck_ws_counter(device: torch.device) -> torch.Tensor:
    buf = _ck_ws_counters.get(device)
    if buf is None:
        buf = torch.zeros(_CK_WS_COUNTER_NUM_SLOTS, dtype=torch.int32, device=device)
        _ck_ws_counters[device] = buf
    return buf


def _num_cus(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


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
        schedule: str = "static",
        **kwargs,
    ) -> bool:
        supported = True
        supported &= build_ck()
        supported &= not is_gfx1250()
        supported &= a.dim() == 2 and b.dim() == 3
        supported &= a.dtype in _COMMON_SUPPORTED_DTYPES and b.dtype in _COMMON_SUPPORTED_DTYPES
        supported &= not trans_a
        supported &= schedule in _WS_SUPPORTED_SCHEDULES
        # The CK WS kernel needs 4 extra bytes of LDS for the atomicAdd
        # broadcast slot, which fits on gfx950 (MI355X) but overflows
        # gfx942's 64 KB LDS budget. The device-side WS body is stubbed
        # out on gfx942, so refuse to dispatch WS to CK on that arch.
        if schedule == "work_steal" and not is_gfx950():
            supported = False
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
        schedule: str = "static",
        **kwargs,
    ) -> torch.Tensor:
        work_steal = schedule == "work_steal"
        ws_counter: torch.Tensor | None = None
        resolved_lpx = 0
        if work_steal:
            ws_counter = _get_ck_ws_counter(a.device)
            # Sync-free: derive total_tiles from tensor metadata. Use the
            # transpose-aware M dimension (``can_handle`` already rejects
            # trans_a=True on this backend, but keep the shape logic honest
            # so it stays correct if that constraint is ever relaxed).
            total_m = a.size(1) if trans_a else a.size(0)
            n = b.size(1) if trans_b else b.size(2)
            total_tiles = approximate_ck_standard_total_tiles(
                total_m,
                group_lens.numel(),
                n,
            )
            resolved_lpx = resolve_ck_ws_local_per_xcd(
                "auto",
                total_tiles,
                num_cu or _num_cus(a.device),
                kernel_kind="standard",
            )
        return torch.ops.primus_turbo_cpp_extension.ck_grouped_gemm(
            a,
            b,
            group_lens,
            group_offs,
            trans_a,
            trans_b,
            num_cu,
            work_steal,
            ws_counter,
            resolved_lpx,
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
        schedule: str = "static",
        inplace_add_to_out: bool = False,
        **kwargs,
    ) -> bool:
        supported = True
        # This backend has no beta=1 accumulate epilogue.
        supported &= not inplace_add_to_out
        supported &= build_ck()
        supported &= not is_gfx1250()
        supported &= a.dim() == 2 and b.dim() == 2
        supported &= a.dtype in _COMMON_SUPPORTED_DTYPES and b.dtype in _COMMON_SUPPORTED_DTYPES
        supported &= trans_a and not trans_b
        supported &= schedule in _WS_SUPPORTED_SCHEDULES
        # See GroupedGEMMCKBackend.can_handle: the CK WS kernel is
        # gfx950-only due to LDS budget.
        if schedule == "work_steal" and not is_gfx950():
            supported = False
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
        schedule: str = "static",
        **kwargs,
    ) -> torch.Tensor:
        if trans_c:
            lhs, rhs = b, a
            trans_lhs, trans_rhs = not trans_b, not trans_a
        else:
            lhs, rhs = a, b
            trans_lhs, trans_rhs = trans_a, trans_b
        work_steal = schedule == "work_steal"
        ws_counter: torch.Tensor | None = None
        resolved_lpx = 0
        if work_steal:
            ws_counter = _get_ck_ws_counter(lhs.device)
            # Variable-K total_tiles depends only on tensor shapes
            # (lhs.size, rhs.size, group_lens.numel) -- exact and sync-free.
            m_out = lhs.size(1) if trans_lhs else lhs.size(0)
            n_out = rhs.size(0) if trans_rhs else rhs.size(1)
            total_tiles = compute_ck_variable_k_total_tiles(
                group_lens.numel(),
                m_out,
                n_out,
            )
            resolved_lpx = resolve_ck_ws_local_per_xcd(
                "auto",
                total_tiles,
                num_cu or _num_cus(lhs.device),
                kernel_kind="variable_k",
            )
        return torch.ops.primus_turbo_cpp_extension.ck_grouped_gemm_variable_k(
            lhs,
            rhs,
            group_lens,
            group_offs,
            trans_lhs,
            trans_rhs,
            num_cu,
            work_steal,
            ws_counter,
            resolved_lpx,
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
        schedule: str = "static",
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.dim() == 2 and b.dim() == 3
        supported &= a.dtype in _COMMON_SUPPORTED_DTYPES and b.dtype in _COMMON_SUPPORTED_DTYPES
        supported &= not trans_a
        supported &= schedule in _NON_WS_SUPPORTED_SCHEDULES
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
        schedule: str = "static",
        **kwargs,
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
        schedule: str = "static",
        inplace_add_to_out: bool = False,
        out: torch.Tensor | None = None,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.dim() == 2 and b.dim() == 2
        supported &= a.dtype in _COMMON_SUPPORTED_DTYPES and b.dtype in _COMMON_SUPPORTED_DTYPES
        supported &= trans_a and not trans_b
        supported &= schedule in _NON_WS_SUPPORTED_SCHEDULES

        if inplace_add_to_out:
            supported &= out is not None and out.is_contiguous()
            supported &= out is not None and (out.dtype == a.dtype or out.dtype == torch.float32)

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
        schedule: str = "static",
        inplace_add_to_out: bool = False,
        out: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if trans_c:
            lhs, rhs = b, a
            trans_lhs, trans_rhs = not trans_b, not trans_a
        else:
            lhs, rhs = a, b
            trans_lhs, trans_rhs = trans_a, trans_b

        beta = 1.0 if inplace_add_to_out else 0.0
        return torch.ops.primus_turbo_cpp_extension.hipblaslt_grouped_gemm(
            lhs, rhs, group_lens, group_offs, trans_lhs, trans_rhs, maybe_pre_sync, beta, out
        )


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
        schedule: str = "static",
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.dim() == 2 and b.dim() == 3
        supported &= a.dtype in _COMMON_SUPPORTED_DTYPES and b.dtype in _COMMON_SUPPORTED_DTYPES
        supported &= not trans_a
        supported &= schedule in _WS_SUPPORTED_SCHEDULES
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
        schedule: str = "static",
        **kwargs,
    ) -> torch.Tensor:
        return grouped_gemm_triton_kernel(
            a,
            b,
            group_offs,
            trans_b=trans_b,
            num_cu=num_cu,
            work_steal=(schedule == "work_steal"),
            ws_mode="auto",
        )


def _cap_cu(num_cu: int | None, device: torch.device) -> int:
    """Turn the dispatcher's CU count into a budget the kernels act on. Callers routinely pass
    the full device count meaning "no limit", and honouring that literally would launch a
    persistent grid where the tuned one-tile-per-WG launch is both simpler and faster, so it
    reads as 0."""
    if num_cu is None or num_cu >= _num_cus(device):
        return 0
    return int(num_cu)


class GroupedGEMMFlyDSLBackend(KernelBackend):
    """FlyDSL bf16 grouped GEMM backend (gfx950).

    M-grouped operator, both directions of the forward pair:
      - trans_b=True  -> NT, b is [G, N, K] (forward)
      - trans_b=False -> NN, b is [G, K, N] (dgrad)

    Tiles are cut per expert rather than on a global row-block grid, so neither the token
    count nor the per-expert run lengths have to land on the tile boundary.
    """

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        num_cu: int | None,
        schedule: str = "static",
        **kwargs,
    ) -> bool:
        supported = True
        # gfx950 (CDNA4) only: the body is built on mfma_f32_16x16x32_bf16.
        supported &= is_gfx950()
        # No work-stealing variant, and the grid is sized from the tile count, so a CU budget
        # could only be ignored -- better to decline than to accept and not honour it.
        supported &= schedule in _NON_WS_SUPPORTED_SCHEDULES
        supported &= a.dim() == 2 and b.dim() == 3
        # Both 16-bit float formats: same pipeline, the mfma atom picks bf16 vs f16. Mixed
        # operand types have no atom, so the pair has to agree.
        supported &= a.dtype in (torch.bfloat16, torch.float16) and b.dtype == a.dtype
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
        schedule: str = "static",
        **kwargs,
    ) -> torch.Tensor:
        from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel import (
            grouped_gemm_bf16_nn_flydsl_kernel,
            grouped_gemm_bf16_nt_flydsl_kernel,
        )

        kernel = grouped_gemm_bf16_nt_flydsl_kernel if trans_b else grouped_gemm_bf16_nn_flydsl_kernel
        return kernel(a, b, group_offs, out_dtype=a.dtype, cap_cu=_cap_cu(num_cu, a.device))


_GROUPED_GEMM_BACKENDS = {
    BackendType.CK: BackendEntry(GroupedGEMMCKBackend),
    BackendType.HIPBLASLT: BackendEntry(GroupedGEMMHipblasltBackend, autotune=False),
    BackendType.TRITON: BackendEntry(GroupedGEMMTritonBackend),
    BackendType.FLYDSL: BackendEntry(GroupedGEMMFlyDSLBackend),
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
        schedule: str = "static",
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.dim() == 2 and b.dim() == 2
        supported &= a.dtype in _COMMON_SUPPORTED_DTYPES and b.dtype in _COMMON_SUPPORTED_DTYPES
        supported &= trans_a and not trans_b
        supported &= schedule in _WS_SUPPORTED_SCHEDULES
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
        schedule: str = "static",
        inplace_add_to_out: bool = False,
        out: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if trans_c:
            lhs, rhs = b, a
        else:
            lhs, rhs = a, b
        return grouped_gemm_variable_k_triton_kernel(
            lhs,
            rhs,
            group_offs,
            num_cu=num_cu,
            work_steal=(schedule == "work_steal"),
            ws_mode="auto",
            beta=(1.0 if inplace_add_to_out else 0.0),
            out=out,
        )


class GroupedGEMMVariableKFlyDSLBackend(KernelBackend):
    """FlyDSL bf16 variable-K grouped GEMM backend (gfx950).

    wgrad: C[g] = a[offs[g] : offs[g] + lens[g]]^T @ b[the same rows] -- the contraction
    length varies per group. The kernel walks it with a runtime scf.for and reads the group
    table on-device, so a padded ``group_offs`` paired with the true ``group_lens`` costs no
    host sync.

    """

    MAX_G = 64

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
        schedule: str = "static",
        inplace_add_to_out: bool = False,
        **kwargs,
    ) -> bool:
        supported = True
        # gfx950 (CDNA4) only: the body is built on mfma_f32_16x16x32_bf16.
        supported &= is_gfx950()
        # This backend has no beta=1 accumulate epilogue.
        supported &= not inplace_add_to_out
        supported &= schedule in _NON_WS_SUPPORTED_SCHEDULES
        supported &= a.dim() == 2 and b.dim() == 2 and a.shape[0] == b.shape[0]
        supported &= a.dtype in (torch.bfloat16, torch.float16) and b.dtype == a.dtype
        supported &= trans_a and not trans_b
        # Measured boundary: clean through G=65, wrong past it (G=80 and G=96 both fail).
        # 64 is the conservative cut, and matches the expert bound the NT path documents.
        supported &= group_lens.numel() <= GroupedGEMMVariableKFlyDSLBackend.MAX_G
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
        schedule: str = "static",
        **kwargs,
    ) -> torch.Tensor:
        from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel import (
            grouped_gemm_bf16_variable_k_flydsl_kernel,
        )

        # trans_c is native here -- the store writes the transposed tile -- so unlike the CK
        # and Triton backends there is no operand swap. group_lens goes in as the per-group
        # valid contraction length, which is what lets group_offs carry padding.
        return grouped_gemm_bf16_variable_k_flydsl_kernel(
            a,
            b,
            group_offs,
            masked_k=group_lens,
            out_dtype=a.dtype,
            trans_c=trans_c,
            cap_cu=_cap_cu(num_cu, a.device),
        )


_GROUPED_GEMM_VARIABLE_K_BACKENDS = {
    BackendType.CK: BackendEntry(GroupedGEMMVariableKCKBackend),
    BackendType.HIPBLASLT: BackendEntry(GroupedGEMMVariableKHipblasltBackend, autotune=False),
    BackendType.TRITON: BackendEntry(GroupedGEMMVariableKTritonBackend),
    BackendType.FLYDSL: BackendEntry(GroupedGEMMVariableKFlyDSLBackend),
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
    schedule: str = "static",
) -> torch.Tensor:
    default_backend_choice = BackendChoice(backend=BackendType(default_backend))
    user_backend_choice = GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.BF16_FP16_FP32)

    kwargs = dict(
        a=a,
        b=b,
        group_lens=group_lens,
        group_offs=group_offs,
        trans_a=trans_a,
        trans_b=trans_b,
        num_cu=num_cu,
        maybe_pre_sync=maybe_pre_sync,
        schedule=schedule,
    )

    out = GroupedGEMMKernelDispatcher.dispatch(default_backend_choice, user_backend_choice, **kwargs)
    out = grouped_gemm_output_tail_kernel(out, group_offs)
    return out


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
    schedule: str = "static",
) -> torch.Tensor:
    default_backend_choice = BackendChoice(backend=BackendType(default_backend))
    user_backend_choice = GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.BF16_FP16_FP32)
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
        schedule=schedule,
    )
    return GroupedGEMMVariableKKernelDispatcher.dispatch(
        default_backend_choice, user_backend_choice, **kwargs
    )


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
    schedule: str = "static",
) -> torch.Tensor:
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 3, f"b must be 3D, got {b.shape}"
    assert a.dtype in [torch.float16, torch.bfloat16], f"a must be float16 or bfloat16, got {a.dtype}"
    assert b.dtype in [torch.float16, torch.bfloat16], f"b must be float16 or bfloat16, got {b.dtype}"
    assert trans_a == False, "Only trans_a=False is supported."

    m = a.shape[1] if trans_a else a.shape[0]
    n = b.shape[-2] if trans_b else b.shape[-1]
    return torch.empty((m, n), device=a.device, dtype=a.dtype)


@_torch_custom_op_wrapper(
    "primus_turbo::grouped_gemm_variable_k_accum_impl", mutates_args={"out"}, device_types="cuda"
)
def grouped_gemm_variable_k_accum_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    trans_c: bool,
    num_cu: int | None,
    default_backend: int,
    out: torch.Tensor,
    maybe_pre_sync: bool = False,
    schedule: str = "static",
) -> None:
    """Variable-K grouped BF16/FP16 GEMM that accumulates into ``out``.

    Computes ``out += A^T @ B`` per group, folding the accumulation into the GEMM
    epilogue (beta=1)
    """
    default_backend_choice = BackendChoice(backend=BackendType(default_backend))
    user_backend_choice = GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.BF16_FP16_FP32)
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
        schedule=schedule,
        inplace_add_to_out=True,
        out=out,
    )

    # The tuner benchmarks a backend by launching it repeatedly, so letting it tune on
    # the caller's buffer would accumulate the wgrad once per warmup and timing
    # iteration.
    if (
        GlobalBackendManager.auto_tune_enabled()
        and not GroupedGEMMVariableKKernelDispatcher._is_graph_capturing()
    ):
        GroupedGEMMVariableKKernelDispatcher.tune(**{**kwargs, "out": torch.zeros_like(out)})

    GroupedGEMMVariableKKernelDispatcher.dispatch(default_backend_choice, user_backend_choice, **kwargs)


@grouped_gemm_variable_k_accum_impl.register_fake
def grouped_gemm_variable_k_accum_impl_meta(
    a: torch.Tensor,
    b: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    trans_c: bool,
    num_cu: int | None,
    default_backend: int,
    out: torch.Tensor,
    maybe_pre_sync: bool = False,
    schedule: str = "static",
) -> None:
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 2, f"b must be 2D, got {b.shape}"
    assert out.dim() == 3, f"out must be 3D, got {out.shape}"
    return None


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
    schedule: str = "static",
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
