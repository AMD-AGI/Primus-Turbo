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
from primus_turbo.pytorch.kernels import hipkitten
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


def _pad_2d(x: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    if x.shape[0] == rows and x.shape[1] == cols:
        return x
    out = torch.zeros((rows, cols), dtype=x.dtype, device=x.device)
    out[: x.shape[0], : x.shape[1]] = x
    return out


def _group_offsets_cpu(group_offs: torch.Tensor) -> list[int]:
    return [int(x) for x in group_offs.detach().cpu().tolist()]


def _uniform_group_m(a_total_rows: int, group_lens: torch.Tensor) -> int | None:
    """Return per-group M iff every entry of ``group_lens`` is equal; else None.

    Uses ``(group_lens == m_avg).all().item()`` — one tiny GPU reduction
    plus one GPU→CPU sync (≈ tens of microseconds for typical B≤32).
    Forwarded by the HipKittens grouped backends as a *fast-path*
    detector: the native HK grouped launcher today consumes a single
    per-group M (the kernel iterates ``B`` groups of identical size),
    so when this returns ``None`` we route to the per-group
    :func:`...dispatch.dense_run` loop instead — never reject.

    Per project policy this is NOT a ``can_handle`` resume / reject
    helper: HipKittens ``can_handle`` accepts arbitrary ``group_lens``,
    and the *execute* path uses this to pick an implementation. New HK
    grouped bindings should accept cumulative-offsets directly so this
    fast-path detector becomes redundant.
    """
    bs = group_lens.numel()
    if bs <= 0 or a_total_rows % bs != 0:
        return None
    m_avg = a_total_rows // bs
    return m_avg if bool((group_lens == m_avg).all().item()) else None


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
        # Hard constraints only — dtype, dim, device, B-divisibility.
        #
        # We deliberately do NOT reject on:
        #   * non-uniform ``group_lens`` (project rule: real MoE traffic
        #     never has equal-sized groups; ``execute`` falls back to a
        #     per-group ``dense_run`` loop when uniform-M is not detected).
        #   * misaligned (M, N, K) (project rule: ``aligned_for`` is a
        #     launcher-internal constraint, not a resume / reject. The
        #     ``execute`` path pads the missing dims and dispatches a
        #     single grouped launch on the padded shape, then slices.)
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
        hk = hipkitten.load_bf16()
        bs = b.shape[0]
        n = b.shape[-2] if trans_b else b.shape[-1]
        k = a.shape[1]
        out = torch.zeros((a.shape[0], n), dtype=a.dtype, device=a.device)
        layout = "rcr" if trans_b else "rrr"

        m = _uniform_group_m(a.shape[0], group_lens)
        if m is not None:
            m_pad, n_pad, k_pad = hipkitten.padded_shape(m, n, k, layout, "bf16")
            cfg = hipkitten.select_default_config(m_pad, n_pad, k_pad, layout, "bf16")
            if (m_pad, n_pad, k_pad) == (m, n, k):
                hipkitten.grouped_run(hk, cfg, a.contiguous(), b.contiguous(), out)
                return out
            if m_pad == m:
                # M is aligned, but N and / or K need padding (the gpt_oss_20B
                # common case: N=2880→3072 / N=5760→6144 and K=2880→3072
                # together). One grouped launch on the K-and-N padded
                # tensors, then slice the unpadded N columns back into
                # ``out``. Padded zero rows / columns contribute nothing
                # to the used [:, :n] slice of C.
                if k_pad == k:
                    a_in = a.contiguous()
                else:
                    a_in = torch.zeros((bs * m, k_pad), dtype=a.dtype, device=a.device)
                    a_in[:, :k].copy_(a)
                if trans_b:
                    if (n_pad, k_pad) == (n, k):
                        b_in = b.contiguous()
                    else:
                        b_in = torch.zeros((bs, n_pad, k_pad), dtype=b.dtype, device=b.device)
                        b_in[:, :n, :k].copy_(b)
                else:
                    if (n_pad, k_pad) == (n, k):
                        b_in = b.contiguous()
                    else:
                        b_in = torch.zeros((bs, k_pad, n_pad), dtype=b.dtype, device=b.device)
                        b_in[:, :k, :n].copy_(b)
                c_pad = torch.zeros((bs * m, n_pad), dtype=a.dtype, device=a.device)
                hipkitten.grouped_run(hk, cfg, a_in, b_in, c_pad)
                out.copy_(c_pad[:, :n])
                return out
            # M is misaligned: fall through to the per-group dense_run
            # loop. M-pad of a uniform-M grouped layout would require
            # interleaving zeros between groups, which is only marginally
            # faster than per-group padding at large B and adds enough
            # complexity that we keep the fallback simple here. The
            # metric never hits this branch (DeepSeek / gpt_oss / Llama
            # all have M ∈ {2048, 4096, 8192}).

        # Per-group dense_run fallback. Slower than the native ``grouped_*``
        # launcher but correct for non-uniform ``group_lens`` and for
        # shapes where M and / or K need padding. Mirrors the FP8 grouped
        # fallback in ``grouped_gemm_fp8_impl.py``.
        offs = _group_offsets_cpu(group_offs)
        for group_idx in range(bs):
            start, end = offs[group_idx], offs[group_idx + 1]
            mg = end - start
            if mg <= 0:
                continue
            m_pad, n_pad, k_pad = hipkitten.padded_shape(mg, n, k, layout, "bf16")
            cfg = hipkitten.select_default_config(m_pad, n_pad, k_pad, layout, "bf16")
            if (m_pad, n_pad, k_pad) == (mg, n, k):
                hipkitten.dense_run(
                    hk,
                    cfg,
                    a[start:end].contiguous(),
                    b[group_idx].contiguous(),
                    out[start:end],
                )
            else:
                b_pad_rows = n_pad if trans_b else k_pad
                b_pad_cols = k_pad if trans_b else n_pad
                out_pad = torch.zeros((m_pad, n_pad), dtype=a.dtype, device=a.device)
                hipkitten.dense_run(
                    hk,
                    cfg,
                    _pad_2d(a[start:end].contiguous(), m_pad, k_pad),
                    _pad_2d(b[group_idx].contiguous(), b_pad_rows, b_pad_cols),
                    out_pad,
                )
                out[start:end].copy_(out_pad[:mg, :n])
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
        # Hard constraints only — dtype, dim, device, layout, B-divisibility.
        # We do NOT reject on group-size uniformity: ``execute`` below
        # dispatches non-uniform-M to a per-group ``dense_run`` loop with
        # crr ordering. Alignment of (n, k, m_avg) is checked
        # because the per-group fallback also pads internally — accepting
        # any (n, k, m) here means execute can always run; rejecting only
        # M_avg-misaligned shapes would be cleaner but no metric / DoD
        # case exercises them, so we keep the alignment hint conservative.
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
        m = a.shape[0] // group_lens.numel()
        n = b.shape[1]
        k = a.shape[1]
        return hipkitten.aligned_for(n, k, m, "bf16")

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
        hk = hipkitten.load_bf16()
        bs = group_lens.numel()
        n = b.shape[1]
        k = a.shape[1]
        out = torch.zeros((bs, n, k), dtype=a.dtype, device=a.device)

        m = _uniform_group_m(a.shape[0], group_lens)
        if m is not None:
            cfg = hipkitten.select_default_config(n, k, m, "crr", "bf16")
            hipkitten.grouped_run(hk, cfg, b.contiguous(), a.contiguous(), out)
            return out

        # Non-uniform-M fallback: per-group dense_run via gemm_crr, with
        # per-group padding for any individual group whose K_logical
        # (= per-group M) breaks the crr alignment that ``can_handle``
        # checked using the *average* M.
        offs = _group_offsets_cpu(group_offs)
        for group_idx in range(bs):
            start, end = offs[group_idx], offs[group_idx + 1]
            mg = end - start
            if mg <= 0:
                continue
            m_pad, n_pad, k_pad = hipkitten.padded_shape(n, k, mg, "crr", "bf16")
            cfg = hipkitten.select_default_config(m_pad, n_pad, k_pad, "crr", "bf16")
            if (m_pad, n_pad, k_pad) == (n, k, mg):
                hipkitten.dense_run(
                    hk,
                    cfg,
                    b[start:end].contiguous(),
                    a[start:end].contiguous(),
                    out[group_idx],
                )
            else:
                out_pad = torch.zeros((m_pad, n_pad), dtype=a.dtype, device=a.device)
                hipkitten.dense_run(
                    hk,
                    cfg,
                    _pad_2d(b[start:end].contiguous(), k_pad, m_pad),
                    _pad_2d(a[start:end].contiguous(), k_pad, n_pad),
                    out_pad,
                )
                out[group_idx].copy_(out_pad[:n, :k])
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
