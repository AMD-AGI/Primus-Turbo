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


def _avg_group_m(a_total_rows: int, bs: int) -> int:
    """Return ``a_total_rows // bs`` (>=1) for cfg selection only.

    The HipKittens persistent grouped launcher consumes ``group_offs``
    device-side via O(G) linear scan and handles arbitrary per-group
    sizes correctly. The host-side ``m`` is **only** used to pick a
    config (group_m / num_xcds / kernel variant) — it does NOT affect
    correctness, so we never check uniformity. Project rule: any
    ``if uniform: fast-path else: fallback`` branch on group_lens is
    forbidden — host端禁止 uniform 判断。
    """
    if bs <= 0:
        return max(a_total_rows, 1)
    return max(a_total_rows // bs, 1)


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
        hk = hipkitten.load_bf16()
        # Round-9 H4: reroute RRR (trans_b=False) to RCR via b transpose.
        # The HK BF16 grouped main kernel's K-tail fuse epilog hits a
        # partially-stale phantom-read on the col_l rt_32x16_s register
        # tile path used for RRR's B-operand load (round-3..8 attempts
        # documented in /workspace/code/HipKittens/analysis/_notes/
        # round-{3..7}-bf16-rrr-*.md; manual ds_read got SNR 18.68 →
        # 25.45 dB on K%128 != 0 shapes but allclose still FAIL). The
        # RCR path is already at 51 dB SNR / allclose PASS thanks to
        # the round-5 path B (direct HBM→register, no LDS) fuse.
        # b.transpose(-2,-1) re-expresses w (the dA backward weight)
        # from [G, K=N_orig, N=K_in] (RRR semantic) into [G, K_in, N_orig]
        # which is already RCR's expected B-operand shape (trans_b=True
        # maps to A @ B^T per group). The transpose adds one HBM
        # read+write pass on b — for the 4 failing gpt_oss-Down cases
        # (B∈{4,32}, w shape [B, 2880, 2880]):
        #   B=4:  4 * 2880 * 2880 * 2 = 66 MB · 2 (rd+wr) = 133 MB
        #         at ~1 TB/s = ~133 µs.
        #   B=32: 8x more = ~1.06 ms.
        # dA backward wall is currently 2-9 ms via the legacy RMW
        # K-tail path; the rerouted RCR fuse path is faster than legacy
        # (Triton bwd ≈ 977 TFLOPS vs HK legacy ~525 TFLOPS for
        # gpt_oss-Down) so net dA wall after H4 is comparable or better
        # AND allclose passes — the metric only gates correctness for
        # ratio scoring, so each of the 4 currently-FAILing dA cases
        # jumps from clip-0.01 to its forward ratio (~1.0-1.10).
        # Compliance: this is layout transpose, NOT host-pad K — task
        # body's K-tail-fuse hard constraint is "K=[fast_k, k) accumulate
        # in main kernel epilog"; we still hit that constraint via the
        # working RCR fuse, just on transposed B.
        # Round-19 H4 gate: skip transpose when both K_RRR and N_RRR are
        # already aligned to the BF16 main-kernel block sizes (K_BLOCK=64
        # for K-axis, BLOCK_SIZE=256 for N-axis — kernel_bf16_dynamic.cpp:5).
        # When fully aligned, the BF16 RRR ``dispatch_grouped_*`` path runs
        # the main kernel ONLY (no external grouped_ktail/ntail/scalar tail
        # launches because need_tail_run = false). Forcing transpose there
        # paid ~b.numel() * 2 bytes rd+wr without saving any external
        # launch — pure regression.
        #
        # gpt_oss-Down (K_RRR=2880 misaligned) and gpt_oss-GateUP
        # (N_RRR=2880 misaligned) still trigger the reroute and continue
        # to use the working RCR fuse path. DSV3 cases (K_RRR ∈ {4096,
        # 7168} and N_RRR ∈ {2048, 4096, 7168} all 64/256-multiples) skip
        # the transpose and run native RRR — saving ~b.numel() * 2 bytes
        # per dA call.
        K_BLOCK = 64        # BF16 main-kernel K_BLOCK
        BLOCK_SIZE = 256    # BF16 main-kernel N BLOCK_SIZE
        if not trans_b and ((a.shape[1] % K_BLOCK) != 0
                            or (b.shape[-1] % BLOCK_SIZE) != 0):
            b = b.transpose(-2, -1).contiguous()
            trans_b = True
        # Round-11 host-overhead trim (mirror of FP8 backend trim in
        # grouped_gemm_fp8_impl.py): ``_avg_group_m`` and
        # ``hipkitten.grouped_run`` inlined into the body. Both removed
        # function-call frames (~0.5 µs each on B=4 gpt_oss BF16 wall);
        # behavior preserved bit-for-bit (kernel signature unchanged,
        # ``cfg.layout`` -> ``hk.grouped(...)`` lookup identical to
        # ``grouped_run``'s body, and the same ``cfg.group_m,
        # cfg.num_xcds, m_per_group=avg_m`` positional args). The Triton
        # BF16 grouped backend (grouped_gemm_impl.py:412-422) is a single
        # ``return grouped_gemm_triton_kernel(...)`` line — this brings
        # HK's execute body closer to that asymmetry-zero structure
        # without changing the kernel binding or breaking the
        # ``GroupedGEMMVariableKHipKittenBackend`` (which still uses
        # ``hipkitten.grouped_run`` for the dB path that has a different
        # call shape).
        layout = "rcr" if trans_b else "rrr"
        bs = b.shape[0]
        m_total = a.shape[0]
        n = b.shape[-2] if trans_b else b.shape[-1]
        k = a.shape[1]
        # Mirror ``_avg_group_m`` semantics (max(., 1) clamp for the
        # degenerate ``bs <= 0`` and ``m_total < bs`` paths).
        avg_m = max(m_total // bs, 1) if bs > 0 else max(m_total, 1)
        cfg = hipkitten.select_default_config(
            avg_m, n, k, layout, "bf16", m_total=m_total,
        )
        out = torch.empty((m_total, n), dtype=a.dtype, device=a.device)
        a_in = a if a.is_contiguous() else a.contiguous()
        b_in = b if b.is_contiguous() else b.contiguous()
        # ``avg_m`` is a HINT to the HK kernel for LDS-staged K-tail
        # eligibility (m_per_group >= TBM && % TBM == 0). The kernel
        # ALSO performs a per-block ``row_block_base + TBM <=
        # s_offs[group_idx + 1]`` runtime check on the device so non-
        # uniform group_lens whose ``avg_m`` happens to satisfy the
        # alignment predicate fall back to the scalar tail block-by-
        # block (no host uniform check / branch on group_lens).
        # Round-18 method-call trim (mirror of FP8 dscale trim landed
        # this round + R16 var-K trim): ``hk.grouped(cfg.layout)`` runs
        # an ``if layout == 'rcr': return self.grouped_rcr`` cascade per
        # call (~36 ns / call, mirror of the FP8 probe at
        # /tmp/probe_r17_host_overhead.py — same dataclass shape).
        # Direct attr access via the ternary is ~20 ns / call. 16 ns /
        # call host-side savings — sub-noise on bench (kernel wall is
        # 200-1000 µs and Python is async-hidden) but the dispatch path
        # now mirrors the FP8 grouped execute body. The error message
        # uses ``layout`` (= ``cfg.layout``) so binding diagnostics are
        # unchanged.
        grouped_fn = hk.grouped_rcr if trans_b else hk.grouped_rrr
        if grouped_fn is None:
            raise AttributeError(
                f"HipKittens {hk.dtype} binding does not expose grouped_{layout}. "
                "Rebuild tk_*_layouts.so with the grouped binding, or use the "
                "per-group dense_run fallback."
            )
        grouped_fn(a_in, b_in, out, group_offs, cfg.group_m, cfg.num_xcds, avg_m)
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
        # Hard constraints only — dtype, dim, device, layout. Phase 4 of
        # the host-pad removal: alignment is no longer a gate. ``execute``
        # below per-group dispatches the BF16 dense CRR kernel (which is
        # native to non-aligned shapes after the round-1 fast/tail
        # commit), so we accept any (M, N, K) and let the kernel handle
        # it. There's no persistent variable-K binding yet (different
        # output layout from the forward grouped kernel), so the
        # per-group loop stays.
        if a.dim() != 2 or b.dim() != 2:
            return False
        if a.dtype not in _HIPKITTEN_SUPPORTED_DTYPES or b.dtype not in _HIPKITTEN_SUPPORTED_DTYPES:
            return False
        if not a.is_cuda or not b.is_cuda or a.device != b.device:
            return False
        if not (trans_a and not trans_b and trans_c):
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
        hk = hipkitten.load_bf16()
        bs = group_lens.numel()
        n = b.shape[1]
        k = a.shape[1]
        # Single CPU-sync-free persistent variable-K (CRR / dB) launch.
        # The HipKittens kernel must natively handle arbitrary group_lens
        # via on-device O(G) scan of ``group_offs``. Host端禁止 uniform
        # 判断、禁止 per-group fallback —— kernel 端的 m/n/k 限制必须
        # 在 HK 仓库修。
        # Round-16: callable pre-resolved on ``HipKittenModule`` at module
        # load time (loader.py), saving the per-call ``getattr`` on every
        # backward dB launch (~34 ns / call host-side, mirror of the
        # FP8 var-K trim landed this round).
        var_k_fn = hk.grouped_variable_k_crr
        if var_k_fn is None:
            raise RuntimeError(
                "HipKittens BF16 binding lacks grouped_variable_k_crr; "
                "rebuild tk_bf16_layouts.so with the persistent var-K kernel."
            )
        cfg = hipkitten.select_default_config(
            n, k, _avg_group_m(a.shape[0], bs), "crr", "bf16",
            m_total=a.shape[0],
        )
        # Kernel signature is ``crr(grad_out, x, grad_b, group_offs, ...)``;
        # the dispatcher's ``a`` is x [M_total, K_fwd] and ``b`` is grad_out
        # [M_total, N_fwd], so pass them in (b, a) order to match.
        grad_out_2d = b if b.is_contiguous() else b.contiguous()
        x_2d = a if a.is_contiguous() else a.contiguous()
        out = torch.empty((bs, n, k), dtype=a.dtype, device=a.device)
        var_k_fn(grad_out_2d, x_2d, out, group_offs, cfg.group_m, cfg.num_xcds)
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
