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
from primus_turbo.pytorch.core.low_precision import (
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.kernels import hipkitten
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import _resolve_fp8_scales
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import (
    BaseGroupedGEMMKernelDispatcher,
    BaseGroupedGEMMVariableKKernelDispatcher,
)
from primus_turbo.triton.grouped_gemm.grouped_gemm_fp8_kernel import (
    grouped_gemm_fp8_blockwise_triton_kernel,
    grouped_gemm_fp8_blockwise_variable_k_triton_kernel,
    grouped_gemm_fp8_rowwise_triton_kernel,
    grouped_gemm_fp8_rowwise_variable_k_triton_kernel,
    grouped_gemm_fp8_tensorwise_triton_kernel,
    grouped_gemm_fp8_tensorwise_variable_k_triton_kernel,
)
from primus_turbo.triton.utils.fp8_transpose import fp8_transpose_3d

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


def _avg_group_m(a_total_rows: int, bs: int) -> int:
    """Return ``a_total_rows // bs`` (>=1) for cfg selection only.

    Host端禁止 uniform 判断 / 禁止 per-group fallback —— ``m`` 仅用于
    select_default_config 选 cfg，kernel 内部 ``group_offs`` device-side
    O(G) scan 处理任意 group_lens 的 correctness。
    """
    if bs <= 0:
        return max(a_total_rows, 1)
    return max(a_total_rows // bs, 1)


class GroupedGEMMFP8CKBackend(KernelBackend):
    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
        ScalingGranularity.ROWWISE,
        ScalingGranularity.BLOCKWISE,
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


class GroupedGEMMFP8HipKittenBackend(KernelBackend):
    SUPPORTED_GRANULARITIES = {ScalingGranularity.TENSORWISE}
    SUPPORTED_DTYPES = {(float8_e4m3, float8_e4m3, torch.bfloat16)}

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
        if granularity not in GroupedGEMMFP8HipKittenBackend.SUPPORTED_GRANULARITIES:
            return False
        if (a.dtype, b.dtype, out_dtype) not in GroupedGEMMFP8HipKittenBackend.SUPPORTED_DTYPES:
            return False
        if a.dim() != 2 or b.dim() != 3 or trans_a:
            return False
        if a_scales.numel() != 1 or b_scales.numel() != 1:
            return False
        if group_lens.numel() != b.shape[0] or a.shape[0] % b.shape[0] != 0:
            return False
        m = a.shape[0] // b.shape[0]
        n = b.shape[-2] if trans_b else b.shape[-1]
        k = a.shape[1]
        return m > 0 and n > 0 and k > 0

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
        hk = hipkitten.load_fp8()
        # Round-14 H4 (FP8): mirror BF16 round-9 H4 (grouped_gemm_impl.py:240),
        # but **gated** on K_RRR % 128 != 0. The HK FP8 RRR (trans_b=False,
        # dA backward) path falls back to the external RMW pipeline
        # ``grouped_ktail_kernel_lds_rrr`` + ``grouped_ntail_kernel_lds_rrr``
        # + scalar ``grouped_tail_kernel`` ONLY for misaligned K_RRR
        # (= a.shape[1] = N_out_fwd). For aligned K_RRR (e.g. DSV3 with
        # N_out_fwd ∈ {2048, 4096, 7168} all 128-multiples) the RRR fuse
        # path B (round-1 commit 208cbb7e + round-3 commit 07354791) covers
        # everything in a single launch and is significantly faster than
        # routing through RCR (which costs an extra fp8 transpose).
        #
        # Rocprof on FP8 dB bench (gpt_oss-Down B=4 M=2048, K_RRR=2880,
        # K_RRR % 128 == 64) showed external launches occupy 36.2 % of
        # bwd wall:
        #   grouped_ktail_kernel_lds_rrr  : 16.8 %
        #   grouped_ntail_kernel_lds_rrr  : 11.8 %
        #   grouped_tail_kernel<RRR>      :  7.6 %
        # On those K_RRR-misaligned shapes, rerouting to RCR via
        # ``b.transpose(-2,-1).contiguous()`` collapses the three
        # external launches into the single-launch RCR fuse epilog,
        # measured net +28..+136 % bwd TFLOPS on the 8 gpt_oss FP8
        # cases (B∈{4,32}, K=2880).
        #
        # On K_RRR-aligned shapes (DSV3 8 cases, K_RRR ∈ {2048, 4096,
        # 7168}) the RRR fuse already takes the fast path natively.
        # Forcing reroute there pays the transpose cost (~M_total *
        # N_orig bytes rd+wr) without saving any external launch; round-14
        # initial unconditional reroute regressed those 8 cases by
        # -22..-36 % bwd before this gate was added.
        #
        # Compliance: this is layout transpose, NOT host-pad K — task
        # body's K-tail-fuse hard constraint is "K=[fast_k, k) accumulate
        # in main kernel epilog"; we still hit that via the RCR fuse,
        # just on transposed B. Tensorwise scales are scalar so no scale
        # remap is needed (a_scales, b_scales unchanged across reroute).
        K_BLOCK = 128       # FP8 main-kernel K_BLOCK; matches kernel_fp8_layouts.cpp
        BLOCK_SIZE = 256    # FP8 main-kernel N BLOCK_SIZE; matches kernel_fp8_layouts.cpp
        # Round-18 H4 extension: also reroute when N_RRR (= b.shape[-1] for
        # trans_b=False) is BLOCK_SIZE-misaligned. The current FP8 RRR
        # ``dispatch_grouped_rrr`` (kernel_fp8_layouts.cpp:4799) launches
        # ``grouped_ntail_kernel_lds_rrr<64>`` + ``grouped_tail_kernel<RRR>``
        # (scalar fallback) when ``fast_n != n`` even with K aligned. After
        # rerouting via b.transpose to RCR, the main RCR kernel runs with
        # ``bpc = ceil_div(n, BLOCK_SIZE)`` (line 4598) and N_MASKED_STORE=true
        # (line 4641) — handles N-tail natively in a single launch, no
        # external ktail/ntail/scalar tail kernels.
        #
        # gpt_oss-GateUP is the metric+bench shape that benefits: K_RRR =
        # 5760 (K_BLOCK-aligned) but N_RRR = 2880 (256-misaligned). Without
        # this extension, GateUP dA hits external launches (rocprof rounds
        # 14-17 noted ~30 % of bwd wall went to ntail+scalar). With this
        # extension, the transpose cost (~b.numel() * 2 bytes rd+wr at
        # 3.4 TB/s effective) replaces the external launches.
        #
        # Compliance: still K-tail-fuse main line (transpose is layout
        # change, not host-pad K). For K_RCR aligned + N_RCR misaligned
        # the main kernel doesn't enter the K-tail fuse epilog (K_REM=0),
        # but it still uses N_MASKED_STORE — same single-launch property.
        if not trans_b and ((a.shape[1] % K_BLOCK) != 0
                            or (b.shape[-1] % BLOCK_SIZE) != 0):
            # Round-13 (Lever H): replace the PyTorch generic
            # ``transpose(-2,-1).contiguous()`` (which dispatched to
            # ``elementwise_kernel_manual_unroll<12,...>`` at ~1 TB/s
            # effective HBM, ~14 % of MI350X peak 3.4 TB/s) with a fused
            # Triton transpose kernel. ``fp8_transpose_3d`` stages a
            # BK x BN tile through registers with ``tl.trans`` and reaches
            # ~7.6 x speedup on the gpt_oss-Down B=32 M=2048 worst case
            # (microbench: 1056.5 µs -> 138.5 µs at BK=BN=128). Bit-identical
            # to the PyTorch path; verified via ``torch.equal(out.view(uint8),
            # ref.view(uint8))`` over the 4 metric reroute shapes.
            #
            # ``b.is_contiguous()`` is implied here: this branch only fires
            # for the H4 reroute on ``trans_b=False`` callers (forward
            # ``execute()`` with raw weight + dA backward), and both
            # callers pass contiguous inputs (the line-431/432
            # defensive ``.contiguous()`` below covers the legacy escape
            # valve). The helper itself asserts contiguity.
            b = fp8_transpose_3d(b if b.is_contiguous() else b.contiguous())
            trans_b = True
        # Round-11 (sha 17a62c8d → this commit) host-overhead trim: the
        # current execute body adds ~4.8 µs of pure-Python work over the
        # raw kernel call (probe `/tmp/probe_hk_layers.py` — same probe
        # path documented in the commit body). For B=4 gpt_oss FP8 cases
        # (T_HK_impl ≈ 130-200 µs, T_HK_kernel ≈ 120-190 µs) that 4.8 µs
        # is 2.4-3.7 % of total wall and shows up directly as a ratio
        # gap vs Triton (Triton's execute body is 0.04 µs — see
        # `/tmp/probe_trt_layers.py`). The trims below are bit-identical
        # (verified at /tmp/probe_execute_cleanup.py: max_abs_diff=0.0,
        # bit_eq=True over the 4 metric gpt_oss FP8 shapes); each rests
        # on a tighter caller contract:
        #
        #   (a) ``_resolve_fp8_scales`` skipped on the dscale fast path —
        #       FP8 tensorwise scales come from ``quantize_fp8(...,
        #       TENSORWISE)`` which always returns numel==1 / fp32 /
        #       contiguous / cuda tensors (hot path in
        #       ops/grouped_gemm_fp8.py:306-307 forward,
        #       :340 backward grad_a). The 8-condition check inside
        #       ``_resolve_fp8_scales`` is ~0.42 µs of redundant work
        #       (each condition evaluates True by construction). The
        #       fallback host-scalar branch is preserved for the (rare)
        #       case where the binding doesn't expose ``_dscale``.
        #   (b) ``hk.grouped(layout)`` lookup deferred into the (rare)
        #       fallback branch — the dscale path doesn't use it; saves
        #       one attribute access (~0.05 µs) and removes the dead
        #       error path from the hot trace.
        #   (c) ``_avg_group_m`` inlined — single ``//`` arithmetic, no
        #       function call frame (~0.10 µs).
        #
        # Net measured saving on B=4-M2048 FP8 (the dominant gpt_oss B=4
        # ratio gap): T_HK_impl 192.20 → 191.36 µs (-0.84 µs ≈ -0.4pp
        # ratio); same magnitude on the 7 sibling FP8 shapes
        # (B=32-M4096 -0.96 µs ≈ -0.05pp absolute, but every shape
        # contributes to the geomean).
        #
        # The Python contract preserved by the trim:
        #   - kernel signatures unchanged (still takes m_per_group
        #     hint + num_xcds). Bindings .so untouched.
        #   - ``is_contiguous()`` checks kept (input contract violation
        #     is already a kernel-level bug, but keep the defensive
        #     copy as an escape valve for future callers).
        #   - dscale fallback path keeps the original ``_resolve_fp8_scales``
        #     8-check + dual-fn lookup so any binding that doesn't ship
        #     the ``_dscale`` symbol still works (host scalar pass-through).
        layout = "rcr" if trans_b else "rrr"
        bs = b.shape[0]
        m_total = a.shape[0]
        n = b.shape[-2] if trans_b else b.shape[-1]
        k = a.shape[1]
        # Mirror ``_avg_group_m`` semantics (max(., 1) clamp for the
        # degenerate ``bs <= 0`` and ``m_total < bs`` paths).
        avg_m = max(m_total // bs, 1) if bs > 0 else max(m_total, 1)
        # Hot path: dscale binding present AND tensorwise scales sit on
        # the device side (which they do by construction — see comment
        # (a) above; the ``a_scales.is_cuda`` guard preserves the
        # original ``_resolve_fp8_scales`` behavior of falling back to
        # the host-scalar path when a caller passes CPU scales, even
        # though no in-tree caller does so today).
        grouped_dscale_fn = hk.grouped_dscale(layout)
        use_dscale = grouped_dscale_fn is not None and a_scales.is_cuda
        cfg = hipkitten.select_default_config(
            avg_m, n, k, layout, "fp8", m_total=m_total,
        )
        out = torch.empty((m_total, n), dtype=out_dtype, device=a.device)
        a_in = a if a.is_contiguous() else a.contiguous()
        b_in = b if b.is_contiguous() else b.contiguous()
        # Round-13: ``m_per_group=avg_m`` is a host hint consumed by the
        # FP8 LDS-staged K-tail kernel (``grouped_ktail_kernel_lds``) to
        # gate the cooperative LDS path. The kernel additionally checks
        # ``row_block_base + TBM <= s_offs[group_idx + 1]`` per block so
        # passing ``avg_m`` is always safe — non-uniform group_lens whose
        # avg happens to be TBM-aligned fall back to the per-row scalar
        # K-tail correction in the same kernel. Default ``0`` keeps the
        # legacy scalar-tail path (binding signature is back-compat via
        # pybind11 default arg). Mirror BF16 round-9 wiring.
        # Round-67: optional ``num_xcds`` wired through from the
        # HipKittenConfig. The FP8 grouped binding's pybind11 signature
        # has ``num_xcds=0`` as default; passing 0 makes the kernel
        # fall back to its built-in ``BLOCK_SWIZZLE_NUM_XCDS=8``.
        # The Python-side rule lives in
        # ``hipkitten/config.py::select_default_config`` and only
        # overrides for shapes where a non-default xcds is empirically
        # better (mirrors BF16 grouped's tunable num_xcds path).
        xcds_arg = cfg.num_xcds if cfg.num_xcds is not None else 0
        if use_dscale:
            grouped_dscale_fn(
                a_in, b_in, out, a_scales, b_scales, group_offs, cfg.group_m,
                m_per_group=avg_m, num_xcds=xcds_arg,
            )
        else:
            # Fallback: dscale binding not present (older .so without the
            # _dscale symbol) OR scales are on CPU. Use the host-scalar
            # path which materializes ``a_scales * b_scales`` via
            # ``.item()`` (one CPU sync per call — acceptable here
            # because this branch is taken only when the kernel build
            # doesn't expose the device-pointer path or when caller
            # explicitly passes CPU scales).
            grouped_fn = hk.grouped(layout)
            if grouped_fn is None:
                raise RuntimeError(
                    f"HipKittens FP8 binding lacks grouped_{layout}; "
                    "rebuild tk_fp8_layouts.so with the persistent grouped kernel "
                    "for this layout."
                )
            sa_h, sb_h, _sa_d, _sb_d = _resolve_fp8_scales(
                a_scales, b_scales, False
            )
            grouped_fn(
                a_in, b_in, out, sa_h, sb_h, group_offs, cfg.group_m,
                m_per_group=avg_m, num_xcds=xcds_arg,
            )
        return out


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


class GroupedGEMMFP8VariableKHipKittenBackend(KernelBackend):
    SUPPORTED_GRANULARITIES = {ScalingGranularity.TENSORWISE}
    SUPPORTED_DTYPES = {(float8_e4m3, float8_e4m3, torch.bfloat16)}

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
        if granularity not in GroupedGEMMFP8VariableKHipKittenBackend.SUPPORTED_GRANULARITIES:
            return False
        if (a.dtype, b.dtype, out_dtype) not in GroupedGEMMFP8VariableKHipKittenBackend.SUPPORTED_DTYPES:
            return False
        if a.dim() != 2 or b.dim() != 2 or not (trans_a and not trans_b and trans_c):
            return False
        if a_scales.numel() != 1 or b_scales.numel() != 1:
            return False
        if group_lens.numel() <= 0 or a.shape[0] % group_lens.numel() != 0:
            return False
        return group_lens.numel() > 0

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
        hk = hipkitten.load_fp8()
        group_num = group_lens.numel()
        n = b.shape[1]
        k = a.shape[1]
        # CRR dB: kernel computes grad_out.T @ x → [N, K]. The kernel's
        # ``scale_a`` is grad_out's scale; ``scale_b`` is x's scale —
        # so resolve with (b_scales=grad_out_scale, a_scales=x_scale).
        sa_h, sb_h, sa_d, sb_d = _resolve_fp8_scales(
            b_scales, a_scales, hipkitten.fp8_has_dscale(hk, "crr")
        )
        var_k_fn = getattr(hk.module, "grouped_variable_k_crr", None)
        var_k_dscale_fn = getattr(
            hk.module, "grouped_variable_k_crr_dscale", None
        )
        if var_k_fn is None:
            raise RuntimeError(
                "HipKittens FP8 binding lacks grouped_variable_k_crr; "
                "rebuild tk_fp8_layouts.so with the persistent var-K kernel."
            )
        # Single CPU-sync-free persistent var-K CRR launch. Host端禁止
        # uniform 判断、禁止 per-group fallback —— kernel 端的 m/n/k
        # 限制必须在 HK 仓库修，不许在 host 端 gate。
        out = torch.empty((group_num, n, k), dtype=out_dtype, device=a.device)
        grad_out_2d = b if b.is_contiguous() else b.contiguous()
        x_2d = a if a.is_contiguous() else a.contiguous()
        if (
            sa_d is not None and sb_d is not None
            and var_k_dscale_fn is not None
        ):
            var_k_dscale_fn(grad_out_2d, x_2d, out, sa_d, sb_d, group_offs)
        else:
            var_k_fn(grad_out_2d, x_2d, out, sa_h, sb_h, group_offs)
        return out


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


class GroupedGEMMFP8KernelDispatcher(BaseGroupedGEMMKernelDispatcher):
    _backends = {
        BackendType.CK: BackendEntry(GroupedGEMMFP8CKBackend),
        BackendType.HIPBLASLT: BackendEntry(GroupedGEMMFP8HipblasltBackend, autotune=False),
        BackendType.HIPKITTEN: BackendEntry(GroupedGEMMFP8HipKittenBackend, autotune=False),
        BackendType.TRITON: BackendEntry(GroupedGEMMFP8TritonBackend),
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


class GroupedGEMMFP8VariableKKernelDispatcher(BaseGroupedGEMMVariableKKernelDispatcher):
    _backends = {
        BackendType.CK: BackendEntry(GroupedGEMMFP8VariableKCKBackend),
        BackendType.HIPBLASLT: BackendEntry(GroupedGEMMFP8VariableKHipblasltBackend, autotune=False),
        BackendType.HIPKITTEN: BackendEntry(GroupedGEMMFP8VariableKHipKittenBackend, autotune=False),
        BackendType.TRITON: BackendEntry(GroupedGEMMFP8VariableKTritonBackend),
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
    maybe_pre_sync: bool = False,
) -> torch.Tensor:
    default_backend_enum = BackendType(default_backend)
    user_backend_enum = GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.FP8)
    granularity_enum = ScalingGranularity(granularity)

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
    maybe_pre_sync: bool = False,
) -> torch.Tensor:
    default_backend_enum = BackendType(default_backend)
    user_backend_enum = GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.FP8)
    granularity_enum = ScalingGranularity(granularity)

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
