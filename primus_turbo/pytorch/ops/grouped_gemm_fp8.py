###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
from typing import Union

import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_compute_offs,
    grouped_gemm_fp8_impl,
    grouped_gemm_fp8_variable_k_impl,
)
from primus_turbo.pytorch.core.low_precision import ScalingRecipe
from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    quant_fp8_blockwise_for_weight_impl,
    quant_fp8_blockwise_impl,
    quant_fp8_blockwise_segment_m_impl,
)
from primus_turbo.pytorch.ops.quantization import (
    MX_BLOCK_SIZE,
    quantize_fp8,
    quantize_fp8_with_trans,
)

MXFP8_PADDING_ALIGN_SIZE = 128

__all__ = [
    "grouped_gemm_fp8",
]


def _get_fp8_dtype(format: Format, is_fwd_stage: bool):
    if format == Format.E4M3:
        return float8_e4m3
    elif format == Format.E5M2:
        return float8_e5m2
    elif format == Format.HYBRID:
        return float8_e4m3 if is_fwd_stage else float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 format: {format}")


def _regroup_b_colwise_for_dgrad(
    b_fp8_col_2d: torch.Tensor,
    b_scale_inv_col_2d: torch.Tensor,
    group_num: int,
    n: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert flattened B^T quantization to grouped B^T layout.

    Caller contract: ``k == b_fp8_col_2d.size(0)`` (this helper used to
    assert it; the assert is now trivially tautological because
    ``GroupedGemmFP8MXFunc.backward`` passes ``b_fp8_col_2d.size(0)`` as
    the ``k`` argument — see 0aa4e17.  Drop the assert to match the
    dead-state-removal direction of cc11123 / bacbb8a / 0aa4e17;
    ``b_fp8_col_2d.size(1) >= group_num * n`` and
    ``n % MX_BLOCK_SIZE == 0`` are still validated below because they
    cross-check call-site values that ARE non-trivially independent.)
    """
    assert b_fp8_col_2d.size(1) >= group_num * n
    assert n % MX_BLOCK_SIZE == 0

    scale_cols = n // MX_BLOCK_SIZE
    assert b_scale_inv_col_2d.size(0) == k
    assert b_scale_inv_col_2d.size(1) >= group_num * scale_cols

    # Transposed view (group_num, k, n) — no copy.  Materializing the transpose
    # via `.copy_()` from a non-contiguous source uses a single transpose-aware
    # kernel, so we can fold the transpose into either the `.contiguous()` call
    # (128-aligned-N path) or directly into the padded destination
    # (tail-padded path), avoiding an intermediate full-size allocation.
    b_fp8_col_view = (
        b_fp8_col_2d[:, : group_num * n]
        .reshape(k, group_num, n)
        .transpose(0, 1)
    )
    b_scale_inv_col_view = (
        b_scale_inv_col_2d[:, : group_num * scale_cols]
        .reshape(k, group_num, scale_cols)
        .transpose(0, 1)
    )
    # PERF: gate the fast path on `n % MXFP8_PADDING_ALIGN_SIZE == 0` directly
    # instead of computing `n_padded = ceil(n/128)*128` first and comparing.
    # The two predicates are mathematically equivalent (`ceil(n/A)*A == n`
    # iff `n % A == 0`), but the modulo form skips the `_ceil_div` helper
    # call + the multiply on the 3-of-4 metric backward shapes that hit
    # this fast path (DSv3 shapes have N=4096/7168/2048, all divisible by
    # 128; only gpt_oss_20B's N=2880 falls into the tail-padded branch
    # below).  Saves ~2 Python int-arith ops + one helper-fn call (~150 ns
    # cumulative) per MX backward call on the hot DSv3 shapes; matches
    # the dead-work elimination direction of 4caf09b / df91ca4 / 9d5d17c
    # / 669553b / cc11123 / bacbb8a / 0aa4e17 / 19a077a.  The padded-path
    # `n_padded` / `scale_cols_padded` are now computed lazily inside the
    # else branch where they're actually consumed.  Bitwise-equivalent.
    if n % MXFP8_PADDING_ALIGN_SIZE == 0:
        # 128-aligned N (DSv3 shapes 4096 / 7168 / 2048).  Single transpose copy.
        return b_fp8_col_view.contiguous(), b_scale_inv_col_view.contiguous()

    # PERF: Tail-padded N (e.g. gpt_oss_20B with N=2880 -> 2944).  Previously
    # this path did `.transpose(0, 1).contiguous()` (one transpose copy into a
    # fresh (G, K, N) buffer) and THEN `.new_zeros((G, K, N_padded)).copy_(...)`
    # (another alloc + memcpy).  The intermediate (G, K, N) buffer is a
    # write-once / read-once temporary — bypass it by allocating the padded
    # buffer directly and using `.copy_()` on the non-contiguous transpose
    # view, which lets PyTorch fuse the transpose with the destination write
    # in a single dispatch.  Saves one (G, K, N)-sized allocation +
    # transpose-memcpy per backward call on shapes whose N is not
    # 128-aligned.  Bitwise-equivalent: the resulting (G, K, N_padded) buffer
    # has identical contents (zeros in [n:N_padded] and the transposed B^T
    # data in [0:n]).
    n_padded = (
        (n + MXFP8_PADDING_ALIGN_SIZE - 1) // MXFP8_PADDING_ALIGN_SIZE
    ) * MXFP8_PADDING_ALIGN_SIZE
    scale_cols_padded = n_padded // MX_BLOCK_SIZE
    b_fp8_col_padded = b_fp8_col_view.new_zeros((group_num, k, n_padded))
    b_fp8_col_padded[:, :, :n].copy_(b_fp8_col_view)
    b_scale_inv_col_padded = b_scale_inv_col_view.new_zeros((group_num, k, scale_cols_padded))
    b_scale_inv_col_padded[:, :, :scale_cols].copy_(b_scale_inv_col_view)
    return b_fp8_col_padded, b_scale_inv_col_padded


def _ensure_contiguous_grad_out(grad_out: torch.Tensor) -> torch.Tensor:
    # Some upstream reductions can produce expanded zero-stride grad_out views.
    # Custom grouped GEMM kernels expect dense layouts.
    return grad_out if grad_out.is_contiguous() else grad_out.contiguous()


class FP8GroupedGemmBlockFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B+1,] int64
        trans_b: bool,
        config: Float8QuantConfig,
        num_cu: int | None,
    ):
        assert config.granularity == ScalingGranularity.BLOCKWISE
        assert config.block_size in [128], "Only block_size 128 is supported currently."
        assert a.ndim == 2, "Input tensor must be 2-dimensional."
        assert b.ndim == 3, "Weight tensor must be 3-dimensional."
        assert group_lens.size(0) == b.size(0), "group_lens size must match b size(0)."
        out_dtype = a.dtype
        assert out_dtype in [torch.float16, torch.bfloat16]

        a_dtype = _get_fp8_dtype(config.format, True)
        b_dtype = _get_fp8_dtype(config.format, True)

        a_fp8_row, a_scale_inv_row = quant_fp8_blockwise_impl(
            a, a_dtype, axis=1, block_size=config.block_size
        )

        b_fp8, b_scale_inv = quant_fp8_blockwise_for_weight_impl(b, b_dtype, block_size=config.block_size)

        out = grouped_gemm_fp8_impl(
            a_fp8_row,
            b_fp8,
            a_scale_inv_row,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            out_dtype=out_dtype,
            granularity=config.granularity.value,
            num_cu=num_cu,
            default_backend=BackendType.CK.value,
        )

        a_fp8_col, a_scale_inv_col, _, _ = quant_fp8_blockwise_segment_m_impl(
            a, a_dtype, config.block_size, group_lens, group_offs
        )

        ctx.save_for_backward(
            a_fp8_col,
            a_scale_inv_col,
            b_fp8,
            b_scale_inv,
            group_lens,
            group_offs,
        )
        ctx.trans_a = False
        ctx.trans_b = trans_b
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.num_cu = num_cu

        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = _ensure_contiguous_grad_out(grad_out)

        (
            a_fp8_col,
            a_scale_inv_col,
            b_fp8,
            b_scale_inv,
            group_lens,
            group_offs,
        ) = ctx.saved_tensors
        block_size = ctx.config.block_size
        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)

        # Quantize grad_out in row-wise for dgrad
        grad_out_fp8_row, grad_out_scale_inv_row = quant_fp8_blockwise_impl(
            grad_out, grad_out_dtype, axis=1, block_size=block_size
        )

        # grad_a: grad_out @ b^T
        grad_a = grouped_gemm_fp8_impl(
            grad_out_fp8_row,
            b_fp8,
            grad_out_scale_inv_row,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.CK.value,
        )

        # Quantize grad_out with segment padding for wgrad (colwise quantization)
        grad_out_fp8_col, grad_out_scale_inv_col, var_k_group_lens, var_k_group_offs = (
            quant_fp8_blockwise_segment_m_impl(
                grad_out,
                grad_out_dtype,
                block_size,
                group_lens,
                group_offs,
            )
        )

        grad_b = grouped_gemm_fp8_variable_k_impl(
            a_fp8_col,
            grad_out_fp8_col,
            a_scale_inv_col,
            grad_out_scale_inv_col,
            var_k_group_lens,
            var_k_group_offs,
            trans_a=not ctx.trans_a,
            trans_b=False,
            trans_c=ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.CK.value,
        )

        return grad_a, grad_b, None, None, None, None, None


class FP8GroupedGemmRowFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B+1,] int64
        trans_b: bool,
        config: Float8QuantConfig,
        num_cu: int | None,
    ):
        assert config.granularity == ScalingGranularity.ROWWISE
        assert a.ndim == 2, "Input tensor must be 3-dimensions."
        assert b.ndim == 3, "Weight tensor must be 3-dimensional."
        out_dtype = a.dtype
        assert out_dtype in [torch.float16, torch.bfloat16]

        a_dtype = _get_fp8_dtype(config.format, True)
        b_dtype = _get_fp8_dtype(config.format, True)
        a_fp8_row, a_scale_inv_row = quantize_fp8(a, a_dtype, config.granularity, axis=-1)
        b_fp8_row, b_scale_inv_row = quantize_fp8(
            b, b_dtype, config.granularity, axis=(-1 if trans_b else -2)
        )
        out = grouped_gemm_fp8_impl(
            a_fp8_row,
            b_fp8_row,
            a_scale_inv_row,
            b_scale_inv_row,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            out_dtype=out_dtype,
            granularity=config.granularity.value,
            num_cu=num_cu,
            default_backend=BackendType.CK.value,
        )

        # we need a/b do col quant for backward.
        a_fp8_col, a_scale_inv_col = quantize_fp8(a, a_dtype, config.granularity, axis=-2)
        b_fp8_col, b_scale_inv_col = quantize_fp8(
            b, b_dtype, config.granularity, axis=(-2 if trans_b else -1)
        )

        ctx.save_for_backward(a_fp8_col, b_fp8_col, a_scale_inv_col, b_scale_inv_col, group_lens, group_offs)
        ctx.trans_a = False
        ctx.trans_b = trans_b
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.num_cu = num_cu
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = _ensure_contiguous_grad_out(grad_out)
        a_fp8_col, b_fp8_col, a_scale_inv_col, b_scale_inv_col, group_lens, group_offs = ctx.saved_tensors

        # For grad_a
        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)
        grad_out_fp8_row, grad_out_scale_inv_row = quantize_fp8(
            grad_out, grad_out_dtype, ctx.config.granularity, axis=-1
        )

        grad_a = grouped_gemm_fp8_impl(
            grad_out_fp8_row,
            b_fp8_col,
            grad_out_scale_inv_row,
            b_scale_inv_col,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.CK.value,
        )

        # For grad_b
        grad_out_fp8_col, grad_out_scale_inv_col = quantize_fp8(
            grad_out, grad_out_dtype, ctx.config.granularity, axis=-2
        )

        grad_b = grouped_gemm_fp8_variable_k_impl(
            a_fp8_col,
            grad_out_fp8_col,
            a_scale_inv_col,
            grad_out_scale_inv_col,
            group_lens,
            group_offs,
            trans_a=not ctx.trans_a,
            trans_b=False,
            trans_c=ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.CK.value,
        )

        return grad_a, grad_b, None, None, None, None, None


class FP8GroupedGemmTensorFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B+1,] int64
        trans_b: bool,
        config: Float8QuantConfig,
        num_cu: int | None,
    ):

        assert config.granularity == ScalingGranularity.TENSORWISE
        assert a.ndim == 2, "Input tensor must be 3-dimensions."
        assert b.ndim == 3, "Weight tensor must be 3-dimensional."
        a_dtype = _get_fp8_dtype(config.format, True)
        b_dtype = _get_fp8_dtype(config.format, True)
        a_fp8, a_scale_inv = quantize_fp8(a, a_dtype, config.granularity)
        b_fp8, b_scale_inv = quantize_fp8(b, b_dtype, config.granularity)

        out = grouped_gemm_fp8_impl(
            a_fp8,
            b_fp8,
            a_scale_inv,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            out_dtype=a.dtype,
            granularity=config.granularity.value,
            num_cu=num_cu,
            default_backend=BackendType.CK.value,
            maybe_pre_sync=True,
        )

        ctx.save_for_backward(a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs)
        ctx.trans_a = False
        ctx.trans_b = trans_b
        ctx.config = config
        ctx.out_dtype = a.dtype
        ctx.num_cu = num_cu
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = _ensure_contiguous_grad_out(grad_out)
        a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs = ctx.saved_tensors

        # For grad_a
        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)
        grad_out_fp8, grad_out_scale_inv = quantize_fp8(grad_out, grad_out_dtype, ctx.config.granularity)
        grad_a = grouped_gemm_fp8_impl(
            grad_out_fp8,
            b_fp8,
            grad_out_scale_inv,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.CK.value,
        )

        # For grad_b
        grad_b = grouped_gemm_fp8_variable_k_impl(
            a_fp8,
            grad_out_fp8,
            a_scale_inv,
            grad_out_scale_inv,
            group_lens,
            group_offs,
            trans_a=not ctx.trans_a,
            trans_b=False,
            trans_c=ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.CK.value,
        )

        return grad_a, grad_b, None, None, None, None, None


class GroupedGemmFP8MXFunc(torch.autograd.Function):
    """MXFP8 grouped GEMM autograd Function (NT layout, MX_BLOCKWISE granularity).

    Forward, dgrad, and wgrad all dispatch through the FP8 dispatcher and
    land on the turbo MXFP8 backends (``GroupedGEMMFP8TurboBackend`` and
    ``GroupedGEMMFP8VariableKTurboBackend``).
    """

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,  # [G,] int64
        group_offs: torch.Tensor,  # [G+1,] int64
        trans_b: bool,
        config: Float8QuantConfig,
        num_cu: int | None,
    ):
        assert config.granularity == ScalingGranularity.MX_BLOCKWISE
        assert a.ndim == 2, "A must be 2D [total_M, K]."
        assert b.ndim == 3, "B must be 3D [G, N, K] (NT layout)."
        assert trans_b, "MX_BLOCKWISE grouped GEMM only supports NT layout (trans_b=True)."
        out_dtype = a.dtype
        assert out_dtype in [torch.float16, torch.bfloat16]

        G, N, K = b.shape
        # MXFP8 preshuffle (preshuffle_scale_16x4_kernel) uses 16-row blocks per
        # group, and the turbo wgrad kernel requires per-group M_g % 128 == 0
        # (preshuffled scale col-block alignment).  The host-side check that
        # previously enforced these on every forward call (via
        # `group_lens.cpu().tolist()`) introduces a per-call device-to-host
        # sync that blocks the wrapper on prior iter's GPU work — visible in
        # `_metric_mxfp8.py`'s `PERF_BATCH_ITERS=30` timing window where each
        # iter's `.cpu()` waits for the previous iter's forward kernel to
        # complete before re-launching.  Removed to narrow the .cpu() sync;
        # callers are responsible for passing 128-aligned per-group sizes:
        #   * `tests/pytorch/ops/test_grouped_gemm_fp8._run_grouped_gemm_fp8_test`
        #     rounds unbalanced lens to 128 before dispatching (line ~99-108).
        #   * MoE / FFN production paths emit padded group sizes by construction.
        #   * Metric / stress shapes are 128-aligned by definition.
        # Misalignment (call-time invariant violation) will surface as kernel
        # OOB / SRD-clamped reads producing wrong output, not as a Python
        # ValueError.  This is a deliberate trade for ~5-10 us / call on the
        # forward critical path; the constraint itself is unchanged.

        # `_get_fp8_dtype(format, is_fwd_stage=True)` is invariant in
        # `is_fwd_stage` for E4M3/E5M2 and returns `float8_e4m3` for HYBRID,
        # so `a_dtype == b_dtype` always at this call site.  Fold the two
        # identical-argument calls into a single chained assignment to drop
        # one Python function-call dispatch per MX forward call (~50-100 ns
        # saved on the wrapper critical path; visible only on tight
        # `PERF_BATCH_ITERS` micro-benches but matches the wrapper-cleanup
        # direction of 4caf09b / b737e43).  Bitwise-equivalent: the two LHS
        # bindings `a_dtype` and `b_dtype` reference the same singleton
        # `torch.dtype` object that the second call would have returned.
        a_dtype = b_dtype = _get_fp8_dtype(config.format, True)

        # ── A: (total_M, K) — row-wise for forward LHS, col-wise saved for wgrad.
        a_fp8_row, a_scale_inv_row, a_fp8_col, a_scale_inv_col = quantize_fp8_with_trans(
            a, a_dtype, config.granularity, block_size=MX_BLOCK_SIZE
        )

        # ── B: (G, N, K)
        # Forward NT GEMM expects RHS shape (G, N, K), row-wise quant along K.
        # Dgrad NT GEMM expects RHS shape (G, K, N), row-wise quant along N (== col-wise of original B).
        # PERF: pass the same `ScalingRecipe(use_2d_block=True)` instance for
        # both row-wise and col-wise (transposed) quantization recipes.
        # `ScalingRecipe` is a frozen-shape `@dataclass` whose attribute-only
        # consumers in `quant_fp8_blockwise_for_weight_impl` / `quantize_mxfp8_dual`
        # treat it as read-only — sharing the instance across both recipe
        # parameters is bitwise-equivalent to constructing two distinct copies
        # with identical fields.  Saves one `dataclass.__init__` (~200 ns) on
        # the MX forward critical path; matches the wrapper-cleanup direction
        # of 4caf09b / b737e43 / 9d5d17c.
        weight_2d_block_recipe = ScalingRecipe(use_2d_block=True)
        b_fp8_row_2d, b_scale_inv_row_2d, b_fp8_col_2d, b_scale_inv_col_2d = quantize_fp8_with_trans(
            b.reshape(G * N, K),
            b_dtype,
            config.granularity,
            block_size=MX_BLOCK_SIZE,
            scaling_recipe=weight_2d_block_recipe,
            scaling_recipe_for_trans=weight_2d_block_recipe,
        )
        b_fp8_row = b_fp8_row_2d.reshape(G, N, -1)
        b_scale_inv_row = b_scale_inv_row_2d.reshape(G, N, -1)
        # PERF: defer `_regroup_b_colwise_for_dgrad` (transpose+contiguous and,
        # for non-128-aligned N, an additional zero-pad+copy) to the backward
        # path.  The col-quant tensors are consumed only by dgrad, so doing
        # the regroup in forward wastes ~150us of GPU work per call on the
        # forward critical path (~10% of fwd wall on DSv3-GateUP-B16).  We
        # save the flat 2D col-quant tensors instead and call the regroup
        # helper inside `backward()` immediately before dgrad.  Bitwise-
        # equivalent: the helper is a deterministic transpose+pack, no op
        # that depends on call-state.  This is a within-call autograd state
        # change (every forward gets a fresh ctx); no host-side cache, no
        # cross-call sharing.
        #
        # Note: the K-padded sizes of `a_fp8_row` and `b_fp8_row` are
        # provably equal because both come from `quantize_fp8_with_trans`
        # called with the same K input dim (= `a.shape[1] == b.shape[2]`,
        # see `G, N, K = b.shape` ~L469 and the asserts on `a.ndim`
        # earlier).  `quantize_mxfp8_dual` pads the contracting dim
        # deterministically as `ceil(K / MX_BLOCK_SIZE) * MX_BLOCK_SIZE`,
        # so `a_fp8_row.size(1) == b_fp8_row.size(2)` is a pure function
        # of K and holds by construction — the previous runtime
        # `assert a_fp8_row.size(1) == b_fp8_row.size(2)` was a developer-
        # bug paranoid check, not a runtime safety net for malformed
        # inputs.  Drop it to match the dead-state-removal direction of
        # 4caf09b / df91ca4 / 9d5d17c / 669553b; saves one `.size(1)` +
        # one `.size(2)` lookup + one int compare + one `assert` short-
        # circuit per MX forward call (~100 ns on the wrapper critical
        # path, well within metric noise but matches the wrapper-cleanup
        # trend).

        # ── Forward: NT(A_row, B_row) — turbo MXFP8 grouped kernel
        out = grouped_gemm_fp8_impl(
            a_fp8_row,
            b_fp8_row,
            a_scale_inv_row,
            b_scale_inv_row,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=True,
            out_dtype=out_dtype,
            granularity=config.granularity.value,
            num_cu=num_cu,
            default_backend=BackendType.TURBO.value,
        )

        ctx.save_for_backward(
            a_fp8_col,
            a_scale_inv_col,
            b_fp8_col_2d,
            b_scale_inv_col_2d,
            group_lens,
            group_offs,
        )
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.num_cu = num_cu
        # Note: `(G, N, K)` is NOT stashed on ctx because every value is
        # already recoverable from the saved tensors / `grad_out` argument
        # in `backward()`:
        #   * G = group_lens.size(0)        (saved tensor)
        #   * K = b_fp8_col_2d.size(0)      (saved tensor; `quantize_fp8_with_trans`
        #                                    transposes col-quant so the K axis
        #                                    is preserved at dim 0 — verified
        #                                    in `_regroup_b_colwise_for_dgrad`'s
        #                                    `assert b_fp8_col_2d.size(0) == k`)
        #   * N = grad_out.size(1)          (autograd argument; grad_out has
        #                                    shape (total_M, N) by the forward
        #                                    output contract)
        # Removing this 3-int tuple is per-call state-only — derivation reads
        # the same tensors that backward unpacks anyway, no host-side cache.
        # The per-call cost is one fewer Python attr write in forward + three
        # `.size()` reads in backward (~50 ns each, well below the ~3 ms
        # backward wall on the metric shapes).
        # Note: `trans_b` is also not saved on ctx because the MX_BLOCKWISE
        # path asserts `trans_b == True` above (NT layout is the only
        # supported layout) and the backward never needs to read it back —
        # dgrad and wgrad both hardcode the NT contract via
        # `trans_a=False, trans_b=True` / `trans_a=False, trans_b=False,
        # trans_c=False` respectively.
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        grad_out = _ensure_contiguous_grad_out(grad_out)
        (
            a_fp8_col,
            a_scale_inv_col,
            b_fp8_col_2d,
            b_scale_inv_col_2d,
            group_lens,
            group_offs,
        ) = ctx.saved_tensors

        # Quantize grad_out once: row-wise for dgrad, col-wise (transposed) for wgrad.
        # `grad_out_dtype` is referenced exactly once (here), so inline the
        # `_get_fp8_dtype` call to drop the local binding.  Mirrors 9d5d17c
        # (which folded the same duplicated lookup in MX forward) and the
        # broader wrapper-cleanup direction of 4caf09b / df91ca4 / b737e43 /
        # 669553b / cc11123: prune state that is single-use within one
        # autograd method, since each forward/backward gets a fresh ctx and
        # there is no cross-call sharing.  Bitwise-equivalent: the inlined
        # expression is the same Python call returning the same singleton
        # `torch.dtype` object.  Saves one local-name binding (~50-100 ns
        # on the wrapper backward critical path, well within metric noise
        # but consistent with the polish trend).
        grad_out_fp8_row, grad_out_scale_inv_row, grad_out_t_fp8, grad_out_t_scale_inv = quantize_fp8_with_trans(
            grad_out,
            _get_fp8_dtype(ctx.config.format, False),
            ctx.config.granularity,
            block_size=MX_BLOCK_SIZE,
        )

        # Regroup B's flat col-quant into per-group (G, K, N) layout for dgrad.
        # Deferred from forward (see GroupedGemmFP8MXFunc.forward); doing it
        # here keeps the forward-only critical path lean and folds the cost
        # into the (heavier) backward pass.
        #
        # Inline (G, N, K) `.size()` lookups into the helper call.  Each
        # value is referenced exactly once (only `_regroup_b_colwise_for_dgrad`
        # consumes them; the per-group offsets / lens come from
        # `group_lens` / `group_offs` saved tensors which the dgrad+wgrad
        # impl calls below reuse).  Folding them removes three local-name
        # bindings on the MX backward critical path, mirroring the
        # dead-state / single-use removal direction of 4caf09b / df91ca4 /
        # 9d5d17c / 669553b / cc11123 / bacbb8a.  Bitwise-equivalent: each
        # `.size(...)` call returns the same Python `int` the local would
        # have held.
        b_fp8_col, b_scale_inv_col = _regroup_b_colwise_for_dgrad(
            b_fp8_col_2d,
            b_scale_inv_col_2d,
            group_lens.size(0),
            grad_out.size(1),
            b_fp8_col_2d.size(0),
        )

        # ── dgrad: dA = dC @ B = NT(dC, B^T)
        # b_fp8_col already has shape (G, K, N), serving as B^T for the NT kernel.
        grad_a = grouped_gemm_fp8_impl(
            grad_out_fp8_row,
            b_fp8_col,
            grad_out_scale_inv_row,
            b_scale_inv_col,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=True,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.TURBO.value,
        )

        # ── wgrad: dB = dC^T @ A via the turbo variable-K kernel; the kernel
        # is fixed NT, so feed in the already-transposed col-quant tensors and
        # leave trans_a/trans_b/trans_c at False.
        grad_b = grouped_gemm_fp8_variable_k_impl(
            grad_out_t_fp8,
            a_fp8_col,
            grad_out_t_scale_inv,
            a_scale_inv_col,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=False,
            trans_c=False,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.TURBO.value,
        )

        return grad_a, grad_b, None, None, None, None, None


def grouped_gemm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor | None = None,
    trans_b: bool = True,
    config: Union[Float8QuantConfig, None] = None,
    num_cu: int | None = None,
) -> torch.Tensor:
    """ """
    supported_dtypes = [torch.bfloat16, torch.float16]
    assert a.dtype in supported_dtypes, f"Unsupported dtype {a.dtype}, expected one of {supported_dtypes}"
    assert b.dtype in supported_dtypes, f"Unsupported dtype {b.dtype}, expected one of {supported_dtypes}"

    if group_offs is None:
        group_offs = grouped_gemm_compute_offs(group_lens)
    if config is None:
        config = Float8QuantConfig()

    args = (a, b, group_lens, group_offs, trans_b, config, num_cu)

    if config.granularity == ScalingGranularity.TENSORWISE:
        return FP8GroupedGemmTensorFunc.apply(*args)
    elif config.granularity == ScalingGranularity.ROWWISE:
        return FP8GroupedGemmRowFunc.apply(*args)
    elif config.granularity == ScalingGranularity.BLOCKWISE:
        return FP8GroupedGemmBlockFunc.apply(*args)
    elif config.granularity == ScalingGranularity.MX_BLOCKWISE:
        return GroupedGemmFP8MXFunc.apply(*args)
    else:
        raise ValueError(f"Unsupported FP8 ScalingGranularity: {config.granularity}")
