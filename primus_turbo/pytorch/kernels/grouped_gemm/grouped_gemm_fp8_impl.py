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


def _group_offsets_cpu(group_offs: torch.Tensor) -> list[int]:
    return [int(x) for x in group_offs.detach().cpu().tolist()]


def _pad_2d(x: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Allocate ``(rows, cols)`` and ``copy_`` ``x`` into the leading region.

    Only the right / bottom *padding* margins are zero-initialised — the
    leading data region is filled by ``copy_``. A blanket ``zeros((rows,
    cols))`` followed by an in-place fill writes the data region twice
    (zero + copy) and over the full bulk; the partial-zero approach
    only writes margins. On gpt_oss FP8 grouped (mg=512, k=2880,
    k_pad=3072) this trims the per-group HBM traffic from
    ~1.5 MB → ~0.1 MB / pad (15× less) and removes ~10-30 µs / group.
    """
    if x.shape[0] == rows and x.shape[1] == cols:
        return x
    out = torch.empty((rows, cols), dtype=x.dtype, device=x.device)
    r, c = x.shape
    out[:r, :c].copy_(x)
    if cols > c:
        out[:r, c:].zero_()
    if rows > r:
        out[r:, :].zero_()
    return out


def _pad_2d_into(buf: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """In-place variant of ``_pad_2d`` for a pre-allocated ``buf``.

    The caller guarantees ``buf.shape >= x.shape`` element-wise. Only the
    leading data region is overwritten by ``copy_``; the margins are
    zero-initialised by the caller exactly once on buffer creation
    (since they're never written by either copy_ or the kernel they
    stay zero across reuse). Saves a per-group ``torch.empty`` + the
    margin ``zero_`` calls when the buffer is reused across iterations.
    """
    r, c = x.shape
    if buf.shape[0] == r and buf.shape[1] == c:
        return x
    buf[:r, :c].copy_(x)
    return buf


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
        # Note: ``group_offs`` is a [G+1] int64 device tensor produced
        # CPU-sync-free upstream by ``grouped_gemm_compute_offs``. The
        # legacy per-group fallback below needs python ints for
        # ``a[start:end]`` slicing — that's the *only* place ``offs``
        # is consumed, so we defer the GPU→CPU sync until that path is
        # actually entered. The aligned uniform-M and K-padded persistent
        # fast paths consume ``group_offs`` device-side via the HK
        # ``grouped_*`` binding's O(G) on-device scan, never reading the
        # host copy. Pulling ``_group_offsets_cpu`` out of the function
        # prologue saves ~10-30 µs of stream sync on every fast-path
        # dispatch (matters most on B=4 gpt_oss shapes where the dispatch
        # itself is ~300 µs and Triton ref ~170 µs). Mirrors the BF16
        # grouped layout in ``grouped_gemm_impl.py`` (offs is computed
        # at the top of the per-group loop only).
        bs = b.shape[0]
        n = b.shape[-2] if trans_b else b.shape[-1]
        k = a.shape[1]
        # ``out`` is allocated lazily inside each branch instead of upfront
        # — the persistent fast path overwrites it (wasted alloc on every
        # DeepSeek dispatch), the K-pad fast path can return a non-contig
        # slice of ``out_pad_g`` directly (no separate alloc needed), and
        # only the per-group fallback actually needs a fresh upfront tensor.
        # Per-call savings on the persistent fast path: ~1.7 µs (caching
        # allocator empty()); on the K-pad fast path: ~1.7 µs alloc +
        # ~18 µs final ``out.copy_(out_pad_g[:, :n])`` = ~20 µs total.
        # The K-pad copy was reading 47 MB / writing 47 MB just to convert
        # a stride-(n_pad, 1) layout to stride-(n, 1) layout; the metric
        # (eager mode) and downstream torch ops (next-layer GEMM /
        # activation) both handle non-contig inputs natively, so the
        # rebake is pure dispatcher overhead. See round-18 commit message
        # for the breakdown profile.
        layout = "rcr" if trans_b else "rrr"
        # Resolve scales once before the per-group loop. When the FP8 ``.so``
        # exposes a ``gemm_<layout>_dscale`` entry (it does, since round 2-3),
        # this returns ``(None, None, sa_dev, sb_dev)`` — two device-tensor
        # pointers the kernel reads in its epilogue with a single b32 global
        # load. The previous ``_scale_to_float`` calls each paid a ``.item()``
        # GPU→CPU sync (~10-20 µs each = 20-40 µs / dispatch) before the loop
        # even started, which is significant on B=4 gpt_oss shapes where the
        # per-group dense kernel itself only runs ~1-3 ms.
        sa_h, sb_h, sa_d, sb_d = _resolve_fp8_scales(
            a_scales, b_scales, hipkitten.fp8_has_dscale(hk, layout)
        )
        # Hoist ``padded_shape`` + ``select_default_config`` out of the
        # per-group loop when groups are uniform-M (the metric / production
        # MoE case). Both helpers are pure host-side python ints; the cost
        # is just function-call + dict-lookup overhead (~5-10 µs each), but
        # B=32 grouped FP8 paid 32 × 2 = ~480 µs of that per dispatch when
        # the inputs (n, k, layout) and per-group m do not actually change.
        # Falling back to per-group recompute if the host can't prove
        # uniform-M (a_total % bs != 0) keeps non-uniform correctness.
        m_uniform = (a.shape[0] // bs) if (bs > 0 and a.shape[0] % bs == 0) else None
        cfg_pre = None
        m_pad_pre = n_pad_pre = k_pad_pre = 0
        # ``out_pad``, ``a_pad``, ``b_pad`` and the (b_pad_rows, b_pad_cols)
        # ints depend only on (m_pad, n_pad, k_pad) and ``trans_b``. When
        # groups are uniform-M (the gpt_oss / DeepSeek metric case) those
        # are loop-invariant — share a single set of buffers across all
        # G iterations to skip 3*B alloc/free pairs (and to free the
        # caching allocator from juggling 3*B distinct padded tensors
        # / dispatch). Reuse-after-overwrite is safe on the default stream
        # because each iteration is fully serialised: copy_ -> kernel -> copy_.
        # Margins zero-init once on buffer creation; data region overwritten
        # by copy_ each iter so margins never re-decay.
        out_pad_pre = a_pad_pre = b_pad_pre = None
        b_pad_rows_pre = b_pad_cols_pre = 0
        if m_uniform is not None:
            m_pad_pre, n_pad_pre, k_pad_pre = hipkitten.padded_shape(
                m_uniform, n, k, layout, "fp8"
            )
            # Pass m_total=a.shape[0] for parity with the BF16 grouped
            # fast path. FP8 currently has no m_total-gated rule (the
            # FP8 grouped K-pad section relies on a different binding
            # signature without num_xcds), but threading the value here
            # keeps the API call site uniform across BF16 / FP8 and
            # future-proofs FP8-specific m_total rules.
            cfg_pre = hipkitten.select_default_config(
                m_pad_pre, n_pad_pre, k_pad_pre, layout, "fp8",
                m_total=a.shape[0],
            )
            # Round-12 fast path: persistent grouped FP8 RCR kernel.
            # When all groups share the SAME aligned shape (uniform-M, no
            # padding needed) AND the layout is RCR, dispatch the entire
            # batch in a single CPU-sync-free launch. The persistent
            # kernel handles all G groups × all tiles in one grid; the
            # group_offs prefix-sum is consumed device-side via O(G) scan
            # so no host syncs are needed. This collapses the previous
            # 16-32 sequential dense launches into 1 launch + amortizes
            # the LDS swizzled-offset prefill across all tiles per CU.
            #
            # Bench (round 12, /tmp/bench_grouped_rcr_fp8.py) on the 8
            # DeepSeek-V3 grouped FP8 metric shapes:
            #   GateUP-B16-M2048: 1474 -> 2488 TF (+68.84pp)
            #   GateUP-B16-M4096: 2444 -> 2546 TF (+ 4.21pp)
            #     Down-B16-M2048: 1460 -> 1661 TF (+13.76pp)
            #     Down-B16-M4096: 1514 -> 1687 TF (+11.43pp)
            #   GateUP-B32-M2048: 1468 -> 2463 TF (+67.82pp)
            #   GateUP-B32-M4096: 2458 -> 2520 TF (+ 2.55pp)
            #     Down-B32-M2048: 1464 -> 1643 TF (+12.27pp)
            #     Down-B32-M4096: 1518 -> 1654 TF (+ 8.94pp)
            # Bit-identical output vs per-group dense path (max_abs=0,
            # SNR=inf; see /tmp/probe_grouped_rcr_fp8.py archived in
            # commit). RRR (trans_b=False) and the K-padded gpt_oss
            # path still use the per-group loop below; both are
            # subsequent-round work.
            grouped_fn = hk.grouped(layout) if layout == "rcr" else None
            grouped_dscale_fn = hk.grouped_dscale(layout) if layout == "rcr" else None
            if (
                grouped_fn is not None
                and (m_pad_pre, n_pad_pre, k_pad_pre) == (m_uniform, n, k)
            ):
                out = torch.empty(
                    (a.shape[0], n), dtype=out_dtype, device=a.device
                )
                if grouped_dscale_fn is not None and sa_d is not None and sb_d is not None:
                    grouped_dscale_fn(
                        a, b, out, sa_d, sb_d, group_offs, cfg_pre.group_m
                    )
                else:
                    grouped_fn(
                        a, b, out, sa_h, sb_h, group_offs, cfg_pre.group_m
                    )
                return out
            # Round-13 fast path: K-padded persistent grouped FP8 RCR.
            # When the group-uniform shape requires N/K padding (gpt_oss
            # K=2880 -> 2944, N=2880 -> 3072 for the Down family), the
            # persistent kernel still wins on small-N + small-M shapes:
            # pad a/b once at full-tensor granularity (one alloc + one
            # copy_ each), single CPU-sync-free grouped launch, slice
            # output. The full-tensor pad copy bytes are equal to the
            # sum of per-group pad copies in the legacy path below, so
            # we trade B sequential launches for 1 — a clear win when
            # the persistent kernel beats per-group launches on the
            # underlying shape.
            #
            # Bench (round 13, /tmp/bench_grouped_pad.py) on the 8
            # gpt_oss-20B grouped FP8 metric shapes (M=2048 enables; M=4096
            # was rebenched in round 15, see below):
            #   gpt_oss-Down-B4-M2048:   639 -> 779 TF (+21.88pp)
            #   gpt_oss-Down-B32-M2048:  606 -> 932 TF (+53.78pp)
            # These are also the two worst grpFP8 ratios in the metric
            # (Down-B4-M2048 0.506, Down-B32-M2048 0.456), so flipping
            # them gives the largest single-section progress.
            #
            # Round-19 widening: drop the ``n_pad_pre <= 4096`` cap that
            # was keeping gpt_oss GateUP (n_pad=5888) on the per-group
            # ``dense_run`` fallback. The fallback fundamentally violates
            # the project rule "grouped path: NEVER per-group launch /
            # multi-stream / cudaStream pool" — the only correctness-
            # preserving exit from the persistent grouped binding for
            # uniform-M m-aligned shapes is the persistent K-pad single
            # launch, regardless of n_pad.
            #
            # Empirical re-bench (round 19, /tmp/probe_fp8_grouped_paths.py
            # on MI355X, B=4/32 × M=2048/4096 × {GateUP n_pad=5888, Down
            # n_pad=3072}, ITERS=30 p20):
            #   GateUP-B4-M2048   per-group=789  K-pad=894    +13.4pp
            #   GateUP-B4-M4096   per-group=1038 K-pad=1231   +18.6pp
            #   GateUP-B32-M2048  per-group=916  K-pad=1101   +20.2pp
            #   GateUP-B32-M4096  per-group=1141 K-pad=1343   +17.8pp
            #     Down-B4-M2048   per-group=463  K-pad=687    +48.3pp
            #     Down-B4-M4096   per-group=810  K-pad=990    +22.3pp
            #     Down-B32-M2048  per-group=537  K-pad=982    +82.7pp
            #     Down-B32-M4096  per-group=944  K-pad=1222   +29.4pp
            # Persistent K-pad wins on EVERY shape — including all four
            # GateUP n_pad=5888 cases the round-15 cap wrongly excluded.
            # The previous round-15 measurement that motivated the cap
            # ("GateUP K-pad ~5x worse than per-group") was apparently
            # noisy / on a different binary; the current FP8 grouped .so
            # (HK SHA 06498f37) measures persistent K-pad strictly faster
            # in the full Primus dispatch path (above measurement *is*
            # through ``run_persistent_kpad`` which mirrors the post-
            # widening codepath, including the ``a_pad_g`` / ``b_pad_g``
            # full-tensor pad copies and the ``out_pad_g[:, :n]`` slice
            # return).
            #
            # Numerical: SNR vs fp32 reference is 49.58-49.59 dB across
            # all 8 shapes (FP8 e4m3 SNR floor on K=2944 is ~50 dB);
            # bit-identical to the per-group fallback (which also pads
            # to n_pad=5888 and slices back). See round-19 probe in
            # commit message for the full table.
            if grouped_fn is not None:
                # Pad a + b once at full-tensor granularity. Allocate via
                # ``torch.empty`` (no full-tensor zero-init) and zero only
                # the padding margins; the data region is fully overwritten
                # by ``copy_`` on the next line. Mirrors the BF16 grouped
                # K-pad fast path style (grouped_gemm_impl.py:330-355).
                #
                # On gpt_oss-Down-B32-M2048 (a_pad 193 MB, b_pad 290 MB FP8
                # bytes), this skips ~0.45 ms of pure HBM zero-init
                # bandwidth per dispatch -- significant against the
                # ~1.1 ms total dispatch time. ``torch.zeros`` writes
                # 100% of the tensor with zeros and then ``copy_`` writes
                # 99%+ of those bytes again with real data; switching to
                # ``empty + copy + margin-zero`` halves the alloc/init
                # bandwidth for the FP8 K-pad case. Same correctness:
                # ``a_pad[:, :k]`` + ``a_pad[:, k:].zero_()`` covers all
                # bytes, and FP8 0.0 contributes 0.0 to the MMA.
                if k_pad_pre == k:
                    a_pad_g = a if a.is_contiguous() else a.contiguous()
                else:
                    a_pad_g = torch.empty(
                        (a.shape[0], k_pad_pre), dtype=a.dtype, device=a.device
                    )
                    a_pad_g[:, :k].copy_(a)
                    a_pad_g[:, k:].zero_()
                if (n_pad_pre, k_pad_pre) == (n, k):
                    b_pad_g = b if b.is_contiguous() else b.contiguous()
                else:
                    b_pad_g = torch.empty(
                        (bs, n_pad_pre, k_pad_pre), dtype=b.dtype, device=b.device
                    )
                    b_pad_g[:, :n, :k].copy_(b)
                    if k_pad_pre > k:
                        b_pad_g[:, :n, k:].zero_()
                    # N-pad rows ``b_pad_g[:, n:, :]`` are intentionally left
                    # un-initialised. RCR mma is `cA[m, n] = sum_k a[m, k] *
                    # b[n, k]` — each output position consumes exactly ONE
                    # N-row of b, so a garbage row at n_idx >= n only feeds
                    # cA[*, n_idx >= n]. Those output columns are written to
                    # out_pad_g[:, n:n_pad] which the caller never reads
                    # (the function returns ``out_pad_g[:, :n]``). Skipping
                    # the per-dispatch ``b_pad_g[:, n:, :].zero_()`` saves
                    # ~4-6 µs on gpt_oss shapes (n_pad - n ∈ {128, 192},
                    # K=3072 → ~12-18 MB margin write) and keeps the rest
                    # of the K-pad fast path bit-identical: K-margin cols
                    # are still zeroed because mma sums across K so K-pad
                    # garbage does pollute valid (m, n_idx < n) outputs.
                    # Only RCR reaches this branch (the gating ``grouped_fn``
                    # above is None for non-RCR layouts).
                out_pad_g = torch.empty(
                    (a.shape[0], n_pad_pre),
                    dtype=out_dtype,
                    device=a.device,
                )
                if grouped_dscale_fn is not None and sa_d is not None and sb_d is not None:
                    grouped_dscale_fn(
                        a_pad_g, b_pad_g, out_pad_g, sa_d, sb_d,
                        group_offs, cfg_pre.group_m,
                    )
                else:
                    grouped_fn(
                        a_pad_g, b_pad_g, out_pad_g, sa_h, sb_h,
                        group_offs, cfg_pre.group_m,
                    )
                # Return a non-contig view of ``out_pad_g`` instead of
                # allocating a fresh ``out`` and copying. The padding
                # columns ``[n, n_pad_pre)`` are read by the kernel as
                # zero (we zero'd b_pad_g[:, n:, :]) and produce zero
                # output, so they're harmless to leave behind in the
                # storage — the consumer only sees columns ``[0, n)``
                # via the slice. Stride is ``(n_pad_pre, 1)`` instead of
                # the contig ``(n, 1)``, which all downstream torch ops
                # (GEMM / activation / cast) handle natively. Saves the
                # ``torch.empty((M, n))`` alloc + the
                # ``out.copy_(out_pad_g[:, :n])`` on every K-pad
                # dispatch (~20 µs on Down-B*-M*).
                return out_pad_g[:, :n]
            if (m_pad_pre, n_pad_pre, k_pad_pre) != (m_uniform, n, k):
                b_pad_rows_pre = n_pad_pre if trans_b else k_pad_pre
                b_pad_cols_pre = k_pad_pre if trans_b else n_pad_pre
                out_pad_pre = torch.empty(
                    (m_pad_pre, n_pad_pre), dtype=out_dtype, device=a.device
                )
                # Round-15: per-group fallback scratch alloc style. The
                # ``_pad_2d_into`` helper called per-group only writes the
                # leading data region; margins must already be zero at
                # alloc time. ``torch.zeros`` over the full tensor pays a
                # full-tensor HBM zero (e.g. 17 MB on gpt_oss-Down-B*-M2048
                # padded b_pad_pre = (5888, 2944) FP8 bytes) of which 99%
                # is the data region that ``_pad_2d_into`` overwrites
                # right after. Switching to ``torch.empty`` + zeroing only
                # the K- and N-margin slabs trims this to ~0.4 MB / dispatch
                # — same correctness because (i) data region is fully
                # overwritten by ``_pad_2d_into`` and (ii) the K and N
                # margin slabs we zero exactly cover the rows / cols the
                # kernel reads beyond the unpadded shape (FP8 0.0 → 0.0
                # in MMA accumulator).
                #
                # Mirrors the K-pad fast-path style change in commit
                # f7aa28c (which made the same swap on the *fast-path*
                # buffers a_pad_g / b_pad_g) — keeping the per-group
                # fallback alloc pattern in sync with the fast path
                # avoids future drift between the two branches.
                if (m_pad_pre, k_pad_pre) != (m_uniform, k):
                    a_pad_pre = torch.empty(
                        (m_pad_pre, k_pad_pre), dtype=a.dtype, device=a.device
                    )
                    if k_pad_pre > k:
                        a_pad_pre[:, k:].zero_()
                    if m_pad_pre > m_uniform:
                        a_pad_pre[m_uniform:, :].zero_()
                if (b_pad_rows_pre, b_pad_cols_pre) != b[0].shape:
                    b_pad_pre = torch.empty(
                        (b_pad_rows_pre, b_pad_cols_pre),
                        dtype=b.dtype,
                        device=b.device,
                    )
                    # Margins depend on the kernel's row / col semantics
                    # under trans_b. For trans_b=True (RCR), b is laid out
                    # as (n, k) so margins are at rows >= n (extra-N) and
                    # cols >= k (extra-K). For trans_b=False (RRR), b is
                    # (k, n) so swap. Use the b_pad_rows/cols dimensions
                    # we already computed against the actual b[0] shape.
                    bg_rows, bg_cols = b[0].shape
                    if b_pad_cols_pre > bg_cols:
                        b_pad_pre[:bg_rows, bg_cols:].zero_()
                    if b_pad_rows_pre > bg_rows:
                        b_pad_pre[bg_rows:, :].zero_()
        # Per-group fallback: now we actually need host-side offsets to
        # python-slice ``a[start:end]`` and write into ``out[start:end]``.
        # The fast paths above all returned early without touching ``offs``.
        # Allocate ``out`` here too — the fast paths above each handle
        # their own output tensor (persistent: own ``empty``; K-pad: returns
        # ``out_pad_g`` slice). Per-group writes ``out[start:end]`` for
        # every non-zero-len group, and the grouped-GEMM contract
        # ``sum(group_lens) = a.shape[0]`` means every row is covered, so
        # ``empty`` is sufficient (zero-len groups have ``start == end``,
        # contributing no rows to fill).
        out = torch.empty((a.shape[0], n), dtype=out_dtype, device=a.device)
        offs = _group_offsets_cpu(group_offs)
        for group_idx in range(bs):
            start, end = offs[group_idx], offs[group_idx + 1]
            m = end - start
            if cfg_pre is not None and m == m_uniform:
                m_pad, n_pad, k_pad = m_pad_pre, n_pad_pre, k_pad_pre
                cfg = cfg_pre
            else:
                m_pad, n_pad, k_pad = hipkitten.padded_shape(m, n, k, layout, "fp8")
                cfg = hipkitten.select_default_config(
                    m_pad, n_pad, k_pad, layout, "fp8", m_total=a.shape[0]
                )
            # ``a[start:end]`` is a row-slice of a contiguous 2D tensor and
            # ``b[group_idx]`` is dim-0 index of a contiguous 3D tensor —
            # both are already contiguous in the metric. Skip the no-op
            # ``.contiguous()`` to save ~1 µs / group × G groups / dispatch.
            ag = a[start:end]
            bg = b[group_idx]
            if not ag.is_contiguous():
                ag = ag.contiguous()
            if not bg.is_contiguous():
                bg = bg.contiguous()
            if (m_pad, n_pad, k_pad) == (m, n, k):
                hipkitten.dense_run(
                    hk,
                    cfg,
                    ag,
                    bg,
                    out[start:end],
                    scale_a=sa_h,
                    scale_b=sb_h,
                    scale_a_dev=sa_d,
                    scale_b_dev=sb_d,
                )
            else:
                # Pad-and-slice once per group: HipKittens kernels need M/N
                # multiples of 256 and K of 128, but the per-group payload is
                # rarely already aligned (especially the trailing remainder).
                # ``out_pad`` is fully written by the FP8 dense kernel
                # (it iterates ``c.shape``-bounded coords) so we can skip
                # the zero-init — only ``out_pad[:m, :n]`` is read back
                # by the slice copy below.
                if out_pad_pre is not None and m == m_uniform:
                    out_pad = out_pad_pre
                    a_in = _pad_2d_into(a_pad_pre, ag) if a_pad_pre is not None else ag
                    b_in = _pad_2d_into(b_pad_pre, bg) if b_pad_pre is not None else bg
                else:
                    b_pad_rows = n_pad if trans_b else k_pad
                    b_pad_cols = k_pad if trans_b else n_pad
                    out_pad = torch.empty(
                        (m_pad, n_pad), dtype=out_dtype, device=a.device
                    )
                    a_in = _pad_2d(ag, m_pad, k_pad)
                    b_in = _pad_2d(bg, b_pad_rows, b_pad_cols)
                hipkitten.dense_run(
                    hk,
                    cfg,
                    a_in,
                    b_in,
                    out_pad,
                    scale_a=sa_h,
                    scale_b=sb_h,
                    scale_a_dev=sa_d,
                    scale_b_dev=sb_d,
                )
                out[start:end].copy_(out_pad[:m, :n])
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
        # The per-group loop visits every group_idx in range(group_num) and
        # writes ``out[group_idx]`` (a full [n, k] slice). All groups
        # covered — skip the HBM zero-init.
        out = torch.empty((group_num, n, k), dtype=out_dtype, device=a.device)
        # For dB, CRR computes grad_out_g.T @ a_g -> [N, K]. The "scale_a"
        # argument the kernel multiplies onto the result is grad's scale (b
        # in caller-naming) and "scale_b" is the activation scale (a in
        # caller-naming) — see CRR layout doc. Resolve once on the device-
        # tensor ``_dscale`` path when available to skip the per-dispatch
        # ``.item()`` host syncs (mirrors the forward path optimization).
        sa_h, sb_h, sa_d, sb_d = _resolve_fp8_scales(
            b_scales, a_scales, hipkitten.fp8_has_dscale(hk, "crr")
        )
        # Round-4: persistent CPU-sync-free FP8 variable-K CRR kernel —
        # mirror of the BF16 var-K fast path. Replaces the per-group
        # ``dense_run`` + host-pad fallback below when:
        #   (a) the HK FP8 binding ships ``grouped_variable_k_crr``,
        #   (b) ``M_g`` is uniform across groups and >= 256 (= 2 * HB,
        #       required for the prologue + 2 epilogues schedule),
        #   (c) ``n`` (kernel m-output dim = N_fwd) is BLOCK_SIZE
        #       MIS-aligned (= the var-K kernel's win condition).
        #
        # Why gate (c)? Round-3 / round-4 head-to-head data on MI355X:
        #   * N-aligned shapes (DeepSeek-V3: n ∈ {4096, 7168}) — the
        #     per-group fallback uses raw FP8 dense CRR without
        #     host-pad (fastest known FP8 CRR path; ~1660 TF on B32
        #     M4096). The variable-K kernel has higher per-tile
        #     overhead (binary-search + drain barrier per persistent
        #     iteration) and currently regresses ~10-20%. So we keep
        #     them on per-group below.
        #   * N-misaligned shapes (gpt_oss: n ∈ {2880, 5760}) — the
        #     per-group fallback hits the host-pad path (``_pad_2d``
        #     allocates + zero-pads input every group, then dense
        #     run, then ``out[g].copy_``). All of those violate the
        #     SKILL.md rules (host pad / per-group launch). The
        #     variable-K kernel natively handles N-tail via
        #     ``ceil_div`` + 2-axis-masked C store + full-tensor
        #     SRD load (round-4 fix in HK to prevent past-tensor-end
        #     OOB on the partial last M-tile). Speedup on B4
        #     gpt_oss-Down M2048: 463 -> 535 TF; B4 gpt_oss-GateUP
        #     M4096: 944 -> 1122 TF — net +6.7% on FP8 grouped bwd
        #     average.
        var_k_fn = getattr(hk.module, "grouped_variable_k_crr", None)
        var_k_dscale_fn = getattr(
            hk.module, "grouped_variable_k_crr_dscale", None
        )
        m_uniform_fast = (
            (a.shape[0] // group_num)
            if (group_num > 0 and a.shape[0] % group_num == 0)
            else None
        )
        if (
            var_k_fn is not None
            and m_uniform_fast is not None
            and m_uniform_fast >= 256
            and (n % hk.block_size != 0)
        ):
            # CRR variable-K dB pass. The dispatcher's ``a`` is the
            # input activation x_fp8 [M_total, k]; ``b`` is grad_out_fp8
            # [M_total, n] (see autograd:: ``grad_b =
            # grouped_gemm_fp8_variable_k_impl(a_fp8, grad_out_fp8,
            # ...)``). The HK kernel signature is
            # ``crr(grad_out_fp8, x_fp8, grad_b_bf16, scale_a,
            # scale_b, group_offs)`` so we MUST pass ``b`` (=
            # grad_out) as the kernel's A and ``a`` (= x) as the
            # kernel's B — same order the per-group fallback below
            # uses (``dense_run(bg, ag, ...)``, where bg = b[s:e] and
            # ag = a[s:e]). Mirrors the BF16 var-K wiring in
            # ``GroupedGEMMVariableKHipKittenBackend.execute``.
            #
            # Scales: ``_resolve_fp8_scales(b_scales, a_scales, ...)``
            # above already maps to (kernel's scale_a = grad_out's
            # scale, kernel's scale_b = x's scale), so we forward
            # ``(sa_*, sb_*)`` straight through.
            grad_out_2d = b if b.is_contiguous() else b.contiguous()
            x_2d = a if a.is_contiguous() else a.contiguous()
            if (
                sa_d is not None and sb_d is not None
                and var_k_dscale_fn is not None
            ):
                var_k_dscale_fn(grad_out_2d, x_2d, out,
                                sa_d, sb_d, group_offs)
            else:
                var_k_fn(grad_out_2d, x_2d, out, sa_h, sb_h, group_offs)
            return out

        offs = _group_offsets_cpu(group_offs)
        # Hoist padded_shape + select_default_config out of the per-group
        # loop on uniform-K (the CRR layout has K as the per-group axis;
        # mirrors the forward fast-path optimization in the sibling
        # GroupedGEMMFP8HipKittenBackend).
        m_uniform = (a.shape[0] // group_num) if (group_num > 0 and a.shape[0] % group_num == 0) else None
        cfg_pre = None
        m_pad_pre = n_pad_pre = k_pad_pre = 0
        # Reuse a single ``out_pad`` buffer across the per-group loop on
        # uniform-K — same idiom as the forward FP8 grouped path; saves
        # ``group_num`` alloc/free pairs per dispatch.
        out_pad_pre = None
        if m_uniform is not None:
            m_pad_pre, n_pad_pre, k_pad_pre = hipkitten.padded_shape(
                n, k, m_uniform, "crr", "fp8"
            )
            # FP8 variable-K (dB) per-group fallback. Pass m_total=
            # a.shape[0] for parity with the forward FP8 / BF16 grouped
            # fast paths.
            cfg_pre = hipkitten.select_default_config(
                m_pad_pre, n_pad_pre, k_pad_pre, "crr", "fp8",
                m_total=a.shape[0],
            )
            if (m_pad_pre, n_pad_pre, k_pad_pre) != (n, k, m_uniform):
                out_pad_pre = torch.empty(
                    (m_pad_pre, n_pad_pre), dtype=out_dtype, device=a.device
                )
        for group_idx in range(group_num):
            start, end = offs[group_idx], offs[group_idx + 1]
            m = end - start
            if cfg_pre is not None and m == m_uniform:
                m_pad, n_pad, k_pad = m_pad_pre, n_pad_pre, k_pad_pre
                cfg = cfg_pre
            else:
                m_pad, n_pad, k_pad = hipkitten.padded_shape(n, k, m, "crr", "fp8")
                cfg = hipkitten.select_default_config(
                    m_pad, n_pad, k_pad, "crr", "fp8", m_total=a.shape[0]
                )
            # Both ``a[start:end]`` and ``b[start:end]`` are row-slices of
            # contiguous 2D tensors so they're already contiguous in the
            # metric. Skip the no-op ``.contiguous()`` calls.
            ag = a[start:end]
            bg = b[start:end]
            if not ag.is_contiguous():
                ag = ag.contiguous()
            if not bg.is_contiguous():
                bg = bg.contiguous()
            if (m_pad, n_pad, k_pad) == (n, k, m):
                hipkitten.dense_run(
                    hk,
                    cfg,
                    bg,
                    ag,
                    out[group_idx],
                    scale_a=sa_h,
                    scale_b=sb_h,
                    scale_a_dev=sa_d,
                    scale_b_dev=sb_d,
                )
            else:
                # ``out_pad`` is fully written by the FP8 dense kernel
                # (kernel writes [c.shape] in full); skip the zero-init.
                if out_pad_pre is not None and m == m_uniform:
                    out_pad = out_pad_pre
                else:
                    out_pad = torch.empty(
                        (m_pad, n_pad), dtype=out_dtype, device=a.device
                    )
                hipkitten.dense_run(
                    hk,
                    cfg,
                    _pad_2d(bg, k_pad, m_pad),
                    _pad_2d(ag, k_pad, n_pad),
                    out_pad,
                    scale_a=sa_h,
                    scale_b=sb_h,
                    scale_a_dev=sa_d,
                    scale_b_dev=sb_d,
                )
                out[group_idx].copy_(out_pad[:n, :k])
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
