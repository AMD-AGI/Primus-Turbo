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
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import (
    BaseGroupedGEMMKernelDispatcher,
    BaseGroupedGEMMVariableKKernelDispatcher,
)
from primus_turbo.triton.utils.fp8_transpose import fp8_transpose_3d
from primus_turbo.triton.grouped_gemm.grouped_gemm_fp8_kernel import (
    grouped_gemm_fp8_blockwise_triton_kernel,
    grouped_gemm_fp8_blockwise_variable_k_triton_kernel,
    grouped_gemm_fp8_rowwise_triton_kernel,
    grouped_gemm_fp8_rowwise_variable_k_triton_kernel,
    grouped_gemm_fp8_tensorwise_triton_kernel,
    grouped_gemm_fp8_tensorwise_variable_k_triton_kernel,
)



# Per-shape autotune cache for the HK FP8 grouped kernels. Each entry
# is the full tuple of scheduling knobs the candidate set covers — pure
# scheduling levers (bit-equivalent across values), so the sweep is
# correctness-safe. Cache is process-local (module-level dict). Key is
# (op, m_total, n, k); k is included because tile geometry along K
# affects the picker even at fixed (m_total, n).
_HK_FP8_AUTOTUNE: dict[tuple, tuple[int, ...]] = {}

# RCR candidates (group_m, num_xcds). Six points cover the regimes seen
# in production (small/large m_total × tight/loose XCD swizzle) without
# blowing up first-call cost (~6 × 7 launches ≈ 50 ms). The advanced
# binding-internal knobs (num_slots, chunk_size, fuse_ktail_off) were
# experimentally added to the sweep and shown to give no gain over the
# binding's own auto-pick — the optimal values are baked into the
# binding default selector now.
_HK_FP8_RCR_CANDIDATES: tuple[tuple[int, int], ...] = (
    (1, 0), (1, 4), (2, 4), (4, 0), (4, 4), (8, 0), (8, 4),
    # Round-2026-05-14: expanded sweep for big-M-total Down shapes
    # (gpt_oss B32 N=2880 K=2880) where 6-cfg sweep landed on a
    # suboptimal point. Mirror RRR candidate distribution.
    (4, 8), (4, 16), (4, 32),
    (8, 16), (8, 32),
    (16, 0), (16, 4),
    (24, 0),
    # Round-2026-05-14b (round-2): probe (scripts/_probe_rcr_fwd.py,
    # ITERS=30) found (gm=1, xcds=4) was the optimum for
    # Down_B4_M4096_fwd (1928 T vs prior pick 1884 T) — same shape
    # category that benefited from (1, 4) on the RRR side last round.
    # Round-2026-05-14c (round-5): probe (scripts/_probe_extra_cfgs.py)
    # found (gm=12, xcds=4) ties the optimum on Down_B32_M2048_fwd
    # (1808 T vs (16,4) 1807 T) and stays within 1-2% on the other
    # gpt_oss B32 fwd shapes — a safe additional cfg for the autotuner
    # to pick when (16, 4) loses to noise on the (gm=16) end of the sweep.
    (12, 4),
    # Round-2026-05-14d: scripts/_probe_gm_xcds_sweep.py 7×5 grid found
    # gm=2 with high xcds wins on B=4 fwd shapes (GateUP_B4_M2048 fwd:
    # gm=2 xcds=32 = 1889 T vs prior pick 1847 T = +2.3%). Add gm=2
    # variants that weren't in the cfg list before.
    (2, 8), (2, 16), (2, 32),
    # Round 2026-05-15: heavy probe (_probe_down_b4_2048.py 13×10 grid,
    # ITERS=50) found (gm=12, xcds=2) is 1.2% above the previous best
    # (1507 T vs 1489 T) for Down_B4_M2048 fwd (the worst-performing
    # gpt_oss fwd shape). Adding to the candidate set; autotune will pick
    # it where it actually wins.
    (12, 2),
)
_HK_FP8_VARK_CANDIDATES: tuple[tuple[int, int], ...] = (
    (1, 0), (4, 0), (4, 4), (8, 0), (8, 4),
    # 2026-05-14: expanded to mirror RRR/RCR sweep. wgrad CRR previously
    # had only 5 cfgs vs RRR's 15; the missing (1, 4), (2, 4), (16, *)
    # combos are exactly where RRR/RCR found their gpt_oss B=4 wins.
    (1, 4), (2, 4),
    (4, 8), (4, 16), (4, 32),
    (8, 16), (8, 32),
    (16, 0), (16, 4),
    (24, 0),
    # Round-2026-05-14d: probe found gm=2 xcds=16/32 wins on Down_B4_M2048
    # wgrad (1319 T vs prior 1292 T = +2.1%). Add gm=2 high-xcds combos.
    (2, 8), (2, 16), (2, 32),
    # Round 2026-05-15: heavy probe found (gm=2, xcds=0) is 3.1% above
    # the previous best (1313 T vs 1273 T) for Down_B4_M2048 wgrad — the
    # worst-performing gpt_oss wgrad shape. (gm=2, xcds=0) was missing
    # from the candidate list entirely.
    (2, 0),
)
# RRR (dgrad direct, no transpose reroute) candidates: (group_m, num_xcds).
# Wider sweep than RCR because dgrad has more shape diversity (M_total
# 8K-65K × N 2880-7168 × K 2048-7168) and RRR's [G,K,N] B-layout makes
# scheduling more sensitive to swizzle than RCR. Includes large group_m
# (16, 24) for big-M-total shapes and large num_xcds for wider chiplet
# distribution. ~15 entries × ~7 launches = ~100ms first-call cost.
#
# Round-2026-05-14: probe (scripts/_probe_rrr_down_b32.py, ITERS=30)
# found (gm=1, xcds=4) was the optimum for GateUP_B32_M2048_dgrad
# (2185 T vs prior pick 2155 T) and Down_B4_M4096_dgrad (1668 T vs
# prior 1626 T) — neither was reachable with the previous candidate
# set (had (1,0), (2,4), (4,4) but not (1,4)).
_HK_FP8_RRR_CANDIDATES: tuple[tuple[int, int], ...] = (
    (1, 0), (1, 4), (2, 4), (4, 0), (4, 4), (8, 0), (8, 4),
    (4, 8), (4, 16), (4, 32),
    (8, 16), (8, 32),
    (16, 0), (16, 4),
    (24, 0),
    # Round-2026-05-14c (round-5): probe (scripts/_probe_extra_cfgs.py)
    # found (gm=12, xcds=4) is the top-1 cfg in the EXTRAS sweep across
    # all 4 measured B32 dgrad shapes (e.g. Down_B32_M2048_dgrad
    # 1597 T vs current best (16,4) 1594 T — a tie in noise) and
    # never falls more than ~3% below the global probe optimum.
    # Adding it as a defensive candidate so the autotuner has a backup
    # near-optimum when (16,4)/(4,4) lose to timing noise on a given run.
    (12, 4),
)

_AUTOTUNE_WARMUP_ITERS = 5
_AUTOTUNE_TIMED_ITERS = 50  # 2026-05-19 confirmed 100-iter gives same picks as
# 50-iter (geomean within 0.3% noise); the brute-force probe's 298us outlier
# was not reproducible in autotune context — reverted to 50/5 to save first-call
# cost. 2026-05-15: bumped 15→50 + warmup 3→5 because
# heavy probe (_probe_down_b4_2048.py at ITERS=50) found cfgs ~1-3% better
# than 15-iter autotune picks (e.g. fwd RCR Down_B4_M2048: probe pick
# (gm=12,xcds=2)=1507T vs autotune pick (gm=4,xcds=4)=1489T = +1.2%).
# At 15-iter, cfg ranking is dominated by timing noise around the top-K
# candidates. 50-iter narrows to <0.5% variance. First-call cost rises
# ~3.3x but per-shape autotune is a one-time bench.


def _autotune_pick(
    op,
    fixed_args: tuple,
    candidates: tuple[tuple[int, ...], ...],
    cfg_slot_indices: tuple[int, ...],
    key: tuple,
) -> tuple[int, ...]:
    """Time each candidate and cache the fastest.

    ``op``       — the torch.ops callable.
    ``fixed_args`` — full positional args list; the slots named by
                     ``cfg_slot_indices`` get substituted each iter.
    ``cfg_slot_indices`` — positions for the candidate-tuple components.
                          Length must match every candidate's arity.

    Uses cuda-event mean timing only — triton.testing.do_bench was tried
    but destabilizes the pytest sweep (memory access fault under accumulated
    state); per user direction 2026-05-20, do_bench is disallowed here.
    """
    cached = _HK_FP8_AUTOTUNE.get(key)
    if cached is not None:
        return cached
    _do_bench = None
    args = list(fixed_args)
    best_ms = float("inf")
    best_cfg = candidates[0]
    for cfg in candidates:
        for slot, val in zip(cfg_slot_indices, cfg):
            args[slot] = val
        try:
            local_args = tuple(args)
            fn = lambda local_args=local_args: op(*local_args)
            if _do_bench is not None:
                ms = _do_bench(fn, warmup=_AUTOTUNE_WARMUP_ITERS,
                               rep=_AUTOTUNE_TIMED_ITERS, return_mode="median")
            else:
                for _ in range(_AUTOTUNE_WARMUP_ITERS):
                    op(*args)
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(_AUTOTUNE_TIMED_ITERS):
                    op(*args)
                end.record()
                torch.cuda.synchronize()
                ms = start.elapsed_time(end) / _AUTOTUNE_TIMED_ITERS
        except Exception:
            continue
        if ms < best_ms:
            best_ms = ms
            best_cfg = cfg
    _HK_FP8_AUTOTUNE[key] = best_cfg
    return best_cfg


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
        del trans_a, granularity, num_cu, kwargs
        bs = b.shape[0]
        m_total = a.shape[0]
        avg_m = max(m_total // bs, 1) if bs > 0 else max(m_total, 1)
        a_in = a if a.is_contiguous() else a.contiguous()
        # Dgrad path (trans_b=False): b is [G, N_inner, K_out]. Two ways to
        # compute grad_a = grad_out @ b:
        #   * H4 reroute: fp8_transpose_3d(b) → RCR. Costs the transpose
        #     (~80%+ of wall-time on N_inner-large shapes).
        #   * RRR direct: hk_grouped_rrr_fp8 reads b in its native layout.
        #     N_MASKED_STORE handles K_out misalign for free; but K_inner
        #     misalign (K_inner % 128 != 0) needs a scalar K-tail RMW that
        #     becomes pathological for the N-tail strip — fall back to H4.
        # FUSED_KTAIL in the HK kernel handles K_inner misalignment (K_REM=64)
        # so RRR direct works for all dgrad shapes. The BF16 path uses a
        # K_inner threshold (4096) to fall back to H4 reroute on short-K
        # shapes where B's K-strided HBM reads miss L1/L2; for FP8 the
        # smaller per-byte footprint reduces cache pressure enough that RRR
        # direct wins on all measured shapes (gpt_oss + dsv3, B=4..16,
        # M=2048..4096) — keep the predicate trivial.
        k_inner_aligned = (not trans_b)
        if (not trans_b) and not k_inner_aligned:
            b_in = b if b.is_contiguous() else b.contiguous()
            b = fp8_transpose_3d(b_in)
            trans_b = True  # b is now in RCR layout
        b_in = b if b.is_contiguous() else b.contiguous()
        if (not trans_b):
            # RRR direct path: b stays as [G, N_inner, K_out].
            n_inner, k_out = b_in.shape[1], b_in.shape[2]
            op = torch.ops.primus_turbo_cpp_extension.hk_grouped_rrr_fp8
            fixed = (a_in, b_in, a_scales, b_scales, group_offs,
                     0, avg_m, 0, out_dtype)
            gm, xcds = _autotune_pick(
                op, fixed, _HK_FP8_RRR_CANDIDATES, (5, 7),
                key=("rrr_fp8", m_total, n_inner, k_out),
            )
            return op(a_in, b_in, a_scales, b_scales, group_offs,
                      gm, avg_m, xcds, out_dtype)
        # RCR path (fwd or H4-rerouted dgrad). b is [G, N, K] col-major view.
        n_out, k = b_in.shape[1], b_in.shape[2]
        op = torch.ops.primus_turbo_cpp_extension.hk_grouped_rcr_fp8
        # Round-2026-05-14: 3-way autotune adds bn_block ∈ {0, 128} to the
        # (gm, xcds) sweep. bn_block=0 → default 256x256 kernel; bn_block=128
        # → 256x128 (bn128) variant. Both kernels now carry the ceil_div
        # bpr_g + per-group shifted gl view + masked store fix (2026-05-18),
        # so unbalanced / M_g < BLOCK_SIZE shapes are correct in either.
        bn_choices = (0, 128)
        candidates_3way = tuple(
            (gm_, xcds_, bn_)
            for (gm_, xcds_) in _HK_FP8_RCR_CANDIDATES
            for bn_ in bn_choices
        )
        # 10 fixed args: a, b, a_s, b_s, g_offs, gm, m_per, xcds, dtype, bn
        fixed = (a_in, b_in, a_scales, b_scales, group_offs,
                 0, avg_m, 0, out_dtype, 0)
        gm, xcds, bn = _autotune_pick(
            op, fixed, candidates_3way, (5, 7, 9),
            key=("rcr_fp8", m_total, n_out, k),
        )
        return op(a_in, b_in, a_scales, b_scales, group_offs,
                  gm, avg_m, xcds, out_dtype, bn)


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
        # wgrad requires trans_a=True, !trans_b. trans_c can be either; the
        # two layouts are transposes of each other and we handle both by
        # swapping the op-side (a / b) roles in execute() below — mirrors
        # GroupedGEMMVariableKHipKittenBackend in grouped_gemm_impl.py.
        if a.dim() != 2 or b.dim() != 2 or not (trans_a and not trans_b):
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
        del trans_a, trans_b, granularity, num_cu, kwargs

        # Standard CRR var-K path (a, b 2D)
        if a.dim() == 3:
            G, m_g, k = a.shape
            a = a.view(G * m_g, k)
            # b might be (G, M_g, N) [non-transposed] or (G, N, M_g) [transposed]
            if b.dim() == 3 and b.shape[1] == m_g:
                # Non-transposed (G, M_g, N)
                b = b.view(G * m_g, b.shape[2])
            elif b.dim() == 3 and b.shape[2] == m_g:
                # Transposed (G, N, M_g) — un-transpose for CRR
                b = fp8_transpose_3d(b.contiguous())  # (G, M_g, N)
                b = b.view(G * m_g, b.shape[2])
        x_in = a if a.is_contiguous() else a.contiguous()
        grad_out_in = b if b.is_contiguous() else b.contiguous()
        # Kernel computes c[g] = op_a[g]^T @ op_b[g] with output shape
        # [G, op_a.shape[1], op_b.shape[1]]. Two output layouts via swapping
        # the op-side (x, grad_out) roles; mirrors the bf16 backend:
        #   trans_c=True  → c [G, N_fwd, K_fwd] = grad_out^T @ x
        #     op_a=grad_out (PT ``b``), op_b=x (PT ``a``)
        #     scales: a_scales is x's scale, b_scales is grad_out's scale,
        #     so op_a_scales=b_scales, op_b_scales=a_scales.
        #   trans_c=False → c [G, K_fwd, N_fwd] = x^T @ grad_out
        #     op_a=x (PT ``a``), op_b=grad_out (PT ``b``)
        #     op_a_scales=a_scales, op_b_scales=b_scales.
        if trans_c:
            op_a, op_b = grad_out_in, x_in
            op_a_scales, op_b_scales = b_scales, a_scales
        else:
            op_a, op_b = x_in, grad_out_in
            op_a_scales, op_b_scales = a_scales, b_scales
        m_total = op_a.shape[0]
        n_out = op_a.shape[1]
        k = op_b.shape[1]

        op = torch.ops.primus_turbo_cpp_extension.hk_grouped_var_k_crr_fp8
        # Positional layout: a, b, a_scales, b_scales, group_offs,
        #                    group_m(5), num_xcds(6), out_dtype.
        fixed = (op_a, op_b, op_a_scales, op_b_scales, group_offs, 0, 0, out_dtype)
        gm, xcds = _autotune_pick(
            op, fixed, _HK_FP8_VARK_CANDIDATES, (5, 6),
            key=("vark_fp8", m_total, n_out, k),
        )
        return op(op_a, op_b, op_a_scales, op_b_scales, group_offs,
                  gm, xcds, out_dtype)


class GroupedGEMMFP8TritonBackend(KernelBackend):
    """Triton persistent-kernel backend for FP8 grouped GEMM (CPU-sync-free).

    Supports:
      - TENSORWISE: per-tensor scaling, including HYBRID format
      - ROWWISE: per-row/per-col vector scaling
      - BLOCKWISE: block-wise scaling (2D B_scales per group)
    """

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
      - TENSORWISE: per-tensor scaling, including HYBRID format
      - ROWWISE: per-row/per-col vector scaling
      - BLOCKWISE: 1D+1D block-wise scaling (TN/CRR layout)
    """

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


_FP8_GRANULARITY_INT_TO_ENUM = {g.value: g for g in ScalingGranularity}
_FP8_HIPKITTEN_BACKEND_INT = BackendType.HIPKITTEN.value
_FP8_HIPKITTEN_BACKEND_ENUM = BackendType.HIPKITTEN


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
    user_backend_enum = GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.FP8)
    if (
        default_backend == _FP8_HIPKITTEN_BACKEND_INT
        and (user_backend_enum is None or user_backend_enum is _FP8_HIPKITTEN_BACKEND_ENUM)
        and not GlobalBackendManager.auto_tune_enabled()
    ):
        return GroupedGEMMFP8HipKittenBackend.execute(
            a=a,
            b=b,
            a_scales=a_scales,
            b_scales=b_scales,
            group_lens=group_lens,
            group_offs=group_offs,
            trans_a=trans_a,
            trans_b=trans_b,
            out_dtype=out_dtype,
            granularity=_FP8_GRANULARITY_INT_TO_ENUM[granularity],
            num_cu=num_cu,
            maybe_pre_sync=maybe_pre_sync,
        )

    # Slow path: full dispatcher (autotune, user-override-different-from-default,
    # fallback chains). Reuse user_backend_enum from the fast-path probe above.
    default_backend_enum = BackendType(default_backend)
    granularity_enum = _FP8_GRANULARITY_INT_TO_ENUM[granularity]

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
    user_backend_enum = GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.FP8)
    if (
        default_backend == _FP8_HIPKITTEN_BACKEND_INT
        and (user_backend_enum is None or user_backend_enum is _FP8_HIPKITTEN_BACKEND_ENUM)
        and not GlobalBackendManager.auto_tune_enabled()
    ):
        return GroupedGEMMFP8VariableKHipKittenBackend.execute(
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
            granularity=_FP8_GRANULARITY_INT_TO_ENUM[granularity],
            num_cu=num_cu,
            maybe_pre_sync=maybe_pre_sync,
        )

    default_backend_enum = BackendType(default_backend)
    granularity_enum = _FP8_GRANULARITY_INT_TO_ENUM[granularity]

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
    """Compute device-resident cumulative offsets ``[0, l0, l0+l1, ...]``."""
    return torch.ops.primus_turbo_cpp_extension.grouped_gemm_compute_offs(group_lens)


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


grouped_gemm_fp8_impl = grouped_gemm_fp8_impl._init_fn
grouped_gemm_fp8_variable_k_impl = grouped_gemm_fp8_variable_k_impl._init_fn
