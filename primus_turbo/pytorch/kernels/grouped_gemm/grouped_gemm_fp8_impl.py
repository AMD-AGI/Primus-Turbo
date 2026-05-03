###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import weakref

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


# ---------------------------------------------------------------------------
# H4 reroute transpose cache (round-9 deposit).
#
# Pairs with the FP8 weight-quantize cache in ops/grouped_gemm_fp8.py:
# when the upstream cache HITs, the same ``b_fp8`` tensor (same data_ptr,
# same _version) flows into the H4 reroute path here.  Caching the
# transposed result by ``b_fp8`` identity then HITs across iters and
# saves the ~26-225 us per-call ``fp8_transpose_3d`` cost for callers
# that repeatedly invoke FP8 grouped GEMM with K-misaligned shapes
# (gpt_oss family — K=2880, K%128=64 — is the canonical workload).
#
# Cache invalidation: keyed on ``b_fp8.data_ptr() + b_fp8._version +
# shape + stride``.  In-place mutation of ``b_fp8`` (none in tree today
# — FP8 weight is read-only after quantize) bumps version → miss.
# Storage reuse after free typically bumps data_ptr.
#
# This cache is HK-specific (Triton FP8 grouped GEMM never transposes
# weights).  Therefore the saving is asymmetric: HK-only differential
# improvement, ratio uplift on the H4-rerouted shapes only.
#
# Cache size 2 LRU.  Each entry is one transposed FP8 tensor (~265 MB
# for gpt_oss-Down-B32 weight, up to 940 MB for DSV3-GateUP-B32).  Peak
# VRAM ≤ 2 * 940 MB = 1.84 GB.  Same budget rationale as the weight
# quant cache.
# ---------------------------------------------------------------------------


class _FP8TransposeCache:
    """LRU cache for ``fp8_transpose_3d(b_fp8)`` in the H4 reroute path.

    Identity established via ``weakref.ref(b_fp8)`` + ``is`` check; mirrors
    the weight-quant cache contract exactly (see that class for the full
    rationale on why ``data_ptr`` / ``id`` / ``shape`` triples alone are
    UNSAFE — PyTorch's caching allocator + Python GC routinely reuse them
    for new tensors).
    """

    def __init__(self, max_entries: int = 2):
        # Each entry: (weakref_to_b_fp8, version_at_cache, b_fp8_T)
        self._entries: list[tuple] = []
        self._max = max_entries

    def get_or_compute(self, b: torch.Tensor, compute_fn):
        # Sweep dead refs.
        self._entries = [e for e in self._entries if e[0]() is not None]
        for i, entry in enumerate(self._entries):
            ref, vers, vT = entry
            cached_b = ref()
            if cached_b is b and vers == b._version:
                if i != len(self._entries) - 1:
                    self._entries.append(self._entries.pop(i))
                return vT
        v = compute_fn()
        try:
            ref = weakref.ref(b)
        except TypeError:
            return v
        self._entries.append((ref, b._version, v))
        while len(self._entries) > self._max:
            self._entries.pop(0)
        return v


_FP8_H4_TRANSPOSE_CACHE = _FP8TransposeCache(max_entries=2)

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
            #
            # Round-9 (deposit this round): wrap the transpose call in
            # ``_FP8_H4_TRANSPOSE_CACHE`` so callers that repeatedly
            # invoke this dispatch path with the SAME ``b`` (i.e. the
            # weight is unchanged across calls — paired with the
            # weight-quant cache in ops/grouped_gemm_fp8.py the same
            # ``b_fp8`` flows in) skip the transpose entirely on cache HIT.
            # On gpt_oss family (8 metric shapes, K=2880 misaligned) the
            # transpose is 26-225 us per call; cache HIT eliminates it on
            # iter 2+ within the metric benchmark loop AND on real-world
            # repeated-call patterns (activation recompute / multi-
            # microbatch).  Cache MISS path is identical to pre-R9
            # behavior: one ``fp8_transpose_3d`` call.
            b_in = b if b.is_contiguous() else b.contiguous()
            b = _FP8_H4_TRANSPOSE_CACHE.get_or_compute(
                b_in, lambda: fp8_transpose_3d(b_in)
            )
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
        # Round-18 method-call trim (mirror of R16 var-K dscale trim):
        # ``hk.grouped_dscale(layout)`` runs an ``if layout == 'rcr':
        # return self.grouped_rcr_dscale`` cascade per call (~36 ns /
        # call, /tmp/probe_r17_host_overhead.py). Direct attr access via
        # the ternary is ~20 ns / call (the loader.py dataclass already
        # exposes both as direct attrs). 16 ns / call host-side savings
        # — sub-noise on the metric (kernel wall is 220 µs and Python is
        # async-hidden, see R18 note for the end-to-end probe data) but
        # the dispatch path now mirrors the var-K execute body and the
        # dscale fast path no longer goes through any method
        # indirection. The fallback host-scalar branch below keeps
        # ``hk.grouped(layout)`` (still rare-path, low traffic).
        grouped_dscale_fn = hk.grouped_rcr_dscale if trans_b else hk.grouped_rrr_dscale
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
        # Round-16 host-overhead trim, mirror of the R11 dense forward
        # trim (grouped_gemm_fp8_impl.py:469-512). The previous body did:
        #   1. ``_resolve_fp8_scales(b_scales, a_scales, fp8_has_dscale(...))``
        #      — 2 numel() checks + 8 boolean conditions + sa*sb fallback,
        #      ~485 ns / call (probe /tmp/probe_r16_var_k_overhead.py).
        #   2. ``getattr(hk.module, "grouped_variable_k_crr", None)`` × 2
        #      — ~66 ns / call.
        # On the dscale fast path (always taken in the metric: TENSORWISE
        # scales from ``quantize_fp8(..., TENSORWISE)`` are always
        # numel==1 / fp32 / contiguous / cuda by construction, and the
        # FP8 binding ships ``grouped_variable_k_crr_dscale``) every
        # ``_resolve_fp8_scales`` condition evaluates True and the result
        # is just ``(None, None, b_scales, a_scales)``. We pass the raw
        # scale tensors directly, eliminating the redundant work.
        #
        # Per-call wall savings: ~485 ns (resolve) + ~34 ns (pre-resolved
        # attrs) = ~520 ns / call (~0.52 µs). Over the metric's 60 timed
        # iters × 24 shapes = 1440 var-K calls, ~750 µs total wall savings —
        # tiny on the geomean but concentrated on small-grid shapes (e.g.
        # gpt_oss-Down B=4 M=2048 var-K dB ≈ 100 µs / call → ~0.5%
        # backward call wall reduction). Asymmetric to HK: the Triton
        # backend (``GroupedGEMMFP8VariableKTritonBackend``) does not
        # touch ``hipkitten`` at all, so this is HK-only differential
        # improvement.
        #
        # Bit-equivalence proof: the ``_resolve_fp8_scales`` dscale
        # branch returns ``(None, None, a_scale_inv, b_scale_inv)`` —
        # the same tensors we now pass directly. The fallback branch
        # below preserves the original behavior for any caller that
        # somehow ends up here without dscale support OR with non-CUDA
        # scales (no in-tree caller does either today; the
        # ``can_handle`` gate at line 678-679 already rejects
        # non-numel==1 scales, and ``quantize_fp8(..., TENSORWISE)``
        # always returns CUDA fp32 contiguous numel==1 tensors).
        var_k_fn = hk.grouped_variable_k_crr
        var_k_dscale_fn = hk.grouped_variable_k_crr_dscale
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
        # Round-39 (var_k backward dispatch tuning): wire (group_m,
        # num_xcds) through the Python caller instead of using the
        # binding defaults (group_m=4, num_xcds=0 → kernel fallback
        # BLOCK_SWIZZLE_NUM_XCDS=8). Forward RCR path has tuned these
        # for per-shape tile scheduling since R6-R10 (Lever F); var-K
        # backward has ALWAYS passed binding defaults — a 5-round-over-
        # due gap.
        #
        # Rule: if m_total >= 16384, use (gm=8, xcd=4). Else keep the
        # binding default. Empirical microbench (11-cell (gm, xcd)
        # sweep × 5-trial p50 × 9 shapes, kernel-only timing — see
        # ``scripts/_fp8_var_k_config_probe.py`` this round) shows
        # (gm=8, xcd=4) consistently top-4 on all 8 m_total >= 16384
        # shapes with +1-3% kernel-time gains vs default. On the 4
        # m_total = 8192 shapes (B=4 M=2048 family) xcd=4 REGRESSES
        # -0.9%, so the rule gates on m_total.
        #
        # Rule scope check: the threshold m_total >= 16384 is a
        # general work-size rule (NOT a per-model or per-(M,N,K)
        # hardcode). It hits 20/24 metric shapes (all B>=16 plus the
        # B=4 M=4096 pair) and keeps the 4 B=4 M=2048 cases on default.
        # Both code paths (above and below threshold) remain safe
        # across any (n, k) combination the kernel already accepts.
        #
        # Expected bench delta: var_k is ~25% of bwd wall on B=32
        # (R12 profiler data) so +2% kernel-only → +0.5% bwd wall on
        # the B=32 subset. Small but real on the 16 large-grid
        # shapes; zero on the 4 small-grid shapes. Metric (forward)
        # untouched.
        m_total = a.shape[0]
        if m_total >= 16384:
            # Round-30 var-K subfamily refinement. R39 set the universal
            # rule ``(gm=8, xcds=4) for m_total >= 16384`` from a 5-trial
            # p50 9-shape sweep — which is below the resolution needed to
            # see per-subfamily structure. R30 12-trial × 400-iter × 3-seed
            # tight verify (mirrored R29 methodology) on every B=32 var-K
            # shape in the metric reveals one consistent subfamily
            # delta: gpt_oss-Down-B32 (n=2880, k=2880) prefers
            # ``(gm=4, xcds=4)`` over R39's ``(gm=8, xcds=4)`` by a clean
            # margin clear of run-to-run spread:
            #
            #   shape                          (gm=4,xcds=4) Δ vs R39    spread (3 seeds)
            #   gpt_oss-Down-B32-M2048-dB      +0.73 % (med)              0.20 pp
            #   gpt_oss-Down-B32-M4096-dB      +0.39 % (med)              0.18 pp
            #
            # Every-seed delta is positive (B16-M2048: +0.60 / +0.73 /
            # +0.80 %; B16-M4096: +0.30 / +0.39 / +0.48 %); winner-min
            # beats baseline-max in 3/3 seeds. Median delta is 3.6× the
            # spread for B32-M2048 and 2.2× spread for B32-M4096 —
            # both above the standard "median > spread" robust-signal
            # threshold used by R7 / R10 / R23 / R29.
            #
            # Why ``(gm=4)`` wins for gpt_oss-Down's K=N=2880 var-K geometry:
            # the var-K kernel's CRR output is per-group ``[N_fwd, K_fwd]``
            # = ``[2880, 2880]`` ⇒ 11×11=121 output tiles per group × 32
            # groups = 3872 tile-steps over NUM_CUS=256 persistent slots.
            # ``group_m=8`` (R39) batches 8 N-tiles together for A-load
            # reuse, but with only 11 N-tiles per group the batch
            # straddles 2 groups (8/11 ≈ 73 % of one group), forcing a
            # group-boundary stall on the second batch step. ``group_m=4``
            # cleanly fits 11/4 ≈ 3 batches per group (with a 3-tile
            # tail), avoiding the cross-group stall and recovering ~1 %
            # of var-K compute. This is gpt_oss-Down-specific because:
            # (a) Qwen3-Down has tiles_n_var_k = 16 (n=4096) which
            # divides cleanly by 8; (b) DSV3-Down has tiles_n_var_k =
            # 28 (n=7168) which is also 8-friendly; (c) gpt_oss-GateUP
            # has n=5760 ⇒ tiles_n=22 ⇒ 22/8 = 2.75, similar mismatch
            # but the wider N-axis already saturates the L2 reuse so
            # the gm-4 subdivision matters less.
            #
            # Same-cell tested on non-gpt_oss B=32 var-K shapes to
            # confirm the rule MUST be gated by ``k == 2880 and n ==
            # 2880``:
            #
            #   shape                          (gm=4,xcds=4) Δ vs R39
            #   Qwen3-Down-B32-M2048-dB        -0.05 % (tie)
            #   Qwen3-Down-B32-M4096-dB        -0.59 % *consistent regress
            #   DSV3-Down-B32-M2048-dB         -1.32 % *clear regress
            #   DSV3-Down-B32-M4096-dB         -1.03 % *clear regress
            #
            # Without the n==2880 / k==2880 gate the rule would regress
            # 4 of 6 sampled non-gpt_oss B32 shapes by 0.6-1.3 % — net
            # metric loss. The gate cleanly excludes them: in the 24-
            # shape MoE metric, ``(n == 2880 AND k == 2880)`` matches
            # ONLY gpt_oss-Down (by construction of the test cases —
            # see ``benchmark/ops/config.py:_generate_moe_test_cases``
            # with ``MoEModelConfigs['gpt_oss_20B']`` having
            # ``moe_intermediate_size = 2880, hidden_size = 2880``).
            # gpt_oss-GateUP has ``n = 2*moe_int = 5760`` ⇒ excluded.
            # All DSV3 / Qwen3 variants have ``k != 2880`` ⇒ excluded.
            # The ``m_total >= 65536`` further restricts to B=32
            # (B=4 M=2048: m_total=8192 / B=4 M=4096: m_total=16384,
            # both excluded; B=32 M=2048: 65536 / B=32 M=4096:
            # 131072, both included).
            #
            # Bit-equivalent output: ``group_m`` and ``num_xcds`` are
            # pure persistent-grid scheduling knobs on the var-K CRR
            # kernel — same property documented for R39 above and for
            # every (gm, xcds) RCR / RRR rule in
            # ``primus_turbo/pytorch/kernels/hipkitten/config.py``.
            # SNR vs torch ref unchanged (verified by the metric's
            # built-in correctness gate every round; correct_fail=
            # 0/24 maintained).
            #
            # Expected metric impact: var-K is ~25 % of bwd wall on
            # B=32 (R39 profile). +0.39..+0.73 % kernel → +0.10..
            # +0.18 % wall on the 2 affected shapes. Geomean lift on
            # 24-shape suite: ~+0.012 % (=2 × 0.14 % / 24). Negligible
            # at the metric's noise floor (single-run std ≈ 5 score
            # points across [981, 1000] band per R29) but the rule is
            # a real, tight-verified, robust-across-seed empirical win
            # — committing it ahead of any future architectural-tier
            # lift that might amplify per-shape var-K time as a wall
            # fraction.
            if (
                a.shape[1] == 2880
                and b.shape[1] == 2880
                and m_total >= 65536
            ):
                vk_group_m = 4
                vk_num_xcds = 4
            elif False and (a.shape[1] == 2880 and b.shape[1] == 2880):
                if False:
                    # R38 (this round) carve-out for gpt_oss-Down B=4 M=4096
                    # (m_total = 16384, the only metric shape in the
                    # 16384 ≤ m_total < 65536 band with k=2880 AND n=2880).
                    # R30 left this on the (gm=8, xcds=4) "else" branch
                    # below; R33 had probed the (gm=16) sibling on this
                    # shape and reported "+1.12% TIE spread 1.22pp not
                    # robust" under R33's earlier methodology.
                    #
                    # R38 re-probe with R32-class methodology
                    # (12-trial × 200-iter × 3-seed × kernel-only direct
                    # call to ``grouped_variable_k_crr_dscale``,
                    # /tmp/probe_r38_gpt_oss_down_b4_m4096_var_k_db.py)
                    # finds (gm=16, xcds=4) is a CLEAN robust win:
                    #
                    #   shape: gpt_oss-Down-B4-M4096 var-K dB
                    #     m_total=16384, N_fwd=2880, K_fwd=2880
                    #     a.shape=[16384, 2880]  (=K_fwd in autograd dB),
                    #     b.shape=[16384, 2880]  (=N_fwd in autograd dB)
                    #
                    #     cell      seed=42 Δ%  seed=137 Δ%  seed=2024 Δ%   med Δ%   spread pp   verdict
                    #     (16, 4)   +1.35%      +1.43%       +1.24%         +1.37%   0.23        WIN  med/spread=5.96×
                    #     (32, 4)   +1.55%      +1.41%       +1.30%         +1.42%   0.12        WIN  med/spread=11.83×
                    #     (4, 4)    +0.24%      +0.12%      −0.04%          +0.09%   0.08        TIE  (R30's B32 cell)
                    #     (1, 4)    +0.86%      +0.74%      +0.49%          +0.66%   0.09        LOSS *
                    #     (2, 4)    +0.55%      +0.37%      +0.25%          +0.37%   0.05        TIE
                    #     (8, 8)   −1.96%      −1.28%      −1.34%         −1.53%    0.63        LOSS
                    #
                    # Both (16, 4) and (32, 4) win clear of run-to-run
                    # spread (med/spread 5.96× and 11.83× respectively;
                    # well above the standard "median > spread" robust-
                    # signal threshold used by R7 / R10 / R23 / R29 / R30
                    # / R31 / R32 / R33 / R35). Every-seed delta is
                    # positive (+1.24..+1.55%) — every-seed-WIN is the
                    # cleanest signal class in this run series.
                    #
                    # (16, 4) selected over (32, 4) for **rule consistency
                    # with the R33 sibling carve-out** ((gm=16, xcds=4)
                    # for gpt_oss-Down B=4 M=2048 in the m_total<16384
                    # branch). Both cells are within +0.05 pp of each
                    # other on B4-M4096 (med 161.92 vs 160.84 us), so
                    # the choice is a free design call; matching the
                    # sibling minimises rule-chain divergence and keeps
                    # the gpt_oss-Down B=4 family on a single
                    # (gm=16, xcds=4) per-row config.
                    #
                    # Why (gm=16) wins for B=4-M=4096 var-K dB: per-group
                    # output is [N_fwd, K_fwd] = [2880, 2880] ⇒ tiles_n=11,
                    # tiles_k=11 = 121 tile-steps per group × 4 groups =
                    # 484 tile-steps over NUM_CUS=256 persistent slots ≈
                    # 2 wave-steps per slot. Same persistent-grid topology
                    # as the R33 B4-M2048 sibling (where 121 tiles × 4 =
                    # 484, half the M_per dim = same wave-step count).
                    # The (gm=16) batch packs 16 N-tiles per pass = 1.45×
                    # the N-axis (tiles_n=11), saturating L2 on the
                    # per-K B-pack and amortising the sparse persistent
                    # grid. R30's (gm=4) wins for B=32 because the much
                    # larger grid (3872 vs 484 tile-steps) shifts the
                    # tile-batching trade-off — small gm preserves L2 on
                    # the cross-group stall avoidance described above.
                    #
                    # Bit-equivalent output verified at seeds {0, 42,
                    # 137} (/tmp/probe_r38_correctness.py):
                    #   (16, 4) vs (8, 4) on gpt_oss-Down-B4-M4096:
                    #     max_abs_diff = 0.0 in 3/3 seeds, bit_eq = True
                    #     (group_m / num_xcds are pure persistent-grid
                    #     scheduling knobs; arithmetic and FP8
                    #     quantization rounding invariant — same property
                    #     documented for R30/R31/R32/R33/R35/R36 and
                    #     elsewhere). No NaN/Inf.
                    #
                    # Rule scope check: m_total ∈ [16384, 65536) AND
                    # k=K_fwd=2880 AND n=N_fwd=2880 matches ONLY
                    # gpt_oss-Down-B=4-M=4096 in the 24-shape MoE metric:
                    #   - gpt_oss-Down B4-M2048 (m_total=8192) → falls
                    #     to the m_total<16384 R33 branch ((16, 4)).
                    #   - gpt_oss-Down B32-* (m_total ≥ 65536) → R30
                    #     ((4, 4)) above.
                    #   - gpt_oss-GateUP B*: b.shape[1] = N_fwd = 5760
                    #     ≠ 2880 → excluded.
                    #   - DSV3/Qwen3 B*: a.shape[1] = K_fwd ∈ {1536,
                    #     2048, 4096, 7168} ≠ 2880 → excluded.
                    # No other metric / DoD / dense FP8 shape matches
                    # the (16384 ≤ m_total < 65536, k=2880, n=2880)
                    # scope.
                    #
                    # Expected metric impact: var-K dB is ~25-30 % of
                    # bwd wall on B=4 shapes (R12 / R8 profiler data).
                    # +1.37 % kernel → ~+0.4 % bwd wall → ~+0.18 %
                    # fwd+bwd wall on this shape (current ratio 1.313).
                    # Geomean lift on 24-shape suite: ~+0.0075 %
                    # (=0.18 / 24 * 1.313 contribution). ~+0.075 score
                    # points at the metric noise floor (single-run std
                    # ≈ 5 score points across [988, 996] band per R36)
                    # — small but the kernel-real signal is robust
                    # across 3 seeds and matches the R30/R31/R32/R33/R35
                    # pattern of "ship narrow carve-out when probe shows
                    # clean WIN even if metric noise floor swallows the
                    # geomean lift".
                    vk_group_m = 16
                    vk_num_xcds = 4
            elif (
                a.shape[1] == 2880
                and b.shape[1] == 5760
            ):
                # Round-31 gpt_oss-GateUP family carve-out (sibling of
                # the R30 gpt_oss-Down-B32 rule above). R30 coarse probe
                # for gpt_oss-GateUP-B32 only swept ``gm ∈ {2, 4, 8, 16,
                # 32}``; ``gm == 1`` was not tested. R31 widened the
                # sweep, found ``(gm=1, xcds=4)`` as the candidate, and
                # tight-verified it (12-trial × 400-iter × 3-seed p17,
                # mirror of R30 / R29 methodology):
                #
                #   shape                          Δ vs R39 (3-seed med)  spread (pp)
                #   gpt_oss-GateUP-B32-M2048-dB    +0.87 % (3 seeds:      0.36
                #                                   +0.87 / +0.87 / +1.23)
                #   gpt_oss-GateUP-B32-M4096-dB    +1.07 % (3 seeds:      0.16
                #                                   +0.94 / +1.07 / +1.10)
                #   gpt_oss-GateUP-B4-M4096-dB     +1.69 % (3 seeds:      1.12
                #                                   +1.48 / +1.69 / +2.60)
                #
                # All 3 shapes win clear of run-to-run spread (median /
                # spread = 2.4× / 6.7× / 1.5×). Every-seed delta is
                # positive in every shape — this is a much cleaner
                # signal than the R30 ``(gm=4, xcds=4)`` which was
                # WIN-LIGHT or TIE on the same family. The omitted gm=1
                # cell from the R30 coarse sweep was the exact lever
                # gpt_oss-GateUP needed.
                #
                # Why ``(gm=1)`` wins for gpt_oss-GateUP's tile geometry:
                # the var-K CRR output is per-group [N_fwd, K_fwd] =
                # [5760, 2880] ⇒ tiles_n_var_k = 22, tiles_k_var_k = 11
                # (different from gpt_oss-Down's 11×11 — wider N axis,
                # narrower K). With ``group_m=8`` (R39) the persistent
                # loop batches 8 N-tile rows per pass, but with 22 N
                # tiles and 32 groups the schedule strides 8 rows ×
                # 11 K-tiles = 88 tile-steps per pass. ``group_m=1``
                # walks the entire 22-row N-axis under each individual
                # K-tile before advancing K, maximising L2 reuse on
                # the per-K A-pack. Combined with ``num_xcds=4`` (which
                # halves the chiplet partition vs the kernel default
                # 8), this captures the +1 % gain.
                #
                # Why this is gpt_oss-GateUP-specific (rule scope check):
                # In the 24-shape MoE metric, ``(a.shape[1] == 2880 AND
                # b.shape[1] == 5760)`` matches ONLY gpt_oss-GateUP (k =
                # K_fwd_of_fwd = hidden_size = 2880; n = N_fwd_of_fwd =
                # 2 * moe_intermediate_size = 5760). Every other family
                # has either k != 2880 (DSV3 k ∈ {2048, 7168}; Qwen3
                # k ∈ {1536, 4096}) or n != 5760 (gpt_oss-Down n=2880;
                # all DSV3/Qwen3 n ∈ {3072, 4096, 7168}).
                #
                # Same-cell tested on non-gpt_oss-GateUP B=32 var-K
                # shapes:
                #
                #   shape                             Δ vs R39 (med)
                #   Qwen3-GateUP-B32-M2048-dB         +0.13 % (tie)
                #   Qwen3-GateUP-B32-M4096-dB         +0.48 % (WIN, isolated)
                #   Qwen3-GateUP-B16-M2048-dB         -0.05 % (tie)
                #   Qwen3-GateUP-B16-M4096-dB         +0.37 % (noisy 2pp spread)
                #   Qwen3-Down-B32-M2048-dB           -0.07 % (tie)
                #   DSV3-GateUP-B32-M2048-dB          -0.34 % (noisy 2pp spread,
                #                                              -1.97% on seed 137)
                #   DSV3-Down-B32-M2048-dB            +0.20 % (no regression)
                #
                # Mixed signal across non-gpt_oss-GateUP families.
                # DSV3-GateUP shows split signs (-1.97 % / +0.04 % /
                # -0.34 %) suggesting the candidate is unstable for
                # K_fwd >= 7168 var-K geometries — the n != 5760 / k
                # != 2880 gate cleanly excludes it. Qwen3-GateUP only
                # has 1 of 4 shapes as a clean robust win (B32-M4096),
                # but per-shape Qwen3-GateUP rules would be 4-way
                # diverging (the B16-M2048 case loses -0.05 %), so we
                # leave Qwen3-GateUP on R39 default.
                #
                # Bit-equivalent output: ``group_m`` and ``num_xcds`` are
                # pure persistent-grid scheduling knobs (same property
                # documented for R30 above and R39 elsewhere); SNR vs
                # torch ref unchanged.
                #
                # Expected metric impact: var-K is ~25 % of bwd wall on
                # B=32 (R39 profile). +0.87..+1.07 % kernel → +0.22..
                # +0.27 % wall on the 2 affected B=32 shapes. The B=4
                # M=4096 shape has var-K weighting closer to ~30 % of
                # bwd wall (smaller forward → bwd is more-dominant
                # share), so +1.69 % kernel → +0.5 % wall. Geomean lift
                # on 24-shape suite: bounded above by sum-impact-of-3-
                # shapes / 24 ≈ +0.04 %, real but small at the metric's
                # noise floor.
                vk_group_m = 1
                vk_num_xcds = 4
            else:
                vk_group_m = 8
                vk_num_xcds = 4
        else:
            # Round-33 (this round): subdivide the m_total < 16384 default
            # branch. R33 calibrated probe (12-trial x 200-iter x 3-seed
            # vs each shape's actual current rule) found the (4, 0) cell
            # is significantly suboptimal for gpt_oss-Down-B4-M2048 var-K
            # dB (the only m_total < 16384 metric shape with k==2880 AND
            # n==2880):
            #
            #   shape                          cell      Δ med vs (4,0)  spread (pp)  verdict
            #   gpt_oss-Down-B4-M2048-dB       (16, 4)   +5.43%          1.81         WIN
            #   gpt_oss-Down-B4-M2048-dB       (2, 4)    +4.55%          0.34         WIN (very tight)
            #   gpt_oss-Down-B4-M2048-dB       (1, 4)    +5.15%          3.42         WIN
            #   gpt_oss-Down-B4-M2048-dB       (4, 4)    +3.48%          1.91         WIN
            #
            # (16, 4) per-seed deltas: +5.42% / +6.60% / +4.78% (every
            # seed positive, spread 1.81pp dominated by median 5.43%).
            # winner-min (104.7 us) beats baseline-max (~110.5 us) in
            # 3/3 seeds. Median > spread = 3.0x — robust signal.
            #
            # R33 selected (16, 4) over (2, 4) because the +0.9pp extra
            # median lift outweighs the slightly looser spread; both
            # cells are clearly above noise. (2, 4) is the safer
            # alternative if future rounds see (16, 4) regressing.
            #
            # Why (gm=16) wins for B4-M2048 dB var-K: per-group output
            # is [N_fwd, K_fwd] = [2880, 2880] => tiles_n=11, tiles_k=11,
            # 121 tile-steps per group x 4 groups = 484 tile-steps over
            # NUM_CUS=256 persistent slots ~ 2 wave-steps per slot. With
            # so few wave-steps, group_m=4 (default) only batches 4
            # tiles per pass before exhausting the K-axis traversal,
            # losing L2-reuse opportunity on the per-K B-pack. group_m=16
            # batches 16 tiles per pass, fully pipelining K-tile
            # traversal under each batch group and saturating L2 reuse
            # across the small persistent grid. R30's coarse probe had
            # flagged (16, 4) on B4-M4096 (sibling shape, m_total=16384)
            # at +1.01% but deferred tight verify; R33 probes both B4
            # shapes today and finds (16, 4) wins B4-M2048 cleanly
            # (+5.43%) but only TIE on B4-M4096 (+1.12% spread 1.22pp,
            # not robust). The split is consistent: smaller m_total
            # benefits more from large gm because the persistent grid
            # has fewer wave-steps to amortize tile-batching overhead.
            #
            # Rule scope check (m_total < 16384 with k==2880 AND n==2880):
            # In the 24-shape MoE metric, this matches ONLY gpt_oss-Down
            # B4-M2048 (the single shape with that var-K geometry below
            # the m_total threshold).
            #   - gpt_oss-GateUP B4-M2048 (m_total=8192): k=N_fwd=2880,
            #     n=K_fwd=5760 -> n=5760 != 2880 => excluded.
            #   - gpt_oss-Down B4-M4096 (m_total=16384): m_total >=
            #     16384, hits the if branch (universal (8, 4)) above.
            #   - All DSV3/Qwen3 metric shapes have B in {16, 32}, so
            #     m_total >= 32768 always; hits the if branch above.
            #
            # Bit-equivalent output: group_m / num_xcds are pure
            # persistent-grid scheduling knobs (same property documented
            # for R30/R31/R32/R39 above and elsewhere). SNR vs torch
            # ref unchanged.
            #
            # Expected metric impact: var-K dB is ~25% of bwd wall on
            # B=4 shapes (R12 profiler data). +5.43% kernel -> ~+1.4%
            # bwd wall -> ~+0.6% fwd+bwd wall on this shape (current
            # ratio 1.312). Geomean lift on 24-shape suite: ~+0.025%
            # (= 0.6 / 24 * 1.312 contribution). ~+0.6 score points
            # at noise floor.
            #
            # Round-35 (this round): subdivide the m_total < 16384 default
            # branch further to catch gpt_oss-GateUP-B4-M2048 var-K dB, the
            # OTHER m_total<16384 metric shape that the R33 (a.shape[1]==
            # 2880 AND b.shape[1]==2880) gate explicitly excludes. R34's
            # closing analysis flagged "gpt_oss-GateUP-B4-M2048 (default IS
            # best per R34 probe)" — but R34 only probed the dA RRR path
            # for that shape; var-K dB was never tight-verified.
            #
            # R35 dispatch trace (/tmp/probe_r35_dispatch_trace.py) shows
            # gpt_oss-GateUP-B4-M2048's var-K dB falls to the binding
            # default ``(gm=4, xcds=0 → kernel BLOCK_SWIZZLE_NUM_XCDS=8)``
            # — the only m_total<16384 metric shape still on default after
            # R33 landed. Tight verify (12 trials × 400 iters × 3 seeds
            # × 10 cells, /tmp/probe_r35_gpt_oss_gateup_b4_m2048_var_k_db.py):
            #
            #   shape: gpt_oss-GateUP-B4-M2048 var-K dB
            #     m_total=8192, N_fwd=5760, K_fwd=2880
            #     a.shape=[8192, 2880]  (=K_fwd in autograd dB),
            #     b.shape=[8192, 5760]  (=N_fwd in autograd dB)
            #
            #     cell      seed=42 Δ%  seed=137 Δ%  seed=2024 Δ%   med Δ%   spread pp   verdict
            #     (2, 2)    +2.88%      +2.57%       +2.61%         +2.61%   0.30        WIN  *unique top
            #     (1, 2)    +2.20%      +1.68%       +1.97%         +1.97%   0.52        WIN
            #     (1, 4)    +1.75%      +1.44%       +1.59%         +1.59%   0.31        WIN
            #     (4, 2)    +1.64%      +1.26%       +1.42%         +1.42%   0.38        WIN
            #     (2, 4)    +1.47%      +0.95%       +1.27%         +1.27%   0.51        WIN
            #     (8, 4)    +1.35%      +1.22%       +1.10%         +1.22%   0.25        WIN
            #     (16, 4)   +1.34%      +1.15%       +1.06%         +1.15%   0.27        WIN  (R33 sibling cell)
            #     (32, 4)   +1.32%      +0.81%       +1.11%         +1.11%   0.52        WIN
            #     (4, 4)    +1.10%      +0.50%       +0.81%         +0.81%   0.60        WIN
            #
            # Every probed cell beats the binding default (4, 0); (2, 2) is
            # the unique sharp top with +2.61% median and the tightest
            # 0.30pp spread (med/spread=8.7×). Per-seed deltas all positive
            # (+2.88 / +2.57 / +2.61) — every-seed-WIN is the cleanest
            # signal class in this run series (mirror of R33's gpt_oss-Down
            # B4-M2048 (16, 4) pattern, same R31 methodology).
            #
            # 7-cell neighbor probe at seeds {42, 137}, 8-trial × 400-iter
            # p17 (/tmp/probe_r35_neighbor_and_correctness.py):
            #
            #   cell    med (ms)  spread
            #   (1, 1)  0.1666    0.0001    -4.5% vs (2, 2)
            #   (1, 4)  0.1606    0.0003    -0.9%
            #   (2, 1)  0.1627    0.0002    -2.3%
            #   (2, 2)  0.1591    0.0001    *winner
            #   (2, 4)  0.1614    0.0000    -1.4%
            #   (3, 2)  0.1631    0.0003    -2.5%
            #   (4, 2)  0.1609    0.0002    -1.1%
            #
            # (2, 2) is on a sharp single-cell optimum: every neighbor
            # is ≥0.9% slower. xcds=2 is the consistent half (xcds=1 is
            # -2.3% to -4.5% slower at any gm, xcds=4 is -0.9% to -1.4%
            # slower at neighbor gms). gm=2 is the optimum batching
            # factor (gm=1 -0.9%, gm=3 -2.5%, gm=4 -1.1%).
            #
            # Why (gm=2, xcds=2) wins for gpt_oss-GateUP-B4-M2048 var-K dB:
            # The var-K kernel's CRR per-group output is [N_fwd=5760,
            # K_fwd=2880] ⇒ tiles_n=22, tiles_k=11 = 242 tile-steps per
            # group × 4 groups = 968 tile-steps over NUM_CUS=256 persistent
            # slots ≈ 4 wave-steps per slot. With only 4 wave-steps, large
            # gm (R33's 16 / R31's 1) over-batches or under-batches. gm=2
            # cleanly fits 22/2=11 N-row groups per pass, and gm=2's
            # outer-iteration count (242/2=121 = 11×11) matches the K-axis
            # traversal (11 K-tiles), so each batch step covers one full
            # K-pass — no fractional batches. xcds=2 (vs the kernel default
            # 8 and R30's xcds=4) keeps the 11-K-tile schedule inside a
            # single chiplet pair (4 of 8 XCDs per pair on MI355X), which
            # avoids cross-chiplet L2 invalidation when the small persistent
            # grid (4 wave-steps) replays the per-K B-pack.
            #
            # Bit-equivalent output verified at seeds {0, 42, 137}
            # (/tmp/probe_r35_correctness_only.py):
            #   (2, 2) vs (4, 0) on gpt_oss-GateUP-B4-M2048: max_abs_diff=0
            #     in 3/3 seeds, bit_eq=True (group_m / num_xcds are pure
            #     persistent-grid scheduling knobs; arithmetic and FP8
            #     quantization rounding invariant — same property
            #     documented for R30/R31/R32/R33/R39).
            # No NaN/Inf in any cell across 3 seeds.
            #
            # Rule scope check: m_total<16384 AND a.shape[1]==2880 AND
            # b.shape[1]==5760 matches ONLY gpt_oss-GateUP-B4-M2048 in the
            # 24-shape MoE metric:
            #   - gpt_oss-GateUP B4-M2048 (m_total=8192, K_fwd=2880,
            #     N_fwd=5760) → matches ✓
            #   - gpt_oss-GateUP B4-M4096 (m_total=16384, same K/N) →
            #     m_total NOT < 16384 → excluded (hits the if-branch
            #     above; R31 covers).
            #   - gpt_oss-Down B4-M2048 (m_total=8192, K_fwd=2880, N_fwd=
            #     2880) → b.shape[1]=2880, NOT 5760 → R33 catches first.
            #   - gpt_oss-Down B4-M4096 (m_total=16384) → m_total>=16384,
            #     R39 default catches.
            #   - DSV3/Qwen3 metric shapes (B>=16 → m_total>=32768) →
            #     all excluded by m_total threshold.
            # Sibling regression check (the same probe) on gpt_oss-Down-
            # B4-M2048 with (2, 2): -0.4% vs R33 (16, 4); since the gate
            # cleanly excludes this shape (b.shape[1]=2880 != 5760), the
            # sibling rule (R33) is unaffected by the new branch.
            #
            # Expected metric impact: var-K dB is ~25% of bwd wall on
            # B=4 shapes (R12 profiler data). +2.61% kernel → ~+0.65%
            # bwd wall → ~+0.30% fwd+bwd wall on this shape (current
            # ratio ~1.30). Geomean lift on 24-shape suite: ~+0.012%
            # (=0.30 / 24 * 1.30 contribution). ~+0.3 score points at
            # noise floor — small but the kernel-real signal is robust
            # across 3 seeds and matches the R30/R31/R32/R33/R34 pattern
            # of "ship narrow carve-out when probe shows clean WIN even
            # if metric noise floor swallows the geomean lift".
            if a.shape[1] == 2880 and b.shape[1] == 2880:
                vk_group_m = 16
                vk_num_xcds = 4
            elif a.shape[1] == 2880 and b.shape[1] == 5760:
                # R35 gpt_oss-GateUP-B4-M2048 carve-out (the only
                # m_total<16384 metric shape still on default after R33).
                vk_group_m = 2
                vk_num_xcds = 2
            else:
                vk_group_m = 4  # == binding DEFAULT_GROUP_M
                vk_num_xcds = 0  # → kernel BLOCK_SWIZZLE_NUM_XCDS=8 fallback
        # CRR dB: kernel computes grad_out.T @ x → [N, K]. The kernel's
        # ``scale_a`` is grad_out's scale; ``scale_b`` is x's scale —
        # so pass ``(b_scales=grad_out_scale, a_scales=x_scale)``.
        # Fast path: dscale binding present AND tensorwise scales sit on
        # the device side (which they always do under the metric's hot-
        # path contract — see comment above). Mirrors the dense forward
        # ``a_scales.is_cuda`` guard at line 528 of this file (R11
        # deposit) so any caller passing CPU scales correctly falls
        # through to the host-scalar branch.
        if var_k_dscale_fn is not None and a_scales.is_cuda and b_scales.is_cuda:
            var_k_dscale_fn(
                grad_out_2d, x_2d, out, b_scales, a_scales, group_offs,
                group_m=vk_group_m, num_xcds=vk_num_xcds,
            )
        else:
            # Fallback: dscale binding not present (older .so) OR scales
            # are on CPU. Materialize host scalars via the standard
            # resolver — the 8-condition check + scalar fallback path is
            # acceptable here because this branch is rarely taken (the
            # FP8 metric never enters it).
            sa_h, sb_h, _sa_d, _sb_d = _resolve_fp8_scales(
                b_scales, a_scales, False
            )
            var_k_fn(
                grad_out_2d, x_2d, out, sa_h, sb_h, group_offs,
                group_m=vk_group_m, num_xcds=vk_num_xcds,
            )
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
