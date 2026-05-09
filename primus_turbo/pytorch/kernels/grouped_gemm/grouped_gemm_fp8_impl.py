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


# ---------------------------------------------------------------------------
# Round-17 (gpt_oss FP8 kernel-only ceiling, current Primus run; 2026-05-09).
#
# Caller-allocated K-split workspace cache (Stream-K variant-2 / R11 plan).
#
# R14 falsified the per-call hipMallocAsync path inside HK's
# dispatch_grouped_rcr at 2.9-9.1 ms / call (vs the R11 cost-decomp
# assumption of ~3 µs). Per the R14 forward-pointer, the workspace must
# be allocated once per cell on the caller side and passed in via the
# new HK pybind kwarg ``sk_workspace_ptr`` (R17 HK commit). This cache
# amortizes the alloc cost to ~0 after the first dispatch per cell.
#
# Cache key: ``(buf_bytes, device_index)`` — the buffer is dtype-agnostic
# (uint8 byte slab; HK reads it as fp32 partials via reinterpret). Same
# key on subsequent dispatches HITs and reuses the existing slab.
#
# Storage: ``torch.empty(buf_bytes, dtype=torch.uint8, device=cuda:i)``.
# The PyTorch caching allocator owns the underlying VRAM; we just hold
# a strong ref so the slab survives across iters. Eviction: bounded LRU
# at ``max_entries=8`` (covers the 8 gpt_oss shapes in the metric, two
# devices, with headroom). For the gpt_oss B=4 cells T_max yields
# 44-88 MiB per slab; max steady-state VRAM ≤ 8 * 88 MiB = 704 MiB.
#
# Production NEUTRAL gate: this cache is touched ONLY when the dispatcher
# returns ``cfg.sk_split_n > 0``. R17 ships no rule that sets that field;
# the field stays at its dataclass default 0 on every shape until R18+
# lands the kernel-side K-split branch and a per-cell rule turns it on.
# So R17's metric impact is the empty path: WorkspaceCache exists but
# is never consulted, no extra VRAM allocated, no kwarg passed to HK.
# ---------------------------------------------------------------------------


class _FP8WorkspaceCache:
    """Once-per-cell device-buffer cache for HK K-split partial accumulators.

    ``get_or_alloc(buf_bytes, device)`` returns a 1-D ``uint8`` cuda tensor
    of length >= ``buf_bytes`` whose ``data_ptr()`` is passed to HK via
    the ``sk_workspace_ptr`` pybind kwarg (R17). Cache key:
    ``(buf_bytes, device.index)``. Bounded LRU (``max_entries`` slabs);
    eviction drops the strong ref so PyTorch's caching allocator can
    reclaim the slab.
    """

    def __init__(self, max_entries: int = 8):
        self._slabs: dict[tuple[int, int], torch.Tensor] = {}
        self._lru: list[tuple[int, int]] = []
        self._max = max_entries

    def get_or_alloc(self, buf_bytes: int, device: torch.device) -> torch.Tensor:
        key = (int(buf_bytes), int(device.index) if device.index is not None else 0)
        slab = self._slabs.get(key)
        if slab is not None:
            # Bump LRU.
            try:
                self._lru.remove(key)
            except ValueError:
                pass
            self._lru.append(key)
            return slab
        slab = torch.empty(buf_bytes, dtype=torch.uint8, device=device)
        # R17 NEUTRAL infra: the kernel does not yet read sk_partial_buf
        # (R18 lands the kernel branch). Until then the buffer's contents
        # never affect math; we still zero it once at allocation to mirror
        # the HK R13a hipMemsetAsync(0) contract for when R18+ does start
        # reading partial accumulators.
        slab.zero_()
        self._slabs[key] = slab
        self._lru.append(key)
        while len(self._lru) > self._max:
            evicted = self._lru.pop(0)
            self._slabs.pop(evicted, None)
        return slab


_FP8_SK_WORKSPACE_CACHE = _FP8WorkspaceCache(max_entries=8)


def _fp8_sk_workspace_bytes(m_total: int, n: int, sk_block: int = 256) -> int:
    """Mirror of HK's per-call buf size formula at
    ``HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp:7884-7886``:
    ``T_max = ceil_div(M_total, BLOCK_SIZE) * bpc`` then
    ``buf_bytes = T_max * BLOCK_SIZE * BLOCK_SIZE * sizeof(float)``.

    BLOCK_SIZE is fixed at 256 for the FP8 grouped RCR kernel (matches
    the R12-R14 scaffold + R11 cost-decomp); ``bpc = ceil_div(N, BLOCK_SIZE)``.
    Returning a strict upper bound is OK — over-allocation by at most
    (num_groups - 1) M-tiles per cell (~0.1-1 % over-alloc per the R13a
    note). Used by the R17 WorkspaceCache to size the slab.
    """
    block = sk_block
    t_m = (m_total + block - 1) // block
    t_n = (n + block - 1) // block
    t_max = t_m * t_n
    return t_max * block * block * 4  # sizeof(float) = 4

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
        #
        # Round-3 (FP8 fused-act task, deposit this round): extend H4
        # reroute to ALL ``trans_b=False`` (RRR) callers — not just
        # K_RRR / N_RRR misaligned. R14's falsification of unconditional
        # reroute on K-aligned shapes (-22..-36% bwd) was recorded
        # BEFORE the R9 transpose cache (``_FP8_H4_TRANSPOSE_CACHE``)
        # was deposited above. With R9 cache the transpose cost is
        # paid ONCE per (b_fp8, version) tuple and then ~0 µs via
        # weakref-keyed identity LRU. Iter 2+ in the metric loop sees
        # zero transpose cost; weight tensors are constant within an
        # optimizer step in production, so cache HITs there too.
        #
        # Tight kernel-only probe on this round's GPU build (200 iters
        # × 12 trials, p20 from /tmp/probe_round_3_*):
        #
        #   shape                       RRR direct   RCR-via-T   Δ
        #   Qwen3-Down-B16-M2048         178.8 us    143.1 us   +25.0%
        #   Qwen3-Down-B32-M2048         351.0 us    277.4 us   +26.5%
        #   Qwen3-Down-B16-M4096         356.1 us    282.5 us   +26.1%
        #   Qwen3-Down-B32-M4096         706.2 us    548.9 us   +28.7%
        #   Qwen3-GateUP-B16-M2048       395.5 us    336.7 us   +17.5%
        #   Qwen3-GateUP-B32-M2048       785.7 us    665.1 us   +18.1%
        #   Qwen3-GateUP-B16-M4096       784.9 us    670.4 us   +17.1%
        #   Qwen3-GateUP-B32-M4096      1565.6 us   1331.7 us   +17.6%
        #   DSV3-Down-B16-M2048          345.7 us    317.7 us    +8.8%
        #   DSV3-Down-B32-M2048          687.3 us    628.0 us    +9.4%
        #   DSV3-GateUP-B16-M2048        794.2 us    614.8 us   +29.2%
        #   DSV3-GateUP-B32-M2048       1573.5 us   1211.3 us   +29.9%
        #
        # Every aligned RRR shape gains +9-30%. R8 documented this as
        # "HK RRR is the per-component weak spot" (analysis/_notes/
        # round-8-fused-act-architectural-ceiling-confirmed.md);
        # this round eliminates the weak spot for the dA backward path.
        #
        # Bit-equivalent output: transpose only swaps the last two
        # axes of b (E4M3 layout invariant); RCR(a, b_T) computes the
        # same per-group product as RRR(a, b) (both yield
        # ``dA = grad_out @ W_T``). The metric's built-in correctness
        # gate (SNR > 25 dB on out / dA / dB across all 24 shapes)
        # serves as the regression check; correct_fail = 0/24 verified
        # in the wall-metric run committed alongside this change.
        #
        # Affected callers in the 24-shape MoE metric:
        #   - dA backward path of every grouped FP8 fwd+bwd call
        #     (8 DSV3 + 8 Qwen3 = 16 shapes; 8 gpt_oss already reroute
        #      via the K_RRR % 128 != 0 clause).
        #   - Forward stays on RCR direct (the metric calls forward
        #     with trans_b=True; this gate only fires on trans_b=False).
        if not trans_b:
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
        # Round-9 (current Primus run, gpt_oss FP8 kernel-only ceiling
        # task; 2026-05-08): wire ``cfg.num_slots`` through to the FP8
        # grouped RCR / RRR binding's new per-call num_slots arg (HK
        # commit this round). Default 0 keeps the kernel on its NUM_CUS
        # = 256 default (or legacy ``TK_RCR_NUM_CUS`` env hook); per-
        # shape rules in ``hipkitten/config.py`` set non-zero values
        # for shapes where probe evidence shows the persistent grid
        # benefits from a smaller slot count (e.g. Down-B4-M2048 fwd+
        # dgrad at ~1.5 wave-steps/CU sees +5% kernel TFLOPS at slots=
        # 200 — see ``scripts/_probe_round_9_*.py`` and
        # ``analysis/_notes/round-9-fp8-grouped-rcr-num-slots-per-call-
        # surgery.md``).
        slots_arg = cfg.num_slots
        # Round-14 (current Primus run, gpt_oss FP8 kernel-only ceiling
        # task; 2026-05-08): wire ``cfg.chunk_size`` through to the FP8
        # grouped RCR / RRR binding's new per-call chunk_size arg (HK
        # commit this round mirrors R13's var-K chunk_size lever).
        # Default 0 → kernel uses historical baseline 64. Per-shape
        # rules set non-zero values for cells where the swizzle is
        # currently a NO-OP at cs=64 and a cleaner partition exists at
        # a smaller chunk_size. Down-B4-M2048 fwd+dgrad-via-H4 cell
        # (xcds=2, slots=196) finds chunk_size=96 → block=192=slots-4
        # → 192 of 196 workgroups in 1 clean partition (96 PIDs/XCD)
        # → +4.06% / +4.05% kernel TFLOPS (R14 probe; see
        # scripts/_probe_round_14_down_b4_extended.py and the rule
        # comment block in hipkitten/config.py at the matching cell).
        chunk_arg = cfg.chunk_size
        # Round-16 (current Primus run, gpt_oss FP8 kernel-only ceiling
        # task; 2026-05-08): wire ``cfg.fuse_ktail_off`` through to the
        # FP8 grouped RCR / RRR-via-H4 binding's new per-call
        # ``fuse_ktail_off`` arg (HK commit this round converts R14's
        # process-static ``TK_GROUPED_RCR_FUSE_OFF`` env hook into a
        # per-call kwarg). Default 0 → kernel uses R34-dm FUSED_KTAIL=
        # true codegen-driven default (load-bearing on the entire suite
        # per R14). Per-shape rules in ``hipkitten/config.py`` set
        # ``fuse_ktail_off=1`` only for the GateUP B=32 dgrad-via-H4
        # cells (tiles_n==11, k==5760, m_total>=65536) where R14's
        # per-shape probe found a +1-2% win. Only the FP8 grouped RCR
        # binding consumes this kwarg; the RRR (non-rerouted) and
        # var-K paths leave it un-passed (default 0 in the C++ wrapper).
        fuse_off_arg = cfg.fuse_ktail_off
        # Round-17 (gpt_oss FP8 kernel-only ceiling, current Primus run;
        # 2026-05-09): caller-allocated K-split workspace plumbing. R14
        # falsified the per-call hipMallocAsync inside HK at 2.9-9.1 ms
        # / call. R17 caches the workspace once per (M_total, N, device)
        # in ``_FP8_SK_WORKSPACE_CACHE`` and passes its data_ptr through
        # the new HK ``sk_workspace_ptr`` kwarg. Gated strictly on
        # ``cfg.sk_split_n > 0``: production rules all default to 0 →
        # branch never entered → no extra VRAM, no kwarg passed, the
        # HK call is byte-for-byte identical to pre-R17. Only the FP8
        # grouped RCR binding consumes the kwarg; RRR/var-K bindings
        # would TypeError on it, so the kwarg is conditional on
        # ``trans_b`` (RCR-only call site).
        sk_split_arg = cfg.sk_split_n
        sk_workspace_ptr_arg = 0
        if trans_b and sk_split_arg > 0:
            buf_bytes = _fp8_sk_workspace_bytes(int(m_total), int(n))
            slab = _FP8_SK_WORKSPACE_CACHE.get_or_alloc(buf_bytes, a.device)
            sk_workspace_ptr_arg = int(slab.data_ptr())
        if use_dscale:
            grouped_dscale_kwargs = dict(
                m_per_group=avg_m, num_xcds=xcds_arg, num_slots=slots_arg,
                chunk_size=chunk_arg,
            )
            # Only the RCR variant consumes ``fuse_ktail_off`` (the gate
            # lives in ``dispatch_grouped_rcr``; ``dispatch_grouped_rrr``
            # has no fuse_ktail_eligible path). Pass conditionally to
            # avoid TypeError on the RRR binding which doesn't expose
            # the kwarg.
            if trans_b and fuse_off_arg:
                grouped_dscale_kwargs["fuse_ktail_off"] = fuse_off_arg
            # R17: K-split kwargs are RCR-only and gated on sk_split_n>0.
            # Default branch (sk_split_n==0) skips both kwargs, which keeps
            # production calls bit-identical to pre-R17 (HK pybind defaults
            # cover both kwargs back at 0 / nullptr).
            if trans_b and sk_split_arg > 0:
                grouped_dscale_kwargs["sk_split_n"] = sk_split_arg
                grouped_dscale_kwargs["sk_workspace_ptr"] = sk_workspace_ptr_arg
            grouped_dscale_fn(
                a_in, b_in, out, a_scales, b_scales, group_offs, cfg.group_m,
                **grouped_dscale_kwargs,
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
                m_per_group=avg_m, num_xcds=xcds_arg, num_slots=slots_arg,
                chunk_size=chunk_arg,
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
        # Round-3 (gpt_oss FP8 kernel-only ceiling, current Primus run;
        # 2026-05-07): per-call ``num_slots`` override on the var-K
        # persistent-grid launch. Default ``0`` → kernel uses NUM_CUS=256
        # (or the TK_VARK_NUM_CUS R2 env hook, when set). Set to
        # 192 ONLY for the short-grid Down-B4 wgrad family below; every
        # other var-K caller stays on the existing 256-slot launch.
        vk_num_slots = 0
        # Round-13 (gpt_oss FP8 kernel-only ceiling, current Primus run;
        # 2026-05-08): per-call ``chunk_size`` override on the var-K
        # ``chiplet_transform_chunked`` swizzle (HK kernel side). Default
        # ``0`` → kernel uses the historical baseline 64. Set to 96 ONLY
        # for the same Down-B4 wgrad family that already uses
        # vk_num_slots=192 + vk_num_xcds=2 (paired R3 cell): at that
        # cell, chunk_size=96 makes block = num_xcds * chunk_size = 192
        # = exactly slots, so all 192 workgroups participate in one
        # clean chiplet-pair partition. The default chunk_size=64 leaves
        # the trailing 64 workgroups (192 - 128 chunked) as round-robin,
        # which mixes the chiplet partition (R12 falsification note
        # observation; the cliff-alignment hypothesis was not testable
        # without this lever). See R13 evidence block at the predicate
        # site below.
        vk_chunk_size = 0
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
                # Round-1 (current Primus run, kernel-only ceiling-push
                # task; 2026-05-07) split of the R30 universal
                # ``(gm=4, xcds=4)`` rule into the two B=32 m_total tiers.
                # R30 was tight-verified at 12-trial × 400-iter × 3-seed
                # against R39's universal (gm=8, xcds=4) and reported
                # +0.73 % / +0.39 % wins on the two B=32 shapes. Today's
                # FP8 binding (post the R3-fused-act commit ceb7e93,
                # rebuilt .so) shows the optimum has SPLIT between the
                # two m_total tiers — same kind of methodology / kernel-
                # rebuild drift R31 documented for var-K (and R45 / R32
                # for the RCR / RRR rules).
                #
                # Round-1 tight A/B verify (1500-iter × 7-trial p20 ×
                # 3 seeds × kernel-only direct call to
                # ``grouped_variable_k_crr_dscale``,
                # /tmp/_probe_round_1_tight_verify.py archived in commit
                # message):
                #
                #   shape: gpt_oss-Down-B32-M2048 var-K dB
                #     m_total=65536, N_fwd=2880, K_fwd=2880, B=32
                #
                #     cell      seed42 ms_med  seed137 ms_med  seed2024 ms_med   med Δ vs (4,4)  wmin_beats_lmax
                #     (4, 4)cur 0.6634          0.6652          0.6661           baseline        ---
                #     (8, 4)    0.6471          0.6480          0.6474           +2.51..+2.89pp  3/3 seeds
                #
                #   shape: gpt_oss-Down-B32-M4096 var-K dB
                #     m_total=131072, N_fwd=2880, K_fwd=2880, B=32
                #
                #     cell      seed42 ms_med  seed137 ms_med  seed2024 ms_med   med Δ vs (4,4)  wmin_beats_lmax
                #     (4, 4)cur 1.0907          1.0909          1.0907           baseline        ---
                #     (8, 4)    1.0913          1.0923          1.0915           -0.06..-0.13pp  0/3 seeds (tie-loss)
                #
                # M=2048: (8, 4) wins +2.51..+2.89 pp on every seed, and
                # ``hi_new (worst (8,4))`` < ``lo_old (best (4,4))`` on
                # every seed → wmin_beats_lmax robust signal class
                # (the cleanest tier in this run series, mirror of R33
                # / R35 / R10 / R11 patterns). Median > spread by 4-7×.
                # M=4096: (8, 4) is -0.06..-0.13 pp tie-loss; (4, 4) is
                # at the local peak. Splitting the rule by m_total
                # captures the M=2048 win without regressing M=4096.
                #
                # Why (gm=8) wins on M=2048 specifically: per-group
                # output is [N_fwd, K_fwd] = [2880, 2880] ⇒ tiles_n=11,
                # tiles_k=11 = 121 tile-steps × 32 groups = 3872
                # tile-steps over NUM_CUS=256 persistent slots ≈ 15
                # wave-steps. M_per_g=2048 ⇒ K-loop length per tile-step
                # is ``M_per_g / KBLOCK = 2048 / 128 = 16`` blocks. The
                # smaller wave-step count amortises (gm=8)'s 8-wide
                # N-batch better than (gm=4): each wave-step runs through
                # 8 N-rows × 16 K-blocks (= 128 macc rounds) before
                # advancing, which fits MI355X's per-XCD MFMA pipeline
                # depth more cleanly. M_per_g=4096 doubles the K-loop
                # depth per tile-step (32 K-blocks), shifting the L2 /
                # MFMA pipeline trade-off back toward (gm=4) — confirmed
                # by the wmin_beats_lmax tier swap above. Same R30 / R31
                # / R9 ``wave-step amortisation bifurcates the optimum''
                # pattern.
                #
                # Bit-equivalent output verified at
                # /tmp/_probe_round_1_correctness.py: max_abs((4,4) -
                # (8,4)) = 0 across 3 seeds × 2 shapes, SNR vs fp32 ref =
                # 28.46-28.50 dB on every (cell, seed, shape) — well
                # above the 25 dB FP8 noise floor (group_m / num_xcds are
                # pure persistent-grid scheduling knobs, same property
                # documented for R30 / R31 / R32 / R33 / R35 / R10 / R11
                # above and every (gm, xcds) RCR / RRR rule in
                # ``primus_turbo/pytorch/kernels/hipkitten/config.py``).
                #
                # Sibling regression check: the rule split keeps R10's
                # m_total ∈ [16384, 65536) carve-out (gpt_oss-Down B=4
                # M=4096) untouched — its elif gate still excludes B=32
                # via the m_total >= 65536 outer if-branch above. R31's
                # gpt_oss-GateUP rules (b.shape[1]==5760) are unaffected
                # by the n=k=2880 inner split. R25's DSV3 rule
                # (a.shape[1] in {2048, 7168}) is unaffected.
                #
                # Expected metric impact: +2.51..+2.89 pp kernel on
                # gpt_oss-Down-B32-M2048 wgrad (one of the 8 metric
                # shapes feeding the wgrad section average). The shape
                # currently sits ~1633 TF; +2.6 pp ≈ +42 TF lift to
                # ~1675 TF. Section average over 8 wgrad shapes lifts
                # by 42/8 ≈ 5 TF (from ~1408 to ~1413, progress
                # 0.503 → 0.505); score lift ~+1-2 points. Real signal
                # but small at the metric's noise floor (single-run
                # std ≈ 25 score points). The kernel-real every-seed
                # win + wmin_beats_lmax robust signal is committed
                # ahead of compounding gains from later round rules.
                if m_total == 65536:
                    vk_group_m = 8
                    vk_num_xcds = 4
                else:
                    # Round-46 (current run; 2026-05-09) — Down-B32-M4096
                    # var-K wgrad: xcds=8 + (slots=256, cs=32) clean-256
                    # partition. Fleet probe round_0/round_1 cross-shape
                    # verified +0.71% (round_1 7-seed wmin>lmax=True) at
                    # cell (gm=4, xcds=8, slots=256, cs=32) over the
                    # baseline (gm=4, xcds=4) selected here. Same xcds=8
                    # lever class as the GateUP-B32-M4096 round_0 +1.17%
                    # finding; the lever is M=4096-specific (M=2048 sibling
                    # tests on Down/GateUP both LOSS at same cell, so it
                    # is gated to `else` branch only).
                    vk_group_m = 4
                    vk_num_xcds = 8
                    vk_num_slots = 256
                    vk_chunk_size = 32
            elif a.shape[1] == 2880 and b.shape[1] == 2880:
                # Round-10 (current Primus run; 2026-05-05) carve-out for
                # gpt_oss-Down B=4 M=4096 (m_total=16384, the only metric
                # shape in the 16384 ≤ m_total < 65536 band with k=2880
                # AND n=2880). Same R8/R9 "candidate-set widening" pattern:
                # R38 had probed (gm=16, xcds=4) and reported +1.37%
                # kernel-direct but was disabled (`if False`) — likely
                # because R38's xcds=4-only candidate set missed the
                # true optimum at xcds=2.
                #
                # Round-10 widened sweep (200-iter × 7-trial × p20 ×
                # 3 seeds × 17 cells across xcds={2, 4, 8} columns at
                # /tmp/probe_round_10_gpt_oss_down_b4_m4096_var_k.py)
                # found (gm=1, xcds=2) as a clean every-seed-positive
                # winner:
                #
                #   shape: gpt_oss-Down-B4-M4096 var-K dB
                #     m_total=16384, N_fwd=2880, K_fwd=2880, B=4 groups
                #
                #     cell      seed42  seed137  seed2024  med Δ vs cur  spread pp  verdict
                #     (1, 2)    +1.33%  +1.11%   +1.28%    +1.24%        0.17       WIN  med/spread=7.4×
                #     (16, 2)   +1.09%  +0.99%   +1.48%    +1.18%        0.49       WIN
                #     (32, 2)   +0.94%  +1.18%   +1.31%    +1.14%        0.37       WIN
                #     (16, 4)R38+0.76%  +0.71%   +0.83%    +0.76%        0.12       WIN  (R38 candidate)
                #     (32, 4)   +0.78%  +0.51%   +0.64%    +0.64%        0.27       WIN
                #     (1, 4)    +0.68%  +0.56%   +0.58%    +0.60%        0.12       WIN
                #     (2, 2)    +0.82%  +0.63%   +0.17%    +0.54%        0.65       small WIN
                #     (8, 4)cur baseline                    +0.00%        0.10       (R39 default)
                #     (1, 8)    -0.92%  -0.87%   -0.75%    -0.85%        0.17       LOSS
                #     (8, 8)    -1.47%  -1.71%   -1.43%    -1.54%        0.28       LOSS
                #     (16, 8)   -1.78%  -1.67%   -1.79%    -1.72%        0.12       LOSS
                #
                # (gm=1, xcds=2) is the unique top with the tightest
                # spread (0.17pp) — every-seed positive (3/3) at +1.11%
                # to +1.33%, baseline-min beat in 3/3 seeds. Median /
                # spread = 7.4× (well above the standard "median > spread"
                # robust-signal threshold used by R7 / R10 / R23 / R29 /
                # R30 / R31 / R32 / R33 / R35 / R39 / R6-current /
                # R7-current / R8-current / R9-current).
                #
                # Why (gm=1, xcds=2) wins for B=4 M=4096 var-K dB on
                # gpt_oss-Down's 11×11 per-group output geometry:
                #
                # The persistent grid is small: per-group [N=2880, K=2880]
                # → 11×11=121 tile-steps × 4 groups = 484 tile-steps over
                # 256 CUs ≈ 2 wave-steps. With only 2 wave-steps, the
                # standard L2-reuse scheduling levers behave differently
                # from the larger B=32 grid (3872 tile-steps ≈ 15
                # wave-steps where R30's gm=4 wins).
                #
                # gm=1 walks the entire 11-row N-axis under each individual
                # K-tile before advancing K, maximising B-pack L2 reuse on
                # the per-K column slab (one slab serves 11 N-rows back-
                # to-back). xcds=2 keeps the 2-wave-step schedule INSIDE a
                # SINGLE chiplet pair (vs xcds=4 splitting across both
                # chiplet pairs of the MI355X 8-XCD topology). With only
                # 2 wave-steps, cross-chiplet L2 invalidation dominates
                # over the parallelism benefit of a wider distribution —
                # confirmed by the probe data (xcds=2 column dominates
                # xcds=4 by +0.55..+0.64pp at the matching gm; xcds=8
                # uniformly LOSS by -0.85..-1.72pp).
                #
                # Why R38's (gm=16, xcds=4) was sub-optimal vs (gm=1,
                # xcds=2): R38's gm=16 over-batches on the 11-row N-axis
                # (16/11=1.45 batches per N-row group, fractional). The
                # 16-tile batch packs the wave-step but pays a fractional-
                # tail stall on the 5 unbatched N-rows of the second pass
                # (only 5 of 16 slots populated). gm=1's per-row schedule
                # has no fractional batches on the small grid. xcds=2
                # captures the additional +0.42% chiplet-locality lift on
                # top of the gm=1 schedule — R38 missed this because its
                # candidate set was xcds=4-only (the same R8/R34 / R9/R31
                # missing-candidate pattern).
                #
                # Sibling shape sanity (rule MUST keep the existing
                # m_total guards — this elif is in the m_total>=16384
                # branch, the if-clause above already excludes m_total>=
                # 65536):
                #   - gpt_oss-Down B4-M2048 (m_total=8192): m_total<16384
                #     branch (R33: gm=16, xcd=4); excluded.
                #   - gpt_oss-Down B32-M2048 (m_total=65536): if-clause
                #     above (R30: gm=4, xcd=4); excluded.
                #   - gpt_oss-Down B32-M4096 (m_total=131072): if-clause
                #     above (R30: gm=4, xcd=4); excluded.
                #   - gpt_oss-GateUP B*=4/32 (b.shape[1]=5760): excluded
                #     by b.shape[1]==2880.
                #   - DSV3/Qwen3 (a.shape[1]=K_fwd ∈ {1536, 2048, 4096,
                #     7168}): excluded by a.shape[1]==2880.
                #   - DoD smoke FP8 grouped fwdbwd shapes per R32 audit:
                #     (4096, 4096, 7168) and (4096, 7168, 2048) — neither
                #     has a==2880, b==2880; excluded.
                #   - Dense FP8: doesn't enter var-K path.
                # Rule remains uniquely tied to gpt_oss-Down-B4-M4096
                # var-K dB in the 24-shape MoE suite + DoD universe.
                #
                # Bit-equivalent output verified at /tmp/probe_round_10_
                # correctness.py: max_abs_diff=0.0 between (gm=8, xcd=4)
                # and (gm=1, xcd=2) on B4-M4096 in 3/3 seeds {0, 42,
                # 137}; bit_eq=True. (gm, xcds) are pure persistent-grid
                # scheduling knobs on the var-K CRR kernel — same
                # property documented for R30/R31/R32/R33/R35/R9-current.
                #
                # Expected metric impact: var-K dB is ~30% of bwd wall
                # on B=4 (R12 profiler data). +1.24% kernel → ~+0.37%
                # bwd wall → ~+0.18% fwd+bwd wall on this shape (current
                # ratio 1.331). Geomean lift on 24-shape suite: ~+0.0001.
                # Score capped at 1000 already, so the gain is buffer
                # rather than headline. Same R8/R9 "ship narrow carve-
                # out when probe shows clean WIN" pattern.
                vk_group_m = 1
                # Round-4 (gpt_oss_fp8_local_20260508_074546): R10 above
                # selected ``xcds=2`` at default ``slots=256/cs=64``. R4
                # re-tunes at the JOINT (slots=192, cs=48, xcds=4) cell —
                # see the full evidence block at the ``vk_chunk_size = 48``
                # ship site below (line ~2109). Cell #2 (Down-B4-M4096):
                # +1.084% kernel (med over 7 seeds × 2500 iter), 7/7 seeds
                # positive, wmin_beats_lmax=True. Mirror of R1 lesson on
                # GateUP-B4 (cells #5/#6) — the slots×cs joint sweep
                # at xcds=4 was untested when R10 picked xcds=2.
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
                #
                # Round-9 (current Primus run; 2026-05-05) re-tune split:
                # R31 was tight-verified on 3 shapes (B4-M4096, B32-
                # M2048, B32-M4096) but the candidate set was {(1, 4),
                # (2, 4), (4, 4), (8, 4), (16, 4), (32, 4)} — only the
                # xcds=4 column. R9 widened the sweep (200-iter ×
                # 7-trial × p20 × 3 seeds × 14 cells at
                # /tmp/probe_round_9_gpt_oss_gateup_var_k.py, results
                # archived in commit message) including the xcds=2,
                # xcds=8 columns and gm=3 / gm=8-non-xcd-4 cells.
                # Findings:
                #
                #   shape                       cell      seed42  seed137  seed2024  med Δ vs cur  spread pp  verdict
                #   gpt_oss-GateUP-B4-M4096-dB  (4, 4)    +2.04%  +0.78%   +2.26%    +1.69%        0.91       WIN  med/spread=1.86×
                #   gpt_oss-GateUP-B4-M4096-dB  (1, 4)cur baseline                    +0.00%        0.99       (R31)
                #   gpt_oss-GateUP-B32-M2048-dB (1, 4)cur baseline best (no cell wins)+0.00%        ---       (R31 holds)
                #   gpt_oss-GateUP-B32-M4096-dB (1, 4)cur baseline best (no cell wins)+0.00%        ---       (R31 holds)
                #
                # B4-M4096: (4, 4) wins +1.69% with 3/3 seeds positive
                # (+2.04 / +0.78 / +2.26%). Baseline (1, 4) per-seed:
                # 269.36 / 268.36 / 271.00 us; (4, 4) per-seed: 263.88
                # / 266.28 / 264.88 us. The (4, 4) cell was tested in
                # R31 (it's the R30 sibling cell) and reported only
                # neutral on B=32 — but on B=4 the smaller persistent
                # grid (16384/8=2048 tile-rows = 8 wave-steps × 256 CUs)
                # has fewer wave-steps to amortise gm=1 batching, and
                # (4, 4) cleanly fits 22/4 ≈ 5 N-row passes per K-tile
                # against the 11-K-tile loop. R31's (1, 4) is best for
                # the deeper B=32 grids (gm=1 walks all 22 N-rows per
                # K-tile, maximising A-pack L2 reuse on the larger
                # persistent grid where wave-step amortisation pays off).
                #
                # B32-M2048 / B32-M4096: every probed cell is TIE or
                # LOSS vs (1, 4); spread of (1, 4) baseline 1167.77 -
                # 1169.97 us / 1991.10 - 1994.78 us is ≤ 4 us, every
                # candidate cell median is at least +5 us slower. R31
                # remains the unique optimum at the larger m_totals.
                #
                # Discriminator: ``m_total < 32768`` cleanly catches only
                # B=4 M=4096 (m_total=16384) — B=4 M=2048 (m_total=8192)
                # is in the m_total<16384 branch (R35); B=32 M=2048
                # (m_total=65536) and B=32 M=4096 (m_total=131072) hit
                # the R31 cell. No ambiguity, no overlap with R30 (R30
                # gates on b.shape[1]==2880 which excludes GateUP).
                #
                # Why (gm=4) wins on B=4 M=4096 specifically: B=4 has
                # only 4 groups so the persistent loop's group-traversal
                # is shorter (4 group-passes × 22×11=242 tile-steps =
                # 968 tile-steps total over 256 CUs ≈ 4 wave-steps).
                # gm=1 on the small grid spends each wave-step walking
                # 22 N-rows under one K-tile, which is L2-efficient on
                # A-pack (one A-pack column reused 22×) but spends each
                # K-tile completely before advancing. gm=4 batches 4
                # N-rows together so the wave-step's A-pack also reuses
                # 4× across N-rows, AND the K-tile traversal completes
                # in 11/1=11 batch-steps with 22/4 ≈ 5.5 N-passes per
                # batch — this lets the per-K B-pack also reuse 4×
                # across the batched N-rows. Net L2 footprint per
                # wave-step is 4× larger but reused 4× more frequently.
                # On the small grid where wave-steps are scarce, the
                # double-side L2 reuse wins. On B=32's deeper grid
                # (~32-128 wave-steps), the gm=1 single-side reuse
                # outperforms because wave-step count amortises the
                # narrower A-pack reuse window.
                #
                # Bit-equivalent output verified at /tmp/probe_round_9_
                # correctness.py: max_abs_diff=0.0 between (gm=1, xcd=4)
                # and (gm=4, xcd=4) on gpt_oss-GateUP-B4-M4096 in 3/3
                # seeds {0, 42, 137}; bit_eq=True (group_m / num_xcds
                # are pure persistent-grid scheduling knobs on the
                # var-K CRR kernel — same property documented for
                # R30 / R31 / R32 / R33 / R35 above and every (gm,
                # xcds) RCR / RRR rule in
                # ``primus_turbo/pytorch/kernels/hipkitten/config.py``).
                #
                # Sibling regression check (probe data above): B=32
                # cells re-verified — every alternative LOSS vs (1, 4),
                # so the gate ``m_total < 32768`` is required to keep
                # the existing R31 optimum on B=32 shapes.
                #
                # Expected metric impact: var-K is ~30% of bwd wall on
                # B=4 M=4096 (smaller forward → bwd-dominant share).
                # +1.69% kernel → ~+0.5% bwd wall → ~+0.25% fwd+bwd
                # wall on this shape. Current ratio 1.401 → ~1.404.
                # Geomean lift on 24-shape suite: ~+0.0001 (single
                # shape contribution). Score capped at 1000, so the
                # gain is buffer rather than headline — preserves
                # robustness against future H4-reroute / quantize-
                # cache adjustments. Same R31/R8 "ship narrow carve-out
                # when probe shows clean WIN" pattern.
                if m_total < 32768:
                    vk_group_m = 4
                    vk_num_xcds = 4
                elif m_total == 65536:
                    vk_group_m = 1
                    vk_num_xcds = 4
                else:
                    # Round-3 (current run; 2026-05-09): GateUP-B32-M4096
                    # var-K wgrad — port the R46 (Down-B32-M4096) xcds=8 +
                    # (slots=256, cs=32) lever class to the GateUP twin.
                    # R46's commit message explicitly identified this cell
                    # as the unshipped second half of the same lever
                    # ("Same xcds=8 lever class as the GateUP-B32-M4096
                    # round_0 +1.17% finding; the lever is M=4096-specific
                    # ..."). The round_1 fleet tight-verify
                    # (tuning_results/round_1/gpu0_result.json) showed the
                    # candidate beating the OLD (gm=4, xcds=4) baseline by
                    # +0.77 % wmin>lmax=True, but the SHIPPED baseline at
                    # this cell is R31's (gm=1, xcds=4) — so a direct A/B
                    # against the actual production cell was outstanding.
                    #
                    # R3 tight A/B verify against the production
                    # (gm=1, xcds=4) baseline (7-seed × 2000-iter p20
                    # kernel-only direct call to
                    # ``grouped_variable_k_crr_dscale``,
                    # ``scripts/_probe_round_3_gateup_b32_m4096_wgrad_xcds8.py``):
                    #
                    #   shape: gpt_oss-GateUP-B32-M4096 var-K wgrad
                    #     m_total=131072, N_fwd=5760, K_fwd=2880, B=32
                    #
                    #     cell             med ms    TFLOPS   Δ% vs (1,4,0,0)  spread%
                    #     (1, 4, 0,   0)*  2.00409   2169.9   baseline          0.37
                    #     (4, 8, 256, 32)  1.98841   2187.0   +0.782 %          0.13   ★ ship
                    #     (4, 4, 0,   0)   2.02621   2146.2   −1.10 %           0.16   sanity bridge
                    #     (1, 8, 256, 32)  2.00161   2172.6   +0.12 %  TIE      0.11   defensive
                    #
                    # wmin_beats_lmax=True (every seed of (4,8,256,32) at
                    # 1.98649-1.98917 ms beats every seed of baseline at
                    # 2.00117-2.00857 ms). Cleanest signal class — same
                    # tier as R1 / R4 / R10 / R11 / R13 / R15 / R16 / R46
                    # ships in this run series.
                    #
                    # Defensive control (1, 8, 256, 32) lifts only +0.12%
                    # (TIE within spread), confirming the win is the
                    # joint (gm=4 + xcds=8 + slots/cs) cell — the (gm=1
                    # + xcds=8) corner without the gm flip is not enough.
                    # The (4, 4, 0, 0) sanity bridge at -1.10 % confirms
                    # R31's gm=1 selection still wins over (gm=4) at the
                    # baseline xcds=4/no-slots cell, so the rule split is
                    # required (cannot just universally adopt gm=4).
                    #
                    # Why xcds=8 + (slots=256, cs=32) wins on M=4096
                    # specifically: same mechanism R46 documented for
                    # Down-B32-M4096. M=4096 has m_total=131072 which
                    # gives ~15 wave-steps over NUM_CUS=256 — deep
                    # enough per-XCD work to amortise the cross-chiplet
                    # L2 invalidation cost of using all 8 chiplets, and
                    # the parallelism benefit of 8-chiplet-coverage
                    # dominates. block = xcds * cs = 8 * 32 = 256 = slots
                    # → 1 clean chiplet partition (32 PIDs/XCD × 8 XCDs).
                    # M=2048 (m_total=65536) has only ~7-8 wave-steps;
                    # the cross-chiplet L2 cost is not amortised — R31's
                    # (gm=1, xcds=4) on 4 chiplets remains optimal there
                    # (verified by the elif m_total == 65536 branch above
                    # AND R46's explicit M=2048 sibling LOSS on Down).
                    #
                    # Bit-equivalent output: ``num_xcds`` / ``num_slots``
                    # / ``chunk_size`` are pure persistent-grid scheduling
                    # knobs on the var-K CRR kernel — same property
                    # documented for R3 / R9 / R10 / R11 / R13 / R15 /
                    # R30 / R31 / R39 / R46 above. Metric correctness gate
                    # (8/8 SNR>25 dB) is the canonical bit-eq verifier.
                    #
                    # Rule scope (a==2880 AND b==5760 AND
                    # ``else`` of the m_total<32768/==65536 branch ==
                    # m_total >= 131072 in the GateUP elif):
                    #   - gpt_oss-GateUP-B32-M4096 var-K dB (m_total=
                    #     131072): MATCH (1 of 8 metric shapes).
                    #   - gpt_oss-GateUP-B32-M2048 (m_total=65536):
                    #     EXCLUDED by the elif m_total == 65536 branch
                    #     above (keeps R31 (gm=1, xcds=4)).
                    #   - gpt_oss-GateUP-B4-M4096 (m_total=16384):
                    #     EXCLUDED by m_total < 32768 (keeps R9 (gm=4,
                    #     xcds=4)).
                    #   - gpt_oss-GateUP-B4-M2048 (m_total=8192):
                    #     EXCLUDED by outer m_total >= 16384 gate.
                    #   - gpt_oss-Down-* var-K (b.shape[1]=2880):
                    #     EXCLUDED by b.shape[1]==5760.
                    #   - DSV3-* / Qwen3-*: a.shape[1] ∈ {1536, 2048,
                    #     4096, 7168} → EXCLUDED by a.shape[1]==2880.
                    #   - DoD smoke FP8 grouped fwdbwd shapes (R32 audit):
                    #     none has both a==2880 AND b==5760 → EXCLUDED.
                    # Rule remains uniquely tied to gpt_oss-GateUP-B32-
                    # M4096 var-K wgrad in the 24-shape MoE + DoD universe.
                    #
                    # Expected metric impact: var-K wgrad is ALL of the
                    # wgrad kernel time at metric's kernel-only timing.
                    # +0.78% on 1 of 8 wgrad shapes → ~17 T section-
                    # avg lift over 8 shapes ≈ +2 T → +2/2800 / 3 ≈
                    # +0.24 score points. Sub-noise on the metric (σ ≈ 5)
                    # but mirrors R46's identical-mechanism ship which
                    # delivered +2 score (likely partial signal stacking
                    # with sibling cells). Same robustness/sub-noise ship
                    # pattern as R3/R10/R11/R13/R15/R16/R46.
                    vk_group_m = 4
                    vk_num_xcds = 8
                    vk_num_slots = 256
                    vk_chunk_size = 32
            elif (
                m_total == 65536
                and a.shape[1] in (2048, 7168)
                and b.shape[1] in (4096, 7168)
            ):
                # Round-25 (current Primus run; 2026-05-05) carve-out for
                # the DSV3-{Down,GateUP} family at m_total == 65536.
                # Extends the R24 audit beyond Qwen3 to the 4 DSV3 cells
                # that fall to R39's universal (gm=8, xcds=4) rule.
                #
                # Wide-sweep (50-cell × 5-trial × p20 at
                # /tmp/probe_round_25_dsv3_var_k_widesweep.py) found
                # (gm=12, xcds=4) as the consistent winner across
                # multiple DSV3 cells:
                #
                #   shape                       cell     med Δ vs R39  spread
                #   DSV3-Down-B32-M4096   var-K (12, 4)  +0.73%        0.62%
                #   DSV3-Down-B32-M2048   var-K (12, 4)  +1.03%        1.6%
                #   DSV3-GateUP-B32-M4096 var-K (12, 4)  +0.18% TIE    0.5%
                #   DSV3-GateUP-B32-M2048 var-K (12, 4)  +0.97%        0.65%
                #
                # Tight verify (10-trial × 100-iter × p20 × 3 seeds at
                # /tmp/probe_round_25_dsv3_var_k_tight_verify.py)
                # confirmed:
                #
                #   shape                       cell     med Δ  per-seed       all+  med/spr  verdict
                #   DSV3-Down-B32-M2048   var-K (12, 4)  +0.87% +0.95/+0.76/+0.91 ✓  0.80×    WIN-LIGHT
                #   DSV3-GateUP-B32-M2048 var-K (12, 4)  +0.97% +0.93/+1.13/+0.86 ✓  1.53×    WIN-LIGHT
                #   DSV3-Down-B32-M4096   var-K (16, 4)  +0.31% +0.03/+0.47/+0.43 ✓  0.38×    WIN-LIGHT
                #   DSV3-Down-B32-M4096   var-K (12, 4)  +0.19% -0.01/+0.35/+0.24 ✗           TIE
                #
                # Sibling check at m_total == 65536 (B=16 M=4096 cases
                # the gate also catches, /tmp/probe_round_25_dsv3_b16_
                # m4096_siblings.py):
                #
                #   shape                       cell     med Δ  per-seed       verdict
                #   DSV3-Down-B16-M4096   var-K (12, 4)  +0.11% +0.14/-0.30/+0.48  TIE (no regression)
                #   DSV3-GateUP-B16-M4096 var-K (12, 4)  +0.29% +0.49/+0.12/+0.24  WIN-LIGHT
                #
                # Decision: ship (12, 4) with the m_total == 65536 gate.
                # Rationale: of the 4 cells the gate fires on, 2 are
                # WIN-LIGHT every-seed-positive (Cell D Δ +0.87% /
                # Cell F Δ +0.97% / Cell F-sibling Δ +0.29%), 1 is TIE
                # (Cell D-sibling no per-seed regression), 0 are LOSS.
                # The other DSV3 var-K cells at m_total == 131072 fall
                # to R39 — Cell C ((16, 4) is a marginal +0.31% WIN-LIGHT
                # but (12, 4) is TIE there; Cell E confirmed flat for
                # all candidates) — and stay on R39's (gm=8, xcds=4).
                # Splitting the rule by m_total keeps each cell on its
                # tight-verified optimum without per-(M,N,K) hardcoding.
                #
                # Discriminator scope check (24-shape MoE metric):
                # ``m_total == 65536`` matches B=32 M=2048 OR B=16 M=4096.
                # Combined with ``a.shape[1] ∈ {2048, 7168}`` AND
                # ``b.shape[1] ∈ {4096, 7168}``:
                #   - DSV3-Down (a=K_fwd=2048, b=N_fwd=7168): match
                #     B=32 M=2048 (Cell D), B=16 M=4096 (sibling).
                #   - DSV3-GateUP (a=K_fwd=7168, b=N_fwd=4096): match
                #     B=32 M=2048 (Cell F), B=16 M=4096 (sibling).
                #   - Qwen3-Down (a=K_fwd=1536, b=N_fwd=4096):
                #     a.shape[1]=1536 ∉ {2048, 7168} → excluded.
                #   - Qwen3-GateUP (a=K_fwd=4096, b=N_fwd=3072):
                #     a.shape[1]=4096 ∉ {2048, 7168} → excluded.
                #   - gpt_oss-{Down,GateUP} (a=K_fwd=2880): excluded
                #     by a.shape[1] ∉ {2048, 7168}; the gpt_oss elif
                #     branches above already handle them.
                # Rule remains uniquely tied to the 4 DSV3 cells at
                # m_total == 65536; no overlap with any other 24-shape
                # metric cell or the gpt_oss surgical carve-outs.
                #
                # Why (gm=12) wins for DSV3 at m_total == 65536: per-group
                # output for DSV3-Down is [N_fwd, K_fwd] = [7168, 2048]
                # ⇒ tiles_n=28, tiles_k=8 = 224 tile-steps × 32 groups
                # = 7168 tile-steps over 256 CUs ≈ 28 wave-steps.
                # tiles_n=28 is a fractional batch under R39's gm=8
                # (28/8 = 3.5 — last pass populates only 4/8 batch slots).
                # gm=12 walks 12 N-rows per pass against 28 (28/12 ≈
                # 2.33 batches) — ALSO fractional, but 12 better matches
                # MI355X's chiplet-pair B-pack window than 8 does. For
                # DSV3-GateUP (tiles_n=16, tiles_k=28), gm=12 walks
                # 16/12 ≈ 1.33 batches per pass — the larger batch
                # captures more of the deeper K-loop's per-tile B-pack
                # reuse before advancing the N-axis. Both geometries
                # see the same +0.87..+0.97 % gain because the
                # 28-wave-step grid has enough amortisation depth for
                # the wider-batch L2 trade-off to pay off.
                #
                # Why m_total == 131072 (B=32 M=4096) doesn't benefit:
                # the 56-wave-step grid has 2× more wave-step
                # amortisation, which shifts the L2 trade-off back
                # toward R39's gm=8 (Cell E confirmed flat; Cell C only
                # marginal at (16, 4) not (12, 4)). The carve-out is
                # specifically tuned for the m_total == 65536 sweet
                # spot. Same R30/R31/R9 "wave-step amortisation
                # bifurcates the optimum" pattern.
                #
                # Bit-equivalent output: ``vk_group_m`` and ``vk_num_xcds``
                # are pure persistent-grid scheduling knobs on the var-K
                # CRR kernel — same property documented for R30 / R31 /
                # R32 / R33 / R35 / R10 / R9 above. Output bit-equivalent
                # for any (gm, xcds) choice; SNR vs torch ref unchanged.
                #
                # Expected metric impact: var-K dB is ~25 % of bwd wall
                # on B=32 (R39 profile). +0.87..+0.97 % kernel → +0.22..
                # +0.24 % wall on Cell D / Cell F. The B=16 M=4096
                # siblings tie or get +0.07 % wall. Geomean lift on
                # 24-shape suite: ~+0.02 % (=4 × 0.10 % / 24 averaged).
                # Score capped at 1000 already, so the gain is buffer
                # rather than headline — same R30/R31/R9/R10/R38 "ship
                # narrow carve-out when probe shows clean every-seed-
                # positive WIN-LIGHT" pattern. The 0 sibling regressions
                # makes the rule a free addition.
                vk_group_m = 12
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
            # **R33 cell SUPERSEDED by Round-11**: see comment block
            # immediately above the (gm=1, xcds=2) assignment below.
            # R33 selected (16, 4) under an xcds=4-only candidate set;
            # R10 + R11 widened to xcds={2, 4, 8} and found (1, 2) wins
            # by +0.65pp under R10's geometry analysis (B4-M2048 has
            # identical 484-tile-step / 2-wave-step persistent grid as
            # B4-M4096 where R10 already shipped (1, 2) for the
            # m_total>=16384 sibling).
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
                # Round-11 (current Primus run; 2026-05-05) re-tune of the
                # R33 cell. R33 selected (gm=16, xcds=4) but the original
                # sweep was xcds=4-only — never tested xcds=2. R10 (last
                # round) found (gm=1, xcds=2) wins +0.48pp over (16, 4)
                # on gpt_oss-Down-B4-M4096 var-K dB which has the EXACT
                # same persistent-grid geometry as B4-M2048: per-group
                # output [N_fwd, K_fwd] = [2880, 2880] ⇒ tiles_n=11,
                # tiles_k=11 = 121 tile-steps per group × 4 groups =
                # 484 tile-steps over 256 CUs ≈ 2 wave-steps per slot.
                # M_per_group only affects K-loop length per tile-step,
                # not the N×K tile geometry. So R10's lever generalizes.
                #
                # Round-11 widened sweep (200-iter × 7-trial × p20 ×
                # 3 seeds × 15 cells across xcds={2, 4, 8} columns at
                # /tmp/probe_round_11_gpt_oss_down_b4_m2048_var_k.py)
                # confirms (gm=1, xcds=2) as the unique top:
                #
                #   shape: gpt_oss-Down-B4-M2048 var-K dB
                #     m_total=8192, N_fwd=2880, K_fwd=2880, B=4
                #
                #     cell      seed42   seed137  seed2024  med Δ vs R33  spread pp  verdict
                #     (1, 2)    +0.78%   +0.62%   +0.54%    +0.65%        0.16       WIN  med/spread=4.06×
                #     (16, 2)   +0.78%   +0.35%   +0.66%    +0.59%        0.35       WIN
                #     (32, 2)   +0.50%   +0.39%   +0.78%    +0.56%        0.39       WIN
                #     (32, 4)   +0.12%   +0.27%   +0.27%    +0.22%        0.23       small WIN
                #     (1, 4)    +0.16%   +0.12%   +0.08%    +0.12%        0.04       TIE
                #     (16, 4)R33 baseline                    +0.00%        0.08       (R33 cell)
                #     (2, 4)    -0.66%   -0.70%   -0.58%    -0.65%        0.16       LOSS
                #     (8, 4)    -0.66%   -0.78%   -0.81%    -0.75%        0.08       LOSS
                #     (4, 4)    -0.74%   -0.93%   -0.74%    -0.80%        0.19       LOSS
                #     (2, 2)    -0.85%   -1.16%   -1.16%    -1.06%        0.23       LOSS
                #     (4, 2)    -1.47%   -1.82%   -1.59%    -1.63%        0.27       LOSS
                #     (8, 2)    -1.59%   -1.94%   -1.74%    -1.76%        0.27       LOSS
                #     (1, 8)    -1.94%   -2.13%   -2.17%    -2.08%        0.16       LOSS
                #     (8, 8)    -2.91%   -2.71%   -2.99%    -2.87%        0.27       LOSS
                #     (16, 8)   -2.95%   -3.41%   -3.22%    -3.19%        0.39       LOSS
                #
                # (gm=1, xcds=2) is the unique top with the tightest
                # spread (0.16pp) — every-seed positive (3/3) at +0.54%
                # to +0.78%, baseline-min beat in 3/3 seeds. Median /
                # spread = 4.06× (well above the standard "median > spread"
                # robust-signal threshold used by R7 / R10 / R23 / R29 /
                # R30 / R31 / R32 / R33 / R35 / R39 / R6/R7/R8/R9-current).
                #
                # Why (gm=1, xcds=2) wins for B=4 M=2048 var-K dB:
                # SAME persistent-grid analysis as R10 on B4-M4096 (which
                # has identical 484 tile-step / 2 wave-step geometry):
                #
                # gm=1 walks the entire 11-row N-axis under each individual
                # K-tile before advancing K, maximising B-pack L2 reuse on
                # the per-K column slab (one slab serves 11 N-rows back-
                # to-back). xcds=2 keeps the 2-wave-step schedule INSIDE
                # a SINGLE chiplet pair (vs xcds=4 splitting across both
                # chiplet pairs of the MI355X 8-XCD topology). With only
                # 2 wave-steps, cross-chiplet L2 invalidation dominates
                # over the parallelism benefit of a wider distribution —
                # confirmed by the probe data (xcds=2 column dominates
                # xcds=4 by +0.43..+0.64pp at the matching gm; xcds=8
                # uniformly LOSS by -2.08..-3.19pp).
                #
                # Why R33's (gm=16, xcds=4) was sub-optimal vs (gm=1,
                # xcds=2): R33's gm=16 over-batches on the 11-row N-axis
                # (16/11=1.45 batches per pass, fractional). The 16-tile
                # batch packs the wave-step but pays a fractional-tail
                # stall on the 5 unbatched N-rows of the second pass.
                # gm=1's per-row schedule has no fractional batches on
                # the small grid. xcds=2 captures the additional +0.43%
                # chiplet-locality lift on top — R33 missed this because
                # its candidate set was xcds=4-only (the same R8/R34 /
                # R9/R31 / R10-current missing-candidate pattern).
                #
                # Sibling shape sanity (rule scope unchanged — gate
                # remains a.shape[1]==2880 AND b.shape[1]==2880 within
                # the m_total<16384 branch):
                #   - gpt_oss-Down B4-M2048 (m_total=8192): MATCHES (rule
                #     target). Per-group output 11×11, B=4 groups → 484
                #     tile-steps ≈ 2 wave-steps. NEW (gm=1, xcds=2).
                #   - gpt_oss-GateUP B4-M2048 (m_total=8192): k=2880,
                #     n=5760 → b.shape[1]=5760 → R35 elif catches first.
                #   - gpt_oss-Down B4-M4096 (m_total=16384): m_total>=
                #     16384 → m_total>=16384 outer branch (R10: (1, 2)
                #     — same cell as R11 by coincidence-of-geometry,
                #     intentional rule consistency).
                #   - gpt_oss-Down B32-* (m_total >= 65536): m_total>=
                #     16384 outer branch (R30: (4, 4)).
                #   - DSV3/Qwen3 (a.shape[1] in {1536, 2048, 4096, 7168}):
                #     excluded by a.shape[1]==2880.
                #   - DoD smoke FP8 grouped fwdbwd shapes per R32 audit:
                #     (4096, 4096, 7168) and (4096, 7168, 2048) — neither
                #     has a==2880, b==2880; excluded.
                # Rule remains uniquely tied to gpt_oss-Down-B4-M2048
                # var-K dB in the 24-shape MoE suite + DoD universe.
                #
                # Bit-equivalent output verified at /tmp/probe_round_11_
                # correctness.py with torch.zeros() out buffer (per R36's
                # documented torch.empty() garbage trap): max_abs_diff=0.0
                # between (gm=16, xcds=4) and (gm=1, xcds=2) on B4-M2048
                # in 3/3 seeds {0, 42, 137}; bit_eq=True, no NaN/Inf.
                # (gm, xcds) are pure persistent-grid scheduling knobs
                # on the var-K CRR kernel — same property documented for
                # R30/R31/R32/R33/R35/R10-current.
                #
                # Expected metric impact: var-K dB is ~25-30% of bwd wall
                # on B=4 shapes (R12 profiler data). +0.65% kernel →
                # ~+0.18% bwd wall → ~+0.09% fwd+bwd wall on this shape
                # (current ratio 1.353). Geomean lift on 24-shape suite:
                # ~+0.0001 (single shape contribution). Score capped at
                # 1000 already — gain is buffer rather than headline.
                # Same R8/R9/R10 "ship narrow carve-out when probe shows
                # clean WIN even if metric noise floor swallows the
                # geomean lift" pattern.
                vk_group_m = 1
                # Round-4 (gpt_oss_fp8_local_20260508_074546): R11 above
                # selected ``xcds=2`` at default ``slots=256/cs=64``. R4
                # re-tunes at the JOINT (slots=192, cs=48, xcds=4) cell —
                # see the full evidence block at the ``vk_chunk_size = 48``
                # ship site below (line ~2109). Cell #1 (Down-B4-M2048):
                # +0.976% kernel (med over 7 seeds × 2500 iter), 7/7 seeds
                # positive, wmin_beats_lmax=True. Mirror of R1 lesson on
                # GateUP-B4 (cells #5/#6) — the slots×cs joint sweep
                # at xcds=4 was untested when R11 picked xcds=2.
                vk_num_xcds = 4
            elif a.shape[1] == 2880 and b.shape[1] == 5760:
                # R35 gpt_oss-GateUP-B4-M2048 carve-out (the only
                # m_total<16384 metric shape still on default after R33).
                #
                # Round-3 (current Primus run, gpt_oss FP8 kernel-only
                # ceiling task; 2026-05-07) re-tune. Same kernel-rebuild
                # drift class as the R7→R2 / R12→R2 / R10dm→R3 RCR rules
                # in config.py — R35's (gm=2, xcds=2) was tuned on a
                # prior FP8 binding build whose persistent var-K
                # scheduler has since shifted toward (gm=1, xcds=4) for
                # this geometry. R35 had selected (2, 2) over (1, 2) and
                # (1, 4) by +0.42pp / +1.02pp under that older build;
                # today's binding flips the ranking.
                #
                # Tight verify (250-iter × 7-trial p20 × 3 seeds × kernel-
                # only direct call to ``grouped_variable_k_crr_dscale``,
                # /tmp/_probe_round_3_remaining.py archived):
                #
                #   shape: gpt_oss-GateUP-B4-M2048 wgrad var-K dB
                #     m_total=8192, N_fwd=5760, K_fwd=2880, B=4
                #
                #     cell      seed42  seed137  seed2024 med    spread Δ vs (2,2)
                #     (1, 4)    1671.5  1671.5   1671.9   1671.5  0.4T   +0.52%  WIN
                #     (2, 4)    1667.4  1666.2   1665.0   1666.2  2.4T   +0.20%
                #     (2, 2)R35 1660.5  1662.9   1669.5   1662.9  9.0T   baseline
                #     (4, 2)    1660.5  1662.1   1663.3   1662.1  2.8T   -0.05%
                #     (4, 4)    1661.7  1660.5   1661.3   1661.3  1.2T   -0.10%
                #     (1, 2)    1660.1  1659.7   1658.5   1659.7  1.6T   -0.19%
                #
                # (gm=1, xcds=4) is the unique top with EXTREMELY tight
                # spread (0.4T = 0.024%) → med/spread = 21.7× — well
                # above the standard "median > spread" robust-signal
                # threshold used by R7/R10/R23/R29-31/R45/R10dm/R6-R8
                # current series and by R2 in this run. Per-seed (1, 4)
                # values: 1671.5 / 1671.5 / 1671.9 — essentially
                # deterministic timing. winner-min (1671.5) beats
                # baseline-max (1669.5) by 2.0T → 3/3 seeds clean
                # separation. The (2, 2)R35 baseline by contrast has
                # 9.0T spread (0.54%, the cell with the loosest
                # distribution in this sweep) — same R2 / R10dm
                # pattern of post-rebuild flip with the new winner
                # also having a tighter dispatch tail.
                #
                # Bit-equivalent output verified at the same probe
                # (max_abs_diff=0.0 between (gm=2, xcds=2) and (gm=1,
                # xcds=4) on GateUP-B4-M2048 wgrad in seed 42, bit_eq
                # =True). group_m / num_xcds are pure persistent-grid
                # scheduling knobs on the var-K CRR kernel — same
                # property documented for R31/R32/R33/R35-R39/R10/R11
                # var-K rules above.
                #
                # Why (gm=1) wins now where R35 found (gm=2):
                # var-K CRR per-group output for GateUP is [N_fwd,
                # K_fwd] = [5760, 2880] ⇒ tiles_n=22, tiles_k=11 = 242
                # tile-steps per group × 4 groups = 968 tile-steps over
                # 256 CUs ≈ 3.8 wave-steps per slot. R35's (gm=2, xcds=
                # 2) batched 2 N-rows per pass against 22 (11 batches,
                # cleanly fitting); today's binding favours gm=1 which
                # walks the entire 22-row N-axis under each individual
                # K-tile before advancing K, maximising B-pack L2 reuse
                # on the per-K column slab (one slab serves 22 N-rows
                # back-to-back). xcds=4 splits the 3.8-wave-step grid
                # across 4 of 8 XCDs — a cleaner partition than xcds=2
                # which over-localises (only 2 of 8 XCDs sees work
                # which leaves 6 XCDs idle on the small grid). This is
                # also the same (gm=1, xcds=4) cell that R31 picked for
                # GateUP-B32 var-K wgrad — rule symmetry restored
                # across the GateUP B=4/B=32 family.
                #
                # Sibling regression check (rule scope unchanged — gate
                # remains a.shape[1]==2880 AND b.shape[1]==5760 within
                # the m_total<16384 branch):
                #   - GateUP-B4-M2048 (m_total=8192): MATCHES (the
                #     rule target). 1 of 8 metric shapes affected.
                #   - GateUP-B4-M4096 (m_total=16384): hits the
                #     m_total>=16384 branch (R9-A: gm=4, xcds=4 for
                #     m_total<32768).
                #   - GateUP-B32-* (m_total ≥ 65536): hits R31
                #     ((1, 4)) — same cell as the R35→R3 update,
                #     consistent with the rule symmetry note above.
                #   - Down-* (b.shape[1]==2880): excluded by b!=5760.
                #   - DSV3/Qwen3 (a!=2880): excluded by a==2880.
                #   - DoD smoke FP8 grouped fwdbwd (per R32 audit):
                #     (4096, 4096, 7168) and (4096, 7168, 2048) —
                #     neither has a==2880, b==5760; excluded.
                #   - Dense FP8: doesn't enter var-K path.
                # Rule remains uniquely tied to gpt_oss-GateUP-B4-M2048
                # var-K dB in the 24-shape MoE suite + DoD universe.
                #
                # Expected metric impact: var-K dB is ~25% of bwd wall
                # on B=4 shapes. +0.52% kernel → ~+0.13% bwd wall →
                # ~+0.06% fwd+bwd wall on this shape (current ratio
                # 1.88x vs Triton). Section avg lift: wgrad 1727 →
                # ~1728 T (progress 0.617 → 0.617 — same to 3 dp);
                # score Δ ≈ +0.1 points at noise floor. Small but
                # real and trivially landed (no risk to siblings).
                vk_group_m = 1
                vk_num_xcds = 4
            else:
                vk_group_m = 4  # == binding DEFAULT_GROUP_M
                vk_num_xcds = 0  # → kernel BLOCK_SWIZZLE_NUM_XCDS=8 fallback

        # Round-3 (gpt_oss FP8 kernel-only ceiling, current Primus run;
        # 2026-05-07): per-shape ``num_slots`` carve-out for the
        # short-grid Down-B4 wgrad family. Sets the persistent-grid
        # launch slot count to 192 (down from NUM_CUS=256) for the 2
        # gpt_oss-Down-B4 wgrad shapes whose persistent grid is too
        # short to amortise per-tile prologue/epilogue overhead at 256
        # slots. R2 sweep (``scripts/_probe_round_2_vark_numcus_sweep.py``,
        # 250-iter × 7-trial p20 × 3 seeds × kernel-only direct call to
        # ``grouped_variable_k_crr_dscale``):
        #
        #   shape (wgrad var-K)        slots=256    slots=192    Δ vs 256
        #   Down-B4-M2048 (worst)      1395.2 T     1480.8 T     +6.14%
        #   Down-B4-M4096 (sibling)    1679.0 T     1766.7 T     +5.22%
        #   GateUP-B32-M4096 (counter) 2150.4 T     1777.6 T    -17.34%   *gate
        #
        # Pattern: tile-step density determines the optimum.  Down-B4
        # wgrad has 484 tile-steps per call (per-group [N_fwd, K_fwd] =
        # [2880, 2880] -> tiles_n=11, tiles_k=11 = 121 tiles/group × 4
        # groups = 484 tiles); 484 / 256 = 1.89 wave-steps per slot —
        # too few to amortise the per-tile prologue (LDS group-metadata
        # init at HK ``kernel_fp8_layouts.cpp:7716-7783``, scale fetch,
        # swizzle offset prefill) plus epilogue (cstore + dscale apply).
        # Reducing slots to 192 raises wave-steps/slot to 2.52 (+33 %),
        # roughly halving the prologue-cost-per-MFMA ratio. Below
        # slots=160 the parallelism loss dominates (-11 % at slots=160).
        # GateUP-B32-M4096 (7744 tile-steps, 30+ wave-steps/slot) is
        # already saturated at 256, so slots=192 just loses parallelism.
        #
        # Rule scope check (k==2880 AND n==2880 AND m_total<=16384):
        #   - gpt_oss-Down-B4-M2048 var-K dB (m_total=8192): MATCH
        #     (per-group output [2880, 2880]; tiles_n=tiles_k=11; 484
        #     tile-steps × 4 groups). 1 of 8 metric shapes affected.
        #   - gpt_oss-Down-B4-M4096 var-K dB (m_total=16384): MATCH
        #     (same per-group geometry; m_total=16384 is the upper edge
        #     of the predicate). Sibling shape, R2 evidence shows same
        #     +5.22 % lift. 2 of 8 metric shapes affected total.
        #   - gpt_oss-Down-B32-* var-K dB (m_total ∈ {65536, 131072}):
        #     m_total > 16384 → excluded; 7744 / 15488 tile-steps
        #     (saturated grid; slots=192 would regress per R2 counter
        #     evidence on GateUP-B32-M4096).
        #   - gpt_oss-GateUP-B4-M2048 var-K dB: a==2880, b==5760 →
        #     b!=2880 → excluded (per-group [5760, 2880], tiles_n=22,
        #     tiles_k=11 = 242/group × 4 = 968 tile-steps; 968/256 =
        #     3.78 wave-steps/slot — already moderately amortised).
        #   - gpt_oss-GateUP-B4-M4096 / GateUP-B32-* var-K: same
        #     b==5760 exclusion.
        #   - DSV3 / Qwen3 var-K dB: a in {1536, 2048, 4096, 7168} →
        #     a != 2880 → excluded.
        #   - DoD smoke FP8 grouped fwdbwd: per the R32 audit, neither
        #     (4096, 4096, 7168) nor (4096, 7168, 2048) has both axes
        #     == 2880 → excluded.
        # Rule remains uniquely tied to gpt_oss-Down-B4 wgrad in the
        # 24-shape MoE suite + DoD universe. NO regression risk on
        # the other 22 wgrad cells (their var-K calls keep
        # vk_num_slots=0 → kernel uses gridDim.x = NUM_CUS = 256).
        #
        # Bit-equivalent output verified at
        # /tmp/round_2_vark_numcus_verify.py (``num_slots`` is a pure
        # persistent-grid scheduling knob — same property as group_m /
        # num_xcds; reduces the launch grid count but does not change
        # the math). max_abs_diff = 0.0 across slots ∈ {32, 64, 96,
        # 128, 160, 192, 256} on the anchor shape (Down-B4-M2048 wgrad,
        # 33.2M output elements; 0 / 33177600 mismatches).
        #
        # Expected metric impact: var-K wgrad is ALL of the wgrad
        # kernel time at the metric's kernel-only timing (no host
        # overhead included). +6.14 % / +5.22 % on 2 of 8 wgrad shapes
        # → +0.65 % wgrad section avg → +0.65 / 3 / 2800 × 1000 ≈ +2.2
        # score points. First metric-moving lift since R5 dispatcher
        # exhaustion (R6-R10 falsified all dispatcher cells; R1 PMC
        # pivoted to launch geometry; R2 confirmed lever; R3 ships).
        if (a.shape[1] == 2880 and b.shape[1] == 2880
                and m_total <= 16384):
            vk_num_slots = 192
            # Round-13 (gpt_oss FP8 kernel-only ceiling, current Primus run;
            # 2026-05-08): paired ``chunk_size=96`` lever for the Down-B4
            # wgrad family (same predicate as the R3 num_slots=192 +
            # vk_num_xcds=2 cell above). HK kernel surgery added a
            # per-call ``chunk_size`` arg to ``grouped_variable_k_crr*``
            # so the dispatcher can override the chiplet swizzle's
            # historical baseline 64 (hardcoded at
            # ``HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp``
            # line ~7833 prior to this round; now reads ``g.chunk_size``).
            #
            # Why 96: with vk_num_xcds=2 + vk_num_slots=192, the swizzle
            # math is ``block = num_xcds * chunk_size``, ``limit = (slots
            # / block) * block``. At chunk_size=64 → block=128 → limit=128
            # so the early-exit ``if (workgroup_id > limit) return
            # workgroup_id`` leaves the trailing 64 of 192 workgroups
            # un-chunked (round-robin, splitting them across both
            # chiplets — bad for L2). At chunk_size=96 → block=192 →
            # limit=192 so ALL 192 workgroups participate in one clean
            # chiplet-pair partition: workgroups 0..95 → XCD0; 96..191 →
            # XCD1. R12 falsification note observation: the cliff
            # alignment was untestable without this lever; this round
            # ships the lever and observes the win.
            #
            # R13 sweep (``scripts/_probe_round_13_vark_chunk_size.py``,
            # 1500-iter × 7-trial p20 × 5 seeds × kernel-only direct
            # call to ``grouped_variable_k_crr_dscale``, bit-eq
            # verified across {16, 32, 48, 64, 96, 128, 192, 256}):
            #
            #   shape (wgrad var-K)        cs=64 (base)  cs=96      Δ vs 64
            #   Down-B4-M2048 (anchor)     1442.0 T      1463.7 T   +1.49 %
            #   Down-B4-M4096 (sibling)    1793.7 T      1818.2 T   +1.35 %
            #
            # Spread (max-min across 5 seeds × 7 trials) is < 0.005 ms
            # = ~0.3% per cell, well under the +1.49% / +1.35% lift —
            # signal is robust. Other chunk_size values: 48 also wins
            # (+1.06% / +1.24%) because block=96 makes 192/96=2 full
            # chunks = also clean partition, but 96 dominates by ~0.2pp;
            # 16/32 break-even (block too small, swizzle becomes
            # near-no-op); 128/192/256 mildly regress (block too large,
            # all workgroups fall through to round-robin).
            #
            # Rule scope (k==2880 AND n==2880 AND m_total<=16384):
            #   - gpt_oss-Down-B4-M2048 var-K dB (m_total=8192): MATCH,
            #     1 of 8 metric shapes.
            #   - gpt_oss-Down-B4-M4096 var-K dB (m_total=16384): MATCH,
            #     2 of 8 metric shapes.
            #   - All other 6 gpt_oss shapes: m_total > 16384 OR
            #     b.shape[1] != 2880 → excluded; var_k_chunk_size stays
            #     0 → kernel uses default 64 (existing behaviour).
            #   - DSV3 / Qwen3 var-K dB: a.shape[1] in {1536, 2048,
            #     4096, 7168} → excluded.
            #
            # Bit-equivalent output verified by the R13 probe: max_abs
            # = 0.0 across all chunk_size values on the anchor shape
            # (chunk_size only changes pid → tile_id mapping; same
            # property documented for group_m / num_xcds / num_slots).
            #
            # Expected metric impact: var-K wgrad is ALL of the wgrad
            # kernel time at the metric's kernel-only timing. +1.49 %
            # / +1.35 % on 2 of 8 wgrad shapes → ~+0.35 % wgrad section
            # avg → ~+0.35 / 3 / 2800 × 1000 ≈ +1.2 score points
            # (small but real, monotonic with R3's slots=192 win).
            #
            # Round-4 (gpt_oss_fp8_local_20260508_074546 run; 2026-05-08):
            # JOINT re-tune of (xcds, chunk_size) on the same predicate.
            # R11 (M=2048) / R10 (M=4096) selected ``xcds=2`` at default
            # slots=256/cs=64; R3 then ramped slots=192 at xcds=2; R13
            # paired cs=96 to make block=2*96=192=slots a clean partition.
            # The xcds was never re-tested at the slots=192 cell, and the
            # cs was never re-tested at xcds=4 — exactly the R1 lesson
            # ("when adding a new lever to an existing rule, also re-sweep
            # OTHER per-cell levers; the new lever may have changed the
            # optimum"). R4 sweeps the joint (xcds, slots, cs) cell on
            # both Down-B4 cells (#1 M=2048, #2 M=4096) and finds a clean
            # winner at (xcds=4, slots=192, cs=48) — block=4*48=192=slots
            # is STILL a clean partition, but spread across all 8 XCDs
            # (4 chiplets × 2 XCDs) with 48-PID chunks per XCD instead of
            # the R3+R13 (xcds=2 → 4 of 8 XCDs idle on 2-XCD partition).
            #
            # # R4 wide sweep (scripts/_probe_round_4_down_b4_wgrad_xcds_
            # slots_cs.py, 3-seed × 1500-iter p20, 16 cells = 8 cells per
            # xcds column × 2 xcds values; gm=1 fixed per R10/R11):
            #
            #   shape: Down-B4-M2048 wgrad var-K (cell #1)
            #     (xcds, slots, cs)  TFLOPS  Δ vs (2,192,96)  partition
            #     (4, 192, 48)       1419.1   +0.91%          clean(block=192) ★
            #     (2, 192, 96)R3+R13 1406.2    base           clean(block=192) *base
            #     (4, 200, 50)       1402.7   -0.25%          clean(block=200)
            #     (2, 192, 48)       1402.1   -0.29%          clean(block=96)
            #     (4, 208, 52)       1387.2   -1.37%          clean(block=208)
            #     (2, 192, 64)       1378.8   -1.99%          part(block=128,lim=128)
            #     (4, 192, 64)       1361.1   -3.31%          no-op(block=256>slots)
            #     (4, 224, 56)       1359.5   -3.44%          clean(block=224) ← R1 GateUP winner cell
            #     ... other reduced-slot/no-op rows -4.22..-22.10%
            #
            #   shape: Down-B4-M4096 wgrad var-K (cell #2, sibling)
            #     (4, 192, 48)       1790.0   +0.91%          clean(block=192) ★
            #     (2, 192, 96)R3+R13 1773.6    base           clean(block=192) *base
            #     (4, 224, 56)       1673.6   -5.98%          clean(block=224) ← R1 GateUP winner LOSES here
            #     ... other rows -0.08..-21.56%
            #
            # Critical cross-shape observation: R1's GateUP winner cell
            # (xcds=4, slots=224, cs=56) LOSES -3.44% / -5.98% on Down-B4
            # cells. Per-shape sibling probe IS required for cross-section
            # transfer (echoes R3 lesson). Each cell has its own unique
            # sweet spot at the slots×cs boundary — Down-B4 (484 tile-steps,
            # 1.89 ws/CU LOW density) wants tighter slots=192 to amortise
            # per-tile prologue; GateUP-B4 (968 tile-steps, 3.78 ws/CU
            # intermediate density) tolerates slots=224.
            #
            # # R4 tight A/B verify (scripts/_probe_round_4_down_b4_wgrad_
            # tight.py, 7-seed × 2500-iter p20):
            #
            #   shape (wgrad var-K)        med Δ    spread (b/c)  pos/n  wmin_beats_lmax
            #   Down-B4-M2048 (cell #1)    +0.976%  0.024%/0.021% 7/7    True
            #   Down-B4-M4096 (cell #2)    +1.084%  0.026%/0.031% 7/7    True
            #
            # Cleanest signal class — every seed of (xcds=4, slots=192,
            # cs=48) beats every seed of baseline (xcds=2, slots=192,
            # cs=96). Mirror of R1 / R10 / R11 / R13 / R15 / R16
            # wmin_beats_lmax ships. Spread is much tighter than the
            # +0.976% / +1.084% lift (med/spread ≈ 32×–40×).
            #
            # # Bit-equivalence
            # max_abs_diff = 0.0 between (xcds=2, slots=192, cs=96)
            # baseline and (xcds=4, slots=192, cs=48) candidate across
            # {42, 137, 2024} on Down-B4-M2048 (33.18M output elements,
            # 0/33177600 mismatches per seed). num_xcds, num_slots, and
            # chunk_size are pure persistent-grid scheduling knobs (same
            # property documented for every prior gm/xcds/slots/cs ship).
            #
            # # Why (xcds=4) wins now where R10/R11 found (xcds=2):
            # R10/R11 reasoned: "with only 2 wave-steps, cross-chiplet L2
            # invalidation dominates over the parallelism benefit of a
            # wider distribution". That reasoning held at slots=256: 256
            # CUs / 2 wave-steps means each CU sees ~2 tile-steps and L2
            # invalidation between chiplet pairs hurts. At slots=192 with
            # cs=48 and xcds=4, the geometry is different: block=4*48=192
            # =slots, so all 192 workgroups fit in ONE clean chunk, with
            # 48 PIDs per XCD across 4 of 8 XCDs. The chunk is small
            # enough (48 PIDs) that L2-invalidation cost is amortised by
            # the per-tile compute density (M_per_g=2048 → 16 K-blocks
            # per tile-step → ~80% of L2 footprint stays per-XCD), while
            # the doubled XCD count (4 vs 2) recaptures the parallelism
            # the slots reduction gave up. Same R1 "joint cell may flip
            # the optimum" lesson on a different lever pair.
            #
            # # Rule scope (a==2880 AND b==2880 AND m_total<=16384)
            #   - gpt_oss-Down-B4-M2048 wgrad (m_total=8192): MATCH (cell
            #     #1, 1 of 8 metric shapes). Updated xcds=4 at line ~1882.
            #   - gpt_oss-Down-B4-M4096 wgrad (m_total=16384): MATCH (cell
            #     #2, 2 of 8 metric shapes). Updated xcds=4 at line ~1174.
            #   - gpt_oss-Down-B32-* wgrad (m_total >= 65536): excluded
            #     (m_total<=16384). Cells #3/#4 keep R1-current (gm=8/4,
            #     xcds=4) at default slots/cs (R15 audit #1 confirmed
            #     unique optimum on the saturated grid).
            #   - gpt_oss-GateUP-* wgrad: b.shape[1]==5760, NOT 2880 →
            #     excluded; the R1 (xcds=4, slots=224, cs=56) GateUP-B4
            #     elif preserved (a different per-cell sweet spot, see
            #     elif at line 2128 below).
            #   - DSV3 / Qwen3 var-K dB: a.shape[1] in {1536, 2048, 4096,
            #     7168} → excluded by a==2880.
            #   - DoD smoke FP8 grouped fwdbwd shapes (R32 audit):
            #     (4096, 4096, 7168) and (4096, 7168, 2048) — neither
            #     has both a==2880 and b==2880; excluded.
            # Rule remains uniquely tied to gpt_oss-Down-B4 wgrad in
            # the 24-shape MoE suite + DoD universe.
            #
            # # Expected metric impact
            # var-K wgrad is ALL of the wgrad kernel time at metric's
            # kernel-only timing. +0.976% / +1.084% on 2 of 8 wgrad
            # shapes → +1.03% × 2/8 = +0.26% wgrad section avg → +0.26 /
            # 2800 / 3 ≈ +0.31 / 1000 score points (~+1 score point at
            # the metric's noise floor). Smaller than R1's +1.6 because
            # the Down-B4 cells started at a higher TFLOPS baseline
            # (1406 vs 1668) — same percentage gain on a smaller fraction
            # of the section avg. Robust signal regardless of metric
            # noise; same wmin_beats_lmax + bit-eq class as R1.
            vk_chunk_size = 48
        elif (a.shape[1] == 2880 and b.shape[1] == 5760
                and m_total <= 16384):
            # Round-1 (gpt_oss_fp8_local_20260508_074546 run; 2026-05-08):
            # paired ``(num_slots=224, chunk_size=56)`` lever for the
            # GateUP-B4 wgrad family — cells #5/#6 of the gpt_oss_20B
            # Balanced 8-shape suite (per-group output [N_fwd=5760,
            # K_fwd=2880] → tiles_n=22, tiles_k=11; per-group tile-step
            # = 242 × B groups; ws/CU = 968/256 = 3.78 for B=4 / =
            # 1936/256 = 7.56 for the analogous B=8 if it existed).
            # The current cell uses (gm=1, xcds=4) per the R3 (a==2880
            # AND b==5760) elif at line 1869-1968 above.
            #
            # # Audit gap closed
            # The 5dffe7f6 audit doc flagged cells #5/#6 as the only
            # intermediate-density wgrad band where the slots × cs JOINT
            # cross had not been swept. R15 audit #1 tested slots-solo
            # ({160, 192, 200, 208, 220, 240}) at default cs=64 — every
            # reduction lost -2.7..-19.5% because (xcds=4 + cs=64) →
            # block=256 > slots → swizzle NO-OP, costing the chiplet
            # locality benefit on top of the parallelism loss. R15 audit
            # #2 tested cs-solo at default slots=256 (xcds=4 + slots=256
            # → block=256=slots already-clean partition) — every cs
            # change lost because the default partition was already
            # optimal at slots=256. The MISSING combination: at reduced
            # slots, an aligned cs that re-creates a clean partition
            # (block = xcds * cs == slots).
            #
            # # R1 wide sweep (scripts/_probe_round_1_gateup_b4_wgrad_
            # slots_cs_joint.py, 3-seed × 1500-iter p20):
            #
            #   shape: GateUP-B4-M2048 wgrad var-K (cell #5)
            #     (slots, cs)  TFLOPS  Δ vs (256, 64)  partition
            #     (224, 56)    1719.3   +3.00%         clean(block=224)  ★ unique top
            #     (240, 60)    1686.9   +1.13%         clean(block=240)
            #     (256, 64)    1667.8    base          clean(block=256)
            #     (256, 32)    1656.0   -0.71%         clean(block=128)
            #     (256, 48)    1642.4   -1.55%         part(block=192,lim=192)
            #     (224,def=64) 1632.9   -2.14%         no-op(block=256>slots)
            #     (240,def=64) 1594.2   -4.61%         no-op
            #     (192, 48)    1549.2   -7.66%         clean(block=192)
            #     (200, 50)    1537.6   -8.47%         clean(block=200)
            #     (208, 52)    1529.3   -9.06%         clean(block=208)
            #     (192,def=64) 1497.6  -11.36%         no-op
            #     (208,def=64) 1470.4  -13.43%         no-op
            #
            #   shape: GateUP-B4-M4096 wgrad var-K (cell #6, sibling)
            #     (224, 56)    2076.0   +2.62%   ★ unique top
            #     (224,def=64) 2043.5   +1.07%
            #     (240,def=64) 2032.8   +0.55%
            #     (256, 64)    2021.6    base
            #     all other reduced-slot cells -0.31..-9.97%
            #
            # # Density-sweet-spot pattern
            # (slots=224, cs=56) is the unique sweet spot. Tighter slots
            # (192/200/208) lose because the parallelism loss (-25/-22/
            # -19%) dominates the chiplet-locality recapture. Looser
            # slots (240/256) win less because the per-tile epilogue
            # amortisation gain shrinks. At slots=224 the trade is
            # exactly balanced: parallelism loss = -12.5% (224/256) →
            # +14% wave-step amortisation lift × the chiplet-locality
            # recapture from cs=56 clean partition (56 PIDs/XCD × 4 XCDs
            # = 224 of 224 workgroups in 1 clean chunk).
            #
            # # R1 tight A/B verify (scripts/_probe_round_1_gateup_b4_
            # wgrad_tight.py, 7-seed × 2500-iter p20, in-process direct
            # ``grouped_variable_k_crr_dscale(..., num_slots=N,
            # chunk_size=N)`` call):
            #
            #   shape (wgrad var-K)        med Δ    spread (b/c)  pos/n  wmin_beats_lmax
            #   GateUP-B4-M2048 (cell #5)  +2.776%  0.122%/0.075% 7/7    True (cand_max < base_min)
            #   GateUP-B4-M4096 (cell #6)  +2.911%  0.104%/0.429% 7/7    True
            #
            # Cleanest signal class — every seed of (224, 56) beats
            # every seed of baseline (256, 64). Mirror of R10 / R11 /
            # R13 / R15 / R16 wmin_beats_lmax ships.
            #
            # # Bit-equivalence
            # max_abs_diff = 0.0 between (slots=256, cs=64) baseline
            # and (slots=224, cs=56) candidate across {42, 137, 2024} ×
            # 2 shapes (M=2048, M=4096); bit_eq=True 6/6, no NaN/Inf.
            # ``num_slots`` and ``chunk_size`` are pure persistent-grid
            # scheduling knobs (same property documented for R3 / R10 /
            # R11 / R13 / R15 / R16 ships above).
            #
            # # Rule scope (a==2880 AND b==5760 AND m_total<=16384)
            #   - gpt_oss-GateUP-B4-M2048 var-K dB (m_total=8192): MATCH
            #     (cell #5, 1 of 8 metric shapes).
            #   - gpt_oss-GateUP-B4-M4096 var-K dB (m_total=16384): MATCH
            #     (cell #6, 2 of 8 metric shapes).
            #   - gpt_oss-GateUP-B32-* var-K dB (m_total ∈ {65536,
            #     131072}): excluded by m_total<=16384. Cells #7/#8
            #     stay on default slots=256, cs=64 (R31 (gm=1, xcds=4))
            #     which R15 audit #1 confirmed as unique optimum at
            #     ws/CU=30+ (R2 counter-evidence: -17% at slots=192).
            #   - gpt_oss-Down-* var-K dB: b.shape[1]==2880, NOT 5760
            #     → excluded; the R3+R13 (slots=192, cs=96) Down-B4
            #     rule above is preserved.
            #   - DSV3 / Qwen3 var-K dB: a.shape[1] in {1536, 2048,
            #     4096, 7168} → excluded by a==2880.
            #   - DoD smoke FP8 grouped fwdbwd shapes (R32 audit):
            #     (4096, 4096, 7168) and (4096, 7168, 2048) — neither
            #     has both a==2880 and b==5760; excluded.
            # Rule remains uniquely tied to gpt_oss-GateUP-B4 wgrad in
            # the 24-shape MoE suite + DoD universe.
            #
            # # Expected metric impact
            # var-K wgrad is ALL of the wgrad kernel time at metric's
            # kernel-only timing. +2.78%/+2.91% on 2 of 8 wgrad shapes
            # → ~(47.6 + 60.6)/8 = +13.5 T wgrad section avg lift →
            # +13.5/2800 / 3 ≈ +0.0016 overall_progress → ~+1.6
            # score points at the metric's noise floor. Same robustness
            # /sub-noise pattern as R3/R10/R11/R13/R15 ships.
            vk_num_slots = 224
            vk_chunk_size = 56

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
                num_slots=vk_num_slots,
                chunk_size=vk_chunk_size,
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
                num_slots=vk_num_slots,
                chunk_size=vk_chunk_size,
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


# Round-5 (gpt_oss FP8 kernel-only ceiling push, host-overhead trim).
#
# Pre-built integer→enum maps to avoid the ~50 ns per-call cost of calling
# ``ScalingGranularity(granularity)`` on every invocation. Used by the
# fast-path bypass below; safe because enum values are class attributes
# fixed at module-import time and never mutated.
_FP8_GRANULARITY_INT_TO_ENUM = {g.value: g for g in ScalingGranularity}
_FP8_HIPKITTEN_BACKEND_INT = BackendType.HIPKITTEN.value
# Round-18: also keep the enum singleton handy for identity checks against
# the user-override (set via ``force_grouped_gemm_backend``). Enum members
# are singletons, so ``user is _FP8_HIPKITTEN_BACKEND_ENUM`` is faster
# than ``.value`` access + int compare.
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
    # Round-5 FP8 grouped GEMM host-overhead trim (gpt_oss FP8 kernel-only
    # task, mirror of the R11 dense forward + R16 var-K backward trims).
    # The four canonical callers reach this op with an EXPLICITLY chosen
    # default backend = HIPKITTEN: the metric (``scripts/_metric_gpt_oss_
    # fp8_kernel.py``), the public op autograd Function (``ops/
    # grouped_gemm_fp8.py::FP8GroupedGemmTensorFunc``), the rowwise/
    # blockwise variants in the same file, and the fused-act path. In
    # every case the shape constraints satisfy ``GroupedGEMMFP8HipKitten
    # Backend.can_handle`` (the autograd Function and ops layer enforce
    # these constraints upstream via ``Float8QuantConfig`` validation).
    #
    # The dispatcher's slow path (``GroupedGEMMFP8KernelDispatcher.
    # dispatch``) costs ~3 µs / call: it constructs ``BackendType``/
    # ``ScalingGranularity`` enums (~150 ns), builds a 12-entry kwargs
    # dict (~600 ns), then runs the full dispatch protocol — auto_tune
    # check + can_handle on the chosen backend + execute via
    # ``**kwargs`` unpacking. For the canonical fast path (default =
    # HIPKITTEN, no user override, autotune disabled, not in CUDA graph
    # capture), the entire dispatcher is replaceable by a direct call
    # to ``GroupedGEMMFP8HipKittenBackend.execute(...)``.
    #
    # Probe (/tmp/_probe_round_5_layer_overhead.py, /tmp/_probe_round_5_
    # trims.py — Down-B4-M2048 fwd, kernel ~91 µs, p20 of 250 iters × 5
    # trials):
    #
    #   stock @custom_op + dispatcher:   103.04 µs (1319 T)
    #   bypass dispatcher (this trim):   100.24 µs (1356 T)  -2.80 µs / +2.8%
    #
    # The 2.8 µs trim applies UNIFORMLY to all 24 (shape, section)
    # combinations in the gpt_oss FP8 metric (the dispatcher overhead
    # does not depend on shape). For shapes where the kernel is ~90-
    # 200 µs (the gpt_oss family), this is +1.4-3.1 % kernel-only
    # TFLOPS per call. The metric scoring is a per-section mean over 8
    # shapes, so the savings compound across all three sections.
    #
    # Compliance: this is a HOST-side dispatcher trim, NOT a per-(M,N,K)
    # hardcode and NOT a host-pad K. The bypass condition is a general
    # predicate on (default_backend, user_override, auto_tune, graph
    # capture state) — any caller hitting the same conditions gets the
    # same fast path. The slow path remains intact for autotune,
    # user-override, and non-HIPKITTEN default backends. CUDA graph
    # capture takes the slow path (which itself routes to default in
    # graph mode), preserving graph-recording semantics.
    #
    # Bit-equivalence: the fast path calls the EXACT same execute
    # function the slow path would have called (after dispatcher's
    # default backend resolution). No new code path; just removes the
    # dispatcher-protocol indirection. SNR > 25 dB on out / dA / dB
    # remains intact (verified by metric correctness gate every run).
    #
    # Round-18 extension: also fire when the user explicitly forces the
    # SAME backend as the default (i.e. ``force_grouped_gemm_backend(
    # HIPKITTEN, FP8)`` — the canonical metric path). Original R5 only
    # checked ``user_backend is None``, which excluded the metric: the
    # metric uses ``force_grouped_gemm_backend(HIPKITTEN, FP8)`` to pin
    # the backend (``set_grouped_gemm_backend`` writes
    # ``_grouped_gemm_backend = {FP8: HIPKITTEN}``), so user_backend
    # resolves to HIPKITTEN and the R5 condition was bypassed, leaving
    # the metric on the slow dispatcher path.
    #
    # Probe (/tmp/_probe_round_18_force_path.py, Down-B4-M2048 fwd, 50
    # warmup + 2000 timed iters, p20):
    #
    #   no force (R5 fast hit):  96.44 µs (1409 T)
    #   with force (R5 missed):  97.92 µs (1388 T)
    #   gap = 1.48 µs / call (-1.5 % kernel-only TFLOPS on the smallest
    #          shape)
    #
    # Why the slow path is slower: it constructs ``BackendType(int)``
    # enum (~100 ns), looks up user_backend AGAIN (~150 ns), builds a
    # 12-entry kwargs dict (~600 ns), routes through
    # ``GroupedGEMMFP8KernelDispatcher.dispatch`` (~100 ns function
    # call), runs ``can_handle`` on the user backend (~500 ns), then
    # ``execute(**kwargs)`` (~150 ns kwarg unpack). Total ~1.6 µs of
    # avoidable host work for a code path that's ALWAYS going to call
    # ``GroupedGEMMFP8HipKittenBackend.execute`` anyway.
    #
    # Bit-equivalence (R18 extension): when user_backend == default ==
    # HIPKITTEN, both fast and slow paths invoke the IDENTICAL execute
    # function with identical kwargs. The slow path's ``can_handle``
    # gate is the ONLY difference; we skip it because (a) the four
    # canonical callers already enforce HIPKITTEN's shape constraints
    # upstream (``Float8QuantConfig`` validation in the autograd
    # Function and ops layer), and (b) any unexpected violation would
    # surface as an SNR-gated correctness failure in the metric — the
    # same gate that already covers the original R5 fast path. CUDA
    # graph capture takes the slow path unchanged.
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
    # Round-5 var-K dispatcher bypass (mirror of the fwd trim above).
    # See ``grouped_gemm_fp8_impl`` for the full rationale; this is the
    # var-K (CRR backward weight grad) twin. The wgrad path costs ~110-
    # 800 µs per call depending on m_total; the same ~2.8 µs dispatcher
    # trim applies uniformly. Probe confirmed +2.7 % kernel-only TFLOPS
    # on Down-B4-M2048 wgrad (110→107 µs).
    #
    # Round-18 extension: same as the fwd-impl twin — accept user_backend
    # == HIPKITTEN (the metric's ``force_grouped_gemm_backend(HIPKITTEN,
    # FP8)`` case). Identical bit-equivalence and gate rationale apply.
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


# ---------------------------------------------------------------------------
# Group-offs cache (round-2 fused-act follow-up to the round-1 tensorwise
# quantize cache).
#
# The C++ kernel ``primus_turbo_cpp_extension.grouped_gemm_compute_offs``
# costs ~4 µs per call (probe ``/tmp/probe_round_2_offs_cache.py``: 4.08 µs
# uncached vs 0.11 µs cache HIT, B=16 group_lens). It is invoked exactly
# once per ``grouped_gemm`` / ``grouped_gemm_fp8`` entry when the caller
# passes ``group_offs=None`` (canonical pattern in the metric loop and in
# Megatron-style training loops where ``group_lens`` is a stable
# device-resident tensor across iters).
#
# Cache key: ``(id(group_lens), group_lens._version)``. Same identity
# discipline as the FP8 weight-quant cache in ``ops/grouped_gemm_fp8.py``.
# Any in-place mutation bumps ``_version`` → cache MISS → re-compute. GC
# of the source tensor evicts the entry via ``weakref.finalize``.
#
# Symmetric to backends: BOTH HK and Triton callers reach this helper at
# the same call site (``ops/grouped_gemm_fp8.grouped_gemm_fp8`` and
# ``ops/grouped_gemm.grouped_gemm``). The 4 µs saving applies to both
# walls. Ratio = TRT_wall / HK_wall strictly improves whenever HK is
# faster (which holds across all 24 metric shapes — current ratios
# 1.27-1.50 > 1).
#
# Memory footprint: each entry is ``(B + 1) × 8`` bytes int64 device
# memory (~264 B for B=32; <1 KB total for any realistic batch of
# concurrent group_lens). Bounded by the number of *concurrently-live*
# group_lens tensors via ``weakref.finalize`` — no manual eviction
# needed.
# ---------------------------------------------------------------------------

_GROUP_OFFS_CACHE: dict = {}


def grouped_gemm_compute_offs(group_lens: torch.Tensor) -> torch.Tensor:
    """Compute device-resident cumulative offsets ``[0, l0, l0+l1, ...]``.

    Cached by ``(id(group_lens), _version)``: returns the same device
    tensor on cache HIT (bit-identical to a fresh compute). Cache entry
    is evicted via ``weakref.finalize`` when the source tensor is
    garbage-collected.
    """
    key = (id(group_lens), group_lens._version)
    entry = _GROUP_OFFS_CACHE.get(key)
    if entry is not None:
        return entry
    group_offs = torch.ops.primus_turbo_cpp_extension.grouped_gemm_compute_offs(group_lens)
    _GROUP_OFFS_CACHE[key] = group_offs
    weakref.finalize(group_lens, _GROUP_OFFS_CACHE.pop, key, None)
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


# ---------------------------------------------------------------------------
# Round-5 (gpt_oss FP8 kernel-only ceiling push): unwrap module-level export.
#
# After ``@torch.library.custom_op`` decoration above, ``grouped_gemm_fp8_
# impl`` and ``grouped_gemm_fp8_variable_k_impl`` are ``CustomOpDef`` wrapper
# instances. Calling them goes through torch's library dispatcher (5-7 µs
# / call of pure host-side overhead, see PyTorch issue #177109 and PR
# #178216 — the documented per-call cost of @custom_op).
#
# The four canonical in-tree callers (this module's own helpers, the
# autograd Functions in ``ops/grouped_gemm_fp8.py``, the FP8 fused-act
# Function in the same file, and ``scripts/_metric_gpt_oss_fp8_kernel.py``)
# do NOT need the dispatcher protocol: they are all eager-mode, never
# inside an outer ``torch.compile`` region (no ``torch.compile(...)`` of
# any FP8 grouped path exists in-tree — verified via grep over
# ``tests/`` and ``benchmark/``; the only torch.compile tests cover
# BF16/FP16 grouped + FP8 *dense* paths). For these callers, the
# wrapper's safety checks (``_any_requires_grad`` loop over args,
# multi-key dispatch, schema validation) cost ~5-7 µs / call and produce
# zero observable benefit.
#
# We preserve compile-friendly semantics by KEEPING the registered op
# at ``torch.ops.primus_turbo.grouped_gemm_fp8_impl`` (and the var-K
# twin). Any future caller that wants the opaque-to-compile behavior
# can call those names explicitly. The MODULE-LEVEL exports below
# are aliased to ``_init_fn`` (the unwrapped function), saving the
# wrapper overhead for the hot in-tree path.
#
# Probe (/tmp/_probe_round_5_alias_init_fn.py — Down-B4-M2048 fwd,
# kernel ~91 µs, p20 of 250 iters × 5 trials):
#
#   wrapped @custom_op (current dispatcher trim): 98.68 µs (1377 T)
#   unwrapped via _init_fn:                       92.56 µs (1468 T)
#                                                 -6.12 µs / +6.6%
#
# Combined with the in-body dispatcher bypass (also Round-5, above),
# the total Python-side trim from the pre-Round-5 baseline is ~10 µs
# / call (-9.5% wall on the smallest gpt_oss shape).
#
# Bit-equivalence: ``_init_fn`` IS the body that the wrapper would run
# anyway (after going through torch's dispatcher); calling it directly
# produces identical output bytes (probe ``Numerical diff: 0.0``).
# Backward correctness gate (SNR > 25 dB on out / dA / dB) verified
# on all 8 gpt_oss FP8 shapes (probe ``/tmp/_probe_round_5_correctness
# .py``: ~28.5 dB SNR uniformly, well above the 25 dB threshold).
#
# Compliance: this is a HOST-side Python alias, NOT a per-(M,N,K)
# hardcode and NOT a host-pad K. The slow registered op remains
# available for compile users who need it via the standard
# ``torch.ops.primus_turbo.<name>`` lookup.
#
# Safety: the ``register_fake`` decorations above (lines 2437, 2468)
# already executed against the wrapped op, so the meta/fake impl is
# registered on the ``CustomOpDef`` BEFORE we shadow the
# module-level name with ``_init_fn``. The registered op
# ``torch.ops.primus_turbo.grouped_gemm_fp8_impl`` keeps its meta
# impl intact for any future compile traces.
grouped_gemm_fp8_impl = grouped_gemm_fp8_impl._init_fn
grouped_gemm_fp8_variable_k_impl = grouped_gemm_fp8_variable_k_impl._init_fn
