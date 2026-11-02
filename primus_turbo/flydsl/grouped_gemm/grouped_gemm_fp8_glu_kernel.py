###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""FlyDSL fp8 per-tensor grouped GEMM with the SwiGLU activation fused in.

Two entries: the fc1 forward, which emits the pre-activation ``l1`` and the
activation together, and the fc2 dgrad, which consumes ``dact`` in registers so
it never reaches HBM. Both use the shared NT/NN kernel factories in
``grouped_gemm_fp8_kernel`` (``glu`` / ``dglu`` flags); this module holds the
entry layer and its swizzle race.
"""

from typing import NamedTuple

import torch

from primus_turbo.flydsl.autotune import Compile, Op, Race, Refine, staged
from primus_turbo.flydsl.autotune.engine import Entry
from primus_turbo.flydsl.autotune.policies import Point, Synthetic
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8_kernel import (
    _GG_SCHED_HINTS,
    _GROUPED_AGPR,
    _balanced_group_offs,
    _compile_grouped_nn,
    _compile_grouped_nt,
)
from primus_turbo.flydsl.utils.gemm_epilogue_helper import (
    AMAX_PARTIAL_SLOTS,
    _check_activation,
    _check_clamp_limit,
)

# Swizzle arms per direction; each wins at some measured shape (group-major (1, 4) loses 3-20%
# everywhere).
_GLU_NT_CANDS = ((4, 4), (4, 8), (8, 8), (4, 0))
_GLU_NN_CANDS = ((4, 4), (8, 4), (8, 8), (4, 8))


def _glu_probe_m(G: int) -> int:
    """Rows for the race's synthetic probe. One M only: the M sensitivity is under 1.5%."""
    pm = max(1024, -(-32768 // max(G, 1)))  # >= 32768 rows, so the grid fills
    return G * pm


# Swizzle race for the fused kernels: timed on a balanced synthetic distribution so the pick
# depends only on the static shape, each arm checked numerically. (4, 4) is within 0.4% at the
# production MLP widths but up to 9% off elsewhere. The base arm is the fallback when others fail
# to build. Short warmup: these launches are ~100x longer than the plain race's short-K ones.
# The probes (up to 1 GB a side) go back to the device once the entry is built (``Synthetic``);
# left in the caching allocator they make the plain GEMM's later race in the same process mis-pick
# an arm that stays 2x slow.
# The 0.985 margin sits past the noise.
_GLU_RACE = Race.interleaved(1, 20, warmup=40, visit=(0, 3), margin=0.985, check="match")
# The fused entries launch raw, under the schedule hints their race traced them with.
_HINTED_RAW = Compile(raw=True, hints=_GG_SCHED_HINTS)
_GROUPED_GEMM_FP8_TENSORWISE_EPI_GLU = Op("grouped_gemm_fp8_tensorwise_epi_glu", compile=_HINTED_RAW)
_GROUPED_GEMM_FP8_TENSORWISE_EPI_DGLU = Op("grouped_gemm_fp8_tensorwise_epi_dglu", compile=_HINTED_RAW)


def _glu_entry(cfg, key, tuned=True):
    """The fused fc1 launch at ``cfg`` = (num_xcd, group_m) for ``key``'s static shape."""
    I, K, G, out_fp16, cbsz, blgp, num_cu, _amax, activation, clamp_limit = key
    _xcd, _gm = cfg
    _capped = num_cu > 0
    launch = _compile_grouped_nt(
        K=K,
        G=G,
        BLOCK_M=256,
        BLOCK_N=256,
        nt_vmcnt=3,
        # VGPR accumulator (mma mode 3): the SwiGLU epilogue reads it as a
        # VALU source without v_accvgpr_read shuffles.
        acc_mode="vgpr",
        # Raced over _GLU_NT_CANDS: the plain row-major default loses
        # 0.66 ms of L2 reuse at I=5760 (8.2 GB extra HBM reads).
        num_xcd=_xcd,
        agpr_inplace=_GROUPED_AGPR,
        out_fp16=out_fp16,
        cbsz=cbsz,
        blgp=blgp,
        group_m=_gm,
        # CShuffle stages one fragment through LDS; the gate/up pair store needs the scalar path.
        store_cshuffle=False,
        sched_schedbar=True,
        # One tile per WG unless CUs are capped: the scf.for tile loop costs ~0.6 ms here.
        persistent=_capped,
        cap_cu=(num_cu if _capped else -1),
        N=I,
        glu=True,
        glu_i=I,
        # Non-temporal outputs (1.11x): keeps 4.4 GB of write-once data from
        # evicting the A/B tiles. aux=16 instead costs 1.5x.
        cstore_aux=2,
        glu_act_aux=2,
        glu_amax=_amax,
        activation=activation,
        clamp_limit=clamp_limit,
    )
    return Entry(cfg, launch, tuned=tuned)


def _glu_probe(key, a, b, a_scale, b_scale, out_dtype):
    """One balanced synthetic fc1 on the caller's weight; none when it would not fit, which
    keeps the static arm."""
    I, K, G, _out_fp16, _cbsz, _blgp, _num_cu, _amax, _activation, _clamp_limit = key

    def points():
        M_c = _glu_probe_m(G)
        try:
            a_c = torch.empty((M_c, K), device=a.device, dtype=torch.uint8)
            a_c.random_(0, 64, generator=torch.Generator(device=a.device).manual_seed(0))
            l1_c = torch.empty((M_c, 2 * I), device=a.device, dtype=out_dtype)
            act_c = torch.empty((M_c, I), device=a.device, dtype=out_dtype)
            probs_c = torch.ones(M_c, device=a.device, dtype=torch.float32)
            amax_c = torch.zeros(AMAX_PARTIAL_SLOTS, device=a.device, dtype=torch.float32) if _amax else act_c
        except torch.cuda.OutOfMemoryError:
            return None
        args = (
            a_c.view(torch.int8),
            b.view(torch.int8),
            l1_c,
            act_c,
            probs_c,
            amax_c,
            a_scale.float().reshape(1),
            b_scale.float().reshape(1),
            _balanced_group_offs(M_c, G, a.device),
            M_c,
            I,
            torch.cuda.current_stream(),
        )
        return [Point(args, act_c)]

    return Synthetic(points)


def _pick_glu(key, a, b, a_scale, b_scale, out_dtype):
    space = staged(Refine(lambda _prev: list(_GLU_NT_CANDS), _GLU_RACE), base=_GLU_NT_CANDS[0])
    build = _GROUPED_GEMM_FP8_TENSORWISE_EPI_GLU.build(lambda cfg: _glu_entry(cfg, key))
    res = _GROUPED_GEMM_FP8_TENSORWISE_EPI_GLU.pick(
        key, space, probe=_glu_probe(key, a, b, a_scale, b_scale, out_dtype), build=build
    )
    return build.winner(res)


def _dglu_entry(cfg, key, tuned=True):
    """The fused fc2-dgrad launch at ``cfg`` = (num_xcd, group_m) for ``key``'s static shape."""
    I, K, G, out_fp16, cbsz, blgp, num_cu, n_stride, _amax, activation, clamp_limit = key
    _xcd, _gm = cfg
    _capped = num_cu > 0
    launch = _compile_grouped_nn(
        K=K,
        G=G,
        BLOCK_M=256,
        BLOCK_N=_DGLU_BLOCK_N,
        nt_vmcnt=3,
        # Raced over _GLU_NN_CANDS (spread up to 1.09x at I=1024).
        num_xcd=_xcd,
        group_m=_gm,
        agpr_inplace=_GROUPED_AGPR,
        out_fp16=out_fp16,
        cbsz=cbsz,
        blgp=blgp,
        store_cshuffle=False,
        sched_schedbar=False,
        persistent=_capped,
        cap_cu=(num_cu if _capped else -1),
        N=I,
        n_stride=n_stride,
        dglu=True,
        dglu_amax=_amax,
        glu_i=I,
        # Non-temporal pays only because the banded epilogue stores whole
        # lines; half-line stores must merge in L2, which non-temporal prevents.
        cstore_aux=2,
        activation=activation,
        clamp_limit=clamp_limit,
    )
    return Entry(cfg, launch, tuned=tuned)


def _dglu_probe(key, a, b, a_scale, b_scale, out_dtype, n_partial):
    """One balanced synthetic fc2 dgrad on the caller's weight; none when it would not fit."""
    I, K, G, _out_fp16, _cbsz, _blgp, _num_cu, _n_stride, _amax, _activation, _clamp_limit = key

    def points():
        M_c = _glu_probe_m(G)
        try:
            a_c = torch.empty((M_c, K), device=a.device, dtype=torch.uint8)
            a_c.random_(0, 64, generator=torch.Generator(device=a.device).manual_seed(0))
            # Not zeros: all-zero gradients would make the numeric check vacuous.
            l1_c = torch.empty((M_c, 2 * I), device=a.device, dtype=out_dtype)
            l1_c.uniform_(-2.0, 2.0)
            dl1_c = torch.empty((M_c, 2 * I), device=a.device, dtype=out_dtype)
            probs_c = torch.ones(M_c, device=a.device, dtype=torch.float32)
            grad_probs_c = torch.zeros((n_partial, M_c), device=a.device, dtype=torch.float32)
            amax_c = torch.zeros(AMAX_PARTIAL_SLOTS, device=a.device, dtype=torch.float32) if _amax else dl1_c
        except torch.cuda.OutOfMemoryError:
            return None
        args = (
            a_c.view(torch.int8),
            b.view(torch.int8),
            dl1_c,
            a_scale.float().reshape(1),
            b_scale.float().reshape(1),
            _balanced_group_offs(M_c, G, a.device),
            l1_c,
            probs_c,
            grad_probs_c,
            amax_c,
            M_c,
            I,
            M_c,
            torch.cuda.current_stream(),
        )
        return [Point(args, dl1_c)]

    return Synthetic(points)


def _pick_dglu(key, a, b, a_scale, b_scale, out_dtype, n_partial):
    space = staged(Refine(lambda _prev: list(_GLU_NN_CANDS), _GLU_RACE), base=_GLU_NN_CANDS[0])
    build = _GROUPED_GEMM_FP8_TENSORWISE_EPI_DGLU.build(lambda cfg: _dglu_entry(cfg, key))
    res = _GROUPED_GEMM_FP8_TENSORWISE_EPI_DGLU.pick(
        key, space, probe=_dglu_probe(key, a, b, a_scale, b_scale, out_dtype, n_partial), build=build
    )
    return build.winner(res)


def grouped_gemm_fp8_tensorwise_epi_glu_flydsl_kernel(
    a: "torch.Tensor",
    b: "torch.Tensor",
    a_scale: "torch.Tensor",
    b_scale: "torch.Tensor",
    probs: "torch.Tensor",
    group_offs: "torch.Tensor",
    act_out: "torch.Tensor",
    intermediate_out: "torch.Tensor",
    trans_b: bool = False,
    *,
    activation: str = "silu",
    clamp_limit: "float | None" = None,
    out_dtype=torch.bfloat16,
    num_cu: "int | None" = None,
    amax_partial: "torch.Tensor | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """FlyDSL fc1 grouped fp8 GEMM with a fused GLU epilogue, matching the Triton entry.

    Computes ``l1 = [gate|up] = (a[g] @ b[g]^T) * a_scale * b_scale`` [M, 2I] and
    ``act = f(gate) * up * probs`` [M, I] in one launch, both in ``out_dtype``.
    ``l1`` is kept for backward's dswiglu; only ``act`` carries ``probs``.

    The tile's second B pool points at the up half of the weight, so gate and up
    share a lane without a permute (see ``StoreCSwiGLU``). NT only
    (``trans_b=True``, b [G, 2I, K]).

    Args:
        clamp_limit: pre-multiplication clamp bound, or None for no clamp; silu only.
            See :func:`~primus_turbo.flydsl.utils.gemm_epilogue_helper._glu_clamp`.
        act_out: [M_total, I] buffer receiving the activation.
        intermediate_out: [M_total, 2I] buffer receiving ``l1``. Both are fully
            written, so neither needs initialising.
        amax_partial: optional zeroed float32 [``AMAX_PARTIAL_SLOTS``] receiving
            ``act``'s abs-max (exact), saving the tensorwise quantiser a read of ``act``.

    Returns:
        ``(act_out, intermediate_out)``.
    """
    _check_activation(activation)
    _check_clamp_limit(clamp_limit)
    assert trans_b, "FlyDSL fused GLU is NT only: pass b as [G, 2I, K]"
    assert a.ndim == 2 and b.ndim == 3
    M_total, K = a.shape
    G = b.shape[0]
    N2, K_b = b.shape[1], b.shape[2]
    assert K == K_b, f"K mismatch a={K} b={K_b}"
    assert N2 % 2 == 0, f"fc1 width must be even (gate||up), got {N2}"
    I = N2 // 2
    assert probs.ndim == 1 and probs.shape[0] == M_total and probs.dtype == torch.float32

    act, l1 = act_out, intermediate_out
    _amax = amax_partial is not None
    if _amax:
        assert amax_partial.shape == (AMAX_PARTIAL_SLOTS,) and amax_partial.dtype == torch.float32, (
            f"amax_partial must be [{AMAX_PARTIAL_SLOTS}] float32, got "
            f"{list(amax_partial.shape)} {amax_partial.dtype}"
        )
    assert act.shape == (M_total, I) and act.dtype == out_dtype and act.device == a.device
    assert l1.shape == (M_total, N2) and l1.dtype == out_dtype and l1.device == a.device

    _go64 = group_offs if group_offs.dtype == torch.int64 else group_offs.to(torch.int64)
    go32 = _go64.view(torch.int32)
    out_fp16 = out_dtype == torch.float16
    cbsz = 1 if a.dtype == torch.float8_e5m2 else 0
    blgp = 1 if b.dtype == torch.float8_e5m2 else 0
    _capped = num_cu is not None and num_cu > 0
    key = (
        I,
        K,
        G,
        out_fp16,
        cbsz,
        blgp,
        num_cu if _capped else 0,
        _amax,
        activation,
        clamp_limit,
    )

    args = (
        a.view(torch.int8),
        b.view(torch.int8),
        l1,
        act,
        probs,
        amax_partial if _amax else act,
        a_scale.float().reshape(1),
        b_scale.float().reshape(1),
        go32,
        M_total,
        I,  # c_n is the gate width; the kernel derives 2I where it needs it
        torch.cuda.current_stream(),
    )
    entry = _GROUPED_GEMM_FP8_TENSORWISE_EPI_GLU.entry(
        key, lambda: _pick_glu(key, a, b, a_scale, b_scale, out_dtype)
    )
    _GROUPED_GEMM_FP8_TENSORWISE_EPI_GLU.launch(entry, args)
    return act, l1


_DGLU_BLOCK_N = 256


class GradProbsPartialSpec(NamedTuple):
    """The grad_probs partial buffer a fused dgrad expects from its caller."""

    shape: "tuple[int, int]"
    needs_zero: bool


def grouped_gemm_fp8_dglu_grad_probs_partial_spec(
    a: "torch.Tensor", b: "torch.Tensor", i_real: "int | None" = None
) -> GradProbsPartialSpec:
    """The buffer :func:`grouped_gemm_fp8_tensorwise_epi_dglu_flydsl_kernel` expects.

    ``grad_probs_partial.sum(0)`` is the gradient wrt ``probs``; the caller
    allocates and folds it.

    One slice per ``(block_n, 16-lane half)``: waves' rows are disjoint in the
    banded epilogue, so they share a slice.

    ``needs_zero`` is True: a group's last M-tile is clamped at ``m_end``, so rows
    past it are never written and would otherwise poison the fold.

    Returns:
        ``shape`` is ``(n_blocks * 2, M_total)``, float32.
    """
    M_total = a.shape[0]
    # The reduction spans the real I, not the padded storage pitch b.shape[2].
    I = i_real if i_real is not None else b.shape[2]
    n_blocks = -(-I // _DGLU_BLOCK_N)
    return GradProbsPartialSpec(shape=(n_blocks * 2, M_total), needs_zero=True)


def grouped_gemm_fp8_tensorwise_epi_dglu_flydsl_kernel(
    a: "torch.Tensor",
    b: "torch.Tensor",
    a_scale: "torch.Tensor",
    b_scale: "torch.Tensor",
    intermediate: "torch.Tensor",
    group_offs: "torch.Tensor",
    probs: "torch.Tensor",
    out: "torch.Tensor",
    grad_probs_partial: "torch.Tensor",
    trans_b: bool = False,
    *,
    activation: str = "silu",
    clamp_limit: "float | None" = None,
    num_cu: "int | None" = None,
    i_real: "int | None" = None,
    amax_partial: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """FlyDSL fc2 dgrad with the GLU gradient fused into its epilogue.

    Computes ``dact = (a[g] @ b[g]) * a_scale * b_scale`` and consumes it in
    registers, so ``dact`` never reaches HBM:

        dl1 = [f'(gate) * up * (dact * probs) | f(gate) * (dact * probs)]

    Gate and up are read I columns apart from ``intermediate``. ``grad_probs``
    spans all of I, so each wave writes a partial over its columns for the caller
    to sum; no atomics, so the result is bitwise reproducible.

    NN only (``trans_b=False``, b [G, K, I]).

    Args:
        clamp_limit: pre-multiplication clamp bound, or None for no clamp; silu only.
            Must match the forward's.
        out: [M_total, 2I] buffer receiving ``dl1``, in ``intermediate``'s dtype,
            gate gradient in [:, :I] and up in [:, I:].
        grad_probs_partial: zeroed float32 buffer shaped by
            :func:`grouped_gemm_fp8_dglu_grad_probs_partial_spec`.
        amax_partial: optional zeroed float32 [``AMAX_PARTIAL_SLOTS``] receiving
            ``dl1``'s abs-max (exact), saving the tensorwise quantiser a read of ``dl1``.

    Returns:
        ``out``, for call-site convenience.
    """
    _check_activation(activation)
    _check_clamp_limit(clamp_limit)
    assert not trans_b, "FlyDSL fused dGLU is NN only: pass b as [G, K, I]"
    assert a.ndim == 2 and b.ndim == 3 and intermediate.ndim == 2
    M_total, K = a.shape
    G, K_b, N_pitch = b.shape
    # b (w2) may be padded on I to N_pitch; B is read at that stride (n_stride),
    # everything else uses the real I (i_real).
    if i_real is not None and i_real != N_pitch:
        n_stride, I = N_pitch, i_real
    else:
        n_stride, I = 0, N_pitch
    assert K == K_b, f"K mismatch a={K} b={K_b}"
    assert intermediate.shape == (M_total, 2 * I), (
        f"intermediate must be [{M_total}, {2 * I}], got {tuple(intermediate.shape)}"
    )
    assert probs.ndim == 1 and probs.shape[0] == M_total and probs.dtype == torch.float32

    out_dtype = intermediate.dtype
    assert out.shape == (M_total, 2 * I) and out.dtype == out_dtype
    partial_shape = grouped_gemm_fp8_dglu_grad_probs_partial_spec(a, b, i_real=I).shape
    assert grad_probs_partial.shape == partial_shape and grad_probs_partial.dtype == torch.float32, (
        f"grad_probs_partial must be {list(partial_shape)} float32, got "
        f"{list(grad_probs_partial.shape)} {grad_probs_partial.dtype}; "
        "size it with grouped_gemm_fp8_dglu_grad_probs_partial_spec"
    )
    dl1 = out
    _amax = amax_partial is not None
    if _amax:
        assert amax_partial.shape == (AMAX_PARTIAL_SLOTS,) and amax_partial.dtype == torch.float32, (
            f"amax_partial must be [{AMAX_PARTIAL_SLOTS}] float32, got "
            f"{list(amax_partial.shape)} {amax_partial.dtype}"
        )

    _go64 = group_offs if group_offs.dtype == torch.int64 else group_offs.to(torch.int64)
    go32 = _go64.view(torch.int32)
    out_fp16 = out_dtype == torch.float16
    cbsz = 1 if a.dtype == torch.float8_e5m2 else 0
    blgp = 1 if b.dtype == torch.float8_e5m2 else 0
    _capped = num_cu is not None and num_cu > 0
    key = (
        I,
        K,
        G,
        out_fp16,
        cbsz,
        blgp,
        num_cu if _capped else 0,
        n_stride,
        _amax,
        activation,
        clamp_limit,
    )

    args = (
        a.view(torch.int8),
        b.view(torch.int8),
        dl1,
        a_scale.float().reshape(1),
        b_scale.float().reshape(1),
        go32,
        intermediate,
        probs,
        grad_probs_partial,
        amax_partial if _amax else dl1,
        M_total,
        I,
        M_total,
        torch.cuda.current_stream(),
    )
    entry = _GROUPED_GEMM_FP8_TENSORWISE_EPI_DGLU.entry(
        key, lambda: _pick_dglu(key, a, b, a_scale, b_scale, out_dtype, partial_shape[0])
    )
    _GROUPED_GEMM_FP8_TENSORWISE_EPI_DGLU.launch(entry, args)
    return dl1
