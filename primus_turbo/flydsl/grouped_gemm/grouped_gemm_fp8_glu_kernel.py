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
it never reaches HBM. Both go through the shared NT/NN kernel factories in
``grouped_gemm_fp8_kernel``, which carry the fused epilogue behind their ``glu``
and ``dglu`` flags -- the fusion reuses those mainloops rather than forking them,
so what lives here is the entry layer and the swizzle race it needs.
"""

from typing import NamedTuple

import torch
from flydsl.compiler.kernel_function import CompilationContext

from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8_kernel import (
    _GG_SCHED_HINTS,
    _GROUPED_AGPR,
    _balanced_group_offs,
    _compile_grouped_nn,
    _compile_grouped_nt,
)

_GROUPED_GLU_CACHE: dict = {}

# Swizzle arms for the fused entries, per direction. Each of these wins at some
# measured shape; the ones left out are dominated everywhere (group-major (1, 4)
# is 3-20% down at every shape).
_GLU_NT_CANDS = ((4, 4), (4, 8), (8, 8), (4, 0))
_GLU_NN_CANDS = ((4, 4), (8, 4), (8, 8), (4, 8))


def _glu_probe_m(G: int) -> int:
    """Rows for the race's synthetic probe: production scale, and enough to fill.

    One canonical M, unlike the plain path's two: what moves with M is under
    1.5%, not worth a second point in a race this slow per launch.
    """
    pm = max(1024, -(-32768 // max(G, 1)))  # >= 32768 rows, so the grid fills
    return G * pm


def _glu_race_time(launch, args, warmup=40, reps=3, iters=20):
    """Median-of-`reps`. Shorter warmup than ``_robust_time``: its 250 iters are
    for short-K kernels that mis-pick before boost clock, and these are two
    orders of magnitude longer per launch, so they get there in a few."""
    for _ in range(warmup):
        launch(*args)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            launch(*args)
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / iters)
    ts.sort()
    return ts[len(ts) // 2]


def _autotune_glu(build, cands, mk_probe):
    """Race the tile swizzle for a fused GLU kernel and return the winner.

    Same shape as the plain path's race, for the same reasons: time on a
    *balanced* synthetic distribution so the pick depends only on the static
    shape and not on whichever token skew the first real call happens to carry,
    and guard each arm numerically so a bad compile cannot win on speed alone.
    Only the swizzle is raced; an arm compiled unlike production would not
    transfer.

    Worth racing at all because the spread runs both ways: the hardcoded (4, 4)
    is within 0.4% at the two production MLP widths but up to 9% off elsewhere.
    """
    probe = mk_probe()
    if probe is None:  # probe would not fit; keep the static arm
        return build(*cands[0])
    args, out_view = probe

    base = build(*cands[0])
    base(*args)
    torch.cuda.synchronize()
    # Cloned: the other arms overwrite out_view, and .float() is a no-op view
    # when the output is already fp32.
    ref = out_view.detach().float().clone()
    refnorm = float((ref * ref).sum().item()) or 1.0

    def score(launch):
        launch(*args)
        torch.cuda.synchronize()
        o = out_view.detach().float()
        e = float(((o - ref) * (o - ref)).sum().item())
        if (e / refnorm) >= (2e-2**2) or not torch.isfinite(o.view(-1)[:1024]).all().item():
            return None
        return _glu_race_time(launch, args)

    best, bs = base, score(base)
    for cand in cands[1:]:
        # An arm that will not build is one arm lost, not a failed call: the base
        # is the config that shipped, so it is always there to fall back on.
        try:
            cl = build(*cand)
        except Exception:
            continue
        s = score(cl)
        if s is not None and bs is not None and s < bs * 0.985:  # past the noise margin
            best, bs = cl, s

    # Hand the allocator back the probe outright. The probes run to a gigabyte a
    # side, and leaving them in the caching allocator was measured to make the
    # *plain* GEMM's own race, running later in the same process, mis-pick an arm
    # and stay 2x slow for the rest of it.
    probe = args = out_view = ref = None
    torch.cuda.empty_cache()
    return best


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
    out_dtype=torch.bfloat16,
    num_cu: "int | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """FlyDSL fc1 grouped fp8 GEMM with a fused SwiGLU epilogue, matching the Triton entry.

    Computes ``l1 = [gate|up] = (a[g] @ b[g]^T) * a_scale * b_scale`` [M, 2I] and
    ``act = silu(gate) * up * probs`` [M, I] in one launch, both in ``out_dtype``.
    ``l1`` is written because backward's dswiglu needs both halves; ``act`` is the
    only one that carries ``probs``.

    Where the Triton twin computes a 2I-wide tile and permutes it to bring gate
    beside up (its epilogue's largest single cost), this points the tile's second
    B pool at the up half of the weight, so the pair already shares a lane. See
    ``StoreCSwiGLU``.

    NT only (``trans_b=True``, b [G, 2I, K]): the pairing lives in the NT body's
    quadrant registers, and the NN twin has no equivalent hook yet.

    Args:
        act_out: [M_total, I] buffer receiving the activation.
        intermediate_out: [M_total, 2I] buffer receiving ``l1``. Both are the
            caller's to allocate: every slot is written, so neither needs
            initialising.

    Returns:
        ``(act_out, intermediate_out)``.
    """
    assert activation == "silu", f"FlyDSL fused GLU implements silu only, got {activation}"
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
    assert act.shape == (M_total, I) and act.dtype == out_dtype and act.device == a.device
    assert l1.shape == (M_total, N2) and l1.dtype == out_dtype and l1.device == a.device

    _go64 = group_offs if group_offs.dtype == torch.int64 else group_offs.to(torch.int64)
    go32 = _go64.view(torch.int32)
    out_fp16 = out_dtype == torch.float16
    cbsz = 1 if a.dtype == torch.float8_e5m2 else 0
    blgp = 1 if b.dtype == torch.float8_e5m2 else 0
    _capped = num_cu is not None and num_cu > 0
    ckey = (I, K, G, out_fp16, cbsz, blgp, num_cu if _capped else 0)

    launch = _GROUPED_GLU_CACHE.get(ckey)
    if launch is None:

        def _build(_xcd, _gm):
            return _compile_grouped_nt(
                K=K,
                G=G,
                BLOCK_M=256,
                BLOCK_N=256,
                nt_vmcnt=3,
                # "vgpr" keeps the accumulator in VGPR (mma mode 3) so the fused
                # SwiGLU epilogue -- its only consumer -- reads it as a VALU source
                # without the v_accvgpr_read shuffle an AGPR accumulator would need.
                # The NT feed is S2RLoader, not transpose asm, so the register budget
                # is unchanged.
                acc_mode="vgpr",
                # Raced per shape over _GLU_NT_CANDS, not defaulted: taking the
                # plain entry's row-major default literally cost 0.66 ms of L2
                # reuse at I=5760 (8.2 GB of extra HBM reads).
                num_xcd=_xcd,
                agpr_inplace=_GROUPED_AGPR,
                out_fp16=out_fp16,
                cbsz=cbsz,
                blgp=blgp,
                group_m=_gm,
                # The pair store is the scalar path's; CShuffle stages one fragment
                # through LDS and has nowhere to put the second.
                store_cshuffle=False,
                sched_schedbar=True,
                # One tile per WG unless the caller reserves CUs, matching the
                # plain GEMM's fast path: the scf.for tile loop costs ~0.6 ms at
                # this shape against a straight-line one-tile body.
                persistent=_capped,
                cap_cu=(num_cu if _capped else -1),
                N=I,
                glu=True,
                glu_i=I,
                # Non-temporal on both output streams, worth 1.11x: 4.4 GB that
                # nothing here reads again would otherwise evict the A/B tiles the
                # next mainloop wants. Bit 1 is the one that pays -- bit 4
                # (aux=16) costs 1.5x, so this is not a monotone knob.
                cstore_aux=2,
                glu_act_aux=2,
            )

        def _probe():
            M_c = _glu_probe_m(G)
            try:
                a_c = torch.empty((M_c, K), device=a.device, dtype=torch.uint8)
                a_c.random_(0, 64, generator=torch.Generator(device=a.device).manual_seed(0))
                l1_c = torch.empty((M_c, N2), device=a.device, dtype=out_dtype)
                act_c = torch.empty((M_c, I), device=a.device, dtype=out_dtype)
                probs_c = torch.ones(M_c, device=a.device, dtype=torch.float32)
            except torch.cuda.OutOfMemoryError:
                return None
            return (
                a_c.view(torch.int8),
                b.view(torch.int8),
                l1_c,
                act_c,
                probs_c,
                a_scale.float().reshape(1),
                b_scale.float().reshape(1),
                _balanced_group_offs(M_c, G, a.device),
                M_c,
                I,
                torch.cuda.current_stream(),
            ), act_c

        with CompilationContext.compile_hints(_GG_SCHED_HINTS):
            launch = _autotune_glu(_build, _GLU_NT_CANDS, _probe)
        _GROUPED_GLU_CACHE[ckey] = launch

    with CompilationContext.compile_hints(_GG_SCHED_HINTS):
        launch(
            a.view(torch.int8),
            b.view(torch.int8),
            l1,
            act,
            probs,
            a_scale.float().reshape(1),
            b_scale.float().reshape(1),
            go32,
            M_total,
            I,  # c_n is the gate width; the kernel derives 2I where it needs it
            torch.cuda.current_stream(),
        )
    return act, l1


_GROUPED_DGLU_CACHE: dict = {}

_DGLU_BLOCK_N = 256


class GradProbsPartialSpec(NamedTuple):
    """The grad_probs partial buffer a fused dgrad expects from its caller."""

    shape: "tuple[int, int]"
    needs_zero: bool


def grouped_gemm_fp8_dglu_grad_probs_partial_spec(
    a: "torch.Tensor", b: "torch.Tensor", i_real: "int | None" = None
) -> GradProbsPartialSpec:
    """The buffer :func:`grouped_gemm_fp8_tensorwise_epi_dglu_flydsl_kernel` expects.

    ``grad_probs_partial.sum(0)`` is the gradient wrt ``probs``. Allocating and
    folding stay with the caller, so this module owns neither.

    One slice per ``(block_n, 16-lane half)``, rather than the Triton twin's one
    per tile: every wave's rows are disjoint in the banded epilogue, so waves
    share a slice and only the column block and the half of the 32 lanes
    covering a row have to be distinct.

    ``needs_zero`` is True, unlike the Triton twin: a group's last M-tile is
    clamped off at ``m_end``, so the rows past it are never written and would
    otherwise poison the fold.

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
    num_cu: "int | None" = None,
    i_real: "int | None" = None,
) -> "torch.Tensor":
    """FlyDSL fc2 dgrad with the SwiGLU gradient fused into its epilogue.

    Computes ``dact = (a[g] @ b[g]) * a_scale * b_scale`` and consumes it in
    registers, so ``dact`` never reaches HBM:

        dl1 = [f'(gate) * up * (dact * probs) | f(gate) * (dact * probs)]

    Unlike the forward twin this needs no tile geometry changes -- the GEMM's N
    axis is already I, so gate and up are a plain pair of global reads I columns
    apart. What it adds is the ``grad_probs`` reduction, whose sum spans all of
    I while a tile spans 256 columns: each wave folds the columns it owns and
    writes one partial for the caller to sum. No atomics, so the result is
    bitwise reproducible, matching the Triton entry.

    NN only (``trans_b=False``, b [G, K, I]).

    Args:
        out: [M_total, 2I] buffer receiving ``dl1``, in ``intermediate``'s dtype,
            gate gradient in [:, :I] and up in [:, I:].
        grad_probs_partial: float32 buffer receiving the grad_probs partials, as
            :func:`grouped_gemm_fp8_dglu_grad_probs_partial_spec` describes it --
            including that it has to arrive zeroed.

    Returns:
        ``out``, for call-site convenience.
    """
    assert activation == "silu", f"FlyDSL fused dGLU implements silu only, got {activation}"
    assert not trans_b, "FlyDSL fused dGLU is NN only: pass b as [G, K, I]"
    assert a.ndim == 2 and b.ndim == 3 and intermediate.ndim == 2
    M_total, K = a.shape
    G, K_b, N_pitch = b.shape
    # b (w2) may be padded on its I axis to N_pitch; i_real is the tight I. The
    # kernel reads B at the N_pitch storage stride (n_stride) but computes,
    # indexes intermediate, and stores dl1 at the real I width.
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

    _go64 = group_offs if group_offs.dtype == torch.int64 else group_offs.to(torch.int64)
    go32 = _go64.view(torch.int32)
    out_fp16 = out_dtype == torch.float16
    cbsz = 1 if a.dtype == torch.float8_e5m2 else 0
    blgp = 1 if b.dtype == torch.float8_e5m2 else 0
    _capped = num_cu is not None and num_cu > 0
    ckey = (I, K, G, out_fp16, cbsz, blgp, num_cu if _capped else 0, n_stride)

    launch = _GROUPED_DGLU_CACHE.get(ckey)
    if launch is None:

        def _build(_xcd, _gm):
            return _compile_grouped_nn(
                K=K,
                G=G,
                BLOCK_M=256,
                BLOCK_N=_DGLU_BLOCK_N,
                nt_vmcnt=3,
                # Raced per shape over _GLU_NN_CANDS. The arms differ from the
                # forward's -- (8, 4) pays here and row-major does not -- and the
                # spread reaches 1.090x at I=1024.
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
                glu_i=I,
                # Non-temporal, as in the forward, but this only pays because the
                # banded epilogue's accesses are whole lines: at 64 bytes a store
                # is half a line whose other half belongs to the neighbouring
                # wave, and the two have to meet in L2 to form one, which
                # non-temporal prevents. Per-wave staging preferred aux=0.
                cstore_aux=2,
            )

        def _probe():
            M_c = _glu_probe_m(G)
            try:
                a_c = torch.empty((M_c, K), device=a.device, dtype=torch.uint8)
                a_c.random_(0, 64, generator=torch.Generator(device=a.device).manual_seed(0))
                # Not zeros: at gate = up = 0 every gradient is 0 for every arm,
                # which would leave the numeric guard comparing zero to zero.
                l1_c = torch.empty((M_c, 2 * I), device=a.device, dtype=out_dtype)
                l1_c.uniform_(-2.0, 2.0)
                dl1_c = torch.empty((M_c, 2 * I), device=a.device, dtype=out_dtype)
                probs_c = torch.ones(M_c, device=a.device, dtype=torch.float32)
                grad_probs_c = torch.zeros((partial_shape[0], M_c), device=a.device, dtype=torch.float32)
            except torch.cuda.OutOfMemoryError:
                return None
            return (
                a_c.view(torch.int8),
                b.view(torch.int8),
                dl1_c,
                a_scale.float().reshape(1),
                b_scale.float().reshape(1),
                _balanced_group_offs(M_c, G, a.device),
                l1_c,
                probs_c,
                grad_probs_c,
                M_c,
                I,
                M_c,
                torch.cuda.current_stream(),
            ), dl1_c

        with CompilationContext.compile_hints(_GG_SCHED_HINTS):
            launch = _autotune_glu(_build, _GLU_NN_CANDS, _probe)
        _GROUPED_DGLU_CACHE[ckey] = launch

    with CompilationContext.compile_hints(_GG_SCHED_HINTS):
        launch(
            a.view(torch.int8),
            b.view(torch.int8),
            dl1,
            a_scale.float().reshape(1),
            b_scale.float().reshape(1),
            go32,
            intermediate,
            probs,
            grad_probs_partial,
            M_total,
            I,
            M_total,
            torch.cuda.current_stream(),
        )
    return dl1
