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

"""FlyDSL MXFP4 grouped GEMM with the SwiGLU activation fused in.

The entry layer for two shapes: the fc1 forward, emitting the pre-activation
``l1`` alongside the activation, and the fc2 dgrad, consuming ``dact`` in
registers so it never reaches HBM. Both ride the NT kernel factory in
``grouped_gemm_mxfp4_kernel`` behind its ``glu`` / ``dglu`` flags.

Unlike the fp8 twin, whose dgrad is NN, both directions are NT here: the B
operand's preshuffled block scale is only walkable by the NT addressing, so
fc2's weight comes in as [G, I, K].
"""

import torch

from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8_glu_kernel import GradProbsPartialSpec
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_mxfp4_kernel import (
    _BLOCK,
    _PRESHUF_BLK,
    _PRESHUF_FO,
    _bound_caches,
    _compile_grouped_mxfp4_nt_glu,
    _get_grouped_mxfp4_ws,
    _run_mxfp4_sched,
    _select_gmxfp4_nt_cfg,
)
from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import MB, _next_sr_seed
from primus_turbo.flydsl.utils.prims import ceildiv

_GMXFP4_GLU_CACHE: dict = {}  # -> [launch, compiled, n_blocks]

_WLV, _ELGK = 10, 9  # whole-loop / early-load-gap knobs, as the plain NT entry sets them

# The dgrad keeps the plain N tiling, so its grad_probs fold has one slice per
# 256-column block -- see StoreCdSwiGLUQuadCShuffle.
_DGLU_BLOCK_N = _BLOCK


def _glu_entry(
    K256,
    k_real,
    G,
    glu_i,
    out_fp16,
    *,
    glu,
    dglu,
    cfg,
    quant_row=False,
    fuse_act_quant=False,
    total_M=0,
    epi_act_quant=False,
    dglu_epi_quant=False,
    epi_row_sr=False,
    epi_col_sr=False,
):
    """Compiled launch for one fused shape, cached on the static shape + blocking."""
    gm, xcd, gn, span, _nt = cfg
    key = (
        K256,
        k_real,
        G,
        glu_i,
        gm,
        xcd,
        gn,
        span,
        out_fp16,
        glu,
        quant_row,
        fuse_act_quant,
        total_M,
        epi_act_quant,
        dglu_epi_quant,
        epi_row_sr,
        epi_col_sr,
    )
    ent = _GMXFP4_GLU_CACHE.get(key)
    if ent is None:
        launch, n_blocks = _compile_grouped_mxfp4_nt_glu(
            K256,
            G,
            glu_i,
            gm,
            xcd,
            gn,
            _WLV,
            _ELGK,
            out_fp16,
            k_real=k_real,
            span=span,
            glu=glu,
            dglu=dglu,
            glu_i=glu_i,
            glu_quant_row=quant_row,
            fuse_act_quant=fuse_act_quant,
            quant_total_M=total_M,
            epi_act_quant=epi_act_quant,
            dglu_epi_quant=dglu_epi_quant,
            epi_row_sr=epi_row_sr,
            epi_col_sr=epi_col_sr,
        )
        ent = [launch, None, n_blocks]
        _GMXFP4_GLU_CACHE[key] = ent
    return ent


def _pack_operands(a, a_scale, b, b_scale, group_offs):
    """The flat int8/int32 views the mxfp4 launch takes, as the plain NT entry builds them."""
    au = a.contiguous().view(torch.uint8)
    bu = b.contiguous().view(torch.uint8)
    # Keep the fp4 operands multi-dim: a 1D view of a >2^31-element MoE tensor
    # overflows the CABI int32 dim.
    a8 = au.view(torch.int8)
    b8 = bu.view(torch.int8)
    a_raw = a_scale.contiguous().view(torch.uint8).view(torch.int32).reshape(-1)
    b_raw = b_scale.contiguous().view(torch.uint8).view(torch.int32).reshape(-1)
    go64 = group_offs if group_offs.dtype == torch.int64 else group_offs.to(torch.int64)
    return a8, b8, a_raw, b_raw, go64.view(torch.int32)


def glu_epi_quant_supported(K: int, I: int, out_dtype=torch.bfloat16) -> bool:
    """Whether :func:`grouped_gemm_mxfp4_epi_glu_quant_flydsl_kernel` covers this shape.

    The dual-quant epilogue needs the ``l1`` store out of its way -- in the
    mainloop's store slot rather than the epilogue -- and the NT kernel only offers
    that slot for a bf16 output over an odd count of 128-K blocks. Mirrors the
    ``_CSTORE`` gate in ``grouped_gemm_mxfp4_kernel``; shapes outside it are the
    caller's to route through the separate quantiser.

    Args:
        K: the operand's padded contraction, i.e. ``b.shape[2] * 2``.
        I: the gate width.
    """
    if out_dtype != torch.bfloat16 or K % 128 or I % MB:
        return False
    if (K // 128) % 2 == 0:  # no trailing 128-K block
        return False
    ki_loop = K // _BLOCK + 1
    # Whole-loop unroll: an even, short K loop has no spare slot to hide the store in.
    return (ki_loop % 2 == 1) or ki_loop >= 4


def grouped_gemm_mxfp4_epi_glu_quant_flydsl_kernel(
    a: "torch.Tensor",
    a_scale: "torch.Tensor",
    b: "torch.Tensor",
    b_scale: "torch.Tensor",
    probs: "torch.Tensor",
    group_offs: "torch.Tensor",
    intermediate_out: "torch.Tensor",
    row_out: "torch.Tensor",
    row_sc: "torch.Tensor",
    col_out: "torch.Tensor",
    col_sc: "torch.Tensor",
    N: int,
    K: int,
    *,
    activation: str = "silu",
    row_use_sr: bool = False,
    col_use_sr: bool = False,
    out_dtype=torch.bfloat16,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """fc1 GLU whose activation is quantised in the epilogue, never reaching bf16.

    Computes ``l1 = [gate|up] = a[g] @ b[g]^T`` [M, 2I] under the MX block scales, and
    ``act = silu(gate) * up * probs`` [M, I] straight into the two MXFP4 operands the
    MLP actually consumes -- row-wise for fc2 and col-wise (RHT) for the wgrad --
    computed from the accumulators. Against staging ``act`` in bf16 that removes an
    [M, I] write, the quantiser's read of it, and a kernel launch.

    ``l1`` is still written: backward's dswiglu reads it.

    Which operand carries the RHT is fixed -- the col-wise one, which is what a wgrad
    contracts over -- so stochastic rounding is the only per-operand choice.

    The row-wise operand's columns past ``I`` and the col-wise operand's rows past
    a group's last 256-block are the quantiser's zero fill. The epilogue only ever
    writes real columns, so the caller has to have zeroed the column tail; the row
    tail is handled here because a tile covers it.

    Args:
        row_use_sr: stochastic-round the row-wise operand, seeded per micro-block from
            its linear index in the scale tensor, the same id the standalone quantiser
            uses.
        col_use_sr: the same for the col-wise operand, off a salted seed so a block the
            two share does not draw one sequence twice.
    """
    assert activation == "silu", f"FlyDSL fused GLU implements silu only, got {activation}"
    assert a.ndim == 2 and b.ndim == 3
    assert N % 2 == 0, f"fc1 width must be even (gate||up), got {N}"
    I = N // 2
    assert I % 128 == 0 or I % 32 == 0, f"a micro-block must not straddle I, got {I}"
    M_total, G = int(a.shape[0]), int(b.shape[0])
    l1 = intermediate_out
    assert l1.shape == (M_total, N) and l1.dtype == out_dtype

    a8, b8, a_raw, b_raw, go32 = _pack_operands(a, a_scale, b, b_scale, group_offs)
    K256 = ceildiv(K, 256) * 256
    a_sp, b_sp, slab_rows = _get_grouped_mxfp4_ws(M_total, N, K256 // 128, G, a.device)
    ent = _glu_entry(
        K256,
        K,
        G,
        I,
        out_dtype == torch.float16,
        glu=True,
        dglu=False,
        cfg=_select_gmxfp4_nt_cfg(M_total, G),
        epi_act_quant=True,
        epi_row_sr=row_use_sr,
        epi_col_sr=col_use_sr,
    )
    # The col-wise operand's row stride: one 256-block per tile row, per group.
    col_rows = col_out.shape[1] * 2
    args = (
        a8,
        b8,
        l1,
        probs,
        a_raw,
        b_raw,
        a_sp,
        b_sp,
        go32,
        M_total,
        I,
        slab_rows,
        ceildiv(slab_rows * (K256 // 128), _PRESHUF_FO * _PRESHUF_BLK),
        (ceildiv(M_total, _BLOCK) + G) * ent[2],
        row_out.view(torch.int32),
        # E8M0 has no memref element type; the scales are raw bytes to the kernel.
        row_sc.view(torch.uint8),
        col_out.view(torch.int32),
        col_sc.view(torch.uint8),
        col_rows,
        _next_sr_seed() if (row_use_sr or col_use_sr) else 0,
        torch.cuda.current_stream(),
    )
    _run_mxfp4_sched(ent, args, 1)
    _bound_caches(_GMXFP4_GLU_CACHE)
    return l1, row_out, row_sc, col_out, col_sc


def grouped_gemm_mxfp4_dglu_grad_probs_partial_spec(
    a: "torch.Tensor", b: "torch.Tensor"
) -> GradProbsPartialSpec:
    """The buffer :func:`grouped_gemm_mxfp4_epi_dglu_quant_flydsl_kernel` expects.

    ``grad_probs_partial.sum(0)`` is the gradient wrt ``probs``; allocating and folding
    stay with the caller.

    One slice per ``block_n``: the epilogue gives every wave disjoint rows and folds
    both column quadrants before publishing. ``needs_zero`` because a group's last
    M-tile is clamped at ``m_end``, leaving the rows past it unwritten.

    Returns:
        ``shape`` is ``(n_blocks, M_total)``, float32.
    """
    M_total = int(a.shape[0])
    I = int(b.shape[1])  # NT dgrad: b is [G, I, K/2], so I is the row count
    return GradProbsPartialSpec(shape=(ceildiv(I, _DGLU_BLOCK_N), M_total), needs_zero=True)


def dglu_epi_quant_supported(K: int, I: int, out_dtype=torch.bfloat16) -> bool:
    """Whether :func:`grouped_gemm_mxfp4_epi_dglu_quant_flydsl_kernel` covers this shape.

    Unlike the forward's, this epilogue competes for no store slot -- the dgrad
    never had one -- and its band is the parent's, so K is free and only the
    micro-block alignment matters: a row-wise block of 32 columns must not straddle
    the ``dg`` and ``du`` halves of ``grad_l1``.
    """
    del K
    return out_dtype == torch.bfloat16 and I % MB == 0


def grouped_gemm_mxfp4_epi_dglu_quant_flydsl_kernel(
    a: "torch.Tensor",
    a_scale: "torch.Tensor",
    b: "torch.Tensor",
    b_scale: "torch.Tensor",
    intermediate: "torch.Tensor",
    group_offs: "torch.Tensor",
    probs: "torch.Tensor",
    grad_probs_partial: "torch.Tensor",
    row_out: "torch.Tensor",
    row_sc: "torch.Tensor",
    col_out: "torch.Tensor",
    col_sc: "torch.Tensor",
    N: int,
    K: int,
    *,
    activation: str = "silu",
    row_use_sr: bool = False,
    col_use_sr: bool = False,
    out_dtype=torch.bfloat16,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """fc2 dgrad whose ``grad_l1`` is quantised in the epilogue, never reaching bf16.

    Computes the fc2 dgrad and turns it into the pre-activation gradient, whose two
    MXFP4 operands are what backward actually consumes -- row-wise for ``grad_x`` and
    col-wise (RHT) for ``grad_w1``. ``grad_l1`` is [M, 2I], the largest tensor in the
    MLP, so keeping it out of bf16 removes the biggest round trip there was: at
    [131072, 5760] a 1.5 GB write and a 1.5 GB read, against 0.82 GB written once.

    ``grad_probs`` is reduced from the per-tile partials the epilogue leaves.

    Which operand carries the RHT is fixed -- the col-wise one, which is what the
    wgrad contracts over -- so stochastic rounding is the only per-operand choice.

    The col-wise operand's rows past a group's last 256-block are the quantiser's
    zero fill and are handled here, because a tile covers them. The row-wise
    operand's columns past ``2I`` are not, so the caller has to have zeroed them.

    Args:
        N: the weight's row count per expert, i.e. ``I``.
        K: the true contraction, unpadded.
        row_use_sr: stochastic-round the row-wise operand, as the gradient recipe
            asks. Seeded per micro-block from its linear index in the scale tensor,
            the same id the standalone quantiser uses.
        col_use_sr: the same for the col-wise operand, off a salted seed so a block
            the two share does not draw one sequence twice.

    Returns:
        ``(row_out, row_sc, col_out, col_sc)``.
    """
    assert activation == "silu", f"FlyDSL fused dGLU implements silu only, got {activation}"
    assert a.ndim == 2 and b.ndim == 3 and intermediate.ndim == 2
    I = N
    assert dglu_epi_quant_supported(K, I, out_dtype), (
        f"unsupported shape for the fused dglu quant: I={I} K={K}"
    )
    M_total, G = int(a.shape[0]), int(b.shape[0])
    assert int(b.shape[1]) == I, f"b must be [{G}, {I}, K/2], got {tuple(b.shape)}"
    assert intermediate.shape == (M_total, 2 * I), (
        f"intermediate must be [{M_total}, {2 * I}], got {tuple(intermediate.shape)}"
    )
    assert probs.ndim == 1 and probs.shape[0] == M_total and probs.dtype == torch.float32
    want = grouped_gemm_mxfp4_dglu_grad_probs_partial_spec(a, b).shape
    assert tuple(grad_probs_partial.shape) == want and grad_probs_partial.dtype == torch.float32, (
        f"grad_probs_partial must be {list(want)} float32, got "
        f"{list(grad_probs_partial.shape)} {grad_probs_partial.dtype}; "
        "size it with grouped_gemm_mxfp4_dglu_grad_probs_partial_spec"
    )

    a8, b8, a_raw, b_raw, go32 = _pack_operands(a, a_scale, b, b_scale, group_offs)
    K256 = ceildiv(K, 256) * 256
    a_sp, b_sp, slab_rows = _get_grouped_mxfp4_ws(M_total, I, K256 // 128, G, a.device)
    ent = _glu_entry(
        K256,
        K,
        G,
        I,
        out_dtype == torch.float16,
        glu=False,
        dglu=True,
        cfg=_select_gmxfp4_nt_cfg(M_total, G),
        dglu_epi_quant=True,
        epi_row_sr=row_use_sr,
        epi_col_sr=col_use_sr,
    )
    # The col-wise operand's row stride: one 256-block per tile row, per group.
    col_rows = col_out.shape[1] * 2
    args = (
        a8,
        b8,
        intermediate,
        probs,
        grad_probs_partial,
        a_raw,
        b_raw,
        a_sp,
        b_sp,
        go32,
        M_total,
        I,
        slab_rows,
        M_total,  # gp_stride: the partial buffer's row pitch
        ceildiv(slab_rows * (K256 // 128), _PRESHUF_FO * _PRESHUF_BLK),
        (ceildiv(M_total, _BLOCK) + G) * ent[2],
        row_out.view(torch.int32),
        # E8M0 has no memref element type; the scales are raw bytes to the kernel.
        row_sc.view(torch.uint8),
        col_out.view(torch.int32),
        col_sc.view(torch.uint8),
        col_rows,
        _next_sr_seed() if (row_use_sr or col_use_sr) else 0,
        torch.cuda.current_stream(),
    )
    _run_mxfp4_sched(ent, args, 1)
    _bound_caches(_GMXFP4_GLU_CACHE)
    return row_out, row_sc, col_out, col_sc
