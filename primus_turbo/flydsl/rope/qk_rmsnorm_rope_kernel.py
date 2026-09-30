###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# See LICENSE for license information.
###############################################################################

"""GPT-OSS packed-QKV RMSNorm + RoPE FlyDSL kernels.

The production tensor is ``[S, B, NG, (NPG + 2) * D]`` with one packed group
laid out as ``[NPG query heads, key, value]``.  GPT-OSS-20B uses
``NG=8, NPG=8, D=64``.  One wave owns one complete 64-element head row:
lane ``d`` owns element ``d``, making both the RMS reduction and rotate-half
pair exchange wave-local.

Forward consumes packed QKV directly and emits contiguous Q/K/V.  Q and K are
RMS-normalized, explicitly rounded to BF16 (matching the materialized output of
the existing RMSNorm kernel), and then rotated.  V is copied unchanged.

Backward performs inverse RoPE, explicitly rounds that result to BF16 (matching
the materialized gradient between the existing RoPE and RMSNorm kernels), then
applies RMSNorm backward and writes directly into packed dQKV.  Each persistent
wave also writes one FP32 dgamma partial for a deterministic host-side fold.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops
from flydsl.expr import math as fmath

QK_RMSNORM_ROPE_HEAD_DIM = 64
_D = QK_RMSNORM_ROPE_HEAD_DIM
_HALF = _D // 2
_WARP = 64
_WAVES = 4
_BLOCK_THREADS = _WARP * _WAVES

# A cycle contains one persistent wave for every logical head slot.  Forward
# keeps 128 cycles; backward uses 64 longer-lived cycles to halve its dgamma
# partial workspace/finalization cost.  Keeping each grid-wave count a multiple
# of its head width makes every wave stay on one logical head, which gives it a
# single gamma value and dgamma accumulator.
_FWD_GRID_CYCLES = 128
_BWD_GRID_CYCLES = 64

# Megatron passes the same full-sequence rotary table to every transformer
# layer.  Materialize cos/sin once and reuse it across all fused launches rather
# than evaluating transcendental functions independently for every Q/K head.
# Keep only one entry per device so a regenerated table cannot grow the cache.
_ROTARY_TABLE_CACHE = {}


def _cached_cos_sin(freqs):
    import weakref

    device = freqs.device.index
    version = freqs._version
    cached = _ROTARY_TABLE_CACHE.get(device)
    if cached is not None:
        ref, cached_version, cosine, sine = cached
        if ref() is freqs and cached_version == version:
            return cosine, sine
    cosine = freqs.cos()
    sine = freqs.sin()
    _ROTARY_TABLE_CACHE[device] = (weakref.ref(freqs), version, cosine, sine)
    return cosine, sine


def _wave_sum_f32(value):
    """Butterfly sum across one gfx950 wave64."""
    value = fx.arith.ArithValue(value)
    for distance in (1, 2, 4, 8, 16, 32):
        value = value.addf(fx.arith.ArithValue(value.shuffle_xor(distance, _WARP)))
    return value


def _make_fwd_kernel(S: int, B: int, NG: int, NPG: int, eps: float):
    packed_heads = NG * (NPG + 2)
    q_heads = NG * NPG
    qk_heads = NG * (NPG + 1)

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def kernel(PACKED: fx.Tensor, QG: fx.Tensor, KG: fx.Tensor,
               COSINE: fx.Tensor, SINE: fx.Tensor,
               QOUT: fx.Tensor, KOUT: fx.Tensor,
               QRSTD: fx.Tensor, KRSTD: fx.Tensor):
        tid = fx.thread_idx.x
        block_x, _, _ = fx.block_idx
        lane = tid % fx.Int32(_WARP)
        wave = tid // fx.Int32(_WARP)
        global_wave = block_x * fx.Int32(_WAVES) + wave
        qk_slot = global_wave % fx.Int32(qk_heads)
        cycle = global_wave // fx.Int32(qk_heads)
        group = qk_slot // fx.Int32(NPG + 1)
        local = qk_slot % fx.Int32(NPG + 1)
        is_q = local < fx.Int32(NPG)
        # Skip the packed V slot.  V remains a zero-copy strided view of PACKED,
        # exactly like the existing Megatron split path; copying it here costs
        # bandwidth and one more buffer resource without contributing to the
        # Q/K fusion.
        slot = group * fx.Int32(NPG + 2) + local

        packed_rsrc = buffer_ops.create_buffer_resource(PACKED, max_size=True)
        qg_rsrc = buffer_ops.create_buffer_resource(QG, max_size=True)
        kg_rsrc = buffer_ops.create_buffer_resource(KG, max_size=True)
        cosine_rsrc = buffer_ops.create_buffer_resource(COSINE, max_size=True)
        sine_rsrc = buffer_ops.create_buffer_resource(SINE, max_size=True)
        qout_rsrc = buffer_ops.create_buffer_resource(QOUT, max_size=True)
        kout_rsrc = buffer_ops.create_buffer_resource(KOUT, max_size=True)
        qrstd_rsrc = buffer_ops.create_buffer_resource(QRSTD, max_size=True)
        krstd_rsrc = buffer_ops.create_buffer_resource(KRSTD, max_size=True)

        # A persistent wave never changes logical head or lane, so gamma is
        # invariant for its entire token loop.  Hoisting avoids one redundant
        # BF16 buffer load per normalized row (millions of loads per launch).
        gamma = fx.Float32(0.0)
        if is_q:
            gamma = fx.Float32(
                buffer_ops.buffer_load(qg_rsrc, lane, vec_width=1, dtype=fx.T.bf16())
            )
        else:
            # qk_slot contains only Q slots followed by one K slot per group,
            # so the non-Q branch is always K.  Keep this as a single
            # FlyDSL-rewritten branch; a nested condition here loses its local
            # predicate when the AST rewriter outlines the else body.
            gamma = fx.Float32(
                buffer_ops.buffer_load(kg_rsrc, lane, vec_width=1, dtype=fx.T.bf16())
            )
        q_head = group * fx.Int32(NPG) + local

        # All B token rows at one sequence position share the same rotary
        # values.  Process them together so each wave loads cos/sin once per
        # sequence instead of once per batch element.
        seq = cycle
        while seq < fx.Int32(S):
            cosine = fx.Float32(
                buffer_ops.buffer_load(
                    cosine_rsrc, seq * fx.Int32(_D) + lane, vec_width=1, dtype=fx.T.f32()
                )
            )
            sine = fx.Float32(
                buffer_ops.buffer_load(
                    sine_rsrc, seq * fx.Int32(_D) + lane, vec_width=1, dtype=fx.T.f32()
                )
            )
            for batch_idx in range(B):
                token = seq * fx.Int32(B) + fx.Int32(batch_idx)
                src = (token * fx.Int32(packed_heads) + slot) * fx.Int32(_D) + lane
                x = fx.Float32(
                    buffer_ops.buffer_load(packed_rsrc, src, vec_width=1, dtype=fx.T.bf16())
                )

                sumsq = _wave_sum_f32(x * x)
                mean = sumsq / fx.Float32(float(_D))
                rstd = fx.Float32(fmath.rsqrt(mean + fx.Float32(eps)))

                # Match the existing two-kernel contract: RMSNorm writes BF16,
                # then RoPE reads BF16 and promotes it for the rotation math.
                norm = (x * rstd * gamma).to(fx.BFloat16).to(fx.Float32)
                pair = fx.arith.ArithValue(norm).shuffle_xor(_HALF, _WARP)
                low = norm * cosine - fx.Float32(pair) * sine
                high = norm * cosine + fx.Float32(pair) * sine
                rotated = fx.BFloat16(
                    fx.arith.select(lane < fx.Int32(_HALF), low, high)
                )

                if is_q:
                    dst = (token * fx.Int32(q_heads) + q_head) * fx.Int32(_D) + lane
                    buffer_ops.buffer_store(rotated, qout_rsrc, dst)
                    if lane == fx.Int32(0):
                        buffer_ops.buffer_store(
                            rstd, qrstd_rsrc, token * fx.Int32(q_heads) + q_head
                        )
                else:
                    dst = (token * fx.Int32(NG) + group) * fx.Int32(_D) + lane
                    buffer_ops.buffer_store(rotated, kout_rsrc, dst)
                    if lane == fx.Int32(0):
                        buffer_ops.buffer_store(
                            rstd, krstd_rsrc, token * fx.Int32(NG) + group
                        )

            seq = seq + fx.Int32(_FWD_GRID_CYCLES)

    return kernel


def _make_bwd_kernel(S: int, B: int, NG: int, NPG: int):
    packed_heads = NG * (NPG + 2)
    q_heads = NG * NPG

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def kernel(DQ: fx.Tensor, DK: fx.Tensor, DV: fx.Tensor, PACKED: fx.Tensor,
               QG: fx.Tensor, KG: fx.Tensor, COSINE: fx.Tensor, SINE: fx.Tensor,
               QRSTD: fx.Tensor, KRSTD: fx.Tensor, DPACKED: fx.Tensor,
               DQG_PART: fx.Tensor, DKG_PART: fx.Tensor):
        tid = fx.thread_idx.x
        block_x, _, _ = fx.block_idx
        lane = tid % fx.Int32(_WARP)
        wave = tid // fx.Int32(_WARP)
        global_wave = block_x * fx.Int32(_WAVES) + wave
        slot = global_wave % fx.Int32(packed_heads)
        cycle = global_wave // fx.Int32(packed_heads)
        group = slot // fx.Int32(NPG + 2)
        local = slot % fx.Int32(NPG + 2)
        is_q = local < fx.Int32(NPG)
        is_k = local == fx.Int32(NPG)

        dq_rsrc = buffer_ops.create_buffer_resource(DQ, max_size=True)
        dk_rsrc = buffer_ops.create_buffer_resource(DK, max_size=True)
        dv_rsrc = buffer_ops.create_buffer_resource(DV, max_size=True)
        packed_rsrc = buffer_ops.create_buffer_resource(PACKED, max_size=True)
        qg_rsrc = buffer_ops.create_buffer_resource(QG, max_size=True)
        kg_rsrc = buffer_ops.create_buffer_resource(KG, max_size=True)
        cosine_rsrc = buffer_ops.create_buffer_resource(COSINE, max_size=True)
        sine_rsrc = buffer_ops.create_buffer_resource(SINE, max_size=True)
        qrstd_rsrc = buffer_ops.create_buffer_resource(QRSTD, max_size=True)
        krstd_rsrc = buffer_ops.create_buffer_resource(KRSTD, max_size=True)
        dpacked_rsrc = buffer_ops.create_buffer_resource(DPACKED, max_size=True)
        dqg_part_rsrc = buffer_ops.create_buffer_resource(DQG_PART, max_size=True)
        dkg_part_rsrc = buffer_ops.create_buffer_resource(DKG_PART, max_size=True)

        # Like forward, every persistent wave owns one head/lane, so gamma and
        # the Q head index are loop invariant.
        gamma = fx.Float32(0.0)
        if is_q:
            gamma = fx.Float32(
                buffer_ops.buffer_load(qg_rsrc, lane, vec_width=1, dtype=fx.T.bf16())
            )
        else:
            gamma = fx.Float32(
                buffer_ops.buffer_load(kg_rsrc, lane, vec_width=1, dtype=fx.T.bf16())
            )
        q_head = group * fx.Int32(NPG) + local

        dgamma = fx.Float32(0.0)
        seq = cycle
        while seq < fx.Int32(S):
            cosine = fx.Float32(0.0)
            sine = fx.Float32(0.0)
            if is_q | is_k:
                cosine = fx.Float32(
                    buffer_ops.buffer_load(
                        cosine_rsrc, seq * fx.Int32(_D) + lane, vec_width=1, dtype=fx.T.f32()
                    )
                )
                sine = fx.Float32(
                    buffer_ops.buffer_load(
                        sine_rsrc, seq * fx.Int32(_D) + lane, vec_width=1, dtype=fx.T.f32()
                    )
                )
            for batch_idx in range(B):
                token = seq * fx.Int32(B) + fx.Int32(batch_idx)
                dst = (token * fx.Int32(packed_heads) + slot) * fx.Int32(_D) + lane
                if is_q | is_k:
                    grad = fx.Float32(0.0)
                    rstd = fx.Float32(0.0)
                    if is_q:
                        src = (token * fx.Int32(q_heads) + q_head) * fx.Int32(_D) + lane
                        grad = fx.Float32(
                            buffer_ops.buffer_load(dq_rsrc, src, vec_width=1, dtype=fx.T.bf16())
                        )
                        rstd = fx.Float32(
                            buffer_ops.buffer_load(
                                qrstd_rsrc,
                                token * fx.Int32(q_heads) + q_head,
                                vec_width=1,
                                dtype=fx.T.f32(),
                            )
                        )
                    else:
                        src = (token * fx.Int32(NG) + group) * fx.Int32(_D) + lane
                        grad = fx.Float32(
                            buffer_ops.buffer_load(dk_rsrc, src, vec_width=1, dtype=fx.T.bf16())
                        )
                        rstd = fx.Float32(
                            buffer_ops.buffer_load(
                                krstd_rsrc,
                                token * fx.Int32(NG) + group,
                                vec_width=1,
                                dtype=fx.T.f32(),
                            )
                        )

                    pair_grad = fx.arith.ArithValue(grad).shuffle_xor(_HALF, _WARP)
                    low = grad * cosine + fx.Float32(pair_grad) * sine
                    high = grad * cosine - fx.Float32(pair_grad) * sine
                    # Match the materialized BF16 gradient at the RoPE -> RMSNorm boundary.
                    dnorm = fx.Float32(
                        fx.BFloat16(fx.arith.select(lane < fx.Int32(_HALF), low, high))
                    )

                    x = fx.Float32(
                        buffer_ops.buffer_load(packed_rsrc, dst, vec_width=1, dtype=fx.T.bf16())
                    )
                    unit = x * rstd
                    dunit = dnorm * gamma
                    dot = _wave_sum_f32(unit * dunit)
                    dx = (dunit - unit * (dot / fx.Float32(float(_D)))) * rstd
                    buffer_ops.buffer_store(dx.to(fx.BFloat16), dpacked_rsrc, dst)
                    dgamma = dgamma + dnorm * unit
                else:
                    src = (token * fx.Int32(NG) + group) * fx.Int32(_D) + lane
                    grad = buffer_ops.buffer_load(dv_rsrc, src, vec_width=1, dtype=fx.T.bf16())
                    buffer_ops.buffer_store(grad, dpacked_rsrc, dst)

            seq = seq + fx.Int32(_BWD_GRID_CYCLES)

        if is_q:
            part = (cycle * fx.Int32(q_heads) + q_head) * fx.Int32(_D) + lane
            buffer_ops.buffer_store(dgamma, dqg_part_rsrc, part)
        elif is_k:
            part = (cycle * fx.Int32(NG) + group) * fx.Int32(_D) + lane
            buffer_ops.buffer_store(dgamma, dkg_part_rsrc, part)

    return kernel


@flyc.jit
def _compiled_fwd(
    PACKED,
    QG,
    KG,
    COSINE,
    SINE,
    QOUT,
    KOUT,
    QRSTD,
    KRSTD,
    S: fx.Constexpr[int],
    B: fx.Constexpr[int],
    NG: fx.Constexpr[int],
    NPG: fx.Constexpr[int],
    EPS: fx.Constexpr[float],
    stream: fx.Stream,
):
    qk_heads = NG * (NPG + 1)
    assert (_FWD_GRID_CYCLES * qk_heads) % _WAVES == 0
    grid_x = _FWD_GRID_CYCLES * qk_heads // _WAVES
    kernel = _make_fwd_kernel(S, B, NG, NPG, EPS)
    kernel(PACKED, QG, KG, COSINE, SINE, QOUT, KOUT, QRSTD, KRSTD).launch(
        grid=(grid_x, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream
    )


@flyc.jit
def _compiled_bwd(
    DQ,
    DK,
    DV,
    PACKED,
    QG,
    KG,
    COSINE,
    SINE,
    QRSTD,
    KRSTD,
    DPACKED,
    DQG_PART,
    DKG_PART,
    S: fx.Constexpr[int],
    B: fx.Constexpr[int],
    NG: fx.Constexpr[int],
    NPG: fx.Constexpr[int],
    stream: fx.Stream,
):
    packed_heads = NG * (NPG + 2)
    assert (_BWD_GRID_CYCLES * packed_heads) % _WAVES == 0
    grid_x = _BWD_GRID_CYCLES * packed_heads // _WAVES
    kernel = _make_bwd_kernel(S, B, NG, NPG)
    kernel(
        DQ, DK, DV, PACKED, QG, KG, COSINE, SINE,
        QRSTD, KRSTD, DPACKED, DQG_PART, DKG_PART
    ).launch(grid=(grid_x, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)


def flydsl_qkv_rmsnorm_rope_forward(qkv, q_gamma, k_gamma, freqs, split_sizes, eps):
    """Raw forward entry point; validation lives in the PyTorch wrapper."""
    import torch

    S, B, NG, _ = qkv.shape
    q_size, k_size, _ = split_sizes
    npg = q_size // _D
    assert k_size == _D
    q = torch.empty((S, B, NG * npg, _D), device=qkv.device, dtype=qkv.dtype)
    k = torch.empty((S, B, NG, _D), device=qkv.device, dtype=qkv.dtype)
    v = qkv[..., -_D:]
    q_rstd = torch.empty((S, B, NG * npg), device=qkv.device, dtype=torch.float32)
    k_rstd = torch.empty((S, B, NG), device=qkv.device, dtype=torch.float32)
    cosine, sine = _cached_cos_sin(freqs)
    _compiled_fwd(
        PACKED=qkv,
        QG=q_gamma,
        KG=k_gamma,
        COSINE=cosine,
        SINE=sine,
        QOUT=q,
        KOUT=k,
        QRSTD=q_rstd,
        KRSTD=k_rstd,
        S=S,
        B=B,
        NG=NG,
        NPG=npg,
        EPS=float(eps),
        stream=torch.cuda.current_stream(),
    )
    return q, k, v, q_rstd, k_rstd


def flydsl_qkv_rmsnorm_rope_backward(
    dq, dk, dv, qkv, q_gamma, k_gamma, freqs, q_rstd, k_rstd, split_sizes
):
    """Raw backward entry point returning packed dQKV and dgamma partials."""
    import torch

    S, B, NG, _ = qkv.shape
    q_size, k_size, _ = split_sizes
    npg = q_size // _D
    assert k_size == _D
    dqkv = torch.empty_like(qkv)
    dqg_part = torch.empty((_BWD_GRID_CYCLES * NG * npg, _D), device=qkv.device, dtype=torch.float32)
    dkg_part = torch.empty((_BWD_GRID_CYCLES * NG, _D), device=qkv.device, dtype=torch.float32)
    cosine, sine = _cached_cos_sin(freqs)
    _compiled_bwd(
        DQ=dq,
        DK=dk,
        DV=dv,
        PACKED=qkv,
        QG=q_gamma,
        KG=k_gamma,
        COSINE=cosine,
        SINE=sine,
        QRSTD=q_rstd,
        KRSTD=k_rstd,
        DPACKED=dqkv,
        DQG_PART=dqg_part,
        DKG_PART=dkg_part,
        S=S,
        B=B,
        NG=NG,
        NPG=npg,
        stream=torch.cuda.current_stream(),
    )
    return dqkv, dqg_part, dkg_part
