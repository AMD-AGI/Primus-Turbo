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

"""MLA schedule of the gfx950 dual-wave flash-attention forward (``flash_attn_fwd_stock``).

Adds D_QK != D_V (192/128, DeepSeek-V3 MLA): Q/K carry ``head_dim`` columns, V/O carry
``head_dim_v``; strides must be those of contiguous [S, B, H, D] / [T, H, D] tensors. The
MLA traits also cover 16x16 QK / P*V MFMAs, one wave per SIMD (``rows_per_wave`` 64) and a
persistent job loop over a host schedule.
"""

from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from primus_turbo.flydsl.attention.flash_attn_fwd_stock import build_dualwave_swp_launcher
from primus_turbo.flydsl.utils.attn_helper_stock import _make_dualwave_swp_traits


def make_mla_traits(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="bf16",
    num_kv_heads=None,
    waves_per_eu=2,
    daz=True,
    dualwave_swp_setprio=True,
    dualwave_swp_enable_stagger=True,
    varlen=False,
    cross_seqlen=False,
    emit_lse=False,
    window_left=-1,
    block_m=None,
    gqa_merge=None,
    sbhd=False,
    has_sink=False,
    head_dim_v=None,
    k_head=None,
    qk_mfma16=None,
    pv_mfma16=None,
    rows_per_wave=32,
    fold_barriers=None,
    persistent=0,
    sm_scale=None,
):
    """The traits ``build_flash_attn_dualwave_swp_module`` compiles (same arguments), for callers
    that embed its job body (``emit_dualwave_swp_body``)."""
    gpu_arch = get_hip_arch()

    if not gpu_arch.startswith("gfx950"):
        raise RuntimeError(
            f"flash_attn_dualwave_swp requires gfx950+ (uses ds_read_tr16_b64), got {gpu_arch}"
        )
    head_dim_v = head_dim if head_dim_v is None else head_dim_v
    if (head_dim, head_dim_v) not in ((64, 64), (128, 128), (192, 128)):
        raise RuntimeError(
            f"flash_attn_dualwave_swp supports D_QK/D_V 64/64, 128/128, 192/128, got {head_dim}/{head_dim_v}"
        )
    if dtype_str not in ("bf16", "f16"):
        raise RuntimeError(f"flash_attn_dualwave_swp supports bf16/f16 only, got dtype={dtype_str}")

    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_heads % num_kv_heads == 0

    traits = _make_dualwave_swp_traits(
        num_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        dtype_str=dtype_str,
        waves_per_eu=waves_per_eu,
        daz=daz,
        dualwave_swp_setprio=dualwave_swp_setprio,
        dualwave_swp_enable_stagger=dualwave_swp_enable_stagger,
        varlen=varlen,
        cross_seqlen=cross_seqlen,
        emit_lse=emit_lse,
        window_left=window_left,
        block_m=block_m,
        gqa_merge=gqa_merge,
        sbhd=sbhd,
        has_sink=has_sink,
        head_dim_v=head_dim_v,
        qk_mfma16=qk_mfma16,
        pv_mfma16=pv_mfma16,
        rows_per_wave=rows_per_wave,
        xcd_q_order=True,
    )
    traits.BIND_JOB_LAST = True
    traits.FOLD_BARRIERS = traits.NUM_WAVES > 4 if fold_barriers is None else bool(fold_barriers)
    traits.K_HEAD = max(1, traits.K_STEPS_QK - 2) if k_head is None else int(k_head)
    assert 1 <= traits.K_HEAD <= traits.K_STEPS_QK
    traits.PERSISTENT = int(persistent)
    assert not traits.PERSISTENT or not (varlen or cross_seqlen or window_left >= 0)
    # Persistent drains DMA part of the next job's Q through the O stage tiles (stage_q_dma);
    # its geometry is the D_QK 192 paired-step split.
    traits.Q_LDS_STAGE = bool(
        traits.PERSISTENT and traits.O_LDS_STAGE and traits.QK_MFMA16 and traits.HEAD_DIM == 192
    )
    # Persistent dense causal with four q tiles a wave: q tile qt of wave w takes block rows
    # 64 qt + 16 w, not 64 w + 16 qt. Every wave then holds the same share of the diagonal, and
    # the drain's dead q tiles are the same static ones on every wave (`_qk_diag`).
    traits.ROW_INTERLEAVE = bool(
        traits.PERSISTENT
        and traits.CAUSAL
        and traits.Q_TILES == 4
        and traits.QK_MFMA16
        and traits.PV_MFMA16
        and traits.Q_HEADS_PER_WG == 1
        and traits.NUM_WAVES * 16 == traits.BLOCK_M // traits.Q_TILES == traits.BLOCK_N
    )
    if traits.ROW_INTERLEAVE:
        traits.QT_ROW_STRIDE = traits.BLOCK_M // traits.Q_TILES
        traits.WAVE_ROW_STRIDE = 16
    traits.SM_SCALE = None if sm_scale is None else float(sm_scale)
    return traits


def build_flash_attn_dualwave_swp_module(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="bf16",
    num_kv_heads=None,
    waves_per_eu=2,
    daz=True,
    dualwave_swp_setprio=True,
    dualwave_swp_enable_stagger=True,
    varlen=False,
    cross_seqlen=False,
    emit_lse=False,
    window_left=-1,
    block_m=None,
    gqa_merge=None,
    sbhd=False,
    has_sink=False,
    head_dim_v=None,
    k_head=None,
    qk_mfma16=None,
    pv_mfma16=None,
    rows_per_wave=32,
    fold_barriers=None,
    persistent=0,
    sm_scale=None,
):
    """Build a DUALWAVE_SWP flash_attn launcher for (D_QK, D_V) in {64/64, 128/128, 192/128}.

    Supports dense (SBHD) and varlen packed QKV (THD) layouts. has_sink folds a learned
    per-q-head attention sink (SINK[Hq] fp32) into the online-softmax denominator.
    ``k_head`` is the number of QK k-steps whose K packs are read before the first MFMA
    group (default K_STEPS_QK - 2). ``qk_mfma16`` runs QK on 16x16x32 MFMAs (default: on
    for D_QK 192 without a causal window); ``pv_mfma16`` runs P*V on them too (default: with
    ``qk_mfma16``). ``rows_per_wave`` 64 halves the waves per q block (one per SIMD at
    block_m 256). ``fold_barriers`` drops the memory-cluster rendezvous (default: above 4 waves).
    ``persistent`` > 0 launches at most that many CTAs (one per CU), each running the jobs the
    host schedule (``launcher.schedule``) gives it; dense layouts without a window only.
    """
    return build_dualwave_swp_launcher(
        make_mla_traits(
            num_heads=num_heads,
            head_dim=head_dim,
            causal=causal,
            dtype_str=dtype_str,
            num_kv_heads=num_kv_heads,
            waves_per_eu=waves_per_eu,
            daz=daz,
            dualwave_swp_setprio=dualwave_swp_setprio,
            dualwave_swp_enable_stagger=dualwave_swp_enable_stagger,
            varlen=varlen,
            cross_seqlen=cross_seqlen,
            emit_lse=emit_lse,
            window_left=window_left,
            block_m=block_m,
            gqa_merge=gqa_merge,
            sbhd=sbhd,
            has_sink=has_sink,
            head_dim_v=head_dim_v,
            k_head=k_head,
            qk_mfma16=qk_mfma16,
            pv_mfma16=pv_mfma16,
            rows_per_wave=rows_per_wave,
            fold_barriers=fold_barriers,
            persistent=persistent,
            sm_scale=sm_scale,
        )
    )
