###############################################################################
# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
"""dsv4up: DeepSeek-V4 blockwise FP8 dense GEMM for gfx1250 (Gluon, wave32, WMMA only).

C[M,N] = (A[M,K] e4m3 * sA) @ (B[N,K] e4m3 * sB)^T     NT: both operands K-contiguous.

Kernel = upstream Triton third_party/amd/python/examples/gluon/gemm_warp_pipeline_cdna5.py (main 83c92283)
MXFP8 256x256x256 BK256 path (8 waves, TDM async loads, 5-stage warp pipeline, fused TDM scale loads,
wmma_scaled), single CTA (cluster = 1, no multicast), M/N/K/strides constexpr (one compile per shape: the
runtime-shape 1-CTA build spills 11 VGPRs, the static build is ~470 VGPR / 0 spills).
The DSv4 recipe (1x128 activation / 128x128 weight UE8M0 scales) is fed as MX block-32 E8M0 scales replicated
from the 128-wide blocks (exact for power-of-two scales), packed by dsv4up_pack.pack_scales.

Changes vs upstream: cluster barriers dropped (1 CTA), and an optional beta = 1 epilogue (ACCUM): the old C tile
is read in the accumulator layout and added in fp32 before the TDM store, so wgrad can accumulate straight into
an fp32 (or 16-bit) main_grad.  Requires Triton main (gluon cdna5 / tdm); imported lazily by the dispatch.
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.cdna5 import PartitionedSharedLayout, tdm

TILE = 256
BLOCK_K = 256
NUM_XCDS = 8
GROUP_SIZE_M = 4


# ------------------------------------------------------------------ upstream helpers (verbatim, cluster-free)
@gluon.jit
def _chiplet_transform(pid, num_workgroups, num_xcds: gl.constexpr):
    xcd = pid % num_xcds
    pos_in_xcd = pid // num_xcds
    min_per_xcd = num_workgroups // num_xcds
    extra_sms = num_workgroups % num_xcds
    offset = xcd * min_per_xcd + min(xcd, extra_sms)
    return offset + pos_in_xcd


@gluon.jit
def _get_xcd_swizzled_pids(M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, GRID_MN: gl.constexpr,
                           NUM_XCDS_: gl.constexpr, GROUP_M: gl.constexpr):
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    if NUM_XCDS_ != 1:
        pid = _chiplet_transform(pid, GRID_MN, NUM_XCDS_)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    return (pid_m, pid_n)


@gluon.jit
def _mxfp8_load_scale(scale_buffer, slot, start_k: gl.constexpr, LAYOUT: gl.constexpr, BLOCK_NONK: gl.constexpr,
                      BK_SCALE: gl.constexpr, SUBTILE_SCALE_K: gl.constexpr):
    scale_slice = scale_buffer.index(slot).reshape((BLOCK_NONK // 128, BK_SCALE // 4, 32, 4, 4)).permute(
        (0, 3, 2, 1, 4)).reshape((BLOCK_NONK, BK_SCALE))
    return scale_slice.slice(0, BLOCK_NONK, 0).slice(start_k, SUBTILE_SCALE_K, 1).load(layout=LAYOUT)


@gluon.jit
def _mxfp8_issue_b2_parent_data_without_update(a_desc, b_desc, a_buf, b_buf, slot):
    tdm.async_load(a_desc, dest=a_buf.index(slot), warp_used_hint=15)
    tdm.async_load(b_desc, dest=b_buf.index(slot), warp_used_hint=15)


@gluon.jit
def _mxfp8_issue_b2_parent_data(a_desc, b_desc, a_buf, b_buf, slot):
    _mxfp8_issue_b2_parent_data_without_update(a_desc, b_desc, a_buf, b_buf, slot)
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, 256])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 0, 256])
    return (a_desc, b_desc)


@gluon.jit
def _mxfp8_load_b2_parent(buffer, slot, start_n: gl.constexpr, start_k: gl.constexpr, B2_LOAD_LAYOUT: gl.constexpr,
                          DOT_B: gl.constexpr):
    rank3 = buffer.index(slot).slice(start_n, 128, 1).slice(start_k, 128, 2).permute([2, 0,
                                                                                      1]).load(layout=B2_LOAD_LAYOUT)
    rank2 = rank3.reshape((128, buffer.type.shape[1] * 128))
    return gl.convert_layout(rank2, DOT_B)


@gluon.jit
def _mxfp8_issue_leading4_scale(as_desc, bs_desc, as_buf, bs_buf, slot):
    tdm.async_load_fused([(as_desc, as_buf.index(slot), 3), (bs_desc, bs_buf.index(slot), 12)])
    as_desc = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, 1024])
    bs_desc = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, 1024])
    return (as_desc, bs_desc)


@gluon.jit
def _load_scales(as_buf, bs_buf, scale_slot, start_k: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr,
                 SCALE_A_LOAD: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr, SCALE_B_LOAD: gl.constexpr,
                 BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, CTA_N: gl.constexpr):
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    packed_n: gl.constexpr = BLOCK_N // 2
    as_ = _mxfp8_load_scale(as_buf, scale_slot, start_k, SCALE_A_LOAD, BLOCK_M, 8, 4)
    as_ = gl.convert_layout(as_, SCALE_A_LAYOUT, assert_trivial=True)
    bs = _mxfp8_load_scale(bs_buf, scale_slot, start_k, SCALE_B_LOAD, BLOCK_N, 8, 4)
    bs_grid = bs.reshape((CTA_N, cta_n, 4))
    bs_lo = gl.convert_layout(
        gl.amd.slice(bs_grid, [CTA_N, half_n, 4], [0, 0, 0]).reshape((packed_n, 4)), SCALE_B_LAYOUT,
        assert_trivial=True)
    bs_hi = gl.convert_layout(
        gl.amd.slice(bs_grid, [CTA_N, half_n, 4], [0, half_n, 0]).reshape((packed_n, 4)), SCALE_B_LAYOUT,
        assert_trivial=True)
    return as_, bs_lo, bs_hi


@gluon.jit
def _consume_tail(a_buf, b_buf, as_buf, bs_buf, data_slot, scale_slot, acc0, acc1, DOT_A: gl.constexpr,
                  DOT_B: gl.constexpr, B2_LOAD_LAYOUT: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr,
                  SCALE_A_LOAD: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr, SCALE_B_LOAD: gl.constexpr,
                  BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, CTA_N: gl.constexpr):
    with gl.amd.warp_pipeline_stage('b2_tail_load_low'):
        a0 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(0, 128, 1).load(layout=DOT_A)
        as0, bs0_lo, bs0_hi = _load_scales(as_buf, bs_buf, scale_slot, 0, SCALE_A_LAYOUT, SCALE_A_LOAD,
                                           SCALE_B_LAYOUT, SCALE_B_LOAD, BLOCK_M, BLOCK_N, CTA_N)
        b0_lo = _mxfp8_load_b2_parent(b_buf, data_slot, 0, 0, B2_LOAD_LAYOUT, DOT_B)
        b0_hi = _mxfp8_load_b2_parent(b_buf, data_slot, 128, 0, B2_LOAD_LAYOUT, DOT_B)
    with gl.amd.warp_pipeline_stage('b2_tail_compute_low'):
        acc0 = gl.amd.cdna5.wmma_scaled(a0, as0, 'e4m3', b0_lo, bs0_lo, 'e4m3', acc0)
        acc1 = gl.amd.cdna5.wmma_scaled(a0, as0, 'e4m3', b0_hi, bs0_hi, 'e4m3', acc1)
    with gl.amd.warp_pipeline_stage('b2_tail_load_high'):
        a1 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(128, 128, 1).load(layout=DOT_A)
        as1, bs1_lo, bs1_hi = _load_scales(as_buf, bs_buf, scale_slot, 4, SCALE_A_LAYOUT, SCALE_A_LOAD,
                                           SCALE_B_LAYOUT, SCALE_B_LOAD, BLOCK_M, BLOCK_N, CTA_N)
        b1_lo = _mxfp8_load_b2_parent(b_buf, data_slot, 0, 128, B2_LOAD_LAYOUT, DOT_B)
        b1_hi = _mxfp8_load_b2_parent(b_buf, data_slot, 128, 128, B2_LOAD_LAYOUT, DOT_B)
    with gl.amd.warp_pipeline_stage('b2_tail_compute_high'):
        acc0 = gl.amd.cdna5.wmma_scaled(a1, as1, 'e4m3', b1_lo, bs1_lo, 'e4m3', acc0)
        acc1 = gl.amd.cdna5.wmma_scaled(a1, as1, 'e4m3', b1_hi, bs1_hi, 'e4m3', acc1)
    return (acc0, acc1)


@gluon.jit
def _consume_and_refill(a_buf, b_buf, as_buf, bs_buf, data_slot, scale_slot, refill_scale_slot, a_desc, b_desc,
                        as_desc, bs_desc, acc0, acc1, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
                        B2_LOAD_LAYOUT: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr, SCALE_A_LOAD: gl.constexpr,
                        SCALE_B_LAYOUT: gl.constexpr, SCALE_B_LOAD: gl.constexpr, BLOCK_M: gl.constexpr,
                        BLOCK_N: gl.constexpr, CTA_N: gl.constexpr, REFILL_SCALE: gl.constexpr,
                        DATA_SCALE_WAITCNT: gl.constexpr):
    # upstream _mxfp8_bk256_b2_consume_and_refill with the cluster arrive/wait removed (1 CTA)
    tdm.async_wait(DATA_SCALE_WAITCNT)
    next_a_desc = a_desc
    next_b_desc = b_desc
    next_as_desc = as_desc
    next_bs_desc = bs_desc
    with gl.amd.warp_pipeline_stage('mxfp8_stage0_load_low', phase_gap=1):
        as0, bs0_lo, bs0_hi = _load_scales(as_buf, bs_buf, scale_slot, 0, SCALE_A_LAYOUT, SCALE_A_LOAD,
                                           SCALE_B_LAYOUT, SCALE_B_LOAD, BLOCK_M, BLOCK_N, CTA_N)
        a0 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(0, 128, 1).load(layout=DOT_A)
        b0_lo = _mxfp8_load_b2_parent(b_buf, data_slot, 0, 0, B2_LOAD_LAYOUT, DOT_B)
        b0_hi = _mxfp8_load_b2_parent(b_buf, data_slot, 128, 0, B2_LOAD_LAYOUT, DOT_B)
    with gl.amd.warp_pipeline_stage('mxfp8_stage1_compute_low'):
        acc0 = gl.amd.cdna5.wmma_scaled(a0, as0, 'e4m3', b0_lo, bs0_lo, 'e4m3', acc0)
        acc1 = gl.amd.cdna5.wmma_scaled(a0, as0, 'e4m3', b0_hi, bs0_hi, 'e4m3', acc1)
    with gl.amd.warp_pipeline_stage('mxfp8_stage2_load_high_advance'):
        a1 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(128, 128, 1).load(layout=DOT_A)
        as1, bs1_lo, bs1_hi = _load_scales(as_buf, bs_buf, scale_slot, 4, SCALE_A_LAYOUT, SCALE_A_LOAD,
                                           SCALE_B_LAYOUT, SCALE_B_LOAD, BLOCK_M, BLOCK_N, CTA_N)
        b1_lo = _mxfp8_load_b2_parent(b_buf, data_slot, 0, 128, B2_LOAD_LAYOUT, DOT_B)
        b1_hi = _mxfp8_load_b2_parent(b_buf, data_slot, 128, 128, B2_LOAD_LAYOUT, DOT_B)
        next_a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, 256])
        next_b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 0, 256])
        if REFILL_SCALE:
            next_as_desc = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, 1024])
            next_bs_desc = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, 1024])
    with gl.amd.warp_pipeline_stage('mxfp8_stage3_compute_high'):
        acc0 = gl.amd.cdna5.wmma_scaled(a1, as1, 'e4m3', b1_lo, bs1_lo, 'e4m3', acc0)
        acc1 = gl.amd.cdna5.wmma_scaled(a1, as1, 'e4m3', b1_hi, bs1_hi, 'e4m3', acc1)
    with gl.amd.warp_pipeline_stage('mxfp8_stage4_wait_refill'):
        _mxfp8_issue_b2_parent_data_without_update(a_desc, b_desc, a_buf, b_buf, data_slot)
        if REFILL_SCALE:
            tdm.async_load_fused([(as_desc, as_buf.index(refill_scale_slot), 3),
                                  (bs_desc, bs_buf.index(refill_scale_slot), 12)])
        a_desc = next_a_desc
        b_desc = next_b_desc
        as_desc = next_as_desc
        bs_desc = next_bs_desc
    return (a_desc, b_desc, as_desc, bs_desc, acc0, acc1)


@gluon.jit
def _store_half(c_ptr, desc, shared, pid_m, pid_n, half: gl.constexpr, stride_cm: gl.constexpr,
                stride_cn: gl.constexpr, acc, WMMA_LAYOUT: gl.constexpr, ACCUM: gl.constexpr):
    cta_m: gl.constexpr = 256
    half_n: gl.constexpr = 128
    out = acc.reshape((1, cta_m, 1, half_n)).permute((0, 2, 1, 3))
    if ACCUM:  # beta = 1: C = old C + acc (fp32 add, one rounding into C's dtype)
        # old C half-tile: TDM global -> LDS (same buffer / descriptor as the store), LDS -> registers in the
        # accumulator's layout (mirror of the store path), add, then the normal store.
        gl.barrier()  # LDS buffer aliases the main-loop operand buffers / the previous half's store source
        tdm.async_load(desc, [pid_m, pid_n, 0, half * half_n], shared)
        tdm.async_wait(0)
        gl.barrier()
        old = shared.load(layout=out.type.layout)
        gl.barrier()
        if ACCUM == 2:  # fuse2: round the product to C's dtype first (= torch's bf16 add of two bf16 tensors)
            out = out.to(c_ptr.type.element_ty).to(gl.float32)
        out = out + old.to(gl.float32)
    shared.store(out.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [pid_m, pid_n, 0, half * half_n], shared)


@gluon.jit
def _store_split(c_ptr, c1_ptr, shared, shared_layout: gl.constexpr, pid_m, pid_n, M: gl.constexpr, N: gl.constexpr,
                 stride_cm: gl.constexpr, stride_c1m: gl.constexpr, SPLIT_AXIS: gl.constexpr, SPLIT0: gl.constexpr,
                 acc0, acc1, WMMA_LAYOUT: gl.constexpr, ACCUM: gl.constexpr, CTA_M: gl.constexpr,
                 CTA_N: gl.constexpr):
    # Same two half-tile stores as the default epilogue, through the descriptor of the output owning the tile.
    cta_m: gl.constexpr = 256
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    if SPLIT_AXIS == 1:
        M0: gl.constexpr = SPLIT0
        N0: gl.constexpr = N
        M1: gl.constexpr = M - SPLIT0
        N1: gl.constexpr = N
        first = pid_m * cta_m < SPLIT0
    else:
        M0: gl.constexpr = M
        N0: gl.constexpr = SPLIT0
        M1: gl.constexpr = M
        N1: gl.constexpr = N - SPLIT0
        first = pid_n * cta_n < SPLIT0
    if first:
        desc0 = tdm.make_tensor_descriptor(base=c_ptr, shape=(M0 // cta_m, N0 // cta_n, cta_m, cta_n),
                                           strides=(cta_m * stride_cm, cta_n, stride_cm, 1),
                                           block_shape=(CTA_M, CTA_N, cta_m, half_n), layout=shared_layout)
        _store_half(c_ptr, desc0, shared, pid_m, pid_n, 0, stride_cm, 1, acc0, WMMA_LAYOUT, ACCUM)
        tdm.async_wait(0)
        _store_half(c_ptr, desc0, shared, pid_m, pid_n, 1, stride_cm, 1, acc1, WMMA_LAYOUT, ACCUM)
        tdm.async_wait(0)
    else:
        if SPLIT_AXIS == 1:
            pm1 = pid_m - SPLIT0 // cta_m
            pn1 = pid_n
        else:
            pm1 = pid_m
            pn1 = pid_n - SPLIT0 // cta_n
        desc1 = tdm.make_tensor_descriptor(base=c1_ptr, shape=(M1 // cta_m, N1 // cta_n, cta_m, cta_n),
                                           strides=(cta_m * stride_c1m, cta_n, stride_c1m, 1),
                                           block_shape=(CTA_M, CTA_N, cta_m, half_n), layout=shared_layout)
        _store_half(c1_ptr, desc1, shared, pm1, pn1, 0, stride_c1m, 1, acc0, WMMA_LAYOUT, ACCUM)
        tdm.async_wait(0)
        _store_half(c1_ptr, desc1, shared, pm1, pn1, 1, stride_c1m, 1, acc1, WMMA_LAYOUT, ACCUM)
        tdm.async_wait(0)


@gluon.jit
def _xcd_pids(pid, M: gl.constexpr, N: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
              GRID_MN: gl.constexpr, NUM_XCDS_: gl.constexpr, GROUP_M: gl.constexpr):
    # _get_xcd_swizzled_pids with the program id passed in (dual launch: pid local to its GEMM)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    if NUM_XCDS_ != 1:
        pid = _chiplet_transform(pid, GRID_MN, NUM_XCDS_)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    return (pid_m, pid_n)


@gluon.jit
def _gemm_tile(pid_m, pid_n, a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M: gl.constexpr, N: gl.constexpr,
               K: gl.constexpr, stride_am: gl.constexpr, stride_bn: gl.constexpr, stride_cm: gl.constexpr,
               stride_scale_a: gl.constexpr, stride_scale_b: gl.constexpr, ACCUM: gl.constexpr,
               SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B2: gl.constexpr, SHARED_SCALE_A: gl.constexpr,
               SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr, LOAD_WMMA_LAYOUT: gl.constexpr,
               B2_LOAD_LAYOUT: gl.constexpr, OUTPUT_CGA_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr,
               BLOCK_N: gl.constexpr, CTA_M: gl.constexpr, CTA_N: gl.constexpr, UNROLL: gl.constexpr = False,
               EPI2: gl.constexpr = False, c1_ptr=None, stride_c1m: gl.constexpr = 0, SPLIT_AXIS: gl.constexpr = 0,
               SPLIT0: gl.constexpr = 0):
    # One 256x256 output tile (pid_m, pid_n) of C[M,N] = A[M,K] . B[N,K]^T: static-shape body of dsv4_gemm.py
    # (gemmx/dsv4up, measured version): K-contiguous A/B (stride_ak = stride_bk = 1), row-major C (stride_cn = 1).
    # Host guarantees M, N % 256 == 0, K % 512 == 0, K >= 1024.
    stride_ak: gl.constexpr = 1
    stride_bk: gl.constexpr = 1
    stride_cn: gl.constexpr = 1
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    dot_b_load: gl.constexpr = gl.DotOperandLayout(1, LOAD_WMMA_LAYOUT, 16)
    scale_a_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_a, [BLOCK_M, 4])
    scale_b_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_b, [BLOCK_N // 2, 4])
    scale_b_load: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_b_load, [BLOCK_N, 4])
    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [2, BLOCK_M, 256], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [2, CTA_N, 256, 256], SHARED_LAYOUT_B2)
    scale_slots: gl.constexpr = 3
    as_buf = gl.allocate_shared_memory(a_scale_ptr.type.element_ty, [scale_slots, BLOCK_M // 128, 1024], SHARED_SCALE_A)
    bs_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty, [scale_slots, BLOCK_N // 128, 1024], SHARED_SCALE_B)
    a_desc = tdm.make_tensor_descriptor(base=a_ptr + pid_m * BLOCK_M * stride_am, shape=(M, K),
                                        strides=(stride_am, stride_ak), block_shape=(BLOCK_M, 256),
                                        layout=SHARED_LAYOUT_A)
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn
    b_desc = tdm.make_tensor_descriptor(base=b_base, shape=(CTA_N, N // CTA_N, K),
                                        strides=(256 * stride_bn, stride_bn, stride_bk), block_shape=(CTA_N, 256, 256),
                                        layout=SHARED_LAYOUT_B2)
    as_desc = tdm.make_tensor_descriptor(base=a_scale_ptr + pid_m * BLOCK_M // 128 * stride_scale_a,
                                         shape=(M // 128, K // 32 * 128), strides=(stride_scale_a, 1),
                                         block_shape=(BLOCK_M // 128, 1024), layout=SHARED_SCALE_A)
    bs_desc = tdm.make_tensor_descriptor(base=b_scale_ptr + pid_n * BLOCK_N // 128 * stride_scale_b,
                                         shape=(N // 128, K // 32 * 128), strides=(stride_scale_b, 1),
                                         block_shape=(BLOCK_N // 128, 1024), layout=SHARED_SCALE_B)
    as_desc, bs_desc = _mxfp8_issue_leading4_scale(as_desc, bs_desc, as_buf, bs_buf, 0)
    a_desc, b_desc = _mxfp8_issue_b2_parent_data(a_desc, b_desc, a_buf, b_buf, 0)
    as_desc, bs_desc = _mxfp8_issue_leading4_scale(as_desc, bs_desc, as_buf, bs_buf, 1)
    a_desc, b_desc = _mxfp8_issue_b2_parent_data(a_desc, b_desc, a_buf, b_buf, 1)
    as_desc, bs_desc = _mxfp8_issue_leading4_scale(as_desc, bs_desc, as_buf, bs_buf, 2)
    acc0 = gl.zeros((BLOCK_M, BLOCK_N // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    acc1 = gl.zeros((BLOCK_M, BLOCK_N // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max: gl.constexpr = K // 256
    data_scale_waitcnt: gl.constexpr = 4
    if UNROLL:  # fully unrolled K loop (dual launch: a 2nd runtime K loop in the other branch doubles the LDS)
        for _ in gl.static_range((iter_max - 4) // 6):
            for inner in gl.static_range(6):
                a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _consume_and_refill(
                    a_buf, b_buf, as_buf, bs_buf, inner % 2, inner % 3, inner % 3, a_desc, b_desc, as_desc, bs_desc, acc0,
                    acc1, dot_a, dot_b, B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load,
                    BLOCK_M, BLOCK_N, CTA_N, True, data_scale_waitcnt)
        for cleanup_pair in gl.static_range((iter_max - 4) % 6 // 2):
            cleanup_tile = cleanup_pair * 2
            s0 = cleanup_tile % 3
            s1 = (cleanup_tile + 1) % 3
            a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _consume_and_refill(
                a_buf, b_buf, as_buf, bs_buf, 0, s0, s0, a_desc, b_desc, as_desc, bs_desc, acc0, acc1, dot_a, dot_b,
                B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load, BLOCK_M, BLOCK_N, CTA_N, True,
                data_scale_waitcnt)
            a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _consume_and_refill(
                a_buf, b_buf, as_buf, bs_buf, 1, s1, s1, a_desc, b_desc, as_desc, bs_desc, acc0, acc1, dot_a, dot_b,
                B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load, BLOCK_M, BLOCK_N, CTA_N, True,
                data_scale_waitcnt)
    else:
        for _ in range(0, (iter_max - 4) // 6):
            for inner in gl.static_range(6):
                a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _consume_and_refill(
                    a_buf, b_buf, as_buf, bs_buf, inner % 2, inner % 3, inner % 3, a_desc, b_desc, as_desc, bs_desc, acc0,
                    acc1, dot_a, dot_b, B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load,
                    BLOCK_M, BLOCK_N, CTA_N, True, data_scale_waitcnt)
        for cleanup_pair in range(0, (iter_max - 4) % 6 // 2):
            cleanup_tile = cleanup_pair * 2
            s0 = cleanup_tile % 3
            s1 = (cleanup_tile + 1) % 3
            a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _consume_and_refill(
                a_buf, b_buf, as_buf, bs_buf, 0, s0, s0, a_desc, b_desc, as_desc, bs_desc, acc0, acc1, dot_a, dot_b,
                B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load, BLOCK_M, BLOCK_N, CTA_N, True,
                data_scale_waitcnt)
            a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _consume_and_refill(
                a_buf, b_buf, as_buf, bs_buf, 1, s1, s1, a_desc, b_desc, as_desc, bs_desc, acc0, acc1, dot_a, dot_b,
                B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load, BLOCK_M, BLOCK_N, CTA_N, True,
                data_scale_waitcnt)
    residual = iter_max - 4
    s0 = residual % 3
    s1 = (residual + 1) % 3
    a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _consume_and_refill(
        a_buf, b_buf, as_buf, bs_buf, 0, s0, s0, a_desc, b_desc, as_desc, bs_desc, acc0, acc1, dot_a, dot_b,
        B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load, BLOCK_M, BLOCK_N, CTA_N, True,
        data_scale_waitcnt)
    a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _consume_and_refill(
        a_buf, b_buf, as_buf, bs_buf, 1, s1, s1, a_desc, b_desc, as_desc, bs_desc, acc0, acc1, dot_a, dot_b,
        B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load, BLOCK_M, BLOCK_N, CTA_N, False,
        data_scale_waitcnt)
    tdm.async_wait(3)
    acc0, acc1 = _consume_tail(a_buf, b_buf, as_buf, bs_buf, 0, (iter_max - 2) % 3, acc0, acc1, dot_a, dot_b,
                               B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load, BLOCK_M,
                               BLOCK_N, CTA_N)
    tdm.async_wait(0)
    acc0, acc1 = _consume_tail(a_buf, b_buf, as_buf, bs_buf, 1, (iter_max - 1) % 3, acc0, acc1, dot_a, dot_b,
                               B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load, BLOCK_M,
                               BLOCK_N, CTA_N)
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    # epilogue: upstream _tdm_store_split_n2 (one LDS buffer, two N halves), optional beta = 1 read-add
    cta_m: gl.constexpr = 256
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[half_n, 8]], [CTA_M, CTA_N, cta_m, half_n],
                                                                          [3, 2, 1, 0], OUTPUT_CGA_LAYOUT)
    shared = gl.allocate_shared_memory(c_ptr.type.element_ty, [CTA_M, CTA_N, cta_m, half_n], shared_layout)
    if SPLIT_AXIS != 0:
        # Split-output epilogue hook (fuseattn, PRIMUS_TURBO_CONCAT_SPLIT_EPI): C is the concatenation of two
        # row-major outputs along M (SPLIT_AXIS 1) or N (SPLIT_AXIS 2): rows / cols [0, SPLIT0) go to c_ptr
        # (stride_cm), the rest to c1_ptr (stride_c1m). SPLIT0 % 256 == 0, so a 256x256 tile lies in exactly one
        # output; only the store descriptor and the tile coordinate change, the accumulators are untouched.
        _store_split(c_ptr, c1_ptr, shared, shared_layout, pid_m, pid_n, M, N, stride_cm, stride_c1m, SPLIT_AXIS,
                     SPLIT0, acc0, acc1, WMMA_LAYOUT, ACCUM, CTA_M, CTA_N)
    else:
        desc = tdm.make_tensor_descriptor(base=c_ptr, shape=(M // cta_m, N // cta_n, cta_m, cta_n),
                                          strides=(cta_m * stride_cm, cta_n * stride_cn, stride_cm, stride_cn),
                                          block_shape=(CTA_M, CTA_N, cta_m, half_n), layout=shared_layout)
        if EPI2:
            # two LDS staging buffers (the operand ring is dead here, 2 x ~70 KB fit): half 1 is written to LDS while
            # half 0's TDM store is in flight; one wait at the end. Same values, same rounding.
            shared1 = gl.allocate_shared_memory(c_ptr.type.element_ty, [CTA_M, CTA_N, cta_m, half_n], shared_layout)
            _store_half(c_ptr, desc, shared, pid_m, pid_n, 0, stride_cm, stride_cn, acc0, WMMA_LAYOUT, ACCUM)
            _store_half(c_ptr, desc, shared1, pid_m, pid_n, 1, stride_cm, stride_cn, acc1, WMMA_LAYOUT, ACCUM)
            tdm.async_wait(0)
            shared1._keep_alive()
        else:
            _store_half(c_ptr, desc, shared, pid_m, pid_n, 0, stride_cm, stride_cn, acc0, WMMA_LAYOUT, ACCUM)
            tdm.async_wait(0)
            _store_half(c_ptr, desc, shared, pid_m, pid_n, 1, stride_cm, stride_cn, acc1, WMMA_LAYOUT, ACCUM)
            tdm.async_wait(0)


@gluon.jit
def dsv4up_gemm_kernel(a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M: gl.constexpr, N: gl.constexpr,
                       K: gl.constexpr, stride_am: gl.constexpr, stride_bn: gl.constexpr, stride_cm: gl.constexpr,
                       stride_scale_a: gl.constexpr, stride_scale_b: gl.constexpr, GRID_MN: gl.constexpr,
                       SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B2: gl.constexpr, SHARED_SCALE_A: gl.constexpr,
                       SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr, LOAD_WMMA_LAYOUT: gl.constexpr,
                       B2_LOAD_LAYOUT: gl.constexpr, OUTPUT_CGA_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr,
                       BLOCK_N: gl.constexpr, CTA_M: gl.constexpr, CTA_N: gl.constexpr, ACCUM: gl.constexpr,
                       SPLIT_K: gl.constexpr = 1, EPI2: gl.constexpr = False):
    pid_m, pid_n = _get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, 4)
    if SPLIT_K > 1:
        # split-K (wgsmall): grid axis 1 = K slice z of length K (the per-slice K; the strides are the full
        # operand's). Slice z reads A/B columns [z*K, (z+1)*K) and their packed scales (4 bytes per K column), and
        # writes fp32 partial z to C + z*M*stride_cm; the host sums the partials in a fixed order.
        pid_k = gl.program_id(axis=1)
        a_ptr = a_ptr + pid_k * K
        b_ptr = b_ptr + pid_k * K
        a_scale_ptr = a_scale_ptr + pid_k * (K * 4)
        b_scale_ptr = b_scale_ptr + pid_k * (K * 4)
        c_ptr = c_ptr + pid_k * (M * stride_cm)
    _gemm_tile(pid_m, pid_n, a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K, stride_am, stride_bn, stride_cm,
               stride_scale_a, stride_scale_b, ACCUM, SHARED_LAYOUT_A, SHARED_LAYOUT_B2, SHARED_SCALE_A, SHARED_SCALE_B, WMMA_LAYOUT,
                   LOAD_WMMA_LAYOUT, B2_LOAD_LAYOUT, OUTPUT_CGA_LAYOUT, BLOCK_M, BLOCK_N, CTA_M, CTA_N, False, EPI2)


@gluon.jit
def dsv4up_split_kernel(a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, c1_ptr, M: gl.constexpr, N: gl.constexpr,
                        K: gl.constexpr, stride_am: gl.constexpr, stride_bn: gl.constexpr, stride_cm: gl.constexpr,
                        stride_c1m: gl.constexpr, stride_scale_a: gl.constexpr, stride_scale_b: gl.constexpr,
                        GRID_MN: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B2: gl.constexpr,
                        SHARED_SCALE_A: gl.constexpr, SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr,
                        LOAD_WMMA_LAYOUT: gl.constexpr, B2_LOAD_LAYOUT: gl.constexpr, OUTPUT_CGA_LAYOUT: gl.constexpr,
                        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, CTA_M: gl.constexpr, CTA_N: gl.constexpr,
                        ACCUM: gl.constexpr, SPLIT_AXIS: gl.constexpr, SPLIT0: gl.constexpr):
    # dsv4up_gemm_kernel (same tile order and main loop) whose C is split into two outputs, see _store_split.
    pid_m, pid_n = _get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, 4)
    _gemm_tile(pid_m, pid_n, a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K, stride_am, stride_bn, stride_cm,
               stride_scale_a, stride_scale_b, ACCUM, SHARED_LAYOUT_A, SHARED_LAYOUT_B2, SHARED_SCALE_A, SHARED_SCALE_B,
               WMMA_LAYOUT, LOAD_WMMA_LAYOUT, B2_LOAD_LAYOUT, OUTPUT_CGA_LAYOUT, BLOCK_M, BLOCK_N, CTA_M, CTA_N, False,
               False, c1_ptr, stride_c1m, SPLIT_AXIS, SPLIT0)


@gluon.jit
def dsv4up_dual_kernel(a0, b0, c0, as0, bs0, a1, b1, c1, as1, bs1,
                       M0: gl.constexpr, N0: gl.constexpr, K0: gl.constexpr, SAM0: gl.constexpr, SBN0: gl.constexpr,
                       SCM0: gl.constexpr, SSA0: gl.constexpr, SSB0: gl.constexpr, GRID0: gl.constexpr,
                       ACC0: gl.constexpr,
                       M1: gl.constexpr, N1: gl.constexpr, K1: gl.constexpr, SAM1: gl.constexpr, SBN1: gl.constexpr,
                       SCM1: gl.constexpr, SSA1: gl.constexpr, SSB1: gl.constexpr, GRID1: gl.constexpr,
                       ACC1: gl.constexpr,
                       SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B2: gl.constexpr, SHARED_SCALE_A: gl.constexpr,
                       SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr, LOAD_WMMA_LAYOUT: gl.constexpr,
                       B2_LOAD_LAYOUT: gl.constexpr, OUTPUT_CGA_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr,
                       BLOCK_N: gl.constexpr, CTA_M: gl.constexpr, CTA_N: gl.constexpr, EPI2: gl.constexpr = False):
    # Two independent GEMMs in one launch (e.g. a Linear's dgrad and wgrad): programs [0, GRID0) are GEMM 0's tiles
    # (same tile order and per-tile code as dsv4up_gemm_kernel, so every output is bit-identical to its own launch),
    # [GRID0, GRID0 + GRID1) GEMM 1's. GRID0 % 8 == 0 keeps each GEMM's XCD mapping. Put the longer-K GEMM first.
    # GEMM 1's K loop is fully unrolled (same instructions): with a runtime K loop in both branches the LDS
    # allocator keeps both branches' buffers live (582 KB > 320 KB); K1 is limited on the host.
    pid = gl.program_id(axis=0)
    if pid < GRID0:
        pm, pn = _xcd_pids(pid, M0, N0, BLOCK_M, BLOCK_N, GRID0, 8, 4)
        _gemm_tile(pm, pn, a0, b0, c0, as0, bs0, M0, N0, K0, SAM0, SBN0, SCM0, SSA0, SSB0, ACC0,
                   SHARED_LAYOUT_A, SHARED_LAYOUT_B2, SHARED_SCALE_A, SHARED_SCALE_B, WMMA_LAYOUT,
                   LOAD_WMMA_LAYOUT, B2_LOAD_LAYOUT, OUTPUT_CGA_LAYOUT, BLOCK_M, BLOCK_N, CTA_M, CTA_N, False, EPI2)
    else:
        pm, pn = _xcd_pids(pid - GRID0, M1, N1, BLOCK_M, BLOCK_N, GRID1, 8, 4)
        _gemm_tile(pm, pn, a1, b1, c1, as1, bs1, M1, N1, K1, SAM1, SBN1, SCM1, SSA1, SSB1, ACC1,
                   SHARED_LAYOUT_A, SHARED_LAYOUT_B2, SHARED_SCALE_A, SHARED_SCALE_B, WMMA_LAYOUT,
                   LOAD_WMMA_LAYOUT, B2_LOAD_LAYOUT, OUTPUT_CGA_LAYOUT, BLOCK_M, BLOCK_N, CTA_M, CTA_N, True, EPI2)


@triton.jit
def _splitk_reduce_kernel(p_ptr, c_ptr, MN, S: tl.constexpr, ACCUM: tl.constexpr, BLOCK: tl.constexpr):
    # C = [old C +] sum_z partial[z] (z = 0..S-1 in order, fp32), one rounding into C's dtype
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.load(p_ptr + offs, mask=mask, other=0.0)
    for z in tl.static_range(1, S):
        acc += tl.load(p_ptr + z * MN + offs, mask=mask, other=0.0)
    if ACCUM:
        acc = tl.load(c_ptr + offs, mask=mask, other=0.0).to(tl.float32) + acc
    tl.store(c_ptr + offs, acc.to(c_ptr.dtype.element_ty), mask=mask)


# ----------------------------------------------------------------------------------------------- host side
def shape_supported(M, N, K):
    """cluster = 1 constraints: 256x256 CTA tile, BK256 with an even trip count >= 4."""
    return M > 0 and N > 0 and M % TILE == 0 and N % TILE == 0 and K >= 1024 and K % 512 == 0


_LAYOUTS = None


def build_layouts():
    """Upstream make_mxfp8_case layout construction for the 256x256x256 tile, cluster = 1."""
    tile, block_k, cluster = TILE, BLOCK_K, 1
    block_m = block_n = tile
    cga = []
    output_cga = tuple()
    slice_n = tile // 2
    local_a = gl.PaddedSharedLayout.with_identity_for([[block_k, 16]], [block_m, block_k], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for([[block_k, 16]], [block_n, block_k], [1, 0])
    _, _, local_wmma = gl.amd.cdna5.make_partitioned_dot_layouts(block_m, block_n, local_a, local_b, 8, [16, 16, 128],
                                                                 a_transposed=False, b_transposed=True, slice_m=tile,
                                                                 slice_n=slice_n)
    load_wmma = gl.amd.AMDWMMALayout(3, local_wmma.transposed, local_wmma.warp_bases, local_wmma.reg_bases,
                                     local_wmma.instr_shape, cga)
    dot_a = gl.DotOperandLayout(0, load_wmma, 16)
    dot_b = gl.DotOperandLayout(1, load_wmma, 16)
    cga_a = dot_a.cga_layout
    cga_b = tuple((basis[1], basis[0]) for basis in dot_b.cga_layout)
    padded_a = gl.PaddedSharedLayout.with_identity_for([[block_k, 16]], [block_m, block_k], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for([[block_k, 16]], [block_n, block_k], [1, 0], cga_b)
    shared_a, shared_b, _ = gl.amd.cdna5.make_partitioned_dot_layouts(tile, tile, padded_a, padded_b, 8, [16, 16, 128],
                                                                      a_transposed=False, b_transposed=True,
                                                                      slice_m=tile, slice_n=slice_n)
    packed_n = block_n // 2
    local_b = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [packed_n, 256], [1, 0])
    _, _, local_wmma = gl.amd.cdna5.make_partitioned_dot_layouts(block_m, packed_n, local_a, local_b, 8, [16, 16, 128],
                                                                 a_transposed=False, b_transposed=True, slice_m=tile,
                                                                 slice_n=slice_n)
    wmma = gl.amd.AMDWMMALayout(3, local_wmma.transposed, local_wmma.warp_bases, local_wmma.reg_bases,
                                local_wmma.instr_shape, cga)
    shared_as = gl.PaddedSharedLayout.with_identity_for([[256, 8]], [block_m // 128, 1024], [1, 0], cga_a)
    shared_bs = gl.PaddedSharedLayout.with_identity_for([[256, 8]], [block_n // 128, 1024], [1, 0], cga_b)
    inner = shared_b.partition_layout
    parent_cga = tuple((basis[0], 0, basis[1]) for basis in inner.cga_layout)
    padded = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [cluster, 64, 256], [2, 1, 0], parent_cga)
    b2_shared = PartitionedSharedLayout(2, 2, 1, padded)
    b2_load = gl.DistributedLinearLayout(
        reg_bases=[[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [32, 0, 0], [64, 0, 0], [0, 0, 16], [0, 0, 32]],
        lane_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [16, 0, 0]], warp_bases=[[0, 0, 64], [0, 0, 0],
                                                                                        [0, 0, 0]],
        block_bases=[], shape=[128, cluster, 128])
    return dict(SHARED_LAYOUT_A=shared_a, SHARED_LAYOUT_B2=b2_shared, SHARED_SCALE_A=shared_as,
                SHARED_SCALE_B=shared_bs, WMMA_LAYOUT=wmma, LOAD_WMMA_LAYOUT=load_wmma, B2_LOAD_LAYOUT=b2_load,
                OUTPUT_CGA_LAYOUT=output_cga, BLOCK_M=block_m, BLOCK_N=block_n, CTA_M=cluster, CTA_N=cluster)


# (M, N, K, stride_am, stride_bn, stride_cm, c dtype, accumulate) -> launch kwargs (static compile per key)
_PLANS = {}


def _epi2() -> bool:
    """PRIMUS_TURBO_DSV4UP_EPI2=1 (default off): double-buffered epilogue (bit-identical)."""
    import os

    return os.environ.get("PRIMUS_TURBO_DSV4UP_EPI2", "0") == "1"


def _plan(M, N, K, sam, sbn, scm, cdt, accum, split_k=1):
    epi2 = _epi2()
    key = (M, N, K, sam, sbn, scm, cdt, accum, split_k, epi2)
    p = _PLANS.get(key)
    if p is None:
        global _LAYOUTS
        if _LAYOUTS is None:
            _LAYOUTS = build_layouts()
        grid_mn = (M // TILE) * (N // TILE)
        ks = K // split_k  # per-slice K (the kernel's K); the packed-scale row stride stays the full K's
        sk = dict(SPLIT_K=split_k) if split_k > 1 else {}
        p = ((grid_mn, split_k), dict(M=M, N=N, K=ks, stride_am=sam, stride_bn=sbn, stride_cm=scm,
                                      stride_scale_a=K // 32 * 128, stride_scale_b=K // 32 * 128, GRID_MN=grid_mn,
                                      ACCUM=accum, num_warps=8, waves_per_eu=2, num_ctas=1, **sk,
                                      **(dict(EPI2=True) if epi2 else {}), **_LAYOUTS))
        _PLANS[key] = p
    return p


def dsv4up_gemm(a, b, a_packed, b_packed, out=None, out_dtype=torch.bfloat16, accumulate=False, split_k=1):
    """C[M,N] = A[M,K] . B[N,K]^T; a/b fp8 e4m3 K-contiguous; *_packed from dsv4up_pack (contiguous
    [rows/128, K*4] uint8). accumulate=True: out += A.B^T (fp32 add, one rounding into out's dtype).
    split_k > 1: split_k K slices (each a supported K) as fp32 partials in one launch, summed in slice order, then
    rounded once into the output (with accumulate: old C + sum, one rounding). Not bit-identical to split_k = 1."""
    M, K = a.shape
    N = b.shape[0]
    assert shape_supported(M, N, K), (M, N, K)
    assert b.shape[1] == K and a.stride(1) == 1 and b.stride(1) == 1
    assert a_packed.shape == (M // 128, K * 4) and b_packed.shape == (N // 128, K * 4)
    assert a_packed.is_contiguous() and b_packed.is_contiguous()
    # accumulate == 2 (fuse2 dX fan-in): C = bf16(product) + old C in fp32, one rounding -- torch's bf16 add
    acc_mode = 2 if (not isinstance(accumulate, bool) and accumulate == 2) else bool(accumulate)
    if split_k > 1:
        assert acc_mode != 2, "dsv4up: accumulate=2 needs split_k == 1"
        assert K % split_k == 0 and shape_supported(M, N, K // split_k), (M, N, K, split_k)
        part = torch.empty((split_k, M, N), dtype=torch.float32, device=a.device)
        grid, kw = _plan(M, N, K, a.stride(0), b.stride(0), N, torch.float32, False, split_k)
        dsv4up_gemm_kernel[grid](a, b, part, a_packed, b_packed, **kw)
        if out is None:
            out = torch.empty((M, N), dtype=out_dtype, device=a.device)
        assert out.is_contiguous()
        BLK = 4096
        _splitk_reduce_kernel[(triton.cdiv(M * N, BLK),)](part, out, M * N, S=split_k, ACCUM=bool(accumulate),
                                                           BLOCK=BLK, num_warps=8)
        return out
    if out is None:
        assert not accumulate
        out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    assert tuple(out.shape) == (M, N) and out.stride(1) == 1
    grid, kw = _plan(M, N, K, a.stride(0), b.stride(0), out.stride(0), out.dtype, acc_mode)
    dsv4up_gemm_kernel[grid](a, b, out, a_packed, b_packed, **kw)
    return out


def _check_job(a, b, a_packed, b_packed, out):
    M, K = a.shape
    N = b.shape[0]
    assert shape_supported(M, N, K), (M, N, K)
    assert b.shape[1] == K and a.stride(1) == 1 and b.stride(1) == 1
    assert a_packed.shape == (M // 128, K * 4) and b_packed.shape == (N // 128, K * 4)
    assert a_packed.is_contiguous() and b_packed.is_contiguous()
    assert tuple(out.shape) == (M, N) and out.stride(1) == 1
    for t in (a, b, a_packed, b_packed, out):  # never hand the kernel a null / foreign address
        assert t.is_cuda and t.data_ptr() != 0 and t.device == a.device, "dsv4up: bad operand"
    return M, N, K


_DUAL_PLANS = {}
DUAL_MAX_K1 = 8192


def dsv4up_dual(job0, job1):
    """Two independent GEMMs in ONE launch; job = (a, b, a_packed, b_packed, out, accumulate) as in dsv4up_gemm
    (``out`` given, split_k 1). Each output is bit-identical to its own dsv4up_gemm launch (same per-tile code and
    tile order). GEMM 0's tiles get the lower program ids: pass the longer-K GEMM as job0."""
    a0, b0, pa0, pb0, o0, acc0 = job0
    a1, b1, pa1, pb1, o1, acc1 = job1
    M0, N0, K0 = _check_job(a0, b0, pa0, pb0, o0)
    M1, N1, K1 = _check_job(a1, b1, pa1, pb1, o1)
    epi2 = _epi2()
    key = (M0, N0, K0, a0.stride(0), b0.stride(0), o0.stride(0), o0.dtype, bool(acc0),
           M1, N1, K1, a1.stride(0), b1.stride(0), o1.stride(0), o1.dtype, bool(acc1), epi2)
    p = _DUAL_PLANS.get(key)
    if p is None:
        global _LAYOUTS
        if _LAYOUTS is None:
            _LAYOUTS = build_layouts()
        g0, g1 = (M0 // TILE) * (N0 // TILE), (M1 // TILE) * (N1 // TILE)
        assert g0 % 8 == 0, "dsv4up_dual: GEMM 0 tile count must be a multiple of 8 (XCD mapping)"
        assert K1 <= DUAL_MAX_K1, "dsv4up_dual: GEMM 1's K loop is unrolled; K1 <= DUAL_MAX_K1"
        p = ((g0 + g1,), dict(M0=M0, N0=N0, K0=K0, SAM0=a0.stride(0), SBN0=b0.stride(0), SCM0=o0.stride(0),
                              SSA0=K0 // 32 * 128, SSB0=K0 // 32 * 128, GRID0=g0, ACC0=bool(acc0),
                              M1=M1, N1=N1, K1=K1, SAM1=a1.stride(0), SBN1=b1.stride(0), SCM1=o1.stride(0),
                              SSA1=K1 // 32 * 128, SSB1=K1 // 32 * 128, GRID1=g1, ACC1=bool(acc1),
                              num_warps=8, waves_per_eu=2, num_ctas=1, **(dict(EPI2=True) if epi2 else {}),
                              **_LAYOUTS))
        _DUAL_PLANS[key] = p
    grid, kw = p
    dsv4up_dual_kernel[grid](a0, b0, o0, pa0, pb0, a1, b1, o1, pa1, pb1, **kw)
    return o0, o1


_SPLIT_PLANS = {}


def dsv4up_gemm_split(a, b, a_packed, b_packed, out0, out1, axis, accumulate=False):
    """One dsv4up GEMM C[M,N] = A[M,K] . B[N,K]^T whose C is two separate row-major outputs (fuseattn split-output
    epilogue): axis "n" -> out0 [M, N0], out1 [M, N - N0] (columns); axis "m" -> out0 [M0, N], out1 [M - M0, N]
    (rows). The split point (N0 / M0) is a multiple of 256, so every 256x256 tile is stored whole into one output; the
    tiles' accumulators are exactly dsv4up_gemm's (same main loop), so each output has the bits of the GEMM restricted
    to its rows / columns. accumulate: beta 1 into both outputs (same dtype)."""
    M, K = a.shape
    N = b.shape[0]
    assert shape_supported(M, N, K), (M, N, K)
    assert b.shape[1] == K and a.stride(1) == 1 and b.stride(1) == 1
    assert a_packed.shape == (M // 128, K * 4) and b_packed.shape == (N // 128, K * 4)
    assert a_packed.is_contiguous() and b_packed.is_contiguous()
    assert out0.dtype == out1.dtype and out0.stride(1) == 1 and out1.stride(1) == 1
    if axis == "n":
        s0 = out0.shape[1]
        assert out0.shape[0] == M and out1.shape[0] == M and s0 + out1.shape[1] == N, (out0.shape, out1.shape, M, N)
        ax = 2
    else:
        s0 = out0.shape[0]
        assert out0.shape[1] == N and out1.shape[1] == N and s0 + out1.shape[0] == M, (out0.shape, out1.shape, M, N)
        ax = 1
    assert 0 < s0 and s0 % TILE == 0, s0
    for t in (a, b, a_packed, b_packed, out0, out1):  # never hand the kernel a null / foreign address
        assert t.is_cuda and t.data_ptr() != 0 and t.device == a.device, "dsv4up split: bad operand"
    key = (M, N, K, a.stride(0), b.stride(0), out0.stride(0), out1.stride(0), out0.dtype, bool(accumulate), ax, s0)
    p = _SPLIT_PLANS.get(key)
    if p is None:
        global _LAYOUTS
        if _LAYOUTS is None:
            _LAYOUTS = build_layouts()
        grid_mn = (M // TILE) * (N // TILE)
        p = ((grid_mn,), dict(M=M, N=N, K=K, stride_am=a.stride(0), stride_bn=b.stride(0), stride_cm=out0.stride(0),
                              stride_c1m=out1.stride(0), stride_scale_a=K // 32 * 128, stride_scale_b=K // 32 * 128,
                              GRID_MN=grid_mn, ACCUM=bool(accumulate), SPLIT_AXIS=ax, SPLIT0=s0, num_warps=8,
                              waves_per_eu=2, num_ctas=1, **_LAYOUTS))
        _SPLIT_PLANS[key] = p
    grid, kw = p
    dsv4up_split_kernel[grid](a, b, out0, a_packed, b_packed, out1, **kw)
    return out0, out1
