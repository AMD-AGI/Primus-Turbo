###############################################################################
# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
"""dsv4s: smaller-output-tile variants of the dsv4up block-FP8 GEMM (gfx1250, Gluon) -- densegemm line.

Same math and the same per-output K order as dsv4up (wmma_scaled 16x16x128 per 128-wide K chunk, chunks in
ascending K, one fp32 accumulator per output), same packed scale format (dsv4up_pack: [rows/128, K*4] uint8), so
outputs are bit-identical to dsv4up; more output tiles for small-grid shapes (wq_b dgrad: 64 -> 128 / 256 tiles).
Tiles (BLOCK_M x BLOCK_N x BK): 128x128x512, 256x128x256, 128x256x256 (2 data slots, (BM+BN)*BK*2 <= 256 KB).
Main loop: upstream gemm_warp_pipeline_cdna5.py mxfp8_gemm_warp_pipeline_bk512 (five-stage warp pipeline, 2 data
slots), cluster = 1 (cluster arrive/wait dropped), packed dsv4up scales instead of raw [M, K/32].
"""

import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.cdna5 import tdm

TILES = {"128x128": (128, 128, 512, 2), "256x128": (256, 128, 256, 2), "128x256": (128, 256, 256, 2),
         "256x128s3": (256, 128, 256, 3), "128x256s3": (128, 256, 256, 3),
         "256x256": (256, 256, 256, 2)}  # (BM, BN, BK, data slots)


@gluon.jit
def _chiplet(pid, num_workgroups, num_xcds: gl.constexpr):
    xcd = pid % num_xcds
    pos_in_xcd = pid // num_xcds
    min_per_xcd = num_workgroups // num_xcds
    extra_sms = num_workgroups % num_xcds
    offset = xcd * min_per_xcd + min(xcd, extra_sms)
    return offset + pos_in_xcd


@gluon.jit
def _pids(M: gl.constexpr, N: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, GRID_MN: gl.constexpr,
          GROUP_M: gl.constexpr):
    pid = gl.program_id(axis=0)
    num_pid_m: gl.constexpr = M // BLOCK_M
    num_pid_n: gl.constexpr = N // BLOCK_N
    pid = _chiplet(pid, GRID_MN, 8)
    num_pid_in_group: gl.constexpr = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    return pid_m, pid_n


@gluon.jit
def _refill(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, slot: gl.constexpr, BK: gl.constexpr):
    tdm.async_load(a_desc, dest=a_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load(b_desc, dest=b_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load_fused([(as_desc, as_buf.index(slot), 0b00000011), (bs_desc, bs_buf.index(slot), 0b00001100)])
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BK])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BK])
    as_desc = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, BK * 4])  # packed: 4 B per K column
    bs_desc = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, BK * 4])
    return a_desc, b_desc, as_desc, bs_desc


@gluon.jit
def _scale(buf, slot: gl.constexpr, start_k: gl.constexpr, LAYOUT: gl.constexpr, ROWS: gl.constexpr,
           BK: gl.constexpr):
    # packed [ROWS/128, BK/32*128] (one BK-wide K step) -> logical [ROWS, BK/32] block-32 scales; 4 per 128 chunk
    s = buf.index(slot).reshape((ROWS // 128, BK // 128, 32, 4, 4)).permute((0, 3, 2, 1, 4)).reshape((ROWS, BK // 32))
    return s.slice(start_k, 4, 1).load(layout=LAYOUT)


@gluon.jit
def _chunk_load(a_buf, b_buf, as_buf, bs_buf, slot: gl.constexpr, k: gl.constexpr, dot_a: gl.constexpr,
                dot_b: gl.constexpr, sa: gl.constexpr, sb: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                BK: gl.constexpr):
    a = a_buf.index(slot).slice(k, 128, 1).load(layout=dot_a)
    as_ = _scale(as_buf, slot, k // 32, sa, BLOCK_M, BK)
    b = b_buf.index(slot).slice(k, 128, 1).permute([1, 0]).load(layout=dot_b)
    bs = _scale(bs_buf, slot, k // 32, sb, BLOCK_N, BK)
    return a, as_, b, bs


@gluon.jit
def _step(a_buf, b_buf, as_buf, bs_buf, slot: gl.constexpr, a_desc, b_desc, as_desc, bs_desc, acc,
          dot_a: gl.constexpr, dot_b: gl.constexpr, sa: gl.constexpr, sb: gl.constexpr, BLOCK_M: gl.constexpr,
          BLOCK_N: gl.constexpr, BK: gl.constexpr, NS: gl.constexpr):
    # upstream _mxfp8_tiled_bk512_five, 1 CTA: consume slot (its K chunks in order), then refill it NS steps ahead
    tdm.async_wait(3 * (NS - 1))
    if BK == 512:
        with gl.amd.warp_pipeline_stage('s_stage0_load_k0', phase_gap=1):
            a0, as0, b0, bs0 = _chunk_load(a_buf, b_buf, as_buf, bs_buf, slot, 0, dot_a, dot_b, sa, sb, BLOCK_M,
                                           BLOCK_N, BK)
            a1, as1, b1, bs1 = _chunk_load(a_buf, b_buf, as_buf, bs_buf, slot, 128, dot_a, dot_b, sa, sb, BLOCK_M,
                                           BLOCK_N, BK)
        with gl.amd.warp_pipeline_stage('s_stage1_compute_k0'):
            acc = gl.amd.cdna5.wmma_scaled(a0, as0, 'e4m3', b0, bs0, 'e4m3', acc)
            acc = gl.amd.cdna5.wmma_scaled(a1, as1, 'e4m3', b1, bs1, 'e4m3', acc)
        with gl.amd.warp_pipeline_stage('s_stage2_load_k2'):
            a2, as2, b2, bs2 = _chunk_load(a_buf, b_buf, as_buf, bs_buf, slot, 256, dot_a, dot_b, sa, sb, BLOCK_M,
                                           BLOCK_N, BK)
            a3, as3, b3, bs3 = _chunk_load(a_buf, b_buf, as_buf, bs_buf, slot, 384, dot_a, dot_b, sa, sb, BLOCK_M,
                                           BLOCK_N, BK)
        with gl.amd.warp_pipeline_stage('s_stage3_compute_k2'):
            acc = gl.amd.cdna5.wmma_scaled(a2, as2, 'e4m3', b2, bs2, 'e4m3', acc)
            acc = gl.amd.cdna5.wmma_scaled(a3, as3, 'e4m3', b3, bs3, 'e4m3', acc)
    else:
        with gl.amd.warp_pipeline_stage('s_stage0_load_k0', phase_gap=1):
            a0, as0, b0, bs0 = _chunk_load(a_buf, b_buf, as_buf, bs_buf, slot, 0, dot_a, dot_b, sa, sb, BLOCK_M,
                                           BLOCK_N, BK)
        with gl.amd.warp_pipeline_stage('s_stage1_compute_k0'):
            acc = gl.amd.cdna5.wmma_scaled(a0, as0, 'e4m3', b0, bs0, 'e4m3', acc)
        with gl.amd.warp_pipeline_stage('s_stage2_load_k1'):
            a1, as1, b1, bs1 = _chunk_load(a_buf, b_buf, as_buf, bs_buf, slot, 128, dot_a, dot_b, sa, sb, BLOCK_M,
                                           BLOCK_N, BK)
        with gl.amd.warp_pipeline_stage('s_stage3_compute_k1'):
            acc = gl.amd.cdna5.wmma_scaled(a1, as1, 'e4m3', b1, bs1, 'e4m3', acc)
    with gl.amd.warp_pipeline_stage('s_stage4_refill'):
        a_desc, b_desc, as_desc, bs_desc = _refill(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, slot,
                                                   BK)
    return a_desc, b_desc, as_desc, bs_desc, acc


@gluon.jit
def _tail(a_buf, b_buf, as_buf, bs_buf, slot: gl.constexpr, acc, dot_a: gl.constexpr, dot_b: gl.constexpr,
          sa: gl.constexpr, sb: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BK: gl.constexpr):
    for k in gl.static_range(0, BK, 128):
        with gl.amd.warp_pipeline_stage('s_tail_load'):
            a, as_, b, bs = _chunk_load(a_buf, b_buf, as_buf, bs_buf, slot, k, dot_a, dot_b, sa, sb, BLOCK_M, BLOCK_N,
                                        BK)
        with gl.amd.warp_pipeline_stage('s_tail_compute'):
            acc = gl.amd.cdna5.wmma_scaled(a, as_, 'e4m3', b, bs, 'e4m3', acc)
    return acc


@gluon.jit
def _epi(c_ptr, M: gl.constexpr, N: gl.constexpr, stride_cm: gl.constexpr, row0, col0, acc, c_shared,
         c_layout: gl.constexpr, WMMA: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, ACCUM: gl.constexpr):
    # one LDS tile, TDM store (beta = 1: TDM-load the old C tile, add in fp32, one rounding)
    c_desc = tdm.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(stride_cm, 1),
                                        block_shape=(BLOCK_M, BLOCK_N), layout=c_layout)
    out = acc
    if ACCUM:
        gl.barrier()
        tdm.async_load(c_desc, [row0, col0], c_shared)
        tdm.async_wait(0)
        gl.barrier()
        old = c_shared.load(layout=WMMA)
        gl.barrier()
        out = out + old.to(gl.float32)
    c_shared.store(out.to(c_ptr.type.element_ty))
    tdm.async_store(c_desc, [row0, col0], c_shared)
    tdm.async_wait(0)


@gluon.jit
def dsv4s_gemm_kernel(a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M: gl.constexpr, N: gl.constexpr,
                      K: gl.constexpr, stride_am: gl.constexpr, stride_bn: gl.constexpr, stride_cm: gl.constexpr,
                      stride_scale_a: gl.constexpr, stride_scale_b: gl.constexpr, GRID_MN: gl.constexpr,
                      SHARED_A: gl.constexpr, SHARED_B: gl.constexpr, SHARED_AS: gl.constexpr,
                      SHARED_BS: gl.constexpr, WMMA: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                      BK: gl.constexpr, NS: gl.constexpr, ACCUM: gl.constexpr, GROUP_M: gl.constexpr = 4,
                      c1_ptr=None, stride_c1m: gl.constexpr = 0, SPLIT_AXIS: gl.constexpr = 0,
                      SPLIT0: gl.constexpr = 0):
    # C[M,N] (+)= A[M,K] . B[N,K]^T; K-contiguous A/B, row-major C; M % BLOCK_M, N % BLOCK_N, K % BK == 0, K >= 3*BK
    pid_m, pid_n = _pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, GROUP_M)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA, 16)
    sa: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_a, [BLOCK_M, 4])
    sb: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_b, [BLOCK_N, 4])
    SK: gl.constexpr = BK * 4
    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [NS, BLOCK_M, BK], SHARED_A)
    b_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [NS, BLOCK_N, BK], SHARED_B)
    as_buf = gl.allocate_shared_memory(a_scale_ptr.type.element_ty, [NS, BLOCK_M // 128, SK], SHARED_AS)
    bs_buf = gl.allocate_shared_memory(b_scale_ptr.type.element_ty, [NS, BLOCK_N // 128, SK], SHARED_BS)
    a_desc = tdm.make_tensor_descriptor(base=a_ptr + pid_m * BLOCK_M * stride_am, shape=(M, K), strides=(stride_am, 1),
                                        block_shape=(BLOCK_M, BK), layout=SHARED_A)
    b_desc = tdm.make_tensor_descriptor(base=b_ptr + pid_n * BLOCK_N * stride_bn, shape=(N, K), strides=(stride_bn, 1),
                                        block_shape=(BLOCK_N, BK), layout=SHARED_B)
    as_desc = tdm.make_tensor_descriptor(base=a_scale_ptr + pid_m * (BLOCK_M // 128) * stride_scale_a,
                                         shape=(M // 128, K // 32 * 128), strides=(stride_scale_a, 1),
                                         block_shape=(BLOCK_M // 128, SK), layout=SHARED_AS)
    bs_desc = tdm.make_tensor_descriptor(base=b_scale_ptr + pid_n * (BLOCK_N // 128) * stride_scale_b,
                                         shape=(N // 128, K // 32 * 128), strides=(stride_scale_b, 1),
                                         block_shape=(BLOCK_N // 128, SK), layout=SHARED_BS)
    for s0 in gl.static_range(NS):
        a_desc, b_desc, as_desc, bs_desc = _refill(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, s0,
                                                   BK)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA)
    iter_max: gl.constexpr = K // BK
    n_main: gl.constexpr = iter_max - NS
    gl.static_assert(6 % NS == 0)
    for _ in range(0, n_main // 6):
        for inner in gl.static_range(6):
            a_desc, b_desc, as_desc, bs_desc, acc = _step(a_buf, b_buf, as_buf, bs_buf, inner % NS, a_desc, b_desc,
                                                          as_desc, bs_desc, acc, dot_a, dot_b, sa, sb, BLOCK_M,
                                                          BLOCK_N, BK, NS)
    for r in gl.static_range(n_main % 6):
        a_desc, b_desc, as_desc, bs_desc, acc = _step(a_buf, b_buf, as_buf, bs_buf, r % NS, a_desc, b_desc, as_desc,
                                                      bs_desc, acc, dot_a, dot_b, sa, sb, BLOCK_M, BLOCK_N, BK, NS)
    for t in gl.static_range(NS):  # drain: the last NS K steps, no refill
        tdm.async_wait(3 * (NS - 1 - t))
        acc = _tail(a_buf, b_buf, as_buf, bs_buf, (n_main + t) % NS, acc, dot_a, dot_b, sa, sb, BLOCK_M, BLOCK_N, BK)
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    # epilogue: one LDS tile, TDM store (beta = 1: TDM-load the old C tile, add in fp32, one rounding)
    c_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 8]], [BLOCK_M, BLOCK_N], [1, 0],
                                                                     WMMA.cga_layout)
    c_shared = gl.allocate_shared_memory(c_ptr.type.element_ty, [BLOCK_M, BLOCK_N], c_layout)
    if SPLIT_AXIS == 0:
        _epi(c_ptr, M, N, stride_cm, pid_m * BLOCK_M, pid_n * BLOCK_N, acc, c_shared, c_layout, WMMA, BLOCK_M, BLOCK_N,
             ACCUM)
    else:
        # split-output epilogue (same contract as dsv4up_gemm_split): C = [out0 | out1] along M (SPLIT_AXIS 1) or N
        # (SPLIT_AXIS 2) at SPLIT0 (a multiple of the tile edge), so each tile is stored whole into one output.
        if SPLIT_AXIS == 1:
            M0: gl.constexpr = SPLIT0
            N0: gl.constexpr = N
            M1: gl.constexpr = M - SPLIT0
            N1: gl.constexpr = N
            first = pid_m * BLOCK_M < SPLIT0
            r1 = pid_m * BLOCK_M - SPLIT0
            c1 = pid_n * BLOCK_N
        else:
            M0: gl.constexpr = M
            N0: gl.constexpr = SPLIT0
            M1: gl.constexpr = M
            N1: gl.constexpr = N - SPLIT0
            first = pid_n * BLOCK_N < SPLIT0
            r1 = pid_m * BLOCK_M
            c1 = pid_n * BLOCK_N - SPLIT0
        if first:
            _epi(c_ptr, M0, N0, stride_cm, pid_m * BLOCK_M, pid_n * BLOCK_N, acc, c_shared, c_layout, WMMA, BLOCK_M,
                 BLOCK_N, ACCUM)
        else:
            _epi(c1_ptr, M1, N1, stride_c1m, r1, c1, acc, c_shared, c_layout, WMMA, BLOCK_M, BLOCK_N, ACCUM)


def shape_supported(M, N, K, tile="128x128"):
    bm, bn, bk, ns = TILES[tile]
    return M > 0 and N > 0 and M % bm == 0 and N % bn == 0 and K % bk == 0 and K >= (ns + 1) * bk and K % 512 == 0


_LAYOUTS = {}
_PLANS = {}


def build_layouts(tile):
    block_m, block_n, block_k, nslot = TILES[tile]
    cga = []
    local_a = gl.PaddedSharedLayout.with_identity_for([[block_k, 16]], [block_m, block_k], [1, 0])
    local_b = gl.PaddedSharedLayout.with_identity_for([[block_k, 16]], [block_n, block_k], [1, 0])
    _, _, local_wmma = gl.amd.cdna5.make_partitioned_dot_layouts(block_m, block_n, local_a, local_b, 8, [16, 16, 128],
                                                                 a_transposed=False, b_transposed=True,
                                                                 slice_m=block_m, slice_n=block_n)
    load_wmma = gl.amd.AMDWMMALayout(3, local_wmma.transposed, local_wmma.warp_bases, local_wmma.reg_bases,
                                     local_wmma.instr_shape, cga)
    dot_a = gl.DotOperandLayout(0, load_wmma, 16)
    dot_b = gl.DotOperandLayout(1, load_wmma, 16)
    cga_a = dot_a.cga_layout
    cga_b = tuple((basis[1], basis[0]) for basis in dot_b.cga_layout)
    padded_a = gl.PaddedSharedLayout.with_identity_for([[block_k, 16]], [block_m, block_k], [1, 0], cga_a)
    padded_b = gl.PaddedSharedLayout.with_identity_for([[block_k, 16]], [block_n, block_k], [1, 0], cga_b)
    shared_a, shared_b, _ = gl.amd.cdna5.make_partitioned_dot_layouts(block_m, block_n, padded_a, padded_b, 8,
                                                                      [16, 16, 128], a_transposed=False,
                                                                      b_transposed=True, slice_m=block_m,
                                                                      slice_n=block_n)
    sk = block_k * 4
    shared_as = gl.PaddedSharedLayout.with_identity_for([[256, 8]], [block_m // 128, sk], [1, 0], cga_a)
    shared_bs = gl.PaddedSharedLayout.with_identity_for([[256, 8]], [block_n // 128, sk], [1, 0], cga_b)
    return dict(SHARED_A=shared_a, SHARED_B=shared_b, SHARED_AS=shared_as, SHARED_BS=shared_bs, WMMA=load_wmma,
                BLOCK_M=block_m, BLOCK_N=block_n, BK=block_k, NS=nslot)


def _plan(M, N, K, sam, sbn, scm, cdt, accum, tile="128x128", group_m=4, split=None):
    key = (M, N, K, sam, sbn, scm, cdt, accum, tile, group_m, split)
    p = _PLANS.get(key)
    if p is None:
        if tile not in _LAYOUTS:
            _LAYOUTS[tile] = build_layouts(tile)
        bm, bn, _, _ = TILES[tile]
        grid_mn = (M // bm) * (N // bn)
        p = ((grid_mn,), dict(M=M, N=N, K=K, stride_am=sam, stride_bn=sbn, stride_cm=scm,
                              stride_scale_a=K // 32 * 128, stride_scale_b=K // 32 * 128, GRID_MN=grid_mn,
                              ACCUM=accum, GROUP_M=group_m, num_warps=8, waves_per_eu=2, num_ctas=1,
                              **(dict(stride_c1m=split[0], SPLIT_AXIS=split[1], SPLIT0=split[2]) if split else {}),
                              **_LAYOUTS[tile]))
        _PLANS[key] = p
    return p


def dsv4s_gemm(a, b, a_packed, b_packed, out=None, out_dtype=torch.bfloat16, accumulate=False, tile="128x128",
               group_m=4):
    """Same contract as dsv4up_gemm (split_k 1), smaller output tiles (see TILES)."""
    M, K = a.shape
    N = b.shape[0]
    assert shape_supported(M, N, K, tile), (M, N, K, tile)
    assert b.shape[1] == K and a.stride(1) == 1 and b.stride(1) == 1
    assert a_packed.shape == (M // 128, K * 4) and b_packed.shape == (N // 128, K * 4)
    assert a_packed.is_contiguous() and b_packed.is_contiguous()
    if out is None:
        assert not accumulate
        out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    assert tuple(out.shape) == (M, N) and out.stride(1) == 1
    for t in (a, b, a_packed, b_packed, out):
        assert t.is_cuda and t.data_ptr() != 0 and t.device == a.device
    grid, kw = _plan(M, N, K, a.stride(0), b.stride(0), out.stride(0), out.dtype, bool(accumulate), tile, group_m)
    dsv4s_gemm_kernel[grid](a, b, out, a_packed, b_packed, **kw)
    return out


def dsv4s_gemm_split(a, b, a_packed, b_packed, out0, out1, axis, accumulate=False, tile="256x128s3", group_m=4):
    """dsv4up_gemm_split contract (two row-major outputs, split along "n" or "m" at a multiple of the tile edge) on
    the smaller-tile kernel. Bit-identical to dsv4up_gemm_split / dsv4up_gemm restricted to each output."""
    M, K = a.shape
    N = b.shape[0]
    bm, bn, _, _ = TILES[tile]
    assert shape_supported(M, N, K, tile), (M, N, K, tile)
    assert b.shape[1] == K and a.stride(1) == 1 and b.stride(1) == 1
    assert a_packed.shape == (M // 128, K * 4) and b_packed.shape == (N // 128, K * 4)
    assert a_packed.is_contiguous() and b_packed.is_contiguous()
    assert out0.dtype == out1.dtype and out0.stride(1) == 1 and out1.stride(1) == 1
    if axis == "n":
        s0 = out0.shape[1]
        assert out0.shape[0] == M and out1.shape[0] == M and s0 + out1.shape[1] == N, (out0.shape, out1.shape, M, N)
        ax, edge = 2, bn
    else:
        s0 = out0.shape[0]
        assert out0.shape[1] == N and out1.shape[1] == N and s0 + out1.shape[0] == M, (out0.shape, out1.shape, M, N)
        ax, edge = 1, bm
    assert 0 < s0 and s0 % edge == 0, (s0, edge)
    for t in (a, b, a_packed, b_packed, out0, out1):
        assert t.is_cuda and t.data_ptr() != 0 and t.device == a.device
    grid, kw = _plan(M, N, K, a.stride(0), b.stride(0), out0.stride(0), out0.dtype, bool(accumulate), tile, group_m,
                     (out1.stride(0), ax, s0))
    dsv4s_gemm_kernel[grid](a, b, out0, a_packed, b_packed, c1_ptr=out1, **kw)
    return out0, out1
