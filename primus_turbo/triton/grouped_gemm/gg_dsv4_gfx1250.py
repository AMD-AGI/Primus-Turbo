"""Grouped (MoE) DSv4 blockwise-FP8 GEMM for gfx1250, derived from dsv4up/dsv4_gemm.py (upstream Triton
gemm_warp_pipeline_cdna5.py mxfp8 BK256 warp-pipeline, cluster=1, static N/K/strides).  Self-contained: the
upstream helpers it needs are copied verbatim below, so the same file can live in Primus-Turbo.

C[P, N] (bf16) rows of the expert pool:  C[m, :] = (A[m, :] * sA) @ (B[e(m)] * sB[e(m)])^T
  A   [P, K]   fp8 e4m3, the whole pool (P = 32768 static: T*topk + E*256), every expert padded to 256 rows
  B   [E*N, K] fp8 e4m3 (expert weights stacked; NT, K-contiguous)
  sA  packed MX scales of A   [P/128, K/32*128]   (pack_rows(sa [P, K/32] codes))
  sB  packed MX scales of B   [E*N/128, K/32*128] (pack_rows(sb [E*N, K/32] codes))
  mt_expert int16 [P/256]: expert of every 256-row tile;  n_mt int32 [1]: live tiles (pool rows / 256)
Both tables live on the device -> CUDA-graph capturable; no host sync.

Kernels
  gg_dsv4_tile       (a) non-persistent: grid = (P/256) * (N/256); dead tiles (t >= n_mt*NT) return through a
                         scalar branch before any TDM is issued.
  gg_dsv4_persistent (b)+(c) persistent grid G (256): tile t = vpid + i*G (vpid = XCD-contiguous pid remap),
                         order M-major inside groups of 4 N tiles.  Epilogue: next tile's scale slot 0 + K-tile 0
                         are issued first, then C is staged (bf16) in the drained data stage 1 (A half -> a_buf[1],
                         B half -> b_buf[1]) and TDM-stored while that prefetch is in flight.
                     (c) SPLIT > 1: if the last round is at most G/SPLIT tiles, each tail tile is split along K into
                         SPLIT pieces; pieces add fp32 partials with L2 atomics into ws[j] and bump cnt[j] once per
                         warp (acq_rel).  The warp that sees the final ticket makes its CTA the last arriver; it
                         atomic-exchanges ws[j] with 0 (reads the sum and re-arms the workspace), resets cnt[j]
                         and writes bf16 C.  Nobody waits on anybody: no spin loop exists.
"""
import os

import triton  # noqa: F401  (import triton before torch)
import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.cdna5 import PartitionedSharedLayout, tdm

TILE = 256
NUM_XCDS = gl.constexpr(8)



# ------------------------------------------------------------------ upstream helpers (verbatim, 83c92283)
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
def _mxfp8_bk256_b2_consume_tail(a_buf, b_buf, as_buf, bs_buf, data_slot, scale_slot, acc0, acc1, DOT_A: gl.constexpr,
                                 DOT_B: gl.constexpr, B2_LOAD_LAYOUT: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr,
                                 SCALE_A_LOAD: gl.constexpr, SCALE_B_LAYOUT: gl.constexpr, SCALE_B_LOAD: gl.constexpr,
                                 BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, CTA_N: gl.constexpr):
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    packed_n: gl.constexpr = BLOCK_N // 2
    with gl.amd.warp_pipeline_stage('b2_tail_load_low'):
        a0 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(0, 128, 1).load(layout=DOT_A)
        as0 = _mxfp8_load_scale(as_buf, scale_slot, 0, SCALE_A_LOAD, BLOCK_M, 8, 4)
        as0 = gl.convert_layout(as0, SCALE_A_LAYOUT, assert_trivial=True)
        bs0 = _mxfp8_load_scale(bs_buf, scale_slot, 0, SCALE_B_LOAD, BLOCK_N, 8, 4)
        bs0_grid = bs0.reshape((CTA_N, cta_n, 4))
        bs0_lo = gl.convert_layout(
            gl.amd.slice(bs0_grid, [CTA_N, half_n, 4], [0, 0, 0]).reshape((packed_n, 4)), SCALE_B_LAYOUT,
            assert_trivial=True)
        bs0_hi = gl.convert_layout(
            gl.amd.slice(bs0_grid, [CTA_N, half_n, 4], [0, half_n, 0]).reshape((packed_n, 4)), SCALE_B_LAYOUT,
            assert_trivial=True)
        b0_lo = _mxfp8_load_b2_parent(b_buf, data_slot, 0, 0, B2_LOAD_LAYOUT, DOT_B)
        b0_hi = _mxfp8_load_b2_parent(b_buf, data_slot, 128, 0, B2_LOAD_LAYOUT, DOT_B)
    with gl.amd.warp_pipeline_stage('b2_tail_compute_low'):
        acc0 = gl.amd.cdna5.wmma_scaled(a0, as0, 'e4m3', b0_lo, bs0_lo, 'e4m3', acc0)
        acc1 = gl.amd.cdna5.wmma_scaled(a0, as0, 'e4m3', b0_hi, bs0_hi, 'e4m3', acc1)
    with gl.amd.warp_pipeline_stage('b2_tail_load_high'):
        a1 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(128, 128, 1).load(layout=DOT_A)
        as1 = _mxfp8_load_scale(as_buf, scale_slot, 4, SCALE_A_LOAD, BLOCK_M, 8, 4)
        as1 = gl.convert_layout(as1, SCALE_A_LAYOUT, assert_trivial=True)
        bs1 = _mxfp8_load_scale(bs_buf, scale_slot, 4, SCALE_B_LOAD, BLOCK_N, 8, 4)
        bs1_grid = bs1.reshape((CTA_N, cta_n, 4))
        bs1_lo = gl.convert_layout(
            gl.amd.slice(bs1_grid, [CTA_N, half_n, 4], [0, 0, 0]).reshape((packed_n, 4)), SCALE_B_LAYOUT,
            assert_trivial=True)
        bs1_hi = gl.convert_layout(
            gl.amd.slice(bs1_grid, [CTA_N, half_n, 4], [0, half_n, 0]).reshape((packed_n, 4)), SCALE_B_LAYOUT,
            assert_trivial=True)
        b1_lo = _mxfp8_load_b2_parent(b_buf, data_slot, 0, 128, B2_LOAD_LAYOUT, DOT_B)
        b1_hi = _mxfp8_load_b2_parent(b_buf, data_slot, 128, 128, B2_LOAD_LAYOUT, DOT_B)
    with gl.amd.warp_pipeline_stage('b2_tail_compute_high'):
        acc0 = gl.amd.cdna5.wmma_scaled(a1, as1, 'e4m3', b1_lo, bs1_lo, 'e4m3', acc0)
        acc1 = gl.amd.cdna5.wmma_scaled(a1, as1, 'e4m3', b1_hi, bs1_hi, 'e4m3', acc1)
    return (acc0, acc1)


# ------------------------------------------------------------------ dsv4up refill (1 CTA, cluster barrier dropped)
@gluon.jit
def _dsv4_consume_and_refill(a_buf, b_buf, as_buf, bs_buf, data_slot, scale_slot, refill_scale_slot, a_desc, b_desc,
                             as_desc, bs_desc, acc0, acc1, DOT_A: gl.constexpr, DOT_B: gl.constexpr,
                             B2_LOAD_LAYOUT: gl.constexpr, SCALE_A_LAYOUT: gl.constexpr, SCALE_A_LOAD: gl.constexpr,
                             SCALE_B_LAYOUT: gl.constexpr, SCALE_B_LOAD: gl.constexpr, BLOCK_M: gl.constexpr,
                             BLOCK_N: gl.constexpr, CTA_N: gl.constexpr, REFILL_SCALE: gl.constexpr,
                             DATA_SCALE_WAITCNT: gl.constexpr, DYN_SLOT: gl.constexpr = False,
                             SLOT_OFF: gl.constexpr = 0):
    # DYN_SLOT: scale_slot is a loop index p and the slot is (2*p + SLOT_OFF) % 3, computed inside stage 0 (the
    # warp-pipeline pass rejects any op between the previous stage and this async_wait).
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    packed_n: gl.constexpr = BLOCK_N // 2
    tdm.async_wait(DATA_SCALE_WAITCNT)
    with gl.amd.warp_pipeline_stage('mxfp8_stage0_load_low', phase_gap=1):
        if DYN_SLOT:
            scale_slot = (2 * scale_slot + SLOT_OFF) % 3
            refill_scale_slot = scale_slot
        as0 = _mxfp8_load_scale(as_buf, scale_slot, 0, SCALE_A_LOAD, BLOCK_M, 8, 4)
        as0 = gl.convert_layout(as0, SCALE_A_LAYOUT, assert_trivial=True)
        bs0 = _mxfp8_load_scale(bs_buf, scale_slot, 0, SCALE_B_LOAD, BLOCK_N, 8, 4)
        bs0_grid = bs0.reshape((CTA_N, cta_n, 4))
        bs0_lo = gl.convert_layout(
            gl.amd.slice(bs0_grid, [CTA_N, half_n, 4], [0, 0, 0]).reshape((packed_n, 4)), SCALE_B_LAYOUT,
            assert_trivial=True)
        bs0_hi = gl.convert_layout(
            gl.amd.slice(bs0_grid, [CTA_N, half_n, 4], [0, half_n, 0]).reshape((packed_n, 4)), SCALE_B_LAYOUT,
            assert_trivial=True)
        a0 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(0, 128, 1).load(layout=DOT_A)
        b0_lo = _mxfp8_load_b2_parent(b_buf, data_slot, 0, 0, B2_LOAD_LAYOUT, DOT_B)
        b0_hi = _mxfp8_load_b2_parent(b_buf, data_slot, 128, 0, B2_LOAD_LAYOUT, DOT_B)
    with gl.amd.warp_pipeline_stage('mxfp8_stage1_compute_low'):
        acc0 = gl.amd.cdna5.wmma_scaled(a0, as0, 'e4m3', b0_lo, bs0_lo, 'e4m3', acc0)
        acc1 = gl.amd.cdna5.wmma_scaled(a0, as0, 'e4m3', b0_hi, bs0_hi, 'e4m3', acc1)
    with gl.amd.warp_pipeline_stage('mxfp8_stage2_load_high_advance'):
        a1 = a_buf.index(data_slot).slice(0, BLOCK_M, 0).slice(128, 128, 1).load(layout=DOT_A)
        as1 = _mxfp8_load_scale(as_buf, scale_slot, 4, SCALE_A_LOAD, BLOCK_M, 8, 4)
        as1 = gl.convert_layout(as1, SCALE_A_LAYOUT, assert_trivial=True)
        bs1 = _mxfp8_load_scale(bs_buf, scale_slot, 4, SCALE_B_LOAD, BLOCK_N, 8, 4)
        bs1_grid = bs1.reshape((CTA_N, cta_n, 4))
        bs1_lo = gl.convert_layout(
            gl.amd.slice(bs1_grid, [CTA_N, half_n, 4], [0, 0, 0]).reshape((packed_n, 4)), SCALE_B_LAYOUT,
            assert_trivial=True)
        bs1_hi = gl.convert_layout(
            gl.amd.slice(bs1_grid, [CTA_N, half_n, 4], [0, half_n, 0]).reshape((packed_n, 4)), SCALE_B_LAYOUT,
            assert_trivial=True)
        b1_lo = _mxfp8_load_b2_parent(b_buf, data_slot, 0, 128, B2_LOAD_LAYOUT, DOT_B)
        b1_hi = _mxfp8_load_b2_parent(b_buf, data_slot, 128, 128, B2_LOAD_LAYOUT, DOT_B)
        next_a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, 256])
        next_b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 0, 256])
        next_as_desc = as_desc
        next_bs_desc = bs_desc
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
    return (next_a_desc, next_b_desc, next_as_desc, next_bs_desc, acc0, acc1)


# ------------------------------------------------------------------ grouped pieces
@gluon.jit
def _gg_tile_coords(t, n_live, NT: gl.constexpr):
    """tile index -> (mt, nt): groups of NGROUP N tiles, M-major inside a group (A tile reused by 4 neighbours)."""
    G4: gl.constexpr = 4 if NT >= 4 else NT
    per_group = n_live * G4
    g = t // per_group
    r = t - g * per_group
    mt = r // G4
    nt = g * G4 + (r - mt * G4)
    return mt, nt


@gluon.jit
def _gg_descs(a_ptr, b_ptr, as_ptr, bs_ptr, mt, e, nt, kd, ksl, M: gl.constexpr, N: gl.constexpr,
              K: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B2: gl.constexpr,
              SHARED_SCALE_A: gl.constexpr, SHARED_SCALE_B: gl.constexpr):
    """TDM descriptors of one (mt, expert e, nt) tile; data positioned at column kd, scales at k-tile ksl."""
    SK: gl.constexpr = K // 32 * 128  # packed scale row length (= scale row stride)
    a_desc = tdm.make_tensor_descriptor(base=a_ptr + mt * (256 * K), shape=(256, K), strides=(K, 1),
                                        block_shape=(256, 256), layout=SHARED_LAYOUT_A)
    b_desc = tdm.make_tensor_descriptor(base=b_ptr + (e * N + nt * 256) * K, shape=(1, 256, K),
                                        strides=(256 * K, K, 1), block_shape=(1, 256, 256), layout=SHARED_LAYOUT_B2)
    as_desc = tdm.make_tensor_descriptor(base=as_ptr + mt * (2 * SK), shape=(2, SK), strides=(SK, 1),
                                         block_shape=(2, 1024), layout=SHARED_SCALE_A)
    bs_desc = tdm.make_tensor_descriptor(base=bs_ptr + (e * (N // 128) + nt * 2) * SK, shape=(2, SK),
                                         strides=(SK, 1), block_shape=(2, 1024), layout=SHARED_SCALE_B)
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, kd])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 0, kd])
    as_desc = tdm.update_tensor_descriptor(as_desc, add_offsets=[0, ksl * 1024])
    bs_desc = tdm.update_tensor_descriptor(bs_desc, add_offsets=[0, ksl * 1024])
    return a_desc, b_desc, as_desc, bs_desc


@gluon.jit
def _gg_prologue_a(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf):
    """scale slot 0 + data slot 0 (K-tile 0)."""
    as_desc, bs_desc = _mxfp8_issue_leading4_scale(as_desc, bs_desc, as_buf, bs_buf, 0)
    a_desc, b_desc = _mxfp8_issue_b2_parent_data(a_desc, b_desc, a_buf, b_buf, 0)
    return a_desc, b_desc, as_desc, bs_desc


@gluon.jit
def _gg_prologue_b(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf):
    """scale slot 1 + data slot 1 (K-tile 1) + scale slot 2."""
    as_desc, bs_desc = _mxfp8_issue_leading4_scale(as_desc, bs_desc, as_buf, bs_buf, 1)
    a_desc, b_desc = _mxfp8_issue_b2_parent_data(a_desc, b_desc, a_buf, b_buf, 1)
    as_desc, bs_desc = _mxfp8_issue_leading4_scale(as_desc, bs_desc, as_buf, bs_buf, 2)
    return a_desc, b_desc, as_desc, bs_desc


@gluon.jit
def _gg_mainloop(a_buf, b_buf, as_buf, bs_buf, a_desc, b_desc, as_desc, bs_desc, ITERS: gl.constexpr,
                 WMMA_LAYOUT: gl.constexpr, LOAD_WMMA_LAYOUT: gl.constexpr, B2_LOAD_LAYOUT: gl.constexpr):
    """Upstream BK256 main loop + tails over ITERS K-tiles; prologue (2 data + 3 scale slots) already issued and
    the descriptors already point at K-tile 2 / scale k-tile 3."""
    gl.static_assert(ITERS >= 4 and ITERS % 2 == 0)
    BLOCK_M: gl.constexpr = 256
    BLOCK_N: gl.constexpr = 256
    CTA_N: gl.constexpr = 1
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    scale_a_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_a, [BLOCK_M, 4])
    scale_b_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_b, [BLOCK_N // 2, 4])
    dot_b_load: gl.constexpr = gl.DotOperandLayout(1, LOAD_WMMA_LAYOUT, 16)
    scale_b_load: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_b_load, [BLOCK_N, 4])
    acc0 = gl.zeros((BLOCK_M, BLOCK_N // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    acc1 = gl.zeros((BLOCK_M, BLOCK_N // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    W: gl.constexpr = 4
    for _ in range(0, (ITERS - 4) // 6):
        for inner in gl.static_range(6):
            a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _dsv4_consume_and_refill(
                a_buf, b_buf, as_buf, bs_buf, inner % 2, inner % 3, inner % 3, a_desc, b_desc, as_desc, bs_desc, acc0,
                acc1, dot_a, dot_b, B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load,
                BLOCK_M, BLOCK_N, CTA_N, True, W)
    for cleanup_pair in gl.static_range((ITERS - 4) % 6 // 2):
        s0 = (cleanup_pair * 2) % 3
        s1 = (cleanup_pair * 2 + 1) % 3
        a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _dsv4_consume_and_refill(
            a_buf, b_buf, as_buf, bs_buf, 0, s0, s0, a_desc, b_desc, as_desc, bs_desc, acc0, acc1, dot_a, dot_b,
            B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load, BLOCK_M, BLOCK_N, CTA_N, True,
            W)
        a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _dsv4_consume_and_refill(
            a_buf, b_buf, as_buf, bs_buf, 1, s1, s1, a_desc, b_desc, as_desc, bs_desc, acc0, acc1, dot_a, dot_b,
            B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load, BLOCK_M, BLOCK_N, CTA_N, True,
            W)
    r0: gl.constexpr = (ITERS - 4) % 3
    r1: gl.constexpr = (ITERS - 3) % 3
    a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _dsv4_consume_and_refill(
        a_buf, b_buf, as_buf, bs_buf, 0, r0, r0, a_desc, b_desc, as_desc, bs_desc, acc0, acc1, dot_a, dot_b,
        B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load, BLOCK_M, BLOCK_N, CTA_N, True, W)
    a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _dsv4_consume_and_refill(
        a_buf, b_buf, as_buf, bs_buf, 1, r1, r1, a_desc, b_desc, as_desc, bs_desc, acc0, acc1, dot_a, dot_b,
        B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load, BLOCK_M, BLOCK_N, CTA_N, False, W)
    tdm.async_wait(3)
    t0: gl.constexpr = (ITERS - 2) % 3
    t1: gl.constexpr = (ITERS - 1) % 3
    acc0, acc1 = _mxfp8_bk256_b2_consume_tail(a_buf, b_buf, as_buf, bs_buf, 0, t0, acc0, acc1, dot_a, dot_b,
                                              B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout,
                                              scale_b_load, BLOCK_M, BLOCK_N, CTA_N)
    tdm.async_wait(0)
    acc0, acc1 = _mxfp8_bk256_b2_consume_tail(a_buf, b_buf, as_buf, bs_buf, 1, t1, acc0, acc1, dot_a, dot_b,
                                              B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout,
                                              scale_b_load, BLOCK_M, BLOCK_N, CTA_N)
    return acc0, acc1


@gluon.jit
def _gg_mainloop_loops(a_buf, b_buf, as_buf, bs_buf, a_desc, b_desc, as_desc, bs_desc, ITERS: gl.constexpr,
                       WMMA_LAYOUT: gl.constexpr, LOAD_WMMA_LAYOUT: gl.constexpr, B2_LOAD_LAYOUT: gl.constexpr):
    """Same K schedule as _gg_mainloop, but every warp-pipeline stage sits inside an scf.for: the AMD
    warp-pipeline pass only handles flat (loop-free) stage borders in a function's top-level block, so inside the
    persistent tile loop the residual/tail steps must be loops too.  Refill steps s = 0..ITERS-3 in pairs (data
    slot static, scale slot s % 3 dynamic); the last one also refills scale k-tile ITERS, which is outside the
    descriptor bounds (TDM drops it, the slot is never read).  Then the two tails as a 2-trip loop."""
    gl.static_assert(ITERS >= 4 and ITERS % 2 == 0)
    BLOCK_M: gl.constexpr = 256
    BLOCK_N: gl.constexpr = 256
    CTA_N: gl.constexpr = 1
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    scale_a_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_a, [BLOCK_M, 4])
    scale_b_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_b, [BLOCK_N // 2, 4])
    dot_b_load: gl.constexpr = gl.DotOperandLayout(1, LOAD_WMMA_LAYOUT, 16)
    scale_b_load: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_b_load, [BLOCK_N, 4])
    acc0 = gl.zeros((BLOCK_M, BLOCK_N // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    acc1 = gl.zeros((BLOCK_M, BLOCK_N // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    W: gl.constexpr = 4
    for p in range(0, (ITERS - 2) // 2):
        a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _dsv4_consume_and_refill(
            a_buf, b_buf, as_buf, bs_buf, 0, p, p, a_desc, b_desc, as_desc, bs_desc, acc0, acc1, dot_a, dot_b,
            B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load, BLOCK_M, BLOCK_N, CTA_N, True,
            W, True, 0)
        a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _dsv4_consume_and_refill(
            a_buf, b_buf, as_buf, bs_buf, 1, p, p, a_desc, b_desc, as_desc, bs_desc, acc0, acc1, dot_a, dot_b,
            B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout, scale_b_layout, scale_b_load, BLOCK_M, BLOCK_N, CTA_N, True,
            W, True, 1)
    for i in range(0, 2):
        tdm.async_wait(0)
        acc0, acc1 = _mxfp8_bk256_b2_consume_tail(a_buf, b_buf, as_buf, bs_buf, i, (ITERS - 2 + i) % 3, acc0, acc1,
                                                  dot_a, dot_b, B2_LOAD_LAYOUT, scale_a_layout, scale_a_layout,
                                                  scale_b_layout, scale_b_load, BLOCK_M, BLOCK_N, CTA_N)
    return acc0, acc1


@gluon.jit
def _gg_c_desc(c_ptr, M: gl.constexpr, N: gl.constexpr, CN: gl.constexpr, C_LAYOUT: gl.constexpr):
    return tdm.make_tensor_descriptor(base=c_ptr, shape=(M // 256, N // 256, 256, 256),
                                      strides=(256 * N, 256, N, 1), block_shape=(1, 1, 256, CN), layout=C_LAYOUT)


@gluon.jit
def _gg_tile_offs(LD: gl.constexpr, WMMA_LAYOUT: gl.constexpr):
    """int32 element offsets of a [256, 128] accumulator half inside a row-major [*, LD] matrix."""
    rows = gl.arange(0, 256, layout=gl.SliceLayout(1, WMMA_LAYOUT))
    cols = gl.arange(0, 128, layout=gl.SliceLayout(0, WMMA_LAYOUT))
    return rows[:, None] * LD + cols[None, :]


@gluon.jit
def _gg_store_c_direct(c_ptr, mt, nt, acc0, acc1, N: gl.constexpr, WMMA_LAYOUT: gl.constexpr):
    """Persistent epilogue: bf16 C straight from the WMMA registers with buffer stores (32-bit offsets, no wait).
    The data stages are already owned by the next tile's prefetch (issued by the caller before this) and the
    partitioned A/B stage layouts cannot be reinterpreted as a C staging buffer (MemDescReinterpretOp rejects
    PartitionedSharedEncoding); the remaining LDS (36 KB) is too small for a staged half and the WMMA layout
    cannot be register-sliced into quarters.  The stores drain behind the next tile's main loop."""
    offs = _gg_tile_offs(N, WMMA_LAYOUT)
    cbase = c_ptr + mt * (256 * N) + nt * 256
    gl.amd.cdna3.buffer_store(acc0.to(gl.bfloat16), cbase, offs)
    gl.amd.cdna3.buffer_store(acc1.to(gl.bfloat16), cbase + 128, offs)


@gluon.jit
def _gg_store_c_own(c_ptr, mt, nt, acc0, acc1, M: gl.constexpr, N: gl.constexpr, OUTPUT_CGA_LAYOUT: gl.constexpr):
    """Upstream _tdm_store_split_n2 for cluster 1 (own buffer, aliased by the allocator with the dead stages)."""
    C_LAYOUT: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[128, 8]], [1, 1, 256, 128], [3, 2, 1, 0],
                                                                     OUTPUT_CGA_LAYOUT)
    shared = gl.allocate_shared_memory(gl.bfloat16, [1, 1, 256, 128], C_LAYOUT)
    desc = _gg_c_desc(c_ptr, M, N, 128, C_LAYOUT)
    shared.store(acc0.reshape((1, 256, 1, 128)).permute((0, 2, 1, 3)).to(gl.bfloat16))
    tdm.async_store(desc, [mt, nt, 0, 0], shared)
    tdm.async_wait(0)
    shared.store(acc1.reshape((1, 256, 1, 128)).permute((0, 2, 1, 3)).to(gl.bfloat16))
    tdm.async_store(desc, [mt, nt, 0, 128], shared)
    tdm.async_wait(0)


@gluon.jit
def _gg_alloc(a_ptr, b_ptr, as_ptr, bs_ptr, SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B2: gl.constexpr,
              SHARED_SCALE_A: gl.constexpr, SHARED_SCALE_B: gl.constexpr):
    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [2, 256, 256], SHARED_LAYOUT_A)
    b_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [2, 1, 256, 256], SHARED_LAYOUT_B2)
    as_buf = gl.allocate_shared_memory(as_ptr.type.element_ty, [3, 2, 1024], SHARED_SCALE_A)
    bs_buf = gl.allocate_shared_memory(bs_ptr.type.element_ty, [3, 2, 1024], SHARED_SCALE_B)
    return a_buf, b_buf, as_buf, bs_buf


# ------------------------------------------------------------------ (a) non-persistent
@gluon.jit
def gg_dsv4_tile(a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr, mte_ptr, nmt_ptr, M: gl.constexpr, N: gl.constexpr,
                 K: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B2: gl.constexpr,
                 SHARED_SCALE_A: gl.constexpr, SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr,
                 LOAD_WMMA_LAYOUT: gl.constexpr, B2_LOAD_LAYOUT: gl.constexpr, OUTPUT_CGA_LAYOUT: gl.constexpr,
                 BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, CTA_M: gl.constexpr, CTA_N: gl.constexpr):
    NT: gl.constexpr = N // 256
    GRID: gl.constexpr = (M // 256) * NT
    pid = gl.program_id(axis=0)
    xcd = pid % NUM_XCDS  # upstream chiplet_transform: each XCD gets a contiguous range of tiles
    t = xcd * (GRID // NUM_XCDS) + gl.minimum(xcd, GRID % NUM_XCDS) + pid // NUM_XCDS
    n_live = gl.load(nmt_ptr)
    if t >= n_live * NT:  # dead tile: scalar branch, nothing issued
        return
    mt, nt = _gg_tile_coords(t, n_live, NT)
    e = gl.load(mte_ptr + mt).to(gl.int32)
    a_buf, b_buf, as_buf, bs_buf = _gg_alloc(a_ptr, b_ptr, as_ptr, bs_ptr, SHARED_LAYOUT_A, SHARED_LAYOUT_B2,
                                             SHARED_SCALE_A, SHARED_SCALE_B)
    a_desc, b_desc, as_desc, bs_desc = _gg_descs(a_ptr, b_ptr, as_ptr, bs_ptr, mt, e, nt, 0, 0, M, N, K,
                                                 SHARED_LAYOUT_A, SHARED_LAYOUT_B2, SHARED_SCALE_A, SHARED_SCALE_B)
    a_desc, b_desc, as_desc, bs_desc = _gg_prologue_a(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf)
    a_desc, b_desc, as_desc, bs_desc = _gg_prologue_b(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf)
    acc0, acc1 = _gg_mainloop(a_buf, b_buf, as_buf, bs_buf, a_desc, b_desc, as_desc, bs_desc, K // 256, WMMA_LAYOUT,
                              LOAD_WMMA_LAYOUT, B2_LOAD_LAYOUT)
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    _gg_store_c_own(c_ptr, mt, nt, acc0, acc1, M, N, OUTPUT_CGA_LAYOUT)


# ------------------------------------------------------------------ (b)+(c) persistent
@gluon.jit
def _gg_split_piece(a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr, mte_ptr, ws_ptr, wsi_ptr, cnt_ptr, n_live, n_full, u, a_buf, b_buf,
                    as_buf, bs_buf, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, SPLIT: gl.constexpr,
                    SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B2: gl.constexpr, SHARED_SCALE_A: gl.constexpr,
                    SHARED_SCALE_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr, LOAD_WMMA_LAYOUT: gl.constexpr,
                    B2_LOAD_LAYOUT: gl.constexpr, OUTPUT_CGA_LAYOUT: gl.constexpr, NUM_WARPS: gl.constexpr,
                    FLAT: gl.constexpr = False):
    NT: gl.constexpr = N // 256
    KS: gl.constexpr = K // SPLIT
    j = u // SPLIT
    piece = u - j * SPLIT
    t = n_full + j
    mt, nt = _gg_tile_coords(t, n_live, NT)
    e = gl.load(mte_ptr + mt).to(gl.int32)
    a_desc, b_desc, as_desc, bs_desc = _gg_descs(a_ptr, b_ptr, as_ptr, bs_ptr, mt, e, nt, piece * KS,
                                                 piece * (KS // 256), M, N, K, SHARED_LAYOUT_A, SHARED_LAYOUT_B2,
                                                 SHARED_SCALE_A, SHARED_SCALE_B)
    a_desc, b_desc, as_desc, bs_desc = _gg_prologue_a(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf)
    a_desc, b_desc, as_desc, bs_desc = _gg_prologue_b(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf)
    if FLAT:  # top-level of its own kernel: the static (flat) upstream schedule, as kernel (a)
        acc0, acc1 = _gg_mainloop(a_buf, b_buf, as_buf, bs_buf, a_desc, b_desc, as_desc, bs_desc, KS // 256,
                                  WMMA_LAYOUT, LOAD_WMMA_LAYOUT, B2_LOAD_LAYOUT)
    else:
        acc0, acc1 = _gg_mainloop_loops(a_buf, b_buf, as_buf, bs_buf, a_desc, b_desc, as_desc, bs_desc, KS // 256,
                                        WMMA_LAYOUT, LOAD_WMMA_LAYOUT, B2_LOAD_LAYOUT)
    # fp32 partials -> ws[j] (L2 atomics, relaxed); one acq_rel ticket per warp after the warp's own atomics.
    offs = _gg_tile_offs(256, WMMA_LAYOUT)
    wsj = ws_ptr + j * 65536
    gl.amd.cdna4.buffer_atomic_add(wsj, offs, acc0, sem="relaxed", scope="gpu")
    gl.amd.cdna4.buffer_atomic_add(wsj + 128, offs, acc1, sem="relaxed", scope="gpu")
    TL: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    wid = gl.arange(0, NUM_WARPS, layout=TL)  # one element per warp (lanes replicate -> one atomic per warp)
    old = gl.atomic_add(cnt_ptr + j + wid * 0, gl.full([NUM_WARPS], 1, gl.int32, TL), sem="acq_rel", scope="gpu")
    last = gl.max(old, axis=0) == SPLIT * NUM_WARPS - 1  # CTA-uniform (cross-warp reduce)
    if last:
        # read the sum and re-arm the workspace in one L2 exchange (fp32 xchg is not exposed -> int32 view);
        # one half at a time (exchanged sum -> bf16 store) to keep 128 accumulator registers live, not 256.
        gl.atomic_xchg(cnt_ptr + j, 0, sem="relaxed", scope="gpu")  # re-arm the ticket for the next launch
        wij = wsi_ptr + j * 65536
        cbase = c_ptr + mt * (256 * N) + nt * 256
        tot0 = gl.amd.cdna4.buffer_atomic_xchg(wij, offs, offs * 0, sem="acquire", scope="gpu")
        gl.amd.cdna3.buffer_store(tot0.to(gl.float32, bitcast=True).to(gl.bfloat16), cbase, _gg_tile_offs(N, WMMA_LAYOUT))
        tot1 = gl.amd.cdna4.buffer_atomic_xchg(wij + 128, offs, offs * 0, sem="acquire", scope="gpu")
        gl.amd.cdna3.buffer_store(tot1.to(gl.float32, bitcast=True).to(gl.bfloat16), cbase + 128,
                                  _gg_tile_offs(N, WMMA_LAYOUT))

@gluon.jit
def gg_dsv4_persistent(a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr, mte_ptr, nmt_ptr, ws_ptr, wsi_ptr, cnt_ptr, M: gl.constexpr,
                       N: gl.constexpr, K: gl.constexpr, SPLIT: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr,
                       SHARED_LAYOUT_B2: gl.constexpr, SHARED_SCALE_A: gl.constexpr, SHARED_SCALE_B: gl.constexpr,
                       WMMA_LAYOUT: gl.constexpr, LOAD_WMMA_LAYOUT: gl.constexpr, B2_LOAD_LAYOUT: gl.constexpr,
                       OUTPUT_CGA_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                       CTA_M: gl.constexpr, CTA_N: gl.constexpr, NUM_WARPS: gl.constexpr = 8,
                       PHASE: gl.constexpr = 0):
    # PHASE 0: whole tiles t < n_full (persistent loop);  PHASE 1: split-K tail pieces only (second launch, same
    # stream, so its registers are allocated separately: one kernel with both paths spills 14 VGPR / 10 SGPR);
    # PHASE 2: both in one launch.  n_full / rem are computed identically by every phase from n_mt.
    NT: gl.constexpr = N // 256
    ITERS: gl.constexpr = K // 256
    DO_FULL: gl.constexpr = PHASE != 1
    DO_TAIL: gl.constexpr = SPLIT > 1 and PHASE != 0
    pid = gl.program_id(axis=0)
    G = gl.num_programs(axis=0)  # host guarantees G % NUM_XCDS == 0
    vpid = (pid % NUM_XCDS) * (G // NUM_XCDS) + pid // NUM_XCDS
    n_live = gl.load(nmt_ptr)
    total = n_live * NT
    n_full = total
    rem = 0
    if SPLIT > 1:
        rem = total % G
        if rem * SPLIT > G:
            rem = 0
        n_full = total - rem
    my_units = rem * SPLIT
    has_full = vpid < n_full
    has_tail = vpid < my_units
    if not DO_FULL:
        has_full = vpid < 0
    if not DO_TAIL:
        has_tail = vpid < 0
    if (not has_full) & (not has_tail):  # no tile and no tail piece: leave before any TDM
        return
    a_buf, b_buf, as_buf, bs_buf = _gg_alloc(a_ptr, b_ptr, as_ptr, bs_ptr, SHARED_LAYOUT_A, SHARED_LAYOUT_B2,
                                             SHARED_SCALE_A, SHARED_SCALE_B)
    if has_full:
        mt, nt = _gg_tile_coords(vpid, n_live, NT)
        e = gl.load(mte_ptr + mt).to(gl.int32)
        a_desc, b_desc, as_desc, bs_desc = _gg_descs(a_ptr, b_ptr, as_ptr, bs_ptr, mt, e, nt, 0, 0, M, N, K,
                                                     SHARED_LAYOUT_A, SHARED_LAYOUT_B2, SHARED_SCALE_A,
                                                     SHARED_SCALE_B)
        a_desc, b_desc, as_desc, bs_desc = _gg_prologue_a(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf,
                                                          bs_buf)
        a_desc, b_desc, as_desc, bs_desc = _gg_prologue_b(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf,
                                                          bs_buf)
        for t in range(vpid, n_full, G):
            mt, nt = _gg_tile_coords(t, n_live, NT)
            e = gl.load(mte_ptr + mt).to(gl.int32)
            # descriptors already advanced past the prologue (data k = 512, scale k-tile 3)
            a_desc, b_desc, as_desc, bs_desc = _gg_descs(a_ptr, b_ptr, as_ptr, bs_ptr, mt, e, nt, 512, 3, M, N, K,
                                                         SHARED_LAYOUT_A, SHARED_LAYOUT_B2, SHARED_SCALE_A,
                                                         SHARED_SCALE_B)
            acc0, acc1 = _gg_mainloop_loops(a_buf, b_buf, as_buf, bs_buf, a_desc, b_desc, as_desc, bs_desc, ITERS,
                                      WMMA_LAYOUT, LOAD_WMMA_LAYOUT, B2_LOAD_LAYOUT)
            gl.barrier()  # every warp is done reading both stages
            # next tile's whole prologue first (clamped: the last iteration re-fetches its own tile from L2)
            tn = gl.minimum(t + G, n_full - 1)
            mt2, nt2 = _gg_tile_coords(tn, n_live, NT)
            e2 = gl.load(mte_ptr + mt2).to(gl.int32)
            a2, b2, as2, bs2 = _gg_descs(a_ptr, b_ptr, as_ptr, bs_ptr, mt2, e2, nt2, 0, 0, M, N, K, SHARED_LAYOUT_A,
                                         SHARED_LAYOUT_B2, SHARED_SCALE_A, SHARED_SCALE_B)
            a2, b2, as2, bs2 = _gg_prologue_a(a2, b2, as2, bs2, a_buf, b_buf, as_buf, bs_buf)
            a2, b2, as2, bs2 = _gg_prologue_b(a2, b2, as2, bs2, a_buf, b_buf, as_buf, bs_buf)
            # then C of this tile, while that prefetch is in flight
            _gg_store_c_direct(c_ptr, mt, nt, acc0, acc1, N, WMMA_LAYOUT)
        tdm.async_wait(0)
        gl.barrier()
    if DO_TAIL:
        if has_tail:
            _gg_split_piece(a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr, mte_ptr, ws_ptr, wsi_ptr, cnt_ptr, n_live, n_full, vpid, a_buf,
                            b_buf, as_buf, bs_buf, M, N, K, SPLIT, SHARED_LAYOUT_A, SHARED_LAYOUT_B2, SHARED_SCALE_A,
                            SHARED_SCALE_B, WMMA_LAYOUT, LOAD_WMMA_LAYOUT, B2_LOAD_LAYOUT, OUTPUT_CGA_LAYOUT,
                            NUM_WARPS)
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()


@gluon.jit
def gg_dsv4_tail(a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr, mte_ptr, nmt_ptr, ws_ptr, wsi_ptr, cnt_ptr, M: gl.constexpr,
                 N: gl.constexpr, K: gl.constexpr, SPLIT: gl.constexpr, SHARED_LAYOUT_A: gl.constexpr,
                 SHARED_LAYOUT_B2: gl.constexpr, SHARED_SCALE_A: gl.constexpr, SHARED_SCALE_B: gl.constexpr,
                 WMMA_LAYOUT: gl.constexpr, LOAD_WMMA_LAYOUT: gl.constexpr, B2_LOAD_LAYOUT: gl.constexpr,
                 OUTPUT_CGA_LAYOUT: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                 CTA_M: gl.constexpr, CTA_N: gl.constexpr, NUM_WARPS: gl.constexpr = 8):
    """(c) split-K tail pieces as their own launch (after gg_dsv4_persistent PHASE=0 on the same stream): one
    piece per CTA at the kernel's top level, so the flat static K schedule of (a) applies."""
    NT: gl.constexpr = N // 256
    pid = gl.program_id(axis=0)
    G = gl.num_programs(axis=0)
    vpid = (pid % NUM_XCDS) * (G // NUM_XCDS) + pid // NUM_XCDS
    n_live = gl.load(nmt_ptr)
    total = n_live * NT
    rem = total % G
    if rem * SPLIT > G:
        rem = 0
    n_full = total - rem
    if vpid >= rem * SPLIT:  # no piece: leave before any TDM
        return
    a_buf, b_buf, as_buf, bs_buf = _gg_alloc(a_ptr, b_ptr, as_ptr, bs_ptr, SHARED_LAYOUT_A, SHARED_LAYOUT_B2,
                                             SHARED_SCALE_A, SHARED_SCALE_B)
    _gg_split_piece(a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr, mte_ptr, ws_ptr, wsi_ptr, cnt_ptr, n_live, n_full, vpid,
                    a_buf, b_buf, as_buf, bs_buf, M, N, K, SPLIT, SHARED_LAYOUT_A, SHARED_LAYOUT_B2, SHARED_SCALE_A,
                    SHARED_SCALE_B, WMMA_LAYOUT, LOAD_WMMA_LAYOUT, B2_LOAD_LAYOUT, OUTPUT_CGA_LAYOUT, NUM_WARPS, True)
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()


# ------------------------------------------------------------------ host side
def build_layouts():
    """Upstream make_mxfp8_case layouts for BK256, 256x256 tile, cluster 1 (== dsv4_gemm.build_layouts(1))."""
    tile, block_k = TILE, 256
    block_m = block_n = tile
    cga = []
    output_cga = ()
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
    padded = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [1, 64, 256], [2, 1, 0], parent_cga)
    b2_shared = PartitionedSharedLayout(2, 2, 1, padded)
    b2_load = gl.DistributedLinearLayout(
        reg_bases=[[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [32, 0, 0], [64, 0, 0], [0, 0, 16], [0, 0, 32]],
        lane_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [16, 0, 0]],
        warp_bases=[[0, 0, 64], [0, 0, 0], [0, 0, 0]],
        block_bases=[],
        shape=[128, 1, 128])
    return dict(SHARED_LAYOUT_A=shared_a, SHARED_LAYOUT_B2=b2_shared, SHARED_SCALE_A=shared_as,
                SHARED_SCALE_B=shared_bs, WMMA_LAYOUT=wmma, LOAD_WMMA_LAYOUT=load_wmma, B2_LOAD_LAYOUT=b2_load,
                OUTPUT_CGA_LAYOUT=output_cga, BLOCK_M=block_m, BLOCK_N=block_n, CTA_M=1, CTA_N=1)


def pack_rows(codes: torch.Tensor) -> torch.Tensor:
    """MX block-32 E8M0 codes [R, K/32] (R % 128 == 0, K/32 % 4 == 0) -> upstream pack_scale layout
    [R/128, K/32*128].  One torch copy, device-side, capturable."""
    R, KS = codes.shape
    return codes.view(R // 128, 4, 32, KS // 4, 4).permute(0, 3, 2, 1, 4).reshape(R // 128, KS * 128)


def default_split(K):
    return {4096: 4, 2048: 2}.get(K, 1)


def tile_table(poffs: torch.Tensor, P: int):
    """poffs int32 [E+1] (256-aligned pool offsets) -> (mt_expert int16 [P/256], n_mt int32 [1]); device-only."""
    E = poffs.numel() - 1
    starts = torch.arange(P // 256, device=poffs.device, dtype=torch.int32) * 256
    e = torch.searchsorted(poffs.contiguous(), starts, right=True) - 1
    mte = e.clamp(0, E - 1).to(torch.int16)
    nmt = (poffs[E:] // 256).to(torch.int32)
    return mte, nmt


_LAYOUTS = None
_WS = {}


def _ws(device, N, split, grid):
    key = (device, N, split, grid)
    if key not in _WS:
        nt = grid // split
        _WS[key] = (torch.zeros(nt * 65536, device=device, dtype=torch.float32),
                    torch.zeros(nt, device=device, dtype=torch.int32))
    return _WS[key]


def gg_dsv4(a, b, as_packed, bs_packed, mte, nmt, N, out=None, mode=None, grid=256, split=None):
    """a [P, K] fp8, b [E*N, K] or [E, N, K] fp8, as_packed [P/128, K*4], bs_packed [E*N/128, K*4] uint8,
    mte int16 [P/256], nmt int32 [1].  Returns out [P, N] bf16 (rows of dead tiles are not written).
    mode: 'tile' (a), 'persistent' (b, split=1) or 'split' (b+c); default env GG_DSV4_MODE or 'split'."""
    global _LAYOUTS
    mode = mode or os.environ.get("GG_DSV4_MODE", "split")
    P, K = a.shape
    b2 = b.reshape(-1, K)
    assert P % 256 == 0 and N % 256 == 0 and K % 512 == 0 and K >= 1024 and b2.shape[0] % N == 0
    assert a.is_contiguous() and b2.is_contiguous() and as_packed.is_contiguous() and bs_packed.is_contiguous()
    assert as_packed.shape == (P // 128, K * 4) and bs_packed.shape == (b2.shape[0] // 128, K * 4)
    if out is None:
        out = torch.empty((P, N), dtype=torch.bfloat16, device=a.device)
    if _LAYOUTS is None:
        _LAYOUTS = build_layouts()
    L = _LAYOUTS
    if mode == "tile":
        g = (P // 256) * (N // 256)
        gg_dsv4_tile[(g, 1)](a, b2, out, as_packed, bs_packed, mte, nmt, P, N, K, num_warps=8, waves_per_eu=2,
                             num_ctas=1, **L)
        return out
    sp = 1 if mode == "persistent" else (split or default_split(K))
    assert grid % NUM_XCDS == 0 and (K // sp) % 512 == 0 and K // sp >= 1024
    ws, cnt = _ws(a.device, N, sp, grid) if sp > 1 else (out, nmt)  # dummies when SPLIT == 1
    wsi = ws.view(torch.int32) if sp > 1 else nmt
    # whole tiles, then (split) the tail pieces: two launches on one stream (capturable), no host sync
    gg_dsv4_persistent[(grid, 1)](a, b2, out, as_packed, bs_packed, mte, nmt, ws, wsi, cnt, P, N, K, sp,
                                  num_warps=8, waves_per_eu=2, num_ctas=1, PHASE=0, **L)
    if sp > 1:
        gg_dsv4_tail[(grid, 1)](a, b2, out, as_packed, bs_packed, mte, nmt, ws, wsi, cnt, P, N, K, sp, num_warps=8,
                                waves_per_eu=2, num_ctas=1, **L)
    return out
