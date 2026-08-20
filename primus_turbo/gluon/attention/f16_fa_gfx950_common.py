###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# Vendored from https://github.com/AMD-Triton/gluon-kernels
# Source path: kernels/cdna4/fa/f16_fa_gfx950_common.py
# Source branch: bangtian/fa-fwd-gfx950-gluon-optimized
# Source commit: 05b349b545ef713cd0ba41a3d89ddf3e3eb6b2c3
#
# Port delta: retained production JIT, metadata, shape, and stride helpers;
# removed standalone input, reference, benchmark, display, and serialization code.
###############################################################################

"""Shared helpers for the CDNA4 gfx950 Flash Attention Gluon kernel."""

import torch
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async
from triton.experimental.gluon.language.amd.cdna4 import mfma as mfma_cdna4

# Keep formatting disabled below so the performance-sensitive vendored bodies
# remain byte-attributable to the pinned source identified above.
# fmt: off

def get_mma_type_for_arch(arch: str) -> str:
    """MI350 specialization: only gfx950 is supported."""
    if arch == "gfx950":
        return "mfma_cdna4"
    raise ValueError(f"MI350-only kernel: unsupported GPU architecture {arch}")


# ---------------------------------------------------------------------------
# Gluon kernel helpers
# ---------------------------------------------------------------------------

@gluon.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: gl.constexpr = 8):
    """Remap program IDs to distribute work evenly across XCDs."""
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (tall_xcds * pids_per_xcd
               + (xcd - tall_xcds) * (pids_per_xcd - 1)
               + local_pid)
    return pid


@gluon.jit
def _nan_propagating_max(a, b):
    return gl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@gluon.jit
def nan_propagating_max(x, axis):
    """Reduce-max using IEEE 754 maximum (propagates NaN)."""
    return gl.reduce(x, axis, _nan_propagating_max)


@gluon.jit
def do_mma(MMA_TYPE: gl.constexpr, a, b, c):
    """MI350 path: always use CDNA4 MFMA."""
    return mfma_cdna4(a, b, c)


# ---------------------------------------------------------------------------
# Non-pipelined inner loop
# ---------------------------------------------------------------------------

@gluon.jit
def attn_fwd_inner(
    acc, l_i, m_i, q_dot, kt_ptrs, v_ptrs, offs_n, offs_d,
    kt_offs_d, kt_offs_n, start_m,
    stride_kn, stride_vk,
    block_start, block_end,
    kt_smem, v_smem,
    seqlen_q, seqlen_k,
    qk_scale: gl.constexpr,
    MAX_SEQLENS_Q: gl.constexpr, MAX_SEQLENS_K: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,
    PRE_LOAD_V: gl.constexpr, MASK_STEPS: gl.constexpr, IS_CAUSAL: gl.constexpr,
    VARLEN: gl.constexpr,
    MMA_TYPE: gl.constexpr,
    kt_blocked_layout: gl.constexpr, blocked_layout: gl.constexpr,
    kt_dot_layout: gl.constexpr, p_dot_layout: gl.constexpr, v_dot_layout: gl.constexpr,
    mma_layout: gl.constexpr, mma_offs_n_col: gl.constexpr, mma_offs_m_row: gl.constexpr,
    BALANCE_CAUSAL_WAVES: gl.constexpr = False,
):
    """Inner attention loop over K/V blocks with shared memory staging."""
    SEQK = seqlen_k if VARLEN else MAX_SEQLENS_K
    SEQQ = seqlen_q if VARLEN else MAX_SEQLENS_Q
    for block_n in range(block_start, block_end):
        start_n = block_n * BLOCK_N

        if PRE_LOAD_V:
            if MASK_STEPS:
                v_mask = (start_n + offs_n[:, None]) < SEQK
                if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
                    v_mask = v_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
                v_global = gl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                v_global = gl.load(v_ptrs)
            v_smem.store(v_global)

        if MASK_STEPS:
            kt_mask = (start_n + kt_offs_n[None, :]) < SEQK
            if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
                kt_mask = kt_mask & (kt_offs_d[:, None] < ACTUAL_BLOCK_DMODEL)
            kt_global = gl.load(kt_ptrs, mask=kt_mask, other=0.0)
        else:
            kt_global = gl.load(kt_ptrs)
        kt_smem.store(kt_global)

        k_t = kt_smem.load(kt_blocked_layout)
        kt_dot = gl.convert_layout(k_t, kt_dot_layout)
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(MMA_TYPE, q_dot, kt_dot, qk)
        qk = qk * qk_scale

        if MASK_STEPS and IS_CAUSAL:
            causal_offs_n = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
            local_m = gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
            if BALANCE_CAUSAL_WAVES:
                wave_m = local_m // 32
                wave_m = wave_m ^ ((wave_m // 4) * 3)
                local_m = wave_m * 32 + local_m % 32
            causal_offs_m = start_m * BLOCK_M + local_m
            causal_boundary = causal_offs_m[:, None] + (SEQK - SEQQ)
            causal_mask = causal_offs_n[None, :] <= causal_boundary
            qk = gl.where(causal_mask, qk, gl.full([BLOCK_M, BLOCK_N], float("-inf"),
                                                    dtype=gl.float32, layout=mma_layout))

        if MASK_STEPS:
            bound_offs = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
            bound_mask = bound_offs[None, :] < SEQK
            qk = gl.where(bound_mask, qk, gl.full([BLOCK_M, BLOCK_N], float("-inf"),
                                                   dtype=gl.float32, layout=mma_layout))

        m_ij = nan_propagating_max(qk, axis=1)
        m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
        # Varlen ragged-causal rows can attend to zero keys, leaving m_new == -inf.
        if VARLEN:
            m_sub = gl.where(m_new == float("-inf"), 0.0, m_new)
        else:
            m_sub = m_new
        p = gl.exp2(qk - m_sub[:, None])
        l_ij = gl.sum(p, axis=1)
        alpha = gl.exp2(m_i - m_sub)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        m_i = m_new

        if not PRE_LOAD_V:
            if MASK_STEPS:
                v_mask = (start_n + offs_n[:, None]) < SEQK
                if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
                    v_mask = v_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
                v_global = gl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                v_global = gl.load(v_ptrs)
            v_smem.store(v_global)

        v = v_smem.load(blocked_layout)
        p_cast = p.to(v.dtype)
        p_dot = gl.convert_layout(p_cast, p_dot_layout)
        v_dot = gl.convert_layout(v, v_dot_layout)
        acc = do_mma(MMA_TYPE, p_dot, v_dot, acc)

        kt_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk

    return acc, l_i, m_i, kt_ptrs, v_ptrs


# ---------------------------------------------------------------------------
# Pipelined inner loop helpers (CDNA4 async copy path)
# ---------------------------------------------------------------------------

@gluon.jit
def issue_async_load_k(
    kt_smem, k_base, start_n,
    stride_kn, stride_kk,
    seqlen_k,
    MASK_STEPS: gl.constexpr,
    MAX_SEQLENS_K: gl.constexpr,
    VARLEN: gl.constexpr,
    BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr, ACTUAL_BLOCK_DMODEL: gl.constexpr,
    kt_async_layout: gl.constexpr,
):
    SEQK = seqlen_k if VARLEN else MAX_SEQLENS_K
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    kt_offs_d = gl.arange(0, BLOCK_DMODEL, layout=kt_offs_d_layout)
    kt_offs_n = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)
    kt_offsets = kt_offs_d[:, None] * stride_kk + (start_n + kt_offs_n[None, :]) * stride_kn

    if MASK_STEPS:
        kt_mask = (start_n + kt_offs_n[None, :]) < SEQK
        if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
            kt_mask = kt_mask & (kt_offs_d[:, None] < ACTUAL_BLOCK_DMODEL)
        cdna4_async.buffer_load_to_shared(kt_smem, k_base, kt_offsets, mask=kt_mask, other=0.0)
    else:
        cdna4_async.buffer_load_to_shared(kt_smem, k_base, kt_offsets)
    cdna4_async.commit_group()


@gluon.jit
def issue_async_load_v(
    v_smem, v_base, start_n,
    stride_vk, stride_vn,
    seqlen_k,
    MASK_STEPS: gl.constexpr,
    MAX_SEQLENS_K: gl.constexpr,
    VARLEN: gl.constexpr,
    BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr, ACTUAL_BLOCK_DMODEL: gl.constexpr,
    v_async_layout: gl.constexpr,
):
    SEQK = seqlen_k if VARLEN else MAX_SEQLENS_K
    v_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_offs_n = gl.arange(0, BLOCK_N, layout=v_offs_n_layout)
    v_offs_d = gl.arange(0, BLOCK_DMODEL, layout=v_offs_d_layout)
    v_offsets = (start_n + v_offs_n[:, None]) * stride_vk + v_offs_d[None, :] * stride_vn

    if MASK_STEPS:
        v_mask = (start_n + v_offs_n[:, None]) < SEQK
        if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
            v_mask = v_mask & (v_offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
        cdna4_async.buffer_load_to_shared(v_smem, v_base, v_offsets, mask=v_mask, other=0.0)
    else:
        cdna4_async.buffer_load_to_shared(v_smem, v_base, v_offsets)
    cdna4_async.commit_group()


@gluon.jit
def compute_dot1_qk(
    q_dot, kt_dot,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    mma_layout: gl.constexpr,
):
    """Dot1: compute QK^T from register operands. Returns unscaled qk scores."""
    qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
    qk = do_mma("mfma_cdna4", q_dot, kt_dot, qk)
    return qk


@gluon.jit
def compute_softmax(
    acc, l_i, m_i, qk, start_n, start_m,
    seqlen_q, seqlen_k,
    qk_scale: gl.constexpr,
    MAX_SEQLENS_Q: gl.constexpr, MAX_SEQLENS_K: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    MASK_STEPS: gl.constexpr, IS_CAUSAL: gl.constexpr,
    VARLEN: gl.constexpr,
    mma_layout: gl.constexpr, mma_offs_n_col: gl.constexpr, mma_offs_m_row: gl.constexpr,
    BALANCE_CAUSAL_WAVES: gl.constexpr = False,
):
    """Online softmax with optional masking."""
    SEQK = seqlen_k if VARLEN else MAX_SEQLENS_K
    SEQQ = seqlen_q if VARLEN else MAX_SEQLENS_Q
    if MASK_STEPS:
        # For the usual positive attention scale, mask raw scores before the
        # reduction.  max(qk) * scale == max(qk * scale), and forming p with
        # FMA replaces a matrix-wide multiply followed by a subtraction with
        # one vector instruction.  Preserve the original order for unusual
        # zero/negative scales, where the max identity does not hold.
        POSITIVE_SCALE: gl.constexpr = qk_scale > 0.0
        qk_softmax = qk if POSITIVE_SCALE else qk * qk_scale

        if IS_CAUSAL:
            causal_offs_n = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
            local_m = gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
            if BALANCE_CAUSAL_WAVES:
                wave_m = local_m // 32
                # Pair resident BM256 waves as 1+8, 2+7, 3+6, and 4+5.
                wave_m = wave_m ^ ((wave_m // 4) * 3)
                local_m = wave_m * 32 + local_m % 32
            causal_offs_m = start_m * BLOCK_M + local_m
            causal_boundary = causal_offs_m[:, None] + (SEQK - SEQQ)
            causal_mask = causal_offs_n[None, :] <= causal_boundary
            qk_softmax = gl.where(causal_mask, qk_softmax, gl.full([BLOCK_M, BLOCK_N], float("-inf"),
                                                    dtype=gl.float32, layout=mma_layout))

        CHECK_K_BOUNDS: gl.constexpr = (
            VARLEN or MAX_SEQLENS_K % BLOCK_N != 0)
        if CHECK_K_BOUNDS:
            bound_offs = start_n + gl.arange(
                0, BLOCK_N, layout=mma_offs_n_col)
            bound_mask = bound_offs[None, :] < SEQK
            qk_softmax = gl.where(
                bound_mask, qk_softmax,
                gl.full([BLOCK_M, BLOCK_N], float("-inf"),
                        dtype=gl.float32, layout=mma_layout))

        m_ij = nan_propagating_max(qk_softmax, axis=1)
        if POSITIVE_SCALE:
            m_ij = m_ij * qk_scale
        m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
        # Varlen ragged-causal rows can attend to zero keys, leaving m_new == -inf.
        if VARLEN:
            m_sub = gl.where(m_new == float("-inf"), 0.0, m_new)
        else:
            m_sub = m_new
        if POSITIVE_SCALE:
            p = gl.exp2(gl.fma(qk_softmax, qk_scale, -m_sub[:, None]))
        else:
            p = gl.exp2(qk_softmax - m_sub[:, None])
    else:
        # FMA-friendly unmasked path.
        m_ij = nan_propagating_max(qk, axis=1) * qk_scale
        m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
        m_sub = m_new
        p = gl.exp2(qk * qk_scale - m_sub[:, None])

    l_ij = gl.sum(p, axis=1)
    alpha = gl.exp2(m_i - m_sub)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    m_i = m_new
    return acc, l_i, m_i, p


@gluon.jit
def compute_dot2_pv(acc, p, v_smem, p_dot_layout: gl.constexpr, v_dot_layout: gl.constexpr):
    """Dot2: Compute P @ V and accumulate."""
    v_dot = cdna4_async.load_shared_relaxed(v_smem, v_dot_layout)
    p_cast = p.to(v_dot.dtype)
    p_dot = gl.convert_layout(p_cast, p_dot_layout)
    acc = do_mma("mfma_cdna4", p_dot, v_dot, acc)
    return acc


# ---------------------------------------------------------------------------
# Metadata and layout helpers
# ---------------------------------------------------------------------------

class MetaData:
    max_seqlens_q = 0
    max_seqlens_k = 0
    causal = False
    layout = None
    cu_seqlens_q = None
    cu_seqlens_k = None
    varlen = False
    num_contexts = 0
    total_q = 0
    total_k = 0

    def __init__(self, sm_scale=1.0):
        self.sm_scale = sm_scale

    def need_causal(self):
        self.causal = True

    def set_varlen_params(self, cu_seqlens_q, cu_seqlens_k):
        """Enable ragged/thd mode from FAv3-style cu_seqlens indptr tensors."""
        self.varlen = True
        self.layout = 'thd'
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        assert len(cu_seqlens_q) >= 2
        assert len(cu_seqlens_q) == len(cu_seqlens_k)
        self.num_contexts = len(cu_seqlens_q) - 1
        seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        self.max_seqlens_q = int(seqlens_q.max())
        self.max_seqlens_k = int(seqlens_k.max())
        self.total_q = int(cu_seqlens_q[-1])
        self.total_k = int(cu_seqlens_k[-1])

    def check_args(self, q, k, v, o):
        assert self.max_seqlens_q > 0 and self.max_seqlens_k > 0
        if self.varlen:
            assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3, \
                "varlen/thd mode expects 3D (total_tokens, heads, head_dim) q/k/v"
            assert self.cu_seqlens_q is not None and self.cu_seqlens_k is not None
            assert self.layout == 'thd'
            assert self.cu_seqlens_q.is_cuda and self.cu_seqlens_k.is_cuda
            assert self.cu_seqlens_q.dtype == torch.int32 and self.cu_seqlens_k.dtype == torch.int32
            assert self.total_q == q.shape[0], \
                f"cu_seqlens_q[-1]={self.total_q} != q tokens {q.shape[0]}"
            assert self.total_k == k.shape[0], \
                f"cu_seqlens_k[-1]={self.total_k} != k tokens {k.shape[0]}"
        else:
            assert q.dim() == 4
        assert k.shape == v.shape
        assert q.shape[-1] == k.shape[-1]
        assert q.dtype == k.dtype and q.dtype == v.dtype
        assert o.shape == q.shape
        batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q, k, self)
        assert (nheads_q % nheads_k) == 0
        assert head_size <= 256
        assert self.layout is not None


def get_shape_from_layout(q, k, metadata):
    if metadata.layout == 'bhsd':
        batch, nheads_q, _, head_size = q.shape
        nheads_k = k.shape[1]
    elif metadata.layout == 'bshd':
        batch, _, nheads_q, head_size = q.shape
        nheads_k = k.shape[2]
    elif metadata.layout == 'thd':
        _, nheads_q, head_size = q.shape
        nheads_k = k.shape[1]
        batch = metadata.num_contexts
    else:
        raise ValueError(f"Unsupported layout: {metadata.layout}")
    return batch, nheads_q, nheads_k, head_size


def get_strides_from_layout(q, k, v, o, metadata):
    if metadata.layout == 'bhsd':
        q_strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
        k_strides = (k.stride(0), k.stride(1), k.stride(2), k.stride(3))
        v_strides = (v.stride(0), v.stride(1), v.stride(2), v.stride(3))
        o_strides = (o.stride(0), o.stride(1), o.stride(2), o.stride(3))
    elif metadata.layout == 'bshd':
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))
    elif metadata.layout == 'thd':
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
    else:
        raise ValueError(f"Unsupported layout: {metadata.layout}")
    return q_strides, k_strides, v_strides, o_strides


# fmt: on
