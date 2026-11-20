###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Batch-invariant attention forward for RL: one kernel family for rollout (paged KV cache, decode
and prefill) and training (contiguous varlen), with visibility intervals and attention sinks.

Derived from ROCm/aiter @ 892e5d003df529516c864f90b2fc0e0f47f093a2,
aiter/ops/triton/_triton_kernels/attention/unified_attention.py (kernel_unified_attention_2d,
kernel_unified_attention_3d, reduce_segments; MIT), itself adapted from vLLM's
triton_unified_attention.py. Changes from upstream, all for batch invariance and train/infer match:

- The per-tile online softmax keeps the running max at -inf for a row that has seen no key yet and
  uses a 0 stand-in only inside exp2. Upstream resets it to 0, so a fully masked leading tile changed
  a row's rounding depending on how rows were grouped into a block.
- Key loads are always masked by seq_len (upstream skips the mask when TILE_SIZE == page size).
- KV segments are fixed absolute key ranges [s * SEGMENT, (s + 1) * SEGMENT). The 2D kernel (fused
  mode) walks them in order with fresh per-segment state; the 3D kernel (split mode) computes one
  segment per program and the reduce kernel merges them in the same order. The tile loop, the merge
  and the epilogue are shared @triton.jit helpers, so both modes perform the same operations per row.
  Upstream sizes segments from the batch.
- Contiguous K/V (training) is read through the same kernels with a page size of 1.
- Optional per-token visibility intervals [lo, hi] in global key indices (HAS_VIS), and per-head sink
  logits folded once after all keys (HAS_SINK).
- Optional LSE output for the backward.
- fp8, shuffled KV cache, ALiBi, QQ bias and softcap are removed.
"""

import triton
import triton.language as tl

LN2 = tl.constexpr(0.6931471805599453)
RCP_LN2 = tl.constexpr(1.4426950408889634)


@triton.jit
def _find_seq_idx(cu_q_ptr, target_idx, num_seqs, BLOCK_Q: tl.constexpr, use_q_block_mode: tl.constexpr):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(cu_q_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val
        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1


@triton.jit
def _seq_kv(seq_idx, seq_lens_ptr, cu_k_ptr, PAGED: tl.constexpr):
    """(seq_len, first K/V row). Paged caches address rows through the block table instead."""
    if PAGED:
        seq_len = tl.load(seq_lens_ptr + seq_idx)
        kv_base = seq_len * 0
    else:
        kv_base = tl.load(cu_k_ptr + seq_idx)
        seq_len = tl.load(cu_k_ptr + seq_idx + 1) - kv_base
    return seq_len, kv_base


@triton.jit
def _block_tile_range(context_len, qpos_lo, qpos_hi, TILE_SIZE: tl.constexpr, SLIDING_WINDOW: tl.constexpr):
    """Tiles [lo, hi) holding any key a row in [qpos_lo, qpos_hi] attends to."""
    key_hi = context_len + qpos_hi
    if SLIDING_WINDOW > 0:
        key_lo = tl.maximum(context_len + qpos_lo - SLIDING_WINDOW + 1, 0)
    else:
        key_lo = key_hi * 0
    return key_lo // TILE_SIZE, key_hi // TILE_SIZE + 1


@triton.jit
def _load_visibility(vis_lo_ptr, vis_hi_ptr, token, row_valid, vis_base, HAS_VIS: tl.constexpr):
    """Per-row visible key range [lo, hi] (inclusive), converted from global key indices to the
    sequence's own positions by subtracting its first key's global index. Invalid rows get an empty
    range. Without HAS_VIS the values are unused placeholders."""
    if HAS_VIS:
        lo = tl.load(vis_lo_ptr + token, mask=row_valid, other=1) - vis_base
        hi = tl.load(vis_hi_ptr + token, mask=row_valid, other=0) - vis_base
    else:
        lo = token * 0
        hi = token * 0
    return lo, hi


@triton.jit
def _visibility_tile_range(vis_lo, vis_hi, row_valid, TILE_SIZE: tl.constexpr):
    """Tiles [lo, hi) holding any key a row of the block may see; empty if no row sees a key."""
    nonempty = row_valid & (vis_lo <= vis_hi)
    key_lo = tl.min(tl.where(nonempty, vis_lo, 2147483647), axis=0)
    key_hi = tl.max(tl.where(nonempty, vis_hi, -1), axis=0)
    has_keys = key_hi >= 0
    tile_lo = tl.where(has_keys, key_lo // TILE_SIZE, 0)
    tile_hi = tl.where(has_keys, key_hi // TILE_SIZE + 1, 0)
    return tile_lo, tile_hi


@triton.jit
def _attn_tiles(
    Q,
    M,
    L,
    acc,
    tile_lo,
    tile_hi,
    k_ptr,
    v_ptr,
    bt_row_ptr,
    kv_base,
    kv_head_idx,
    seq_len,
    context_len,
    query_pos,
    row_valid,
    vis_lo,
    vis_hi,
    qk_scale,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_k3: tl.constexpr,
    stride_v0,
    stride_v1,
    stride_v2,
    stride_v3: tl.constexpr,
    PAGED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    V_HEAD_SIZE: tl.constexpr,
    V_HEAD_SIZE_PADDED: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    OFFSETS_64: tl.constexpr,
    KV_CACHE_MODIFIER: tl.constexpr,
    HAS_VIS: tl.constexpr,
):
    """Online softmax over tiles [tile_lo, tile_hi). Tiles a row may not see are exact no-ops for
    that row, so the result per row depends only on which of its own keys fall in the range.

    Tiles entirely below seq_len load without a key mask; only the last tile of a sequence needs
    one. Both loops run `_attn_tile` with the same arithmetic."""
    full_hi = tl.minimum(tile_hi, seq_len // TILE_SIZE)
    full_hi = tl.maximum(full_hi, tile_lo)
    for j in range(tile_lo, full_hi):
        M, L, acc = _attn_tile(
            j,
            Q,
            M,
            L,
            acc,
            k_ptr,
            v_ptr,
            bt_row_ptr,
            kv_base,
            kv_head_idx,
            seq_len,
            context_len,
            query_pos,
            row_valid,
            vis_lo,
            vis_hi,
            qk_scale,
            stride_k0,
            stride_k1,
            stride_k2,
            stride_k3,
            stride_v0,
            stride_v1,
            stride_v2,
            stride_v3,
            PAGED,
            BLOCK_SIZE,
            TILE_SIZE,
            HEAD_SIZE,
            HEAD_SIZE_PADDED,
            V_HEAD_SIZE,
            V_HEAD_SIZE_PADDED,
            SLIDING_WINDOW,
            OFFSETS_64,
            KV_CACHE_MODIFIER,
            HAS_VIS,
            False,
        )
    for j in range(full_hi, tile_hi):
        M, L, acc = _attn_tile(
            j,
            Q,
            M,
            L,
            acc,
            k_ptr,
            v_ptr,
            bt_row_ptr,
            kv_base,
            kv_head_idx,
            seq_len,
            context_len,
            query_pos,
            row_valid,
            vis_lo,
            vis_hi,
            qk_scale,
            stride_k0,
            stride_k1,
            stride_k2,
            stride_k3,
            stride_v0,
            stride_v1,
            stride_v2,
            stride_v3,
            PAGED,
            BLOCK_SIZE,
            TILE_SIZE,
            HEAD_SIZE,
            HEAD_SIZE_PADDED,
            V_HEAD_SIZE,
            V_HEAD_SIZE_PADDED,
            SLIDING_WINDOW,
            OFFSETS_64,
            KV_CACHE_MODIFIER,
            HAS_VIS,
            True,
        )
    return M, L, acc


@triton.jit
def _attn_tile(
    j,
    Q,
    M,
    L,
    acc,
    k_ptr,
    v_ptr,
    bt_row_ptr,
    kv_base,
    kv_head_idx,
    seq_len,
    context_len,
    query_pos,
    row_valid,
    vis_lo,
    vis_hi,
    qk_scale,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_k3: tl.constexpr,
    stride_v0,
    stride_v1,
    stride_v2,
    stride_v3: tl.constexpr,
    PAGED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    V_HEAD_SIZE: tl.constexpr,
    V_HEAD_SIZE_PADDED: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    OFFSETS_64: tl.constexpr,
    KV_CACHE_MODIFIER: tl.constexpr,
    HAS_VIS: tl.constexpr,
    MASK_KEYS: tl.constexpr,
):
    offs_t = tl.arange(0, TILE_SIZE)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_vd = tl.arange(0, V_HEAD_SIZE_PADDED)
    seq_offset = j * TILE_SIZE + offs_t
    if MASK_KEYS:
        tile_mask = seq_offset < seq_len
    else:
        tile_mask = tl.full([TILE_SIZE], 1, dtype=tl.int1)
    if PAGED:
        if MASK_KEYS:
            phys = tl.load(bt_row_ptr + seq_offset // BLOCK_SIZE, mask=tile_mask, other=0)
        else:
            phys = tl.load(bt_row_ptr + seq_offset // BLOCK_SIZE)
        in_page = seq_offset % BLOCK_SIZE
    else:
        phys = kv_base + seq_offset
        in_page = seq_offset * 0
    if OFFSETS_64:
        phys = phys.to(tl.int64)
        in_page = in_page.to(tl.int64)

    k_offset = (
        phys[None, :] * stride_k0
        + in_page[None, :] * stride_k1
        + kv_head_idx * stride_k2
        + offs_d[:, None] * stride_k3
    )
    v_offset = (
        phys[:, None] * stride_v0
        + in_page[:, None] * stride_v1
        + kv_head_idx * stride_v2
        + offs_vd[None, :] * stride_v3
    )
    if HEAD_SIZE_PADDED != HEAD_SIZE or MASK_KEYS:
        k_mask = (offs_d < HEAD_SIZE)[:, None] & tile_mask[None, :]
        K = tl.load(k_ptr + k_offset, mask=k_mask, other=0.0, cache_modifier=KV_CACHE_MODIFIER)
    else:
        K = tl.load(k_ptr + k_offset, cache_modifier=KV_CACHE_MODIFIER)
    if V_HEAD_SIZE_PADDED != V_HEAD_SIZE or MASK_KEYS:
        v_mask = (offs_vd < V_HEAD_SIZE)[None, :] & tile_mask[:, None]
        V = tl.load(v_ptr + v_offset, mask=v_mask, other=0.0, cache_modifier=KV_CACHE_MODIFIER)
    else:
        V = tl.load(v_ptr + v_offset, cache_modifier=KV_CACHE_MODIFIER)
    K = K.to(Q.dtype)
    V = V.to(Q.dtype)

    S = qk_scale * tl.dot(Q, K)
    if HAS_VIS:
        allowed = (
            row_valid[:, None]
            & (seq_offset[None, :] >= vis_lo[:, None])
            & (seq_offset[None, :] <= vis_hi[:, None])
        )
    else:
        allowed = row_valid[:, None] & (seq_offset[None, :] <= context_len + query_pos[:, None])
        if SLIDING_WINDOW > 0:
            allowed = allowed & ((context_len + query_pos[:, None] - seq_offset[None, :]) < SLIDING_WINDOW)
    S = tl.where(allowed, S, float("-inf"))

    m_new = tl.maximum(M, tl.max(S, axis=1))
    m_safe = tl.where(m_new > float("-inf"), m_new, 0.0)
    P = tl.math.exp2(S - m_safe[:, None])
    alpha = tl.math.exp2(M - m_safe)
    L = L * alpha + tl.sum(P, axis=1)
    acc = acc * alpha[:, None]
    acc = tl.dot(P.to(V.dtype), V, acc=acc)
    return m_new, L, acc


@triton.jit
def _merge(M, L, A, m, l, a):
    """Merge segment state (m, l, a) into the running state. An empty segment (-inf, 0, 0) is an
    exact no-op on either side of the real ones."""
    m_new = tl.maximum(M, m)
    m_safe = tl.where(m_new > float("-inf"), m_new, 0.0)
    ca = tl.math.exp2(M - m_safe)
    cb = tl.math.exp2(m - m_safe)
    L = L * ca + l * cb
    A = A * ca[:, None] + a * cb[:, None]
    return m_new, L, A


@triton.jit
def _fold_sink(M, L, A, sink2):
    """Fold a per-row sink logit (log2 domain; -inf for none) into the denominator: a virtual key
    with value 0. Applied once, after every key."""
    m_new = tl.maximum(M, sink2)
    m_safe = tl.where(m_new > float("-inf"), m_new, 0.0)
    ca = tl.math.exp2(M - m_safe)
    L = L * ca + tl.math.exp2(sink2 - m_safe)
    A = A * ca[:, None]
    return m_new, L, A


@triton.jit
def _finalize(M, L, A):
    """Normalized output and natural-log LSE (+inf for a row with no keys, so the backward's
    exp(s - lse) is 0 there)."""
    has_keys = L > 0.0
    inv_l = tl.where(has_keys, 1.0 / L, 0.0)
    out = A * inv_l[:, None]
    lse = tl.where(has_keys, (M + tl.math.log2(L)) * LN2, float("inf"))
    return out, lse


@triton.jit
def _load_q(
    q_ptr,
    q_start,
    query_pos,
    head_idx,
    row_valid,
    stride_q0,
    stride_q1,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    q_offset = (
        (q_start + query_pos)[:, None].to(tl.int64) * stride_q0
        + head_idx[:, None] * stride_q1
        + offs_d[None, :]
    )
    return tl.load(q_ptr + q_offset, mask=row_valid[:, None] & (offs_d < HEAD_SIZE)[None, :], other=0.0)


@triton.jit
def _store_out(
    out_ptr,
    lse_ptr,
    out,
    lse,
    token,
    head_idx,
    row_valid,
    stride_o0,
    stride_o1,
    NUM_Q_HEADS: tl.constexpr,
    V_HEAD_SIZE: tl.constexpr,
    V_HEAD_SIZE_PADDED: tl.constexpr,
    STORE_LSE: tl.constexpr,
):
    offs_vd = tl.arange(0, V_HEAD_SIZE_PADDED)
    o_offset = token[:, None].to(tl.int64) * stride_o0 + head_idx[:, None] * stride_o1 + offs_vd[None, :]
    tl.store(
        out_ptr + o_offset,
        out.to(out_ptr.type.element_ty),
        mask=row_valid[:, None] & (offs_vd < V_HEAD_SIZE)[None, :],
    )
    if STORE_LSE:
        tl.store(lse_ptr + token.to(tl.int64) * NUM_Q_HEADS + head_idx, lse, mask=row_valid)


@triton.jit
def attention_fwd_fused_kernel(
    out_ptr,  # [num_tokens, num_q_heads, v_head_size]
    lse_ptr,  # [num_tokens, num_q_heads] fp32, natural log
    q_ptr,  # [num_tokens, num_q_heads, head_size]
    k_ptr,  # paged [num_blocks, block_size, num_kv_heads, head_size]; contiguous [total_k, num_kv_heads, head_size]
    v_ptr,
    block_tables_ptr,  # [num_seqs, max_blocks_per_seq] (paged)
    seq_lens_ptr,  # [num_seqs] (paged)
    cu_k_ptr,  # [num_seqs + 1] (contiguous)
    cu_q_ptr,  # [num_seqs + 1]
    vis_lo_ptr,  # [num_tokens] int32, first visible key (global key index), with HAS_VIS
    vis_hi_ptr,  # [num_tokens] int32, last visible key (inclusive), with HAS_VIS
    sink_ptr,  # [num_q_heads] fp32 sink logits, with HAS_SINK
    scale,
    num_seqs,
    block_table_stride,
    stride_q0,
    stride_q1,
    stride_o0,
    stride_o1,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_k3: tl.constexpr,
    stride_v0,
    stride_v1,
    stride_v2,
    stride_v3: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    QUERIES_PER_KV: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    V_HEAD_SIZE: tl.constexpr,
    V_HEAD_SIZE_PADDED: tl.constexpr,
    PAGED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    TILES_PER_SEGMENT: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    STORE_LSE: tl.constexpr,
    OFFSETS_64: tl.constexpr,
    KV_CACHE_MODIFIER: tl.constexpr,
    HAS_VIS: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    kv_head_idx = tl.program_id(0)
    q_block_global_idx = tl.program_id(1)

    seq_idx = _find_seq_idx(cu_q_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True)
    q_start = tl.load(cu_q_ptr + seq_idx)
    q_len = tl.load(cu_q_ptr + seq_idx + 1) - q_start
    q_block_local_idx = q_block_global_idx - (q_start // BLOCK_Q + seq_idx)
    if q_block_local_idx * BLOCK_Q >= q_len:
        return
    seq_len, kv_base = _seq_kv(seq_idx, seq_lens_ptr, cu_k_ptr, PAGED)
    context_len = seq_len - q_len

    offs_m = tl.arange(0, BLOCK_M)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // QUERIES_PER_KV
    head_idx = kv_head_idx * QUERIES_PER_KV + offs_m % QUERIES_PER_KV
    row_valid = (offs_m // QUERIES_PER_KV < BLOCK_Q) & (query_pos < q_len)

    Q = _load_q(
        q_ptr, q_start, query_pos, head_idx, row_valid, stride_q0, stride_q1, HEAD_SIZE, HEAD_SIZE_PADDED
    )
    qk_scale = scale * RCP_LN2
    bt_row_ptr = block_tables_ptr + seq_idx.to(tl.int64) * block_table_stride
    vis_base = tl.load(cu_k_ptr + seq_idx) if HAS_VIS else 0
    vis_lo, vis_hi = _load_visibility(
        vis_lo_ptr, vis_hi_ptr, q_start + query_pos, row_valid, vis_base, HAS_VIS
    )

    if HAS_VIS:
        tile_lo, tile_hi = _visibility_tile_range(vis_lo, vis_hi, row_valid, TILE_SIZE)
    else:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(qpos_lo + BLOCK_Q - 1, q_len - 1)
        tile_lo, tile_hi = _block_tile_range(context_len, qpos_lo, qpos_hi, TILE_SIZE, SLIDING_WINDOW)

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.zeros([BLOCK_M], dtype=tl.float32)
    A = tl.zeros([BLOCK_M, V_HEAD_SIZE_PADDED], dtype=tl.float32)
    for seg in range(tile_lo // TILES_PER_SEGMENT, (tile_hi - 1) // TILES_PER_SEGMENT + 1):
        t0 = tl.maximum(seg * TILES_PER_SEGMENT, tile_lo)
        t1 = tl.minimum((seg + 1) * TILES_PER_SEGMENT, tile_hi)
        m = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l = tl.zeros([BLOCK_M], dtype=tl.float32)
        a = tl.zeros([BLOCK_M, V_HEAD_SIZE_PADDED], dtype=tl.float32)
        m, l, a = _attn_tiles(
            Q,
            m,
            l,
            a,
            t0,
            t1,
            k_ptr,
            v_ptr,
            bt_row_ptr,
            kv_base,
            kv_head_idx,
            seq_len,
            context_len,
            query_pos,
            row_valid,
            vis_lo,
            vis_hi,
            qk_scale,
            stride_k0,
            stride_k1,
            stride_k2,
            stride_k3,
            stride_v0,
            stride_v1,
            stride_v2,
            stride_v3,
            PAGED,
            BLOCK_SIZE,
            TILE_SIZE,
            HEAD_SIZE,
            HEAD_SIZE_PADDED,
            V_HEAD_SIZE,
            V_HEAD_SIZE_PADDED,
            SLIDING_WINDOW,
            OFFSETS_64,
            KV_CACHE_MODIFIER,
            HAS_VIS,
        )
        M, L, A = _merge(M, L, A, m, l, a)

    if HAS_SINK:
        sink2 = tl.load(sink_ptr + head_idx, mask=row_valid, other=float("-inf")).to(tl.float32) * RCP_LN2
        M, L, A = _fold_sink(M, L, A, sink2)
    out, lse = _finalize(M, L, A)
    _store_out(
        out_ptr,
        lse_ptr,
        out,
        lse,
        q_start + query_pos,
        head_idx,
        row_valid,
        stride_o0,
        stride_o1,
        NUM_Q_HEADS,
        V_HEAD_SIZE,
        V_HEAD_SIZE_PADDED,
        STORE_LSE,
    )


@triton.jit
def attention_fwd_split_kernel(
    segm_m_ptr,  # [num_tokens, num_q_heads, num_segments] fp32
    segm_l_ptr,  # [num_tokens, num_q_heads, num_segments] fp32
    segm_acc_ptr,  # [num_tokens, num_q_heads, num_segments, V_HEAD_SIZE_PADDED] fp32
    q_ptr,
    k_ptr,
    v_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    cu_k_ptr,
    cu_q_ptr,
    vis_lo_ptr,
    vis_hi_ptr,
    sink_ptr,
    scale,
    num_seqs,
    block_table_stride,
    stride_q0,
    stride_q1,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_k3: tl.constexpr,
    stride_v0,
    stride_v1,
    stride_v2,
    stride_v3: tl.constexpr,
    NUM_SEGMENTS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    QUERIES_PER_KV: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    V_HEAD_SIZE: tl.constexpr,
    V_HEAD_SIZE_PADDED: tl.constexpr,
    PAGED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    TILES_PER_SEGMENT: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    OFFSETS_64: tl.constexpr,
    KV_CACHE_MODIFIER: tl.constexpr,
    HAS_VIS: tl.constexpr,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    seg = tl.program_id(2)

    seq_idx = _find_seq_idx(cu_q_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True)
    q_start = tl.load(cu_q_ptr + seq_idx)
    q_len = tl.load(cu_q_ptr + seq_idx + 1) - q_start
    q_block_local_idx = q_block_global_idx - (q_start // BLOCK_Q + seq_idx)
    if q_block_local_idx * BLOCK_Q >= q_len:
        return
    seq_len, kv_base = _seq_kv(seq_idx, seq_lens_ptr, cu_k_ptr, PAGED)
    context_len = seq_len - q_len

    offs_m = tl.arange(0, BLOCK_M)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // QUERIES_PER_KV
    head_idx = kv_head_idx * QUERIES_PER_KV + offs_m % QUERIES_PER_KV
    row_valid = (offs_m // QUERIES_PER_KV < BLOCK_Q) & (query_pos < q_len)
    vis_base = tl.load(cu_k_ptr + seq_idx) if HAS_VIS else 0
    vis_lo, vis_hi = _load_visibility(
        vis_lo_ptr, vis_hi_ptr, q_start + query_pos, row_valid, vis_base, HAS_VIS
    )

    if HAS_VIS:
        tile_lo, tile_hi = _visibility_tile_range(vis_lo, vis_hi, row_valid, TILE_SIZE)
    else:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(qpos_lo + BLOCK_Q - 1, q_len - 1)
        tile_lo, tile_hi = _block_tile_range(context_len, qpos_lo, qpos_hi, TILE_SIZE, SLIDING_WINDOW)
    t0 = tl.maximum(seg * TILES_PER_SEGMENT, tile_lo)
    t1 = tl.minimum((seg + 1) * TILES_PER_SEGMENT, tile_hi)
    if t0 >= t1:
        return

    Q = _load_q(
        q_ptr, q_start, query_pos, head_idx, row_valid, stride_q0, stride_q1, HEAD_SIZE, HEAD_SIZE_PADDED
    )
    qk_scale = scale * RCP_LN2
    bt_row_ptr = block_tables_ptr + seq_idx.to(tl.int64) * block_table_stride

    m = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    a = tl.zeros([BLOCK_M, V_HEAD_SIZE_PADDED], dtype=tl.float32)
    m, l, a = _attn_tiles(
        Q,
        m,
        l,
        a,
        t0,
        t1,
        k_ptr,
        v_ptr,
        bt_row_ptr,
        kv_base,
        kv_head_idx,
        seq_len,
        context_len,
        query_pos,
        row_valid,
        vis_lo,
        vis_hi,
        qk_scale,
        stride_k0,
        stride_k1,
        stride_k2,
        stride_k3,
        stride_v0,
        stride_v1,
        stride_v2,
        stride_v3,
        PAGED,
        BLOCK_SIZE,
        TILE_SIZE,
        HEAD_SIZE,
        HEAD_SIZE_PADDED,
        V_HEAD_SIZE,
        V_HEAD_SIZE_PADDED,
        SLIDING_WINDOW,
        OFFSETS_64,
        KV_CACHE_MODIFIER,
        HAS_VIS,
    )

    row = ((q_start + query_pos).to(tl.int64) * NUM_Q_HEADS + head_idx) * NUM_SEGMENTS + seg
    tl.store(segm_m_ptr + row, m, mask=row_valid)
    tl.store(segm_l_ptr + row, l, mask=row_valid)
    offs_vd = tl.arange(0, V_HEAD_SIZE_PADDED)
    tl.store(segm_acc_ptr + row[:, None] * V_HEAD_SIZE_PADDED + offs_vd[None, :], a, mask=row_valid[:, None])


@triton.jit
def attention_fwd_reduce_kernel(
    out_ptr,
    lse_ptr,
    segm_m_ptr,
    segm_l_ptr,
    segm_acc_ptr,
    seq_lens_ptr,
    cu_k_ptr,
    cu_q_ptr,
    vis_lo_ptr,
    vis_hi_ptr,
    sink_ptr,
    num_seqs,
    stride_o0,
    stride_o1,
    NUM_SEGMENTS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    QUERIES_PER_KV: tl.constexpr,
    GROUP_PADDED: tl.constexpr,
    V_HEAD_SIZE: tl.constexpr,
    V_HEAD_SIZE_PADDED: tl.constexpr,
    PAGED: tl.constexpr,
    SEGMENT_SIZE: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    STORE_LSE: tl.constexpr,
    HAS_VIS: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    """Merges a token's segments in increasing order, for the GQA group of one KV head. Only the
    segments holding the token's own keys are read; the others are no-ops in fused mode too."""
    token = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = _find_seq_idx(cu_q_ptr, token, num_seqs, 1, False)
    q_start = tl.load(cu_q_ptr + seq_idx)
    q_len = tl.load(cu_q_ptr + seq_idx + 1) - q_start
    seq_len, _ = _seq_kv(seq_idx, seq_lens_ptr, cu_k_ptr, PAGED)
    if HAS_VIS:
        vis_base = tl.load(cu_k_ptr + seq_idx)
        key_lo = tl.load(vis_lo_ptr + token) - vis_base
        key_hi = tl.load(vis_hi_ptr + token) - vis_base
    else:
        key_hi = seq_len - q_len + (token - q_start)
        if SLIDING_WINDOW > 0:
            key_lo = tl.maximum(key_hi - SLIDING_WINDOW + 1, 0)
        else:
            key_lo = key_hi * 0
    num_token_segments = tl.where(key_lo <= key_hi, key_hi // SEGMENT_SIZE - key_lo // SEGMENT_SIZE + 1, 0)

    offs_h = tl.arange(0, GROUP_PADDED)
    head_idx = kv_head_idx * QUERIES_PER_KV + offs_h
    row_valid = offs_h < QUERIES_PER_KV
    offs_vd = tl.arange(0, V_HEAD_SIZE_PADDED)

    M = tl.full([GROUP_PADDED], float("-inf"), dtype=tl.float32)
    L = tl.zeros([GROUP_PADDED], dtype=tl.float32)
    A = tl.zeros([GROUP_PADDED, V_HEAD_SIZE_PADDED], dtype=tl.float32)
    for seg in range(key_lo // SEGMENT_SIZE, key_lo // SEGMENT_SIZE + num_token_segments):
        row = (token.to(tl.int64) * NUM_Q_HEADS + head_idx) * NUM_SEGMENTS + seg
        m = tl.load(segm_m_ptr + row, mask=row_valid, other=float("-inf"))
        l = tl.load(segm_l_ptr + row, mask=row_valid, other=0.0)
        a = tl.load(
            segm_acc_ptr + row[:, None] * V_HEAD_SIZE_PADDED + offs_vd[None, :],
            mask=row_valid[:, None],
            other=0.0,
        )
        M, L, A = _merge(M, L, A, m, l, a)

    if HAS_SINK:
        sink2 = tl.load(sink_ptr + head_idx, mask=row_valid, other=float("-inf")).to(tl.float32) * RCP_LN2
        M, L, A = _fold_sink(M, L, A, sink2)
    out, lse = _finalize(M, L, A)
    tokens = token + offs_h * 0
    _store_out(
        out_ptr,
        lse_ptr,
        out,
        lse,
        tokens,
        head_idx,
        row_valid,
        stride_o0,
        stride_o1,
        NUM_Q_HEADS,
        V_HEAD_SIZE,
        V_HEAD_SIZE_PADDED,
        STORE_LSE,
    )
