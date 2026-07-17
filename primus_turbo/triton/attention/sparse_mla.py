###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plain-Triton DeepSeek-V4 sparse-MLA attention (the "triton v2" backend): forward + backward.

Fused single MQA latent (K = V = the first kv_lora_rank channels), per-token absolute top-k
indices, optional per-head softmax sink; the QK / PV GEMMs lower to MFMA via tl.dot. The
backward uses the non-atomic chunked-gather scheme (per rank-chunk dQ + dKV-intermediate
kernels, then a CSR inverted-topk reduce into dkv). Public entry points sparse_mla_fwd_triton
/ sparse_mla_bwd_triton are the reference oracle for the flydsl sparse-MLA path.
"""

import torch
import triton
import triton.language as tl

from primus_turbo.triton.utils.triton_knobs_helper import scoped_amd_triton_knobs_disabled


def _get_fwd_configs():
    # Focused around the autotune winners from the wide sweep (TILE_K=16,
    # num_stages=3 dominated -> latency-bound; deep pipelining is the lever), plus
    # num_stages=4 to probe deeper pipelining. Keeps first-call autotune cheap.
    return [
        triton.Config({"BLOCK_H": bh, "TILE_K": tk, "waves_per_eu": wpe}, num_warps=4, num_stages=ns)
        for bh in (32, 64)
        for tk in (16, 32)
        for ns in (2, 3, 4)
        for wpe in (0, 1)
    ]


@triton.autotune(configs=_get_fwd_configs(), key=["num_heads", "TOPK", "D_V", "D_ROPE", "HAS_ROPE"])
@triton.jit
def _sparse_mla_fwd_tr_kernel(
    Q_ptr,  # [total_tokens, num_heads, D_QK] bf16
    KV_ptr,  # [num_kv, 1, D_QK]               bf16
    TopK_ptr,  # [total_tokens, TOPK]            int32
    Sink_ptr,  # [num_heads]                     fp32 (guarded by HAS_SINK)
    O_ptr,  # [total_tokens, num_heads, D_V]  bf16
    LSE_ptr,  # [total_tokens, num_heads]       fp32 (sink-inclusive)
    stride_q_t,
    stride_q_h,
    stride_kv_t,
    stride_o_t,
    stride_o_h,
    stride_topk_t,
    scale,
    num_heads,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TILE_K: tl.constexpr,
    D_V: tl.constexpr,
    D_ROPE: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    tok = tl.program_id(0)
    hg = tl.program_id(1)

    offs_h = hg * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < num_heads
    offs_v = tl.arange(0, D_V)
    offs_r = tl.arange(0, D_ROPE)

    q_base = tok.to(tl.int64) * stride_q_t + offs_h.to(tl.int64)[:, None] * stride_q_h
    q_lora = tl.load(Q_ptr + q_base + offs_v[None, :], mask=mask_h[:, None], other=0.0)
    if HAS_ROPE:
        q_rope = tl.load(Q_ptr + q_base + (D_V + offs_r)[None, :], mask=mask_h[:, None], other=0.0)

    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D_V], dtype=tl.float32)

    topk_base = tok.to(tl.int64) * stride_topk_t
    for kt in range(0, TOPK, TILE_K):
        offs_k = kt + tl.arange(0, TILE_K)
        idx = tl.load(TopK_ptr + topk_base + offs_k, mask=offs_k < TOPK, other=-1)
        valid = idx >= 0
        safe = tl.where(valid, idx, 0).to(tl.int64)

        kv_base = safe[:, None] * stride_kv_t
        k_lora = tl.load(KV_ptr + kv_base + offs_v[None, :], mask=valid[:, None], other=0.0)

        # S = q @ k^T over [lora ++ rope]  -> [BLOCK_H, TILE_K]
        s = tl.dot(q_lora, tl.trans(k_lora))
        if HAS_ROPE:
            k_rope = tl.load(KV_ptr + kv_base + (D_V + offs_r)[None, :], mask=valid[:, None], other=0.0)
            s += tl.dot(q_rope, tl.trans(k_rope))
        s = s * scale
        s = tl.where(valid[None, :] & mask_h[:, None], s, float("-inf"))

        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        m_new = tl.where(m_new > float("-inf"), m_new, 0.0)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(k_lora.dtype), k_lora)
        m_i = m_new

    if HAS_SINK:
        sink = tl.load(Sink_ptr + offs_h, mask=mask_h, other=float("-inf"))
        m_f = tl.maximum(m_i, sink)
        af = tl.exp(m_i - m_f)
        l_t = l_i * af + tl.exp(sink - m_f)
        acc = acc * af[:, None]
        acc = acc / l_t[:, None]
        lse = m_f + tl.log(l_t)
    else:
        acc = acc / l_i[:, None]
        lse = m_i + tl.log(l_i)

    o_base = tok.to(tl.int64) * stride_o_t + offs_h.to(tl.int64)[:, None] * stride_o_h
    tl.store(O_ptr + o_base + offs_v[None, :], acc.to(O_ptr.dtype.element_ty), mask=mask_h[:, None])
    tl.store(LSE_ptr + tok.to(tl.int64) * num_heads + offs_h, lse, mask=mask_h)


def sparse_mla_fwd_triton(q, kv, topk_indices, attn_sink=None, kv_lora_rank=512, scale=None):
    """DeepSeek-V4 sparse-MLA forward (plain Triton / MFMA). API mirrors the gluon path.

    Args:
        q:            [total_tokens, num_heads, d_qk] bf16
        kv:           [num_kv, 1, d_qk] bf16 (or [num_kv, d_qk]); single MQA latent
        topk_indices: [total_tokens, topk] int32 (SWA + sparse, -1 = invalid)
        attn_sink:    [num_heads] fp32 optional per-head learnable sink
        kv_lora_rank: int, default 512
        scale:        float, default 1/sqrt(d_qk)

    Returns:
        o:   [total_tokens, num_heads, kv_lora_rank] (q.dtype)
        lse: [total_tokens, num_heads] fp32 (sink-inclusive when attn_sink given)
    """
    assert q.is_contiguous() and topk_indices.is_contiguous()
    total_tokens, num_heads, d_qk = q.shape
    rope_rank = d_qk - kv_lora_rank
    topk = topk_indices.shape[1]
    if scale is None:
        scale = 1.0 / (d_qk**0.5)
    if kv.dim() == 2:
        kv = kv.unsqueeze(1)
    assert kv.is_contiguous()
    assert kv.shape[0] >= total_tokens and kv.shape[-1] == d_qk

    has_sink = attn_sink is not None
    if has_sink:
        assert attn_sink.is_contiguous() and attn_sink.dtype == torch.float32
        assert attn_sink.shape == (num_heads,)
        sink_ptr = attn_sink
    else:
        sink_ptr = torch.empty(1, dtype=torch.float32, device=q.device)

    o = torch.empty(total_tokens, num_heads, kv_lora_rank, dtype=q.dtype, device=q.device)
    lse = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=q.device)

    # V4 single-latent form: the D_ROPE block of q/kv is a zero pad (RoPE baked
    # in-place over the 512 latent), so the rope QK term is provably zero — skip it.
    has_rope = False

    grid = lambda META: (total_tokens, triton.cdiv(num_heads, META["BLOCK_H"]))
    # The AMD ping-pong / async-copy knobs primus_turbo enables globally are a pessimization for
    # this fwd kernel. They are read at compile time and are not in Triton's cache key, so compiling
    # under this scope pins the faster non-ping-pong schedule without touching any other kernel.
    with scoped_amd_triton_knobs_disabled():
        _sparse_mla_fwd_tr_kernel[grid](
            Q_ptr=q,
            KV_ptr=kv,
            TopK_ptr=topk_indices,
            Sink_ptr=sink_ptr,
            O_ptr=o,
            LSE_ptr=lse,
            stride_q_t=q.stride(0),
            stride_q_h=q.stride(1),
            stride_kv_t=kv.stride(0),
            stride_o_t=o.stride(0),
            stride_o_h=o.stride(1),
            stride_topk_t=topk_indices.stride(0),
            scale=scale,
            num_heads=num_heads,
            TOPK=topk,
            D_V=kv_lora_rank,
            D_ROPE=rope_rank,
            HAS_ROPE=has_rope,
            HAS_SINK=has_sink,
        )
    return o, lse


# ---- CSR inverted-topk builder (merged from _csr_helper) ----
def _build_inverted_topk_slice(topk_indices_slice, r_start, R_CHUNK, num_kv=None):
    """Build a CSR-style inverted index for a topk slice, excluding invalid (-1).

    Args:
        topk_indices_slice: [T, R_CHUNK] int32 — ``topk_indices[:, r_start:r_start+R_CHUNK]``.
          May contain -1 padding (last chunk shorter than R_CHUNK).
        r_start:  int — first rank index in this slice (documentation only).
        R_CHUNK:  int — number of ranks in this slice (constexpr width).
        num_kv:   int or None — number of KV tokens. When the KV buffer holds MORE
          rows than query tokens (V4 ``[local ++ pool]``, ``num_kv = S + P``), pass
          it so ``inv_ptr`` has length ``num_kv + 1``. Defaults to ``T``.

    Returns:
        inv_ptr:  [num_kv+1] int32 — CSR row pointers (kv_token -> range in inv_data).
        inv_data: [T*R_CHUNK] int32 — flat indices ``q*R_CHUNK + local_r`` sorted by KV
          token; invalid (-1) entries sort to the front and are skipped by inv_ptr[0].
    """
    T, RC = topk_indices_slice.shape
    n_kv = T if num_kv is None else int(num_kv)
    # Keep the sort keys int32 (was .long()): the stable radix sort is faster on int32 than
    # int64 and yields a BIT-IDENTICAL permutation (same key values, stable) -> inv_ptr/inv_data
    # byte-for-byte identical, so the gather stays deterministic and the SNR reference holds.
    flat_kv = topk_indices_slice.reshape(-1)  # [T*R_CHUNK] int32; -1 marks invalid

    inv_data = torch.argsort(flat_kv, stable=True).to(torch.int32)  # [T*R_CHUNK]
    counts = torch.bincount(flat_kv + 1, minlength=n_kv + 1)  # [num_kv+1]; bin0 = #invalid
    inv_ptr = torch.cumsum(counts, dim=0).to(torch.int32)  # [num_kv+1]; inv_ptr[0]=#invalid

    return inv_ptr, inv_data


# ---- backward compute kernels (merged from dsa_bwd_kernels) ----


@triton.jit
def _bwd_chunk_dq_store_ds(
    Q_ptr,  # [T, H, D] bf16
    KV_ptr,  # [T, 1, D] bf16
    dO_ptr,  # [T, H, D_V] bf16
    TopK_ptr,  # [T, TOPK] int32
    LSE_ptr,  # [T, H] fp32
    Delta_ptr,  # [T, H] fp32 (computed here on the first chunk, read after)
    O_ptr,  # [T, H, D_V] bf16 (fwd output, for Delta = rowsum(O*dO))
    dQ_ptr,  # [T, H, D] bf16 — read-modify-write across chunks
    dS_ptr,  # [T, H, R_CHUNK] bf16 — output chunk dS
    P_ptr,  # [T, H, R_CHUNK] bf16 — output chunk P
    stride_q_t: tl.int64,
    stride_q_h: tl.int64,
    stride_kv_t: tl.int64,
    stride_do_t: tl.int64,
    stride_do_h: tl.int64,
    stride_o_t: tl.int64,
    stride_o_h: tl.int64,
    stride_dq_t: tl.int64,
    stride_dq_h: tl.int64,
    stride_topk_t: tl.int64,
    stride_ds_t: tl.int64,
    stride_ds_h: tl.int64,
    scale: tl.float32,
    num_heads: tl.int32,
    R_START: tl.int32,
    R_CHUNK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TILE_K: tl.constexpr,
    D_V: tl.constexpr,
    D_ROPE: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    IS_FIRST_CHUNK: tl.constexpr,
):
    """dQ accumulation for rank chunk [R_START, R_START+R_CHUNK), plus stores
    chunk dS and P to [T, H, R_CHUNK] buffers for _bwd_compute_dkv_intermediate.
    Grid: (total_tokens, num_hg).

    Delta = rowsum(O*dO) is fused here (computed + stored on the first chunk,
    reloaded on later chunks) instead of a separate preprocess kernel — dO is
    already resident, so this only adds the O load and drops a whole kernel + a
    duplicate dO read.

    HAS_ROPE=False (V4 zero-pad): skips all rope MMAs/loads and writes the dQ rope
    columns as zero (they are discarded downstream but kept well-defined)."""
    token_idx = tl.program_id(0)
    hg_idx = tl.program_id(1)
    offs_h = hg_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < num_heads
    offs_v = tl.arange(0, D_V)
    offs_r = tl.arange(0, D_ROPE)

    q_base = token_idx * stride_q_t
    Q_lora = tl.load(
        Q_ptr + q_base + offs_h[:, None] * stride_q_h + offs_v[None, :], mask=mask_h[:, None], other=0.0
    )
    if HAS_ROPE:
        Q_rope = tl.load(
            Q_ptr + q_base + offs_h[:, None] * stride_q_h + (D_V + offs_r[None, :]),
            mask=mask_h[:, None],
            other=0.0,
        )
    do_base = token_idx * stride_do_t
    dO_val = tl.load(
        dO_ptr + do_base + offs_h[:, None] * stride_do_h + offs_v[None, :], mask=mask_h[:, None], other=0.0
    )
    lse = tl.load(LSE_ptr + token_idx * num_heads + offs_h, mask=mask_h, other=0.0)
    if IS_FIRST_CHUNK:
        # Delta = rowsum(O * dO): fold the preprocess in (dO already loaded).
        O_val = tl.load(
            O_ptr + token_idx * stride_o_t + offs_h[:, None] * stride_o_h + offs_v[None, :],
            mask=mask_h[:, None],
            other=0.0,
        )
        delta = tl.sum(O_val.to(tl.float32) * dO_val.to(tl.float32), axis=1)
        tl.store(Delta_ptr + token_idx * num_heads + offs_h, delta, mask=mask_h)
    else:
        delta = tl.load(Delta_ptr + token_idx * num_heads + offs_h, mask=mask_h, other=0.0)

    dq_base = token_idx * stride_dq_t
    if IS_FIRST_CHUNK:
        dQ_lora = tl.zeros([BLOCK_H, D_V], dtype=tl.float32)
    else:
        dQ_lora = tl.load(
            dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + offs_v[None, :],
            mask=mask_h[:, None],
            other=0.0,
        ).to(tl.float32)
    if HAS_ROPE:
        if IS_FIRST_CHUNK:
            dQ_rope = tl.zeros([BLOCK_H, D_ROPE], dtype=tl.float32)
        else:
            dQ_rope = tl.load(
                dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + (D_V + offs_r[None, :]),
                mask=mask_h[:, None],
                other=0.0,
            ).to(tl.float32)

    NUM_TILES: tl.constexpr = (R_CHUNK + TILE_K - 1) // TILE_K
    topk_base = token_idx * stride_topk_t + R_START
    offs_tile = tl.arange(0, TILE_K)
    ds_base = token_idx * stride_ds_t + hg_idx * BLOCK_H * stride_ds_h

    for t in range(NUM_TILES):
        tile_start = t * TILE_K
        tile_offs = tile_start + offs_tile
        valid = tile_offs < R_CHUNK
        topk_pos = tl.load(TopK_ptr + topk_base + tile_offs, mask=valid, other=-1)
        valid = valid & (topk_pos != -1)
        safe_pos = tl.where(valid, topk_pos, 0)

        K_lora_T = tl.load(
            KV_ptr + safe_pos[None, :] * stride_kv_t + offs_v[:, None], mask=valid[None, :], other=0.0
        )

        S = tl.dot(Q_lora, K_lora_T)
        if HAS_ROPE:
            K_rope_T = tl.load(
                KV_ptr + safe_pos[None, :] * stride_kv_t + (D_V + offs_r[:, None]),
                mask=valid[None, :],
                other=0.0,
            )
            S += tl.dot(Q_rope, K_rope_T)
        S = tl.where(valid[None, :] & mask_h[:, None], S * scale, float("-inf"))
        P = tl.exp(S - lse[:, None])
        P = tl.where(valid[None, :] & mask_h[:, None], P, 0.0)
        dP = tl.dot(dO_val, K_lora_T)
        dS = P * (dP - delta[:, None]) * scale
        dS = tl.where(valid[None, :] & mask_h[:, None], dS, 0.0)

        dS_bf = dS.to(tl.bfloat16)
        dQ_lora += tl.dot(dS_bf, tl.trans(K_lora_T)).to(tl.float32)
        if HAS_ROPE:
            dQ_rope += tl.dot(dS_bf, tl.trans(K_rope_T)).to(tl.float32)

        local_h = tl.arange(0, BLOCK_H)
        tl.store(
            dS_ptr + ds_base + local_h[:, None] * stride_ds_h + tile_offs[None, :],
            dS_bf,
            mask=mask_h[:, None] & valid[None, :],
        )
        tl.store(
            P_ptr + ds_base + local_h[:, None] * stride_ds_h + tile_offs[None, :],
            P.to(tl.bfloat16),
            mask=mask_h[:, None] & valid[None, :],
        )

    tl.store(
        dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + offs_v[None, :],
        dQ_lora.to(Q_lora.dtype),
        mask=mask_h[:, None],
    )
    if HAS_ROPE:
        tl.store(
            dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + (D_V + offs_r[None, :]),
            dQ_rope.to(Q_lora.dtype),
            mask=mask_h[:, None],
        )
    else:
        tl.store(
            dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + (D_V + offs_r[None, :]),
            tl.zeros([BLOCK_H, D_ROPE], dtype=Q_lora.dtype),
            mask=mask_h[:, None],
        )


@triton.jit
def _bwd_compute_dkv_intermediate(
    Q_ptr,  # [T, H, D_QK] bf16  (UNtransposed; loaded transposed via strided index)
    dO_ptr,  # [T, H, D_V]  bf16 (UNtransposed)
    dS_ptr,  # [T, H, R_CHUNK] bf16
    P_ptr,  # [T, H, R_CHUNK] bf16
    Interm_ptr,  # [T, R_CHUNK, D_QK] bf16 — output, one writer per (q, rank)
    stride_q_t: tl.int64,
    stride_q_h: tl.int64,
    stride_do_t: tl.int64,
    stride_do_h: tl.int64,
    stride_ds_t: tl.int64,
    stride_ds_h: tl.int64,
    stride_interm_t: tl.int64,  # R_CHUNK * D_QK
    stride_interm_k: tl.int64,  # D_QK
    num_heads: tl.int32,
    R_CHUNK: tl.constexpr,
    TILE_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_HG: tl.constexpr,
    D_V: tl.constexpr,
    D_ROPE: tl.constexpr,
    HAS_ROPE: tl.constexpr,
):
    """dKV intermediate for one chunk, REUSING stored dS/P (no recompute).
    Loads Q/dO UNtransposed with a [D, BLOCK_H] strided index pattern (contiguous
    along D per head), removing the external q.transpose(1,2).contiguous() /
    do.transpose(...).contiguous() copies. Grid: (total_tokens,).

    HAS_ROPE=False (V4 zero-pad): skips Q_rope load, the dKV_rope MMA and the
    interm rope store (interm rope columns are never read by the gather)."""
    token_idx = tl.program_id(0)

    NUM_TILES: tl.constexpr = (R_CHUNK + TILE_K - 1) // TILE_K
    offs_tile = tl.arange(0, TILE_K)
    offs_v = tl.arange(0, D_V)
    offs_r = tl.arange(0, D_ROPE)

    interm_base_t = token_idx * stride_interm_t
    q_base = token_idx * stride_q_t
    do_base = token_idx * stride_do_t
    ds_base = token_idx * stride_ds_t

    for t in range(NUM_TILES):
        tile_start = t * TILE_K
        tile_offs = tile_start + offs_tile
        valid = tile_offs < R_CHUNK

        dKV_lora = tl.zeros([D_V, TILE_K], dtype=tl.float32)
        if HAS_ROPE:
            dKV_rope = tl.zeros([D_ROPE, TILE_K], dtype=tl.float32)

        for hg in range(NUM_HG):
            offs_h = hg * BLOCK_H + tl.arange(0, BLOCK_H)
            mask_h = offs_h < num_heads

            # [D_V, BLOCK_H] loaded transposed: rows=offs_v (stride 1, contiguous),
            # cols=offs_h (stride stride_q_h). No HBM transpose copy needed.
            Q_lora_T = tl.load(
                Q_ptr + q_base + offs_h[None, :] * stride_q_h + offs_v[:, None],
                mask=mask_h[None, :],
                other=0.0,
            )
            dO_T = tl.load(
                dO_ptr + do_base + offs_h[None, :] * stride_do_h + offs_v[:, None],
                mask=mask_h[None, :],
                other=0.0,
            )

            dS_val = tl.load(
                dS_ptr + ds_base + offs_h[:, None] * stride_ds_h + tile_offs[None, :],
                mask=mask_h[:, None] & valid[None, :],
                other=0.0,
            )
            P_val = tl.load(
                P_ptr + ds_base + offs_h[:, None] * stride_ds_h + tile_offs[None, :],
                mask=mask_h[:, None] & valid[None, :],
                other=0.0,
            )

            dKV_lora += tl.dot(Q_lora_T, dS_val.to(Q_lora_T.dtype)).to(tl.float32)
            dKV_lora += tl.dot(dO_T, P_val.to(dO_T.dtype)).to(tl.float32)
            if HAS_ROPE:
                Q_rope_T = tl.load(
                    Q_ptr + q_base + offs_h[None, :] * stride_q_h + (D_V + offs_r[:, None]),
                    mask=mask_h[None, :],
                    other=0.0,
                )
                dKV_rope += tl.dot(Q_rope_T, dS_val.to(Q_rope_T.dtype)).to(tl.float32)

        interm_lora_ptrs = Interm_ptr + interm_base_t + tile_offs[None, :] * stride_interm_k + offs_v[:, None]
        tl.store(interm_lora_ptrs, dKV_lora.to(tl.bfloat16), mask=valid[None, :])

        if HAS_ROPE:
            interm_rope_ptrs = (
                Interm_ptr + interm_base_t + tile_offs[None, :] * stride_interm_k + D_V + offs_r[:, None]
            )
            tl.store(interm_rope_ptrs, dKV_rope.to(tl.bfloat16), mask=valid[None, :])


@triton.jit
def _bwd_dkv_gather_acc(
    Interm_ptr,  # [T, R_CHUNK, D] bf16 — chunk intermediate
    InvPtr_ptr,  # [T+1] int32 — CSR row pointers
    InvData_ptr,  # [T*R_CHUNK] int32 — encoded as q*R_CHUNK + local_r
    dKV_acc_ptr,  # [T, D] fp32 — accumulator (read-modify-write across chunks)
    stride_interm_r: tl.int64,  # D
    stride_acc_t: tl.int64,  # D
    D_V: tl.constexpr,
    D_ROPE: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    BLOCK_K: tl.constexpr = 64,
):
    """Gather one chunk's bf16 intermediate into the fp32 dKV accumulator.
    Grid: (num_kv,) — one CTA per KV token, no atomics. Tiled BLOCK_K rows/iter.

    HAS_ROPE=False (V4 zero-pad): interm rope columns are never written, so skip
    reading/accumulating them; dKV_acc rope stays at its zero-init value."""
    k = tl.program_id(0)
    offs_v = tl.arange(0, D_V)
    offs_r = tl.arange(0, D_ROPE)
    offs_k = tl.arange(0, BLOCK_K)

    start = tl.load(InvPtr_ptr + k)
    end = tl.load(InvPtr_ptr + k + 1)

    acc_base = k.to(tl.int64) * stride_acc_t
    dkv_acc_lora = tl.load(dKV_acc_ptr + acc_base + offs_v).to(tl.float32)
    if HAS_ROPE:
        dkv_acc_rope = tl.load(dKV_acc_ptr + acc_base + D_V + offs_r).to(tl.float32)

    n_entries = end - start
    for ti in range(0, tl.cdiv(n_entries, BLOCK_K)):
        e_local = ti * BLOCK_K + offs_k
        valid = e_local < n_entries
        entry = tl.load(InvData_ptr + start + e_local, mask=valid, other=0).to(tl.int64)
        rp = entry[:, None] * stride_interm_r
        rows_v = tl.load(Interm_ptr + rp + offs_v[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        dkv_acc_lora += tl.sum(rows_v, axis=0)
        if HAS_ROPE:
            rows_r = tl.load(Interm_ptr + rp + D_V + offs_r[None, :], mask=valid[:, None], other=0.0).to(
                tl.float32
            )
            dkv_acc_rope += tl.sum(rows_r, axis=0)

    tl.store(dKV_acc_ptr + acc_base + offs_v, dkv_acc_lora)
    if HAS_ROPE:
        tl.store(dKV_acc_ptr + acc_base + D_V + offs_r, dkv_acc_rope)


def sparse_mla_bwd_triton(q, kv, o, do, topk_indices, lse, attn_sink=None, kv_lora_rank=512, scale=None):
    """DeepSeek-V4 sparse-MLA backward (plain Triton / MFMA, non-atomic).

    Returns ``(dq, dkv, d_sink)`` with ``dkv`` shaped like ``kv`` and ``d_sink``
    ``[num_heads]`` fp32 (or ``None`` when ``attn_sink`` is None).
    """
    assert q.is_contiguous() and o.is_contiguous() and do.is_contiguous()
    assert topk_indices.is_contiguous() and lse.is_contiguous()
    total_tokens, num_heads, d_qk = q.shape
    rope_rank = d_qk - kv_lora_rank
    topk = topk_indices.shape[1]
    if scale is None:
        scale = 1.0 / (d_qk**0.5)
    if kv.dim() == 2:
        kv = kv.unsqueeze(1)
    assert kv.is_contiguous()
    num_kv = kv.shape[0]

    has_sink = attn_sink is not None
    if has_sink:
        assert attn_sink.dtype == torch.float32 and attn_sink.shape == (num_heads,)

    # Delta = rowsum(O*dO) is fused into the first dQ chunk (no preprocess kernel).
    delta = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=q.device)

    # ---- config (mirror the gluon bwd chunking) ----
    # R_CHUNK (rank-chunk width): dQ is read-modify-written across chunks, so more chunks = more
    # redundant dq reloads + repeated launches/CSR builds. H>=128 has large dq RMW volume, so one
    # chunk over the whole topk (memory-bounded) wins; H<=64's smaller dq RMW is outweighed by the
    # larger per-chunk buffers/occupancy, so 256 stays the cap.
    if num_heads >= 128:
        R_CHUNK = min(topk, 1536)
    else:
        R_CHUNK = min(256, topk)
    BH_DQ, TK_DQ = 64, 16
    # dKV-intermediate tiling. Default (BH_DKV=32, TK_DKV=64) is best for H>=128 and non-128-aligned
    # chunks. For H<=64 with a 128-aligned chunk, a wider TILE_K=128 over one head-group (BH_DKV=64,
    # NUM_HG=1) cuts Q/dO re-loads and issues fuller MMAs; guarded off for H>=128 and non-128 chunks.
    # Its LDS overflows 160 KB only with ping-pong on, but the bwd compiles under
    # scoped_amd_triton_knobs_disabled() below, so it fits and is the default where it applies.
    use_wide_dkv = num_heads <= 64 and R_CHUNK % 128 == 0
    BH_DKV, TK_DKV = (64, 128) if use_wide_dkv else (32, 64)
    num_hg_dq = triton.cdiv(num_heads, BH_DQ)
    num_hg_dkv = triton.cdiv(num_heads, BH_DKV)

    # In the V4 single-latent form the D_ROPE block of q/kv is a zero pad (RoPE is
    # baked in-place over the 512 latent) and its gradient is discarded by the
    # adapter (dq[..., :D_V], dkv[..., :D_V]). So all rope MMAs / K_rope loads /
    # interm-rope traffic compute a provably-zero result that is thrown away.
    # Skip them: bit-identical non-rope outputs, zero rope outputs (== gluon).
    HAS_ROPE = False

    dq = torch.empty_like(q)
    chunk_dS = torch.empty(total_tokens, num_heads, R_CHUNK, dtype=torch.bfloat16, device=q.device)
    chunk_P = torch.empty(total_tokens, num_heads, R_CHUNK, dtype=torch.bfloat16, device=q.device)
    dkv_acc = torch.zeros(num_kv, d_qk, dtype=torch.float32, device=q.device)
    interm = torch.empty(total_tokens, R_CHUNK, d_qk, dtype=torch.bfloat16, device=q.device)

    # pad topk to an R_CHUNK multiple (-1 = invalid).
    topk_padded_len = ((topk + R_CHUNK - 1) // R_CHUNK) * R_CHUNK
    if topk_padded_len != topk:
        pad = torch.full((total_tokens, topk_padded_len - topk), -1, dtype=torch.int32, device=q.device)
        topk_padded = torch.cat([topk_indices, pad], dim=1).contiguous()
    else:
        topk_padded = topk_indices

    all_csr = [
        _build_inverted_topk_slice(topk_padded[:, rs : rs + R_CHUNK], rs, R_CHUNK, num_kv=num_kv)
        for rs in range(0, topk, R_CHUNK)
    ]

    # The AMD ping-pong / async-copy knobs primus_turbo enables globally are a
    # pessimization for the whole triton_v2 backward, and it overflows the 160 KB LDS limit for
    # wide dKV tiling. Compile the entire bwd (dQ / dKV-intermediate / gather) with them
    # disabled; the knobs are read at compile time and are not in Triton's cache key, so
    # this pins the faster non-ping-pong schedule without touching any kernel elsewhere.
    with scoped_amd_triton_knobs_disabled():
        for chunk_idx, r_start in enumerate(range(0, topk, R_CHUNK)):
            is_first = r_start == 0

            _bwd_chunk_dq_store_ds[(total_tokens, num_hg_dq)](
                q,
                kv,
                do,
                topk_padded,
                lse,
                delta,
                o,
                dq,
                chunk_dS,
                chunk_P,
                q.stride(0),
                q.stride(1),
                kv.stride(0),
                do.stride(0),
                do.stride(1),
                o.stride(0),
                o.stride(1),
                dq.stride(0),
                dq.stride(1),
                topk_padded.stride(0),
                chunk_dS.stride(0),
                chunk_dS.stride(1),
                scale,
                num_heads,
                r_start,
                R_CHUNK=R_CHUNK,
                BLOCK_H=BH_DQ,
                TILE_K=TK_DQ,
                D_V=kv_lora_rank,
                D_ROPE=rope_rank,
                HAS_ROPE=HAS_ROPE,
                IS_FIRST_CHUNK=is_first,
                num_warps=4,
                waves_per_eu=1,
            )

            # ping-pong is already off for the whole loop (outer scope), which the wide
            # dKV tiling needs to fit LDS.
            _bwd_compute_dkv_intermediate[(total_tokens,)](
                q,
                do,
                chunk_dS,
                chunk_P,
                interm,
                q.stride(0),
                q.stride(1),
                do.stride(0),
                do.stride(1),
                chunk_dS.stride(0),
                chunk_dS.stride(1),
                interm.stride(0),
                interm.stride(1),
                num_heads,
                R_CHUNK=R_CHUNK,
                TILE_K=TK_DKV,
                BLOCK_H=BH_DKV,
                NUM_HG=num_hg_dkv,
                D_V=kv_lora_rank,
                D_ROPE=rope_rank,
                HAS_ROPE=HAS_ROPE,
                num_warps=4,
            )

            inv_ptr, inv_data = all_csr[chunk_idx]
            _bwd_dkv_gather_acc[(num_kv,)](
                interm,
                inv_ptr,
                inv_data,
                dkv_acc,
                interm.stride(1),
                dkv_acc.stride(0),
                D_V=kv_lora_rank,
                D_ROPE=rope_rank,
                HAS_ROPE=HAS_ROPE,
                num_warps=4,
            )

    d_sink = None
    if has_sink:
        d_sink = -(torch.exp(attn_sink.unsqueeze(0) - lse) * delta).sum(0)

    dkv = dkv_acc.to(kv.dtype).unsqueeze(1)
    return dq, dkv, d_sink
