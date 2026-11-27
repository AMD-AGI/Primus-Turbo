###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Host-side (PyTorch) helpers of the FlyDSL flex attention kernels: FA4's block-sparse
list layout, fp8 inputs in the GPU's native format, and the split-KV combine."""

from typing import NamedTuple, Optional, Tuple

import torch

# --- Block-sparse lists -----------------------------------------------------
# Block-sparse KV lists, FA4's ``BlockSparseTensorsTorch`` (fixed-length layout).
#
# For each (batch, head, sparse Q block): ``mask_block_cnt`` / ``mask_block_idx`` list
# the sparse KV blocks that are partly masked (mask_mod is evaluated there) and
# ``full_block_cnt`` / ``full_block_idx`` the blocks that are entirely kept (mask_mod
# is skipped). Counts are ``[B, H, M]`` int32, indices ``[B, H, M, N]`` int32 packed to
# the left; ``H`` may be 1 to share one pattern across heads. ``block_size`` is the
# sparse ``(q, kv)`` block, a multiple of the kernel's ``(tile_m, tile_n)``; ``None``
# means the kernel tile.


class BlockSparseTensors(NamedTuple):
    mask_block_cnt: torch.Tensor
    mask_block_idx: torch.Tensor
    full_block_cnt: Optional[torch.Tensor] = None
    full_block_idx: Optional[torch.Tensor] = None
    block_size: Optional[Tuple[int, int]] = None


def subtile_factors(block_size, tile_m: int, tile_n: int) -> Tuple[int, int]:
    """``(q, kv)`` number of kernel tiles per sparse block."""
    sparse_q, sparse_kv = (tile_m, tile_n) if block_size is None else block_size
    if sparse_q % tile_m or sparse_kv % tile_n:
        raise ValueError(
            f"sparse block size {(sparse_q, sparse_kv)} must be a multiple of the tile {(tile_m, tile_n)}"
        )
    return sparse_q // tile_m, sparse_kv // tile_n


def broadcast_heads(tensors: BlockSparseTensors, num_heads: int) -> BlockSparseTensors:
    """Expand a head dimension of 1 to ``num_heads`` (a stride-0 view, no copy)."""

    def expand(t):
        if t is None or t.shape[1] == num_heads:
            return t
        if t.shape[1] != 1:
            raise ValueError(f"block-sparse head dim must be 1 or {num_heads}, got {t.shape[1]}")
        return t.expand(t.shape[0], num_heads, *t.shape[2:])

    return tensors._replace(
        mask_block_cnt=expand(tensors.mask_block_cnt),
        mask_block_idx=expand(tensors.mask_block_idx),
        full_block_cnt=expand(tensors.full_block_cnt),
        full_block_idx=expand(tensors.full_block_idx),
    )


def _lists_to_dense(cnt, idx, num_cols):
    sel = torch.arange(idx.shape[-1], device=idx.device) < cnt.unsqueeze(-1)
    out = torch.zeros(*idx.shape[:-1], num_cols, dtype=torch.bool, device=idx.device)
    return out.scatter(-1, idx.long().clamp(0, num_cols - 1), sel)


def _dense_to_lists(sel):
    cnt = sel.sum(-1).to(torch.int32)
    order = torch.argsort((~sel).to(torch.int8), dim=-1, stable=True).to(torch.int32)
    return cnt.contiguous(), order.contiguous()


def transpose(tensors: BlockSparseTensors, num_kv_blocks: int) -> BlockSparseTensors:
    """KV-major lists ``[B, H, N(, M)]`` (per sparse KV block, the sparse Q blocks that
    see it) from query-major ones, for the backward's dK / dV kernel."""
    num_q_blocks = tensors.mask_block_cnt.shape[-1]

    def flip(cnt, idx):
        if cnt is None:
            return None, None
        return _dense_to_lists(_lists_to_dense(cnt, idx, num_kv_blocks).transpose(-1, -2))

    mask_cnt, mask_idx = flip(tensors.mask_block_cnt, tensors.mask_block_idx)
    full_cnt, full_idx = flip(tensors.full_block_cnt, tensors.full_block_idx)
    if mask_idx.shape[-1] != num_q_blocks:
        raise AssertionError("transpose lost query blocks")
    return BlockSparseTensors(mask_cnt, mask_idx, full_cnt, full_idx, tensors.block_size)


def dense_to_block_sparse(dense_mask: torch.Tensor, block_size: Tuple[int, int]) -> BlockSparseTensors:
    """Classify a dense ``[B, H, Sq, Sk]`` bool mask into partial / full block lists
    (a testing and fallback utility; FA4's ``compute_block_sparsity`` does this on GPU
    from a mask_mod)."""
    B, H, Sq, Sk = dense_mask.shape
    bq, bk = block_size
    M, N = -(-Sq // bq), -(-Sk // bk)
    padded = torch.zeros(B, H, M * bq, N * bk, dtype=torch.bool, device=dense_mask.device)
    valid = torch.zeros_like(padded)
    padded[..., :Sq, :Sk] = dense_mask
    valid[..., :Sq, :Sk] = True
    blocks = padded.view(B, H, M, bq, N, bk)
    valid_blocks = valid.view(B, H, M, bq, N, bk)
    any_kept = blocks.any(dim=(3, 5))
    all_kept = (blocks | ~valid_blocks).all(dim=(3, 5)) & any_kept
    partial = any_kept & ~all_kept
    mask_cnt, mask_idx = _dense_to_lists(partial)
    full_cnt, full_idx = _dense_to_lists(all_kept)
    return BlockSparseTensors(mask_cnt, mask_idx, full_cnt, full_idx, block_size)


# --- FP8 inputs -------------------------------------------------------------
# FP8 inputs in the GPU's native format: FNUZ on gfx942, OCP on gfx950.
#
# OCP -> FNUZ (gfx942) reinterprets the bits: the same code is exactly half the value in
# FNUZ (one more exponent bias), so the factor of 2 goes into the descale and the result
# is exact; OCP's -0 is FNUZ's NaN and becomes +0. FNUZ -> OCP (gfx950) cannot do the
# same, because FNUZ's largest codes would double past OCP's range, so it converts by
# value instead: exact except that FNUZ's smallest subnormal rounds. NaN / inf inputs are
# not preserved.

_OCP = {"fp8_e4m3": torch.float8_e4m3fn, "fp8_e5m2": torch.float8_e5m2}
_FNUZ = {"fp8_e4m3": torch.float8_e4m3fnuz, "fp8_e5m2": torch.float8_e5m2fnuz}


def _convert(t: torch.Tensor, dtype: str, to_ocp: bool):
    """``(tensor in the target format, factor)`` with ``value_in = factor * value_out``."""
    target = _OCP[dtype] if to_ocp else _FNUZ[dtype]
    if t.dtype == target:
        return t, 1.0
    source = _FNUZ[dtype] if to_ocp else _OCP[dtype]
    if t.dtype != source:
        raise TypeError(f"expected {_OCP[dtype]} or {_FNUZ[dtype]}, got {t.dtype}")
    if to_ocp:
        return t.to(torch.float32).to(target), 1.0
    bits = t.view(torch.uint8)
    return torch.where(bits == 0x80, torch.zeros_like(bits), bits).view(target), 2.0


def dequantize_fp8(q, k, v, descales, qhead_per_kvhead: int, cu_seqlens_q=None, cu_seqlens_k=None):
    """bf16 ``x * descale`` for q / k / v with ``[B, Hk]`` descales (ones where absent);
    dense ``[B, S, H, D]`` or varlen ``[T, H, D]`` with ``cu_seqlens``."""

    def scale(t, d, cu, rep):
        if d is None:
            return t.to(torch.bfloat16)
        d = d.float().repeat_interleave(rep, dim=1)  # [B, H]
        if cu is None:
            return (t.float() * d[:, None, :, None]).to(torch.bfloat16)
        batch_of_token = torch.repeat_interleave(
            torch.arange(cu.numel() - 1, device=t.device), (cu[1:] - cu[:-1]).long(), output_size=t.shape[0]
        )
        return (t.float() * d[batch_of_token][:, :, None]).to(torch.bfloat16)

    q_d, k_d, v_d = descales
    return (
        scale(q, q_d, cu_seqlens_q, qhead_per_kvhead),
        scale(k, k_d, cu_seqlens_k, 1),
        scale(v, v_d, cu_seqlens_k, 1),
    )


def to_native_fp8(q, k, v, descales, dtype: str, ocp: bool):
    """Convert q / k / v to the native format; returns them with ``[B, Hk]`` fp32
    descales (ones where absent) that include the conversion factors."""
    q, fq = _convert(q, dtype, ocp)
    k, fk = _convert(k, dtype, ocp)
    v, fv = _convert(v, dtype, ocp)
    batch_hk = None
    for d in descales:
        if d is not None:
            batch_hk = tuple(d.shape)
    if batch_hk is None:
        raise ValueError("pass q_descale / k_descale / v_descale (shape [batch, num_kv_heads]) for fp8")
    out = []
    for d, factor in zip(descales, (fq, fk, fv)):
        d = torch.ones(batch_hk, dtype=torch.float32, device=q.device) if d is None else d.float()
        out.append((d * factor).contiguous() if factor != 1.0 else d.contiguous())
    return q, k, v, tuple(out)


# --- Split-KV combine -------------------------------------------------------
# Split-KV combine, FA4's ``flash_attn_combine`` (as torch ops for now).


def flash_attn_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    return_lse: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Merge per-split normalized outputs by their LSEs.

    ``out_partial``: ``[num_splits, B, S, H, Dv]`` or ``[num_splits, T, H, Dv]`` (fp32).
    ``lse_partial``: ``[num_splits, B, S, H]`` or ``[num_splits, T, H]`` (FA4's layout).
    Returns ``(out, lse)`` with ``lse`` shaped like one split of ``lse_partial``. A row
    no split saw gives O = 0 and LSE = -inf.
    """
    lse = torch.logsumexp(lse_partial, dim=0)
    weight = torch.exp(lse_partial - lse.unsqueeze(0)).nan_to_num(0.0)
    combined = (out_partial * weight.unsqueeze(-1)).sum(0)
    out_dtype = out_dtype or (out.dtype if out is not None else torch.bfloat16)
    if out is None:
        out = combined.to(out_dtype)
    else:
        out.copy_(combined)
    return out, (lse if return_lse else None)
