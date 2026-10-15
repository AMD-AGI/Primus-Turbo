###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Block-sparsity plans for flex attention.

A plan says, per ``(batch, head, q_block)``, which KV blocks to visit, split into two
ordered-sparse lists (counts plus left-packed indices, the representation the kernels
read):

  * ``full``   -- every position kept; the kernel skips masking entirely.
  * ``masked`` -- some positions kept; the kernel applies ``mask_mod`` per element.

Empty blocks appear in neither list and are never visited.

The backward's dK/dV phase walks the transpose (per KV block, the Q blocks that attend
it), and both backward phases evaluate ``mask_mod`` on every block they visit, so they
read one combined list per direction; :func:`backward_block_plans` derives and memoizes
both from the forward plan.
"""

import weakref
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from primus_turbo.triton.flex_attention.flex_attention_block_mask_kernel import (
    BLOCK_FULL,
    BLOCK_PARTIAL,
    classify_blocks_kernel,
)

__all__ = ["BlockList", "BlockPlan", "backward_block_plans", "classify_blocks"]


@dataclass(frozen=True)
class BlockList:
    """Ordered-sparse block indices.

    ``count``: int32 ``[B, H, num_rows]``; ``index``: int32 ``[B, H, num_rows, max_count]``,
    packed left (entries at or past ``count`` are ignored). ``num_cols`` is the width of
    the grid it was built from, kept so callers can bound-check without a device sync.
    """

    count: torch.Tensor
    index: torch.Tensor
    num_cols: int

    @classmethod
    def from_flags(cls, flags: torch.Tensor, full_width: bool = False) -> "BlockList":
        """Build from a ``[B, H, num_rows, num_cols]`` bool grid.

        The index tensor is trimmed to the largest count, which is read back from the
        device (one sync). With ``full_width`` it spans every column instead, so nothing
        is read back.
        """
        counts = flags.sum(dim=-1).to(torch.int32)
        num_cols = flags.shape[-1]
        if full_width:
            max_count = max(num_cols, 1)
        else:
            max_count = max(int(counts.max().item()) if counts.numel() else 0, 1)
        order = torch.argsort(
            (~flags).to(torch.int32) * num_cols + torch.arange(num_cols, device=flags.device),
            dim=-1,
            stable=True,
        )
        return cls(counts, order[..., :max_count].to(torch.int32).contiguous(), num_cols)

    def to_dense(self) -> torch.Tensor:
        """Inverse of :meth:`from_flags`."""
        b, h, rows, max_entries = self.index.shape
        valid = torch.arange(max_entries, device=self.index.device).view(1, 1, 1, -1) < self.count.unsqueeze(
            -1
        )
        dense = torch.zeros(b, h, rows, self.num_cols + 1, dtype=torch.bool, device=self.index.device)
        dense.scatter_(-1, torch.where(valid, self.index.long(), self.num_cols), True)
        return dense[..., : self.num_cols]

    def expand(self, batch: int, heads: int) -> "BlockList":
        """Broadcast size-1 batch/head dims with stride-0 views; the kernels index the
        lists by their strides, so a size-1 dim must not be read past its end."""
        return BlockList(
            self.count.expand(batch, heads, -1),
            self.index.expand(batch, heads, -1, -1),
            self.num_cols,
        )


@dataclass(frozen=True)
class BlockPlan:
    """Which KV blocks each Q block visits. ``full`` is None when no block is fully kept.

    A ``sync_free`` plan, and every plan derived from it, is built without reading
    anything back from the device: its index tensors span the full grid width and its
    ``full`` list is always present (possibly empty). Varlen masks are rebuilt for every
    batch, so they use this; dense masks are built once and trimmed.
    """

    block_size: Tuple[int, int]
    masked: BlockList
    full: Optional[BlockList] = None
    sync_free: bool = False

    # Kernel-facing names.
    @property
    def mask_block_cnt(self) -> torch.Tensor:
        return self.masked.count

    @property
    def mask_block_idx(self) -> torch.Tensor:
        return self.masked.index

    @property
    def full_block_cnt(self) -> Optional[torch.Tensor]:
        return None if self.full is None else self.full.count

    @property
    def full_block_idx(self) -> Optional[torch.Tensor]:
        return None if self.full is None else self.full.index

    @property
    def num_kv_blocks(self) -> int:
        return self.masked.num_cols

    def blocks_fit_within(self, seqlen_k: int) -> bool:
        """True when no listed KV block can straddle ``seqlen_k``, so the kernel may drop
        its per-block bounds masking. Checked on the grid width: no device sync."""
        return self.num_kv_blocks * self.block_size[1] <= seqlen_k

    @classmethod
    def from_categories(
        cls, categories: torch.Tensor, block_size: Tuple[int, int], sync_free: bool = False
    ) -> "BlockPlan":
        """Build from an int8 ``[B, H, num_q_blocks, num_kv_blocks]`` category grid."""
        full = categories == BLOCK_FULL
        has_full = sync_free or bool(full.any())
        return cls(
            block_size=tuple(block_size),
            masked=BlockList.from_flags(categories == BLOCK_PARTIAL, full_width=sync_free),
            full=BlockList.from_flags(full, full_width=sync_free) if has_full else None,
            sync_free=sync_free,
        )

    def expand(self, batch: int, heads: int) -> "BlockPlan":
        return BlockPlan(
            self.block_size,
            self.masked.expand(batch, heads),
            None if self.full is None else self.full.expand(batch, heads),
            self.sync_free,
        )

    def combined(self) -> "BlockPlan":
        """Every visited block in one masked list (what both backward phases read)."""
        dense = self.masked.to_dense()
        if self.full is not None:
            dense = dense | self.full.to_dense()
        return BlockPlan(
            self.block_size, BlockList.from_flags(dense, full_width=self.sync_free), sync_free=self.sync_free
        )

    def transposed(self) -> "BlockPlan":
        """Per-KV-block lists of the Q blocks that attend it (the dK/dV direction)."""
        flags = self.masked.to_dense().transpose(-2, -1).contiguous()
        return BlockPlan(
            self.block_size, BlockList.from_flags(flags, full_width=self.sync_free), sync_free=self.sync_free
        )


def classify_blocks(
    mask_mod,
    batch: int,
    heads: int,
    num_q_blocks: int,
    num_kv_blocks: int,
    block_size: Tuple[int, int],
    device,
    q_len: int = 0,
    kv_len: int = 0,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run ``mask_mod`` over every tile and return the int8 category grid.

    Dense: pass ``q_len``/``kv_len``. Varlen: pass ``cu_seqlens_q``/``cu_seqlens_k``;
    ``batch`` is then the number of sequences and indices are sequence-local.
    """
    out = torch.empty(batch, heads, num_q_blocks, num_kv_blocks, dtype=torch.int8, device=device)
    is_varlen = cu_seqlens_q is not None
    classify_blocks_kernel[(num_kv_blocks, num_q_blocks, batch * heads)](
        out,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        q_len,
        kv_len,
        cu_seqlens_q,
        cu_seqlens_k,
        MASK_MOD=mask_mod,
        NUM_HEADS=heads,
        BLOCK_Q=block_size[0],
        BLOCK_KV=block_size[1],
        IS_VARLEN=is_varlen,
        num_warps=4,
    )
    return out


# Deriving the backward lists costs a few small kernels, plus the device sync that sizes
# the packed index tensor unless the plan is sync-free. A reused mask should pay that
# once, so memoize per plan, keyed by the identity of its index tensor and dropped when
# the plan is freed.
_BWD_CACHE: dict = {}


def backward_block_plans(plan: BlockPlan) -> Tuple[BlockPlan, BlockPlan]:
    """``(dq_plan, dkdv_plan)`` for the backward, memoized per plan."""
    key = id(plan.mask_block_idx)
    hit = _BWD_CACHE.get(key)
    if hit is None:
        dq_plan = plan.combined()
        hit = (dq_plan, dq_plan.transposed())
        _BWD_CACHE[key] = hit
        weakref.finalize(plan.mask_block_idx, _BWD_CACHE.pop, key, None)
    return hit
