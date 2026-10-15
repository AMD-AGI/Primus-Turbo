###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Block masks for flex attention, modelled on ``torch.nn.attention.flex_attention``.

A ``mask_mod(b, h, q_idx, kv_idx) -> bool`` is a ``@triton.jit`` function: the same
predicate builds the block lists (``create_block_mask`` evaluates it per tile on the GPU)
and masks individual positions of partially kept blocks inside the attention kernel.
``q_idx`` / ``kv_idx`` are broadcastable index tiles; ``b`` and ``h`` are scalars (0 for
a dimension the mask is broadcast over). Under varlen the indices are sequence-local and
``b`` is the sequence index.

``causal_mask``, ``sliding_window_mask(...)`` and ``noop_mask`` are recognized: for
equal query/key lengths they can also run on the kernel's dedicated causal / window
path, which needs no block lists.

``BLOCK_SIZE`` is the granularity of the exposed block lists. The kernels may run each
pass at a different granularity -- every block size is exact, since partial blocks
evaluate ``mask_mod`` per element. With autotuning on (the default;
``PRIMUS_TURBO_FLEX_ATTENTION_AUTOTUNE=0`` disables it) the first call for a given mask
and input shape times the fast path and block sizes 64 / 128 separately for the forward
and the backward, and reuses the winners: the forward usually prefers big tiles, the
backward smaller ones.

Dense block masks are memoized: a ``mask_mod`` sees only indices (constants are baked
into its source, and it cannot read tensors), so the lists depend on nothing but the
mask, the shape and the block size, and ``create_block_mask`` returns the same object
for a repeated call. Varlen lists depend on ``cu_seqlens``, so they are rebuilt per
call, without reading anything back from the device.
"""

import functools
from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
import triton

from primus_turbo.pytorch.kernels.flex_attention.flex_attention_block_mask_utils import (
    BlockPlan,
    classify_blocks,
)
from primus_turbo.pytorch.kernels.flex_attention.flex_attention_jit_utils import (
    build_jit,
    literal_tag,
)

__all__ = [
    "BlockMask",
    "causal_mask",
    "create_block_mask",
    "create_block_mask_varlen",
    "noop_mask",
    "sliding_window_mask",
]

# Built-in mask_mod -> (causal, window_left, window_right) for the kernel's fast path.
_FAST_PATH = {}


@triton.jit
def noop_mask(b, h, q_idx, kv_idx):
    """Keep every position."""
    return (q_idx >= 0) & (kv_idx >= 0)


@triton.jit
def causal_mask(b, h, q_idx, kv_idx):
    """Keep keys at or before the query: ``kv_idx <= q_idx`` (top-left aligned)."""
    return q_idx >= kv_idx


_FAST_PATH[noop_mask] = (False, -1, -1)
_FAST_PATH[causal_mask] = (True, -1, -1)


@functools.lru_cache(maxsize=None)
def sliding_window_mask(window_left: int, window_right: int = 0):
    """Keep keys with ``q_idx - window_left <= kv_idx <= q_idx + window_right``.

    ``window_right=0`` is a causal sliding window.
    """
    left, right = int(window_left), int(window_right)
    if left < 0 or right < 0:
        raise ValueError(f"window bounds must be non-negative, got ({left}, {right})")
    name = f"sliding_window_mask_{literal_tag(left)}_{literal_tag(right)}"
    src = f"""
@triton.jit
def {name}(b, h, q_idx, kv_idx):
    rel = q_idx - kv_idx
    return (rel <= {left}) & (rel >= {-right})
"""
    fn = build_jit(src, name, name)
    _FAST_PATH[fn] = (right == 0, left, right)
    return fn


def _block_size(block_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    q_bs, kv_bs = (block_size, block_size) if isinstance(block_size, int) else tuple(block_size)
    if q_bs != kv_bs:
        # The backward needs BLOCK_N % BLOCK_M == 0 in its dK/dV sweep and the reverse
        # in its dQ sweep, and both are pinned to the block size.
        raise ValueError(f"flex attention block masks must be square, got BLOCK_SIZE={block_size}")
    if q_bs < 16 or q_bs & (q_bs - 1):
        raise ValueError(f"BLOCK_SIZE must be a power of two >= 16, got {q_bs}")
    return q_bs, kv_bs


class BlockMask:
    """Which (query-block, key-block) tiles to visit, plus the ``mask_mod`` for partial ones.

    Build with :func:`create_block_mask` or :func:`create_block_mask_varlen`. The block
    lists (``kv_num_blocks``, ``kv_indices``, ``full_kv_num_blocks``,
    ``full_kv_indices``) are ``[B, H, num_q_blocks, ...]`` int32, as in PyTorch; for a
    built-in mask on the fast path they are only computed if read.
    """

    def __init__(
        self,
        mask_mod,
        shape: Tuple[int, int, int, int],
        block_size: Tuple[int, int],
        device,
        fast_path: Optional[Tuple[bool, int, int]] = None,
        plan: Optional[BlockPlan] = None,
        cu_seqlens: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        self.mask_mod = mask_mod
        self.shape = shape
        self.BLOCK_SIZE = block_size
        self.device = device
        self._fast_path = fast_path
        self._plans = {} if plan is None else {plan.block_size[0]: plan}
        self._cu_seqlens = cu_seqlens

    @property
    def seq_lengths(self) -> Tuple[int, int]:
        """``(Q_LEN, KV_LEN)``; the longest sequences for a varlen mask."""
        return self.shape[2], self.shape[3]

    @property
    def is_varlen(self) -> bool:
        return self._cu_seqlens is not None

    @property
    def plan(self) -> BlockPlan:
        """The block lists at ``BLOCK_SIZE``."""
        return self.plan_at(self.BLOCK_SIZE[0])

    def plan_at(self, block_size: int) -> BlockPlan:
        """The block lists at another (square) block size, built on first use.

        Any block size is exact -- partial blocks evaluate ``mask_mod`` per element -- so
        the kernels may run the forward and backward at different granularities.
        """
        if block_size not in self._plans:
            batch, heads, q_len, kv_len = self.shape
            cu_q, cu_k = self._cu_seqlens if self._cu_seqlens is not None else (None, None)
            categories = classify_blocks(
                self.mask_mod,
                batch,
                heads,
                triton.cdiv(q_len, block_size),
                triton.cdiv(kv_len, block_size),
                (block_size, block_size),
                self.device,
                q_len=q_len,
                kv_len=kv_len,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
            )
            self._plans[block_size] = BlockPlan.from_categories(
                categories, (block_size, block_size), sync_free=self.is_varlen
            )
        return self._plans[block_size]

    @property
    def kv_num_blocks(self) -> torch.Tensor:
        return self.plan.mask_block_cnt

    @property
    def kv_indices(self) -> torch.Tensor:
        return self.plan.mask_block_idx

    @property
    def full_kv_num_blocks(self) -> Optional[torch.Tensor]:
        return self.plan.full_block_cnt

    @property
    def full_kv_indices(self) -> Optional[torch.Tensor]:
        return self.plan.full_block_idx

    def materialize(self) -> "BlockMask":
        """Build the block lists now (they are otherwise built on first use)."""
        _ = self.plan
        return self

    def to_dense(self) -> torch.Tensor:
        """Bool ``[B, H, num_q_blocks, num_kv_blocks]``: the tiles the kernel visits."""
        dense = self.plan.masked.to_dense()
        if self.plan.full is not None:
            dense = dense | self.plan.full.to_dense()
        return dense

    def sparsity(self) -> float:
        """Percentage of tiles skipped."""
        return 100.0 * (1.0 - self.to_dense().float().mean().item())

    def __repr__(self) -> str:
        path = "fast path" if self._fast_path is not None else "block-sparse"
        varlen = ", varlen" if self.is_varlen else ""
        return f"BlockMask(shape={self.shape}, BLOCK_SIZE={self.BLOCK_SIZE}, {path}{varlen})"


# Dense masks by everything that determines their lists, least recently used first.
_DENSE_MASKS: "OrderedDict[tuple, BlockMask]" = OrderedDict()
_DENSE_MASK_CACHE_SIZE = 64


def _device_key(device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def create_block_mask(
    mask_mod,
    B: Optional[int],
    H: Optional[int],
    Q_LEN: int,
    KV_LEN: int,
    device: Union[str, torch.device] = "cuda",
    BLOCK_SIZE: Union[int, Tuple[int, int]] = 128,
) -> BlockMask:
    """Block mask for :func:`flex_attention`. ``B`` / ``H`` = None broadcasts the mask
    over that dimension (``mask_mod`` then sees 0 for it).

    Memoized: a repeated call with the same arguments returns the same object, along
    with every block list it has built since (at any block size, and for the backward).
    The cache keys on the ``mask_mod`` object, so build parameterized masks through
    ``functools.lru_cache`` (as ``sliding_window_mask`` does) to keep hitting it.
    """
    block_size = _block_size(BLOCK_SIZE)
    shape = (B or 1, H or 1, int(Q_LEN), int(KV_LEN))
    device = _device_key(device)
    key = (mask_mod, shape, block_size, device)
    mask = _DENSE_MASKS.get(key)
    if mask is not None:
        _DENSE_MASKS.move_to_end(key)
        return mask
    fast = _FAST_PATH.get(mask_mod) if Q_LEN == KV_LEN else None
    mask = BlockMask(mask_mod, shape, block_size, device, fast_path=fast)
    # Build the lists up front, as torch does, unless the fast path never needs them.
    if fast is None:
        mask.materialize()
    _DENSE_MASKS[key] = mask
    if len(_DENSE_MASKS) > _DENSE_MASK_CACHE_SIZE:
        _DENSE_MASKS.popitem(last=False)
    return mask


def create_block_mask_varlen(
    mask_mod,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    H: Optional[int] = None,
    BLOCK_SIZE: Union[int, Tuple[int, int]] = 128,
) -> BlockMask:
    """Block mask for :func:`flex_attention_varlen` over packed sequences.

    Block indices are sequence-local, padded to the longest sequence. A built-in mask
    takes the fast path when queries and keys share one ``cu_seqlens`` tensor
    (self-attention, so every sequence's query and key lengths agree).

    Not memoized, since the lists depend on the ``cu_seqlens`` values. They are built
    without a device sync (index tensors span the full block grid), so a per-batch
    rebuild does not stall the host.
    """
    block_size = _block_size(BLOCK_SIZE)
    shape = (cu_seqlens_q.numel() - 1, H or 1, int(max_seqlen_q), int(max_seqlen_k))
    fast = _FAST_PATH.get(mask_mod) if cu_seqlens_q is cu_seqlens_k else None
    mask = BlockMask(
        mask_mod,
        shape,
        block_size,
        cu_seqlens_q.device,
        fast_path=fast,
        cu_seqlens=(cu_seqlens_q, cu_seqlens_k),
    )
    return mask if fast is not None else mask.materialize()
