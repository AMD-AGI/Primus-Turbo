###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Low-level probing primitives for ``mask_mod`` / ``score_mod`` callables.

These are the only places that actually *call* a user-supplied flex callable. They
normalise the two calling styles torch allows (vectorised tensor arguments and
scalar-style Python callables that use ``and`` / ``or``) behind one interface, and
sample single rows rather than materialising a whole ``[B, H, Sq, Skv]`` mask so
classification of a long sequence stays O(S log S) rather than O(S**2).
"""

from typing import Any, Callable

import torch


def _to_scalar_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"mask_mod must return a scalar boolean, got shape={tuple(value.shape)}")
        return bool(value.item())
    return bool(value)


def _to_scalar_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"score_mod must return a scalar score, got shape={tuple(value.shape)}")
        return float(value.item())
    return float(value)


def _call_mask_mod(mask_mod: Callable, b: int, h: int, q_idx: int, kv_idx: int) -> bool:
    """Call a mask_mod at a single index, tolerating int- or tensor-only signatures."""
    try:
        return _to_scalar_bool(mask_mod(b, h, q_idx, kv_idx))
    except (TypeError, RuntimeError):
        return _to_scalar_bool(
            mask_mod(
                torch.tensor(b),
                torch.tensor(h),
                torch.tensor(q_idx),
                torch.tensor(kv_idx),
            )
        )


def _call_score_mod(score_mod: Callable, score: float, b: int, h: int, q_idx: int, kv_idx: int) -> float:
    try:
        return _to_scalar_float(score_mod(score, b, h, q_idx, kv_idx))
    except (TypeError, RuntimeError):
        return _to_scalar_float(
            score_mod(
                torch.tensor(float(score)),
                torch.tensor(b),
                torch.tensor(h),
                torch.tensor(q_idx),
                torch.tensor(kv_idx),
            )
        )


def _probe_mask_grid(mask_mod: Callable, b: int, h: int, q_probe: int, kv_probe: int) -> torch.Tensor:
    """Evaluate ``mask_mod`` over the top-left ``q_probe x kv_probe`` grid.

    Tries a single vectorised broadcast call first (torch mask_mods used through
    ``create_block_mask`` are vectorisable by construction); falls back to an
    element-wise loop for scalar-only lambdas (e.g. those using python ``and``).
    """
    q_idx = torch.arange(q_probe).view(q_probe, 1)
    kv_idx = torch.arange(kv_probe).view(1, kv_probe)
    try:
        raw = mask_mod(b, h, q_idx, kv_idx)
        mask = torch.as_tensor(raw, dtype=torch.bool)
        if mask.shape != (q_probe, kv_probe):
            mask = mask.broadcast_to((q_probe, kv_probe))
        return mask.contiguous()
    except Exception:
        out = torch.empty((q_probe, kv_probe), dtype=torch.bool)
        for qi in range(q_probe):
            for ki in range(kv_probe):
                out[qi, ki] = _call_mask_mod(mask_mod, b, h, qi, ki)
        return out


def _probe_mask_row(mask_mod: Callable, q_pos: int, kv_len: int) -> torch.Tensor:
    """Evaluate ``mask_mod`` for a single query row ``q_pos`` over ``kv in [0, kv_len)``.

    Returns a 1D bool tensor of length ``kv_len``. Vectorised first (one broadcast
    call), falling back to an element-wise loop for scalar-only lambdas. Used by the
    sliding-window locator to verify a candidate window exactly on sampled rows.
    """
    q_idx = torch.tensor([[q_pos]])
    kv_idx = torch.arange(kv_len).view(1, kv_len)
    try:
        raw = mask_mod(0, 0, q_idx, kv_idx)
        row = torch.as_tensor(raw, dtype=torch.bool)
        if row.shape != (1, kv_len):
            row = row.broadcast_to((1, kv_len))
        return row.reshape(kv_len).contiguous()
    except Exception:
        out = torch.empty(kv_len, dtype=torch.bool)
        for ki in range(kv_len):
            out[ki] = _call_mask_mod(mask_mod, 0, 0, q_pos, ki)
        return out


def _mask_is_bh_dependent(
    mask_mod: Callable, base: torch.Tensor, B: int, H: int, q_probe: int, kv_probe: int
) -> bool:
    candidates = []
    if B > 1:
        candidates.append((B - 1, 0))
    if H > 1:
        candidates.append((0, H - 1))
    if B > 1 and H > 1:
        candidates.append((B - 1, H - 1))
    for b, h in candidates:
        other = _probe_mask_grid(mask_mod, b, h, q_probe, kv_probe)
        if not torch.equal(other, base):
            return True
    return False


def _probe_mask_pairs(
    mask_mod: Callable, q_positions: torch.Tensor, kv_positions: torch.Tensor
) -> torch.Tensor:
    """Evaluate ``mask_mod`` on the *element-wise pairs* ``zip(q_positions, kv_positions)``.

    Both inputs are 1D int64 tensors of equal length; the result is a 1D bool tensor of
    the same length. One vectorised broadcast call first (the shape passed to the
    mask_mod is ``[n, 1]`` for both indices, which every ``create_block_mask``-style
    mask_mod handles), falling back to an element-wise loop for scalar-only lambdas.
    Used to read a diagonal / sub-diagonal in O(1) kernel-free calls instead of O(S).
    """
    n = int(q_positions.numel())
    try:
        raw = mask_mod(0, 0, q_positions.view(n, 1), kv_positions.view(n, 1))
        out = torch.as_tensor(raw, dtype=torch.bool)
        if out.shape != (n, 1):
            out = out.broadcast_to((n, 1))
        return out.reshape(n).contiguous()
    except Exception:
        out = torch.empty(n, dtype=torch.bool)
        for i in range(n):
            out[i] = _call_mask_mod(mask_mod, 0, 0, int(q_positions[i]), int(kv_positions[i]))
        return out
