###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch

from primus_turbo.pytorch.core.backend import AutoKernelDispatcher


def _lb_group_lens(group_lens: torch.Tensor, total: int) -> torch.Tensor:
    """Evenly distribute total across num_groups (for tuning only).

    NOTE: This is intentionally workload-agnostic; it is used to stabilize
    profiling for autotune by reducing variance from highly imbalanced groups.
    """
    num_groups = int(group_lens.numel())
    if num_groups <= 0:
        return group_lens
    base, rem = divmod(int(total), num_groups)
    out = torch.full((num_groups,), base, dtype=torch.int64, device=group_lens.device)
    if rem:
        out[:rem] += 1
    return out


def _group_offs_from_lens(group_lens: torch.Tensor) -> torch.Tensor:
    """Compute prefix-sum offsets from group lengths."""
    lens = group_lens.to(dtype=torch.int64)
    offs = torch.empty((lens.numel() + 1,), dtype=torch.int64, device=lens.device)
    offs[0] = 0
    offs[1:] = torch.cumsum(lens, dim=0)
    return offs


class BaseGroupedGEMMKernelDispatcher(AutoKernelDispatcher):
    """Unified grouped-GEMM autotune dispatcher.

    During autotune/tuning, to make profiling more stable, we load-balance
    `group_lens` and use the balanced `group_lens/group_offs` for profiling and
    backend selection (profiling only).

    Subclasses must implement `make_key()` and provide `_backends` / `_cache`.
    """

    @classmethod
    @torch.no_grad()
    def tune(cls, **kwargs):
        """Tune with load-balanced group_lens (profiling only)."""
        key = cls.make_key(**kwargs)

        cached_backend = cls._cache.get(key) if cls._cache is not None else None
        if cached_backend is not None:
            return cached_backend

        a: torch.Tensor = kwargs["a"]
        group_lens: torch.Tensor = kwargs["group_lens"]
        lb_group_lens = _lb_group_lens(group_lens, int(a.size(0)))
        lb_group_offs = _group_offs_from_lens(lb_group_lens)

        prof_kwargs = dict(kwargs)
        prof_kwargs["group_lens"] = lb_group_lens
        prof_kwargs["group_offs"] = lb_group_offs

        best_backend = None
        best_time = float("inf")
        for backend in cls._backends.values():
            if backend.can_handle(**kwargs):
                torch.cuda.synchronize()
                try:
                    cur_time = cls.profile(backend, **prof_kwargs)
                except Exception:
                    cur_time = float("inf")
                finally:
                    torch.cuda.synchronize()
                if cur_time < best_time:
                    best_time = cur_time
                    best_backend = backend

        if best_backend is not None and cls._cache is not None:
            cls._cache.put(key, best_backend)
        return best_backend


class BaseGroupedGEMMVariableKKernelDispatcher(AutoKernelDispatcher):
    """Unified variable-K grouped-GEMM autotune dispatcher.

    During autotune/tuning, we compute `total_k` and load-balance `group_lens`
    accordingly, then use the balanced `group_lens/group_offs` for profiling and
    backend selection (profiling only).

    Subclasses must implement `make_key()` and provide `_backends` / `_cache`.
    """

    @classmethod
    @torch.no_grad()
    def tune(cls, **kwargs):
        """Tune with load-balanced variable-K group_lens (profiling only)."""
        key = cls.make_key(**kwargs)

        cached_backend = cls._cache.get(key) if cls._cache is not None else None
        if cached_backend is not None:
            return cached_backend

        a: torch.Tensor = kwargs["a"]
        b: torch.Tensor = kwargs["b"]
        group_lens: torch.Tensor = kwargs["group_lens"]
        trans_a: bool = kwargs["trans_a"]
        trans_b: bool = kwargs["trans_b"]
        trans_c: bool = kwargs["trans_c"]

        if trans_c:
            lhs = b
            trans_lhs = not trans_b
        else:
            lhs = a
            trans_lhs = trans_a

        total_k = int(lhs.size(0) if trans_lhs else lhs.size(1))
        lb_group_lens = _lb_group_lens(group_lens, total_k)
        lb_group_offs = _group_offs_from_lens(lb_group_lens)

        prof_kwargs = dict(kwargs)
        prof_kwargs["group_lens"] = lb_group_lens
        prof_kwargs["group_offs"] = lb_group_offs

        best_backend = None
        best_time = float("inf")
        for backend in cls._backends.values():
            if backend.can_handle(**kwargs):
                torch.cuda.synchronize()
                try:
                    cur_time = cls.profile(backend, **prof_kwargs)
                except Exception:
                    cur_time = float("inf")
                finally:
                    torch.cuda.synchronize()
                if cur_time < best_time:
                    best_time = cur_time
                    best_backend = backend

        if best_backend is not None and cls._cache is not None:
            cls._cache.put(key, best_backend)
        return best_backend
