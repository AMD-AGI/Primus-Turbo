###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import functools
from typing import Any, Callable, Hashable, Tuple

import torch

from primus_turbo.common.logger import logger

# Per-shape memo of whether a FlyDSL backend successfully built/ran for a shape.
_FLYDSL_BUILD_CACHE: dict = {}


def flydsl_can_build(key: Hashable, run: Callable[[], Any]) -> bool:
    """Per-shape build/run probe for a FlyDSL backend's ``can_handle``.

    Runs ``run`` once per shape (JIT compile + autotune + one launch); on any
    exception the shape is memoized unsupported so the dispatcher falls back to
    another backend (Triton for the grouped MX path). Only exceptions are caught
    (never masks a wrong result), warned once. Skipped under CUDA graph capture
    (its D2H/timing is illegal there).
    """
    cached = _FLYDSL_BUILD_CACHE.get(key)
    if cached is not None:
        return cached
    if torch.cuda.is_current_stream_capturing():
        return True
    try:
        run()
        buildable = True
    except Exception as e:
        logger.warning(
            f"FlyDSL build/run probe failed for shape {key}: "
            f"{type(e).__name__}: {e}; falling back to another backend.",
            once=True,
        )
        buildable = False
    _FLYDSL_BUILD_CACHE[key] = buildable
    return buildable


@functools.lru_cache
def _get_device_compute_capability(device: torch.device) -> Tuple[int, int]:
    props = torch.cuda.get_device_properties(device)
    return (props.major, props.minor)


def get_device_compute_capability() -> Tuple[int, int]:
    """CUDA compute capability of current GPU"""
    return _get_device_compute_capability(torch.cuda.current_device())


def is_gfx950() -> bool:
    return get_device_compute_capability() == (9, 5)


def is_gfx942() -> bool:
    return get_device_compute_capability() == (9, 4)


@functools.lru_cache
def get_num_cus() -> int:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())

    return props.multi_processor_count
