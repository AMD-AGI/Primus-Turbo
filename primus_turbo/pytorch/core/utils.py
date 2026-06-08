###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import functools
from typing import Tuple

import torch


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
