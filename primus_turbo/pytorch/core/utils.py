import functools
from typing import Tuple

import torch
import triton.language as tl


def mapping_triton_language_dtype(torch_dtype):
    DTYPE_MAPPING = {
        # ------- float -------
        torch.half: tl.float16,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
        torch.float64: tl.float64,
        # ------- int -------
        torch.int8: tl.int8,
        torch.int16: tl.int16,
        torch.int32: tl.int32,
        torch.int64: tl.int64,
        # ------- uint -------
        torch.uint8: tl.uint8,
        torch.uint16: tl.uint16,
        torch.uint32: tl.uint32,
        torch.uint64: tl.uint64,
        # ------- float8 ------
        torch.float8_e4m3fn: tl.float8e4nv,
        torch.float8_e5m2: tl.float8e5,
        torch.float8_e4m3fnuz: tl.float8e4b8,
        torch.float8_e5m2fnuz: tl.float8e5b16,
    }
    assert torch_dtype in DTYPE_MAPPING.keys(), f"Not support mapping dtype: {torch.dtype}"

    return DTYPE_MAPPING[torch_dtype]


@functools.lru_cache
def _get_device_compute_capability(device: torch.device) -> Tuple[int, int]:
    props = torch.cuda.get_device_properties(device)
    return (props.major, props.minor)


def get_device_compute_capability() -> Tuple[int, int]:
    """CUDA compute capability of current GPU"""
    return _get_device_compute_capability(torch.cuda.current_device())
