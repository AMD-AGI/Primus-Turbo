###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
import warnings

import torch

import primus_turbo.pytorch._C.runtime as runtime

__all__ = ["TurboStream"]


class TurboStream:
    """
    A class to create and manage a HIP stream with CU masks,
    exposed as a Python interface.

    This class wraps the low-level `create_stream_with_cu_masks` binding
    and provides a convenient `torch.cuda.ExternalStream` for use with
    PyTorch stream APIs.

    Example:
        >>> cu_masks = [0x0000000F, 0x00001234]  # Example CU mask values
        >>> ts = TurboStream(device="cuda:0", cu_masks=cu_masks)
        >>> with torch.cuda.stream(ts.torch_stream):
        ...     # Your CUDA operations here

    Note:
        - The underlying HIP stream pointer is managed by this class.
        - The user is responsible for ensuring that work on the stream
          has completed before the stream is destroyed.
    """

    def __init__(self, device: torch.device | str | int | None, cu_masks: list[int] | None):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/ROCm is not available")

        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device("cuda", device)
        elif not isinstance(device, torch.device):
            raise TypeError(f"Unsupported device type: {type(device)}")

        # Check
        if device.type != "cuda":
            raise ValueError(f"device.type must be 'cuda', got {device.type}")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        # Mask
        if not isinstance(cu_masks, (list, tuple)) or not all(isinstance(x, int) for x in cu_masks):
            num_cu = torch.cuda.get_device_properties(device.index).multi_processor_count
            cu_masks = [0xFFFFFFFF] * ((num_cu + 31) // 32)
        else:
            cu_masks = [max(0, min(int(x), 0xFFFFFFFF)) for x in cu_masks]

        self._device = device
        self._raw_stream = runtime.create_stream_with_cu_masks(self._device.index, cu_masks)
        self._torch_stream = torch.cuda.ExternalStream(self._raw_stream, device=self._device)

    @property
    def torch_stream(self):
        return self._torch_stream

    def __del__(self):
        if hasattr(self, "_raw_stream"):
            try:
                runtime.destroy_stream(self._device.index, self._raw_stream)
            except Exception as e:
                warnings.warn(f"TurboStream destroy failed: {e}", RuntimeWarning)
