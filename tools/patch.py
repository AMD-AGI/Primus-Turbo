import os
import warnings
from typing import Optional

import torch
from torch.utils import cpp_extension


def patch_torch_extension():
    # copy from torchv2.8.0-rc8
    def _get_rocm_arch_flags(cflags: Optional[list[str]] = None) -> list[str]:
        # If cflags is given, there may already be user-provided arch flags in it
        # (from `extra_compile_args`). If user also specified -fgpu-rdc or -fno-gpu-rdc, we
        # assume they know what they're doing. Otherwise, we force -fno-gpu-rdc default.
        has_gpu_rdc_flag = False
        if cflags is not None:
            has_custom_flags = False
            for flag in cflags:
                if "amdgpu-target" in flag or "offload-arch" in flag:
                    has_custom_flags = True
                elif "gpu-rdc" in flag:
                    has_gpu_rdc_flag = True
            if has_custom_flags:
                return [] if has_gpu_rdc_flag else ["-fno-gpu-rdc"]
        # Use same defaults as used for building PyTorch
        # Allow env var to override, just like during initial cmake build.
        _archs = os.environ.get("PYTORCH_ROCM_ARCH", None)
        if not _archs:
            archFlags = torch._C._cuda_getArchFlags()
            if archFlags:
                archs = archFlags.split()
            else:
                archs = []
        else:
            archs = _archs.replace(" ", ";").split(";")
        flags = [f"--offload-arch={arch}" for arch in archs]
        flags += [] if has_gpu_rdc_flag else ["-fno-gpu-rdc"]
        return flags

    cpp_extension._get_rocm_arch_flags = _get_rocm_arch_flags

    warnings.warn("Patch torch.utils.cpp_extension._get_rocm_arch_flags to support -fgpu-rdc flag!")
