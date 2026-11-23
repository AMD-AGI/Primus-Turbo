###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared skip marks and small helpers for the gfx1250 (MI455X) kernel tests."""

from __future__ import annotations

import importlib

import pytest
import torch


def _arch() -> str:
    if not torch.cuda.is_available():
        return ""
    return getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or ""


def is_gfx1250() -> bool:
    return _arch().startswith("gfx1250")


def gluon_available() -> bool:
    """Triton with the Gluon gfx1250 (cdna5 / TDM) API, i.e. the Gluon GEMM kernels import."""
    try:
        importlib.import_module("triton.experimental.gluon")
        importlib.import_module("primus_turbo.gluon.gemm.gemm_fp8blk_kernel")
    except Exception:
        return False
    return True


def flydsl_available() -> bool:
    try:
        importlib.import_module("flydsl")
        importlib.import_module("primus_turbo.pytorch.kernels.gemm.gemm_fp8_flydsl_gfx1250_impl")
    except Exception:
        return False
    return True


requires_gfx1250 = pytest.mark.skipif(not is_gfx1250(), reason="requires an AMD gfx1250 GPU")
requires_gluon = pytest.mark.skipif(
    not (is_gfx1250() and gluon_available()), reason="requires gfx1250 and a Triton with Gluon cdna5 support"
)
requires_flydsl = pytest.mark.skipif(
    not (is_gfx1250() and flydsl_available()), reason="requires gfx1250 and the FlyDSL backend"
)

_INT_OF_SIZE = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def as_bits(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret ``t`` as an integer tensor of the same element size."""
    return t.contiguous().view(_INT_OF_SIZE[t.element_size()])


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """True if shapes match and every element has the same bit pattern (NaN payloads and -0.0 included)."""
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(as_bits(a), as_bits(b)))


def bit_diff_count(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((as_bits(a) != as_bits(b)).sum().item())


def rel_l2(a: torch.Tensor, ref: torch.Tensor) -> float:
    a, ref = a.double(), ref.double()
    return ((a - ref).norm() / ref.norm().clamp_min(1e-30)).item()
