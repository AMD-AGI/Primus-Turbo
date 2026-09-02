###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MXFP6 (E2M3) GEMM: ``out[M, N] = A[M, K] @ B[N, K].T``.

The signature deliberately diverges from ``gemm_fp4_impl``. FP4 passes strided operands
plus separate scales and derives M/N/K from the shapes, with ``trans_a`` / ``trans_b``
selecting a layout. Here both operands are opaque 1-D blobs in AITER's
``mxfp6_c0c1_256_padk2`` layout, so:

- M, N and K must be passed explicitly -- they are not recoverable from the blobs, and
  guessing them wrong is not caught by shape checks. (It *is* caught by the kernel's
  size check, but only after our fix made that check exact rather than ``>=``.)
- There is no transpose option. The packed layout already fixes the contraction axis:
  the direction is chosen when packing, not when multiplying. So the caller picks
  ``quantize_mxfp6_row`` or ``quantize_mxfp6_col`` per operand and this op always
  computes ``A @ B.T``.

AITER is the only backend. HipBLASLt has no MXFP6 entry point, and FlyDSL cannot express
an FP6 B operand at all (its ``b_dtype`` branches only fp8-or-fp4, so a "fp6" B silently
takes the fp4 path) -- it is A6W4, never A6W6.
"""

import torch

from primus_turbo.common.aiter_utils import get_aiter
from primus_turbo.pytorch.core.backend import (
    BackendEntry,
    BackendType,
    GlobalBackendManager,
    KernelBackend,
    PrecisionType,
)
from primus_turbo.pytorch.core.low_precision import (
    MXFP6_K_TILE_SIZE,
    MXFP6_TILE_SIZE,
    ScalingGranularity,
)
from primus_turbo.pytorch.core.utils import is_gfx950_device
from primus_turbo.pytorch.kernels.quantization.mxfp6_pack import mxfp6_pack_sizes

_torch_custom_op_wrapper = torch.library.custom_op

__all__ = ["GEMMFP6AITERBackend", "gemm_fp6_impl", "gemm_fp6_out_impl"]


class GEMMFP6AITERBackend(KernelBackend):
    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.MX_BLOCKWISE,
    }

    SUPPORTED_OUT_DTYPES = {torch.bfloat16}

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        a_scale: torch.Tensor,
        b: torch.Tensor,
        b_scale: torch.Tensor,
        m: int,
        n: int,
        k: int,
        out_dtype: torch.dtype,
        granularity: ScalingGranularity,
        **kwargs,
    ) -> bool:
        supported = is_gfx950_device(a.device)
        supported &= granularity in GEMMFP6AITERBackend.SUPPORTED_GRANULARITIES
        # The A6W6 asm only writes bf16.
        supported &= out_dtype in GEMMFP6AITERBackend.SUPPORTED_OUT_DTYPES
        # Blobs are uint8 byte streams, not typed operands.
        supported &= a.dtype == torch.uint8 and b.dtype == torch.uint8
        supported &= a_scale.dtype == torch.uint8 and b_scale.dtype == torch.uint8

        # Alignment. gemm_a6w6 does pad internally and is correct on unaligned shapes,
        # so this is a padding-waste guard rather than a correctness one -- an unaligned
        # M/N/K would silently do work on padding. Every Flux 12B GEMM satisfies it.
        supported &= m % MXFP6_TILE_SIZE == 0 and n % MXFP6_TILE_SIZE == 0 and k % MXFP6_K_TILE_SIZE == 0
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        a_scale: torch.Tensor,
        b: torch.Tensor,
        b_scale: torch.Tensor,
        m: int,
        n: int,
        k: int,
        out_dtype: torch.dtype,
        granularity: ScalingGranularity,
    ) -> torch.Tensor:
        del out_dtype, granularity  # already gated by can_handle
        return get_aiter().gemm_a6w6(a, b, a_scale, b_scale, m, n, k)


_GEMM_FP6_BACKENDS = {
    BackendType.AITER: BackendEntry(GEMMFP6AITERBackend, autotune=False),
}


def _resolve_backend() -> BackendType:
    """MXFP6 has exactly one backend, so honour an explicit request and otherwise
    use AITER. This does not go through AutoKernelDispatcher: with one backend there
    is nothing to tune between, and the dispatcher's key derives M/N/K from operand
    shapes, which opaque blobs do not carry."""
    choice = GlobalBackendManager.get_gemm_backend(PrecisionType.FP6)
    if choice is not None and choice.backend is not None:
        if choice.backend not in _GEMM_FP6_BACKENDS:
            raise ValueError(
                f"{choice.backend} has no MXFP6 GEMM. Supported: {sorted(b.name for b in _GEMM_FP6_BACKENDS)}"
            )
        return choice.backend
    return BackendType.AITER


def _validate_blobs(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    m: int,
    n: int,
    k: int,
) -> None:
    """Reject malformed packed operands before they reach AITER.

    The blobs are opaque byte streams carrying neither shape nor dtype, so a wrong
    M/N/K is invisible to every check downstream of here: the kernel derives its strides
    from the dimensions it was handed and reads whatever lies past the end of a blob
    that is too short. Sizing them with the same helper the packer used is the only
    point at which that is catchable from Python.
    """
    for name, blob in (("a", a), ("a_scale", a_scale), ("b", b), ("b_scale", b_scale)):
        if blob.dtype != torch.uint8:
            raise TypeError(f"MXFP6 GEMM operand {name} must be a uint8 blob, got {blob.dtype}.")
        if blob.ndim != 1:
            raise ValueError(f"MXFP6 GEMM operand {name} must be a 1-D blob, got {blob.ndim}-D.")
        if not blob.is_contiguous():
            raise ValueError(f"MXFP6 GEMM operand {name} must be contiguous.")

    devices = {blob.device for blob in (a, a_scale, b, b_scale)}
    if len(devices) != 1:
        raise ValueError(f"MXFP6 GEMM operands must share one device, got {sorted(str(d) for d in devices)}.")

    for name, dim in (("M", m), ("N", n), ("K", k)):
        if dim <= 0:
            raise ValueError(f"MXFP6 GEMM needs positive dimensions, got {name}={dim}.")

    # a is the [M, K] operand and b the [N, K] one, in every direction: the backward
    # GEMMs permute which logical tensor plays which role but not this relationship.
    for name, (operand, scale), (want_operand, want_scale) in (
        ("a", (a, a_scale), mxfp6_pack_sizes(m, k)),
        ("b", (b, b_scale), mxfp6_pack_sizes(n, k)),
    ):
        if operand.numel() != want_operand or scale.numel() != want_scale:
            raise ValueError(
                f"MXFP6 GEMM operand {name} does not match M={m} N={n} K={k}: expected "
                f"{want_operand} operand and {want_scale} scale bytes, got "
                f"{operand.numel()} and {scale.numel()}."
            )


@_torch_custom_op_wrapper("primus_turbo::gemm_fp6_impl", mutates_args=(), device_types="cuda")
def gemm_fp6_impl(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    m: int,
    n: int,
    k: int,
    out_dtype: torch.dtype,
    granularity: int,
) -> torch.Tensor:
    granularity_enum = ScalingGranularity(granularity)
    _validate_blobs(a, a_scale, b, b_scale, m, n, k)
    backend = _resolve_backend()
    impl = _GEMM_FP6_BACKENDS[backend].impl

    kwargs = dict(
        a=a,
        a_scale=a_scale,
        b=b,
        b_scale=b_scale,
        m=m,
        n=n,
        k=k,
        out_dtype=out_dtype,
        granularity=granularity_enum,
    )
    if not impl.can_handle(**kwargs):
        raise ValueError(
            f"{backend.name} cannot handle this MXFP6 GEMM: M={m} N={n} K={k} "
            f"out_dtype={out_dtype} granularity={granularity_enum}. MXFP6 needs gfx950, "
            f"a bf16 output, MX_BLOCKWISE scaling, and M/N a multiple of "
            f"{MXFP6_TILE_SIZE} with K a multiple of {MXFP6_K_TILE_SIZE}."
        )
    return impl.execute(**kwargs)


@gemm_fp6_impl.register_fake
def gemm_fp6_impl_meta(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    m: int,
    n: int,
    k: int,
    out_dtype: torch.dtype,
    granularity: int,
) -> torch.Tensor:
    # Pure arithmetic on purpose: this must not reach into AITER, whose kernel
    # selection does lru_cached pandas lookups that SymInts would break.
    return torch.empty(m, n, dtype=out_dtype, device=a.device)


def _pad_k(k: int) -> int:
    """K rounded up to AITER's A6W6 K tile, which is what the asm expects."""
    return (k + MXFP6_K_TILE_SIZE - 1) // MXFP6_K_TILE_SIZE * MXFP6_K_TILE_SIZE


@_torch_custom_op_wrapper(
    "primus_turbo::gemm_fp6_out_impl", mutates_args=("out",), device_types="cuda"
)
def gemm_fp6_out_impl(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    m: int,
    n: int,
    k: int,
    granularity: int,
) -> None:
    """``out[M, N] = A[M, K] @ B[N, K].T``, writing into a caller-owned buffer.

    This exists so a weight gradient can land straight in ``param.main_grad`` instead of
    being allocated and then added in by Megatron's DDP hook. Two things make that safe
    to do here but not in general:

    - The A6W6 asm stores with beta=0, so this **overwrites** ``out`` rather than
      accumulating into it. It is therefore only correct for the last (or only)
      microbatch of a step. The caller owns that decision.
    - ``gemm_a6w6`` normally allocates a tile-padded output and slices the result, so a
      caller-provided buffer is only writable when the launch needs no padding. Hence the
      exact-alignment requirement below, which is stricter than ``can_handle``'s
      padding-waste guard.
    """
    granularity_enum = ScalingGranularity(granularity)
    _validate_blobs(a, a_scale, b, b_scale, m, n, k)

    if granularity_enum not in GEMMFP6AITERBackend.SUPPORTED_GRANULARITIES:
        raise ValueError(f"MXFP6 out-GEMM needs MX_BLOCKWISE scaling, got {granularity_enum}.")
    if out.dtype not in GEMMFP6AITERBackend.SUPPORTED_OUT_DTYPES:
        raise TypeError(f"MXFP6 out-GEMM writes bf16 only, got out dtype {out.dtype}.")
    if out.ndim != 2 or tuple(out.shape) != (m, n):
        raise ValueError(f"MXFP6 out-GEMM expects a 2-D [{m}, {n}] out, got {tuple(out.shape)}.")
    if not out.is_contiguous():
        raise ValueError("MXFP6 out-GEMM needs a contiguous out; the asm writes it directly.")
    if out.device != a.device:
        raise ValueError(f"MXFP6 out-GEMM out is on {out.device}, operands on {a.device}.")
    if m % MXFP6_TILE_SIZE != 0 or n % MXFP6_TILE_SIZE != 0:
        raise ValueError(
            f"MXFP6 out-GEMM needs M and N exact multiples of {MXFP6_TILE_SIZE} so the launch "
            f"needs no padding and can write out in place, got M={m} N={n}."
        )
    if not is_gfx950_device(a.device):
        raise RuntimeError("MXFP6 out-GEMM requires gfx950.")

    aiter = get_aiter()
    config = aiter.get_GEMM_A6W6_config(m, n, k)
    kernel_name = str(config["kernelName"]) if config is not None else None
    aiter.gemm_a6w6_asm(a, b, a_scale, b_scale, out, _pad_k(k), kernel_name)


@gemm_fp6_out_impl.register_fake
def gemm_fp6_out_impl_meta(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    m: int,
    n: int,
    k: int,
    granularity: int,
) -> None:
    return None
