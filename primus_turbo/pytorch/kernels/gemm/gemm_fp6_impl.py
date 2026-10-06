###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MXFP6 (E2M3) GEMM: ``out[M, N] = A[M, K] @ B[N, K].T (+ bias[N])``.

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

The optional ``bias`` is folded into the A6W6 store epilogue, where it is free: that asm
is store-bound rather than VALU-bound, so it saves the whole of the separate elementwise
pass a caller would otherwise do. Whether the installed aiter can do this is a capability
we probe, not a switch we expose -- an older aiter gets the separate pass added here, so
callers pass a bias unconditionally and never branch on aiter's version themselves.
"""

import functools
import os
from typing import Optional

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
from primus_turbo.pytorch.kernels.quantization.mxfp4_pack import (
    aiter_has_a6w4_bias_epilogue,
    mxfp4_gemm_pack_sizes,
)
from primus_turbo.pytorch.kernels.quantization.mxfp6_pack import (
    aiter_has_bias_epilogue,
    mxfp6_pack_sizes,
)

_torch_custom_op_wrapper = torch.library.custom_op

__all__ = [
    "GEMMFP6AITERBackend",
    "a4w4_ts_shapes",
    "a6w6_ts_available",
    "a6w6_ts_shapes",
    "a6w4_ts_available",
    "tilescale_table",
    "tilescale_table_has",
    "set_a6w6_backend",
    "gemm_fp6_impl",
    "gemm_fp6_out_impl",
]


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
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del out_dtype, granularity  # already gated by can_handle
        aiter = get_aiter()
        if bias is None:
            # Kept as a call with no bias argument at all, rather than bias=None, so that
            # an aiter predating the epilogue takes exactly the path it always did.
            return aiter.gemm_a6w6(a, b, a_scale, b_scale, m, n, k)
        if not aiter_has_bias_epilogue():
            # Same result, one extra pass over the output: this is what MXFP6 did before
            # the epilogue existed. Adding in bf16 after the GEMM rounds twice where the
            # epilogue rounds once, so the two differ by up to a last-bit step -- the
            # epilogue being the more accurate. That is a property of which aiter is
            # installed, not of anything the caller chose, so it is not switchable here.
            return aiter.gemm_a6w6(a, b, a_scale, b_scale, m, n, k) + bias
        return aiter.gemm_a6w6(a, b, a_scale, b_scale, m, n, k, bias=bias)


class GEMMA6W4AITERBackend(GEMMFP6AITERBackend):
    """A6W4: the same GEMM with the B operand in MXFP4 instead of MXFP6.

    Everything `can_handle` tests is identical -- gfx950, bf16 out, MX_BLOCKWISE, uint8
    blobs, M/N on 256 and K on 128 -- because A6W4 shares A6W6's tile geometry exactly.
    What differs is which blob `b` is (sized by `mxfp4_gemm_pack_sizes`, checked in
    `_validate_blobs`) and that there is no bias epilogue to fold into.

    Only the forward and dgrad GEMMs can reach this. wgrad contracts M, so neither of its
    operands is the weight and there is no MXFP4 blob to hand it.
    """

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
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del out_dtype, granularity  # already gated by can_handle
        aiter = get_aiter()
        if bias is None:
            return aiter.gemm_a6w4(a, b, a_scale, b_scale, m, n, k)
        if aiter_has_a6w4_bias_epilogue():
            return aiter.gemm_a6w4(a, b, a_scale, b_scale, m, n, k, bias=bias)
        out = aiter.gemm_a6w4(a, b, a_scale, b_scale, m, n, k)
        # Unlike A6W6, the upstream A6W4 asm has no bias epilogue, so the add is a
        # separate pass here. That is a real cost on the forward -- the epilogue is worth
        # a measurable cost on A6W6 -- and the reason porting BIAS into the A6W4
        # generator is tracked as the next step rather than skipped. Correctness is
        # unaffected: adding in bf16 after the GEMM rounds twice where the epilogue rounds
        # once, so the two differ by at most a last-bit step.
        #
        # Deliberately NOT `out.add_(bias)`, which looks strictly cheaper -- one allocation
        # and one write of the whole [M, N] output saved -- and measures slower end to
        # end. The in-place op is a fusion barrier: the functional add gets folded
        # into the Inductor region that consumes this output, and the in-place one cannot
        # be. The allocation is cheaper than the fusion it costs.
        return out + bias


_GEMM_FP6_BACKENDS = {
    BackendType.AITER: BackendEntry(GEMMFP6AITERBackend, autotune=False),
}

class GEMMA4W6AITERBackend(GEMMFP6AITERBackend):
    """A4W6: the A operand in MXFP4, the B operand in MXFP6 -- the mirror of A6W4.

    This exists for wgrad. `grad_w = g_col @ a_col` contracts the token dimension, so
    neither operand is the weight and A6W4 cannot reach it; narrowing A instead puts the
    fp4 on the *gradient* and leaves the activation at fp6, where A6W4 on the same GEMM
    would put it on the activation. Measured on the Flux wgrad shapes, A6W4 is the faster
    of the two (1.0982x against 1.0767x) and A4W6 the more accurate (weight-gradient
    cosine 0.99326 against 0.99279), so which to run is a numerics call and both are wired.

    No bias: wgrad produces a weight gradient and has none.
    """

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
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del out_dtype, granularity
        if bias is not None:
            raise ValueError("A4W6 is the wgrad path and takes no bias")
        return get_aiter().gemm_a4w6(a, b, a_scale, b_scale, m, n, k)


_GEMM_A6W4_BACKENDS = {
    BackendType.AITER: BackendEntry(GEMMA6W4AITERBackend, autotune=False),
}

_GEMM_A4W6_BACKENDS = {
    BackendType.AITER: BackendEntry(GEMMA4W6AITERBackend, autotune=False),
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
    weight_is_fp4: bool = False,
    a_is_fp4: bool = False,
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
    #
    # Under A6W4 only b changes format, so only b's expected size does. Getting this
    # wrong is exactly the failure the whole function exists to catch -- an MXFP6-sized
    # blob handed to gemm_a6w4 is 1.5x too long and would be read at the wrong stride
    # with no error anywhere -- so the two formats are sized by their own helpers.
    a_pack_sizes = mxfp4_gemm_pack_sizes if a_is_fp4 else mxfp6_pack_sizes
    b_pack_sizes = mxfp4_gemm_pack_sizes if weight_is_fp4 else mxfp6_pack_sizes
    fmt = "A6W4" if weight_is_fp4 else "A4W6" if a_is_fp4 else "MXFP6"
    for name, (operand, scale), (want_operand, want_scale) in (
        ("a", (a, a_scale), a_pack_sizes(m, k)),
        ("b", (b, b_scale), b_pack_sizes(n, k)),
    ):
        if operand.numel() != want_operand or scale.numel() != want_scale:
            raise ValueError(
                f"{fmt} GEMM operand {name} does not match M={m} N={n} K={k}: expected "
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
    bias: torch.Tensor | None = None,
    weight_is_fp4: bool = False,
    a_is_fp4: bool = False,
    a4w4: int = 0,
    a6w6_ts: bool = False,
    a6w4_ts: bool = False,
) -> torch.Tensor:
    granularity_enum = ScalingGranularity(granularity)
    if a6w4_ts:
        # A6W4 on the tilescale layout: A as ts6_fmt packed it, B as with_ts4_row packed it (K128-blocked MXFP4
        # codes); aiter gemm_a6w4_tilescale, bias in its store epilogue. weight_is_fp4 (the A6W4 tile blob) is
        # not consulted: a6w4_ts names the weight format itself.
        if a_is_fp4 or a4w4 or a6w6_ts:
            raise ValueError("a6w4_ts excludes a_is_fp4 / a4w4 / a6w6_ts")
        return _a6w4_ts(a, a_scale, b, b_scale, m, n, k, out_dtype, bias)
    if a6w6_ts:
        # MXFP6 both operands in the FlyDSL A6W6 GEMM's layout (ts6_fmt packs) on the assembly ports of that
        # kernel in aiter; bias in its store epilogue.
        if weight_is_fp4 or a_is_fp4 or a4w4:
            raise ValueError("a6w6_ts excludes weight_is_fp4 / a_is_fp4 / a4w4: both operands are MXFP6")
        return _a6w6_ts(a, a_scale, b, b_scale, m, n, k, out_dtype, bias)
    if a4w4 in (2, 3):
        # Both operands MXFP4 on FlyDSL's GEMM: plain scales (2, fmt 8 / 9 / 12) or scales
        # already in its packed per-tile layout (3, ts_fmt).
        return _a4w4_flydsl(a, a_scale, b, b_scale, m, n, k, out_dtype, bias, packed=a4w4 == 3)
    if a4w4 == 4:
        # Same operands as 3 (ts_fmt packs, scales in the packed per-tile layout), on the assembly ports
        # of FlyDSL's 256-wide kernel in aiter: AOT, no FlyDSL at runtime.
        if weight_is_fp4 or a_is_fp4:
            raise ValueError("a4w4 excludes weight_is_fp4 / a_is_fp4: both operands are MXFP4")
        return _a4w4_ts(a, a_scale, b, b_scale, m, n, k, out_dtype, bias)
    if a4w4 == 5:
        # Both operands in the A4W4 tile blob (MX_FMT_BLOB_* rows) on aiter's tile-blob kernel: any shape, the same
        # exact fp32 sum as a4w4=4 -- the fallback where no fly code object exists for (M, N, K).
        if weight_is_fp4 or a_is_fp4:
            raise ValueError("a4w4 excludes weight_is_fp4 / a_is_fp4: both operands are MXFP4")
        return _a4w4_aiter_blob(a, a_scale, b, b_scale, m, n, k, out_dtype, bias)
    if a4w4:
        # Both operands MXFP4 in AITER's f4gemm layout (quantize_mx_* fmt 1-4): A plain, B
        # (16, 16)-shuffled, scales shuffle_scale()'d. gemm_a4w4 picks the tuned kernel.
        if weight_is_fp4 or a_is_fp4:
            raise ValueError("a4w4 excludes weight_is_fp4 / a_is_fp4: both operands are MXFP4")
        from primus_turbo.pytorch.kernels.quantization.mx_a4w4_pack import a4w4_operand

        A, As = a4w4_operand(a, a_scale, m, k)
        B, Bs = a4w4_operand(b, b_scale, n, k)
        out = get_aiter().gemm_a4w4(A, B, As, Bs, bpreshuffle=True)
        out = out[:m, :n]
        if out_dtype != out.dtype:
            out = out.to(out_dtype)
        return out if bias is None else out + bias
    if weight_is_fp4 and a_is_fp4:
        raise ValueError(
            "weight_is_fp4 selects A6W4 and a_is_fp4 selects A4W6; they are mutually "
            "exclusive and there is no all-fp4 entry point here."
        )
    _validate_blobs(a, a_scale, b, b_scale, m, n, k, weight_is_fp4, a_is_fp4)
    if bias is not None and (bias.dim() != 1 or bias.numel() != n):
        raise ValueError(f"MXFP6 GEMM bias must be a 1D tensor of length N={n}, got {tuple(bias.shape)}.")
    if not (weight_is_fp4 or a_is_fp4) and _a6w6_flydsl_ok(m, n, k, out_dtype, granularity_enum):
        return _a6w6_flydsl(a, a_scale, b, b_scale, m, n, k, bias)
    backend = _resolve_backend()
    # One flag rather than a second op: the two differ only in which aiter entry point
    # runs and how b is sized, and sharing the op keeps every caller's autograd, fake
    # kernel and Dynamo behaviour identical between the formats.
    impl = (
        _GEMM_A6W4_BACKENDS
        if weight_is_fp4
        else _GEMM_A4W6_BACKENDS if a_is_fp4 else _GEMM_FP6_BACKENDS
    )[backend].impl

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
    # bias is not part of the can_handle key: it is added in the store epilogue and does not
    # change which shapes a backend supports.
    return impl.execute(**kwargs, bias=bias)


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
    bias: torch.Tensor | None = None,
    weight_is_fp4: bool = False,
    a_is_fp4: bool = False,
    a4w4: int = 0,
    a6w6_ts: bool = False,
    a6w4_ts: bool = False,
) -> torch.Tensor:
    # Pure arithmetic on purpose: this must not reach into AITER, whose kernel
    # selection does lru_cached pandas lookups that SymInts would break. The output
    # geometry does not depend on the weight format, so weight_is_fp4 is unused here.
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
    weight_is_fp4: bool = False,
    bias: Optional[torch.Tensor] = None,
    a4w4: int = 0,
    a6w6_ts: bool = False,
    a6w4_ts: bool = False,
) -> None:
    """``out[M, N] = A[M, K] @ B[N, K].T (+ bias)``, writing into a caller-owned buffer.

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

    ``bias`` (bf16 ``[N]``) is added in the store epilogue, exactly as ``gemm_fp6_impl``
    adds it, so a projection written into a shared buffer stays bit-identical to the
    allocating call. A6W6 only.
    """
    granularity_enum = ScalingGranularity(granularity)
    if a6w4_ts:
        if a4w4 or a6w6_ts:
            raise ValueError("a6w4_ts out-GEMM excludes a4w4 / a6w6_ts")
        _a6w4_ts(a, a_scale, b, b_scale, m, n, k, out.dtype, bias, out=out)
        return
    if a6w6_ts:
        if weight_is_fp4 or a4w4:
            raise ValueError("a6w6_ts out-GEMM excludes weight_is_fp4 / a4w4: both operands are MXFP6")
        _a6w6_ts(a, a_scale, b, b_scale, m, n, k, out.dtype, bias, out=out)
        return
    if a4w4 in (2, 3):
        if weight_is_fp4 or bias is not None:
            raise ValueError("FlyDSL a4w4 out-GEMM takes no bias and excludes weight_is_fp4")
        _a4w4_flydsl(a, a_scale, b, b_scale, m, n, k, out.dtype, None, out=out, packed=a4w4 == 3)
        return
    if a4w4 == 4:
        if weight_is_fp4 or bias is not None:
            raise ValueError("aiter fly a4w4 out-GEMM takes no bias and excludes weight_is_fp4")
        _a4w4_ts(a, a_scale, b, b_scale, m, n, k, out.dtype, None, out=out)
        return
    if a4w4:
        # A4W4 (f4gemm layouts). Like A6W6, the asm stores with beta=0 (overwrites), so the
        # same one-microbatch-per-step contract applies; the tuned row picks the kernel.
        if weight_is_fp4 or bias is not None:
            raise ValueError("a4w4 out-GEMM takes no bias and excludes weight_is_fp4")
        if out.dtype != torch.bfloat16 or tuple(out.shape) != (m, n) or not out.is_contiguous():
            raise ValueError(
                f"a4w4 out-GEMM expects a contiguous bf16 [{m}, {n}] out, got {tuple(out.shape)} {out.dtype}"
            )
        if m % 32 != 0:
            raise ValueError(f"a4w4 out-GEMM needs M a multiple of 32, got {m}")
        from aiter.ops.gemm_op_a4w4 import get_GEMM_config

        from primus_turbo.pytorch.kernels.quantization.mx_a4w4_pack import a4w4_operand

        A, As = a4w4_operand(a, a_scale, m, k)
        B, Bs = a4w4_operand(b, b_scale, n, k)
        cfg = get_GEMM_config(m, n, k)
        kernel = "" if cfg is None else str(cfg["kernelName"])
        split_k = 0 if cfg is None or cfg.get("splitK") is None else int(cfg["splitK"])
        if kernel and kernel.find("_ZN") == -1:
            raise RuntimeError(f"a4w4 out-GEMM: tuned row for {(m, n, k)} is a CK kernel; asm only")
        get_aiter().gemm_a4w4_asm(A[:m], B, As, Bs, out, kernel, None, bpreshuffle=True, log2_k_split=split_k)
        return
    _validate_blobs(a, a_scale, b, b_scale, m, n, k, weight_is_fp4)
    if bias is not None:
        if weight_is_fp4:
            raise ValueError("MXFP6 out-GEMM bias is supported for A6W6 only, not A6W4.")
        if bias.dim() != 1 or bias.numel() != n:
            raise ValueError(f"MXFP6 out-GEMM bias must be a 1D tensor of length N={n}, got {tuple(bias.shape)}.")

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

    if not weight_is_fp4 and _a6w6_flydsl_ok(m, n, k, out.dtype, granularity_enum):
        _a6w6_flydsl(a, a_scale, b, b_scale, m, n, k, bias, out=out)
        return
    aiter = get_aiter()
    if weight_is_fp4:
        # wgrad with the B operand narrowed. gemm_a6w4_asm takes the same physical buffers
        # and picks its own kernel, and like A6W6 it stores with beta=0 -- so the same
        # one-microbatch-per-step contract applies unchanged.
        aiter.gemm_a6w4_asm(a, b, a_scale, b_scale, out, _pad_k(k))
        return
    config = aiter.get_GEMM_A6W6_config(m, n, k)
    kernel_name = str(config["kernelName"]) if config is not None else None
    if bias is None:
        aiter.gemm_a6w6_asm(a, b, a_scale, b_scale, out, _pad_k(k), kernel_name)
    else:
        # aiter refuses a kernel with no bias epilogue rather than silently dropping it.
        aiter.gemm_a6w6_asm(a, b, a_scale, b_scale, out, _pad_k(k), kernel_name, 1.0, bias)


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
    weight_is_fp4: bool = False,
    bias: Optional[torch.Tensor] = None,
    a4w4: int = 0,
    a6w6_ts: bool = False,
    a6w4_ts: bool = False,
) -> None:
    return None


def _a4w4_aiter_blob(a, a_scale, b, b_scale, m, n, k, out_dtype, bias):
    """A4W4 on aiter's tile-blob kernel (``a4w4=5``, ``gemm_a4w4_blob_asm``, its default ``stnt_allk``): both operands
    as the packers write MX_FMT_BLOB_* rows (C0 tile blob, +2 guard K tiles). M / N are padded to the 256 tile."""
    from aiter.ops.gemm_op_a4w4_blob import gemm_a4w4_blob_asm

    if out_dtype != torch.bfloat16:
        raise ValueError(f"aiter blob a4w4 writes bf16, got {out_dtype}")
    mp, np_, kp = -(-m // 256) * 256, -(-n // 256) * 256, -(-k // 128) * 128
    out = torch.empty(mp, np_, dtype=torch.bfloat16, device=a.device)
    flat = lambda t: t.view(torch.uint8).reshape(-1)  # noqa: E731
    gemm_a4w4_blob_asm(flat(a), flat(b), flat(a_scale), flat(b_scale), out, kp)
    out = out[:m, :n] if (mp, np_) != (m, n) else out
    return out if bias is None else out + bias


def tilescale_table(a_fmt: int, b_fmt: int, b_codes: int = 0, b_ilv: int = 0):
    """aiter's tilescale kernels for one format pair as plain data, for callers that decide per GEMM inside compiled
    regions (read it once, outside them): (exact {(M, N, K, bias)}, generic {(bias, K-loop class): kmin}). A GEMM has a
    kernel if its (M, N, K, bias) is exact, or M, N are multiples of 256, K a multiple of 512, and the generic entry of
    (bias, K/128 mod 12) exists with K >= kmin -- see ``tilescale_table_has``."""
    try:
        from aiter.ops.gemm_op_tilescale import _generic, _rows
    except ImportError:
        return frozenset(), {}
    key = (a_fmt, b_fmt, b_codes, b_ilv)
    exact = frozenset((r[5], r[6], r[7], bool(r[4])) for r in _rows() if r[:4] == key and r[5])
    generic = {(bool(k[4]), k[5]): v for k, v in _generic().items() if k[:4] == key}
    return exact, generic


def tilescale_table_has(table, m: int, n: int, k: int, has_bias: bool) -> bool:
    """Whether ``table`` (``tilescale_table``) has a kernel for the GEMM; plain arithmetic, safe to trace."""
    exact, generic = table
    if (m, n, k, has_bias) in exact:
        return True
    if m % 256 or n % 256 or k % 512:
        return False
    for kcls in ((k // 128) % 12, 12):  # 12: a generic row serving every K-loop class
        kmin = generic.get((has_bias, kcls))
        if kmin is not None and k >= kmin:
            return True
    return False


def _tilescale_rows() -> frozenset:
    """aiter's tilescale GEMM manifest as (a_fmt, b_fmt, b_codes, b_ilv, bias, M, N, K) tuples (empty without it)."""
    try:
        from aiter.ops.gemm_op_tilescale import _rows
    except ImportError:
        return frozenset()
    return _rows()


# A6W6 backend: "aiter" (the tuned asm table) or "flydsl" (Turbo's FlyDSL MXFP6 GEMM compiled at runtime,
# reading the same mxfp6_c0c1_256_padk2 blobs, one tile per WG; bit-identical to AITER's A6W6, bias included). Shapes the
# FlyDSL kernel does not take (K not a multiple of 512, M / N off the 256 tile) stay on AITER, on the same operands.
_A6W6_BACKEND = os.environ.get("PRIMUS_TURBO_A6W6_BACKEND", "aiter")


def set_a6w6_backend(name: str) -> None:
    """Select the A6W6 (MXFP6 x MXFP6) GEMM backend: "aiter" or "flydsl"."""
    global _A6W6_BACKEND
    if name not in ("aiter", "flydsl"):
        raise ValueError(f"A6W6 backend must be 'aiter' or 'flydsl', got {name!r}")
    _A6W6_BACKEND = name


def _a6w6_flydsl_ok(m, n, k, out_dtype, granularity_enum) -> bool:
    return (
        _A6W6_BACKEND == "flydsl"
        and out_dtype == torch.bfloat16
        and granularity_enum == ScalingGranularity.MX_BLOCKWISE
        and m % 256 == 0
        and n % 256 == 0
        and k % 512 == 0
    )


def _a6w6_flydsl(a, a_scale, b, b_scale, m, n, k, bias, out=None):
    """A6W6 on Turbo's FlyDSL MXFP6 GEMM (``layout="aiter"``, one tile per WG): the standard MXFP6 tile blobs, beta 0
    into ``out`` if given, bias as fp32(acc) + fp32(bias) with one rounding (A6W6's epilogue)."""
    from primus_turbo.flydsl.gemm.gemm_mxfp6_kernel import gemm_mxfp6_persistent

    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=a.device)
    flat = lambda t: t.view(torch.uint8).reshape(-1)  # noqa: E731
    gemm_mxfp6_persistent(
        flat(a),
        None,
        flat(b),
        None,
        flat(a_scale),
        flat(b_scale),
        out=out,
        tpw=1,
        bias=bias,
        layout="aiter",
        m=m,
        n=n,
        k=k,
    )
    return out


@functools.lru_cache(maxsize=None)
def a4w4_ts_shapes() -> frozenset:
    """{(M, N, K)} with a code object in aiter's f4flygemm family (``a4w4=4``); empty for an aiter without it. Reads
    files: call it outside compiled regions."""
    return frozenset((r[5], r[6], r[7]) for r in _tilescale_rows() if r[:3] == (4, 4, 0) and not r[4] and r[5])


@functools.lru_cache(maxsize=None)
def a6w6_ts_shapes() -> frozenset:
    """{(M, N, K, bias)} with a code object in aiter's f6flygemm family (empty for an aiter without it). Reads files:
    call it outside compiled regions (a caller deciding per GEMM inside torch.compile should hold the set)."""
    return frozenset((r[5], r[6], r[7], bool(r[4])) for r in _tilescale_rows() if r[:4] == (6, 6, 0, 0))


def a6w6_ts_available(m: int, n: int, k: int, has_bias: bool) -> bool:
    """Whether aiter has the A6W6 fly kernel (``gemm_a6w6_fly_asm``) for exactly this GEMM: the caller packs the
    operands in ``ts6_fmt`` only then, and in the A6W6 tile blob otherwise."""
    return (int(m), int(n), int(k), bool(has_bias)) in a6w6_ts_shapes()


def _a6w6_ts(a, a_scale, b, b_scale, m, n, k, out_dtype, bias, out=None):
    """A6W6 on the assembly ports of the FlyDSL MXFP6 GEMM (``a6w6_ts``, aiter ``gemm_a6w6_fly_asm``).

    Operands as ``ts6_fmt`` packed them: one buffer per operand with the K128-blocked C0 then C1 planes, scales in
    FlyDSL's packed slab. Bit-identical to the A6W6 tile-blob kernels on the same values (bias: fp32(acc) + fp32(bias),
    one rounding). One code object per (M, N, K, bias); see ``a6w6_ts_available``."""
    from aiter.ops.gemm_op_tilescale import gemm_a6w6_tilescale

    if out_dtype != torch.bfloat16:
        raise ValueError(f"a6w6_ts writes bf16, got {out_dtype}")
    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=a.device)
    flat = lambda t: t.view(torch.uint8).reshape(-1)  # noqa: E731
    gemm_a6w6_tilescale(flat(a), flat(b), flat(a_scale), flat(b_scale), out, k, bias)
    return out


@functools.lru_cache(maxsize=None)
def _a6w4_ts_rows() -> frozenset:
    try:
        from aiter.ops.gemm_op_tilescale import _rows
    except ImportError:
        return frozenset()
    return frozenset((r[5], r[6], r[7], bool(r[4])) for r in _rows() if r[:4] == (6, 4, 1, 0))


def a6w4_ts_available(m: int, n: int, k: int, has_bias: bool) -> bool:
    """Whether aiter has the A6W4 tilescale kernel for exactly this GEMM (K128-blocked FP4 weight, role-B scales
    without interleave). Reads the manifest once; call it outside compiled regions or hold the answer."""
    return (int(m), int(n), int(k), bool(has_bias)) in _a6w4_ts_rows()


def _a6w4_ts(a, a_scale, b, b_scale, m, n, k, out_dtype, bias, out=None):
    """A6W4 on the tilescale layout (``a6w4_ts``, aiter ``gemm_a6w4_tilescale``): A = MXFP6 K128-blocked C0 / C1
    planes (``ts6_fmt``), B = MXFP4 K128-blocked codes (``with_ts4_row``), scales FlyDSL's packed slab (B without
    interleave). Bit-identical to A6W6 on the FP6 re-encoding of B; bias: fp32(acc) + fp32(bias), one rounding."""
    from aiter.ops.gemm_op_tilescale import gemm_a6w4_tilescale

    if out_dtype != torch.bfloat16:
        raise ValueError(f"a6w4_ts writes bf16, got {out_dtype}")
    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=a.device)
    flat = lambda t: t.view(torch.uint8).reshape(-1)  # noqa: E731
    gemm_a6w4_tilescale(flat(a), flat(b), flat(a_scale), flat(b_scale), out, k, bias)
    return out


@functools.lru_cache(maxsize=None)
def _a4w4_table(ilv):
    return tilescale_table(4, 4, 0, ilv)


def _a4w4_ts(a, a_scale, b, b_scale, m, n, k, out_dtype, bias, out=None):
    """A4W4 on the assembly ports of FlyDSL's 256-wide MXFP4 GEMM (``a4w4=4``, aiter ``gemm_a4w4_fly_asm``).

    Operands exactly as ``a4w4=3``: the packers wrote plain MXFP4 rows and the scales in the packed per-tile layout
    for the 256-wide N tile (``ts_fmt``, ``ts_b_params``), on aiter's A4W4 tilescale kernels: an exact-shape one
    or the K-generic one (K a multiple of 512, at least its minimum)."""
    from aiter.ops.gemm_op_tilescale import a4w4_b_ilv, gemm_a4w4_tilescale

    from primus_turbo.pytorch.kernels.quantization.mx_a4w4_pack import ts_operand

    if out_dtype != torch.bfloat16:
        raise ValueError(f"aiter fly a4w4 writes bf16, got {out_dtype}")
    ilv = a4w4_b_ilv(m, n, k)
    if not tilescale_table_has(_a4w4_table(ilv), int(m), int(n), int(k), False):
        raise ValueError(f"no aiter A4W4 tilescale kernel for {m}x{n}x{k} (K must be a multiple of 512, >= 1024)")
    A, As = ts_operand(a, a_scale, m, k)
    B, Bs = ts_operand(b, b_scale, n, k)
    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
    gemm_a4w4_tilescale(A, B, As, Bs, out, k, ilv)
    return out if bias is None else out + bias


def _a4w4_flydsl(a, a_scale, b, b_scale, m, n, k, out_dtype, bias, out=None, packed=False):
    """A4W4 on FlyDSL's MXFP4 GEMM (``a4w4=2``): plain-layout operands, beta 0 into ``out`` if given.

    FlyDSL repacks the plain E8M0 scales into its per-tile layout itself (its own workspace,
    sized per tile). Deliberately not the prepacked entry: Turbo's FlyDSL backend wrapper sizes a
    caller's packed slab per 256 rows, 1.33x short for block_n = 192."""
    from primus_turbo.flydsl.gemm.gemm_mxfp4_kernel import gemm_mxfp4_flydsl_kernel
    from primus_turbo.pytorch.kernels.quantization.mx_a4w4_pack import ts_operand, plain_operand

    if packed:
        # The packers stored the scales in the layout this GEMM reads (sized per tile, as
        # FlyDSL's own workspace is), so no repack kernel runs.
        A, As = ts_operand(a, a_scale, m, k)
        B, Bs = ts_operand(b, b_scale, n, k)
        res = gemm_mxfp4_flydsl_kernel(A, As, B, Bs, out_dtype=out_dtype, out=out, scales_prepacked=True, k=k)
        return res if bias is None else res + bias
    A, As = plain_operand(a, a_scale, m, k)
    B, Bs = plain_operand(b, b_scale, n, k)
    res = gemm_mxfp4_flydsl_kernel(A, As.contiguous(), B, Bs.contiguous(), out_dtype=out_dtype, out=out)
    return res if bias is None else res + bias


# Pre-tilescale names.
a6w6_fly_shapes, a6w6_fly_available, a4w4_fly_shapes = a6w6_ts_shapes, a6w6_ts_available, a4w4_ts_shapes
