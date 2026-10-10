###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch

_torch_custom_op_wrapper = torch.library.custom_op

from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendChoice,
    BackendEntry,
    BackendType,
    GlobalBackendManager,
    KernelBackend,
    PrecisionType,
    TuneCache,
)
from primus_turbo.triton.gemm.gemm_kernel import gemm_triton_kernel

_COMMON_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_HIPBLASLT_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


class GEMMHipBLASLtBackend(KernelBackend):
    @staticmethod
    def can_handle(
        a: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        inplace_add_to_out: bool = False,
        out: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> bool:
        supported = bias is None
        supported &= a.ndim == 2 and b.ndim == 2
        supported &= a.dtype in _HIPBLASLT_SUPPORTED_DTYPES and a.dtype == b.dtype

        d_dtype = out.dtype if out is not None else out_dtype
        supported &= d_dtype == a.dtype or d_dtype == torch.float32

        if inplace_add_to_out:
            supported &= out is not None
        if out is not None:
            supported &= out.is_contiguous()

        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        inplace_add_to_out: bool = False,
        out: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        beta = 1.0 if inplace_add_to_out else 0.0
        return torch.ops.primus_turbo_cpp_extension.hipblaslt_gemm(
            a, b, out_dtype, trans_a, trans_b, trans_c, beta, out
        )


class GEMMTritonBackend(KernelBackend):
    @staticmethod
    def can_handle(
        a: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> bool:
        supported = bias is None
        supported &= a.ndim == 2 and b.ndim == 2
        supported &= a.dtype in _COMMON_SUPPORTED_DTYPES and b.dtype in _COMMON_SUPPORTED_DTYPES
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        inplace_add_to_out: bool = False,
        out: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        beta = 1.0 if inplace_add_to_out else 0.0
        return gemm_triton_kernel(a, b, trans_a, trans_b, out_dtype, trans_c, beta=beta, out=out)


_AITER_GEMM_M = 32
_AITER_GEMM_PASSES = {0: "fwd", 1: "dgrad", 2: "wgrad"}  # aiter's adaln_gemm pass ids


class GEMMAiterBackend(KernelBackend):
    """bf16 GEMMs with a 32-row operand, on aiter's ``adaln_gemm`` kernels (built for the modulation linear of a DiT's
    adaptive LayerNorm at a 32-row micro-batch). With ``x[32, K]``, ``w[N, K]`` and ``g[32, N]`` it runs exactly

    * ``x @ w.T + bias`` (``bias[N]``, added in the epilogue: one rounding, bitwise ``torch.addmm``),
    * ``g @ w``,
    * ``g.T @ x``, returned or stored into ``out`` (``gemm_accum_impl(..., accumulate=False)``; the kernel
      overwrites, so it declines ``accumulate=True``),

    in any (trans_a, trans_b, trans_c) spelling, at the (N, K) aiter has a kernel for (``shapes()``). The kernels,
    their manifest and the workspace and counters of the second product live in aiter; this backend only calls it.

    ``can_handle`` is plain arithmetic on shapes, dtypes and a set, so callers may evaluate it inside compiled regions
    -- once ``shapes()`` has been read outside them (it imports aiter on first use, as every aiter-backed op here).
    The backend is strict (``BackendEntry.strict``): a call naming it as the default backend runs on it or raises.
    """

    _shapes: frozenset | None = None

    @classmethod
    def shapes(cls) -> frozenset:
        """The products aiter has kernels for, as {(pass, N, K)} with pass "fwd" (``x @ w.T + bias``), "dgrad"
        (``g @ w``) or "wgrad" (``g.T @ x``). Read once, for the current GPU's architecture; empty if the installed
        aiter has no ``adaln_gemm`` op or no kernels for this architecture."""
        if cls._shapes is None:
            try:
                from aiter.ops.adaln_gemm import _manifest

                # this GPU's kernels: on an architecture without them the manifest is empty and nothing is handled
                arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]
                manifest = _manifest(arch)
            except (ImportError, RuntimeError, AssertionError):
                manifest = {}
            cls._shapes = frozenset(
                (_AITER_GEMM_PASSES[p], n, k) for (p, n, k) in manifest if p in _AITER_GEMM_PASSES
            )
        return cls._shapes

    @staticmethod
    def _product(a, trans_a, b, trans_b, trans_c):
        """The stored result as ``op(a) @ op(b)`` without trans_c, ``(op(a) @ op(b)).T == op(b).T @ op(a).T``, and
        its (pass, N, K) -- None if it is none of the three products."""
        if trans_c:
            a, trans_a, b, trans_b = b, not trans_b, a, not trans_a
        m = _AITER_GEMM_M
        key = None
        if not trans_a and trans_b:  # x[32, K] @ w[N, K].T
            if a.shape[0] == m and a.shape[1] == b.shape[1]:
                key = ("fwd", b.shape[0], b.shape[1])
        elif not trans_a:  # g[32, N] @ w[N, K]
            if a.shape[0] == m and a.shape[1] == b.shape[0]:
                key = ("dgrad", b.shape[0], b.shape[1])
        elif not trans_b:  # g[32, N].T @ x[32, K]
            if a.shape[0] == m and b.shape[0] == m:
                key = ("wgrad", a.shape[1], b.shape[1])
        return a, b, key

    @classmethod
    def can_handle(
        cls,
        a: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        inplace_add_to_out: bool = False,
        out: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> bool:
        if inplace_add_to_out or a.ndim != 2 or b.ndim != 2 or not a.is_cuda:
            return False
        if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16 or out_dtype != torch.bfloat16:
            return False
        _, _, key = cls._product(a, trans_a, b, trans_b, trans_c)
        if key is None or key not in cls.shapes():
            return False
        pass_, n, k = key
        if (bias is not None) != (pass_ == "fwd"):
            return False
        if bias is not None and (bias.dtype != torch.bfloat16 or tuple(bias.shape) != (n,)):
            return False
        if out is not None:
            shape = {"fwd": (_AITER_GEMM_M, n), "dgrad": (_AITER_GEMM_M, k), "wgrad": (n, k)}[pass_]
            if out.dtype != torch.bfloat16 or tuple(out.shape) != shape or not out.is_contiguous():
                return False
        return True

    @classmethod
    def execute(
        cls,
        a: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        inplace_add_to_out: bool = False,
        out: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        from aiter.ops.adaln_gemm import adaln_dgrad, adaln_fwd, adaln_wgrad

        a, b, (pass_, n, k) = cls._product(a, trans_a, b, trans_b, trans_c)
        a, b = a.contiguous(), b.contiguous()
        if pass_ == "fwd":
            out = a.new_empty(_AITER_GEMM_M, n) if out is None else out
            adaln_fwd(a, b, bias.contiguous(), out)
        elif pass_ == "dgrad":
            out = a.new_empty(_AITER_GEMM_M, k) if out is None else out
            adaln_dgrad(a, b, out)
        else:
            out = a.new_empty(n, k) if out is None else out
            adaln_wgrad(a, b, out)
        return out


_GEMM_BACKENDS = {
    BackendType.HIPBLASLT: BackendEntry(GEMMHipBLASLtBackend),
    BackendType.TRITON: BackendEntry(GEMMTritonBackend),
    # Only where a call names it: it has kernels for a handful of 32-row shapes, and auto-tuning must not move other
    # GEMMs that happen to match one of them onto it.
    BackendType.AITER: BackendEntry(GEMMAiterBackend, autotune=False, strict=True),
}


class GEMMKernelDispatcher(AutoKernelDispatcher):
    _backends = _GEMM_BACKENDS
    _cache = TuneCache(1024)

    @classmethod
    def make_key(cls, a, b, trans_a, trans_b, out_dtype, trans_c, **kwargs):
        M = a.shape[1] if trans_a else a.shape[0]
        Ka = a.shape[0] if trans_a else a.shape[1]
        N = b.shape[0] if trans_b else b.shape[1]
        # The epilogue is part of the key: not every backend has a bias, an accumulating or a store-into-out one, and
        # backends differ in the bias and output dtypes and the output layouts they take.
        bias, out = kwargs.get("bias"), kwargs.get("out")
        epilogue = (
            None if bias is None else bias.dtype,
            bool(kwargs.get("inplace_add_to_out", False)),
            None if out is None else (out.dtype, out.is_contiguous()),
        )
        return (M, N, Ka, a.dtype, b.dtype, out_dtype, trans_a, trans_b, trans_c, epilogue)


@_torch_custom_op_wrapper("primus_turbo::gemm_impl", mutates_args=(), device_types="cuda")
def gemm_impl(
    a: torch.Tensor,
    trans_a: bool,
    b: torch.Tensor,
    trans_b: bool,
    out_dtype: torch.dtype,
    trans_c: bool,
    default_backend: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """BF16/FP16/FP32 GEMM ``op(A) @ op(B)`` (transposed if ``trans_c``), plus ``bias`` ([N], added in the epilogue
    before the one rounding to ``out_dtype``) if given. Only backends with a bias epilogue accept a bias."""
    default_backend_choice = BackendChoice(backend=BackendType(default_backend))
    user_backend_choice = GlobalBackendManager.get_gemm_backend(PrecisionType.BF16_FP16_FP32)

    kwargs = dict(
        a=a,
        trans_a=trans_a,
        b=b,
        trans_b=trans_b,
        out_dtype=out_dtype,
        trans_c=trans_c,
    )
    if bias is not None:
        kwargs["bias"] = bias

    return GEMMKernelDispatcher.dispatch(default_backend_choice, user_backend_choice, **kwargs)


@gemm_impl.register_fake
def gemm_impl_meta(
    a: torch.Tensor,
    trans_a: bool,
    b: torch.Tensor,
    trans_b: bool,
    out_dtype: torch.dtype,
    trans_c: bool,
    default_backend: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2, (
        f"Expected both a and b to be 2D tensors, but got a.ndim={a.ndim}, b.ndim={b.ndim}"
    )
    M = a.shape[1] if trans_a else a.shape[0]
    N = b.shape[0] if trans_b else b.shape[1]
    if trans_c:
        M, N = N, M
    return torch.empty(M, N, dtype=out_dtype, device=a.device)


@_torch_custom_op_wrapper("primus_turbo::gemm_accum_impl", mutates_args={"out"}, device_types="cuda")
def gemm_accum_impl(
    a: torch.Tensor,
    trans_a: bool,
    b: torch.Tensor,
    trans_b: bool,
    out_dtype: torch.dtype,
    trans_c: bool,
    out: torch.Tensor,
    default_backend: int,
    accumulate: bool = True,
) -> None:
    """BF16/FP16 GEMM that accumulates into ``out`` instead of returning.

    Computes ``out += op(A) @ op(B)``, folding the accumulation into the GEMM
    epilogue (beta=1). With ``accumulate=False`` it stores ``out = op(A) @ op(B)``
    instead (beta=0; e.g. a weight gradient written straight into a buffer whose
    previous contents are not part of the result).
    """
    default_backend_choice = BackendChoice(backend=BackendType(default_backend))
    user_backend_choice = GlobalBackendManager.get_gemm_backend(PrecisionType.BF16_FP16_FP32)

    kwargs = dict(
        a=a,
        trans_a=trans_a,
        b=b,
        trans_b=trans_b,
        out_dtype=out_dtype,
        trans_c=trans_c,
        inplace_add_to_out=accumulate,
        out=out,
    )

    # The tuner benchmarks a backend by launching it repeatedly, so letting it tune on
    # the caller's buffer would accumulate the wgrad once per warmup and timing
    # iteration. Prime the cache on a scratch buffer first: the tune key ignores `out`,
    # so the dispatch below hits that cache and runs exactly once on the real buffer.
    # Zeroed, not empty -- beta=1 reads the buffer back and NaNs would skew the timings.
    if GlobalBackendManager.auto_tune_enabled() and not GEMMKernelDispatcher._is_graph_capturing():
        GEMMKernelDispatcher.tune(**{**kwargs, "out": torch.zeros_like(out)})

    GEMMKernelDispatcher.dispatch(default_backend_choice, user_backend_choice, **kwargs)


@gemm_accum_impl.register_fake
def gemm_accum_impl_meta(
    a: torch.Tensor,
    trans_a: bool,
    b: torch.Tensor,
    trans_b: bool,
    out_dtype: torch.dtype,
    trans_c: bool,
    out: torch.Tensor,
    default_backend: int,
    accumulate: bool = True,
) -> None:
    assert a.ndim == 2 and b.ndim == 2, (
        f"Expected both a and b to be 2D tensors, but got a.ndim={a.ndim}, b.ndim={b.ndim}"
    )
    M = a.shape[1] if trans_a else a.shape[0]
    N = b.shape[0] if trans_b else b.shape[1]
    if trans_c:
        M, N = N, M
    assert tuple(out.shape) == (M, N), f"out shape {tuple(out.shape)} must equal {(M, N)}"
    return None
