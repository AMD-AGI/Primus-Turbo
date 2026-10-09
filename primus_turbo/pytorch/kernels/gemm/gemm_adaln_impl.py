###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The bf16 GEMMs of a DiT's AdaLN modulation linear at a 32-row micro-batch, on aiter's ``adaln_gemm`` kernels.

For ``y = x @ w.T + b`` with ``x[32, K]``, ``w[N, K]``:

* ``adaln_gemm_fwd(x, w, b) -> y``        ``y[32, N] = x @ w.T + b``
* ``adaln_gemm_dgrad(go, w) -> dx``       ``dx[32, K] = go @ w``
* ``adaln_gemm_wgrad_out(go, x, out)``    ``out[N, K] = go.T @ x``, stored into ``out`` (e.g. a parameter's
  ``main_grad``), which the op mutates.

The kernels, their manifest and the dgrad workspace live in aiter; these ops only launch them. Each is an opaque
custom op with a fake, so it is safe inside ``torch.compile(fullgraph=True)``. Kernels exist for exact (pass, N, K)
only: ``adaln_gemm_table`` lists them as plain data, to be read once outside compiled regions. Calling an op at a shape
not in the table is an error.
"""

import torch

_torch_custom_op_wrapper = torch.library.custom_op

# aiter's pass ids, by name
_PASS_NAMES = {0: "fwd", 1: "dgrad", 2: "wgrad"}

ADALN_GEMM_M = 32
"""The micro-batch rows (M) every ``adaln_gemm`` kernel is built for."""


def adaln_gemm_table() -> frozenset:
    """The shapes the AdaLN GEMM ops have kernels for, as plain data for callers that decide inside compiled regions
    (read it once, outside them): {(pass, N, K)} with pass "fwd" / "dgrad" / "wgrad", each at M = ``ADALN_GEMM_M``.
    Empty if the installed aiter has no ``adaln_gemm`` op or no kernels for this GPU."""
    try:
        from aiter.ops.adaln_gemm import _manifest
    except ImportError:
        return frozenset()
    return frozenset((_PASS_NAMES[p], n, k) for (p, n, k) in _manifest().keys() if p in _PASS_NAMES)


@_torch_custom_op_wrapper("primus_turbo::adaln_gemm_fwd", mutates_args=(), device_types="cuda")
def adaln_gemm_fwd(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``x[32, K] @ w[N, K].T + b[N]`` in bf16. See ``adaln_gemm_table`` for the (N, K) it has kernels for."""
    from aiter.ops.adaln_gemm import adaln_fwd

    out = torch.empty(x.shape[0], w.shape[0], dtype=x.dtype, device=x.device)
    adaln_fwd(x.contiguous(), w.contiguous(), b.contiguous(), out)
    return out


@adaln_gemm_fwd.register_fake
def _adaln_gemm_fwd_meta(x, w, b):
    return x.new_empty(x.shape[0], w.shape[0])


@_torch_custom_op_wrapper("primus_turbo::adaln_gemm_dgrad", mutates_args=(), device_types="cuda")
def adaln_gemm_dgrad(go: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``go[32, N] @ w[N, K]`` in bf16 (deterministic). See ``adaln_gemm_table`` for the (N, K) it has kernels for."""
    from aiter.ops.adaln_gemm import adaln_dgrad

    out = torch.empty(go.shape[0], w.shape[1], dtype=go.dtype, device=go.device)
    adaln_dgrad(go.contiguous(), w.contiguous(), out)
    return out


@adaln_gemm_dgrad.register_fake
def _adaln_gemm_dgrad_meta(go, w):
    return go.new_empty(go.shape[0], w.shape[1])


@_torch_custom_op_wrapper("primus_turbo::adaln_gemm_wgrad_out", mutates_args=("out",), device_types="cuda")
def adaln_gemm_wgrad_out(go: torch.Tensor, x: torch.Tensor, out: torch.Tensor) -> None:
    """``out[N, K] = go[32, N].T @ x[32, K]`` in bf16, stored into ``out`` (contiguous; overwritten, not accumulated).
    See ``adaln_gemm_table`` for the (N, K) it has kernels for."""
    from aiter.ops.adaln_gemm import adaln_wgrad

    adaln_wgrad(go.contiguous(), x.contiguous(), out)


@adaln_gemm_wgrad_out.register_fake
def _adaln_gemm_wgrad_out_meta(go, x, out) -> None:
    return None
