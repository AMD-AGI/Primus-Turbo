###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Standalone gfx950 FlyDSL GEMMs for BF16 operands.

The tile implementation is shared with the grouped BF16 kernels.  This thin
launcher supplies the dense grid and supports the NN, NT, and TN layouts used
by the language-model head in forward and backward.  Accumulation is always
FP32 in the MFMA pipeline; ``out_dtype=torch.float32`` preserves those values
in the output instead of narrowing them to BF16.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from primus_turbo.flydsl.gemm.gemm_bf16_kernel import (
    _make_shared_storage,
    gemm_bf16_nn_tile,
    gemm_bf16_nt_tile,
    gemm_bf16_tn_tile,
)
from primus_turbo.flydsl.utils.gemm_helper import make_value_attrs
from primus_turbo.flydsl.utils.prims import ceildiv

_LAYOUT_TO_TILE = {
    "nn": gemm_bf16_nn_tile,
    "nt": gemm_bf16_nt_tile,
    "tn": gemm_bf16_tn_tile,
}
_SUPPORTED_OUTPUT_DTYPES = (torch.bfloat16, torch.float32)
_COMPILED_DENSE_CACHE: dict = {}


@functools.lru_cache(maxsize=256)
def _compile_dense_bf16(
    layout: str,
    k: int,
    n_tail: int,
    out_fp32: bool,
    block_m: int = 256,
    block_n: int = 256,
    group_m: int = 1,
    num_xcd: int = 8,
    waves_per_eu: int = 2,
    agpr_alloc: int = 0,
    nt_vmcnt: int = 3,
):
    assert layout in _LAYOUT_TO_TILE
    assert block_m >= 128 and block_m % 128 == 0
    assert block_n >= 256 and block_n % 256 == 0
    tile_fn = _LAYOUT_TO_TILE[layout]
    shared_storage = _make_shared_storage(block_m, block_n)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_dense_bf16(
        a: fx.Tensor,
        b: fx.Tensor,
        c: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        # Materializing thread_idx here keeps the kernel launch geometry visible
        # to FlyDSL before the shared tile helper is expanded.
        _ = str(fx.thread_idx.x)
        n_blocks = ceildiv(c_n, block_n)
        lds = fx.SharedAllocator().allocate(shared_storage).peek()
        tile_fn(
            a,
            b,
            c,
            c_m,
            c_n,
            lds,
            K=k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            n_blocks=n_blocks,
            GROUP_M=group_m,
            num_xcd=num_xcd,
            out_fp32=out_fp32,
            nt_vmcnt=nt_vmcnt,
            n_tail=n_tail,
        )

    @flyc.jit
    def launch_dense_bf16(
        a: fx.Tensor,
        b: fx.Tensor,
        c: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, block_m) * ceildiv(c_n, block_n)
        kernel_dense_bf16(
            a,
            b,
            c,
            c_m,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_dense_bf16


def _static_layout(args):
    return tuple(flyc.from_torch_tensor(arg) if isinstance(arg, torch.Tensor) else arg for arg in args)


def _get_compiled_dense(launch, args):
    # The queue handle selects the launch stream and is deliberately excluded.
    key = (id(launch),) + tuple(
        (tuple(arg.shape), arg.stride(), arg.dtype) if isinstance(arg, torch.Tensor) else arg
        for arg in args[:-1]
    )
    compiled = _COMPILED_DENSE_CACHE.get(key)
    if compiled is None:
        compiled = flyc.compile(launch, *_static_layout(args))
        _COMPILED_DENSE_CACHE[key] = compiled
    return compiled


def gemm_bf16_flydsl_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    layout: str,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Compute a contiguous dense BF16 GEMM in ``layout`` and return MxN.

    Layout names describe the physical operands: NN is ``[M,K] @ [K,N]``,
    NT is ``[M,K] @ [N,K].T``, and TN is ``[K,M].T @ [K,N]``.
    """
    assert layout in _LAYOUT_TO_TILE, f"unsupported BF16 FlyDSL layout: {layout}"
    assert a.ndim == 2 and b.ndim == 2
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    assert out_dtype in _SUPPORTED_OUTPUT_DTYPES
    assert a.is_cuda and b.is_cuda and a.device == b.device

    if layout == "nn":
        m, k = a.shape
        kb, n = b.shape
    elif layout == "nt":
        m, k = a.shape
        n, kb = b.shape
    else:
        k, m = a.shape
        kb, n = b.shape
    assert k == kb, f"{layout.upper()} K mismatch: a {a.shape}, b {b.shape}"

    a = a.contiguous()
    b = b.contiguous()
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    launch = _compile_dense_bf16(layout, k, n % 256, out_dtype == torch.float32)
    args = (a, b, out, m, n, torch.cuda.current_stream())
    _get_compiled_dense(launch, args)(*args)
    return out
