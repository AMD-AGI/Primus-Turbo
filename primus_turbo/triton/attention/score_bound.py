###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""Key-side input to the online-softmax reference bound."""

import torch
import triton
import triton.language as tl


@triton.jit
def _k_sq_norm_max_kernel(
    K,
    OUT,
    n_rows,
    chunk_rows,
    n_groups,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    N_CHUNKS: tl.constexpr,
):
    g = tl.program_id(0)
    c = tl.program_id(1)
    r_end = tl.minimum((c + 1) * chunk_rows, n_rows)
    acc = 0.0
    for r0 in range(c * chunk_rows, r_end, BLOCK_R):
        rows = r0 + tl.arange(0, BLOCK_R)
        # Masked rows contribute a zero norm, which never displaces a real maximum.
        k = tl.load(
            K + (rows[:, None] * n_groups + g) * D + tl.arange(0, D)[None, :],
            mask=rows[:, None] < r_end,
            other=0.0,
        )
        k = k.to(tl.float32)
        acc = tl.maximum(acc, tl.max(tl.sum(k * k, axis=1)))
    tl.store(OUT + g * N_CHUNKS + c, acc)


_WORKSPACE = {}
_LAUNCHERS = {}
_UNBOUND = object()


def _bind_launch(compiled, grid, tail, k, out):
    """Bind the compiled pass to its grid and constants so a launch only moves the pointers."""
    try:
        runner = compiled[grid]
        runner(k, out, *tail)
        return lambda K, OUT: runner(K, OUT, *tail)
    except Exception:
        return None


def _workspace(device, n_out):
    """Output scratch, reused across calls."""
    key = (device, n_out)
    ws = _WORKSPACE.get(key)
    if ws is None:
        ws = torch.empty(n_out, device=device, dtype=torch.float32)
        _WORKSPACE[key] = ws
    return ws


def k_sq_norm_max(k: torch.Tensor, n_chunks: int) -> torch.Tensor:
    """``max_j ||k_j||^2`` per (row group, kv chunk), as an fp32 [G * n_chunks] tensor."""
    d = k.shape[-1]
    k = k if k.is_contiguous() else k.contiguous()
    n_rows = k.shape[0]
    n_groups = k.numel() // (d * n_rows)
    out = _workspace(k.device, n_groups * n_chunks)
    chunk_rows = triton.cdiv(n_rows, n_chunks)
    block_r = max(1, 16384 // d)
    # The bound launcher holds a device-specific kernel handle, so it is cached per device.
    key = (k.device, d, n_rows, n_groups, n_chunks)
    launch = _LAUNCHERS.get(key, _UNBOUND)
    if launch is not None and launch is not _UNBOUND:
        launch(k, out)
        return out
    compiled = _k_sq_norm_max_kernel[(n_groups, n_chunks)](
        k,
        out,
        n_rows,
        chunk_rows,
        n_groups,
        D=d,
        BLOCK_R=block_r,
        N_CHUNKS=n_chunks,
        num_warps=8,
    )
    if launch is _UNBOUND:
        _LAUNCHERS[key] = _bind_launch(
            compiled,
            (n_groups, n_chunks, 1),
            (n_rows, chunk_rows, n_groups, d, block_r, n_chunks),
            k,
            out,
        )
    return out
