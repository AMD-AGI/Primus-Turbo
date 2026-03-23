###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Triton FP8 tensorwise quantization kernels — optimized for MI355X.

Current policy:
  - Main path: amax partials -> reduce scale -> quant with scalar scale.
    This avoids the legacy fused reduce-and-quant kernel repeatedly re-reading
    the full partial buffer from every CTA.
  - Legacy 2-kernel fused path is kept for future retuning experiments, but is
    disabled by default until both n-threshold and tile-threshold are swept
    again on MI355X.

Configs are data-driven from per-shape sweeps on MI355X.
"""

import torch
import triton
import triton.language as tl

FP8_MAX_MAP = {
    torch.float8_e4m3fn: 448.0,
    torch.float8_e4m3fnuz: 240.0,
    torch.float8_e5m2: 57344.0,
    torch.float8_e5m2fnuz: 57344.0,
}

FUSE_THRESHOLD = 75_500_000
FUSE_NUM_TILES_THRESHOLD = 0


@triton.jit
def _amax_partial_kernel(
    x_ptr, partial_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tile_max = tl.max(tl.abs(x))
    tl.store(partial_ptr + pid, tile_max)


@triton.jit
def _reduce_and_quant_kernel(
    x_ptr, out_ptr, partial_ptr, scale_inv_ptr,
    n_elements, num_partials,
    FP8_MAX: tl.constexpr, BLOCK_SIZE: tl.constexpr, RED_BS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    global_max = tl.zeros([1], dtype=tl.float32)
    for start in range(0, num_partials, RED_BS):
        idx = start + tl.arange(0, RED_BS)
        mask = idx < num_partials
        p = tl.load(partial_ptr + idx, mask=mask, other=0.0)
        block_max = tl.max(p)
        global_max = tl.maximum(global_max, tl.full([1], block_max, tl.float32))
    amax = tl.reshape(global_max, [])
    amax = tl.maximum(amax, 1e-12)
    scale = FP8_MAX / amax
    if pid == 0:
        tl.store(scale_inv_ptr, 1.0 / scale)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x_q = x * scale
    x_q = tl.clamp(x_q, min=-FP8_MAX, max=FP8_MAX)
    tl.store(out_ptr + offs, x_q.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _reduce_scale_kernel(
    partial_ptr, scale_ptr, scale_inv_ptr, num_partials,
    FP8_MAX: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    global_max = tl.zeros([1], dtype=tl.float32)
    for start in range(0, num_partials, BLOCK_SIZE):
        idx = start + offs
        mask = idx < num_partials
        p = tl.load(partial_ptr + idx, mask=mask, other=0.0)
        block_max = tl.max(p)
        global_max = tl.maximum(global_max, tl.full([1], block_max, tl.float32))
    amax = tl.reshape(global_max, [])
    amax = tl.maximum(amax, 1e-12)
    scale = FP8_MAX / amax
    tl.store(scale_ptr, scale)
    tl.store(scale_inv_ptr, 1.0 / scale)


@triton.jit
def _quant_scaled_kernel(
    x_ptr, out_ptr, scale_ptr, n_elements,
    FP8_MAX: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    scale = tl.load(scale_ptr)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x_q = x * scale
    x_q = tl.clamp(x_q, min=-FP8_MAX, max=FP8_MAX)
    tl.store(out_ptr + offs, x_q.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _quant_with_known_scale_kernel(
    x_ptr, out_ptr, amax_ptr, n_elements,
    FP8_MAX: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    amax = tl.load(amax_ptr)
    amax = tl.maximum(amax, 1e-12)
    scale = FP8_MAX / amax
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x_q = x * scale
    x_q = tl.clamp(x_q, min=-FP8_MAX, max=FP8_MAX)
    tl.store(out_ptr + offs, x_q.to(out_ptr.dtype.element_ty), mask=mask)


class _BufferCache:
    __slots__ = ('partial_buf', 'scale_buf', 'max_tiles')

    def __init__(self):
        self.partial_buf = None
        self.scale_buf = None
        self.max_tiles = 0

    def get(self, device, num_tiles):
        if self.partial_buf is None or self.max_tiles < num_tiles or \
                self.partial_buf.device != device:
            self.max_tiles = max(num_tiles, 1024)
            self.partial_buf = torch.empty(
                self.max_tiles, dtype=torch.float32, device=device)
            self.scale_buf = torch.empty(
                2, dtype=torch.float32, device=device)
        return self.partial_buf[:num_tiles], self.scale_buf


_cache = _BufferCache()


def _select_config(n: int):
    """Data-driven config from MI355X sweeps."""
    if n <= 40_000_000:
        return 8192, 4, 1
    elif n <= FUSE_THRESHOLD:
        return 16384, 8, 1
    elif n < 200_000_000:
        return 16384, 16, 1
    elif n < 900_000_000:
        return 16384, 16, 2
    else:
        return 8192, 8, 1


def _use_reduce_and_quant_fuse(n: int, num_tiles: int) -> bool:
    """Return whether the legacy fused reduce-and-quant path should be used."""
    return n <= FUSE_THRESHOLD and num_tiles <= FUSE_NUM_TILES_THRESHOLD


def quantize_fp8_tensorwise(
    x: torch.Tensor,
    out_dtype: torch.dtype = torch.float8_e4m3fn,
    scale=None,
    out: torch.Tensor = None,
    buf_cache: _BufferCache = None,
):
    """Triton-based FP8 tensorwise quantization. Handles any-dim tensors (2D, 3D, etc.).

    Args:
        out: Optional pre-allocated FP8 output tensor (same shape as x).
        buf_cache: Optional separate BufferCache instance for stream-safe concurrent use.
    """
    fp8_max = FP8_MAX_MAP[out_dtype]
    orig_shape = x.shape
    if not x.is_contiguous():
        x = x.contiguous()
    n = x.numel()
    x_flat = x.reshape(-1)
    if out is not None:
        out_flat = out.reshape(-1)
    else:
        out_flat = torch.empty(n, dtype=out_dtype, device=x.device)

    if scale is not None:
        bs, nw, ns = _select_config(n)
        num_tiles = (n + bs - 1) // bs
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor([float(scale)], dtype=torch.float32, device=x.device)
        scale = scale.to(device=x.device, dtype=torch.float32).reshape(1)
        _quant_with_known_scale_kernel[(num_tiles,)](
            x_flat, out_flat, scale, n,
            FP8_MAX=fp8_max, BLOCK_SIZE=bs,
            num_warps=nw, num_stages=ns,
        )
        scale_inv = (scale / fp8_max).clone()
        return out_flat.reshape(orig_shape), scale_inv

    if buf_cache is None:
        buf_cache = _cache

    bs, nw, ns = _select_config(n)
    num_tiles = (n + bs - 1) // bs
    partials, scale_buf = buf_cache.get(x.device, num_tiles)

    _amax_partial_kernel[(num_tiles,)](
        x_flat, partials, n,
        BLOCK_SIZE=bs, num_warps=nw, num_stages=ns,
    )

    if _use_reduce_and_quant_fuse(n, num_tiles):
        si_ptr = scale_buf[1:2]
        rbs = min(4096, triton.next_power_of_2(num_tiles))
        _reduce_and_quant_kernel[(num_tiles,)](
            x_flat, out_flat, partials, si_ptr,
            n, num_tiles,
            FP8_MAX=fp8_max, BLOCK_SIZE=bs, RED_BS=rbs,
            num_warps=nw, num_stages=ns,
        )
        return out_flat.reshape(orig_shape), si_ptr.clone()

    scale_ptr = scale_buf[0:1]
    si_ptr = scale_buf[1:2]
    rbs = min(4096, triton.next_power_of_2(num_tiles))
    rnw = min(16, max(1, rbs // 64))

    _reduce_scale_kernel[(1,)](
        partials, scale_ptr, si_ptr, num_tiles,
        FP8_MAX=fp8_max, BLOCK_SIZE=rbs, num_warps=rnw,
    )

    _quant_scaled_kernel[(num_tiles,)](
        x_flat, out_flat, scale_ptr, n,
        FP8_MAX=fp8_max, BLOCK_SIZE=bs,
        num_warps=nw, num_stages=ns,
    )

    return out_flat.reshape(orig_shape), si_ptr.clone()
