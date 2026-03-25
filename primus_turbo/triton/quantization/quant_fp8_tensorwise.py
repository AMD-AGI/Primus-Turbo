###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Triton FP8 tensorwise quantization kernels optimized for MI355X.

This implementation is kept aligned with the sweep-backed `fp8_quant_opt`
prototype: use the fused 2-kernel path for small/medium tensors, switch to the
3-kernel path when partial-buffer rereads become expensive, and apply
MI355X-specific `waves_per_eu` tuning in the profitable ranges.
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

FUSE_THRESHOLD = 70_000_000
TRITON_USE_MIN_NUMEL = 8_000_000
TRITON_USE_MAX_NUMEL = 600_000_000

@triton.jit
def _amax_partial_kernel(
    x_ptr,
    partial_ptr,
    n_elements,
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
    x_ptr,
    out_ptr,
    partial_ptr,
    scale_inv_ptr,
    n_elements,
    num_partials,
    FP8_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RED_BS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    global_max = tl.zeros([1], dtype=tl.float32)
    for start in range(0, num_partials, RED_BS):
        idx = start + tl.arange(0, RED_BS)
        mask = idx < num_partials
        partials = tl.load(partial_ptr + idx, mask=mask, other=0.0)
        block_max = tl.max(partials)
        global_max = tl.maximum(global_max, tl.full([1], block_max, tl.float32))
    amax = tl.reshape(global_max, [])
    amax = tl.maximum(amax, 1e-12)
    scale = FP8_MAX / amax
    if pid == 0:
        tl.store(scale_inv_ptr, 1.0 / scale)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x_q = tl.clamp(x * scale, min=-FP8_MAX, max=FP8_MAX)
    tl.store(out_ptr + offs, x_q.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _reduce_scale_kernel(
    partial_ptr,
    scale_ptr,
    scale_inv_ptr,
    num_partials,
    FP8_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    global_max = tl.zeros([1], dtype=tl.float32)
    for start in range(0, num_partials, BLOCK_SIZE):
        idx = start + offs
        mask = idx < num_partials
        partials = tl.load(partial_ptr + idx, mask=mask, other=0.0)
        block_max = tl.max(partials)
        global_max = tl.maximum(global_max, tl.full([1], block_max, tl.float32))
    amax = tl.reshape(global_max, [])
    amax = tl.maximum(amax, 1e-12)
    scale = FP8_MAX / amax
    tl.store(scale_ptr, scale)
    tl.store(scale_inv_ptr, 1.0 / scale)


@triton.jit
def _quant_scaled_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    n_elements,
    FP8_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    scale = tl.load(scale_ptr)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x_q = tl.clamp(x * scale, min=-FP8_MAX, max=FP8_MAX)
    tl.store(out_ptr + offs, x_q.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _quant_with_known_scale_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    n_elements,
    FP8_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    scale = tl.load(scale_ptr)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x_q = tl.clamp(x * scale, min=-FP8_MAX, max=FP8_MAX)
    tl.store(out_ptr + offs, x_q.to(out_ptr.dtype.element_ty), mask=mask)


class _BufferCache:
    __slots__ = ("partial_buf", "scale_buf", "max_tiles")

    def __init__(self):
        self.partial_buf = None
        self.scale_buf = None
        self.max_tiles = 0

    def get(self, device, num_tiles):
        if (
            self.partial_buf is None
            or self.max_tiles < num_tiles
            or self.partial_buf.device != device
        ):
            self.max_tiles = max(num_tiles, 1024)
            self.partial_buf = torch.empty(self.max_tiles, dtype=torch.float32, device=device)
            self.scale_buf = torch.empty(2, dtype=torch.float32, device=device)
        return self.partial_buf[:num_tiles], self.scale_buf


_cache = _BufferCache()


def _should_use_triton_fp8_tensorwise(x: torch.Tensor) -> bool:
    """Conservative MI355 gate from direct C++ vs Triton quant sweeps."""
    n = x.numel()
    return TRITON_USE_MIN_NUMEL <= n <= TRITON_USE_MAX_NUMEL


def _select_strategy(n: int):
    """Select path + launch config from MI355X sweep data."""
    if n <= 3_000_000:
        return "2k", 8192, 8, 1, 0
    if n <= 6_000_000:
        return "2k", 16384, 16, 1, 0
    if n <= 12_000_000:
        return "2k", 8192, 4, 1, 0
    if n <= FUSE_THRESHOLD:
        return "2k", 16384, 8, 1, 1
    if n <= 80_000_000:
        return "3k", 16384, 16, 2, 1
    if n <= 140_000_000:
        return "3k", 8192, 16, 1, 1
    if n <= 210_000_000:
        return "3k", 16384, 16, 2, 0
    if n < 350_000_000:
        return "3k", 16384, 16, 1, 1
    if n < 600_000_000:
        return "3k", 16384, 16, 2, 1
    return "3k", 8192, 8, 1, 1


def quantize_fp8_tensorwise(
    x: torch.Tensor,
    out_dtype: torch.dtype = torch.float8_e4m3fn,
    scale=None,
    out: torch.Tensor = None,
    buf_cache: _BufferCache = None,
):
    """Triton-based FP8 tensorwise quantization for contiguous 2D/3D tensors.

    The kernel only depends on the flattened element count, so grouped GEMM
    weights in ``(G, N, K)`` layout are handled by preserving ``orig_shape``
    across the flatten/restore path.
    """
    fp8_max = FP8_MAX_MAP[out_dtype]
    orig_shape = x.shape
    if not x.is_contiguous():
        x = x.contiguous()
    n = x.numel()
    x_flat = x.reshape(-1)
    out_flat = out.reshape(-1) if out is not None else torch.empty(n, dtype=out_dtype, device=x.device)
    path, bs, nw, ns, waves = _select_strategy(n)

    if scale is not None:
        num_tiles = (n + bs - 1) // bs
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(float(scale), dtype=torch.float32, device=x.device)
        scale = scale.to(device=x.device, dtype=torch.float32).reshape(())
        scale_view = scale.view(1)
        _quant_with_known_scale_kernel[(num_tiles,)](
            x_flat,
            out_flat,
            scale_view,
            n,
            FP8_MAX=fp8_max,
            BLOCK_SIZE=bs,
            num_warps=nw,
            num_stages=ns,
            waves_per_eu=waves,
        )
        return out_flat.reshape(orig_shape), scale.reciprocal()

    if buf_cache is None:
        buf_cache = _cache

    num_tiles = (n + bs - 1) // bs
    partials, scale_buf = buf_cache.get(x.device, num_tiles)
    scale_inv_out = torch.empty((), dtype=torch.float32, device=x.device)
    scale_inv_ptr = scale_inv_out.view(1)

    _amax_partial_kernel[(num_tiles,)](
        x_flat,
        partials,
        n,
        BLOCK_SIZE=bs,
        num_warps=nw,
        num_stages=ns,
        waves_per_eu=waves,
    )

    if path == "2k":
        red_bs = min(4096, triton.next_power_of_2(num_tiles))
        _reduce_and_quant_kernel[(num_tiles,)](
            x_flat,
            out_flat,
            partials,
            scale_inv_ptr,
            n,
            num_tiles,
            FP8_MAX=fp8_max,
            BLOCK_SIZE=bs,
            RED_BS=red_bs,
            num_warps=nw,
            num_stages=ns,
            waves_per_eu=waves,
        )
        return out_flat.reshape(orig_shape), scale_inv_out

    scale_ptr = scale_buf[0:1]
    red_bs = min(4096, triton.next_power_of_2(num_tiles))
    red_nw = min(16, max(1, red_bs // 64))

    _reduce_scale_kernel[(1,)](
        partials,
        scale_ptr,
        scale_inv_ptr,
        num_tiles,
        FP8_MAX=fp8_max,
        BLOCK_SIZE=red_bs,
        num_warps=red_nw,
        waves_per_eu=waves,
    )

    _quant_scaled_kernel[(num_tiles,)](
        x_flat,
        out_flat,
        scale_ptr,
        n,
        FP8_MAX=fp8_max,
        BLOCK_SIZE=bs,
        num_warps=nw,
        num_stages=ns,
        waves_per_eu=waves,
    )

    return out_flat.reshape(orig_shape), scale_inv_out
