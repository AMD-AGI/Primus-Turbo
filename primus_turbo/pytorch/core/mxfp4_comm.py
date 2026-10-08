###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Experimental byte transport for unshuffled, deterministic MXFP4 weights.

The wire payload contains BOTH independently quantized weight orientations and
their E8M0 scales. Packing is a bit-preserving copy, not another quantization.
The two orientations cost 17/16 bytes per logical weight, versus 2 for BF16.
Only equal, 32-row-aligned shards of [experts, rows, columns] are supported.
Flat optimizer shards must first be assembled into complete scaling tiles.
"""

from dataclasses import dataclass
from math import prod

import torch


def _aligned_bf16_input(weight):
    """Make the int32 view used by FlyDSL legal without copying aligned shards."""
    weight = weight.contiguous()
    # contiguous() returns the original tensor when its strides already match,
    # even if a flat-buffer slice starts at an odd BF16 storage offset.
    return weight.clone() if weight.storage_offset() % 2 else weight


@dataclass(frozen=True)
class MXFP4WireLayout:
    """Out-of-band metadata; every collective participant must use the same layout."""

    shape: tuple[int, int, int]
    scale_rounding_mode: int = 0

    def __post_init__(self):
        if len(self.shape) != 3 or any(d <= 0 for d in self.shape):
            raise ValueError("expected positive [experts, rows, columns] dimensions")
        if any(d % 32 for d in self.shape[-2:]):
            raise ValueError("weight rows and columns must be multiples of 32")
        if self.scale_rounding_mode not in (0, 1, 2):
            raise ValueError("unsupported MXFP4 scale rounding mode")

    @property
    def component_shapes(self):
        g, n, k = self.shape
        return ((g, n, k // 2), (g, n, k // 32), (g, k, n // 2), (g, k, n // 32))

    @property
    def nbytes(self):
        return sum(prod(shape) for shape in self.component_shapes)

    def views(self, payload):
        """Return zero-copy uint8 component views of one contiguous wire payload."""
        if payload.dtype != torch.uint8 or payload.ndim != 1 or not payload.is_contiguous():
            raise ValueError("payload must be a contiguous 1D uint8 tensor")
        if payload.numel() != self.nbytes:
            raise ValueError("payload length does not match the wire layout")
        return tuple(
            part.view(shape)
            for part, shape in zip(
                payload.split([prod(s) for s in self.component_shapes]), self.component_shapes
            )
        )

    def pack(self, components):
        """Strip kernel padding and concatenate raw bytes (never cast numerically)."""
        if len(components) != 4:
            raise ValueError("expected row data, row scales, column data, column scales")
        parts = []
        for tensor, shape in zip(components, self.component_shapes):
            if tensor.element_size() != 1 or tensor.ndim != 3:
                raise ValueError("each component must be a 3D, one-byte tensor")
            if tuple(tensor.shape[:-1]) != shape[:-1] or tensor.shape[-1] < shape[-1]:
                raise ValueError("component shape does not match the wire layout")
            parts.append(tensor.view(torch.uint8)[..., : shape[-1]].contiguous().view(-1))
        return torch.cat(parts)

    def quantize(self, weight):
        """Quantize a BF16 shard using the existing dual-direction weight recipe."""
        if tuple(weight.shape) != self.shape or weight.dtype != torch.bfloat16:
            raise ValueError("expected a BF16 weight matching the layout; cast master shards first")
        from primus_turbo.pytorch.core.low_precision import (
            ScalingGranularity,
            ScalingRecipe,
            float4_e2m1fn_x2,
        )
        from primus_turbo.pytorch.ops.quantization import quantize_fp4_with_trans

        recipe = ScalingRecipe(use_2d_block=True)
        with torch.no_grad():
            components = quantize_fp4_with_trans(
                _aligned_bf16_input(weight),
                float4_e2m1fn_x2,
                ScalingGranularity.MX_BLOCKWISE,
                block_size=32,
                scaling_recipe=recipe,
                scaling_recipe_for_trans=recipe,
                scale_rounding_mode=self.scale_rounding_mode,
            )
        return self.pack(components)

    def assemble(self, gathered):
        """Reorder rank-major wire data and restore zero kernel padding.

        Rank r owns rows [r*N, (r+1)*N) of EVERY expert. This is not Megatron's
        flat-bucket ownership. Returned tensors are bytes; no dequantization or
        requantization is performed. This reference path allocates new buffers.
        """
        if gathered.ndim != 2 or gathered.shape[0] == 0 or not gathered.is_contiguous():
            raise ValueError("expected contiguous [world_size, payload_bytes] storage")
        shards = [self.views(payload) for payload in gathered.unbind(0)]
        merged = []
        for index in range(4):
            logical = torch.cat([s[index] for s in shards], dim=1 if index < 2 else 2)
            # Turbo GEMM requires the contracted extent padded to 128 elements.
            alignment = 64 if index % 2 == 0 else 4
            padded_size = (logical.shape[-1] + alignment - 1) // alignment * alignment
            padded = logical.new_zeros((*logical.shape[:-1], padded_size))
            padded[..., : logical.shape[-1]].copy_(logical)
            merged.append(padded)
        return tuple(merged)

    def as_quantized_pair(self, gathered):
        """Create compute-ready cached wrappers, without reconstructing BF16 weights.

        Cache lifetime and invalidation after optimizer updates belong to the
        caller. Autograd/main_grad bridging also belongs to the training layer.
        """
        g, n, k = self.shape
        return self.wrap_components(self.assemble(gathered), (g, n * gathered.shape[0], k))

    def assemble_strip_shards(self, shards):
        """Assemble (first_strip, count, wire) pieces into this full weight.

        A strip is 32 complete rows. Ownership may vary by rank and cross expert
        boundaries. Scales and both orientations are copied without quantizing.
        """
        g, n, k = self.shape
        count = g * n // 32
        pieces = sorted(shards, key=lambda shard: shard[0])
        cursor = 0
        for first, length, _ in pieces:
            if first != cursor or length <= 0:
                raise ValueError("strip shards must cover the weight exactly once")
            cursor += length
        if cursor != count:
            raise ValueError("strip shards do not cover the weight")
        components = [[], [], [], []]
        for _first, length, payload in pieces:
            layout = MXFP4WireLayout((length, 32, k), self.scale_rounding_mode)
            for index, part in enumerate(layout.views(payload)):
                components[index].append(part)
        merged = []
        for index, parts in enumerate(components):
            strips = torch.cat(parts, dim=0)
            if index < 2:
                logical = strips.reshape(g, n, -1)
            else:
                logical = strips.reshape(g, n // 32, k, -1).permute(0, 2, 1, 3).reshape(g, k, -1)
            alignment = 64 if index % 2 == 0 else 4
            padded_size = (logical.shape[-1] + alignment - 1) // alignment * alignment
            padded = logical.new_zeros((*logical.shape[:-1], padded_size))
            padded[..., : logical.shape[-1]].copy_(logical)
            merged.append(padded)
        return tuple(merged)

    def wrap_components(self, components, shape=None):
        """Wrap already assembled kernel-layout buffers as compute operands."""
        from primus_turbo.pytorch.core.low_precision import (
            ScalingGranularity,
            ScalingRecipe,
            float4_e2m1fn_x2,
        )
        from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor, QuantizedTensorPair

        rd, rs, cd, cs = components
        common = dict(
            shape=torch.Size(self.shape if shape is None else shape),
            orig_dtype=torch.bfloat16,
            dest_dtype=float4_e2m1fn_x2,
            granularity=ScalingGranularity.MX_BLOCKWISE,
            block_size=32,
            scaling_recipe=ScalingRecipe(use_2d_block=True),
            scale_rounding_mode=self.scale_rounding_mode,
        )
        return QuantizedTensorPair(
            QuantizedTensor(
                rd.view(float4_e2m1fn_x2), rs.view(torch.float8_e8m0fnu), quantized_axis=2, **common
            ),
            QuantizedTensor(
                cd.view(float4_e2m1fn_x2), cs.view(torch.float8_e8m0fnu), quantized_axis=1, **common
            ),
        )
