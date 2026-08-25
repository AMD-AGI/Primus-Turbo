###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple, Optional, Tuple

import torch
from torch._library.opaque_object import register_opaque_type

from primus_turbo.pytorch.core.utils import get_device_compute_capability

__all__ = ["float8_e4m3", "float8_e5m2"]


def is_fp8_dtype(dtype):
    TORCH_FP8_DTYPE = [
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    ]
    return dtype in TORCH_FP8_DTYPE


def is_fp4_dtype(dtype):
    TORCH_FP4_DTYPE = [
        torch.float4_e2m1fn_x2,
    ]
    return dtype in TORCH_FP4_DTYPE


def check_fp8_support() -> Tuple[bool, str]:
    """Return if fp8 support is available"""
    if get_device_compute_capability() >= (9, 4):
        return True, ""
    return (
        False,
        "Device compute capability gfx942 or higher required for FP8 execution.",
    )


def check_mxfp4_support() -> Tuple[bool, str]:
    """Return if fp4 support is available"""
    if get_device_compute_capability() >= (9, 5):
        return True, ""
    return (
        False,
        "Device compute capability gfx950 or higher required for FP4 execution.",
    )


def check_fp8_ocp_support() -> Tuple[bool, str]:
    """Return if fp8 ocp support is available"""
    if get_device_compute_capability() >= (9, 5):
        return True, ""
    return (
        False,
        "Device compute capability gfx950 or higher required for FP8 OCP format.",
    )


def check_mxfp8_support() -> Tuple[bool, str]:
    """Return if mxfp8 support is available"""
    if get_device_compute_capability() >= (9, 5):
        return True, ""
    return (
        False,
        "Device compute capability gfx950 or higher required for MXFP8 execution.",
    )


###################################################

try:
    if check_fp8_ocp_support()[0]:
        float8_e4m3 = torch.float8_e4m3fn
        float8_e5m2 = torch.float8_e5m2
    else:
        float8_e4m3 = torch.float8_e4m3fnuz
        float8_e5m2 = torch.float8_e5m2fnuz
    if check_mxfp4_support()[0]:
        float4_e2m1fn_x2 = torch.float4_e2m1fn_x2
    else:
        float4_e2m1fn_x2 = None
except AttributeError:
    raise RuntimeError("Your PyTorch build does not support FP8 types.") from None

###################################################

# Block size for MXFP4
MXFP4_BLOCK_SIZE = 32
# Padding align size for MXFP4
MXFP4_PADDING_ALIGN_SIZE = 128
# Block size for MXFP6
MXFP6_BLOCK_SIZE = 32
# Padding align size for MXFP6. The A6W6 ASM consumes K in units of 128.
MXFP6_PADDING_ALIGN_SIZE = 128
# Block size for MXFP8
MXFP8_BLOCK_SIZE = 32
# Padding align size for MXFP8
MXFP8_PADDING_ALIGN_SIZE = 128
# Block size for BLOCKWISE scaling
DEFAULT_BLOCK_SIZE = 128

# Geometry of AITER's "mxfp6_c0c1_256_padk2" packed blob, which the A6W6 ASM kernels
# consume directly. These are not tunable: the assembly has them baked in, and it
# derives its row-tile stride from MXFP6_K_TILES(K) + MXFP6_GUARD_K_TILES.
MXFP6_TILE_SIZE = 256
MXFP6_K_TILE_SIZE = 128
# Trailing K-tiles the blob must be sized for. Their contents are never read -- see
# the guard-tile probe -- but omitting the space makes every row-tile stride wrong.
MXFP6_GUARD_K_TILES = 2
MXFP6_PACKED_TILE_BYTES = 24576
MXFP6_SCALE_TILE_BYTES = 1024

# Elementwise epilogues the MXFP6 packer can fold into its staging read, so the tensor the
# epilogue produces never reaches HBM. Must match enum MXFP6Prologue in
# csrc/include/primus_turbo/quantization.h.
MXFP6_PROLOGUE_IDENTITY = 0
MXFP6_PROLOGUE_BIAS_GELU = 1
MXFP6_PROLOGUE_BIAS_GELU_BACKWARD = 2

# M-rows per row of the packer's bias-gradient partial buffer, i.e. its M-tile height.
# Must match MXFP6_COL_SUM_TILE_M in csrc/include/primus_turbo/quantization.h.
MXFP6_COL_SUM_TILE_M = 64


class Format(Enum):
    """
    Supported FP8/FP6/FP4 formats.
    """

    E4M3 = auto()
    E5M2 = auto()
    E2M1_X2 = auto()
    E2M3 = auto()
    HYBRID = auto()


class ScaleDtype(Enum):
    """
    Supported FP8/FP4 Scale data type.
    """

    FP32 = auto()
    E8M0 = auto()


class ScalingGranularity(Enum):
    """
    Supported FP8/FP4 scaling granularity.
    """

    TENSORWISE = auto()
    ROWWISE = auto()
    BLOCKWISE = auto()
    MX_BLOCKWISE = auto()


class ScalingStrategy(Enum):
    """
    Supported FP8/FP4 scaling strategy.
    """

    DYNAMIC = auto()
    # DELAYED_SCALING = auto() # TODO: undetermined


class ScalingRecipe(NamedTuple):
    """
    Supported MXFP8/MXFP4 scaling recipe.

    - use_2d_block: Whether to use 2D block in quantization. Available in blockwise, MXFP8 and MXFP4.
    - use_sr: Whether to use stochastic rounding in quantization. Available in MXFP4.
    - use_rht: The tensor will be apply by random Hadamard transform. Available in MXFP4.
    - shuffle_scale: Whether to shuffle the scale tensor. Available in MXFP4.
    - shuffle_output: Whether to shuffle the output tensor. Available in MXFP4.
    """

    use_2d_block: bool = False
    use_sr: bool = False
    use_rht: bool = False

    # Memory Layout Shuffle
    shuffle_scale: bool = False
    shuffle_out: bool = False

    def __fx_repr__(self) -> Tuple[str, dict]:
        return _quant_config_fx_repr(self)


def _quant_config_fx_repr(config) -> Tuple[str, dict]:
    """An evaluable repr plus its globals, for FX codegen of an opaque argument.

    Required by ``register_opaque_type(typ="value")``: torch.compile bakes the config
    into the graph as a constant, guarded on ``__eq__``, and regenerates it from this
    string. Enum fields have no evaluable ``repr``, so they are spelled out by name.

    Takes dataclasses and NamedTuples alike; the latter carry their fields in
    ``_asdict`` rather than ``__dict__``.
    """
    values = config._asdict() if hasattr(config, "_asdict") else config.__dict__
    fields = ", ".join(
        f"{name}={type(value).__name__}.{value.name}" if isinstance(value, Enum) else f"{name}={value!r}"
        for name, value in values.items()
    )
    globals_ = {type(config).__name__: type(config)}
    globals_.update({type(v).__name__: type(v) for v in values.values() if isinstance(v, Enum)})
    return f"{type(config).__name__}({fields})", globals_


@dataclass(unsafe_hash=True)  # hashable so it can be an opaque custom-op argument
class Float8QuantConfig:
    format: Format = Format.E4M3
    granularity: ScalingGranularity = ScalingGranularity.TENSORWISE
    strategy: ScalingStrategy = ScalingStrategy.DYNAMIC
    scale_dtype: ScaleDtype = ScaleDtype.FP32
    block_size: Optional[int] = None  # Default: not used for tensorwise/rowwise

    def __fx_repr__(self) -> Tuple[str, dict]:
        return _quant_config_fx_repr(self)

    def __post_init__(self):
        if self.granularity == ScalingGranularity.BLOCKWISE:
            assert self.block_size is not None, "block_size must be set when granularity is BLOCKWISE"

        if self.granularity == ScalingGranularity.MX_BLOCKWISE:
            mx_support_block_size = [MXFP8_BLOCK_SIZE]
            assert self.block_size in mx_support_block_size, (
                f"block_size should be {mx_support_block_size} when granularity is MX_BLOCKWISE"
            )

            mx_support_scale_dtype = ScaleDtype.E8M0
            assert self.scale_dtype == mx_support_scale_dtype, (
                f"scale_dtype should be {mx_support_scale_dtype} when granularity is MX_BLOCKWISE"
            )

    def tensorwise_scaling(self) -> bool:
        return (
            self.granularity == ScalingGranularity.TENSORWISE
            and self.strategy == ScalingStrategy.DYNAMIC
            and self.scale_dtype == ScaleDtype.FP32
        )

    def rowwise_scaling(self) -> bool:
        return self.granularity == ScalingGranularity.ROWWISE and self.scale_dtype == ScaleDtype.FP32

    def blockwise_scaling(self) -> bool:
        return self.granularity == ScalingGranularity.BLOCKWISE and self.scale_dtype == ScaleDtype.FP32

    def mxfp8_scaling(self) -> bool:
        return self.granularity == ScalingGranularity.MX_BLOCKWISE and self.scale_dtype == ScaleDtype.E8M0


@dataclass(unsafe_hash=True)  # hashable so it can be an opaque custom-op argument
class Float4QuantConfig:
    format: Format = Format.E2M1_X2
    granularity: ScalingGranularity = ScalingGranularity.MX_BLOCKWISE
    strategy: ScalingStrategy = ScalingStrategy.DYNAMIC
    scale_dtype: ScaleDtype = ScaleDtype.E8M0
    block_size: int = 32
    use_gradient_sr: bool = False
    use_preshuffle: bool = False

    def __fx_repr__(self) -> Tuple[str, dict]:
        return _quant_config_fx_repr(self)

    def __post_init__(self):
        assert self.granularity == ScalingGranularity.MX_BLOCKWISE, (
            "Float4QuantConfig currently only supports MX_BLOCKWISE granularity"
        )

        mx_support_block_size = [MXFP4_BLOCK_SIZE]
        assert self.block_size in mx_support_block_size, (
            f"block_size should be {mx_support_block_size} when granularity is MX_BLOCKWISE"
        )
        assert self.format == Format.E2M1_X2, "Format must be E2M1_X2 for Float4QuantConfig"

        mx_support_scale_dtype = ScaleDtype.E8M0
        assert self.scale_dtype == mx_support_scale_dtype, (
            f"scale_dtype should be {mx_support_scale_dtype} when granularity is MX_BLOCKWISE"
        )

    def mxfp4_scaling(self) -> bool:
        return self.granularity == ScalingGranularity.MX_BLOCKWISE and self.scale_dtype == ScaleDtype.E8M0


@dataclass
class Float6QuantConfig:
    """MXFP6 (E2M3) quantization config.

    Deliberately narrower than ``Float4QuantConfig``, because MXFP6 has far fewer
    degrees of freedom:

    - The 32-point Hadamard rotation is **mandatory**, not a recipe flag. It is fused
      into the packer and the GEMM relies on it cancelling between the two operands
      (``(A H)(B H)^T == A B^T``), so there is no un-rotated MXFP6 to opt into.
    - Scaling is strictly per-1x32 along the contraction axis, so a 2D block has no
      meaning.
    - There is no un-shuffled layout to choose. The A6W6 kernels read the packed
      C0/C1 tile blob directly, so the layout is part of the format rather than an
      option applied on top of it.
    """

    format: Format = Format.E2M3
    granularity: ScalingGranularity = ScalingGranularity.MX_BLOCKWISE
    strategy: ScalingStrategy = ScalingStrategy.DYNAMIC
    scale_dtype: ScaleDtype = ScaleDtype.E8M0
    block_size: int = MXFP6_BLOCK_SIZE
    # Accepted but not yet implemented; asserted off rather than silently ignored, so
    # a caller asking for SR gets an error instead of round-to-nearest.
    use_gradient_sr: bool = False

    def __post_init__(self):
        assert self.granularity == ScalingGranularity.MX_BLOCKWISE, (
            "Float6QuantConfig currently only supports MX_BLOCKWISE granularity"
        )
        assert self.format == Format.E2M3, "Format must be E2M3 for Float6QuantConfig"

        mx_support_block_size = [MXFP6_BLOCK_SIZE]
        assert self.block_size in mx_support_block_size, (
            f"block_size should be {mx_support_block_size} when granularity is MX_BLOCKWISE"
        )

        mx_support_scale_dtype = ScaleDtype.E8M0
        assert self.scale_dtype == mx_support_scale_dtype, (
            f"scale_dtype should be {mx_support_scale_dtype} when granularity is MX_BLOCKWISE"
        )

        assert not self.use_gradient_sr, (
            "use_gradient_sr is not implemented for MXFP6 yet. With 3 mantissa bits the "
            "MXFP4 motivation for stochastic rounding largely does not apply; remove this "
            "assert together with the SR path in the packer."
        )


# Lets a config travel through a torch.library custom op as a single argument rather
# than being flattened into scalars. A "value" type is specialized into the compiled
# graph and guarded on equality, which is what a static recipe wants. Note the schema
# admits these only as required parameters: neither a default nor an Optional of an
# opaque type is inferrable. Float6QuantConfig is absent because the MXFP6 ops take
# packed blobs rather than a config, so it never crosses a custom-op boundary.
for _opaque_cls in (Float8QuantConfig, Float4QuantConfig, ScalingRecipe):
    register_opaque_type(_opaque_cls, typ="value")
