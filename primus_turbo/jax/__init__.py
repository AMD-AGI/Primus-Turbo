###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import jax
from jax.interpreters import ad, batching, mlir

from primus_turbo.jax._C import registrations
from primus_turbo.jax.primitive import (
    ABSTRACT_EVAL_TABLE,
    BATCHING_TABLE,
    IMPL_TABLE,
    LOWERING_TABLE,
    TRANSPOSE_TABLE,
)

# Import Turbo grouped GEMM module
from primus_turbo.jax import modules


def initialize():
    """the function of jax_plugins entry_point"""

    # print("[Primus-Turbo/Jax] : register_ffi_target")
    for name, target in registrations().items():
        jax.ffi.register_ffi_target(name, target, platform="ROCM")

    # print("[Primus-Turbo/Jax] : def_impl")
    for primitive, func in IMPL_TABLE.items():
        primitive.def_impl(func)

    # print("[Primus-Turbo/Jax] : def_abstract_eval")
    for primitive, func in ABSTRACT_EVAL_TABLE.items():
        primitive.def_abstract_eval(func)

    # print("[Primus-Turbo/Jax] : register_lowering")
    for primitive, func in LOWERING_TABLE.items():
        mlir.register_lowering(primitive, func, platform="rocm")

    # print("[Primus-Turbo/Jax] : primitive_transposes")
    # for primitive, func in TRANSPOSE_TABLE.items():
    #     ad.primitive_transposes[primitive] = func

    # print("[Primus-Turbo/Jax] : primitive_batchers")
    # for primitive, func in BATCHING_TABLE.items():
    #     batching.primitive_batchers[primitive] = func


# Export Turbo grouped GEMM API
__all__ = [
    "modules",
]
