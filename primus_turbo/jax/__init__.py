###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
_initialized = False


def initialize():
    """Register FFI targets, primitives, and lowerings with JAX.

    Called automatically by JAX's ``jax_plugins`` entry-point discovery,
    or explicitly before first use of primus_turbo JAX ops.  Safe to call
    multiple times (subsequent calls are no-ops).
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    import jax
    from jax.interpreters import mlir

    from primus_turbo.jax._C import registrations
    from primus_turbo.jax.primitive import (
        ABSTRACT_EVAL_TABLE,
        IMPL_TABLE,
        LOWERING_TABLE,
    )

    for name, target in registrations().items():
        jax.ffi.register_ffi_target(name, target, platform="ROCM")

    for primitive, func in IMPL_TABLE.items():
        primitive.def_impl(func)

    for primitive, func in ABSTRACT_EVAL_TABLE.items():
        primitive.def_abstract_eval(func)

    for primitive, func in LOWERING_TABLE.items():
        mlir.register_lowering(primitive, func, platform="rocm")

