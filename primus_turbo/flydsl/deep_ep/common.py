###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""Device helpers shared by the FlyDSL DeepEP intranode kernels."""

import flydsl.expr as fx

from primus_turbo.flydsl.utils.buffer_ops import (
    _unwrap_value,
    buffer_load,
    extract_base_index,
)
from primus_turbo.flydsl.utils.prims import cast

WARP_SIZE = 64


def load_index(res, elem_idx, element_type):
    """Load one i32 / i64 index element as i32."""
    value = buffer_load(res, elem_idx, vec_width=1, dtype=element_type)
    if element_type.width == 64:
        value = fx.arith.ArithValue(fx.arith.trunci(fx.T.i32, _unwrap_value(value)), signed=True)
    return value


def to_i64(value):
    """Sign-extend an i32 to i64 (``cast`` follows the value's signedness flag, which a select drops)."""
    return fx.arith.ArithValue(fx.arith.extsi(fx.T.i64, _unwrap_value(value)), signed=True)


def tensor_addr(tensor):
    """Base address of a kernel tensor argument as i64."""
    return cast(fx.arith.ArithValue(extract_base_index(tensor, address_space=1)), fx.T.i64)


_COMPILED_LAUNCHERS = {}


def run_launcher(launcher, constexprs, *args):
    """Run a ``@flyc.jit`` launcher through a ``flyc.compile``'d object, one per launcher, ``constexprs``
    and tensor dtype / rank.

    A compiled call skips ``@flyc.jit``'s per-call signature bind and cache-key build (~40 us -> ~5 us
    of host time), which is most of a small launch. The kernels address tensors through their base
    pointers only, so one compiled object serves every shape. ``args`` are all of the launcher's
    arguments, positionally (stream last); ``constexprs`` must hold every ``fx.Constexpr`` among them.
    Under CUDA-graph capture this takes the raw launcher, as a compiled object regresses there."""
    import flydsl.compiler as flyc
    import torch

    if torch.cuda.is_current_stream_capturing():
        return launcher(*args)
    key = (id(launcher), constexprs) + tuple((a.dtype, a.dim()) for a in args if isinstance(a, torch.Tensor))
    compiled = _COMPILED_LAUNCHERS.get(key)
    if compiled is None:
        # the compile runs the launch once
        _COMPILED_LAUNCHERS[key] = flyc.compile(launcher, *args)
    else:
        compiled(*args)
