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

"""Launch ``@flyc.jit`` functions through cached ``flyc.compile`` objects."""

import inspect

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

_COMPILED = {}
_CONSTEXPR_POSITIONS = {}


def _constexpr_positions(launcher):
    positions = _CONSTEXPR_POSITIONS.get(id(launcher))
    if positions is None:
        params = inspect.signature(launcher.func).parameters.values()
        positions = tuple(
            i
            for i, param in enumerate(params)
            if inspect.isclass(param.annotation) and issubclass(param.annotation, fx.Constexpr)
        )
        _CONSTEXPR_POSITIONS[id(launcher)] = positions
    return positions


def run_launcher(launcher, *args):
    """Run ``launcher(*args)`` through a ``flyc.compile``'d object, one per launcher, constexpr values
    and tensor dtype / rank.

    A compiled call skips ``@flyc.jit``'s per-call signature bind and cache-key build (~40 us -> ~5 us
    of host time), which is most of a small launch. A tensor that is only addressed through its base
    pointer compiles layout-dynamic, so one compiled object serves every shape. ``args`` are all of the
    launcher's arguments, positionally (stream last). Under CUDA-graph capture this takes the raw
    launcher, as a compiled object regresses there."""
    if torch.cuda.is_current_stream_capturing():
        launcher(*args)
        return
    key = (id(launcher),) + tuple(args[i] for i in _constexpr_positions(launcher))
    key += tuple((a.dtype, a.dim()) for a in args if isinstance(a, torch.Tensor))
    compiled = _COMPILED.get(key)
    if compiled is None:
        # the compile runs the launch once
        _COMPILED[key] = flyc.compile(launcher, *args)
    else:
        compiled(*args)
