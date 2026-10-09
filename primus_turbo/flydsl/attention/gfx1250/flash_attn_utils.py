###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""Helpers shared by the gfx1250 attention kernels: ``LOG2E``, ``WAVE_SIZE``, ``create_llvm_ptr``
(a raw LLVM pointer from an integer byte address) and ``_run_compiled`` (the compile-once launch
path of every forward and backward kernel).
"""

from .flydsl_version import require_flydsl

require_flydsl()

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

# UNSTABLE(gfx1250): raw MLIR context, only to unwind the one flyc.compile leaves open on failure.
from flydsl._mlir import ir

LOG2E = 1.4426950408889634
# Lanes per wave. gfx1250 runs wave32, and every layout and reshape in these kernels assumes it;
# FlyDSL only exposes the wave size compiler-side (GPUTarget.warp_size, from the arch), not as a
# trace-time Python int.
WAVE_SIZE = 32

# LLVM address-space numbers (1 global, 3 LDS) as fx address spaces; fx.to_llvm_ptr maps fx's
# Shared space back to !llvm.ptr<3>.
_FX_ADDRESS_SPACE = {1: fx.AddressSpace.Global, 3: fx.AddressSpace.Shared}


def create_llvm_ptr(value, address_space=1):
    """Raw ``!llvm.ptr`` for an integer byte address, for the LLVM / ROCDL ops that take one.

    ``address_space`` is the LLVM number: 1 (global) or 3 (LDS). An integer address has no
    ``fly.ptr`` form, so it goes through ``fx.inttoptr`` before ``fx.to_llvm_ptr``.
    """
    pt = fx.PointerType.get(fx.Int32.ir_type, address_space=_FX_ADDRESS_SPACE[address_space], alignment=4)
    return fx.as_ir_value(fx.to_llvm_ptr(fx.inttoptr(pt, value)))


# The package's one compiled-function cache: (flyc.jit launcher, device, (dtype, rank) of each
# tensor argument) -> the CompiledFunction flyc.compile returned for it.
_COMPILED = {}


def _run_compiled(exe, *args):
    """Launch the ``flyc.jit`` launcher ``exe`` on ``args`` through a cached ``CompiledFunction``.

    A plain ``flyc.jit`` call re-derives flydsl's whole cache key on every launch, a flat
    ~0.27 ms of host time; the ``CompiledFunction`` that ``flyc.compile`` returns only refreshes
    the argument storage. Every shape-dependent argument of these launchers is a runtime value,
    so one compiled function serves every shape; it is cached per launcher, device and tensor
    signature (each tensor's dtype and rank, which its argument storage is built for). flydsl's
    own JIT cache key has no device, and driving two devices from one process is untested.

    The first call, ``flyc.compile(exe, *args)``, compiles **and** launches. Under flydsl's
    ``COMPILE_ONLY`` it only compiles and returns None, which is not cached, so every call
    compiles and none launches.
    """
    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    key = (exe, tensors[0].device, tuple((t.dtype, t.dim()) for t in tensors))
    cf = _COMPILED.get(key)
    if cf is not None:
        cf(*args)
        return
    try:
        cf = flyc.compile(exe, *args)
    except Exception:
        # flyc.compile leaks ir.Context on failure; pop it so a retry takes the right path.
        try:
            while ir.Context.current is not None:
                ir.Context.current.__exit__(None, None, None)
        except Exception:  # noqa: BLE001, S110
            pass
        raise
    if cf is not None:
        _COMPILED[key] = cf
