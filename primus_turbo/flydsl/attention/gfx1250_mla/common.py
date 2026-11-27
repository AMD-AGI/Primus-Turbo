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

"""Helpers shared by the gfx1250 MLA kernels' host code and FlyDSL shells."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

# UNSTABLE(gfx1250): raw LLVM pointers, volatile loads / stores and inline asm have no stable API in FlyDSL 0.3.4.1.
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr

_NUM_CUS = {}  # device index -> CU count


def num_cus(device) -> int:
    """Compute units of ``device`` (256, an MI455X, for the CPU tensors of a compile-only run)."""
    if device.type != "cuda":
        return 256
    cus = _NUM_CUS.get(device.index)
    if cus is None:
        cus = _NUM_CUS[device.index] = torch.cuda.get_device_properties(device).multi_processor_count
    return cus


def lds_ptr(addr):
    """Raw ``!llvm.ptr<3>`` for an LDS byte address."""
    pt = fx.PointerType.get(fx.Int32.ir_type, address_space=fx.AddressSpace.Shared, alignment=4)
    return fx.to_llvm_ptr(fx.inttoptr(pt, addr))


def global_ptr(addr):
    """Raw ``!llvm.ptr<1>`` for a global byte address."""
    pt = fx.PointerType.get(fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4)
    return fx.to_llvm_ptr(fx.inttoptr(pt, addr))


def inline_region(sched, name, *operands):
    """The generated region ``name`` of the generator module ``sched`` (fwd_sched / bwd_dkdv_sched) as one inline asm
    statement on ``operands``."""
    asm, cons = sched.region_asm(name)
    llvm.inline_asm(
        res=None,
        operands_=[fx.as_ir_value(x) for x in operands],
        asm_string=asm,
        constraints=cons,
        has_side_effects=True,
    )


def reserve_vgprs(cond, lds, count):
    """Makes the wave's VGPR allocation reach 1024 VGPRs for the generated regions, whose state lives in VGPRs up to
    v1023: inline asm constraints cannot name VGPRs above v255, so LLVM does not count them. A block that is never
    taken (``cond`` is false at run time, which LLVM cannot tell) keeps ``count`` 16-byte values live at once
    (volatile LDS loads at immediate offsets from ``lds``, stored back in reverse order): LLVM's register allocation
    then reaches 4 * count VGPRs on top of the kernel's own. This rests on how LLVM allocates registers, which
    nothing in the language pins down: a new compiler must show at least 1009 VGPRs (the kernel descriptor rounds up
    to 1024: a gfx1250 wave gets its VGPRs in blocks of 16) in the kernel's final ISA (``.num_vgpr``: 1018 and 1022
    in the builds that were validated)."""

    @flyc.jit
    def reserve(cond):
        if cond:
            v4 = fx.Vector.make_type(4, fx.Int32)
            vals = [
                llvm.load(v4, lds_ptr(lds + fx.Int32(16 * i)), volatile_=True) for i in range_constexpr(count)
            ]
            for i in range_constexpr(count):
                j = count - 1 - i
                llvm.store(vals[j], lds_ptr(lds + fx.Int32(16 * j)), volatile_=True)

    reserve(cond)
