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

"""Helpers shared by the gfx1250 FP8 block-scaled grouped GEMM kernels (nt, mx, vk, vkp, vkq)."""

from types import MappingProxyType

import flydsl.compiler as flyc
import flydsl.expr as fx

WAVE = 32
_A_ID = MappingProxyType({0: fx.Float8E4M3FN, 1: fx.Float8E5M2})


def _make_lds_load(bits):
    layout = fx.make_layout(bits // 32, 1)
    atom = fx.make_copy_atom(fx.UniversalCopy(bits), fx.Int32)
    ptr_ty = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, bits // 8)

    def load(base, byte_offset):
        ptr = fx.inttoptr(ptr_ty, fx.Int32(base) + fx.Int32(byte_offset))
        reg = fx.make_rmem_tensor(layout, fx.Int32)
        fx.copy_atom_call(atom, fx.Tensor(fx.make_view(ptr, layout)), reg)
        return reg.load()

    return load


def _ptr(t):
    return flyc.from_c_void_p(fx.Int8, t.data_ptr(), assumed_align=16)
