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
from flydsl.expr.primitive import get_dyn_shared
from flydsl.expr.primitive import ptrtoint as _fly_ptrtoint

from primus_turbo.flydsl.utils.buffer_ops import (
    _unwrap_value,
    buffer_load,
    extract_base_index,
)
from primus_turbo.flydsl.utils.prims import cast, ld, st

WARP_SIZE = 64


def load_index(res, elem_idx, element_type):
    """Load one i32 / i64 index element as i32."""
    value = buffer_load(res, elem_idx, vec_width=1, dtype=element_type)
    if element_type.width == 64:
        value = fx.arith.ArithValue(fx.arith.trunci(fx.T.i32, _unwrap_value(value)), signed=True)
    return value


def load_i32(res, idx):
    return buffer_load(res, idx, vec_width=1, dtype=fx.T.i32)


def tensor_addr(tensor):
    """Base address of a kernel tensor argument as i64."""
    return cast(fx.arith.ArithValue(extract_base_index(tensor, address_space=1)), fx.T.i64)


def lds_base():
    """Base address of the dynamic LDS allocation."""
    return _unwrap_value(_fly_ptrtoint(get_dyn_shared()))


def ld_lds(base, idx):
    return ld(base, fx.Int32(idx) if isinstance(idx, int) else idx, scope="workgroup", space=3)


def st_lds(base, idx, value):
    st(base, fx.Int32(idx) if isinstance(idx, int) else idx, value, scope="workgroup", space=3)
