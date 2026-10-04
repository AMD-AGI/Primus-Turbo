###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The flydsl compatibility shims installed by ``primus_turbo.flydsl`` (no GPU needed)."""

import importlib

import pytest

flydsl = pytest.importorskip("flydsl")
# Installs the shims; must run before anything resolves flydsl.expr.buffer_ops.
importlib.import_module("primus_turbo.flydsl")

# The flydsl 0.2.4 buffer_ops API the Turbo kernels are written against.
_BUFFER_OPS_API = (
    "BufferResourceDescriptor",
    "buffer_load",
    "buffer_store",
    "create_buffer_resource",
    "create_buffer_resource_from_addr",
    "create_llvm_ptr",
    "extract_base_index",
    "get_element_ptr",
    "_unwrap_value",
    "_create_i32_constant",
    "_create_i64_constant",
)


def test_buffer_ops_api_available():
    import flydsl.expr.buffer_ops as by_path
    from flydsl.expr import buffer_ops

    assert buffer_ops is by_path
    missing = [name for name in _BUFFER_OPS_API if not hasattr(buffer_ops, name)]
    assert not missing, missing


def test_buffer_ops_vendored_only_when_flydsl_lacks_it():
    from flydsl.expr import buffer_ops

    vendored = buffer_ops.__name__ == "primus_turbo.flydsl._compat.buffer_ops"
    assert vendored == (not flydsl.__version__.startswith("0.2."))


def test_arith_value_alias():
    import flydsl.expr as fx
    from flydsl.expr.arith import ArithValue

    assert fx.ArithValue is ArithValue
