# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Smoke test for the migrated ODC rocSHMEM backends.

The single-node host-API backend (originally librs_host5.so) and the multi-node
GPU-direct backend (originally librs_host_gda.so) were migrated into Turbo. The
implementations live in the kernels library (csrc/kernels/odc_rocshmem/*.cu,
declared in primus_turbo/odc_rocshmem/api.h); thin pybind bindings in
csrc/pytorch/dist/odc_rocshmem_*.cpp re-expose them as the ``odc_rocshmem_host`` /
``odc_rocshmem_gda`` submodules of _C. Both are compiled behind
``#ifndef DISABLE_ROCSHMEM``: on a toolchain without rocSHMEM the submodules are
simply absent from _C, so every test here skips instead of failing.

These are pure binding-surface checks: we verify the submodules exist, expose the
expected ABI symbols, and that the init-free ``rs_uid_bytes()`` sanity call is
callable and returns the rocSHMEM unique-id size (128). We deliberately do NOT
call anything that needs ``rocshmem_init`` (rs_init_uid / rs_malloc / gather /
reduce-scatter) so the test runs in a plain single-process pytest with no launcher.
"""

import primus_turbo.pytorch._C as C
import pytest

import primus_turbo

# rocSHMEM host-API symbols (mirrors _rocshmem_backend.py's ctypes surface for
# librs_host5.so): uid bootstrap + symmetric heap + peer-ptr resolve + PE query.
_HOST_SYMBOLS = [
    "rs_uid_bytes",
    "rs_get_uid",
    "rs_init_uid",
    "rs_malloc",
    "rs_ptr",
    "rs_barrier",
    "rs_finalize",
    "rs_my_pe",
    "rs_n_pes",
]

# GDA symbols: the host-compatible surface (so _rocshmem_backend can reuse it)
# plus the device gather / reduce-scatter launchers.
_GDA_SYMBOLS = [
    "rs_uid_bytes",
    "rs_get_uid",
    "rs_init_uid",
    "rs_malloc",
    "rs_ptr",
    "rs_barrier",
    "rs_finalize",
    "rs_my_pe",
    "rs_n_pes",
    "gda_gather",
    "gda_reduce_scatter_acc",
]

# rocSHMEM unique-id size (sizeof(rocshmem_uniqueid_t)); rs_uid_bytes() must match.
_EXPECTED_UID_BYTES = 128

# DISABLE_ROCSHMEM fallback: if the extension was built without rocSHMEM the
# submodules never get registered, so skip the whole module rather than fail.
_rocshmem_available = hasattr(C, "odc_rocshmem_host") and hasattr(C, "odc_rocshmem_gda")

pytestmark = pytest.mark.skipif(
    not _rocshmem_available,
    reason="rocSHMEM backends absent from _C (built with DISABLE_ROCSHMEM)",
)


def test_primus_turbo_importable():
    assert primus_turbo is not None
    assert hasattr(primus_turbo, "pytorch")


def test_submodules_present():
    assert hasattr(C, "odc_rocshmem_host"), "_C is missing odc_rocshmem_host submodule"
    assert hasattr(C, "odc_rocshmem_gda"), "_C is missing odc_rocshmem_gda submodule"


@pytest.mark.parametrize("symbol", _HOST_SYMBOLS)
def test_host_symbol_present(symbol):
    sub = C.odc_rocshmem_host
    assert hasattr(sub, symbol), f"odc_rocshmem_host missing symbol {symbol}"
    assert callable(getattr(sub, symbol)), f"odc_rocshmem_host.{symbol} not callable"


@pytest.mark.parametrize("symbol", _GDA_SYMBOLS)
def test_gda_symbol_present(symbol):
    sub = C.odc_rocshmem_gda
    assert hasattr(sub, symbol), f"odc_rocshmem_gda missing symbol {symbol}"
    assert callable(getattr(sub, symbol)), f"odc_rocshmem_gda.{symbol} not callable"


def test_host_rs_uid_bytes_sanity():
    # rs_uid_bytes() is sizeof(rocshmem_uniqueid_t): a compile-time constant that
    # needs no rocshmem_init, so it's the one host op safe to call in a smoke test.
    n = C.odc_rocshmem_host.rs_uid_bytes()
    assert isinstance(n, int)
    assert n == _EXPECTED_UID_BYTES, f"host rs_uid_bytes()={n}, expected {_EXPECTED_UID_BYTES}"


def test_gda_rs_uid_bytes_sanity():
    n = C.odc_rocshmem_gda.rs_uid_bytes()
    assert isinstance(n, int)
    assert n == _EXPECTED_UID_BYTES, f"gda rs_uid_bytes()={n}, expected {_EXPECTED_UID_BYTES}"


def test_host_gda_uid_bytes_agree():
    # Both backends bootstrap over the SAME unique-id, so their uid sizes must match.
    assert C.odc_rocshmem_host.rs_uid_bytes() == C.odc_rocshmem_gda.rs_uid_bytes()
