###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Unit tests for SymmetricMemory input-validation paths.

PR #276 introduced a large new Python module (symm_mem.py) including
SymmetricMemory with several guard checks.  The guards themselves are pure
Python and do not call any HIP runtime function; we test them by constructing
a mock group and mocking the HIP lib so that hipMalloc raises, which lets
us reach the early-rejection code in __init__.

These tests are CPU-only and do not require a GPU or HIP driver.
"""

import ctypes
import unittest.mock as mock

import pytest
import torch

_cuda_available = torch.cuda.is_available()


def _make_fake_group(rank=0, world_size=1):
    """Return a minimal mock object that looks like a ProcessGroup."""
    g = mock.MagicMock()
    g.rank.return_value = rank
    g.size.return_value = world_size
    g.group_name = "fake_group"
    return g


@pytest.mark.skipif(_cuda_available, reason="Validation tested via input guards below; "
                    "skip pure-guard path on GPU builds to avoid real HIP calls")
class TestSymmetricMemoryInputValidation:
    """
    Tests that guard checks in SymmetricMemory.__init__ reject bad arguments
    before any HIP allocation happens.  The checks fire before the first HIP
    call so no GPU is needed.
    """

    def test_zero_alloc_size_raises_value_error(self):
        from primus_turbo.pytorch.core.symm_mem import SymmetricMemory

        group = _make_fake_group()
        with pytest.raises(ValueError, match="alloc size must be greater than 0"):
            SymmetricMemory(group, alloc_size=0)

    def test_negative_alloc_size_raises_value_error(self):
        from primus_turbo.pytorch.core.symm_mem import SymmetricMemory

        group = _make_fake_group()
        with pytest.raises(ValueError, match="alloc size must be greater than 0"):
            SymmetricMemory(group, alloc_size=-1)

    def test_zero_signal_pad_size_raises_value_error(self):
        from primus_turbo.pytorch.core.symm_mem import SymmetricMemory

        group = _make_fake_group()
        with pytest.raises(ValueError, match="signal_pad_size must be greater than 0"):
            SymmetricMemory(group, alloc_size=1024, signal_pad_size=0)

    def test_negative_signal_pad_size_raises_value_error(self):
        from primus_turbo.pytorch.core.symm_mem import SymmetricMemory

        group = _make_fake_group()
        with pytest.raises(ValueError, match="signal_pad_size must be greater than 0"):
            SymmetricMemory(group, alloc_size=1024, signal_pad_size=-8)


class TestCVoidPToInt:
    """Test the internal ctypes NULL-pointer conversion helper.

    This is pure Python logic; no GPU required.
    """

    def test_valid_pointer_returns_int(self):
        from primus_turbo.pytorch.core.symm_mem import SymmetricMemory

        ptr = ctypes.c_void_p(12345)
        result = SymmetricMemory._c_void_p_to_int(ptr)
        assert result == 12345

    def test_null_pointer_raises_value_error(self):
        from primus_turbo.pytorch.core.symm_mem import SymmetricMemory

        null_ptr = ctypes.c_void_p(None)  # NULL
        with pytest.raises(ValueError, match="NULL device pointer"):
            SymmetricMemory._c_void_p_to_int(null_ptr)

    def test_zero_pointer_raises_value_error(self):
        from primus_turbo.pytorch.core.symm_mem import SymmetricMemory

        zero_ptr = ctypes.c_void_p(0)
        with pytest.raises(ValueError, match="NULL device pointer"):
            SymmetricMemory._c_void_p_to_int(zero_ptr)


class TestGetSymmMemWorkspaceErrors:
    """
    get_symm_mem_workspace() should raise RuntimeError during CUDA graph capture.
    We mock is_current_stream_capturing to simulate this without needing a GPU.
    """

    def test_raises_during_graph_capture(self, monkeypatch):
        import primus_turbo.pytorch.core.symm_mem as symm_mem_mod

        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True, raising=False)

        group = _make_fake_group()
        with pytest.raises(RuntimeError, match="cannot resize the symmetric-memory workspace during CUDA graph capture"):
            symm_mem_mod.get_symm_mem_workspace(group, min_size=1024)
