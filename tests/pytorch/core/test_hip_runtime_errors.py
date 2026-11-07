###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""HIP FFI diagnostics and serialized IPC-handle validation without unsafe launches."""

import ctypes

import pytest

from primus_turbo.pytorch.core.pyhip_runtime_wrapper import HIPRuntimeLibrary, hipIpcMemHandle_t


def test_missing_runtime_library_preserves_loader_error(tmp_path, capsys):
    missing = str(tmp_path / "missing-hip.so")
    with pytest.raises(OSError, match="missing-hip.so"):
        HIPRuntimeLibrary(missing)
    assert missing in capsys.readouterr().out
    assert missing not in HIPRuntimeLibrary.path_to_library_cache


@pytest.mark.parametrize("message", [None, b"invalid device ordinal"])
def test_driver_error_is_reported(message):
    runtime = HIPRuntimeLibrary.__new__(HIPRuntimeLibrary)
    runtime._funcs = {"hipGetErrorString": lambda result: message}
    expected = "Unknown HIP runtime error code: 101" if message is None else message.decode()
    assert runtime.hipGetErrorString(101) == expected
    runtime.HIP_CHECK(0)
    with pytest.raises(RuntimeError, match=expected):
        runtime.HIP_CHECK(101)


@pytest.mark.parametrize("size", [0, 1, 63, 65])
def test_ipc_handle_rejects_truncated_or_extended_payload(size):
    assert ctypes.sizeof(hipIpcMemHandle_t) == 64
    with pytest.raises(ValueError, match=f"Expected 64 bytes.*got {size}"):
        HIPRuntimeLibrary.mem_handle_from_bytes(bytes(size))
