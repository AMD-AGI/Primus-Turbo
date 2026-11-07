###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from primus_turbo.pytorch.core import stream


@pytest.fixture
def runtime(monkeypatch):
    # Exercise the wrapper contract without the ROCm CU-mask creation call that
    # currently deadlocks under parallel GPU load (see test_stream.py).
    create = Mock(return_value=1234)
    destroy = Mock()
    external = Mock(return_value=object())
    monkeypatch.setattr(stream.runtime, "create_stream_with_cu_masks", create)
    monkeypatch.setattr(stream.runtime, "destroy_stream", destroy)
    monkeypatch.setattr(torch.cuda, "ExternalStream", external)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(multi_processor_count=65)
    )
    return create, destroy, external


@pytest.mark.parametrize(
    "device, index", [(None, 2), (1, 1), ("cuda", 2), ("cuda:3", 3), (torch.device("cuda:4"), 4)]
)
@pytest.mark.parametrize(
    "masks, expected",
    [
        (None, [0xFFFFFFFF] * 3),
        (["invalid"], [0xFFFFFFFF] * 3),
        ([-1, 7, 1 << 40], [0, 7, 0xFFFFFFFF]),
        ((3, 5), [3, 5]),
    ],
)
def test_stream_device_and_masks(runtime, device, index, masks, expected):
    create, destroy, external = runtime
    obj = stream.TurboStream(device, masks)
    create.assert_called_once_with(index, expected)
    external.assert_called_once_with(1234, device=torch.device("cuda", index))
    assert obj.torch_stream is external.return_value
    del obj
    destroy.assert_called_once_with(index, 1234)


@pytest.mark.parametrize(
    "device, error, message",
    [("cpu", ValueError, "must be 'cuda'"), (object(), TypeError, "Unsupported device")],
)
def test_invalid_device_never_allocates(runtime, device, error, message):
    with pytest.raises(error, match=message):
        stream.TurboStream(device, None)
    runtime[0].assert_not_called()
    runtime[1].assert_not_called()


def test_unavailable_runtime(runtime, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="not available"):
        stream.TurboStream(None, None)
    runtime[0].assert_not_called()


def test_destroy_failure_warns(runtime):
    obj = stream.TurboStream(0, [1])
    runtime[1].side_effect = RuntimeError("driver failure")
    with pytest.warns(RuntimeWarning, match="TurboStream destroy failed: driver failure"):
        del obj
