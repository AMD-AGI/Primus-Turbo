###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

import pytest

from primus_turbo.pytorch.deep_ep._compat import PyConfig


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"num_sms": -1}, "num_sms"),
        ({"num_max_nvl_chunked_send_tokens": 0}, "nvl chunked"),
        ({"num_max_nvl_chunked_send_tokens": 256}, "nvl chunked"),
        ({"num_max_rdma_chunked_send_tokens": 0}, "rdma chunked"),
        ({"num_max_rdma_chunked_recv_tokens": 0}, "rdma chunked"),
        ({"num_max_rdma_chunked_send_tokens": 8, "num_max_rdma_chunked_recv_tokens": 8}, "at most half"),
    ],
)
def test_invalid_config(kwargs, message):
    with pytest.raises(RuntimeError, match=message):
        PyConfig(**kwargs)


def test_rdma_receive_capacity_rounds_to_send_chunk():
    config = PyConfig(num_max_rdma_chunked_send_tokens=6, num_max_rdma_chunked_recv_tokens=13)
    assert config.num_max_rdma_chunked_recv_tokens == 18


@pytest.mark.parametrize("ranks", [1, 2, 8, 16])
def test_nvl_capacity_alignment_and_growth(ranks):
    # One receive slot per peer and half of the 20 SMs; per-token metadata is
    # 128 entries each for int64 indices, float32 weights and float32 scales.
    config = PyConfig()
    slots = 10 * min(ranks, 8)
    tokens = slots * 256
    header = slots * (2 * max(ranks // 8, 1) + 3) * 4
    expected = ((header + tokens * (512 + 128 * (8 + 4 + 4)) + 127) // 128) * 128
    assert config.get_nvl_buffer_size_hint(512, ranks) == expected
    assert config.get_nvl_buffer_size_hint(640, ranks) - expected == tokens * 128


def test_unsupported_rank_count_and_odd_sms():
    with pytest.raises(RuntimeError, match="unsupported num_ranks"):
        PyConfig().get_nvl_buffer_size_hint(512, 9)
    with pytest.raises(RuntimeError, match="num_sms must be even"):
        PyConfig(num_sms=3).get_nvl_buffer_size_hint(512, 16)


def test_rdma_requires_cpp_only_for_internode():
    config = PyConfig()
    assert config.get_rdma_buffer_size_hint(512, 8) == 0
    with pytest.raises(RuntimeError, match=r"needs the C\+\+ DeepEP"):
        config.get_rdma_buffer_size_hint(512, 16)


@pytest.mark.parametrize("explicit_stream", [False, True])
def test_event_handle_waits_for_recorded_work(explicit_stream):
    import torch

    from primus_turbo.pytorch.deep_ep._compat import PyEventHandle

    source = torch.cuda.Stream() if explicit_stream else torch.cuda.current_stream()
    with torch.cuda.stream(source):
        value = torch.arange(128, device="cuda") * 2
    event = PyEventHandle(source if explicit_stream else None)
    event.current_stream_wait()
    torch.testing.assert_close(value.cpu(), torch.arange(128) * 2)
    assert event.event.query()
