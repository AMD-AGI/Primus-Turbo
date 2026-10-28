###############################################################################
# Copyright (c) 2025 DeepSeek. All rights reserved.
#
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The C++ DeepEP module when it is built, else Python stand-ins for what the FlyDSL
intranode backend needs of it (``Config``, ``EventHandle``).

The C++ DeepEP does not run on gfx1250 (MI455X) even where ``_C`` carries it, and ``_C``
has no ``deep_ep`` where it is not compiled; the FlyDSL intranode backend runs without it.
"""

from typing import Optional

import torch

try:
    import primus_turbo.pytorch._C.deep_ep as deep_ep_cpp
except ImportError:
    deep_ep_cpp = None

if deep_ep_cpp is not None:
    # rocm_agent_enumerator, so importing this does not initialize the device
    from flydsl.runtime.device import get_rocm_arch

    if get_rocm_arch().startswith("gfx1250"):
        deep_ep_cpp = None

NUM_MAX_NVL_PEERS = 8


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class PyConfig:
    """``primus_turbo::deep_ep::Config`` (``csrc/include/primus_turbo/deep_ep/config.hpp``)."""

    def __init__(
        self,
        num_sms: int = 20,
        num_max_nvl_chunked_send_tokens: int = 6,
        num_max_nvl_chunked_recv_tokens: int = 256,
        num_max_rdma_chunked_send_tokens: int = 6,
        num_max_rdma_chunked_recv_tokens: int = 256,
    ) -> None:
        if num_sms < 0:
            raise RuntimeError("Config: num_sms must be non-negative")
        if not (0 < num_max_nvl_chunked_send_tokens < num_max_nvl_chunked_recv_tokens):
            raise RuntimeError("Config: need 0 < nvl chunked send tokens < nvl chunked recv tokens")
        if num_max_rdma_chunked_send_tokens <= 0 or num_max_rdma_chunked_recv_tokens <= 0:
            raise RuntimeError("Config: rdma chunked tokens must be positive")
        num_max_rdma_chunked_recv_tokens = _align(
            num_max_rdma_chunked_recv_tokens, num_max_rdma_chunked_send_tokens
        )
        if not (num_max_rdma_chunked_send_tokens <= num_max_rdma_chunked_recv_tokens // 2):
            raise RuntimeError("Config: rdma chunked send tokens must be at most half the recv tokens")
        self.num_sms = num_sms
        self.num_max_nvl_chunked_send_tokens = num_max_nvl_chunked_send_tokens
        self.num_max_nvl_chunked_recv_tokens = num_max_nvl_chunked_recv_tokens
        self.num_max_rdma_chunked_send_tokens = num_max_rdma_chunked_send_tokens
        self.num_max_rdma_chunked_recv_tokens = num_max_rdma_chunked_recv_tokens

    def get_nvl_buffer_size_hint(self, hidden_bytes: int, num_ranks: int) -> int:
        """The C++ hint without the internode source-meta term (no rocSHMEM here)."""
        num_max_topk = num_max_scales = 128
        if not (num_ranks < NUM_MAX_NVL_PEERS or num_ranks % NUM_MAX_NVL_PEERS == 0):
            raise RuntimeError(f"Config: unsupported num_ranks {num_ranks}")
        if not (num_ranks <= NUM_MAX_NVL_PEERS or self.num_sms % 2 == 0):
            raise RuntimeError("Config: num_sms must be even")
        num_rdma_ranks = max(num_ranks // NUM_MAX_NVL_PEERS, 1)
        num_nvl_ranks = min(num_ranks, NUM_MAX_NVL_PEERS)
        slots = (self.num_sms // 2) * num_nvl_ranks
        tokens = slots * self.num_max_nvl_chunked_recv_tokens
        num_bytes = slots * (2 * num_rdma_ranks + 3) * 4
        num_bytes += tokens * hidden_bytes
        num_bytes += tokens * num_max_topk * 8
        num_bytes += tokens * num_max_topk * 4
        num_bytes += tokens * num_max_scales * 4
        return _align(num_bytes, 128)

    def get_rdma_buffer_size_hint(self, hidden_bytes: int, num_ranks: int) -> int:
        if num_ranks <= NUM_MAX_NVL_PEERS:
            return 0
        raise RuntimeError("Config: the internode RDMA buffer needs the C++ DeepEP")


class PyEventHandle:
    """``deep_ep::EventHandle``: an event recorded on ``stream`` (default: the current stream)."""

    def __init__(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        self.event = torch.cuda.Event()
        self.event.record(torch.cuda.current_stream() if stream is None else stream)

    def current_stream_wait(self) -> None:
        torch.cuda.current_stream().wait_event(self.event)


Config = deep_ep_cpp.Config if deep_ep_cpp is not None else PyConfig
EventHandle = deep_ep_cpp.EventHandle if deep_ep_cpp is not None else PyEventHandle
