###############################################################################

# Copyright (c) 2025 DeepSeek. All rights reserved.

# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.

#

# See LICENSE for license information.

###############################################################################


import torch

from .buffer import Buffer
from .fake_cpp_cls import FakeCppBuffer, FakeEventHandle
from .utils import EventOverlap

CppConfig = torch.classes.primus_turbo_cpp_extension.Config


@torch._library.register_fake_class("primus_turbo_cpp_extension::Config")
class Config:
    def __init__(
        self,
        num_sms,
        num_max_nvl_chunked_send_tokens,
        num_max_nvl_chunked_recv_tokens,
        num_max_rdma_chunked_send_tokens,
        num_max_rdma_chunked_recv_tokens,
    ):
        self.cfg = CppConfig(
            num_sms,
            num_max_nvl_chunked_send_tokens,
            num_max_nvl_chunked_recv_tokens,
            num_max_rdma_chunked_send_tokens,
            num_max_rdma_chunked_recv_tokens,
        )

    def get_nvl_buffer_size_hint(self, hidden_bytes, num_ranks):
        return self.cfg.get_nvl_buffer_size_hint(hidden_bytes, num_ranks)

    def get_rdma_buffer_size_hint(self, hidden_bytes, num_ranks):
        return self.cfg.get_rdma_buffer_size_hint(hidden_bytes, num_ranks)

    @classmethod
    def __obj_unflatten__(cls, flattened_obj):
        return cls(**dict(flattened_obj))


__all__ = ["Buffer", "EventOverlap", "Config"]
