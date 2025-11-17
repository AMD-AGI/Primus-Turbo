from typing import List, Optional, Tuple

import torch

from .utils import EventHandle

NUM_MAX_NVL_PEERS = 8

Config = torch.classes.primus_turbo_cpp_extension.Config


@torch._library.register_fake_class("primus_turbo_cpp_extension::Buffer")
class FakeCppBuffer:

    def __init__(
        self,
        rank: int,
        num_ranks: int,
        num_nvl_bytes: int,
        num_rdma_bytes: int,
        low_latency_mode: bool,
        explicitly_destroy: bool,
        use_default_stream_as_comm_stream: bool,
    ):
        self.rank = rank
        self.num_ranks = num_ranks
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.low_latency_mode = low_latency_mode
        self.explicitly_destroy = explicitly_destroy
        self.use_default_stream_as_comm_stream = use_default_stream_as_comm_stream

        self.available = True
        self.rdma_rank = rank // NUM_MAX_NVL_PEERS
        self.nvl_rank = rank % NUM_MAX_NVL_PEERS

    def is_available(self) -> bool:
        return self.available

    def get_num_rdma_ranks(self) -> int:
        return self.num_ranks // NUM_MAX_NVL_PEERS

    def get_rdma_rank(self) -> int:
        return self.rdma_rank

    def get_root_rdma_rank(self, is_global: bool) -> int:
        return self.nvl_rank if is_global else 0

    def get_local_device_id(self) -> int:
        return self.nvl_rank

    def get_local_ipc_handle(self) -> torch.Tensor:
        return torch.Tensor().cpu()

    def get_local_nvshmem_unique_id(self) -> torch.Tensor:
        return torch.Tensor().cpu()

    def get_local_buffer_tensor(self, dtype: torch.dtype, offset: int, use_rdma_buffer: bool) -> torch.Tensor:
        return torch.Tensor().cuda()

    def get_comm_stream(self) -> torch.Stream:
        return torch.cuda.Stream()

    def sync(
        self,
        device_ids: List[int],
        all_gathered_handles: torch.Tensor,
        root_unique_id_opt: Optional[torch.Tensor],
    ) -> None:
        pass

    def destroy(self) -> None:
        pass

    def get_dispatch_layout(
        self,
        topk_idx: torch.Tensor,
        num_experts: int,
        previous_event: Optional[EventHandle],
        async_finish: bool,
        allocate_on_comm_stream: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, Optional[EventHandle]]:
        return torch.Tensor().cuda(), None, torch.Tensor().cuda(), torch.Tensor().cuda(), None

    def intranode_dispatch(
        self,
        x: torch.Tensor,
        x_scales: Optional[torch.Tensor],
        topk_idx: Optional[torch.Tensor],
        topk_weights: Optional[torch.Tensor],
        num_tokens_per_rank: Optional[torch.Tensor],
        is_token_in_rank: torch.Tensor,
        num_tokens_per_expert: Optional[torch.Tensor],
        cached_num_recv_tokens: int,
        cached_rank_prefix_matrix: Optional[torch.Tensor],
        cached_channel_prefix_matrix: Optional[torch.Tensor],
        expert_alignment: int,
        num_worst_tokens: int,
        config: Optional[Config],
        previous_event: Optional[EventHandle],
        async_finish: bool,
        allocate_on_comm_stream: bool,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[int],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[EventHandle],
    ]:
        return (
            torch.Tensor().cuda(),
            None,
            None,
            None,
            [],
            torch.Tensor().cuda(),
            torch.Tensor().cuda(),
            torch.Tensor().cuda(),
            torch.Tensor().cuda(),
            torch.Tensor().cuda(),
            torch.Tensor().cuda(),
            None,
        )

    def intranode_combine(
        self,
        x: torch.Tensor,
        topk_weights: Optional[torch.Tensor],
        bias_0: Optional[torch.Tensor],
        bias_1: Optional[torch.Tensor],
        src_idx: torch.Tensor,
        rank_prefix_matrix: torch.Tensor,
        channel_prefix_matrix: torch.Tensor,
        send_head: torch.Tensor,
        config: Optional[Config],
        previous_event: Optional[EventHandle],
        async_finish: bool,
        allocate_on_comm_stream: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[EventHandle]]:
        return torch.Tensor().cuda(), None, None

    def internode_dispatch(
        self,
        x: torch.Tensor,
        x_scales: Optional[torch.Tensor],
        topk_idx: Optional[torch.Tensor],
        topk_weights: Optional[torch.Tensor],
        num_tokens_per_rank: Optional[torch.Tensor],
        num_tokens_per_rdma_rank: Optional[torch.Tensor],
        is_token_in_rank: torch.Tensor,
        num_tokens_per_expert: Optional[torch.Tensor],
        cached_num_recv_tokens: int,
        cached_num_rdma_recv_tokens: int,
        cached_rdma_channel_prefix_matrix: Optional[torch.Tensor],
        cached_recv_rdma_rank_prefix_sum: Optional[torch.Tensor],
        cached_gbl_channel_prefix_matrix: Optional[torch.Tensor],
        cached_recv_gbl_rank_prefix_sum: Optional[torch.Tensor],
        expert_alignment: int,
        num_worst_tokens: int,
        config: Optional[Config],
        previous_event: Optional[EventHandle],
        async_finish: bool,
        allocate_on_comm_stream: bool,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[int],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[EventHandle],
    ]:
        return (
            torch.Tensor().cuda(),
            None,
            None,
            None,
            [],
            torch.Tensor().cuda(),
            torch.Tensor().cuda(),
            torch.Tensor().cuda(),
            None,
            torch.Tensor().cuda(),
            None,
            torch.Tensor().cuda(),
            None,
            None,
            None,
            None,
        )

    def internode_combine(
        self,
        x: torch.Tensor,
        topk_weights: Optional[torch.Tensor],
        bias_0: Optional[torch.Tensor],
        bias_1: Optional[torch.Tensor],
        src_meta: torch.Tensor,
        is_combined_token_in_rank: torch.Tensor,
        rdma_channel_prefix_matrix: torch.Tensor,
        rdma_rank_prefix_sum: torch.Tensor,
        gbl_channel_prefix_matrix: torch.Tensor,
        combined_rdma_head: torch.Tensor,
        combined_nvl_head: torch.Tensor,
        config: Optional[Config],
        previous_event: Optional[EventHandle],
        async_finish: bool,
        allocate_on_comm_stream: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[EventHandle]]:
        return torch.Tensor().cuda(), None, None

    def __getstate__(self) -> dict:
        return {
            "rank": self.rank,
            "num_ranks": self.num_ranks,
            "num_nvl_bytes": self.num_nvl_bytes,
            "num_rdma_bytes": self.num_rdma_bytes,
            "low_latency_mode": self.low_latency_mode,
            "explicitly_destroy": self.explicitly_destroy,
            "use_default_stream_as_comm_stream": self.use_default_stream_as_comm_stream,
        }

    @classmethod
    def __obj_unflatten__(cls, flattened_obj):
        return cls(**dict(flattened_obj))
