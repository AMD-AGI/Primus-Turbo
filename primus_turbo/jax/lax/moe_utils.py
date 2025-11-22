from typing import NamedTuple


class Config(NamedTuple):
    num_sms: int
    num_max_nvl_chunked_send_tokens: int
    num_max_nvl_chunked_recv_tokens: int
    num_max_rdma_chunked_send_tokens: int
    num_max_rdma_chunked_recv_tokens: int
