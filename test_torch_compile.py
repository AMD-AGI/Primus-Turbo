import torch

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.deep_ep import Config

EventHandle = torch.classes.primus_turbo_cpp_extension.EventHandle


def test_buffer(group):
    num_tokens, hidden, num_topk, num_experts = 4096, 4096, 8, 128
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * rank
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device="cuda") * rank
    out = turbo.ops.deepep_dispatch(x, topk_idx, topk_weights, num_experts, group)
    return out


def test_config(
    num_sms,
    num_max_nvl_chunked_send_tokens,
    num_max_nvl_chunked_recv_tokens,
    num_max_rdma_chunked_send_tokens,
    num_max_rdma_chunked_recv_tokens,
):

    config = Config(
        num_sms,
        num_max_nvl_chunked_send_tokens,
        num_max_nvl_chunked_recv_tokens,
        num_max_rdma_chunked_send_tokens,
        num_max_rdma_chunked_recv_tokens,
    )
    return config


def test_event_overlap():
    event_overlap = EventHandle()
    return event_overlap


# compiled_event_overlap = torch.compile(test_event_overlap, fullgraph=True)
compiled_cfg = torch.compile(test_config, fullgraph=True)
# print(compiled_event_overlap())
print(compiled_cfg(20, 24, 256, 6, 128).get_nvl_buffer_size_hint(7168, 8))

# compiled_buffer = torch.compile(test_buffer)

# rank = int(os.getenv("LOCAL_RANK", 0))
# torch.cuda.set_device(rank)

# dist.init_process_group(backend="nccl", rank=rank, world_size=8)
# group = dist.group.WORLD

# print(compiled_buffer(group))
