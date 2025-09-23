# 💡 Primus-Turbo Example

This page shows usage of **Primus-Turbo**.


## 1. Operators

### 1.1 Gemm
```python
import torch
import primus_turbo.pytorch as turbo

device = "cuda:0"
dtype = torch.bfloat16
M = 128
N = 256
K = 512

# a [M, K]
a = torch.randn((M, K), dtype=dtype, device=device)
# b [K, N]
b = torch.randn((K, N), dtype=dtype, device=device)
# c [M, N]
c = turbo.ops.gemm(a, b, trans_a=False, trans_b=False, out_dtype=dtype)

print(c)
print(c.shape)
```

### 1.2 Attention

+ Simple Attention
```python
import torch
import primus_turbo.pytorch as turbo

device = "cuda:0"
dtype = torch.bfloat16

B = 4
S = 4096
H = 32
D = 128

q = torch.randn((B, S, H, D), dtype=dtype, device=device)
k = torch.randn((B, S, H, D), dtype=dtype, device=device)
v = torch.randn((B, S, H, D), dtype=dtype, device=device)
softmax_scale = q.shape[-1] ** (-0.5)

o = turbo.ops.flash_attn_func(q, k, v, softmax_scale=softmax_scale, causal=True)

print(o)
print(o.shape)
```

+ Attention with CP
```python
import os
import torch
import primus_turbo.pytorch as turbo

dtype = torch.bfloat16

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ.get("LOCAL_RANK", 0))

torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
torch.distributed.init_process_group(
    backend='nccl',
    world_size=world_size,
    rank=rank,
)

cp_group = torch.distributed.group.WORLD
cp_param_bundle = {
    "cp_group": cp_group,
    "cp_comm_type": "a2a"
}

B = 4
S = 4096
H = 256
D = 128

q = torch.randn((B, S, H, D), dtype=dtype, device=device)
k = torch.randn((B, S, H, D), dtype=dtype, device=device)
v = torch.randn((B, S, H, D), dtype=dtype, device=device)
softmax_scale = q.shape[-1] ** (-0.5)

o = turbo.ops.flash_attn_func(q, k, v, softmax_scale=softmax_scale, causal=True, cp_param_bundle=cp_param_bundle)

torch.distributed.destroy_process_group()
# run with torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12355 this_code.py
```


### 1.3 Grouped Gemm
```python
import torch
import primus_turbo.pytorch as turbo

device = "cuda:0"
dtype = torch.bfloat16

G = 4
M = 128  # 128=32+16+48+32
N = 256
K = 512

group_lens = torch.tensor([32, 16, 48, 32], dtype=torch.long, device=device)
a = torch.randn(M, K, device=device, dtype=dtype)
b = torch.randn(G, K, N, device=device, dtype=dtype)
c = turbo.ops.grouped_gemm(a, b, group_lens, trans_b=False)

print(c)
print(c.shape) # [128, 256]
```

## 2. Modules

### 2.1 Linear
```python
import torch
import primus_turbo.pytorch as turbo

device = "cuda:0"
dtype = torch.bfloat16

in_features = 512
out_features = 256
bias = True

input = torch.randn(128, in_features, device=device, dtype=dtype)
model = turbo.modules.Linear(
    in_features, out_features, bias=bias, device=device, dtype=dtype
)

# If you want to use torch.compile.
model = torch.compile(model, fullgraph=True, mode="max-autotune")

output = model(input)

print(model)
print(output)
print(output.shape)
```

## 3. FP8

This section introduces the **FP8 quantization config** and usage of **FP8 GEMM** and **FP8 GroupedGEMM** in Primus-Turbo.


### 3.1 Quantization Config (Float8QuantConfig)

FP8 quantization is configured through `Float8QuantConfig`:

- **format**
  - `Format.E4M3` (default)
  - `Format.E5M2`
- **granularity**
  - `ScalingGranularity.TENSORWISE` (default)
  - `ScalingGranularity.ROWWISE`

### 3.2 FP8 GEMM

Computation flow:

`FP16/BF16 → Quantize → FP8 → GEMM(FP8 × FP8) → FP16/BF16`

Example:

```python
import torch
import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.float8 import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)

device = "cuda:0"
dtype = torch.bfloat16

M, N, K = 128, 256, 512
# a [M, K]
a = torch.randn((M, K), dtype=dtype, device=device)
# b [N, K]
b = torch.randn((N, K), dtype=dtype, device=device)
# c [M, N]

# Set quant config through Float8QuantConfig class.
fp8_cfg = Float8QuantConfig(
    format=Format.E4M3,
    granularity=ScalingGranularity.TENSORWISE,  # or ROWWISE
)

c = turbo.ops.gemm_fp8(a, b, trans_a=False, trans_b=True, out_dtype=dtype, config=quant_config)
print(c)
print(c.shape) # [128, 256]
```

### 3.3 FP8 GroupedGEMM

Grouped GEMM supports multiple sub-matrices with different row sizes in a single call. The workflow is below:

`FP16/BF16 -> Quantize -> FP8 -> GroupedGEMM(FP8 × FP8) -> FP16/BF16`

Example:

```python
import torch
import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.float8 import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)

device = "cuda:0"
dtype = torch.bfloat16

# 4 groups, total rows M = 128
G, M, N, K = 4, 128, 256, 512
group_lens = torch.tensor([32, 16, 48, 32], device=device)

a = torch.randn(M, K, device=device, dtype=dtype)
b = torch.randn(G, N, K, device=device, dtype=dtype)  # shape [G, N, K] if trans_b=True

fp8_cfg = Float8QuantConfig(
    format=Format.E4M3,
    granularity=ScalingGranularity.TENSORWISE,  # or ROWWISE
)

c = turbo.ops.grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=fp8_cfg)
print(c)
print(c.shape)  # [128, 256]
```

**Quantization Config (Float8QuantConfig)**
* format
    * Format.E4M3 (default)
    * Format.E5M2
* granularity
    * ScalingGranularity.TENSORWISE (default)
    * ScalingGranularity.ROWWISE

## 4. DeepEP

We added some new params for DeepEP buffer and dispatch. The normal kernels can be used in model training as the below example code shows:

```python
import torch
import torch.distributed as dist
from typing import List, Tuple, Optional, Union

from primus_turbo.pytorch.deep_ep import Buffer, EventOverlap

# Communication buffer (will allocate at runtime)
_buffer: Optional[Buffer] = None

# Set the number of SMs to use
# NOTES: this is a static variable
Buffer.set_num_sms(24)


# You may call this function at the framework initialization
def get_buffer(group: dist.ProcessGroup, hidden_bytes: int, use_default_stream_as_comm_stream: bool = True) -> Buffer:
    global _buffer

    # NOTES: you may also replace `get_*_config` with your auto-tuned results via all the tests
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (Buffer.get_dispatch_config(group.size()), Buffer.get_combine_config(group.size())):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)

    # Allocate a buffer if not existed or not enough buffer size
    if _buffer is None or _buffer.group != group or _buffer.num_nvl_bytes < num_nvl_bytes or _buffer.num_rdma_bytes < num_rdma_bytes:
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes, use_default_stream_as_comm_stream=use_default_stream_as_comm_stream)
    return _buffer


def get_hidden_bytes(x: torch.Tensor) -> int:
    t = x[0] if isinstance(x, tuple) else x
    return t.size(1) * max(t.element_size(), 2)


def dispatch_forward(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                     num_experts: int, previous_event: Optional[EventOverlap] = None,
                     num_recv_tokens_per_expert_as_cuda: bool=False,) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor, List, Tuple, EventOverlap]:
    # NOTES: an optional `previous_event` means a CUDA event captured that you want to make it as a dependency
    # of the dispatch kernel, it may be useful with communication-computation overlap. For more information, please
    # refer to the docs of `Buffer.dispatch`
    global _buffer

    # Calculate layout before actual dispatch
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, previous_event = \
        _buffer.get_dispatch_layout(topk_idx, num_experts,
                                    previous_event=previous_event, async_finish=True,
                                    allocate_on_comm_stream=previous_event is not None)
    # Do MoE dispatch
    # NOTES: the CPU will wait for GPU's signal to arrive, so this is not compatible with CUDA graph
    # Unless you specify `num_worst_tokens`, but this flag is for intranode only
    # For more advanced usages, please refer to the docs of the `dispatch` function
    recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = \
        _buffer.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights,
                         num_tokens_per_rank=num_tokens_per_rank, num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                         is_token_in_rank=is_token_in_rank, num_tokens_per_expert=num_tokens_per_expert,
                         previous_event=previous_event, async_finish=True,
                         allocate_on_comm_stream=True,
                         num_recv_tokens_per_expert_as_cuda = False,)
    # For event management, please refer to the docs of the `EventOverlap` class
    return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event


def dispatch_backward(grad_recv_x: torch.Tensor, grad_recv_topk_weights: torch.Tensor, handle: Tuple) -> \
        Tuple[torch.Tensor, torch.Tensor, EventOverlap]:
    global _buffer

    # The backward process of MoE dispatch is actually a combine
    # For more advanced usages, please refer to the docs of the `combine` function
    combined_grad_x, combined_grad_recv_topk_weights, event = \
        _buffer.combine(grad_recv_x, handle, topk_weights=grad_recv_topk_weights, async_finish=True)

    # For event management, please refer to the docs of the `EventOverlap` class
    return combined_grad_x, combined_grad_recv_topk_weights, event


def combine_forward(x: torch.Tensor, handle: Tuple, previous_event: Optional[EventOverlap] = None) -> \
        Tuple[torch.Tensor, EventOverlap]:
    global _buffer

    # Do MoE combine
    # For more advanced usages, please refer to the docs of the `combine` function
    combined_x, _, event = _buffer.combine(x, handle, async_finish=True, previous_event=previous_event,
                                           allocate_on_comm_stream=previous_event is not None)

    # For event management, please refer to the docs of the `EventOverlap` class
    return combined_x, event


def combine_backward(grad_combined_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     handle: Tuple, previous_event: Optional[EventOverlap] = None) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], EventOverlap]:
    global _buffer

    # The backward process of MoE combine is actually a dispatch
    # For more advanced usages, please refer to the docs of the `dispatch` function
    grad_x, _, _, _, _, event = _buffer.dispatch(grad_combined_x, handle=handle, async_finish=True,
                                                 previous_event=previous_event,
                                                 allocate_on_comm_stream=previous_event is not None)

    # For event management, please refer to the docs of the `EventOverlap` class
    return grad_x, event
```
