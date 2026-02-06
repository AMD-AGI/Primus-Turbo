# 💡 Primus-Turbo Example

This page shows usage of **Primus-Turbo**.

## Table of Contents

- [1. Operators](#1-operators)
  - [1.1 Gemm](#11-gemm)
  - [1.2 Attention](#12-attention)
  - [1.3 Grouped Gemm](#13-grouped-gemm)
- [2. Modules](#2-modules)
  - [2.1 Linear](#21-linear)
- [3. Low-Precision](#3-low-precision)
  - [3.1 Quantization Config](#31-quantization-config)
  - [3.2 FP8 GEMM](#32-fp8-gemm)
  - [3.3 FP8 GroupedGEMM](#33-fp8-groupedgemm)
  - [3.4 FP4 GEMM](#34-fp4-gemm)
- [4. DeepEP](#4-deepep)
- [5. Backend and AutoTune](#5-backend-and-autotune)
  - [5.1 Backend Selection](#51-backend-selection)
  - [5.2 AutoTune](#52-autotune)
  - [5.3 Environment Variables](#53-environment-variables)


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

from torch.distributed.device_mesh import init_device_mesh

dtype = torch.bfloat16

world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ.get("LOCAL_RANK", 0))

torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

ulysses_degree = 4
ring_degree = 2
device_mesh = init_device_mesh(
    "cuda",
    (ring_degree, ulysses_degree),
    mesh_dim_names=("ring", "ulysses"))

B = 4
S = 4096
H = 256
D = 128

q = torch.randn((B, S, H, D), dtype=dtype, device=device)
k = torch.randn((B, S, H, D), dtype=dtype, device=device)
v = torch.randn((B, S, H, D), dtype=dtype, device=device)
softmax_scale = q.shape[-1] ** (-0.5)

o = turbo.ops.flash_attn_usp_func(q,
        k,
        v,
        softmax_scale=softmax_scale,
        ulysses_group=device_mesh["ulysses"].get_group(),
        ring_group=device_mesh["ring"].get_group())

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

## 3. Low-Precision

This section introduces the **FP8/FP4 quantization config** and usage of **FP8/FP4 GEMM** and **FP8 GroupedGEMM** in Primus-Turbo.


### 3.1 Quantization Config

FP8 quantization is configured through `Float8QuantConfig`:

- **format**
  - `Format.E4M3` (default)
  - `Format.E5M2`
  - `Format.HYBRID` (backward calculation of dgrad is e5m2 other is e4m3.)
- **granularity**
  - `ScalingGranularity.TENSORWISE` (default)
  - `ScalingGranularity.ROWWISE`
  - `ScalingGranularity.BLOCKWISE`
  - `ScalingGranularity.MX_BLOCKWISE`
- **scale dtype**
  - `ScaleDtype.FP32` (default)
  - `ScaleDtype.E8M0` (for mx)
- **block_size**
  - Specifies the size of each block when using BLOCKWISE and MX_BLOCKWISE granularity.
  - This parameter must be explicitly specified in BLOCKWISE and MX_BLOCKWISE mode; otherwise, an error will be raised.

FP4 quantization is configured through `Float4QuantConfig`:

- **format**
  - `Format.E2M1_X2` (default)
- **granularity**
  - `ScalingGranularity.MX_BLOCKWISE` (default)
- **scale dtype**
  - `ScaleDtype.E8M0` (default)
- **block_size**
  - Specifies the size of each block when using MX_BLOCKWISE granularity.


### 3.2 FP8 GEMM

Computation flow:

`FP16/BF16 → Quantize → FP8 → GEMM(FP8 × FP8) → FP16/BF16`

Example:

```python
import torch
import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.low_precision import (
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

c = turbo.ops.gemm_fp8(a, b, trans_a=False, trans_b=True, out_dtype=dtype, config=fp8_cfg)
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
from primus_turbo.pytorch.core.low_precision import (
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

### 3.4 FP4 GEMM

For FP4 GEMM, it will apply some recipe (such as 2D block quantize and random hadamard transform) in quantization. Reference: https://arxiv.org/pdf/2509.25149


Computation flow:

`FP16/BF16 → Quantize → FP4 → GEMM(FP4 × FP4) → FP16/BF16`

Example:

```python
import torch
import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.low_precision import (
    Float4QuantConfig,
    Format,
    ScalingGranularity,
    ScaleDtype,
)

device = "cuda:0"
dtype = torch.bfloat16

M, N, K = 128, 256, 512
# a [M, K]
a = torch.randn((M, K), dtype=dtype, device=device)
# b [N, K]
b = torch.randn((N, K), dtype=dtype, device=device)
# c [M, N]

# Set quant config through Float4QuantConfig class.
fp4_cfg = Float4QuantConfig(
    format=Format.E2M1_X2,
    granularity=ScalingGranularity.MX_BLOCKWISE,
    block_size=32,
    scale_dtype=ScaleDtype.E8M0,
)

c = turbo.ops.gemm_fp4(a, b, trans_a=False, trans_b=True, out_dtype=dtype, config=fp4_cfg)
print(c)
print(c.shape) # [128, 256]
```

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
def get_buffer(group: dist.ProcessGroup,
               hidden_bytes: int,
               use_comm_stream: bool = False) -> Buffer:
    global _buffer

    # NOTES: you may also replace `get_*_config` with your auto-tuned results via all the tests
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (Buffer.get_dispatch_config(group.size()), Buffer.get_combine_config(group.size())):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)

    # Allocate a buffer if not existed or not enough buffer size
    if _buffer is None or _buffer.group != group or _buffer.num_nvl_bytes < num_nvl_bytes or _buffer.num_rdma_bytes < num_rdma_bytes:
        _buffer = Buffer(group,
                         num_nvl_bytes,
                         num_rdma_bytes,
                         use_default_stream_as_comm_stream=not use_comm_stream)
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
                         num_recv_tokens_per_expert_as_cuda=num_recv_tokens_per_expert_as_cuda,)
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

## 5. Backend and AutoTune

Some Primus-Turbo operators provide multiple backends. You can force a backend (for reproducibility/debugging) or enable AutoTune to pick the fastest backend for your shapes on the current GPU.

Priority (high to low):

1. Code settings (`GlobalBackendManager.set_*_backend(...)`)
2. Environment variables (`PRIMUS_TURBO_*_BACKEND`)
3. AutoTune (`PRIMUS_TURBO_AUTO_TUNE=1`)
4. Operator defaults
5. Fallback: try all registered backends

### 5.1 Backend Selection

You can set backends in code via `GlobalBackendManager`.

```python
import torch
import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)

device = "cuda:0"
dtype = torch.bfloat16

# Set GEMM backend to CK
GlobalBackendManager.set_gemm_backend(BackendType.CK)

M, N, K = 128, 256, 512
a = torch.randn((M, K), dtype=dtype, device=device)
b = torch.randn((N, K), dtype=dtype, device=device)  # [N, K] when trans_b=True

fp8_cfg = Float8QuantConfig(
    format=Format.E4M3,
    granularity=ScalingGranularity.TENSORWISE,
)

out = turbo.ops.gemm_fp8(a, b, trans_a=False, trans_b=True, out_dtype=dtype, config=fp8_cfg)
print(out.shape)

# Unset GEMM backend
GlobalBackendManager.set_gemm_backend(None)
```

### 5.2 AutoTune

AutoTune profiles compatible backends on the first call (per input "key", usually shape/dtype/layout) and caches the fastest choice. This is useful for stable shapes, but may add overhead if shapes change frequently.

NOTE: AutoTune is skipped during CUDA graph capture.

```python
import torch
import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.backend import GlobalBackendManager

GlobalBackendManager.set_auto_tune(True)  # or set PRIMUS_TURBO_AUTO_TUNE=1

# First call may be slower due to profiling; later calls hit the cache.
# Use fixed shapes for best results.
device = "cuda:0"
dtype = torch.bfloat16
G, M, N, K = 4, 128, 256, 512
group_lens = torch.tensor([32, 16, 48, 32], dtype=torch.long, device=device)
a = torch.randn(M, K, device=device, dtype=dtype)
b = torch.randn(G, K, N, device=device, dtype=dtype)
out = turbo.ops.grouped_gemm(a, b, group_lens, trans_b=False)
print(out.shape)

# Clear backend settings + all AutoTune caches.
GlobalBackendManager.reset()
```

### 5.3 Environment Variables

You can also control backend selection and AutoTune via environment variables:

```bash
export PRIMUS_TURBO_AUTO_TUNE=1
export PRIMUS_TURBO_GEMM_BACKEND=HIPBLASLT
export PRIMUS_TURBO_GROUPED_GEMM_BACKEND=CK
export PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND=DEEP_EP
```
