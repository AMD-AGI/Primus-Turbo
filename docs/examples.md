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
