## DeepEP (experimental)

DeepEP of Primus-Turbo is in the experimental stage.
The kernel code of DeepEP is primarily derived from ROCm internal DeepEP (it's still under development). It's **only used for training** and doesn't support low-latency kernels.

### Installation

#### 1. Dependencies

Hardware
- MI300 (gfx942) and MI355 (gfx950)
- MI455X (gfx1250): intranode only, with the FlyDSL backends (see [Intranode backend](#intranode-backend))

#### 2. Docker (Recommended)

Use AMD ROCm image:
```
docker.io/rocm/megatron-lm:v25.5_py310
```

#### 3. Install rocSHMEM (optional)

rocSHMEM is required for internode api of experimental DeepEP. Please refer to [our rocSHMEM Installation Guide](../../../docs/install_dependencies.md) for instructions.

> **Please Note: rocSHMEM is under development, no guarantee of full compatibility and performance for bnxt,mlx5 and ionic NIC driver.**

#### 4. Install from source
Please following [Primus-Turbo Install from Source](../../../README.md#3-install-from-source) instructions to install.

### Intranode backend

The intranode path (one node, 2, 4 or 8 ranks, no low-latency mode) has three kernel backends behind
the same `Buffer` API. Internode (rocSHMEM) and low-latency kernels always use `cpp`.

| Backend | Kernels | Runs on | Default on |
| --- | --- | --- | --- |
| `cpp` | HIP V1 (channel rings): `csrc/kernels/deep_ep/{layout,intranode}.cu` | gfx942, gfx950; not loaded on gfx1250 | gfx942, gfx950 |
| `flydsl` | FlyDSL V1, the same channel protocol: `primus_turbo/flydsl/deep_ep/intranode_kernel.py` | gfx942, gfx950, gfx1250 | builds without the C++ DeepEP |
| `flydsl_v2` | FlyDSL V2, every token written straight into a fixed slot on its destination rank, no rings: `primus_turbo/flydsl/deep_ep/intranode_v2_kernel.py` | gfx1250 (TDM); gfx942, gfx950 experimental (register copy) | gfx1250 (MI455X) |

All three return the same tensors. `cpp` and `flydsl` also return the same handles, so a handle from
one can drive the other's cached dispatch or combine; `flydsl_v2` handles work with `flydsl_v2` only.
On MI455X, `flydsl_v2` is 2.2–5.1× faster than `flydsl` at the kernel level and as fast or faster end
to end (`docs/flydsl_deepep_mi455.md`).

#### Choosing a backend

The first of these that is set wins:

1. `Buffer(group, num_nvl_bytes, intranode_backend="flydsl")`, for one buffer.
2. `PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND=cpp|flydsl|flydsl_v2`, for the process. `turbo.ops.moe_dispatch` /
   `moe_combine` (and so Primus's DeepEP dispatcher), `Buffer.get_dispatch_config` / `get_combine_config`
   and `Buffer.get_token_buffer_size_hint` read it too.
3. The default: `flydsl_v2` on gfx1250, `cpp` elsewhere (`flydsl` if the C++ DeepEP is not built).

`buffer.intranode_backend` tells which one a buffer runs. Nothing falls back silently: `flydsl_v2` off
gfx1250 and `cpp` on gfx1250 raise an error. When the intranode path does not apply (internode,
low-latency mode, other rank counts, `num_nvl_bytes == 0`), `Buffer` uses `cpp`, so on gfx1250 that
raises as well.

#### When to use `flydsl` instead of `flydsl_v2` on MI455X

- CUDA graphs: `flydsl_v2` raises under capture, because every call advances host-side epochs that a
  replay would reuse. Set `PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND=flydsl`, which keeps no host state
  between calls; as with `cpp`, a dispatch waits on the host for the receive counts unless it gets a
  cached handle or `num_worst_tokens`. `flydsl` under capture has not been tested on MI455X yet.
- Ranks that dispatch different numbers of tokens (THD packed sequences, dynamic context parallel):
  `flydsl_v2` sizes its buffer from the tokens per rank, which must be the same on every rank.

#### Buffer size

- `cpp`, `flydsl`: the channel rings, `Config.get_nvl_buffer_size_hint(hidden_bytes, num_ranks)`.
- `flydsl_v2`: the whole batch, `Buffer.get_token_buffer_size_hint(num_max_tokens_per_rank, hidden_bytes,
  num_ranks, num_topk)` (0 for the ring backends), with the same `num_max_tokens_per_rank` on every rank.
  `turbo.ops.moe_dispatch` sizes the buffer from the group's largest first dispatch (an all-reduce at
  allocation, so ranks may dispatch different numbers of tokens) and reallocates it for a larger one
  (after a device sync and a barrier, which waits for every rank to grow it); for token counts that
  change per step, `set_buffer_global_config(num_max_tokens_per_rank=...)` allocates it up front. If a rank receives more than the buffer holds, that rank raises
  `V2 dispatch receives ... > the heap's ...; raise num_nvl_bytes` and the others stop at the kernels'
  timeout (300 s on gfx1250, as the C++ DeepEP).

#### Number of CUs

`Buffer.set_num_sms(n)`, `set_buffer_global_config(num_use_cu=n)`, or `turbo_deepep_num_cu` in Primus.
On MI455X (EP2 / EP4) `flydsl_v2` saturates at 160 CUs and 64–160 CUs train within 0.6% of each other;
at 32 CUs its kernels take about twice as long as at 128. Use 128–160 there, and set it explicitly in
Primus, whose default is 32 (Primus also limits it to 32 at EP >= 16, for the internode kernels).

#### Examples

```bash
# MI455X: flydsl_v2 by default
torchrun --nproc-per-node 4 train.py
# MI455X with CUDA graphs, or token counts that change per step without a known maximum
PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND=flydsl torchrun --nproc-per-node 4 train.py
```

```python
from primus_turbo.pytorch.deep_ep import Buffer

# gfx1250: flydsl_v2, sized for 8192 tokens per rank, hidden 7168 bf16, top-8
buffer = Buffer(group, Buffer.get_token_buffer_size_hint(8192, 7168 * 2, group.size(), 8))
# a ring backend, sized by its rings
buffer = Buffer(group, num_nvl_bytes, intranode_backend="flydsl")
print(buffer.intranode_backend)
```

### Example

See [DeepEP example](../../../docs/examples.md#4-deepep)

### Benchmark Usage

 See [DeepEP benchmark](../../../benchmark/README.md#deepep)
