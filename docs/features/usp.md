# Context-parallel flash attention (USP)

## Summary

`flash_attn_usp_func` and its varlen / FP8 variants run flash attention under context
parallelism (CP). Each rank holds a contiguous shard of the sequence. An all-to-all in the
style of [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509) exchanges heads so that every
rank sees the full sequence on a subset of the heads; an optional ring group then walks the
sequence shards with the same flash-attention kernels that the non-CP ops use. The output
all-to-all brings the local token shard back. Forward and backward are both implemented.

Three entry points:

- `flash_attn_usp_func` — dense BF16 / FP16, Ulysses all-to-all plus optional ring;
- `flash_attn_varlen_usp_func` — packed variable-length sequences (THD), Ulysses only;
- `flash_attn_fp8_usp_func` — FP8 blockwise, Ulysses only.

`TurboAttention` picks the USP path when it is constructed with an `ulysses_group`.

## Support matrix

| | gfx942 (MI300X / MI325X) | gfx950 (MI350X / MI355X) | gfx1250 (MI455X) |
|---|:-:|:-:|:-:|
| `flash_attn_usp_func` (BF16 / FP16) | ✅ | ✅ | — |
| `flash_attn_varlen_usp_func` (BF16) | ✅ | ✅ | — |
| `flash_attn_fp8_usp_func` (FP8 blockwise) | ✅ | ✅ | — |

Ulysses degrees covered by the tests: 2 / 4 / 8. Dense BF16 also covers a combined mesh
(`ulysses_degree` × `ring_degree` = world size), including ring-only (`ulysses_degree=1`).

Frameworks: PyTorch. See [Supported Operators § Attention](../supported_ops.md#23-attention).

## Usage

Install Primus-Turbo as in the [Quick Start](../../README.md#-quick-start). Needs a process
group; run under `torchrun`.

Dense BF16, Ulysses 4 × ring 2 on 8 GPUs:

```python
import os
import torch
import primus_turbo.pytorch as turbo
from torch.distributed.device_mesh import init_device_mesh

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("ring", "ulysses"))

B, S, H, D = 4, 4096, 256, 128
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)

o = turbo.ops.flash_attn_usp_func(
    q, k, v,
    softmax_scale=D ** (-0.5),
    causal=True,
    ulysses_group=mesh["ulysses"].get_group(),
    ring_group=mesh["ring"].get_group(),
)
```

Ulysses only (no ring): pass `ring_group=None`, or a process group of size 1.

Module form:

```python
attn = turbo.modules.TurboAttention(
    causal=True,
    ulysses_group=mesh["ulysses"].get_group(),
    ring_group=mesh["ring"].get_group(),
)
o = attn(q, k, v)
```

Varlen and FP8 use the same Ulysses groups; see
[Examples § Attention with Context Parallel](../examples.md#attention-with-context-parallel).

## Configuration

### Groups

- `ulysses_group`: the Ulysses all-to-all group. Its size must divide both the query-head
  count and the KV-head count (GQA / MQA included).
- `ring_group`: optional. Dense BF16 / FP16 can combine Ulysses with ring attention. FP8 and
  varlen require it unset or of size 1.

### Layouts

Dense ops accept `bshd` (default), `sbhd` and `bhsd`. The layout is inferred from strides, or
passed as `qkv_format`. Each rank's sequence length is the local shard; the full sequence is
`local_S * ulysses_group.size()`.

Varlen takes packed `[t_local, h, d]` with the **global** `cu_seqlens` / `max_seqlen` of the
full sequence (`t = t_local * cp_size`).

### FP8

`flash_attn_fp8_usp_func` takes an optional `fp8_config`. Only blockwise granularity with
`block_size=64` is supported (that is also the default).

### Environment

| Variable | Role |
|---|---|
| `PRIMUS_TURBO_ATTN_BACKEND` | Backend of the underlying dense / varlen flash attention (same key as the non-CP ops). See [env.md](../env.md#backend-selection). |
| `PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32` | Whether the attention V3 backward accumulates with FP32 atomics. Default on; set to `"0"` to disable. See [env.md](../env.md#kernel-behavior). |

## Performance

No published end-to-end numbers for the USP attention ops themselves. The related fused
Ulysses GEMM + all-to-all ops (`ulysses_qkv_gemm_a2a`, `ulysses_a2a_o_gemm`) remove the pack /
collective / unpack around the QKV and output projections on MI355X; see
[#624](https://github.com/AMD-AGI/Primus-Turbo/pull/624) and the benches below.

Benchmarks: `benchmark/ops/training/bench_ulysses_a2a.py`,
`benchmark/ops/training/bench_ulysses_overlap.py`.

## End-to-end impact

Primus-LM's Megatron backend selects the USP ops when `use_turbo_attention` is on and
`context_parallel_size > 1` (`PrimusTurboAttention` → `flash_attn_usp_func` or
`flash_attn_fp8_usp_func`). Today that path passes the CP group as `ulysses_group` and a
singleton as `ring_group` (Ulysses only; ring is not wired yet). Sink attention is turned off
when CP > 1, because the USP ops do not take a sink parameter.

## Known limitations

- Not supported on gfx1250 (MI455X). Dense non-CP flash attention and MLA are; USP is not.
- `dropout_p` must be 0; `bias` must be `None`.
- Sink attention is not available on the USP path.
- FP8 USP and varlen USP are Ulysses only: `ring_group` unset or size 1.
- FP8 USP: blockwise `block_size=64` only.
- Ulysses degree must divide both `num_head_q` and `num_head_kv`.
- Varlen: BF16, packed THD inputs, global `cu_seqlens`.

## Further reading

- Examples: [§ Attention with Context Parallel](../examples.md#attention-with-context-parallel)
- Related: Ulysses GEMM + all-to-all fusion (`ulysses_qkv_gemm_a2a`, `ulysses_a2a_o_gemm`),
  [#624](https://github.com/AMD-AGI/Primus-Turbo/pull/624)
- Tests: `tests/pytorch/ops/test_attention_with_cp.py`,
  `tests/pytorch/ops/test_attention_varlen_with_cp.py`
- DeepSpeed Ulysses: https://arxiv.org/abs/2309.14509
