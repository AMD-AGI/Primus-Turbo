# Ulysses GEMM + all-to-all fusion

## Summary

Ulysses context parallelism runs an all-to-all on each side of attention: after the QKV projection,
to trade the sequence split for a head split, and before the output projection, to trade it back.
Done one step after another, each exchange is a GEMM, a pack copy, an RCCL all-to-all and an unpack
copy, and the backward does the same again. On Qwen3-30B-A3B with CP 4, about 70% of the time in
these four places is the exchange itself.

Two ops fuse each projection GEMM with its all-to-all, forward and backward:

- `ulysses_qkv_gemm_a2a` (GemmAll2All): the QKV GEMM writes each finished tile to a staging buffer
  in symmetric memory, and peer ranks read the tile from there, straight into the q / k / v layout
  that attention takes. Its backward is an All2AllGemm that computes dx and dW_qkv in one kernel.
- `ulysses_a2a_o_gemm` (All2AllGemm): each rank sends rows of the attention output, in place, into
  the peers' receive buffers, and the output-projection GEMM starts on each tile as its rows
  arrive. The residual is added in fp32 before the single bf16 rounding. Its backward is a
  GemmAll2All that computes dO and dW_o in one kernel.

There is no pack, collective, unpack, global barrier or host sync. Synchronization is per tile, so
the GEMMs overlap the transfers over XGMI. Results are bitwise reproducible.

## Support matrix

| | gfx942 (MI300X / MI325X) | gfx950 (MI350X / MI355X) | gfx1250 (MI455X) |
|---|:-:|:-:|:-:|
| `ulysses_qkv_gemm_a2a`, `ulysses_a2a_o_gemm`, forward and backward | — | ✅ | — |

| | Supported |
|---|---|
| Precision | BF16 |
| Ulysses group | 2, 4 or 8 ranks on one node |
| Layout | `sbhd`, `bshd` |
| Head dim | 64, 128 |
| Heads | GQA and MHA; query and KV head counts divisible by the group size |
| QKV weight | Megatron's interleaved layout or `[q; k; v]` |

Frameworks: PyTorch.

## Usage

Install Primus-Turbo as in the [Quick Start](../../README.md#-quick-start).

The two ops replace the projections around attention. Attention in between is the plain
single-GPU `flash_attn_func`, since every rank holds the full sequence for its heads:

```python
import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.ops.attention.ulysses_fused_a2a import (
    ulysses_a2a_o_gemm,
    ulysses_qkv_gemm_a2a,
)

# x: this rank's chunk of the sequence, [b, s/n, hidden], bf16
q, k, v = ulysses_qkv_gemm_a2a(
    x, w_qkv, num_q_heads, num_kv_heads, head_dim, cp_group, qkv_format="bshd"
)
# q, k, v: [b, s, heads/n, head_dim], this rank's heads over the full sequence.
# Apply q / k norm and RoPE here, with positions over the full sequence.
o = turbo.ops.flash_attn_func(q, k, v, causal=True)
y = ulysses_a2a_o_gemm(o, w_o, residual, cp_group, qkv_format="bshd")
# y = o @ w_o^T + residual, [b, s/n, hidden]
```

- `cp_group` is the Ulysses process group, `n` ranks on one node.
- Rank `r` holds the `r`-th contiguous chunk of the sequence, as with `flash_attn_usp_func`. A
  framework that splits the sequence for load balance, as Megatron does by default for context
  parallelism, has to switch to the contiguous split.
- `w_qkv` is `[(num_q_heads + 2 * num_kv_heads) * head_dim, hidden]`. `weight_layout="interleaved"`
  (the default) is Megatron's row order, `weight_layout="concat"` is `[q; k; v]`.
- `w_o` is `[hidden, num_q_heads * head_dim]`; `residual` can be `None`.

## Configuration

`fwd_tuning` and `bwd_tuning` take a `GemmA2ATuning` or an `A2AGemmTuning` (same module) and override
the launch parameters of one kernel, such as the tile rows and the number of blocks that move data.
Fields left at `None` keep the shape heuristics.

The symmetric-memory buffers are two banks per group, sized for the larger of the QKV and output
exchanges, and are allocated on the first call: 96 MiB for Llama 3.1 8B at 32K with 8 ranks.

## Performance

From [#624](https://github.com/AMD-AGI/Primus-Turbo/pull/624): one node of 8 × MI355X, BF16, no
CUDA graph.

Kernels, Qwen3-30B-A3B, 4 ranks: the four fused places of one layer, unfused → fused, in µs.

| Sequence | Forward QKV | Forward output projection | Backward output projection | Backward QKV | Sum |
|---|---|---|---|---|---|
| 16K | 439 → 256 | 319 → 236 | 386 → 240 | 531 → 302 | 1,675 → 1,034 (1.62×) |
| 32K | 829 → 471 | 579 → 410 | 714 → 412 | 981 → 514 | 3,103 → 1,807 (1.72×) |
| 64K | 1,573 → 867 | 1,112 → 731 | 1,341 → 781 | 1,901 → 951 | 5,927 → 3,330 (1.78×) |
| 128K | 3,057 → 1,821 | 2,163 → 1,391 | 2,552 → 1,558 | 3,717 → 1,828 | 11,489 → 6,598 (1.74×) |

At 64K and 128K the fused kernels run within 1.00–1.13× of the roofline: the larger of the GEMM time
at 1,242 TFLOP/s and the off-rank bytes at 145 GB/s.

Benchmarks (`torchrun --nproc_per_node <n>`):

```bash
# each fused op against its unfused steps, with the ideal overlap and the roofline
torchrun --nproc_per_node 4 benchmark/ops/training/bench_ulysses_overlap.py --model qwen3-30b-a3b --seq 32768
# a whole layer (attention + dense MLP), forward + backward, unfused vs fused
torchrun --nproc_per_node 8 benchmark/ops/training/bench_ulysses_a2a.py --model llama3.1-8b --seq 32768 --fused
```

## End-to-end impact

Megatron pretraining through a Primus-LM patch that is still in review
([Primus #1416](https://github.com/AMD-AGI/Primus/pull/1416), opt-in with
`turbo_ulysses_fused_a2a: 1`). Qwen3-30B-A3B, all 48 layers, TP 1 / CP 4 / EP 8 / DP 2, micro-batch 1,
global batch 16, mock data, median of steps 4–12, from
[#624](https://github.com/AMD-AGI/Primus-Turbo/pull/624):

| Sequence | Base ms/iter | Fused ms/iter | Change |
|---|---|---|---|
| 16K | 3,157 | 2,815 | −10.8% |
| 32K | 6,581 | 5,979 | −9.2% |
| 64K | 17,350 | 16,004 | −7.8% |
| 128K | 55,942 | 53,005 | −5.3% |

The relative gain falls with length because attention grows quadratically, but the time saved per
step grows, from 342 ms to 2,937 ms. The step-12 loss is within 2e-3 of the base run, and memory is
unchanged.

Part of this gain is not the fusion. The base run calls `flash_attn_usp_func`, which pins the AITER
attention kernel, while the fused path calls `flash_attn_func`, which picks the FlyDSL kernel on
this shape. On the other models below, GLM-4.7 and MiniMax-M2.5 run AITER attention on both sides,
and the Llama 3.1 8B run pins it on both, so their gains are the fusion alone.

| Model | CP / EP | 16K | 32K |
|---|---|---|---|
| Qwen3-235B-A22B, 4-layer proxy | 4 / 8 | −5.9% | −6.6% |
| GLM-4.7, 4-layer proxy | 4 / 8 | −6.7% | −6.2% |
| MiniMax-M2.5, 4-layer proxy | 4 / 8 | −4.8% | −5.2% |
| Llama 3.1 8B, all 32 layers | 4 / 1 | −4.3% | −3.6% |
| GLM-4.7, 4-layer proxy | 8 / 8 | −1.9% | −3.7% |

The 4-layer proxies understate the full-model gain, because the embedding, output layer and
optimizer weigh more in them.

## Known limitations

- gfx950 only, and the Ulysses group must sit inside one node: the ops exchange through XGMI peer
  memory.
- BF16 only, and the inputs' dtype is not checked yet: other dtypes give wrong results without an
  error.
- Not supported: FP8, MLA, biases on the projections, gradient accumulation into `main_grad` (the
  weight gradients come back as plain bf16 `.grad`), and CUDA graphs (not tested).
- `weight_layout="concat"` caches a reordered copy of the weight, matched by the weight's address
  and `_version`. An update that does not bump `_version`, such as a write through `.data`, leaves
  the cache stale. The default interleaved layout has no cache.
- Shape limits: the sequence length must divide by the group size; with a residual, `hidden` must
  be a multiple of 256; each GEMM operand must have fewer than 2^31 elements. When a dimension is not
  a multiple of 256, the weight gradient runs after the kernel instead of inside it.
- The GEMM tile is fixed at 256 × 256 with no autotune, about 15% slower than hipBLASLt on the same
  GEMM alone. With 8 ranks the launch parameters are not tuned yet, so the gain is smaller than
  with 4.

## Further reading

- Design and full results: [#624](https://github.com/AMD-AGI/Primus-Turbo/pull/624)
- Primus-LM integration: [Primus #1416](https://github.com/AMD-AGI/Primus/pull/1416)
- Tests: `tests/pytorch/ops/test_ulysses_fused_a2a.py`
  (`torchrun --nproc_per_node 8 tests/pytorch/ops/test_ulysses_fused_a2a.py`)
- Benchmarks: `benchmark/ops/training/bench_ulysses_overlap.py`,
  `benchmark/ops/training/bench_ulysses_a2a.py`
