# Async tensor parallelism (Async-TP)

> **Maturity:** Experimental ([what this means](./maturity.md))
>
> **Since:** v0.1.0

## Summary

Async-TP overlaps the communication of tensor parallelism with the GEMM next to it. The
all-gather before a column-parallel GEMM and the reduce-scatter after a row-parallel GEMM are split
into chunks, and the GEMM on one chunk runs while the next chunk is in flight. The ops cover the
forward only; the training integration supplies the backward.

## Support matrix

| | gfx942 (MI300X / MI325X) | gfx950 (MI350X / MI355X) | gfx1250 (MI455X) |
|---|:-:|:-:|:-:|
| `fused_all_gather_matmul` | ✅ | ✅ | — |
| `fused_all_gather_scaled_matmul` (FP8, `torch._scaled_mm`) | ✅ | ✅ | — |
| `fused_matmul_reduce_scatter` | ✅ | ✅ | — |

Frameworks: PyTorch. See [Supported Operators § Communication](../supported_ops.md#27-communication-and-overlap).

## Usage

```python
import primus_turbo.pytorch as turbo

# A = all_gather(A_shard, dim=0) over the TP group; outputs[i] = A @ Bs[i]
A, outputs = turbo.ops.fused_all_gather_matmul(
    A_shard, [B], ["NN"], 0, tp_group.group_name,
    gemm_streams=[torch.cuda.current_stream()],
    comm_streams=[torch.cuda.Stream() for _ in range(tp_group.size() - 1)],
    copy_streams=[torch.cuda.Stream()],
)

# C = reduce_scatter(A @ B, "sum", dim=0) over the TP group
C = turbo.ops.fused_matmul_reduce_scatter(
    A, B, "NN", "sum", 0, tp_group.group_name,
    gemm_streams=[torch.cuda.current_stream()],
    comm_streams=[torch.cuda.Stream() for _ in range(tp_group.size())],
)
```

## Configuration

Keyword arguments, no environment variables:

- `comm_method`: `"pipeline"` (the default) splits the work into `num_splits` chunks (default 2).
  All-gather also takes `"ring_exchange"` and `"auto"`; reduce-scatter also takes `"tile"`.
- `enable_sdma=True` does the peer copies on the SDMA engines instead of CUs, leaving the CUs to
  the GEMM.
- Layouts: `NN`, `NT` and `TN` for all-gather; `NN` and `NT` for reduce-scatter.

## Performance

No published numbers and no benchmark in `benchmark/` yet.

## End-to-end impact

Integrated in Primus as the TransformerEngine communication-overlap backend and as a TorchTitan
async-TP patch. Measured training-step gain not recorded yet. Required for Stable.

## Known limitations

- Forward only.
- Ranks exchange data through PyTorch symmetric memory, so the TP group must be within one node.
- Reduce-scatter with `comm_method="tile"` needs M, N and K divisible by 256, and M / TP size of at
  least 256.
- The stream lists are passed in by the caller, and the right number depends on `comm_method` and
  the TP size; see the docstrings in `primus_turbo/pytorch/ops/async_tp.py`.

## Further reading

- Tests: `tests/pytorch/ops/test_fused_all_gather_matmul.py`, `test_fused_matmul_reduce_scatter.py`
