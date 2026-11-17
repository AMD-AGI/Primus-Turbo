# DeepEP

> **Maturity:** Beta on gfx942 / gfx950; Experimental on gfx1250 ([what this means](./maturity.md))
>
> **Since:** v0.1.0; the gfx1250 FlyDSL intranode backend is on `main`, not yet in a release

## Summary

DeepEP moves MoE tokens between expert-parallel (EP) ranks: `dispatch` sends each token to the
ranks that hold its experts, and `combine` brings the expert outputs back and sums them per token.
A token goes to each destination GPU once and is copied per expert on the receiver, so it moves
fewer bytes than an all-to-all that sends one copy per expert. It is for training only: it has
dispatch / combine backward, and no low-latency (inference) kernels. The kernels derive from ROCm's
DeepEP.

## Support matrix

| | gfx942 (MI300X / MI325X) | gfx950 (MI350X / MI355X) | gfx1250 (MI455X) |
|---|:-:|:-:|:-:|
| Intranode (one node, 2 / 4 / 8 ranks) | ✅ `cpp` | ✅ `cpp` | ✅ `flydsl_v2` |
| Internode (rocSHMEM) | ✅ | ✅ | — |

Frameworks: PyTorch, and JAX (`primus_turbo.jax.lax.moe.moe_dispatch` / `moe_combine`, intra- and
inter-node). See [Supported Operators § MoE](../supported_ops.md#24-moe).

## Usage

Install Primus-Turbo as in the [Quick Start](../../README.md#-quick-start). The internode path also
needs rocSHMEM: see [Install rocSHMEM](../install_dependencies.md).

- `turbo.ops.moe_dispatch` / `turbo.ops.moe_combine`, or the `DeepEPTokenDispatcher` module, which
  also permutes the tokens per expert. These size the buffer from the dispatched tokens.
- `primus_turbo.pytorch.deep_ep.Buffer` for direct control of the buffer:

  ```python
  from primus_turbo.pytorch.deep_ep import Buffer

  # MI455X: flydsl_v2, sized for 8192 tokens per rank, hidden 7168 bf16, top-8
  buffer = Buffer(group, Buffer.get_token_buffer_size_hint(8192, 7168 * 2, group.size(), 8))
  # MI300X / MI355X: cpp, num_nvl_bytes from Config.get_nvl_buffer_size_hint (its rings)
  buffer = Buffer(group, num_nvl_bytes)
  print(buffer.intranode_backend)
  ```

Full example: [Examples § DeepEP](../examples.md#4-deepep).

## Configuration

### Backends

The intranode path runs the kernels of the GPU's architecture, as `get_rocm_arch()` reports it.
Internode (rocSHMEM) and low-latency kernels are C++ only.

| GPU | Backend | Kernels |
|---|---|---|
| MI300X / MI325X (gfx942), MI355X (gfx950) | `cpp` | HIP V1 (channel rings): `csrc/kernels/deep_ep/{layout,intranode}.cu` |
| MI455X (gfx1250) | `flydsl_v2` | FlyDSL V2, every token written straight into a fixed slot on its destination rank, no rings: `primus_turbo/flydsl/deep_ep/intranode_v2_kernel.py` |

The lists are `CPP_DEEPEP_ARCHS` and `FLYDSL_DEEPEP_ARCHS` in `primus_turbo/flydsl/deep_ep/arch.py`.
`buffer.intranode_backend` tells which one a buffer runs. Both backends return the same tensors, but
`flydsl_v2` handles work with `flydsl_v2` only.

FlyDSL V1 (`flydsl`, the channel protocol of `cpp` on FlyDSL:
`primus_turbo/flydsl/deep_ep/intranode_v1_kernel.py`) is kept for the tests, through
`Buffer(..., intranode_backend="flydsl")`. `intranode_backend` and the `backend` argument of
`get_dispatch_config` / `get_combine_config` / `get_token_buffer_size_hint` are for the tests only;
users leave them unset. On MI455X V1 is 2.2–5.1× slower than `flydsl_v2`; on gfx942 / gfx950 it is not
tested.

`PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND` chooses the implementation behind `moe_dispatch` /
`moe_combine`: `TURBO` (the default) runs this DeepEP, `DEEP_EP` an installed upstream `deep_ep`
package.

### Buffer size

- `cpp`: the channel rings, `Config.get_nvl_buffer_size_hint(hidden_bytes, num_ranks)`.
- `flydsl_v2`: the whole batch, `Buffer.get_token_buffer_size_hint(num_max_tokens_per_rank,
  hidden_bytes, num_ranks, num_topk)` (0 for the ring backends), with the same
  `num_max_tokens_per_rank` on every rank.

`turbo.ops.moe_dispatch` sizes the buffer from the group's largest first dispatch (an all-reduce at
allocation, so ranks may dispatch different numbers of tokens) and reallocates it for a larger one,
after a device sync and a barrier that waits for every rank to grow it. With a known maximum,
`set_buffer_global_config(num_max_tokens_per_rank=...)` allocates for it up front.

If a rank receives more than the buffer holds, that rank raises
`V2 dispatch receives ... > the heap's ...; raise num_nvl_bytes`, and the others stop at the
kernels' timeout (300 s on gfx1250, as the C++ DeepEP).

### Number of CUs

`Buffer.set_num_sms(n)`, `set_buffer_global_config(num_use_cu=n)`, or `turbo_deepep_num_cu` in
Primus. On MI455X (EP2 / EP4) `flydsl_v2` saturates at 160 CUs, and 64–160 CUs train within 0.6% of
each other; at 32 CUs its kernels take about twice as long as at 128. Use 128–160 there, and set it
explicitly in Primus, whose default is 32 (Primus also limits it to 32 at EP >= 16, for the
internode kernels).

### Environment variables

See [Environment variables](../env.md):

- `PRIMUS_TURBO_DEEPEP_CPU_TIMEOUT`: seconds the host waits for the other ranks before it raises
  (default 300).
- `PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM`: run the kernels on the caller's stream.
- `PRIMUS_TURBO_DEEPEP_DISABLE_CHEAP_FENCE`: use the full memory fence in the intranode channels.
- `PRIMUS_TURBO_JAX_DEEPEP_MODE`: JAX launch mode.

## Performance

MI455X, EP2 of one node, Primus's DeepSeek-V3 proxy shape (8192 tokens per rank, hidden 7168 BF16,
top-8 of 128 experts), 128 CUs. Notify + dispatch / combine:

| | µs |
|---|---|
| `flydsl_v2` | 183 / 150 |
| RCCL all-to-all as Megatron's alltoall dispatcher runs it | 652 / 603 |

Stack, timing method, and the other CU counts: [FlyDSL DeepEP on MI455X](../design/flydsl_deepep_mi455.md#performance).
No published numbers for gfx942 / gfx950 yet. Benchmark: see [Benchmarks § DeepEP](../../benchmark/README.md#deepep).

## End-to-end impact

Used as the MoE token dispatcher in Primus-LM's MoE recipes (`use_turbo_deepep: true`). Measured
training-step gain not recorded yet. Required for Stable.

## Known limitations

- No low-latency kernels.
- Internode needs rocSHMEM, which is still under development; compatibility and performance on the
  bnxt, mlx5 and ionic NIC drivers are not guaranteed.
- On an architecture outside the two lists above, a `Buffer` logs an error and raises. It also raises
  when the reported architecture is not that of its GPU: `FLYDSL_GPU_ARCH` set to another one, or
  `rocm_agent_enumerator` failing, which makes it report gfx942.
- gfx1250: the C++ DeepEP is not loaded, so the internode and low-latency paths raise, and so does an
  intranode buffer the FlyDSL kernels do not cover (rank counts other than 2, 4 or 8,
  `num_nvl_bytes == 0`).
- gfx1250, CUDA graphs: `flydsl_v2` raises under capture, because every call advances host-side
  epochs that a replay would reuse.
- gfx1250, token counts that change per step without a known maximum (packed THD sequences, dynamic
  context parallelism): the buffer holds whole batches, and a larger dispatch grows it only once
  every rank of the group grows it.

In these cases use the all-to-all (RCCL) token dispatcher; in Primus, `use_turbo_deepep: false`.

## Further reading

- Design of the gfx1250 backend: [FlyDSL DeepEP on MI455X](../design/flydsl_deepep_mi455.md)
- Tests: `tests/pytorch/deep_ep/`
