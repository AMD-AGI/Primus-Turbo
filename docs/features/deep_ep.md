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

- `turbo.ops.moe_dispatch` / `turbo.ops.moe_combine`, or the `DeepEPTokenDispatcher` module, which
  also permutes the tokens per expert. These size the buffer from the dispatched tokens.
- `primus_turbo.pytorch.deep_ep.Buffer` for direct control of the buffer.

Example: [Examples § DeepEP](../examples.md#4-deepep). Installation, including rocSHMEM for the
internode path: [DeepEP guide](../../primus_turbo/pytorch/deep_ep/README.md).

## Configuration

- `PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND` chooses the implementation behind `moe_dispatch` /
  `moe_combine`: `TURBO` (the default) runs this DeepEP, `DEEP_EP` an installed upstream `deep_ep`
  package.
- Number of CUs: `Buffer.set_num_sms(n)` or `set_buffer_global_config(num_use_cu=n)`.
- Environment variables, see [Environment variables](../env.md):
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

Stack, timing method, and the other CU counts: [FlyDSL DeepEP on MI455X](../flydsl_deepep_mi455.md#performance).
No published numbers for gfx942 / gfx950 yet. Benchmark: see [Benchmarks § DeepEP](../../benchmark/README.md#deepep).

## End-to-end impact

Used as the MoE token dispatcher in Primus-LM's MoE recipes (`use_turbo_deepep: true`). Measured
training-step gain not recorded yet. Required for Stable.

## Known limitations

- No low-latency kernels.
- Internode needs rocSHMEM, which is still under development; compatibility and performance on the
  bnxt, mlx5 and ionic NIC drivers are not guaranteed.
- gfx1250: intranode only, and only 2, 4 or 8 ranks. `flydsl_v2` raises under CUDA Graph capture.
  Use an all-to-all (RCCL) token dispatcher for those cases.
- Token counts that change every step without a known maximum (packed THD sequences, dynamic
  context parallelism) can force buffer reallocations; pass `num_max_tokens_per_rank` to
  `set_buffer_global_config` when the maximum is known.

## Further reading

- [DeepEP guide](../../primus_turbo/pytorch/deep_ep/README.md): installation, backends, buffer sizing
- Design of the gfx1250 backend: [FlyDSL DeepEP on MI455X](../flydsl_deepep_mi455.md)
- Tests: `tests/pytorch/deep_ep/`
