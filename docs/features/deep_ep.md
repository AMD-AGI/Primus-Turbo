# DeepEP

## Summary

DeepEP moves MoE tokens between expert-parallel (EP) ranks: `dispatch` sends each token to the
ranks that hold its experts, and `combine` brings the expert outputs back and sums them per token.
A token goes to each destination GPU once and is copied per expert on the receiver, so it moves
fewer bytes than an all-to-all that sends one copy per expert. It is for training only: it has
dispatch / combine backward, and no low-latency (inference) kernels. The kernels derive from
[DeepEP](https://github.com/deepseek-ai/DeepEP).

The intranode kernels implement the data transfer of the two versions of upstream DeepEP:

- **V1**, after DeepEP V1's normal mode: the tokens are split into channels, and each channel has a
  ring on every destination rank. The sender writes a chunk and advances the ring's tail; the
  receiver copies it out into the output and advances the head. The rings have a fixed size,
  independent of the token count.
- **V2**, after DeepEP V2's direct mode (`ElasticBuffer`): a notify step computes how many tokens
  each rank receives from each rank, so every token has a fixed slot on each destination rank. The
  sender writes the token straight into its slot without waiting for the receiver; the combine
  pushes the rows back the same way and reduces them as they land. The buffer holds the whole batch,
  so it is sized by the token count and is larger than V1's rings.

Both versions take the same `Buffer` API and give bit-identical outputs. Today Primus-Turbo runs V1
on MI300X / MI325X / MI355X and V2 on MI455X; the version is picked by the GPU, not by the user.

## Support matrix

| | gfx942 (MI300X / MI325X) | gfx950 (MI350X / MI355X) | gfx1250 (MI455X) |
|---|:-:|:-:|:-:|
| Intranode (one node, 2 / 4 / 8 ranks) | ✅ V1 | ✅ V1 | ✅ V2 |
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

  # MI455X (V2): sized for 8192 tokens per rank, hidden 7168 bf16, top-8
  buffer = Buffer(group, Buffer.get_token_buffer_size_hint(8192, 7168 * 2, group.size(), 8))
  # MI300X / MI355X (V1): num_nvl_bytes from Config.get_nvl_buffer_size_hint
  buffer = Buffer(group, num_nvl_bytes)
  ```

Full example: [Examples § DeepEP](../examples.md#4-deepep).

## Configuration

### Buffer size

- V1 (MI300X / MI355X): the rings, `Config.get_nvl_buffer_size_hint(hidden_bytes, num_ranks)`.
- V2 (MI455X): the whole batch, `Buffer.get_token_buffer_size_hint(num_max_tokens_per_rank,
  hidden_bytes, num_ranks, num_topk)` (0 where V1 runs), with the same `num_max_tokens_per_rank`
  on every rank.

`turbo.ops.moe_dispatch` sizes the buffer from the group's largest first dispatch (an all-reduce at
allocation, so ranks may dispatch different numbers of tokens) and reallocates it for a larger one,
after a device sync and a barrier that waits for every rank to grow it. With a known maximum,
`set_buffer_global_config(num_max_tokens_per_rank=...)` allocates for it up front.

If a rank receives more than the buffer holds, that rank raises
`V2 dispatch receives ... > the heap's ...; raise num_nvl_bytes`, and the others stop at the
kernels' timeout (300 s).

### Number of CUs

`Buffer.set_num_sms(n)`, `set_buffer_global_config(num_use_cu=n)`, or `turbo_deepep_num_cu` in
Primus, whose default is 32 (Primus also limits it to 32 at EP >= 16, for the internode kernels).

- MI355X, EP8 (V1): 64 CUs are within 4% of 128 and 14–18% faster than 32 (see
  [Performance](#performance)).
- MI455X, EP2 / EP4 (V2): V2 saturates at 160 CUs, and 64–160 CUs train within 0.6% of each other;
  at 32 CUs its kernels take about twice as long as at 128. Use 128–160 there.

### Environment variables

See [Environment variables](../env.md):

- `PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND`: the implementation behind `moe_dispatch` /
  `moe_combine`: `TURBO` (the default) runs this DeepEP, `DEEP_EP` an installed upstream `deep_ep`
  package.
- `PRIMUS_TURBO_DEEPEP_CPU_TIMEOUT`: seconds the host waits for the other ranks before it raises
  (default 300).
- `PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM`: run the kernels on the caller's stream.
- `PRIMUS_TURBO_DEEPEP_DISABLE_CHEAP_FENCE`: use the full memory fence in the intranode channels.
- `PRIMUS_TURBO_JAX_DEEPEP_MODE`: JAX launch mode.

## Performance

Against the RCCL all-to-all of Megatron's alltoall token dispatcher, which sends a token once per
expert; DeepEP sends it once per GPU, so it moves about half the bytes at EP8. The DeepEP times
include the layout, the notify and its host wait; the all-to-all times include the all-gather of the
counts and the device-to-host copy of the splits that Megatron does before it. Neither includes the
local permute.

### V1: 8 × MI355X, EP8

DeepSeek-V3 MoE layer: hidden 7168 BF16, top-8 of 256 experts, routing limited to the top 4 of 8
expert groups, random router logits. Notify + dispatch / combine, µs:

| | 4096 tokens per rank | 8192 tokens per rank |
|---|---|---|
| DeepEP V1, 32 CUs | 929 / 912 | 1732 / 1673 |
| DeepEP V1, 64 CUs | 785 / 756 | 1426 / 1432 |
| DeepEP V1, 128 CUs | 777 / 735 | 1391 / 1384 |
| RCCL all-to-all | 1357 / 1292 | 2605 / 2542 |
| MB received per GPU, DeepEP / all-to-all | 235 / 475 | 469 / 946 |

At 64 CUs and more, DeepEP is 1.7–1.9× faster than the all-to-all; at 32 CUs, 1.4–1.5×.

- Stack: Primus-Turbo `72ae7f91`, torch 2.14.0 with ROCm 10.1, RCCL 2.30.7.
- Timing: a 512 MB flush before every iteration, the slowest rank of each iteration, median of 30
  iterations, mean of 3 rounds.
- Reproduce: `python benchmark/ops/training/bench_deepep_vs_alltoall.py --tokens 4096 8192 --num-cus 32 64 128`

### V2: 2 × MI455X, EP2

Primus's DeepSeek-V3 proxy shape: 8192 tokens per rank, hidden 7168 BF16, top-8 of 128 experts,
routing limited to the top 4 of 8 expert groups. 128 CUs. Notify + dispatch / combine, µs:

| | µs |
|---|---|
| DeepEP V2 | 183 / 150 |
| RCCL all-to-all | 652 / 603 |

Stack, timing method, and the other CU counts: [DeepEP V2 on MI455X](../design/deep_ep_v2_mi455.md#performance).

Other benchmarks: [Benchmarks § DeepEP](../../benchmark/README.md#deepep).

## End-to-end impact

Used as the MoE token dispatcher in Primus-LM's MoE recipes (`use_turbo_deepep: true`). Measured
training-step gain not recorded yet.

## Known limitations

- No low-latency kernels.
- Internode needs rocSHMEM, which is still under development; compatibility and performance on the
  bnxt, mlx5 and ionic NIC drivers are not guaranteed.
- Architectures other than gfx942, gfx950 and gfx1250 are not supported: a `Buffer` logs an error
  and raises. It also raises when the detected architecture is not that of its GPU, for example when
  `rocm_agent_enumerator` fails, which makes it report gfx942.
- MI455X: intranode only, with 2, 4 or 8 ranks and `num_nvl_bytes > 0`; other buffers raise.
- V2 (MI455X), CUDA graphs: V2 raises under capture, because every call advances host-side state
  that a replay would reuse.
- V2 (MI455X), token counts that change per step without a known maximum (packed THD sequences,
  dynamic context parallelism): the buffer holds whole batches, and a larger dispatch grows it only
  once every rank of the group grows it.

In these cases use the all-to-all (RCCL) token dispatcher; in Primus, `use_turbo_deepep: false`.

## Further reading

- Design of V2: [DeepEP V2 on MI455X](../design/deep_ep_v2_mi455.md)
- Tests: `tests/pytorch/deep_ep/`
