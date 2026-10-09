# MoE gather and MXFP4 quantization fusion

This experimental, opt-in path avoids materializing the BF16 expert-ordered
activation and output-gradient tensors before grouped MXFP4 quantization.
The quantizer reads original token rows through a destination-to-source map.
Required expert output combination and input-gradient reduction remain unchanged.

## Supported configuration

The measured path is GPT-OSS on MI355X/gfx950, with TP=1, EP=1, BF16 activations,
32 experts, top-4 routing, hidden width 2880, and Turbo's fused grouped MXFP4 MLP.
Backward fusion additionally restricts the source to at most 32768 tokens, no
merging probabilities or capacity padding at unpermute, and ordinary first-order
backward outside CUDA graph capture. Unsupported backward consumers materialize
the existing gather. Tensor hooks and aliases share that materialized storage.
Changing routing/source tensors before consumption raises an error.

Both repositories need their paired gather-fusion implementation. Enable in the
training process only when the permutation output goes directly to the supported
Turbo grouped MLP/quantizer (storage-preserving EP1 identity transport is allowed):

```bash
export PRIMUS_TP=1
export PRIMUS_EP=1
export MOE_SKIP_IDENTITY_SORT=1
export GPTOSS_FUSED_PERMUTE_QUANT=1
export GPTOSS_FUSED_BACKWARD_PERMUTE_QUANT=1
```

Set the backward flag to `0` for forward-only fusion. Both feature flags are off
by default; they are not a general-purpose replacement for every permutation
consumer. The forward placeholder has no activation values until its registered
quantizer consumes the source and map. Do not apply ordinary value operations to
it. Missing or mismatched handoff metadata raises an error rather than reading
uninitialized activation storage. Primus loads the paired Turbo helpers lazily,
so disabling the flags keeps compatibility with installations without them.

## Evidence and validation limits

The original pinned MLPerf campaign measured two controls and two candidates,
with 1920 clean training steps per run: 482.948 to 477.264 ms/step, or +1.191%
throughput. Profiled throughput improved +1.387%. Forty-eight gather launches
per step disappeared, with 11.08–11.18 ms of net raw target savings. The estimated
72.48 GB/rank/step avoided intermediate reads/writes are logical tensor traffic,
not measured HBM traffic or peak-memory savings.

- [First candidate CI](https://github.com/AMD-MLPerf/mlperf-training/actions/runs/37915713696)
- [Repeated candidate CI](https://github.com/AMD-MLPerf/mlperf-training/actions/runs/37925568140)
- [Original patches and full preflights](https://github.com/AMD-MLPerf/mlperf-training/tree/990dffd0acdd24693918010636e833621dc8320f/small_llm_moe_pretraining/primus/dev/experiments/permute_quant_fusion)

Those workflows passed six forward and fourteen backward exactness cases,
including stochastic-rounding progression, defined quantized payload/scales,
BF16 main-gradient ownership, hooks, aliases and outstanding backward calls.
They used older pinned source revisions. The latest-main port and its composition
with router-backward optimization need new paired GPU correctness/profiling
runs; the historical throughput result must not be attributed to that stack.
Full training-to-target/convergence equivalence has not been established.

## Implementation and tests

`primus_turbo/pytorch/ops/moe_gather.py` owns the handoff and deferred-gradient
state. `grouped_mlp_fp4.py` consumes it in forward and backward. The dual
quantizer accepts `dest2src`, optional `permuted_probs`, and `total_m`; the FlyDSL
kernel applies the map during tile loads and preserves the destination layout,
zero-probability padding convention, and quantization recipe. Unsupported fused
recipes use an explicit gather before the existing quantizer.

CPU contract tests (no built Turbo extension required):

```bash
python tests/pytorch/ops/test_moe_gather_handoff.py -v
```

GPU payload/scale parity, including stochastic rounding, repeated indices,
zero/signed-zero probabilities, empty groups, BF16 and FP16:

```bash
pytest tests/pytorch/ops/test_moe_gather_quantization.py -v
```

The companion Primus integration test exercises the complete grouped MLP and
its autograd path.
