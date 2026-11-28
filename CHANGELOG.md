# Changelog

Changes in each Primus-Turbo release, newest first; what has merged since the last release is under
"Unreleased". The headline features of every release are in
[docs/history.md](./docs/history.md); each release's GitHub page lists every merged pull request.

## Unreleased (main)

[changes since v0.7.0](https://github.com/AMD-AGI/Primus-Turbo/compare/v0.7.0...main)

### Highlights

- **Collectives fused into GEMMs and attention on MI355X (gfx950).**
  - `fused_collective_gemm`: GemmA2A, A2AGemm, AgGemm and GemmRS run their all-to-all, all-gather or
    reduce-scatter in the GEMM launch ([#669](https://github.com/AMD-AGI/Primus-Turbo/pull/669), [#671](https://github.com/AMD-AGI/Primus-Turbo/pull/671), [#673](https://github.com/AMD-AGI/Primus-Turbo/pull/673), [#674](https://github.com/AMD-AGI/Primus-Turbo/pull/674); guide [#676](https://github.com/AMD-AGI/Primus-Turbo/pull/676)).
  - The Ulysses ops run on that package: QKV / o_proj for GQA ([#677](https://github.com/AMD-AGI/Primus-Turbo/pull/677)) and MLA through the latents,
    `ulysses_mla_latent_up_proj` ([#678](https://github.com/AMD-AGI/Primus-Turbo/pull/678)).
  - `ulysses_mega_attn`: the Ulysses all-to-alls and causal attention in one kernel, forward and
    backward, GQA and MLA ([#708](https://github.com/AMD-AGI/Primus-Turbo/pull/708)).
  - `flash_attn_agkv_func`: all-gather-KV context parallelism with the exchange inside the attention
    kernels ([#696](https://github.com/AMD-AGI/Primus-Turbo/pull/696)).
- **MLA attention on MI355X (gfx950)**: `flash_attn_func` runs the DeepSeek-V3 / Kimi-K2 head dims
  (q / k 192, v 128) on FlyDSL kernels, forward and backward, with `softmax_scale` ([#707](https://github.com/AMD-AGI/Primus-Turbo/pull/707)).
- **Fused DiT block ops for FLUX (gfx950)**: `adaln_modulate`, `gated_residual_adaln`, `linear_gelu`
  ([#649](https://github.com/AMD-AGI/Primus-Turbo/pull/649)) and `dit_qk_norm_rope` ([#652](https://github.com/AMD-AGI/Primus-Turbo/pull/652)).
- **JAX grouped GEMM backends for MoE**: hipBLASLt and FlyDSL, under `jit` / `shard_map` ([#638](https://github.com/AMD-AGI/Primus-Turbo/pull/638)).

### New features

- Tensorwise `gemm_fp8` is traced by `torch.compile` ([#662](https://github.com/AMD-AGI/Primus-Turbo/pull/662)).
- `Float8QuantConfig(default_backend=...)`: an opt-in backend for the dense FP8 GEMMs of one config
  ([#663](https://github.com/AMD-AGI/Primus-Turbo/pull/663)).
- `quantize_fp8(..., scale=, amax_out=)`: the tensorwise FP8 cast with a given scale also reports its
  input's amax ([#698](https://github.com/AMD-AGI/Primus-Turbo/pull/698)).
- `fused_qkv_rmsnorm_rope` runs on gfx942 ([#642](https://github.com/AMD-AGI/Primus-Turbo/pull/642)).
- Docs: the [MXFP4 training guide](./docs/features/mxfp4.md) ([#666](https://github.com/AMD-AGI/Primus-Turbo/pull/666)) and the
  [fused collective GEMM guide](./docs/features/fused_collective_gemm.md) ([#676](https://github.com/AMD-AGI/Primus-Turbo/pull/676)).

### Performance

- BF16 grouped GEMM on MI455X: retuned Muse Spark keys with nearest-key matching, and keys for
  EP16 / EP32 ([#685](https://github.com/AMD-AGI/Primus-Turbo/pull/685)).

### Bug fixes

- Works with FlyDSL 0.5.0, which newer AITER requires ([#653](https://github.com/AMD-AGI/Primus-Turbo/pull/653)).
- The Ulysses ops rebuild their rearranged projection weights on every call: an optimizer
  can update a weight in place without changing the version the cache was keyed on ([#679](https://github.com/AMD-AGI/Primus-Turbo/pull/679)).
- The CK grouped GEMM tile no longer spills registers on gfx942, and the JAX FP8 grouped GEMM works
  under `jit` ([#638](https://github.com/AMD-AGI/Primus-Turbo/pull/638)).
- The PyTorch unit tests pass on gfx950, including MXFP4 `shuffle_weight` / `dequantize_mxfp4` on
  inputs with a padded row pitch ([#659](https://github.com/AMD-AGI/Primus-Turbo/pull/659)).
- Lookup tables read by gfx1250 FlyDSL launches are read-only, so they no longer change the launch
  cache key ([#650](https://github.com/AMD-AGI/Primus-Turbo/pull/650)).

## v0.7.0 — 2026/11/18

[Release](https://github.com/AMD-AGI/Primus-Turbo/releases/tag/v0.7.0) ·
[changes since v0.6.0](https://github.com/AMD-AGI/Primus-Turbo/compare/v0.6.0...v0.7.0)

### Highlights

- **MoE training on MI455X (gfx1250).**
  - BF16 MegaMoE forward and backward, fused and staged ([#582](https://github.com/AMD-AGI/Primus-Turbo/pull/582)).
  - DeepEP intranode dispatch / combine: the V1 kernels ported to gfx1250 and a new V2 that writes
    every token straight into a fixed slot on its destination ([#583](https://github.com/AMD-AGI/Primus-Turbo/pull/583),
    [#648](https://github.com/AMD-AGI/Primus-Turbo/pull/648)).
  - MLA attention for DeepSeek-V3 ([#567](https://github.com/AMD-AGI/Primus-Turbo/pull/567), rewritten in
    [#647](https://github.com/AMD-AGI/Primus-Turbo/pull/647)) with its backward ([#632](https://github.com/AMD-AGI/Primus-Turbo/pull/632)),
    flash attention for head dim 128 and sparse-MLA for DeepSeek-V4 ([#580](https://github.com/AMD-AGI/Primus-Turbo/pull/580)).
  - Expert wgrads accumulated straight into Megatron's `main_grad`: MegaMoE's BF16 staged ops into a
    bf16 `main_grad` ([#632](https://github.com/AMD-AGI/Primus-Turbo/pull/632)), the BF16 grouped GEMM
    into an fp32 `main_grad` ([#647](https://github.com/AMD-AGI/Primus-Turbo/pull/647)).
- **GPT-OSS in MegaMoE**: hidden / intermediate 2880 and per-expert fc1 / fc2 biases, BF16 and MXFP8
  ([#571](https://github.com/AMD-AGI/Primus-Turbo/pull/571)).
- **Ulysses GEMM + all-to-all fusion on MI355X (gfx950)**: `ulysses_qkv_gemm_a2a` and
  `ulysses_a2a_o_gemm` ([#624](https://github.com/AMD-AGI/Primus-Turbo/pull/624)).
- **Backend choice report and strict dispatch**: `PRIMUS_TURBO_DISPATCH_REPORT=<path>` and
  `PRIMUS_TURBO_STRICT_DISPATCH=1` ([#646](https://github.com/AMD-AGI/Primus-Turbo/pull/646)).

### New features

- MXFP4 GEMM: the Llama2 LoRA adapter folded into the epilogue ([#609](https://github.com/AMD-AGI/Primus-Turbo/pull/609)).
- The Flux MLPerf kernels build on FlyDSL 0.3 ([#633](https://github.com/AMD-AGI/Primus-Turbo/pull/633)).
- One autotune engine for every FlyDSL GEMM ([#587](https://github.com/AMD-AGI/Primus-Turbo/pull/587)).
- DeepEP V2 guards against TDM operations lost while KFD evicts a queue, opt-in through
  `PRIMUS_TURBO_DEEPEP_V2_TDM_GUARD` ([#647](https://github.com/AMD-AGI/Primus-Turbo/pull/647)).
- Docs: [`docs/env.md`](./docs/env.md) generated from `primus_turbo/common/envs.py`
  ([#645](https://github.com/AMD-AGI/Primus-Turbo/pull/645)), the
  [supported operators matrix](./docs/supported_ops.md) ([#586](https://github.com/AMD-AGI/Primus-Turbo/pull/586)),
  and feature cards for [DeepEP](./docs/features/deep_ep.md) ([#648](https://github.com/AMD-AGI/Primus-Turbo/pull/648))
  and [MegaMoE](./docs/features/mega_moe.md) ([#655](https://github.com/AMD-AGI/Primus-Turbo/pull/655)).

### Performance

- MegaMoE BF16 and MXFP8 per-layer memory sized to the rows dispatched to each rank
  ([#566](https://github.com/AMD-AGI/Primus-Turbo/pull/566), [#568](https://github.com/AMD-AGI/Primus-Turbo/pull/568)).
- MXFP8 grouped GEMM training step for DeepSeek-V4-Flash on MI355X, with the wgrad accumulated into
  an fp32 `main_grad` ([#536](https://github.com/AMD-AGI/Primus-Turbo/pull/536)).
- BF16 grouped GEMM on MI455X tuned for the DeepSeek-V3 steady state
  ([#597](https://github.com/AMD-AGI/Primus-Turbo/pull/597)), and for MiniMax-M3 and Muse Spark shapes
  ([#632](https://github.com/AMD-AGI/Primus-Turbo/pull/632), [#647](https://github.com/AMD-AGI/Primus-Turbo/pull/647)).

### Bug fixes

- Three silent wrong-answer races in the FlyDSL dense GEMMs on gfx950 ([#570](https://github.com/AMD-AGI/Primus-Turbo/pull/570)).
- Bit-reproducible MXFP8 / MXFP4 / FP8 FlyDSL GEMMs, with a determinism test suite ([#603](https://github.com/AMD-AGI/Primus-Turbo/pull/603)).
- Exact split-K window divide in the FlyDSL FP8 TN GEMM ([#589](https://github.com/AMD-AGI/Primus-Turbo/pull/589)).
- int32 row-offset overflow in the unmasked SwiGLU / GeGLU kernels ([#594](https://github.com/AMD-AGI/Primus-Turbo/pull/594)).
- The DEEP_EP backend of `moe_dispatch` / `moe_combine` records `previous_event` on the current
  stream, as uccl-ep needs
  ([#608](https://github.com/AMD-AGI/Primus-Turbo/pull/608)).
- DeepEP V2 on MI455X no longer hangs when a row's TDM retry never succeeds ([#647](https://github.com/AMD-AGI/Primus-Turbo/pull/647)).
- Ring attention works with newer torch ([#637](https://github.com/AMD-AGI/Primus-Turbo/pull/637)).
- The `moe_gather` test loads from the installed package when the source tree is absent ([#569](https://github.com/AMD-AGI/Primus-Turbo/pull/569)).

### Behavior changes

- Environment variables are read through `primus_turbo.common.envs` and cached at their first read
  ([#605](https://github.com/AMD-AGI/Primus-Turbo/pull/605)). Set them before the first Primus-Turbo
  call; changing `os.environ` afterwards has no effect.
- The DeepEP intranode versions are named `v1` / `v2`, and the older names raise a `ValueError`
  ([#648](https://github.com/AMD-AGI/Primus-Turbo/pull/648)). Only the test-only `intranode_backend`
  argument of `deep_ep.Buffer` takes them; the default choice per GPU is unchanged.
- Unused Triton GLU grouped GEMM kernels removed ([#606](https://github.com/AMD-AGI/Primus-Turbo/pull/606)); no public API changes.

### Compatibility

- FlyDSL 0.3.4.1 and AITER `v0.1.24.post1` ([#544](https://github.com/AMD-AGI/Primus-Turbo/pull/544)).
- Recommended image `rocm/primus:v26.8` ([#572](https://github.com/AMD-AGI/Primus-Turbo/pull/572)).

### Known issues

- DeepEP V2 on MI455X runs within one host, up to 8 ranks, and is tested at EP2 / EP4; see the
  [DeepEP card](./docs/features/deep_ep.md).
- MegaMoE runs within one host and cannot be captured in a CUDA graph, and MXFP8 MegaMoE is MI355X only; see
  [MegaMoE § Known limitations](./docs/features/mega_moe.md#known-limitations).

## Earlier releases

See the [GitHub Releases](https://github.com/AMD-AGI/Primus-Turbo/releases) and
[docs/history.md](./docs/history.md) for v0.6.1 and earlier.
