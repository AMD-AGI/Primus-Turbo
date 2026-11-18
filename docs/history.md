# Primus-Turbo History

Every What's New entry, grouped by the release it first shipped in, newest first. The [README](../README.md#-whats-new) shows the latest release and the latest seven entries; each new entry goes both there and under "Unreleased" here, and at a release that section takes the release's name. An entry is a new feature or a capability a new model needs; optimizations, wider shape or dtype coverage, fixes and dependency bumps are left to the [CHANGELOG](../CHANGELOG.md) and the release notes, which list every merged pull request.

## Unreleased (main)

- **[2026/11/18]** 🔥 **MXFP8 training on MI455X (gfx1250)** — `gemm_fp8`, `grouped_gemm_fp8`, `grouped_mlp_fp8` and the staged MXFP8 MegaMoE run on FlyDSL MXFP8 kernels on gfx1250 (E4M3, BF16 activations), with the weight gradients added straight into a bf16 or fp32 `main_grad` ([Supported Operators](./supported_ops.md)).

## v0.7.0

[v0.7.0](https://github.com/AMD-AGI/Primus-Turbo/releases/tag/v0.7.0), 2026/11/18 · [changes since v0.6.0](https://github.com/AMD-AGI/Primus-Turbo/compare/v0.6.0...v0.7.0) · [CHANGELOG](../CHANGELOG.md#v070--20261118)

- **[2026/11/17]** 🔥 **Backend choice report and strict dispatch** — every operator dispatch records which backend it picked and why (user, autotune, default or fallback); `PRIMUS_TURBO_DISPATCH_REPORT=<path>` writes the per-shape report as JSON at exit, and `PRIMUS_TURBO_STRICT_DISPATCH=1` turns a fallback into an error ([#646](https://github.com/AMD-AGI/Primus-Turbo/pull/646), [env](./env.md#dispatch-diagnostics)).
- **[2026/11/10–11/17]** 🔥 **Gradient-accumulation fusion on MI455X (gfx1250)** — expert wgrads added straight into Megatron's `main_grad`, with no separate dW buffer or add: MegaMoE's BF16 staged ops into a bf16 `main_grad` ([#632](https://github.com/AMD-AGI/Primus-Turbo/pull/632), [MegaMoE](./features/mega_moe.md)), and the BF16 grouped GEMM into an fp32 `main_grad` ([#647](https://github.com/AMD-AGI/Primus-Turbo/pull/647)).
- **[2026/11/09]** 🔥 **Ulysses GEMM + all-to-all fusion on MI355X (gfx950)** — `ulysses_qkv_gemm_a2a` and `ulysses_a2a_o_gemm` fuse the QKV projection with the Ulysses all-to-all and the all-to-all with the output projection, forward and backward: GEMM tiles go straight to their peers in the post-all-to-all layout, with no pack, collective or unpack ([#624](https://github.com/AMD-AGI/Primus-Turbo/pull/624)).
- **[2026/11/02]** 🔥 **DeepEP on MI455X (gfx1250)** — intranode dispatch / combine on MI455X: the V1 kernels ported to gfx1250, and a new V2 that writes every token straight into a fixed slot on its destination, the version MI455X runs. EP4 training on 4× MI455X is 1.08–1.10× faster than Megatron's alltoall dispatcher on a DeepSeek-V3 proxy and 1.04–1.07× on GPT-OSS 20B, with 12–16 GB less memory per GPU ([#583](https://github.com/AMD-AGI/Primus-Turbo/pull/583), [DeepEP](./features/deep_ep.md), [design](./design/deep_ep_v2_mi455.md)).
- **[2026/10/28]** 🔥 **MegaMoE on MI455X (gfx1250)** — BF16 fused MegaMoE forward and backward on gfx1250, including the staged op with per-expert biases for GPT-OSS shapes; validated on 4× MI455X (EP4) ([#582](https://github.com/AMD-AGI/Primus-Turbo/pull/582), [MegaMoE](./features/mega_moe.md)).
- **[2026/10/19–10/27]** 🔥 **Attention on MI455X (gfx1250)** — FlyDSL MLA attention for DeepSeek-V3 (qk 192 / v 128) ([#567](https://github.com/AMD-AGI/Primus-Turbo/pull/567)); FlyDSL flash attention for head dim 128 (Llama-3.1-8B) and sparse-MLA for DeepSeek-V4 ([#580](https://github.com/AMD-AGI/Primus-Turbo/pull/580)).
- **[2026/10/20–10/21]** 🔥 **GPT-OSS in MegaMoE** — hidden / intermediate 2880 and per-expert fc1 / fc2 biases in the BF16 and MXFP8 MegaMoE ops ([#571](https://github.com/AMD-AGI/Primus-Turbo/pull/571)), with the per-layer memory sized to the rows dispatched to each rank ([#566](https://github.com/AMD-AGI/Primus-Turbo/pull/566), [#568](https://github.com/AMD-AGI/Primus-Turbo/pull/568), [MegaMoE](./features/mega_moe.md)).

## v0.6.x

[v0.6.0](https://github.com/AMD-AGI/Primus-Turbo/releases/tag/v0.6.0) and [v0.6.1](https://github.com/AMD-AGI/Primus-Turbo/releases/tag/v0.6.1) (v0.6.0 plus a test fix), both 2026/10/21 · [changes since v0.5.0](https://github.com/AMD-AGI/Primus-Turbo/compare/v0.5.0...v0.6.1)

- **[2026/10/08]** 🔥 **MiniMax-M3 in MegaMoE** — MiniMax-M3's SwiGLU-OAI activation in the BF16 and MXFP8 MegaMoE kernels ([#538](https://github.com/AMD-AGI/Primus-Turbo/pull/538)).
- **[2026/09/21]** 🔥 **Grouped GEMM on MI455X (gfx1250)** — FlyDSL BF16 grouped GEMM for MoE, the first grouped GEMM on gfx1250 ([#508](https://github.com/AMD-AGI/Primus-Turbo/pull/508)).
- **[2026/09/17]** 🔥 **`mlp_fp4` and `rmsnorm_residual_fp4`** — MXFP4 dense MLP with SwiGLU fused into its GEMMs, and RMSNorm + residual in the MXFP4 path ([#509](https://github.com/AMD-AGI/Primus-Turbo/pull/509)).

## v0.5.0

[v0.5.0](https://github.com/AMD-AGI/Primus-Turbo/releases/tag/v0.5.0), 2026/09/16 · [changes since v0.4.0](https://github.com/AMD-AGI/Primus-Turbo/compare/v0.4.0...v0.5.0)

- **[2026/08/25–09/14]** 🔥 **Fused GroupedMLP (FP8 / MXFP8 / MXFP4)** — expert MLP with SwiGLU folded into grouped GEMM; pad-aware paths for GPT-OSS-20B-style MoE ([#476](https://github.com/AMD-AGI/Primus-Turbo/pull/476), [#488](https://github.com/AMD-AGI/Primus-Turbo/pull/488), [#503](https://github.com/AMD-AGI/Primus-Turbo/pull/503)); `grouped_mlp_fp4` for MXFP4 ([#483](https://github.com/AMD-AGI/Primus-Turbo/pull/483)).
- **[2026/08/18]** 🔥 **MegaMoE (MXFP8)** — fused forward and backward in the MXFP8 MegaMoE path: dispatch+FC1 and FC2+combine with intra-node EP ([#456](https://github.com/AMD-AGI/Primus-Turbo/pull/456)).

## v0.4.0

[v0.4.0](https://github.com/AMD-AGI/Primus-Turbo/releases/tag/v0.4.0), 2026/07/30 · [changes since v0.3.1](https://github.com/AMD-AGI/Primus-Turbo/compare/v0.3.1...v0.4.0)

- **[2026/07/27]** 🔥 **MXFP4 GEMM & grouped GEMM** — FlyDSL and hipBLASLt stacks for dense and expert GEMMs in low-precision training ([#424](https://github.com/AMD-AGI/Primus-Turbo/pull/424)).
- **[2026/07/15]** 🔥 **MegaMoE (BF16)** — FlyDSL fused MoE layer with comm–compute overlap inside the grouped GEMMs ([MegaMoE](./features/mega_moe.md)) ([#412](https://github.com/AMD-AGI/Primus-Turbo/pull/412)).

## v0.3.x

[v0.3.0](https://github.com/AMD-AGI/Primus-Turbo/releases/tag/v0.3.0), 2026/04/13, and [v0.3.1](https://github.com/AMD-AGI/Primus-Turbo/releases/tag/v0.3.1), 2026/06/12 · [changes since v0.2.0](https://github.com/AMD-AGI/Primus-Turbo/compare/v0.2.0...v0.3.1)

- **[2026/05/27]** 🔥 **DeepEP** — JAX intranode/internode token dispatch ([#344](https://github.com/AMD-AGI/Primus-Turbo/pull/344)); PyTorch `DeepEPTokenDispatcher` (introduced in [#114](https://github.com/AMD-AGI/Primus-Turbo/pull/114)). The rocSHMEM communication backend followed in v0.4.0 ([#409](https://github.com/AMD-AGI/Primus-Turbo/pull/409)) ([DeepEP](./features/deep_ep.md)).
- **[2026/04/17–05/11]** 🔥 **Attention layouts and varlen** — `sbhd` / `bhsd` layouts and AITER `flash_attn_varlen_func` ([#275](https://github.com/AMD-AGI/Primus-Turbo/pull/275), [#304](https://github.com/AMD-AGI/Primus-Turbo/pull/304), [#332](https://github.com/AMD-AGI/Primus-Turbo/pull/332)).
- **[2026/01/26–04/10]** 🔥 **Sink attention** — attention sinks with sliding window, as GPT-OSS uses ([#208](https://github.com/AMD-AGI/Primus-Turbo/pull/208), [#262](https://github.com/AMD-AGI/Primus-Turbo/pull/262)).
- **[2025/12/15–2026/01/05]** 🔥 **MXFP4 GEMM** — MXFP4 GEMM and MXFP8 / MXFP4 quantization with fused transpose and Hadamard transform ([#179](https://github.com/AMD-AGI/Primus-Turbo/pull/179), [#195](https://github.com/AMD-AGI/Primus-Turbo/pull/195)).
- **[2025/12/04]** 🔥 **Backend management and auto-tune** — selectable kernel backends per operator and an auto-tune infrastructure ([#167](https://github.com/AMD-AGI/Primus-Turbo/pull/167)).
- **[2025/12/01]** 🔥 **JAX DeepEP** — single-node `moe_dispatch` / `moe_combine` on JAX ([#162](https://github.com/AMD-AGI/Primus-Turbo/pull/162)).

## v0.2.0

[v0.2.0](https://github.com/AMD-AGI/Primus-Turbo/releases/tag/v0.2.0), 2025/12/04 · [changes since v0.1.1](https://github.com/AMD-AGI/Primus-Turbo/compare/v0.1.1...v0.2.0)

- **[2025/11/24]** 🔥 **JAX grouped GEMM** ([#157](https://github.com/AMD-AGI/Primus-Turbo/pull/157)).
- **[2025/11/19]** 🔥 **MXFP8 GEMM** — MXFP8 GEMM and quantization ([#154](https://github.com/AMD-AGI/Primus-Turbo/pull/154)).
- **[2025/10/14]** 🔥 **`DeepEPTokenDispatcher`** — a DeepEP token dispatcher for MoE layers ([#114](https://github.com/AMD-AGI/Primus-Turbo/pull/114)).

## v0.1.x

[v0.1.0](https://github.com/AMD-AGI/Primus-Turbo/releases/tag/v0.1.0), the initial release, 2025/09/11, and [v0.1.1](https://github.com/AMD-AGI/Primus-Turbo/releases/tag/v0.1.1), 2025/10/15 · [changes since v0.1.0](https://github.com/AMD-AGI/Primus-Turbo/compare/v0.1.0...v0.1.1)

- **[2025/09/22]** 🔥 **ROCm 7** support ([#107](https://github.com/AMD-AGI/Primus-Turbo/pull/107)).
- **[2025/09/15]** 🔥 **DeepEP internode** — DeepEP dispatch / combine across nodes ([#94](https://github.com/AMD-AGI/Primus-Turbo/pull/94)).

## Blogs

- **[2025/12/16]** 🔥 [MoE training best practices on AMD GPUs](https://rocm.blogs.amd.com/software-tools-optimization/primus-moe-package/README.html) — DeepEP, grouped GEMM, and Primus recipes end to end.
- **[2025/12/01]** 🔥 [Efficient MoE Pre-training at Scale on 1K AMD GPUs with TorchTitan](https://pytorch.org/blog/efficient-moe-pre-training-at-scale-with-torchtitan/).
- **[2025/09/19]** [Primus-Turbo introduction blog](https://rocm.blogs.amd.com/software-tools-optimization/primus-large-models/README.html).
