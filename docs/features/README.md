# Feature Index

The maturity level of each Primus-Turbo feature, and its Feature Card. The levels and what each one
requires are in [Feature Maturity Levels](./maturity.md). To add a card, copy
[the template](./_template.md).

A level holds on the architectures its row lists. On any other architecture the feature supports
(see [Supported Operators](../supported_ops.md)), it is Experimental. Features not listed here are
Experimental.

*Provisional* levels come from how a feature is used today; its evidence and Feature Card are due by
the next minor release (see [Initial levels](./maturity.md#initial-levels)).

| Feature | Level | Architectures | Feature Card |
|---|---|---|---|
| GEMM, BF16 / FP16 | Stable, provisional | gfx942, gfx950 | not yet |
| GEMM, FP8 tensorwise / rowwise / blockwise | Stable, provisional | gfx942, gfx950 | not yet |
| Grouped GEMM, BF16 / FP16 | Stable, provisional | gfx942, gfx950 | not yet |
| Grouped GEMM, FP8 tensorwise / rowwise / blockwise | Stable, provisional | gfx942, gfx950 | not yet |
| Flash Attention, BF16 / FP16 (`flash_attn_func`, `flash_attn_varlen_func`) | Stable, provisional | gfx942, gfx950 | not yet |
| GEMM and grouped GEMM, MXFP8 | Beta, provisional | gfx950 | not yet |
| Grouped MLP, FP8 tensorwise / MXFP8 (`grouped_mlp_fp8`) | Beta, provisional | gfx950 | not yet |
| DeepEP (`moe_dispatch` / `moe_combine`, `deep_ep.Buffer`) | Beta | gfx942, gfx950 | [DeepEP](./deep_ep.md) |
| MegaMoE, BF16 | Beta | gfx950, gfx1250 | [MegaMoE](./mega_moe.md) |
| MegaMoE, MXFP8 | Experimental | gfx950 | [MegaMoE](./mega_moe.md) |
| MXFP4 GEMM (`gemm_fp4`) | Beta | gfx950 | [MXFP4 training](./mxfp4.md) |
| MXFP4 grouped GEMM and MLPs (`grouped_gemm_fp4`, `mlp_fp4`, `grouped_mlp_fp4`) | Experimental | gfx950, gfx1250 | [MXFP4 training](./mxfp4.md) |
| Async-TP (`fused_all_gather_matmul`, `fused_matmul_reduce_scatter`) | Experimental | gfx942, gfx950 | [Async-TP](./async_tp.md) |
| MoE gather fused into MXFP4 quantization | Experimental | gfx950 | [design doc](../moe_gather_quantization.md) |
| JAX (`primus_turbo.jax`) | Experimental | gfx942, gfx950 | not yet |
