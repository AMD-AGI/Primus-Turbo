# Primus-Turbo Documentation

## Get started

- [Installation](../README.md#-quick-start): Docker, prebuilt wheels, building from source
- [rocSHMEM](./install_dependencies.md): the optional dependency of internode DeepEP
- [Examples](./examples.md): usage of each operator, backend selection, AutoTune

## Reference

- [Supported Operators](./supported_ops.md): every operator, its maturity level on each
  architecture, and its kernel backends
- [Feature Maturity Levels](./features/maturity.md): what Experimental, Beta and Stable promise,
  and how a feature moves between them
- [Environment variables](./env.md): every build-time and runtime variable
- [Benchmarks](../benchmark/README.md): operator and end-to-end benchmarks

## Feature Cards

One page per feature: support, usage, configuration, performance, and limitations.

- [MegaMoE](./features/mega_moe.md)
- [DeepEP](./features/deep_ep.md)
- [MXFP4 training](./features/mxfp4.md)
- [Async-TP](./features/async_tp.md)

To add one, copy [the template](./features/_template.md).

## Design

- [MegaMoE](./design/mega_moe.md)
- [FlyDSL DeepEP on MI455X](./design/flydsl_deepep_mi455.md)
- [MoE gather and MXFP4 quantization fusion](./design/moe_gather_quantization.md)
- [Single-rank fused cross entropy](./design/cross_entropy.md)
