# Changelog

Release highlights are listed under "What's New" in the [README](README.md); this file records
changes that have not been released yet.

## Unreleased

### Added

- **FlyDSL attention for DeepSeek-V3 MLA on gfx1250 (MI455X).** `flash_attn_func` routes
  q/k head dim 192, v head dim 128, bf16, MHA, bottom-right causal attention with
  `seqlen_q % 64 == 0`, `seqlen_kv % 64 == 0` and `seqlen_q <= seqlen_kv` to new FlyDSL forward
  and backward kernels (`primus_turbo/flydsl/attention/gfx1250_mla_fwd/`, `gfx1250_mla_bwd/`).
  Megatron's SBHD tensors are read with no copy, the backward is deterministic, and the custom
  ops support `torch.compile`. The kernels need flydsl >= 0.3.4, < 0.3.5; with any other flydsl
  the call falls back to another backend. Support matrix, version requirement and
  `PRIMUS_TURBO_ATTN_BACKEND` usage: [docs/attention_backends.md](docs/attention_backends.md).
