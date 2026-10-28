## DeepEP (experimental)

DeepEP of Primus-Turbo is in the experimental stage.
The kernel code of DeepEP is primarily derived from ROCm internal DeepEP (it's still under development). It's **only used for training** and doesn't support low-latency kernels.

### Installation

#### 1. Dependencies

Hardware
- MI300 (gfx942) and MI355 (gfx950)

#### 2. Docker (Recommended)

Use AMD ROCm image:
```
docker.io/rocm/megatron-lm:v25.5_py310
```

#### 3. Install rocSHMEM (optional)

rocSHMEM is required for internode api of experimental DeepEP. Please refer to [our rocSHMEM Installation Guide](../../../docs/install_dependencies.md) for instructions.

> **Please Note: rocSHMEM is under development, no guarantee of full compatibility and performance for bnxt,mlx5 and ionic NIC driver.**

#### 4. Install from source
Please following [Primus-Turbo Install from Source](../../../README.md#3-install-from-source) instructions to install.

### Intranode backend: HIP (C++) or FlyDSL

The intranode path (a single node, `num_ranks` in {2, 4, 8}, no low-latency mode) has these kernel
backends with the same `Buffer` API:

- `cpp` (default where the C++ DeepEP is loaded): the HIP kernels in
  `csrc/kernels/deep_ep/{layout,intranode}.cu`.
- `flydsl`: FlyDSL ports of the same kernels in `primus_turbo/flydsl/deep_ep/`
  (`get_dispatch_layout`, `notify_dispatch`, `dispatch`, `cached_notify_combine`, `combine`).
  It runs on gfx942 / gfx950 / gfx1250 and owns its own uncached HIP-IPC heap of `num_nvl_bytes`.
  On gfx1250 the C++ DeepEP is not loaded and `flydsl` is the default.
- `flydsl_v2` (gfx1250 only): FlyDSL kernels in `primus_turbo/flydsl/deep_ep/intranode_v2_kernel.py`
  that write every token straight into a fixed slot on each destination rank instead of streaming it
  through channel rings. Same outputs as `flydsl`, with handles of its own. Its heap grows with the
  token count: `moe_dispatch` sizes it from the dispatched tokens, which must be the same on every
  rank, or `set_buffer_global_config(num_max_tokens_per_rank=...)` sets it up front
  (see `docs/flydsl_deepep_mi455.md`). It cannot be captured in a CUDA graph.

Pick one per `Buffer` with `intranode_backend="cpp" | "flydsl" | "flydsl_v2"`, or globally with
`PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND=cpp|flydsl|flydsl_v2`. `cpp` and `flydsl` use the same
channel protocol, so the returned tensors and handles are bit-identical: a handle from one backend can
drive the other's cached dispatch or combine. Internode (rocSHMEM) and low-latency kernels stay on the
C++ path whatever the setting.

### Example

See [DeepEP example](../../../docs/examples.md#4-deepep)

### Benchmark Usage

 See [DeepEP benchmark](../../../benchmark/README.md#deepep)
