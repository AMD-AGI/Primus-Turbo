# FlyDSL Blockwise FP8 GEMM for MI355X

## Summary

- Replace the public gfx950 blockwise FP8 GEMM path with row-major 4-wave and
  8-wave/3-stage FlyDSL kernels.
- Route forward GEMMs by shape, K depth, A-scale layout, fold schedule, and XCD
  grouping.
- Keep backward on direction-specific 4-wave NN/TN kernels using tiled transpose
  workspaces or zero-copy transpose views.
- Emit K-major forward A-scales directly from the blockwise quantizer where that
  layout is profitable.
- Support partial N, BF16/FP16 output, deterministic execution, and A/B/C buffers
  above 4 GiB.
- Remove the deprecated v1 kernel, its Triton transpose helper, and unused
  pre-shuffled quantization APIs.
- Move the implementation into
  `primus_turbo/flydsl/gemm/blockscale_fp8_gemm/` and consolidate shared data
  movement helpers with `primus_turbo/flydsl/utils/gemm_helper.py`.

## Motivation

The original v1 implementation depended on pre-shuffled operands and duplicated
kernel, host-dispatch, layout, and tuning logic. It also left unsupported or
slow paths for partial-N and large-buffer shapes.

The replacement uses plain row-major FP8 operands, keeps blockwise scaling
semantics explicit for every K128 partial, and selects measured kernel geometry
and schedules from M/N/K.

## Public behavior

The public PyTorch API remains `gemm_fp8`. The internal FlyDSL facade keeps the
same forward, dgrad, and wgrad entry points.

The removed direct-import modules are:

```text
primus_turbo.flydsl.gemm.gemm_fp8_blockwise_kernel
primus_turbo.triton.gemm.preshuffle_fp8
```

The old direct-import compatibility surface is intentionally removed. Public
backend dispatch does not use those modules.

## Implementation layout

```text
primus_turbo/flydsl/gemm/blockscale_fp8_gemm/
├── __init__.py
├── dispatch.py
├── four_wave_blockwise_fp8_gemm_kernel.py
├── eight_wave_blockwise_fp8_gemm_kernel.py
└── utils.py
```

- `dispatch.py`: support checks, measured forward selector, compile cache, and
  host launchers.
- `four_wave_blockwise_fp8_gemm_kernel.py`: 4-wave NT/NN/TN compilers.
- `eight_wave_blockwise_fp8_gemm_kernel.py`: 8-wave/3-stage forward compiler.
- `utils.py`: blockscale fold scheduling, XCD mapping, state helpers, large
  buffer views, and tiled FP8 transpose.
- `gemm_helper.py`: shared G2S/S2R loaders, swizzle, packing, and barrier
  primitives reused by dense and grouped GEMMs.

## Tensor and scale semantics

For each K128 block `q`:

```text
C[m,n] += dot(A[m, 128q:128q+128], B[n, 128q:128q+128])
          * A_scale[m,q]
          * B_scale[n//128,q]
```

Arithmetic:

```text
input       FP8 E4M3
scale       FP32 inverse scale
accumulator FP32
output      BF16 or FP16
target      gfx950 / CDNA4
```

## Forward routing

| Condition | Kernel | A-scale layout | Schedule | Purpose |
|---|---|---|---|---|
| `N % 128 != 0` | 4-wave BM128 | row-major | fold6, unroll4 | partial N |
| `K == 29568` | 4-wave BM128 | K-major | fold6, unroll6 | 231 K-block tail |
| preferred 8-wave, `K >= 32768` | 8-wave/3-stage | row-major | fold5, iw2, delay8 | deep K |
| preferred 8-wave, `K < 32768` | 8-wave/3-stage | K-major | K/fold/group registry | main route |
| aligned, `M % 192 == 0` | 4-wave BM192 | K-major | fold4, unroll2 | compute-dense M |
| aligned fallback | 4-wave BM128 | K-major | fold4, unroll2 | general fallback |

The 8-wave preference requires:

```text
M >= 4096
K >= 3584
(K / 128) % 3 != 0
and one of:
  K >= 32768
  N >= 65536 and K >= 8192
  M <= 32768 and M % 192 != 0
  M <= 8192 and N <= 32768
  K == 4096 and N == 28672 and M % 192 != 0
```

Measured K-major scheduling:

```text
fold4   K=3584, 11008, 14336, 18944
fold8   K=28672
fold12  K=16384
fold8   K=4096 and N>=28672
fold8   K=5120 and N==32768
group1  K=3584 and N==37888
group2  K=4096, N==28672, M>32768
```

## Backward routing

| Direction | Equation | Kernel | Layout handling | Output |
|---|---|---|---|---|
| dgrad NN | `dY[M,N] @ W[N,K]` | 4-wave BM128/192 | tiled weight transpose workspace | `dX[M,K]` |
| wgrad TN production | `dY.T[N,M] @ A[M,K]` | 4-wave BM128/192 | dual-quantized transpose views | `dW[N,K]` |
| wgrad TN fallback | `dY.T[N,M] @ A[M,K]` | 4-wave BM128/192 | two tiled transpose workspaces | `dW[N,K]` |

Backward does not use the 8-wave kernel. The 4-wave implementation supports the
required NN/TN scale conventions while retaining two-workgroup LDS residency.

## Kernel characteristics

| Kernel | Threads / waves | BM × BN × BK | LDS / workgroup | Residency | Main use |
|---|---:|---:|---:|---:|---|
| 4-wave BM128 | 256 / 4 | 128 × 128 × 128 | 64 KiB | 2 WG/CU | partial N, shallow K, fallback |
| 4-wave BM192 | 256 / 4 | 192 × 128 × 128 | 80 KiB | 2 WG/CU | aligned compute-dense M |
| 8-wave/3-stage | 512 / 8 | 256 × 128 × 128 | 144 KiB | 1 WG/CU | main forward, wide N, deep K |

## Shared optimization techniques

- Per-K128 FP32 scale fold.
- 128-bit buffer-to-LDS DMA for FP8 operands.
- XOR-swizzled row-major LDS tiles.
- Double or triple buffered K pipelines.
- Partial `vmcnt` waits with one workgroup barrier per stage.
- G2S and S2R operations interleaved with grouped MFMA folds.
- Short `s_setprio` regions around MFMA issue.
- XCD-aware workgroup remapping.
- Static M/N specialization and predicated tails.
- Per-tile 64-bit resource bases for inputs and outputs above 4 GiB.
- Shape/config keyed compiled-function cache.

## K-major A-scale producer

The dual blockwise quantizer can write row scales directly as
`[K/128, M]`. This is byte-equivalent to a contiguous transpose of the original
`[M, K/128]` scales, but requires no extra allocation or kernel launch.

The forward GEMM then replaces four strided scalar scale loads with one
contiguous vec4 load. Explicit FlyDSL execution receives this layout directly.
Autotune, pre-quantized tensors, and compatibility paths can still supply
row-major scales; the FlyDSL facade converts them only when required.

## Large-buffer support

AMDGPU buffer instructions use 32-bit residual offsets. For A/B/C buffers above
4 GiB, each workgroup folds its large tile offset into a 64-bit resource base:

```text
base_byte_offset = tile_origin * leading_dimension * element_size
local_offset     = row_within_tile * leading_dimension + column
```

The residual offset remains tile-local and within the descriptor limit.

## Optimization history

| Stage | Retained change | Forward geo | Backward geo | Step geo |
|---|---|---:|---:|---:|
| unified 4-wave | plain row-major operands and unified public route | 1.028× | 1.285× | 1.204× |
| 8-wave + large I/O | deep-K route and 64-bit resource bases | 1.028× | 1.285× | 1.204× |
| K-major A-scale | producer-native scale layout | 1.056× | 1.283× | 1.213× |
| K-specific fold | measured fold4/8/12 registry | 1.064× | 1.285× | 1.217× |
| group/geometry specials | K3584 group1 and large-M K4096 8-wave | 1.066× | 1.285× | 1.218× |

The largest broad forward gain came from producer-native K-major A-scales.

### Explored but not retained

| PoC | Result | Limiting mechanism |
|---|---:|---|
| loop-carried scale prefetch | -5.9% to -21% | VGPR pressure and scratch spills |
| row-major scale in LDS | 8-wave geo 0.965× | barrier and LGKM stalls exceeded VMEM savings |
| LDS C-shuffle | -0.7% to -6.6% | extra LDS traffic exceeded store savings |
| BM128 × BN256 | -1.4% to -7.2% | A-scale duplication and weaker B reuse |
| BM256 × BN64 | -26% to -37% | half the MFMA work per workgroup |
| 32 × 32 × 64 MFMA | -21% to -30% | long dependency chain and resource pressure |
| K-major B-scale | -0.9% to -4.9% | lost contiguous K traversal for scalar loads |

## Performance

MI355X, 175-shape training suite, 174 common valid rows:

| Metric | FlyDSL / Triton |
|---|---:|
| forward geometric mean | 1.066× |
| backward geometric mean | 1.285× |
| forward + backward step | 1.218× |
| forward wins | 173 / 174 |
| backward wins | 174 / 174 |
| step wins | 174 / 174 |
| forward shapes at or above 1.10× | 46 / 174 |

The sole forward loss is TestID 110, `4096×202048×5120`, at approximately
0.995×. Its full training step remains faster than Triton.

Selected cases:

| Test | Shape | Route | Triton | FlyDSL | Speedup |
|---:|---:|---|---:|---:|---:|
| 110 | 4096×202048×5120 | 4-wave partial-N | 6.30 ms | 6.34 ms | 0.995× |
| 133 | 16384×37888×3584 | 8-wave K-major group1 | 2.94 ms | 2.85 ms | 1.029× |
| 63 | 65536×28672×4096 | 8-wave K-major group2 | 9.95 ms | 9.69 ms | 1.028× |
| 154 | 16384×8192×29568 | 4-wave K-major unroll6 | 5.43 ms | 5.02 ms | 1.082× |
| 74 | 16384×16384×53248 | 8-wave row-major delay8 | 18.82 ms | 18.28 ms | 1.030× |

Both backends report the same OOM for TestID 145, whose requested allocation is
approximately 185.62 GiB.

## Correctness and validation

- BF16 and FP16 output.
- NT forward, NN dgrad, and TN wgrad.
- Row-major and K-major A-scale layouts.
- Partial M/N and dgrad contraction tails.
- Runtime deep-K SCF loops.
- Bitwise 4-wave/8-wave comparisons.
- 100-launch determinism checks.
- Exact 4 GiB boundary samples.
- 4.875 GiB and 6.166 GiB production outputs.

Validation results:

```text
blockscale regression matrix: 117 passed, 32 skipped
full benchmark suite:         174 passed, 1 common OOM
Black / Ruff / py_compile:    passed
```

Refactor validation on a second MI355X node showed no systematic performance
regression; the measured center shifted approximately +1.5% for both forward
and backward, consistent with a node-level offset.

## Test plan

- [x] Run the complete blockscale correctness matrix with disk cache disabled.
- [x] Run the 175-shape FlyDSL benchmark in 8 shards.
- [x] Compare every common PASS row with the Triton baseline.
- [x] Validate partial-N, deep-K, K29568, and large-output routing.
- [x] Validate row-major compatibility for autotune and pre-quantized inputs.
- [x] Run Black, Ruff, syntax, IDE diagnostics, and diff checks.

## Benchmark artifacts

```text
benchmark/ops/training/output_kmajor_scale_poc_20260801/
benchmark/ops/training/output_kmajor_scale_run2_20260801/
benchmark/ops/training/output_kmajor_scale_run3_20260801/
benchmark/ops/training/output_kmajor_group_final_20260801/
benchmark/ops/training/output_refactor_20260803/
```
