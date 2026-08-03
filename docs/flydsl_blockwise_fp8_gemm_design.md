# FlyDSL Blockwise FP8 GEMM Design and Tuning Record

## Scope

This document records the gfx950 blockwise FP8 GEMM work completed for the
Primus-Turbo FlyDSL backend. It covers:

- row-major forward GEMM without a weight preshuffle,
- NN dgrad and TN wgrad,
- four-wave and eight-wave kernel geometries,
- partial-N and output buffers larger than 4 GiB,
- measured tuning results,
- the implemented shape-to-kernel dispatch policy,
- the deprecation status of the legacy preshuffle kernel.

The supported arithmetic is:

```text
input       FP8 E4M3
scale       FP32 inverse scale
scale K     128 elements
accumulator FP32
output      BF16 or FP16
target      gfx950 / CDNA4
```

Every K128 partial is scaled before it is accumulated. Unscaled partials from
different scale blocks must never be combined.

## Tensor and scale semantics

### Forward NT

```text
C[M,N] = A[M,K] @ B[N,K].T
A scale: [M, ceil(K/128)]       (1x128)
B scale: [ceil(N/128), K/128]   (128x128)
```

For scale block `q`:

```text
C[m,n] +=
    dot(A[m, 128q:128q+128], B[n, 128q:128q+128])
    * A_scale[m,q]
    * B_scale[n//128,q]
```

### Dgrad NN

```text
dA[M,K] = dY[M,N] @ W[N,K]
dY scale: [M, ceil(N/128)]
W scale:  [ceil(N/128), K/128]
```

The physical NN implementation transposes the FP8 weight into an N-major
workspace with the tiled FP8 transpose kernel.

### Wgrad TN

```text
dW[N,K] = dY[M,N].T @ A[M,K]
dY column scale: [ceil(M/128), N]
A column scale:  [ceil(M/128), K]
```

Dual-layout quantization emits transposed contiguous storage. The normalized TN
kernel consumes those views without an extra transpose dispatch.

## Production source layout

The current implementation is split across:

```text
primus_turbo/flydsl/gemm/blockscale_fp8_gemm/
  __init__.py
  dispatch.py
  four_wave_blockwise_fp8_gemm_kernel.py
  eight_wave_blockwise_fp8_gemm_kernel.py
  utils.py
primus_turbo/flydsl/gemm/__init__.py
primus_turbo/pytorch/ops/gemm_fp8.py
```

## Four-wave row-major kernel

### Geometry

The four-wave kernel uses:

```text
threads             256
waves               4
BLOCK_N             128
BLOCK_K             128
BLOCK_M             128 or 192
MFMA                16x16x128
wave ownership      2M x 2N
```

Each wave computes four disjoint output regions. A and B both use row-major
XOR-swizzled LDS. Current and next K tiles are double-buffered.

LDS usage:

```text
BM128  64 KiB
BM192  80 KiB
```

Both geometries permit two resident workgroups per 160 KiB CU.

### Main optimizations retained

- partial `vmcnt` waits,
- G2S/S2R operations interleaved with MFMA,
- grouped K128 partial folding,
- short `s_setprio` regions,
- scalar B-scale loads for 128x128 weight scales,
- XCD-aware workgroup remapping,
- static M/N specialization,
- runtime SCF loops for deep K.

### BM128 versus BM192

BM192 is effective when its additional accumulator state is amortized by a
large, compute-dense shape. It is not a universal replacement for BM128.

For `4096x202048x5120`:

```text
BM128 tuned  5.154 ms  1644 TFLOPS
BM192        6.580 ms  1288 TFLOPS
```

For `8192x202048x5120`:

```text
BM128 tuned  10.435 ms  1624 TFLOPS
BM192        12.659 ms  1339 TFLOPS
```

The M-tail waste is only 3.0% and 0.8%, respectively. The larger loss comes from
resource pressure and reduced issue density:

```text
metric          BM128 tuned   BM192
MFMA util       33.3%         27.6%
occupancy       24.0%         23.8%
VGPR            104           128
LDS             64 KiB        80 KiB
```

### BM256 four-wave experiment

BM256 requires 96 KiB LDS, so only one four-wave workgroup can reside on a CU.

For `4096x202048x5120`:

```text
BM256 best   5.780 ms  1466 TFLOPS
BM128 best   5.154 ms  1644 TFLOPS
```

BM256 is therefore not retained in the four-wave kernel.

## Partial-N support

The original Primus route rejected any N that was not divisible by 128. Shapes
such as `N=202048` therefore used the legacy preshuffle kernel even though the
last tile contains only a 64-column tail.

The row-major kernel already had:

- descriptor-based OOB-zero B loads,
- `ceil(N/128)` B-scale rows,
- predicated output stores.

The public route now permits `N % 16 == 0`. For partial-N shapes it selects:

```text
BLOCK_M           128
k_loop_unroll     4
fold_group_size   6
GROUP_M           4
partial waits     enabled
```

Measured result for `4096x202048x5120`:

```text
legacy preshuffle   9.12 ms
four-wave initial   6.12 ms
four-wave tuned     5.94 ms
Triton              5.85 ms
```

For `8192x202048x5120`:

```text
FlyDSL tuned  11.12 ms
Triton        12.11 ms
```

Opening N-tail support moved four suite cases from legacy preshuffle to the
row-major four-wave path.

## Eight-wave, three-stage kernel

### Geometry and LDS

The independent proof of concept uses:

```text
BLOCK_M          256
BLOCK_N          128
BLOCK_K          128
threads          512
waves            8
wave ownership   4M x 2N
output/wave      64 x 64
MFMA/wave/K128   16
```

Each stage contains:

```text
A  256 x 128 FP8 = 32 KiB
B  128 x 128 FP8 = 16 KiB
```

Three stages consume 144 KiB. One workgroup resides per CU; its eight waves
provide two waves per SIMD.

### Three-stage pipeline

The prologue loads tile 0 into stage 0 and tile 1 into stage 1. Stage 2 starts
empty.

The runtime loop executes three statically rotated steps per SCF iteration:

```text
(current, next, future)
(stage0, stage1, stage2)
(stage1, stage2, stage0)
(stage2, stage0, stage1)
```

The best schedule is:

```text
issue future-stage G2S
scheduler fence
waitcnt vmcnt(6)
workgroup barrier
scheduler fence
issue next-stage S2R
compute current-stage MFMA
```

`vmcnt(6)` drains the six older G2S operations while leaving the six newly
issued future-stage operations in flight.

Current deep-K defaults:

```text
fold_group_size     5
interleave_width    2
wait_delay_thunks   8
pipeline_wait_count 6
interleave_mode     g2s_wait6_s2r
```

### Scheduling evolution

For `16384x16384x53248`:

```text
four-wave tuned                  18.41 ms  1553 TFLOPS
eight-wave initial               18.43 ms  1551 TFLOPS
G2S-first                        17.60 ms  1624 TFLOPS
interleaved vmcnt(6)             16.87 ms  1695 TFLOPS
fold/interleave tuning           16.23 ms  1762 TFLOPS
scheduler-fenced final           15.97 ms  1790 TFLOPS
Triton GEMM                      16.68 ms  ~1714 TFLOPS
```

Final PMC:

```text
MFMA util       35.8%
occupancy       24.5%
VALU busy       21.3%
LDS util        17.9%
VGPR            128
LDS             144 KiB
scratch         32 B
```

The initial eight-wave kernel had approximately 30.8% MFMA utilization. The
fenced three-stage schedule increased useful MFMA issue density by about five
percentage points.

### Shape dependence

The eight-wave kernel is not a universal replacement:

```text
shape                    4-wave TFLOPS  8-wave TFLOPS  8w/4w
8192x16384x53248          1554            1733          1.115
16384x16384x53248         1553            1790          1.153
32768x16384x53248         1527            1745          1.143
8192x106496x16384         1569            2047          1.305
8192x8192x29568           1584            1572          0.992
4096x202048x5120          1644            1564          0.951
49152x12288x4096          2162            1884          0.871
```

The dispatch must select both geometry and schedule parameters. In particular,
K-stage remainders and the efficient BM192 large-M path materially change the
winner.

## Buffers larger than 4 GiB

### Failure mechanism

AMDGPU buffer instructions use a 32-bit vector offset. Flattening a BF16 output
larger than 4 GiB also creates a host tensor dimension that can reach or exceed
`2^31`, which does not fit the C ABI shape field used by the JIT adapter.

### Specialized resource-base design

The new four-wave and eight-wave kernels automatically use a per-M-tile output
resource when `M*N*2 > 0xffffffff`:

```text
tile_row_origin   = tile_m * BLOCK_M
base_byte_offset  = tile_row_origin * N * sizeof(C)
records_bytes     = min(total_bytes - base_byte_offset, 0xffffffff)
local_offset      = row_in_tile * N + column
```

`base_byte_offset` is computed with 64-bit index arithmetic. The residual
element offset is tile-local and remains far below 4 GiB.

Large outputs must be passed to the JIT as two-dimensional tensors. Flattening a
4 GiB BF16 tensor produces a length of `2^31` and fails before kernel launch.

Static range proof:

```text
0 <= row_in_tile < BLOCK_M
0 <= column < N
local_offset <= BLOCK_M*N - 1
```

For the tested shapes, `BLOCK_M*N*sizeof(C)` is below the descriptor limit.

The same design is used for FP8 A and B inputs:

```text
A base = tile_m * BLOCK_M * K
B base = tile_n * BLOCK_N * K
```

The local buffer views are limited to `BLOCK_M*K` and `BLOCK_N*K`; global
dimensions never enter a 32-bit descriptor offset. This enables dgrad
contraction tensors above 4 GiB without changing the MFMA pipeline.

### Boundary validation

The exact 4 GiB shape `16384x131072` was filled by both kernels with K=128 and
unit FP8 inputs. Samples were checked at:

```text
rows    0, 127, 128, 191, 192, 255, 256, 16383
columns 0, 63, 64, 127, 128, 131071
```

All 48 samples from each kernel matched the analytical result.

### Large-output performance

Aligned deep-K case:

```text
shape                  24576x106496x16384
output                 4.875 GiB
four-wave GEMM         50.91 ms  1685 TFLOPS
eight-wave GEMM        40.98 ms  2093 TFLOPS
FlyDSL full forward    42.80 ms  2004 TFLOPS
Triton full forward    54.99 ms  1559 TFLOPS
FlyDSL/Triton          1.285x
```

Partial-N ultra-wide case:

```text
shape                  16384x202048x5120
output                 6.166 GiB
four-wave GEMM         20.51 ms  1653 TFLOPS
eight-wave GEMM        21.16 ms  1602 TFLOPS
FlyDSL full forward    21.30 ms  1591 TFLOPS
Triton full forward    23.38 ms  1450 TFLOPS
FlyDSL/Triton          1.097x
```

The large-output path is therefore viable and outperforms Triton in both tested
classes. Geometry remains shape-dependent.

## Final full-suite results

The final no-preshuffle suite produced 175 rows per backend. Both backends pass
174 rows and report OOM for TestID 145. On the 174 common valid rows:

```text
forward geometric-mean speedup   1.028x
backward geometric-mean speedup  1.285x
step geometric-mean speedup      1.204x
forward wins                     129 / 174
backward wins                    174 / 174
step wins                        174 / 174
```

Average throughput:

```text
stage       Triton TFLOPS  FlyDSL TFLOPS
forward     1385.27         1425.10
backward    1021.14         1312.01
```

Artifacts:

```text
benchmark/ops/training/output_dispatch_20260731/
benchmark/ops/training/output_dispatch_largeio_20260731/
gemm_fp8_blockwise_triton_benchmark.csv
gemm_fp8_blockwise_flydsl_benchmark.csv
```

Compared with the first integrated result, valid coverage increased from 159 to
174 and FlyDSL errors decreased from 16 to the same single OOM case as Triton.

## Implemented dispatch design

```text
condition                                      kernel             parameters
N % 128 != 0                                  4-wave BM128       row-scale, fold6, unroll4
aligned, K == 29568                            4-wave BM128       K-major scale, fold6, unroll6
aligned, prefer_8wave, K >= 32768             8-wave 3-stage    row-scale, fold5, iw2, delay8
aligned, prefer_8wave, K < 32768              8-wave 3-stage    K-major scale, K/group registry
aligned, !prefer_8wave, M % 192 == 0           4-wave BM192      K-major scale, fold4, unroll2
aligned, !prefer_8wave, M % 192 != 0           4-wave BM128      K-major scale, fold4, unroll2
```

The measured eight-wave predicate is:

```text
M >= 4096
K >= 3584
(K/128) % 3 != 0
and one of:
    K >= 32768
    N >= 65536 and K >= 8192
    M <= 32768 and M % 192 != 0
    M <= 8192 and N <= 32768
    K == 4096 and N == 28672 and M % 192 != 0
```

Suite dispatch counts:

```text
4-wave BM128       20 total / 19 valid
4-wave BM192       12 total / 12 valid
8-wave 3-stage    143 total / 143 valid
```

Per-family geometric-mean speedups:

```text
family             forward   backward   step
4-wave BM128       0.995x    1.259x     1.175x
4-wave BM192       1.086x    1.371x     1.281x
8-wave 3-stage     1.028x    1.281x     1.202x
```

## Preshuffle deprecation status

The public gfx950 blockwise dispatcher and autograd producer no longer import or
generate preshuffled operands:

- forward consumes plain row-major FP8 weight data,
- dgrad transposes plain weight data with the tiled FP8 transpose,
- wgrad consumes dual-quantized transposed views,
- large A/B/C tensors use per-tile 64-bit resource bases.

The public Primus-Turbo FlyDSL FP8 backend already rejects gfx942, so gfx942 is
not affected by this gfx950 migration.

## Forward +10% target analysis

On the final 174 common valid suite rows:

```text
forward speedup >= 1.10   9
forward speedup <  1.10   165

ratio range         count
< 0.95                  5
0.95 to 1.00           40
1.00 to 1.05           77
1.05 to 1.10           43
>= 1.10                 9
```

The 45 forward losses have a 0.973x geometric mean. Raising all losses to
parity would move the overall forward geometric mean from 1.028x to 1.036x,
and the training-step geometric mean from 1.204x to 1.207x. Requiring every
shape to exceed 1.10 is therefore a different target from maximizing aggregate
training throughput.

PMC samples from the main regression classes:

```text
shape class          kernel   MFMA util  occupancy  VGPR  LDS
K3584 N37888         4-wave      35.4%      24.1%    104   64 KiB
K3584 N37888         8-wave      31.5%      23.2%    128  144 KiB
K29568 N8192         4-wave      33.3%      24.5%    100   64 KiB
K4096 N28672         8-wave      35.0%      24.1%    128  144 KiB
```

Cross-tuning `fold_group_size`, K-loop unroll, interleave order, wait depth,
MFMA priority, GROUP_M, and 4w/8w geometry did not produce a stable end-to-end
1.10 ratio for these classes. Constant-input GEMM-only measurements
over-predicted gains relative to the complete quantize+GEMM path.

Likely structural work:

- alias existing LDS for a C-shuffle/vectorized BF16 epilogue on shallow K,
- add a narrow-N or BN256 geometry for K3584/K5120 classes,
- specialize the 231-K-block (`K=29568`) pipeline tail,
- investigate quantization/GEMM fusion for short forward kernels,
- use ATT traces to reduce barrier/VMEM stalls and raise MFMA issue density,
- validate strict per-shape targets with repeated medians, since duplicate
  shapes show up to 6.7% run-to-run ratio variation.

### Follow-up structural PoCs

The structural candidates above were evaluated with alternating repeated
measurements before changing production dispatch.

Scale loop-carried prefetch was rejected. Carrying two compact scale sets
increased the SCF state from 24 to 34 values and regressed four representative
shapes by 7.7% to 21.3%. A one-set pipeline reduced the state to 29 values but
still regressed by 5.9% to 20.1%. The kernel already allocates 256 VGPRs; the
deep-K private segment increased from 28 bytes to 140 bytes.

Row-major A-scale LDS staging was evaluated separately. The correct PoC used
three 2 KiB FP32 stages and `BufferCopyLDS32b` global-to-LDS DMA, adding 6 KiB
without changing the two-wave-per-SIMD occupancy. It passed all 174 valid suite
rows, but 127 of 142 eight-wave forward rows regressed. Their geometric-mean
speedup was 0.96466x and total eight-wave forward time increased by 4.59%.
ATT showed that buffer-load instructions decreased from 161 to 56, while
barrier stall increased from about 13% to 27% and LDS plus LGKM wait stall
increased from 3.6% to 11.5%. No K or shape group met both a 1.01x geometric
mean and a maximum 1% regression, so the staging path was removed.

The synchronization scan retained one deep-K-only change. Keeping
`pipeline_wait_count=6` and moving `wait_delay_thunks` from 2 to 8 gives the
future G2S requests more MFMA issue distance before the partial wait. Complete
forward medians from three complete runs at K53248 improved by 1.68%, 1.15%,
1.36%, and 1.40% at M8192, M16384, M24576, and M32768. The same delay regressed
a K4096 sample by 2.95%, so shallow-K dispatch remains at delay 0.

An LDS-aliased C-shuffle was also rejected. It reused an input stage and emitted
128-bit BF16 stores without increasing LDS allocation. The extra LDS writes,
reads, and synchronization regressed the K3584/K4096 samples by 0.7% to 3.2%.
Row padding of 4 or 8 elements increased the regression to as much as 6.6%.

The `BM128 x BN256 x BK128` eight-wave geometry was bitwise correct, but was
1.4% to 7.2% slower than `BM256 x BN128`. Increasing `GROUP_M` from 4 to 16
recovered most of the K3584 loss but did not beat the existing geometry.
Fold/interleave tuning did not change the result.

The K29568 specialization was retained. `K / 128 = 231`; using six statically
unrolled stages per SCF iteration with `fold_group_size=6` reduces the runtime
loop iterations while preserving the one-step plus two-step static tail. It
improved GEMM-only time by 6.1% at `8192x8192x29568` and 8.2% at
`32768x8192x29568`. Complete forward time improved by 5.4% and 7.4%,
respectively. The remaining gap to Triton is 1.5% and 0.5%; this specialization
does not by itself meet the 1.10 target.

The `BM256 x BN64 x BK128` eight-wave geometry was rejected. Although it reduces
three-stage LDS from 144 KiB to 120 KiB, it halves the MFMA work per workgroup
and doubles the N-grid size. It regressed the K18944/N3584 samples by 26% to
37%.

The post-PoC 175-shape suite retained only the K29568 specialization:

```text
common PASS                         174
common ERROR                          1  (ID145, OOM in both backends)
forward geometric-mean speedup    1.0285x
backward geometric-mean speedup   1.2837x
step geometric-mean speedup       1.2032x
```

These values use the per-TestID median of three complete runs. For the three
K29568 rows, the median new/old FlyDSL geometric mean is 1.02052x. Their median
new FlyDSL/Triton geometric mean is 0.96777x, so the retained
specialization improves the implementation but does not satisfy the 1.10
per-shape target. Across all 174 rows, the run-to-run forward range has a 1.16%
p90 and 5.54% maximum. The result artifacts are:

```text
benchmark/ops/training/output_structural_poc_20260731/
benchmark/ops/training/output_structural_poc_run2_20260731/
benchmark/ops/training/output_structural_poc_run3_20260731/
benchmark/ops/training/output_structural_poc_delay8_20260731/
benchmark/ops/training/output_structural_poc_delay8_run2_20260731/
benchmark/ops/training/output_structural_poc_delay8_run3_20260731/
```

### K-major A-scale producer and final selector

The retained broad optimization changes the dual blockwise quantizer so the
forward A scales can be written directly as `[K/128, M]`. This is byte-equivalent
to transposing the original `[M, K/128]` scale tensor, but it requires no extra
kernel or workspace copy. The GEMM then replaces four strided scalar loads per
MFMA row group with one contiguous vec4 load.

The layout is enabled for aligned outputs with `K < 32768`; partial-N and
deep-K routes retain row-major scales. The K-major schedule registry uses:

```text
fold4   K=3584, 11008, 14336, 18944
fold8   K=28672
fold12  K=16384
fold8   K=4096 and N>=28672
fold8   K=5120 and N==32768
group1  K=3584 and N==37888
group2  K=4096, N==28672, M>32768
```

Explicit FlyDSL execution receives K-major scales directly from the producer.
Pre-quantized tensors, NN compatibility paths, and autotune inputs may still
arrive as `[M,K/128]`; the FlyDSL facade converts those inputs only when the
selected kernel requires K-major layout.

Three complete K-major runs plus full selector validation produced:

```text
common PASS                         174
common ERROR                          1  (ID145, OOM in both backends)
forward FlyDSL / Triton geo       1.06595x
backward FlyDSL / Triton geo      1.28504x
step FlyDSL / Triton geo          1.21764x
forward wins / losses              173 / 1
forward speedup >= 1.10             46 / 174
forward geo vs prior production    1.03697x
```

The only remaining forward loss is partial-N `4096x202048x5120` at about
0.54%; no valid step is slower than Triton. K29568 reaches a 1.11046x
geometric-mean improvement over the prior production selector.

Artifacts:

```text
benchmark/ops/training/output_kmajor_scale_poc_20260801/
benchmark/ops/training/output_kmajor_scale_run2_20260801/
benchmark/ops/training/output_kmajor_scale_run3_20260801/
benchmark/ops/training/output_kmajor_group_final_20260801/
```

## Correctness and profiling coverage

Verified cases include:

- BF16 and FP16 output,
- K=128/384/640 and runtime SCF deep-K,
- M/N tails,
- row-major and K-major A scale layouts,
- exact four-wave/eight-wave comparison,
- 100 repeated deterministic launches,
- exact 4 GiB output boundary samples,
- production outputs of 4.875 GiB and 6.166 GiB.

Focused test command:

```bash
python3 -m pytest tests/pytorch/ops/test_gemm_fp8.py -q \
  -k "8wave_3stage_poc or routes_new_kernels or blockwise_flydsl_dgrad_tail"
```

Profiling uses:

```bash
rocprofv3 --pmc MfmaUtil OccupancyPercent VALUBusy LdsUtil -- <command>
```

The benchmark comparison canvas is maintained at:

```text
~/.cursor/projects/.../canvases/blockwise-fp8-performance.canvas.tsx
```

