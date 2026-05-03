# Round 4 — gpt_oss dA H4 fast Triton transpose (LANDED)

## Selected target

Per round-4 baseline metric:
- Lowest-progress shape: **gpt_oss-GateUP-B32-M2048** (ratio=0.773,
  weight=3, progress=0.618). Same shape as rounds 1–3.
- Round-4 starting baseline: **777** (mean ~779, GPU contention stable)
- Best-historical (per harness): 817

## Lever D step 1 — H4 transpose elimination via Triton fast kernel

Per round-2 + round-3 falsification notes, the dispatch surface for
gpt_oss is saturated; the real dA bottleneck is the H4 reroute's
`b.transpose(-2, -1).contiguous()` (~265 µs B=4 to ~2.1 ms B=32 per
dA call on the gpt_oss b shapes). Round 3 recommended kernel-side
elimination as a multi-round project. Step 1 lands here.

### Probe — PyTorch contiguous() runs at 28-36 % of HBM peak

Probe `/tmp/probe_h4_transpose_speed.py` measured PyTorch
`b.transpose(-2,-1).contiguous()` for the 4 gpt_oss BF16 shapes:

| shape          | bytes (rd+wr) | wall    | effective BW | floor 2 TB/s | slowdown |
|----------------|---------------|---------|--------------|--------------|----------|
| GateUP-B4      | 265.4 MB      | 0.310 ms| 856 GB/s     | 0.133 ms     | 2.34x    |
| GateUP-B32     | 2.12 GB       | 2.441 ms| 870 GB/s     | 1.062 ms     | 2.30x    |
| Down-B4        | 132.7 MB      | 0.152 ms| 872 GB/s     | 0.066 ms     | 2.29x    |
| Down-B32       | 1.06 GB       | 1.149 ms| 924 GB/s     | 0.531 ms     | 2.17x    |

PyTorch's generic `at::native::copy_kernel` runs at ~870 GB/s for the
post-transpose strided copy — bottlenecked at single-channel HBM
bandwidth despite MI355X having 8 channels (advertised peak 8 TB/s,
streaming-kernel achievable 2-3 TB/s).

### Solution — reuse FP8 Triton transpose template for BF16

`primus_turbo/triton/utils/fp8_transpose.py` already contained
`_fp8_transpose_3d_kernel` (round-13, used by FP8 grouped GEMM dA
H4 path). The kernel is dtype-agnostic — Triton infers element type
from the pointer; the FP8 wrapper uses `b.view(torch.uint8)` to
collapse 3 different fp8 dtypes (e4m3fn / e4m3fnuz / e5m2) onto the
same kernel template, but BF16 (always 2-byte) doesn't need the
byte-cast.

Added `bf16_transpose_3d` wrapper in same file: identical strides
and grid as `fp8_transpose_3d`, just passes BF16 tensor pointers
directly to the kernel (no `.view(uint8)`). Reuses the same
`_select_block_shape(K, N)` heuristic (K>N → BK=128 BN=256, K==N
→ BK=256 BN=128, K<N → BK=BN=128) which empirically remains optimal
on the gpt_oss BF16 corner.

Probe `/tmp/probe_bf16_triton_transpose.py` confirms:

| shape          | PyTorch wall | Triton wall | speedup | bit_eq |
|----------------|--------------|-------------|---------|--------|
| GateUP-B4      | 0.306 ms     | 0.055 ms    | 5.61x   | True   |
| GateUP-B32     | 2.433 ms     | 0.416 ms    | 5.85x   | True   |
| Down-B4        | 0.153 ms     | 0.026 ms    | 5.92x   | True   |
| Down-B32       | 1.152 ms     | 0.218 ms    | 5.30x   | True   |

Triton hits ~5 TB/s effective (~60 % of MI355X HBM peak, ~2.5x
PyTorch's effective BW). Bit-identical via `torch.equal`.

### Integration — dispatch H4 path through new helper

`GroupedGEMMHipKittenBackend.execute` H4 reroute body changed from
```python
b = b.transpose(-2, -1).contiguous()
```
to
```python
b = bf16_transpose_3d(b) if b.is_contiguous() \
    else b.transpose(-2, -1).contiguous()
```

The `is_contiguous()` guard preserves the legacy path when the
caller passes a non-contiguous `b` (Triton kernel only supports
contig source). In the metric / bench, callers always pass
contiguous `w` so the fast path always fires.

H4 reroute conditions unchanged (round-19 gate intact: only fires
when `(K_RRR % 64 != 0) OR (N_RRR % 256 != 0)`; gpt_oss-Down hits
N_RRR cond, gpt_oss-GateUP hits N_RRR cond, DSV3 / Qwen3 fall
through and run native RRR). DSV3 / Qwen3 unchanged.

## Bench results — gpt_oss bwd TFLOPS

`bench_grouped_gemm_turbo.py --dtype bf16` (100-iter
`torch.utils.benchmark`, 20 warmup, 24-shape MoE suite). Backward
TFLOPS = `2 * fwd_FLOPS / bwd_mean_time`.

```
shape                            base_bwd_TF  R4_bwd_TF   Δ TF    Δ%
gpt_oss-GateUP-B4-M2048               619.4      758.8   +139.5  +22.5%
gpt_oss-Down-B4-M2048                 563.6      680.1   +116.5  +20.7%
gpt_oss-GateUP-B4-M4096               856.3      978.3   +122.0  +14.2%
gpt_oss-Down-B4-M4096                 753.1      833.7    +80.6  +10.7%
gpt_oss-GateUP-B32-M2048              649.0      903.6   +254.6  +39.2%
gpt_oss-Down-B32-M2048                618.0      811.3   +193.3  +31.3%
gpt_oss-GateUP-B32-M4096              845.4     1039.0   +193.6  +22.9%
gpt_oss-Down-B32-M4096                789.3      942.7   +153.4  +19.4%

DSV3 / Qwen3 backwards (control, no H4 fires): unchanged within
±5 TF bench noise.

Average BF16 backward TFLOPS: 949.2 -> 980.2 (+30.97 TF)
Forward TFLOPS: ~unchanged (1269.8 -> 1245.5, contention noise)
Correctness: 24/24 PASS (fwd, bwd_x, bwd_w all True per shape)
```

The biggest wins (B=32 GateUP +39 %, B=32 Down +31 %) match the
pattern: larger B → larger transpose cost → larger absolute savings
when the transpose accelerates 5.85x.

## Metric numbers

Full 24-shape weighted-wall metric, GPU 3 (auto-pinned):

```
                       baseline (start)   R4 with fast transpose
Score                  777 single         884.8 mean (5 runs: 887/886/891/882/878)
gpt_oss family geomean 0.8715             1.0992  (+22.8pp)
DSV3 family geomean    1.1186             1.1219  (~unchanged, +0.3pp noise)
Qwen3 family geomean   1.0970             1.1218  (+2.5pp; small movement, noise)

Δ score vs baseline:    +107
Δ score vs hist best:    +67.8
```

Per-shape (best run; gpt_oss highlighted):

```
  gpt_oss-GateUP-B4-M2048   0.850 -> 1.100  +25pp ✓
  gpt_oss-Down-B4-M2048     0.871 -> 1.160  +29pp ✓
  gpt_oss-GateUP-B4-M4096   0.953 -> 1.109  +16pp ✓
  gpt_oss-Down-B4-M4096     0.959 -> 1.156  +20pp ✓
  gpt_oss-GateUP-B32-M2048  0.773 -> 1.040  +27pp ✓
  gpt_oss-Down-B32-M2048    0.785 -> 1.065  +28pp ✓
  gpt_oss-GateUP-B32-M4096  0.893 -> 1.085  +19pp ✓
  gpt_oss-Down-B32-M4096    0.908 -> 1.084  +18pp ✓

All 8 gpt_oss shapes now > 1.0 (was all < 1.0).
```

## Side-metric regression check

* `_metric_grouped_fused_wall.py` (FP8 fused-act wall):
  **score 927** (≥ 920 floor). FP8 path unchanged (only added
  new BF16 function, FP8 callers still use the existing
  `fp8_transpose_3d`).
* `_metric_grouped_only.py` (BF16 + FP8 grouped fwd-only):
  **score 983** (≥ 980 floor).
* No new rejects, no new correctness FAILs in any metric.

## Files touched

- Primus-Turbo:
  - `primus_turbo/triton/utils/fp8_transpose.py`: added
    `bf16_transpose_3d` function (mirrors `fp8_transpose_3d` for
    BF16 — same kernel template, no byte-cast).
  - `primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_impl.py`:
    - imported `bf16_transpose_3d`
    - in `GroupedGEMMHipKittenBackend.execute`'s H4 reroute, replaced
      `b.transpose(-2, -1).contiguous()` with `bf16_transpose_3d(b)`
      when `b.is_contiguous()`; legacy fallback retained for
      defensive non-contig case.

- HipKittens: NO changes (kernel-side strided-B work explicitly
  deferred to a future round; this round's win is pure host+Triton
  speed-up of the existing transpose+copy step).

## Recommendation for round 5

Round 4 closed the H4 transpose bottleneck. gpt_oss family geomean
is now at 1.10 (vs 1.25 target = +14 % gap). Remaining headroom
on gpt_oss is in the K-tail forward kernel and the var-K dB kernel
itself — both kernel-side work.

Next attack vectors (in approx priority order):

1. **Lever B1 (DSV3/Qwen3 MFMA scheduling)**: 16 shapes at 1.10–1.13
   each; pushing them across 1.25 adds ~0.3 score per pp. Full
   1.25 saturation = +50–80 score. Lower per-shape leverage but
   broad reach (16 shapes); needs MFMA pipeline scheduling work in
   the K%128==0 fast path of `kernel_bf16_dynamic.cpp`.
2. **Lever A1 (in-kernel masked K-tail loop)**: gpt_oss forward
   K-tail (K%128=64) fuse is already enabled (round-3 path A) but
   the per-iter cost of the K-tail accumulation tile remains
   measurable. Re-profile with `rocprofv3 valuMfmaUtil` to see if
   MFMA saturation drops on the K-tail cells. Expected +5–10 pp on
   gpt_oss family if the K-tail tile can be merged into the
   penultimate K-block's MFMA pipeline.
3. **Lever D step 2 (var-K kernel)**: var-K dB is now ~70-80 % of
   gpt_oss bwd time (transpose was ~30 %, now ~5 %). The var-K
   kernel for small-grid shapes (Down-B4: ~1.9 tiles/CU × 32
   K-iters/tile) under-utilizes CUs. Multi-round work on the
   var-K tile topology.

Round 5 recommendation: **Lever B1** — investigate DSV3 / Qwen3
MFMA scheduling. The K%128==0 fast path is the cleanest target
(no K-tail complexity), and the 16 affected shapes give predictable
score deltas per pp gained.
