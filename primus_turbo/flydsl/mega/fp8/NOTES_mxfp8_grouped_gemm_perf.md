# MXFP8 grouped GEMM perf note (MI355X / gfx950)

Investigation of the MXFP8 (per-1x32 E8M0 block-scaled) grouped GEMM throughput vs the
per-tensor fp8 and bf16 ceilings, on AMD Instinct MI355X. DeepSeek-V3 MoE shapes:
L1 up/gate `N=2I=4096, K=H=7168`; L2 down `N=H=7168, K=I=2048`; `G=32` experts/rank (EP8).

## TL;DR

- **Block-scaling is NOT a throughput bottleneck.** The dense standard mxfp8 kernel
  (`flydsl/gemm/mxfp8_gemm_kernel.py`) already runs at **~2.7k TFLOPS (~1.85x bf16,
  ~0.9x of the per-tensor fp8 ceiling)** — the per-1x32 E8M0 scale loads + scaled MMA
  (`v_mfma_scale_f32_16x16x128_f8f6f4`) are essentially free.
- The grouped mxfp8 kernel *looked* 2x slower (~1216 TF) but that was a **measurement
  artifact**, not a kernel problem: `grouped_gemm_mxfp8_flydsl_kernel` re-ran
  `flyc.compile(launch, *args)` on **every call** (no compiled-object cache), so the
  per-call JIT/compile stall leaked into the CUDA-event timing window.
- **Fix**: cache the compiled launch per shape (`[raw, compiled]`, mirroring the dense
  mxfp8 kernel and `grouped_gemm_fp8_tensorwise_flydsl_kernel`). After the fix the
  grouped mxfp8 GEMM runs at **~2.0x bf16 / ~0.9x of per-tensor fp8**, as expected.

## How we found it (differential, each isolates one variable)

| experiment | result | conclusion |
|---|---|---|
| dense standard mxfp8 vs bf16 / per-tensor fp8 | 2727 / 1463 / 2607 TF | block-scaling ≈ free |
| grouped mxfp8, sweep `group_m`/`num_xcd` | all ~1150–1216 TF | not the L2-reuse swizzle |
| grouped mxfp8 `G=1` (single group, same shape) | 1372 TF (still ~2x off) | not per-group B streaming |
| grouped: scan prologue vs lookup (no O(G) scan) | equal (~1160–1200 TF) | not the O(G) group scan |
| **split the fused stub: preshuffle vs gemm-only (cached compile)** | **gemm-only = 2502 TF**, preshuffle = 0.05 ms | **the kernel is full-speed; the wrapper's per-call `flyc.compile` was the cost** |

## The fix

`primus_turbo/flydsl/mega/fp8/grouped_gemm_mxfp8_kernel.py`:
`_get_grouped_mxfp8_launch` now returns a cached `[raw, compiled]` entry; the host
wrapper compiles once per shape (eager) / uses the raw `@flyc.jit` closure under CUDA-graph
capture — identical to the dense mxfp8 path. (Also added an optional `lookup=True`
tile→group prologue that avoids the O(G) scan; measured perf-neutral, kept as an option.)

## Perf: grouped mxfp8 GEMM, before vs after the cache fix (G=32, MI355X)

TFLOPS (FLOPs = 2·M·N·K). SNR vs bf16 ≈ 28 dB throughout.

| stage | M_per | M     | bf16 | per-tensor fp8 | mxfp8 **before** | mxfp8 **after** | after / fp8-tw |
|-------|-------|-------|------|----------------|------------------|-----------------|----------------|
| L1    | 256   | 8192  | 929  | 1976           | 772              | **1510**        | 0.76x |
| L1    | 512   | 16384 | 1096 | 2369           | 1025             | **1999**        | 0.84x |
| L1    | 2048  | 65536 | 1197 | 2635           | 1216             | **2411**        | 0.91x |
| L2    | 256   | 8192  | 684  | 1498           | 641              | **1221**        | 0.82x |
| L2    | 512   | 16384 | 918  | 1764           | 830              | **1568**        | 0.89x |
| L2    | 2048  | 65536 | 1022 | 2115           | 1000             | **1976**        | 0.93x |

After the fix the large-M (M_per=2048) grouped mxfp8 is ~2.0x bf16 and within ~7–9% of the
per-tensor fp8 ceiling; small-M closes the rest as the grid fills. Correctness unchanged
(`tests/pytorch/ops/test_grouped_gemm_mxfp8_flydsl.py` 7 passed).

## Downstream: DSv3 8K top8 EP8 dispatch + L1 GEMM (bf16 vs fp8), MI355X

Kernel-level (`bench_dispatch_grouped_gemm_mxfp8.py`, load_balanced), after the grouped
cache fix:

| leg | bf16 | fp8 |
|---|---|---|
| dispatch_only (XGMI push) | 2.05 ms (402 GB/s) | **1.07 ms** (1.94x fewer bytes) |
| gemm_only (grouped L1)    | 3.19 ms (1282 TF) | **1.76 ms (2321 TF, preshuffled scale)** |
| fused single-kernel       | 3.58 ms | 5.02 ms ✗ (raw scale + L2 fences) |

- **Decoupled fp8** = dispatch_only + preshuffled gemm_only = **1.07 + 1.76 ≈ 2.83 ms**,
  which **beats bf16 fused (3.58 ms) by ~1.26x** — a direct payoff of the cache fix (the
  fp8 gemm was "1175 TF" before, is 2321 TF now).
- The **fused single-kernel (5.02 ms)** is *slower than running the two legs serially*.
  Isolated via a `no_fence` diagnostic (skip the L2 wb/inv, wrong answer, timing only):

  | fused variant | ms |
  |---|---|
  | fused (with fences) | 5.02 |
  | fused (no fence)    | 4.80 |
  | **fence cost**      | **0.22** |

  Adding a `no_gate` diagnostic (comm off, pool pre-filled → the fused kernel's internal
  gemm alone):

  | fused variant | ms | isolates |
  |---|---|---|
  | fused (full)        | 5.02 | |
  | fused (no fence)    | 4.79 | fence cost = **0.23 ms** |
  | fused (gemm-only)   | 4.58 | comm-overlap gap = **0.21 ms** |

  So the fused kernel is **architecturally fine**: comm is ~fully overlapped (0.21 ms exposed)
  and the L2 coherence fences are cheap (0.23 ms) — the earlier "~1.5 ms fence" guess was
  wrong. The entire 5 ms is the **internal raw-scale gemm (4.58 ms)**. NOTE the benchmark's
  `gemm_only` line (1.74 ms, 2321 TF) is the *preshuffled decoupled* kernel, NOT what the
  fused kernel runs — comparing fused(raw) vs gemm_only(preshuffled) is what made fused look
  "broken". The fused's internal gemm uses the **raw on-the-fly `ScaleS2RRaw`** (measured
  **0.30–0.39x** of preshuffled: 1034 vs 2762 TF dense, compile-cached — a real kernel gap;
  4.58 ms raw ≈ 0.38x of the 1.74 ms preshuffled, consistent).
  Root cause: in the raw loader lane `r=lane%16` indexes the scale ROW, and rows are `K//128`
  dwords apart, so a wave's 16 distinct scale words land on ~16 scattered cachelines per
  sub-tile (vs preshuffled: 64 lanes read 64 contiguous dwords). The scattered global-latency
  loads stall the scaled MMA.
  Ideal fused (preshuffled scale + comm overlap; fences stay, they're cheap) ≈ max(gemm 1.76,
  comm 1.07) ≈ **1.8 ms (~2x bf16)**.

**Fix path for the fused kernel** (keeps fusion + overlap): feed the gemm role *preshuffled*
scale instead of raw. B (weights) are static → preshuffle once on host, read with
`ScaleBComb`. A (pool) is filled dynamically by the comm role → have the **comm push the
scale already in the `ScaleS2R` broadcast layout** (scatter each token's `K//128` E8M0 words
into the layout-1 positions for its dest row), grow the `pool_scale` symm region ~4x
(broadcast int32), and read with `ScaleS2R`. LDS staging is not viable (the fp8 A/B buffers
already use ~128 KB of the 160 KB CDNA4 LDS; a full-K scale tile is ~57 KB).

**Recommendation**: ship the **decoupled fp8** path now (1.26x over bf16, `comm="fp8"` default);
the fused ~2x needs the comm-push-preshuffled-scale change above (bounded but multi-file:
`ep_fp8` push, `sym_layout`/`symm_buffer` region size, tile loaders).

## Fused quant-in-push attempt (correct, but not a win) — DSv3 8K top8 EP8

Implemented the fused path end-to-end (all pieces validated bit-exact):
- `quant_flydsl.py`: FlyDSL MXFP8 quant (amax + E8M0 round-even + soft-clamp + `cvt_pk_fp8_f32`),
  bit-exact vs torch/HIP; optional `preshuffle=True` writes the scale directly in the ScaleS2R
  broadcast layout (bit-exact vs `build_preshuffle_ab_kernel`).
- `ep_fp8.dispatch_fp8_tile`: reads bf16 tokens, quantizes per 1x32 block in-warp, pushes fp8 +
  broadcast-layout E8M0 (no separate quant op, no separate preshuffle pass).
- `sym_layout`/`symm_buffer`: `pool_scale_ps` broadcast region (raw `pool_scale` kept for decoupled).
- `gemm_mxfp8_nt_tile(preshuffled=True)`: `ScaleS2R`/`ScaleBComb` (fast); `preshuffle_b_scale` for B.

Correctness: EP8 `comm="fp8_fused"` = **22.97 dB** (matches decoupled + bf16). Perf (after the
same compile-cache fix; ndcu swept):

| | ms | note |
|---|---|---|
| fused gemm-only (no comm) | **1.62** (2521 TF) | fast — at the fp8 ceiling |
| fused fp8 (best, ndcu=64) | 4.03 | comm-overlap gap ~1.6 + fence ~0.85 on top |
| bf16 fused (best, ndcu=16) | **3.53** | |
| decoupled fp8 (kernel) | ~2.87 | comm 1.13 + gemm 1.74 |

**Negative result**: fusing quant into the push puts the *heavy* quant (bf16 read = 2x bytes +
amax/E8M0/cvt compute) on the comm->gate->gemm critical path, so it cannot hide under the (now
1.62 ms) gemm even at ndcu=64. Best fused fp8 (4.03 ms) does NOT beat bf16's best (3.53 ms) or the
decoupled fp8 (~2.87 ms). The fp8 comm/compute savings are roughly offset by the quant cost at
this scale.

**Takeaway**: keep quant *separate and light* (the decoupled path). A future fused win would push
*pre-quantized* fp8 + *scatter the small pre-computed scale* into the broadcast layout (light comm
that hides under the gemm), rather than quantizing in the push — but the modeled gain over decoupled
is small (~0.4 ms). The `quant_flydsl` quant/preshuffle kernels + `preshuffled=True` tile are validated
and reusable regardless.

## Caveat & remaining item

- Any earlier EP8 mega numbers that routed through the *uncached* grouped kernel (e.g. the
  decoupled `comm="fp8"` L1 gemm-only baseline) were similarly polluted by the per-call
  compile stall and should be re-measured.
- The **fused dispatch kernel**'s gemm role uses `gemm_mxfp8_nt_tile` with a **raw
  on-the-fly** E8M0 scale loader (`ScaleS2RRaw`), which measures **~2.6x slower** than the
  preshuffled `ScaleS2R`/`ScaleBComb` path (≈458 vs ≈1200 TF at the polluted scale; the
  gap is a real kernel difference, not the compile artifact). To make the fused fp8 dispatch
  competitive, switch its gemm role to the preshuffled scale layout or optimize
  `ScaleS2RRaw`, then re-benchmark DSv3 EP8 fused vs bf16.

## Repro

Dev container `xiaoming-dev-slime` (`tasimage/primus:pr-836`, flydsl preinstalled), 8×MI355X:

```bash
cd /mnt/shared/xiaoming/Primus-Turbo
PYTHONPATH=$PWD HIP_VISIBLE_DEVICES=0 python benchmark/ops/bench_grouped_gemm_mxfp8.py   # grouped
PYTHONPATH=$PWD HIP_VISIBLE_DEVICES=0 python benchmark/ops/bench_gemm_mxfp8_dense.py     # dense ceiling
```
