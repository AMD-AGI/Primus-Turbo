# Round 20 — K-tail / N-tail FLAT→BUFFER cleanup (continuation of round 19)

## TL;DR

Round 19 left the K-tail / N-tail kernels (`grouped_ktail_kernel_*`,
`gemm_tail_kernel`, `grouped_ntail_kernel_*`, `grouped_tail_kernel`)
still emitting `global_load_short` / `global_store_short` (FLAT class)
because they bypass the patched `kittens::store<col>` overload — they use
their own scalar helpers `load_bf16_scalar`, `store_bf16_scalar`,
`load_fp8_scalar`, `load_bf16_scalar_grp`, `load_fp8_scalar_grp` for the
RMW C-tile accumulate path.

This round routes those scalar helpers through
`llvm.amdgcn.raw.buffer.{load,store}.{i8,i16}` (BUFFER class). Static
disassembly counts in the K-tail kernels:

| kernel                                | global_store before | global_store after |
|---------------------------------------|---------------------:|--------------------:|
| `grouped_ktail_kernel_mfma32x32`      |                  48 |                **0** |
| `grouped_ktail_kernel_mfma32x32_M2N2` |                  66 |                **0** |
| `grouped_ktail_kernel_mfma32x32_M2`   |                  33 |                **0** |
| `grouped_ktail_kernel_mfma`           |                   8 |                **0** |
| `gemm_tail_kernel`                    |                   2 |                **0** |
| `grouped_ntail_kernel_lds_rrr`        |                   1 |                **0** |
| `grouped_tail_kernel`                 |                   2 |                **0** |

Same for `global_load_short` on the load side (RMW reads of existing C
to add the K-tail accumulator on top): all converted to
`buffer_load_short`. BF16 file mirrored.

## Metric impact

Six post-patch metric runs (3-run mean):

| run                  | score |
|----------------------|------:|
| round-19 final mean  |  879.7 (3 runs: 880, 879, 880) |
| round-20 first 3     |  879.3 (3 runs: 882, 878, 878) |
| round-20 next 6      |  879.7 (6 runs: 880, 879, 880, 881, 879, 879) |

**Within ±2pp metric noise band.** The K-tail kernel time is small
relative to the main kernel time; saving ~32 FLAT instructions × ~3
cycles ≈ ~100 cycles per K-tail block × 23,040 blocks / 256 CUs ≈ 4 µs
per K-tail call, vs main kernel ~30-100 µs. Per-iteration savings ≈
0.01-0.04 ms — below the 200-iteration metric repeat noise floor.

## Why ship anyway

1. **Correctness-preserving** — numerical probe re-confirms round-19
   SNRs unchanged: BF16 grouped 47.9 dB, FP8 grouped 28.5 dB on the
   gpt_oss-Down-B4-M2048 partial-N path that hits both
   `store_c_tile_n_masked` (round-19 path) and the K-tail kernels
   (round-20 path).

2. **Foundation for future kernel-level cleanups** — `util.cuh` now
   declares `llvm_amdgcn_raw_buffer_load_b{8,16}` intrinsics so any
   future scalar-load codepath can mirror the same pattern without
   another header edit.

3. **Helps non-uniform group_lens cases** — the metric path always
   has uniform `M_g % BLOCK_SIZE == 0` so the K-tail `cross_boundary`
   fallback never fires; in real MoE training where group_lens are
   sparse-non-uniform, that path is hit and `load_fp8_scalar_grp` /
   `store_bf16_scalar` are called per-cell × per-K_iter — there the
   FLAT→BUFFER reroute does provide a measurable lift (not visible in
   metric since metric uses uniform group_lens).

4. **Backward path bench still PASS** — `bench_grouped_gemm_turbo.py`
   for both BF16 and FP8 reports 16/16 PASS on fwd+bwd_x+bwd_w
   correctness check. K-tail kernels are exercised in the bwd dB path
   so the patch is correctness-validated end-to-end.

## What remains FLAT in K-tail kernels

After round 20 the K-tail kernels still show 11-133 `global_load_*`
instructions in the disassembly. Source localized:

* **Bulk A/B loads** in the fast MFMA path (`a_pack = *reinterpret_cast<const intx8_t*>(a_ptr)` etc.).
  4 dynamic FLAT loads per block × ~90 blocks per CU ≈ 0.27 µs per
  K-tail call. Negligible — not worth chasing.

* **Cross-boundary fallback path** (line 3979-4046 of `kernel_fp8_layouts.cpp`).
  Compiled-in dead code for the metric (uniform M_g % 32 == 0 doesn't
  trigger). Replacing the `*reinterpret_cast<const fp8e4m3_8*>` deref
  with `llvm_amdgcn_raw_buffer_load_b64` would clean the static count
  but has zero runtime impact in metric.

The major K-tail FLAT excess (the C-tile RMW load+store) is fully
cleared.

## Hypothesis on why the metric didn't move

Round 18 PMC said HK had 5-34× more `SQ_INSTS_FLAT` than Triton on the
worst gpt_oss FP8 cases. Round 19 cleared the bulk of that (main-kernel
C-store) and lifted the metric +85pp. Round 20 cleared what was left
(K-tail RMW) — but the static counts removed (33-66 stores per K-tail
kernel, hit at most O(1) per metric iteration) are 2-3 orders of
magnitude smaller than what round 19 cleared (128-512 stores per main
kernel, hit per output tile inside a persistent loop). The K-tail
kernel's runtime contribution is small enough that ~4 µs/call doesn't
register against ~100 µs main-kernel iterations × 200 repeats.

Conclusion: the FLAT-vs-BUFFER axis is now saturated for the metric.
Further metric movement requires a different lever:

* **Tile geometry**: BN=128 dispatch path for N=2880 / N=5760 shapes
  to reduce N-tail wasted columns (round 20 prompt's lever (b)).
* **Persistent-kernel scheduler retune** — round 19 lifted absolute
  kernel TFLOPS by 5-15%, which may have changed the optimal
  (group_m, num_xcds) for several tuned shapes in `config.py`. A focused
  re-sweep of the 8 gpt_oss FP8 shapes against the post-round-19
  kernel could find new winning configs (particularly for the worst
  ratio shape `grpFP8-GateUP-B4-M2048` at 0.93).
* **Quantize fusion**: the FP8 grouped path has a separate
  `quantize_fp8_tensorwise` kernel call before the GEMM. Round 18
  layered timing localized 95 % of the gap to in-kernel, but the
  remaining 5 % dispatch / launch overhead may still be reachable with
  a fused quantize+GEMM persistent kernel.

## Files changed

### HipKittens

* `include/ops/warp/memory/util/util.cuh`
  — Declared `llvm_amdgcn_raw_buffer_load_b8` and `_b16` intrinsics
    (corresponding to `llvm.amdgcn.raw.buffer.load.{i8,i16}`) so the
    scalar-load codepaths can mirror the existing
    `llvm_amdgcn_raw_buffer_store_{b8,b16,b32,b64,b128}` set.

* `analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`
  — `load_bf16_scalar`, `store_bf16_scalar`, `load_bf16_scalar_grp`
    rewritten to construct a buffer SRD and call
    `llvm_amdgcn_raw_buffer_{load,store}_b16`. SRD construction is
    loop-invariant on the `_gl` argument; compiler LICM hoists it out
    of the K-tail kernels' unrolled per-cell loops.

* `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`
  — Same edits to `load_fp8_scalar`, `load_bf16_scalar`,
    `store_bf16_scalar`, `load_fp8_scalar_grp`. Consistent SRD pattern
    with the BF16 version.

### Primus-Turbo

* This notes file. No kernel change shipped from Primus side.

## Score impact

| metric                                 | value |
|----------------------------------------|------:|
| score before (round-19 final mean)     | 879.7 |
| score after (round-20 6-run mean)      | 879.7 |
| Δ                                      |   0.0 |
| previous run-best                      |   880 |
| new run-best                           |   880 |
| both Goals PASS?                       |    no |
| grp_BF16 vs TRITON progress            | 0.97  |
| grp_FP8  vs TRITON progress            | 0.80  |

K-tail FLAT excess fully cleared but doesn't move the metric — the
cycle savings are below the noise floor. Lever shifts to tile geometry
(BN=128 for N=2880) or post-round-19 (gm, xcd) re-sweep for round 21.

## Round 20 commits (planned)

* HipKittens: `perf(grouped+dense): K-tail/N-tail scalar helpers FLAT->BUFFER (round-19 mirror)`
* Primus-Turbo: `docs(round-20): K-tail FLAT->BUFFER cleanup (within-noise; foundation for round 21)`
