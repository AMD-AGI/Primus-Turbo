# Round 18 (death-march) — FP8 grouped SRD hoist: SALU savings not on the wall-time critical path

## Round 17 hypothesis tested

Round-17 rocprof identified two structural costs in HK FP8 `grouped_rcr_kernel`
relative to Triton on `DSV3-Down-B16-M4096`:

1. **Scratch spills (FLAT instructions)**: 2.9× Triton (2.99M vs 1.03M per
   launch) — direct consequence of 67-VGPR spill + 272-byte/lane scratch.
2. **SALU instructions**: 1.53× Triton (60.9M vs 39.8M) — the round-17 doc
   speculated ~33% (~6.9M) of the 21M SALU excess came from per-call
   `make_srsrc(tensor_base, src.batch*src.depth*src.rows*src.cols*sizeof(T))`
   inside `rcr_8w_load_hoist`, called 12× per persistent-loop iteration
   inside `grouped_rcr_kernel` — 3 multiplies + 4-element vector construct
   per call.

The round-17 plan recommended **option 2** (cheap, single-round) before
**option 1** (b0/b1 reload pattern, multi-round, attacks the 67-VGPR spill
directly): hoist `make_srsrc` out of `rcr_8w_load_hoist` to once-per-kernel.
Estimated +2-4 score, "validates the round-17 diagnostic was load-bearing".

## Implementation (HK kernel, ~163 lines added, then reverted)

Added new helper `rcr_8w_load_hoist_srd<...>(dst, src, idx, swizzled_offsets,
i32x4 srsrc)` after the existing `rcr_8w_load_hoist` (kernel_fp8_layouts.cpp
line ~570). Identical body except: skips `make_srsrc(tensor_base,
total_bytes)` construction, uses caller-provided SRD as the buffer-resource
operand of the inline-asm `buffer_load_dwordx4 ... offen lds`. Per-call
`tile_byte_offset` (idx-dependent) stays inside the helper.

In `grouped_rcr_kernel` body, computed cached SRDs once per kernel entry
(after the swizzled-offsets prefill, before the persistent loop):

```c++
const uint32_t a_total_bytes_grp = static_cast<uint32_t>(
    size_t(g.a.batch()) * size_t(g.a.depth()) *
    size_t(g.a.rows())  * size_t(g.a.cols())  * sizeof(fp8e4m3));
const uint32_t b_total_bytes_grp = static_cast<uint32_t>(
    size_t(g.b.batch()) * size_t(g.b.depth()) *
    size_t(g.b.rows())  * size_t(g.b.cols())  * sizeof(fp8e4m3));
i32x4 a_srsrc_grp = make_srsrc((fp8e4m3*)g.a.raw_ptr, a_total_bytes_grp);
i32x4 b_srsrc_grp = make_srsrc((fp8e4m3*)g.b.raw_ptr, b_total_bytes_grp);
```

Replaced 12 call sites in `grouped_rcr_kernel`:
* 7 in prologue (lines 2277-2280, 2286-2288)
* 4 in main K-loop body (lines 2313, 2320, 2327, 2333)
* 1 in epilog 1 (line 2343)

Dense kernel + `dispatch_grouped_var_k_fp8` paths intentionally untouched
(out of scope this round).

## Build resource usage — **unchanged**

`grouped_rcr_kernel<0,0,0>` (vanilla DSV3 template spec):

| Metric | Round-17 baseline | Round-18 SRD hoist |
|---|---|---|
| TotalSGPRs | 65 | **65** (identical) |
| VGPRs | 256 (max) | **256** |
| ScratchSize bytes/lane | 272 | **272** (identical) |
| VGPRs Spill | 67 | **67** (identical) |
| LDS bytes/block | 139796 | **139796** |
| Occupancy waves/SIMD | 2 | **2** |

All 4 template specializations `<0,0,0>` `<0,1,0>` `<0,0,1>` `<0,1,1>` showed
**identical resource numbers** before and after the source change. The
explicit SRD hoist did NOT free any registers, did NOT reduce scratch, and
did NOT change the compiler's spill decisions.

## Wall-time — **flat within noise**

Four metric runs after rebuild:

| Run | score | geomean grpFP8 |
|---|---|---|
| 1 | 824 | 0.9889 |
| 2 | 820 | — |
| 3 | 828 | — |
| 4 | 823 | — |
| **mean** | **823.75 ± 3.4** | — |

Round-17 baseline (no source change, two runs): 823, 830 → mean 826.5 ± 3.5.
Round-15/16 historical: 822, 825, 825 → mean 824 ± 1.5. Round-18 result is
**indistinguishable from baseline noise** (overlapping 1-σ bands).

Per-shape: DSV3-Down-B16-M4096 (the rocprof-localized worst shape) moved from
0.943 → 0.946 (+0.003, well below the per-shape ratio jitter of ±0.01-0.02).
No shape moved more than ±0.02. No correctness regressions
(`correct_fail=0/16`, SNR=28.48 dB on both DSV3-Down-B16-M4096 and
gpt_oss-Down-B4-M2048).

## Falsification root cause

Two consistent explanations for the null result, ranked by likelihood:

**(a) Compiler was already CSEing `make_srsrc` across the 12 inlined
expansions.** Both `rcr_8w_load_hoist` and `rcr_8w_load_hoist_srd` are
`__forceinline`, so the compiler sees one big function with 12 `make_srsrc`
calls all reading the same uniform `src.batch/depth/rows/cols/raw_ptr`
expressions. CSE would collapse them automatically. Evidence: identical
SGPR + scratch + VGPR-spill + LDS counts across the 4 template specs.

**(b) Even if the SALU savings were real (~6.9M instructions), they were
overlapped with HBM loads + MFMA issue and not on the wall-time critical
path.** The MI350X SIMD has independent SALU vs VALU vs MFMA issue ports;
a fully memory-bound + MFMA-bound kernel can absorb 6.9M SALU "for free"
inside the existing HBM/LDS bubbles. Round-17 PMC showed `MemUnitStalled` ≈
0.24% (HK) vs 0.56% (Triton) — very low memory stall, consistent with a
balanced pipeline that has SALU slack.

In either case: **the round-17 hypothesis "SALU overhead is on the critical
path" is FALSIFIED**. The 21M-SALU-excess vs Triton is a real instruction
count delta but NOT a wall-time-relevant bottleneck.

## Implication for round 19+

Re-rank round-17's option list:

| Option | Round-17 priority | Round-18 evidence | New priority |
|---|---|---|---|
| **Option 1**: b0/b1 reload pattern (attacks 67-VGPR spill directly) | "multi-round, higher yield" | Untested; the 67-VGPR spill is unchanged → still the best-attested bottleneck | **#1 (was #2)** |
| **Option 2**: SRD hoist (SALU reduction) | "single-round, +2-4 score" | **Falsified — flat within noise** | **CUT** |
| **Option 3**: Prefetch depth tuning (BF16 has deeper unroll) | not in round-17 doc | open | **#2** |
| **Option 4**: N-tail masked C-store port from BF16 (gpt_oss only) | round-17 prep mentioned | open | **#3 (gpt_oss-only)** |

Round 19 should attempt **option 1 (b0/b1 reload pattern)** as planned.
Specifically:
* Profile `grouped_rcr_kernel` register liveness via `llvm-mca` or
  `--save-temps` to identify exactly which VGPRs are causing the 67-spill
  (likely the `b_tile`/`As` LDS tile addresses + the 4 accumulator
  registers `cA, cB, cC, cD` × `RBM × RBN` = 16 registers each).
* Try splitting `cA, cB` and `cC, cD` into separate `RBM=2 × RBN=1`
  half-tile accumulators, with explicit register-allocation hints to keep
  one half live at a time (mirror BF16 dense's dual-accumulator pattern at
  `kernel_bf16_dynamic.cpp:567+`).
* Or: use `raw_buffer_load_b128` directly into VGPR registers for the
  per-K-iter B reload (skip LDS staging for the outer `b1` tile), at the
  cost of LDS bandwidth. BF16 grouped tested this pattern in round 27.

## What stays untouched

* Round-17-dm rocprof PMC findings remain valid (reaffirmed by this
  round's null result — the FLAT/scratch story is structural, not an SALU
  byproduct).
* Architecture: persistent single-launch, no host sync, no per-group loops
  — INVARIANT.
* Dense kernel + `dispatch_grouped_var_k_fp8` SRD construction left as-is
  (no evidence they're a bottleneck).

## Falsified levers (cumulative, for next round's prompt)

1. BF16 K_STEP=64→32 port (round 26) — out of scope.
2. FP8 chunk_size sweep (round 22) — dead-end.
3. FP8 MFMA cell-shape migration (rounds 14-15) — falsified by rocprof.
4. FP8 main-loop barrier/setprio/vmcnt micro-knobs (rounds 13-16) — saturated.
5. **FP8 grouped SRD hoist for SALU reduction (THIS ROUND, round 18)** —
   compiler already CSEs; SALU is not on critical path.

## Status

**Reverted**. Doc-only commit in Primus-Turbo. No HipKittens commit this
round (kernel changes were tested, confirmed flat, reverted in-tree).

Round-18 metric SHA: this commit (Primus-Turbo).
HK SHA: unchanged (`a0644b80` baseline + reverted experiment).

Score: 823.75 ± 3.4 (mean of 4 runs, indistinguishable from rolling
baseline 824 ± 1.5).
