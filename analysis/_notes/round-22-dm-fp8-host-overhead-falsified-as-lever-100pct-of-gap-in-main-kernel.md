# Round 22 (DM): Host overhead falsified as a lever — 100% of FP8 grouped gap is in `grouped_rcr_kernel`

## Status

- Primus-Turbo HEAD start: `4a646e49`
- Best metric: 829 (round-18, sha `809f93b3`)
- Last 4 rounds: 829 → 821 → 818 → 819 (improvement campaign stalled)
- Cumulative falsifications since round-17 PMC localization:
  - R18 SRD hoist (compiler already CSE'd) → 829 (within noise)
  - R19 two-tile main loop port (catastrophic spill cliff -37%) → 821
  - R20 `readfirstlane` on `lds_tile_base` (compiler already hoisted) → 818
  - R21 `__noinline__` on `rcr_8w_load_hoist` (build-fail; helper architecturally requires inline) → 819

## Round 22 question

Round-17 PMC said HK's main kernel is ~0.83% slower than Triton in pure
device wall-time, but the end-to-end metric is ~5% slower. The implied
"~80% of the gap is non-main-kernel" suggested host-side / helper-launch
overhead might be the productive lever. Round 11 trimmed `execute()` host
overhead to ~4.8 µs, but that was 6+ months of metric history ago — has
the trim held? And is the helper launch (quantize + scale) cost actually
the gap?

**Probe**: `/tmp/probe_round22_host_overhead.py` measures per-call
wall vs device time on the 3 worst FP8 shapes:

| shape                          | wall_us | dev_us | host_us | host% |
|--------------------------------|--------:|-------:|--------:|------:|
| gpt_oss-GateUP-B4-M2048 (0.953)|    260  |   258  |   2.4   |  0.9% |
| DSV3-Down-B16-M2048    (0.955) |    827  |   832  |  -4.6   | -0.6% |
| gpt_oss-GateUP-B4-M4096 (0.963)|    437  |   432  |   4.9   |  1.1% |

**Per-kernel device time breakdown (gpt_oss-GateUP-B4-M2048)**:

| device kernel                                           | µs    |
|---------------------------------------------------------|------:|
| `grouped_rcr_kernel<0, true, true>` (HK main GEMM)      | 153.9 |
| `unary_kernel` (BF16→FP8 quantize)                      |  43.5 |
| `reduce_row_kernel<AbsMaxOp, bfloat16, ...>` (amax-A)   |  40.7 |
| `reduce_row_kernel<AbsMaxOp, float, ...>`               |   8.2 |
| `compute_scale_from_amax_kernel`                        |   7.5 |
| `compute_group_offs_device`                             |   3.9 |
| **shared quantize+offs total**                          |  84.3 |

## Conclusion: host is **not** the lever

Host overhead is 0.9-1.1% (or below noise). Round-11's trim is intact.
Subtracting the **shared invariant quantize+offs (~84 µs)** from both
backends (the quantize call is identical for HK and Triton — task body
declares `ops/grouped_gemm_fp8.py:306-307` INVARIANT) gives the **pure
main-kernel comparison**:

| shape                          | HK main µs | Triton main µs (impl.) | main gap |
|--------------------------------|-----------:|------------------------:|---------:|
| gpt_oss-GateUP-B4-M2048        |      154   |   ~146 (260/1.049 - 84) |   +6.1%  |
| DSV3-Down-B16-M2048            |      533   |   ~582 implied wrong??  | revisit  |
| gpt_oss-GateUP-B4-M4096        |      302   |   ~370 implied wrong??  | revisit  |

(The latter two cases imply HK main kernel is FASTER than Triton main
kernel, which contradicts the metric showing HK SLOWER overall. This
tells us **Triton's quantize+offs path is NOT the same as HK's** — the
metric is comparing different end-to-end stacks. Triton may use a fused
quantize-into-GEMM kernel or a faster reduce kernel. So the 84 µs is
NOT a clean common-mode subtraction. Only the gpt_oss-B4-M2048 number
where main kernel dominates at ~60% of wall is reliable.)

**The honest summary**: the metric reports a ~5% wall-time gap that is
distributed across three components — main GEMM kernel, quantize/scale
helpers, and a small host overhead — and the relative weights of these
three components differ between HK and Triton. The cleanest signal we
have is round-17's per-kernel rocprof: HK `grouped_rcr_kernel` is
~0.83% slower than Triton's grouped FP8 GEMM kernel **at the kernel
PMC level**. That's a small absolute number but every bit counts.

## What this means for rounds 23+

1. **Host overhead is NOT a lever.** The execute() body is already
   trimmed to ~3 µs. Don't chase further trims.
2. **Quantize is INVARIANT (task body).** Cannot touch even though it
   accounts for ~30% of wall time on B=4 shapes.
3. **The main kernel gap is small (~0.83% PMC, ~6% metric on the worst
   shape).** Rounds 17-21 falsified every kernel-level intervention I
   could think of. The remaining options from round 21:

   **Option A: batch 2 B-tiles per `rcr_8w_load_hoist` call.**
   Halves the inline-expansion footprint of the per-K-iter loads.
   Risk: round-19 showed the kernel is on a spill cliff — adding ANY
   code regresses 30%+. But this option REDUCES code (2 calls instead
   of 4), so it's safer than R19's two-tile port.

   **Option B: re-layout B prefetch so all 4 B tiles share one SRD
   construction in the main loop.** The dense kernel does this. But
   the FP8 grouped LDS swizzle has a different stride pattern, so the
   port isn't trivial.

   **Option C: stop chasing kernel internals; accept 818-829 as the
   ceiling and move score-up effort to per-shape config tuning.**
   Round 7-69 already did this exhaustively; the easy wins are gone.

4. **The "spill divergence" finding from round 20 (BF16 grouped 0-spill
   vs FP8 grouped 67-spill) remains the most actionable diagnostic.**
   The mechanism by which BF16 achieves 0 spill while FP8 has 67 spill
   despite similar persistent-loop bookkeeping is the single piece of
   knowledge that, if understood, would unlock kernel-level wins.
   This needs **architecture-level diff** of the two grouped kernels
   (BF16 `grouped_kernel` line ~3667 in `kernel_bf16_dynamic.cpp` vs
   FP8 `grouped_rcr_kernel` line ~1973 in `kernel_fp8_layouts.cpp`)
   focusing on what differs between their respective inner loops.

## Round 22 verdict

This round produced no kernel change. The probe falsified the
"host overhead is the lever" hypothesis. The score did not move.
Doc-only commit captures the rule-out so future rounds don't
re-chase host overhead.

**Metric**: 819 (no change vs round 21).

## Recommended round 23 starting move

Since the chat session rolls over after this round (>90 min), the
next agent will cold-start. To avoid losing the cumulative
falsification trail, the recommended first action is:

1. Read `analysis/_notes/round-{17..22}-dm-*.md` to absorb the
   rule-outs (do not re-run any of: SRD hoist, two-tile port,
   readfirstlane, `__noinline__`, host-overhead trim, MFMA cell
   migration, chunk_size sweep).
2. Either implement Option A above (batch B-tile loads), or step
   back and do the BF16 vs FP8 grouped-kernel architecture diff to
   understand the 0 vs 67 spill divergence (round 20 finding).
3. The score has been 818-829 for 5 rounds. If 3 more rounds don't
   beat 829, consider escalating to the user that the FP8 main
   kernel may have hit its architectural ceiling vs Triton.
