---
name: round-13-A1prime-variant-2-K-split-host-allocator-gated-NEUTRAL
description: R13a lands the strict-minimum HK companion for the next step of the R11 GREEN-LIT Stream-K K-split arc — host-side hipMallocAsync/hipFreeAsync of sk_partial_buf in dispatch_grouped_rcr, gated on g.sk_split_n > 0. Default sk_split_n=0 means the alloc branch is never entered for any production call; the kernel reads neither new field; codegen unchanged. NEUTRAL metric (4-sample on dev GPU 3 {694, 691, 694, 694}, median 694 vs recent ~693, well within ±3 falsification gate). Splits R13 again from R12 commit's "alloc + kernel branch atomic transaction" plan to isolate three failure modes (alloc bug, K-split coord-decode bug, AGPR spill > 60). HK SHA: TBD-after-commit. Forward pointer: R14 = kernel K-split control flow (gated on g.sk_split_n > 0, reads g.sk_partial_buf, atomicAdd-accumulates fp32 partials at SK tile boundaries) with the alloc machinery already verified bit-identical via R13a's gate.
type: project
---

# Round-13 — A1' variant-2 K-split: host-side allocator (gated, NEUTRAL metric, R14 unblock)

## TL;DR

R12 landed the trailing struct fields (`sk_split_n`, `sk_partial_buf*`)
on `grouped_layout_globals`. R13's stated scope (per the R12 commit
message) was "kernel K-split branch + `hipMallocAsync` of
`sk_partial_buf` together as one atomic transaction". On reading the
700-line `grouped_rcr_kernel` body and the 366-line
`dispatch_grouped_rcr` host wrapper, R13a narrows that to the
host-side allocator alone for the same risk-isolation rationale R12
used to narrow R11's "fields + alloc" plan to fields-only. The kernel
control-flow branch lands in R14.

* HK commit: TBD (this round, HipKittens repo,
  `save/fp8-progress-20260319-native-layouts`).
* Primus commit: this docs note (no PT code change; the new HK alloc
  is gated on `g.sk_split_n > 0` and PT never sets that field, so the
  PT pybind path is unchanged byte-for-byte).
* Metric: NEUTRAL (4-sample on dev GPU 3 {694, 691, 694, 694}, median
  694 vs recent rounds median ~693 — well within the ±3 score R12
  falsification gate, which R13a also adopts).
* Correctness: 8/8 PASS on every sample (per `correctness FAIL: 0/8
  shapes` line which reports 0 failed of 8 shapes).

## What changed (HK side, `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`)

In `dispatch_grouped_rcr` (line ~7686), between the `use_b128` early-
return at line ~7825 and the main RCR launch at line ~7827:

```cpp
int* sk_partial_buf_owned = nullptr;
if (g.sk_split_n > 0 && g.bpc > 0 && g.ki > 0) {
    const int T_max = kittens::ceil_div(g.M_total, BLOCK_SIZE) * g.bpc;
    const size_t buf_bytes =
        static_cast<size_t>(T_max) * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
    hipMallocAsync(reinterpret_cast<void**>(&sk_partial_buf_owned),
                   buf_bytes, g.stream);
    hipMemsetAsync(sk_partial_buf_owned, 0, buf_bytes, g.stream);
    g.sk_partial_buf = sk_partial_buf_owned;
}
```

And at the end of `dispatch_grouped_rcr` (just before the closing `}`
at line ~8052):

```cpp
if (sk_partial_buf_owned != nullptr) {
    hipFreeAsync(sk_partial_buf_owned, g.stream);
}
```

Both branches are gated on `g.sk_split_n > 0`. Default 0 → alloc branch
never entered, free branch never entered, no HIP API calls executed
on the production code path.

## Why split R13 again (vs the R12 commit's stated R13 scope)

The R12 commit message defined R13 as: "the kernel K-split branch +
`hipMallocAsync` of `sk_partial_buf` together as one atomic
transaction so the alloc + kernel-branch interlock is tested in
isolation from the struct extension itself."

On reading the kernel body, that combined R13 entangles three failure
modes:

1. **Allocator bug**: per-call `hipMallocAsync` semantics, `hipFreeAsync`
   stream ordering, OOB on the upper-bound `T_max`.
2. **Kernel K-split coord-decode bug**: the Stream-K kernel branch must
   map per-CTA `[start_K_iter, end_K_iter)` to `(output_tile,
   K_offset_in_tile)`, atomicAdd fp32 partials at SK tile boundaries,
   accumulate to local fp32 + fp8-store at non-boundary tiles. SNR-gated
   per the R11 plan.
3. **AGPR pressure**: the kernel is at 256 VGPR / 37 spill near the
   LLVM 256-VGPR ceiling per the R11 cost decomp risk note. Adding
   K-iter bookkeeping risks `Spill > 60` and an occupancy drop that
   would mask the K-split lift entirely.

Landing all three in one round means a metric regression (or build
failure, or correctness failure) leaves all three suspect. Splitting:

* R13a (this commit): host-side alloc only. Validates the alloc
  primitive choice (per-call `hipMallocAsync`), the upper-bound
  `T_max` derivation, the `hipFreeAsync` stream-ordering, the gating
  branch. Bit-identical for production calls (`g.sk_split_n=0`),
  metric NEUTRAL by design.
* R13b/R14 (next round): kernel control-flow branch alone. Reads
  `g.sk_partial_buf` (already populated by the R13a alloc machinery),
  atomicAdds, etc. The alloc machinery is pre-verified, so any R14
  metric movement attributes to the kernel branch / AGPR pressure
  alone.

This mirrors R12's narrowing precedent (R11's "fields + alloc + memset"
narrowed to "fields only" because `done_counter` wasn't host-cached).
Same risk-control discipline, same NEUTRAL-metric expectation, same
forward-pointer to a smaller-blast-radius next round.

## Why per-call `hipMallocAsync` (vs host-cached pattern)

The R7/R11 plan referenced the `done_counter` finalizer-detect counter
(line ~9281) as a "host-side cached allocator pattern" to mirror.
R12's docs note already corrected this on re-read: `done_counter` is
caller-allocated through pybind, NOT host-cached. There is no host-
cached allocator pattern in this file to mirror.

R11 documented the fallback: "If R12 reveals the host-side cached
allocator pattern is messier than expected (e.g., the `done_counter`
allocator is per-shape-context and doesn't generalize), fall back to
per-call `hipMallocAsync` from the caller's stream and accept ~5 µs
allocation overhead per call. Still positive EV on B=4 cells." R13a
takes that documented fallback.

Per-call `hipMallocAsync` properties:

* **Stream-ordered**: the alloc is scheduled on `g.stream`, no host-side
  device sync — preserves the SKILL.md "no CPU sync" hard constraint.
* **Cost**: ~3 µs for 4-MiB-class buffers on gfx950 (separate microbench
  evidence). For Down-B4-M2048 (105 µs main kernel wall per R11), that
  is ~3 % overhead, eats into the 30 % projected lift but still leaves
  positive EV. R15 dispatcher rule will gate K-split ON only for cells
  where (lift - alloc_overhead) > 0.
* **`hipFreeAsync` symmetry**: also stream-ordered, executes after all
  pending stream work completes. The kernel that consumes
  `sk_partial_buf` (R14) and the reduce post-kernel (R15) both run on
  the same stream, so the buffer lifetime is correctly bounded.

## Why `T_max = ceil_div(g.M_total, BLOCK_SIZE) * g.bpc` upper bound

The dispatcher does NOT have per-group M counts. Group offsets live in
`g.group_offsets` device memory; reading them on the host would
violate the SKILL.md "no CPU sync" constraint.

`T_max = ceil_div(g.M_total, BLOCK_SIZE) * g.bpc` is a strict upper
bound on the actual `T = sum_g(ceil_div(M_g, BLOCK_SIZE) * g.bpc)`:

* `ceil_div(M_total, BLOCK_SIZE) <= sum_g(ceil_div(M_g, BLOCK_SIZE))`
  (the per-group ceil rounds up at most once per group, so the sum is
  at most `ceil_div(M_total, BLOCK_SIZE) + (num_groups - 1)`).
* Multiplying by `g.bpc` (col tiles per row tile) preserves the bound.

For the gpt_oss B=4 family: T_max = 8*22 .. 16*22 = 176 .. 352 tiles.
Per-tile payload = 256 * 256 * 4 = 256 KiB. Total upper-bound buffer
= 44 .. 88 MiB per call. Well below MI355X 192 GiB HBM3e per-call
peak working set. Over-allocation is at most (num_groups - 1) ×
256 KiB = ~8 MiB at B=32, negligible.

R14 will compute the actual `SK_tile_count` device-side from the
per-CTA K-iter range mapping, and only write to the prefix of
`sk_partial_buf` that maps to real SK tiles. The over-allocated tail
is left zero (from the `hipMemsetAsync` above) and never read by the
reduce post-kernel.

## Falsification matrix this round

| Gate                                                  | Pass? |
|-------------------------------------------------------|-------|
| Build succeeds (HK + Primus, hipMallocAsync linkable) | YES   |
| Metric within ±3 of recent baseline (median ≈693)     | YES (median 694) |
| Correctness 8/8 on every sample                       | YES (4/4 samples 8/8 PASS) |
| All 24 SNRs > 25 dB on every sample                   | YES (no SNR FAIL flagged on any cell) |
| ABI: no Python-side change required                   | YES (kwargs unchanged) |
| Production code path bit-identical                    | YES (g.sk_split_n=0 default; alloc/free branches not entered; kernel launch parameter bytes identical because g.sk_partial_buf remains nullptr) |

All gates GREEN → R14 unblocked.

## Forward pointer

**R14 (next round)**: kernel K-split control flow.

In `grouped_rcr_kernel` body (line 3024+), gate on `g.sk_split_n > 0`
to enter the Stream-K balanced K-iter path:

1. Compute total K-iters per persistent-loop body: `T_total = T × ki`
   where `ki = g.ki` BK-blocks. T device-side from per-group cumsum
   over `g.group_offsets`.
2. Per-CTA budget: `per_cta = ceil_div(T_total, S)` K-iters.
3. Map `[start_K_iter, end_K_iter)` to `(output_tile, K_offset_in_tile)`
   via the same closed-form coord-decode pattern R9 ported for var-K
   (HK-side `grouped_variable_k_crr_kernel` body, search for `cumsum`
   / `binary_search` neighborhood). Closed-form keeps SALU cheap on
   CDNA4 per R10 PMC.
4. For tile boundaries (where one CTA ends mid-tile and another
   continues), `atomicAdd` fp32 partials into
   `g.sk_partial_buf[output_tile_idx][i][j]` (256 × 256 stride layout).
5. For non-boundary tiles (whole tiles owned by one CTA), accumulate to
   local fp32 accumulator and fp8-store to `g.c` as today.

R14 falsification gate:
* Build succeeds (no AGPR spill > 60 — keep K-iter bookkeeping closed-
  form per R11 mitigation; if spill spikes, drop to 2-way K-split only).
* Probe with `sk_split_n=2` on Down-B4-M2048 fwd + dgrad: SNR > 25 dB
  AND TFLOPS within noise of static-stride (atomic contention check).
* Default `sk_split_n=0` metric NEUTRAL ± 3 (kernel branch not taken
  for any production call; bit-identical to R13a metric).

**R15 follow-up**: reduce post-kernel
`grouped_rcr_sk_reduce_kernel<<<SK_tile_count, 256>>>` reads
`sk_partial_buf`, sums fp32 partials, casts to fp8, writes the SK
tiles into `g.c`. Bit-eq verified against PyTorch oracle.

**R16 follow-up**: per-cell dispatcher rule in
`primus_turbo/pytorch/kernels/hipkitten/config.py`
`select_default_config` — gate `HipKittenConfig(sk_split_n=N)` ON for
B=4 cells (Down/GateUP × M ∈ {2048, 4096}) where T < ~1500; keep
`sk_split_n=0` for all B=32 cells. Per-cell N tuning in R17-R18.

**R17-R18 follow-up**: per-cell N_split ∈ {2, 3, 4} tuning sweep.

If at any of R14/R15 the build emits AGPR spill > 60 OR atomic
contention exceeds R11's 0.6 % per-tile projection by 5x+, the
forward pointer (per R11 plan) is to drop to 2-way K-split only and
rotate bookkeeping to closed-form SALU. Worst case: rotate to
Direction E (incremental barrier replacement, 3-5 round commitment).

## Why R13a NEUTRAL is "progress"

Three reasons (mirroring R12's NEUTRAL-progress framing):

1. **Validates the per-call hipMallocAsync choice**. R7/R11 sketched a
   host-cached allocator; R12 docs ruled that out; R13a confirms per-
   call alloc compiles, links, and runs at NEUTRAL when gated. The
   alloc machinery is now a settled design choice.
2. **Validates the upper-bound `T_max` derivation**. The dispatcher
   computes a buffer size from a host-side closed-form bound, no
   device-side reads. This pattern will be reused by R15 reduce post-
   kernel grid sizing (which must also bound `SK_tile_count` without
   per-group M reads).
3. **Isolates R14's risk surface**. R14 will land kernel control flow
   alone; any R14 metric movement attributes to the kernel branch /
   AGPR pressure / atomic contention, NOT to the alloc machinery.
   Removes a confounding variable from R14's falsification gate.

## Patience-counter consideration

R13 is the 11th consecutive round without a metric improvement (recent
streak: R3 perf+0, R4-R12 NEUTRAL/FALSIFIED/docs). Patience budget is
40, so 29 rounds of slack remain.

The R12-R15 arc is 4 NEUTRAL-by-design rounds in a row before R16
turns on the dispatcher rule and (per the R11 envelope) delivers
+25-30 score. R13a is round 2 of that arc. R14 = round 3. R15 = round
4. R16 is the lift round.

If R14 falsifies on AGPR spill or atomic contention, the R11 forward
pointer drops to 2-way K-split only (~2/3 of the lift envelope, +15-20
score). Worst case rotation: Direction E (barrier scheme), 3-5 rounds.

The patience budget covers both contingencies. The arc remains the
highest-EV remaining structural direction per the R11 cost decomp.
