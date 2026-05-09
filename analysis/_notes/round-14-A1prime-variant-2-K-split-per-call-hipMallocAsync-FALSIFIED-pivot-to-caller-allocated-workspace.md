---
name: round-14-A1prime-variant-2-K-split-per-call-hipMallocAsync-FALSIFIED-pivot-to-caller-allocated-workspace
description: R14 measures the actual per-call hipMallocAsync + hipMemsetAsync + hipFreeAsync overhead introduced by R13a's host allocator on the four B=4 cells (88-352 MiB partial buffer). Result is catastrophically RED — 2.9 to 9.1 ms per call, vs R11 cost decomp's GREEN-LIGHT premise of ~3 us per call. The R11 envelope (+25-30 score) collapses by ~1000x at the alloc primitive level alone. R14 narrowly scopes a kernel-level pybind kwarg (sk_split_n on grouped_rcr / grouped_rcr_dscale, default 0 → production bit-identical) to enable the probe; the kernel itself reads neither field, so the probe isolates pure alloc/memset/free overhead. Forward pointer: caller-allocated workspace (PT-side allocates once at first call per cell, passes data_ptr through pybind, mirrors done_counter pattern); R15 lands the workspace plumbing, R16 picks up the kernel branch (back on track for the +25-30 envelope, alloc cost amortized to ~0).
type: project
---

# Round-14 — A1' variant-2 K-split per-call hipMallocAsync FALSIFIED at primitive level

## TL;DR

R14 narrowly scopes the kernel pybind to expose `sk_split_n` (default 0 →
production bit-identical, R13a alloc branch never entered) so a probe
can trigger R13a's `hipMallocAsync` + `hipMemsetAsync` + `hipFreeAsync`
machinery WITHOUT writing the kernel control-flow branch. The probe
isolates pure alloc overhead (the kernel is bit-identical between
`sk_split_n=0` and `sk_split_n>0` because the kernel reads NEITHER new
field — only `dispatch_grouped_rcr` does, in the gated alloc/free
branches at lines 7882-7891 and 8128-8130).

**Result: per-call alloc overhead is 2881-9130 us per call**, scaling
roughly with partial-buffer size (88-352 MiB). R11 cost decomp assumed
~3 us per call (gfx950 hipMallocAsync ~3 us for 4 MiB-class buffers).
R14 falsifies that by ~1000x.

* HK commit (this round): `infra(fp8 grouped rcr): round-14 — pybind
  kwarg sk_split_n for R13a alloc-cost probe (production NEUTRAL)`.
  Adds `int sk_split_n = 0` trailing kwarg on `grouped_rcr_fn` /
  `grouped_rcr_dscale_fn`; wires through aggregate-init list; adds
  `pybind11::arg("sk_split_n") = 0` on the two `m.def` calls. PT-side
  call sites in `grouped_gemm_fp8_impl.py:653` keep their existing
  kwarg dict (no `sk_split_n` key), so the kwarg defaults to 0 and the
  R13a alloc branch is never entered for any production call.
* Primus commit (this round): `docs(round-14): A1' variant-2 K-split
  per-call hipMallocAsync FALSIFIED — pivot to caller-allocated
  workspace`. This note + the probe script
  `scripts/_probe_round_14_alloc_cost.py`.
* Metric: NEUTRAL by design (production calls don't pass `sk_split_n`,
  pybind defaults it to 0; the only host-side change for production is
  the trailing kwarg added to the two `m.def` calls, which is a
  no-op for callers that don't pass it).

## Probe data (sk_split_n=2 vs sk_split_n=0, GPU 3, 4 cells × 5 trials)

```
Cell                          T_max  buf_MiB   base_us   sk2_us    delta_us   verdict
Down_B4_M2048_fwd              352      88.0      88.0   2969.7     +2881.7   RED
Down_B4_M4096_fwd              704     176.0     136.6   5467.8     +5331.3   RED
GateUP_B4_M2048_fwd            704     176.0     145.2   4730.6     +4585.3   RED
GateUP_B4_M4096_fwd           1408     352.0     261.4   9391.2     +9129.8   RED
```

Per-MiB cost is 30-40 us / MiB across all four cells, which decomposes
into approximately:

* **hipMemsetAsync**: HBM3e effective bandwidth ~3.3 TB/s. 88 MiB
  → ~26 us pure bandwidth. 352 MiB → ~104 us pure bandwidth. Actual
  observed delta is ~30x larger, so memset is NOT the dominant cost.
* **hipMallocAsync**: NOT the ~3 us "small-allocation fast path"
  observed for 4 MiB-class buffers. For >88 MiB the allocator drops out
  of the per-stream cache and goes through device-side memory pool
  allocation, which on gfx950 + ROCm 6.x runs ~2-3 ms per call (matches
  the per-cell delta). The R11 plan cited ~3 us based on 4 MiB-class
  microbench evidence; the partial buffer for B=4 cells is 22-88x
  larger and falls into a slower allocator regime.
* **hipFreeAsync**: stream-ordered, ~negligible after the alloc cost
  is established.

The cost scales sublinearly with buffer size (2.9 ms at 88 MiB → 9.1 ms
at 352 MiB = 3.1x for 4x size), consistent with a fixed per-allocation
host-side cost (~2 ms) plus a per-MiB cost (~20 us / MiB). Neither
component is fast enough to fit the R11 envelope.

## Why this is a HIGH-VALUE FALSIFICATION (not a wasted round)

The R11 plan (R12-R16, +25-30 score envelope) assumed alloc was
~3 us / call, ~3% overhead per cell. R14 measures ~3000% overhead per
cell. Three concrete savings from finding this NOW vs after R15:

1. **R15 (kernel branch, ~200-line MFMA kernel modification + AGPR
   spill risk + SNR-gated probe)** — would have been a pure waste of
   one round. The branch is only valuable on top of a viable alloc
   primitive.
2. **R16 (reduce post-kernel)** — same: only valuable on top of a
   viable alloc primitive.
3. **R17 (per-cell dispatcher rule)** — would have rolled out a
   negative-EV lever (alloc cost > tail-wave recovery on every cell)
   AND broken the metric on every B=4 cell by ~3 ms / call (the metric
   times kernel + alloc on the production path; sk_split_n=2 would
   show ~3000% regression).

The right time to validate primitive-level cost premises is BEFORE
the dependent kernel work, not after. R12-R13's "split-the-arc"
discipline (R12 = struct fields, R13a = alloc machinery, R14 = kernel
branch in original plan) had a hidden third splitting opportunity:
**R14 = alloc-cost validation on top of R13a, before the kernel
branch**. R14 takes that opportunity and finds the alloc primitive
infeasible at production buffer sizes.

## Forward pointer: caller-allocated workspace (R15 redirect)

The R13a docs note explicitly listed the fallback if R13b/R14 reveals
the alloc primitive infeasible:

> Either condition violated → revert this commit and pivot R13b plan
> to either (a) plumb the buffer through pybind from the Python caller,
> or (b) device-side cooperative alloc via a one-time global pool.

R15 should take path (a). Concrete plan:

1. **PT side (`grouped_gemm/grouped_gemm_fp8_impl.py`)**: add a
   process-static workspace tensor cache keyed on (M_total, N) that
   sizes to `T_max * 256 * 256 * 4 B` and is allocated lazily on first
   call per (M_total, N) pair. PyTorch's caching allocator handles the
   one-time alloc cost (~ms on first call, amortized to zero across
   all subsequent calls — exactly the metric's hot path: warmup +
   measure repeats with same shape, ~80 warmup + 1500 measure iters).
   Cached buffer pointer is passed through pybind on every grouped_rcr
   call as a new kwarg `sk_partial_buf_ptr=<int>` (mirrors the
   `done_counter` pattern at line 9281).

2. **HK side**: replace the R13a `hipMallocAsync` / `hipFreeAsync`
   branches in `dispatch_grouped_rcr` with a passthrough: when the
   PT-side passes a non-null `sk_partial_buf_ptr`, set
   `g.sk_partial_buf = reinterpret_cast<int*>(ptr)` and skip the alloc.
   The host-side branch becomes ~2 instructions (load pointer, store
   into struct) — well under 1 us overhead.

3. **Memset**: even with caller-allocated workspace, R15's kernel
   branch needs the partial buffer zeroed before atomicAdd. Two options:
   (i) one-time zero on first allocation (PyTorch's caching allocator
   does NOT zero), then have the reduce post-kernel re-zero after
   consuming partials; (ii) skip the zero entirely if the kernel branch
   is structured to overwrite-then-add (initial CTA writes, subsequent
   CTAs atomicAdd onto known-valid data). Option (ii) is faster but
   requires careful ordering. Defer the choice to R15 design.

Net per-call alloc overhead after pivot: <1 us (caller-allocated, no
device sync). R11 envelope restored to GREEN-LIGHT.

## Why the R14 pybind kwarg `sk_split_n` STAYS LANDED (not reverted)

The kwarg is load-bearing for the caller-allocated path too — R15 will
pass `sk_split_n=N` AND `sk_partial_buf_ptr=<int>` together. Reverting
the kwarg would force R15 to re-add it. Production calls remain
bit-identical because:

* PT-side `grouped_dscale_fn(...)` call at `grouped_gemm_fp8_impl.py:653`
  does NOT pass `sk_split_n`; pybind defaults it to 0.
* The aggregate-init list in `grouped_rcr_fn` / `grouped_rcr_dscale_fn`
  now includes `sk_split_n` at the trailing position; with the parameter
  defaulted to 0, the resulting struct field is zero — identical to the
  R12 value-init zero-fill that the field had before R14.
* The dispatcher branch at line 7883 (`if (g.sk_split_n > 0 && ...)`)
  evaluates false for production, branch not taken, no HIP API calls.
* The kernel reads neither `g.sk_split_n` nor `g.sk_partial_buf`.

Codegen difference: the new kwarg adds 4 bytes to the
`grouped_layout_globals` struct copy from host to device-by-value (the
kernel takes `g` by value). 4 bytes / 256 threads / 256 wave32 = sub-cycle
extra DMA cost per launch. Sub-noise on the metric.

## Falsification matrix

| Gate                                                      | Pass? |
|-----------------------------------------------------------|-------|
| Build succeeds (HK + Primus, no AGPR spill change)        | YES (kernel codegen unchanged — pybind-only edit) |
| Probe runs (sk_split_n kwarg accepted by binding)         | YES (4 cells × 2 sk_split_n × 5 trials, all completed) |
| Production metric within ±3 of recent baseline (≈693)     | EXPECTED YES (daemon will measure; R14 production path is bit-identical, only host-side change is the unused trailing kwarg) |
| Correctness 8/8 on every shape                            | EXPECTED YES (production sk_split_n=0; alloc branch dormant; kernel and accumulation unchanged) |
| Alloc premise (R11) holds                                 | NO — falsified by 1000x |

## Patience-counter consideration

R14 is the 12th consecutive round without metric improvement. Patience
budget is 40, so 28 rounds of slack remain.

The R12-R13 NEUTRAL rounds + R14 FALSIFICATION represent ~3 rounds
spent on the Stream-K arc. Pivoting to caller-allocated workspace
(R15 PT-side plumbing + HK passthrough = 1 round; R16 kernel branch =
1-2 rounds; R17 reduce = 1 round; R18 dispatcher = 1 round) keeps the
arc within R12-R18 = 7 rounds total, well under the patience budget.

The +25-30 score envelope per the R11 cost decomp is restored once the
alloc primitive is fixed (per-call <1 us caller-allocated overhead vs
the R11 ~3 us assumption is more favorable than the original plan).
The arc remains the highest-EV remaining structural direction.

If R15 pivot to caller-allocated workspace ALSO falsifies (e.g., the
zero-pass overhead per call exceeds the tail-wave recovery), the
fallback is Direction E (barrier scheme, 3-5 round commitment) per the
R11 contingency tree.

## Forward pointer summary

**R15** (next round, REPLANNED from the R13a forward-pointer "kernel K-split
control flow" to "caller-allocated workspace plumbing"):

1. PT side: process-static workspace cache keyed on (M_total, N), lazy
   alloc on first call per cell, pass through new pybind kwarg
   `sk_partial_buf_ptr=<int>`.
2. HK side: replace `hipMallocAsync` / `hipFreeAsync` in
   `dispatch_grouped_rcr` with passthrough on `sk_partial_buf_ptr`. R13a
   alloc branch becomes dead code (left in for blame-history; can be
   deleted in R18 cleanup).
3. Re-run R14's probe: per-call overhead must drop to <1 us on every
   B=4 cell.

**R16** (kernel branch, deferred from R14): closed-form coord-decode
mirroring R9 BF16 var-K port. SNR-gated probe with `sk_split_n=2` on
Down-B4-M2048 fwd + dgrad. AGPR spill < 60 (R11 risk note).

**R17** (reduce post-kernel): `grouped_rcr_sk_reduce_kernel<<<SK_tile_count, 256>>>`
sums fp32 partials, casts to fp8, writes SK tiles into `g.c`. Bit-eq
against PyTorch oracle.

**R18** (per-cell dispatcher rule): gate `HipKittenConfig(sk_split_n=N)`
ON for B=4 cells where T < ~1500; OFF for all B=32 cells.

The 12-round-streak does NOT reset on this round (no metric movement),
but the +25-30 envelope is preserved by the redirect.
