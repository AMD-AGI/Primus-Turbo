---
name: round-12-A1prime-variant-2-K-split-HK-scaffolding-NEUTRAL
description: R12 lands the HK-side strict-minimum scaffolding for the R11 GREEN-LIT A1' variant-2 (Stream-K K-split) arc — two new trailing fields (sk_split_n, sk_partial_buf*) on grouped_layout_globals, no dispatcher logic, no kernel body changes. Bit-identical compile, NEUTRAL metric (4-sample median 696 vs recent ~693, well within ±3 falsification gate). HK SHA bc5df92d. Forward pointer: R13 = kernel K-split control flow + sk_partial_buf alloc on g.sk_split_n > 0 dispatch path.
type: project
---

# Round-12 — A1' variant-2 K-split: HK-side scaffolding (NEUTRAL metric, R13 unblock)

## TL;DR

R11 GREEN-LIT the multi-round A1' variant-2 (Stream-K K-split) arc with a
refined cost decomposition (mid-point +25-30 score envelope, mixed per-cell
verdict requiring an R15 dispatcher rule). R12 executes the first step
of that arc: lands two new trailing fields on the FP8
``grouped_layout_globals`` struct so R13 can wire the kernel-side K-split
control flow + per-call partial-buffer alloc against existing struct fields.

* HK commit: ``bc5df92d`` (HipKittens repo,
  ``save/fp8-progress-20260319-native-layouts``).
* Primus commit: this docs note (no PT code change this round; the new HK
  fields default-init to 0/nullptr through C++ aggregate value-init — the
  Primus pybind path leaves the Python kwargs unchanged).
* Metric: NEUTRAL (4-sample on dev GPU 3 {691, 698, 695, 697}, median 696
  vs recent rounds median 693 — well within the ±3 score R12 falsification
  gate from the R11 plan section "Falsification gates for R12").
* Correctness: 8/8 PASS on every sample, all 24 (8 shapes × 3 sections)
  SNRs above the 25 dB gate.

## What changed (HK side, ``analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp``)

Added two trailing fields on ``struct grouped_layout_globals`` (after the
existing ``int fuse_ktail_off`` Round-16 field, before the ``dim3 block()``
helper):

```cpp
int  sk_split_n;       // 0 = static stride (production); >0 = K-split active
int* sk_partial_buf;   // fp32 partial-accumulator buffer, nullptr until R13
                       // alloc lands
```

Both fields are pure declarations:
1. R12 dispatcher does NOT allocate (no ``hipMallocAsync`` / ``hipMemsetAsync``
   added to ``dispatch_grouped_rcr`` this round).
2. R12 kernel body does NOT read either field.
3. The 5 existing positional aggregate initializers in the file
   (``grouped_rcr_fn`` line 8925, ``grouped_rcr_dscale_fn`` 8955,
   ``grouped_rrr_fn`` 9034, ``grouped_rrr_dscale_fn`` 9061,
   ``test_4w_real_coords`` 9770) leave the trailing fields zero/nullptr
   via C++ aggregate value-init rules — same backward-compat property used
   by the R9 ``num_slots``, R14 ``chunk_size``, and R16 ``fuse_ktail_off``
   ABI extensions.

R12 is FIELDS-ONLY by design: the smallest atomic step that R13 can build
on, so any R12-side bug surfaces in isolation rather than tangled with
R13's kernel-side K-split logic.

## Why FIELDS-ONLY (vs R11 plan's "fields + alloc + memset")

R11's R12 PR template called for adding both the struct fields AND the
host-side allocator/memset gated on ``g.sk_split_n > 0``. R12 narrows
this to fields-only for two reasons:

1. **Allocator pattern not portable from done_counter**. R11 referenced
   the ``done_counter`` finalizer-detect counter (line ~9281) as a
   "host-side cached allocator pattern" to mirror. On re-reading, that
   counter is NOT host-cached — it's a pre-zeroed scalar fp32 device
   tensor allocated by the **caller** (Python side) and passed in via the
   pybind11 ``done_obj`` argument of ``max_abs_bf16_to_fp8_scale_fn``
   (line 9317). There is NO existing host-side cached allocator pattern
   in this file to mirror; either we plumb a buffer through pybind (more
   plumbing) or use ``hipMallocAsync`` per-call (ranges 1-5 µs overhead,
   acceptable but introduces failure modes).

2. **Smallest atomic R12 step**. With the fields declared but no allocator
   added, R13 will land the K-split kernel branch + ``hipMallocAsync``
   together as a single atomic transaction. If R13 fails, revert is one
   commit on the kernel side; the struct fields stay (cheap to keep,
   ABI-extending only). If R13 succeeds, the alloc and the kernel branch
   land together with their interlock tested in one round.

The R11 plan's falsification gate ("metric within ±3 score; SNR > 25 dB
on every shape") is satisfied by R12 fields-only; R13 will re-test
against the same gate when it adds the alloc + kernel branch.

## R13 entry brief (next round)

For round 13:

1. In ``dispatch_grouped_rcr`` (line ~7640), gate on ``g.sk_split_n > 0``
   AND a basic shape-validity check (``g.bpc > 0``, ``g.ki > 0``,
   ``g.k % K_BLOCK == 0``). On true: compute ``SK_tile_count = T - S *
   (ceil(T / S) - 1)`` where T = total tiles per call (sum over groups
   of ``ceil_div(M_g, BLOCK_SIZE) * g.bpc``) and S = ``rcr_slots``;
   ``hipMallocAsync`` an ``SK_tile_count × 256 × 256 × 4``-byte fp32
   buffer onto ``g.stream``; ``hipMemsetAsync(0)`` it; assign to
   ``g.sk_partial_buf``.

2. In ``grouped_rcr_kernel`` body, gate on ``g.sk_split_n > 0`` to enter
   the Stream-K balanced K-iter path:
   - Compute total K-iters per persistent-loop body: ``T_total = T × ki``
     where ki = ``g.ki`` BK-blocks.
   - Per-CTA budget: ``per_cta = ceil_div(T_total, S)`` K-iters.
   - Map ``[start_K_iter, end_K_iter)`` to (output_tile, K_offset_in_tile)
     via the same coord-decode pattern R9 ported for var-K (HK-side
     ``grouped_variable_k_crr_kernel`` body, search for ``cumsum`` /
     ``binary_search``).
   - For tile boundaries, ``atomicAdd`` fp32 partial into
     ``g.sk_partial_buf[output_tile_idx][i][j]`` (256×256 stride layout).
   - For non-boundary tiles (whole tiles owned by one CTA), accumulate
     to local fp32 accumulator and fp8-store to ``g.c`` as today.

3. Bit-equivalent gate: probe with ``sk_split_n=2`` (simplest non-trivial
   split) on Down-B4-M2048 fwd + dgrad; SNR > 25 dB and TFLOPS within
   noise of static-stride. If TFLOPS regresses sharply (atomic
   contention worse than the 0.6 % per-tile R11 projected), the R11
   forward pointer says "rotate K-iter bookkeeping to closed-form SALU
   (cheap on CDNA4 per R10's PMC), as the BF16 var-K coord-decode does
   (R9 port). Worst case, drop to 2-way K-split only."

4. Risk: AGPR pressure from extra K-iter bookkeeping (current main
   kernel is 256 VGPR / 37 spill, near LLVM's 256-VGPR ceiling). If
   build emits ``Spill > 60`` on the new kernel template, drop the
   K-iter bookkeeping into closed-form SALU per R11 mitigation.

R14 follow-up: reduce post-kernel
``grouped_rcr_sk_reduce_kernel<<<SK_tile_count, 256>>>`` reads
``sk_partial_buf``, sums fp32 partials, casts to fp8, writes the SK
tiles into ``g.c``. Bit-eq verified against PyTorch oracle.

R15 follow-up: per-cell dispatcher rule in
``primus_turbo/pytorch/kernels/hipkitten/config.py``
``select_default_config`` — gate ``HipKittenConfig(sk_split_n=N)``
ON for B=4 cells (Down/GateUP × M ∈ {2048, 4096}) where T < ~1500;
keep ``sk_split_n=0`` for all B=32 cells.

## Why R12 NEUTRAL is "progress"

The R11 GREEN-LIT analysis confirmed Stream-K is the highest-EV
remaining structural direction. R12 lands the smallest atomic step
toward implementing it — a strict prerequisite for R13's kernel
control flow. The metric staying NEUTRAL within ±3 confirms:

1. The struct extension is ABI-safe (no caller broke).
2. The compile is bit-identical at the kernel level (kernel reads
   neither new field, codegen unchanged).
3. The dispatch path is unchanged (no allocator added, no new branch).

This isolates R13's risk: any metric movement in R13 attributes to the
kernel branch + allocator, NOT to the struct extension itself. R12 is
"progress" because it removes a confounding variable from R13's
falsification gate.

## Falsification matrix this round

| Gate                                                  | Pass? |
|-------------------------------------------------------|-------|
| Build succeeds (HK + Primus)                          | YES   |
| Metric within ±3 of recent baseline (median ≈693)     | YES (median 696) |
| Correctness 8/8 on every sample                       | YES (4/4 samples 8/8 PASS) |
| All 24 SNRs > 25 dB on every sample                   | YES   |
| ABI: no Python-side change required                   | YES (kwargs unchanged) |

All gates GREEN → R13 unblocked.

## Forward pointer

R13: kernel K-split control flow + sk_partial_buf alloc on
``g.sk_split_n > 0`` dispatch path. SNR-gated. See "R13 entry brief"
above for the implementation outline.

R14: reduce post-kernel.

R15: per-cell dispatcher rule (B=4 cells ON, B=32 cells OFF).

If at any of R13/R14 the build emits AGPR spill > 60 OR atomic
contention exceeds R11's 0.6 % per-tile projection by 5x+, the
forward pointer (per R11 plan) is to drop to 2-way K-split only and
rotate bookkeeping to closed-form SALU. Worst case: rotate to
Direction E (incremental barrier replacement, 3-5 round commitment).
