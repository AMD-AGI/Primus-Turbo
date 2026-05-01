# Round 19 (death-march) — FP8 grouped two-tile main-loop port: catastrophic regression from code-size-induced VGPR spill

## Round 18 plan revisited

Round-18 falsified the SRD-hoist (option 2) and re-ranked the round-17
option list. The new #1 priority was "option 1: b0/b1 reload / accumulator
split" — a multi-round VGPR-spill attack. This round explored an ALTERNATE
option-1 lever that's a single-round port: bringing the **two-tile main
loop** (8 mma/iter, 4 LDS slots) from the FP8 dense kernel into the FP8
grouped kernel.

## Hypothesis

The grouped FP8 `grouped_rcr_kernel` only has a **single-tile** main loop
body (4 mma/iter, 2 LDS slots, lines 2174-2202 of `kernel_fp8_layouts.cpp`).
The dense `gemm_kernel<RCR>` has BOTH single-tile (lines 1397-1426) AND
two-tile (lines 1325-1395) bodies, gated at `RCR_TWO_TILE_MIN_KI = 28`.

Round-15 falsified lowering this threshold from 28 → 20 on the **dense**
kernel for ki=22 (NOOP, ±0.23% noise). But round-15 explicitly noted:
*"compile-time codegen is identical, runtime branch only"* — the dense
kernel already had both paths compiled, so changing the threshold just
changed which body executed at runtime. No spill/codegen cost.

For grouped, two-tile is **net-new code**. Hypothesis: porting the
two-tile body into grouped should give grouped ≥1.20× Triton on shapes
with even ki ≥ 16 (DSV3-Down ki=16, gpt_oss-K2880 ki=22, gpt_oss-K4096
ki=32 — i.e. ALL 16 grouped FP8 metric shapes). The deeper pipeline
should hide more LDS bubbles per mma issue.

## Implementation (~97 lines added, then reverted)

In `grouped_rcr_kernel` body, around the existing single-tile main loop
(line 2174), wrapped with a runtime branch:

```c++
if ((ki_dyn & 1) == 0 && ki_dyn >= 16) {
    auto main_loop_iter = [&](int tile) {
        // 8-mma body: cA, cB, cC, cD on (tile) then cA, cB, cC, cD on (tile+1)
        // mirrors dense kernel_fp8_layouts.cpp:1326-1387
        // uses Bs[0][0..1] / Bs[1][0..1] / As[0..1][0..1] directly (no tic/toc)
        load_b(b0, Bs[0][0], wn);
        load_a(a, As[0][0], wm);
        rcr_8w_load_hoist<...>(As[1][1], g.a, a_co(br*2+1, tile+1), soA);
        ... 8 mma + 8 prefetches + 8 barriers ...
    };
    TK_PRAGMA_UNROLL(RCR_MAIN_UNROLL)
    for (int tile = 0; tile < ki_dyn - 2; tile += 2) {
        main_loop_iter(tile);
    }
    TK_WAIT_VMCNT(0); __builtin_amdgcn_s_barrier();
} else {
    // existing single-tile body unchanged
}
```

Threshold rationale: 16 covers all 16 grouped FP8 metric shapes (DSV3
ki=16, gpt_oss ki=22 / 32 all even), so two-tile fires for everything.
The dense empirical break-even of 28 is for dense's per-launch overhead
amortization; grouped's persistent kernel pays per-tile overhead, so the
break-even might be lower.

After main loop, tic=0/toc=1 (initial state, since two-tile body doesn't
swap). Bs[0][0..1] holds tile=ki_dyn-2 and Bs[1][0..1] holds tile=ki_dyn-1
— matches the existing epilog 1's `b_tile(tic=0, ...)` and epilog 2's
`b_tile(toc=1, ...)` expectations.

## Build resource usage — **structural spill increase**

`grouped_rcr_kernel<0,0,0>` (vanilla DSV3 template spec):

| Metric | Round-18 baseline | Round-19 two-tile port |
|---|---|---|
| TotalSGPRs | 65 | **70** (+5) |
| ScratchSize bytes/lane | 272 | **328** (+56) |
| VGPRs Spill | 67 | **81** (+14) |
| LDS bytes/block | 139796 | 139796 |
| Occupancy waves/SIMD | 2 | 2 |

The added two-tile body forced the compiler to spill 14 more VGPRs and
allocate 56 more bytes of scratch per lane. Per-thread cost is paid even
for shapes that would take the runtime `else` branch — but in this metric,
ALL 16 shapes hit the two-tile branch (all have even ki ≥ 16).

## Wall-time — **catastrophic 37% regression**

3 metric runs after rebuild:

| Run | score |
|---|---|
| 1 | 523 |
| 2 | 522 |
| 3 | 518 |
| **mean** | **521 (-37% vs 824 baseline)** |

Per-shape (single representative run):

| Shape | Baseline ratio | Two-tile ratio | Δ |
|---|---|---|---|
| DSV3-Down-B16-M2048 | 0.966 | 0.602 | -37.7% |
| DSV3-Down-B16-M4096 | 0.957 | 0.579 | -39.5% |
| DSV3-GateUP-B16-M4096 | 1.004 | 0.517 | -48.5% |
| DSV3-GateUP-B32-M4096 | 1.054 | 0.531 | -49.6% |
| gpt_oss-Down-B4-M2048 | 1.025 | 0.767 | -25.2% |
| gpt_oss-GateUP-B32-M4096 | 0.979 | 0.672 | -31.4% |

**Every single shape regressed 25-50%.** Zero correctness failures
(`correct_fail=0/16`, `reject=0/16`) — the port is numerically correct.
The regression is purely from the +14 VGPR spill / +56 byte scratch cost
of the larger compiled kernel.

## Falsification root cause

Round-15 explicitly warned: *"compile-time codegen is identical, runtime
branch only"* for dense. That note quantified the dense path's cost as
zero because the dense kernel was already compiling both paths. For
grouped, the two-tile path was net-new code → the compiler had to
register-allocate across a 50% larger function body → hit a hard spill
cliff.

The grouped kernel was ALREADY at 67-VGPR-spill / 272-byte-scratch
baseline (round-17 rocprof's `SQ_INSTS_FLAT 2.99M` localized this). It is
operating at a **spill cliff**: adding ANY net-new instruction stream to
the function body provokes a non-linear spill regression because the
register allocator's heuristic flips from "barely spilling" to "deeply
spilling".

This implication generalizes:

> **Any intervention that ADDS code to `grouped_rcr_kernel` (rather than
> REPLACING existing code) has a non-trivial chance of triggering the same
> 14-VGPR-spill cliff.** The kernel cannot absorb more code without losing
> 30-50% of its performance.

This is a stronger falsification than just "two-tile doesn't help": it
falsifies the whole **option-1 family** of "add a register-tile-rewrite
alongside the existing path" approaches. The only safe interventions are:

1. **Replace** the existing main loop body (no net code addition).
2. **Reduce** kernel code size (might tip back below the spill cliff and
   improve baseline).
3. **Template-specialize** alternative paths so each instantiation only
   compiles ONE body (zero runtime branch, zero net code growth per spec).

## Implication for round 20+

Re-rank with the spill-cliff constraint:

| Option | Round-18 priority | Round-19 evidence | New priority |
|---|---|---|---|
| Two-tile port (this round) | not in round-18 doc | **-37% catastrophic regression, falsified** | CUT |
| **Template-specialize two-tile** as `<KI_HINT, N_MASKED_STORE, FUSED_KTAIL, USE_TWO_TILE>` | new option | reduces per-spec code size; each spec only compiles ONE main loop body | **#1 candidate** |
| **Reduce existing kernel code size** (e.g. fold N-tail masked C-store + FUSED_KTAIL into helper templates) | new option | might tip back below spill cliff, improving baseline | **#2 candidate** |
| Option 1 (b0/b1 reload pattern, register tile rewrite) | round-18 #1 | risky if implemented additively; only safe if it REPLACES existing reg-tile pattern | demoted to #3 |
| Option 3 (prefetch depth tuning port from BF16) | round-18 #2 | open, but requires careful code-size accounting | #4 |
| Option 4 (N-tail masked C-store port from BF16) | round-18 #3 | requires code-size accounting | #5 (gpt_oss only) |

## What to try next round (round 20)

Two recommended single-round directions, ranked:

**A. Template-specialize the two-tile body** (revisit this round's port,
but as a compile-time gate rather than runtime branch). Add a 4th
template parameter `bool USE_TWO_TILE` to `grouped_rcr_kernel`. Each
spec only compiles ONE main loop body (single-tile OR two-tile, never
both). Then dispatch picks the right spec per shape (`USE_TWO_TILE = true`
when `ki_dyn % 2 == 0 && ki_dyn >= 16`, else `false`). Risk:
template-spec count doubles (currently 4: `<0,0,0>`, `<0,1,0>`, `<0,0,1>`,
`<0,1,1>` — would become 8). Each spec compiles independently with its
own register allocation, so two-tile spill cost is contained. Expected:
two-tile-spec gets ~67 spill (same as current single-tile) since the
function body is similar size, and the deeper pipeline could give +1-3
score on DSV3-Down shapes. If neutral, cut and move to B.

**B. Audit kernel code-size and reduce** (reduce the 67-VGPR-spill
baseline). Concrete candidates:
* Move FUSED_KTAIL block into a separate `__device__` helper called from
  `grouped_rcr_kernel` body (probably no-op since `__forceinline` will
  re-inline it, but worth probing).
* Move N-tail masked C-store path into a helper.
* Audit the prologue's `s_offs` cumsum init — see if more compact code
  is possible.
* Revisit the 6-step branch-free binary search — could it be a single
  iteration if g.G ≤ 8?

## What stays untouched

* Round-17 / round-18 rocprof findings still valid (FLAT scratch is the
  bottleneck; this round CONFIRMED that ADDING code amplifies the
  bottleneck).
* Architecture: persistent single-launch, no host sync, no per-group
  loops — INVARIANT.
* All previous falsifications still hold.

## Falsified levers (cumulative, for next round's prompt)

1. BF16 K_STEP=64→32 port (round 26) — out of scope.
2. FP8 chunk_size sweep (round 22) — dead-end.
3. FP8 MFMA cell-shape migration (rounds 14-15) — falsified by rocprof.
4. FP8 main-loop barrier/setprio/vmcnt micro-knobs (rounds 13-16) — saturated.
5. FP8 grouped SRD hoist for SALU reduction (round 18) — compiler CSEs;
   SALU not on critical path.
6. **FP8 grouped two-tile main-loop port via runtime branch (THIS ROUND,
   round 19)** — net-new code triggers +14 VGPR spill cliff → -37%
   catastrophic regression. Future grouped main-loop alternatives MUST
   be either (a) compile-time template-specialized OR (b) replace
   (not add to) existing code.

## Status

**Reverted**. Doc-only commit in Primus-Turbo. No HipKittens commit this
round (kernel changes were tested, confirmed -37%, reverted in-tree).

Round-19 metric SHA: this commit (Primus-Turbo).
HK SHA: unchanged.

Score after revert: 818 (within noise of round-18 baseline 824).
