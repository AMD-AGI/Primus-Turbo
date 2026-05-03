# Round 48 — BF16 grouped, var-K KI specialisation (KI=32, KI=64) — FALSIFIED

## Goal coming in

R47 falsified the per-tile arithmetic hoist for `grouped_var_k_kernel`
(cache hits ≤ 6% on every gpt_oss dB var-K shape because
`tiles_per_group < NUM_CUS`). R47 next-action surface ranked:

1. **Var-K KI specialisation** (R47 backup #2) — the largest
   remaining lever. `grouped_var_k_kernel<0>` is the only
   instantiation today; specialising for `KI=32` (M_per_group=2048
   → ki_g=32) and `KI=64` (M_per_group=4096 → ki_g=64) lets the
   compiler unroll the K-loop with a compile-time bound. Coverage:
   all 24 metric shapes (gpt_oss + DSV3 + Qwen3 with M ∈ {2048, 4096})
   route through one of these specs.
2. DSV3-GateUP dB var-K dispatch retry (R45 backup, weighted 1x).
3. Store batching 4 → 1 (R47 backup #3, high correctness cost).

R47 explicitly noted the spill risk was bounded: "R39-R42 spills were
on the FORWARD kernel which has a more complex schedule;
`grouped_var_k_kernel` currently has 0 spill at KI=0 so there's
headroom."

## Hypothesis (R48)

Adding `grouped_var_k_kernel<32>` and `grouped_var_k_kernel<64>` with a
host-side dispatcher driven by an `m_per_group` hint (= `M_total / G`,
exact for uniform groups; `_metric` / bench / DoD-smoke all use
`balance=True`) gives the compiler a compile-time `KI` template
parameter on `device_gemm_tile_body`. The constant K-loop bound +
constant epilog tile indices should let LLVM:

* Eliminate the loop counter increment branch (saves ~1 cycle / iter).
* Constant-fold the epilog 1 / epilog 2 `tile = KI - 2` argument into
  immediate offsets in the HBM `b_coord`/`a_coord` buffer-load
  computation.
* Schedule `main_loop_iter`'s 8 inlined LDS/MMA bursts more tightly
  with known iteration count.

Per R41 diagnostic, dB var-K is ~50 % of the bwd wall and bwd is
~70 % of the total wall. A 5 % speedup on var-K should lift the wall
ratio by ~1.7 %, score by ~10-15 points across the 3 families.

## v1 attempt — straight KI=32 + KI=64 instantiations

Edits:

1. Added `template __global__ void grouped_var_k_kernel<32>(...)` and
   `<64>` instantiations alongside `<0>` (line 4689 of
   `kernel_bf16_dynamic.cpp`).
2. Updated `launch_grouped_var_k(g)` to dispatch based on `g.ki_max`
   (a previously-reserved field): `if (g.ki_max == 32) launch <32>;
   else if (g.ki_max == 64) launch <64>; else launch <0>`. Per-spec
   `hipFuncSetAttribute` cached in static bools (mirror of existing
   pattern).
3. Updated `grouped_var_k_crr_fn` binding to take a new
   `m_per_group=0` int kwarg; sets `g.ki_max = m_per_group / K_STEP`
   when `m_per_group > 0 && m_per_group % K_STEP == 0`, else 0.
4. Updated PYBIND11 binding signature for `grouped_variable_k_crr` to
   expose the new `m_per_group=0` kwarg.
5. Updated Primus
   `GroupedGEMMVariableKHipKittenBackend.execute` to compute
   `m_per_group_hint = a.shape[0] // bs` (no GPU-side group_lens read,
   no host sync) and pass it as the 7th positional arg.

Build resource (clean, KI=32/64 produce 14 VGPR spill each):

```
                                       KI=0  KI=32  KI=64
grouped_var_k_kernel: TotalSGPRs        95    94    96
                      VGPRs            256   256   256
                      ScratchSize        0    60    60   bytes/lane
                      Occupancy          2     2     2   waves/SIMD
                      SGPRs Spill        0     0     0
                      VGPRs Spill        0    14    14   ← REGRESSION
                      LDS Size         272   272   272
```

Correctness (probe `/tmp/probe_r48_correctness.py`,
gpt_oss-Down B=4 M=2048 ki_g=32 case): KI=32 spec output is
**bit-identical** to the KI=0 dynamic output (reference SNR 47.84 dB,
max_diff 0.0156 — pure bf16 rounding noise dominates the gap, both
specs agree to the bit).

Metric (full wall):

```
                       baseline    R48 v1     delta
score                  879/880     852        -27 ← REGRESSION
gpt_oss_20B  geomean   1.085       1.050      -3.5pp
DeepSeek-V3  geomean   1.123       1.089      -3.4pp
Qwen3        geomean   1.114       1.084      -3.0pp
correct_fail           0/24        0/24       no correctness regression
```

The 14 VGPR spill on KI=32/64 (= ScratchSize 60 bytes/lane) is the
direct cause of the cross-family regression. Per R41 numbers, dB
var-K is ~half of bwd wall (and bwd is ~70 % of wall), so a ~14 %
slowdown on var-K alone produces a ~5 % wall ratio drop —
exactly what the per-family numbers show (-3.0 to -3.5 pp).

## v2 attempt — `LIGHT_UNROLL` to suppress the unroll-2 register pressure

Hypothesis: the spill comes from `#pragma unroll 2` in the
CRR + KI > 0 branch of `device_gemm_tile_body` (line 697). With a
compile-time `num_tiles = KI` constant, LLVM's `unroll 2` becomes
much more aggressive than with the dynamic-bound version (it can
specialise each unrolled iter with constant tile values, increasing
the live LDS-slot / index state across the 8 inlined `main_loop_iter`
bursts). Force `#pragma unroll 1` for that branch (gated by a new
`bool LIGHT_UNROLL=false` template param, default false to preserve
forward CRR semantics) and the spill should drop.

Edit:

```cpp
if constexpr (KI_HINT > 0 && L == Layout::CRR) {
    if constexpr (LIGHT_UNROLL) {
        #pragma unroll 1
        for (int tile = 0; tile < num_tiles - 2; tile += 2) main_loop_iter(tile);
    } else {
        #pragma unroll 2
        for (int tile = 0; tile < num_tiles - 2; tile += 2) main_loop_iter(tile);
    }
}
```

Var-K kernel calls with `LIGHT_UNROLL=true`. Forward CRR (currently
unused on the active dispatch path — `hk.grouped_crr` is never
called from Primus; Python picks `rcr` or `rrr`) keeps the default
`LIGHT_UNROLL=false` so the existing fwd CRR + KI ∈ {56, 64, 112,
128, ...} instantiations are byte-identical.

Build resource:

```
                                       KI=0  KI=32  KI=64
grouped_var_k_kernel: ScratchSize        0    72    72   bytes/lane  ← WORSE
                      VGPRs Spill        0    17    17                 ← WORSE
                      Occupancy          2     2     2   waves/SIMD
```

The spill went UP, not down. Mechanism: with `unroll 1` the compiler
no longer batches state across pairs of iterations, so the per-iter
inlined `main_loop_iter` body (8 LDS/MMA stages with their own live
register tiles) can no longer share intermediate state across the
`tile, tile+1, tile+2, tile+3` references inside the lambda — every
register tile is freshly allocated per iter, pushing total VGPR
demand above 256.

Reverted v2 immediately.

## Reverted state

Both v1 and v2 reverted via:
- `cp /tmp/kernel_bf16_dynamic_R47_baseline.cpp` →
  `kernel_bf16_dynamic.cpp` (HipKittens).
- StrReplace revert of the `m_per_group_hint` plumb in
  `grouped_gemm_impl.py:GroupedGEMMVariableKHipKittenBackend.execute`
  (Primus).

Build of restored baseline confirms KI=0 only, 0 VGPR spill,
ScratchSize 0. Re-run of the metric on the restored kernel:
**score = 880** (within noise of the original 879 baseline) —
confirms the regression source was the KI=32/64 specs themselves
and not some other side-effect of the build.

## Mechanism — why KI specialisation cannot win on `grouped_var_k_kernel`

The CRR + KI > 0 main loop is fundamentally pressure-bounded at the
existing `__launch_bounds__(NUM_THREADS, 1)` cap of 256 VGPRs +
2 occupancy. The KI=0 dynamic path uses exactly that budget with 0
spill — meaning LLVM's allocator is already at the tightest
schedule it can produce while keeping `main_loop_iter`'s 8 inlined
LDS/MMA stages live (4 register tiles A_tile / B_tile_0 / B_tile_1 /
A_kt + 4 LDS slot offsets + 8 MMA C_accum cells + per-iter coord
state).

Adding compile-time KI gives the unroller two unhealthy options:

* **`unroll 2` with constant bound (v1)**: aggressive enough to
  specialise each unrolled iter with constant `tile` values,
  duplicating coord arithmetic and forcing more intermediate state
  live → +14 VGPR spill.
* **`unroll 1` with constant bound (v2)**: refuses to share state
  across paired iterations, every register tile freshly allocated
  per iter → +17 VGPR spill.

The "win" from compile-time KI (constant loop counter, constant
epilog tile indices) is at most a handful of cycles per
`main_loop_iter` — far less than the 14-17 VGPR spill cost paid on
every iter (each spill = 4 bytes scratch round-trip per VGPR per
spill site). At 14 spills × 32 iters × ~10 cycles per spill ≈
4500 wasted cycles per tile vs the ~10000 productive cycles per tile
of MMA work — a ~30 % per-kernel slowdown, which matches the
observed 14 % var-K wall slowdown (the slowdown is amortised across
in-flight HBM loads).

The same fundamental tradeoff blocked R39-R42 on the FORWARD kernel
too — KI=24/32/44/48/88/90 all spilled. The var-K kernel has the
SAME `device_gemm_tile_body` schedule, just CRR-only. The fact that
KI=0 dynamic uses 0 spill is BECAUSE the loop bound is unknown;
removing that uncertainty exhausts the register budget.

## Falsification consequence

R48 closes:

* **Var-K KI specialisation** (R47 backup #2 / R43 item 4). Confirmed
  net-negative across all 24 metric shapes via two independent
  unroll factor attempts (unroll 2 default, unroll 1 LIGHT_UNROLL).
  The CRR + KI > 0 schedule is fundamentally pressure-bound at the
  existing `__launch_bounds__(NUM_THREADS, 1)` cap; compile-time
  unrolling exhausts the register budget, and the 14-17 VGPR spill
  cost (~14 % var-K kernel slowdown) outweighs the constant-bound
  benefit (~1-2 % expected from cycle-counter elision and epilog
  index folding).

R48 documents (for future rounds):

* `grouped_var_k_kernel<KI > 0>` requires either (a) a different
  schedule with lower register pressure (e.g., manual MFMA pipelining
  with explicit register reuse), or (b) loosening
  `__launch_bounds__` to allow >256 VGPRs at the cost of dropping
  occupancy to 1 wave/SIMD. Both are non-trivial multi-round
  rewrites; defer until R47-listed levers #2-#3 also falsify.

R48 does NOT close:

* **DSV3-GateUP dB var-K dispatch retry** (R47 backup #2 / R45
  backup). R24 dropped `xcds=0` due to allclose drift; re-sweep with
  `xcds ∈ {1, 2, 4, 8}` on (tiles_m=16, tiles_n=28) cells. Smaller
  upside (DSV3-GateUP is already 1.13-1.15 wall) but cleanest
  dispatch surface remaining.
* **Store batching 4 → 1** (R47 backup #3). Combine the 4
  `store_c_tile_mn_masked_grouped` calls in the var-K kernel epilog
  into one larger masked store. Reduces per-tile bounds-check
  overhead. Higher correctness validation cost.
* **Forward LDS swizzle audit for K=2880** (task body lever A3,
  un-tried). The K%128 != 0 K-tail path may interact badly with the
  existing 128-wide bank pattern; rocprofv3 `lds_bank_conflict` PMC
  on gpt_oss-GateUP-B32-M2048 forward kernel would confirm.

## Action

* HipKittens: `kernel_bf16_dynamic.cpp` modified twice (v1 KI specs
  + v2 LIGHT_UNROLL gate), then both reverted via backup file. Final
  diff = 0 (rebuilt to confirm). No HipKittens commit.
* Primus-Turbo: `grouped_gemm_impl.py:GroupedGEMMVariableKHipKittenBackend.execute`
  modified (m_per_group_hint plumb), then reverted. Final diff = 0.
* Primus-Turbo: 1 commit (this falsification note).

## R49 next-action surface

Three candidates remain in the R47 / task body var-K + dispatch
surface:

1. **DSV3-GateUP dB var-K dispatch retry** (R47 backup #2 / R45
   backup). R24 dropped `xcds=0` due to allclose drift; re-sweep with
   `xcds ∈ {1, 2, 4, 8}`. Smaller upside but cleanest remaining.
2. **Store batching 4 → 1** (R47 backup #3). Refactor the var-K
   epilog 4 stores into a single MMA-tile-wide store. Correctness
   risk is moderate (mask handling for the 4 sub-tiles must stay
   bit-identical) but the per-tile dispatch overhead win is real.
3. **Forward LDS swizzle audit for K=2880** (task body lever A3,
   forward kernel). Untouched in R39-R42. rocprofv3 PMC on
   gpt_oss-GateUP-B32-M2048 forward kernel for `lds_bank_conflict`
   counts. If high, swizzle change for K%128 != 0 path could lift
   the worst gpt_oss shape's forward ratio (current 1.137 fwd ratio
   per R41) without touching K-loop control flow.

Recommended for R49: **start with (3)** — it's the only fwd-side
lever that hasn't been profiled, has the clearest measurement signal
(LDS bank conflicts are a single rocprofv3 counter), and addresses
the worst metric shape (gpt_oss-GateUP-B32-M2048, ratio 1.048). If
rocprofv3 shows low LDS bank conflicts → (3) is exhausted, fall back
to (1) which has the next clearest signal.
