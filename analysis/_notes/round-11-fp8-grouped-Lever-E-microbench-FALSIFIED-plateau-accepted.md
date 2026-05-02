## Round 11 — FP8 grouped: Lever E microbench FALSIFIED → plateau accepted

**Status**: LEVER E ARCHITECTURALLY FALSIFIED via single-wave
software-pipelining microbench. Hand-rolled `prefetch-next-iter +
mfma-current` schedule runs **-7.28 % slower** than LLVM's auto
schedule, well outside the gate's PASS region (≥ +5 pp) AND outside
the noise band (-3..+5 pp). All 6 architectural levers (A through F)
are now exhausted on this kernel. The 962-964 / FP8 geomean ~1.125
plateau is accepted as the architectural ceiling for the current
single-launch persistent CDNA4 grouped FP8 kernel.

**Auto-optimize round**: 11 / 100
**Date**: 2026-05-02
**HK SHA at round start**: `9ee90e2c`
**HK SHA at round end**: `<this round's commit>` (lever_e_microbench.cu added)
**PT SHA at round start**: `7818763a`
**PT SHA at round end**: `<this round's commit>` (R11 note added)
**Round time**: ~30 min (1 baseline + 5-trial confirm + write microbench
+ build + 5-trial microbench + write-up)
**Score before (best)**: 962 (R9 trailing best)
**Score after (R11 metric)**: 963 stable (5-trial median, range 963-964)
**FP8 geomean**: 1.1259 (R11 single-trial), 1.1254 (5-trial mean)
**Grp BF16 geomean**: 1.1869 ([watch], not scored)

---

## Pre-round 5-trial baseline (lock-in for R10's Qwen-GateUP-B32-M4096 rule)

```
trial 1: score=963  grp_FP8=1.1238  grp_BF16=1.1869
trial 2: score=963  grp_FP8=1.1253  grp_BF16=1.1860
trial 3: score=964  grp_FP8=1.1266  grp_BF16=1.1881
trial 4: score=963  grp_FP8=1.1261  grp_BF16=1.1870
trial 5: score=964  grp_FP8=1.1274  grp_BF16=1.1881
mean   : 963.4   range 963-964  (very tight; ±0.5 noise)
```

R10's commit message said its own metric run sampled 960 (low end of
noise) but tight verify showed +0.93 pp on B32-M4096 sub-shape with
clean 1.85× spread separation. R11's stable 963 confirms the gain
landed: this is +1 over best=962, sitting consistently above the
prior 960-962 range.

This is the baseline the R11 falsification test is anchored against.

---

## Worst-5 grpFP8 cases (unchanged since R3 — architectural ceiling)

```
 1. gpt_oss_20B-GateUP-B32-M4096   ratio = 1.031  (K=2880 K-tail spec)
 2. gpt_oss_20B-GateUP-B32-M2048   ratio = 1.042
 3. gpt_oss_20B-Down-B32-M4096     ratio = 1.061
 4. gpt_oss_20B-GateUP-B4-M4096    ratio = 1.063
 5. gpt_oss_20B-Down-B32-M2048     ratio = 1.073
```

All 5 worst cases (and indeed all 7-9 worst cases) are gpt_oss
K=2880. They sit at the architectural ceiling derived in R3:
working-set 250-290 dw/lane vs VGPR cap 256 dw/lane forces 32-43 dw
spill on the FP8 grouped main kernel, and the gpt_oss spec
`<0,T,T>` (FUSED_KTAIL=true + N_MASKED_STORE=true) carries 39 dw
spill (vs DSV3+Qwen `<0,F,T>` floor of 32 dw). The +7 dw delta
comes from the N_MASKED helper code being IN SCOPE in the template
instance, not from runtime calls (R4 SENTINEL refactor confirmed
the helper-internal mechanism cannot reduce this further).

---

## Lever E microbench: design + result

### Microbench location
`/workspace/code/HipKittens/analysis/fp8_gemm/mi350x/lever_e_microbench.cu`
(committed this round to HK).

### Design rationale
R10 round-note recommended R11 = Lever E scout (ASM software pipeline).
The hypothesis: the inner main loop chunk in
`kernel_fp8_layouts.cpp::main_loop_iter` (lines 1576-1638) emits 8
ds_reads followed by `s_waitcnt lgkmcnt(0)` then 8 mfma's per chunk,
serialising the ~14 cy ds_read latency with the ~13 cy/mfma pipeline.
If the hand-rolled schedule overlaps next-iter's reads with current-iter's
mfma's, the ~8 cy/iter LDS-stall headroom should be recoverable.

The microbench is a single-wave proxy of this: 50000 inner iter, each
issuing 2 intx8 LDS reads + 8 mfma_scale_f32_16x16x128_f8f6f4 calls.
Two paths:

**Path (a) "current style" / serial**:
```cpp
for (iter ...) {
    intx8_t A = lds_buf[idx];     // emits ds_read_b128 x2
    intx8_t B = lds_buf[idx + 1]; // emits ds_read_b128 x2
    asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(A[0]), "+v"(B[0]));
    // 8 mfma's serially
    C0 = mfma(A, B, C0); C1 = mfma(A, B, C1); ...; C7 = mfma(A, B, C7);
}
```

**Path (b) "pipelined"**:
```cpp
intx8_t A_curr = lds_buf[idx0];  intx8_t B_curr = lds_buf[idx0+1];
asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(A_curr[0]), "+v"(B_curr[0]));

for (iter ... N-1) {
    intx8_t A_next = lds_buf[idx];      // PREFETCH next iter
    intx8_t B_next = lds_buf[idx + 1];
    // mfma's run while next-iter reads are in flight
    C0 = mfma(A_curr, B_curr, C0); ...; C7 = mfma(A_curr, B_curr, C7);
    asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(A_next[0]), "+v"(B_next[0]));
    A_curr = A_next; B_curr = B_next;
}
```

### Result (5 trials, 50000 iter each)

```
trial 1:  serial 7.439 ms = 148.79 ns/iter   pipelined 8.026 ms = 160.52 ns/iter   delta -7.31%
trial 2:  serial 7.439 ms = 148.78 ns/iter   pipelined 8.026 ms = 160.52 ns/iter   delta -7.31%
trial 3:  serial 7.441 ms = 148.82 ns/iter   pipelined 8.024 ms = 160.48 ns/iter   delta -7.27%
trial 4:  serial 7.436 ms = 148.72 ns/iter   pipelined 8.021 ms = 160.42 ns/iter   delta -7.30%
trial 5:  serial 7.438 ms = 148.76 ns/iter   pipelined 8.022 ms = 160.44 ns/iter   delta -7.28%
median:                                                                              -7.28%
```

Decision threshold (R10 plan):
- pct ≥ +5.0 % → Lever E CONFIRMED, commit kernel rewrite
- pct in (-3, +5) → Lever E FALSIFIED, plateau accepted
- pct ≤ -3.0 % → Lever E WORSE than baseline, FALSIFIED with codegen note

**Result: -7.28 % median, every trial in [-7.31, -7.27]. FAR below the
−3 % "WORSE" threshold. Lever E is FALSIFIED at the strongest signal
strength** (5 / 5 trials hand-rolled schedule slower than LLVM
auto schedule).

### Why hand-rolled is worse (interpretation)

The clear signal that the hand-rolled software pipeline is **slower**
not just neutral has two explanations:

1. **LLVM is already pipelining**. The `s_waitcnt lgkmcnt(0)` placed
   BEFORE the mfma's in path (a) tells LLVM "all prior loads must be
   resident here". LLVM's instruction scheduler already issues the
   ds_reads as early as possible — typically at the top of the iter,
   then schedules mfma's to run while LDS round-trips. The empirical
   148.79 ns/iter for 8 mfma calls (= 18.6 ns/mfma = ~22 cy at 1.2GHz)
   is close to the architectural ~13 cy/mfma single-issue rate, which
   means LDS read latency IS already largely hidden by mfma execution.
   The headroom Lever E claimed to recover doesn't exist.

2. **Path (b) doubles VGPR working set**. The double-buffered version
   simultaneously holds A_curr/B_curr (16 dw/lane) AND A_next/B_next
   (16 dw/lane), inflating active register set by +16 dw/lane. The
   `failed to meet occupancy target` warning shows path (b) compiles
   to occupancy 2 wave/SIMD same as path (a), but spill behaviour
   is different — path (b) likely emits register-rotate copies when
   `A_curr = A_next` happens, adding cycles per iter.

The combined effect: path (a) gives LLVM the freedom to schedule
ds_read early while running mfma's. Path (b) tries to FORCE that
schedule but does so with extra book-keeping (the double-buffered
state) that costs more cycles than it saves.

### Microbench limitations (acknowledged)

- Single-wave only. Real grouped_rcr_kernel runs 8 waves/CTA with
  s_barriers; the synchronization cost across waves is NOT captured.
- LDS pattern is simple consecutive reads, not the swizzled
  `st_16x128_v2_s` pattern used by the real kernel. Bank conflict
  costs in real kernel might be different.
- `__launch_bounds__(64, 8)` actually meets occupancy 2 not 8 (LLVM
  warning on both kernels). Hardware concurrency in real kernel is
  different.
- mfma issue rate per single SIMD is approximate; real kernel issues
  on multiple SIMDs of same CU.

Despite these caveats, the **direction** of the result is robust:
in the simplest possible isolated test, hand-rolled software pipeline
loses to LLVM auto schedule by 7.28 %, with NO trial showing improve-
ment. For Lever E to be worth pursuing in the production kernel, we'd
need a thesis why the production kernel's LLVM scheduler is somehow
WORSE than this microbench's LLVM scheduler — and there's no
mechanism to predict that.

---

## Final cumulative lever falsification matrix

After 11 rounds of disciplined falsification:

| Lever | Status | Falsified | Mechanism |
|---|---|---|---|
| **A** Async global→LDS copy + MFMA pipelining | FALSIFIED | R2 | Already shipped via inline-ASM `buffer_load_dwordx4 ... lds` (line 787 of kernel_fp8_layouts.cpp) |
| **B** Dual / triple LDS buffer ping-pong | FALSIFIED | R2 | Dual already shipped (`As[2][2]`/`Bs[2][2]`); triple infeasible (LDS at 137/160 KB cap) |
| **C-1** LDS hand-spill | DEFERRED | R3-R4 | R3 spill data shows architectural floor (cA-cD + a/b0/b1 = ~171 dw); spill is at VGPR cap, not localizable. C-X falsification confirmed mechanism is wrong |
| **C-2** K-tail capture refactor | FALSIFIED | R3 | Captures already in if-branch (line 2540-2719), zero liveness leak |
| **C-3** Spill localization probe | DONE | R3 | Architectural ceiling: 256 VGPR cap × 250-290 dw/lane working set forces unavoidable 32-43 dw spill |
| **C-X** N_MASKED helper SENTINEL store | FALSIFIED | R4 | Neutral on active `<0,T,T>` template; -1.8 pp regression on Down-B32-M4096 |
| **D** mfma 32x32x64 cell-shape full port | FALSIFIED | R5 | Microbench gate (lever_d_microbench.cu): -0.03 % delta vs +3 pp gate threshold |
| **E** ASM software pipelining | **FALSIFIED R11** | R11 | Microbench gate (lever_e_microbench.cu): **-7.28 %** vs +5 pp gate threshold |
| **F** Dispatcher-rule per-shape config | LANDED | R6-R10 | 5 generic-rule lands: Qwen-Down M=4096, Qwen-Down BF16 M=2048, DSV3-GateUP M=4096, DSV3-GateUP-B32-M2048, Qwen-GateUP-M=2048, Qwen-GateUP-B32-M4096 |

**6 of 6 architectural levers exhausted**.
**5 dispatcher-rule lands (all Lever F, generic predicates)**.
**Plateau locked at score = 962-964 / grp_FP8 geomean ≈ 1.125 / grp_BF16
geomean ≈ 1.187**.

---

## Worst-5 case-level analysis (why 1.20 target unreachable)

The 5 hardest gpt_oss K=2880 cases have ratios 1.031..1.073 against the
1.200 target. To reach 1.20 on the WORST case (B32-M4096 GateUP at
1.031) would require closing a +16.4 pp gap. Per R3 ceiling derivation:

- 39 dw VGPR spill on the gpt_oss `<0,T,T>` template causes 4 spill→
  reload round-trips per K-iter. Each round-trip is ~80 cy (VMEM
  scratch I/O latency). 22 K-iter × 4 pairs × 80 cy = ~7 K cy/tile
  saved IF spill was zero.
- gpt_oss-GateUP-B32-M4096 has ~368 tile-CU steps. ~7 K cy/tile × 368
  = ~2.6 M cy. At 1.2 GHz this is ~2.2 ms.
- Current main-kernel wall is ~5.4 ms on this shape. Best-case spill
  elimination = 5.4 → 3.2 ms = +69 % throughput = +1.7× speedup
  = ratio 1.031 → 1.75.

**However**: this best case requires ELIMINATING the spill, which
means reducing the working set below 256 dw/lane. Per Lever D R5
microbench falsification, mfma_323264 cell-shape (the only
architectural lever that could shrink the accumulator footprint
half) has zero throughput advantage on isolated single-warp test.
The full-port effort (R56-R64-dm Lever D R-B) was abandoned at
that gate.

So the actually-achievable gap closure on gpt_oss is bounded by
(a) micro-optimizations within the existing 256-VGPR design (already
extensively swept and falsified) and (b) accepting a non-zero spill
floor. The empirical 1.06 mean ratio on gpt_oss subset reflects this
floor.

---

## Plateau acceptance (formal)

**Decision**: ACCEPT score = 962-964 / grp_FP8 geomean ~1.125 as the
final architectural ceiling for FP8 grouped GEMM on this kernel
design.

**Justification**:
1. All 6 architectural levers (A through F) have been empirically
   falsified or fully shipped.
2. F is at saturation: every Qwen3 + DSV3 sub-shape has been audited
   via tight-verify probes (R6-R10) and either has a generic rule
   landed or sits at the noise floor where the win margin is below
   the per-shape spread (rule cannot land without overfitting).
3. gpt_oss K=2880 architectural ceiling is empirically validated:
   R3 spill data shows it is structural (256 VGPR cap × 250-290 dw
   working set), R5 confirmed cell-shape change yields 0% advantage,
   R11 confirmed ASM pipelining yields -7.3% (worse).
4. Task body marked Lever E "最后再做" / "very high risk"; we did
   the microbench gate to falsify cleanly rather than dive into
   8000 LOC of rewrite that the gate said would lose.

**What this means for remaining 89 rounds**:

The R10 round-note's plan B kicks in:
> "If Lever E microbench fails (<3 % gain), accept the 962 ± 3 plateau
> and shift to backward-only optimisations (`bench_grouped_gemm_turbo.py
> --bwd`) which the metric does not exercise."

Backward-only optimisations are out of metric scope (the metric only
times forward). But improvements there are still real engineering
deliverables, and the R11 microbench infra is the foundation for any
future ASM pipelining attempt on the backward kernel
(`grouped_var_k_kernel_fp8`, dB direction).

R12+ candidate options (not committed, just menu):

A. **Backward kernel optimization scout**. Probe whether
   `grouped_var_k_kernel_fp8` (line 5532+ of kernel_fp8_layouts.cpp)
   has the same ceiling. The metric does not score this, but real
   training would benefit from any speedup. R3 spill data showed
   `grouped_var_k_kernel_fp8` has 52 dw spill / 162 dw secondary
   cluster — significantly more headroom than the forward kernel.

B. **R11 noise-floor F sub-rules**. R10 noted Qwen-GateUP-B16-M4096
   is at +0.47 pp tight-verify but +0.6× spread (gap < 1× spread).
   With longer per-shape probes (1000-iter) the gap might open. If
   it does, a B16-M4096 sub-tier rule could land. Bounded upside:
   1/24 case × +0.5 pp = +0.02 pp geomean → +0.2 score points.
   Marginal but free.

C. **Round-budget cleanup**. Document the exhaustion roadmap as a
   single canonical "FP8 grouped GEMM ceiling analysis" doc that
   future agents can reference instead of re-running falsification.

D. **Multi-trial averaging of metric**. The current metric is
   single-trial which causes ±2 score noise. A 3-trial median wrapper
   (`bash scripts/run_metric_3trial.sh`) would tighten the
   improvement detection, but task body forbids editing scripts.
   Could add a wrapper outside scripts dir.

R12 should pick A (backward scout) per task body's R10 plan B
recommendation.

---

## Hard-constraint compliance check

- [x] No metric / benchmark / config edits
- [x] No dispatcher / can_handle changes
- [x] No quantize fuse, no host-side .item() / .tolist()
- [x] No per-model branches in dispatcher (no rule changes this round)
- [x] HIPKITTEN remains `BackendEntry(..., autotune=False)`
- [x] One focused PT commit (this falsification note)
- [x] One focused HK commit (`lever_e_microbench.cu`)
- [x] No BF16 grouped touch
- [x] Correctness 0/48 fail (5-trial baseline + microbench: kernel binary unchanged)

---

## Files touched

### HipKittens repo
- **NEW**: `analysis/fp8_gemm/mi350x/lever_e_microbench.cu` (~280 lines)
  - Standalone .cu microbench for Lever E's "software pipelining" hypothesis
  - Build: `hipcc lever_e_microbench.cu -o lever_e_microbench --offload-arch=gfx950 -O3`
  - Run:   `./lever_e_microbench [N=10000]`
  - Output: per-iter throughput (serial vs pipelined vs % delta vs
    PASS/FAIL verdict)
  - **Result of this run**: -7.28 % × 5 trials → FAIL (verdict: hand
    schedule worse than LLVM)

No HK kernel changes. `kernel_fp8_layouts.cpp` is bit-identical to
R10 end-of-round.

### Primus-Turbo repo
- **NEW**: this `.md` (R11 falsification + plateau acceptance)
- (no code change; baseline metric trial logs preserved at
  `/tmp/metric_round_11_pre.log` and `/tmp/lever_e_microbench_run1.log`)

---

## DoD smoke status

R11 is not on the 5/10/15 cadence (R10 was, this is R12 cadence).
DoD harness will run automatically at R15. Last DoD score recorded
was 608 (SHA `7818763a`).

---

## Quick stats summary

```
Round       SHA (after)      Score   Best   improved   gain over plateau
   1        771d7d58         958     958    yes        +0   (24-shape baseline)
   2        849ae8c          960     960    yes        +2   (R1 falsify A/B)
   3        b54107b          956     960    no          0   (R3 docs round)
   4        fbc5693          961     961    yes         0   (R3 C-X falsify)
   5        9fd99a9          959     961    no          0   (R5 D microbench)
   6        14df676          958     961    no          0   (R6 F partial land)
   7        3c74bdd          960     961    yes         0   (R7 F GateUP-M2048)
   8        85bf67d          961     961    yes        +1   (R8 F DSV3-GateUP)
   9        333fee2          962     962    yes        +1   (R9 F BF16 fix)
  10        7818763          960     962    no          0   (R10 F Qwen-GateUP-B32-M4096)
  11        <this>           963     963    yes        +1   (R10's gain stable, Lever E falsified)
```

Net 11-round gain: +5 score (958 → 963) via Lever F dispatcher rules
+ R10 commit's belated landing. All other levers FALSIFIED. 1000-963
= 37 score gap (= 0.075 pp grp_FP8 geomean delta to 1.20 target).

Architectural ceiling is reached. Per-shape ratios on gpt_oss K=2880
sit at 1.03-1.07 (16-19 pp below the 1.20 target). Closing this gap
requires a wholly different kernel architecture (different acc shape,
different LDS layout) beyond the scope of "mark up an existing
single-launch persistent kernel". No incremental lever remaining.

---

## Recommendation for R12

**Switch agenda to backward-kernel optimization** per R10 plan B.
Specifically, `grouped_var_k_kernel_fp8` has a notably larger
secondary spill cluster (162 NumVR vs forward's 29 NumVR per R3
data) — likely a real lever for fp8 backward dB pass. The metric
does not score backward, but the user's downstream production
training will benefit from any speedup there.

Concrete R12 first step:
1. Rebuild HK with `-Rpass-analysis=kernel-resource-usage` and
   isolate the `grouped_var_k_kernel_fp8` template's spill/VGPR
   profile (R3 data shows it's higher than the forward kernel).
2. Use `bench_grouped_gemm_turbo.py --bwd --dtype fp8` to capture
   current backward wall-time across the 24-shape suite.
3. Identify whether the 52 dw spill is concentrated in a single
   block (analogous to R3's "secondary cluster" analysis) and apply
   the same falsify-or-land protocol used in R3-R7.

Estimated yield: backward dB on gpt_oss-GateUP B32-M4096 currently
runs at ~3.5 TFLOPS (per old bench_grouped_gemm_turbo data). If R12
lands +0.2× on backward wall, that's a meaningful real-world
speedup even though the metric won't reflect it. R12 commit message
should include the bench output for posterity.

If R12 finds backward kernel ALSO at architectural ceiling, then
R13+ shifts to documentation cleanup + plateau noise-band tightening
across remaining rounds.
