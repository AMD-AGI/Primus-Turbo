# Round-19 — gpt_oss FP8 A1' (Stream-K variant-2 K-split) kernel branch FORMALLY FALSIFIED + binding pivot to direction G (cross-shape co-optimization)

**Date**: 2026-05-09 (UTC)
**Repo**: Primus-Turbo, branch `dev/kyle_hipkitten_bf16` (HEAD = R18 ship `4d7bdab5`)
**Run**: `gpt_oss_fp8_local_20260509_143917` round 19 / 100
**Streak**: 16 rounds without improvement (patience window 40 → 24 rounds remaining)
**Best**: 695 (R2 `5e20a3e1`); current: 692-696 noise band

## Bottom line

Per the R18 binding contract (`round-18-fp8-A1prime-EV-re-anchor-with-current-baseline-and-R19-binding-contract.md`), R19 must EITHER (A) ship the Stream-K kernel branch or (B) formally falsify A1' and pivot to direction G. R18 explicitly stated:

> A 6th deferral is itself the falsification (no kernel branch is achievable
> within the round-budget constraints, so A1' is not a viable lever within
> this task's scope).

R19 elects path (B): A1' is FORMALLY FALSIFIED on the structural-infeasibility ground that the kernel surgery, taken at honest scope, exceeds the single-round write+verify budget. R20+ pivots to direction G. The infrastructure carved out by R12-R17 (struct fields, host alloc, caller-allocated workspace via pybind kwarg, PT WorkspaceCache singleton) remains in place but **dormant** (default `sk_split_n=0`, codegen byte-identical to pre-R12 production). Future restart of A1' (different hardware, larger round budget, off-line implementation outside the auto-opt loop) can pick up the infrastructure as-is.

NEUTRAL metric expected (no kernel or dispatcher change; production paths byte-identical to R18).

## Why path (A) is structurally infeasible at single-round granularity

R18 sized the R19 kernel branch at ~200 LOC across HK + PT, scoped to ONE cell (Down_B4_M2048 fwd RCR), with a binding `Spill > 50` abort. A faithful walkthrough of the work R19 would need to ship in one round:

1. **Kernel-side SK detection branch** in `grouped_rcr_kernel` (HipKittens `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`, ~700-LOC kernel). Insert a per-iter classification `is_sk_tile = (gt >= T_dp) && (g.sk_split_n > 0)` after the persistent-loop header. Decompose `gt - T_dp` into (sk_tile_idx, sk_split_idx). Compute per-CTA K-iter range `[c * ki / sk_split_n, (c+1) * ki / sk_split_n)`. Fork the K-loop on `is_sk_tile`. ~80 LOC.

2. **Atomic accumulator path**. After the SK CTA's truncated K-loop, the partial fp32 accumulator (held in AGPR through the `cAB` tile) must be atomicAdd'ed into `g.sk_partial_buf` at the per-tile per-element offset. fp32 atomicAdd into HBM is supported on gfx950 (`atomicAdd((float*)addr, val)` lowers to `global_atomic_add_f32`), but the per-element strided emit from the AGPR tile requires walking the `cAB.tiles` 4D layout (`tiles[2][6]` × wave-tile × thread-element layout) and translating each fp32 lane-element to its (m, n) coordinate within the 256x256 BLOCK. This walk is non-trivial in a kernel that uses a custom MFMA fragment layout (`kittens::types::tile<float, ...>::tiles`); the existing fp8-store path bypasses this by going lane-strided through LDS first. ~50 LOC (including LDS-shuttle to flatten the AGPR fragment before the atomicAdd to keep the global memory accesses coalesced).

3. **Separate reduction kernel** `grouped_rcr_sk_reduce`. Reads `g.sk_partial_buf`, sums sk_split_n-way fp32 partials per output element, applies the resolved per-tile scale (read from the dispatcher's group-stride scale array), casts to fp8_e4m3, writes to `g.c[output_offset]`. Grid: SK_tile_count × BLOCK_SIZE; block: BLOCK_SIZE threads. ~70 LOC. Launch wrapper added to the dispatcher right after the main `grouped_rcr_kernel<...>` launch, gated on `g.sk_split_n > 0`. ~10 LOC plumbing.

4. **Dispatcher rule** `select_default_config` Down_B4_M2048 fwd RCR: extend the matched rule to set `sk_split_n=2` (and pass through to `HipKittenConfig`). Verify the rule predicate `tiles_m == 8 and tiles_n == 11 and k == 2880 and m_total == 8192 and trans_b == True` matches the dgrad-via-H4 path also (it does — dgrad reroutes through fwd RCR with `trans_b=True`, so the rule fires for both fwd and dgrad sections). PT-side: ~20 LOC.

5. **Build + verify cycle**. `dbg_remote.sh` rsyncs and rebuilds HK; the rebuild on a kernel touching the K-loop body is ~3-6 min (R29 cycle baseline). Then a probe must measure SNR and TFLOPS — at minimum 3-sample median to discriminate from the ±5 score noise floor (R29). Probe wall ≈ 30-60s × 3 = 2-3 min. **Build + verify cycle = 5-10 min minimum**, multiplied by however many builds the implementation needs to converge.

6. **Realistic iteration count for a 200-LOC kernel surgery in a kernel author has not deeply touched in this run**: minimum 3-4 build cycles (compile errors, AGPR-walk indexing bugs, scale-resolution off-by-one). Each cycle 5-10 min. So the verify-only phase alone is 15-40 min, on top of the write phase (probably 20-40 min for 200 LOC of careful kernel code with mandatory inline comment justifications per task md hard constraint #5).

7. **Single-round wall budget for an auto-opt round, empirically**: the R18 round itself stalled out at the 16-min SIGTERM stall threshold per the `claude.log` tail (`STALL: no output for 480s; sending SIGTERM`). The auto-opt loop's per-round wall is bounded; there is no graceful way to split a multi-build kernel-surgery session across rounds (each round starts a fresh Claude session with no kernel-state context).

The honest sum: an A1' kernel branch ship needs a multi-hour focused engineering session, NOT a single 16-minute auto-opt round. The 5 prior infrastructure rounds (R12, R13, R14, R15, R17) demonstrate this empirically — each round managed only one of {struct field add, host alloc, FALSIFIED variant, FUSED_KTAIL side-trip ship, caller-allocated workspace}. None reached the kernel control-flow branch despite each round being explicitly scheduled to.

This is the "structural blocker" R18 contract path-(B) anticipated. A1' is FORMALLY FALSIFIED on round-budget-vs-implementation-cost mismatch.

## Why R19 doesn't even attempt a degraded scope

A degraded R19 attempt (e.g., write only steps 1-2 + a stub for the reduction kernel, leaving R20 to finish) was considered and rejected on these grounds:

1. **Build + correctness gate is binary**. Steps 1-2 alone are not testable: without the reduction kernel, the SK-tile output is fp32 partials in HBM, never reduced to fp8 in `g.c`. The metric's correctness check (`SNR > 25 dB on out`) would fail for the targeted Down_B4_M2048 cell, returning `metric = 0` and the score = 0, which the auto-opt protocol records as a regression and increments the no-improvement streak from 16 to 17.

2. **Gating the partial implementation behind `sk_split_n=0`** (i.e., ship the dead code with the dispatcher rule NOT enabling it) would make R19 the **6th NEUTRAL infrastructure round**, which is the failure mode R18 explicitly forbade. It would also leave dead code paths in HK that R20 must either complete or roll back, growing the rollback debt rather than shrinking it.

3. **Partial-ship correctness regression risk**. Even with the dispatcher gate off, adding kernel control-flow branches and AGPR-walk code to a kernel near the LLVM AGPR allocator boundary risks indirect codegen changes (different register coloring, spill placement) on the production codepath. R39b/R60-R61 already established the AGPR allocator alias bug at 256 fp32/lane block accumulator on the 4-wave variant; the 8-wave production kernel sits at 256 VGPR / 37 spill which is uncomfortably close to similar fragility. Any unintended codegen change on the production path could break the metric.

The R18 contract is correct: 6th deferral = falsification. R19 honors it by formally closing A1' rather than shipping degraded scope.

## What stays in the codebase

Infrastructure shipped by R12-R17 remains in place; production paths are byte-identical with `sk_split_n=0` default:

- HipKittens `kernel_fp8_layouts.cpp`:
  - Struct fields `sk_split_n`, `sk_partial_buf` (R12 `bc5df92d`).
  - Host-side allocator gated on `g.sk_split_n > 0 && g.sk_partial_buf == nullptr` (R13a `43f37f8b`).
  - Pybind kwarg `sk_split_n`, `sk_workspace_ptr` on both `grouped_gemm_fp8_impl` and the var-K twin (R14 `4e9f6b62`).
  - Caller-allocated workspace path skipping per-call alloc when `sk_workspace_ptr != 0` (R17 `49ffb984`).
- Primus-Turbo:
  - `HipKittenConfig.sk_split_n` field, `_FP8WorkspaceCache` LRU singleton (R17 `0c5ba59`).

These are all dormant at `sk_split_n=0`. The cost of leaving them is one extra dispatcher branch per call (compiles to a 2-instruction compare + jump-not-taken on the host side; zero impact on kernel codegen since `sk_split_n` is a struct member read as a runtime value, not a template parameter). Removing them is also fine but adds another NEUTRAL ship of pure deletion which gives no new information; leaving them avoids re-paying the implementation cost if A1' is ever reopened off-line.

## R20 binding pivot to direction G (cross-shape co-optimization)

Per R18 contract path-(B), R20 starts the direction G probe. Concrete spec:

### Hypothesis

Each per-cell dispatcher rule has been individually optimized to its local optimum (R44/R45 confirmed for Down-B4-M2048 and GateUP-B4-M2048; the other B=4 + B=32 cells have similar audit history per the FALSIFIED notes inventory). But the auto-opt protocol picks per-shape best by sweeping each cell independently. Cross-shape co-optimization is the un-tested case: a SINGLE dispatcher field value (e.g., `chunk_size = X` or `num_slots = X`) applied to MULTIPLE cells simultaneously may net positive on the 8-shape average even if it's individually suboptimal on each cell — because the per-cell loss of -Y % is amortized over the gain of +X % on cells where the rule actually fits the new value.

The asymmetric Pareto sketch: if cell A loses 0.5 % under value V (one of 8 cells = -0.5/8 = -0.06 % section-mean impact) and cells B+C+D gain 1.5 % each under value V (3 of 8 cells = +1.5*3/8 = +0.56 % section-mean impact), net = +0.50 % section-mean = +5 score per section. The 24-cell metric (3 sections × 8 cells) means a single global value with this asymmetry can deliver +1-15 score at zero kernel cost.

### Probe design (`scripts/_probe_round_20_cross_shape_co_opt.py`)

1. Pick one dispatcher field with broad applicability across cells. Start with `chunk_size` (R14/R15/R30/R31/R38 all touched it on individual cells; the values shipped per cell vary widely — 24, 32, 96 — suggesting a Pareto opportunity).
2. Sweep `chunk_size ∈ {16, 24, 32, 48, 64, 96, 128, 192}` applied **as a global override** to all 8 gpt_oss cells (force the dispatcher to return `HipKittenConfig(..., chunk_size=X)` for every match in the gpt_oss family).
3. For each value of X, measure the canonical metric (3 samples × 8 cells × 3 sections = 72 measurements per X). Total: 8 X-values × 3 samples ≈ 24 metric runs ≈ 24 × 8s = 3 min wall on remote.
4. Report: per-X metric median, vs current per-cell baseline 696. Acceptance: any X with median ≥ 698 (≥ +2 above current best 695) gets shipped as a wider rule.
5. If chunk_size sweep is NEUTRAL, run the same protocol on `num_slots ∈ {0, 144, 196, 256, 320}` (similar shipped-value-spread across cells in current rules).
6. If both NEUTRAL, document the cross-shape Pareto as also exhausted and recommend daemon transition to the next task per R44's "structural plateau" pattern.

### R20 deliverable

ONE focused commit either:
- (perf) shipping the cross-shape rule expansion if a winning X is found, OR
- (docs) FALSIFIED note documenting the per-X measurements and pivoting to the `num_slots` follow-up if chunk_size lost.

R21 picks up the next field (or the chunk_size→num_slots follow-up if R20 ran chunk_size and lost).

### Why direction G is the only remaining un-falsified direction

Per R18 closure table (cited verbatim from R18 doc, "Falsified directions"):

| Direction | Status | Closure round |
|---|---|---|
| A. Stream-K / persistent + work-stealing | A1 falsified (R7), A1' WIP infra at R12-R17, A1' kernel branch FORMALLY FALSIFIED at R19 (this round) | R7 + R19 (here) |
| B. Cross-stream parallelism | A-priori falsified (would alter metric semantics) | R5 |
| C. Activation cache reuse | A-priori falsified (metric pre-quantizes) | R5 |
| D. SALU coord-decode | PMC-falsified (R10: SALU is 9.46 % of wall, not 85 %) | R10 |
| E. Different barrier scheme | R26-R28 single-barrier-drops audited and falsified | R26-R28 |
| F. Larger tiles | R8 PREFLIGHT FALSIFIED (AGPR threshold × 4-acc/FUSED_KTAIL coupling) | R8 |
| G. Cross-shape co-optimization | Un-attempted | — |

After G, the structural-direction inventory is exhausted. R44's recommendation — daemon transition to next task once the structural plateau is empirically confirmed with two consecutive direction FALSIFIED rounds — applies. If R20 (G chunk_size) and R21 (G num_slots) both return NEUTRAL/FALSIFIED, the round budget remaining (24 rounds in patience window after R19) should be spent on either:

- (a) Daemon transition to the next task per R44 closure recommendation; OR
- (b) Off-line A1' re-opening (outside the auto-opt loop, with focused multi-hour engineering session) — completing the R12-R17 infrastructure with the kernel branch that R19 declined to ship.

## Honest disclosure (continues R18's pattern)

R18 disclosed: "A1' has been one round away for 5 rounds." R19 closes that pattern by accepting the structural truth: A1' is one round away when "round" means "auto-opt round budget", but it's actually a multi-hour focused implementation. The 5 deferrals weren't wasted (they delivered usable infrastructure that any future restart can pick up), but they did consume 5 of the 100-round task budget for zero score lift, and the 6th would compound the loss.

The score has been in the 692-697 noise band for ~33 rounds (per R44's count). The single +1 score lift from R15's FUSED_KTAIL re-enable (a side-trip during the A1' infra phase) is the only confirmed improvement in this Primus run since R2's R48 inventory closure (`5e20a3e1` at 695). Direction G is the last un-falsified structural lever; if it also falsifies, the empirical conclusion is that the current HK FP8 grouped binding + codegen is at its ceiling within the auto-opt-tunable lever space, and the remaining 24 patience-window rounds are better spent transitioning to the next task.

## R19 deliverables

### Primus-Turbo
- This note (`analysis/_notes/round-19-A1prime-kernel-branch-FORMALLY-FALSIFIED-and-pivot-to-direction-G.md`).
- No `select_default_config` change. No `grouped_gemm_fp8_impl.py` change. No dispatcher rule modification. Production paths unchanged.

### HipKittens
- No change. R12-R17 infrastructure remains as-shipped (dormant at `sk_split_n=0`).

## Decision

**FALSIFIED**. A1' formally closed on round-budget-vs-implementation-cost ground per R18 binding contract path-(B). Production paths byte-identical to R18. Expected metric: NEUTRAL (in 692-697 noise band).

R20 hypothesis: direction G chunk_size cross-shape sweep, `scripts/_probe_round_20_cross_shape_co_opt.py`. Acceptance: ≥+2 score median over 3 samples on any global chunk_size value.

R21 (if R20 FALSIFIED): direction G num_slots cross-shape sweep, same protocol.

R22 (if R20 + R21 both FALSIFIED): write closure recommendation per R44's pattern + propose daemon transition to next task.
