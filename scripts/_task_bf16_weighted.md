# Task: HipKittens BF16 grouped GEMM — full 24-shape MoE wall, weighted toward gpt_oss

## What is the optimization target?

`turbo.ops.grouped_gemm` with `BackendType.HIPKITTEN` on the full
24-shape MoE BF16 suite (DeepSeek-V3 + gpt_oss_20B + Qwen3-235B-A22B,
each 8 shapes, fwd + bwd wall). The task: lift HK / Triton ratio to
**≥ 1.25 on every one of the 24 shapes**.

Baseline (verified just now, MI355X, idle GPU): per-family un-weighted
geomean of the HK / Triton ratio:

```
DeepSeek-V3       n=8  geomean = 1.129    gap to 1.25  +10.7 %
gpt_oss_20B       n=8  geomean = 0.887    gap to 1.25  +40.9 %
Qwen3-235B-A22B   n=8  geomean = 1.121    gap to 1.25  +11.5 %
```

`gpt_oss_20B` is by far the worst — every shape there is HK / Triton
< 1.0 because `K = 2880, K % 128 = 64` routes through the multi-launch
"K-tail" kernel path while Triton fuses K-tail into a single inner-K
loop. DSV3 / Qwen3 are already HK-favored (+10-13 %) on the K%128==0
fast path; getting them to 1.25 is harder kernel-quality work
(LDS swizzle / MFMA scheduling / dispatch tile selection), not a
single-shot kernel rewrite.

## Metric (consumed by `auto_optimize.py`)

```
python3 scripts/_metric_grouped_bf16_weighted_wall.py
```

* Runs the full 24-shape MoE BF16 wall (fwd + bwd, FLOPs = `6 M N K`).
* **Per-shape progress**, capped:
  ```
  progress_i = min(ratio_i / 1.25, 1.0)
  ```
  No shape can over-contribute past its own target — over-shooting one
  shape can't mask another being below.
* **Weighted average** with hand-picked per-family weights:
  ```
  gpt_oss_20B       weight 3   (worst gap, highest priority)
  DeepSeek-V3       weight 1
  Qwen3-235B-A22B   weight 1
  ```
  Total weight = 8*3 + 8*1 + 8*1 = 40. gpt_oss owns 60 % of the
  headroom; DSV3 + Qwen3 each own 20 %.
* **Score**:
  ```
  score = int(weighted_avg(progress_i) * 1000)
  ```
  * Phase 0 (today): **score = 788**, all 24 correctness PASS, 0/24
    above target.
  * Theoretical ceiling without DSV3 / Qwen3 progress (gpt_oss alone
    reaches 1.25, others stay at 1.12-1.13): score ≈ **957-960**
    — agent will see this plateau and naturally pivot to the
    K%128==0 path.
  * Score = 1000 ⇔ every shape's HK / Triton ratio ≥ 1.25.
* Correctness on every shape: HK fwd+bwd cross-checked against Triton
  fwd+bwd on a downsized version of the shape (B' = min(B, 4),
  M' = min(M, 256)); bfloat16 `check_allclose` must agree on `out`,
  `dA`, `dB`. FAIL clips that shape's progress to 0 (severe, since
  weight is 1 or 3).
* Iteration order = canonical suite order (DSV3 → gpt_oss → Qwen3).
  DSV3 first incidentally warms the HipKittens runtime around the
  K%128==0 path before the K-tail (K%128==64) gpt_oss launches —
  workaround for the HK BF16 K-tail cold-start sync-fault bug.
  **Don't reorder.**

Override knobs (env, for offline experimentation only — do not commit
metric tweaks):
* `METRIC_BF16_WEIGHTED_TARGET=1.25` (default)
* `METRIC_BF16_WEIGHT_GPT_OSS=3.0`
* `METRIC_BF16_WEIGHT_DSV3=1.0`
* `METRIC_BF16_WEIGHT_QWEN3=1.0`

## Score landscape (rough, for round planning)

| Phase | gpt_oss geomean | DSV3 / Qwen3 geomean | Score |
|---|---|---|---|
| Today (baseline) | 0.887 | 1.129 / 1.121 | 788 |
| gpt_oss → 1.05 | 1.05 | unchanged | ~875 |
| gpt_oss → 1.25 (capped 1.0) | 1.25+ | unchanged | ~960 |
| gpt_oss = 1.25, DSV3+Qwen3 → 1.20 | 1.25 | 1.20 | ~984 |
| All → 1.25 | 1.25+ | 1.25+ | **1000** |

Do not expect each round to move the score by 30+; kernel surgery on
the K-tail path is hard, +5-15 score per round of useful work is the
norm during the gpt_oss attack phase.

## Hard constraints (non-negotiable, FROZEN — same as prior tasks)

1. **No caching of any kind.** No weight quant cache, no activation
   quant cache, no delayed scale cache, no transposed-B cache,
   anything that survives across `.forward()` calls. The previous
   FP8 run was force-cleaned because R9-R11 caches gamed the metric
   without reflecting real training-step savings.
2. **No per-(M, N, K) hardcodes.** Dispatch rules in
   `select_default_config` / `BF16BackendDispatcher` must be
   expressible as general predicates (`if K % 128 != 0`,
   `if M_per_group >= 4096`, `if B >= 16`) — never `if M==2048
   and N==5760 and K==2880`.
3. **No `can_handle` tightening to dodge hard shapes.** Reject =
   ratio 0 = score penalty proportional to weight.
4. **No host syncs in the hot path.** No `.item()`, `.tolist()`,
   `torch.cuda.synchronize()` inside `grouped_gemm` /
   `grouped_gemm_fp8` dispatch / kernel call.
5. **Numerical equivalence** (bfloat16 `check_allclose`) on every
   shape every round. Validated by the metric's correctness gate
   (downsized) AND by DoD smoke (full shapes) every 5 rounds.
6. **Don't break neighboring metrics.** Every 5 rounds, verify:
   * `python3 scripts/_metric_grouped_fused_wall.py` (FP8 fused-act
     wall) — score must stay ≥ 920 (current ≈ 929).
   * `python3 scripts/_metric_grouped_only.py` (BF16 + FP8 grouped
     fwd-only) — score must stay ≥ 980.
   * DoD smoke (auto'd by `auto_optimize.py --dod-every 5`) —
     failed = 0.
7. **Don't modify metric files**:
   * `scripts/_metric_grouped_bf16_weighted_wall.py`
   * `scripts/_metric_grouped_fused_wall.py`
   * `scripts/_metric_grouped_only.py`
   * `scripts/_metric_hk_ratio.py`
   * `benchmark/ops/config.py`
   These are read-only inputs to the auto-loop.

## Where to attack — investigation surface

These are *suggested* attack vectors. **Profile first** every round
(metric per-shape table picks the lowest-ratio shape for you) — many
will turn out to be partially exhausted; that's a successful round if
the falsification note nails it.

### Phase A — gpt_oss K-tail (rounds 1-15 expected)

Highest score-per-round leverage. All 8 gpt_oss shapes have ratio
< 1.0; B=32 shapes (4 of them) are 0.76-0.91 — worst.

* **Lever A1**: in-kernel masked K-tail loop (compile-time gated via a
  `FUSED_KTAIL` template; load-side mask on the last K iteration).
  Eliminates the 2-launch K-main + K-tail synchronization in the
  output tile. Triton's win on K=2880 comes from this.
* **Lever A2**: dB var-K kernel for B=32 — persistent kernel +
  work-stealing across groups, or group-coalescing for `min(group_lens)`
  small. The B=32 ratio drop from B=4 (e.g. 0.85 → 0.76 on
  GateUP-M2048) suggests the var-K dispatch issues too many small
  launches per group.
* **Lever A3**: LDS swizzle audit for K=2880. `K % 128 = 64` may
  interact badly with the existing 128-wide bank pattern. If
  `rocprofv3 lds_bank_conflict` shows high counts, an alternate
  swizzle for K%128 != 0 fixes it without touching the K-loop control
  flow.
* **Lever A4**: tile-dispatch for B=32 + K=2880 — current
  `select_default_config` may pick a sub-MFMA-saturating
  `(BM, BN, BK)`. Don't tighten `can_handle`; change the
  selection rule (general predicate).

### Phase B — DSV3 / Qwen3 push (rounds 16-30 expected)

Once gpt_oss is mostly capped (progress ≥ 0.95), the score plateau
sits ~960 and the remaining ~40 score points live on the K%128==0
fast path.

* **Lever B1**: MFMA pipeline scheduling — likely the biggest single
  win on K%128==0. Profile `rocprofv3 valuMfmaUtil` on DSV3 / Qwen3
  baseline; if it's < 90 % the kernel is not MFMA-saturated.
* **Lever B2**: register pressure / occupancy — DSV3 K=7168 /
  Qwen3 K=4096 are both large enough that the K-loop register
  footprint matters. Consider `__launch_bounds__` tuning,
  spill-to-LDS for less hot regs.
* **Lever B3**: better pre-shuffle / B-side load patterns — depending
  on layout (RCR / RRR / CRR) the B-tile load may have unaligned
  swizzle. Audit per-layout.
* **Lever B4**: persistent kernel / work-stealing for the dB var-K
  path on DSV3 K=2048 — same idea as A2 but on the K%128==0 case.

### Phase C — closing 990 → 1000 (final rounds)

Last 10 score points typically need a few coordinated wins, not one
big lever. Expect 1-2 % improvements per round, possibly a per-shape
Lever A* / B* combination that didn't fit cleanly into one phase.

## Workflow per round (strict)

1. **Read necessary docs** if not in context:
   * `/root/.cursor/skills/hipkittens-primus-turbo-backend/SKILL.md`
   * Most recent `analysis/_notes/round-*-bf16-*.md`.
   * Current target file in HipKittens — verify path before patching:
     ```
     ls /workspace/code/HipKittens/analysis/bf16_gemm/mi350x/
     ```
     primary kernel surface is a .cpp under that directory (currently
     `kernel_bf16_dynamic.cpp` builds into `tk_bf16_layouts.so` via the
     local `Makefile`; the file may be renamed across rounds — re-grep
     before each kernel edit).
2. **First action of every round = run the metric.**
   ```
   python3 scripts/_metric_grouped_bf16_weighted_wall.py 2>&1 \
     | tee /tmp/metric_bf16_round_N.log
   ```
   From the per-shape table, pick the lowest-progress shape that's
   PASS (correctness ok) and treat it as the round's attack target.
   Don't guess.
3. **Pick a focused change**: ONE lever from the surface above (A1
   alone, B2 alone, ...). Don't bundle "K-tail kernel + LDS swizzle"
   in one round; you can't disambiguate which moved the score.
4. **Build HipKittens** if you touched the C++:
   ```
   cd /workspace/code/HipKittens/analysis/bf16_gemm/mi350x
   make -j8 tk_bf16_layouts.so 2>&1 | tail -20
   ```
5. **Quick correctness probe** (one shape, fwd+bwd vs Triton ref,
   bfloat16 `check_allclose`). If FAIL, REVERT and write a
   falsification round note.
6. **Run the metric** (~17 s on idle MI355X). Decide:
   * Score ≥ prior best + 5 AND all 24 correctness PASS → commit +
     write round note. Add the per-family geomean line to the commit
     message.
   * Flat or down → revert + falsification round note. Don't keep a
     change that doesn't move the score.
   * Any correctness FAIL → revert.
7. **Every 5 rounds** (auto'd by the loop, also worth eyeballing):
   FP8 fused wall metric ≥ 920, grouped-only metric ≥ 980, DoD
   smoke failed = 0.

## GPU + repo discipline

* GPU pool: `HIPKITTEN_GPU_POOL=3,4,6,7` (auto-pinned by
  `auto_optimize.py`). Don't manually export `HIP_VISIBLE_DEVICES`.
* Two repos, both writable:
  * Primus-Turbo: `/workspace/code/Primus-Turbo`
  * HipKittens: `/workspace/code/HipKittens`
* One focused commit per repo per round; `feat:` / `fix:` / `perf:` /
  `refactor:` style; NEVER `git push`.
* If both repos touched in one round, list HipKittens commit SHA in
  the Primus-Turbo commit body.
* Round notes go to `analysis/_notes/round-N-bf16-*.md` in
  Primus-Turbo (Primus side) and / or HipKittens (kernel side).

## Output requirements (per round)

A short markdown summary at the end:
* Selected lever / target shape (lowest-progress row from the metric)
* Files touched in each repo
* Metric before / after:
  * Score
  * Per-family geomean (gpt_oss / DSV3 / Qwen3)
  * Correctness FAIL count
  * Number of shapes that moved (above_target / below_target shifts)
* Commit SHAs (Primus + HipKittens if both)
* Suggestion for the next round
