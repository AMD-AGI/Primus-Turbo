# Task: HipKittens BF16 grouped GEMM — gpt_oss_20B K-tail (K=2880, K%128=64) speedup

## What is the optimization target?

`turbo.ops.grouped_gemm` with `BackendType.HIPKITTEN` is currently
**~12 % SLOWER** than Triton on every gpt_oss_20B BF16 grouped MoE
shape (8/24 cases) while it is **+10 %** on DeepSeek-V3 / Qwen3-235B-A22B
(16/24 cases). Per-shape baseline (geomean of fwd + bwd wall ratios,
HK / Triton):

```
gpt_oss_20B-GateUP-B4-M2048   0.872   GateUP-B4-M4096   0.934
gpt_oss_20B-Down-B4-M2048     0.922   Down-B4-M4096     0.969
gpt_oss_20B-GateUP-B32-M2048  0.761   GateUP-B32-M4096  0.889
gpt_oss_20B-Down-B32-M2048    0.794   Down-B32-M4096    0.916
geomean (8 shapes)                                       0.880
```

Common factor: `K = 2880` → `K % 128 == 64`, the HipKittens BF16 grouped
RRR / RCR kernels currently dispatch this through a multi-launch
"K-tail" path (a sliced K-aligned kernel + a small tail kernel) while
Triton hits a single fused kernel with native K-tail handling. The B=32
shapes are worst because the launch overhead amortizes worse over the
shorter dB var-K path.

**Goal**: close the gap and pull HK BF16 grouped on gpt_oss to a
geomean ratio ≥ 1.25 (i.e. +25 % vs Triton, matching the FP8
grouped-fused-act ratio HK already holds). This is hard kernel surgery
on the K-tail path; no Path-A-style fusion exists for BF16 (no quantize
step to fuse).

**Out of scope** (FROZEN — same rules as the prior tasks):

* **Caching of any kind is forbidden**: weight quant cache, activation
  quant cache, delayed scale cache, transposed-B cache, **anything**
  that survives across `.forward()` calls. The previous FP8 run was
  cleaned up specifically because R9-R11 cache implementations gamed
  the metric without reflecting real training-step savings.
* **Per-(M, N, K) hardcodes are forbidden**. Dispatch rules go in
  `select_default_config` and must be expressible as general predicates
  (`if K % 128 != 0`, `if M_per_group >= 4096`, `if B >= 16`) — never
  `if M==2048 and N==5760 and K==2880`.
* **Tightening `can_handle` to reject hard shapes is forbidden**.
  Rejecting a shape clips its ratio to 0 → drops the geomean → the
  metric will catch it. The metric script also reports
  `n_correct_fail` and `n_reject` separately so the agent can't sneak
  an `assert K % 128 == 0` past it.
* **No host syncs in the hot path**: no `.item()` / `.tolist()` /
  `torch.cuda.synchronize()` inside `grouped_gemm` / `grouped_gemm_fp8`
  hot paths. Probes are fine for offline measurement only.
* **Don't break the FP8 grouped fused-act path** that ships on this
  branch. Run `python3 scripts/_metric_grouped_fused_wall.py` every 5
  rounds; the score must stay ≥ 920 (current ≈ 929, regression
  budget −10).
* **Don't break DSV3 / Qwen3 BF16** (the 16 "warmup-only" rows in the
  metric table). The metric prints their HK / Triton ratios; if any
  row drops below 1.05 it's a regression even though they don't feed
  the geomean. Watch the `[warmup]` rows.

## Metric (consumed by `auto_optimize.py`)

```
python3 scripts/_metric_grouped_bf16_gpt_oss_wall.py
```

* Runs the full 24-shape MoE BF16 grouped suite (DSV3 + gpt_oss + Qwen3).
* **Scores only the 8 gpt_oss_20B shapes**. The other 16 are still
  benchmarked + their HK/Triton ratios are printed (with a `warmup-only`
  flag in the status column) so silent DSV3/Qwen3 regressions surface
  immediately, but they do NOT feed the geomean. This is by design:
  diluting the gpt_oss signal across 24 shapes hides a +20 % gpt_oss
  win behind a +6 % geomean change.
* DSV3/Qwen3 are run BEFORE gpt_oss in the suite. This is a metric
  necessity, not a free design choice: HK BF16 K-tail kernels have a
  cold-start memory-fault bug if the first BF16 grouped op the process
  launches has K%128≠0. Running K%128==0 shapes first warms the
  runtime. **Don't try to "fix" by reordering or removing the warmup
  shapes — kill the metric's correctness instead.**
* Per-shape: `ratio = hk_tflops / trt_tflops` over `fwd + bwd` wall
  (FLOPs = `6 * M_total * N * K` = 2 fwd + 2 dA + 2 dB).
* Correctness on each gpt_oss shape: HipKittens fwd+bwd cross-checked
  against Triton fwd+bwd on a downsized version of the shape
  (`B' = min(B, 4)`, `M' = min(M, 256)`); `check_allclose` for
  bfloat16 must agree on `out`, `dA`, `dB`. FAIL on any clip the
  shape's ratio to 0 (geomean uses 0.01 floor so penalty is large but
  not exit-poisonous).
* `geomean(ratio_i)` across the 8 gpt_oss shapes only.
* `score = int(min(geomean / 1.25, 1.0) * 1000)`. Target ratio 1.25
  (overridable via `METRIC_BF16_GPT_OSS_TARGET` env).

**Phase 0 baseline** (verified just now, MI355X, GPU 6 idle):
* score = **704**
* geomean = **0.8797**
* correctness FAIL = 0/8
* below_target = 8/8

The gap to 1.25 is large (geomean must rise by +44 %). Don't expect
ratios to jump by half the gap in one round — kernel-surgery work
typically lands +3-8 % per landed change for the first 4-6 rounds.

## Where to attack — investigation surface

These are *suggested* attack vectors, not a forced order. Profile
first; many will turn out to be partially exhausted.

### Lever 1: K-tail kernel design — single-launch path for K%128==64

This is the most likely big win. Current HK behavior on gpt_oss K=2880:
the dispatcher calls the main kernel with `K_main = 2816` (largest
multiple of 128 below 2880) plus a separate tail kernel for `K_tail =
64`. Two launches, two K-loops, two synchronizations of partial sums
in the output tile. Triton bakes K-tail into a single inner-K loop with
a partial-mask on the last iteration.

* Kernel surface: `/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/`
  (currently `kernel_bf16_dynamic.cpp`, builds into `tk_bf16_layouts.so`).
  Look for `grouped_rcr_kernel<...>` (forward + dA path) and
  `grouped_variable_k_crr_kernel<...>` (dB var-K path). Confirm with
  `rg "grouped_.*_kernel" -t cpp` before editing — file names have
  shifted across major refactors.
* Sub-lever A: **In-kernel masked K-tail loop** — let the existing
  kernel handle `K not divisible by 128` by issuing a partial-K MFMA
  with a load-side mask on the last iteration. Compile-time gated via
  `FUSED_KTAIL` template.
* Sub-lever B: **Compile-specialized K-padding** — precompile the
  kernel for `K_padded = 2816 + 128` (a K-bucket with K%128==0) and
  short-circuit the LDS write of the padding columns. Cheaper to write
  but wastes ~4 % of MFMAs.
* Sub-lever C: **Two-stream K-split** — issue the K_main and K_tail
  GEMMs on separate streams to overlap. Likely loses to the launch
  overhead on B=4 small shapes.

### Lever 2: dB var-K kernel for B=32

The B=32 gpt_oss shapes (ratio 0.76-0.92) are the worst — driven by the
var-K dB path issuing many small launches per group. Possible attacks:

* Persistent kernel + work-stealing across groups (similar to what
  the FP8 dB var-K path does).
* Fuse small groups into a single launch (group-coalescing), keyed
  off `min(group_lens)` heuristics.
* Audit MFMA / LDS occupancy on `K=2880, M_per_group=2048` — the
  current allocation may be sub-MFMA-saturating.

### Lever 3: LDS layout / swizzle

`K = 2880` may interact badly with the existing 128-wide LDS bank
swizzle pattern. Profile bank conflicts via `rocprofv3 lds_bank_conflict`
on a single launch. If they're high, an alternate swizzle for K%128≠0
could help without changing the K-tail control flow.

### Lever 4: Tile dispatch / kernel selection rules

Inspect `select_default_config` / `BF16BackendDispatcher` (HipKittens
side) for whether B=32 + K=2880 picks a sub-optimal `(BM, BN, BK)`
tile. Ratio drops from 0.88 (B=4) → 0.76 (B=32) at the same M_per_group
suggests the high-B dispatch is using a tile that wastes dB
amortization. Don't tighten `can_handle` here — change the tile
selection.

## Hard constraints (non-negotiable)

1. **Numerical equivalence** (bfloat16 `check_allclose` agreement) for
   every shape, every round. Validated by the metric's correctness
   gate.
2. **No caching of any kind.** This includes anything that lives in a
   module-level dict / weakref / @lru_cache and persists state across
   `.forward()` calls.
3. **No host syncs in the hot path.** No `.item()`, `.tolist()`,
   `torch.cuda.synchronize()` inside `grouped_gemm` dispatch /
   `_unfused_*` / kernel call.
4. **No per-(M,N,K) hardcodes.** Predicates in `select_default_config`
   must be general (`if K % 128 != 0`, `if K >= 4096`, `if B * M >= 32k`,
   ...).
5. **No `can_handle` tightening to dodge hard shapes.** Reject = score
   penalty.
6. **Don't modify metric files**:
   - `scripts/_metric_grouped_bf16_gpt_oss_wall.py`
   - `scripts/_metric_grouped_fused_wall.py`
   - `scripts/_metric_grouped_only.py`
   - `scripts/_metric_hk_ratio.py`
   - `benchmark/ops/config.py`
7. **Don't regress neighboring metrics**: every 5 rounds also run
   - `python3 scripts/_metric_grouped_fused_wall.py` (FP8 fused wall) —
     score must stay ≥ 920 (current ≈ 929).
   - `python3 scripts/_metric_grouped_only.py` (BF16+FP8 grouped fwd
     only) — score must stay ≥ 980.
   - DoD smoke (auto every 5 rounds via `auto_optimize.py`) — failed = 0.

## Workflow per round (strict)

1. **Read necessary docs** if not in context:
   * `/root/.cursor/skills/hipkittens-primus-turbo-backend/SKILL.md`
   * Most recent `analysis/_notes/round-*-bf16-gpt-oss-*.md`
     round notes.
   * Current target file in HipKittens — verify path before patching:
     ```
     ls /workspace/code/HipKittens/analysis/bf16_gemm/mi350x/
     ```
     primary kernel surface is a .cpp under that directory (currently
     `kernel_bf16_dynamic.cpp` builds into `tk_bf16_layouts.so` via the
     local `Makefile`; the file may be renamed across rounds — re-grep
     before each kernel edit).
2. **First action of every round = run the metric** to get the
   per-shape ratio table. From the table pick the lowest-ratio
   gpt_oss shape that's still PASS as the round's attack target.
   No "I'll guess" shape selection.
   ```
   python3 scripts/_metric_grouped_bf16_gpt_oss_wall.py 2>&1 | tee /tmp/metric_bf16_round_N.log
   ```
3. **Pick a focused change**: ONE of the levers above. One lever per
   round. Don't bundle K-tail rewrite + dispatcher tile changes in
   one round.
4. **Build HipKittens** if you touched the C++:
   ```
   cd /workspace/code/HipKittens/analysis/bf16_gemm/mi350x
   make -j8 tk_bf16_layouts.so 2>&1 | tail -20
   ```
5. **Quick correctness probe** (one shape, fwd+bwd vs Triton ref,
   bfloat16 `check_allclose`):
   `/tmp/probe_bf16_gpt_oss_round_N.py`. If FAIL, REVERT and write a
   falsification round note.
6. **Run the metric** (wall ~16 s on idle MI355X). Decide:
   * Score ≥ prior best + 5 AND all 8 PASS → commit + write round
     note.
   * Flat or down → revert + falsification round note. Don't keep a
     change that doesn't move the score; the auto-loop is
     improvement-gated and will record the lack of progress.
   * Any FAIL on gpt_oss → revert.
   * Any DSV3/Qwen3 ratio drops below 1.05 → revert (silent
     regression on the un-scored rows).
7. **Every 5 rounds** (auto'd by the loop, but also worth eyeballing):
   FP8 fused wall metric ≥ 920, grouped-only metric ≥ 980, DoD smoke
   failed = 0.

## GPU + repo discipline

* GPU pool: `HIPKITTEN_GPU_POOL=3,4,6,7` (auto-pinned by
  `auto_optimize.py`). Don't manually `export HIP_VISIBLE_DEVICES`.
* Two repos, both writable:
  * Primus-Turbo: `/workspace/code/Primus-Turbo`
  * HipKittens: `/workspace/code/HipKittens`
* One focused commit per repo per round; `feat:` / `fix:` / `perf:` /
  `refactor:` style; NEVER `git push`.
* If both repos touched in one round, list HipKittens commit SHA in the
  Primus-Turbo commit body.
* Round notes go to `analysis/_notes/round-N-bf16-gpt-oss-*.md` in
  Primus-Turbo (Primus side) and / or HipKittens (kernel side).

## Output requirements (per round)

A short markdown summary at the end:
* Selected lever / target shape
* Files touched in each repo
* Metric before / after (score, geomean, correctness pass/fail counts,
  any below_target shape moves)
* Commit SHAs (Primus + HipKittens if both)
* Suggestion for the next round
