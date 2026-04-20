# Optimization Process Specification

This file defines the **detailed optimization process** for `kernel-optimize`.

The default prerequisite is (agent proceeds through: project skill → `../SKILL.md` (understand requirements) → project skill (collect information) → `../SKILL.md` (execute) → this file):
- All required information has been collected from the project skill per `../SKILL.md`'s "Prerequisite Information"
- DEFINE_TARGET (parameter structuring) and PREPARE_ENVIRONMENT (campaign directory setup) have been completed in `../SKILL.md`
- The campaign directory for this round has been established, `manifest.yaml` has been filled in, and `quick_test_bench.py` has been generated
- `related_work.md` has been created by the `SURVEY_RELATED_WORK` step in `../SKILL.md`
- If `agent/historical_experience/<target_gpu>/<target_op>/<target_backend_lower>/tips.md` exists, it has been read after `related_work.md` and before `round-1`
- `target_op`, `target_backend`, `target_lang`, `target_gpu`, and `execution_mode` are clearly defined

Therefore, this file is **not responsible for**:
- Explaining how to extract parameters from user instructions (see DEFINE_TARGET in [`../SKILL.md`](../SKILL.md))
- Specifying campaign directory structure (see PREPARE_ENVIRONMENT in [`../SKILL.md`](../SKILL.md))
- Running the related-work / SOTA survey itself (done in `../SKILL.md` before this file starts)
- Providing build, test, or benchmark commands for any specific project (provided by the project skill)

## Iteration Contract

Before running `ENVIRONMENT_BASELINE`, read [`../../../rules/iteration_rules.mdc`](../../../rules/iteration_rules.mdc). Treat it as a hard constraint throughout the loop:

- one hypothesis and one meaningful kernel change per round
- correctness before performance
- benchmark the full active validation set
- accept or roll back cleanly to the previous accepted baseline

## Input Parameters

Before starting the optimization process, at minimum define:

| Parameter | Meaning |
|-----------|---------|
| `target_op` | Target operator |
| `target_backend` | Target backend |
| `target_lang` | Implementation language |
| `target_gpu` | Target GPU architecture |
| `execution_mode` | `repo` / `workspace` |
| `campaign_dir` | Campaign directory for this round |
| `target_shapes` | Full shape set of interest for this round; quick validation uses `representative_shapes` as the active subset |
| `performance_target` | Target performance; `null` means there is no explicit numeric target |
| `primary_metric` | Primary comparison metric(s); depends on operator type (e.g., GEMM: `Forward TFLOPS`, elementwise: `Forward GB/s`) |
| `git_commit` | Whether to git commit accepted versions |
| `git_branch` | Current optimization branch (or `none`) |
| `max_iterations` | Maximum iteration count (optional) |
| `max_duration` | Maximum runtime budget (optional, e.g. `"4h"`) |
| `project_skill` | Corresponding project skill |

## Confirm Previously Obtained Information

The following information should have been obtained during the project skill phase and DEFINE_TARGET phase. Do a final confirmation before starting the optimization loop:

| Information | Purpose | Example |
|-------------|---------|---------|
| **Kernel source file path** | Read code during ANALYZE | `primus_turbo/triton/gemm/gemm_fp8_kernel.py` |
| **Focused test command** | Full correctness validation | `pytest tests/pytorch/ops/test_gemm_fp8.py -v -k "blockwise and TRITON"` |
| **Focused benchmark command** | Full performance evaluation | `PRIMUS_TURBO_GEMM_BACKEND=TRITON python benchmark/ops/bench_gemm_turbo.py --dtype fp8 --granularity blockwise` |
| **Quick validation command** | Fast correctness + benchmark each round | `python <campaign_dir>/quick_test_bench.py` (generated in PREPARE_ENVIRONMENT, filled with `representative_shapes` after BASELINE) |
| **Benchmark output column names** | Parse results, compute scores | `Forward TFLOPS`, `Backward TFLOPS` (primary_metric), `Check` (correctness gate) |
| **Rebuild requirements** | Whether rebuild is needed after code changes | Triton: no rebuild needed; HIP: incremental rebuild required |
| **Related-work report** | Seed early hypotheses with external and internal baselines | `<campaign_dir>/related_work.md` |

If anything is missing, refer back to the project skill or `../SKILL.md`'s manifest to fill in the gaps.

## Historical Tips Bootstrap

After `related_work.md` is written and before `round-1` / `ENVIRONMENT_BASELINE` starts, check whether a reusable tips file already exists for this hardware / op / backend combination.

Path convention:

`agent/historical_experience/<target_gpu>/<target_op>/<target_backend_lower>/tips.md`

Example:

`agent/historical_experience/gfx950/gemm_fp8_blockwise/triton/tips.md`

Rules:

- Normalize the backend directory to lowercase, e.g. `TRITON -> triton`, `CK -> ck`
- If the file exists, read it once before the first round
- Treat it as prior experience, not as permission to skip measurement, profiling, or validation
- Revisit it later only when the round history suggests a related pattern may apply again

## Two-Level Validation

The optimization loop uses two levels of validation:

| Level | When to use | Content |
|-------|------------|---------|
| **quick** | Default for each VALIDATE round | Run `quick_command`: representative-shape correctness + benchmark in one step |
| **full** | BASELINE, end of a direction, final acceptance, or when risk/noise requires it | Focused test + focused benchmark on all `target_shapes` |

The quick validation script is generated during PREPARE_ENVIRONMENT while the project API context is still fresh. After BASELINE, fill it with `representative_shapes` and record the invocation in the manifest as `quick_command`. Full validation uses the focused test / benchmark commands from the project skill.

Active validation set by level:
- **quick** → all `representative_shapes`
- **full** → all `target_shapes`

Within a chosen validation level, do not cherry-pick a smaller subset.

Quick validation is for small-step iteration within one direction. When a direction completes, when quick results are borderline (improvement < 5%), or when changes are high-risk (control flow, data layout, etc.), upgrade to full validation for confirmation.

## Overall Flow

Before `ENVIRONMENT_BASELINE`, if the corresponding historical tips file exists, read it.

```text
ENVIRONMENT_BASELINE
  -> [ANALYZE -> OPTIMIZE -> VALIDATE -> ACCEPT/ROLLBACK] (iteration loop)
  -> TERMINATION_CHECK (exit only if at least one termination condition is satisfied)
  -> REPORT
```

Round numbering is strict: `round-1` is BASELINE, and optimization attempts start at `round-2`.

ACCEPT and ROLLBACK both return to ANALYZE for the next round. Do not jump directly to REPORT unless TERMINATION_CHECK passes.

In `repo-mode`, VALIDATE validates directly in the main repo with no SYNC_BACK step.
In `workspace-mode`, VALIDATE is split into a local gate and an integration gate, with SYNC_BACK in between.

## Scoring Operations Specification

### From Benchmark Output to Aggregate Score

**Step 1**: Run benchmark (quick or full), produce output

**Step 2**: Extract each row's `primary_metric` (e.g., `Forward TFLOPS` or `Forward GB/s`, depending on operator type; if multiple metrics, extract each separately) and `Check` column

**Step 3**: Correctness gate
- Any row with `Check = FAIL` → candidate `aggregate score = 0`, reject immediately

**Step 4**: Compute aggregate score
- If `target_shapes` has only 1 configuration: `aggregate score = that configuration's primary_metric`
- If `target_shapes` has multiple configurations: `aggregate score = geometric_mean(all configurations' primary_metric)`

```
geometric_mean = (x1 * x2 * ... * xn) ^ (1/n)
```

**Step 5**: Retain the score vector (raw values per shape) to avoid the total score masking localized regressions

**Multi-metric handling**:
- If `primary_metric` is a forward+backward pair for the same end-to-end workload (for example, GEMM training where benchmark output includes both forward and backward time), derive a **single combined-step primary score** and use that for accept/rollback.
- Default combined-step derivation:
  - `Combined Step Time (ms) = Forward Time (ms) + Backward Time (ms)`
  - `Combined Step Score = 1 / Combined Step Time (ms)` (higher is better)
- If the project skill defines a workload-specific equivalent throughput metric (for example, `Combined Step TFLOPS`), use that instead of the generic inverse-time score.
- Keep `Forward TFLOPS`, `Backward TFLOPS`, or other component metrics as **secondary diagnostics** to explain trade-offs, but do not require every component aggregate to improve independently when the campaign's true objective is end-to-end step performance.
- If no combined-step metric is defined and the metrics are genuinely independent objectives, fall back to the old rule: all metric aggregates must not regress, and at least one must improve.

### Noise Assessment

- When improvement is < 2%, it is considered near the noise range
- In this case, re-measure at least 3 times, compute mean and standard deviation
- If mean improvement > 1% and standard deviation < half of the improvement magnitude, consider it a valid improvement
- Otherwise, treat as noise and do not accept

### Acceptance Rules

- `aggregate score` must not be lower than the current best
- If it only matches the current best, there must be a clear additional benefit (e.g., broader applicability, more stable results)
- If any core shape regresses > 5% on the **primary accept metric**, default to rejection; unless this round is explicitly a targeted shape optimization
- When using a combined-step metric, judge per-shape regressions on `Combined Step Time (ms)` (or the project-defined equivalent primary score), not on the sign of individual forward/backward component metrics alone

### Computation Example

Benchmark results (3 shapes):

| M | N | K | Forward TFLOPS | Backward TFLOPS | Check |
|---|---|---|----------------|-----------------|-------|
| 4096 | 4096 | 4096 | 320 | 160 | PASS |
| 2048 | 8192 | 4096 | 305 | 152 | PASS |
| 1024 | 4096 | 8192 | 290 | 145 | PASS |

```
forward aggregate  = (320 * 305 * 290) ^ (1/3)         = 304.8
backward aggregate = (160 * 152 * 145) ^ (1/3)         = 152.3
step geomean       = sqrt(304.8 * 152.3)               = 215.5
```
Equivalently, the step geomean is the geometric mean of all six TFLOPS
values (forward ∪ backward) across PASS shapes.

If the kernel has no backward path, fill the Backward TFLOPS column with `-`,
skip the backward aggregate, and let `step geomean = forward aggregate`.

Baseline: forward=285.0, backward=140.0, step=sqrt(285.0 * 140.0)=199.8.
Candidate step = 215.5 → improvement = (215.5 - 199.8) / 199.8 = +7.857% → accept.
`vs Baseline` cell on the trend row: `step +7.857%, fwd +6.947%, bwd +8.786%`.

## Phase Descriptions

### 1. ENVIRONMENT_BASELINE (`round-1`)

**Goal**: Freeze the starting point and establish a unified comparison baseline.

**What "focused" means**:
- Focused test = test subset limited to `target_op` + `target_backend` (e.g., `-k "blockwise and TRITON"`)
- Focused benchmark = benchmark limited to `target_backend`, covering all `target_shapes`

**Steps**:
1. Confirm the current environment can build and run correctly
2. Run **full** focused test, confirm all PASS
3. Run **full** focused benchmark, write the canonical baseline summary to `<campaign_dir>/rounds/round-1/summary.md`, and store any raw command outputs / CSVs under `<campaign_dir>/rounds/round-1/artifacts/`
4. Compute baseline `score vector` and `aggregate score` per scoring specification
5. Select 3-5 `representative_shapes` from PASS-only benchmark rows, covering small / medium / large behavior
6. Update `quick_test_bench.py` with those `representative_shapes` and record `quick_command` in the manifest
7. Copy the current kernel into `<campaign_dir>/rounds/round-1/kernel_snapshot/` to establish the rollback root
8. Record backend configuration and key environment state

BASELINE always uses full validation to ensure the starting data is complete and reliable.

**Baseline record template** (write to `<campaign_dir>/logs/optimize.md`):

```markdown
## Baseline
- Time: <YYYY-MM-DD HH:MM>
- Backend: <target_backend>
- GPU: <target_gpu>
- Commit: <git_hash>
- Validation level: full

- Aggregate score (geomean): 278.0
- All Check: PASS
- Detailed data: rounds/round-1/summary.md
```

**Output**:
- `<campaign_dir>/rounds/round-1/summary.md` + baseline aggregate score
- `<campaign_dir>/rounds/round-1/kernel_snapshot/` as the baseline rollback root
- Current best = baseline (`round-1`)
- The first optimization attempt starts at `round-2`

The baseline round uses the same `summary.md` structure as later rounds. Its "Single change" section must explicitly say that no code change was made, and its "Decision" section must be `BASELINE`.

### 2. ANALYZE

**Goal**: Find the most worthwhile direction for the next round, rather than tuning blindly.

**Required actions**:
- Read `<campaign_dir>/related_work.md` before proposing new directions
- Read the core implementation of the current best version
- Review recent accepted versions and failed attempts (from campaign log)
- Profile or analyze benchmark metrics as needed to identify the current main bottleneck
- Read language/hardware/profiling skills as needed

**Bottleneck Classification and Optimization Direction Mapping**:

When profiler data is available, use the following framework to classify bottlenecks:

| Bottleneck signal | Classification | Optimization direction |
|-------------------|---------------|----------------------|
| Low ALU utilization, low MFMA instruction ratio | Compute bound | Tile size adjustment, instruction selection (e.g., MFMA vs WMMA), algorithm simplification, reduce redundant computation |
| Memory throughput near hardware peak, many global load/store stalls | Memory bound | Data layout optimization, prefetch / software pipelining, reduce redundant memory access, LDS utilization |
| Low occupancy, excessive register or LDS usage | Resource bound | Reduce register pressure, adjust LDS allocation, lower tile size to trade for more waves |
| High kernel launch overhead ratio, small total compute | Launch/overhead bound | Persistent kernel, batch multiple small kernels, reduce dispatch count |

When profiler data is unavailable, infer indirectly from benchmark results:
- If TFLOPS is far below theoretical peak and efficiency improves significantly with larger shapes → likely launch overhead or occupancy issue
- If TFLOPS improves significantly as K increases → likely memory bound (higher compute-to-memory ratio improves efficiency)
- If efficiency difference across shapes is small and overall low → likely compute bound

**Each candidate direction must answer at minimum**:
- What is the current bottleneck
- What will be changed this round
- What is the expected benefit
- What are the risks
- What signal will verify success or failure

**Output**:
- Prioritized hypothesis list
- Primary hypothesis for this round

### 3. OPTIMIZE

**Goal**: Implement a single attributable, rollback-able small-step change.

**Required actions**:
- Advance only one primary hypothesis per round
- After modification, be able to clearly answer "what exactly was changed this round"
- If compiled artifacts are involved, rebuild per project skill instructions
- Record modified files, key parameters, and expected impact for this round

**Constraints**:
- Do not mix unrelated cleanup into optimization attempts
- Do not introduce multiple orthogonal major changes simultaneously
- Do not break key interface semantics agreed upon with the upstream project

**Output**:
- Candidate diff
- Build status

### 4. VALIDATE

**Goal**: Decide whether the candidate passes validation.

**Under `repo-mode`** (common path, no SYNC_BACK):

1. Ensure backend settings are correct if needed (env var or reset)
2. Run `quick_command` (combined representative-shape correctness + benchmark)
3. Compute `score vector` and `aggregate score` per scoring specification
4. Compare against current best
5. Upgrade to **full** validation when the direction completes, when results are borderline (improvement < 5%), or when changes are high-risk

**Command completion protocol**:
- A benchmark or test command does **not** count as completed when it has merely been launched. It counts as completed only after the process exits and the expected output artifact (log, CSV, etc.) exists on disk.
- If the shell tool backgrounds a command because of timeout, the agent MUST poll the process and/or output file until completion before doing anything else in the optimization loop.
- Do **not** treat partial terminal output, partial CSV content, or an in-progress log file as a finished validation result.
- Do **not** start the next `ANALYZE`, `OPTIMIZE`, or `VALIDATE` step while the current round's benchmark or test command is still running.

**Polling cap**:
- A single `sleep` / `wait` / polling call MUST NOT exceed **15 minutes (900 seconds)**. Long-running benchmarks (which commonly take 30-90 minutes on this kernel family) MUST be polled in repeated `sleep <= 900` windows, not in one multi-hour sleep.
- Between poll windows, the agent MUST re-inspect the backgrounded terminal file and/or the expected output artifact to confirm progress (e.g. grep the latest `TestID:` line, `wc -l` the CSV, check for `exit_code` in the terminal metadata).
- If a poll window finds no forward progress across two consecutive checks, treat the command as hung and investigate (re-read the terminal, check for OOM / driver errors) instead of continuing to sleep blindly.
- This bounds worst-case blocking if the shell session is interrupted or if the benchmark deadlocks, and keeps the agent responsive for mid-run corrections from the user.

Write the canonical round record to `<campaign_dir>/rounds/round-N/summary.md` (for `N >= 2`) and place any raw benchmark/test outputs under `<campaign_dir>/rounds/round-N/artifacts/`.

**Hard gates**:
- Build failure → reject immediately
- Correctness failure → reject immediately
- `Check = FAIL` in benchmark → reject immediately
- Aggregate score regression → default reject
- Core shape regression > 3% → default reject

**After passing**:
1. Write this round's detailed results to `rounds/round-N/summary.md`
2. Immediately update `logs/optimize.md` (optimization history, current best, directions to try)
3. Update current best
4. If `git_commit=true`: git commit (see git integration specification below)
5. If the round produced a reusable technical lesson, append a concise tip to `agent/historical_experience/<target_gpu>/<target_op>/<target_backend_lower>/tips.md`

`rounds/round-N/summary.md` and `logs/optimize.md` must be updated in the same VALIDATE round. Do not leave placeholders or defer the log update.

Before issuing any command for the next round, the agent MUST have completed **all** of the following for the current round:
- benchmark/test completion confirmed
- accept/rollback decision made from finished data
- `rounds/round-N/summary.md` written
- `rounds/round-N/kernel_snapshot/` populated
- `logs/optimize.md` updated
- `logs/performance_trend.md` updated
- reusable tip appended when applicable

**Under `workspace-mode`**:
Validation is split into a local gate (within minimal environment) and an integration gate (after syncing back to main repo).
Only passing the integration gate counts as a truly accepted version.
SYNC_BACK step: only sync accepted core changes — do not carry over scaffolding or temporary code.

### 5. ACCEPT / REPORT

**Goal**: Update lineage and leave reusable context for the next round.

**ACCEPT required actions**:
- Confirm `logs/optimize.md` already reflects the latest accepted round
- Record cumulative improvement relative to baseline
- Mark which directions were effective, ineffective, or need revisiting
- Produce candidate directions for the next round

**REPORT** (output when campaign terminates, written to the `## Final Report` section at the end of `logs/optimize.md`):
- Baseline vs final best comparison (with full validation data)
- Total cumulative improvement
- List of key effective optimizations
- List of verified ineffective directions
- If continuing optimization, top three recommended next steps
- Detailed data references to the corresponding `rounds/round-N/summary.md` files

## Rollback Rules

The following situations require rollback of the current round's candidate:
- Build failure
- Correctness failure
- `Check` failure in benchmark
- Clear regression compared to current best
- Results are too volatile to confirm whether improvement is real

Rollback operations:
1. Revert the candidate change (`repo-mode`: `git checkout -- <modified_files>` or `git revert <commit>`; `workspace-mode`: roll back the local candidate without affecting the main repo)
2. Write `rounds/round-N/summary.md` and mark the round as rollback with the failure reason
3. Immediately update `logs/optimize.md`, including the optimization history and verified ineffective directions
4. Return to ANALYZE with a new direction; rollback is not, by itself, a campaign termination signal
5. If the failed round exposed a reusable pitfall or signal, append a concise tip to `agent/historical_experience/<target_gpu>/<target_op>/<target_backend_lower>/tips.md`

## Historical Experience Capture

After every completed round, whether accepted or rolled back, evaluate whether the round is worth recording in the long-lived tips store:

`agent/historical_experience/<target_gpu>/<target_op>/<target_backend_lower>/tips.md`

If the path does not exist yet, create the missing directories and `tips.md`, then append the new entry.

Append only when the round yielded reusable knowledge beyond the local campaign, such as:

- a hardware-specific constraint or backend/compiler quirk
- a config or parameter pattern that consistently helps or hurts
- a failure signature with a clear root cause
- a profiler signal that maps cleanly to an actionable next step
- a validation / measurement trick that avoids wasted rounds

Do not copy the whole round summary into `tips.md`. Append only the reusable takeaway.

Recommended append format:

```markdown
### <YYYY-MM-DD HH:MM> round-N — ACCEPTED / ROLLED BACK
- Context: <shape family / kernel path / hardware scope>
- Signal: <what was observed>
- Takeaway: <reusable lesson>
- Applicability: <when this tip should be reused, and when it should not>
```

## Git Integration Specification

When `git_commit=true`, each accepted version corresponds to a git commit, forming a lineage (referencing AVO's design). When `git_commit=false`, skip the commit but still record the accepted version in the log.

**Commit timing**: After VALIDATE passes, check the `git_commit` flag; if `true`, commit immediately.

**Commit message format**:

```
[optimize] <target_op> <target_backend> round-<N>: <one-line summary>

Hypothesis: <optimization hypothesis for this round>
Result: <aggregate score change>
Details: <campaign_dir>/logs/optimize.md
```

Example:

```
[optimize] gemm_fp8_blockwise TRITON round-3: increase num_stages from 2 to 3

Hypothesis: Increase software pipelining depth to hide memory access latency
Result: geomean 301 -> 319 TFLOPS (+6.0%)
Details: agent/workspace/gemm_fp8_blockwise_triton_gfx942_20260412/logs/optimize.md
```

**Rollback**: `git revert <commit>` to roll back a single version.

**Note**: The campaign directory (`agent/workspace/`) is not tracked by git by default. Only kernel code changes enter the git lineage. The `rounds/` summaries and raw artifacts are stored only in the campaign directory.

## Optimization Log Template

Each campaign maintains a `logs/optimize.md`, updated in real-time so humans can check progress at any time.

`logs/optimize.md` and `logs/performance_trend.md` are append-only history files:

- Never delete, truncate, or rewrite existing content
- Only append new sections, rows, or correction notes
- If a prior entry is inaccurate, append a correction instead of erasing the original record

```markdown
# <target_op> <target_backend> Optimization Log

## Basic Information
- Target operator: <target_op>
- Implementation language: <target_lang>
- Backend: <target_backend>
- Target GPU: <target_gpu>
- Campaign: <campaign_dir>
- Start time: <YYYY-MM-DD HH:MM>
- Current status: Optimizing (round-N)

## Baseline
| Shape (MxNxK) | Forward TFLOPS | Backward TFLOPS | Check |
|---------------|---------------|-----------------|-------|
| ... | ... | ... | ... |
- Forward aggregate: <baseline_forward_score>
- Backward aggregate: <baseline_backward_score_or_->
- Detailed data: rounds/round-1/summary.md

## Optimization History

### round-2 — <one-line description of changes>
- Time: <YYYY-MM-DD HH:MM>
- Validation level: quick / full
- Hypothesis: <why this change was made>
- Changes: <which files and parameters were modified>
- Result: <aggregate score change> ✅/❌
- Test: PASS/FAIL
- Decision: accept / rollback
- Detailed data: rounds/round-2/summary.md
- Notes: <failure reason or key observations>

### round-3 — ...

## Current Best
| Shape (MxNxK) | Baseline | Current Best | Improvement |
|---------------|----------|-------------|-------------|
| ... | ... | ... | ... |
- Aggregate score: baseline <X> → current <Y> (+Z%)

## Directions to Try
- [ ] <Direction 1>
- [ ] <Direction 2>
- [x] ~~<Verified ineffective direction>~~ (verified in round-N)

## Verified Ineffective Directions
| Direction | Version | Failure Reason |
|-----------|---------|---------------|
| ... | round-N | ... |

## Final Report
(Filled in when campaign terminates)
- Baseline aggregate score: <X>
- Final best aggregate score: <Y> (+Z%)
- Total iterations: <N> (accepted: <A>, rollback: <B>)
- Key effective optimizations: ...
- Verified ineffective directions: ...
- If continuing optimization, recommended next three steps: ...
```

## Round Summary Template

The canonical detailed artifact for each round is `<campaign_dir>/rounds/round-N/summary.md`.

- `round-1` is the baseline round
- `round-2+` are optimization attempts
- If a round emits raw benchmark CSVs, command captures, or other detailed outputs, store them under `<campaign_dir>/rounds/round-N/artifacts/`

The summary must follow the same round-oriented structure used by the iteration rules:

```markdown
## Hypothesis

<For round-1: "Baseline round. No optimization change yet; record the starting correctness/performance and choose representative shapes.">
<For round-N >= 2: optimization hypothesis for this round>

## Single change

<For round-1: "No code change. Freeze the starting kernel and environment state.">
<For round-N >= 2: exact code/config change>

## Results

### Validation
- Time: <YYYY-MM-DD HH:MM>
- Validation level: quick / full
- Test command: <test command>
- Benchmark command: <benchmark command>
- Raw artifacts: `artifacts/<file>` (if any)

| Shape / label | M | N | K | Result | Forward TFLOPS | Backward TFLOPS | Delta vs current best |
|---|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... | ... |

### Aggregate
- Forward aggregate (geomean): <score>
- Backward aggregate (geomean): <score> (if applicable)
- vs baseline: +X%
- vs current best: +Y%

## Decision

`BASELINE` / `ACCEPTED` / `ROLLED BACK`

<Why this round was accepted, rolled back, or recorded as the baseline>

## Attribution

<Why this specific round behaved this way>

## Next direction

<What to try next>

## Rollback Analysis

<Only for rolled-back rounds>
```

## Stagnation Detection and Conditional Intervention

Drawing from AVO's continuous evolution approach, stagnation is not a stop signal but a trigger for intervention.

Any of the following conditions can trigger intervention:
- Two consecutive candidates were rolled back
- Multiple consecutive rounds made only minor adjustments in the same direction with no measurable improvement
- The profiler-identified bottleneck has remained unchanged for an extended period
- Recent rounds only show parameter jitter with no structurally new hypotheses

Once triggered, a `stagnation review` must be performed:

1. Review the benefit curve of recent accepted versions
2. Review failed attempts to identify proven ineffective directions
3. Re-examine profiler results, reference implementations, and hardware documentation; if profiling has not been done yet, do it now
4. Generate at least 3 fundamentally different new directions
5. Prioritize directions that have not been explored recently

Recommended direction-switching categories:
- Tile / launch parameters
- Memory layout / data movement
- Software pipelining / overlap
- Occupancy / register / LDS resource allocation
- Backend switching or reference implementation comparison
- Algorithm-level reordering, branch elimination, kernel fusion

## Termination Conditions

The campaign may terminate only when at least one of the following conditions is satisfied:

| ID | Condition | Notes |
|----|-----------|-------|
| `T1` | `performance_target` has been reached | Only applies when `performance_target` is not `null` |
| `T2` | An acceptable hardware-efficiency range has been reached | For example, a clearly acceptable fraction of peak efficiency |
| `T3` | `max_iterations` limit has been reached | Only if configured; if set, it must be `< 120` |
| `T4` | `max_duration` limit has been reached | Only if configured |
| `T5` | The user explicitly requests stop | Always valid |

Before writing REPORT, add a termination-check block to `logs/optimize.md` and state which condition passed:

```markdown
### Termination Check
- T1 performance_target: ❌ / ✅
- T2 hardware efficiency: ❌ / ✅
- T3 max_iterations reached: ❌ / ✅
- T4 max_duration reached: ❌ / ✅
- T5 user requested stop: ❌ / ✅
-> Satisfied condition(s): T<N>
```

If none of the conditions are satisfied, the campaign must return to ANALYZE instead of terminating.

A REPORT must be output upon termination (see ACCEPT / REPORT phase).

## Cross-Skill Reference Paths

| Phase | What you need | Where to read |
|-------|---------------|---------------|
| Full workflow entry point | Project information collection → DEFINE_TARGET → PREPARE_ENVIRONMENT | Project skill → [`../SKILL.md`](../SKILL.md) |
| ENVIRONMENT_BASELINE / VALIDATE | Project test commands, benchmark commands | Corresponding project skill |
| ANALYZE | Profiling methods | [`../../tool-rocprof/SKILL.md`](../../tool-rocprof/SKILL.md) |
| ANALYZE | Architecture constraints | `../../hardware/<arch>/SKILL.md` + `optimization-guide.md` |
| ANALYZE / OPTIMIZE | Language-level optimization techniques | [`../triton/SKILL.md`](../triton/SKILL.md) or [`../hip/SKILL.md`](../hip/SKILL.md) |
| ANALYZE / OPTIMIZE | CK template and pipeline tuning | [`../hip/ck.md`](../hip/ck.md) |
| ANALYZE | Historical examples and cross-generation comparison | [`../examples.md`](../examples.md) and [`../../hardware/hardware-comparison.md`](../../hardware/hardware-comparison.md) |
| Before round-1 / after each round | Historical reusable tips | `agent/historical_experience/<target_gpu>/<target_op>/<target_backend_lower>/tips.md` (if present) |

## Execution Reminders

- Read [`../../../rules/iteration_rules.mdc`](../../../rules/iteration_rules.mdc) before the first round and keep it active for the whole campaign.
- Correctness first, then performance.
- Advance only one primary hypothesis per round.
- When `git_commit=true`, accepted versions must be git committed, forming a traceable lineage.
- When stagnated, switch directions — do not endlessly fine-tune in the same direction.
- Keep logs updated in real-time so humans can check current progress and historical decisions at any time.
- `logs/optimize.md` and `logs/performance_trend.md` are append-only; never delete old content, only append new entries or correction notes.
- Every round, including the baseline, must keep `rounds/round-N/summary.md` up to date; every VALIDATE round must update that summary and `logs/optimize.md` together.
- After `related_work.md` is done and before the first round starts, read `agent/historical_experience/<target_gpu>/<target_op>/<target_backend_lower>/tips.md` if it exists.
- After every accepted or rolled-back round, append a reusable lesson to that tips file when the round is worth preserving.
- All timestamps must be recorded to minute precision in the format `YYYY-MM-DD HH:MM`. **Always obtain the current time by running `date '+%Y-%m-%d %H:%M'` in the shell**; do not rely on the system prompt date or any other source, as they may lack time-of-day precision.
- **Never pipe benchmark or test commands through `| tail`, `| head`, `| grep`, or any filter.** Piping changes stdout from line-buffered to fully-buffered, hiding all intermediate output. This makes it look like the command is hung, leading to premature kills and lost data. If you need to inspect a subset of the output, redirect the full output to a file first (`> out.txt 2>&1`), then read or search the file after the command completes.
- **Never treat a backgrounded command as a completed round.** If a shell command times out and continues in the background, that is an unfinished round-state, not a result.
- **Never leave a round half-closed.** If `summary.md`, `kernel_snapshot/`, `logs/optimize.md`, or `logs/performance_trend.md` are missing for the current round, the agent must finish that bookkeeping before any new optimization action.