# Optimization Process Specification

This file defines the **detailed optimization process** for `kernel-optimize`.

The default prerequisite is (agent proceeds through: project skill → `../SKILL.md` (understand requirements) → project skill (collect information) → `../SKILL.md` (execute) → this file):
- All required information has been collected from the project skill per `../SKILL.md`'s "Prerequisite Information"
- DEFINE_TARGET (parameter structuring) and PREPARE_ENVIRONMENT (campaign directory setup) have been completed in `../SKILL.md`
- The campaign directory for this round has been established, and `manifest.yaml` has been filled in
- `related_work.md` has been created by the `SURVEY_RELATED_WORK` step in `../SKILL.md`
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
| `performance_target` | Target performance |
| `primary_metric` | Primary comparison metric(s); depends on operator type (e.g., GEMM: `Forward TFLOPS`, elementwise: `Forward GB/s`) |
| `git_commit` | Whether to git commit accepted versions |
| `git_branch` | Current optimization branch (or `none`) |
| `max_iterations` | Maximum iteration count (optional) |
| `project_skill` | Corresponding project skill |

## Confirm Previously Obtained Information

The following information should have been obtained during the project skill phase and DEFINE_TARGET phase. Do a final confirmation before starting the optimization loop:

| Information | Purpose | Example |
|-------------|---------|---------|
| **Kernel source file path** | Read code during ANALYZE | `primus_turbo/triton/gemm/gemm_fp8_kernel.py` |
| **Focused test command** | Full correctness validation | `pytest tests/pytorch/ops/test_gemm_fp8.py -v -k "blockwise and TRITON"` |
| **Focused benchmark command** | Full performance evaluation | `PRIMUS_TURBO_GEMM_BACKEND=TRITON python benchmark/ops/bench_gemm_turbo.py --dtype fp8 --granularity blockwise` |
| **Quick test command** | Fast correctness validation each round | `pytest tests/pytorch/ops/test_gemm_fp8.py -v -k "blockwise and TRITON" --maxfail=3` |
| **Quick benchmark command** | Fast performance evaluation each round | Run full benchmark, then extract `representative_shapes` data; or use script's shape filter parameter (if supported) |
| **Benchmark output column names** | Parse results, compute scores | `Forward TFLOPS`, `Backward TFLOPS` (primary_metric), `Check` (correctness gate) |
| **Rebuild requirements** | Whether rebuild is needed after code changes | Triton: no rebuild needed; HIP: incremental rebuild required |
| **Related-work report** | Seed early hypotheses with external and internal baselines | `<campaign_dir>/related_work.md` |

If anything is missing, refer back to the project skill or `../SKILL.md`'s manifest to fill in the gaps.

## Two-Level Validation

The optimization loop uses two levels of validation:

| Level | When to use | Content |
|-------|------------|---------|
| **quick** | Default for each VALIDATE round | Test + benchmark on a representative shape subset (3-5 shapes), for fast feedback |
| **full** | BASELINE, final acceptance, or when agent deems necessary | Test + benchmark on all shapes |

The specific commands for quick and full validation are provided by the project skill and recorded in the manifest.

Active validation set by level:
- **quick** → all `representative_shapes`
- **full** → all `target_shapes`

Within a chosen validation level, do not cherry-pick a smaller subset.

The quick validation aggregate score can be used for fast accept/reject decisions. When quick results are borderline (improvement < 5%) or involve high-risk changes like loop structures, the agent should proactively upgrade to full validation for confirmation.

## Overall Flow

```text
ENVIRONMENT_BASELINE
  -> [ANALYZE -> OPTIMIZE -> VALIDATE -> ACCEPT/ROLLBACK] (iteration loop)
  -> REPORT
```

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

**Multi-metric handling**: When `primary_metric` contains multiple metrics (e.g., `Forward TFLOPS, Backward TFLOPS`), compute the aggregate score for each metric separately. Acceptance condition: all metrics' aggregate scores must not regress, and at least one metric must show improvement.

### Noise Assessment

- When improvement is < 2%, it is considered near the noise range
- In this case, re-measure at least 3 times, compute mean and standard deviation
- If mean improvement > 1% and standard deviation < half of the improvement magnitude, consider it a valid improvement
- Otherwise, treat as noise and do not accept

### Acceptance Rules

- `aggregate score` must not be lower than the current best
- If it only matches the current best, there must be a clear additional benefit (e.g., broader applicability, more stable results)
- If any core shape regresses > 3%, default to rejection; unless this round is explicitly a targeted shape optimization

### Computation Example

Benchmark results (3 shapes):

| M | N | K | Forward TFLOPS | Check |
|---|---|---|---------------|-------|
| 4096 | 4096 | 4096 | 320 | PASS |
| 2048 | 8192 | 4096 | 305 | PASS |
| 1024 | 4096 | 8192 | 290 | PASS |

```
aggregate score = (320 * 305 * 290) ^ (1/3) = 304.8
```

Baseline aggregate score = 285.0 → improvement = (304.8 - 285.0) / 285.0 = +6.9% → accept.

## Phase Descriptions

### 1. ENVIRONMENT_BASELINE

**Goal**: Freeze the starting point and establish a unified comparison baseline.

**What "focused" means**:
- Focused test = test subset limited to `target_op` + `target_backend` (e.g., `-k "blockwise and TRITON"`)
- Focused benchmark = benchmark limited to `target_backend`, covering all `target_shapes`

**Steps**:
1. Confirm the current environment can build and run correctly
2. Run **full** focused test, confirm all PASS
3. Run **full** focused benchmark, write results to `<campaign_dir>/results/baseline.md`
4. Compute baseline `score vector` and `aggregate score` per scoring specification
5. Record backend configuration and key environment state

BASELINE always uses full validation to ensure the starting data is complete and reliable.

**Baseline record template** (write to `<campaign_dir>/logs/optimize.md`):

```markdown
## Baseline
- Time: <timestamp>
- Backend: <target_backend>
- GPU: <target_gpu>
- Commit: <git_hash>
- Validation level: full

- Aggregate score (geomean): 278.0
- All Check: PASS
- Detailed data: results/baseline.md
```

**Output**:
- `results/baseline.md` (detailed data) + baseline aggregate score
- Current best = baseline

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
2. Run **quick** test
3. After all correctness passes, run **quick** benchmark
4. Compute `score vector` and `aggregate score` per scoring specification
5. Compare against current best
6. When results are borderline (improvement < 5%) or involve high-risk changes, upgrade to **full** validation for confirmation

Write results to `<campaign_dir>/results/v<N>.md` (note the validation level).

**Hard gates**:
- Build failure → reject immediately
- Correctness failure → reject immediately
- `Check = FAIL` in benchmark → reject immediately
- Aggregate score regression → default reject
- Core shape regression > 3% → default reject

**After passing**:
- Update current best
- If `git_commit=true`: git commit (see git integration specification below)
- Write this round's results to campaign log

**Under `workspace-mode`**:
Validation is split into a local gate (within minimal environment) and an integration gate (after syncing back to main repo).
Only passing the integration gate counts as a truly accepted version.
SYNC_BACK step: only sync accepted core changes — do not carry over scaffolding or temporary code.

### 5. ACCEPT / REPORT

**Goal**: Update lineage and leave reusable context for the next round.

**ACCEPT required actions**:
- Update accepted history in the campaign log
- Record cumulative improvement relative to baseline
- Mark which directions were effective, ineffective, or need revisiting
- Produce candidate directions for the next round

**REPORT** (output when campaign terminates, written to the `## Final Report` section at the end of `logs/optimize.md`):
- Baseline vs final best comparison (with full validation data)
- Total cumulative improvement
- List of key effective optimizations
- List of verified ineffective directions
- If continuing optimization, top three recommended next steps
- Detailed data references to corresponding `.md` files under `results/`

## Rollback Rules

The following situations require rollback of the current round's candidate:
- Build failure
- Correctness failure
- `Check` failure in benchmark
- Clear regression compared to current best
- Results are too volatile to confirm whether improvement is real

Rollback operations:
- `repo-mode`: `git checkout -- <modified_files>` or `git revert <commit>`
- `workspace-mode`: Roll back this round's local changes without affecting the main repo
- Rollback reason must be written to the log to avoid repeating the same mistake

## Git Integration Specification

When `git_commit=true`, each accepted version corresponds to a git commit, forming a lineage (referencing AVO's design). When `git_commit=false`, skip the commit but still record the accepted version in the log.

**Commit timing**: After VALIDATE passes, check the `git_commit` flag; if `true`, commit immediately.

**Commit message format**:

```
[optimize] <target_op> <target_backend> v<N>: <one-line summary>

Hypothesis: <optimization hypothesis for this round>
Result: <aggregate score change>
Details: <campaign_dir>/logs/optimize.md
```

Example:

```
[optimize] gemm_fp8_blockwise TRITON v3: increase num_stages from 2 to 3

Hypothesis: Increase software pipelining depth to hide memory access latency
Result: geomean 301 -> 319 TFLOPS (+6.0%)
Details: agent/workspace/gemm_fp8_blockwise_triton_gfx942_20260412/logs/optimize.md
```

**Rollback**: `git revert <commit>` to roll back a single version.

**Note**: The campaign directory (`agent/workspace/`) is not tracked by git by default. Only kernel code changes enter the git lineage. `.md` files under `results/` are stored only in the campaign directory.

## Optimization Log Template

Each campaign maintains a `logs/optimize.md`, updated in real-time so humans can check progress at any time.

```markdown
# <target_op> <target_backend> Optimization Log

## Basic Information
- Target operator: <target_op>
- Implementation language: <target_lang>
- Backend: <target_backend>
- Target GPU: <target_gpu>
- Campaign: <campaign_dir>
- Start time: <timestamp>
- Current status: Optimizing (v<N>)

## Baseline
| Shape (MxNxK) | Forward TFLOPS | Check |
|---------------|---------------|-------|
| ... | ... | ... |
- Aggregate score: <baseline_score>

## Optimization History

### v1 — <one-line description of changes>
- Time: <timestamp>
- Validation level: quick / full
- Hypothesis: <why this change was made>
- Changes: <which files and parameters were modified>
- Result: <aggregate score change> ✅/❌
- Test: PASS/FAIL
- Decision: accept / rollback
- Detailed data: results/v1.md
- Notes: <failure reason or key observations>

### v2 — ...

## Current Best
| Shape (MxNxK) | Baseline | Current Best | Improvement |
|---------------|----------|-------------|-------------|
| ... | ... | ... | ... |
- Aggregate score: baseline <X> → current <Y> (+Z%)

## Directions to Try
- [ ] <Direction 1>
- [ ] <Direction 2>
- [x] ~~<Verified ineffective direction>~~ (verified in vN)

## Verified Ineffective Directions
| Direction | Version | Failure Reason |
|-----------|---------|---------------|
| ... | v<N> | ... |

## Final Report
(Filled in when campaign terminates)
- Baseline aggregate score: <X>
- Final best aggregate score: <Y> (+Z%)
- Total iterations: <N> (accepted: <A>, rollback: <B>)
- Key effective optimizations: ...
- Verified ineffective directions: ...
- If continuing optimization, recommended next three steps: ...
```

## Result File Template

Detailed data from each VALIDATE round is written to `results/v<N>.md` (BASELINE is written to `results/baseline.md`).

```markdown
# v<N> — <one-line description of changes>

- Time: <timestamp>
- Validation level: quick / full
- Hypothesis: <optimization hypothesis for this round>

## Correctness
- Command: <test command>
- Result: <X passed, Y failed>

## Benchmark
- Command: <benchmark command>

| M | N | K | Forward TFLOPS | Backward TFLOPS | Check |
|---|---|---|---------------|----------------|-------|
| ... | ... | ... | ... | ... | ... |

## Score
- Forward aggregate (geomean): <score>
- Backward aggregate (geomean): <score> (if applicable)
- vs baseline: +X%
- vs current best: +Y%
```

## Stagnation Detection and Conditional Intervention

Drawing from AVO's continuous evolution approach, stagnation is not a stop signal but a trigger for intervention.

Any of the following conditions can trigger intervention:
- `N` consecutive candidates were not accepted, default `N = 5`
- Multiple consecutive rounds of minor adjustments in the same direction with no measurable improvement
- Profiler-identified bottleneck has remained unchanged for an extended period
- Recent rounds only show parameter jitter with no structurally new hypotheses

Once triggered, a `stagnation review` must be performed:

1. Review the benefit curve of recent accepted versions
2. Review failed attempts to identify proven ineffective directions
3. Re-examine profiler results, reference implementations, and hardware documentation
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

Any of the following conditions can terminate the current optimization campaign:
- `performance_target` has been reached
- An acceptable hardware efficiency range has been reached
- Recent accepted versions' gains are below the noise level
- The hypothesis pool has been largely exhausted
- Remaining directions have excessive risk and insufficient expected benefit
- `max_iterations` limit has been reached (if set)
- User requests stop, or time / compute budget is exhausted

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

## Execution Reminders

- Read [`../../../rules/iteration_rules.mdc`](../../../rules/iteration_rules.mdc) before the first round and keep it active for the whole campaign.
- Correctness first, then performance.
- Advance only one primary hypothesis per round.
- When `git_commit=true`, accepted versions must be git committed, forming a traceable lineage.
- When stagnated, switch directions — do not endlessly fine-tune in the same direction.
- Keep logs updated in real-time so humans can check current progress and historical decisions at any time.
