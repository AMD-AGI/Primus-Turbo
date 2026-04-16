---
name: kernel-optimize
description: AI-driven operator performance optimization framework. Defines the optimization loop, execution environment selection, knowledge routing, and logging conventions to drive agent-autonomous iteration toward hardware limits.
---

# Operator Performance Optimization Framework

This skill defines the **general-purpose operator optimization loop**, driving the agent to autonomously iterate toward hardware limits.

This skill is responsible for defining:
- Prerequisites needed for optimization (interface contract)
- Parameter confirmation and directory structure for optimization campaigns
- High-level optimization loop (reference: AVO — Agentic Variation Operators)
- Knowledge routing approach
- General principles for logging, lineage, and stagnation handling

This skill is **not responsible for**:
- Hardcoding source paths, test commands, or benchmark commands for any specific project (provided by the project skill)
- Replacing language-specific, hardware-specific, or profiling-specific documentation

## Prerequisite Information

**This section is the interface contract between kernel-optimize and the project skill.** Before starting optimization, the agent must collect the following information from the corresponding project skill:

| Requirement | Description | Where to find in project skill |
|-------------|-------------|-------------------------------|
| **Kernel source file path** | Location of the kernel code to optimize | Code structure / file mapping table |
| **Focused test command** | Correctness test limited to the target operator + backend (full) | Testing section |
| **Focused benchmark command** | Performance test limited to the target backend (full) | Benchmark section |
| **Quick validation script template** | Self-contained correctness + benchmark script template generated into the campaign directory during PREPARE_ENVIRONMENT; representative shapes are filled in after BASELINE | Quick validation section in project skill |
| **Benchmark output format** | CSV column names, which columns are performance metrics (`Forward TFLOPS`, `Backward TFLOPS`, etc.), which column is the correctness gate (`Check`) | Benchmark output description |
| **Scoring rules** | How to compute `aggregate score` from benchmark output (e.g., geometric mean) | Operator optimization scoring section in project skill |
| **execution_mode recommendation** | `repo-mode` vs `workspace-mode`, and the corresponding build/rebuild approach | Operator optimization environment section in project skill |
| **Rebuild requirements** | Whether rebuild is needed after code changes, and the build command | Build section |

After the agent has collected all the above information, return to this file to execute DEFINE_TARGET.

Before entering the optimization loop, read [`../../rules/iteration_rules.mdc`](../../rules/iteration_rules.mdc). Those rules are hard constraints for every backend: one hypothesis per round, correctness before performance, benchmark the full active validation set, and accept-or-rollback lineage.

For validation scope, interpret that contract as:
- **full validation** → run all `target_shapes`
- **quick validation** → run all `representative_shapes`

Within a chosen validation level, the agent must not cherry-pick a smaller subset.

## Input Parameters

During the DEFINE_TARGET phase, the user instruction + prerequisite information must be organized into the following structured parameters:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `target_op` | Target operator | `gemm_fp8_blockwise` |
| `target_backend` | Target backend | `TRITON` |
| `target_lang` | Implementation language | `TRITON` / `HIP` |
| `target_gpu` | Target GPU architecture | `gfx942` / `gfx950` |
| `target_shapes` | Full shape set of interest for the campaign; quick validation uses a separate representative subset recorded in `representative_shapes` | A full shape list or `all` (use benchmark default shape set) |
| `performance_target` | Performance target | `>500 TFLOPS`, `>60% peak efficiency`, or `null`; defaults to `null` if unspecified |
| `primary_metric` | Primary performance metric(s), depending on operator type | GEMM: `"Forward TFLOPS"` / `"Forward TFLOPS, Backward TFLOPS"`; elementwise: `"Forward GB/s"` |
| `project_skill` | Corresponding project skill | `primus-turbo-develop` |
| `execution_mode` | Execution environment, referencing project skill recommendation, decided by agent | `repo` / `workspace` |
| `git_commit` | Whether to git commit accepted versions | `true` (default) / `false` |
| `git_branch` | Optimization branch strategy | `auto` (default, auto-creates `optimize/<campaign>` branch) / `none` / `<custom branch name>` |
| `max_iterations` | Maximum iteration count (optional) | `10`; if unspecified, leave `null` and let termination conditions decide; if set, it must be `< 120` |
| `max_duration` | Maximum campaign runtime (optional) | `"4h"` / `"90m"`; if unspecified, runtime is unbounded |

## Overall Loop

```text
DEFINE_TARGET
  -> PREPARE_ENVIRONMENT
  -> SURVEY_RELATED_WORK
  -> READ_HISTORICAL_TIPS
  -> BASELINE
  -> [ANALYZE -> OPTIMIZE -> VALIDATE]  (iteration loop)
  -> REPORT
```

| Phase | What to do |
|-------|-----------|
| **DEFINE_TARGET** | Organize user instruction + project skill information into structured parameters, confirm completeness, **confirm target with user before starting** |
| **PREPARE_ENVIRONMENT** | Set up campaign directory, record metadata, and generate the quick validation script scaffold |
| **SURVEY_RELATED_WORK** | Survey current SOTA implementations, docs, and competitor baselines; write findings to `related_work.md` before the first baseline run |
| **READ_HISTORICAL_TIPS** | If `agent/historical_experience/<target_gpu>/<target_op>/<target_backend_lower>/tips.md` exists, read it after related work and before the first round |
| **BASELINE** | Record starting correctness and performance |
| **ANALYZE** | Read code, profile, consult skill knowledge, generate optimization hypotheses |
| **OPTIMIZE** | Implement a single primary hypothesis with small incremental changes |
| **VALIDATE** | Correctness hard gate + benchmark comparison; pass → accept (+ git commit if `git_commit=true`), fail → rollback; keep `rounds/round-N/summary.md` and `logs/optimize.md` synchronized round by round |
| **REPORT** | Summarize best version, effective directions, failed directions, and next-step recommendations; hand back to project skill for final acceptance |

For detailed optimization process, gating rules, rollback, stagnation detection, and log templates, see [`workflow/optimize-loop.md`](workflow/optimize-loop.md).

## DEFINE_TARGET

When the agent reaches this point, it should have already collected all required information from the project skill per the "Prerequisite Information" section. This phase organizes the user instruction + prerequisite information into structured parameters.

**Step 1: Populate parameters**

| Parameter | Extraction method |
|-----------|-------------------|
| `target_op` | Identify operator name and precision from user instruction |
| `target_backend` | Identify from user instruction; if unspecified, select from the project skill's backend table |
| `target_lang` | Determined by `target_backend`: `TRITON` → Triton, `CK` / `HIPBLASLT` / `TURBO` → HIP |
| `target_gpu` | Identify GPU model from user instruction, map to architecture codename (e.g., MI300X → `gfx942`, MI355X → `gfx950`) |
| `target_shapes` | Use if specified by user; otherwise use the benchmark default shape set |
| `performance_target` | Use if specified by user; otherwise default to `null` |
| `primary_metric` | Get available metrics from the project skill's scoring section; use if specified by user |
| `execution_mode` | Reference the project skill's recommendation, decided by agent based on task characteristics |
| `git_commit` | Default `true`; set to `false` if user specifies no commit |
| `git_branch` | Default `auto`; use if specified by user |
| `max_iterations` | Use if specified by user and validate that it is `< 120`; otherwise leave empty, controlled by termination conditions |
| `max_duration` | Use if specified by user; otherwise leave empty, controlled by termination conditions |

**Step 2: Confirm prerequisite information is complete**

Do a final check against the "Prerequisite Information" section:
- [ ] Kernel source file path
- [ ] Focused test command
- [ ] Focused benchmark command
- [ ] Quick validation script template
- [ ] Benchmark output format and available performance metric columns
- [ ] Scoring rules (e.g., geometric mean)
- [ ] `execution_mode` decision
- [ ] Whether rebuild is needed after changes, and the rebuild command

If anything is missing, go back to the project skill to fill in the gaps.

**Step 3: Confirm target with user**

List the agent's inferred key parameters and confirm with the user before starting. At minimum include:

- `target_op`, `target_backend`, `target_gpu`
- `primary_metric`: Optimize forward only? Or forward + backward? Or custom metric?
- `performance_target`: Specific number or `null`?
- `execution_mode`: repo or workspace?
- `git_commit` / `git_branch`
- `max_iterations` / `max_duration` (if applicable)
- Special constraints (e.g., cannot modify certain interfaces)

The user can confirm directly or adjust parameters. After confirmation, proceed to PREPARE_ENVIRONMENT.

## PREPARE_ENVIRONMENT

Set up the campaign directory for this optimization round, and create an optimization branch based on the `git_branch` parameter.

**Step 1: Create optimization branch** (if `git_branch` is not `none`)

- `git_branch=auto`: `git checkout -b optimize/<campaign_name>`
- `git_branch=<custom>`: `git checkout -b <custom branch name>`
- `git_branch=none`: Do not switch branches, work on the current branch

**Step 2: Set up campaign directory**

```text
agent/workspace/<campaign_name>/
├── logs/
│   ├── optimize.md       # Optimization log (main file)
│   └── performance_trend.md
├── profiles/              # Profiler output
├── related_work.md        # Related-work / SOTA survey for this campaign
├── rounds/
│   ├── round-1/
│   │   ├── summary.md     # Baseline round summary
│   │   ├── kernel_snapshot/
│   │   └── artifacts/     # Optional raw benchmark/test outputs for this round
│   └── round-N/
│       ├── summary.md
│       ├── kernel_snapshot/
│       └── artifacts/
└── manifest.yaml          # Metadata
```

Campaign naming convention: `<op>_<backend>_<gpu>_<date>`, e.g., `gemm_fp8_blockwise_triton_gfx942_20260412`.

**Step 3: Write manifest.yaml**

```yaml
target_op: <target_op>
target_backend: <target_backend>
target_lang: <target_lang>
target_gpu: <target_gpu>
execution_mode: <repo | workspace>
project_skill: <project_skill_name>
performance_target: <null | "performance target description">
primary_metric: "<primary performance metric(s), comma-separated if multiple>"
target_shapes: <all | shape list>
kernel_source: <kernel source file path>
test_command: "<focused test command>"
benchmark_command: "<focused benchmark command>"
quick_command: "python <campaign_dir>/quick_test_bench.py"
representative_shapes: <representative shape list selected during BASELINE, used for quick validation>
related_work_file: <campaign_dir>/related_work.md
git_commit: <true | false>
git_branch: <branch name | none>
max_iterations: <integer < 120 | null>
max_duration: <"Nh" | null>
created: <YYYY-MM-DD HH:MM>
```

All campaign timestamps must be recorded to minute precision in the format `YYYY-MM-DD HH:MM`.

All per-round artifacts live under `<campaign_dir>/rounds/`. `round-1` is the baseline round, and optimization attempts start at `round-2`. The running comparison table lives at `<campaign_dir>/logs/performance_trend.md`.

**Step 4: Generate `quick_test_bench.py`**

Use the template from the project skill's quick validation section to generate `<campaign_dir>/quick_test_bench.py` while the project API context is still fresh.

- Leave `SHAPES` empty or fill it with temporary placeholders during PREPARE_ENVIRONMENT
- After BASELINE, select `representative_shapes` and update both `quick_test_bench.py` and `manifest.yaml`
- Prefer a single self-contained script that runs correctness + benchmark together for quick iteration

## SURVEY_RELATED_WORK

After PREPARE_ENVIRONMENT creates the campaign directory, perform a short related-work survey before BASELINE.

**Goal**: learn from current best-known implementations before spending time on blind local iteration.

**What the agent may do**:

- Search the project tree for existing implementations and nearby historical results.
- Search AMD / ROCm documentation for op-specific guidance, ISA features, and tuning constraints.
- Search public web sources for relevant open-source implementations and reported performance.
- Search competitor implementations and published operator performance on other stacks or chips when that comparison can reveal useful techniques or realistic ceilings.
- If code inspection is worthwhile, clone selected GitHub repositories into `agent/tmp/<campaign_name>/related-work/repos/`.

**What the agent must produce**:

- Write `<campaign_dir>/related_work.md` summarizing:
  - candidate implementations and libraries reviewed
  - reported performance claims and the hardware / shape context behind them
  - transferable implementation ideas worth trying in this campaign
  - caveats about reproducibility, hidden fusion, datatype mismatch, or incompatible hardware assumptions
  - a short shortlist of concrete optimization directions to test locally

**Constraints**:

- Treat `agent/tmp/<campaign_name>/related-work/` as ephemeral scratch space, not part of the accepted optimization lineage.
- Do not allow the survey to turn into open-ended browsing; stop once the agent has enough information to guide early hypotheses.
- The survey informs the campaign, but it does not replace the local baseline and validation loop.

Use `related-work-template.md` as the default output structure for `<campaign_dir>/related_work.md`.

## READ_HISTORICAL_TIPS

After `SURVEY_RELATED_WORK` finishes and before `round-1` starts, check whether a reusable tips file already exists for the same hardware / op / backend combination.

Use this path convention:

`agent/historical_experience/<target_gpu>/<target_op>/<target_backend_lower>/tips.md`

Example:

`agent/historical_experience/gfx950/gemm_fp8_blockwise/triton/tips.md`

Rules:

- Normalize the backend directory name to lowercase, e.g. `TRITON -> triton`, `CK -> ck`
- If the file exists, read it before BASELINE so the first hypothesis benefits from prior experience
- Treat it as reusable guidance, not as a substitute for current measurements, profiling, or validation
- If the first worthy lesson has no existing tips file yet, create the missing directories and `tips.md`, then append to it
- After every completed round, if the round produced a reusable technical lesson, append a concise tip to this same file

## Execution Environment

Optimization can be performed in two modes:

- **`repo-mode`**: Modify and validate directly in the upstream project. Code changes, tests, and benchmarks are all done in the main repository.
- **`workspace-mode`**: First set up a minimal development environment, iterate rapidly within it, then integrate back into the upstream project once optimization targets are met.

The project skill provides a recommendation, but the agent makes the final decision based on task characteristics. General guidelines:
- Small scope of changes, mainly parameter tuning, fast builds → `repo-mode`
- Extensive trial-and-error needed, writing new kernel from scratch, heavy main repo build pipeline → `workspace-mode`

### workspace-mode Minimal Development Environment

When the agent selects `workspace-mode`, set up a minimal development environment within the campaign directory, containing at least:

```text
agent/workspace/<campaign_name>/
├── src/                   # Minimal kernel implementation extracted from upstream
├── tests/                 # Targeted correctness tests
├── bench/                 # Targeted benchmarks
├── logs/
│   ├── optimize.md
│   └── performance_trend.md
├── profiles/
├── related_work.md
├── rounds/
│   ├── round-1/
│   │   ├── summary.md
│   │   ├── kernel_snapshot/
│   │   └── artifacts/
│   └── round-N/
│       ├── summary.md
│       ├── kernel_snapshot/
│       └── artifacts/
└── manifest.yaml
```

Setup principles:
- **Minimal**: Extract only the target kernel and its direct dependencies, not the entire project
- **Reproducible**: Record which commit the code was extracted from and which files were extracted
- **Faithful**: Tests and benchmarks must be equivalent to their upstream counterparts, ensuring trustworthy results
- **Clear integration path**: After optimization targets are met, only sync core kernel changes back to upstream — do not carry over scaffolding or temporary code

How to extract code from a specific project, build minimal tests, and benchmarks is guided by the corresponding project skill.

Regardless of mode, optimization artifacts (logs, profiles, benchmark results) are uniformly stored in the campaign directory.

## Workflow

The agent's full path: **project skill → this file (understand requirements) → project skill (collect information) → this file (DEFINE_TARGET / PREPARE_ENVIRONMENT / SURVEY_RELATED_WORK) → `../../rules/iteration_rules.mdc` → `workflow/optimize-loop.md` → project skill (acceptance)**.

Typical interaction sequence:

1. Agent is directed to this file from the project skill.
2. Read "Prerequisite Information" to understand what the optimization framework needs.
3. Return to the project skill and collect project information per the requirement checklist.
4. Return to this file with the information, execute DEFINE_TARGET → PREPARE_ENVIRONMENT.
5. Run `SURVEY_RELATED_WORK`: create `<campaign_dir>/related_work.md`, using `agent/tmp/<campaign_name>/related-work/` for any temporary repo clones or downloaded notes.
6. If `agent/historical_experience/<target_gpu>/<target_op>/<target_backend_lower>/tips.md` exists, read it before the first round.
7. Read [`../../rules/iteration_rules.mdc`](../../rules/iteration_rules.mdc) before the first BASELINE / VALIDATE round.
8. Read [`workflow/optimize-loop.md`](workflow/optimize-loop.md) and execute the BASELINE → ANALYZE → OPTIMIZE → VALIDATE loop.
9. Read language, hardware, profiling, related-work survey outputs, and historical tips as needed during ANALYZE / OPTIMIZE phases.
10. Output REPORT and hand back to the project skill for final acceptance.

## Knowledge Reference Table

Read the corresponding skill as needed based on `target_lang`, `target_gpu`, and the current phase. **Do not read everything at once.**

| What you need | Where to find it |
|---------------|-----------------|
| Linear iteration contract | [../../rules/iteration_rules.mdc](../../rules/iteration_rules.mdc) |
| Optimization process and gating rules | [workflow/optimize-loop.md](workflow/optimize-loop.md) |
| General Triton optimization techniques | [triton/SKILL.md](triton/SKILL.md) |
| General HIP optimization techniques | [hip/SKILL.md](hip/SKILL.md) |
| CK template / pipeline tuning | [hip/ck.md](hip/ck.md) |
| Hardware parameters and optimization strategies | [../hardware/\<arch\>/SKILL.md](../hardware/) + `optimization-guide.md` |
| Cross-generation hardware comparison | [../hardware/hardware-comparison.md](../hardware/hardware-comparison.md) |
| Profiling methods | [../tool-rocprof/SKILL.md](../tool-rocprof/SKILL.md) |
| Project code structure / build / test / benchmark / integration | Corresponding project skill, e.g., [../primus-turbo-develop/SKILL.md](../primus-turbo-develop/SKILL.md) |
| Related-work report structure | [related-work-template.md](related-work-template.md) |
| Historical reusable tips | `agent/historical_experience/<target_gpu>/<target_op>/<target_backend_lower>/tips.md` (if present) |
| Historical optimization cases | [examples.md](examples.md) |

## Logging and Lineage General Principles

The optimization process must maintain structured history (referencing AVO's lineage design) to support long-term iteration.

- When `git_commit=true`, each accepted version corresponds to a git commit, forming a lineage
- When `git_commit=false`, accepted versions are still recorded in logs but without git commits
- Failed attempts are recorded in the campaign log but do not enter the accepted lineage
- Accepted versions must have clear hypotheses, validation results, and acceptance rationale
- Logs serve both humans (can check progress at any time) and the agent (trace history to avoid repeated attempts)
- Every round must update its own `rounds/round-N/summary.md`, and every VALIDATE round must keep that summary synchronized with `logs/optimize.md`
- If a round reveals a reusable hardware / op / backend lesson, append a concise tip to `agent/historical_experience/<target_gpu>/<target_op>/<target_backend_lower>/tips.md`
- Logs and profiling results are stored in `agent/workspace/<campaign_name>/`
- For detailed log format, see [`workflow/optimize-loop.md`](workflow/optimize-loop.md)

## Stagnation Handling General Principles

Referencing AVO's continuous evolution mechanism: stagnation is not a stop signal, but a trigger for strategy switching.

When there is no improvement for multiple consecutive rounds, the agent should not just make minor adjustments in the same direction, but instead:
- Review recent versions and failed attempts
- Re-identify bottlenecks based on profiler results
- Switch to a fundamentally different optimization direction
- Revisit reference implementations and hardware documentation as needed
- Two consecutive rollbacks should trigger a stagnation review by default
- Continuous rollback is a signal to switch direction, not a reason to terminate early

For detailed stagnation detection and direction-switching rules, see [`workflow/optimize-loop.md`](workflow/optimize-loop.md).

## End-to-End Example

**User instruction**: "Please optimize the blockwise FP8 GEMM Triton implementation in Primus-Turbo, target GPU is MI300X."

**Step 1: Understand requirements**

Agent is directed from `primus-turbo-develop/SKILL.md` to this file. Reads "Prerequisite Information" and learns the optimization framework needs: kernel source file, focused test, focused benchmark, quick validation script template, benchmark output format, scoring rules, execution_mode recommendation, and rebuild requirements.

**Step 2: Collect project information**

Agent returns to `primus-turbo-develop/SKILL.md` and collects per the requirement checklist:
- Kernel: `primus_turbo/triton/gemm/gemm_fp8_kernel.py`
- Focused test: `pytest tests/pytorch/ops/test_gemm_fp8.py -v -k "blockwise and TRITON"`
- Focused benchmark: `PRIMUS_TURBO_GEMM_BACKEND=TRITON python benchmark/ops/bench_gemm_turbo.py --dtype fp8 --granularity blockwise`
- Quick validation template: generate `quick_test_bench.py` during PREPARE_ENVIRONMENT, then fill representative shapes after BASELINE
- Scoring: `Forward TFLOPS` geometric mean, `Check` as correctness gate
- Environment recommendation: Triton → `repo-mode`, no rebuild needed

**Step 3: DEFINE_TARGET**

Organize parameters: `target_op=gemm_fp8_blockwise`, `target_backend=TRITON`, `target_gpu=gfx942`, `execution_mode=repo`, `performance_target=null`. Verify completeness against the prerequisite information checklist.

Confirm with user:

> Confirm optimization target:
> - Operator: blockwise FP8 GEMM, backend: TRITON, GPU: MI300X (gfx942)
> - Primary metric: Forward TFLOPS (do you also want to optimize Backward?)
> - Performance target: `null` unless you want to set a concrete target
> - Execution mode: repo-mode, git_commit=true, git_branch=auto
> - Please confirm or adjust.

Proceed to PREPARE_ENVIRONMENT after user confirmation.

**Step 4: PREPARE_ENVIRONMENT**

1. Create optimization branch: `git checkout -b optimize/gemm_fp8_blockwise_triton_gfx942_20260412`
2. Create `agent/workspace/gemm_fp8_blockwise_triton_gfx942_20260412/`
3. Create subdirectories `logs/`, `profiles/`, `rounds/`
4. Write `manifest.yaml`
5. Create `rounds/round-1/` as the baseline round scaffold
6. Generate `quick_test_bench.py` with placeholder `SHAPES`; fill it after BASELINE selects representative shapes

**Step 5: SURVEY_RELATED_WORK**

1. Review local project implementations and AMD / ROCm references
2. Search external related work and competitor baselines as needed
3. If code inspection is useful, clone temporary repos into `agent/tmp/gemm_fp8_blockwise_triton_gfx942_20260412/related-work/repos/`
4. Write `agent/workspace/gemm_fp8_blockwise_triton_gfx942_20260412/related_work.md` using `related-work-template.md`

**Step 6: READ_HISTORICAL_TIPS**

Read `agent/historical_experience/gfx942/gemm_fp8_blockwise/triton/tips.md` if it exists, carrying forward only reusable lessons that still need to be validated in the current environment.

**Step 7: Enter optimization loop**

Read `workflow/optimize-loop.md` and execute as defined:
1. BASELINE (`round-1`): Run focused test (confirm PASS) → run focused benchmark → write `rounds/round-1/summary.md` → select representative shapes and update `quick_test_bench.py`
2. ANALYZE: Read `related_work.md`, kernel source, consult `triton/SKILL.md` and `hardware/gfx942/`, generate optimization hypotheses
3. OPTIMIZE → VALIDATE loop (`round-2+`): Modify kernel → run test → run benchmark → write `rounds/round-N/summary.md` → compare → accept or rollback → append a reusable tip if the round taught something worth preserving
4. After reaching target or exhausting directions, output REPORT

**Step 8: Acceptance** (return to project skill)

Project skill runs full tests, reviews report, confirms commits.

## Related Skills

| Skill | Description |
|-------|-------------|
| `workflow/optimize-loop` | Detailed optimization process, gating, rollback, stagnation handling |
| `triton` | General Triton optimization techniques |
| `hip` | General HIP/CK optimization techniques |
| `hardware/gfx942` | MI300X/MI325X hardware parameters and optimization strategies |
| `hardware/gfx950` | MI350X/MI355X hardware parameters and optimization strategies |
| `tool-rocprof` | rocprof profiling tool usage |
| `primus-turbo-develop` | Code structure, build, test, benchmark, and integration for the Primus-Turbo project |
