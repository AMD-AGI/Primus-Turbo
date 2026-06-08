---
name: kernel-optimize
description: AI-driven operator performance optimization framework. Defines the optimization loop, execution environment selection, knowledge routing, and logging conventions to drive agent-autonomous iteration toward hardware limits.
---

# Operator Performance Optimization Framework

This skill defines the **general-purpose operator optimization loop**, driving the agent to autonomously iterate toward hardware limits. It covers the prerequisite interface contract, campaign directory layout, the high-level loop, knowledge routing, and general principles for logging and stagnation handling.

Project-specific source paths and build/test/benchmark commands live in the corresponding project skill; runtime profiling lives in the profiling skill. Operator-, hardware-, and backend-level tuning knowledge lives in this skill's in-repo knowledge base under [`knowledge/`](knowledge/); route into it by `target_op` / `target_gpu` / `target_lang` during ANALYZE (see the Knowledge Reference Table). It currently covers GEMM and Attention (`knowledge/ops/`), gfx942 and gfx950 (`knowledge/hardware/`), and the FlyDSL backend (`knowledge/backend/`); for targets not yet covered there, draw on the agent's own knowledge.

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

Before entering the optimization loop, read [`../../rules/iteration_rules.mdc`](../../rules/iteration_rules.mdc). It is a hard constraint for every backend: one hypothesis per round, correctness before performance, benchmark the full active validation set, accept-or-rollback lineage, and **every accepted gain must transfer to a real LLM training step (Rule 11 — no benchmark idiom over-fitting via `id(...)`-keyed activation / grad_out / scale caches)**. The operational layer for Rule 11 lives in the "Avoiding Benchmark Over-Fitting" section below.

For validation scope, interpret that contract as:
- **full validation** → run all `target_shapes`
- **quick validation** → run all `representative_shapes`

Within a chosen validation level, the agent must not cherry-pick a smaller subset.

## Input Parameters

During the DEFINE_TARGET phase, the user instruction + prerequisite information must be organized into the following structured parameters:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `target_op` | Target operator | `gemm_fp8_blockwise` |
| `target_backend` | Target backend | `TRITON` / `CK` / `FLYDSL` |
| `target_lang` | Implementation language | `TRITON` / `HIP` / `FLYDSL` |
| `target_gpu` | Target GPU architecture | `gfx942` / `gfx950` |
| `target_shapes` | Full shape set of interest for the campaign; quick validation uses a separate representative subset recorded in `representative_shapes` | A full shape list or `all` (use benchmark default shape set) |
| `performance_target` | Performance target | `>500 TFLOPS`, `>60% peak efficiency`, or `null`; defaults to `null` if unspecified |
| `primary_metric` | Primary performance metric(s), depending on operator type | GEMM forward-only: `"Forward TFLOPS"`; GEMM forward+backward: `"Combined Step TFLOPS"` (or project-defined equivalent derived from forward/backward time), with forward/backward TFLOPS retained as diagnostics; elementwise: `"Forward GB/s"` |
| `project_skill` | Corresponding project skill | `primus-turbo-develop` |
| `execution_mode` | Execution environment, referencing project skill recommendation, decided by agent | `repo` / `workspace` |
| `git_commit` | Whether to git commit accepted versions | `true` (default) / `false` |
| `git_branch` | Optimization branch strategy | `auto` (default, auto-creates `optimize/<campaign>` branch) / `none` / `<custom branch name>` |
| `max_iterations` | Maximum iteration count (optional) | `10`; if unspecified, leave `null` and let termination conditions decide; if set, it must be `< 120` (a practical cost/benefit ceiling, not a hardware limit) |

## Overall Loop

```text
DEFINE_TARGET
  -> PREPARE_ENVIRONMENT
  -> READ_HISTORICAL_TIPS
  -> BASELINE
  -> [ANALYZE -> OPTIMIZE -> VALIDATE]  (iteration loop)
  -> REPORT
```

| Phase | What to do |
|-------|-----------|
| **DEFINE_TARGET** | Organize user instruction + project skill information into structured parameters, confirm completeness, **confirm target with user before starting** |
| **PREPARE_ENVIRONMENT** | Set up campaign directory, record metadata, and generate the quick validation script scaffold |
| **READ_HISTORICAL_TIPS** | If `agent/historical_experience/<target_gpu>/<target_op>/<target_backend_lower>/tips.md` exists, read it after PREPARE_ENVIRONMENT and before the first round |
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
| `target_lang` | Determined by `target_backend`: `TRITON` → Triton, `CK` / `HIPBLASLT` / `TURBO` → HIP, `FLYDSL` → FlyDSL |
| `target_gpu` | Identify GPU model from user instruction, map to architecture codename (e.g., MI300X → `gfx942`, MI355X → `gfx950`) |
| `target_shapes` | Use if specified by user; otherwise use the benchmark default shape set |
| `performance_target` | Use if specified by user; otherwise default to `null` |
| `primary_metric` | Get available metrics from the project skill's scoring section; use if specified by user. If the user wants both forward and backward GEMM optimized together and the project skill exposes both times, default to the project-defined combined-step metric instead of requiring forward/backward aggregates to improve independently |
| `execution_mode` | Reference the project skill's recommendation, decided by agent based on task characteristics |
| `git_commit` | Default `true`; set to `false` if user specifies no commit |
| `git_branch` | Default `auto`; use if specified by user |
| `max_iterations` | Use if specified by user and validate that it is `< 120`; otherwise leave empty, controlled by termination conditions |

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
- `max_iterations` (if applicable)
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
git_commit: <true | false>
git_branch: <branch name | none>
max_iterations: <integer < 120 | null>
created: <YYYY-MM-DD HH:MM>
```

All campaign timestamps must be recorded to minute precision in the format `YYYY-MM-DD HH:MM`.

All per-round artifacts live under `<campaign_dir>/rounds/`. `round-1` is the baseline round, and optimization attempts start at `round-2`. The running comparison table lives at `<campaign_dir>/logs/performance_trend.md`.

**Step 4: Generate `quick_test_bench.py`**

Use the template from the project skill's quick validation section to generate `<campaign_dir>/quick_test_bench.py` while the project API context is still fresh.

- Leave `SHAPES` empty or fill it with temporary placeholders during PREPARE_ENVIRONMENT
- After BASELINE, select `representative_shapes` and update both `quick_test_bench.py` and `manifest.yaml`
- Prefer a single self-contained script that runs correctness + benchmark together for quick iteration

## Avoiding Benchmark Over-Fitting

This section is the operational layer for [`../../rules/iteration_rules.mdc`](../../rules/iteration_rules.mdc) Rule 11. The rule defines the hard constraints — bucket classification (K1–K4, W1–W3), forbidden patterns, the W1 gain bound, the workload-distribution constraint for GroupGemm / MoE, the required `Real-training transfer check` summary section, and REPORT re-attribution. This section says **when** to apply them and gives the lightweight procedures the agent runs during each phase.

### When to invoke

- **ANALYZE**: tag every candidate direction with its Rule 11 bucket; drop `W2` / `W3`, only `K1`–`K4` and bounded `W1` directions may advance. Before proposing any wrapper-level change involving `dict`, `OrderedDict`, `weakref`, `lru_cache`, `id(...)`, or any "skip work if we have seen this tensor before" idea, run the id-audit from Rule 11. For GroupGemm work, also check whether the change assumes `M_per_group` is constant, divisible by a tile dim, or that every expert is non-empty, and confirm the `representative_shapes` selected at BASELINE include at least one skewed expert distribution.
- **VALIDATE**: whenever a round shows benchmark gain larger than what the kernel change can structurally produce, re-check the bucket and the 4-step trace before accepting. For GroupGemm rounds, re-run the validation on at least one skewed shape and confirm the gain holds.
- **REPORT**: while writing `## Final Report` in `logs/optimize.md`, re-attribute the campaign's `baseline → final best` delta into `S_real`, `W_real`, `R_real` per Rule 11.

### Pen-and-paper hit-rate trace

For any cache that survives the bucket / id-audit gates, trace its hit rate across a hypothetical 4-step real-training loop:

```
step 1: a1, w, grad_out_1   ->  fwd(a1, w),   bwd(grad_out_1)
step 2: a2, w, grad_out_2   ->  fwd(a2, w),   bwd(grad_out_2)
step 3: a3, w, grad_out_3   ->  fwd(a3, w),   bwd(grad_out_3)
step 4: a4, w, grad_out_4   ->  fwd(a4, w),   bwd(grad_out_4)
```

Expected hit rates:

- W1 weight cache keyed on `(id(w), w._version)`: 1 hit per step (backward reuses forward-time entry; `optim.step()` invalidates before next forward).
- W2 activation cache keyed on `(id(a), a._version)`: 0 hits (`a` differs across steps).
- W2 grad_out cache: 0 hits.

If the real-training hit rate falls below 50%, the round MUST be rolled back even if the benchmark accepted it.

### Checklist — VALIDATE

Before recording a round as ACCEPT:

- No new `dict` / `OrderedDict` / `lru_cache` keyed on `id(t)` for any non-weight `t`.
- No new code path uses `weakref.ref(activation)` or `weakref.ref(grad_out)`.
- If the round adds a weight cache: the `_version` invalidation is tested (mutate a clone of the weight in-place, expect cache miss).
- For GroupGemm rounds: no `tl.constexpr` / compile-time constant `M_per_group`, no static assumption that every expert is non-empty, and the benchmark gain reproduces on at least one skewed-distribution shape.
- The summary's `Real-training transfer check` section is filled in and explicitly states the estimated real-training gain, plus (for GroupGemm) the skew robustness line.
- Any reusable tip appended does not advertise an `id(activation)` / `id(grad_out)` pattern or a "fixed `M_per_group`" GroupGemm shortcut as a useful technique.

### Checklist — REPORT

Before publishing `## Final Report` in `logs/optimize.md`:

- Every accepted round has a `Real-training transfer check` section in its `summary.md` with a concrete `Decision`.
- No accepted round has `Decision: REJECT-as-overfit`; if one does, roll it back first, then re-run the audit on the remaining lineage.
- The `Real-training applicability audit` table is filled per Rule 11's re-attribution procedure (`S_real`, `W_real`, `R_real`), not copied from the headline benchmark number.
- The inflation gap (`headline benchmark gain` minus `real-training-equivalent gain`) is reported. If the gap exceeds 1%, the report attributes it to benchmark-loop residual and recommends rollback rather than shipping the inflated headline.

## READ_HISTORICAL_TIPS

After `PREPARE_ENVIRONMENT` finishes and before `round-1` starts, check whether a reusable tips file already exists for the same hardware / op / backend combination.

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

When the agent selects `workspace-mode`, extend the campaign directory from PREPARE_ENVIRONMENT with three additional subdirectories for the minimal local stack:

- `src/` — minimal kernel implementation extracted from upstream
- `tests/` — targeted correctness tests equivalent to their upstream counterparts
- `bench/` — targeted benchmarks equivalent to their upstream counterparts

Setup principles:
- **Minimal**: extract only the target kernel and its direct dependencies, not the entire project.
- **Reproducible**: record which upstream commit and files were extracted.
- **Faithful**: local tests/benchmarks must match their upstream counterparts so results are trustworthy.
- **Clear integration path**: after targets are met, sync only core kernel changes back upstream — never the scaffolding.

The project skill provides how-to for extraction and minimal-stack construction. Regardless of mode, all optimization artifacts (logs, profiles, benchmark results) live in the campaign directory.

## Knowledge Reference Table

Read the corresponding skill as needed based on `target_lang`, `target_gpu`, and the current phase. **Do not read everything at once.**

| What you need | Where to find it |
|---------------|-----------------|
| Linear iteration contract + no benchmark over-fitting (hard rules) | [../../rules/iteration_rules.mdc](../../rules/iteration_rules.mdc) |
| Optimization process and gating rules | [workflow/optimize-loop.md](workflow/optimize-loop.md) |
| Operator-level anatomy / bottleneck model / optimization directions | [knowledge/ops/](knowledge/ops/) — `gemm/`, `attention/` (read each area's `overview.md` first, then `optimization-directions.md`); for ops not yet covered, use the agent's own knowledge |
| Backend / language-level tuning expression (Triton / HIP / CK / FlyDSL) | [knowledge/backend/](knowledge/backend/) — currently `flydsl/` only (`overview.md` → `programming-model.md` → `optimization-directions.md`); for Triton / HIP / CK draw on the agent's own knowledge until in-repo docs land |
| Hardware budgets / occupancy / ISA notes / cross-generation diff | [knowledge/hardware/](knowledge/hardware/) — top-level `overview.md` for the cross-generation diff + porting, then per-generation `gfx942/`, `gfx950/` (`overview.md` → `kernel-implementation-notes.md`) |
| Profiling methods | Via the project skill, e.g. [../primus-turbo-develop/run_profile/tool-rocprof/SKILL.md](../primus-turbo-develop/run_profile/tool-rocprof/SKILL.md) |
| Project code structure / build / test / benchmark / integration | Corresponding project skill, e.g., [../primus-turbo-develop/SKILL.md](../primus-turbo-develop/SKILL.md) |
| Historical reusable tips | `agent/historical_experience/<target_gpu>/<target_op>/<target_backend_lower>/tips.md` (if present) |

## Logging and Lineage General Principles

The optimization process must maintain a structured, lineage-style history to support long-term iteration.

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

Stagnation is a continuous-evolution trigger: it is not a stop signal, but a prompt for strategy switching.

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

1. **Understand requirements** — from the project skill, learn the framework needs: kernel source, focused test/benchmark commands, quick validation template, benchmark output format, scoring rules, execution mode, rebuild requirements.
2. **Collect project information** (illustrative paths/commands below; the project skill provides the real ones) — kernel `primus_turbo/triton/gemm/gemm_fp8_kernel.py`; focused test `pytest tests/pytorch/ops/test_gemm_fp8.py -v -k "blockwise and TRITON"`; focused benchmark `PRIMUS_TURBO_GEMM_BACKEND=TRITON python benchmark/ops/bench_gemm_turbo.py --dtype fp8 --granularity blockwise`; scoring is `Forward TFLOPS` geomean with `Check` as the correctness gate; Triton → `repo-mode`, no rebuild.
3. **DEFINE_TARGET** — `target_op=gemm_fp8_blockwise`, `target_backend=TRITON`, `target_gpu=gfx942`, `execution_mode=repo`, `performance_target=null`. Confirm key parameters with the user before starting.
4. **PREPARE_ENVIRONMENT** — `git checkout -b optimize/gemm_fp8_blockwise_triton_gfx942_<date>`; create `agent/workspace/<campaign_name>/` with `logs/`, `profiles/`, `rounds/round-1/`; write `manifest.yaml`; generate `quick_test_bench.py` with placeholder `SHAPES`.
5. **READ_HISTORICAL_TIPS** — read `agent/historical_experience/gfx942/gemm_fp8_blockwise/triton/tips.md` if present.
6. **Optimization loop** — follow `workflow/optimize-loop.md`: BASELINE (`round-1`, full validation, select representative shapes, fill `quick_test_bench.py`), then ANALYZE → OPTIMIZE → VALIDATE rounds, each with `summary.md`, kernel snapshot, and trend update.
7. **Acceptance** — hand back to the project skill for full tests, report review, and commit confirmation.
