# Primus Optimizer — Design Specification

> Claude Code plugin for automated multi-agent operator optimization in Primus-Turbo.

**Date**: 2026-04-07
**Status**: Draft
**Approach**: Claude Code Native Multi-Agent (MVP), evolving toward Python orchestrator + Claude Code workers

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Task Configuration](#3-task-configuration)
4. [Worker Agent](#4-worker-agent)
5. [Coordinator](#5-coordinator)
6. [Review Agent](#6-review-agent)
7. [Bottleneck Escalation](#7-bottleneck-escalation)
8. [Knowledge Agent](#8-knowledge-agent)
9. [Terminal Dashboard](#9-terminal-dashboard)
10. [State Storage](#10-state-storage)
11. [Final Output](#11-final-output)
12. [User Guide](#12-user-guide)

---

## 1. Overview

### Problem

Primus-Turbo contains 6 operator families (GEMM, Attention, Grouped GEMM, MoE, Quantization, Normalization) across 3 backends (CK, HipBLASLt, Triton) targeting 2 hardware platforms (MI300X, MI355X). Manually optimizing each combination is slow and doesn't scale.

### Solution

A Claude Code skill plugin (`primus-optimizer`) that orchestrates multiple parallel optimization agents — each responsible for a single operator x backend combination — with automated profiling, bottleneck detection, knowledge mining, and cross-pollination review.

### Prior Art

- **Internal**: `.cursor/` rules and skills already implement a single-agent serial optimization loop (Profile -> Analyze -> Plan -> Implement -> Verify -> Decide -> Log) with 11 successful rounds.
- **External**: NVIDIA's AVO (arXiv:2603.24517) demonstrates 7-day autonomous agent-driven kernel evolution on Blackwell GPUs, achieving 3.5-10.5% over cuDNN/FA4.

### Design Principles

- **Atomicity**: One optimization point per round, enabling clear attribution and rollback.
- **Verifiability**: Every claim backed by reproducible commands and data.
- **Traceability**: Complete optimization history in structured files.
- **Parallelism**: Operator x backend level concurrency, GPU-isolated.
- **Evolutionary**: Terminal dashboard MVP, data format designed for future web UI.

---

## 2. Architecture

### Component Layout

```
primus-optimizer/
├── skills/
│   ├── optimize/                    # Main entry skill (/optimize)
│   │   └── SKILL.md
│   ├── optimize-worker/             # Worker agent skill
│   │   └── SKILL.md
│   ├── optimize-review/             # Review agent skill
│   │   └── SKILL.md
│   └── optimize-knowledge/          # Knowledge mining skill
│       └── SKILL.md
│
├── tools/
│   ├── gpu_pool.py                  # GPU resource pool management
│   ├── benchmark_runner.py          # Standardized benchmark executor
│   ├── profiler.py                  # Fine-grained profiling wrapper
│   ├── bottleneck_detector.py       # Bottleneck detection (multi-dimensional)
│   ├── state_store.py               # Optimization state persistence (JSON)
│   └── dashboard.py                 # Terminal dashboard (rich)
│
├── hooks/
│   ├── pre-benchmark.sh             # Environment check before benchmark
│   └── post-round.sh               # State update after each round
│
├── templates/
│   ├── round-report.md              # Per-round report template
│   ├── pr-description.md            # PR description template
│   └── final-summary.md             # Final summary template
│
└── config/
    └── optimizer.yaml               # Global config (targets, GPU mapping, profiles)
```

### Data Flow

```
User: /optimize --tasks gemm:triton,attn:ck --hw mi355x

                    ┌─────────────────────────────────────────────┐
                    │            Coordinator (optimize skill)      │
                    │  1. Parse config -> task matrix              │
                    │  2. Check GPU pool -> allocate GPUs          │
                    │  3. Dispatch Worker Agents (background,      │
                    │     worktree-isolated)                       │
                    │  4. Monitor loop -> update Dashboard         │
                    │  5. On bottleneck/completion -> Review Agent │
                    │  6. Finalize -> merge plan + PR descriptions │
                    └──────────────┬───────────────────────────────┘
               ┌───────────────────┼───────────────────┐
               ▼                   ▼                   ▼
        ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
        │ Worker Agent │     │ Worker Agent │     │ Worker Agent │
        │ gemm:triton  │     │ gemm:ck      │     │ attn:triton  │
        │ GPU:0        │     │ GPU:1        │     │ GPU:2        │
        │ worktree:A   │     │ worktree:B   │     │ worktree:C   │
        └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
               │                   │                   │
               ▼                   ▼                   ▼
        Profile -> Analyze -> Implement -> Verify -> Log (loop)
```

---

## 3. Task Configuration

### CLI Interface

Three ways to specify optimization targets:

**Explicit task list (preferred)**:
```bash
/optimize --tasks gemm:ck,attention:triton --hw mi355x
```

**Cartesian product expansion**:
```bash
/optimize --operators gemm,attention --backends triton,ck --hw mi355x
# Expands to: gemm:triton, gemm:ck, attention:triton, attention:ck
```

**Mixed**:
```bash
/optimize --tasks gemm:ck,gemm:hipblaslt --operators attention --backends triton --hw mi355x
# Result: gemm:ck, gemm:hipblaslt, attention:triton
```

**Named profile**:
```bash
/optimize --profile mi355x-priority
```

**Resume interrupted session**:
```bash
/optimize --resume                          # Resume last session
/optimize --resume --session 2026-04-07-001 # Resume specific session
```
Coordinator reads `optimizer-state.json`, detects each Worker's last committed round, and restarts from there.

### Configuration File (`config/optimizer.yaml`)

```yaml
# GPU resource configuration
gpu_pool:
  device_type: hip              # hip or cuda
  device_ids: [0, 1, 2, 3]     # Available GPU IDs

# Global defaults
defaults:
  max_rounds: 10                # Max optimization rounds per worker
  bottleneck_threshold: 0.02    # 2% — consecutive gain below this triggers bottleneck
  bottleneck_patience: 3        # Number of consecutive low-gain rounds before bottleneck
  accuracy_snr_bf16: 30         # dB threshold for BF16
  accuracy_snr_fp8: 20          # dB threshold for FP8
  monitor_interval: 60          # Seconds between Dashboard refreshes

# Named profiles
profiles:
  mi355x-priority:
    hw: mi355x
    tasks:
      - operator: gemm
        backend: ck
        max_rounds: 10
      - operator: attention
        backend: triton
        max_rounds: 8

  mi300x-full-sweep:
    hw: mi300x
    operators: [gemm, grouped_gemm, attention]
    backends: [triton, ck, hipblaslt]
    max_rounds: 5

  blockwise-fp8:
    hw: mi355x
    tasks:
      - operator: gemm
        backend: triton
        dtype: fp8
        quant: blockwise
        max_rounds: 12

# Knowledge agent search configuration
knowledge:
  search_templates:
    gemm:
      queries:
        - "{backend} GEMM optimization AMD {arch} site:github.com"
        - "matrix multiplication kernel optimization rocm 2026"
        - "GEMM persistent kernel split-k AMD"
      repos_to_check:
        - ROCm/composable_kernel
        - ROCm/aiter
        - ROCm/hipBLASLt
        - triton-lang/triton

    attention:
      queries:
        - "flash attention optimization AMD MI300 MI350"
        - "triton attention kernel CDNA"
      repos_to_check:
        - Dao-AILab/flash-attention
        - ROCm/aiter
        - linkedin/Liger-Kernel

    grouped_gemm:
      queries:
        - "grouped GEMM MoE optimization AMD"
        - "batched matrix multiplication variable sizes GPU"
      repos_to_check:
        - ROCm/composable_kernel
        - vllm-project/vllm
```

---

## 4. Worker Agent

### Optimization Loop State Machine

```
                    ┌──────────┐
                    │  INIT    │ Load skill knowledge base + establish baseline
                    └────┬─────┘
                         ▼
              ┌─────────────────────┐
              │     PROFILE         │ Run benchmark -> collect performance data
              └─────────┬───────────┘
                        ▼
              ┌─────────────────────┐
              │     ANALYZE         │ Identify bottleneck type (compute/memory/latency)
              │                     │ Compare against roofline / SOTA
              └─────────┬───────────┘
                        ▼
              ┌─────────────────────┐
              │     PLAN            │ Select strategy from skill knowledge base
              │                     │ Design single atomic modification
              └─────────┬───────────┘
                        ▼
              ┌─────────────────────┐
              │     IMPLEMENT       │ Modify code (single atomic change)
              │                     │ git commit on worktree branch
              └─────────┬───────────┘
                        ▼
              ┌─────────────────────┐
              │     VERIFY          │ Level 1: Quick accuracy check
              │                     │ Level 2: Full benchmark
              │                     │ Level 3: Regression detection
              └─────────┬───────────┘
                   ┌────┴────┐
                   ▼         ▼
              [PASS]      [FAIL] --> git revert -> back to PLAN (switch strategy)
                   │                  3 consecutive fails -> report to Coordinator
                   ▼
              ┌─────────────────────┐
              │     DECIDE          │ Record gain, decide whether to continue
              └─────────┬───────────┘
                   ┌────┴──────────┐
                   ▼               ▼
           [Continue]         [Bottleneck]
           back to PROFILE    Notify Coordinator
```

### Skill Loading

Each Worker loads operator-specific knowledge on startup:

```
Worker(gemm:triton) loads:
  ├── skills/triton-gemm-optimize/SKILL.md      # Optimization playbook
  ├── skills/amd-gpu-architecture/SKILL.md       # Hardware constraints
  ├── skills/kernel-profiling/SKILL.md           # Profiling methodology
  └── skills/sota-knowledge-base/SKILL.md        # SOTA references

Worker(attention:triton) loads:
  ├── skills/triton-attention-optimize/SKILL.md
  ├── skills/amd-gpu-architecture/SKILL.md
  └── ...
```

### Worker Prompt Structure

```markdown
## Your Task
You are a Primus-Turbo optimization loop Worker Agent.
- Target operator: {operator} ({backend} backend)
- Target hardware: {hw}
- Assigned GPU: HIP_VISIBLE_DEVICES={gpu_id}
- Work branch: {worktree_branch}

## Constraints
- One atomic modification per round
- Must pass accuracy check before and after modification
- Max rounds: {max_rounds}
- Report bottleneck after {bottleneck_patience} consecutive rounds with gain < {bottleneck_threshold}

## Current State
- Completed rounds: {completed_rounds}
- Current baseline: {current_tflops} TFLOPS
- Tried strategies: {tried_strategies}

## Knowledge Base
{loaded_skill_content}

## Output Requirements
After each round, write to:
  agent_docs/{hw}/{operator}-{backend}/round-{N}/
    ├── report.md        # Round report (see template below)
    ├── baseline.json    # Pre-modification performance
    ├── optimized.json   # Post-modification performance
    └── accuracy.log     # Accuracy verification log
```

### GPU Isolation

Each Worker receives an exclusive GPU via environment variable:

```python
# tools/gpu_pool.py
class GPUPool:
    def __init__(self, gpu_ids: list[int]):
        self.available = set(gpu_ids)
        self.allocated = {}  # task_id -> gpu_id

    def acquire(self, task_id: str) -> int:
        gpu_id = self.available.pop()
        self.allocated[task_id] = gpu_id
        return gpu_id

    def release(self, task_id: str):
        gpu_id = self.allocated.pop(task_id)
        self.available.add(gpu_id)
```

Worker benchmark execution:
```bash
HIP_VISIBLE_DEVICES={gpu_id} python benchmark/ops/bench_gemm_turbo.py ...
```

### Termination Conditions

| Condition | Behavior |
|-----------|----------|
| Reached `max_rounds` | Normal completion, report final results |
| 3 consecutive rounds with gain <2% | Report `bottleneck`, await Coordinator decision |
| 3 consecutive VERIFY failures | Report `stuck`, Coordinator intervenes |
| Roofline utilization >80% | Report `near_optimal`, recommend termination |
| Gap to SOTA <5% | Report `competitive`, recommend termination |

### Round Report Template (`round-{N}/report.md`)

```markdown
# Round {N}: {Optimization Strategy Name}

## Optimization Rationale

### Problem Analysis
{Specific bottleneck identified from profiling data, e.g.:
"The current persistent kernel with BLOCK_M=128 only assigns 1-2 tiles per CU
in decode scenarios where M<512, leaving >60% of 304 CUs idle,
resulting in compute utilization of only 9.8%"}

### Optimization Strategy
{Technical principle of the strategy, e.g.:
"Reduce BLOCK_M from 128 to 64, doubling tile count. For M=256, N=7168,
tiles increase from 2x56=112 to 4x56=224, CU utilization from 36.8% to 73.7%.
The tradeoff is lower per-tile compute density, but for small M the CU
utilization gain far outweighs this cost."}

### Theoretical Expectation
{Expected improvement and basis, e.g.: "Expected +30-50% based on linear
extrapolation of CU utilization increase from 36.8% to 73.7%"}

## Code Changes

### Modified Files
- `primus_turbo/triton/gemm/gemm_kernel.py:L142-L158`

### Diff Summary
{Key diff, not exhaustive}

### Commit
- Branch: `opt/{operator}-{backend}-{session_id}`
- Commit: `{hash}` - `{message}`

## Reproduction Commands

### Accuracy Verification
```bash
# Generates accuracy.log
HIP_VISIBLE_DEVICES={gpu_id} python benchmark/accuracy/eval_gemm_accuracy.py \
    --dtype {dtype} --quant {quant} \
    --shapes "{shape_list}" \
    --backend {backend} \
    2>&1 | tee agent_docs/{hw}/{operator}-{backend}/round-{N}/accuracy.log
```

### Baseline Performance
```bash
# Generates baseline.json (run from pre-modification state)
git stash  # or git checkout HEAD~1
HIP_VISIBLE_DEVICES={gpu_id} python benchmark/ops/bench_{operator}_turbo.py \
    --dtype {dtype} --quant {quant} \
    --shapes "{shape_list}" \
    --backend {backend} --warmup 10 --rep 50 \
    --output agent_docs/{hw}/{operator}-{backend}/round-{N}/baseline.json
git stash pop  # or git checkout {worktree_branch}
```

### Optimized Performance
```bash
# Generates optimized.json (run from post-modification state)
HIP_VISIBLE_DEVICES={gpu_id} python benchmark/ops/bench_{operator}_turbo.py \
    --dtype {dtype} --quant {quant} \
    --shapes "{shape_list}" \
    --backend {backend} --warmup 10 --rep 50 \
    --output agent_docs/{hw}/{operator}-{backend}/round-{N}/optimized.json
```

## Results

| Shape (M,N,K) | Baseline (TFLOPS) | Optimized (TFLOPS) | Change |
|---|---|---|---|
| 256,7168,7168 | 488.1 | 793.2 | +62.5% |
| ... | ... | ... | ... |

- **Geomean improvement**: +X.X%
- **Peak improvement**: +X.X% @ shape (M,N,K)
- **Regression detection**: No regressions / {shape} regressed X%

## Conclusion
{Success/failure, whether to adopt, next direction}
```

---

## 5. Coordinator

### Lifecycle

```
INIT --> DISPATCH --> MONITOR --> REVIEW --> FINALIZE

INIT:     Load optimizer.yaml / CLI args
DISPATCH: Parse config -> task matrix, allocate GPUs, launch Worker Agents
MONITOR:  Poll Worker states, update Dashboard, manage GPU queue
REVIEW:   Trigger Review Agent on milestones / bottlenecks / completion
FINALIZE: Merge results, generate PR descriptions, final summary
```

### Monitor Loop

Every `monitor_interval` seconds (default 60):

```
1. Read optimizer-state.json
2. Check each Worker status:
   - running     -> Update Dashboard display
   - bottleneck  -> Trigger bottleneck escalation flow
   - stuck       -> Trigger Review Agent intervention
   - completed   -> Reclaim GPU, check if queued tasks remain
3. If Worker completed and tasks queued -> allocate freed GPU, start new Worker
4. All Workers completed -> enter FINALIZE
```

### GPU Queue Management

When task count > GPU count, Coordinator maintains a FIFO queue:

```
GPU Pool: [0, 1, 2, 3]
Task Queue: [gemm:triton, gemm:ck, attn:triton, attn:ck, grouped_gemm:triton]

Phase 1: First 4 tasks each occupy 1 GPU, run in parallel
Phase 2: gemm:triton finishes first -> GPU:0 freed -> grouped_gemm:triton starts
```

---

## 6. Review Agent

### Trigger Conditions

| Trigger | Review Scope |
|---------|-------------|
| Any Worker completes 5 rounds | Milestone review: strategy direction, missed opportunities |
| Worker reports `bottleneck` | Bottleneck diagnosis + decision (see Section 7) |
| Worker reports `stuck` | Failure analysis, suggest new strategy or terminate |
| All Workers completed | Final summary: cross-pollination + merge report |

### Review Agent Prompt Structure

```markdown
## Your Role
You are the Primus-Turbo optimization loop Review Agent. You are responsible for:
1. Reviewing Worker optimization history, judging strategy quality
2. Identifying cross-pollination opportunities (Worker A's successful strategy
   may transfer to Worker B)
3. Making action decisions during bottleneck diagnosis

## All Worker States
{Loaded from optimizer-state.json}

## Worker Under Review
{Complete round history of the Worker that triggered this review}

## Cross-Pollination Checklist
- Worker A used strategy X successfully -> Can Worker B also apply it?
  (e.g., gemm:triton's persistent kernel -> applicable to grouped_gemm:triton?)
- Performance gap between backends for the same operator -> Does this suggest
  certain backends are better suited for specific shape ranges?
  (e.g., gemm:ck outperforms at M<512 -> should backend dispatch rules be updated?)

## Output Requirements
1. Assessment of each Worker (strategy direction, efficiency, missed opportunities)
2. Cross-pollination suggestions (if any)
3. Decision recommendation: continue / switch strategy / terminate
```

---

## 7. Bottleneck Escalation

When a Worker reports `bottleneck` (3 consecutive rounds with <2% gain), the system triggers a three-level escalation:

### Level 1: Fine-Grained Profiling

Coordinator instructs the Worker to run deep profiling:

```
1. rocprof --stats: kernel-level timing
2. omniperf: hardware counter collection
   - Compute utilization
   - Memory bandwidth utilization
   - LDS bank conflicts
   - MFMA pipe utilization
3. Roofline analysis generation
```

Profiling results are injected into Worker context. Worker re-enters ANALYZE -> PLAN with new data.

### Level 2: Knowledge Mining

If Level 1 fails to break through (bottleneck triggered again), Coordinator launches a Knowledge Agent:

```
1. Web Search for latest optimization techniques:
   - "{operator} optimization {hw_arch} 2025 2026"
   - "triton kernel optimization AMD CDNA"
   - GitHub: recent commits in related repos
2. Fetch and analyze:
   - AITER / vLLM / FlashAttention latest changelog
   - AMD ROCm documentation updates
   - Related papers (arXiv)
3. Distill actionable strategy list, inject into Worker context
```

Knowledge Agent output:
```
knowledge_docs/{operator}-{backend}/
  ├── web_findings.md     # Search discoveries
  ├── new_strategies.md   # Distilled new strategies
  └── references.json     # Source URL list
```

### Level 3: Review Agent Final Verdict

If Level 2 fails to break through (third bottleneck), Review Agent makes a comprehensive judgment:

1. Is the kernel near hardware theoretical limit? (roofline >80%)
2. Is the gap to SOTA <5%?
3. Are there cross-Worker cross-pollination opportunities?

Decisions:
- **TERMINATE**: Current operator x backend is near limit, stop.
- **PIVOT**: Switch to a fundamentally different strategy direction.
- **ESCALATE**: Mark as requiring human intervention.

---

## 8. Knowledge Agent

### Search Strategy Configuration

Defined in `config/optimizer.yaml` under the `knowledge` section. Per-operator search templates include:

- **Keyword queries** for web search (operator + backend + architecture specific)
- **GitHub repos to check** for recent commits and releases
- **arXiv search** for relevant papers

### Output Structure

```markdown
# Knowledge Mining Report: {operator}:{backend}

## Search Queries Executed
1. "{query_1}" -> {N} results
2. "{query_2}" -> {N} results

## Key Findings

### Finding 1: {Title}
- **Source**: {URL}
- **Relevance**: {Why this matters for our optimization}
- **Applicable Strategy**: {Concrete optimization idea derived from this finding}

### Finding 2: ...

## Recommended New Strategies (Priority Ordered)
1. {Strategy}: {Description} — Expected impact: {estimate}
2. ...

## References
- [{title}]({url}) — {one-line relevance note}
```

---

## 9. Terminal Dashboard

### Layout

```
┌─ Primus Optimizer ──────────────────────────────────────────────────────────┐
│ Session: 2026-04-07-001 │ HW: MI355X │ Elapsed: 02:34:17 │ GPUs: 4/4 busy │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Worker Status                                                              │
│  ┌──────────────────┬────────┬─────┬──────────┬───────────┬────────────┐   │
│  │ Task             │ Status │ GPU │ Round    │ TFLOPS    │ Gain       │   │
│  ├──────────────────┼────────┼─────┼──────────┼───────────┼────────────┤   │
│  │ gemm:triton      │ ▶ RUN  │  0  │ 5/10     │ 488→793   │ +62.5%     │   │
│  │ gemm:ck          │ ▶ RUN  │  1  │ 3/10     │ 429→512   │ +19.3%     │   │
│  │ attn:triton      │ ⚠ BTNK │  2  │ 7/8  L1  │ 301→452   │ +50.2%     │   │
│  │ grouped_gemm:ck  │ ◻ WAIT │  -  │ -        │ -         │ -          │   │
│  └──────────────────┴────────┴─────┴──────────┴───────────┴────────────┘   │
│                                                                             │
│  Live Activity                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ 12:34:17 [gemm:triton  ] Round 5 VERIFY passed (+8.3%)              │  │
│  │ 12:33:42 [attn:triton  ] BOTTLENECK L1: running omniperf...         │  │
│  │ 12:32:08 [gemm:ck      ] Round 3 IMPLEMENT: M-aware tile select    │  │
│  │ 12:31:55 [gemm:triton  ] Round 5 IMPLEMENT: BLOCK_K=128->256       │  │
│  │ 12:30:01 [attn:triton  ] Round 7 gain +0.8% < 2%, bottleneck hit   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  Performance Trend (gemm:triton)                                            │
│  TFLOPS                                                                     │
│  800 ┤                                                          ╭──● R5    │
│  700 ┤                                              ╭───────────╯          │
│  600 ┤                              ╭───────────────╯                      │
│  500 ┤              ╭───────────────╯                                      │
│  488 ┤──────●───────╯                                                      │
│      └──────┬───────┬───────────────┬───────────────┬───────────┬───       │
│            R0      R1              R2              R3           R4          │
│                                                                             │
│  [q] Quit  [r] Refresh  [w] Switch Worker  [p] Pause Worker  [d] Detail   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Data Source | Refresh Rate |
|-----------|-----------|-------------|
| Header | `optimizer-state.json` session info | Every refresh |
| Worker Status table | `optimizer-state.json` workers field | 60s |
| Live Activity | Worker `activity.jsonl` files (aggregated) | 10s |
| Performance Trend | Round `optimized.json` files | On round completion |

### Status Icons

```
▶ RUN   — Running normally
⚠ BTNK  — Bottleneck handling (L1/L2/L3 annotated)
✓ DONE  — Optimization completed
✗ FAIL  — Failed / needs human intervention
◻ WAIT  — Queued, waiting for GPU
⏸ PAUSE — Paused by user
```

### Activity Log Format

Each Worker appends to its own `activity.jsonl`:

```jsonl
{"ts":"2026-04-07T12:34:17Z","worker":"gemm:triton","phase":"VERIFY","round":5,"msg":"accuracy check passed, 3344/3344 tests OK"}
{"ts":"2026-04-07T12:34:18Z","worker":"gemm:triton","phase":"VERIFY","round":5,"msg":"benchmark complete: 793.2 TFLOPS (+8.3%)"}
{"ts":"2026-04-07T12:34:19Z","worker":"gemm:triton","phase":"DECIDE","round":5,"msg":"improvement +8.3% > 2%, continuing"}
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Graceful exit (wait for current round to finish, then stop all Workers) |
| `Ctrl+C` | Immediate interrupt (Worker worktrees preserved, session resumable) |
| `w` | Switch Performance Trend chart to show a different Worker |
| `d` | Expand selected Worker's detailed round history |
| `p` | Pause / resume selected Worker |

### Future Web UI Upgrade Path

All data flows through JSON files — Dashboard is read-only. Adding a Web UI later requires only a thin API layer:

- `optimizer-state.json` -> REST `GET /api/state`
- `activity.jsonl` -> WebSocket push
- Round JSON files -> `GET /api/workers/{id}/rounds/{n}`

No changes needed to Worker or Coordinator logic.

---

## 10. State Storage

### `optimizer-state.json`

Central state file at `agent_docs/{hw}/optimizer-state.json`:

```json
{
  "session_id": "2026-04-07-001",
  "config": {
    "hw": "mi355x",
    "tasks": [
      {"operator": "gemm", "backend": "triton", "max_rounds": 10},
      {"operator": "gemm", "backend": "ck", "max_rounds": 10},
      {"operator": "attention", "backend": "triton", "max_rounds": 8}
    ]
  },
  "gpu_pool": {
    "total": [0, 1, 2, 3],
    "allocated": {"gemm:triton": 0, "gemm:ck": 1, "attn:triton": 2}
  },
  "workers": {
    "gemm:triton": {
      "status": "running",
      "gpu_id": 0,
      "worktree_branch": "opt/gemm-triton-001",
      "current_round": 5,
      "rounds": [
        {
          "round": 1,
          "strategy": "persistent_kernel",
          "baseline_tflops": 488.1,
          "result_tflops": 614.2,
          "improvement_pct": 25.8,
          "status": "success"
        }
      ],
      "bottleneck": null,
      "started_at": "2026-04-07T10:00:00Z"
    }
  },
  "task_queue": ["grouped_gemm:ck"],
  "review_log": [
    {
      "trigger": "milestone",
      "worker": "gemm:triton",
      "round": 5,
      "decision": "continue",
      "cross_pollination": [],
      "timestamp": "2026-04-07T12:00:00Z"
    }
  ],
  "started_at": "2026-04-07T10:00:00Z",
  "updated_at": "2026-04-07T12:30:00Z"
}
```

### Round Output Directory

```
agent_docs/{hw}/{operator}-{backend}/
├── round-0/                   # Baseline round
│   ├── report.md
│   ├── baseline.json
│   └── accuracy.log
├── round-1/
│   ├── report.md
│   ├── baseline.json          # Pre-modification (= round-0 optimized)
│   ├── optimized.json         # Post-modification
│   └── accuracy.log
├── round-N/
│   └── ...
└── activity.jsonl             # Worker activity log (append-only)
```

---

## 11. Final Output

### Directory Structure

```
agent_docs/{hw}/session-{id}/
├── final-summary.md           # Global summary report
├── optimizer-state.json       # Complete state snapshot
├── merge-plan.md              # Worktree merge plan
└── pr-descriptions/           # PR description per recommended merge
    ├── gemm-triton.md
    ├── gemm-ck.md
    └── attention-triton.md
```

### PR Description Template (`pr-descriptions/{operator}-{backend}.md`)

```markdown
## Summary

Optimize {operator} ({backend} backend) on {hw} through {N} rounds
of iterative optimization, achieving **+{geomean}% geomean improvement**
({baseline_tflops} -> {optimized_tflops} TFLOPS).

### Key Changes

- **Round 1 -- {strategy_1}**: {one-line description} (+{pct_1}%)
- **Round 3 -- {strategy_3}**: {one-line description} (+{pct_3}%)
- ...

### Optimization Rationale

{2-3 paragraph technical explanation covering:
  - What bottleneck was identified (e.g., CU underutilization at small M)
  - What approach was taken and why (e.g., tile size downgrade trades
    per-tile compute density for higher CU occupancy)
  - Why this is safe (accuracy verification results)}

## Performance Results

| Shape (M,N,K) | Baseline (TFLOPS) | Optimized (TFLOPS) | Change |
|---|---|---|---|
| 256,7168,7168 | 488.1 | 793.2 | +62.5% |
| ... | ... | ... | ... |

**Geomean**: +{X.X}% | **Peak**: +{X.X}% @ {shape} | **Regressions**: None

## Accuracy Verification

- {dtype} accuracy: all {N} test cases passed (SNR > {threshold} dB)
- Reproduce: `{accuracy_command}`

## Reproduce Benchmarks

```bash
# Baseline
{baseline_command}

# Optimized
{optimized_command}
```

## Test Plan

- [ ] Accuracy check: `{accuracy_command}`
- [ ] Performance benchmark: `{benchmark_command}`
- [ ] Regression check: `python tools/check_regression.py --baseline {path}`
```

### Merge Plan (`merge-plan.md`)

```markdown
# Merge Plan — Session {session_id}

## Recommended Merges

| Branch | Operator:Backend | Geomean Gain | Regressions | PR Description |
|--------|-----------------|-------------|-------------|----------------|
| opt/gemm-triton-001 | gemm:triton | +62.5% | None | pr-descriptions/gemm-triton.md |
| opt/gemm-ck-001 | gemm:ck | +19.3% | None | pr-descriptions/gemm-ck.md |

## Merge Order
1. `opt/gemm-triton-001` — no conflicts expected
2. `opt/gemm-ck-001` — may conflict with gemm-triton in `gemm_impl.py`, manual review needed

## Branches NOT Recommended for Merge
| Branch | Reason |
|--------|--------|
| opt/attn-triton-001 | Regression on seq_len=8192 (-3.2%), needs investigation |

## Cleanup
After merging, remove worktrees:
```bash
git worktree remove .claude/worktrees/opt-gemm-triton-001
git worktree remove .claude/worktrees/opt-gemm-ck-001
```
```

---

## 12. User Guide

### Prerequisites

- Claude Code CLI installed and authenticated
- Primus-Turbo repository cloned with GPU access
- Python environment with `rich` installed (`pip install rich`)
- AMD GPU with ROCm and HIP toolkit
- `rocprof` and `omniperf` available for profiling (optional, for bottleneck L1)

### Installation

```bash
# From Primus-Turbo repo root
cd primus-optimizer

# Install the plugin as Claude Code skills
# (Copy skills to Claude Code skill directory or configure in settings.json)
cp -r skills/* ~/.claude/skills/

# Install Python dependencies for tools
pip install rich pyyaml

# Verify GPU availability
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### Quick Start

**1. Run a single operator optimization:**
```bash
# In Claude Code, optimize GEMM with Triton backend on MI355X
/optimize --tasks gemm:triton --hw mi355x
```

**2. Run parallel optimization on multiple targets:**
```bash
# Optimize GEMM (CK) and Attention (Triton) in parallel
/optimize --tasks gemm:ck,attention:triton --hw mi355x
```

**3. Use a predefined profile:**
```bash
# Run the MI355X priority optimization profile
/optimize --profile mi355x-priority
```

**4. Full sweep with custom round limit:**
```bash
# All operators x all backends, max 5 rounds each
/optimize --operators gemm,attention,grouped_gemm \
          --backends triton,ck \
          --hw mi300x \
          --max-rounds 5
```

### Configuration

Edit `config/optimizer.yaml` to customize:

- **GPU pool**: Which GPU IDs are available for optimization
- **Profiles**: Predefined optimization target sets for common scenarios
- **Defaults**: Round limits, bottleneck thresholds, accuracy standards
- **Knowledge search**: Web search templates per operator (used during bottleneck L2)

### Monitoring

The terminal Dashboard starts automatically when `/optimize` is invoked. Key interactions:

| Key | Action |
|-----|--------|
| `q` | Graceful shutdown — finishes current rounds, then stops |
| `Ctrl+C` | Immediate stop — worktrees preserved for later resume |
| `w` | Cycle the performance chart between Workers |
| `d` | Show detailed round history for the selected Worker |
| `p` | Pause or resume a Worker |

### Inspecting Results

During and after optimization, results are stored in structured directories:

```bash
# View current session state
cat agent_docs/mi355x/optimizer-state.json | python -m json.tool

# Read a specific round report
cat agent_docs/mi355x/gemm-triton/round-3/report.md

# Compare baseline vs optimized performance
diff <(cat agent_docs/mi355x/gemm-triton/round-3/baseline.json | python -m json.tool) \
     <(cat agent_docs/mi355x/gemm-triton/round-3/optimized.json | python -m json.tool)

# View live activity across all Workers
tail -f agent_docs/mi355x/*/activity.jsonl
```

### Resuming a Session

If a session is interrupted (Ctrl+C or crash), worktrees and state are preserved:

```bash
# Resume the last session — Coordinator reads optimizer-state.json
# and restarts Workers from their last completed round
/optimize --resume

# Resume a specific session
/optimize --resume --session 2026-04-07-001
```

The Coordinator detects each Worker's last committed round and resumes from there.

### Merging Results

After optimization completes, review the merge plan:

```bash
# The merge plan and PR descriptions are in:
cat agent_docs/mi355x/session-2026-04-07-001/merge-plan.md
cat agent_docs/mi355x/session-2026-04-07-001/pr-descriptions/gemm-triton.md

# To merge a recommended branch:
git merge opt/gemm-triton-001

# To create a PR directly (uses the generated description):
gh pr create --title "Optimize GEMM Triton on MI355X (+62.5%)" \
  --body-file agent_docs/mi355x/session-2026-04-07-001/pr-descriptions/gemm-triton.md
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| "No GPUs available" | Check `HIP_VISIBLE_DEVICES` and `config/optimizer.yaml` gpu_pool |
| Worker stuck in VERIFY | Check `accuracy.log` for failures; may need to relax SNR thresholds for experimental changes |
| Dashboard not updating | Verify `optimizer-state.json` is being written; check Worker process is alive |
| Bottleneck L1 fails (no omniperf) | Install omniperf or skip to L2 (knowledge mining) via `--skip-profiling` |
| Worktree conflicts on merge | Follow `merge-plan.md` merge order; resolve conflicts in `*_impl.py` dispatch files |
| Session resume fails | Delete stale lock: `rm agent_docs/{hw}/optimizer-state.lock` |

### Best Practices

1. **Start small**: Run a single `--tasks gemm:triton` first to validate the loop before scaling to parallel.
2. **Use profiles**: Define commonly-used target sets in `optimizer.yaml` to avoid typing long CLI args.
3. **Review before merge**: Always read `merge-plan.md` — not all branches are recommended for merge.
4. **Monitor GPU memory**: Large Triton autotune can exhaust GPU memory. Set `TRITON_AUTOTUNE_MAX_CONFIGS` to limit compilations.
5. **Save successful strategies**: After a successful session, update the skill knowledge base (`skills/*/SKILL.md`) with newly discovered optimization techniques for future runs.
