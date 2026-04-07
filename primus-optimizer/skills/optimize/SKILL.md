---
name: optimize
description: Coordinator skill that orchestrates parallel multi-agent operator optimization for Primus-Turbo
invocable: true
---

# Primus-Turbo Optimization Coordinator

You are the Coordinator for the Primus-Turbo multi-agent optimization system. You orchestrate parallel Worker agents, each optimizing a single operator x backend combination on an exclusive GPU.

## CLI Arguments

Parse arguments from the user's invocation:

| Argument | Description | Example |
|----------|-------------|---------|
| `--tasks` | Explicit task list (operator:backend pairs) | `gemm:triton,gemm:ck` |
| `--operators` | Operator list (combined with --backends) | `gemm,attention` |
| `--backends` | Backend list (combined with --operators) | `triton,ck` |
| `--hw` | Target hardware | `mi355x` |
| `--profile` | Named profile from optimizer.yaml | `blockwise-fp8` |
| `--max-rounds` | Override default max rounds | `10` |
| `--resume` | Resume a previous session | `2026-04-07-001` |
| `--gpu-ids` | Override GPU list | `0,1,2,3` |

## Phase 1: INIT

### Step 1: Load Configuration

```bash
python -c "
import sys, json; sys.path.insert(0, 'primus-optimizer')
from tools.config_loader import load_config, parse_tasks, resolve_profile

cfg = load_config()
defaults = cfg.get('defaults', {})
gpu_ids = cfg.get('gpu_pool', {}).get('device_ids', [0, 1, 2, 3])

# Parse tasks from CLI args or profile
# Option A: --profile
# hw, tasks = resolve_profile(cfg, '{profile}')

# Option B: --tasks
# tasks = parse_tasks(tasks_str='{tasks}')

# Option C: --operators + --backends
# tasks = parse_tasks(operators='{operators}', backends='{backends}')

print(json.dumps({'hw': hw, 'tasks': tasks, 'gpu_ids': gpu_ids, 'defaults': defaults}))
"
```

### Step 2: Generate Session ID

```python
import datetime
session_id = datetime.datetime.now().strftime('%Y-%m-%d') + '-001'
```

If resuming (`--resume`), load existing state instead.

### Step 3: Initialize State Store

```bash
python -c "
import sys; sys.path.insert(0, 'primus-optimizer')
from tools.state_store import StateStore
store = StateStore('agent_docs/{hw}/optimizer-state.json')
store.init_session('{session_id}', '{hw}', {tasks_json}, {gpu_ids})
"
```

### Step 4: Initialize GPU Pool

```bash
python -c "
import sys; sys.path.insert(0, 'primus-optimizer')
from tools.gpu_pool import GPUPool
pool = GPUPool({gpu_ids})
print(f'GPU Pool initialized: {pool.available_count()} GPUs available')
"
```

### Step 5: Print Dashboard Attach Command

```
Dashboard ready. Open another terminal and run:
  python primus-optimizer/tools/dashboard.py --state agent_docs/{hw}/optimizer-state.json
```

## Phase 2: DISPATCH

For each task in the task list, allocate a GPU and launch a Worker agent:

### GPU Allocation

```bash
python -c "
import sys; sys.path.insert(0, 'primus-optimizer')
from tools.gpu_pool import GPUPool
pool = GPUPool({gpu_ids})
gpu = pool.acquire('{operator}:{backend}')
print(f'Allocated GPU {gpu} to {operator}:{backend}')
"
```

### Launch Worker Agent

For each task that gets a GPU, use Claude Code's Agent tool:

```
Agent(
    description="Optimize {operator}:{backend}",
    isolation="worktree",
    run_in_background=true,
    prompt="""
    Read the Worker skill at primus-optimizer/skills/optimize-worker/SKILL.md
    and follow its instructions with these parameters:

    - operator: {operator}
    - backend: {backend}
    - hw: {hw}
    - gpu_id: {gpu_id}
    - worktree_branch: opt/{operator}-{backend}-{session_id}
    - max_rounds: {max_rounds}
    - dtype: {dtype}
    - bottleneck_threshold: {bottleneck_threshold}
    - bottleneck_patience: {bottleneck_patience}

    Execute the full optimization loop as defined in the Worker skill.
    """
)
```

### Queue Management

If there are more tasks than GPUs:
- Launch tasks up to the number of available GPUs
- Remaining tasks go into the task queue in optimizer-state.json
- When a Worker completes, its GPU is released and the next queued task starts

## Phase 3: MONITOR

Poll Worker states periodically. Use the state store to check progress:

```bash
python -c "
import sys, json; sys.path.insert(0, 'primus-optimizer')
from tools.state_store import StateStore
store = StateStore('agent_docs/{hw}/optimizer-state.json')
state = store.load()
for wid, w in state['workers'].items():
    print(f\"{wid}: status={w['status']} round={w['current_round']} bottleneck={w.get('bottleneck')}\")
print(f\"Queue: {state.get('task_queue', [])}\")
"
```

### Monitor Loop Actions

Check each Worker's status and act accordingly:

| Status | Action |
|--------|--------|
| `running` | No action, Worker is progressing |
| `bottleneck` | Trigger bottleneck escalation (see below) |
| `stuck` | Launch Review Agent in `stuck` mode |
| `completed` | Reclaim GPU, start next queued task (if any) |
| `failed` | Log failure, reclaim GPU, start next queued task |

### Milestone Review Trigger

When any Worker completes 5 rounds, launch the Review Agent:

```
Agent(
    description="Milestone review for {worker_id}",
    prompt="""
    Read the Review skill at primus-optimizer/skills/optimize-review/SKILL.md
    and perform a milestone review with:
    - review_mode: milestone
    - hw: {hw}
    - worker_id: {worker_id}
    - round: {current_round}
    """
)
```

## Phase 4: BOTTLENECK ESCALATION

Three-level escalation when a Worker reports `bottleneck`:

### Level 1: Fine-Grained Profiling

Instruct the Worker to run deep profiling:

```
Send message to Worker agent:
"Run deep profiling this round using rocprof and omniperf.
Execute these commands:

HIP_VISIBLE_DEVICES={gpu_id} rocprof --stats \
    -o agent_docs/{hw}/{op}-{be}/round-{N}/rocprof_stats.csv \
    python benchmark/ops/bench_{op}_turbo.py --dtype {dtype}

HIP_VISIBLE_DEVICES={gpu_id} omniperf profile \
    --path agent_docs/{hw}/{op}-{be}/round-{N}/omniperf \
    -- python benchmark/ops/bench_{op}_turbo.py --dtype {dtype}

Analyze the results and re-enter ANALYZE -> PLAN with this new data."
```

Update Worker status:
```bash
python -c "
import sys; sys.path.insert(0, 'primus-optimizer')
from tools.state_store import StateStore
store = StateStore('agent_docs/{hw}/optimizer-state.json')
store.update_worker_status('{worker_id}', 'bottleneck', 'L1')
"
```

### Level 2: Knowledge Mining

If the Worker reports bottleneck again after L1:

```
Agent(
    description="Knowledge mining for {operator}:{backend}",
    run_in_background=true,
    prompt="""
    Read the Knowledge Mining skill at primus-optimizer/skills/optimize-knowledge/SKILL.md
    and search for new optimization strategies with:
    - operator: {operator}
    - backend: {backend}
    - hw: {hw}
    - arch: {arch}
    - bottleneck_description: {bottleneck_details}
    - current_tflops: {current_tflops}
    - tried_strategies: {tried_list}
    """
)
```

When Knowledge Agent completes, inject its findings into the Worker's context.

Update status to `L2`.

### Level 3: Review Agent Final Verdict

If the Worker reports bottleneck a third time:

```
Agent(
    description="Bottleneck verdict for {worker_id}",
    prompt="""
    Read the Review skill at primus-optimizer/skills/optimize-review/SKILL.md
    and perform a bottleneck review with:
    - review_mode: bottleneck
    - hw: {hw}
    - worker_id: {worker_id}
    - round: {current_round}
    """
)
```

The Review Agent will decide: TERMINATE, PIVOT, or ESCALATE (to human).

## Phase 5: FINALIZE

When all Workers have reached a terminal state (completed, failed, or terminated):

### Step 1: Launch Final Review

```
Agent(
    description="Final optimization review",
    prompt="""
    Read the Review skill at primus-optimizer/skills/optimize-review/SKILL.md
    and perform a final review with:
    - review_mode: final
    - hw: {hw}
    - session_id: {session_id}

    Generate:
    1. merge-plan.md
    2. PR descriptions for each recommended merge
    3. final-summary.md
    """
)
```

### Step 2: Print Summary

After the Review Agent completes, print:
- Number of Workers, rounds completed, and total elapsed time
- Top-line performance improvements per Worker
- Location of merge-plan.md and PR descriptions
- Any branches recommended for merge

### Step 3: Cleanup Guidance

```
Optimization session {session_id} complete.

Results:
  agent_docs/{hw}/session-{id}/final-summary.md
  agent_docs/{hw}/session-{id}/merge-plan.md
  agent_docs/{hw}/session-{id}/pr-descriptions/

To merge recommended branches:
  See merge-plan.md for order and conflict notes.

To clean up worktrees:
  git worktree list
  git worktree remove <path>
```

## Error Handling

- If a Worker agent crashes, catch the error, update its status to `failed`, release its GPU, and continue with remaining Workers
- If the state file becomes corrupted, the StateStore uses file locking to prevent concurrent write issues
- If all GPUs are busy and tasks are queued, the queue is persisted in optimizer-state.json and survives session restarts
