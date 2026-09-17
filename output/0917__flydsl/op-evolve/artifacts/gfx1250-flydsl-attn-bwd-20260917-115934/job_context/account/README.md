# LLM accounts

Paths and variable names only. **No key, token or file content is recorded here, anywhere
under `job_context/`, or in the setup report.** Job Setup never opened these files to read
their values: `tools/check_llm_account.py` sources them itself and puts the values into the
environment of the process that talks to the model and nowhere else.

## Which roles need which SDK

`agent.roles` in `../gfx1250-flydsl-attn-bwd_final.yaml` names two SDKs:

| Role | SDK | Model | Effort |
| --- | --- | --- | --- |
| setup | `claude` | `claude-opus-5` | xhigh |
| profiler | `claude` | `claude-opus-5` | xhigh |
| planner | `claude` | `claude-opus-5` | max |
| reviewer | `codex` | `gpt-5.6-sol` | xhigh |

## Which file each SDK reads

The mapping is fixed in `op_evolve/agents/account.py`: sdk `claude` reads
`agent.account.anthropic`, sdk `codex` reads `agent.account.openai`, sdk `cursor` would read
`agent.account.cursor`.

| SDK | Spec key | File |
| --- | --- | --- |
| `claude` | `agent.account.anthropic` | `/home/lihuzhan/.op_evolve_anthropic` |
| `codex` | `agent.account.openai` | `/home/lihuzhan/.op_evolve_openai` -- a **symlink** to `.op_evolve_anthropic` |

One file therefore serves both SDKs, which is what the user's comment in the job file
intended. If single-SDK operation is ever wanted instead, set `roles.reviewer: null` rather
than deleting the symlink -- `check_llm_account.py` treats a null role as nothing to check.

## Which variables the file must define

The file is a **shell script and is sourced, never parsed**, so a value written in terms of
another variable (e.g. `ANTHROPIC_CUSTOM_HEADERS="x-api-key: $LLM_GATEWAY_KEY"`) arrives
expanded. A regex parser would hand the literal `$LLM_GATEWAY_KEY` to the gateway and get a
401 that says nothing about why.

| SDK | Required | Optional |
| --- | --- | --- |
| `claude` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL`, `ANTHROPIC_CUSTOM_HEADERS` |
| `codex` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| `cursor` (unused here) | `CURSOR_API_KEY` | none -- the Python package reads no endpoint variable, so offering one would promise a redirection that never happens |

The variable **names** `/home/lihuzhan/.op_evolve_anthropic` defines (values not read):
`ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_CUSTOM_HEADERS`, `OPENAI_API_KEY`,
`OPENAI_BASE_URL`, `CURSOR_API_KEY`, `LLM_GATEWAY_KEY`. That covers the required variable of
every SDK this project supports, which is why one file can back both roles here.

## Where to obtain the keys

Both SDKs are pointed at an internal LLM gateway by the `*_BASE_URL` variables rather than at
the vendors directly, and `LLM_GATEWAY_KEY` is the credential the other variables are written
in terms of. Obtain or rotate it through the gateway that issued it and re-run the check
below; do not hand-edit the derived variables. `CURSOR_API_KEY` is present but unused by this
job.

## How to verify, and what "verified" means

    python3 /home/lihuzhan/code/2026_0910__op-evolve/op-evolve/tools/check_llm_account.py \
      --config /home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/job_context/gfx1250-flydsl-attn-bwd_final.yaml
    # --role <name> narrows it to one role

For each role it makes **one real call and then a second that resumes the first**. Both
matter. A key that parses but cannot reach the model fails the first; a session that cannot
be handed back fails the second -- and the second is not a small degradation, because 3-Act
continues the planner's session from 2-Plan and 4-Reflect continues the reviewer's.

All four roles passed both calls at setup. Result: `../logs/step3_llm_accounts.log`.
