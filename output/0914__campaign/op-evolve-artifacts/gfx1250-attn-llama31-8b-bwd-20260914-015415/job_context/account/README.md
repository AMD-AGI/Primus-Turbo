# LLM accounts

Which file each SDK's credentials come from, and which variables that file must
define. **Paths and variable names only. No value from any of these files is
recorded here, anywhere else under `job_context/`, or in any report.** The
framework never parses them either: `op_evolve/agents/account.py` *sources* the
file in a subshell and reads the resulting environment, so a value written in
terms of another variable arrives expanded.

Verify with the shipped script — not with a check of your own, and never by
`cat`-ing a credentials file:

    python3 tools/check_llm_account.py \
      --config <job_context>/gfx1250-attn-llama31-8b-bwd_final.yaml [--role planner]

It makes **one real call per role and then a second that resumes the first**. Both
matter: a key that parses but cannot reach the model fails the first, and a session
that cannot be handed back fails the second. 3-Act continues the planner's session
from 2-Plan and 4-Reflect continues the reviewer's, so a lost session id is two
modules losing the context they were designed around — not a small degradation.

## What this job needs

`agent.roles` names two SDKs across four roles:

| Role | SDK | Model | Effort |
| --- | --- | --- | --- |
| setup | claude | claude-opus-5 | xhigh |
| profiler | claude | claude-opus-5 | xhigh |
| planner | claude | claude-opus-5 | max |
| reviewer | codex | gpt-5.6-sol | xhigh |

| SDK | `agent.account` key | File the spec names | Must define | Optional | Where the key comes from |
| --- | --- | --- | --- | --- | --- |
| `claude` | `anthropic` | `~/claude/.amd_llm_env` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL`, `ANTHROPIC_CUSTOM_HEADERS` | the AMD internal LLM gateway, or console.anthropic.com for a direct vendor key |
| `codex` | `openai` | `~/claude/.amd_llm_env` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` | the same AMD internal LLM gateway, or platform.openai.com |
| `cursor` | `cursor` | `~/claude/.amd_llm_env` | `CURSOR_API_KEY` | — (the Cursor SDK offers no base-url variable its Python package reads) | cursor.com dashboard |

All three keys point at the same file. That is what the user wrote and it is fine —
one file may set all of them.

**`cursor` is configured but unused.** No role in this job names `sdk: cursor`, so
`CURSOR_API_KEY` is never read and its absence would not fail anything here. It was
left in place rather than removed.

Gateway note: the optional `ANTHROPIC_BASE_URL` / `ANTHROPIC_CUSTOM_HEADERS` pair is
how a gateway is addressed, and `account.py`'s docstring calls out the exact trap —
a line like `ANTHROPIC_CUSTOM_HEADERS="x-api-key: $LLM_GATEWAY_KEY"` must be
*sourced* so the inner variable expands. A regex parser hands the gateway the literal
`$LLM_GATEWAY_KEY` and gets back a 401 with nothing explaining why. This is the reason
to use the script rather than reading the file.

## Result, 2026-09-14

All four roles passed both the call and the resume. Full output in
`../logs/03_check_llm_account.log`.

| Role | SDK / model | Call | Resume |
| --- | --- | --- | --- |
| setup | claude / claude-opus-5 | ok, 2.187 s | ok, context intact |
| profiler | claude / claude-opus-5 | ok, 2.447 s | ok, context intact |
| planner | claude / claude-opus-5 | ok, 3.662 s | ok, context intact |
| reviewer | codex / gpt-5.6-sol | ok, 2.219 s | ok, context intact |

The codex reply carried no cost figure. That is the SDK not reporting one, not a
free call — and it is worth knowing, because `cost_usd` is read from the SDK and, per
the job file, is neither accumulated nor persisted. `evolve.max_rounds: 40` is the
only real bound on spend, at roughly $13–20 per optimize round.
