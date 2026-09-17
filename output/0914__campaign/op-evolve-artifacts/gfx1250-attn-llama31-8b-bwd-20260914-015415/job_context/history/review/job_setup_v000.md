# Job setup — gfx1250-attn-llama31-8b-bwd, v000

**SUCCESS.** All four LLM roles pass call *and* resume; all five runner checks pass
including the optional decoder; the environment is as the job file describes it.
Nothing blocks the job from starting.

Four things below need your attention before Op Setup runs, and one of them may be
the most valuable thing in the job. None of them is a setup failure.

---

## Account checks

`python3 tools/check_llm_account.py --config …_final.yaml` → exit 0.
Full output: `job_context/logs/03_check_llm_account.log`. No key, token or
credentials-file content was read, copied or recorded anywhere.

| Item | Expected | Observed | Pass |
| --- | --- | --- | --- |
| SDKs named by `agent.roles` | claude, codex | claude, codex | ✅ |
| `agent.account.anthropic` → `ANTHROPIC_API_KEY` | set in `~/claude/.amd_llm_env` | present, sourced, expanded | ✅ |
| `agent.account.openai` → `OPENAI_API_KEY` | set in `~/claude/.amd_llm_env` | present, sourced, expanded | ✅ |
| `setup` — claude/claude-opus-5, call | token returned | ok, 2.187 s, $0.0145 | ✅ |
| `setup` — resume | session handed back, context intact | ok | ✅ |
| `profiler` — claude/claude-opus-5, call | token returned | ok, 2.447 s, $0.0145 | ✅ |
| `profiler` — resume | context intact | ok | ✅ |
| `planner` — claude/claude-opus-5, call | token returned | ok, 3.662 s, $0.0379 | ✅ |
| `planner` — resume (3-Act continues this session) | context intact | ok | ✅ |
| `reviewer` — codex/gpt-5.6-sol, call | token returned | ok, 2.219 s, no cost reported | ✅ |
| `reviewer` — resume (4-Reflect continues this session) | context intact | ok | ✅ |
| `agent.account.cursor` | — | configured but **unused**; no role names `sdk: cursor`. Inert, left alone. | n/a |

The codex reply carries no `cost_usd`. That is the SDK not reporting one, not a free
call — worth knowing because the job file notes cost is neither accumulated nor
persisted, so `max_rounds: 40` is the only real bound on spend (~$13–20/optimize round).

---

## Environment checks

`python3 tools/check_runner.py …` → exit 0 (`logs/05_check_runner.log`).
Host and GPU observations in `logs/04a`–`04d`. Everything below is what was seen,
not what was asked for.

| Item | Expected | Observed | Pass |
| --- | --- | --- | --- |
| ssh hops | `runtime.runner.ssh` absent → none | none; controller = node = container host = `ctheliosp-1b112-a37-1.mnb.dcgpu` | ✅ |
| GPU present | yes | `/dev/kfd` + 32 render nodes; **4** gfx1250 cards (kfd nodes 2–5) | ✅ |
| `runtime.gpu_arch` | gfx1250 | `gfx1250`, `gfx_target_version 120500` on all four | ✅ |
| `runtime.gpu_id: 1` usable | yes | `HIP_VISIBLE_DEVICES=1` → torch sees exactly 1 device, `gcnArchName gfx1250` | ✅ |
| wave32 / LDS / CUs | 32 / 320 KB / 256 CUs | `wave_front_size 32`, `lds_size_in_kb 320`, `simd_count 1024` → 256 CU / 128 WGP | ✅ |
| docker image | pulls **or** container exists | ⚠️ **pull DENIED** (needs `docker login`); image is present locally (`sha256:6e656de79e6c…`) and `docker run` works | ✅ with caveat |
| docker options | all 9 from spec | all accepted verbatim, incl. `CAP_SYS_ADMIN` + `seccomp=unconfined` | ✅ |
| container | `docker.container: null` → runner-owned | created `op-evolve-gfx1250-attn-llama31-8b-bwd`, running, provisioned | ✅ |
| `rocm-smi` installed and runs | yes | `/opt/venv/bin/rocm-smi`, SMI 4.0.0+feb9c98 / LIB 7.8.0; ran, did not hang | ✅ |
| ROCm version | — | **7.14.60850**, pip wheel SDK (not `/opt/rocm`) | ✅ |
| torch imports, sees GPU, runs a GPU op | yes | **2.11.0+rocm7.14.0a20260625**, hip 7.14.60850, bf16 matmul on GPU 1 OK | ✅ |
| sudo | `runtime.sudo: true` | **available, passwordless** (`sudo -n true`) | ✅ |
| runner: reach | machine answers | `ctheliosp-1b112-a37-1.mnb.dcgpu (kfd) via docker` | ✅ |
| runner: torch gemm | a GPU GEMM runs | `4096³ bf16 1.116 ms 123.2 TFLOP/s` — see caveat 2 | ✅ |
| runner: shared path | same dir both sides | verified **both directions** at the job dir | ✅ |
| runner: counters | rocprofv3 writes rows | `GRBM_COUNT, 27 rows` | ✅ |
| runner: decoder (optional) | recorded either way | reachable | ✅ |
| dmesg `MES(` / `GPU Hang` / `wait for reset ack` | clean | **0 before and 0 after** every check | ✅ |
| GPU idle before/after | idle | all 4 cards at 0% | ✅ |

Per runtime hazard 1 no PC sampling was run; per hazard 5 `tools/clock_probe.py` was
not run; per hazard 2 `dmesg` was grepped *before* `rocm-smi` was ever called.

---

## Resolved op spec

Written to `job_context/gfx1250-attn-llama31-8b-bwd_final.yaml`. The user's job file is
byte-for-byte intact above the `setup_resolution:` line; six values were made explicit
inline and the rest is a new block. **No `[user]` field was changed**, and in particular
`op.target.tflops` and `op.target.roofline_pct` are still `null` as the file demands.

| Field | Value | Source |
| --- | --- | --- |
| `job.name` / `op.type` / `op.backend` | gfx1250-attn-llama31-8b-bwd / attention / [triton] | user |
| `op.config.causal` | bottom-right (≡ top-left at sq=skv=8192) | user |
| `op.config` heads / dims / dtype | Hq 32, Hkv 8, G=4, D=128, bf16, BSHD, fp32 acc | user |
| `op.config.direction` | [fwd, bwd] | user |
| `op.precision_sqnr_db` | 50, per-tensor on out/dq/dk/dv, no sampling | user |
| shape | b=4, sq=skv=8192, Hq=32, Hkv=8 | user |
| shape `name` | `b4_s8192_hq32_hkv8_d128` | **inferred** — spec shapes carried no name; matching is positional, so this is a label for people and `validation.py`, not a key |
| shape `head_dim` | 128 | **inferred** — loop.py falls back to `op.config.q_head_dim`; identical, written out so the fallback isn't load-bearing |
| shape `window_left` | -1 | **inferred** — op_flops.py's spelling of `sliding_window: null` |
| `op.target.beat` | flex_attention, block_causal, same run, beaten by 50% | user |
| derived **margin** | **1.50**, relative bar LIVE | computed by calling `loop.py::_margin()` on the final file |
| `evolve.max_rounds` / `max_timeout` | 40 / 36h (129600 s) | user |
| `evolve.turn_timeout` | `null` → module default 3h/turn | **inferred** (unset) |
| `evolve.min_gain` | `0.0` → any gain >1.0x counts | **inferred** (unset; 0.0 is the default) — see caveat 4 |
| derived **schedule** | deep rounds **[18, 24, 30, 36]** = 36 fast / 4 deep | computed by enumerating `schedule.mode_for()` |
| `runtime.runner.type` | docker (DockerRunner over LocalRunner) | user |
| `runtime.runner.ssh` | `null`, zero hops | **inferred** (absent, and type is `docker` not `ssh_docker`) |
| derived container | `op-evolve-gfx1250-attn-llama31-8b-bwd`, runner-**owned** | computed from `docker.container: null` |
| derived mount | `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve` | `resources.container_mounts()` |
| `runtime.gpu_id` / `gpu_pool` | 1 / `[]` → profiling pool is `[1]` | user + derived |
| `review.auto_approve_after` | 10m = 600 s | user |
| `op.reference.impl` | still `null` | **Op Setup's**, not setup's |

---

## What I had to infer, and what is still open

Seven open questions are recorded in full in `setup_resolution.open_questions`. The
four that need you:

**1 — The FLOP basis does not agree with the job file. (high)**
`tools/op_flops.py` counts an FA-2 backward as 2.5× the forward and returns *that
alone* for `backward=True`; `loop.py` passes `backward=True` because `direction`
contains `bwd`. So the ledger will divide a **forward-plus-backward** measured time by
a **backward-only** FLOP count. Every TFLOP/s in your job file is on the 3.5× fwd+bwd
basis — I checked all four anchors and all four reproduce exactly on it and none on
the framework's (59.642 ms → 129.06 vs your 129.0; 31.107 → 247.45 vs 247.4;
31.337 → 245.63 vs 245.6; 24.435 → 315.03 vs 315.0). **The ledger's TFLOP/s column will
read 1.400× below every number you read in the spec, the JIRA and the corpus.**
It does *not* affect pass/fail — the bar is relative, so the factor divides out — but it
invalidates every absolute and roofline statement, including "20.8 ms = 370 TFLOP/s =
37% of the roof", which reads 264 / 26% on the framework's basis. I did **not** change
`op_flops.py`: it is shared by every job in this repo and re-scaling it silently
rewrites other jobs' histories. Op Setup must pin one basis and say which; my
recommendation is to time fwd and bwd separately (which `op.reference.api` already
demands for its own reasons) and charge 2.199 / 3.299 TFLOP, which makes the basis
unambiguous by construction.

**2 — The card is not at the 1100 MHz this file was written against. (high, and this
may be the finding the header predicted)**
Under a sustained 45 s load, GPU 1 held **1701 / 1702 / 1702 MHz** at 100% utilisation.
That is the header's own *un-throttled* 2026-09-04 figure (1699–1703) to within 1 MHz,
and 1.55× the 1100 MHz state every absolute number in the file was taken in. It would
also explain the c1325c7e contradiction the header calls the highest-value probe in the
job (220.6 / 131.7 = 1.675×, the throttle ratio to within noise).
**This is necessary but not sufficient, and I want to be plain about that**: the load I
used was `torch.matmul`, which on this image runs at 113–124 TFLOP/s — a VALU-rate,
low-power load that may simply never approach the VR limit. What is established is that
the card is not *statically* capped at 1100 MHz. What is not established is the clock
under a real attention kernel. The header's paired re-measurement is now mandatory, and
it must record the clock **under the kernel being measured**.

**3 — The seed kernel and the mandated harness are not in the pinned image. (high —
blocks Op Setup)**
None of `fused_mha_bwd_kernel.py`, `attention_fused_bwd_impl.py`,
`tools/gfx1250/tune_attention.py` or `tools/gfx1250/exclusive.sh` exist in
`amdprimus:gfx1250-20260910`. Its Primus-Turbo is commit `f857e429` with only the
in-tree **unfused** path and no `tools/gfx1250/` at all. All four are in your host
checkout `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo` @ `d2f75576`, which is not
mounted — and `import primus_turbo` inside the container resolves to
`/workspace/Primus-Turbo`, so a bare `PYTHONPATH` would mix the two trees. Op Setup must
resolve this before it can build `op/baseline`. I did not choose between vendoring into
`op/` (my recommendation — `op/` is already on the mount) and adding a mount, because it
is Op Setup's structure to decide.

**4 — The seed contradicts the margin's justification. (high — needs your call)**
Your SEED block (dated 2026-09-13, "after a day of measurement") says to seed at the
fused kernel, 24.435 ms, and explicitly retires aiter's shipped config. Your
`op.target` block justifies the 50% margin by assuming the *opposite* seed — it argues
for aiter's shipped 34.111 ms precisely so "round 0 starts behind the anchor", and warns
that seeding at the tuned point means "any margin below that is not a bar at all". Both
are in the file; the SEED block is dated later and says it supersedes, but the target
prose was never updated. With the fused seed, round 0 begins at **1.27–1.28× the
anchor** — the exact situation that comment calls not a bar — and the 50% bar is
1.50/1.282 = **1.17× beyond the seed** (~20.9 ms). Still a real bar, but a 17% job
reported as a 50% one. I changed nothing: `beat` is `[user]` and the file forbids
touching the two null fields. **My recommendation is to keep the fused seed and the 50%
margin as-is** — it is the honest seed, the bar still sits above anything measured, and
re-seeding backwards to make a percentage look better is what this whole file argues
against — but you should decide before Op Setup builds `op/baseline`.

Also recorded, lower stakes: `evolve.min_gain` is 0.0 and **could not** be set from a
measured floor spread, because there is no `op/baseline` to repeat-measure yet (the
0.26% in your file is carried from a prior campaign, different seed, different clock) —
Op Setup owes ≥5 processes and a number in `env/README.md`; your schedule comment says
"~34 fast / 6 deep" but the code gives **36 fast / 4 deep** (the first deep is round 18,
not 13 — `fast_per_deep: 3` would give 6); the pinned image **cannot be re-pulled**
(access denied) so the one local copy is all there is; this node has **four** gfx1250
cards, not the one `gpu_pool`'s comment claims, and the 0/2/3 fence is a convention with
no mechanism behind it — per your hazard 6, check the card is idle before every round.

One trap worth repeating because it is now written into `env/README.md` as a passing
check: **`check_runner`'s `123.2 TFLOP/s` is not a roof.** It is `torch.matmul` →
hipBLASLt, and it is your runtime hazard 3 reproduced exactly (I re-measured: 123.8 at
4096³, 113.5 at 8192³ — flat in size, the signature you describe). The check means "a
GPU GEMM runs through the runner", which is all it claims. I did not measure a Triton
roof: writing an untuned Triton GEMM and reporting it as "the roof" would be worse than
reporting none.

---

*Written by 0-Setup / job_setup. `state.yaml`, `history/v000_original.yaml` and `op/`
were not created or edited.*


---

## Review verdict

- 2026-09-14T02:05:13 -- approved (non_interactive)


---

## Review verdict

- 2026-09-14T03:40:35 -- approved (non_interactive)
