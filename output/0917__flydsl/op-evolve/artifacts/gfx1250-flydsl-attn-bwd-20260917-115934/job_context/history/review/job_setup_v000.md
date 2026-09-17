# Job Setup -- gfx1250-flydsl-attn-bwd, v000

**SUCCESS.** The spec is complete on disk, all four LLM roles answered and resumed, every
environment check passed, and `check_runner.py` passed all five of its checks including the
optional ATT decoder. `op/` was not created -- that is Op Setup's.

Two things the user should read before approving, neither of which blocks the job:
the `torch gemm` figure `check_runner` recorded is **not** a roof for this card (§5), and
`TORCH_BLAS_PREFER_HIPBLASLT=1` makes bf16 `torch.matmul` **raise** on this image (§4).

---

## 1. Account checks

`python3 tools/check_llm_account.py --config .../gfx1250-flydsl-attn-bwd_final.yaml` --
exit 0. One real call per role, then a second that resumes the first. Log:
`logs/step3_llm_accounts.log`. No key, token or file content was read by me or is recorded
anywhere under `job_context/`.

| Item | Expected | Observed | Pass |
| --- | --- | --- | --- |
| `agent.account.anthropic` exists | a readable file | `/home/lihuzhan/.op_evolve_anthropic`, 423 B | pass |
| `agent.account.openai` exists | a readable file | `/home/lihuzhan/.op_evolve_openai` -> symlink to `.op_evolve_anthropic`, as the user's comment said | pass |
| required variables defined | `ANTHROPIC_API_KEY` (claude), `OPENAI_API_KEY` (codex) | both present, plus both `*_BASE_URL`, `ANTHROPIC_CUSTOM_HEADERS`, `CURSOR_API_KEY`, `LLM_GATEWAY_KEY` (names only) | pass |
| `setup` -- call | `claude` / `claude-opus-5` answers | answered in 2.048 s, $0.0149 | pass |
| `setup` -- resume | session handed back, context intact | session `87a9e9a3…` resumed, context intact | pass |
| `profiler` -- call | `claude` / `claude-opus-5` answers | answered in 2.067 s, $0.0149 | pass |
| `profiler` -- resume | session handed back | session `d0fe1052…` resumed, context intact | pass |
| `planner` -- call | `claude` / `claude-opus-5` answers | answered in 2.467 s, $0.0421 | pass |
| `planner` -- resume | **3-Act continues this session** | session `a818cfc8…` resumed, context intact | pass |
| `reviewer` -- call | `codex` / `gpt-5.6-sol` answers | answered in 2.315 s | pass |
| `reviewer` -- resume | **4-Reflect continues this session** | session `01a0af40…` resumed, context intact | pass |

The two-SDK setup is kept. `reviewer: null` was offered by the user as an alternative and is
not needed: the `codex` side works.

---

## 2. Environment checks (step 4 -- the machine)

Logs: `logs/step4_environment.log`, `logs/step4_clock_witness.log`,
`logs/step4_dmesg_gpu_health.log`. Full detail in `env/README.md`.

| Item | Expected | Observed | Pass |
| --- | --- | --- | --- |
| ssh hops (`runtime.runner.ssh`) | -- | **key absent: zero hops.** Local docker; controller, node and container are all `heliosr-1b114-c07-1` | pass (n/a) |
| container `fa-repro` exists | a pre-existing, running container | **Up**, 3 h old, image `fa-tune:deps` | pass |
| docker image pull | not attempted -- `image: null` and `container:` is set, so the runner attaches and never creates | nothing pulled | pass (n/a) |
| `docker.options` applied | `[]` | none applied, and none *would* be: `options` is read only on the create path. Actual options came from whoever started the container: binds `/home/lihuzhan:/home/lihuzhan`, devices `/dev/kfd` + `/dev/dri`, `ipc=host`, `net=host`, `privileged=false`, `seccomp=unconfined`, `group_add=[video,993]`, `shm=16 GiB` | pass |
| GPU present | yes | yes, exactly **one** (`device_count() == 1`) | pass |
| `runtime.gpu_arch` | `gfx1250` | **`gfx1250`** from all three independent sources: `rocm-smi` GFX Version, `rocminfo` `amdgcn-amd-amdhsa--gfx1250`, torch `gcnArchName`. Part is `AMD Eng Sample: 100-000001046-04` | pass |
| `runtime.gpu_id` | `0` usable | usable: allocation + bf16 GEMM both ran on device 0 | pass |
| `runtime.gpu_pool: []` | one card | correct, there is no second card | pass |
| `rocm-smi` installed and runs | yes | yes, at `/opt/venv/bin/rocm-smi`. **Driver version `7.1.1.31300009`** | pass |
| torch imports | yes | `2.11.0+rocm7.14.0a20260625`, `torch.version.hip 7.14.60850` | pass |
| torch sees the GPU | yes | yes, `AMD Radeon Graphics` / `gfx1250` | pass |
| torch executes a GPU op | yes | 1024^3 bf16 matmul, all-finite result | pass |
| sudo | spec says `sudo: false` | `sudo -n true` **succeeds -- sudo IS available.** Kept `false`: the field is a policy, not a capability | pass (noted) |
| `rocprofv3` present | needed by step 5 | `1.3.2` (`git feb9c98f`) | pass |
| flydsl 0.3.2 resolves | `flydsl.__file__` must be the 0.3.2 copy | `0.3.2` at `/home/lihuzhan/.local/flydsl032/flydsl/__init__.py`; `flydsl.expr.buffer_ops` **absent** as expected; image's `0.2.4` still intact and untouched | pass |
| aiter imports | baseline needs it | yes, off `/home/lihuzhan/code/aiter-src` -- **it is not installed in the container** | pass |
| baseline sources exist | 3 files | `odo_gfx1250.py`, `dkdv_loop_gfx1250.py`, `dq_gfx1250.py`, all present | pass |
| beat launcher exists | `asm_bwd_launcher.py` | present and executable | pass |
| kernel log during the checks | no faults | **clean** -- no amdgpu line, no `MES`, no reset. A `dmesg` monitor was armed *before* the GPU work | pass |

## 3. Runner checks (step 5 -- the runner driving the machine)

`python3 tools/check_runner.py --config … --job-dir …` -- exit 0. Log:
`logs/step5_runner.log`; the table it wrote is the top of `env/README.md` and the script it
ran is `env/run_torch_gemm.sh`.

| Item | Expected | Observed | Pass |
| --- | --- | --- | --- |
| reach | the machine answers | `heliosr-1b114-c07-1 (kfd) via docker` | pass |
| torch gemm | a GPU GEMM runs | `4096^3 bf16, 5.060 ms, 27.2 TFLOP/s` -- **see §5, this is not a roof** | pass |
| shared path | job dir identical from both sides | verified both directions at the job dir | pass |
| counters | `rocprofv3` collects and writes rows | `GRBM_COUNT`, 29 rows | pass |
| decoder (allowed to fail) | recorded either way | **reachable** -- the profiling agent may fetch it, so ATT thread traces are available | pass |

---

## 4. Resolved op spec

`gfx1250-flydsl-attn-bwd_final.yaml` is the user's input plus everything below; the reasons
are in `history/changelog.md`. "user" = written by the user; "inferred" = implied by the
input and now written out; "observed" = measured on this machine today.

| Field | Value | Source |
| --- | --- | --- |
| `job.name` / `id` / `started_at` | `gfx1250-flydsl-attn-bwd` / `20260917-115934` / `2026-09-17T11:59:34` | user / framework |
| `op.type` | `attention` | user |
| `op.config.causal` | `bottom-right` | user |
| `op.config.sliding_window` / `sparsity` / `sink` | `-1` / `dense` / `none` | user |
| `op.config.q_head_dim` / `kv_head_dim` | `128` / `128` | user |
| `op.config.heads_q_per_kv` | `4` | user -- **checked**: consistent with all three shapes (8/2, 32/8, 32/8) |
| `op.config.format` | `bshd`, `[b, s, h, d]` | user |
| `op.config.direction` | `[bwd]`; the forward is run to produce `o`/`lse` and is **excluded from the measured region** | user / inferred |
| `op.config.softmax_scale` | `head_dim ** -0.5` | user |
| `op.config.softmax_scale_value` | `0.08838834764831845` | **inferred** -- written out so no stage re-derives it |
| `op.config.dtype` | qkv/out bf16; qk_acc/softmax/pv_acc/lse fp32 | user |
| `op.config.determinism_gate` | no atomics on any output; dq/dk/dv **bitwise identical across 200 runs** at the fast-iteration shape | **inferred** from the user's prose, made checkable |
| `op.reference.impl` / `logic` / `api` | as written | user |
| `op.reference.impl_dtype` | `fp32` | **inferred** from "a dense fp32 score tensor is 34 GB" |
| `op.baseline.root` | `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0917__flydsl/kernels` | **inferred + observed** -- the job file said only "Primus-Turbo", which is *not* in the op-evolve checkout |
| `op.baseline.sources` | `odo_gfx1250.py` (159.24/156.49 dB), `dkdv_loop_gfx1250.py` (141.9/141.0 dB), `dq_gfx1250.py` (144.90 dB) | user; all three **verified present** |
| `op.baseline.imports` | `torch`, `aiter` -- **never `primus_turbo`** | user |
| `op.precision_sqnr_db` | `50` | user |
| `op.precision_gate` | dq, dk, dv **each separately** >= 50 dB; NaN-prefilled buffers; full `isfinite` coverage asserted **before** SQNR | **inferred** from the HINTS file, which is stricter than the bare number |
| `op.shape.mode` | `sweep`, all three shapes scored | user |
| shape 1 | b1 sq/skv 1024, hq 8, hkv 2, d 128 -- fast-iteration; **5.37395e9 FLOP** | user + **inferred** FLOP |
| shape 2 | b1 sq/skv 4096, hq 32, hkv 8, d 128 -- quarter-FLOP proxy; **3.43681e11 FLOP** | user + **inferred** FLOP |
| shape 3 | b4 sq/skv 8192, hq 32, hkv 8, d 128 -- production; **5.49823e12 FLOP** | user + **inferred** FLOP |
| `op.guards` | `[]` -- no shape is a guard | **inferred** (sibling jobs do carry a `guards:` block, so the absence is worth stating) |
| `op.target.beat` | aiter prebuilt gfx1250 ASM backward, `dkdv_heads=q` + host reduction, **by 0%** | user |
| `op.target.beat_launcher` | `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/tools/gfx1250/asm_bwd_launcher.py` | **inferred + observed** -- *not* op-evolve's own `tools/` |
| `op.target.beat_margin_pct` | `0` -- parity is the bar | **inferred**, matches "REACH the ASM backward, not beat it by a margin" |
| `op.target.beat_measured_same_run` | `true` | **inferred** -- mandatory on a card that drifts 1100 -> 967 MHz inside one timing window |
| `op.target.tflops` / `roofline_pct` | `null` / `null` | user -- **confirmed still null**; filling either silently disables the relative gate |
| `op.backend` | `[flydsl]`, pinned | user |
| `evolve.max_rounds` / `max_timeout` | `40` / `48h` (= 172800 s) | user |
| `evolve.schedule` | `fast_rounds: 12`, `fast_per_deep: 5` | user |
| `evolve.min_gain` | `0.0` | **inferred** default, justified: the gap is ~93x, so gains are expected as integer factors, far above run-to-run spread |
| `evolve.turn_timeout` | `null` -- each module keeps its 3 h default | **inferred** default |
| `runtime.runner.type` / `ssh` | `docker` / `null` -- local, **zero hops** | user / **inferred** |
| `runtime.runner.docker.container` / `image` / `owned` | `fa-repro` / `null` / `false` | user / **inferred**; running image is `fa-tune:deps`, recorded in a comment only |
| `runtime.sudo` | `false` (policy) | user -- though sudo **is** available |
| `runtime.gpu_arch` / `gpu_id` / `gpu_pool` | `gfx1250` / `0` / `[]` | user, all three **observed and confirmed** |
| `runtime.host` | `heliosr-1b114-c07-1` | **observed** |
| `runtime.python_path` | `/home/lihuzhan/.local/flydsl032`, then `/home/lihuzhan/code/aiter-src` | **observed -- new, and load-bearing** |
| `runtime.env.TORCH_BLAS_PREFER_HIPBLASLT` | `'0'` | **observed -- see §4 below** |
| `runtime.observed.*` | torch, hip, rocm-smi path, rocprofv3, both flydsl versions, gpu count | **observed** |
| `review.auto_approve_after` | `10m` (= 600 s) | user |

---

## 5. What I inferred, and what you should look at

**Nothing in the job file was contradictory.** Everything below is either a path the job file
named relatively, or something it left implicit that a later stage would otherwise guess.

**The two path inferences worth checking.** The job file writes "Primus-Turbo/output/…" and
"tools/gfx1250/asm_bwd_launcher.py" as if relative to the op-evolve checkout. They are not:
that checkout has its own `tools/` and no `Primus-Turbo/`. I resolved both to
`/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/` and verified all four files exist. If
you meant a different tree, this is the one line to correct.

**`runtime.python_path` and `runtime.env` are new sections I added, and the baseline does not
import without them.** The job file says flydsl 0.3.2 lives at `/home/lihuzhan/.local/flydsl032`
but never says how a process picks it up, and never mentions that **`aiter` is not installed
in the container at all** -- it is imported off `/home/lihuzhan/code/aiter-src`, which the
baseline kernels add to `sys.path` themselves. Both entries, in that order, are now in the
spec. Also: the job file's instruction to `pip install --target /home/lihuzhan/.local/flydsl032`
is **already done** -- 0.3.2 is there, the image's 0.2.4 is intact, and I verified
`flydsl.__version__ == "0.3.2"`, `flydsl.expr.buffer_ops` absent, `import aiter` OK.

**`TORCH_BLAS_PREFER_HIPBLASLT=1` makes bf16 `torch.matmul` raise on this image.** hipBLASLt's
gfx1250 Tensile library files are missing. Measured, 4096^3 bf16: `=1` -> `RuntimeError:
HIPBLAS_STATUS_INVALID_VALUE` from `hipblasLtMatmulAlgoGetHeuristic`, the matmul never runs;
`=0` -> 27.3 TFLOP/s; unset -> 27.2 TFLOP/s. The spec now sets `'0'` explicitly, as the
baseline kernels already do. Anything written later that does a torch GEMM must do the same.

**⚠ `check_runner`'s `torch gemm 27.2 TFLOP/s` is NOT a roof for this card.** It is the
rocBLAS fallback path and it is broken-slow here. This card's measured bf16 compute roof in
its current degraded state is **1002.7 TFLOP/s** and HBM is **6.46 TB/s** (ridge 155
FLOP/byte). The check proves *a GPU GEMM executes*, which is all it claims -- do not let a
later round use 27 TFLOP/s as a ceiling, a reference or a baseline.

**Clock state, confirmed at setup.** The DPM table is truncated to `500Mhz / 1100Mhz` -- the
VR throttle is real and `THROTTLE_STATUS` reads `N/A` on this part, so the table is the only
tell. Current sclk 1100 MHz, 851 W, 43 C junction. This is why the target is purely relative
and same-run, and why `tflops`/`roofline_pct` stay `null`. Witness:
`logs/step4_clock_witness.log`.

**GPU health.** I armed a `dmesg` monitor before any GPU work; the window was clean. The
monitor did replay one historical event from the ring buffer: a `GCVM_L2_PROTECTION_FAULT`
plus `MES(0,0) failed to respond` about **2.9 h before setup** (kernel ts ~5044 against an
uptime of 15406). That is the event the HINTS file already records -- timing
`dkdv_heads="kv"` at s=8192 -- and it **recovered**: no `wait for reset ack`, and the card
serves GEMMs and `rocprofv3` counters now. Not a live condition, and not caused by any check
I ran. Separately, the box logs a CPU-side **corrected** machine check every 20-30 min
(`CPU:8`, `L3/GEN`, `Corrected error, no action required`); pre-existing, unrelated to the
GPU, but worth passing to whoever owns the hardware.

**Still unresolved -- none that blocks the job.** One judgement call you may want to
override: `evolve.min_gain`. I set the framework default `0.0`, on the argument that a ~93x
gap produces gains far above this machine's run-to-run spread. The framework's own guidance
is to set it from measured spread, which no round has produced yet. If round 1's floor
spread turns out comparable to its gains, raise it then.

**Not done, by instruction:** `op/` -- baseline, beat, unit tests and `validation.py` are Op
Setup's. `state.yaml` was not touched.


---

## Review verdict

- 2026-09-17T12:09:53 -- approved (non_interactive)
