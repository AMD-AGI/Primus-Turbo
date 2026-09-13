# Phase 0 status — all GPU-free work, 2026-09-13

Branch: `gfx1250-attn-dispatch-and-tuning` (Primus-Turbo).
**No GPU was used.** Everything below is source reading, offline unit tests of pure-Python
logic, and writing. Nothing here is a performance claim.

---

## What changed in the code

### The dispatch fix — without this, no kernel work reaches a training step

`DenseAttnFwdTritonBackend` is the only dense attention backend that runs on gfx1250, and
**it was unreachable by default dispatch.** `AutoKernelDispatcher`'s fallback scan walks
`_DENSE_FWD_BACKENDS` in insertion order, AITER is registered ahead of TRITON, and
`DenseAttnFwdAiterBackend.can_handle` accepted any 4-D fp16/bf16 tensor on any arch. So
AITER always won — and on gfx1250 it reaches CK, which is CDNA-only: the forward succeeds
and the backward rejects the call *mid training step*.

- `attention_impl.py` — AITER's dense backend now declines on gfx1250. Varlen is
  deliberately untouched (separate class, and there is no Triton varlen path, so the
  forward-only behaviour there is unchanged).

### A live latent bug, found while reading

`_flydsl_common_ok` gated on `get_device_compute_capability() >= (9, 5)`. gfx1250 reports
`(12, 5)`, so **the gate was open**. Every FlyDSL FA builder hard-raises on a non-gfx950
arch (it emits `ds_read_tr16_b64`, a CDNA4 LDS transpose load). Nothing had hit this only
because `_gqa_group_ok` happens to refuse Llama-3.1-8B's G=4 — any model with G ∈ {8, 16, …}
would have dispatched to FlyDSL on gfx1250 today and raised inside a gfx950 JIT.

- Changed to an exact `is_gfx950()` check.

### An import-time fragility on gfx1250 builds

`setup.py:529-535` skips installing `flydsl` for a gfx1250 build, but
`attention_flydsl_impl.py` imported it unconditionally at module scope — and both
`attention_impl` and `flash_attn_interface` import that module at *their* module scope. On
such a build, `from primus_turbo.pytorch.ops.attention import flash_attn_func` would fail
outright, taking down the Triton path along with it.

- The flydsl import is now guarded; absence degrades to "FlyDSL declines" (`FLYDSL_AVAILABLE`),
  and the impl functions raise a diagnostic if ever reached.

### CI could not see any of this

`tests/conftest.py` skipped the **entire** suite on gfx1250, including the gfx1250-specific
tests written for it — the commit that added the Triton backend notes its 19 passing tests
were run "with the blanket skip lifted".

- The skip is now opt-out via `@pytest.mark.gfx1250`. **13 tests now run on that arch**,
  including two new dispatch-regression tests that pin the two fixes above.

### The tuning surface

The forward's `@triton.autotune` list held exactly one config (`num_stages=1, num_warps=4`);
so did the backward's; `_bwd_preprocess_use_o` had no decorator at all. Those values came
verbatim from a **CDNA**-targeted perf-kernel — wave64, MFMA, 64 KB LDS — while gfx1250 is
wave32 / WMMA / 320 KB LDS. `@triton.autotune` was functioning as a compile cache, never a
search.

- `PRIMUS_TURBO_ATTN_TRITON_TUNE` opens the space: `sweep` for a grid, or
  `fwd:num_stages=2;bwd:num_warps=2` to pin exactly one config (the mode an A/B round
  wants). **The default is unchanged** — unset, the kernel offers the same single config it
  always shipped, and a test asserts that.
- A malformed spec **raises** rather than falling back, because silently measuring the
  default produces a flat sweep that reads as "this knob does nothing".

**Deliberately not changed:** `FIXED_BLOCK_M`. It is a cross-kernel ABI — the forward writes
LSE and the backward preprocess writes delta *interleaved per BLOCK_M* into one buffer, and
`_lse_delta_views` reconstructs that indexing host-side. Retiling one without the others
mis-reads delta and produces smoothly-wrong gradients. Documented rather than touched.

---

## Tooling

| file | what it is |
|---|---|
| `tools/gfx1250/tune_attention.py` | one process, one config, one JSON line. Correctness before timing. |
| `tools/gfx1250/sweep_attention.py` | drives it over a candidate list; resumable fsynced JSONL ledger. |

Four defences are built in, each against a failure that was actually observed:

1. **SQNR separately for `out`, `dq`, `dk`, `dv`** against an fp32 reference, threshold
   50 dB. Not a stricter version of an output-only check — a *different* one. aiter's
   `BLOCK_N1=256`-alone config measures **1.31× faster with dq at 9.59 dB and dk/dv
   perfect**; an output-only or dk/dv-only gate *rewards* it. The existing bench's max-abs
   check returns an identical `8.011e-03` for every backend and never looks at gradients.
2. **Assert the config that ran is the config that was asked for**, before timing.
3. **`rc=139` is a retry, not a verdict** — the flex anchor page-faults ~2 runs in 12, and
   it is re-measured every round.
4. **Health checks read `dmesg`, never `rocm-smi`** — a wedged card leaves tasks stuck in
   `amdgpu_info_ioctl`, which *is* `rocm-smi`, so it hangs rather than alerting. PC
   sampling is not used at all: three attempts, three GPU faults, one needing a reboot.

Offline-tested: the spec parser (15 cases), the axis/spec round-trip between the two tools,
the report's exclusion of fast-but-wrong candidates, and endpoint-winner detection.

---

## Documents

| file | for whom |
|---|---|
| `output/0913__opt_plan__claude/index.html` | the plan, explained for a non-specialist reader |
| `docs/gfx1250-attention-tuning.md` | what is safe to retile, the traps, the measured priors, the candidate list |
| `agent/skills/.../hardware/gfx1250/overview.md` + `optimization-directions.md` | **new** — there was no gfx1250 hardware doc at all, only gfx942 and gfx950 |
| `output/.../phase0/gfx1250-attn-llama31-8b-bwd.yaml` | op-evolve job spec |
| `output/.../phase0/RUNBOOK.md` | exact commands for Phase 1 and 2 |
| `output/.../phase0/PLATFORM-ESCALATION.md` | the throttle, priced in this project's numbers |

Four wrong lines in the existing FlyDSL knowledge base were corrected: the gfx1250 WMMA
operand width (v8 → **v16**, because K=32), the TDM gating mechanism (`pipeline_fence` →
`s_wait_tensorcnt` + split barriers), the claim that LDS transpose-loads are gfx950-only
(gfx1250 has `ds_load_tr16_b128`), and the blank gfx1250 row of the arch matrix.

---

## Corrections to things stated earlier in this effort

Recorded because they were propagated before being checked.

- **The fp32 gradients are not a bug.** `dense_backward` passes `dq/dk/dv=None` so all
  three are allocated fp32, but PyTorch's autograd *casts* floating-point grads to the
  input dtype — visible in the bench trace as `bfloat16_copy_kernel` ×3, 0.182 ms. It is a
  memory and cast cost (2× gradient bytes, three casts, fp32 zero-fills), and the fp32
  buffer may well be load-bearing for the atomic dQ accumulation. **Measure before
  changing** — it is a Phase 2 item, not a Phase 0 fix.
- **"~32 exps per WMMA, ~16× over budget" was wrong by the wave width.** On wave32 a
  32-element row segment is *one* `v_exp_f32`. The correct exp count is
  `256 / (D_qk + D_v)` = **1.0 per WMMA** at D=128, inside the 1–2 budget. Cycle-weighting
  the whole softmax VALU stream still leaves D=128 softmax-bound, but by **~1.2–1.4×**, not
  an order of magnitude. The direction held; the magnitude did not.
- **The JIRA's "patch is skipped" root cause is wrong**, and its "enable AITER fmha on
  gfx1250" request is not reachable at all (CK is CDNA-only). The patch *is* applied and
  dies on `import aiter` one layer down.
- **The JIRA's 245 ms elementwise is not attention.** Measured elementwise inside a flex
  fwd+bwd is 1.25%, ≈12.8 ms/step, not 245 — a 19× gap. It is unfused eager RoPE plus
  layout transposes, and replacing the FA kernel will not remove it. Separate ticket.

---

## What Phase 0 could not do

- **Vendor the aiter Triton MHA seed.** aiter is not installed on this host and there is no
  source copy; it lives in the container images. Network access to `github.com/ROCm/aiter`
  works, so this can be done — but which seed wins is a round-0 measurement (aiter 34.111 ms
  vs turbo 58.424 ms were taken on *different images* and never cross-checked), so vendoring
  before that bake-off would be guessing.
- **Verify any of the code changes at runtime.** No torch outside the containers. The
  changes are structured so the default path is bit-identical where possible, and the two
  fixes that do change behaviour are covered by the new tests. `pytest -m gfx1250` is Step 1
  of the runbook for exactly this reason.
- **Settle the 1.675× discrepancy.** `c1325c7e` reports 220.6 TFLOP/s; the in-tree bench
  measures 131.7 for the same kernel at the same shape. The ratio matches the VR throttle to
  within noise. This is the highest-value single measurement waiting on hardware.
