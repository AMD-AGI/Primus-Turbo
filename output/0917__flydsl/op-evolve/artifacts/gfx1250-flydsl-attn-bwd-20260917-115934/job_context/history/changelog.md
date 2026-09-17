# Spec changelog

## v000 -- 2026-09-17T11:59:34

original job file, jobs/gfx1250-flydsl-attn-bwd.yaml

### v000 completed by Job Setup -- 2026-09-17

`gfx1250-flydsl-attn-bwd_final.yaml` is the user's input plus the values below. Nothing
the user wrote was changed or removed; everything added is marked `[resolved]` (implied by
the input, or derived from it) or `[observed]` (measured on this machine today).

**Paths the job file named relatively and Job Setup resolved to absolute, after verifying
each one exists:**

- `Primus-Turbo` -> `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo`. It is *not* inside
  the op-evolve checkout, so `op.baseline.root` now carries the absolute kernels directory
  and lists the three source files with the dB figures the job file quoted.
- `tools/gfx1250/asm_bwd_launcher.py` (the beat target) -> `op.target.beat_launcher`, in
  the same Primus-Turbo tree, not in op-evolve's own `tools/`. Both were ambiguous: the
  op-evolve checkout has a `tools/` of its own and no `Primus-Turbo/`.
- `~/.op_evolve_anthropic` / `~/.op_evolve_openai` -> `/home/lihuzhan/...`. Contents never
  read by Job Setup.
- The HINTS file, in the header comment.

**Implicit things now written out:**

- `runtime.runner.ssh: null` and `runtime.host` -- the job file has no `ssh` block, so this
  is a *local* docker runner with zero hops. Worth stating, because step 4 of setup checks
  ssh hops and there are none to check.
- `runtime.runner.docker.owned: false` and a note that `options: []` is never applied --
  `container:` is set, so the runner attaches to a user-owned container and the create-path
  (which is the only reader of `image` and `options`) is dead here.
- `runtime.python_path` and `runtime.env` -- NEW, and load-bearing. The job file says flydsl
  0.3.2 lives at `/home/lihuzhan/.local/flydsl032` but never says how a process picks it up,
  and does not mention that `aiter` is **not installed in the container** at all: it is
  imported off `/home/lihuzhan/code/aiter-src`, which the baseline kernels add to `sys.path`
  themselves. Without both entries, in that order, the baseline does not import.
  `TORCH_BLAS_PREFER_HIPBLASLT=0` is recorded for the same reason: hipBLASLt's gfx1250
  Tensile library is missing from this image and the first bf16 GEMM prints a wall of
  rocblaslt errors before falling back.
- `op.config.softmax_scale_value: 0.08838834764831845` -- the job file gives the expression.
- `op.config.determinism_gate` -- the user's prose turned into a checkable statement
  (no atomics on outputs; 200-run bitwise identity at the fast-iteration shape).
- `op.precision_gate` -- `precision_sqnr_db: 50` alone is weaker than what the HINTS file
  requires. Written out: dq/dk/dv each separately >= 50 dB, NaN-prefilled buffers and full
  `isfinite` coverage asserted *before* SQNR.
- `op.reference.impl_dtype: fp32`, implied by "a dense fp32 score tensor is 34 GB".
- `op.guards: []` -- `shape.mode: sweep` with no guard block means all three shapes are
  scored. Sibling jobs in `jobs/` do carry a `guards:` block, so its absence is worth
  making explicit rather than leaving to a reader.
- `op.target.beat_margin_pct: 0` and `beat_measured_same_run: true` -- "beaten by 0%" is
  parity, which matches the description's "REACH the ASM backward". Same-run measurement is
  not optional on a card that drifts 1100 -> 967 MHz inside one timing window.
- `evolve.min_gain: 0.0` and `evolve.turn_timeout: null` -- the framework defaults, written
  down with the reason they are right here rather than left to be looked up.
- `evolve.max_timeout: 48h` = 172800 s and `review.auto_approve_after: 10m` = 600 s, as
  parsed by `op_evolve.core.spec.parse_duration`.
- `job.id` / `job.started_at`, from the framework.

**Per-shape FLOP and compulsory-byte counts** added to each entry in `op.shape.shapes`,
computed with `tools/op_flops.py` rather than by hand, so no later stage re-derives them:
5.37395e9 / 3.43681e11 / 5.49823e12 FLOP. The production shape's 5.49823e12 FLOP over the
job file's quoted 10.160 ms anchor gives 541.2 TFLOP/s, which confirms its "~540".

**Observed values that contradict nothing but are worth stating:** `runtime.sudo: false` is
kept as the user wrote it, but `sudo -n true` *succeeds* on this host -- the field is a
policy, not a capability. `runtime.runner.docker.image` is kept `null`; the running
container's actual image, `fa-tune:deps`, is recorded in a comment only, because moving it
into the field would flip the runner to owning the container.

**Also recorded** in `runtime`, from knowledge already on this machine: the four safety
rules for this card (no rocprofv3 PC sampling, no autotuner on a training path, toy-shape
first launch with `AMD_SERIALIZE_KERNEL=3`, ABAB interleaving with a clock witness).

Nothing was found to be contradictory. `heads_q_per_kv: 4` is consistent with all three
shapes (8/2, 32/8, 32/8), and `op.target.tflops` / `roofline_pct` remain `null` as the
job file requires.
