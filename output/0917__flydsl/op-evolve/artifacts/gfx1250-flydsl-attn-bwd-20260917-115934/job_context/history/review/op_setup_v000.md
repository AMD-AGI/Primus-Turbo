# Op Setup — gfx1250-flydsl-attn-bwd (v000)

**SUCCESS** — baseline, beat, eager, unit tests, benchmark and validation all built,
all run on the card through `runtime.runner`. `validation.py` exits 2 at round 0, which is
the expected and valid result: the baseline is correct and deterministic and **0.068x beat**.

---

## 1. Which branch step 1 took — check this first

**Branch B: developed from the backend corpus. No best-practice recipe was adopted, because
none matches `op.config`.**

- **Source repository:** `https://github.com/AMD-AGI/Primus-Turbo`
  (checkout `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo`)
- **Commit:** `5690a771874a0fe8ab6277c245a411a1f746e9ff` (2026-09-17 12:01:15 +0000)
- **Files built from:** `output/0917__flydsl/kernels/{README.md, odo_gfx1250.py,
  dkdv_loop_gfx1250.py, dq_gfx1250.py, wmma_layout_probe.py}` — the three sources
  `op.baseline.sources` names, plus the layout probe every index derives from.

`knowledge/backends/flydsl/attention/` holds exactly two recipes and both were diffed field
by field. Full tables are in `op/baseline/PROVENANCE.md`; **the fields that differ:**

| field | `recipes/hd64.md` | `recipes/hd128.md` | `op.config` |
|---|---|---|---|
| `arch` | gfx950 | gfx950 | **gfx1250** |
| `heads_q_per_kv` | power of two in [8, 256] | power of two in [8, 256] | **4** |
| `format` | sbhd / thd (bshd at batch 1 only) | same | **bshd at batch 4** |
| `determinism` | dQ not reproducible by default | dQ atomic bf16, `deterministic=True` costs 22.7% | **atomic-free, every element written once** |
| `q_head_dim` / `kv_head_dim` | **64 / 64** | 128 / 128 ✓ | 128 / 128 |

Matching on both: `causal: bottom-right`, `sliding_window: none`, `sparsity: dense`,
`direction` includes bwd, `dtype.qkv: bf16`, `softmax_scale: 1/sqrt(D)`, `sink: none`.

The arch mismatch is not a formality. Measured by in-container LLVM probe: on gfx1250
`llvm.amdgcn.mfma.*`, `llvm.amdgcn.ds.read.tr16.b64` and `llvm.amdgcn.permlane32.swap` all
**cannot select** — gfx1250 has no MFMA at all — and `warp_size = 64` is hardcoded in the
recipe's helper. `heads_q_per_kv` is separately a hard *backward correctness* gate there
(`LD_VEC >= 2`), not a tiling preference. Nothing in either recipe was retargetable.

Every referenced implementation built and ran in this environment, so the "stop and report a
reference that will not build" rule did not trigger.

---

## 2. What was created

| path | what it is | how verified |
|---|---|---|
| `op/baseline/kernels.py` | three FlyDSL gfx1250 kernels — `k_delta_bshd` (verbatim from `odo_gfx1250.py`), `k_dkdv`, `k_dq` — plus `@flyc.jit` launchers | built and ran on the card; 52.5–52.8 dB on every shape |
| `op/baseline/impl.py` | `attn_bwd(do,q,k,v,o,lse,softmax_scale=None,causal=True) -> (dq,dk,dv)` | the API `op.reference.api` names; used by ut, benchmark and validation |
| `op/baseline/_env.py` | flydsl 0.3.2 + aiter on `sys.path`, `TORCH_BLAS_PREFER_HIPBLASLT=0`; asserts version and `gfx1250` | asserts fire at import |
| `op/baseline/PROVENANCE.md` | branch, source, commit, files read, full config diff, known weaknesses | — |
| `op/beat/impl.py` | aiter prebuilt gfx1250 ASM backward, `dkdv_heads="q"` + host GQA reduction, same API | PASS on 5 shapes; measured every session |
| `op/beat/PROVENANCE.md` | library and versions | — |
| `op/eager/impl.py` | plain torch fp32 chunked reference, **explicit** bottom-right mask | the only precision reference; SDPA deliberately unused |
| `op/ut/common.py` | 9 shapes, `make_inputs`, `forward_reference`, path-based `load_impl`, NaN allocator poisoning, `sqnr_db` | — |
| `op/ut/test_correctness.py` | per-tensor ≥50 dB vs eager, both causal modes | ran: all PASS |
| `op/benchmark.py` | measures and prints; decides nothing | ran; 9 rows over 3 arms × 3 shapes |
| `op/validation.py` | the sole success criterion | ran once; exit 2 as expected |
| `op/runner_util.py`, `op/drive.py` | detached-run helper: `setsid` + rc sentinel + poll, attaches to IN-FLIGHT, re-runs FINISHED | every measurement in this report went through it |

Not created, by instruction: `op/current/`, `rounds/000/`, `state.yaml`.

**Shapes covered by `op/ut/`**: all three spec shapes, plus `toy`, `gqa4_small`, `mha`
(G=1), `unequal_seqlen` (512/1024), `unequal_seqlen_2` (1024/2048) and `sq_gt_skv`
(1024/512, **non-causal only** — under bottom-right causal with `Sq > Skv` the first
`Sq−Skv` queries have an empty window, so `lse = -inf` and `p = NaN` in *any* implementation
including the fp32 reference; that is an undefined input, not a kernel defect, and the reason
is written into `common.py`). Both causal modes are exercised where defined.

---

## 3. Measurements

One session, `runtime.runner`, palindromic order, median of 51, 3 s continuous warmup,
256 MiB L2 flush, sclk witness on every line. Raw: `op/.runs/bench_v000.json`.

| shape | baseline ms | baseline TF/s | current | beat ms | beat TF/s | baseline ÷ beat | SQNR dq/dk/dv | gate |
|---|--:|--:|---|--:|--:|--:|---|---|
| fast (1×1024×1024, 8/2, 128) | 1.7665 | **3.04** | = baseline, 0.53% | 0.1054 | 51.0 | 0.061x | 52.61 / 52.65 / 52.83 | PASS |
| proxy (1×4096×4096, 32/8, 128) | 9.4310 | **36.44** | = baseline, 0.12% | 0.5947 | 577.9 | 0.063x | 52.52 / 52.57 / 52.67 | PASS |
| prod (4×8192×8192, 32/8, 128) | 96.0533 | **57.24** | = baseline, 0.05% | 7.6134 | 722.2 | 0.080x | 52.56 / 52.60 / 52.71 | PASS |

sclk 1048–1100 MHz across the session (the card is VR-throttled to 1100 and drifts down
inside a window; palindromic ordering is what keeps the ratios valid through it).

`current` was not measured from `op/current/` — the framework has not created it yet and
Op Setup must not. Instead a **copy** of `op/baseline/` was measured as a third arm in the
same session, which is exactly what round 0's `current` will be. It agreed with `baseline`
to 0.53% / 0.12% / 0.05%. The copy was deleted afterwards.

**One thing that had to be fixed to get that agreement, and it matters for every round.** At
the first attempt (20 iterations) the two identical directories disagreed by **7.8%** at the
`fast` shape, while their `min_ms` differed by 0.24%. A 101-iteration re-run put them
0.011% apart. `fast` is 1.8 ms and launch-bound (3 TF/s on a ~1000 TF/s card), and a
20-iteration median there is not converged. **The default was raised to 51 iterations**, with
the reason written into `benchmark.py`. Had it been left at 20, the evolve loop would have
been handed a phantom 8% regression to chase on its smallest and cheapest shape.

---

## 4. What `validation.py` checks

> **Success condition:** dq, dk and dv each ≥ 50 dB against `op/eager/` on all three spec
> shapes with full `isfinite` coverage asserted first, **and** dq/dk/dv bitwise identical
> across 200 consecutive runs at the fast shape, **and** the geometric mean of
> (candidate TFLOP/s ÷ beat TFLOP/s) across the three shapes ≥ 1.00.

Benchmark statistic: **median of 51 timed CUDA-event iterations** per arm per shape, after
3 s of continuous warmup, arms in palindromic order, L2 flushed outside the event window,
one input shared by all arms. `validation.py` contains no timing loop — every number it
prints comes from a `benchmark.py` subprocess.

It takes one optional argument, the implementation **directory**, defaulting to
`op/current/`, and hands it to `benchmark.py` as `--arm-path candidate=<abs dir>`. It is
never turned into a name and looked up: `rounds/<n>/op/` has basename `op`, and a name
lookup would measure `op/op`. Because the benchmark takes a path directly there is no
staging step at all, so there is also no symlink to resolve back out of `op/`. Outside
`job_context` it depends only on the shipped `tools/op_flops.py`, imported from there and
not copied.

The determinism gate is `op.config.determinism_gate` run as written. Bitwise identity is the
observable form of "atomic-free": a split-k or atomic reduction breaks it within a few runs
even when it is faster.

---

## 5. Live anchor

**Yes.** `op.target.beat_measured_same_run: true`, and `validation.py` measures `op/beat/`
in the same process invocation as the candidate, every round. No stored anchor is read
anywhere, so clock drift cannot masquerade as progress. The spec's
`beat_reference_figure` of 10.160 ms at the production shape is a sanity check only; the
same-session measurement came out **7.61 ms**, i.e. faster than the quoted figure, so
nothing is being flattered by a stale number.

---

## 6. Inferred, and known weaknesses

**Inferred (not stated in the spec):**

- The spec names three source files but not how to combine them. The wiring — a delta pass,
  a dkdv pass and a dq pass, in that order — was chosen here.
- `k_dkdv` **flattens its runtime loop over `(q head, q tile)` so the GQA reduction happens
  in registers.** This departs from the ASM anchor, which writes `[B,Skv,Hq,D]` and reduces
  on the host. It was chosen because it is simpler, and it makes `op.config.determinism`
  true by construction. It is also most of why the baseline is far stronger than the spec
  expected (below).
- The aggregation rule for `op.shape.mode: sweep` is not spelled out in the spec.
  **Geometric mean of the per-shape ratios** was chosen, so no single shape dominates.
- `validation.py` gates causal only, matching `op.config.causal: bottom-right`. Non-causal
  correctness is covered by `op/ut/` but is not part of the success criterion.

**⚠ The baseline is roughly 10x stronger than `op.baseline.reported_tflops` anticipated, and
the gap to beat is roughly 10x smaller than the job description assumed.**
The spec records 5.8 TFLOP/s and a "~93x" gap. Measured: **57.2 TFLOP/s at the production
shape, 0.080x beat — a 12.5x gap, not 93x.** The difference is the in-kernel GQA reduction
above, worth about 4x over the naive wiring the job author had in mind, plus the rest from
building on `dkdv_loop_gfx1250.py` (which post-dates the README and already has the runtime
q loop) rather than on the single-tile probes. **This makes the job harder than the spec
implies: there is 12.5x to find, not 93x, and `min_gain: 0.0` with `max_rounds: 40` is now
aimed at a much narrower target.** Flagging it now rather than after forty rounds have been
measured against it.

**Known baseline weaknesses — all deliberate, all round-1 material:**

1. **No causal tile-skipping in either main kernel.** Both do full `S²` work while causal
   masks about half of it. This is the largest single structural inefficiency and the
   obvious first target — roughly 2x is sitting there.
2. **One wave (32 lanes) per workgroup**, inherited from the bring-up probes: no multi-wave
   scheduling, no TDM async pipeline, no `sched_barrier` placement. The baseline reaches
   5.7% of the measured 1002.7 TF/s bf16 roof; beat reaches 72%.
3. **`k_dkdv` pads a 16-wide contraction to the WMMA's 32**, wasting half that GEMM.
4. **No bank-conflict swizzle** on the LDS staging.
5. Shape constraints asserted, not handled: `D == 128`, `Sq % 16 == 0`, `Skv % 32 == 0`,
   `Hq % Hkv == 0`, `B*S*H % 32 == 0`. Every spec shape divides; a new shape might not.

**Shaky measurement to keep an eye on:**

- **Precision margin is thin by design.** Returning bf16 grads caps SQNR near 52–53 dB
  against a 50 dB gate, so there is only ~2.5 dB of headroom. This is the expected regime
  (the spec's own precedent quotes 52.24 dB), but any round that adds an approximation will
  hit the gate quickly.
- **The beat arm is thinner still.** `dkdv_heads="q"` plus the host reduction sums four bf16
  slices, and beat's dk/dv measure **50.2–51.0 dB** — barely clear of the same gate, against
  the baseline's uniform ~52.5 dB. The configuration is the one `op.target.beat` specifies,
  but the anchor is closer to failing precision than the code being optimised is.
- **The card drifts.** sclk fell to 1048 MHz inside the production window and recovered.
  Ratios are protected by palindromic ordering and same-session beat; absolute TFLOP/s
  figures are only meaningful with the clock witness printed beside them, which every
  `RESULT` line carries.


---

## Review verdict

- 2026-09-17T12:57:23 -- approved (non_interactive)
