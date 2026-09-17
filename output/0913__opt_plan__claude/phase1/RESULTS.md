# Phase 1 results — first GPU session, 2026-09-13

Node `heliosr-1b114-c07-1`, one gfx1250 (MI455X). Container
`amdprimus/amdprimus:gfx1250-20260910`, torch 2.11.0+rocm7.14.0a20260625, triton 3.6.0.
Shape everywhere: **b=4 s=8192 hq=32 hkv=8 d=128 bf16 causal** unless stated.
FLOP fwd+bwd = 7.697e12 per call; a training step is 32 of them.

---

## 0. Card state: still VR-throttled

The reboot did **not** clear it.

```
amdgpu 0001:01:00.0: WARN: GPU is throttled, expect performance decrease. VR.
pp_dpm_sclk:  0: 500Mhz    1: 1100Mhz *
```

Two sclk levels, ceiling 1100 MHz, against 1699–1703 MHz on this same card on 2026-09-04.
**Every absolute number below is in the throttled regime.** 0 GPU faults across the session.

Also note: `amdgpu` is blacklisted at boot on this node and must be `modprobe`d by hand after
a reboot. It was not loaded when the session started.

---

## 1. The rig reproduces the Phase-1 corpus

| | this session | Phase-1 log | Δ |
|---|---:|---:|---:|
| flex fwd+bwd | 31.337 ms | 31.508 ms | 0.5% |
| flex TFLOP/s | 245.6 | 244.3 | 0.5% |
| sdpa:FLASH fwd+bwd | 97.063 ms | 96.581 ms | 0.5% |
| turbo:TRITON fwd+bwd | 60.209 ms | 58.424 ms | 3.0% |

Correct-backend SQNR lands at **53.67 / 52.24 / 52.31 / 52.71** dB for out/dq/dk/dv against
the corpus's 53.67 / 52.25 / 52.31 / 52.73 — agreement to 0.02 dB, which independently
validates the new fp32 reference implementation.

---

## 2. SETTLED: the 220.6 vs 131.7 TFLOP/s contradiction was the clock

Commit `c1325c7e` reported 220.6 TFLOP/s at b=2 s=8192; the in-tree bench measured 131.7 for
the same kernel at b=4. The candidate explanations were "batch moves TFLOP/s" or "the clock
state differed". Measured today, same kernel, same session:

| shape | total ms | TFLOP/s |
|---|---:|---:|
| b=2 s=8192 | 29.735 | **129.42** |
| b=4 s=8192 | 60.209 | **127.83** |

**Batch does not move TFLOP/s** — 1.2% apart. That hypothesis is dead. The residual gap to
the commit's figure is **220.6 / 129.4 = 1.70×**, against a VR throttle whose own ratio is
~1.65×, on a card confirmed throttled right now.

**Conclusion: `c1325c7e` was measured on an un-throttled card, and this is the same kernel at
1100 MHz.** If the throttle is lifted, expect everything here to scale by ~1.65–1.70×.

---

## 3. Tuning: 1.66× over the shipped config, from two integers

Sweeps at the production shape, each candidate in its own process, correctness gated on all
four tensors before timing.

| config | fwd ms | bwd ms | total ms | TFLOP/s | ms/step |
|---|---:|---:|---:|---:|---:|
| shipped default | 10.781 | 48.537 | 59.319 | 129.7 | 1898 |
| `fwd:num_stages=2` only | 4.227 | 48.5 | ~52.8 | ~146 | ~1690 |
| **`fwd:num_stages=2; bwd:num_warps=2`** | **4.227** | **31.477** | **35.704** | **215.6** | **1142** |
| torch flex (the path to beat) | 7.506 | 23.831 | 31.337 | 245.6 | 1003 |

- **Forward `num_stages` 1 → 2 is worth 2.42–2.55×** on the forward alone (10.781 → 4.227 ms
  at b=4; 5.567 → 2.298 ms at b=2), at **bit-identical SQNR**. The prior from aiter's kernel
  was 2.06×; this is larger.
- **Backward `num_warps` 4 → 2 is worth 1.54×** (48.537 → 31.477 ms). Prior was 1.32×.
- `waves_per_eu`: 0 is best, 1 is neutral, **2 and 4 fall off a cliff** (54.9 ms and 104.4 ms
  at num_warps=2) — matching the corpus's `waves_per_eu=4 → 120.6 ms` cliff on aiter.
- `bwd num_stages` > 1 is consistently worse (39.3 ms at 2, 39.5 at 3). Unlike the forward.
- Repeatability: the champion measured 35.704 and 35.797 ms in two independent sweeps (0.3%).

**Status against the anchor: 35.7 ms vs flex's 31.3 ms — still 1.14× behind.** The whole
remaining gap is the backward (31.5 ms vs flex's 23.8 ms); the forward is now *ahead* of
flex (4.2 vs 7.5 ms).

---

## 4. FOUND: an intermittent dk/dv corruption in the Triton backward

The correctness gate fired on `fwd:num_stages=2; bwd:num_warps=2,waves_per_eu=1`:

```
out  53.67 dB  OK        dq  52.24 dB  OK
dk   -inf dB   WRONG     dv  -inf dB   WRONG
```

`out` and `dq` are **perfect** while dk/dv are destroyed — the mirror image of the known
aiter `BLOCK_N1` trap (where dq breaks and dk/dv stay perfect). An output-only check, or a
forward-only check, passes this silently. This is the concrete justification for gating on
all four tensors separately rather than on the output.

**It is not deterministic.** Three immediate repeats of the identical config all passed. Two
further observations point the same way:

- dk appeared to vary run to run in the 9th significant digit (52.30926834784377 vs
  52.30926798392079) while out and dq looked stable.
  **CORRECTED in §12: that was the fp32 reference wobbling, not the kernel.** The
  reference uses `torch.matmul` -> rocBLAS, which is itself non-deterministic, so the SQNR
  moved while the kernel output did not. Measuring determinism against a re-derived
  reference cannot separate "the kernel wobbles" from "the yardstick wobbles".

A 25-repeat study across the default config, the champion, and the failing config is running
to establish the rate and whether the champion and the *default* are also affected. **Nothing
should land until that is known** — an intermittent wrong gradient that survives a
single-shot check is exactly the failure mode that ships.

---

## 5. Environment findings worth recording

- **hipBLASLt is not merely slow here, it is absent.** The gfx1250 Tensile library
  (`TensileLibrary_lazy_gfx1250.dat`) is missing from the image, so any fp32 `torch.matmul`
  raises `HIPBLAS_STATUS_INVALID_VALUE`. The correctness reference cannot run without
  `TORCH_BLAS_PREFER_HIPBLASLT=0`, which the harness now sets by default.
- **Repo main does not import under the image's torch.** `low_precision.py` calls
  `register_opaque_type` on classes that torch 2.11 requires to carry `OpaqueBaseMeta`.
  All four candidate images ship the same torch with the strict check. The image's own
  newer checkout contains the fix (a `try: from torch._opaque_base import OpaqueBaseMeta`
  shim plus metaclasses on `Float8QuantConfig` / `Float4QuantConfig` / `ScalingRecipe`);
  it has been applied here. **This is a main-branch incompatibility, not something this
  work introduced**, and it is worth upstreaming.
- **`3rdparty/hipify_torch` must be initialised** before `build_ext` will run.
- **A script's directory, not the repo root, goes on `sys.path`.** The first harness runs
  silently measured the image's editable install at `/workspace/Primus-Turbo` rather than
  the checkout being edited. The harness now forces its own checkout onto `sys.path` and
  records `turbo_path` in every result row so a ledger line is self-describing. The
  baseline numbers in §1–2 are unaffected — they are the shipped kernel either way.

---

## 6. What this does to the target ladder

Unchanged in structure, but now with a measured champion:

| milestone | FA ms/step | vs flex | status |
|---|---:|---|---|
| shipped turbo:TRITON | 1898 | 1.89× worse | was a regression |
| **champion today (throttled)** | **1142** | 1.14× worse | measured |
| flex, today (throttled) | 1003 | — | the anchor |
| champion, if throttle lifts (÷1.65) | ~692 | — | inferred, not measured |
| JIRA 30k tps target | 192 | — | 128% of the throttled compute roof; unreachable |

The backward has to come down a further ~25% to reach flex parity at this clock.

---

## 7. Kernel edits attempted, measured, and reverted

A ranked candidate list came out of a five-way source analysis of the backward. The top
hypothesis was that `_bwd_kernel_dkdv` spills registers at the shipped `num_warps=4`, which
would have pointed all subsequent work at tile sizes and lane counts. **It is refuted.**

| num_warps | `.vgpr_count` | `.vgpr_spill_count` | `scratch_*` ops |
|---|---:|---:|---:|
| 1 | 1024 | **512** | 326 |
| 2 | 941 | 0 | 0 |
| 4 | 667 | 0 | 0 |
| 8 | 512 | 0 | 0 |

dkdv spills only at `num_warps=1`, which is exactly why that arm measures 47.8 ms. At 2/4/8
there is no spilling at all, so dkdv's cost is **per-program efficiency, not register
pressure**. (Read these from the `.amdgcn` metadata; this Triton version's cache JSON no
longer carries `n_spills`.)

That pointed at the softmax VALU stream, so three edits were made to `_bwd_kernel_dkdv` and
measured:

| edit | result |
|---|---|
| Skip the causal mask on non-diagonal blocks | **−9% (regression)**: bwd 31.5 → 34.4 ms |
| Tighten `lo` by one `BLOCK_M` block | neutral (within 1–2% noise) |
| Fold `log_p_scale` into the per-row term | neutral |

All three are correct (SQNR unchanged to 2 dp). **All three were reverted.** The mask-skip
condition is uniform across the workgroup so it compiles to a scalar branch, but the branch
costs more than the `[BLOCK_M, BLOCK_N]` compare-and-select it avoids — most likely by
breaking the software pipeline that `num_stages` depends on.

The `lo` bound genuinely is one block too loose (dkdv subtracts an extra `BLOCK_M` before
flooring where `_bwd_kernel_dq` uses a tight `cdiv`, so every `start_n >= 1` spends one
iteration on an all-`-inf` tile, ~1.5% of iterations). It is a real inconsistency between
the two kernels and worth fixing for hygiene, but it is **not** a performance lever.

**Net: the shipped change is config-only, with zero kernel risk.**

## 8. Three claims in `docs/gfx1250-attention-tuning.md` were wrong and are corrected

Written before hardware access, refuted by the source analysis:

1. **dkdv's `BLOCK_M` was listed as a free knob. It is not** — it is bound to the LSE ABI by
   `2 * start_m + tl.arange(0, 2 * BLOCK_M)` plus two `tl.gather`s. Setting it to 128 returns
   lse-rows concatenated with delta-rows: no fault, smoothly wrong dk/dv.
2. **"Price `sequence_parallel=False`" was dangerous advice.** It is not a slower-but-correct
   mode: it sets grid dim 1 to 1 while `program_id(1)` is the output tile index, so it
   computes 1/128 of the gradient and leaves the rest at the `zeros` init. **A timing-only
   sweep would have reported it as a spectacular win.**
3. **The XCD remap would hurt, not help.** `gridDim0` is already a multiple of 8 in both
   backward kernels, so all tiles sharing a (batch, head) already land on one XCD for free.
   Chunking would scatter them.

## 9. Open, and what to do next

- **The one dk/dv corruption is unexplained.** 1 occurrence in ~100 runs; 0 in a targeted
  75-rep study. The shipped default's dk is also non-deterministic run-to-run (3 distinct
  SQNR values in 25 reps) while out/dq/dv are bitwise stable. 75 reps is far below the ≥500
  the corpus says is needed to claim determinism. **Do not claim determinism; re-run the
  study at ≥500 reps before anything ships.**
- **Productizing the champion needs one prerequisite.** Making it the default means either
  an arch-gated default (these kernels are shared with the fp8 path, which runs on other
  arches) or shipping a two-entry autotune list. The latter is cleaner but **the autotune
  key lists must be fixed first**: they name `BLOCK_DMODEL`, which is not a parameter of
  either backward kernel, and omit HQ/HK and seqlen. Triton silently drops unknown key names
  and only computes the key at all when `len(configs) > 1`, so today the lists are inert —
  the moment a second config ships, they start mattering and they are wrong.
- **The backward still needs ~25% to reach flex parity** (31.5 vs 23.8 ms). The cheap
  scheduling knobs are exhausted; what remains is structural.

---

## 10. End-to-end: the attention win is neutral E2E, because E2E is bottlenecked elsewhere

**This node had never completed a single training step.** All three prior runs (run1, run3,
run_smoke, 2026-09-10) log **zero** step lines. So the JIRA's 19,795 tps did not come from
this machine, and there was no local baseline to regress against.

Getting a step out required fixing three things, none of them architectural:

1. The image is missing torchtitan's dependency set: `tyro`, `torchdata`, `datasets`,
   `tabulate`, `tokenizers`, `safetensors`, `einops`, `pillow`, `tensorboard`, `wandb`.
2. **`tensorboard` in particular explains the JIRA's `enable_gqa` error.** The patch runner
   reports a missing dependency as *"patch failed"* and continues:
   `[Patch] ✗ Patch 'torchtitan.primus_turbo.turbo_attention' failed due to missing
   dependency: No module named 'tensorboard'`. The patch that rewrites the call site never
   applies, so torchtitan's flex call site still passes `enable_gqa` into `TurboAttention`
   and raises `TypeError`. With tensorboard installed: `✓ Applied`, 12/12 patches.
   **This is a third root cause distinct from the JIRA's account**, alongside the missing
   aiter and CK being CDNA-only.
3. `_hfassets` was not mounted into the container.

### The measurement

| path | steady tps | tflops | mfu | peak memory |
|---|---:|---:|---:|---:|
| turbo:TRITON, tuned config | **245** | 14.17 | 4.54% | 379.50 GiB (87.85%) |
| flex (baseline, same config) | **244** | 14.12 | 4.53% | 378.00 GiB (87.50%) |

Both train correctly (loss descends). Steady from step 2; memory plateaus, so there is no
leak.

**The two paths are identical to 0.4%.** Two conclusions:

- **Enabling turbo attention is NOT an E2E regression** — it matches flex, while being
  1.66x faster in the kernel.
- **The E2E number here cannot validate attention work at all.** 245 tps is 133.7 s/step;
  the entire attention path is 35.7 ms x 32 = **1.14 s**, under 1% of the step. Something
  outside attention is ~100x larger than attention and dominates both paths equally.

Note the signature: **100% GPU utilisation at 14 TFLOPS** against a measured ~1000 TFLOP/s
roof. The GPU is busy doing something that is not useful matrix work. Both runs sit at
~88% of the 432 GiB HBM, which is the leading suspect (allocator pressure / eviction), and
is being tested by re-running with activation checkpointing on.

**What this does NOT license.** It does not license a claim that the kernel win is worth
1.66x of a training step — the attention path is too small a fraction here to show, and on
a healthy machine it would be ~45% of the step (the JIRA's 755 of 1655 ms). It licenses
exactly two claims: the kernel is 1.66x faster, and turning it on costs nothing E2E.

### A process-hygiene trap worth recording

The first attempt at the activation-checkpointing comparison died with a confusing OOM:

```
Tried to allocate 896.00 MiB. GPU has 432.00 GiB total, of which 446.00 MiB is free.
Of the allocated memory 50.56 GiB is allocated by PyTorch
```

PyTorch holding 50 GiB while only 446 MiB is free reads like a leak from a dead process.
It was not. `rocm-smi --showpids` named the holder: **the flex baseline run, still alive**,
holding 407,519,252,480 bytes = 379.5 GiB. The shell command that had been *waiting* for
four step lines exited once it had them; the training it was watching did not.

Two consequences:

- `rocm-smi --showpids` is the check that distinguishes "leaked" from "still running", and
  it is worth running before every E2E launch. `pkill` on the launcher is not enough --
  confirm VRAM returns to 0%.
- **The 245-vs-244 comparison is unaffected**, and the arithmetic shows why: flex peaked at
  378 GiB by itself. Had turbo's 379.5 GiB still been resident, the two would need 757 GiB
  on a 432 GiB card. So turbo had been released before flex started and each ran alone.

Note also that ~378 GiB is genuinely what this configuration costs: 8B parameters with Adam
plus full activations at mbs 4 / seq 8192 with `activation_checkpoint.mode: none`. It is not
an artifact, which is why the activation-checkpointing variant is the right next probe.

## 11. ROOT CAUSE of the E2E bottleneck: this image has no working BLAS for gfx1250

Activation checkpointing settled that memory pressure was not the cause:

| config | peak memory | tps |
|---|---:|---:|
| `activation_checkpoint: none` | 379.50 GiB (87.9%) | 245 |
| `activation_checkpoint: full` | **174.79 GiB (40.5%)** | **215** |

Halving memory made it marginally *slower* (AC trades memory for recompute, so that is the
expected sign). Memory was not the bottleneck.

The GEMMs are. Measured on an idle card (`rocm-smi` confirmed 0 KFD processes), bf16:

| GEMM | time | achieved |
|---|---:|---:|
| 8192 x 8192 x 8192 | 40.12 ms | **27.4 TFLOP/s** |
| qkv proj 32768 x 4096 x 6144 | 68.77 ms | 24.0 TFLOP/s |
| mlp up 32768 x 4096 x 14336 | 161.00 ms | 23.9 TFLOP/s |
| mlp down 32768 x 14336 x 4096 | 147.20 ms | 26.1 TFLOP/s |

**Put that beside two numbers measured on the same card in the same session:**

- the tuned attention kernel achieves **215 TFLOP/s**
- Phase 1's Triton GEMM established a **1002.7 TFLOP/s** bf16 roof (throttled)

A dense GEMM running **8x slower than a flash-attention kernel** on the same silicon is
prima facie broken — attention does strictly more work per FLOP (softmax, masking, online
rescaling) and should be the slower of the two. rocBLAS is roughly **37x off** the roof.

And there is no alternative: hipBLASLt's gfx1250 Tensile library is **absent from the
image** (`TensileLibrary_lazy_gfx1250.dat` does not exist), so torch either raises
`HIPBLAS_STATUS_INVALID_VALUE` or, with `TORCH_BLAS_PREFER_HIPBLASLT=0`, falls back to the
rocBLAS path measured above. **This image has no working BLAS for this architecture.**

### What this explains, and what it means for the JIRA

It explains every E2E observation at once: 14 TFLOPS overall, 133.7 s/step, 100% GPU
utilisation doing almost no useful matrix work, and flex and turbo landing within 0.4% of
each other — both are GEMM-bound by a factor that dwarfs attention.

It also means **the JIRA's GEMM figures (MI455X 655 ms/step, 1.51x faster than MI355X)
cannot have come from this image**, and neither can its 19,695 tps. Whatever environment
produced those numbers has a working BLAS; this one does not.

**Consequence for this project: E2E validation of attention work is impossible on this
image, for reasons that have nothing to do with attention.** It needs an image with a
functioning hipBLASLt/rocBLAS for gfx1250. That is a platform request, and it belongs in
the same escalation as the VR throttle.

## 12. Determinism: the kernel is bitwise deterministic; the earlier signal was my own yardstick

The study was initially built the wrong way: it re-derived the fp32 reference every rep and
compared SQNR, at ~30 s/rep, which put the >=500 reps the corpus asks for at four hours.
Determinism does not need a reference — it needs run-to-run **bitwise** equality. Comparing
each rep against rep 0 instead costs **0.47 s/rep**, a 64x speedup, and asks the sharper
question. Added as `--determinism-reps`.

**Result, production shape, bitwise:**

| config | reps | out | dq | dk | dv | non-finite |
|---|---:|---:|---:|---:|---:|---:|
| `fwd:num_stages=2; bwd:num_warps=2` | 1000 | 0 | 0 | 0 | 0 | 0 |
| shipped default | 1000 | 0 | 0 | 0 | 0 | 0 |

Zero mismatches on every tensor, both configs, well past the >=500 bar. Plus 150 fresh
processes x 2 reps (the original corruption appeared in a fresh process, so in-process reps
alone would not have caught it): **0 non-deterministic processes, 0 non-finite reps.** Study complete: **2,300 kernel invocations, zero failures of any kind.**

### The correction this forces

Earlier in this session I reported that the shipped default's dk was non-deterministic,
on the evidence of 3 distinct dk SQNR values across 25 reps. **That was wrong, and the
mechanism is worth recording because it is an easy trap:** each of those reps recomputed
the fp32 reference, and that reference is built from `torch.matmul`, which on this stack
goes to rocBLAS and is *itself* non-deterministic. The SQNR moved because the denominator
moved. The kernel output was bitwise identical the whole time.

**Rule: never measure determinism against a freshly-derived reference.** Compare the
subject to itself.

### What remains genuinely open

The single dk/dv = -inf event (1 occurrence in ~100 runs earlier in the session) is now
**un-reproduced across 2,300 further kernel invocations** (1000 + 1000 in-process, 300
cross-process over 150 fresh processes). It is not a property of any config tested, and it is not a determinism
problem in the kernel. Remaining candidates, in order: a transient hardware fault on a card
with a documented fault history (four wedges in eight days, and a page-fault rate of ~2-in-12
measured on another path), or something outside the kernel entirely.

**Assessment: this is no longer a ship blocker for the config change**, which touches no
kernel code at all -- it selects `num_stages` and `num_warps`. It should stay on the record
as an unexplained single event, and the harness keeps the four-tensor gate that caught it.

---

## 13. Round-0 bake-off: the in-tree backend loses to aiter's

Full write-up in `BAKEOFF.md`; the numbers and the consequences are folded in here.

One session, one image, identical fp32 reference, identical four-tensor SQNR gate,
identical timer. Production shape.

| impl | fwd ms | bwd ms | total ms | TFLOP/s | ms/step |
|---|---:|---:|---:|---:|---:|
| **aiter + 2 knobs** | **3.260** | 27.947 | **31.207** | **246.6** | 999 |
| torch flex (anchor) | 7.506 | **23.831** | 31.337 | 245.6 | 1003 |
| aiter shipped | 6.676 | 27.888 | 34.565 | 222.7 | 1106 |
| **turbo champion (§3)** | 4.152 | 32.252 | 36.405 | 211.4 | 1165 |
| turbo shipped | 10.673 | 48.969 | 59.642 | 129.0 | 1909 |

All four contenders passed the SQNR gate on out/dq/dk/dv. No fast-but-wrong entries.

**The in-tree turbo backend is currently slower than aiter's.** 36.405 ms vs 31.207 ms
(1.17x), and it also loses to aiter's *untouched shipped* config at 34.565 ms (1.05x).
Our forward is strong — 4.152 ms, 1.81x ahead of flex, though still 1.27x behind aiter's
3.260. **The whole deficit is the backward: 32.252 vs 27.888 ms, 1.16x.**

Run-to-run spread for calibration: the champion measured 36.405 here vs 35.704 in §3
(2.0%); flex measured 31.337 both times. The 1.17x gap to aiter is far outside noise.

### The sequencing was wrong, and it cost time

§8/Phase 1 of the plan named Round 0 as the *first* step and as the highest-value probe.
It was not run first — turbo tuning started immediately instead. **That was a mistake.**
The two knobs would have been found either way (they transfer), but the rounds spent on
turbo's scheduling knobs — `waves_per_eu`, the mask-skip edit, the `lo`-bound edit, the
`log_p_scale` fold, the register-spill investigation, all of §7 — were spent on the wrong
axis. Every one of them landed neutral or worse. **Never tune an implementation before
establishing it is the right implementation to tune.**

### The structural reason: asymmetric block shapes, blocked by an ABI

| | turbo backward | aiter backward |
|---|---|---|
| dkdv tile | `FIXED_BLOCK_M = FIXED_BLOCK_N = 64` | `BLOCK_M1 = 32`, `BLOCK_N1 = 128` |
| dq tile | the same two constants | `BLOCK_M2 = 128`, `BLOCK_N2 = 32` |
| shape | symmetric, hardcoded module constants | **asymmetric**, paired `N1 == M2`, `M1 == N2` |
| other | — | `waves_per_eu = 1`, `BLK_SLICE_FACTOR = 2` |

aiter's two halves are transposes of each other, so each gets a long inner axis where it
needs one. turbo has one symmetric 64x64 tile for both halves and serves neither well.

turbo cannot currently express this, for the reason already recorded in §8.1: **dkdv's
`BLOCK_M` is bound to a cross-kernel LSE/delta ABI.** `attn_fwd` writes LSE at
`m * BLOCK_M * 2`; `_bwd_preprocess_use_o` writes delta at `+ BLOCK_M` of the same block;
host-side `_lse_delta_views` reconstructs it; `_bwd_kernel_dkdv` reads
`2 * start_m + tl.arange(0, 2 * BLOCK_M)` and splits it with two `tl.gather`s. Changing
the constant alone does not fault — it returns lse rows concatenated with delta rows and
yields smoothly wrong dk/dv. **Decoupling that ABI is the work in progress**, and it is the
prerequisite for anything aiter-shaped.

Arithmetic only, not measured: turbo's forward (4.152) plus aiter's backward (27.888)
would be **32.04 ms / 240.2 TFLOP/s**, within ~3% of both anchors. That is the honest size
of the prize — **parity, not a win.**

### CORRECTION to §6 and to the plan: 316.1 TFLOP/s did not reproduce

The archived op-evolve figure for aiter + 2 knobs, **24.347 ms / 316.1 TFLOP/s**, has been
carried through the plan as aiter's established performance. Measured here: **31.207 ms /
246.6 TFLOP/s.**

| | then | now |
|---|---:|---:|
| fwd | ~3.272 | 3.260 (reproduces) |
| bwd | **21.075** | **27.947** |
| total | 24.347 | 31.207 |

The forward reproduces exactly; the entire 6.86 ms delta is the backward.

**Mechanism: upstream aiter retuned its shipped backward.** The old number came from a
different aiter revision. Evidence: aiter's *forward* `num_stages` 1 -> 2 still reproduces
exactly (6.676 -> 3.260, 2.05x), while aiter's *backward* `num_warps` 4 -> 2 **no longer
helps at all** on current aiter main (27.888 -> 27.947, noise). That knob was a real win on
the older revision; upstream has since moved the backward to a config where it is already
spent.

**Retire 24.347 / 316.1 from every target and comparison.** Any ladder entry derived from
it needs recomputing.

### What changes, and what does not

Changes:

- **The champion config still ships.** 1.64x over the shipped default, config-only, zero
  kernel risk, bitwise deterministic (§12), E2E-neutral (§10). It is simply not the finish
  line.
- **Stop spending rounds on turbo scheduling knobs.** Exhausted, per §7.
- **Next work item is the LSE/delta ABI decoupling.** Everything else in the backward is
  downstream of it.
- **Re-anchor the ladder**: the near-term realistic backward target is aiter's 27.9 ms, not
  flex's 23.8 ms. flex still owns the fastest backward on this card, 1.17x ahead of aiter,
  and nothing yet explains why.
- **"Dispatch to aiter's Triton backend on gfx1250" is now a legitimate option** and should
  be costed against continuing. The in-tree backend's real value was always "a gfx1250 path
  that is correct in both directions and that we can edit". That survives losing the
  bake-off; it is just not a performance argument.

Does not change:

- **The E2E result still means only what §10 said it means**: turbo 245 tps vs flex 244 tps
  shows that enabling turbo attention **is not a regression**, and nothing more. The whole
  attention path is 36.4 ms x 32 = 1.16 s of a 133.7 s step — **under 1%**. The step is
  GEMM-bound (§11): rocBLAS bf16 at 27.4 TFLOP/s against a 1002.7 roof, with hipBLASLt's
  gfx1250 Tensile library absent. **E2E on this image could not have detected that turbo
  lost this bake-off**, in either direction.
- **The throttle still dominates.** Every number is at 1100 MHz vs 1699-1703 on 2026-09-04.
- §8's refutations stand: `sequence_parallel=False` is silently wrong, the XCD remap would
  hurt, FlyDSL is unavailable and rejects G=4, AITER's CK fmha is CDNA-only.
