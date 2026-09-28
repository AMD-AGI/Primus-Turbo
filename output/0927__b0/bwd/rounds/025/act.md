> **Carried forward** from `rounds/025/3-act.stale-20260927T153639/` after the framework
> restarted the act step at 2026-09-28T01:51Z. Every measurement in this file was taken on
> **physical GPU 1** via container `fa-g1`, which no longer exists. The runner now points at
> `fa-g3` = **physical GPU 3**.


---

## Step 4 — report, and how the round ended

`act.yaml` written from what was already on disk; no new measurement was taken for it.

### Final validation

Run through the runner against `rounds/025/op` explicitly — not the default, which is
`op/current/` and is round 20's code until this round is accepted. Left on the default it
would have graded the incumbent and printed the number as this round's, with nothing in the
output to say which one it had measured.

| gate | result |
|---|---|
| correctness | pass |
| determinism | pass |
| speed | **FAIL**, geomean 0.943× beat |
| exit code | **2** |

Exit 2 is the speed gate specifically, not a correctness failure. An earlier independent run
of the same binary gave 0.946× — a 0.3% spread, consistent with the session's 0.32% floor.

### The container fleet was destroyed 2 s after the run finished

The post-validation idle check **could not be taken**, and this is recorded rather than
silently omitted. `docker exec fa-g1` returned `No such container`. Investigated instead of
assumed:

- `docker events` shows `destroy` for `fa-g0`, `fa-g1`, `fa-g2`, `fa-g3` all at epoch
  `1790523843` = `2026-09-27T15:44:03Z`.
- My validation `rc` sentinel was written at `15:44:01Z`. **The run completed two seconds
  before the teardown**, so the result is intact — it is not a partial run reported as a
  whole one.
- `dmesg` shows only veth teardown: no GPU reset, no ring timeout, no MES hang. This was a
  deliberate external fleet operation, not a card wedge.

A clean idle check before the run was taken and is in `provenance.yaml`; the after-check is
absent for the reason above.

### The runner has moved to a different physical GPU

The fleet came back as `fa-g0`, `fa-g2`, `fa-g3`; **`fa-g1` does not exist any more**. The
round's runner line now names `container=fa-g3`, and `docker inspect fa-g3` gives
`/dev/dri/renderD152` = **physical GPU 3**.

Every number in this file was measured on **physical GPU 1** via `fa-g1`. Nothing here needs
re-measuring — the round is closed and internally consistent, all arms measured against each
other in one process on one card — but **no new measurement may be appended to these tables
from `fa-g3`**. It is a different card on a host the campaign has recorded as degraded and
per-card in its clock behaviour. A number from GPU 3 placed in a GPU 1 table is exactly the
silent machine-change the standing rules warn about.

---

# Step 2 — RE-MEASURE on physical GPU 3

The framework rotated `3-act` aside at 01:51 and re-issued the round's steps against a new
runner. This is not a formality: `job_context/gfx1250-flydsl-attn-bwd_final.yaml:247` now
reads `container: fa-g3` with the comment *"B0 2026-09-28: per-card container (only
renderD152 = physical GPU 3); day 1 was fa-g1"*. `docker inspect fa-g3` confirms
`/dev/dri/renderD152`.

**Every number from the first attempt was taken on physical GPU 1 via `fa-g1`, which no
longer exists.** The standing rule that stored numbers are not comparisons — "clocks, image
and machine state all move between them" — is exactly the condition here, and a card swap is
the strongest form of it. So step 2 is re-run in full on GPU 3: this round, the incumbent,
and every distinct champion. The GPU-1 tables are kept in
`3-act.stale-20260927T153639/` and are not merged with these.

## Device state before the run

- `fa-g3` up 10 min, image `fa-tune:b0-snap`, devices `/dev/kfd /dev/dri/renderD152`.
- Physical GPU 3: `rocm-smi --showuse` = 0%, no python alive in `fa-g3`.
- **A foreign process is on the board but not on my card.** PID 2036864,
  `python3 benchmark.py --arms current,beat ... r009b-scratch`, cgroup
  `92bb52de5a27…` = **`fa-g2` → renderD144 → physical GPU 2**. Same hazard class as the
  `fa-g0` tenant in the first attempt: it cannot bias a within-shape ratio, but this board
  couples power and clock, so it can depress absolute TF/s. Waiting it out rather than
  recording it as a caveat this time.
- `/tmp/flycache` deleted in-container and every `rounds/*/op/__pycache__` removed, so all
  five arms are from-scratch compiles. A stale compiled kernel would have this round measure
  a previous round's binary with no error anywhere.

## Form

**Two passes, back to back, five arms each, all three shapes** — `step2_g3` (rc 0) and
`step2_g3_rep` (rc 0), both launched detached with a sentinel through
`job_context/op/runner_util.py` → `fa-g3` → physical GPU 3. Every reference is re-measured
now; nothing is read from a previous round's record, and nothing is read from the GPU-1
attempt.

Arms: `r25` = `rounds/025/op` (this round), `r20` = `rounds/020/op` (incumbent,
`state.yaml: best_round: 20`), `r19` = `champions.prod`, `r17` = `champions.fast` and
`champions.proxy`, `beat` = `job_context/op/beat` (aiter ASM; `BEAT_MARGIN = 0.0` makes it
the target). **All four reference arms rebuilt and ran.** No shape had to be guarded against
the incumbent alone.

`benchmark.py:132-134` interleaves the arms per iteration with alternating order, so all five
arms of a shape share one clock excursion — visible as identical `sclk_start`/`sclk_end`
columns within each shape.

## Raw, both passes, and the noise floor for this card

| shape | arm | pass 1 | pass 2 | spread |
|---|---|---|---|---|
| fast | r25 | 97.351 | 97.703 | 0.36% |
| fast | r20 | 92.009 | 92.579 | 0.62% |
| fast | r19 | 97.280 | 97.703 | 0.43% |
| fast | r17 | 85.882 | 87.735 | 2.16% |
| fast | beat | 50.018 | 50.679 | 1.32% |
| proxy | r25 | 516.381 | 515.018 | 0.26% |
| proxy | r20 | 575.783 | 572.097 | 0.64% |
| proxy | r19 | 593.714 | 592.238 | 0.25% |
| proxy | r17 | 590.811 | 588.663 | 0.36% |
| proxy | beat | 754.544 | 752.161 | 0.32% |
| prod | r25 | 614.657 | 616.604 | 0.32% |
| prod | r20 | 628.025 | 628.796 | 0.12% |
| prod | r19 | 636.870 | 638.265 | 0.22% |
| prod | r17 | 634.159 | 634.678 | 0.08% |
| prod | beat | 827.185 | 828.483 | 0.16% |

**Prod noise floor on GPU 3: 0.08% to 0.32%**, which reproduces the 0.32% established on
GPU 1 — so the floor is a property of the measurement, not of the card. `fast` is the noisy
shape (up to 2.16%) because the kernel is 55 µs and the clock is at 2.33 GHz boost; nothing
at `fast` should be read to better than about 2%.

The per-pass score is 0.8091 and 0.8097. The reported table is the mean of the two passes.

**This card is about 2% slower than GPU 1 across every arm** (prod `beat` 827.8 here against
847.2 there; sclk 1758 against 1793). Every arm moves together, so the ratios are stable to
about 0.1% — which is exactly why the tables must be swapped wholesale rather than merged.
Had the GPU-1 `beat` number been kept and this round's GPU-3 number placed beside it, the
round would have scored roughly 2% low for a reason that has nothing to do with the code.

## Result

| shape | this round | incumbent r20 | target (beat) | champion now | ratio | vs_champion |
|---|---|---|---|---|---|---|
| fast | 97.527 | 92.294 | 50.348 | r17 86.808 | 1.0000 (raw 1.937) | **1.1235** |
| proxy | 515.700 | 573.940 | 753.353 | r17 589.737 | 0.6845 | **0.8745** |
| prod | 615.630 | 628.411 | 827.834 | r19 637.567 | 0.7437 | **0.9656** |

`score = 0.8094`.

## Champion bookkeeping

`state.yaml.champions` still carries four keys for three spec shapes: `fast: 17`,
`proxy: 17`, `prod: 19`, `prod_b4_s8192_hq32_hkv8_d128: 20`. The fourth names exactly the
production geometry and points at a different round from `prod`. Both were measured rather
than choosing between them. The table uses r19, which is the stricter reading
(0.9656); against the `prod_b4_…` key, `vs_champion` is 0.9797. Both are inside the 5% band,
so the guard verdict is the same either way. Flagged again for reconciliation; not resolved
here.

## Regressions

- **proxy, 0.8745 — outside the band.** A 12.55% loss against its champion, 10.1% against
  the incumbent. This is the shape that fails the guard.
- **prod, 0.9656 — inside the band, and this is the one that matters.** 3.44% below the
  champion, which is **eleven times this card's prod noise floor**. Campaign correction 4
  applies exactly here: `fast` gained 12.35% and the arithmetic mean would happily let that
  pay for the prod loss, and prod is what this job exists to close.
- **fast, 1.1235 — a gain**, listed because the instruction is to report every regression
  and the completeness of the set is the point, not its sign.

## Idle

Before: physical GPU 3 at 0% use, no python alive in `fa-g3`; the one foreign process on the
board (`fa-g2`) had exited before pass 1 started. After: 0% use, no python in `fa-g3`. Two
KFD processes were listed at the end — PID 2094117 in `fa-g0` (renderD128, GPU 0) and PID
2094685 in `fa-g2` (renderD144, GPU 2), both forward-attention jobs, established by cgroup
id against `docker inspect`. **Neither is on my card**, but both were alive during pass 2,
so the pass-1-to-pass-2 spread above is an upper bound on noise that conflates true
measurement noise with board-level power coupling. It is 0.32% at prod even so.

No `dmesg` event fired during either pass: no reset, no ring timeout, no MES hang.

---

# Step 3 — did the predicted thing happen? (re-measured on GPU 3)

## What could not be measured, said before anything that could

`plan.yaml`'s mechanism is an L2 residency claim: the reversed scan was meant to phase-align
the dispatch cohort so its Q/dO working set collapses into the 4 MiB device-wide L2. **There
is no counter on this part that can see that.** Round-25 profiling's `non_findings` settled
it: TCC, TCP, TA and TD are gone, and the replacement CHC/GL1C/CHA/GLARBC blocks are
*accepted* and read exactly zero. So the memory-side half of the hypothesis is unmeasurable
by anyone this round. It is not refuted below; it is unobserved, and nothing here is claimed
about L2.

What *can* be measured is whether the loss is localised in `k_dkdv` and whether it is extra
**issued work**. `SQC_ICACHE_REQ` is the one deterministic dynamic-work proxy this card
gives — good as a build-to-build gate, never as a time predictor (correction 3).

`measurements/kdkdv_locus_g3/`, `rocprofv3 --pmc SQC_ICACHE_REQ,SQC_ICACHE_MISSES,
SQC_ICACHE_MISSES_DUPLICATE`, one hardware block, prod, 2 warmup + 5 measured, arms
`op_base` (r24 + the h33 fixes, so the h33 fixes are measured *out* of the arm) and
`rounds/025/op`. Both CSVs checked present and non-empty first — a `--pmc` list naming an
undefined counter writes no CSV and still exits 0.

## The result, and it reproduces bit-for-bit across two cards

| kernel | base | g77 | Δ | spread |
|---|---|---|---|---|
| `k_delta_bshd_0` | 28,672 | 28,672 | **0.0000%** | 0 |
| `k_dkdv_0` | 275,152,320 | 317,021,440 | **+15.2167%** | 0 |
| `k_dq_0` | 141,914,496 | 141,914,496 | **0.0000%** | 0 |

Every one of these six numbers is **identical to the GPU-1 measurement**, digit for digit.
An instruction-fetch counter should be architectural rather than a property of a device, and
this is the direct confirmation: the +15.2167% is a fact about the build, not about a card.

The two kernels g77 does not touch are a perfect control — bit-identical between arms, on
both cards. The perturbation is localised in `k_dkdv` and nowhere else.

Profiled durations (locus only; **not comparable to step 2** — dispatches serialise and the
clock rises): `k_dkdv` +8.49%, carrying 139% of the total profiled prod delta while being
58% of base prod kernel time. `k_delta_bshd` moved −13.5% and `k_dq` −3.4% despite
byte-identical code, which is exactly why attribution uses the zero-spread counter column
and not the duration column.

## The verdict on each predicted row

| predicted | outcome | measured |
|---|---|---|
| `prod_vs_champion_same_session ≥ 1.030` | **failed** | 0.9656 |
| `qmask_probe_delta_prod ≥ 5%` (PRIMARY falsifier) | **failed** | 1.83% raw, ≈0 clock-normalised |
| `k_dkdv_body_isa` identical but for bound/prologue/sign | **failed** | +19 and +16 instrs, 6 → 7 blocks, `s_set_vgpr_msb` 393 → 495 |
| `last_prefetch_load_index_in_body` unchanged | **failed** | 189 → 192 |
| `prod_B1 ≥ 1.5×` the end-to-end delta | **sign inverted** | B1 −5.57%, B2 −3.01%, B4 −1.80%: magnitude 3.1×, but worst where alignment is cleanest |
| `fast` inside floor (h5) | **failed** | +12.35% vs champion |
| `spill: 0` | **held** | 0 |
| `wmma_per_body: 64` | **held** | 64 |

## The reading

The regression is **issue/fetch-side, not the predicted memory-side effect**. The memory
instruction mix did not change at all — 160 `buffer_load_b128`, 176 loads, 128 `wmma` in both
arms. All static growth is 102 extra `s_set_vgpr_msb` bank prefixes, +19/+16 instructions,
and one extra block. Hot-body bytes grew only 2.64% (FULL) and 3.58% (MASK), so **code size
does not explain +15.22%**. A trip-mix model of the residual required a shift that the
verified-identical coverage rules out, so the microscopic cause is recorded as **not
established** — only its locus and its determinism are.

**NOT claimed:** that the +15.22% fetch growth *caused* the duration growth. Correction 3
forbids that conversion and it is not made here.

The consequence for round 26 is the round's actual value: **g77 does not test the L2
phase-alignment hypothesis.** It carries a 15.22% fetch perturbation the hypothesis never
called for, so its loss is confounded and refutes nothing on that axis. The only clean
evidence bounding that family remains **P77** — 1.83% raw, ≈0 clock-normalised.

---

# Step 4 — report, and how the round ends

`act.yaml` and this file are written from what is on disk. No new measurement was taken for
them; the one command run in this step is `validation.py`.

## Final validation

Run through the runner against **`rounds/025/op` explicitly** (`.runs/step4_validation_g3`),
on physical GPU 3. Left on its default it tests `op/current/`, which is round 20's code until
this round is accepted — it would have graded the incumbent and printed the number as this
round's, with nothing in the output to say which one it measured.

| shape | candidate | beat | × beat |
|---|---|---|---|
| fast | 90.1 TF/s | 53.2 | 1.693 |
| proxy | 525.6 | 775.3 | 0.678 |
| prod | 614.2 | 840.7 | 0.731 |

geomean over 3 shapes: **0.943× beat**

| gate | result |
|---|---|
| correctness | **pass** — prod dq 52.56 dB, dk 52.60, dv 52.71 against a 50 dB gate |
| determinism | **pass** — dk/dv bitwise identical across 200 runs |
| speed | **FAIL** |
| exit code | **2** |

Exit 2 is the speed gate specifically, not 0 or 1. Three independent full validations now
agree — 0.946 and 0.943 on GPU 1, 0.943 here on GPU 3 — so the verdict is stable across both
cards and is not an artefact of the move. The prod candidate figure (614.2) also sits inside
step 2's two passes (614.66, 616.60), which is the cross-check that the two instruments are
measuring the same binary.

## Disposition

The losing implementation **stays in `rounds/025/op`**. Reverting it would destroy the only
copy of what was learned and make the round hash-identical to one that built nothing —
same empty `files_changed`, same gain of exactly 1.0000. Correction 4 independently forbids
promoting it: prod `vs_champion` is 0.9656, eleven times this card's prod noise floor.

Acceptance is Python's arithmetic from step 2's numbers, not a judgement made here.

## What this round is worth

Not the code, which loses. Three things:

1. **A pure index-direction change is not ISA-neutral in this backend.** Mirroring an
   induction variable with net-zero arithmetic and no added memory operations still moved
   dynamic instruction fetch by 15.22% — bit-identically on two different cards. Any future
   candidate arguing "this is just a reindex, the code is the same" is making a claim that
   has now been measured false, and `SQC_ICACHE_REQ` is the one-pass gate that catches it.
2. **g77 does not test its own hypothesis.** The confound is larger than the effect sought,
   so the L2 phase-alignment family is neither confirmed nor refuted by this round. The only
   clean bound on it remains P77: 1.83% raw, ≈0 clock-normalised.
3. **Pool rule 1 survives an eighth test, by accident** — through the register allocator
   rather than through the source, on an arm chosen partly because it was believed to sit
   off that axis.

Round 26 should not read this as "reversed scans are dead." It should read it as "the
backend's register allocator is a confounder you must gate for before you can test anything
about loop order at all."
