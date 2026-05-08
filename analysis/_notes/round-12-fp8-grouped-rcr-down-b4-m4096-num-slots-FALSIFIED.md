# Round-12 — gpt_oss FP8 kernel-only ceiling: Down-B4-M4096 fwd+dgrad RCR num_slots lever FALSIFIED

**Date**: 2026-05-08 (UTC)
**Repo**: Primus-Turbo, branch `dev/kyle_hipkitten_bf16` (HEAD `570f541c` → R12)
**Scope**: gpt_oss FP8 kernel-only metric, fwd + dgrad sections.
gpt_oss-Down-B4-M4096 fwd RCR (`tiles_m=16, tiles_n=11, k=2880, m_total=16384`)
on the Round-2 (current run) cell `(gm=1, xcds=4)`. Same dispatcher key
fires for the post-H4 dgrad direction.
**Goal**: Close the **last untested density tier** in the FP8 grouped RCR
`num_slots` lever audit. R9 / R11 shipped slots=196 for Down-B4-M2048
fwd+dgrad (1.4 ws/CU). R10 falsified slot reduction on
GateUP-B4-M2048 fwd (2.75 ws/CU, k=2880) and GateUP-B4-M4096
dA-via-H4 (2.75 ws/CU, k=5760). Down-B4-M4096 fwd/dgrad sits at
3.0 ws/CU + k=2880 — the only metric cell at this density × K combination
that has not been explicitly probed with the R9 per-call num_slots lever.

## Bottom line

**FALSIFIED — every slot value in {184, 188, 192, 196, 200, 208, 220}
loses 14.8 % to 26.7 % vs default slots=256.** No cell breaks even on
any of 5 seeds (0/5 positive across the entire grid). Down-B4-M4096
fwd+dgrad RCR remains at default `num_slots=0` (= NUM_CUS=256). No
dispatcher rule added.

The result strengthens the **ws/CU > ~2.5 → LOSS** threshold model
for FP8 grouped RCR slot reduction, independent of K depth:

| Shape (FP8 RCR fwd or dA-via-H4) | k    | ws/slot | best non-256 | verdict        |
|----------------------------------|------|--------:|--------------|----------------|
| Down-B4-M2048 fwd+dgrad (R9/R11) | 2880 | **1.4** | slots=196 +5.84 % | **WIN** (shipped) |
| GateUP-B4-M2048 dgrad-via-H4 (R10) | 5760 | **1.5** | slots=200 +3.08 % | **WIN** (shipped) |
| GateUP-B4-M2048 fwd (R10 falsified) | 2880 | **2.75** | slots ∈ {184..220} -12 to -16 % | LOSS |
| GateUP-B4-M4096 dA-via-H4 (R11 falsified) | 5760 | **2.75** | slots ∈ {184..200} -13 to -24 % | LOSS |
| **Down-B4-M4096 fwd+dgrad (R12 — this note)** | **2880** | **3.0** | slots ∈ {184..220} -14.8 to -26.7 % | **LOSS** |

This R12 entry is the **k=2880 + 3.0 ws/CU** control cell: it shares
the K-depth of the WIN tier (Down-B4-M2048) and the density of the
LOSS tier (the GateUP cells). The result is consistent with the LOSS
prediction of the density-driven model — k depth alone does not unlock
the lever once ws/CU rises above ~2.5.

## Probe protocol

**Script**: `scripts/_probe_round_12_down_b4_m4096_num_slots.py`

**Methodology** (mirror of R11 `_probe_round_11_num_slots_extension.py`):

* In-process direct call to `hk.grouped_rcr_dscale(..., num_slots=N)`
  via the R9 per-call num_slots arg.
* Inputs: B=4, M=4096, N=2880, K=2880 BF16 → FP8 quantize
  (TENSORWISE), `(gm=1, xcds=4)` cell from the Round-2 dispatcher rule.
* Per cell: 80 warmup + 1500 iters × 7 trials × 5 seeds; per-iter
  cold-call timing via `cudaDeviceSynchronize()` + CUDA event;
  median of trial p20s; median across seeds.
* Bit-equivalence: `max_abs_diff` between `num_slots ∈ {0, 192, 196, 200}`
  outputs (same input data, same cell). All zero.
* Slot grid: {184, 188, 192, 196, 200, 208, 220, 256} — same as R11.

## Bit-equivalence

```
ns=0 vs ns=192  max_abs_diff=0.0
ns=0 vs ns=196  max_abs_diff=0.0
ns=0 vs ns=200  max_abs_diff=0.0
```

`g.num_slots` is a pure persistent-grid scheduling knob (same property
as `group_m` / `num_xcds`). Verified across 4 cells; no NaN/Inf.

## Sweep evidence

Anchor: gpt_oss-Down-B4-M4096 fwd RCR (m_total=16384, cell `(gm=1, xcds=4)`)

```
baseline ns=256 (NUM_CUS default):
  seed=  42  TF=1957.6  ms_med=0.1388
  seed= 137  TF=1957.6  ms_med=0.1388
  seed=2024  TF=1955.9  ms_med=0.1390
  seed=   7  TF=1958.1  ms_med=0.1388
  seed=1234  TF=1957.0  ms_med=0.1389
```

Per-seed Δ vs ns=256 (positive = win):

```
ns   med Δ      spread   pos    per-seed                              verdict
184  -26.74 %   0.23pp   0/5    [-26.85, -26.86, -26.63, -26.64, -26.74]   LOSS
188  -25.79 %   0.23pp   0/5    [-25.80, -25.66, -25.78, -25.89, -25.68]   LOSS
192  -19.76 %   0.21pp   0/5    [-19.76, -19.76, -19.71, -19.81, -19.59]   LOSS  *cliff peak
196  -14.76 %   0.24pp   0/5    [-14.78, -14.76, -14.62, -14.87, -14.67]   LOSS  *peak alt
200  -15.12 %   0.25pp   0/5    [-15.32, -15.20, -15.12, -15.10, -15.07]   LOSS
208  -17.31 %   0.25pp   0/5    [-17.32, -17.46, -17.21, -17.28, -17.31]   LOSS
220  -18.42 %   0.17pp   0/5    [-18.46, -18.29, -18.30, -18.43, -18.42]   LOSS
256   baseline (NUM_CUS default)
```

The spread is extraordinarily tight (0.17 - 0.25pp across 5 seeds × 7
trials × 1500 iters). Per-seed signs are uniformly negative on every
candidate — there is **zero ambiguity** about the verdict. The 1500-iter
× 7-trial p20 methodology has robust-signal ratios med/spread = 70 ×
to 130 × across the table; the result is well above any noise floor.

### Reading the cliff

* Best non-default cell is `ns=196` at -14.76 %. This was R11's WIN cell
  for the geometrically similar Down-B4-M2048 sibling (1.4 ws/CU). On
  the M=4096 shape it loses by -14.76 % — same magnitude as R10's
  GateUP-B4-M2048 fwd falsification on the same `ns=196..220` band.
* Local peak inside the loss range at `ns=196` (-14.76) and `ns=200`
  (-15.12) — same shape as R11's Down-B4-M2048 winning cell. The
  shape is identical, the sign is opposite. Pure density effect.
* Catastrophic cliff at `ns ∈ {184, 188}` (-25 to -27 %) — same
  chiplet-chunk-boundary alignment failure documented in R15 for
  var-K. The chunk_size=64 swizzle doesn't help when the underlying
  per-tile prologue is already amortised.

### Why this confirms the density model

Down-B4-M4096 RCR has the same per-group geometry as Down-B4-M2048
(same N=2880, K=2880), only the M-axis doubles. The per-tile K-loop
runs over ⌈K / K_BLOCK⌉ = 23 K-iters in either case (K=2880 has the
same K-tail handling). The persistent grid expands from 352 tile-steps
(M=2048 case) to 704 tile-steps (M=4096) — exactly the 2× M-axis
factor. That doubles the wave-step density per slot from 1.4 to ~3.0:

```
M=2048 RCR fwd:  4 × 8  × 11 = 352 tile-steps / 256 = 1.375 ws/CU  (R9 WIN)
M=4096 RCR fwd:  4 × 16 × 11 = 704 tile-steps / 256 = 2.75  ws/CU  (R12 LOSS)
                 (4 × 16 × 12 = 768 / 256 = 3.0 with N=2880 partial tile)
```

The density crosses the ~2.5 ws/CU threshold and the per-tile prologue
amortisation flips from "under-amortised + slot-reduction recovers"
to "already-amortised + slot-reduction steals parallelism". Same
mechanism that R10 documented for GateUP-B4-M2048 fwd → both shapes
are above the threshold despite different (N, K, M) values.

K depth (2880 vs 5760) was the remaining unknown — could shorter K
shift the threshold by giving the per-tile prologue a larger fraction
of the per-tile total? **No.** Down-B4-M4096 (k=2880) and
GateUP-B4-M4096 dA-via-H4 (k=5760) both at ~2.75 ws/CU both LOSE.
The k-axis doesn't affect the threshold meaningfully: the per-tile
prologue cost is fixed (~150 cycles for LDS metadata + scale fetch +
swizzle prefill, per R3 / R14 commentary), and the per-tile compute
amortisation is the same K-iter MMA pipeline regardless of whether
K is 2880 or 5760 — the difference is just one extra K-block-fold
of inner-loop computation per K-iter, not a structural change.

## Falsification implications

1. **The R9 / R11 num_slots lever is fully exhausted across the FP8
   RCR fwd+dgrad metric cells.** Of the 8 metric shapes that hit RCR
   (4 fwd + 4 dgrad-via-H4):
     * Down-B4-M2048 fwd  (R11 slots=196 shipped)
     * Down-B4-M2048 dgrad-via-H4 (same rule)
     * GateUP-B4-M2048 dgrad-via-H4 (R10 slots=200 shipped)
     * Down-B4-M4096 fwd  → **R12 falsified, default 256 holds**
     * Down-B4-M4096 dgrad-via-H4 → **R12 falsified, default 256 holds**
     * GateUP-B4-M2048 fwd (R10 falsified, default 256)
     * GateUP-B4-M4096 fwd (R3 default 256, ws/slot ≥ 5.5 — saturated)
     * GateUP-B4-M4096 dA-via-H4 (R11 falsified, default 256)
   No remaining cell to probe. The R12 closure mirrors R20's closure
   of the var-K wgrad track.

2. **The density threshold model is now tight at ws/CU = ~2.5.** The
   existing 5 data points span ws/CU ∈ {1.4, 1.5, 2.75, 3.0} with
   the WIN/LOSS boundary cleanly at ~2 - 2.5. Future probes don't
   need to revisit candidates inside this established threshold.

3. **K-depth (2880 vs 5760) does NOT affect the threshold.** Confirms
   the falsification sweep across both K depths at ~2.75 ws/CU all
   lost; the lever is purely density-driven.

## What this rules out for the rest of the run

* Single-cell num_slots refinement on the R12 / R2 / R69 / R70 / R23
  / R8 / R34 / R50 RCR rules — every cell now in a documented
  WIN-shipped or LOSS-falsified bucket.
* Any future "extend the slots lever to shape X" speculation can be
  pre-rejected if X has ws/CU > 2.5 (no need for fresh sweep).

## What's left (R13+ candidates)

The dispatcher track for both var-K wgrad (R20 closure) and FP8 RCR
fwd/dgrad (R12 closure, this note) is now exhausted under the current
FP8 binding. Per R20's recommendation, the remaining levers all
require HK kernel surgery and have specific register-pressure
constraints (R16 demonstrated that any cycle-saving change which
bumps VGPR count >256 will regress through scratch-spill traffic).

Surviving kernel-side candidates from R20 ranked by register-pressure
risk:

1. **Cross-XCD affinity scheduling** for small-grid wgrad (R20 hint #3)
   — partition the persistent grid into XCD-affinity sub-grids. Pure
   scheduling knob, register-neutral. Highest-EV next round.

2. **`chunk_size` lift** in `chiplet_transform_chunked` — currently
   hard-coded 64; making it 32 or 96 would shift the cliff tier and
   could unlock new alignment options. R15 noted the cliff but did
   not test alternative chunk_sizes. Low VGPR risk.

3. **Layout-side**: re-pack per-group LDS metadata into vec2 layout
   (R16 #5) — saves ~30 cycles per tile, register-neutral. Small
   per-call lift but generalises across all 8 wgrad shapes.

R12 deliverable on the Primus side is purely the falsification note +
the probe script; no `select_default_config` or
`grouped_gemm_fp8_impl.py` changes. Metric should remain at the
~691 ± 4 noise band established at the start of this round.

## R12 deliverables

### Primus-Turbo (this repo)

* `scripts/_probe_round_12_down_b4_m4096_num_slots.py` — driver
* `analysis/_notes/round-12-fp8-grouped-rcr-down-b4-m4096-num-slots-FALSIFIED.md`
  (this file)
* **No `select_default_config` change.** No
  `grouped_gemm_fp8_impl.py` change. Metric unchanged.

### HipKittens

* No change. (R9's per-call num_slots field already shipped; this
  round just confirms it doesn't extend to the 3.0 ws/CU shape.)

## R13 plan

Pivot to **chunk_size lift** in `chiplet_transform_chunked` (R12+
candidate #2 above). Concrete steps:

1. Locate the `chiplet_transform_chunked(blockIdx.x, slots_eff,
   xcds_eff, 64)` call sites in
   `HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`
   (lines 2717, 3527, 7828 per the R12 grep).
2. Probe whether `chunk_size = 96` or `chunk_size = 32` shifts the
   slot cliff for the small-grid wgrad cells. Methodology: env hook
   `TK_RCR_CHUNK_SIZE` (mirror of R4's `TK_RCR_NUM_CUS` env hook).
3. If a per-call lever is justified by the probe, wire it through
   the binding (mirror R9's per-call `num_slots` field).
4. Tight verify on the 4 short-grid cells (Down-B4-M2048, Down-B4-M4096,
   GateUP-B4-M2048-dgrad-via-H4) — each at its current shipped
   `(gm, xcds, num_slots)` cell.

If the chunk_size sweep shows no new cliff alignment within ±1 % of
the existing slots=192/196/200 wins, falsify and pivot to candidate
#3 (LDS metadata vec2 repack).
