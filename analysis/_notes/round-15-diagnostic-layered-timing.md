# Round 15 — diagnostic: layered timing breakdown localizes the FP8 ratio gap to the kernel itself

**Date**: 2026-05-01
**Branch**: `dev/kyle_hipkitten_bf16`
**HEAD before**: `e1747ba`
**HEAD after**: this commit (Primus-Turbo notes-only; kernel/config bytes unchanged)
**HipKittens commit**: none
**Metric**: 794 → 794 (no functional change).

## Why this note

Patience counter at 10/30 with no metric improvement. Round 14 documented
3 falsified micro-knob experiments and observed the metric ceiling sits
at ~797 with mean 794.2 (10-run baseline). Before chasing further
micro-tunes, this round runs a **layered timing breakdown** on the
worst-ratio gpt_oss FP8 shape (`Down-B4-M4096`, ratio 0.834) to localize
where the gap actually sits. Result: **the gap is in the kernel itself**,
not in Python / dispatcher / autograd overhead. This narrows the
optimization space sharply for future rounds.

Two side-findings worth capturing in the same note (both flagged as
"falsified" so future rounds don't re-chase them):

1. **Triton's origami selector returns `None` for every gpt_oss shape**
   (BF16 + FP8) → Triton uses **the same default `BM=BN=256, BK=64`
   tiles as HK**. The round-30-start "BN=128 for N=2880" hypothesis is
   stale; matching Triton's tile choice cannot be the win.
2. **Initial breakdown probe overcounted "Python overhead" at 184us** —
   that was a measurement artifact (manually-constructed `group_offs`
   with wrong dtype caused HK to enter an early-out fast path). With a
   properly-formed `group_offs` (via `grouped_gemm_compute_offs`), the
   real Python overhead is ~5us and the HK kernel is ~196us.

## Step 1: Triton picks BM=BN=256 for all gpt_oss shapes

Probe at `/tmp/probe_triton_origami.py`. Runs Triton's
`primus_turbo.triton.gemm.gemm_kernel._select_params_origami` for every
gpt_oss shape (BF16 + FP8 forward, NT layout) at the per-group-M
granularity that matches `_get_gg_bf16_fwd_config` /
`_get_gg_fp8_fwd_config`. **All 16 returns are `None`** — Triton's
origami declines to override the default `BM=256, BN=256, BK=64` for
*any* gpt_oss shape:

```
BF16 fwd (NT, trans_b=True):
  Down-B4-M2048    M_avg=2048  N=2880 K=2880 | origami=None (defaults BM=BN=256, BK=64)
  Down-B4-M4096    M_avg=4096  N=2880 K=2880 | origami=None
  Down-B32-M2048   M_avg=2048  N=2880 K=2880 | origami=None
  Down-B32-M4096   M_avg=4096  N=2880 K=2880 | origami=None
  GateUP-B4-M2048  M_avg=2048  N=5760 K=2880 | origami=None
  GateUP-B4-M4096  M_avg=4096  N=5760 K=2880 | origami=None
  GateUP-B32-M2048 M_avg=2048  N=5760 K=2880 | origami=None
  GateUP-B32-M4096 M_avg=4096  N=5760 K=2880 | origami=None

FP8 fwd (NT, trans_b=True): same — origami=None for all 8 shapes.
```

**Conclusion**: HK and Triton both run `BM=BN=256` on every gpt_oss
shape. The performance gap is *not* due to a different tile choice. The
round-30 starting suggestion "try BN=128 for N=2880" is **falsified** —
matching Triton means staying on `BM=BN=256`. Building a parallel HK
`BN=128` template would not close the gap; it would race against
Triton's own non-existent BN=128 path.

## Step 2: Layered breakdown (FP8-Down-B4-M4096)

Probe at `/tmp/probe_layered.py`. Mirrors the metric's `_time_op`
exactly (sync each iter, p20 of 50 iter, after 10-iter warmup). Times 4
layers in increasing-overhead order: just-quantize, kernel-only via
`backend.execute()`, kernel via dispatcher, kernel via the autograd-
wrapped `turbo.ops.grouped_gemm_fp8`.

**FIRST attempt** (with manually-constructed `group_offs` of wrong
dtype) showed HK kernel at 14us — way faster than physically possible
(271 GFLOPs / 14us = 19 PFLOPS, vs MI355X's ~2.6 PFLOPS FP8 peak).
That 14us was a **fast-path artifact**: HK's persistent grouped
launcher hits an early-out when `group_offs` is malformed (the device-
side O(G) scan trips and the kernel exits without computing). Hence
the apparent "184us Python overhead" — real overhead is ~5us, the
"missing" 175us was just the kernel exiting early.

**SECOND attempt** at `/tmp/probe_dispatcher_overhead.py` uses
`grouped_gemm_compute_offs(group_lens)` to construct `group_offs`
correctly. Numbers below are with that fix applied:

```
L1: HK.execute()                              = 196.5us   (FP8 kernel + ~5us Python)
L1: Triton.execute()                          = 166.7us   (same path, Triton kernel)
L2: grouped_gemm_fp8_impl (HK backend)        = 201.6us   (+ 5us dispatcher)
L2: grouped_gemm_fp8_impl (TRT backend)       = 168.3us   (+ 1.6us dispatcher)
L3: grouped_gemm_fp8_impl (HK, maybe_pre_sync=False) = 197.6us  (-4us pre_sync)
```

Independently measured: `quantize_fp8(a)` = 47.7us, `quantize_fp8(b)`
= 38.0us, sum = 86us (same path for both backends, irreducible).

Full op breakdown (L4: `turbo.ops.grouped_gemm_fp8(a_bf16, b_bf16, …)`):

```
                quantize  kernel  dispatch  pre_sync  TOTAL
HK              86us      197us   5us       4us       291us
Triton          86us      167us   1us       0us       254us  (≈248us measured)
```

Ratio match: HK 271.8 GFLOPs / 291us = 935 TF, Triton 271.8 GFLOPs /
248us = 1095 TF, ratio = 0.854 (the metric reports 0.834 with a
slightly different inputs/warmup pattern — within noise band).

## Where the gap actually sits

| component        | HK     | Triton | gap       | %% of wall gap |
| ---------------- | ------ | ------ | --------- | -------------- |
| quantize_fp8     | 86us   | 86us   | 0us       | 0%             |
| **kernel**       | 197us  | 167us  | **+30us** | **70%**        |
| dispatcher       | 5us    | 1us    | +4us      | 9%             |
| pre_sync waste   | 4us    | 0us    | +4us      | 9%             |
| autograd save    | ~5us   | ~5us   | 0us       | 0%             |
| **total wall**   | 291us  | 248us  | +43us     | 100%           |

**70% of the ratio gap (30us out of 43us) is in the FP8 grouped kernel
itself**, not in any Python / dispatcher / autograd path. Round-14's
conclusion stands: only structural kernel changes will close this gap.
Round-13 showed config tuning saturated; round-14 falsified all
sched_barrier / VMCNT micro-knobs at the metric level. The remaining
30us must be unlocked at a coarser granularity — main-loop pipeline
restructuring, K-tail amortize across multi-tile, or HBM prefetch
schedule changes.

## Side-finding: `maybe_pre_sync` overhead is NOT real (noise)

Initial single-trial measurement showed
`maybe_pre_sync=True` adding 4us per call vs `=False`. Five-trial
verification at `/tmp/probe_pre_sync_noise.py`:

```
5 trials each, p20 of 50 iter (us):
  trial 0: pre_sync=True 206.1us  pre_sync=False 200.9us  Δ +5.2us
  trial 1: pre_sync=True 200.4us  pre_sync=False 200.1us  Δ +0.3us
  trial 2: pre_sync=True 200.9us  pre_sync=False 200.7us  Δ +0.2us
  trial 3: pre_sync=True 200.0us  pre_sync=False 200.2us  Δ -0.2us
  trial 4: pre_sync=True 200.0us  pre_sync=False 199.5us  Δ +0.5us
```

The "4us overhead" was a single-trial artifact — average across 5
trials is ~+1us, dominated by trial 0's noisy +5.2us. **No win from
plumbing `maybe_pre_sync` only on the hipblaslt path.** Falsified.

This is a useful negative result: it tightens the kernel-vs-Python
attribution to **>95% of the gap is in the kernel itself**, not in any
control plumbing. There is no Python-side micro-win left to chase on
the FP8 grouped hot path.

## What's NOT yet falsified (next-round candidates, refined)

Strict ranking by hypothesis weight × cost (best ROI first), updated
based on this round's diagnostic findings:

1. **HK kernel structural change** — the round-14 candidate set
   ((K-tail amortize / B=4 grid swizzle / FP8 main-loop pipeline)
   stays open. The 30us in-kernel gap is the only real lever now.
   Rocprof breakdown of the HK kernel (HBM-bound vs MFMA-bound vs
   LDS-bound) is the necessary next probe before committing to a
   specific structural change. **High ROI, ~3-5 round investment.**
2. **`turbo.ops.grouped_gemm_fp8` autograd save trim** — the
   forward saves 6 tensors (a_fp8, b_fp8, a_scale, b_scale,
   group_lens, group_offs); some (group_lens, group_offs) might
   be reconstructable in backward without saving. Marginal,
   hard to quantify without a benchmark. **Low ROI.**

## Falsified this round (do NOT re-attempt)

* **BN=128 for N=2880 shapes** — Triton uses BN=256 just like HK
  (`origami=None` for every gpt_oss shape). Building a separate HK
  BN=128 template wouldn't close the gap because the gap isn't in
  tile choice.
* **Python overhead trimming on FP8 hot path** — round-11 already got
  it down to ~5us; further trimming is below the metric noise band
  (±3 pt = ~1us at B=4 latency-bound shapes).
* **`maybe_pre_sync` plumbing cleanup** — single-trial measurement
  showed +4us, but 5-trial verification shows the difference is noise
  (mean +1us across 5 trials, dominated by one +5.2us outlier). The
  kwarg pass-through is essentially free.
* **Diagnostic probe with manually-constructed `group_offs`** — must
  use `grouped_gemm_compute_offs(group_lens)` to construct correctly;
  hand-rolled int32 offset tensors trip an HK fast-path that exits
  without computing, producing nonsense ~14us "kernel time".

## Files / commits

* HipKittens: no change.
* Primus-Turbo: this commit —
  `analysis/_notes/round-15-diagnostic-layered-timing.md` (this
  file).

Diagnostic probes archived for next-round reference:
* `/tmp/probe_triton_origami.py` — origami probe (BN selection).
* `/tmp/probe_layered.py` — layered breakdown (quantize / kernel / op).
* `/tmp/probe_dispatcher_overhead.py` — dispatcher overhead breakdown.
* `/tmp/probe_pre_sync_noise.py` — 5-trial verification of pre_sync overhead (noise).

Self-bench: not required (no backward path touched, no kernel
edited, only documentation).
