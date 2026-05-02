round-46-fp8-grouped-NEW-24-shape-suite-baseline.md
====================================================

Round: 46 / 100
Date: 2026-05-02
SHA: 7186b919 (no commit this round besides this doc)
Task: grouped FP8 tensorwise (fwd + var-K bwd + dA bwd) on the NEW 24-shape
      suite (DeepSeek-V3 + gpt_oss_20B + **Qwen3-235B-A22B (new 8 shapes)**)

## TL;DR

This is the **first round on the NEW 24-shape suite** (the suite was just
expanded from 16 → 24 by adding 8 Qwen3-235B-A22B shapes). Per task body's
explicit first-round guidance:

  > 第一轮强烈推荐 baseline metric + 把 24 row table 写入 round note，
  > 不要直接改 kernel —— 没数据基础选 lever 是赌博

So: **baseline-only round, no kernel changes**. Doc-only commit.

**Authoritative metric is GPU-blocked** (idle hard-check refuses card3 — VRAM
shows 19.49 GB used though no KFD pid lists it; same zombie-VRAM situation
as R36-R45). Substituting `scripts/_fp8_grouped_nogate_probe.py` (single-trial
±2-8 pp noise) over **3 full runs** for stability sanity:

  Run 1 GEOMEAN = 1.2034   (extrapolated score ~994)
  Run 3 GEOMEAN = 1.1492   (extrapolated score ~973)
  Run 4 GEOMEAN = 1.1792   (extrapolated score ~985)
  median geomean ≈ **1.179**, range ≈ 1.149-1.203, spread ≈ 5.4 pp

If the metric *could* run, it would likely score in the **973-994 range**
(median estimate ~985). Best historical metric score = 981 (still on the
old 16-shape suite from R37 best). The probe data does NOT prove a
regression vs the 16-shape best — but does NOT cleanly confirm progress
either.

**Critical finding — task body's Qwen3 cold-start prediction was wrong**:
all 8 Qwen3 shapes register ratio ≥ 1.10 in median; worst Qwen3 is
`Qwen3-Down-B16-M4096` at 1.106. **Qwen3 is NOT a bottleneck.** The
existing kernel handles Qwen3 (128-aligned, K=4096 / K=1536) without
cold-start issues — likely because the K=128 multiple keeps the K-tail
codepath dormant.

**Real bottleneck: gpt_oss K=2880 (K-tail K_REM=64) cluster.** Top 5
worst cases by median are *all* gpt_oss shapes — same cluster the old
R5-R55 saturation rounds attacked.

## GPU state (R46)

The persistent zombie KFD VRAM leak is **still there** (19th consecutive
round including the early-stop / new-suite gap):

    card3,309220868096,20924743680   # 19.49 GB used, baseline ~320 MB

`rocm-smi --showpids` lists *no* KFD process holding card3, yet the VRAM
is reserved. Same situation on cards 4, 6, 7 (all in `HIPKITTEN_GPU_POOL`):

    card0: 18.92 GB    card1: 20.34 GB (real workload, KFD pids confirm)
    card2: 21.39 GB    card3: 19.49 GB ← pinned to me, no KFD pids
    card4: 19.27 GB    card5: 21.24 GB
    card6: 19.21 GB    card7: 19.20 GB

`auto_optimize.py`'s GPU-pin uses `(use% < 30) AND (KFD VRAM = 0)` — both
satisfied on card3, so it pins. But `_metric_hk_ratio._assert_gpu_truly_idle`
(R45-shipped defense-in-depth) demands **VRAM used ≤ 320 MB** which can
NEVER pass on this box without a kernel module reload.

Result: **5 consecutive rounds (R41-R45) recorded `metric=None`** by the
auto_optimize tracker. R46 follows the same pattern unless infrastructure
unblocks.

The two "idle" definitions disagree on this exact failure mode (compute-
idle but VRAM-leaked card). Per task body's GPU rules I cannot edit either
checker, and I cannot `export HIP_VISIBLE_DEVICES=0` (already pinned to 3).

**ACTIONABLE for user / infra**: `sudo rmmod amdkfd && sudo modprobe amdkfd`
on this MI355X box (or migrate to a non-leaked GPU outside the pool, e.g.
GPUs 1-N if any are clean). 19 rounds of leaked VRAM is the dominant
blocker on this run — every round currently lands as `metric=None` and
the auto_optimize tracker can't ratchet `best`.

## 24-shape baseline table (per-shape median across 3 probe runs)

Sorted ASC (worst first):

```
rank  shape                              med    R1     R3     R4     spread  cluster
  1   gpt_oss-GateUP-B4-M4096           1.056  1.056  1.056  1.027   2.9pp   small-grid
  2   gpt_oss-GateUP-B32-M2048          1.061  1.056  1.084  1.061   2.8pp   *worst-anchor
  3   gpt_oss-Down-B32-M4096            1.067  1.067  1.061  1.072   1.1pp   *tightest
  4   gpt_oss-GateUP-B32-M4096          1.074  1.075  1.054  1.074   2.1pp
  5   gpt_oss-Down-B32-M2048            1.077  1.077  1.064  1.085   2.1pp
  6   gpt_oss-GateUP-B4-M2048           1.103  1.095  1.113  1.103   1.8pp
  7   Qwen3-Down-B16-M4096              1.106  1.106  1.112  1.102   1.0pp   <-- worst Qwen3
  8   Qwen3-Down-B16-M2048              1.145  1.150  1.144  1.145   0.6pp
  9   gpt_oss-Down-B4-M4096             1.391  1.866  1.087  1.391  77.9pp   ! noisy
 10   Qwen3-Down-B32-M4096              1.163  1.163  1.155  1.164   0.9pp
 11   Qwen3-GateUP-B32-M2048            1.164  1.164  1.149  1.178   2.9pp
 12   Qwen3-Down-B32-M2048              1.168  1.179  1.078  1.168  10.1pp   ! noisy
 13   DSV3-GateUP-B32-M4096             1.170  1.170  1.191  1.169   2.2pp
 14   DSV3-GateUP-B32-M2048             1.169  1.193  1.168  1.169   2.5pp
 15   DSV3-GateUP-B16-M4096             1.173  1.201  1.173  1.164   3.7pp
 16   Qwen3-GateUP-B32-M4096            1.174  1.193  1.174  1.169   2.4pp
 17   Qwen3-GateUP-B16-M2048            1.176  1.237  1.160  1.176   7.7pp
 18   DSV3-Down-B16-M4096               1.195  1.195  1.179  1.243   6.4pp
 19   gpt_oss-Down-B4-M2048             1.206  1.231  1.137  1.206   9.4pp
 20   DSV3-GateUP-B16-M2048             1.219  1.296  1.161  1.219  13.5pp   ! noisy
 21   DSV3-Down-B16-M2048               1.246  1.246  1.266  1.224   4.2pp
 22   DSV3-Down-B32-M4096               1.289  1.286  1.435  1.289  14.9pp   ! noisy
 23   Qwen3-GateUP-B16-M4096            1.376  1.410  1.152  1.376  25.8pp   ! noisy
 24   DSV3-Down-B32-M2048               1.395  1.395  1.299  1.430  13.1pp   ! noisy
```

**WORST 5 (signal-tight, all-gpt_oss-K2880 cluster):**

  1. `gpt_oss-GateUP-B4-M4096`        1.056   N=5760 K=2880 M=4096 B=4
  2. `gpt_oss-GateUP-B32-M2048`       1.061   N=5760 K=2880 M=2048 B=32
  3. `gpt_oss-Down-B32-M4096`         1.067   N=2880 K=2880 M=4096 B=32
  4. `gpt_oss-GateUP-B32-M4096`       1.074   N=5760 K=2880 M=4096 B=32
  5. `gpt_oss-Down-B32-M2048`         1.077   N=2880 K=2880 M=2048 B=32

**ALL 5 share K=2880 = 22 × 128 + 64** → `K_REM = 64` → K-tail dominates
the last iter. This cluster is exactly what R3 / R4 / R6-dm / R8-grouped
attacked (K-tail vmcnt / sched_barrier / load-reorder rounds). Every
falsified micro-knob in the FROZEN list landed on this cluster's K-tail
inner loop.

The three large-grid B=32 cases (rank 2, 4, 5) have spread ≤ 2.1pp across
runs — the +1.06-1.08 ratio is **tight signal, not noise**.

## Per-cluster summary

```
cluster                             count  median ratio  range
gpt_oss K=2880  (K-tail dominant)     8    1.077         1.05-1.39 (small-grid noisy)
Qwen3   K=4096/1536  (clean K%128=0)  8    1.165         1.10-1.38 (small-grid noisy)
DSV3    K=7168/2048  (clean K%128=0)  8    1.207         1.16-1.43 (Down highly noisy)
```

**Qwen3 verdict (vs task body prediction)**: The task body warned that
Qwen3 GateUP/Down might mass-FAIL ("Qwen 大面积低于 1.0"). Reality: Qwen3
median 1.165, all 8 shapes ≥ 1.10. The kernel handles Qwen3 cleanly — its
N/K=128 alignment means the K-tail / N-tail codepath stays dormant, so
existing main-loop performance is what we see (similar to DSV3 cluster).

**DSV3 verdict**: 1.169-1.430 range, all healthy. R42-R45 dispatch tunes
landed on this cluster (RRR narrow/wide-N + RCR B16 m_total relax) and the
results show in DSV3-Down which dominates 1.29-1.43.

**gpt_oss verdict**: this is the bottleneck cluster. Median 1.077 is
**exactly the same** as R37 metric data on the old 16-shape suite (gpt_oss
geomean 1.07-1.09). **No regression, no progress** — the K-tail saturation
plateau is intact.

## What doesn't apply / what FROZEN list says

The full FROZEN list (task body) is intact. NEW data point: even with the
24-shape suite expansion, the **5 worst** are all the same gpt_oss K=2880
cluster from the 16-shape era. None of the lever falsifications land
differently:

  ✗ K-tail vmcnt sweep / split-vmcnt    — already shipped, can't redo
  ✗ K-tail sched_barrier mask           — falsified R31-32 (mask insensitive)
  ✗ K-tail load-reorder                 — `[b0,b1,a,a_kt1]` already shipped
  ✗ MFMA cell-shape 32x32x64 (Lever D)  — falsified R34 (1.5x microbench loss)
  ✗ MFMA cell-shape 32x32x64 K-tail-only — falsified R62 (-9 pp on probe)
  ✗ async g→LDS (Lever A)               — already shipped (rcr_8w_load_hoist)
  ✗ dual LDS ping-pong (Lever B)        — already shipped (`As[2][2]/Bs[2][2]`)
  ✗ register restrict / lifetime hints  — saturated R12, R54
  ✗ noinline / builtin_expect           — falsified R54

The only architectural directions NOT proven empty:

  - **Lever D R-B FULL main-loop port** to `mfma_323264` (4-6 round commit)
    Steps 1-4 of infra are landed (`st_32x64`, `rt_32x64_s`, `rcr_mma_32`,
    K-tail loaders) per HK commits `c2abba2..78415fb0`. Step 5 (the
    actual main-loop body rewrite) was never started. Per R34 microbench
    falsification, the *isolated* `mfma_323264` is 1.5× *slower* than
    `mfma_1616128` at single-warp, so this is unlikely to net win even
    with full port. **Recommended: do NOT start without re-running the
    Lever D microbench gate (`tk_lever_d_microbench` per HK commit
    `9ee90e2c`) and confirming a ≥ 3 pp single-warp throughput edge for
    323264.** R34 confirmed -1.5× edge; nothing in 24-shape data
    invalidates that.

  - **Lever E hand-written ASM main-loop**: 2-3 round commit, very high
    risk. Build-time / debuggability cost. Per R63 plateau-accepted note,
    explicit user buy-in needed before starting.

## Additional probe-derived insights (not previously documented)

Comparing the 4 large-grid gpt_oss B=32 worst cases to their B=4 siblings:

```
shape                      med    grid_size  HK TFLOPS  TR TFLOPS  ratio gap
gpt_oss-GateUP-B32-M2048  1.061   65536      1997       1841       small
gpt_oss-GateUP-B4-M4096   1.056    16384      1827       1730       small
gpt_oss-GateUP-B32-M4096  1.074   131072     2040       1855       small
gpt_oss-Down-B32-M2048    1.077    65536     1831       1700       small
gpt_oss-Down-B32-M4096    1.067   131072     1892       1784       small
```

All 5 have absolute HK TFLOPS in the 1827-2040 range — that's well below
DSV3 GateUP's 2727 (K=7168). The **HK kernel's absolute throughput on
K=2880 caps out near ~2 TFLOPS** while DSV3 / Qwen3 (K%128=0) reach
2540-2750. The K-tail epilog burn is real.

Triton TFLOPS on the same 5 cases is also reduced (1700-1900 vs 2300+
for K%128=0 shapes), confirming K=2880 is hard for Triton too. The
**ratio** stays at 1.05-1.08 because both kernels lose the same fraction
to K-tail. Closing the ratio gap further means HK has to lose LESS to
K-tail than Triton does — unclear if there's headroom there.

## Why "do nothing" this round is correct

The task body explicitly says:

  > 第一轮强烈推荐 baseline metric + 把 24 row table 写入 round note，
  > 不要直接改 kernel —— 没数据基础选 lever 是赌博

This round is round 1 of the new 24-shape suite (different from R45 which
operated on the 16-shape suite). The only data point that justified
prior dispatch tunes (R42-R45) was tight per-shape probes targeted at
specific rule fall-throughs. Now we have:

  * 24-shape worst-5 all in same gpt_oss K=2880 cluster
  * NO new rule fall-throughs spotted (the dispatch table from R45's
    audit covers all 24 shapes via existing rules)
  * No new lever surfaced by the Qwen3 expansion (Qwen3 is healthy)

So: **shipping a kernel change this round would be "lever roulette" against
saturated knobs**. Doc-only round acknowledging the baseline is the
right action.

## Next round (R47) action ladder

1. **Highest priority — escalate GPU cleanup**: ask user (or infra) to
   `sudo rmmod amdkfd && sudo modprobe amdkfd`. Without it, the next 5+
   rounds will keep landing `metric=None`. The probe is single-trial
   noisy; if score truly is in the 973-985 band, the metric needs to
   confirm so the auto_optimize tracker can ratchet from 981 to ~985.

2. **If GPU unblocks**: run authoritative `_metric_grouped_only.py` and
   re-confirm the score. R45's accumulated dispatch changes (RRR narrow/
   wide-N + RCR B16 m_total relax) likely already account for any uplift;
   R45 wasn't proven by probe.

3. **If GPU still blocked AND nothing else improves**: recommend running
   the Lever D R-B microbench gate again (currently has R34 result of
   -1.5× single-warp). If gate stays ≥ -3 pp, formally close Lever D
   full-port roadmap and accept ~985 plateau.

4. **No new tight probes warranted on the worst-5**: the gpt_oss K=2880
   K-tail saturation is the same kernel signature R3-R55 attacked
   exhaustively. Spending another round on micro-knob sweep there
   without a NEW hypothesis (architecture change, data flow change)
   is the textbook definition of the user-FROZEN micro-knob class.

5. **DO NOT** re-attempt: any lever in the FROZEN ✗ list, dispatch.py /
   can_handle / config.py edits, kernel field hardcodes, model-specific
   shape table hardcodes, fallback to triton for difficult shapes,
   metric / test file edits.

## Falsification register (rebased on 24-shape suite)

| Lever / approach                       | Status (carried) | Round   |
|----------------------------------------|------------------|---------|
| Lever A (async g→LDS) — base shipped   | SHIPPED          | R54-dm  |
| Lever B (dual LDS) — base shipped      | SHIPPED          | early   |
| Lever C (restrict / lifetime hints)    | SATURATED        | R12,R54 |
| Lever D K-tail-only port               | FALSIFIED        | R62-dm  |
| Lever D full main-loop port (R-B 5+)   | NOT STARTED      | —       |
| Lever D microbench gate                | -1.5× single warp| R34-dm  |
| Lever E (ASM main-loop)                | NOT STARTED      | —       |
| Lever F (Qwen3 K=1536 short-K variant) | FALSIFIED        | R35-grp |
| sched_barrier / LICM / anti-CSE class  | FALSIFIED        | R31-32  |
| K-tail micro-knobs (vmcnt / reorder)   | SATURATED        | R3-R55  |
| FP8 var_k init parallelize             | SHIPPED          | R38     |
| FP8 var_k bwd dispatch (8, 4)          | SHIPPED          | R39     |
| FP8 rrr init parallelize               | SHIPPED          | R41     |
| FP8 RRR narrow-N dispatch (16, 4)      | SHIPPED          | R42     |
| FP8 RRR wide-N dispatch (16, 4)        | SHIPPED          | R43-R44 |
| FP8 RCR B=16 M=2048 m_total relax      | SHIPPED          | R45     |
| FP8 RCR B=16 M=4096 m_total relax      | SHIPPED          | R45     |
| **NEW**: 24-shape baseline + Qwen3-OK  | DOCUMENTED       | **R46** |

## Attribution

- HipKittens HEAD: `78415fb0` — UNCHANGED this round (no kernel edits)
- Primus-Turbo: only this doc note added
- No `config.py` / `dispatch.py` / kernel changes
- No metric / test edits

## Probe data files

Saved for the next round to compare against:

  /tmp/probe_round_46.log     (run 1, geomean 1.2034)
  /tmp/probe_round_46_v3.log  (run 3, geomean 1.1492)
  /tmp/probe_round_46_v4.log  (run 4, geomean 1.1792)
  /tmp/metric_round_46_baseline.log  (FATAL: GPU not idle)
