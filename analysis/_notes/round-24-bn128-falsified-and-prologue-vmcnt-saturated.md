# Round 24 — BN=128 path FALSIFIED via Triton tile-config audit + prologue VMCNT counters confirmed saturated

## TL;DR

Two falsifications shipped this round; neither moves score by design:

1. **BN=128 dispatch path is dead-end** — Triton's grouped GEMM kernel
   uses **BM=256, BN=256 on every gpt_oss shape**, identical to HK.
   N=2880 last-tile 25% utilization is paid by Triton too, yet Triton
   still beats HK. The wedge is **not** N-tile geometry.
2. **`RCR_INIT0_VMCNT` / `RCR_INIT1_VMCNT` prologue wait counters
   saturated** — 4-cell sweep on B=4 FP8 shapes. Increasing past
   baseline `(4, 6)` flips into a correctness race (1/5 runs reject:
   vmcnt threshold > actual outstanding ⇒ no-wait ⇒ main loop reads
   stale LDS); decreasing is at-best score-neutral.

Net: code unchanged this round, docs only. Score stays at the 880 ± 3
plateau established in rounds 19-23.

## (1) Triton vs HK tile config — gpt_oss-specific audit

### What

Probed Triton's per-shape config selection via the lru-cached
`_get_gg_bf16_fwd_config` / `_get_gg_fp8_tw_fwd_config` for all 16
gpt_oss BF16 + FP8 forward shapes (M ∈ {2048, 4096}, N ∈ {2880,
5760}, K = 2880, B ∈ {4, 32}). Triton's `_select_params_origami`
**returns `None` for every gpt_oss shape** (the origami auto-tune
database has no entries for K = 2880), so Triton falls back to its
default per-arch dispatch:

| precision    | Triton (gpt_oss default fallback)           | HipKittens (kernel template constants)                          |
|--------------|----------------------------------------------|------------------------------------------------------------------|
| BF16 fwd RCR | `BM=256, BN=256, BK=32, num_stages=3, gm=4`  | `BM=256, BN=256, BK=64, double-buffer (≈ ns=2), gm=per-shape`  |
| FP8 tw fwd RCR| `BM=256, BN=256, BK=128, num_stages=2, gm=4`| `BM=256, BN=256, BK=128, double-buffer (≈ ns=2), gm=per-shape` |

Verified at `/tmp/audit_triton_gpt_oss_tile.py` archived alongside;
the printed table covers all 16 shapes and is identical for every row
(N=2880 / N=5760 / M=2048 / M=4096 all collapse to the same default).

### Why this falsifies the BN=128 path

User & previous round-22 docs flagged "BN=256 → BN=128 for N=2880
(Down) shapes" as the highest-leverage remaining structural lever:
the rationale was last-N-tile utilization 64/256 = 25% ⇒ 64/128 = 50%
on the masked column. **But Triton uses BN=256 on the same N=2880
shapes and still beats HK by 1.0-3.5pp** — i.e. Triton pays the same
75% N-tail waste as HK. If N-tile geometry were the wedge, Triton
would also be slow; it isn't, so the wedge is elsewhere (BF16:
BK=32+ns=3 deeper pipeline; FP8: tile-identical, must be main-loop
schedule / MFMA cell pattern / chunk_size=32 vs HK's 64).

A BN=128 HK port — estimated 1-2 round budget in round 22 docs —
would, *at best*, reproduce the same 25% N-tail waste as Triton with
twice the grid (12→23 N-tiles) ⇒ launch overhead doubled. The
expected outcome is neutral-to-negative on B=4 launch-bound shapes
(where launch overhead already dominates per round-12 rocprof) and
flat on B=32 throughput-bound shapes. **Recommend: do not pursue.**

### What this implies for round 25+

The remaining wedge for BF16 grouped is structural: **port HK's BF16
grouped main loop to BK=32 + 3-stage pipeline** (matching Triton).
Direct cost estimates:

* Reduce `K_BLOCK` from 64 to 32 in the BF16 kernel template, which
  halves the tic/toc LDS slot footprint and adds 1 more stage before
  the tic/toc rotation. Requires re-deriving every register tile
  shape (`A_row_reg`, `B_row_reg`, `cA..cD` etc.) for the new BK.
* LDS budget: BF16 ST currently 8 KB/tile × 4 tiles (As[2][2] +
  Bs[2][2]) = 32 KB. Adding stage 3 + halving BK = 16 KB original
  + 16 KB stage 3 = 32 KB. LDS-neutral, but register tile count
  doubles.
* VGPR budget: HK BF16 main loop currently spills ~67 bytes/lane.
  Adding more register tiles is *prohibited* per round-5 docs
  (round-5 added 3 tiles, +3 spills nominally identical, but the
  spill victims shifted onto hot-path variables ⇒ **−55 score
  regression**). Stage-3 port must use **template-conditional**
  register tiles so dense / FP8 / non-stage3 variants don't pay.

This is a 3-4 round project, not 1. Round 25 should not begin until
a concrete roll-back plan is in place (round-5 lesson: incremental
1-tile-at-a-time, full metric verify between).

For FP8 grouped: tile config is already identical to Triton, so the
wedge is purely the in-kernel main-loop schedule (round 12 rocprof
attributed all 50 µs/iter Triton-vs-HK gap to the GEMM kernel
itself). No further rounds 1-15 micro-knobs apply (all saturated).
The only remaining structural change is the **MFMA cell-shape
rewrite** noted in round-15 docs: `mfma_scale_f32_16x16x128_f8f6f4`
→ `mfma_scale_f32_32x32x64_f8f6f4`. Round-12 confirmed the cell
count is identical (no MFMA waste), but the larger 32×32 cell
amortizes more accumulator registers per MFMA issue, which may be
the wedge for FP8. Estimated 2-3 rounds.

## (2) RCR_INIT0_VMCNT / RCR_INIT1_VMCNT prologue counter sweep

### Why test these

Round-5 docs covered `RCR_STEADY_VMCNT` ∈ {4, 8, 12} and
`RCR_PREFETCH_LGKM` ∈ {2, 4, 8} on the **main-loop** wait counters —
both saturated within ±1 TF. The **prologue** wait counters
(`RCR_INIT0_VMCNT = 4`, `RCR_INIT1_VMCNT = 6` at
`kernel_fp8_layouts.cpp:56-57`) were never swept. For B=4 / small-
batch shapes, prologue accounts for a relatively larger fraction of
wall (per round-12 rocprof: B=4 main loop only ~55 K-iters total
across the whole kernel ⇒ prologue = 4-loads × 2 × ~150 cyc HBM ≈
1200 cyc out of ~10k total ≈ 12% of wall). Prologue counter wins
should preferentially help the worst-ratio B=4 FP8 shapes.

### Sweep methodology

4-cell sweep `(INIT0, INIT1) ∈ {(2,4), (4,6) baseline, (6,8), (8,10)}`,
each → rebuild `tk_fp8_layouts.so` (~9 sec), → 1 metric run (~14 sec).
Then 5-run verify on each non-baseline candidate. All probe scripts
archived as `/tmp/probe_init_vmcnt_round24.txt`.

### Results

| (INIT0, INIT1) | 1-run sweep score | 5-run scores       | reject rate | verdict                          |
|----------------|-------------------|--------------------|-------------|----------------------------------|
| **(4, 6)**     | 881               | 881, 878, 881, 884, 879  | 0/5     | **baseline (kept)**              |
| (2, 4)         | 882               | 880, 880, 882, 880, 878  | 0/5     | **−1 vs baseline median**, safe   |
| (6, 8)         | 883               | 881, 881, 880, 661, 881  | **1/5** | **CORRECTNESS RACE — UNSAFE**     |
| (8, 10)        | 663               | (sweep already 663) | 1/1   | **CORRECTNESS RACE — UNSAFE**     |

### Why higher VMCNT thresholds break correctness

A `s_waitcnt vmcnt(N)` instruction on AMDGPU waits **until ≤ N
outstanding vmem operations**. If the actual outstanding count is
already < N at issue time, the wait is a no-op — execution proceeds
immediately. The grouped RCR prologue issues exactly 7 vmem
operations across init0 + init1:

```
prologue (line 2135-2147 of kernel_fp8_layouts.cpp):
  init0:
    rcr_8w_load_hoist(b_tile(tic, 0), ...)  ← 1 vmem
    rcr_8w_load_hoist(As[tic][0],     ...)  ← 1 vmem
    rcr_8w_load_hoist(b_tile(tic, 1), ...)  ← 1 vmem
    rcr_8w_load_hoist(As[tic][1],     ...)  ← 1 vmem
    TK_WAIT_VMCNT(RCR_INIT0_VMCNT=4)  ← wait ≤ 4 outstanding (passes once init1's 3 loads have not yet been issued)
    [...barrier, more setup...]
  init1:
    rcr_8w_load_hoist(b_tile(toc, 0), ...)  ← 5th vmem
    rcr_8w_load_hoist(As[toc][0],     ...)  ← 6th vmem
    rcr_8w_load_hoist(b_tile(toc, 1), ...)  ← 7th vmem
    TK_WAIT_VMCNT(RCR_INIT1_VMCNT=6)  ← wait ≤ 6 outstanding (1 of the 7 must be done)
    [...barrier, main loop...]
```

* `INIT1=8` means "wait until ≤ 8 outstanding". With only 7 issued
  so far, the wait is immediately satisfied with **0 of 7 actually
  done** — the main loop's first `load_a` / `load_b` (which read
  `As[tic][0]` / `b_tile(tic, 0)` from the prologue's first two
  loads) reads stale LDS (the prior tile's leftover bytes) for the
  ~150-cycle HBM round-trip window.
* `INIT1=10` is the same race, just earlier. The kernel may "get
  away with it" most of the time when L2 happens to be hot for the
  prior tile's data, then randomly fail when L2 is cold —
  explaining the 1/5 (= 20%) reject rate observed for `(6, 8)`.
* `(2, 4)` is the conservative direction: wait until ≤ 2 / ≤ 4
  outstanding — i.e. wait for **5 of 7 done** at the init1 boundary.
  This is more synchronization than the baseline `(4, 6)` (which
  waits for **3 of 7 done**). The extra wait costs ~1 score on the
  median; safe but not a win.

### Updated saturated-knob inventory

After this round, the FP8 grouped RCR kernel's per-call wait/sched/
unroll knob space is **fully audited** (no more cells left to sweep
without a structural change):

| knob                     | sweep round | range tested        | result                |
|--------------------------|-------------|---------------------|------------------------|
| `RCR_PREFETCH_LGKM`      | round 5     | {2, 4, 8}           | saturated ±0.16%       |
| `RCR_STEADY_VMCNT`       | round 5     | {4, 8, 12}          | saturated ±0.08%       |
| `RCR_INIT0_VMCNT`        | round 24    | {2, 4, 6, 8}        | (4) baseline; (6,8) race |
| `RCR_INIT1_VMCNT`        | round 24    | {4, 6, 8, 10}       | (6) baseline; (8,10) race |
| `RCR_TWO_TILE_MID_VMCNT` | n/a         | not exercised — grouped uses single-tile path | round-15 confirmed two-tile MIN_KI lowering is no-op |
| `RCR_EPILOGUE_VMCNT`     | n/a         | epilog vmcnt — same race risk as INIT1 if raised | unswept (presumed similar) |
| `RCR_MAIN_UNROLL`        | round 17    | {1, 2, 4}           | saturated, =2 optimal  |
| `RCR_TWO_TILE_MIN_KI`    | round 15    | {20, 22, 28}        | no-op for ki=22 (grouped) |
| `chunk_size` (chiplet)   | round 22    | {32, 64}            | saturated (32 regresses B=4) |
| `(group_m, num_xcds)`    | rounds 7-23 | wide                | saturated per shape    |

Remaining unswept counters: `RCR_EPILOGUE_VMCNT`. A future round
could try lowering it (raising would race), but the epilog is
already inside one persistent-tile boundary so the leverage is
small (round 5 already showed steady-state vmcnt is saturated).

## Verification artifacts

* `/tmp/audit_triton_gpt_oss_tile.py` — Triton tile config audit
  (16 BF16 + 16 FP8 shapes; all return BM=256, BN=256)
* `/tmp/probe_init_vmcnt_round24.txt` — 4-cell sweep log
* `/tmp/build_init_*.log` — per-cell rebuild output (no spill change)

## Round-25 recommendation

Score plateau at 880 ± 3 for 5 consecutive rounds (19-24). All
"small lever" sweeps in the FP8 grouped kernel are saturated. The
remaining wedges all require multi-round structural rewrites:

1. **BF16 grouped: BK=64 → BK=32 + num_stages=3** (Triton match,
   3-4 rounds, must be incremental per round-5 spill lesson)
2. **FP8 grouped: MFMA cell-shape 16×16×128 → 32×32×64 rewrite**
   (2-3 rounds, requires register tile re-derivation)

Either is high-effort, high-risk; both have a roll-back plan
prerequisite (round-5 spill cascade lesson). Until then, score
will remain at the 880 ± 3 plateau.

**Do NOT** pursue: BN=128 path (this round, Triton-tile-audit
falsified), prologue VMCNT raises (this round, race-condition
falsified), `chunk_size=32` (round 22, B=4 regression), 2-tile
main-loop port (round 6, −144 score catastrophic fail), 2-tile
MIN_KI threshold (round 15, no-op), full K-tail hoist (round 5,
−55 score), kernel-side RCR sched_barrier removal (round 8,
falsified for grouped).
