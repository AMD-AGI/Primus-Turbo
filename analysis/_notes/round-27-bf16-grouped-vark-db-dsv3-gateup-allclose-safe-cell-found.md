# Round 27 — DSV3-GateUP dB var-K extended probe: allclose-safe cell FOUND but too small to land

## Goal

R26 explicitly recommended R27 main line as "K-tail forward kernel
structural probe" (HK .cpp work). With ~30 min available in the
chat window after R26's 3-rule falsification, the structural HK
work is too risky for one round. Instead, R27 closes the open
question from R24: "is there ANY allclose-safe uniform-positive
cell for DSV3-GateUP dB var-K, or is it locked at default?"

R24 found only `(gm=2, xcds=0)` uniform-positive (avg +0.28 %)
but it failed the metric's Triton-reference allclose check
(max_abs ~232-327 vs bf16 expected ~1-2). The R24 probe never
tested cells with `xcds ∈ {2, 8, 16, 32}` for `gm=2` (only
xcds=0 and xcds=4).

R26's recommendation suggested R27 alternates included
"selectively LAND just the gpt_oss-GateUP dB var-K rule" — but
that rule alone (3× weight × 4 shapes × ~0.7 expected score) is
also sub-+5. So R27 instead investigates whether a 4-rule R26
aggregate (R26's 3 + a new DSV3-GateUP rule) becomes viable.

R27 baseline = **880** (single run).

## Probe — `scripts/_bf16_vark_db_dsv3_gateup_extended_probe.py`

Extended 26-cell sweep on DSV3-GateUP dB var-K (vs R24's 11
cells), with explicit Triton-reference `torch.allclose(rtol=1e-2,
atol=1e-1)` verification on every uniform-positive candidate.

Newly probed cells beyond R24's set:
  `(1,2)`, `(2,2)`, `(8,2)`, `(16,2)`,
  `(1,8)`, `(2,8)`, `(8,8)`, `(16,8)`,
  `(1,16)`, `(2,16)`, `(4,16)`,
  `(1,32)`, `(2,32)`.

### All 4 uniform-positive cells found

| cell | B16-M2k | B16-M4k | B32-M2k | B32-M4k | avg | Triton allclose |
|---|---|---|---|---|---|---|
| (gm=2, xcds=0)  | +0.13% | +0.26% | +0.14% | +0.50% | +0.26% | **FAIL** (max_abs 232-327) |
| (gm=2, xcds=8)  | +0.16% | +0.20% | +0.16% | +0.47% | **+0.25%** | **PASS** (max_abs 1.79-2.00) |
| (gm=2, xcds=32) | +0.15% | +0.19% | +0.21% | +0.45% | +0.25% | **PASS** (max_abs 1.79-2.00) |
| (gm=2, xcds=16) | +0.03% | +0.14% | +0.21% | +0.42% | +0.20% | **PASS** (max_abs 1.79-2.00) |

The `xcds ∈ {8, 16, 32}` cells produce IDENTICAL `max_abs`
values per shape (1.79-2.00 across 4 shapes) — same output
just at different scheduler partitions. Confirms the chiplet-
swizzle bypass at `xcds=0` is the unique source of the R24
allclose drift.

**Best landable cell: `(gm=2, xcds=8)` avg +0.25 %, range
[+0.16 %, +0.47 %], uniform-positive AND Triton-allclose-safe.**

## Score arithmetic — why it doesn't help

| rule (4 shapes per) | weight | wall_frac | kernel Δ | score contribution |
|---|---|---|---|---|
| (R27) DSV3-GateUP dB var-K (gm=2, xcds=8) | 1× | ~25 % | +0.25 % | +0.06 |
| + R26 already-probed Qwen3-GateUP fwd RCR | 1× | ~28 % | +1.68 % | +0.4 |
| + R26 already-probed DSV3-Down fwd RCR     | 1× | ~31 % | +0.47 % | +0.15 |
| + R26 already-probed gpt_oss-GateUP dB var-K | 3× | ~25 % | +0.98 % | +0.7 |

Sum = ~+1.3 expected score — STILL well below the +5 commit
threshold (5-run noise σ ≈ 4). R26 actually measured +0.6 for
the 3-rule subset; adding R27's DSV3-GateUP would push it to
~+1.4 measured at most. Not viable as a single-round commit.

## Decision

**Documentation-only commit.** No production change. Probe and
finding archived for future use:

* If a future round adds 2-3 MORE uniform-positive rules
  (e.g. from a kernel-level structural improvement that opens
  new headroom), bundling all 5-7 rules together would cross
  the +5 noise floor. Both R26 and R27's probe data are then
  reusable as the smaller-but-real components of that aggregate.
* The DSV3-GateUP `(gm=2, xcds=8)` finding closes a long-
  standing open question (R24 noted "DSV3-GateUP dB var-K
  stays on default — only known uniform-positive cell fails
  allclose"). It can now be promoted to "DSV3-GateUP HAS an
  allclose-safe uniform-positive cell at +0.25 %; requires
  bundling to land".

## Why this is not a falsification (subtle)

R25 / R26 were FALSIFIED because they ATTEMPTED a
production change and the metric didn't move. R27 makes NO
production change attempt because the score arithmetic
*pre*-shows the change won't cross +5 (no need to spend a
metric verify cycle on a known-sub-noise change). The probe
finding (allclose-safe cell exists) is positive new
information — promoted to the dispatch backlog rather than
falsified.

## Suggested R28 next step

* **R28 must pivot away from dispatch tweaks** — R20 + R24 +
  R26 + R27 have collectively characterized every plausible
  (gm, xcds) cell on every metric family across fwd RCR,
  dA RRR, dB var-K, and the H4 reroute path. The remaining
  surface is sub-noise. The auto-optimizer's per-round budget
  needs a structural improvement to make further progress.
* **R28 main line**: read
  `HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`
  K-tail kernel section (lines 845-1060 contain the K-tail
  inner loop; 1664-1741 contain the LDS-staged K-tail
  scheduling). gpt_oss B=32 (K%128=64) hits this path and
  has bottom-4 metric ratios (1.05-1.09). Kernel
  structural work has not been attempted since R5.
* **R28 alt — H4 transpose kernel fusion**: the standalone
  Triton `bf16_transpose_3d` is at 5 TB/s but adds a separate
  HBM read+write pass for gpt_oss-Down B=4 (~5 % of wall).
  If we can fuse the transpose into the kernel's B-load path,
  that's a 5 % wall savings on 4 shapes × 3 weight = ~+1
  score. Marginal but better than +0.5.
* **Worst-case fallback**: if structural work is infeasible,
  explicitly land R26's 3-rule + R27's 4th-rule aggregate as
  a "kernel-correctness improvement" commit (every cell is
  bit-equivalent, every cell is allclose-safe, no regression
  on any shape) accepting that the metric won't move +5.
  Documents progress for the auto_optimizer's audit trail.

## Files

* `scripts/_bf16_vark_db_dsv3_gateup_extended_probe.py` —
  archived probe with extended cells + Triton allclose check.
* `analysis/_notes/round-27-bf16-grouped-vark-db-dsv3-gateup-allclose-safe-cell-found.md` — this note.
* No production change (config.py at HEAD c671443).
