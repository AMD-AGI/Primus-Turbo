# Round 6 (optimize) — wgrad 4-wave: the tuner measures where it used to model

Git is forbidden this round, so this file stands in for `agent_diff.txt`.

## What is on disk

One file, `primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py`, three edits, all inside
the `kernel_grouped_tn_wgrad_4wave` candidate list. Nothing else in the tree moved (the
`gemm_epilogue_helper.py` hunk in the working tree is round 5's LDS swizzle, already kept).

1. **`_wgrad_4wave_cands`, boundary body becomes the incumbent.** Was:

   ```python
   cands += tuple(c[:4] + (3, True) for c in cands[:2])
   ```

   Now:

   ```python
   if OUT_M % block or OUT_N % block:
       bnd = tuple(c[:4] + (3, True) for c in cands[:2])
       cands = bnd[:1] + cands + bnd[1:]
   ```

   `half_bnd=3` forces `_HALF_M`/`_HALF_N` on regardless of launch depth, so the short last
   M/N block skips its all-masked MFMA and loads half its operand slab. At the deploy shapes
   `_HALF_N` already fired through the cost gate; the delta is `_HALF_M`, which `_BND_GATED`
   turns off for every launch at or past `_WGRAD_AFF_ROUNDS * _NCU`. Where the tile grid has no
   short block the two builds are the same kernel, so the duplicates are no longer offered at
   all and those shapes race two fewer arms.

2. **`_wgrad_aff_widths`, new 6-line helper**, and one line in `_wgrad_4wave_cands` that races
   the two band widths adjacent to the affine geometry's pick. `_wgrad_xcd_aff_geom` minimises
   `max(rows, cols)` of the per-XCD rectangle — an operand-bandwidth proxy — and the wall does
   not follow it here.

3. **`_WGRAD_RACE_MARGIN` 0.985 -> 0.995.** Without it neither new candidate can be reached:
   0.985 asks a challenger for 1.5%, and the two that matter are worth 0.9% and 0.4%.

## Measurements

Scored bench, this tree, five reads:

| read | spd | step_ms | t_mlp | ratio_mlp | gate |
|---|---|---|---|---|---|
| 1 | 1.039271 | 680.3806 | 10.9651 | 0.968546 | PASS |
| 2 | 1.040679 | 679.4602 | 10.9511 | 0.967308 | PASS |
| 3 | 1.039331 | 680.3412 | 10.9644 | 0.968480 | PASS |
| 4 | 1.039170 | 680.4472 | 10.9724 | 0.969187 | PASS |
| 5 | 1.039388 | 680.3042 | 10.9608 | 0.968164 | PASS |

median spd **1.039331**, mean 1.039568; ratio_mlp mean **0.968337**.
Round 5 three reads: 1.038095 / 1.038187 / 1.037848, ratio_mlp 0.972749 / 0.971362 / 0.971574.
All five ratio_mlp readings sit below all three of the previous round's; mlp is -0.366%.
`proj` 0.8371-0.8418 and `perm` 0.7996-0.8105 stay inside their bands — neither was touched.

Correctness: `grad_x`, `grad_w1`, `grad_w2` byte-identical between `half_bnd=3` and the gated
form, balanced and ragged, two seeds. The skipped rows/cols lie outside the store's row/column
clamp, so this is byte equality rather than a dB floor.

Family check (pitfalls/06 §static-lead: a lead acts on every M its predicate reaches). Two wgrad
calls timed alone, geometry pinned, `half_bnd` the only variable, discard burn after each build:

| M/expert | gated (ms) | boundary (ms) | ratio |
|---|---|---|---|
| 1024 | 1.0420 / 1.0447 | 1.0277 / 1.0398 | 0.9838 |
| 2048 | 1.5818 / 1.5850 | 1.5625 / 1.5618 | 0.9853 |
| 4096 (deploy) | 2.7480 / 2.7437 | 2.6996 / 2.6817 | 0.9774 |
| 8192 | 5.1402 / 5.1588 | 5.0818 / 5.0513 | 0.9792 |

The lead is only the base: the gated form stays in the list right behind it, so anywhere the
boundary body is the wrong pick by more than half a point the race takes the slot back.

Race pick, dumped from the shipped closure, stable over two runs and both distributions:
fc1 `(5760,2880)` -> `(1, 6, 1, 1, 3, True)` rank 0 of 8 when the neighbours are absent,
`(1, 4, 1, 1, 3, True)` rank 6 of 8 when they are present; fc2 `(2880,2880)` ->
`(1, 4, 1, 1, 3, True)` rank 0 of 9.

## Reverted this round

* **Boundary lead alone** (margin left at 0.985, no width neighbours): bench spd 1.037381,
  ratio_mlp 0.972092 — indistinguishable from round 5. The work deletion needs the width the
  race can only reach once the margin is fine enough to resolve it.
* **Tiered plain dispatch** (`_WGRAD_TIER_BARS = ()`): 13440 of the two calls' grid ids are dead
  on a balanced load, 52-59% of each grid. Removing them is worth 0.9995 balanced and 1.0007
  ragged — about 5 µs. Empty workgroups on this part are nearly free; not a lever.
* **Non-temporal wgrad C store** (`store_aux` 2 and 3 on `StoreCPerTensor`/`RowN`): 1.0029 and
  1.0018 balanced, 1.0009 and 1.0023 ragged. Both lose. The output is write-once and nothing in
  the step reads it, but the store is the scalar/row-merged path emitting short runs, and L2
  write-combining those is worth more than the operand residency it displaces — the opposite of
  round 4's Triton permute result, where the stores were already full rows.
* **Band widths past the two adjacent divisors** (1, 2, 3, 6, 12 at both calls): every reading
  within 0.32% of width 4 with 0.14% control drift, and the ranking it produced was wrong — see
  pitfalls below.
