# Round 15 — gpt_oss FP8 grouped: dispatcher-lever exhaustion audit (4 falsifications)

After R13 (var-K chunk_size=96 for Down-B4 wgrad, +1.49% / +1.35%) and
R14 (RCR chunk_size=96 for Down-B4-M2048 fwd+dgrad-via-H4, +4.06% /
+4.05%) shipped the chunk_size lever for the only two cells where it
moves performance, this round audits the remaining candidate cells to
close the dispatcher-lever search.

R15 baseline metric: 693 (median of 5 runs). All 4 audits below
returned **NO WIN** (every variant ties or regresses vs the existing
rule). Documenting here so future rounds don't re-burn the same compute.

## R14 best metric: 693

R14 shipped chunk_size=96 for Down-B4-M2048 fwd+dgrad-via-H4 cell
(xcds=2 + slots=196). +4.06% / +4.05% probe lift. R14 metric 693 = NEW
BEST.

R15 entry baseline: 693 (same as R14).

## Audit 1 — GateUP-B4 wgrad var-K num_slots (FALSIFIED)

Cell: vk_group_m=1, vk_num_xcds=4, slots=NUM_CUS=256 (R31/R35→R3
rule). Tile-step density 968/256 = 3.78 ws/CU (borderline; R3 said
"already moderately amortised" but never directly tested).

Probe: scripts/_probe_round_15_gateup_b4_wgrad_slots.py.
Methodology: 1500-iter × p20 × 3 seeds × kernel-only via the impl path.

```
slots   GateUP-B4-M2048 wgrad   GateUP-B4-M4096 wgrad
   0    1610.5 T  baseline      2003.4 T  baseline   (=NUM_CUS=256)
 160    1347.4 T  -19.53%       1682.7 T  -19.06%
 192    1474.9 T  -9.20%        1852.7 T  -8.14%
 200    1456.8 T  -10.55%       1831.7 T  -9.38%
 208    1444.2 T  -11.52%       1815.1 T  -10.38%
 220    1423.0 T  -13.18%       1802.1 T  -11.17%
 240    1567.8 T  -2.73%        1990.5 T  -0.65%
 256    1613.6 T  +0.19%        2007.6 T  +0.21%
```

Every num_slots reduction REGRESSES; default 256 is unique optimum.
Confirms the ws/CU ≥ 2.5 saturation threshold from R3 (Down-B4 wgrad
at 1.89 ws/CU benefits +6%; this cell at 3.78 is on the saturated
side).

## Audit 2 — var-K chunk_size on 6 unaudited wgrad cells (FALSIFIED)

R13 shipped chunk_size=96 only for the Down-B4 cell (xcds=2 +
slots=192). The other 6 metric var-K wgrad cells use xcds=4 +
slots=256, which gives a clean 1-chunk partition (block=4*64=256=slots
→ all chunked, 64 PIDs/XCD) at the default cs=64. Probe asks: do
finer chunk_size values (16/32: 2-4 chunks) or larger ones (96/128:
swizzle NO-OP) help?

Probe: scripts/_probe_round_15_vark_chunk_size_audit.py. 6 cells ×
6 cs values = 36 measurements.

```
cell                      cs=64 (base)   best cs   lift
GateUP-B4-M2048 wgrad     1650.4 T       cs=64     +0.00%  *NO WIN
GateUP-B4-M4096 wgrad     2023.7 T       cs=64     +0.00%  *NO WIN
Down-B32-M2048 wgrad      1644.3 T       cs=64     +0.00%  *NO WIN
Down-B32-M4096 wgrad      1995.1 T       cs=64     +0.00%  *NO WIN
GateUP-B32-M2048 wgrad    1840.9 T       cs=64     +0.00%  *NO WIN
GateUP-B32-M4096 wgrad    2166.9 T       cs=64     +0.00%  *NO WIN
```

Every cell: cs=64 is unique winner. cs=32 (2-chunk interleave)
regresses 0.41-1.19%; cs=96/128 (swizzle NO-OP) regresses 1.92-5.13%.

Conclusion: the chiplet swizzle's L2 reuse benefit is maximised when
ALL PIDs land on consecutive XCDs in a single chunk. Splitting into 2+
chunks interleaves the assignment across chunks and breaks locality.
chunk_size=64 with xcds=4 + slots=256 gives the maximal-locality
1-chunk partition. The R13 chunk_size lever is therefore RULE-LIMITED
to cells where the default cs=64 produces a NON-clean partition (only
the xcds=2 + slots=196 cell satisfies this in the FP8 metric universe).

## Audit 3 — GateUP-B32-M{2048,4096} fwd RCR (gm, xcds) wide retune (FALSIFIED)

The lowest-ratio shape in the R15 metric is GateUP-B32-M4096 fwd at
1.087 (2097 vs Triton 1929). Current rule (R70): gm=8, xcds=4, picked
from a (gm ∈ {1..24}) × (xcds ∈ {1,2,4,8,16,32}) sweep with +1.39pp
on M=4096. Multiple notes in this codebase document "kernel-rebuild
drift" (R30/R31/R32/R45/R50) where the optimum can shift between
binding rebuilds; today's binding hasn't been re-tuned for this cell
since R70.

Probe: scripts/_probe_round_15_gateup_b32_m4096_fwd_retune.py.
21-cell wide grid × 3 seeds × 1500-iter p20.

```
GateUP-B32-M4096 fwd                GateUP-B32-M2048 fwd
gm xcds  TFLOPS  Δ vs (8,4)         gm xcds  TFLOPS  Δ vs (8,4)
 8   4   2092.1  +0.00 *R70 win     8   4   2038.8  +0.00 *R70 win
 4   4   2063.6  -1.38              8   2   1985.4  -2.69
 8   2   2060.3  -1.55             14   4   1979.5  -2.99
10   4   2051.6  -1.98             16   4   1978.3  -3.06
12   4   2046.4  -2.23              6   4   1977.0  -3.12
14   4   2041.3  -2.49              4   4   1976.1  -3.17
 6   4   2037.5  -2.68             12   4   1969.8  -3.50
 4   8   2025.4  -3.29             10   4   1964.3  -3.79
 2   2   2001.8  -4.51              4   8   1957.5  -4.15
 ... 12 more cells -4.6 to -10.3% ...
```

R70's (gm=8, xcds=4) is still the unique top on BOTH shapes; every
other cell -1.38..-10.31% off. Today's binding has NOT shifted the
optimum from R70 — the cell is at the dispatcher ceiling.

## Audit 4 — GateUP-B4-M4096 dgrad-via-H4 num_slots × chunk_size (FALSIFIED)

R10 set num_slots=200 for the SIBLING cell (M=2048, tiles_m=8) at
1.5 ws/CU. The M=4096 cell (R8 rule: gm=1, xcds=4) at 2.75 ws/CU
hadn't been audited. Probe sweeps num_slots ∈ {192, 200, 208, 220,
240, 256} × chunk_size variants (default 64 + clean-partition cs
matched to each slots count).

Probe: scripts/_probe_round_15_gateup_b4_m4096_dg.py.

```
slots   cs    TFLOPS    Δ vs default
def    def   2491.6     +0.00  *base (slots=256, cs=64)
256     32   2491.6     +0.00  (2-chunk, ties baseline)
256     16   2465.4     -1.06
200     50   2250.3    -10.73  (clean 1-chunk at slots=200)
192     48   2232.5    -11.61
208     52   2218.0    -12.34
... others -13 to -26% ...
```

Default unique winner. The 2.75 ws/CU is firmly above the num_slots-
help threshold (which lies between 1.5 and 2.5 ws/CU based on the R3
/ R9 / R10 / R15 evidence). Even paired with cleanly-aligned
chunk_size (slots/xcds = clean partition), the parallelism loss
dominates.

## Refined ws/CU threshold for num_slots lever (R15 update)

| Cell | ws/CU | num_slots win? | Source |
|------|-------|----------------|--------|
| Down-B4-M2048 fwd+dgrad-via-H4 RCR | 1.50 | YES (+5%) | R9, R11 |
| GateUP-B4-M2048 dgrad-via-H4 RCR   | 1.50 | YES (+3.08%) | R10 |
| Down-B4 wgrad var-K (gen)          | 1.89 | YES (+5-6%) | R3 |
| GateUP-B4-M4096 dgrad-via-H4 RCR   | 2.75 | NO  | R15 |
| Down-B4-M4096 fwd RCR              | 3.0  | NO  | R12 |
| GateUP-B4 wgrad var-K              | 3.78 | NO  | R15 |
| GateUP-B32-M4096 fwd RCR           | 22.0 | NO (assumed) | high density |

The threshold for num_slots help is firmly in the **1.5 < ws/CU <
2.75** band. Cells at 2.75+ ws/CU are saturated and num_slots
reduction always loses parallelism without compensating amortisation
gain.

## Refined chunk_size threshold for RCR/var-K lever (R15 update)

The chiplet-swizzle ``chunk_size`` lever requires the cell to produce
a NON-clean partition at the default cs=64. The condition is:

  block_default = num_xcds * 64
  limit_default = (slots / block_default) * block_default

If `limit_default == slots`, the default is already clean → cs lever
can't help (R15 confirmed on 6 var-K wgrad cells with xcds=4+slots=256).

If `limit_default < slots` (e.g. xcds=8+slots=200 → block=512>slots →
limit=0 → swizzle NO-OP), an aligned chunk_size CAN produce a clean
partition AND the existing default leaves work mis-scheduled. R13
shipped Down-B4 wgrad (xcds=2+slots=192 → limit=128, leaving 64/192
unchunked). R14 shipped Down-B4-M2048 fwd (xcds=2+slots=196 →
limit=128, leaving 68/196 unchunked).

In the gpt_oss FP8 metric universe, only those 2 cells satisfy the
"non-clean default" condition. The chunk_size lever is therefore
EXHAUSTED.

## Next-direction guidance for R16+

1. **Dispatcher levers exhausted.** All FP8 RCR / var-K rule cells in
   the 24-shape MoE suite have been audited at the (gm, xcds, slots,
   chunk_size) granularity. The remaining 31% gap to 2800T target
   per-section requires kernel-source changes.

2. **Kernel surgery candidate list (R16 attack order, by expected
   impact + safety)**:

   a. **K-tail epilog optimisation** — every gpt_oss shape pays a
      K-tail (K=2880, K%128=64). The fused-K-tail epilog (path B,
      FUSED_KTAIL=true) currently uses `raw_buffer_load_b128` per
      lane. Better SRD or bf16-pack-to-fp8 conversion ordering may
      shave 1-2% off every shape.

   b. **B-pack prefetch reorder for var-K** — the var-K kernel's
      B-pack (per-K column slab) is currently loaded just-in-time
      per K-block. Pre-loading 1-2 K-blocks ahead may close 1-2% of
      memory stall on the saturated B=32 wgrad cells.

   c. **LDS metadata vec2 repack** — the per-CTA group-metadata LDS
      caches (s_offs / s_cum_tiles) are loaded as int32 scalars
      (R9-dm parallel HBM init). Repacking as int64-vec2 doubles the
      HBM transaction width for the same volume; ~1us savings per
      launch (probe needed).

   d. **R16's binary-search-to-divide REVERTED** in past rounds (37
      VGPR spill). Skip unless register pressure can be reduced
      first.

3. **The 6 unaudited wgrad cells (R15 audit 2) are at kernel-template
   ceiling**. Future probes should NOT re-test their (gm, xcds,
   slots, chunk_size) cells; only kernel-source changes can move
   them.

## Score impact: 0 (documentation-only round)

R14 baseline = 693, R15 entry = 693 (same), R15 exit = 693 (same).
Audit-only round; the value is in eliminating future-round candidate
cells.
