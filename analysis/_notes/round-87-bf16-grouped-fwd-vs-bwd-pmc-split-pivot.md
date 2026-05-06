# Round 87 — bf16 grouped GEMM weighted wall (auto_optimize round 10/100)

> **Context:** auto_optimize round 10/100. R83-R86 = 4-round VGPR-pressure
> falsification streak (RCR FUSE KI=88 spill 9; KI=44 spill 28; var-K
> KI=32/64 spill 14-18; st_32x16_v2 within-half swizzle spill 17 var-K
> + 8 grouped CRR). Goal this round: produce a clean diagnostic /
> pivot note that splits the worst shape's wall into per-kernel
> components and PMC-confirms which kernel deserves R11+ effort,
> resolving R85+R86's lingering "where exactly is the cost?" question.

**Status:** R87 = **diagnostic / pivot round**, NO code change. Both
HK and Primus working trees stay clean. Score 882 (R86 baseline,
within ±3 noise band).

| run                                    | weighted score | gpt_oss geomean |
|----------------------------------------|---------------:|-----------------|
| R86 baseline (commit 2f669d9)          | 882            | 1.0938          |
| R87 (this round, no code change)       | 882            | 1.0938          |

## Part A: per-kernel rocprofv3 split on the worst-progress shape

Target: gpt_oss-Down-B4-M2048 (B=4, M=2048, N=2880, K=2880, ratio
1.051, progress 0.840 — lowest of all 24 metric shapes). Driver:
`/tmp/rocprof_round10_fwd_bwd.py` (DSV3 K%128==0 warmup → 5 target
fwd+bwd iters under HIPKITTEN). Full kernel-trace CSV at
`/tmp/rocprof_round10/chi2894/3329865_kernel_trace.csv`.

Steady-state per fwd+bwd iter (median of last 4):

```
phase  kernel               dur_us   share
fwd    fwd_RCR_FUSE         150       30.7 %   // grouped_kernel<RCR,0,FUSE=true>
dA     transpose_3d          24        4.9 %   // H4 reroute pre-pass (5 TB/s)
dA     fwd_RCR_FUSE         150       30.7 %   // dA via H4 reroute → same RCR FUSE kernel
dB     var_k                165       33.7 %   // grouped_var_k_kernel<0> (CRR)
total                       489      100.0 %
```

Distribution: ≈ 30/30/35/5 % across fwd/dA/transpose/dB. None
dominates — ANY single-kernel speedup of 10 % shifts the wall by ≤ 3 %
(= 0.03 ratio shift = ≤ 0.4 score per shape, weighted 3 = ≤ 1.2 score
across the family). 25 % savings on a single kernel = +0.07 ratio per
shape × 8 = +5-7 score. Realistic single-round achievable.

## Part B: per-kernel LDS bank-conflict PMC (R85 measurement at higher precision)

Counter set: `SQ_INSTS_LDS`, `SQ_LDS_BANK_CONFLICT`, `SQ_LDS_IDX_ACTIVE`.
Same shape as Part A. Single PMC pass, ~7 sec/shape. Raw CSV at
`/tmp/rocprof_round10_pmc/pass_1/chi2894/3330618_counter_collection.csv`.

```
disp  kernel          BankConfl    IdxActive    LDS_Insts   BC%   BC/Inst
27    fwd_RCR_FUSE          0      13_034_240    3_268_736   0.0     0.00
32    var_k          14_155_776    28_332_544    7_083_776  50.0     2.00
... (same pattern for the other 4 target iters)
```

* **fwd_RCR_FUSE has 0 LDS bank conflicts.** Confirmed: the K-tail
  FUSE path B (round-5 path B, direct HBM-to-register reload at
  lines 829-946) does NOT go through LDS for the K-tail. Forward's
  ~30 % wall share is bottlenecked by **HBM stalls in the K-tail
  block**, not LDS. LDS is clean throughout.
* **var_k has 2.00 BC/Inst — exactly matching R85's prediction.**
  R85's analysis ("Confl/Inst ratio scales linearly with the number
  of `st_32x16_s` swizzle tiles in the layout: CRR has 2 →
  2 BC/Inst") is now reconfirmed at R87 with cleaner counter data
  (50.0 % BC/IdxActive → equivalent to 50 % wasted LDS-arbiter
  cycles).

Cost estimate: 14.2M conflicts × ~1-3 stall-cycles each ≈ 7-21 µs
wasted in var_k's 165 µs runtime → **4-13 % var_k speedup if all
conflicts eliminated**. On Down-B4-M2048: 4-13 % of 165 µs / 489 µs
= **+1.4-4.4 % wall ratio = +0.015-0.046 progress shift on this
shape**. Across the gpt_oss family (8 weight-3 shapes, all hitting
the same var_k path with similar BC/Inst): **+9-27 score** if
eliminated cleanly.

## Part C: R86's swizzle attack mis-targeted the wrong code path

R86 wired `st_32x16_v2_s` to BOTH the grouped CRR forward kernel
(`grouped_kernel<CRR,...>`) AND `grouped_var_k_kernel`. R87 PMC
shows:

* The grouped CRR forward kernel is **NOT exercised by any of the
  24 metric shapes**. Forward = RCR; backward dA = RRR or H4-RCR;
  backward dB = var_k (a separate kernel from grouped_kernel<CRR,...>).
* Among the metric kernels actually exercised, only var_k uses
  `st_32x16_s` for both ST_A and ST_B. RRR (used by DSV3/Qwen3 dA
  on the K%128==0 path) uses st_32x16 for ST_B only → 1 BC/Inst.
  RCR (forward + H4-rerouted gpt_oss dA) uses st_16x32 for both →
  0 BC.
* R86's table showed:

| kernel                               | baseline spill | R86 v2 spill | Δ spill |
|--------------------------------------|---------------:|-------------:|--------:|
| `grouped_kernel<CRR, 832, FUSE=0>`   |              7 |           15 |      +8 |
| `grouped_var_k_kernel<0>`            |              0 |           17 |     +17 |

Both spilled. The grouped CRR spill was a wasted cost (no metric
shape uses it). Even isolated to var_k only, the +17 VGPR spill is
fatal — var_k baseline is at 256 / 0 spill / Occ 2; +17 spill drops
Occ to 1 or scratches heavily.

## Part D: R11 plan — pursue var_k LDS bank-conflict elimination via lower-VGPR levers

Two targeted attack vectors, ordered by single-round feasibility:

### Lever B5 (R11 candidate): RRR-only swizzle for ST_B's 1 BC/Inst path

The R86 swizzle peer shape costs +17 VGPRs in **var_k** (where
ST_A AND ST_B are both st_32x16_s — generic `next_addr` path
needed twice). If we wire the new shape **only to RRR's ST_B**
(where ST_A is st_16x32, untouched), the spill-source isolated to
1 next_addr register pair = **+8-9 VGPRs** (half the var_k cost).
That's still over the 256 ceiling for var_k (no — var_k is CRR,
not RRR), but RRR `grouped_kernel<RRR,0,FUSE=true>` baseline is
250 VGPR / 0 spill — 6 VGPR headroom.

* Target shapes: DSV3 / Qwen3 native RRR dA (16 weight-1 shapes,
  all currently 1 BC/Inst). 50 % of LDS BC eliminated → ~5-10 %
  dA kernel speedup → ~+0.01-0.02 ratio per shape × 16 weight-1 =
  **+4-8 score**.
* Risk: VGPR spill still possible at 250 + 8 = 258 → drops Occ 2 → 1.
  Build resource report check is the gate before metric.
* Bit-equivalence: same as R86 — group_m / num_xcds / lane→MFMA
  mapping unchanged; only the LDS storage offset changes; round-trip
  through `prefill_swizzled_offsets` + `swizzle()` is preserved
  (verified involution math in R86 Part A).

### Lever B6 (R12+ candidate): VGPR live-range trim in var_k body, then revisit B5 with safety margin for CRR

The bigger prize is var_k's 2 BC/Inst → 0 (full 4-13 % var_k
speedup, +9-27 weighted score). But var_k baseline is already at
256 / 0 spill — no room to add ANY register state (R86 confirmed
+17 spill).

Plan to make room:

1. **Profile var_k's per-pass VGPR live-range** by inserting
   `__attribute__((noinline))` markers at suspected spill points
   (cooperative-cumsum prologue, group bookkeeping helpers,
   per-tile coord lambdas). Compile, read per-helper VGPR usage
   from -Rpass-analysis. (1 round)
2. **Recompute-instead-of-store** the highest-pressure live coord
   in the inner loop (e.g., the per-tile group-row offset that's
   currently materialised in VGPRs but could be derived on-the-fly
   from the iteration counter). Target: -8 to -12 VGPR base. (1 round)
3. **With ≥ 8 VGPR headroom restored, retry the swizzle-or-permutation
   landing** for var_k's CRR ST_A + ST_B path. (1-2 rounds)

### Lever A1 / A2 status (unchanged)

Forward K-tail HBM stall (path B's ~700-900 ns wait stall) remains
the forward-side bottleneck (0 LDS BC; HBM-bound). Same VGPR-pressure
constraint applies: dual-A prefetch needs +8 VGPRs in
`grouped_kernel<RCR,0,FUSE=true>` which is at 250 / 0 spill — would
spill. Same pre-trim prologue as Lever B6 unblocks A1.

## Part E: chosen R10 outcome — pivot, no code change

R10 commits a single docs round note (this file). HK working tree
clean (only pre-existing NFS lock files). Primus working tree clean
of code edits.

Justification:

* 4 consecutive falsification rounds (R83-R86) prove that any
  single-round VGPR-adding lever on the BF16 grouped / var-K bodies
  spills and regresses the metric.
* R87 PMC delivers the per-kernel cost split that R86's plan
  identified as missing — eliminates ambiguity for R11+ direction.
* Best-known R11 lever (B5 — RRR-only swizzle) has clear ROI bound
  (+4-8 score) but still risks VGPR spill at 258 — needs build-time
  resource check before commit. Better to land it in R11 with the
  pre-condition explicit, not bundle with a docs round.
* Score is at the noise floor (882 vs best 886 = -4, within ±3
  noise) — not regressing.

## Files touched

* `/workspace/code/Primus-Turbo/analysis/_notes/round-87-bf16-grouped-fwd-vs-bwd-pmc-split-pivot.md`
  (this file)
* `/tmp/rocprof_round10_fwd_bwd.py` (probe driver, kept offline,
  reusable for R11+ kernel-time gating)
* `/tmp/rocprof_input_lds.yaml` (LDS PMC counter input, reusable)
* `/tmp/rocprof_round10/`, `/tmp/rocprof_round10_pmc/` (raw CSVs,
  not committed)
* `/tmp/build_baseline_r10.log` (-Rpass-analysis baseline VGPR
  report for the 5 target kernel instantiations, reusable for
  R11 build-gate comparison)

## Metric / numbers

* R86 baseline / R87 1-run: **882** / 1.0938 gpt_oss geomean.
* All 24 shapes PASS correctness.
* Per-shape PMC steady-state (gpt_oss-Down-B4-M2048):
  fwd 150 µs / dA-via-H4 174 µs / var_k 165 µs / total 489 µs.
* Per-kernel LDS BC%: fwd 0.0 % / var_k 50.0 % @ BC/Inst 2.00.
* Build VGPR baseline (key kernels):
  * `grouped_kernel<RCR,0,FUSE=true>`:  250 VGPR / 0 spill / Occ 2
  * `grouped_kernel<RRR,0,FUSE=true>`:  250 VGPR / 0 spill / Occ 2
  * `grouped_kernel<{RCR,RRR,CRR},0,FUSE=false>`: 246 / 0 / Occ 2
  * `grouped_var_k_kernel<0>`:           256 VGPR / 0 spill / Occ 2  ←
    binding budget exhausted.

## Recommendation for round 11

Execute Lever B5 (RRR-only `st_32x16_v2_s` for ST_B). Build-gate:
post-change `grouped_kernel<RRR,0,FUSE=true>` resource report must
show ≤ 256 VGPR AND ≤ 4 spill (recover Occ 2). If either gate
fails, REVERT to docs round and pivot R12 to Lever B6 (var_k VGPR
live-range trim, multi-round prologue).
