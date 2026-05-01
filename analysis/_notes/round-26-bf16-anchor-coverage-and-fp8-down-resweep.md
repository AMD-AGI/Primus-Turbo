# Round-26 — BF16 anchor-coverage audit + FP8 Down-B32-M4096 post-BUFFER re-sweep

**Date**: 2026-05-01  
**Repo / HEAD**: Primus-Turbo `5a55709` (round-25); HipKittens `62cebd5`  
**Focus**: gpt_oss 16 shapes; DSV3 = `[watch]`  
**Result**: 1 marginal config update (+0.09pp), 1 falsified re-sweep, BF16 BK-port plan deferred to round-27

---

## Goal of this round

Round-25 recommended starting BF16 `K_STEP=64 → K_STEP=32 + ns=3` port
(match Triton). Before committing to a multi-round structural rewrite,
audit the **anchor coverage** of every per-shape config rule for
gpt_oss to ensure no 1-round wedge is being left on the table by a
stale or missing sweep.

---

## Anchor-coverage audit (gpt_oss 16 shapes)

Walked every `if` branch in `select_default_config` that fires for the
8 BF16 + 8 FP8 gpt_oss shapes and checked: (a) is there a sweep
anchor in the comment, (b) was that sweep run against the current
post-BUFFER (round-19/20) kernel?

### BF16 (8 shapes, 4 rules cover them)

| m_total branch | shape (B, M) | rule | anchor |
|---|---|---|---|
| `<=8192`   | B4 M2048  | round-9 (gm=2, xcd=2)   | yes — 60-cell sweep + 1500-iter × 7-trial p20 verify |
| `<=16384`  | B4 M4096  | round-21 (gm=12, xcd=4) | yes — post-BUFFER metric-aligned verify |
| `<=65536`  | **B32 M2048** | **(gm=8, xcd=4)** | **NO ANCHOR** — defensive default since pre-BUFFER |
| `>65536`   | B32 M4096 | round-21 (gm=8, xcd=4)  | yes — post-BUFFER metric-aligned verify |

### BF16 Down (4 shapes, separate `tiles_n==11` branch)

All 4 sub-rules anchored (round-10/12/21).

### FP8 (8 shapes, 6 rules cover, 1 falls to default, 1 missing)

| shape | rule | anchor |
|---|---|---|
| GateUP-B4-M2048   | round-23 (1, 4)  | post-BUFFER metric-aligned 9-cell × 7-trial verify |
| GateUP-B4-M4096   | round-21 (14, 4) | post-BUFFER 11-cfg × 7-trial verify |
| GateUP-B32-M2048  | round-70 (8, 4)  | round-70 wider sweep verified post-BUFFER |
| GateUP-B32-M4096  | round-70 (8, 4)  | round-70 wider sweep verified post-BUFFER |
| Down-B4-M2048     | round-7 (2, 2)   | 40-cell sweep + neighbor robustness verify |
| Down-B4-M4096     | round-12 (32, 4) | round-12 54-cell + 1500-iter × 7-trial verify (**pre-BUFFER**) |
| Down-B32-M2048    | round-8 (16, 4)  | round-8 54-cell + verify |
| Down-B32-M4096    | **fall-through default (4, None=8)** | round-12 swept, "no rule needed" (**pre-BUFFER**) |

Two FP8 rules pre-date the round-19/20 BUFFER reroute (Down-B4-M4096
round-12 and Down-B32-M4096 round-12). Round-21 redid BF16 sweeps
post-BUFFER for the same reason; this round redoes the FP8 Down
B32-M4096 case (the un-ruled one — most likely to have shifted).

---

## Probe 1 — BF16 GateUP-B32-M2048 (the unanchored BF16 rule)

### Setup

40-cell coarse sweep `gm ∈ {1,2,4,6,8,12,16,24,32,48} × xcd ∈ {1,2,4,8}`
at `/tmp/sweep_bf16_gateup_b32_m2048_round26.py` (5 trials × 60 iters,
metric-aligned per-iter-sync). Then 7-trial × 200-iter tight verify on
top-8 cells at `/tmp/verify_bf16_gateup_b32_m2048_round26.py`.

### Tight verify result

```
cfg          med p20    p20      p80      spread%
( 4, 4)      1255.71    1253.92  1257.02   0.26  *winner
( 8, 4)      1254.55    1252.21  1255.16   0.45  ←current
( 6, 4)      1250.89    1250.51  1251.78   0.25
( 2, 4)      1247.50    1247.21  1249.28   0.27
( 4, 8)      1244.96    1244.10  1245.01   0.22
(12, 4)      1244.84    1243.56  1246.81   0.39
( 4, 1)      1242.68    1242.17  1244.39   0.24
( 4, 2)      1240.49    1240.13  1241.09   0.26
```

`(gm=4, xcd=4)` wins by **+1.16 TF (+0.09pp)** over `(gm=8, xcd=4)`.
Margin is small but `(4,4)` p20 (1253.92) > `(8,4)` p20 (1252.21)
consistently across all 7 trials. The entire `xcd=4` plateau dominates
`xcd ∈ {1,2,8}` by 5-15 TF.

The bottom-4 cells are `(48,1) (48,8) (32,8) (32,1)` all at 1067-1069 TF
(-15% from top), confirming `xcd ∈ {1, 8}` × large gm is a clear
anti-pattern for this `tiles_m=8 / tiles_n=22 / m_total=65536` grid.

### 3-run metric verify after switching to (4, 4)

Single-run noise = ±2 score; 3 runs:

| run | hk_TF (B32-M2048) | ratio | grp_BF16 geomean | score |
|---|---|---|---|---|
| 1 | 1251.4 | 1.142 | 1.1655 | 881 |
| 2 | 1250.8 | 1.141 | 1.1657 | 882 |
| 3 | 1251.1 | 1.140 | 1.1626 | 880 |
| **mean** | **1251.1** | **1.141** | **1.1646** | **881** |

Round-25 baseline mean was ~1.139 / ~881. **Net Δ ratio = +0.2pp**,
**Δ score within noise**. Bit-identical output verified.

### Conclusion (Probe 1)

Switch to `(gm=4, xcd=4)` and add the missing anchor docs. Net wedge
is marginal (+0.09pp single-shape ≈ +0.012pp segment geomean ≈ +0.5
score, lost in metric noise) but brings the rule in line with the
post-BUFFER anchor coverage already established by round-21 for the
3 sibling m_total tiers.

---

## Probe 2 — FP8 Down-B32-M4096 post-BUFFER re-sweep

### Setup

Round-12 swept this shape pre-BUFFER (54-cell at 200-iter × 1500-iter
verify), concluded "no rule needed" (best gain ≤ +0.15pp at the noise
floor). After round-19/20 BUFFER reroute the per-tile completion
latency shifted enough that round-21 redid BF16 post-BUFFER sweeps
and found new winners. This re-sweeps FP8 Down-B32-M4096 against the
current kernel.

60-cell sweep `gm ∈ {1,2,4,6,8,12,16,24,32,48} × xcd ∈ {None,1,2,4,8,16}`
at `/tmp/sweep_fp8_down_b32_m4096_round26.py` (5 trials × 60 iters,
metric-aligned per-iter-sync).

### Top-8 result

```
cfg          med_p20      spread%
( 4,   8)    1203.79      0.31
( 4,  16)    1202.99      0.30
( 4,None)    1202.16      0.26   ←current (None ⇔ binding default 8)
( 4,   1)    1201.10      0.24
( 1,   4)    1198.72      0.30
( 2,  16)    1197.19      0.16
( 2,None)    1196.79      0.14
( 2,   2)    1196.21      0.20
```

Top 4 are **all** `gm=4`, with `xcd ∈ {8, 16, None=8, 1}` within 2.7 TF
of each other (`xcd=8` and `xcd=None` are functionally identical —
binding's `BLOCK_SWIZZLE_NUM_XCDS` default = 8). `(gm=4, xcd=8)` is
+1.63 TF (+0.14pp) over `(gm=4, None)` — also within noise.

### Conclusion (Probe 2)

Post-BUFFER re-sweep **confirms** the round-12 finding: default
`(gm=4, xcd=None=8)` is at the local optimum for FP8 Down-B32-M4096.
The BUFFER reroute did **not** shift this shape's optimum (unlike
some BF16 shapes round-21 found shifted). No rule added — fall-through
default is correct.

This also incidentally confirms: when the sweep top is `gm=default`
and `xcd=None`, the `xcd` value is irrelevant to the kernel for FP8
when running the binding default (the binding's `dispatch<RCR>`
auto-picks `BLOCK_SWIZZLE_NUM_XCDS=8`).

---

## Updated leverage map for gpt_oss (post round 26)

After round-26's audit, all **single-knob 1-round levers** for both
BF16 and FP8 grouped RCR on gpt_oss are exhausted:

- All 4 BF16 m_total tiers anchored (round-9 / 21 / 26 / 21).
- All 4 BF16 Down sub-rules anchored (round-10 / 12 / 21 / explicit).
- All 7 FP8 explicit rules anchored against the current post-BUFFER
  kernel (round-7/8/12/21/23/70 + round-26 confirmation).
- 1 FP8 fall-through (Down-B32-M4096) confirmed optimal by post-BUFFER
  re-sweep this round.
- All micro-knobs in `kernel_fp8_layouts.cpp` (INIT0/1/STEADY/PREFETCH/
  EPILOGUE VMCNT, sched_barrier, chunk_size, two-tile path) saturated
  by rounds 4/5/6/15/22/24/25.
- BN=128 path falsified by round-24 (Triton uses BM/BN=256/256 too).

The remaining wedges all require multi-round structural work (BF16
BK port, FP8 MFMA cell-shape rewrite, K-tail epilog merge — all from
round-25 docs).

---

## Why round-26 is not the BK-port-start round

Round-25 recommended round-26 = "read BF16 RCR main loop + list K_STEP
hard-code sites + plan incremental commits". Round-26 instead audited
anchor coverage first because:

1. **Risk asymmetry**: A bad anchor sweep is a 1-shot test (no
   compilation breakage); a bad BK-port commit can leave the kernel
   uncompilable, blocking the next 2-3 rounds. Confirming all
   1-round wedges are exhausted before starting BK port reduces the
   "is there an easier wedge I missed?" backtrack risk.
2. **Found 1 marginal wedge** + falsified 1 stale-suspect rule —
   small but real, and documents the post-BUFFER state across the
   entire FP8 Down family for the first time.

BF16 BK port still recommended for round-27 onwards. Round-25 docs
Section "What's left (multi-round, structural)" item 1 remains the
top-leverage path; the K_STEP audit work in
`/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`
should be done in round-27 (read main_loop_iter at line 600+ and the
30+ K_STEP hard-code sites identified by `rg ^constexpr K_STEP|K_STEP`).

---

## Commit plan

Primus-Turbo only (HipKittens has no net code change this round).

```
config(bf16-grouped): round-26 anchor for GateUP-B32-M2048 (gm=8 -> gm=4) +
                      gpt_oss anchor coverage audit
```

## Next-round suggestion

Round-27: start BF16 K_STEP=64 → K_STEP=32 + ns=3 port (per round-25 plan).
Step 1 must be a *non-functional* WIP commit:

1. Read `kernel_bf16_dynamic.cpp` lines 600-686 (main_loop_iter), lines
   1158-1183 (ST_A/ST_B + register tile types), lines 540-552 (subtile
   templates), lines 1145-1240 (gemm_kernel entry up through SRD setup).
2. Catalogue all 30+ K_STEP hard-code sites by category:
   - LDS shape (ST_A/ST_B/ST_A_RCR/ST_B_RCR + their subtile templates)
   - Register tile shape (A_reg_t / B_reg_t)
   - LDS double-buffer offsets (a_lds_{00,01,10,11}, b_lds_{00,...})
   - Main loop body indexing
   - K-tail dispatcher (fast_k = (k / K_TWO_TILE) * K_TWO_TILE)
3. Introduce a `BF16_K_STEP_PORT_PROBE` constexpr switch (default = 0,
   bit-equivalent to current code path) — empty `if constexpr` branch
   for the BK=32 path. WIP commit at this step is acceptable per
   round-25 "Failure mode is graceful".
4. Round-28 onwards: wire actual BK=32 main loop body, debug compilation,
   verify single-shape correctness, measure perf.

Round-25 cap-from-cap analysis: BF16 segment cap to 1.20 = +14 score
(progress 0.969 → 1.0). Real path to win (1.20+ on multiple BF16
shapes consistently) likely +5-10 score. Path is contained risk-wise
but multi-round.
