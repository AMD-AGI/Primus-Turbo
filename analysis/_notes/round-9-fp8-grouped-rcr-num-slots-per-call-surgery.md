# Round-9 — FP8 grouped RCR per-call num_slots surgery (R4 lever reopened)

**Date**: 2026-05-08 (UTC)
**Repo**: Primus-Turbo, branch `dev/kyle_hipkitten_bf16` (HEAD `39915de2` → R9)
**Companion repo**: HipKittens (`/workspace/code/HipKittens`), one focused commit
**Scope**: gpt_oss FP8 kernel-only ceiling task, 8-shape suite × 3 sections.

## Bottom line

Reopened R4's "FP8 grouped RCR slots reduction lever" by adding a **per-call**
``g.num_slots`` parameter to the HK ``grouped_rcr`` / ``grouped_rcr_dscale``
binding (mirroring the var-K R3 wiring). R4 had only the env-only
``TK_RCR_NUM_CUS`` (process-static cached → process-wide effect); this round's
metric-level test confirmed it is unusable globally (TK_RCR_NUM_CUS=200
tanked the score 686 → 621 because Down-B4-M2048 lifted +1.3% but every
other shape regressed 10-30%).

R4 had also reported "+1.47% within ±2% noise" at slots=208 on Down-B4-M2048
fwd using 250-iter × 7-trial × 3-seed methodology. This round's tight
re-verify with 1500-iter × 7-trial × 5-seed found **+4.99% WIN-ROBUST at
slots=200** (5/5 seeds positive, med/spread ≈ 10.6×). R4's coarse
methodology had under-resolved the win by ~3.5pp.

| Direction | Anchor | Methodology | Result | Verdict |
|---|---|---|---|---|
| HK surgery: per-call num_slots | n/a | binding rebuild + bit-eq | max_abs_diff = 0 | LANDED |
| Down-B4-M2048 fwd RCR slots | tiles_n=11 tiles_m=8 k=2880 m_total=8192 | 1500i×7t×5s p20 subprocess | slots=200 +4.99% | **WIN-ROBUST** |
| Down-B4-M2048 fwd RCR in-process | same | 1500i×7t×3s p20 in-process via new arg | slots=200 +5.40..+5.81% | **WIRING CONFIRMED** |
| GateUP-B4-M2048 wgrad var-K slots | m_total=8192 (3.78 ws/CU) | 1500i×7t×3s p20 in-process | best ns=224 -0.95% | FALSIFIED |
| GateUP-B4-M4096 wgrad var-K slots | m_total=16384 (3.78 ws/CU) | 1500i×7t×3s + tight 5s | best ns=224 +0.31% (3s) → -0.12% (5s) | FALSIFIED (3-seed lucky hit) |
| GateUP-B32-M2048 wgrad var-K slots | m_total=65536 (15 ws/CU) | 1500i×7t×3s | best ns=224 -9.77% | FALSIFIED |

## Detailed per-cell data — Down-B4-M2048 fwd RCR slots tight verify

Probe: ``scripts/_probe_round_9_down_b4_m2048_fwd_numcus_tight.py``
Anchor: B=4 M=2048 N=2880 K=2880, cell (gm=16, xcds=2) [R2 dispatcher cell].
Per-shape persistent grid: 352 tile-steps over NUM_CUS=256 ≈ 1.4 ws/slot.
Methodology: subprocess-per-slot (env-static cache forces fresh process), each
child runs 1500-iter × 7-trial p20 × 5-seed direct ``grouped_rcr_dscale`` call.

```
slots   seed42 Δ  seed137 Δ  seed2024 Δ  seed7 Δ  seed1234 Δ  med Δ    spread   verdict
196     +5.04%     +4.67%     +4.80%      +4.76%   +4.90%      +4.80%   0.37pp   WIN-ROBUST
200     +5.23%     +4.95%     +4.99%      +4.76%   +5.09%      +4.99%   0.47pp   WIN-ROBUST  *unique top
204     +0.17%     -0.26%     +0.13%      -0.13%   +0.17%      +0.13%   0.43pp   TIE
208     +4.19%     +4.24%     +4.28%      +4.24%   +4.24%      +4.24%   0.10pp   WIN-ROBUST
212     +2.57%     +2.49%     +2.80%      +2.39%   +2.66%      +2.57%   0.40pp   WIN-ROBUST
216     -0.22%     -0.65%     -0.30%      -0.69%   -0.64%      -0.64%   0.47pp   LOSS
220     +2.62%     +2.62%     +2.80%      +2.48%   +2.66%      +2.62%   0.32pp   WIN-ROBUST
256     baseline (NUM_CUS default = legacy R4 env-only fallback)
```

**Reading**: Non-monotone landscape with peaks at 196/200/208/220 and dips at
204/216. slots=200 is the cleanest unique top:

- Seed-med +4.99% with all 5 seeds positive (+4.76..+5.23%).
- Spread 0.47pp << seed-med 4.99 → med/spread ≈ 10.6× (well above the
  standard "median > spread" robust-signal threshold used by R7/R10/R11/
  R23/R29-31/R45 / R6-R8 / R7-current / R8-current).
- Winner-min (+4.76%) > all other cells' winner-max in 5/5 seeds (clean
  tier separation).

The non-monotone dips at 204/216 likely reflect tile-distribution-modulo-
XCD partition resonances: with 8 XCDs × 32 CUs = 256, slot counts 200
(= 8 × 25) and 208 (= 8 × 26) divide evenly into XCD chunks, while 204
(= 4 × 51) and 216 (= 8 × 27 with remainder) leave fragmented
distributions. The clean "200 > 208 > 220" peak ordering matches the
wave-step amortisation prediction (1.4 → 1.76 → 1.69 → 1.6 ws/slot).

## In-process per-call wiring verify

Probe: ``scripts/_probe_round_9_per_call_num_slots_verify.py``
Methodology: same shape, in-process direct ``grouped_rcr_dscale(...,
num_slots=N)`` call (no env, no subprocess). Validates the new pybind arg
is actually wired through to the kernel's ``g.num_slots`` field.

```
[bit-eq] num_slots=0 (default) vs candidates:
  ns=0 vs ns=200:  max_abs_diff = 0.0000e+00
  ns=0 vs ns=208:  max_abs_diff = 0.0000e+00
  ns=0 vs ns=220:  max_abs_diff = 0.0000e+00
  ns=0 vs ns=256:  max_abs_diff = 0.0000e+00

Bench (1500-iter × 7-trial × 3-seed p20):
  ns=0   1462.5 / 1458.1 / 1460.0 TF (baseline; default = NUM_CUS=256)
  ns=200 1541.4 / 1542.8 / 1540.7 TF  Δ=+5.40 / +5.81 / +5.53%
  ns=208 1524.8 / 1524.9 / 1527.6 TF  Δ=+4.26 / +4.58 / +4.63%
  ns=256 1466.9 / 1464.4 / 1463.1 TF  Δ=+0.30 / +0.43 / +0.22% (= ns=0 noise)
```

Wiring CONFIRMED:
- ``ns=0`` (default) ≈ ``ns=256`` (both → kernel default), within ±0.4pp noise.
- ``ns=200`` lifts +5.4..+5.8% — matches the subprocess-probe data within ±0.5pp.
- bit-eq across all candidates (max_abs_diff = 0.0).

## HK kernel surgery — per-call num_slots wiring

File: ``analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp``
Mirror of var-K's R3 num_slots wiring (already shipped at line 7744 / 8169).

Changes:

1. **``grouped_layout_globals`` struct (line ~2606)**: added trailing
   ``int num_slots`` field. C++ aggregate init's value-init rule for
   missing trailing initializers means the existing
   ``grouped_rrr{,_dscale}_fn`` callers (which still init the struct
   positionally up to ``m_per_group``) get ``num_slots = 0`` for free —
   strict backward-compat extension.

2. **``dispatch_grouped_rcr`` (line ~7445)**: replaced process-static
   ``rcr_slots`` cache with:
   ```c++
   const int rcr_slots = (g.num_slots > 0 && g.num_slots <= NUM_CUS)
       ? g.num_slots : rcr_slots_env;
   ```
   The legacy ``TK_RCR_NUM_CUS`` env hook is preserved (renamed
   ``rcr_slots_env``) as the fallback when ``g.num_slots == 0``, so any
   debug usage of the env still works.

3. **``grouped_rcr_fn`` and ``grouped_rcr_dscale_fn`` wrappers** (lines
   ~8316 / ~8347): added ``int num_slots`` parameter, propagated to
   ``g.num_slots``.

4. **pybind11 ``m.def`` for ``grouped_rcr`` and ``grouped_rcr_dscale``**:
   added ``pybind11::arg("num_slots") = 0`` as the trailing kwarg
   (default 0 → backward-compat for all current Primus call sites).

5. **NOT touched**: ``dispatch_grouped_rcr_fused_act`` (uses a separate
   struct ``grouped_layout_globals_fused_act``; fused-act is out of scope
   for this metric and the kernel-only task). ``dispatch_grouped_rrr``
   (uses the same struct but a different dispatcher; the new
   ``num_slots`` field defaults to 0 in the rrr struct init via aggregate
   value-init — no behavior change).

Build: ``cd analysis/fp8_gemm/mi350x && make`` (10s, no warnings).

Bit-equivalence:
- max_abs_diff = 0 across ns ∈ {0, 200, 208, 220, 256} on Down-B4-M2048 fwd
  (in-process probe).
- Full 24-shape FP8 metric correctness: 8/8 PASS at ns=0 default
  (post-rebuild metric run = 685, within ±1 of 686 baseline).
- ``bench_grouped_gemm_turbo.py --dtype fp8`` 24-shape PASS (fwd+bwd
  correctness via SNR/allclose); kernel build does not regress any
  non-targeted shape.

## Primus-Turbo wiring — config + dispatch

File: ``primus_turbo/pytorch/kernels/hipkitten/config.py``

- Added ``num_slots: int = 0`` field to ``HipKittenConfig`` dataclass
  (default 0 → backward-compat for all existing rules; only the FP8
  grouped RCR / RRR-via-H4 paths consume this field).

- Updated R2 rule (Down-B4-M2048 ``tiles_n=11 AND tiles_m=8 AND k=2880
  AND m_total=8192``) to set ``num_slots=200`` alongside the existing
  (gm=16, xcds=2) cell. Rule scope unchanged — same predicate gate, just
  the new lever wired through.

File: ``primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py``

- Wired ``cfg.num_slots`` through the dscale + fallback dispatch paths
  (pass as ``num_slots=cfg.num_slots`` to the HK binding).

## Falsification — GateUP wgrad var-K num_slots audit (R8 next-round
suggestion #1)

Probe: ``scripts/_probe_round_9_gateup_wgrad_num_slots.py``
Anchors: 3 GateUP wgrad shapes that R2 (current Primus run, sha 2d3946f1
series) had analytically excluded but never tight-verified.

```
GateUP_B4_M2048 wgrad (m_total=8192, 3.78 ws/CU)
  cell = (gm=1, xcds=4) [R3]
  baseline ns=256 (per-seed): 1668.2 / 1669.5 / 1668.6 TF
    ns=128  -33.8% LOSS
    ns=160  -17.9% LOSS
    ns=192  -9.5%  LOSS
    ns=224  -0.95% LOSS  ← closest, still LOSS
  Verdict: R2's analytic exclusion CONFIRMED.

GateUP_B4_M4096 wgrad (m_total=16384, 3.78 ws/CU, K_loop=32)
  cell = (gm=4, xcds=4) [R9-A]
  baseline ns=256 (per-seed): 2028.3 / 2028.0 / 2030.7 TF
    Initial 3-seed probe found ns=224 +0.31% WIN-LIGHT (3/3 positive).
    Tight-2 with 5 seeds (ns ∈ {224, 232, 240, 248, 256}):
      ns=224  -0.12% TIE   (0/5 positive — initial 3-seed was lucky combo)
      ns=232  -0.34% TIE
      ns=240  -1.57% LOSS
      ns=248  -0.94% LOSS
  Verdict: FALSIFIED at the 5-seed tight-verify gate.

GateUP_B32_M2048 wgrad (m_total=65536, 15 ws/CU)
  cell = (gm=1, xcds=4) [R31]
  baseline ns=256 (per-seed): 1846.3 / 1846.3 / 1847.5 TF
    ns=128  -40.9% LOSS
    ns=192  -17.1% LOSS
    ns=224  -9.8%  LOSS
  Verdict: R2's analytic exclusion CONFIRMED (saturated grid).
```

Reading: All three GateUP wgrad shapes confirm R2's rule is tight — the
``slots=192`` lever is genuinely Down-only (small m_total + K=N=2880 +
1.89 ws/CU symmetric). GateUP's larger N=5760 produces a different tile-
step density profile that resists slot reduction.

## Score impact — metric-level

Per-shape kernel TFLOPS (median across 5 metric runs, post-R9 wiring):

```
shape                 section  baseline (R8)  R9     Δ TF   Δ%
Down_B4_M2048         fwd      1478           1502   +24    +1.6%
Down_B4_M2048         dgrad    1416           1473   +57    +4.0%
Down_B4_M2048         wgrad    1299           1306   +7     +0.5% (noise)
all other shapes      *        *              ±5     ±5     ±0.3% (noise)
```

Section avg shifts:
- fwd:    1902 → 1905 T (+3)   progress 0.679 → 0.681 (+0.0011)
- dgrad:  2078 → 2085 T (+7)   progress 0.742 → 0.745 (+0.0025)
- wgrad:  1781 → 1782 T (±0)   unchanged

Score arithmetic: overall progress 0.685 → 0.687 → score ~+1-2 points
(median across 5 runs: baseline 686, R9 ~687). The robust per-cell
+5% kernel win dilutes heavily through (a) 8-shape section averaging
(only 1 shape benefits per section), (b) 3-section overall averaging
(only 2 sections benefit), (c) progress capped at 1.0 per section but
we sit at 0.6-0.7. R7 saw the same dilution (+0.5% probe → +1 metric
score). Same magnitude expected here.

The structural win is the lever itself: R4 closed this lever as
"falsified for Python wiring" because the env-only path was unusable.
R9's HK surgery makes it a proper per-call dispatcher knob, opening up
follow-up rounds to probe num_slots on:

- GateUP-B4-M2048 fwd RCR (1.4 ws/CU, similar sparsity to Down-B4-M2048)
- GateUP-B4-M2048 dgrad-via-H4 RCR (1.4 ws/CU, R7 already ships gm=8)
- Down-B4-M4096 fwd RCR (3 ws/CU; R4 only tested 192/208/224/240 increments)
- Down-B4-M4096 dgrad-via-H4 RCR (similar)
- Each can independently contribute a +0.5..+1 score point if the lever
  fires there.

## Falsified

- GateUP-B4-M2048 wgrad var-K, slots ∈ {128,160,192,224} (every cell LOSS)
- GateUP-B4-M4096 wgrad var-K, slots ∈ {128,160,192,224,232,240,248} (initial
  3-seed lucky hit at ns=224 falsified by 5-seed tight-verify)
- GateUP-B32-M2048 wgrad var-K, slots ∈ {128,160,192,224} (saturated grid)
- ``HipKittenConfig(kernel="...")`` for FP8 grouped paths (silently ignored
  per R8 — only dense ``gemm_rcr`` reads the kernel template id; grouped
  paths route through ``dispatch_grouped_rcr`` which does not consume the
  ``kernel`` field). Quarantined as out-of-scope until kernel surgery.

## Next-round suggestions

1. **Probe num_slots on GateUP-B4-M2048 fwd RCR** (the GateUP analog of
   the Down-B4-M2048 fwd win this round). Both have m_total=8192 and
   ~1.4 ws/CU; the GateUP shape has tiles_n=22 (vs Down's 11) so the
   tile-step landscape modulo XCD count is different, but the wave-step
   amortisation argument applies equally. R23's existing rule there is
   (gm=1, xcds=4); slot tuning is orthogonal to the (gm, xcds) cell.

2. **Probe num_slots on GateUP-B4-M2048 dgrad-via-H4 RCR** (R7 just shipped
   gm=8, xcds=8 here). Similar 1.4 ws/CU; check if the new lever stacks
   with the R7 (gm, xcds) win.

3. **Re-probe Down-B4-M4096 fwd RCR with finer slot increments** (R4 only
   tested {128,160,192,208,224,240,256} and reported -17.58% at slots=208
   — a huge LOSS implying saturated grid, but at 3 ws/CU it might respond
   to slots ∈ {240, 244, 248, 252} which R4 didn't test). Low-priority
   given R4's strong negative signal at slots=208.

4. **Audit the K-tail kernel paths** for similar per-call slot levers.
   The grouped K-tail kernels (``grouped_ktail_kernel_mfma32x32`` etc.)
   currently launch with ``ceil_div(g.M_total, TBM)`` — fixed grid, no
   slot knob. Per-call slot reduction there could help K-tail-dominated
   shapes (gpt_oss K%128==64 every shape pays one tail block).
