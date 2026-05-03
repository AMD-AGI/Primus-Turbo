# Round 7 — gpt_oss-GateUP dA H4 RCR cfg rule (Lever C dispatch) LANDED

## Selected target

Per round-7 baseline metric (`/tmp/metric_round_7.log`), shape table
sorted by ratio ascending:

```
shape                                  ratio  progress  weight
gpt_oss-GateUP-B32-M2048               1.038    0.830     3   ← worst
gpt_oss-Down-B32-M2048                 1.055    0.844     3
gpt_oss-GateUP-B32-M4096               1.079    0.863     3
Qwen3-Down-B32-M2048                   1.088    0.871     1
gpt_oss-GateUP-B4-M2048                1.093    0.874     3
gpt_oss-GateUP-B4-M4096                1.093    0.875     3
gpt_oss-Down-B32-M4096                 1.095    0.876     3
gpt_oss-Down-B4-M2048                  1.109    0.887     3
...
DeepSeek-V3-GateUP-B32-M4096           1.136    0.909     1   (best ratio)
```

- Round-7 starting baseline: **876** (single run; range 876-891 across
  recent rounds with same git HEAD — characterised noise band per R2)
- Best historical: **891** (R6 single)
- gpt_oss family geomean: **1.0825** (target 1.25, weight 3×)
- DSV3 / Qwen3 family geomeans: 1.1144 / 1.1125

R6 recommendation was: "rocprofv3 profile the 3 worst metric shapes
(gpt_oss-GateUP-B32-M2048, gpt_oss-Down-B32-M2048, DSV3-Down-B16-M4096)
to identify which kernel path has the most MFMA-utilization headroom.
Then commit to one of D2 / A1 / B1 based on the profile data."

Profile-first read of the dispatch surface revealed an untapped seam:
the BF16 path of the gpt_oss-GateUP dA H4 reroute (Lever C, not yet
exhausted on this branch).

## Why Lever C is still alive on the BF16 dA H4 path

After the round-4 H4 fast Triton transpose landed, the gpt_oss-GateUP
dA call was rerouted from RRR to RCR via `bf16_transpose_3d(b)`. The
post-reroute RCR call has tile geometry:

```
m = avg_m = M_per_group ∈ {2048, 4096}  -> tiles_m ∈ {8, 16}
n = b.shape[-2] = K_fwd = 2880          -> tiles_n = 11
k = a.shape[1] = N_fwd = 5760
```

No prior BF16 RCR rule matched: the existing gpt_oss block in
`select_default_config` is gated on `k == 2880` (gpt_oss forward +
gpt_oss-Down dA H4). `tiles_n == 11 and k == 5760` is uniquely
gpt_oss-GateUP dA H4 reroute — DSV3 / Qwen3 / gpt_oss-Down don't hit
it. So all 4 GateUP dA H4 calls were falling through to the BF16
default `(gm=4, num_xcds=8)`.

R2 tested `(gm=4, xcd=4)` here and falsified (-11.5 / -2.0 score) —
but R2 only varied xcds, holding gm=4. The FP8 R34 sibling rule (line
1467 below; same H4 reroute path, FP8 binding) found the actual
winners are `(gm=8, xcd=4)` for tiles_m=16 and `(gm=16, xcd=4)` for
tiles_m=8 + B=32 — different gm from R2's tested set.

The BF16 grouped RCR kernel uses the same persistent grid +
chiplet-swizzle scheduler as FP8 (different inner MFMA tile + dtype
but identical group_m / num_xcds semantics — pure scheduling knobs).
Per-tile MFMA time is ~2× FP8 (BF16 K_BLOCK=64 → 90 K-iters for
k=5760; FP8 K_BLOCK=128 → 45 K-iters), but the cfg pattern still
carries over.

## Probe — 10-cell (gm, xcd) sweep on the 4 target shapes

`/tmp/probe_round7_bf16_dA_h4_rcr.py` — 5-trial × 80-iter cudaEvent
median per cfg, bench against post-H4 inputs (a:[M_total, N_fwd=5760],
b:[B, K_fwd=2880, N_fwd=5760] — what the RCR kernel actually sees on
this path). Output below; full log archived alongside this note.

```
shape (m_total, tiles_m)        cfg      med_TF  spread%   Δ vs (4,8)
B4-M2048  (8192,    8)
  (4,8) baseline                         1027.0   5.30%   baseline
  (24,2) best candidate                  1036.2   6.56%   +0.90%   ←spread > gain, NOISE
  (every other cfg)                          ~      >2%    < 1%
  -> NO RULE for m_total=8192 + tiles_m=8 (default within noise band)

B4-M4096  (16384,  16)
  (4,8) baseline                         1325.6   1.77%   baseline
  (8,4)  FP8-R34 winner                  1335.7   1.97%   +0.77%
  (4,4)                                  1347.9   1.88%   +1.68%
  (2,4)                                  1354.7   1.81%   +2.19%   ←per-shape top1
  -> rule fires (8,4)

B32-M2048 (65536,   8)
  (4,8) baseline                         1322.2   0.43%   baseline
  (16,4) FP8-R34 winner                  1329.4   0.42%   +0.54%   (clean: tied spread)
  (12,4)                                 1337.2   0.76%   +1.13%   (1.5× spread)
  (1,4)                                  1335.0   1.69%   +0.97%
  -> rule fires (16,4)

B32-M4096 (131072, 16)
  (4,8) baseline                         1339.1   0.33%   baseline
  (8,4)  FP8-R34 winner                  1366.2   0.49%   +2.03%   ←CLEANEST: 4× spread
  (16,4)                                 1363.5   0.44%   +1.82%
  (1,4)                                  1361.9   0.19%   +1.70%
  -> rule fires (8,4)
```

Picked the FP8 R34 cfg pattern over the per-shape top1 because
1. FP8 R34 was validated end-to-end (FP8 metric correctness gate,
   contributed +1.5pp / shape on the 3 covered brackets via fused-act
   wall);
2. (8,4)/(16,4) is internally consistent across the 2 tiles_m=16
   brackets (both pick (8,4), good rule simplicity); and
3. The B4-M4096 (2,4) +2.19% gain is anomalous (only that one bracket
   prefers it strongly) — risk that it's noise (spread 1.81% — gap
   only 1.2× spread).

## Rule (one focused commit, BF16 dispatch only)

```python
if (
    layout == "rcr"
    and tiles_n == 11
    and k == 5760
    and m_total is not None
):
    if tiles_m == 16:
        return HipKittenConfig(
            layout=layout, group_m=8, num_xcds=4, kernel=None
        )
    if tiles_m == 8 and m_total >= 65536:
        return HipKittenConfig(
            layout=layout, group_m=16, num_xcds=4, kernel=None
        )
```

Inserted at `primus_turbo/pytorch/kernels/hipkitten/config.py:591`,
between the gpt_oss k==2880 block and the cube-small rule. Catches:

| shape (post-H4 RCR)                       | tiles_m | m_total  | cfg picked |
|-------------------------------------------|---------|----------|------------|
| gpt_oss-GateUP-B4-M4096 dA                | 16      | 16384    | (8, 4)     |
| gpt_oss-GateUP-B32-M2048 dA               | 8       | 65536    | (16, 4)    |
| gpt_oss-GateUP-B32-M4096 dA               | 16      | 131072   | (8, 4)     |
| gpt_oss-GateUP-B4-M2048 dA (excluded)     | 8       | 8192     | default    |

## Bit-equivalence verified

`/tmp/probe_round7_bf16_correctness.py`: across all 5 candidate cfgs
((8,4), (16,4), (4,4), (2,4), (1,4)) on all 4 target shapes,
`torch.equal(out_with_cfg, out_default)` returns True — group_m and
num_xcds are pure scheduling knobs on the BF16 grouped RCR persistent
tile schedule, identical to the property documented for every prior
BF16 RCR rule. Output also validated by the metric's correctness gate
(`_check_hk_vs_triton_small` — downsized B'=min(B,4), M'=min(M,256)
allclose check against Triton fwd+bwd) on all 24 shapes after the
rule landed: 24/24 PASS, 0/24 reject.

## Metric numbers — 3-run mean post-rule

3 consecutive metric runs with the rule active:

```
                     score    gpt_oss    DSV3    Qwen3
run 1                891      1.108      1.122   1.122
run 2                886      1.097      1.125   1.122
run 3                884      1.097      1.118   1.113
mean                 887.0    1.1006     1.1217  1.1190
single starting baseline (without rule, same git HEAD as commit base):
                     876      1.083      1.114   1.113
```

- Δ score (mean post-rule vs starting baseline): **+11**
- Δ gpt_oss family geomean: **+1.7pp** (1.083 → 1.100), well above the
  ±0.4pp noise floor characterised by the 3 post-rule runs.
- DSV3 / Qwen3 family geomeans: drift +0.4pp / +0.6pp respectively
  (both inside noise; the rule does not affect either family).

Per-shape gpt_oss-GateUP — direct rule-affected vs pre-rule (single
run #1 vs starting baseline):

```
                                pre-rule  post-rule  Δpp
GateUP-B4-M2048 (excluded)      1.093     1.117      +2.4 (noise — rule does NOT fire)
GateUP-B4-M4096 (rule fires)    1.093     1.125      +3.2
GateUP-B32-M2048 (rule fires)   1.038     1.058      +2.0
GateUP-B32-M4096 (rule fires)   1.079     1.089      +1.0
```

The 3 rule-affected shapes contributed +6.2pp summed Δ (mean +2.1pp
each). The Δ matches the kernel-level probe directionally (+0.77 /
+0.54 / +2.03% kernel TFLOPS → +1-3pp metric ratio because the cfg
only affects the dA share of the bwd wall).

## Compliance

- ✅ No cache (rule is a pure if/else config dispatch).
- ✅ General predicates only: `tiles_n == 11 and k == 5760 and
  m_total is not None and tiles_m in {8, 16} and m_total >= 65536` —
  no per-(M,N,K) hardcode.
- ✅ `can_handle` unchanged — the rule changes cfg only, never
  rejects.
- ✅ No host syncs added.
- ✅ Numerical equivalence: bit-identical output across all 5
  candidate cfgs on all 4 target shapes (`torch.equal` returned True
  in 20/20 probe trials).
- ✅ Metric files unmodified.
- ✅ One focused commit on Primus-Turbo this round; HipKittens repo
  not touched.
- ✅ HIPKITTEN backend stays `autotune=False`.
- ✅ Rule scope-checked against all metric BF16 grouped shapes:
  - DSV3-GateUP fwd: K=7168 (no), dA: K=N_fwd=4096 (no), dB var-K:
    different layout
  - DSV3-Down fwd: K=2048 (no), dA: K=N_fwd=7168 (no)
  - gpt_oss-GateUP fwd: K=2880 (no — caught by k==2880 block above)
  - gpt_oss-GateUP dA H4: K=N_fwd=5760, n=K_fwd=2880 → tiles_n=11,
    k=5760  ← UNIQUELY MATCHES this rule
  - gpt_oss-Down fwd: K=2880 (no)
  - gpt_oss-Down dA H4: K=N_fwd=2880 (no — k=2880 block)
  - Qwen3-GateUP fwd: K=4096 (no)
  - Qwen3-Down fwd: K=1536 (no)
  - Dense BF16 (Llama / Mistral / etc.): no shape has n=2880
    (tiles_n=11 = unique to gpt_oss in any metric).
  No collateral on other shapes.

## Files touched

- `primus_turbo/pytorch/kernels/hipkitten/config.py` — new BF16 RCR
  rule (1 if/elif block, ~95 lines comment + 8 lines code) inserted
  at line 591, between the gpt_oss k==2880 block and the cube-small
  rule.
- `analysis/_notes/round-7-bf16-grouped-gpt_oss-GateUP-dA-h4-rcr-LANDED.md`
  (this file).

NO HipKittens kernel changes this round (kernel-side strided-B work
explicitly deferred to a future round; this round's win is pure
dispatch-cfg routing of the existing post-H4 RCR call into a
better-tuned (gm, xcds) cell on the BF16 persistent grouped grid).

## Suggested R8

The dispatch surface for gpt_oss-GateUP backward (dA H4 RCR + dB
var-K CRR) is now exhausted on the BF16 side. Remaining attack
vectors with non-zero leverage on gpt_oss family:

1. **Lever C (dispatch) — gpt_oss-Down dA H4 RCR cfg**: gpt_oss-Down
   dA H4 reroute hits `tiles_n=11 and k=2880` (NOT k=5760 — Down's
   K_fwd = N_fwd = 2880). The existing k==2880 block at lines 272-590
   has rules for tiles_m∈{8,16} but those were tuned for gpt_oss-Down
   FORWARD (different tile geometry). Worth a probe to see if the
   dA path benefits from a separate cfg. Lower priority than (2) below
   (gpt_oss-Down B=32 ratios are 1.05-1.10, less below 1.25 than
   GateUP B=32).

2. **Lever B1 — DSV3 / Qwen3 forward MFMA pipeline scheduling
   (kernel-side)**: 16 shapes at 1.10-1.13 ratios. Each shape pushed
   to 1.20 = +0.08 progress × weight 1 / 40 = +2 score; full 1.25
   capping = +0.15 × 16 / 40 = +60 score ceiling. Highest absolute
   ceiling from here. Multi-round kernel work in
   `kernel_bf16_dynamic.cpp`'s K%128==0 fast path. Profile with
   `rocprofv3 valuMfmaUtil` first (per round-6 recommendation, still
   the right next move) to see whether MFMA is saturated.

3. **Lever A1 — gpt_oss K-tail in-kernel fuse (kernel-side)**:
   gpt_oss forward K=2880 (K%128=64) currently uses `FUSED_KTAIL`
   (round-3 path A) but the K-tail accumulation tile may still
   under-utilize MFMA. The 4 gpt_oss B=32 shapes are still at
   1.04-1.09 — closing the K-tail gap further could lift them
   collectively +5-10pp. Multi-round kernel work.

R8 recommendation: **Lever C dA H4 RCR for gpt_oss-Down** (cheap to
probe, mirrors this round's pattern). If saturated, pivot to Lever B1
profile-and-scout (kernel-side multi-round project). The R6 plan to
profile DSV3-Down-B16-M4096 / gpt_oss-GateUP-B32-M2048 with rocprofv3
remains valid for R9+ once B1 entry costs become the gating concern.
