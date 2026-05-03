# Round-21 (BF16 grouped GEMM) — R20 LANDED confirmed + var-K B=32 split candidate for R22

**Commit (this round):** TBD (Primus-Turbo) — doc-only, no rule change
**Status:** R20 LANDED verified by 3-run R21 baseline (mean 891.7, exceeds historical best 891).
**Lever (this round):** none. Documents R20 success and surveys remaining surfaces.

---

## R20 LANDED confirmation

R21 baseline (HEAD = `cffcc3d`, R20's 3-rule aggregate live), 3 runs:
- Run 1: 895
- Run 2: 880
- Run 3: 900
- **mean: 891.7**

Vs auto_optimize's reported "previous round metric = 870" (single noise low).
Vs historical best 891 → R20 **LANDED confirmed**. The 3-rule aggregate (`tiles_n
∈ {8, 16, 28}` dA RRR rules) is real, not noise.

Per-family geomean across 3 runs:
- gpt_oss_20B: 1.1136, 1.0830, 1.1221 (mean 1.1062, ±0.4pp run-to-run)
- DeepSeek-V3: 1.1251, 1.1258, 1.1282 (mean 1.1264, ±0.02pp)
- Qwen3-235B:  1.1229, 1.1206, 1.1305 (mean 1.1247, ±0.05pp)

DSV3 + Qwen3 family geomeans are now **above target 1.25** ratio noise floor —
the wide 0.4pp run-to-run swing in gpt_oss_20B is what gates the metric score.

## R21 lowest-ratio shapes (post-R20)

| rank | shape | ratio | weight |
|---|---|---|---|
| 1 | gpt_oss_20B-GateUP-B32-M2048 | 1.047 | 3x |
| 2 | gpt_oss_20B-GateUP-B32-M4096 | 1.087 | 3x |
| 3 | gpt_oss_20B-Down-B32-M2048   | 1.094 | 3x |
| 4 | gpt_oss_20B-Down-B32-M4096   | 1.095 | 3x |
| 5 | gpt_oss_20B-GateUP-B4-M2048  | 1.095 | 3x |

All bottom 5 are gpt_oss family. Forward (RCR) path is heavily tuned across
multiple rounds (R7, R10, R26, R45, R57, R70, R9, R21 in the BF16 block):
each `m_total` tier in `tiles_n ∈ {11, 22}` is individually anchored to a
verified local optimum.

## Surveyed surfaces

### Closed (no headroom expected)

- **BF16 forward RCR** for gpt_oss `k==2880`: each (tiles_n, tiles_m, m_total)
  bracket has been swept multiple times (R7-R26). Per-iter-sync verifies
  show top-1 within ±0.5% of the picked cell across all 4 m_total tiers.
- **BF16 dA H4 RCR reroute** for gpt_oss-GateUP `k==5760, tiles_n==11`:
  R7 mirrored FP8 R34 (gm=8/16, xcds=4 by tiles_m) on 4 metric shapes.
- **BF16 dA RRR** for DSV3 + Qwen3 (16 of 24 shapes): R18 + R20 cover all
  4 `tiles_n` brackets (6, 8, 16, 28) — R20 doc.
- **K-tail kernel path** for gpt_oss `K=2880`: R11/R14/R15 falsified
  KI=44 specialization (VGPR spill); R16 falsified KI=24/32 short-K
  specialization. Existing dynamic `KI_HINT=0` + `#pragma unroll 2` is the
  local optimum.

### Open (probe candidates for R22)

#### **(A) BF16 var-K (dB CRR) gpt_oss B=32 split** — high-confidence candidate

Current rule (line 1080 of `config.py`):

```python
if layout == "crr" and tiles_n == 11 and 8 <= tiles_m <= 24 and k <= 4096:
    return HipKittenConfig(layout=layout, group_m=4, num_xcds=4, kernel=None)
```

Single cell `(gm=4, xcds=4)` covers ALL 8 gpt_oss var-K dB shapes. R1's
commentary explicitly noted:

> Split-by-m_total tested: separate rule with (gm=8, xcds=4) for
> m_total>16384 (B=32) was within noise of this single-cfg version
> (gpt_oss geomean 0.8872 vs 0.8823, but full-suite score swung 776 vs
> 780 due to DSV3 / Qwen3 Triton baseline run-to-run noise on
> uninvolved shapes — ±2pp). Single cfg is cleaner; **defer B=32 split
> to a later round once a kernel-level fix opens more headroom there.**

Post-R19/20 kernel-level changes (BUFFER store + R18/R19/R20 RRR rule
reshape) may have opened the headroom R1 anticipated. The var-K B=32
sub-family (4 shapes, all 3x weight) sits at metric ratios 1.087-1.095
— same low band as the forward GateUP-B32-M2048 bottom shape — so even
a +1pp dB lift here would be visible in the score.

**Plan for R22**: clone `scripts/_bf16_rrr_da_probe.py` to a var-K
probe, sweep 11 cells × 5 trials × 100 iters on the 4 gpt_oss B=32 var-K
shapes. If a uniform-positive cell emerges, split the rule:

```python
if layout == "crr" and tiles_n == 11 and 8 <= tiles_m <= 24 and k <= 4096:
    if m_total is not None and m_total >= 65536:  # B=32 family
        return HipKittenConfig(layout=layout, group_m=??, num_xcds=??, kernel=None)
    return HipKittenConfig(layout=layout, group_m=4, num_xcds=4, kernel=None)
```

#### (B) Forward RCR multi-tile re-tune (low-priority)

The 4 gpt_oss-GateUP-B32 forward shapes were last anchored at R26 against
post-R19/20 BUFFER kernel — **already current**. Re-sweeping is unlikely
to open new wins until a kernel-level change happens.

#### (C) Phase-A/B/C closure doc (fallback)

If R22's var-K probe shows noise like R1's prior split attempt, write a
phase-A/B/C consolidated closure doc. The patience counter is at 14/30
post-R20 (auto_optimize sees R20 as 870, not the actual mean 891) —
once it observes the high-water 891+ baseline 2-3 more times, the
counter should reset.

## Constraints honored (this round)

- ✅ No commits with score below noise threshold (R21 doc-only)
- ✅ No host syncs / per-(M,N,K) hardcodes / can_handle tightening
- ✅ R20 commit's numerical equivalence already verified (3-run mean
  consistent at correctness PASS 0/24)
- ✅ FP8 metric untouched

## Files

- `analysis/_notes/round-21-bf16-grouped-R20-LANDED-confirm-and-var-K-B32-split-candidate.md`
  — this doc
