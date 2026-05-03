# Round-22 (BF16 grouped GEMM) — gpt_oss var-K (dB CRR) B=32 split FALSIFIED (noise)

**Commit (this round):** docs-only (no rule change committed).
**Status:** R21's high-confidence candidate (var-K B=32 split) **FALSIFIED**
at the metric level. Kernel-only signal real (+0.32 to +0.99 % bit-eq
on all 4 B=32 shapes, top-1 each), but the wall-time delta is well
below the metric's ±5-8 score noise floor.
**Lever (this round):** A2 / B4 (var-K dB persistent kernel cfg split by m_total).

---

## Setup

R22 baseline run (HEAD = `2b07f005`, post-R20 + R21 docs):
- Score = **884**, all 24 PASS.
- Per-family geomean: gpt_oss = 1.0940, DSV3 = 1.1298, Qwen3 = 1.1137.
- Bottom 4 shapes (lowest ratio):
  | rank | shape | ratio | weight |
  |---|---|---|---|
  | 1 | gpt_oss_20B-GateUP-B32-M2048 | 1.044 | 3x |
  | 2 | gpt_oss_20B-Down-B32-M2048   | 1.065 | 3x |
  | 3 | gpt_oss_20B-GateUP-B32-M4096 | 1.083 | 3x |
  | 4 | gpt_oss_20B-Down-B32-M4096   | 1.090 | 3x |

All 4 are gpt_oss B=32 var-K dB-touching (per the metric's fwd+bwd
wall) — exactly the family R21 proposed splitting from the existing
single-cfg rule.

## Probe (kernel-only)

`scripts/_bf16_vark_db_probe.py` — clone of `_bf16_rrr_da_probe.py`
adapted for the var-K signature `grouped_variable_k_crr(a, b, c,
group_offs, group_m, num_xcds)`. 11 cells × 5 trials × 100 iters per
cell, **kernel-only** timing (no fwd, no Triton dA), bit-eq check
vs default (4, 4) on every cell.

**Important workaround**: HK BF16 var-K crashes with a memory access
fault on cold gpt_oss-GateUP-B=32 (M_total=65536, N=5760, K=2880).
Confirmed by bisection — the metric works around this by running
DSV3 fwd+bwd via autograd before gpt_oss launches; the probe needs
the same. Direct `var_k_fn(...)` warmup on K%128==0 shapes (e.g.
DSV3-Down) is INSUFFICIENT. The probe now warms via the
`turbo.ops.grouped_gemm` autograd path on (DSV3-GateUP B=16, DSV3-Down
B=16, gpt_oss-GateUP B=4, gpt_oss-Down B=4, gpt_oss-Down B=32) before
hitting gpt_oss-GateUP-B=32. This is the same K-tail cold-start
sync-fault bug the metric calls out in its iteration-order comment.

### Kernel-only sweep results

```
gpt_oss-GateUP B=32 M=2048 (m_total=65536, tiles_m=22, tiles_n=11):
   gm= 1 xcd= 4   1124.9 TF  +0.84 %  (top-1)
   gm= 4 xcd= 4   1115.5 TF   *def*
   gm= 2 xcd= 4   1115.0 TF
   gm= 8 xcd= 4   1114.3 TF
   gm=16 xcd= 4   1107.5 TF

gpt_oss-GateUP B=32 M=4096 (m_total=131072):
   gm= 1 xcd= 4   1214.3 TF  +0.99 %  (top-1)
   gm= 4 xcd= 4   1202.4 TF   *def*

gpt_oss-Down B=32 M=2048 (m_total=65536, tiles_m=11, tiles_n=11):
   gm= 1 xcd= 4   1068.7 TF  +0.32 %  (top-1)
   gm=16 xcd= 4   1067.2 TF  +0.18 %
   gm= 8 xcd= 4   1066.0 TF
   gm= 4 xcd= 4   1065.3 TF   *def*

gpt_oss-Down B=32 M=4096 (m_total=131072):
   gm= 1 xcd= 4   1155.5 TF  +0.61 %  (top-1)
   gm= 8 xcd= 4   1154.4 TF
   gm=16 xcd= 4   1153.0 TF
   gm= 4 xcd= 4   1148.5 TF   *def*
```

**Aggregate** (from `/tmp/bf16_vark_db_probe_round22.log`):

| cell | GU-2k | GU-4k | Dn-2k | Dn-4k | avg | min | max | uniform |
|---|---|---|---|---|---|---|---|---|
| `(gm=1, xcds=4)` | +0.84 | +0.99 | +0.32 | +0.61 | +0.69 | +0.32 | +0.99 | **+** |
| `(gm=8, xcds=4)` | -0.10 | -0.60 | +0.06 | +0.51 | -0.03 | -0.60 | +0.51 | mixed |
| `(gm=16, xcds=4)`| -0.71 | -0.91 | +0.18 | +0.39 | -0.26 | -0.91 | +0.39 | mixed |
| `(gm=2, xcds=4)` | -0.04 | -0.18 | -0.75 | -0.90 | -0.47 | -0.90 | -0.04 | mixed |

Only `(gm=1, xcds=4)` is uniformly positive on all 4 shapes
(important — R21 explicitly required uniformity to commit).
Bit-identical output (max_abs=0, bit_eq=True) at every cell.

## Patch tested

```python
if layout == "crr" and tiles_n == 11 and 8 <= tiles_m <= 24 and k <= 4096:
    if m_total is not None and m_total >= 65536:  # B=32 family
        return HipKittenConfig(layout=layout, group_m=1, num_xcds=4, kernel=None)
    return HipKittenConfig(layout=layout, group_m=4, num_xcds=4, kernel=None)
```

Scope check: `m_total >= 65536` is a general work-size predicate
(not a per-(M,N,K) hardcode). In the BF16 metric it captures exactly
the 4 gpt_oss B=32 var-K shapes (B=32, M_per ∈ {2048, 4096} →
m_total ∈ {65536, 131072}). The B=4 family stays on the original
`(gm=4, xcds=4)` cell (m_total ∈ {8192, 16384}). No dense LLaMA CRR
caller passes `m_total` (dense uses `m_total=None` by construction)
so collateral exposure is zero.

## Result (R22 metric, after rule)

| run | score |
|---|---|
| baseline (HEAD `2b07f005`, no rule) | 884 |
| with rule, run 1 | 878 |
| with rule, run 2 | 886 |
| with rule, mean | **882** |
| revert confirmation | 883 |

Net effect: **−2 score** (within ±8 noise band; equiv to flat).

Per-shape gpt_oss B=32 ratio change (before → after, single sample):

| shape | before | after | Δ |
|---|---|---|---|
| GateUP-B32-M2048 | 1.044 | 1.051 | +0.7 pp ✓ |
| Down-B32-M2048   | 1.065 | 1.052 | -1.3 pp ✗ |
| GateUP-B32-M4096 | 1.083 | 1.084 | +0.1 pp tie |
| Down-B32-M4096   | 1.090 | 1.084 | -0.6 pp ✗ |

Mixed at the metric, despite the kernel-only sweep being uniform-
positive on all 4. DSV3 / Qwen3 (uninvolved by the rule) also
moved ±1 pp run-to-run. The +0.69 % kernel-only signal does not
clear the metric's run-to-run noise floor.

## Why the metric noise eats the signal

Decomposition of full fwd+bwd wall on a B=32 shape:
- forward grouped GEMM ≈ 50 % of wall (1 launch)
- backward dA (RRR) ≈ 25 % of wall (1 launch)
- backward dB (var-K CRR) ≈ 25 % of wall (1 launch — this rule)

A +0.69 % var-K kernel-only delta therefore lands as ~+0.17 % wall
delta on the bench → ~+0.002 to ratio (i.e. 1.044 → 1.046). Below
the metric's ~±0.005-0.010 single-sample run-to-run swing on
ratio. Even averaged over 5 trials the signal-to-noise is ≈ 1:3.
R1's identical conclusion 21 rounds ago (the original deferral
quote) anticipated this exactly.

## Constraints honored

- ✅ No commits with score below noise threshold (R22 doc-only,
  rule reverted before commit)
- ✅ No host syncs / per-(M,N,K) hardcodes / can_handle tightening
- ✅ Bit-identical correctness (max_abs=0, bit_eq=True at every
  probed cell; metric correctness gate 0/24 fail at every run)
- ✅ FP8 metric untouched
- ✅ Probe script archived for future re-test (`scripts/_bf16_vark_db_probe.py`)

## Implications for next rounds

1. **Var-K (CRR / dB) cfg dispatch is now CLOSED for gpt_oss**:
   the only uniform-positive cell at the kernel level is too small
   (+0.7 % avg) to clear the metric noise. Future kernel-level
   surgery on the var-K path (LDS swizzle, MFMA pipeline, register
   schedule) is required to open materially larger gains here.
2. **K-tail cold-start fault in HK BF16 var-K** is real and not
   currently documented in the SKILL.md gotchas — adding the
   workaround note (run DSV3 fwd+bwd via autograd before gpt_oss
   var-K direct calls) would save the next agent a debug round.
3. **Suggested R23 attack**: Phase B (DSV3 / Qwen3 push) levers,
   since gpt_oss B=4 ratios (1.10-1.15) are now better than B=32
   (1.04-1.09) and the B=32 cfg surface is exhausted at this
   kernel version. Specifically:
   - Lever B1: MFMA pipeline scheduling on K%128==0 forward
     (DSV3-Down K=2048 has shallowest tile budget).
   - Lever B2: register pressure / spill audit on DSV3-GateUP
     K=7168 forward — the deepest K in the suite, most likely
     to be regfile-limited.
4. **Auto_optimize patience counter** is at 15/30 (R20 LANDED
   counted as 870, real mean 891). Auto-loop's 3-run average
   eventually catches up.

## Files

- `scripts/_bf16_vark_db_probe.py` — new var-K B=32 probe
  (kept for future re-runs).
- `analysis/_notes/round-22-bf16-grouped-vark-B32-split-FALSIFIED-noise.md`
  — this falsification note.
- `primus_turbo/pytorch/kernels/hipkitten/config.py` — UNCHANGED
  (rule reverted before commit).
