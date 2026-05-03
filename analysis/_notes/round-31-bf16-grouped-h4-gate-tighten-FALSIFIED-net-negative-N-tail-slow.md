# Round 31 — H4 gate tighten FALSIFIED (-79 score): native RRR N-tail is much slower than RCR fuse, not H4 overhead

## Goal

R30 characterized H4 `bf16_transpose_3d` as a 6.9-7.1 % wall
overhead on gpt_oss B=32 and pitched the R31 main line as
"investigate RRR B-load bug to eliminate H4". R31 opened that
investigation by first testing whether the HK RRR K-tail path
was already correct by monkey-patching the `execute` to skip H4
unconditionally, then running the metric's downsized allclose on
all 8 gpt_oss H4-reroute shapes.

## Correctness probe finding

`/tmp/probe_round31_rrr_bypass.py` ran downsized allclose per
shape:

```
PASS  gpt_oss-GateUP-B4-M2048     (a.shape[1] = N_fwd = 5760 % 128 == 0)
PASS  gpt_oss-GateUP-B32-M2048
PASS  gpt_oss-GateUP-B4-M4096
PASS  gpt_oss-GateUP-B32-M4096
FAIL  gpt_oss-Down-B4-M2048       (a.shape[1] = N_fwd = 2880 % 128 == 64)  dA-allclose
FAIL  gpt_oss-Down-B32-M2048      dA-allclose
FAIL  gpt_oss-Down-B4-M4096       dA-allclose
FAIL  gpt_oss-Down-B32-M4096      dA-allclose
```

**Correctness finding**: the RRR K-tail phantom-read bug only fires
when kernel-K = a.shape[1] = N_fwd has K_TWO_TILE-misaligned tail
(N_fwd % 128 != 0). gpt_oss-GateUP (N_fwd=5760 % 128 == 0) bypass
H4 PASSES allclose because no K-tail path is entered; gpt_oss-Down
(N_fwd=2880 % 128 == 64) FAILS because the K-tail fuse block with
the stale-capture bug fires.

This is the R19 gate's over-triggering: R19 used the coarser
`K_RRR % 64 != 0 OR N_RRR % 256 != 0` predicate; the actual bug
trigger is `a.shape[1] % 128 != 0`.

## Metric verification — where it went wrong

R31 tightened the H4 gate from R19's coarse predicate to the
bug-exact `a.shape[1] % 128 != 0`. All 9 canonical shapes passed
correctness. Metric (single-run):

| shape (gpt_oss-GateUP, weight 3x)       | R30 baseline ratio | R31 post-tighten ratio | Δ |
|-----------------------------------------|-------------------:|-----------------------:|--:|
| B=4-M2048  (now skips H4)               | 1.100              | **0.859**              | -0.241 |
| B=32-M2048 (now skips H4)               | 1.049              | **0.666**              | -0.383 |
| B=4-M4096  (now skips H4)               | 1.099              | **0.821**              | -0.278 |
| B=32-M4096 (now skips H4)               | 1.087              | **0.648**              | -0.439 |

```
gpt_oss_20B geomean:  1.086 → 0.902   (-17 %)
score:                  880 → 801     (-79)
```

**Not the hypothesized +12 score; it is −79 score.** The mid-bwd
GateUP family regressed by 30-40 % wall.

## Root cause

H4 is NOT a 393 µs overhead — it is the **fast-path switch** that
reroutes the bwd dA from a broken+slow native RRR path onto the
working+fast RCR fuse path. The 393 µs transpose pays for itself
~10x over by avoiding:

  1. The native RRR K-tail phantom-read bug (what R9 H4 was
     originally written to dodge). R31's correctness probe shows
     this fires only for N_fwd % 128 != 0, so gpt_oss-GateUP
     doesn't hit it — hence PASS — but…
  2. **The native RRR N-tail slow path**, which IS entered for
     `b.shape[-1] % 256 != 0` (= K_fwd=2880 % 256 = 64). The HK
     kernel's N-tail handler produces correct output but at
     **dramatically** reduced throughput. From the measured
     hk_tflops collapse:

     GateUP-B32-M2048 hk: 1141 TF (post-transpose RCR) →
                           718 TF (native RRR+N-tail)
     wall:                5710 µs → 9080 µs (+3370 µs per iter)

     The H4 "cost" (393 µs) is dwarfed by the +3370 µs native-RRR-N-tail
     penalty. Net: transposing B saves ~3000 µs / iter on this shape.

The R19 comment ("N-tail is native and handled correctly") was
*correctness*-correct but *performance*-wrong: the N-tail handler
is correct but much slower than the padded RCR fuse.

## Decision

**Revert. Documentation-only commit.** The R19 gate with both
K and N alignment checks is preserved exactly as-is. R31 probes
and this note are archived.

**This is a positive new finding, not a pure FALSIFIED.** We now
know:

  * **R31 correctness finding** (stays in the backlog): the HK RRR
    K-tail fuse has a real bug only when N_fwd % 128 != 0. For
    gpt_oss-Down (N_fwd=2880), the R9 H4 workaround is still
    required for correctness.
  * **R31 performance finding**: the HK RRR kernel's N-tail handler
    runs at about 0.6× the RCR fuse path's throughput on gpt_oss
    K_fwd=2880. Removing H4 exposes the slow path.

Post-revert metric confirms: 878 (matches R31 baseline 880 within
noise floor).

## Implications for R32+

The H4 wall fraction on gpt_oss B=32 is not 6.9 % of "transposing
overhead"; it is the **net saving** of redirecting bwd dA through
RCR. To eliminate the H4 call AND keep the fast path, **either**:

  A. **Fix RRR's K-tail phantom-read bug** (round-3..7 territory
     in kernel_bf16_dynamic.cpp lines 917-1134). R31 probe shows
     this blocks only 4 shapes (gpt_oss-Down); a fix would let
     those shapes go native RRR. But the native RRR N-tail path
     is still slow — R31 showed this is a separate issue.
  B. **Fix RRR's N-tail slow path** (the HK kernel's per-cell
     tail handler). This is the DOMINANT speed gap: GateUP B=32
     hits 0.6× throughput on the N-tail vs RCR. Optimizing this
     kernel branch would benefit the 4 gpt_oss-GateUP shapes
     directly, even before fixing the K-tail bug.
  C. **Both A and B** are needed for a full H4 elimination. Neither
     is a single-round change.

**R32 recommendation**: do NOT touch the H4 gate. Pivot to the
kernel C++ side. Start with (B) — native RRR N-tail handler is
the 80 % lever. `kernel_bf16_dynamic.cpp` uses an external
`grouped_ntail` launcher for N%256 != 0; profile what that
launcher does and look for why it's 0.6× on K=2880 shapes.

**If C++ work infeasible**: document this round as a knowledge
deposit, keep code at HEAD, and accept score plateau ~880.

## Files

* `analysis/_notes/round-31-bf16-grouped-h4-gate-tighten-FALSIFIED-net-negative-N-tail-slow.md` — this note.
* `/tmp/probe_round31_rrr_bypass.py` — the monkey-patch correctness
  probe (not committed).
* No production change (`grouped_gemm_impl.py` reverted to HEAD d4096bd).
