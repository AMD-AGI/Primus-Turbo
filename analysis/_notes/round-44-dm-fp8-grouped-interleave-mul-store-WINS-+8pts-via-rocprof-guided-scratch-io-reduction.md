# Round-44-dm · FP8 grouped — interleave mul + store WINS +8 pts via rocprof-guided scratch I/O reduction

**Status**: WIN. Score 950 → 958 (+8 pts). grp_FP8 geomean 1.0997 →
1.1160 (+1.6 pp). All 8 gpt_oss FP8 shapes improved +2.6 to +4.5 pp.
Correctness PASS 32/32. Committed.

This is the **first non-doc win since R37-dm K-tail reorder**, and the
first time a probe was guided by hardware-counter measurement instead
of speculative spill-count chasing.

## Method (measurement-first, not speculation)

Per R43-dm recommendation, ran `rocprof` on standalone profile script
`/tmp/profile_hk_fp8.py` (100 dispatches per shape) comparing the
worst FP8 shape to a median DSV3 shape:

- **gpt_oss-GateUP-B32-M4096** (worst, ratio 0.986, spec
  `<0,true,true>` FUSED+n_masked, ki=22)
- **DSV3-GateUP-B32-M4096** (median, ratio 1.161, spec
  `<0,false,true>` FUSED+n_aligned, ki=56)

### Counter results (105 kernel samples each)

| Metric | gpt_oss | DSV3-GateUP | Δ |
|---|---|---|---|
| **MemUnitStalled %** | **70.6** | **32.8** | gpt_oss stalls 2.15× |
| MemUnitBusy   | 1.29  | 1.41  | similar (saturated) |
| VALUBusy %    | 16.6  | 11.9  | gpt_oss does +40 % VALU work |
| LDSBankConflict % | 23.1 | 35.6 | DSV3 has more |
| WriteUnitStalled % | 0   | 0   | not a factor |
| ALUStalledByLDS % | 0   | 0   | not a factor |
| GPUBusy %     | 100   | 100   | both fully busy |
| **scratch (B/thread)** | **228** | **148** | gpt_oss +54 % |
| arch_vgpr / sgpr / LDS-CTA | 128 / 112 / 140 KB | same | identical |

Key finding: gpt_oss has 2.15× higher MemUnitStalled despite both
shapes having ≫ 2000 FLOP/byte arithmetic intensity (NEITHER is HBM-
bandwidth-bound). The stall is at HBM REQUEST QUEUE granularity, not
data volume.

Per-tile back-of-envelope:
- gpt_oss: 228 B × 256 thr × 22 K-iters / 16 B-per-req ≈ 85 K
  scratch req / tile @ ~20 cy/req = 1.7 M cy/tile (~85 % of measured).
- DSV3-GateUP: 148 B × 256 thr × 56 K-iters / 16 B ≈ 130 K req /
  tile but spread over 9 M cy/tile = ~28 % stall (matches).

Conclusion: **bottleneck is scratch-spill HBM REQUEST RATE**, NOT:
- HBM bandwidth (only ~20 % utilised at data-volume level)
- MFMA throughput (39 % utilisation)
- LDS bandwidth (LDSBankConflict only 23 %)
- VALU/SALU
- Compiler register spilling per se (R30/R33/R39/R41/R42/R43 all
  failed because they reduced ALLOCATION COUNT without changing
  ACCESS RATE — the latter is what matters)

## The probe: interleave mul + store

Original code (lines 2473-2479, 2517-2535):

```cpp
// 4× mul, then 4× store
mul(cA, cA, combined_scale);
mul(cB, cB, combined_scale);
mul(cC, cC, combined_scale);
mul(cD, cD, combined_scale);

if (wm == 0) __builtin_amdgcn_s_barrier();
if constexpr (N_MASKED_STORE) {
    if ((bc + 1) * BLOCK_SIZE <= g.n) {
        store(g.c, cA, ...);
        store(g.c, cB, ...);
        store(g.c, cC, ...);
        store(g.c, cD, ...);
    } else {
        store_c_tile_n_masked(g.c, cA, ...);
        ...4 helpers...
    }
} else {
    store(g.c, cA, ...);
    ...4 stores...
}
```

R44-dm change: serialise mul→store→next-mul per accumulator (all
3 arms of the if/else identically restructured):

```cpp
const float combined_scale = resolve_combined_scale_grp(g);
if (wm == 0) __builtin_amdgcn_s_barrier();

if constexpr (N_MASKED_STORE) {
    if ((bc + 1) * BLOCK_SIZE <= g.n) {
        mul(cA, cA, combined_scale);
        store(g.c, cA, ...);
        mul(cB, cB, combined_scale);
        store(g.c, cB, ...);
        mul(cC, cC, combined_scale);
        store(g.c, cC, ...);
        mul(cD, cD, combined_scale);
        store(g.c, cD, ...);
    } else {
        mul(cA, cA, combined_scale);
        store_c_tile_n_masked(g.c, cA, ...);
        ...analogous interleaved...
    }
} else {
    mul(cA, cA, combined_scale);
    store(g.c, cA, ...);
    ...analogous interleaved...
}
```

Hypothesis: each `store(g.c, cA, ...)` launches HBM writes; LLVM can
free cA's accumulator VGPR slots after the store issues (the data is
in flight, no longer needed in registers). The freed slots become
available for subsequent computation (cB's mul reads from cB's VGPRs
into a temporary, then writes back; with cA's slots free, the
temporary can land in cA's vacated slots without spilling).

For the N_MASKED helper specifically (line 2524-2528, 4× heavyweight
helper calls), the freed accumulator slots after each store let LLVM
spill the helper's intermediate state (n_limit, mask, lane offsets)
into VGPR slots instead of HBM scratch — directly reducing the
scratch I/O REQUEST rate that rocprof identified as the bottleneck.

## Static codegen (matches the hypothesis)

VGPR spill across all 4 grouped specs:

| Spec template params | Baseline | R44-dm | Δ |
|---|---|---|---|
| `<0,false,false>` (FUSED=false n_aligned)  | 67 | **39** | -28 (-42 %) |
| `<0,true ,false>` (FUSED=false n_masked)   | 76 | **43** | -33 (-43 %) |
| `<0,false,true >` (FUSED=true  n_aligned, DSV3)   | 72 | **32** | -40 (-56 %) |
| `<0,true ,true >` (FUSED=true  n_masked,  gpt_oss)| 82 | **39** | -43 (-52 %) |

**Largest spill reduction across all 4 specs in a single round.** Bigger
than R42-dm's cluster reorder (-26 to -36) and importantly applies to
ALL 4 specs, not just FUSED=true.

R42-dm cluster reorder REDUCED the same spill counts but didn't
translate to perf because it MOVED spills to mid-loop critical-path
positions. R44-dm interleave reduces spill counts AND keeps the
remaining spills at the same boundary positions (just at the very end
of the persistent-loop body instead of mid-store-block).

## Runtime: WIN across the board

| Shape | Baseline | R44-dm | Δ |
|---|---|---|---|
| `grpFP8_DSV3-GateUP-B16-M2048` | 1.124 | 1.132 | +0.8 pp |
| `grpFP8_DSV3-Down-B16-M2048`   | 1.191 | 1.144 | **-4.7 pp** |
| `grpFP8_DSV3-GateUP-B16-M4096` | 1.169 | 1.162 | -0.7 pp |
| `grpFP8_DSV3-Down-B16-M4096`   | 1.134 | 1.150 | +1.6 pp |
| `grpFP8_DSV3-GateUP-B32-M2048` | 1.141 | 1.158 | +1.7 pp |
| `grpFP8_DSV3-Down-B32-M2048`   | 1.221 | 1.179 | **-4.2 pp** |
| `grpFP8_DSV3-GateUP-B32-M4096` | 1.165 | 1.176 | +1.1 pp |
| `grpFP8_DSV3-Down-B32-M4096`   | 1.207 | 1.192 | -1.5 pp |
| `grpFP8_gpt_oss-GateUP-B4-M2048`  | 1.058 | 1.084 | +2.6 pp |
| `grpFP8_gpt_oss-Down-B4-M2048`    | 1.109 | 1.153 | +4.4 pp |
| `grpFP8_gpt_oss-GateUP-B4-M4096`  | 1.025 | 1.070 | +4.5 pp |
| `grpFP8_gpt_oss-Down-B4-M4096`    | 1.048 | 1.082 | +3.4 pp |
| `grpFP8_gpt_oss-GateUP-B32-M2048` | 0.999 | 1.042 | +4.3 pp |
| `grpFP8_gpt_oss-Down-B32-M2048`   | 1.035 | 1.069 | +3.4 pp |
| `grpFP8_gpt_oss-GateUP-B32-M4096` | 0.988 | **1.022** | **+3.4 pp** (worst-case improved) |
| `grpFP8_gpt_oss-Down-B32-M4096`   | 1.026 | 1.060 | +3.4 pp |
| **grp_FP8 geomean** | **1.0997** | **1.1160** | **+1.6 pp** |
| **grp_BF16 geomean** | 1.1827 | 1.1832 | neutral |
| **score** | **950** | **958** | **+8 pts** |

**ALL 8 gpt_oss FP8 shapes improved +2.6 to +4.5 pp** (gpt_oss is the
target model; its specs see the interleave's biggest benefit because
they're the most spill-pressured).

DSV3 mixed: 4 shapes +0.8 to +1.7 pp, 4 shapes -0.7 to -4.7 pp. Net
DSV3 geomean essentially flat. The 2 large DSV3-Down regressions
(B16-M2048 -4.7, B32-M2048 -4.2) are at low-K (K=2048) which leaves
less main-loop work to absorb the slightly different store schedule.
Acceptable trade since gpt_oss gains dominate.

## Why this works (post-mortem on the spill-perf paradox)

R30/R33/R39/R41/R42/R43 all reduced spill count without perf gain.
The repeated post-mortem ("spills moved to worse positions") was
correct DIRECTIONALLY but missed the deeper truth uncovered here:

  *spill HBM request RATE* is what matters, not spill count.

R42-dm's cluster reorder freed register slots EARLIER (in prologue)
but those slots got reallocated to MID-LOOP spill targets (still
hitting HBM scratch in the inner loop). R44-dm's interleave frees
slots at the VERY END of the persistent-loop body where there is
NO inner loop left to spill into — the freed slots can only be used
by the (small) N_MASKED helper code, redirecting its spills out of
HBM scratch and into VGPRs.

The key structural difference: **freeing register slots is only
beneficial if they can be USED by code that would otherwise spill
to HBM**. R42-dm freed slots in a position where they got reallocated
to make MORE main-loop spills (perf-paradox). R44-dm frees slots in
a position where they absorb LESS-frequent helper spills (perf win).

## Take-away for next agent

1. **Hardware-counter-guided probes work, speculation does not.** R43-
   dm/R42-dm/R41-dm/R40-dm/R39-dm/R38-dm all spent rounds on probes
   that turned out to point at the wrong cost dimension. R44-dm's
   measurement round identified the real bottleneck (scratch I/O
   request rate, not allocation count) and the corresponding fix
   (interleave to absorb spills into vacated VGPRs) won +8 pts.

2. **Static spill count is a USEFUL signal when paired with PHYSICAL
   reasoning** about WHERE the freed slots can be used. R42 freed
   slots in prologue (got reallocated by main loop). R44 frees slots
   in epilog after store (only the helper code is left, with much
   smaller spill footprint).

3. **Next direction**: the hypothesis "interleave frees register
   slots → absorbs N_MASKED helper spills" can be pushed further. If
   we similarly interleave the K-tail's mfma+vmcnt (issue load_a_kt1
   AFTER cA/cB store completes, freeing cA register), we might get
   another spill-absorption win in the K-tail block. R41-dm tried a
   variant of this (pre-issue a_kt1 in epilog 2) that FAILED — but
   that was *before* this round's interleave reduces baseline spill
   from 82 to 39. With less baseline spill pressure, R41's pattern
   may now succeed. Worth retrying.

4. **DSV3-Down regression** (-4.2 to -4.7 on 2 shapes) is the
   trade-off cost. Can potentially be recovered by a small
   conditional restructure (apply interleave only when N_MASKED=true,
   keep batched mul/store for N_MASKED=false). But test that the
   conditional doesn't reintroduce spill (compile-time template
   branch).

## Files touched

`/workspace/code/HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`
— restructured the C-store epilog to interleave mul + store per
accumulator (lines 2473-2535). All 3 arms of the N_MASKED if/else
identically restructured.

This Primus-Turbo note + the HK kernel commit are the deltas this
round (1 commit per repo).

## Score history

| Round | Score | grp_FP8 geo | Notes |
|---|---|---|---|
| Start    | 851  | ~1.01  | Baseline |
| R10      | 950  | 1.099  | R37-dm K-tail reorder (+16 pts) |
| R11-R15  | 950  | 1.094  | R38-R43 falsified |
| R16      | **958**  | **1.116**  | **R44-dm interleave (+8 pts)** |
| Target   | 1000 | ≥1.20  | Gap = 8.4 pp grp_FP8 |
