# Round-36-dm · FP8 grouped — gpt_oss bottleneck is K-tail RUNTIME execution, not interleaved spills

**Status**: ANALYSIS ROUND (no code change).
Score unchanged: R8 935 → R9 936 (±1 noise).
Correctness all PASS.

## Post-R34/R35 state

After R34's `FUSED_KTAIL=true` extension and R35's correctly-falsified
N_MASKED_STORE collapse, the FP8 dispatch landscape stabilises at:

| Shape family | Spec | Interleaved spills | Perf ratio |
|---|---|---|---|
| DSV3-GateUP (4×) | `<0,false,true>` | 28 | 1.090-1.147 |
| DSV3-Down (4×)   | `<0,false,true>` | 28 | 1.055-1.137 |
| gpt_oss-GateUP (4×) | `<0,true,true>` | **22** (strictly better) | **0.967-1.035** |
| gpt_oss-Down (4×)   | `<0,true,true>` | **22** (strictly better) | 1.008-1.097 |

**gpt_oss is on the BETTER-codegen spec yet performs WORSE** than DSV3 by 5-10 pp.
The interleaved-spill proxy has NEGATIVE correlation here — static ISA advantage
does not translate to runtime advantage.

## Deep ISA analysis of the 21 interleaved-spill groups

Inspected `spec_<0,true,true>.s` in detail. Example of the pattern
(first group, around line 1748):

```asm
s_setprio 1
v_mfma_f32_16x16x128_f8f6f4 v[96:99], v[144:151], v[128:135], v[96:99]
v_mfma_f32_16x16x128_f8f6f4 v[76:79], v[144:151], v[136:143], v[76:79]
v_mfma_f32_16x16x128_f8f6f4 v[184:187], v[152:159], v[128:135], v[52:55]
v_mfma_f32_16x16x128_f8f6f4 v[188:191], v[152:159], v[136:143], v[40:43]
v_mfma_f32_16x16x128_f8f6f4 v[4:7], v[168:175], v[128:135], v[4:7]
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[168:175], v[136:143], v[0:3]
s_nop 10                                      ; ← mfma RAW hazard nops
scratch_store_dwordx4 off, v[4:7], off offset:16 ; Folded Spill (HIDDEN in nop)
v_mfma_f32_16x16x128_f8f6f4 v[192:195], v[160:167], v[128:135], v[24:27]
scratch_store_dwordx4 off, v[0:3], off offset:32 ; Folded Spill (HIDDEN in nop)
v_mfma_f32_16x16x128_f8f6f4 v[222:225], v[160:167], v[136:143], v[12:15]
s_setprio 0
s_barrier
```

**Key finding**: The interleaved `scratch_store_dwordx4` instructions fit
INSIDE the required MFMA `s_nop 10` hazard slot. They do NOT add wall-clock
latency beyond what `s_nop 10` already costs. LLVM placed spills
opportunistically during mandatory MFMA stall cycles.

This explains why R35's attempted 28→22 interleaved move regressed DSV3-Down
by 2.9-8.8 pp: the "savings" from removing 6 interleaved spills were
hidden latency already; meanwhile the runtime N-masked branch cost became
un-hidden.

**Implication**: the interleaved-spill proxy has DIMINISHING RETURNS
below ~30. R34's 62→28 move paid off because the 34 excess spills
overran the `s_nop` budget. Going 28→22 is in the noise zone.

## Full ISA instruction counts (per spec, total static counts)

| Spec | mfma | buffer_load_dwordx4 | scratch_load | scratch_store | vmcnt(0) waits |
|---|---|---|---|---|---|
| `<0,false,false>` | 648 | — | 507 | 170 | 505 |
| `<0,true,false>`  | 552 | — | 423 | 137 | 495 |
| `<0,false,true>`  (DSV3) | 456 | 144 | 344 | 164 | 469 |
| `<0,true,true>`   (gpt_oss) | **328** | 96 | **274** | **128** | 446 |

gpt_oss's spec has:
- 28% fewer MFMAs (328 vs 456) — but these run at peak throughput
- 20% fewer spill loads
- Strictly better on every static metric

YET gpt_oss ratio is ~0.98, while DSV3 is ~1.10.

## Identified runtime bottleneck for gpt_oss

The gap is NOT codegen — it's **genuine runtime execution cost**:

1. **K-tail fires for every tile** (gpt_oss K_REM=64, DSV3 K_REM=0).
   K-tail sequence = 24 × `buffer_load_b128` (HBM) + 2 vmcnt waits + 4
   MFMAs ≈ 300-400 cycles per tile. DSV3 skips this entirely.

2. **N-masked store branch** fires for every C-tile output. Cheap per-fire
   (wave-uniform) but accumulates across tens of tiles per group.

3. **Non-aligned N (5760 / 2880)** → `bpc = ceil_div(N, 256)` = {23, 12}
   giving partial last col-tile on every row — masked-store helper body
   is in scope (even if fast-path is taken).

Quantitative comparison:
- DSV3-GateUP-B32-M4096: K=7168, 56 K-iters, 0 K-tail; ratio 1.147
- gpt_oss-GateUP-B32-M4096: K=2880, 22 K-iters + 1 K-tail; ratio 0.967
- Δ K-tail pct of work: ~4.3% extra MFMA+load on gpt_oss
- Δ ratio: −18 pp
- Overhead multiplier: ~4x (K-tail is expensive per cycle of work)

The K-tail sequence issues 24 HBM loads with vmcnt(8)+vmcnt(0) waits
sandwiching only 2 MFMAs (~64 cy overlap). The 8 `a_kt1` loads cost
HBM latency (~500 cy) with only 64 cy hidden → ~436 cy unhidden per
tile from vmcnt(0) alone. Across thousands of tiles per group, this
dominates gpt_oss wall time.

## R10 plan: K-tail overlap via separate register tile + early issue

Concrete design:

1. Hoist the K-tail SRD setup (a_srsrc_kt / b_srsrc_kt / K_tail_base_bytes)
   and a_kt1 declaration to BEFORE epilog 2 (just after epilog 1 closes).
2. Issue the 8 `load_a_kt(a_kt1, 1)` calls DURING epilog 2's merged
   mfma bracket (line 2251-2253). These 8 HBM loads begin draining while
   the 2 MFMAs execute (~64 cy head start, plus draining through the
   barrier overhead ~32 cy).
3. In the K-tail block, SKIP the a_kt1 load (already issued) and drop
   the vmcnt(0) before `rcr_mma(cC, a_kt1, b0)` to a smaller count
   accounting only for the remaining 16 loads.
4. Add `RCR_SCHED_BARRIER()` between the early-issued `load_a_kt(a_kt1)`
   and the merged MFMA, to prevent LLVM from pulling the load into the
   mfma bracket (which would defeat the overlap).

### Risk analysis

Primary risk: LLVM spill backlash (R33-pattern). Extending `a_kt1`'s
live range from ~6 lines to ~20 lines may cause LLVM to spill
`a_kt1`'s 16 VGPRs across the epilog 2 MFMA. Mitigations:
- Keep the early-issue INSIDE `if constexpr (FUSED_KTAIL)` (no impact on
  non-fused specs).
- Keep the early-issue INSIDE runtime `if (g.fast_k < g.k)` branch, so
  DSV3's K-tail-dead path gains nothing (avoiding R35-style false-win
  on shapes that weren't meant to benefit).
- Verify with ISA dump: if the interleaved-spill count spikes on
  `<0,true,true>`, abandon and revert.

Secondary risk: out-of-order HBM completion. The current `vmcnt(8)`
wait assumes in-issue-order retirement (per R12-dm comment). Moving
`a_kt1` to earlier issue position means its drain time is longer —
should help with hiding, but if vmcnt semantics do NOT guarantee
in-order retirement, results could be nondeterministic. Numerical
probe required.

### Expected payoff

If a_kt1 HBM latency is ~500 cy and epilog 2's mfma bracket is ~128 cy
(merged 2-mfma + barrier), the overlap hides 128 cy of a_kt1 drain.
Current K-tail vmcnt(0) stalls ~436 cy → with overlap, ~308 cy.
Savings: ~128 cy per tile ≈ 4% per-tile speedup on gpt_oss.
At 8 gpt_oss shapes averaging 1.03 ratio, +4% → ~1.07 avg → +4 pp
on grp_FP8 geomean.

## Alternative R10 plan: dispatcher-level group_m tweak for gpt_oss

All micro-knob config sweeps are frozen per task body, BUT the ISA
dispatch logic for `fuse_ktail_eligible` could be extended with a
runtime "gpt_oss-specific" path. Risk: per-shape branching in dispatch
is dangerous (breaks uniformity of compile-time specialisation).

## Falsified this round

R9 re-verified that spec `<0,true,true>` interleaved-spill count is 22
(unchanged from R34 measurement). The ISA proxy was pointing us away
from the real bottleneck.

## Chat-window note

This chat has been running ~60 min (7 rounds). Next round may be a
cold-start new chat. If so, the R10 agent should read:
- `round-34-dm-fp8-isa-guided-dispatch-dsv3-to-fused-ktail-spec-wins-plus17pts.md` (win)
- `round-35-dm-fp8-n_masked_spec_collapse-falsified-by-runtime-branch-overhead-on-dsv3.md` (falsification)
- THIS NOTE (R10 plan + bottleneck analysis)

Tool: `analysis/tools/dump_fp8_grouped_isa.sh` at
`/workspace/code/Primus-Turbo/analysis/tools/` (emits per-spec ISA + spill/mfma histograms).
