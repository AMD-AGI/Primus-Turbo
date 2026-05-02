# Round 19 — FP8 grouped: Direction B falsified by prior RRR path A round-28 attempts

**Status**: R18's proposed Lever H Direction B (in-kernel fused dA-transpose)
**FALSIFIED** before any new code changes. Two independent prior attempts
already reached the same dead-end with documented compiler register
aliasing across cooperative-op gaps:

* HK round 28 (`round-28-fp8-rrr-path-a-aliasing-fixes-fail.md`):
  5 distinct fix attempts on FP8 RRR in-kernel K-tail fuse, **all reach
  SNR ~15 dB** (below the 16.56 dB no-K-tail floor).
* HK round 29 (`round-29-bf16-rrr-path-a-address-derivation-confirmed.md`):
  BF16 RRR path A probe with corrected address derivation — **SNR 19.59 dB**,
  identical to manual ds_read baseline. Same fundamental block.

Both rounds documented the failure mode: cooperative LDS load for B
register tile (col_l layout) inherently retires the A register tile, the
compiler reassigns A's VGPRs to the live c register pressure pool, and
load_a_kt subsequently writes corrupt c bytes regardless of any inline-asm
pin pattern.

**Auto-optimize round**: 19 / 100
**Date**: 2026-05-02
**HK SHA at round start / end**: `0f14b165` (no kernel rebuild this round)
**PT SHA at round start**: `74fdead4`
**Reported best (forward)**: 966 (R15 / R18, high-tail of noise band)
**R19 baseline metric**: 960 (single trial); **4-trial median = 962** (959-966
trials this round: 960, 965, 961, 962)
**R19 patience**: 4 rounds at noise floor (R16=964, R17=963, R18=966, R19=960)

---

## What R18 proposed and what's actually possible

### R18 Direction B (as proposed)

> Add a new `B_INLINE_TRANSPOSE = true` template specialisation to
> `grouped_rcr_kernel` that swaps `(k_idx, n_idx)` voffset →
> `(n_idx, k_idx)` at load time, eliminating the explicit
> `fp8_transpose_3d` pre-pass for the gpt_oss reroute subset.

### Why it doesn't work — TWO independent dead-ends

**Dead-end A (memory access pattern)**: voffset swap in RCR's b-load
would require uncoalesced reads. `b` is stored `(N_fwd, K_fwd)`
row-major; reading `b_logical[k_idx][n_offset..n_offset+K_BLOCK]` (the
swapped pattern) maps to physical offsets `(n_offset+i)*K_fwd + k_idx`
for `i ∈ [0, K_BLOCK)` — `K_BLOCK` separate `K_fwd`-strided HBM reads
per K-iter per warp. This is exactly the access pattern that drove R13
to introduce `fp8_transpose_3d` in the first place. No voffset
trick converts an uncoalesced layout into a coalesced one.

**Dead-end B (compiler register aliasing)**: The "in-kernel K-tail fuse
on RRR" reformulation — which is functionally equivalent and the
intended target of R18's Direction B — was attempted in HK round 28
(FP8) and round 29 (BF16). Both rounds:

| Attempt | Layout | Floor SNR | Best probe SNR | Verdict |
|---|---|---|---|---|
| Production (external launches) | RRR | n/a | 43.99 dB (FP8) / 44.34 dB (BF16) | reference baseline |
| Path A in-kernel fuse | FP8 RRR | 16.56 dB | 15.05 dB (5 fixes tried) | FALSIFIED |
| Path A in-kernel fuse | BF16 RRR | n/a | 19.59 dB (4 variants tried) | FALSIFIED |

Root cause (from round-28 analysis):

> RRR's B is col_l register — fundamentally requires LDS-staged load
> via ds_read_b64_tr_b8 (transpose-from-LDS) for the col_l layout. No
> way to avoid cooperative LDS load for B, hence no way to keep `a`
> live across the K-tail boundary.

Even the most aggressive fix (pin all `cA/cB/cC/cD` dwords with `+v` +
fresh `A_row_reg a_kt` declaration) failed — the compiler maps
`a_kt`'s VGPRs to whichever slots are "free" at issue time, and after
cooperative ops those slots overlap c.

---

## Implications for R19+ planning

R18's note proposed Direction B as the highest-EV next lever (3-5 rounds,
+5-8 % bwd wall on gpt_oss reroute). With both dead-ends documented,
**this lever is FALSIFIED at design time**, not requiring any HK kernel
attempts in R19.

The R19+ lever space is now:

### Untried FORWARD-track levers
None remain within the current `grouped_rcr_kernel` design. R5/R11/R17/
R28/R29 cumulative falsification matrix:

| Lever | Verdict | Round | Mechanism |
|---|---|---|---|
| **A** Async global→LDS | FALSIFIED | R2 | Already shipped via inline ASM |
| **B** Triple LDS slab | FALSIFIED | R2 | LDS at 137/160 KB cap |
| **C-X** N_MASKED helper SENTINEL | FALSIFIED | R4 | Spill neutral, -1.8 pp regression |
| **C-2** K-tail capture refactor | FALSIFIED | R3 | Already correctly scoped |
| **D** mfma_32x32x64 cell-shape | FALSIFIED | R5 | Microbench -0.03 % |
| **E** ASM software pipelining (iter-level) | FALSIFIED | R11 | Microbench -7.28 % |
| **R** Stage-level pipelining | FALSIFIED | R17 | Microbench -0.07 % (LLVM auto-overlaps) |
| **H/B-rcr** voffset swap | FALSIFIED | R19 | Uncoalesced HBM reads (this note) |
| **H/B-rrr** in-kernel K-tail | FALSIFIED | R28 (FP8) / R29 (BF16) | Compiler aliases A→c VGPRs across coop gap |
| **F** Per-shape dispatcher rules | LANDED & SATURATED | R6-R10 | 5 rules landed, R10-dm audit confirmed top-1 |

### Untried BACKWARD-track levers (no metric movement)

1. **CRR (var-K dB) main-line fuse** — round-28 alternative #2.
   *Status*: STRUCTURALLY MOOT. `grouped_var_k_kernel_fp8` already runs
   as a single launch with `store_c_tile_mn_masked_grouped` for partial
   tiles (kernel_fp8_layouts.cpp:5896-5898). No external K-tail kernels
   exist for var-K — R14 Lever K already trimmed the spill from 52 →
   37 dw. No fuse left to attempt.

2. **Custom HIP `fp8_transpose` kernel** to replace Triton
   `fp8_transpose_3d`. *Estimated*: Triton at 3.4 TB/s eff vs HBM peak
   5.3 TB/s; a HIP kernel could close the 36 % gap → ~38 µs saved per
   call → ~1.3 % bwd wall improvement. Low EV but tractable in 1 round.

3. **External RRR tail kernels** (`grouped_ktail_kernel_lds_rrr`,
   `grouped_ntail_kernel_lds_rrr`, `grouped_tail_kernel<RRR>`) — not
   triggered by any metric shape (all metric shapes either align cleanly
   or take the H4 reroute). Optimising these is dead-code maintenance.

### Untried MAJOR-rewrite levers (multi-round, high risk)

1. **Wave-specialisation** (4-8 rounds, novel for HK). Splits warps
   into producer/consumer roles. No precedent.

2. **Block-CCR layout** (5-15 rounds). Replaces RCR with CCR in the
   forward path. Wide regression surface across all 24 shapes.

3. **Hand-written ASM mfma+vmcnt schedule** outside the kittens
   register-tile abstraction (round-28 alternative for sidestepping
   compiler aliasing). Very high effort; round-28 explicitly noted
   "very invasive (requires deriving fp8 mfma instruction encoding)".

---

## R19 recommended action: ACCEPT plateau, PIVOT to backward maintenance

After R5-R19's 14 rounds of disciplined falsification, the FP8 grouped
forward kernel is at an architectural ceiling. The score band 959-966
(median ~963) represents the kernel plateau. Per-trial variance (~5
score points) dominates any sub-pp lever-class improvement.

The user-budget remaining (R20 .. R100, 81 rounds) is best spent on:

* **Backward-track maintenance** — keep `bench_grouped_gemm_turbo` ahead
  of TRT (currently +27 % avg bwd). One round of custom HIP fp8_transpose
  if that's the lowest-hanging fruit. Each round is concrete, low-risk,
  metric-invariant.

* **BF16 grouped maintenance** — same suite, 24 shapes, currently at
  geomean 1.189 (R19 = within 1.183-1.191 band). Symmetric to FP8;
  same lever exhaustion.

* **OR pause auto-optimize** — given the score plateau, future optimisation
  rounds will not change the metric. Continued runs only confirm the
  plateau; the user may want to redirect budget elsewhere.

R20 should pick the **custom HIP fp8_transpose** lever as the only
remaining backward-track item with positive expected value, OR explicitly
acknowledge the plateau and stop further rounds. Both are valid.

---

## R19 metric data

```
$ python3 scripts/_metric_grouped_only.py 2>&1 | tail -1   # initial
960
$ for i in 1 2 3; do python3 scripts/_metric_grouped_only.py 2>&1 | tail -1; done
965
961
962
```

R14-R19 cumulative score history: 962, 966, 964, 963, 966, 964, 962, 964, 964, 959, 960, 965, 961, 962. Min = 959, max = 966, range = 7, median = 963.

Worst FP8 case (R19): gpt_oss-GateUP-B32-M4096 at 1.027 (HK=1976, TRT=1924).
Same as R16-R18; 6-case gpt_oss FUSED_KTAIL+N_MASKED cluster
(architectural ceiling per R3 spill analysis: 67 dw spill on
`<0,T,T>` template).

---

## Files touched in R19

* `analysis/_notes/round-19-fp8-grouped-direction-b-falsified-by-prior-rrr-path-a.md` (NEW)

No HK kernel changes, no PT runtime changes, no rebuild.
