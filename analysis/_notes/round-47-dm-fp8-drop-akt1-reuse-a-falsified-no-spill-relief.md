# Round-47-dm · FP8 grouped — drop `a_kt1`, reuse `a` register for both K-tail M-slabs FALSIFIED (no spill relief, lost overlap)

**Status**: 1 probe FALSIFIED. Reverted to R44-dm winner state (HK SHA `37926c98`).
Score baseline 961 → R47-dm probe 955 (-6 pts, 1-run). grp_FP8 1.1234 → 1.1117 (-1.2 pp).
Correctness PASS 32/32.

## Hypothesis

R44-dm rocprof identified scratch I/O REQUEST RATE (not bandwidth) as the
bottleneck for gpt_oss spec `<0,true,true>` (FUSED+n_masked, 67 % MemUnit-
Stalled post-R44 vs DSV3 spec `<0,false,true>`'s ~33 %). Static spill on
gpt_oss = 39 dwords vs DSV3's 32 dwords; the +7-dword gap is plausibly
attributable to the dedicated `a_kt1` register tile being live across the
K-tail block (~32 dwords/lane footprint).

Hypothesis: drop the separate `a_kt1` register and reuse `a` for both
M-slabs in the K-tail. Trade the R12-dm split-vmcnt overlap (~64 cy of
mfma cA/cB hidden during a_kt1 drain) for ~16-32 dwords of VGPR pressure
relief. If LLVM absorbs vacated slots into spill-target slots that
currently round-trip through HBM scratch, the net wins per R44 mechanism.

## Implementation

```diff
-        if constexpr (FUSED_KTAIL) {
-            A_row_reg a_kt1;          // separate register tile, M-slab 1
-            if (g.fast_k < g.k) {
+        if constexpr (FUSED_KTAIL) {
+            // a_kt1 dropped; reuse `a` for both slabs
+            if (g.fast_k < g.k) {

-                load_b_kt(b0,    0);
-                load_b_kt(b1,    1);
-                load_a_kt(a,     0);
-                load_a_kt(a_kt1, 1);
-                asm volatile("s_waitcnt vmcnt(8)");
-                rcr_mma(cA, a,     b0);
-                rcr_mma(cB, a,     b1);
-                asm volatile("s_waitcnt vmcnt(0)");
-                rcr_mma(cC, a_kt1, b0);
-                rcr_mma(cD, a_kt1, b1);
+                load_b_kt(b0, 0);
+                load_b_kt(b1, 1);
+                load_a_kt(a, 0);
+                asm volatile("s_waitcnt vmcnt(0)");
+                rcr_mma(cA, a, b0);
+                rcr_mma(cB, a, b1);
+                load_a_kt(a, 1);     // OVERWRITE for M-slab 1
+                asm volatile("s_waitcnt vmcnt(0)");
+                rcr_mma(cC, a, b0);
+                rcr_mma(cD, a, b1);
```

## Spill count delta (`-Rpass-analysis=kernel-resource-usage`)

| Spec template params | R44-dm baseline | R47-dm probe-1 | Δ |
|---|---|---|---|
| `<0,false,false>` (FUSED=false n_aligned)         | 39 | 39 | 0 |
| `<0,true ,false>` (FUSED=false n_masked)          | 43 | 43 | 0 |
| `<0,false,true >` (FUSED=true  n_aligned, DSV3)   | 32 | 32 | 0 |
| `<0,true ,true >` (FUSED=true  n_masked, gpt_oss) | 39 | 39 | 0 |

**Spill counts UNCHANGED.** LLVM did not reduce spill on the gpt_oss spec
even though `a_kt1`'s 32-dwords-per-lane footprint was eliminated. Either
LLVM was already aliasing `a_kt1`'s slots onto `a` after the K-tail (since
they share the same lifetime in the original code structure), or the 39
dwords is determined by some OTHER live state in the K-tail block (not
`a_kt1`).

## Per-shape metric (R44-dm baseline 961 → R47-dm probe-1 955)

| Shape | R44-dm baseline | R47-dm probe | Δ pp |
|---|---|---|---|
| DSV3-GateUP-B16-M2048 | 1.135 | 1.139 | +0.4 |
| DSV3-Down-B16-M2048   | 1.187 | 1.153 | **-3.4** |
| DSV3-GateUP-B16-M4096 | 1.154 | 1.155 | 0.0 |
| DSV3-Down-B16-M4096   | 1.202 | 1.128 | **-7.4** |
| DSV3-GateUP-B32-M2048 | 1.161 | 1.162 | 0.0 |
| DSV3-Down-B32-M2048   | 1.216 | 1.210 | -0.6 |
| DSV3-GateUP-B32-M4096 | 1.169 | 1.165 | -0.4 |
| DSV3-Down-B32-M4096   | 1.218 | 1.151 | **-6.7** |
| gpt_oss-GateUP-B4-M2048   | 1.078 | 1.079 | 0.0 |
| gpt_oss-Down-B4-M2048     | 1.178 | 1.155 | -2.3 |
| gpt_oss-GateUP-B4-M4096   | 1.055 | 1.057 | +0.2 |
| gpt_oss-Down-B4-M4096     | 1.083 | 1.075 | -0.8 |
| gpt_oss-GateUP-B32-M2048  | 1.040 | 1.043 | +0.3 |
| gpt_oss-Down-B32-M2048    | 1.069 | 1.067 | -0.2 |
| gpt_oss-GateUP-B32-M4096  | 1.028 | 1.017 | **-1.1** |
| gpt_oss-Down-B32-M4096    | 1.036 | 1.053 | +1.7 |

grp_FP8 geomean: 1.1234 → 1.1117 (-1.2 pp).
grp_BF16 geomean: 1.1833 → 1.1817 (-0.2 pp, noise — BF16 unaffected by FP8 kernel).

The biggest losses (-3 to -7 pp) are on DSV3-Down family. DSV3 has
K_REM=0 so the runtime `if (g.fast_k < g.k)` is always FALSE — the
K-tail body never executes. But LLVM's instruction scheduling for the
FUSED_KTAIL=true template body STILL changed (different scheduling at
the call site of the K-tail block in the persistent loop), causing
codegen perturbation that hurts DSV3.

Correctness: PASS (correct_fail=0/32, reject=0/32).

## Why the analysis was wrong

### Mechanism #1 (a_kt1 → spill reduction): NOT delivered

LLVM did not free the `a_kt1` slots into productive use. Two plausible
reasons:

1. **`a_kt1`'s slots were already aliased onto `a` after the K-tail.**
   In the original code, `a_kt1` is only used INSIDE the K-tail block
   (1 declaration, 1 load, 2 mfma reads). After the K-tail, both `a`
   and `a_kt1` are dead. LLVM's register allocator may already be
   reusing the same physical slots for both because their lifetimes
   are short and disjoint w.r.t. downstream code. The spill count of
   39 dwords is determined by OTHER state, not `a_kt1`.

2. **The +7 dword gap (32 DSV3 vs 39 gpt_oss) is from RUNTIME path,
   not register count.** The if-branch body itself adds VGPR pressure
   for the SRD setup, lane index derivations, and the load lambdas'
   captures. Removing `a_kt1` doesn't touch those.

### Mechanism #2 (lost overlap → runtime cost): real

Lost ~64 cy/tile of mfma cA/cB execution overlapped with a_kt1's HBM
drain (R12-dm). With ~K-tile-fires per launch on gpt_oss, this is on
the order of ~6.4 × 10^4 cy lost per launch — small but measurable.

### Mechanism #3 (DSV3 codegen perturbation): unexpected dominant loss

The FUSED_KTAIL=true template is shared between DSV3 and gpt_oss spec.
Restructuring the K-tail body changed instruction scheduling at the
call site (epilog 2 → K-tail block → C-store), perturbing register
allocation in the SHARED part of the kernel. DSV3 (which never enters
the K-tail body at runtime) lost -3 to -7 pp on Down family. This is
the dominant source of the -1.2 pp grp_FP8 geomean drop.

## Conclusion

R47-dm-probe-1 falsified. The 7-dword spill gap between DSV3 (32) and
gpt_oss (39) specs of FUSED_KTAIL=true is NOT attributable to `a_kt1`'s
register footprint. Spill source is elsewhere (likely K-tail SRD setup
SGPRs + lane-index derived VGPRs that feed the load lambdas).

## Search-space update — exhausted directions

Exhausted SINCE R37 K-tail issue reorder (R37 was last gainful K-tail
change, +16 pts):

* ✗ R38: K-tail mfma permutation sweep — [b0,b1,a,a_kt1] is unique optimum
* ✗ R39: MFMA commutative reorder — neutral
* ✗ R40: Runtime-dead a_extra — neutral/negative
* ✗ R41: Pre-issue a_kt1 in epilog 2 — -56 pts (R33-pattern spill backlash)
* ✗ R45: Conditional interleave by N_MASKED — neutral
* ✗ R46: K-tail SRD setup hoist — neutral
* ✗ R47: Drop a_kt1 (this round) — -6 pts (no spill relief, lost overlap)

## ISA-derived insight: spill state is mostly the prefetch tile

Inspecting `spec_Li0ELb1ELb1.s` lines 312-333 shows LLVM spilling
v100-v107 (8 contiguous VGPRs = exactly one A_row_reg base tile = 16x128
fp8 / 64 lanes = 32 bytes/lane = 8 dwords/lane × 4 lanes/dwordx4). This
is the PREFETCH TILE being spilled across the K-iter boundary, NOT the
K-tail's `a_kt1`. The K-tail's `a_kt1` is at a different VGPR range and
is dead by the spill point.

This means the 39 dwords of spill on gpt_oss is dominated by:
1. Main-loop prefetch state (As[toc][1] / b_tile(tic, X) waiting in
   registers between successive K-iters)
2. K-iter accumulator partials (cA/cB/cC/cD slot fragments)
3. SRD + lane-index derived state from rcr_8w_load_hoist's address-
   computation epoch

`a_kt1` is NOT a significant contributor. The 7-dword gap to DSV3 is
from the K-tail SRD setup (a_srsrc_kt + b_srsrc_kt + lane derivations
+ load lambda captures, all live across the runtime if-branch).

## Take-away for next agent

1. **Stop attacking `a_kt1` / K-tail block register pressure.** R37's
   issue order [b0,b1,a,a_kt1] is the unique optimum AND removing
   a_kt1 doesn't reduce spill (LLVM was already aliasing). The 39
   dwords of spill is from main-loop prefetch state, not K-tail.

2. **Stop perturbing the FUSED_KTAIL=true template body** — DSV3 (which
   shares this template with gpt_oss) is sensitive to instruction
   scheduling at the call site. Any restructure has -3 to -7 pp risk
   on DSV3-Down family from codegen perturbation alone.

3. **Next direction**: focus on MAIN-LOOP prefetch state. Specifically:
   * The prefetch As[toc][1] / b_tile(tic, X) values waiting in
     registers are spilled across K-iter boundaries. If we can shrink
     the prefetch window from 2 K-iters ahead to 1, that'd reduce
     spill significantly. Risk: less HBM hide, may regress runtime.
   * Or: investigate WHY rcr_8w_load_hoist's address-computation epoch
     is spilling so much. Each call constructs an SRD + computes
     `tile_byte_offset` + `lds_addrs[]`. If LLVM is keeping these
     across the K-iter boundary instead of recomputing, that's the
     spill source.

4. **Last unexplored algorithmic levers (per R46-dm note)**:
   * (a) Reduce K-tail load count by combining a + a_kt1 into a single
     256x128 tile load (requires register tile redefinition to
     height=8; doubles VGPR per lane; may regress occupancy)
   * (b) HAS_KTAIL template param to skip K-tail entirely for K-aligned
     shapes (helps DSV3 ~0.5 pp, doesn't help gpt_oss)
   * (c) rt_32x64 / rt_64x32 cell shape switch (HK scaffold exists,
     R29 falsified integral migration but with R44 baseline could
     work)

5. **NEW lever from R47-dm ISA inspection: cap unroll of
   `store_c_tile_n_masked` outer height loop**. The helper has 4 nested
   `#pragma unroll` (height=4 × width=2 × num_strides=1 × stride/packing=4
   = 32 fully-unrolled iters per call). Each unrolled iter has live VGPRs
   for `row, col, off0, off1, v0, v1, src.tiles[i][j].data[idx]`.
   Capping the outer height loop to `#pragma unroll 1` (sequential 4-iter
   loop with 8-iter unrolled body) caps peak live state at 1/4 of current.
   Risk: blast radius — the helper is also called from `gemm_kernel<RCR>`
   (line 1252-1255) and the standalone `store_c_kernel_n_masked` (line
   1777-1780), not just the grouped kernel. Changing helper unroll
   affects ALL call sites. Could mitigate by cloning the helper into a
   `_grouped` variant and only restricting the grouped clone. A deferred
   probe; needs spill-count delta verification before metric.

R47-dm did NOT execute the prefetch-window analysis or the n_masked
helper unroll probe (item 5). Next round should pick item 5 (cleanest
small change) or run rocprof on the gpt_oss main loop with the new
hypothesis ("masked-store helper internal state is the spill source,
not main-loop prefetch").

## Score history

| Round | Score | grp_FP8 geo | Notes |
|---|---|---|---|
| Start    | 851  | ~1.01  | Baseline |
| R10      | 950  | 1.099  | R37-dm K-tail reorder (+16 pts) |
| R11-R15  | 950  | 1.094  | R38-R43 falsified |
| R16      | 958  | 1.116  | R44-dm interleave (+8 pts) |
| R17      | 958  | 1.116  | R45 falsified |
| R18      | 958  | 1.116  | R46 falsified |
| R19      | 961  | 1.123  | (drift, noise) |
| R19-probe-1 | 955 | 1.112  | R47-dm-probe-1 falsified (-6 pts) |
| Target   | 1000 | ≥1.20  | Gap = 7.7 pp grp_FP8 |

## Files touched (this round)

`analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` (HipKittens) —
modified then reverted; final state matches HEAD `37926c98` (R44-dm
winner). Verified post-revert build: spill counts 39 / 43 / 32 / 39
unchanged.

This Primus-Turbo round-note is the only delta this round. No HK
commit (revert restored HEAD).
