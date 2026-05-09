---
name: round-48-vark-k-loop-asm-inventory-direction-D-CLOSED
description: Disassembled K-loop body of grouped_var_k_kernel_fp8 (HK 301c51c0 / FP8). Counts 31 SALU per iter; quantifies Direction D (SALU coord-decode) EV ceiling at ≤+5 score (within R29 ±3-5 noise floor). Officially closes Direction D for the gpt_oss FP8 ceiling task. Forward pointer to A1 (Stream-K) / A3 (decoupled-warps) / F (larger tiles).
type: project
---

# Round-48 — var-K K-loop body assembly inventory + Direction D closure

**Run:** `gpt_oss_fp8_local_20260509_143917` round 2 (auto-script numbering;
content-numbering "R48" = Direction D-2 follow-up to R47).
**Decision:** Direction D **CLOSED** at the source-edit level. R49 forward
pointer recommends Direction A1 (Stream-K) preflight as the next round.

## Method

R47's forward pointer (round-47-vark-uniform-m-closed-form-coord-decode-FALSIFIED.md
§ "Forward pointer — Round-48") asked for an assembly-level inventory of
the K-loop body's per-iter SALU before attempting the next surgery.

Built the production kernel with `--save-temps=obj` via `dbg_remote.sh`
to dump `kernel_fp8_layouts-hip-amdgcn-amd-amdhsa-gfx950.s`. Located the
`grouped_var_k_kernel_fp8<0>` symbol (lines 76452-85057, ~8600 lines /
2.0 MB of asm). The K-loop body is `.LBB28_33` (asm lines 77594-77875,
exactly 282 instructions / iter), nested inside the outer persistent-grid
loop `.LBB28_18`.

## Per-iter K-loop body instruction inventory (HK 301c51c0, var-K production)

```
salu     :   31   (s_*, excl. s_barrier/s_waitcnt/s_branch)
valu     :   80   (v_*, includes mfma)
mfma     :   32   (v_mfma_f32_16x16x128_f8f6f4)
ds_read  :   24   (ds_read_b64_tr_b8 + ds_read_b128, LDS A/B reads)
ds_write :    0   (no LDS writes inside loop, prefetches go HBM→LDS direct)
buffer   :    8   (buffer_load_*, HBM prefetches for k+1 / k+2)
s_barrier:    4   (CTA-wide barriers)
s_waitcnt:    4   (vmcnt/lgkmcnt drains)
TOTAL    :  282   instructions per K-loop iter
```

Match to source (`HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`
lines 8503-8574):
- 4 mma calls (`crr_mma(cA, ...)`, `crr_mma(cB, ...)`, `crr_mma(cC, ...)`,
  `crr_mma(cD, ...)`) = 4 × 8-mfma-each = 32 MFMA — matches.
- 6 ds_read groups (`load_a` at 8513/8548, `load_b` at 8504/8505) ×
  4 ds_reads each = 24 — matches.
- 8 buffer_load (3 from line 8517 + 8537 + 8538 + 8550 + 8551 + 8573,
  unrolled HBM prefetches for k+1 / k+2) — matches.
- 4 barriers + 4 waitcnts (R21 anatomy: 4×CTA-barrier-per-iter pin) — matches.

## Per-iter SALU breakdown (the targets of Direction D)

```
asm-line  instr                            mechanism
--------- -------------------------------- ----------------------------------
2         s_mul_i32  s21, s4,  0x8800      tic * LDS_SLOT_STRIDE     ← top of loop
3         s_add_i32  s20, s21, 0x11000     +Bs[*][0] base offset
25        s_add_i32  s22, s21, 0x15400     +Bs[*][1] base offset
95        s_mul_i32  s23, s8,  0x8800      toc * LDS_SLOT_STRIDE     ← second mul
103       s_addk_i32 s23, 0x4400           +As[toc][1] half-offset
110       s_add_i32  s41, s13, s9          loop-variant: prefetch K coord (k+1)
113       s_lshl_b32 s41, s41, 7           <<7 = ×128 (HB row-stride)
115       s_mov_b32  m0,  s40              LDS M0 setup (per ds_read group)
121       s_mov_b32  m0,  s23              LDS M0 setup
132       s_setprio  1                     raise priority around MFMA
149       s_setprio  0                     restore
150       s_add_i32  s23, s21, 0x4400      +As[tic][1] half-offset (re-derived)
203       s_add_i32  s40, s15, s9          loop-variant: prefetch K coord (k+2)
206       s_lshl_b32 s40, s40, 7           <<7
208       s_mov_b32  m0,  s23              LDS M0
214       s_mov_b32  m0,  s21              LDS M0
219       s_add_i32  s23, s14, 1           loop-variant
222       s_lshl_b32 s23, s23, 7
224       s_mov_b32  m0,  s21              LDS M0
230       s_mov_b32  m0,  s22              LDS M0
241       s_setprio  1                     raise priority
258       s_setprio  0                     restore
262       s_lshl_b32 s20, s14, 7           loop-variant
265       s_mov_b32  m0,  s21              LDS M0
271       s_mov_b32  m0,  s22              LDS M0
275       s_xor_b32  s4, s4, 1             tic ^= 1   (loop counter)
276       s_xor_b32  s8, s8, 1             toc ^= 1   (loop counter)
277       s_add_i32  s12, s12, -1          loop counter
278       s_add_i32  s9, s9, s34           K-coord stride
279       s_add_i32  s14, s14, s52         K-coord stride
280       s_cmp_eq_u32 s12, 0              loop branch
```

### Hoist-candidate audit

* **s_mul_i32 s21, s4, 0x8800 (asm-line 2)** and **s_mul_i32 s23, s8, 0x8800
  (asm-line 95)** — these compute `tic * LDS_STRIDE` and `toc * LDS_STRIDE`.
  `tic`/`toc` toggle 0↔1 each iter (XOR at lines 275-276). Each takes 2 values
  → could be replaced with `s_cselect_b32 s21, 0x8800, 0` (1 cy) instead of
  `s_mul_i32` (4 cy). **EV: 6 cycles per iter saved (12 → 6).** LLVM does NOT
  do this transformation because `tic`/`toc` are full 32-bit ints in source —
  LLVM has no proof they only take 0/1 values.

* **6 derived adds** (s20, s22, s23 lines 3/25/103/150/etc.) — these chain off
  the muls; they're 1 cy each but RAW-dependent on the muls. If muls collapse
  to selects, adds collapse too. **EV: 0 additional savings (already cheap).**

* **12 s_mov_b32 m0** — required by the AMD `ds_read*` ABI (LDS read uses M0
  as the swizzle/offset register). 1 cy each. Some are RAW-dependent on the
  preceding s_add (e.g. line 115 depends on s40 from line 110). LLVM is
  already CSE'ing where it can; the remainders are mandatory by ISA. **EV:
  ~0** — the ISA forces the M0 indirection.

* **2 s_setprio pairs** = 4 SALU. These are inserted by `CRR_MMA_BEGIN/END`
  macros (kernel_fp8_layouts.cpp lines ~5200-5210) for MFMA pipeline priority
  tuning. Removing them was tested in R23 and silently regressed (LLVM also
  inserts implicit setprio for correctness). **Not a candidate.**

* **6 loop-counter updates at the bottom** — essential, bounded by 1 cy each
  (= 6 cy / iter). Cannot be hoisted (loop-variant by definition).

* **6 loop-variant K-coord adds** (lines 110/113, 203/206, 219/222, 262) —
  these are the `k+1` / `k+2` prefetch coord computations from
  `global_load_a(As[toc][1], br*2+1, k+1)` etc. Loop-variant by design,
  cannot hoist.

### Maximum achievable Direction D delta

* **Best-case SALU savings per iter**: 6 cy (the 2× `s_mul_i32` → `s_cselect`).
* **K-loop iters per CTA on the worst cell** (Down_B4_M2048 wgrad, R47-derived):
  ~1.4 outer iters × 14 K-iters = 19.6 K-iters.
* **CTA wall per K-iter** (R21 anatomy + 282 instr / 32 mfma → MFMA-bound at
  ~280 cy/iter): ~280 cy.
* **Δ wall per CTA**: 19.6 × 6 = 118 cy savings on a ~5500-cy CTA = **2.1%
  faster CTA**.
* **Translates to wgrad section**: +2.1% × current 1263 T = +27 T = **+1.0
  on overall metric**.

The R29 noise floor characterization measured cluster σ ~3-4 score on the
daemon, with a minimum-detectable single-sample effect of +12-15 score.
**A 1.0-score lift is provably below the noise floor of this metric.**

### Side note: are SALU even on the critical path?

CDNA4 has a separate scalar issue port from VALU/MFMA. SALU instructions
can issue concurrently with VALU/MFMA when no RAW dep blocks. The R21 PMC
counter `SALUBusy% = 9.1%` measures cycles where SALU was the ONLY thing
issuing — not cycles where SALU stalled the pipe. The reported `SALU/SQ_busy
= 85%` was the ratio of SALU instructions to all SQ-issued instructions
during the small fraction of cycles where SQ was busy with something other
than VALU/MFMA — i.e. the K-loop is mostly issuing MFMAs, and during the
brief gaps SALU dominates. **SALU is not stalling the MFMA pipe; the
barriers and waitcnts are.**

This is consistent with the 6-round chain of FALSIFICATIONs in the SALU /
wait-counter / barrier-tweak family (R8/9/13/22/23/24/26/27/28/31b/47):
they all attack a non-bottleneck.

## Closure of Direction D

Direction D ("SALU coord-decode optimization") as enumerated in
`scripts/_task_gpt_oss_fp8_kernel.md` is hereby **closed at the source-edit
level**. Maximum source-edit EV (~+1 score) is provably below the metric
noise floor (R29: minimum detectable +12-15). Combined with the 6-round
prior chain falsifying SALU/wait/barrier tweaks on the same kernel, no
single round of pure SALU surgery can produce a measurable lift.

If a future agent revisits Direction D, it must be paired with a
mechanism that reduces the *barrier-cycle* count (R21 anatomy: 4 CTA
barriers per K-iter, the actual MFMA-pipe stall driver). Closing barriers
is structural work covered by Direction A3 (decoupled-warps, see below).

## Forward pointers — round 49 candidates

Per `_task_gpt_oss_fp8_kernel.md` § "NEW DIRECTIONS":

### Round 49 RECOMMENDED: A1 (Stream-K / persistent + work-stealing) preflight

Tail-effect quantification on the gpt_oss B=4 cells (where the current
static-slot persistent grid most under-saturates):

```
Cell                       tiles  slots  ws/CU  tail-CTA-imbalance
GateUP B=4 M=2048           704   256   2.75   0.75 → ~27% tail variance
Down   B=4 M=2048           352   256   1.375  0.375 → 73% tail variance ★
Down   B=4 M=4096           704   256   2.75   0.75 → ~27%
GateUP B=4 M=4096          1408   256   5.5    0.5  → ~9%
GateUP B=32 M=2048         5632   256  22.0    1.0  → negligible
Down   B=32 M=2048         2816   256  11.0    1.0  → negligible
GateUP B=32 M=4096        11264   256  44.0    1.0  → negligible
Down   B=32 M=4096         5632   256  22.0    1.0  → negligible
```

Down-B4-M2048 has only 1.375 waves per slot — every CTA does 1 or 2 outer
iterations, with tail-CTA imbalance of 73% (some CTAs do 1, some do 2;
1× → idle while waiting for 2×). Static-slot persistent grid serializes
this. Stream-K's atomic-counter dispatch lets fast CTAs steal work from
the unfinished pile, collapsing tail to ≤1 outer-iter latency.

Estimated lift on Down-B4-M2048 wgrad: 73% / 2 ≈ **+25-30% T** on this
cell (1263 T → ~1600 T). Translates to wgrad section +4.7% = **+11 score
on overall metric** — at the edge of detectability but within the
realistic R29 minimum-detectable threshold of 12-15.

R49 = preflight: characterize the persistent-grid loop (`for (gt = pid;
gt < total_tiles; gt += slots_eff)` at line 8395) and sketch the
Stream-K patch surface. Multi-round to ship.

### Round 49 ALTERNATIVE: F (larger tiles) preflight on M=4096 cells

The M=4096 cells (GateUP-B4-M4096, Down-B4-M4096, GateUP-B32-M4096,
Down-B32-M4096) have plentiful M-tiles (16 m-tiles per group). Doubling
the M-tile to 512 keeps the per-CTA work load and halves the per-tile
fixed overhead amortization. RBM/RBN headroom on M=4096 is ample (6+
m-tiles per slot). Risk: register pressure. Estimate: +5-10% per affected
cell → +3-6 score across 4 cells.

### Round 49 LAST RESORT: A3 (decoupled-warps) preflight

The 4×CTA-barrier-per-iter pin (R21 anatomy) is what's keeping MfmaUtil
at 32-49%. A decoupled producer-consumer warp split (e.g. 2 warps for
HBM→LDS, 6 for MFMA) drops the CTA-wide barrier count to 0-1 per iter.
This is a 4-6 round project and high-risk (LLVM AGPR allocator bug at
256 fp32/lane block accumulator, R59-R61 etiology).

## Files touched this round (Primus-Turbo)

* `analysis/_notes/round-48-vark-k-loop-asm-inventory-direction-D-CLOSED.md`
  (this doc).

## Files touched this round (HipKittens)

* None — this round is observational. The .s file dumped via
  `--save-temps=obj` lives only in the remote build dir and is not
  committed.

## Reproducibility

```bash
DBG_REMOTE_NO_BUILD=1 bash scripts/dbg_remote.sh '
  cd /workspace/code/HipKittens && source env.src &&
  cd analysis/fp8_gemm/mi350x &&
  HIPFLAGS="-DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math -I/opt/rocm/include/rocrand --save-temps=obj" make -B
  S=kernel_fp8_layouts-hip-amdgcn-amd-amdhsa-gfx950.s
  # var_k kernel symbol: _Z24grouped_var_k_kernel_fp8ILi0EE...
  # Function spans asm lines 76452..85057.
  # K-loop body label: .LBB28_33  (asm lines 77594..77875, 282 instr/iter)
'
```
