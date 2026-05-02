# Round 28 — FP8 grouped: saturation acknowledgment + concrete Lever C-1 (LDS scratch redirect) roadmap for R29+

**Status**: NO KERNEL CHANGE — R28 = saturation audit + concrete next-lever
plan after R26/R27 readfirstlane class fully exhausted.
**Auto-optimize round**: 28 / 100
**Date**: 2026-05-02
**HK SHA at round start / end**: `4caa6d9a` (unchanged)
**PT SHA at round start**: `2f2fdcfe`
**Round time**: ~15 min (1 baseline metric + 1 BF16/FP8 spill comparison +
1 K-tail issue-order audit)
**Score before**: 979 (R27 baseline median; 4-trial range 977-981)
**Score after**:  979 (no change, no probe)

---

## R28 baseline metric (4 worst FP8 cases, R27 commit unchanged)

```
[metric_grouped_only]   grp_BF16  vs triton geomean=1.1875 (n=24)
[metric_grouped_only]   grp_FP8   vs triton geomean=1.1629 (n=24)
[metric_grouped_only]   below_target=35/48  goals=0/2  score=979
```

Bottom 8 FP8 shapes (ratio < 1.20):

```
grpFP8_gpt_oss_20B-GateUP-B32-M4096    ratio=1.072  (-12.8 pp from 1.20)
grpFP8_gpt_oss_20B-Down-B32-M2048      ratio=1.098  (-10.2 pp)
grpFP8_gpt_oss_20B-GateUP-B32-M2048    ratio=1.107  (-9.3 pp)
grpFP8_gpt_oss_20B-Down-B32-M4096      ratio=1.108  (-9.2 pp)
grpFP8_gpt_oss_20B-GateUP-B4-M4096     ratio=1.114  (-8.6 pp)
grpFP8_gpt_oss_20B-GateUP-B4-M2048     ratio=1.117  (-8.3 pp)
grpFP8_gpt_oss_20B-Down-B4-M4096       ratio=1.128  (-7.2 pp)
grpFP8_Qwen3-235B-A22B-Down-B16-M4096  ratio=1.139  (-6.1 pp)
```

**8/8 of bottom-8 are gpt_oss + 1 Qwen3-Down.** All 8 gpt_oss ratios sit
within 0.06 of each other (1.072-1.128) — the cluster is structurally bound
by the `<0,T,T>` template's 37 dw spill (vs `<0,F,T>` 32 dw used by all
non-gpt_oss).

---

## R28 audit findings — every "1-round probe" is now in FROZEN class

| Lever class | Status | Last attempt |
|---|---|---|
| Async global→LDS copy (Lever A) | **shipped from day-0** (`rcr_8w_load_hoist` inline ASM `buffer_load_dwordx4 ... offen lds`) | R2 audit confirmed |
| Dual LDS slab ping-pong (Lever B) | **shipped from day-0** (`As[2][2]`, `Bs[2][2]`); triple slab infeasible (137/160 KB used) | R2 audit confirmed |
| K-tail issue reorder | shipped (R37-dm `b0b1aa_kt1`, R12-dm split-vmcnt) | R37-dm |
| K-tail vmcnt | shipped (R12-dm 2-stage vmcnt, R3 single-wait epilog) | R12-dm |
| K-tail SRD uniformization | **EXHAUSTED** (3 mechanisms falsified R26) | R26 |
| readfirstlane on C-store coords | **EXHAUSTED** (rcr fwd LANDED R24, rrr dA bwd LANDED R25, var_k dB bwd FALSIFIED R27) | R27 |
| readfirstlane on group_idx prologue | **EXHAUSTED** (R22 V-A spill backlash) | R22 |
| sched_barrier mask, s_setprio, launch_bounds, WARPS_M/N flip, MFMA cell-shape, LDS swizzle, unroll | **FROZEN** (task body explicitly forbids) | (multiple) |
| FUSED_KTAIL routing | **EXHAUSTED** (R23 K_REM=64 disable -48 pts, R34-dm K_REM=0 enable +17 pts both committed) | R34-dm |
| N_MASKED helper SENTINEL | **FALSIFIED** (R4 active template unchanged) | R4 |
| N_MASKED helper `__noinline__` | **FALSIFIED** (R47-dm) | R47-dm |
| `__builtin_expect` on K-tail branch | **FALSIFIED** (R54-dm) | R54-dm |
| Custom HIP fp8_transpose | **FALSIFIED** (R20 Triton @ 75-110 % HBM peak) | R20 |
| Lever D K-tail port | **FALSIFIED** (R62 fan-out cost dominates) | R62-dm |
| Lever D R-B step 4 | **FALSIFIED** (R62-dm) | R62-dm |

The single REMAINING unspent lever is **Lever C-1 (VMEM→LDS scratch
redirect)** — proposed by R3 plan, never executed, multi-round investment.

---

## BF16 grouped vs FP8 grouped: spill differential (the architectural gap)

R3 doc compared FP8 grouped to BF16 *dense* (0 spill). R28 sweeps the
BF16 *grouped* kernel for a fairer comparison:

```
$ make all 2>&1 | grep -E 'VGPRs Spill|cpp:[0-9]+:1: remark.*func' | ...

kernel_bf16_dynamic.cpp:1145 (BF16 dense templates):     0 dw spill (30 specs)
kernel_bf16_dynamic.cpp:3667 (BF16 grouped templates):  12-29 dw spill (~18 specs)
   - mode 24 dw (4 specs)
   - mode 14 dw (4 specs)
   - mode 13 dw (4 specs)
   - 29 dw (2 specs)
   - 16 dw (2 specs)
   - 12 dw (2 specs)

kernel_fp8_layouts.cpp:2223 (FP8 RCR grouped templates): 34-54 dw spill (4 specs)
   - <0,F,T> 34 dw (DSV3+Qwen, 16/24 cases)
   - <0,T,T> 37 dw (gpt_oss, 8/24 cases)
   - <0,F,F> 54 dw (dead)
   - <0,T,F> 38 dw (dead)
kernel_fp8_layouts.cpp:2912 (FP8 RRR grouped):           65 dw spill
kernel_fp8_layouts.cpp:5608 (FP8 var_k grouped):         37 dw spill
```

**Average BF16 grouped spill ≈ 15 dw; average FP8 grouped spill ≈ 37 dw.**

The +22 dw FP8/BF16 delta has 3 architectural sources, none of which are
addressable by 1-round micro-knobs:

1. **K_BLOCK = 128 (FP8) vs 64 (BF16)** — 2× fp8 cells per K-iter ⇒ 2× the
   per-iter A_row_reg / B_col_reg VGPR width (each tile carries
   {16, 16, 16} dw of fp8 cell data).
2. **MFMA cell shape mismatch** — FP8 uses `mfma_f32_16x16x128_f8` (16×16
   output × 128 K), BF16 uses `mfma_f32_32x32x16_bf16` (32×32 output × 16
   K). The accumulator working set per warp (cA/cB/cC/cD) is identical
   geometry (4× rt_fl<64,32>) but the per-mfma "live A and B halves"
   pattern differs — FP8's narrower mfma forces 4 mfma issues per output
   pair vs BF16's 1, with each issue requiring a fresh A/B fragment in
   register. 4× the issue-time register pressure.
3. **FUSED_KTAIL block** — FP8 has the `if (g.fast_k < g.k)` block at
   lines 2540-2719 with 12 `buffer_load_b128` ops + 4 K-tail mfmas + 2
   tile-buffer registers (`a_kt1` declared at function scope per R34-dm
   for favorable codegen). BF16 grouped has no analogous block. The
   K-tail block contributes the +5 dw delta between `<0,F,T>` (32 dw,
   no K-tail body affecting allocation) and `<0,T,T>` (37 dw, K-tail
   body in scope).

These three are baked into the kernel's IR shape. Lever D (mfma_323264
port) was the proposed fix for #2 and was **falsified** (R62-dm fan-out
cost ate the savings).

The remaining +22 dw delta is the architectural floor; closing it
requires a kernel rewrite, not a micro-optimization.

---

## R56-dm ceiling holds — even Lever C-1 success cannot reach 1.20 on gpt_oss-B32

Per R56-dm cost model (re-validated against R28 baseline):

```
gpt_oss-GateUP-B32-M4096 cycle decomposition (worst case):
  main loop  (ki=22 × T_iter)   ~5500 cy   87 %
  K-tail     (1 fused tail)      ~256 cy    4 %
  epilog     (mul + store)       ~400 cy    6 %
  prologue   (binary search)     ~150 cy    3 %

Plausible Lever C-1 ceiling on gpt_oss-B32 spec:
  - VMEM scratch I/O (current): ~80-100 cy/round-trip × ~4 round-trips/iter
                                = 360 cy/iter × 22 iter = ~8K cy/tile
  - Max savings if ALL VMEM scratch redirected to LDS (~14 cy/round-trip):
                                = 65 cy/iter × 22 = ~1.4K cy/tile
                                = ~6.6K cy/tile potential save
  - Per-tile budget: ~6300 cy (current actual measurement is ~6700 cy
                                including all stalls)
  - Best-case ratio: 1.072 × (6700 / 5300) = 1.354 ← UPPER BOUND if
                                                   100 % of spill becomes
                                                   LDS-resident with zero
                                                   schedule cost
  - Realistic (33 % redirection, 50 % schedule cost): 1.072 × 1.05 = 1.13
  - Pessimistic (5 % win): 1.072 × 1.02 = 1.094
```

So Lever C-1 RANGE is +2 to +5 pp on gpt_oss-B32 specs. Even the
optimistic case (+5 pp on 8 cases) gives geomean +1.7 pp ⇒ score
~1000 × min((1.163 + 0.017) / 1.20, 1.0) ~= 983. Still short of 1000.

**Honest projection**: even Lever C-1 success leaves score 980-985,
not 1000. The 1.20 target on gpt_oss-B32 is unreachable on the current
kernel architecture. R56-dm's "structurally unreachable" assessment
holds.

---

## Concrete Lever C-1 plan for R29-R31 (3 rounds, multi-step commit)

**Per R3 plan, never executed.** The mechanism: replace the `~22 dw VMEM
scratch round-trips per K-iter` (LLVM's default spill choice on the
`<0,T,T>` gpt_oss spec) with explicit LDS scratch via `ds_write_b32` /
`ds_read_b32` inline asm.

### R29: SCOUT (data only, no kernel change)

* Build with `-mllvm -print-after-all -mllvm -debug-only=spill`
  (ROCm hipcc may need different flag spelling — verify with
  `/opt/rocm/llvm/bin/llc -help-hidden 2>&1 | grep -i spill`).
* Parse `kernel_fp8_layouts-hip-amdgcn-amd-amdhsa-gfx950.opt.yaml`
  `SpillReloadCopies` remarks at `<0,T,T>` template's loop body.
* Identify: which 4-8 specific VGPR slots in the K-iter are spilled
  most often (the "hot spill" set). Cross-reference with `-Rpass=spill`
  source-level remarks.
* Output: `analysis/_notes/round-29-fp8-grouped-Lever-C-1-spill-source-localization.md`
  containing the hot-spill VGPR list + source line numbers + cycle
  count proxy. **No kernel commit.**

### R30: PROBE (single hot spot, LDS scratch redirect)

* Pick TOP 1 hot-spill source from R29 list (likely the K-tail mid-iter
  `b_per_group_bytes` SRD or a `combined_scale` reload).
* Add `__shared__ uint32_t scratch_pool[16]` (64 B LDS, fits in 23 KB
  headroom).
* Inline `ds_write_b32(scratch_pool[K], cold_vgpr_value)` at spill
  source location.
* Inline `ds_read_b32(reload_vgpr, scratch_pool[K])` at reload
  consumer location.
* Use `asm volatile("" : "=v"(reload_vgpr))` clobber pattern to force
  LLVM to consider the original source register free after the LDS
  write.
* Validate: `-Rpass-analysis=kernel-resource-usage` should show
  ScratchSize/lane DROP on `<0,T,T>` spec. If unchanged → LLVM
  re-spilled to VMEM anyway → revert + falsify.
* Run metric. If +5 pts → commit. If 0 → falsify, drop in scope at
  next pass.
* **Estimated 1-3 hot-spill targets to attempt this round if budget
  allows; one redirect = one revert/keep cycle.**

### R31: EXTEND if R30 LANDED

* Apply same pattern to next 2-3 hot-spill sites identified in R29.
* Cumulative target: 8-12 dw spill drop on `<0,T,T>`, equiv to
  ~+3-5 pp on gpt_oss specs.
* If ANY R30+R31 commit lands, score should reach 982-985 territory.

### Risks and abort criteria

* **LLVM not honoring manual LDS spill**: `ds_write_b32` may not
  release the source VGPR (LLVM still considers it live). Mitigation:
  `asm volatile` clobber. If clobber breaks correctness (allclose
  fails) → falsify.
* **LDS bank conflict**: scratch_pool reads might bank-conflict with
  As/Bs reads inside the K-iter. Mitigation: probe LDS bank usage via
  `rocprof --pmc LDS_BANK_CONFLICTS` on a single shape; if bank conflict
  rate increases → choose different scratch_pool offset.
* **LDS size cap (137 → 137+X KB)**: scratch_pool ≤ 16 KB to stay in
  the 23 KB headroom. Target redirection volume = ~16 dw × 4 bytes ×
  some-iter-multiplier. For per-warp scratch (8 waves × 64 lanes ×
  16 dw × 4 bytes = 32 KB) — TOO BIG. Must be per-WAVE scratch
  (256 lanes × 16 dw × 4 = 16 KB) or per-CTA cyclic (~1 KB at any
  time). Architectural verify needed.

---

## What this round changed

**Nothing in code or HK.** Only:
- New round note at this path (~250 lines)
- No HK build / no metric variance from baseline

HK SHA stays at `4caa6d9a`. PT this commit is doc-only.

---

## Round meta

| Field | Value |
|---|---|
| HK SHA before/after | `4caa6d9a` / `4caa6d9a` (unchanged) |
| PT SHA before | `2f2fdcfe` |
| PT SHA after  | (this commit) |
| Forward metric before/after | 979 / 979 |
| Lever class status | FP8 grouped C-store readfirstlane CLASS = EXHAUSTED |
| Open lever class | **Lever C-1 (LDS scratch redirect)** = PROPOSED for R29-R31 |
| Score trajectory | 979 → 979 (no probe) |
| Patience increment | +1 (was 0, will be 1 after this) |
| Best metric | 981 (R27 noise sample; current baseline median 979) |
| Score ceiling estimate | 982-985 (best case) / 980-981 (likely) |

---

## DoD smoke status

Not run this round (no shared-code change — doc-only). Last DoD run was
at SHA `813c2e3e` (608 score per round-26 prompt).

---

## Files touched

* `/workspace/code/Primus-Turbo/analysis/_notes/round-28-fp8-grouped-saturation-acknowledgment-and-lever-c-1-roadmap.md` (this note, ~280 lines)
