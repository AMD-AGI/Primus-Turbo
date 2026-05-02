# Round 4 — FP8 grouped: Lever C-X (SENTINEL store) NEUTRAL → revert

**Status**: KERNEL CHANGE TRIED + REVERTED — Lever C-X SENTINEL voffset
masking in `store_c_tile_n_masked` neither reduced spill on the
ACTIVE template (`<0,T,T>` gpt_oss) nor moved the metric. Reverted
per task body's "score 持平或跌 → revert + falsify" rule.
**Auto-optimize round**: 4 / 100
**Date**: 2026-05-02
**HK SHA at round start**: `ecbead9a`
**HK SHA at round end**: `ecbead9a` (unchanged after revert)
**PT SHA at round start**: `b54107b`
**Round time**: ~30 min (2 metric runs + 1 build + 1 metric + 1 revert + 1 metric)
**Score before**: 958 (R4 baseline)
**Score during refactor**: 956 (Δ=-2)
**Score after revert**: 961 (back in noise band)

---

## Hypothesis (from R3 plan)

Replace `store_c_tile_n_masked`'s per-lane divergent skip:
```cpp
const int col = j * src.base_tile_cols + col_offset;
if (n0 + col >= n_limit) continue;
```
with SENTINEL voffset masking, mirroring the proven K-tail pattern
at line 2637-2638:
```cpp
const int col = j * src.base_tile_cols + col_offset;
const bool col_oob = (n0 + col) >= n_limit;
constexpr uint32_t SENTINEL = 0xFFFF0000u;
...
const uint32_t off0 = col_oob ? SENTINEL : off0_real;
const uint32_t off1 = col_oob ? SENTINEL : off1_real;
llvm_amdgcn_raw_buffer_store_b16(v0_bits, srsrc, off0, 0, 0);
llvm_amdgcn_raw_buffer_store_b16(v1_bits, srsrc, off1, 0, 0);
```

R3 spill-localization predicted +44 NumVR / +24 cost reduction on
`<0,T,T>` template's secondary spill cluster (R3 yaml parse showed
this template's secondary cluster = 117 NumVR vs `<0,F,T>` floor = 29
NumVR, with the +88 NumVR delta attributed to the masked-store
helper's 4× inlined invocations in epilog).

---

## Resource-usage delta (LLVM remarks)

| Template | Spill before | Spill after | ScratchSize before | ScratchSize after | Δ Spill |
|---|--:|--:|--:|--:|--:|
| `<0,F,F>` (DEAD) | 39 | 39 | 160 | 160 | 0 |
| `<0,T,F>` (DEAD) | 43 | **41** | 176 | **168** | **-2** |
| `<0,F,T>` (DSV3+Qwen, 16/24) | 32 | 32 | 132 | 132 | 0 (helper not called) |
| `<0,T,T>` (gpt_oss, 8/24) ← TARGET | **39** | **39** | **160** | **160** | **0** |
| `grouped_var_k_kernel_fp8` | 52 | 52 | 212 | 212 | 0 |

**Refactor only affected the DEAD `<0,T,F>` template** (dropped 43→41
spill, -8 B scratch). The active gpt_oss target template (`<0,T,T>`)
**did NOT change** — bit-identical 256 VGPR / 39 spill / 160 B
ScratchSize / 137 KB LDS.

### Why the refactor didn't reduce spill on the active template

The N_MASKED_STORE=true template carries the helper's body code
regardless of the inner branch path. The 4× inlined invocations
in epilog (lines 2785-2791) compete for VGPR slots even when
the n_aligned fast-path (`(bc + 1) * BLOCK_SIZE <= g.n` at line
2774) is taken at runtime — the helper code is in scope and LLVM
extends its register requirements across the entire epilog scope.

The SENTINEL refactor replaces a **divergent control-flow** primitive
(`if (col_oob) continue;`) with a **uniform v_cndmask** primitive
(`col_oob ? SENTINEL : off_real`). The two consume similar register
slots (EXEC mask machinery vs col_oob predicate + 2× v_cndmask result
slots). LLVM keeps spill at the same level on `<0,T,T>` because the
COLD path (FUSED_KTAIL=true main code) still dominates the live state
budget independent of how the masked store handles OOB columns.

The fact that `<0,T,F>` (FUSED_KTAIL=false, less hot path) DID drop
2 dw confirms the SENTINEL refactor IS doing something — just not
enough to register on the FUSED_KTAIL=true active template where the
K-tail block's live state dominates.

---

## Metric delta

```
R4 baseline:                      score=958  grp_FP8=1.1132
After SENTINEL refactor:          score=956  grp_FP8=1.1109   (Δ=-2 / -0.23 pp)
After revert (back to ecbead9a):  score=961  grp_FP8=1.1214   (in noise)
```

### Per-shape gpt_oss subset (8/24 case, the targeted subset)

| Case | before | after | Δ |
|---|--:|--:|--:|
| GateUP-B4-M2048 | 1.075 | 1.079 | +0.4 pp |
| Down-B4-M2048   | 1.139 | 1.154 | +1.5 pp |
| GateUP-B4-M4096 | 1.066 | 1.060 | -0.6 pp |
| Down-B4-M4096   | 1.082 | 1.081 | -0.1 pp |
| GateUP-B32-M2048| 1.048 | 1.053 | +0.5 pp |
| Down-B32-M2048  | 1.052 | 1.062 | +1.0 pp |
| GateUP-B32-M4096| 1.023 | 1.022 | -0.1 pp |
| Down-B32-M4096  | 1.059 | 1.041 | **-1.8 pp** |

Net: +0.1 pp average (4 up, 4 down, with -1.8 pp tail on B32-M4096).
Within ±2 pp shape-noise band typical for these kernels — the
refactor is **statistically NEUTRAL** on gpt_oss.

The smaller-batch (B=4) cases trend slightly positive (+0.4 to +1.5);
the larger-batch (B=32) cases trend slightly negative (-1.8 worst).
This is consistent with: SENTINEL adds a fixed v_cndmask cost per
inner iter (~2 cyc), divergent-branch save scales with EXEC management
overhead per per-warp iteration; with more output tiles per CU on
B=32, the EXEC overhead amortizes better, narrowing the win
margin to negative.

---

## Falsification: Lever C-X is NEUTRAL

R3's expectation of "+0.5 pp geomean / +5 score" was based on the
+88 NumVR secondary cluster delta predicting a 5-7 dw spill reduction.
That prediction relied on LLVM's register allocator using the EXEC
mask saving to free VGPR slots. Empirically, LLVM did NOT free those
slots (spill stayed at 39 dw on `<0,T,T>`) because the v_cndmask
predicate + SENTINEL constants take up slots equivalent to the EXEC
mask machinery.

**Updated lever roadmap**:

| Lever | Status |
|---|---|
| **A** Async global→LDS copy | FALSIFIED R2 (already shipped) |
| **B** Triple LDS slab | FALSIFIED R2 (LDS 137/160 KB used) |
| **C-1** LDS hand-spill | DEFERRED — R3 said gated on C-X first; C-X falsified, but C-1's premise (single targetable expression) was already weak |
| **C-2** K-tail capture refactor | FALSIFIED R3 (captures already in if-branch) |
| **C-3** Spill source localization | DONE R3 (data only) |
| **C-X** N_MASKED helper SENTINEL refactor | **FALSIFIED R4** — neutral on active template, slight regression on Down-B32-M4096 |
| **D** mfma_323264 cell-shape | OPEN — mandatory R64-dm microbench gate not yet run |
| **E** ASM software pipelining | LAST RESORT |
| **F** Qwen-Down K=1536 short-K | DEFERRED — only +0.8 pp geomean even if successful |

---

## R5 plan: **Lever D microbench gate** (R64-dm leftover)

After 4 rounds of falsification (R1 lever-A/B-misjudged, R2 ditto,
R3 C-2 prerequisite-falsified, R4 C-X neutral), all "incremental"
levers are exhausted. Only architectural Lever D and Lever E remain.

**R5 = R64-dm microbench gate** (the explicit pre-requisite for
committing to Lever D's 4-6 round full port):

1. Write a standalone `.cu` microbench:
   - Allocates M=64, N=32, K=128 device tensors (fp8 input, fp32 acc)
   - Path (a): runs N=10000 iter of (8 × `mfma_f32_16x16x128_f8f6f4`)
     into 4 × rt_fl<16,16> accumulators
   - Path (b): runs N=10000 iter of (2 × `mfma_f32_32x32x64_f8f6f4`)
     into 2 × rt_fl<32,32> accumulators (architecturally equivalent)
   - Wall-clock per-iter via `__builtin_amdgcn_s_memtime`
2. Build via `hipcc --offload-arch=gfx950 -O3`
3. Run in a free GPU slot, capture per-iter wall (median over 10k iter)
4. Decision threshold: if (b) is **≥ 3 pp faster per-iter** than (a),
   commit to R6+ Lever D full port (4-6 rounds). Else **abandon Lever
   D entirely** and accept 956-962 / FP8 geomean ~1.12 as the final
   plateau.

**If Lever D fails**: remaining options are
(i) Lever E (ASM software pipeline, R20+ last resort, very high risk)
or (ii) accept plateau and direct remaining rounds at:
   - cleanup / documentation
   - Lever F (Qwen-Down K=1536, +0.8 pp geomean ceiling)
   - whatever incidental wins emerge from BF16 [watch] segment review
     (forbidden to MODIFY but reviewing for cross-pollination is OK)

---

## What this round changed in code

**Nothing committed.** Tried + reverted SENTINEL refactor in
`kernel_fp8_layouts.cpp:909-972`. HK working tree is clean
(`git status` shows only `.nfs0000000dc0f357f5000027df` orphan from
NFS-deleted .opt.yaml, harmless).

**Primus-Turbo repo**: this `.md` is the only diff.
**HipKittens repo**: no change (HK SHA stays at `ecbead9a`).

---

## Hard-constraint compliance check

- [x] No metric / benchmark / config edits
- [x] No dispatcher / can_handle changes
- [x] No quantize fuse, no host-side .item() / .tolist()
- [x] No per-model branches in dispatcher
- [x] HIPKITTEN remains `BackendEntry(..., autotune=False)`
- [x] One focused PT commit (the falsification note)
- [x] Zero HK commits this round (tried, reverted, no commit)
- [x] No BF16 grouped touch
- [x] Correctness 0/48 fail throughout

---

## DoD smoke status

Not run this round (kernel change tried + reverted; .so is bit-identical
to R3 end-state). Last DoD run was at SHA `94fc3121` (R64). Next R5
checkpoint per task body cadence will run it.
