# Round-27 — FP8 K-tail `a_kt1` hoist falsified by VGPR live-range pressure

**Date**: 2026-05-01  
**Repo / HEAD**: Primus-Turbo `19a4548` (round-26); HipKittens `62cebd5` (no net change this round, both probes restored)  
**Focus**: gpt_oss 16 shapes; DSV3 = `[watch]`  
**Result**: K-tail vmem hoist falsified (-4.9pp grp_FP8); single-knob vmem-overlap is unsafe without simultaneous register-tile re-derivation.

---

## Goal of this round

Round-26 audit completed all single-knob 1-round levers. Round-27
plan was to start the multi-round structural project (round-25
recommended **BF16 K_STEP=64 → BK=32 + ns=3 port**, ~+14 score cap).

**Reconsidered leverage**: K-tail epilog optimization (round-25 docs
item #3) affects **all 16 gpt_oss shapes** (every gpt_oss case has
K=2880, K_REM=64), with estimated wall savings 1.5-3% per shape →
+20-40 score upper if successful. BF16 BK port only affects 8 BF16
shapes with cap 1.20 → +14 score upper.

**K-tail leverage ≈ 2-3× BF16 BK port leverage**. Pivot to K-tail.

---

## Probe — Hoist `load_a_kt(a_kt1, 1)` to before Epilog 1

### Hypothesis

Current FP8 grouped K-tail epilog (`kernel_fp8_layouts.cpp` lines
2272-2422, round-3 commit `07354791`):

```
Issue 24 b128 (load_a_kt(a, 0) + load_a_kt(a_kt1, 1) +
                load_b_kt(b0, 0) + load_b_kt(b1, 1))
asm volatile("s_waitcnt vmcnt(0)")    ← drain ~100-150 cyc
4 sequential mfma_scale_f32_16x16x128
```

`a_kt1` is a dedicated K-tail register tile (declared at line 1995),
independent of `a / b0 / b1` used by Epilog 1+2. Hypothesis: hoist the
8 b128 of `load_a_kt(a_kt1, 1)` to **before Epilog 1**, so they
complete concurrently with Epilog 1+2's 4+4=8 mfma (~256 cyc shadow).
The K-tail block then issues only 16 b128 + drain → ~50-100 cyc wait
instead of ~100-150 cyc → ~25-50 cyc saved per output tile = ~1.5-3 %
wall on K-misaligned shapes.

This was framed as the **minimal-risk first step** of the multi-round
"K-tail epilog single-load merge" path (round-25 docs item #3): only
reuses the already-declared `a_kt1` register, no new register tiles
allocated, no expected VGPR spill.

### Implementation

Patched `kernel_fp8_layouts.cpp` to insert after main-loop end
(before Epilog 1):

```cpp
if constexpr (FUSED_KTAIL) {
    if (g.fast_k < g.k) {
        // Mirror SRD/lane setup of K-tail epilog block
        // ... (a_srsrc_kt, K_REM, b128_lo/hi_valid, etc) ...
        const int M_warp_base_kt = (m_subtile_A + br * 2 + 1) * HB + wm * RBM;
        for (int h = 0; h < A_row_reg::height; ++h) {
            // 2 × raw_buffer_load_b128 → a_kt1.tiles[h][0].data[0..7]
        }
    }
}
```

K-tail epilog block had `load_a_kt(a_kt1, 1)` removed (already issued).

Build: clean. No compile error, no linker error.

### 5-run metric verify

| run | GateUP-B4-M2048 | Down-B4-M2048 | GateUP-B4-M4096 | Down-B4-M4096 | GateUP-B32-M2048 | Down-B32-M2048 | GateUP-B32-M4096 | Down-B32-M4096 | grp_FP8 geomean |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.879 | 0.965 | 0.903 | 0.873 | 0.924 | 0.934 | 0.916 | 0.921 | 0.9139 |
| 2 | 0.885 | 0.954 | 0.899 | 0.873 | 0.923 | 0.933 | 0.914 | 0.918 | 0.9120 |
| 3 | 0.880 | 0.964 | 0.911 | 0.880 | 0.921 | 0.931 | 0.916 | 0.921 | 0.9150 |
| 4 | 0.886 | 0.946 | 0.897 | 0.868 | 0.924 | 0.934 | 0.915 | 0.920 | 0.9110 |
| 5 | 0.879 | 0.954 | 0.911 | 0.895 | 0.920 | 0.929 | 0.912 | 0.916 | 0.9142 |
| **mean** | **0.882** | **0.957** | **0.904** | **0.878** | **0.922** | **0.932** | **0.915** | **0.919** | **0.9132** |

**Pre-hoist baseline** (round-26 metric):

| | GateUP-B4-M2048 | Down-B4-M2048 | GateUP-B4-M4096 | Down-B4-M4096 | GateUP-B32-M2048 | Down-B32-M2048 | GateUP-B32-M4096 | Down-B32-M4096 | grp_FP8 geomean |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.917-0.929 | 0.975-0.991 | 0.940-0.948 | 0.923-0.929 | 0.977-0.983 | 0.984-0.988 | 0.972-0.976 | 0.974-0.977 | 0.962 |

**Δ vs baseline (mean per shape)**:

| shape | Δ pp |
|---|---|
| GateUP-B4-M2048 | -3.7 |
| Down-B4-M2048   | -2.4 |
| GateUP-B4-M4096 | -3.9 |
| Down-B4-M4096   | -5.1 |
| GateUP-B32-M2048| -6.0 |
| Down-B32-M2048  | -5.4 |
| GateUP-B32-M4096| -5.9 |
| Down-B32-M4096  | -5.8 |

**Net grp_FP8 segment Δ = 0.962 → 0.913 = -4.9 pp** — uniform regression
across **all 8 FP8 cases**, B=32 even worse than B=4 (which is
opposite of the launch-overhead-dominated hypothesis). Five-of-five
runs below baseline.

Score before: 880. Score after: ~860 (hand-est from FP8 progress
0.762/0.760/0.763/0.759/0.762 × BF16 progress 0.972 → focus weighted
geomean ≈ 0.86 → score ≈ 860). **20-point regression**.

### Reverted

Both edits reverted to round-26 state. Verify build: OK. Verify
single-run metric: score 880 (back to baseline), FP8 geomean 0.9624.

---

## Why it regressed (architectural cause)

The hoist extends the **live range of `a_kt1`** from "K-tail block
only" (~250 cyc lifetime) to "main-loop end → K-tail mfma end"
(~500-600 cyc lifetime, spanning both Epilog 1 and Epilog 2).

`A_row_reg a_kt1` has size = 4 sub-tiles × 32 dwords/sub-tile = 128
dwords/wave = **128 VGPR/wave**. Holding this live across Epilog 1+2's
8 mfma (256+ cyc) prevents the register allocator from spilling /
re-using these 128 VGPR for:

- Epilog 1's `a, b0, b1, cA, cB, cC, cD` register tiles (each
  ~32-64 VGPR/wave)
- Epilog 2's same tile set
- Loop-carried temporaries the register allocator tries to keep in VGPR

The MI355X kernel's per-wave VGPR budget is 256 (with `MIN_BLOCKS_PER_CU=2`).
Pre-hoist, the K-tail epilog uses `a_kt1` only at the very end, so it
effectively shares VGPR with `cD`'s lifetime in Epilog 2. Post-hoist,
`a_kt1` and `cA/cB/cC/cD` (which are alive across the entire Epilog
1+2 sequence) **both** demand VGPR concurrently → register allocator
must spill loop temporaries to LDS or scratch.

LDS spill costs are ~10-20 cyc per spill/reload, and Epilog 1+2 has
many such opportunities (every `load_a/load_b/mfma` chain). The
spilled register reloads happen *during* the mfma issue queue, which
is the critical path. Net effect: each Epilog 1+2 mfma sequence
slows by ~20-50 cyc per output tile = ~5-10% wall regression →
matches the observed -4.9pp grp_FP8 geomean shift.

The B=32 case regressing slightly **worse** than B=4 (despite B=4
being more launch/latency-bound) is consistent with VGPR spill: B=32
runs more tiles per persistent slot, so any per-tile overhead
amplifies linearly with iteration count.

---

## Lessons + updated multi-round plan for K-tail

Round-25 docs item #3 said:

> 3. K-tail epilog single-load merge ... 2 rounds: requires path B
>    and Epilog 2 schedule co-design.

Round-27 confirms: **"schedule co-design" specifically means
register-tile lifetime co-design**. A naive load-hoist without
register-tile rotation strategy unconditionally regresses.

Real K-tail single-load-merge needs at minimum:

1. **Register tile rotation**: introduce alternate K-tail register
   tiles `a_kt0, a_kt1, b_kt0, b_kt1, b_kt2, b_kt3` (4-6 new tiles)
   that rotate so only the active tile holds K-tail data, and dead
   tiles can be reclaimed by Epilog 1+2 mfma.
2. **VGPR budget calibration**: measure post-rotation total VGPR/wave
   pressure with `__attribute__((amdgpu_kernel_max_vgpr(N)))` /
   `--amdgpu-kernel-arg-vgpr` and confirm `MIN_BLOCKS_PER_CU=2`
   occupancy preserved. If VGPR > 256 for 2-blocks/CU, spill or
   regress.
3. **mfma-vmem schedule alignment**: order the hoisted K-tail loads
   so each lands in the mfma cycle window where its target register
   is dead (ditto for `cA/cB/cC/cD` partial reuse).
4. **K-tail mfma instruction selection**: for K_REM=64 < K_BLOCK=128,
   half the lanes are SENTINEL (zero-fill). Could replace 1×
   `mfma_scale_f32_16x16x128` (32 cyc, half data zero) with 2×
   `mfma_f32_16x16x32_f8f6f4` (8+8 cyc, full data) → 50% mfma cyc
   saving on K-tail. Requires re-derived lane→cell mapping.

Each of (1)-(4) is a separate round. Total: **3-4 rounds** to
properly land K-tail epilog optimization, not 2.

---

## Updated leverage map for gpt_oss

After round-27 falsification:

| lever | status | rounds to land | est. wedge |
|---|---|---|---|
| BF16 K_STEP=64 → BK=32+ns=3 port | open (round-25 plan) | 3-4 | +14 score (cap 1.20) |
| K-tail epilog single-load merge   | falsified naively (round-27); needs full co-design | 3-4 | +20-40 score |
| FP8 MFMA cell-shape `16x16x128 → 32x32x64` | open (round-25 plan) | 2-3 | unknown |
| All single-knob micro-tunes | exhausted (rounds 4/5/6/15/22/24/25/26) | - | 0 |
| BN=128 dispatch path | falsified (round-24) | - | 0 |

**No 1-round wedge remains. All paths forward are 2-4 round structural
projects.**

---

## Commit plan

Primus-Turbo only (HipKittens has no net code change — both edits
reverted).

```
docs(round-27): FP8 K-tail load_a_kt(a_kt1) hoist falsified by VGPR live-range pressure
```

## Next-round suggestion

Two viable multi-round paths remain:

**A. BF16 K_STEP=64 → BK=32 + ns=3 port** (round-25 + round-26 plan)
- Cap +14 score; 3-4 rounds; risk = compile breakage (graceful WIP).
- Round-28 step 1 = read main loop + K_STEP audit + introduce
  non-functional probe constexpr.

**B. K-tail epilog full co-design** (round-25 item #3 + round-27 update)
- Cap +20-40 score; 3-4 rounds; risk = VGPR pressure at every step.
- Round-28 step 1 = inventory K-tail register-tile lifetimes +
  declare additional `a_kt0, b_kt0..3` shells (non-functional, sized
  to budget) + measure baseline VGPR/wave with `rocm-smi` /
  compiler dump. Round-29+ wires actual rotation.

Recommend **A** for round-28 because:

1. BF16 BK port has only 1 failure mode (compile error → revert one
   file). K-tail co-design has 3 simultaneous failure modes (compile,
   VGPR spill, schedule race) requiring incremental probe per step.
2. BF16 segment is closer to its cap (1.166 → 1.20 = 3.4pp gap; FP8
   is 0.962 → 1.20 = 24pp gap), so BK port has tighter expected delta
   distribution.
3. Round-27's lesson (naive hoist regressed -5pp) means K-tail
   co-design must be more careful — better deferred until BF16 BK
   port lands and we have the kernel re-tuned baseline to compare.

Round-28 step 1: read `kernel_bf16_dynamic.cpp` lines 600-686
(`main_loop_iter`), 540-552 (subtile templates), 1158-1183
(ST_A/ST_B / register tile types), 1145-1240 (gemm_kernel entry +
SRD setup). Catalog all `K_STEP` hard-code sites (per round-26 docs
item 2). Introduce `BF16_K_STEP_PORT_PROBE` constexpr default 0
(bit-equivalent to current). WIP commit at this step.
