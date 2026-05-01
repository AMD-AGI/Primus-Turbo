# Round 34-dm (R7 DM probe) — FP8 grouped: ISA-guided dispatch rewrite — route DSV3 (K_REM=0) shapes through the FUSED_KTAIL=true template spec for their free-lunch register allocation **(+17 pts, first perf win since R4 baseline)**

Status: **SHIPPED** — score 918 → 935 (+17), grp_FP8 geomean 1.0247 → 1.0637 (+3.9pp), all 32 shapes correctness PASS. HipKittens dispatcher change + Primus-Turbo doc.

## The insight

R33-dm and R3 both failed the same way — structural changes to epilog 1 or main loop shifted LLVM's register allocator in ways that hurt DSV3 specifically. The pattern pointed at a DSV3-specific register-pressure problem, but the prior work lacked the ISA visibility to pinpoint WHERE the cost actually was.

This round started with getting `hipcc --save-temps=obj` working for `kernel_fp8_layouts.cpp`. That gave a 3.4 MB gfx950 GCN assembly dump (vs the broken `llvm-objdump` path from R31-dm). Per-spec scratch-op analysis revealed:

| Template spec | Shape class | MFMA count | Total scratch ops | **Interleaved scratch** (within ±3 lines of v_mfma) |
|---|---|---|---|---|
| `<0, false, false>` | DSV3-GateUP (K-aligned) | 648 | 741 | **62** |
| `<0, true,  false>` | DSV3-Down   (K-aligned, N-masked) | 552 | 661 | 44 |
| `<0, false, true>`  | gpt_oss-GateUP (K=2880, K-tail fuse) | 456 | 559 | 28 |
| `<0, true,  true>`  | gpt_oss-Down   (K=2880, K-tail fuse, N-masked) | 328 | 444 | **22** |

The **interleaved-spill** metric (scratch ops within 3 ASM lines of an `v_mfma`) turned out to be a far better proxy for kernel performance than the `-Rpass-analysis=kernel-resource-usage` VGPRs Spill count. Interleaved spills directly block the MFMA issue pipeline because the scratch_store/load needs the same VGPRs that the mfma is consuming/producing. Main K-loop has ZERO interleaved spills (confirmed by sampling lines 1050-1300 of spec <0,0,0>); epilog 1 is where the pressure peaks — 5+ accumulator cells spill and reload WITHIN the mfma block.

## The free-lunch observation

The FUSED_KTAIL=true template specs have HALF to a THIRD of the interleaved-spill count of their FUSED_KTAIL=false counterparts, even though FUSED_KTAIL=true adds MORE code (the in-kernel K-tail accumulation branch). This is a pure codegen artifact: the extra `A_row_reg a_kt1;` declaration + the in-kernel K-tail mfma loop forces LLVM's register allocator down a different greedy path that ends up placing fewer scratch ops in the hot epilog region. The K-tail runtime branch `if (g.fast_k < g.k)` is dead when K is aligned to K_BLOCK=128, so for DSV3 (K=7168 = 56·128) the extra code is executed ZERO times per tile.

## The fix

One-line dispatcher edit in `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` at `dispatch_grouped_rcr` (~line 4963):

```cpp
// Before:
const bool fuse_ktail_eligible =
    (g.bpc > 0) && (g.ki > 0) &&
    (K_rem_for_fuse == 64) &&
    lds_k_tail_safe_for_fuse;

// After:
const bool fuse_ktail_eligible =
    (g.bpc > 0) && (g.ki > 0) &&
    ((K_rem_for_fuse == 64) || (K_rem_for_fuse == 0)) &&
    lds_k_tail_safe_for_fuse;
```

This routes all K-aligned shapes (DSV3: K=7168) through `grouped_rcr_kernel<0, *, true>`. The `if (g.fast_k < g.k)` branch inside the `if constexpr (FUSED_KTAIL)` block is runtime-dead for these shapes, so numerical output is byte-identical.

Safety conditions remain enforced:
- `g.bpc > 0 && g.ki > 0` — require at least one K-tile
- `lds_k_tail_safe_for_fuse` — if the K-tail branch ever DID run (K_REM==64 for gpt_oss), the `m_per_group` alignment is still enforced

No changes to the kernel template itself; no changes to the runtime semantics; no changes to autograd / Python / dispatch.py / tests / metric.

## Per-shape delta (baseline → probe, both this-round runs)

DSV3 FP8 (all K-aligned, all moved onto the fused spec):
```
grpFP8_DSV3-GateUP-B16-M2048   1.038 → 1.098   +6.0 pp
grpFP8_DSV3-Down-B16-M2048     0.995 → 1.072   +7.7 pp
grpFP8_DSV3-GateUP-B16-M4096   1.065 → 1.103   +3.8 pp
grpFP8_DSV3-Down-B16-M4096     0.980 → 1.120   +14.0 pp   ← biggest jump
grpFP8_DSV3-GateUP-B32-M2048   1.052 → 1.111   +5.9 pp
grpFP8_DSV3-Down-B32-M2048     1.026 → 1.121   +9.5 pp
grpFP8_DSV3-GateUP-B32-M4096   1.069 → 1.139   +7.0 pp
grpFP8_DSV3-Down-B32-M4096     1.011 → 1.139   +12.8 pp
```

gpt_oss FP8 (already on fused spec, no dispatch change): within ±0.02x noise, no measurable effect.

BF16 grouped: unchanged (geomean 1.183-1.184 pre/post, not a code path touched by this commit).

Two metric runs to confirm: 934, 935. `best=918` before, so a real +17 floor improvement.

## Tool also committed: `analysis/tools/dump_fp8_grouped_isa.sh`

Reproducible wrapper that emits `spec_{tag}.s` per template spec + 4-row interleaved-spill summary. Works standalone with `THUNDERKITTENS_ROOT=/workspace/code/HipKittens`. This is the tool that enabled the R34-dm insight and will support future ISA-guided probes in rounds 35+.

## What's next (ISA-guided probe ideas)

Now that interleaved-spill count is a validated hot-path proxy, the remaining FP8 ratio gap breaks down as:

1. **gpt_oss-GateUP-B32-M{2048,4096}** (ratios 0.987 / 0.969 after this commit): spec `<0, false, true>` with 28 interleaved spills is ALREADY the lowest-spill spec. Additional reduction requires either:
   - Lowering its interleaved spill count further via structural change to epilog 1 (needs co-design with LLVM's greedy allocator; no simple template-switch trick available).
   - Lever D (32x32x64 MFMA cell shape) — halves the accumulator cell count (from 8 → 4 for `rt_fl<64,32>`), which directly reduces the accumulator-VGPR footprint. R15-dm's naive migration failed; with ISA-guided validation available now, a piece-wise migration starting from epilog 1 is tractable.

2. **DSV3-Down-B{16,32}-M* ratios 1.072-1.139**: already excellent, but still have 44 interleaved spills. If dropped to 22 (matching spec 3), another +2-3 pp possible per shape → could push geomean toward 1.09-1.10.

3. **Regression protection**: BF16 grouped kernel is in a separate file (`kernel_bf16_dynamic.cpp`) and separate dispatcher; this commit does not touch it. grp_BF16 confirmed stable.

## Concrete commit trail

- HipKittens: **1 commit** in `/workspace/code/HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` — the 3-line `fuse_ktail_eligible` condition extension. No template changes, no kernel body changes.
- Primus-Turbo: **1 commit** adding this note + `analysis/tools/dump_fp8_grouped_isa.sh`.
- Both commits must land this round.

## Verdict

First real perf win since the R4 baseline. Score 918 → 935. 17 pts closer to the 1000 target (needing grp_FP8 geomean 1.20, currently at 1.0637 post-fix, needing +13.6 more pp). Path forward: continue ISA-guided probes on remaining high-interleave regions + Lever D piecewise migration.
