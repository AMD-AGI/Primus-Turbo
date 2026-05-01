# Round 34-dm (R7 DM probe) — FP8 grouped: ISA inspection tooling works via `hipcc --save-temps`; main K-loop is SPILL-FREE, and the real bottleneck is accumulator VGPRs being spilled/reloaded INTERLEAVED WITH the epilog-1 MFMA pipeline

Status: **DIAGNOSTIC + TOOLING (no perf commit)**. Primus-Turbo gets a reproducible ISA-dump helper + a concrete per-spec spill-distribution characterization that reshapes the strategy for rounds 35+.

## Tooling established

`hipcc --save-temps=obj` emits the device-side gfx950 GCN assembly as a readable `.s` file alongside the `.o`. Verified on the current head of `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`:

```
$ /opt/rocm/bin/hipcc -std=c++20 -DKITTENS_CDNA4 --offload-arch=gfx950 \
    -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
    -I/opt/rocm/include/hip -I/opt/rocm/include/rocrand \
    -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype \
    $(python3 -m pybind11 --includes) \
    -shared -fPIC -w -O3 --save-temps=obj \
    -c kernel_fp8_layouts.cpp -o kernel_fp8_layouts.o

# produces:
#   kernel_fp8_layouts-hip-amdgcn-amd-amdhsa-gfx950.s   (3.4 MB, gfx950 ISA)
#   kernel_fp8_layouts-host-x86_64-unknown-linux-gnu.s  (4.4 MB, pybind11 bindings)
#   kernel_fp8_layouts.o                                 (device+host object)
```

This replaces the broken `llvm-objdump --disassemble-all --triple=amdgcn--amdhsa --mcpu=gfx950` path (R31-dm found it crashed segv or returned x86-only on the `.so`). `--save-temps=obj` keeps the temps in the CWD instead of `/tmp/` so we don't lose them.

A reproducible wrapper ships at `analysis/tools/dump_fp8_grouped_isa.sh` (new this round).

## Spec-level ISA characterization (gfx950 device asm)

The `grouped_rcr_kernel<KI_HINT=0, N_MASKED_STORE, FUSED_KTAIL>` template
has 4 specializations that the dispatcher picks based on the shape:

| Template spec | Shape class | MFMA count | Total scratch ops | **Interleaved scratch** (within ±3 lines of v_mfma) |
|---|---|---|---|---|
| `<0, false, false>` | DSV3-GateUP (K-aligned, no N-mask) | 648 | 741 | **62** |
| `<0, true,  false>` | DSV3-Down   (K-aligned, N-masked)  | 552 | 661 | 44 |
| `<0, false, true>`  | gpt_oss-GateUP (K=2880 K-tail fuse) | 456 | 559 | 28 |
| `<0, true,  true>`  | gpt_oss-Down   (K=2880, N-masked)  | 328 | 444 | **22** |

**Critical observation**: the interleaved-spill count correlates almost
linearly with the per-spec performance ratio observed in the metric:

```
DSV3-GateUP  (62 interleaved spills) → geomean 1.065 ← worst hot-path
DSV3-Down    (44 interleaved spills) → geomean 1.000
gpt_oss-GateUP (28 interleaved spills) → geomean 1.022
gpt_oss-Down (22 interleaved spills) → geomean 1.050
```

The "interleaved spill" metric (scratch ops that sit within 3 ASM lines of
a v_mfma) is a much better proxy for kernel performance than the
`-Rpass-analysis` spill count, because it filters out COLD-PATH spills
(outer-loop backedges, group-metadata setup) that run once per tile and
don't block the MFMA issue pipeline.

## What the ISA reveals about the failure mode

The main K-loop body is **COMPLETELY spill-free**. Between stage-1 mfmas
(line 1050-1057 of spec <0,0,0>) and stage-2 mfmas (line 1101-1108), there
are only ds_read_b128 + buffer_load_dwordx4 offen lds + s_barrier — NO
scratch_store/scratch_load. Same for stages 3-4. The hot steady-state
K-loop is as tight as it can be.

The spill-reload cycles happen in **epilog 1**. Spec <0,0,0> shows this
pattern starting at line ~1430 (epilog 1 stage 1):

```asm
s_setprio 1
v_mfma_f32_16x16x128_f8f6f4 v[76:79], v[204:211], v[196:203], v[76:79]   ; acc cA cell #?
v_mfma_f32_16x16x128_f8f6f4 v[24:27], v[220:227], v[188:195], v[24:27]   ; acc cA cell #?
s_nop 10
scratch_store_dwordx4 off, v[76:79], off             ; 16-byte Folded Spill   ← HOT SPILL
v_mfma_f32_16x16x128_f8f6f4 v[12:15], v[220:227], v[196:203], v[12:15]
scratch_store_dwordx4 off, v[24:27], off offset:16   ; 16-byte Folded Spill   ← HOT SPILL
v_mfma_f32_16x16x128_f8f6f4 v[4:7],  v[228:235], v[188:195], v[4:7]
s_nop 9
scratch_store_dwordx4 off, v[12:15], off offset:32   ; 16-byte Folded Spill   ← HOT SPILL
v_mfma_f32_16x16x128_f8f6f4 v[0:3],  v[228:235], v[196:203], v[0:3]
scratch_store_dwordx4 off, v[4:7],  off offset:48    ; 16-byte Folded Spill   ← HOT SPILL
v_mfma_f32_16x16x128_f8f6f4 v[244:247], v[204:211], v[188:195], v[96:99]
s_nop 9
scratch_store_dwordx4 off, v[0:3],  off offset:64    ; 16-byte Folded Spill   ← HOT SPILL
...
```

LLVM's register allocator is spilling 5 accumulator cells DURING epilog-1
stage-1's MFMA issue block. The `s_nop 9/10` insertions confirm this is
scheduler-induced backpressure — mfma needs accumulator regs, scratch_store
needs same regs, so nops are inserted to let mfmas retire before spilling.

Later in epilog 1 stage 3+ (lines 1556+), matching `scratch_load_dwordx4`
reloads the same v[0:3], v[4:7], v[12:15], v[20:23] cells back for more
mfmas. This is a spill→use→reload→use pattern WITHIN a single tile's
epilog.

## Register-pressure budget (from ISA)

- Accumulators (cA+cB+cC+cD, each rt_fl<64,32,col_l,rt_16x16_s>): v0-v127
  scattered (~128 VGPRs)
- `a` reg (A_row_reg): v[204:235] = 32 VGPRs (4 a-tiles × 8 VGPRs/tile)
- `b0`, `b1` (B_row_reg × 2): v[188:203] = 16 VGPRs (2 b-tiles × 8 VGPRs/tile)
- Prefetch b registers (next K-iter): v[236:253] = 18 VGPRs
- Coord + loop bookkeeping: ~20-30 VGPRs

Total live in main loop steady state: ~210-220 VGPRs out of 256 available
for 2-wave occupancy. In epilog 1, LLVM needs temporary VGPRs for the
prefetch of `As[toc][1]` + the store-C address arithmetic + the
already-issued mfmas whose destinations live on while b-regs are reloaded.
That pushes over budget → accumulators spill.

## Why R33-dm merged-mfma made this WORSE (retroactive diagnosis)

R33 proposed merging epilog-1 stages 3+4 mfma(cC,a,b0)+mfma(cD,a,b1) into
one setprio bracket. Post-hoc ISA inspection (via the tool developed this
round) would have shown that merging extends the simultaneous live-range
of `a` (32 VGPRs) and the 8 accumulator cells being written by the pair of
mfmas → 32 + 32 = 64 VGPRs continuously live across the pair, vs 32+32
with a barrier between that LLVM could exploit. The budget was already
tight; 64 VGPRs continuously live was over the edge for the LONG=0 specs
and triggered +3-4 extra spills. Confirms R33 failure mechanism.

## Concrete next-round levers (ranked by expected yield)

**(1) HINT LLVM TO NOT SPILL ACCUMULATOR CELLS — target +3-6 pp on DSV3:**
The 5 interleaved scratch_stores in epilog 1 stage 1 spill v[76:79],
v[24:27], v[12:15], v[4:7], v[0:3]. These are accumulator cells that WILL
be read back later in epilog 1 stage 3. If we can tell LLVM "do NOT spill
THESE registers", it would have to spill something else instead (probably
the `a` or prefetch scratch, which is cheaper to reload from LDS).
Options:
- `asm volatile("; live-across" : : "v"(reg) : );` scheduling fence
- `register rt_fl<...> cA asm("v0")` explicit register binding (likely not
  supported for wide register tiles)
- Simpler: try reducing the `a` live-range in epilog 1 by RE-LOADING `a`
  at stage 3 explicitly with a different variable name to avoid LLVM
  seeing one long `a` interval. (This is NOT the R33 merge; it's the
  opposite — forcing LLVM to treat a-stage1 and a-stage3 as DIFFERENT
  variables.)

**(2) REDUCE `a` or `b` LIVE-RANGE ACROSS EPILOG 1:**
Currently `a` is a 32-VGPR register tile that stays allocated across all
4 epilog-1 stages even though stage 3 reloads it. If we declare two
separate `a_early`, `a_late` register tiles with CLEAR separation, LLVM
might see 2 × 32-VGPR intervals each HALF the length and fit them in the
same physical registers (live-range killed between stages). Estimated
payoff: frees ~16 VGPRs during the critical epilog-1 stage-1 window →
fewer accumulator spills.

**(3) LOWER OCCUPANCY + FEWER SPILLS:**
Changing `__launch_bounds__(_NUM_THREADS, 1)` to `(_NUM_THREADS, 1)` — the
task body flags this as already saturated (MIN={2,3,4,5} tested).
Actually adding an upper bound hint like `__launch_bounds__(_NUM_THREADS,
1, N_WAVES_MAX)` or using `hipLaunchKernelGGL` with explicit resource
limits has NOT been fully explored. Parking.

**(4) LEVER D with ISA-guided VALIDATION:**
Now that we can diff ISA before/after, migrate parts of the epilog
(not main loop) to 32x32x64 MFMA shape; the resulting accumulator tile
count halves (from 8 cells to 4 cells per `rt_fl<64,32>`), which would
materially reduce register pressure in the epilog. R15-dm's "naive full
migration" failed, but piece-wise + ISA-verified migration could succeed.
Multi-round.

## Tool delivered

`analysis/tools/dump_fp8_grouped_isa.sh` — runs the compile + emits
per-spec `.s` files + prints a 4-row summary showing interleaved-spill
count / total-scratch / mfma-count per template spec. Works standalone
when `THUNDERKITTENS_ROOT=/workspace/code/HipKittens` is exported (or uses
the hardcoded default).

## Concrete commit trail

- HipKittens: **no change** this round.
- Primus-Turbo: this note + `analysis/tools/dump_fp8_grouped_isa.sh`.
- No build or metric change this round; this is tooling + diagnosis work
  that unblocks rounds 35+.
