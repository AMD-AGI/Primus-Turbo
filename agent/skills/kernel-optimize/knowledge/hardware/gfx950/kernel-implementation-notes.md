# gfx950 (CDNA4) — Kernel Implementation Notes

Per-subsystem guidance for writing and tuning CDNA4 kernels (HIP, Composable Kernel, ROCDL, Triton-MI, or hand assembly). Read [`overview.md`](overview.md) first for the hardware budget and occupancy model.

Everything below is CDNA4 / gfx950 behavior. Instruction names refer to the public CDNA4 ISA. Sections 2 and 4's transpose-load are the headline new capabilities versus CDNA3 — most CDNA4-specific wins come from them.

---

## 1. Matrix Core (MFMA)

One `V_MFMA_<acc>_MxNxK[_BB]_<input>` op computes a full `M×N×K` block outer product `D = C + A·B`. The dense catalog covers FP64, FP32, TF32, FP16, BF16, INT8, FP8, BF8.

**Wider K than CDNA3.** CDNA4 adds wide-K dense forms: F16/BF16 reach `32x32x16` (K=16) and `16x16x32` (K=32); INT8 reaches `16x16x64` / `32x32x32`. Pick the largest K that divides your K-loop so the loop issues the fewest MFMAs. A K=64 loop in F16 is 4 issues at `32x32x16` vs 8 at `32x32x8` — half the dynamic MFMA count. The chosen shape fixes the A/B/C/D register layout, so decide it **before** laying out LDS tiles and the epilogue swizzle.

**Accumulators in AccVGPRs.** Keep C/D in the AccVGPR pool and shuttle to/from arch-VGPRs only at prologue/epilogue with `V_ACCVGPR_READ` / `V_ACCVGPR_WRITE` / `V_ACCVGPR_MOV_B32`. This roughly doubles effective per-lane register space at the cost of explicit copies; AccVGPR pressure caps occupancy.

**Operand broadcast.** `CBSZ`/`ABID` broadcast an A-block across lane groups; `BLGP` swizzles B. Reuse them to share A/B across blocks and cut VGPR/LDS bandwidth in grouped-GEMM and small-K tiles. (On `F8F6F4` opcodes these fields take an alternate meaning — see §2.)

**Structured sparsity.** `V_SMFMAC_*` runs 4:2 structured-sparse `A · dense B` with per-lane 2-bit non-zero indices packed into the SRC2 VGPR layout, doubling throughput on a pre-pruned A.

**MFMA ignores MODE.** Forced RNE, denorms kept, EXEC ignored, no FP exceptions. Mask matrix lanes by tile selection, never with EXEC. Scale modifiers are not supported on FP16/BF16/INT8 MFMA — scale those outside the op.

**Required-independent scheduling (hazard).** Between an MFMA write and a dependent read, schedule enough independent VALU/MFMA work to cover the per-opcode required-wait count, hiding accumulator latency without `S_NOP`. The CDNA4 dependency table is finer-grained than CDNA3's: same-opcode SrcC accumulation forwards with **0** waits, but SrcA/B reuse, cross-opcode reads, and VMEM/LDS reads of an MFMA result need progressively more waits (commonly 2–20 depending on overlap and opcode class). A `V_CMPX` write to EXEC before an MFMA needs 4 waits (no EXEC forwarding into the matrix pipe).

---

## 2. Block-Scaled (MX) & Mixed-Precision Matrix — the CDNA4 headline

**Block-scaled MX MFMA.** `V_MFMA_SCALE_F32_16X16X128_F8F6F4` and `V_MFMA_SCALE_F32_32X32X64_F8F6F4` are 4-DWORD instructions: a "Load-Scale" half (`SRC0` = A scale, `SRC1` = B scale) fused with a normal F8/F6/F4 MFMA. The hardware multiplies one **8-bit E8M0 (bias 127) exponent scale per 32-element K-block** of A and of B into the mantissa dot product before F32 accumulation:

```
D[m][n] += A_scale[m, kblk] * B_scale[n, kblk] * sum_{k in kblk} A[m,k] * B[k,n]
```

This is exactly the MXFP8 / MXFP6 / MXFP4 contract (one E8M0 per 32 mantissas). It is the only hardware path for MX-format GEMM on gfx950.

Key rules:
- **The scale lasts exactly one instruction.** A following plain `V_MFMA_F32_*_F8F6F4` is unscaled; every K-block needs its own SCALE op.
- **Scale byte addressing.** Each lane's `SRC0`/`SRC1` packs four candidate 8-bit scales; the 2-bit `{OP_SEL_HI, OP_SEL}` selects byte 0–3. Wrong selection silently scales by the wrong K-block.
- **Lane→(M, K-block) layout is fixed and asymmetric** and differs between `16x16x128` and `32x32x64`. Use the documented layout verbatim; do not infer it.
- **E8M0 `0xFF` is NaN** — clamp quantized exponents to a valid range before packing or the result is poisoned.
- Precompute per-32-lane E8M0 block scales into a compact scale tile (quantize → pack → MFMA-repack) so the inner loop just streams scales to VGPRs and never recomputes dequant factors.

**`F8F6F4` mixed-precision MFMA.** `V_MFMA_F32_*_F8F6F4` lets A and B **independently** select FP8 / BF8 / FP6 / BF6 / FP4 via the alternate `CBSZ`/`BLGP` field encoding (and `ABID[0]` for the SCALE variants), halving operand register/bandwidth pressure versus a wider dtype. The format is selected by how mantissas are packed in the operand VGPRs, not by a different opcode — mis-packing yields silently wrong arithmetic. Cycle count is variable ("16 or 32"): F8×F8 does **not** get the small-cycle path.

---

## 3. Low Precision & Numerics

**FP8 is OCP on CDNA4.** FP8 = E4M3 with **bias 7, max 448** (`0x7F`/`0xFF` are NaN); BF8 = E5M2 with bias 15, max 57344, **with Inf encodings**. This is **not** the FNUZ encoding CDNA3 uses (bias 8 / max 240 / no Inf). FP8 format is a correctness gate when porting — re-derive scales rather than reusing CDNA3 ones.

**New narrow formats.** FP6 (E2M3, max 7.5), BF6 (E3M2, max 28), FP4 (E2M1, max 6.0). All are denormal-permissive and have no Inf/NaN encodings — out-of-range values saturate to ±max, sub-minimum values flush to zero.

**Scaled convert family.** `V_CVT_SCALEF32_*` (de)quantizes between FP32/FP16/BF16 and FP8/BF8/FP6/BF6/FP4 with one E8M0 scale broadcast across a packed block. Pick the SR (stochastic-round) variant for narrow targets where RNE bias hurts. Notes: conversions from F4/F6/F8 ignore input modifiers; conversion to F4/F6 does not support `FP16_OVFL` (to F8 does); multipass F32/F16/BF16→FP4/FP6 SR uses an internally-generated random value (the PRNG VGPR is read but not written). `CVT_*` accept VGPRs only.

**Hardware PRNG.** `V_PRNG_B32` advances a per-lane 32-bit LFSR in one VALU op, seeding from any vector register without external RNG state — feed it to the SR converters.

**Gotchas (shared with CDNA3):**
- `SH_MEM_CONFIG` bit[8] must be 1 for correct FP8/BF8 results.
- `CVT_*_F32` up-converts have **no 4-cycle forwarding** — insert a NOP or unrelated VGPR write between two converts writing different bytes/halves of the same destination.
- `FP16_OVFL` is a MODE bit (saturate vs NaN/Inf), not per-instruction.

**MODE register.** Program `FP_ROUND` / `FP_DENORM` (and `FP16_OVFL` / `DX10_CLAMP`) via `S_SETREG` once, and combine with the per-instruction `CLAMP` bit for per-op saturation control. Plan float-atomic reducers around the fixed RNE rounding, per-opcode denormal-flush table, and SNaN→QNaN propagation for determinism.

**Transcendentals.** Use single-issue `V_RCP`/`V_RSQ`/`V_SQRT`/`V_EXP`/`V_LOG`/`V_SIN`/`V_COS` (~1 ULP) for softmax/norm/activation hot paths; assemble the full IEEE divide macro (`V_DIV_SCALE` → `V_RCP` + Newton–Raphson → `V_DIV_FMAS` → `V_DIV_FIXUP`) only when true IEEE divide semantics are required.

**Packed math throughput.** `V_PK_*` / `V_DOT*` (VOP3P) issue two FP16/BF16/INT16 ops — or 2/4/8 FP32/INT8/INT4 dot-product lanes — per 32-bit VGPR per cycle. Lower elementwise/activation epilogues and reductions to these for 2×+ throughput, using `op_sel`/`op_sel_hi`/`neg` for in-instruction lane control.

---

## 4. LDS & Shared Memory

LDS is 160 KiB/CU across 64 banks (640 Dwords/bank, 32-bit wide) — 2.5× the capacity and 2× the banks of CDNA3. The larger budget makes deeper multi-buffer pipelines and larger staged tiles realistic.

**MFMA Transpose-Load from LDS (`DS_READ_*_TR`) — CDNA4-only.** `DS_READ_B64_TR_B{16,8,4}` and `DS_READ_B96_TR_B6` transpose 16/8/6/4-bit data while moving it from LDS to VGPRs, delivering MFMA A/B operand tiles directly in the lane layout MFMA expects and **eliminating the explicit in-register VGPR transpose / shuffle pass** that CDNA3 needs. Rules: EXEC must be all-1s before issue; the LDS address must be aligned to the data size; 64-bit-or-larger DS ops need even-aligned VGPRs (the `B96_TR_B6` form is exempt). Two instructions with different addresses load one complete matrix. This is a primary lever for sub-byte/half-word GEMM and attention operand staging — port loaders to use it instead of a manual transpose.

**Bank-conflict avoidance.** Prefer XOR swizzle (column index XOR row-index bits → 64 distinct banks, zero overhead); padding the row stride also works. Size all formulas to **64 banks** — reusing CDNA3's 32-bank formulas is a common porting bug. Co-design LDS layouts so column-style MFMA reads hit zero bank conflicts and `ds_read`/`ds_write` overlap with global loads and MFMA compute.

**Coalesce / reduce LDS traffic.** Prefer paired `DS_READ2`/`DS_WRITE2`, lane-stride `DS_*_ADDTID`, and in-LDS `DS_*_ATOMIC` ops to maximize LDS throughput and fold reductions into LDS.

**Direct device→LDS staging.** `BUFFER_/GLOBAL_/SCRATCH_LOAD_LDS_*` stream tiles from memory directly into LDS, bypassing the VGPR roundtrip so VGPRs stay free for MFMA accumulators. The backbone of double-buffered prologues; combine with the transpose-load above for sub-byte operands.

**Tile budgeting.** Size tiles so each fits the 160 KiB LDS budget against the workgroup/wavefront × per-CU LDS × HBM hierarchy; remember the 1280 B allocation block (no wrap) and the `MIN(SPI size, M0)` clamp.

---

## 5. Memory System & Atomics

**FLAT segment selection.** Emit `GLOBAL_*` or `SCRATCH_*` instead of generic `FLAT_*` whenever the address provably lives in one aperture — `FLAT` pays an aperture-decode tax and double-counts both `VM_CNT` and `LGKM_CNT`, hurting `s_waitcnt` scheduling.

**Buffer / MUBUF / MTBUF addressing.** Address HBM through a V# buffer-resource descriptor so the hardware combines base + SGPR offset + stride×(index+TID) + inst/VGPR offset with built-in range checking, optional AoS swizzling (via `stride`/`element_size`/`index_stride`), and optional D16 packing. The range-check silently drops OOB lanes — use it to replace per-lane epilogue bounds branches.

**HBM-resident atomic accumulators.** The `BUFFER_ATOMIC_*` family (int, f32/f64, packed `pk_add_f16`/`pk_add_bf16`) updates HBM accumulators directly via a range-checked V#; bracket with `BUFFER_WBL2` / `BUFFER_INV` to establish cross-CU coherency. For split-K accumulation, bias each warp's per-lane byte offset (e.g. `warpid * stride`) so concurrent warps target disjoint packed slots and never collide on the same atomic address.

**Scalar atomics — CDNA4 addition.** `S_ATOMIC_*` / `S_BUFFER_ATOMIC_*` update wave-uniform counters and shared state on the scalar path, landing in SGPRs and bypassing the vector memory pipe — cheaper than a VGPR atomic for uniform state.

**Cache scope (`sc0`/`sc1`).** Pick the memory-op family (`global_`/`flat_`/`buffer_`+`lds`) and `sc0`/`sc1` cache-scope flags to match each access's data lifetime and producer-consumer coherence domain, avoiding stale-cache stalls and silent dropped stores. Issue wave-uniform reads through the scalar path (`S_LOAD_*` / `S_BUFFER_LOAD_*`) so all 64 lanes share one broadcast value; bracket producer/consumer boundaries with `S_DCACHE_INV` / `S_DCACHE_WB`.

**XCD / L2 locality.** The 256 CUs are 8 XCD chiplets, each with its **own 4 MB L2 slice**. The dispatcher round-robins work-groups across XCDs, so consecutive `blockIdx` land on different L2s. Remap the work-group id (keep `chunk_size` consecutive ids on the same XCD) so adjacent tiles share a warm L2 — a real win for GEMM/attention with spatial locality. With persistent kernels, remap the persistent work-id, not `blockIdx.x`.

---

## 6. Cross-Lane Data Movement

CDNA4 wavefronts are 64 lanes. Pick the cheapest primitive per pattern:

- **DPP** (quad-perm, row shift/rotate/mirror, wave shift/rotate, row-broadcast appended to a VOP1/VOP2/VOPC): fuses a neighbor-lane fetch into the same VALU issue — reductions, prefix sums, stencils with no LDS round-trip.
- **DS crossbar** (`DS_SWIZZLE_B32` fixed compile-time patterns, `DS_PERMUTE_B32` / `DS_BPERMUTE_B32` arbitrary lane-indexed): shuffle across the 64 lanes through the LDS crossbar **without reading/writing LDS storage** — replaces LDS round-trips for reductions, transposes, RoPE partner-lane swaps.
- **Register-tile layout swap** (`v_permlane16_swap`): reinterpret a register tile between row/col MFMA layouts **in place**, so the same tile feeds both A and B operands without an LDS round-trip.
- **Lane↔scalar** (`V_READFIRSTLANE/READLANE/WRITELANE_B32`, `V_SWAP_B32`): single-lane moves between VGPR and SGPR, ignoring EXEC.
- **Scalar wave-mask introspection** (`S_WQM`/`S_QUADMASK`, `S_BCNT0/1`, etc.): helper-lane / quad-mask transforms and active/inactive-lane popcounts on the SALU for divergence-aware bookkeeping.
- **3-input bitwise collapse** (`V_BITOP3_B32/B16`): encode any 3-input bitwise function's 8-bit truth table as the SIMM8 immediate, replacing 2–3 chained AND/OR/XOR/NOT ops — CDNA4 addition.

---

## 7. Scheduling & Hazards

**Manual wait states.** For hazards the hardware does not interlock, insert `S_NOP` / `V_NOP` / `DS_NOP` (or appropriate `S_WAITCNT`) per the required-software-wait-state table — e.g. VALU-writes-SGPR → VMEM-reads-SGPR needs 5 waits, VALU-writes-VGPR → DPP-reads-VGPR has its own count. Insert the exact count rather than over-padding; use `S_SLEEP` only for coarse throttling.

**`S_WAITCNT` counter scheduling.** Schedule VMEM/LDS/export traffic against `vmcnt` / `lgkmcnt` / `expcnt`: hoist long-latency loads, keep up to each counter's max depth in flight, and fence only the specific counter the next consumer needs (never a blanket `S_WAITCNT 0`). Avoiding `FLAT` (see §5) keeps these counters un-double-counted.

**Manual scheduler steering.** In MFMA hot loops, `__builtin_amdgcn_sched_group_barrier` / ROCDL `sched_mfma` / `sched_dsrd` / `sched_vmem` / `sched_dswr` hints let you hand-interleave MFMA / LDS-read / VMEM / LDS-write groups. Derive group counts from compile-time tile/wave algebra (MFMA cluster count, buffer-load count, ds-read count) so a tile-shape change auto-retunes the interleave.

**Software pipelining.** The 160 KiB LDS enables deeper multi-buffer ping-pong than CDNA3. A common structure: a small producer warp-group streams global→LDS while a larger consumer warp-group issues MFMA, coupled by a multi-stage LDS ring buffer so the matrix math never stalls on HBM. Pair with `DS_READ_*_TR` (§4) to skip operand transposes.

**Joint co-saturation.** Co-select tile shape, MFMA instruction width, and waves/SIMD so VGPR+AGPR budget, LDS footprint, and matrix-unit issue rate all saturate together rather than one resource starving the others. A VGPR liveness pass over compiled AMDGCN (find dead VGPR windows, remap above-boundary registers into holes) can bump waves/SIMD at the next occupancy boundary (64/73/85/102/128/170/256 next_free_vgpr steps).

---

## 8. Control Flow & Divergence

- **Uniform decisions** ride `SCC`: produce with `S_CMP*`, consume via `S_CBRANCH_SCC0/1` / `S_CSELECT` / `S_CMOV` so the branch never touches VALU/VGPRs.
- **Divergent regions**: push/pop EXEC via the `SAVEEXEC` family with `S_CBRANCH_EXECZ` early-exit; `V_CMPX_*` fuses the compare with the EXEC update, eliding `S_AND_SAVEEXEC`. `WREXEC` accelerates waterfall loops.
- **PC control flow**: `S_GETPC` / `S_SETPC` / `S_SWAPPC` / `S_CALL` give cheap intra-kernel subroutines; the SOPP lifecycle primitives (`S_ENDPGM`, `S_TRAP`, `S_SETPRIO`, `S_SETHALT`, …) manage wave lifecycle. A trap handler can read `TRAPSTS`/`STATUS`/`HW_ID` via `S_GETREG` plus the saved `{TTMP1,TTMP0}` to classify memory-violation exceptions.

---

## 9. Pitfall Checklist

- Reusing CDNA3 32-bank LDS swizzle/padding — CDNA4 has 64 banks and a 1280 B (not 512 B) allocation block.
- Treating 160 KiB LDS as "free" — large tiles still trade against occupancy; the 1280 B block rounding can demote a tier.
- Treating CDNA4 FP8 as interchangeable with CDNA3 FP8 — OCP vs FNUZ is a correctness gate.
- Assuming a block-scale persists across MFMAs — it lasts exactly one SCALE instruction.
- Mis-selecting the E8M0 scale byte (`OP_SEL`) or inferring the lane→(M,K-block) layout instead of using the documented one.
- Mis-packing `F8F6F4` mantissas — the format is chosen by packing, not opcode; errors are silent.
- Skipping `DS_READ_*_TR` and keeping a manual VGPR transpose — you leave a CDNA4-specific win on the table.
- Forgetting EXEC must be all-1s before a transpose-load, or that the LDS address must be data-size aligned.
- Porting old TF32 assumptions — there is no hardware-native TF32 matrix path on CDNA4; do not import a CDNA3 TF32 path.
- Masking MFMA lanes with EXEC — MFMA ignores it.
- Comparing utilization against the wrong-generation peak table.
- Chasing instruction-level tweaks before classifying memory- vs compute- vs stall-bound via the profiler.
