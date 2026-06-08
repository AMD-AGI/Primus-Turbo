# gfx942 (CDNA3) — Kernel Implementation Notes

Per-subsystem guidance for writing and tuning CDNA3 kernels (HIP, Composable Kernel, ROCDL, Triton-MI, or hand assembly). Read [`overview.md`](overview.md) first for the hardware budget and occupancy model.

Everything below is CDNA3 / gfx942 behavior. Instruction names refer to the public MI300 CDNA3 ISA.

---

## 1. Matrix Core (MFMA)

MFMA is the single largest performance lever on gfx942. One `V_MFMA_<acc>_MxNxK[_BB]_<input>` op computes a full `M×N×K` block outer product `D = C + A·B`, with A/B in (Acc)VGPRs and C/D ideally pinned in AccVGPRs across the K-loop.

**Tile / dtype selection.** Pick the opcode whose per-instruction K is the largest that divides your K-loop, so the loop issues the fewest MFMAs. The dense catalog covers FP64, FP32, TF32/XF32, FP16, BF16, INT8, FP8, BF8. CDNA3's widest F16/BF16 forms are `32x32x8` (K=8) and `16x16x16` (K=16); INT8 reaches K=32. Choosing the MFMA shape fixes the A/B/C/D register layout, so decide it **before** laying out LDS tiles and epilogue swizzles.

**Accumulators in AccVGPRs.** Keep C/D in the AccVGPR pool (`ACC_CD=1`) so each K-step is `C += A·B` with no spill. Shuttle to/from arch-VGPRs only at prologue/epilogue with `V_ACCVGPR_READ` / `V_ACCVGPR_WRITE` (and `V_ACCVGPR_MOV_B32` for acc↔acc). AccVGPR pressure caps occupancy — trade tile size against waves/CU.

**Operand broadcast.** `CBSZ`/`ABID` broadcast one A-block across `2^CBSZ` peer blocks; `BLGP` swizzles B's lane groups. Use these for grouped-GEMM and small-K tiles to avoid re-loading A/B from LDS/VGPR. `CBSZ`/`ABID` must be ≤ `log2(blocks)`; on F64 MFMAs the `BLGP` field is repurposed as a NEG modifier, not a swizzle.

**Lane-native layout.** Plan A/B loads and C/D writeback so each lane already holds the exact VGPR item the MFMA layout demands, eliminating post-LDS shuffles before the issue. SRC0/SRC1 must be VGPR-only; only SRC2 (the C input) accepts an inline constant.

**Structured sparsity.** `V_SMFMAC_*` runs 4:2 structured sparsity on the A K-axis (two of every four K-elements pruned to zero) with a packed per-lane 2-bit index VGPR, doubling effective throughput when A can be pruned.

**MFMA ignores MODE.** The matrix core forces round-to-nearest-even, ignores the denorm and EXEC-mask MODE bits, and raises no FP exceptions. Do **not** try to mask matrix lanes with `EXEC` — mask via tile selection. `FP16_OVFL` still controls F32 clamp-to-±MAX vs ±INF, and INT32 saturate-vs-wrap. XF32/TF32 silently truncates mantissas to 10 bits — never use it where full FP32 precision is required.

**Required-independent window (hazard).** Between an MFMA write and its dependent read, insert exactly the number of independent VALU/MFMA instructions the latency table mandates so the matrix pipe issues back-to-back without RAW/WAW stalls. Schedule independent work into that window rather than padding with `S_NOP` when possible; under-filling stalls, over-padding wastes issue slots.

**Small-K alternative.** When K is tiny, the MFMA setup cost may not pay off — `V_DOT2/4/8` (packed 2×f16 / 4×i8 / 8×i4 dot products with an FP32/INT32 accumulator) fuse a small inner product into one VALU issue and can beat MFMA in that regime.

---

## 2. Low Precision & Numerics

**FP8 is FNUZ on CDNA3.** FP8 = E4M3 with **bias 8, max 240, no Inf encoding**; BF8 = E5M2 with bias 16, max 57344. This is **not** the OCP encoding CDNA4 uses (bias 7 / max 448). Treat FP8 format as a correctness gate when porting — re-derive per-tensor/per-block scales rather than reusing CDNA4 ones.

**Packed conversions.** `V_CVT_PK_FP8_F32` / `V_CVT_PK_BF8_F32` pack two F32 lanes into one 16-bit half of a VGPR (half selected by `OPSEL[3]`, the other half preserved), laying the result out directly for an MFMA FP8/BF8 input register. The up-convert path `V_CVT_PK_F32_{FP8,BF8}` reads a 16-bit word via SDWA and writes two F32 into an even-aligned VGPR pair.

**Hardware stochastic rounding.** `V_CVT_SR_{FP8,BF8}_F32` adds the top 20/21 bits of a U32 seed into the F32 mantissa then truncates — training-quality requantization with no software RNG. Destination byte is steered by `OPSEL[3:2]`, so four `_SR_` ops can pack four results into one VGPR. The seed quality is the kernel's responsibility: drive it from a per-lane PRNG (Philox/LCG/threefry) and advance every call, or "stochastic" rounding becomes biased deterministic rounding.

**Gotchas:**
- `SH_MEM_CONFIG` bit[8] must be set to 1 for the dispatch, or BF8/FP8 results are wrong.
- The down-convert / SR variants are VOP3A — **no SDWA**; lane steering is `OPSEL` only.
- `CVT_*_F32` up-converts have **no 4-cycle forwarding**: insert a NOP or an unrelated VGPR write between two converts writing different bytes/halves of the same destination, or you get stale bytes.
- `CVT_SR_*` / `CVT_PK_*` accept VGPRs only (no SGPR/literal/inline).
- `FP16_OVFL` is a MODE bit (saturate vs NaN/Inf on overflow), not per-instruction — switching it affects every later convert in the wave.
- FP8 (max ≈240) saturates well inside typical training ranges; pair it with an appropriate scale, or use BF8 (max ≈57344) for gradients.

**MODE register prologue.** Program `MODE.FP_ROUND` / `FP_DENORM` / `FP16_OVFL` once at kernel entry with `S_SETREG_IMM32_B32`; every subsequent `V_ADD`/`V_MUL`/`V_FMA` then inherits the rounding/denorm behavior with zero per-instruction cost. Plan float-atomic reducers around the fixed RNE rounding, per-opcode denormal-flush table, and quiet-NaN propagation so concurrent reducers stay deterministic.

**Transcendentals for softmax / norm hot paths.** Lower softmax/GELU/LayerNorm/RMSNorm inner loops to single-issue `V_EXP` / `V_LOG` / `V_RCP` / `V_RSQ` / `V_SQRT` / `V_SIN` / `V_COS` and accept their ~1-ULP error. Reserve the full IEEE FP64 divide macro (`V_DIV_SCALE_F64` → `V_RCP_F64` + Newton–Raphson `V_FMA_F64` → `V_DIV_FMAS_F64` → `V_DIV_FIXUP_F64`) for cases that genuinely need it.

**Packed epilogues.** `V_PK_*` (VOP3P) issues paired 16-bit (and paired 32-bit) ops with `OPSEL`/`OPSEL_HI`/`NEG_HI` to halve VALU cost on bias/activation/elementwise epilogues and pre-MFMA casts. `V_MAD_MIX_F32` / `V_MAD_MIXLO/HI_F16` fuse per-lane f16↔f32 conversion with a multiply-add.

---

## 3. LDS & Shared Memory

LDS is 64 KiB/CU across 32 banks (512 Dwords/bank, 32-bit wide). A bank conflict (two lanes hitting the same bank in one phase) serializes the access and cuts effective bandwidth.

**Bank-conflict avoidance.** Prefer XOR swizzle — XOR the column index with bits of the row index so a wavefront hits 32 distinct banks (zero overhead). Padding the row stride to break bank alignment also works but wastes LDS. Size all swizzle/padding formulas to **32 banks** — this is the most common CDNA3-vs-CDNA4 porting bug (CDNA4 has 64).

**Coalesce LDS traffic.** `DS_READ2` / `DS_WRITE2` (and the `_ST64` forms) issue two LDS accesses in one instruction, halving LDS instruction count on tile loads/stores.

**In-LDS reductions.** `DS_*` atomics (`DS_ADD/MIN/MAX/AND/OR/XOR`, plus `DS_PK_ADD_F16/BF16` for packed halves) fold per-wave partial reductions directly into LDS, removing the explicit VGPR-shuffle + barrier + broadcast stage. `DS_APPEND` / `DS_CONSUME` against an LDS/GDS counter give a hardware wavefront-ordered FIFO without synthesizing one from atomic-add + ballot.

**Direct global→LDS staging.** `GLOBAL_LOAD_LDS_*` / `SCRATCH_LOAD_LDS_*` (and the MUBUF `BUFFER_LOAD_*_LDS` path) stream device memory or per-lane scratch directly into LDS, bypassing the VGPR detour so VGPRs stay free for MFMA accumulators. This is the backbone of double-buffered GEMM prologues.

**Cross-workgroup sync.** `DS_GWS_*` (GDS) barrier/semaphore opcodes — initialized once with `DS_GWS_INIT` — synchronize waves across work-groups on-chip without round-tripping host-coherent global memory.

**Note (vs CDNA4):** CDNA3 has **no `DS_READ_*_TR` transpose-load**. To feed MFMA A/B in the expected lane layout you still need an explicit VGPR transpose / cross-lane shuffle (DPP or DS-permute), or a pre-swizzled LDS write. Do not port a CDNA4 transpose-load tile loader unchanged.

---

## 4. Memory System & Atomics

**FLAT segment selection.** Prefer `GLOBAL_*` over `FLAT_*` for HIP pointer loads/stores, and reserve `SCRATCH_*` for swizzled per-lane spills. Select the segment by where the pointer actually lives instead of letting the compiler default to `FLAT` (which pays an aperture-decode tax).

**Buffer-resource addressing.** Configure a buffer V# with `const_stride` / `const_index_stride` / `const_element_size` so the hardware address engine emits AoS-interleaved (swizzled) layout, replacing per-lane address VALU for register-blocked, LDS-free streaming. The V#'s `NumRecords` range-check silently drops OOB lanes — align ragged-tile starts to natural alignment and let the range-check replace per-lane bounds-test branches in the epilogue.

**Float / integer atomics.** Plan f16/f32/f64 LDS and L2 float atomics around the fixed RNE rounding, per-opcode denormal-flush table, and QNaN propagation so concurrent reducers are deterministic. `ATOMIC_INC` / `ATOMIC_DEC` do a single-instruction wrapping increment/decrement against a user limit — ideal for global counters / queue indices.

**Targeted cache control.** Use `BUFFER_INV` / `BUFFER_WBL2` (vector L1/L2) and `S_DCACHE_INV/WB/...` (scalar L0) to enforce just the coherency edge a producer→consumer handoff needs, instead of a blanket `S_WAITCNT 0` + barrier flush.

**XCD / L2 locality.** The 304 CUs are 8 XCD chiplets, each with its **own 4 MB L2 slice**. The dispatcher round-robins work-groups across XCDs, so consecutive `blockIdx` land on different L2s. Remapping the work-group id (keep `chunk_size` consecutive ids on the same XCD via a chiplet-chunked transform) makes adjacent tiles share a warm L2 — a meaningful win for GEMM/attention with spatial locality. This relies on the dispatcher's `% num_xcds` policy; combine it with persistent kernels by remapping the persistent work-id, not `blockIdx.x`.

---

## 5. Cross-Lane Data Movement

CDNA3 wavefronts are 64 lanes. Pick the cheapest primitive per pattern:

- **DPP** (`row_shr` / `quad_perm` / `wf_sr1` / `row_bcast`, appended as a modifier dword to a VOP1/VOP2/VOPC): fuses a neighbor-lane fetch into the same VALU issue — reductions, shifts, stencils with no LDS round-trip.
- **DS crossbar** (`DS_SWIZZLE_B32` for fixed compile-time patterns, `DS_PERMUTE_B32` / `DS_BPERMUTE_B32` for arbitrary lane-indexed shuffles): a zero-LDS-byte cross-lane crossbar for shuffles, reductions, and transposes.
- **Lane↔scalar** (`V_READFIRSTLANE_B32` / `V_READLANE_B32` / `V_WRITELANE_B32` / `V_SWAP_B32`): move single-lane values between VGPR and SGPR (or swap two VGPRs) with no temp, ignoring EXEC.
- **Prefix-popcount** (`V_MBCNT_LO_U32_B32` then `V_MBCNT_HI_U32_B32`): each lane's exclusive prefix-popcount over a wave64 EXEC/VCC mask in two ALU ops — no LDS, no permute. Backbone of stream-compaction / MoE scatter.
- **Byte crossbar** (`V_PERM_B32` with a packed selector): permute any 4 bytes from two source dwords into a destination dword in one VOP3A — INT8 transpose, byte-quant pack, format swaps.

---

## 6. Scheduling & Hazards

**Exact hazard NOP counts.** When the hardware does not interlock a producer→consumer pair, insert the **exact** `S_NOP` / `V_NOP` wait-state count from the CDNA3 hazard table rather than letting the assembler conservatively over-pad. Over-padding directly wastes issue slots in hot loops.

**Per-pipe `S_WAITCNT`.** Use separate `vmcnt` (VMEM), `lgkmcnt` (LDS/SMEM/GDS), and `expcnt` (export) thresholds to overlap the three streams, instead of a blanket `S_WAITCNT 0`. Hoist long-latency loads early, keep up to each counter's max depth in flight, and fence only the specific counter the next consumer needs.

**Manual scheduler steering.** In MFMA hot loops, `__builtin_amdgcn_sched_group_barrier` / ROCDL `sched_mfma` / `sched_dsrd` / `sched_vmem` / `sched_dswr` hints (terminated by `sched_barrier(0)`) let you hand-interleave MFMA / LDS-read / VMEM / LDS-write groups into a chosen pattern. Derive the group counts from compile-time tile algebra (MFMA cluster count, buffer-load count, ds-read count) so changing the tile shape auto-retunes the interleave.

**Software pipelining.** The standard CDNA GEMM body is a 2-stage LDS ping-pong: toggle `write_stage = read_stage ^ 1` each K-iteration, separated by one barrier, prefetching the next K-tile (often via `BUFFER_LOAD_*_LDS` async DMA) while MFMA consumes the current stage. Pair it with three-level tiling + XOR-16 swizzle + the sched hints above, and optionally a CShuffle LDS epilogue that re-tiles accumulators for coalesced vectorized global stores.

---

## 7. Control Flow & Divergence

- **Uniform (scalar) decisions** should ride `SCC`: produce it with `S_CMP*` / `S_BITCMP*` and consume it directly via `S_CSELECT` / `S_CMOV` / `S_CBRANCH_SCC*` so the branch never touches VALU/VGPRs.
- **Divergent regions**: open with a single `S_*_SAVEEXEC_B64` (fuses the EXEC update with saving old EXEC) and use EXEC as the live predicate. `V_CMPX_*` fuses the compare with the EXEC update, eliding the `V_CMP` + `S_AND_SAVEEXEC` pair. `S_CBRANCH_EXECZ` early-exits empty masks.
- When EXEC is provably all-zero over a stretch (e.g. a fully-masked tail tile), `S_SETVSKIP` makes the hardware skip issue entirely instead of issuing-and-masking.

---

## 8. Pitfall Checklist

- Launching too little work — fill all 304 CUs or use persistent scheduling.
- Reusing NVIDIA warp-32 assumptions — wavefront is 64; reductions and subgroup logic must match.
- Treating CDNA3 FP8 as interchangeable with CDNA4 FP8 — FNUZ vs OCP is a correctness gate.
- Sizing LDS swizzle/padding for the wrong bank count (32 here, 64 on CDNA4).
- Filling LDS aggressively without checking the occupancy trade-off (64 KiB is tight).
- Masking MFMA lanes with EXEC — MFMA ignores it; mask by tile selection.
- Forgetting `SH_MEM_CONFIG[8]=1` before FP8/BF8 conversions.
- Forgetting the no-4-cycle-forwarding NOP between two converts into the same VGPR.
- Chasing instruction-level tweaks before classifying the kernel as memory-, compute-, or stall-bound via the profiler.
- Porting a CDNA4 transpose-load or block-scaled MFMA loader to gfx942 — neither exists here.
