# gfx950 / flash_attn_fwd_mla (d_qk 192, d_v 128) / FlyDSL tips

- 2026-11-09 (campaign flash_attn_fwd_mla_flydsl_gfx950_20261109, round 2): at block_m 256
  (8 waves / CU, 256-VGPR cap per wave) the dual-wave kernel is register-bound. Adding
  `sched_group_barrier` DS-read groups to pace the QK tail K reads did not change their issue
  point: the register allocator gave the tail only three register quads, so post-RA false
  dependencies pin each read behind the previous MFMA. Check the ISA register reuse before
  spending a round on read-pacing hints; free VGPRs first. Measured: geomean -0.59 %, rolled back.
- 2026-11-09 (same campaign, round 3): k_head (K k-steps read before the first QK MFMA) 4 -> 2
  cut spills 105 -> 90 and raised forward geomean +3.10 % (4K shapes +4.8 / +5.4 %). At the
  256-VGPR cap, fewer head K packs beats deeper K prefetch for d_qk 192.
- 2026-11-09 (same campaign, round 5): the dual-wave kernel's `fold_mem_barriers` was gated on
  `Q_HEADS_PER_WG > 1`, but its hazard argument holds for any WG whose waves share the
  cooperative K/V DMA. Enabling it for block_m 256 (8 waves, one head) cut barriers to one per
  tile: +4.77 % geomean, output bitwise unchanged. For sync-only changes, verify with a bitwise
  comparison against the previous round's snapshot (and confirm the two builds really differ).
- 2026-11-09 (same campaign, rounds 6-7): the entire 90-VGPR spill came from one 96-wide
  `llvm.fptrunc` of scaled Q (d_qk 192 -> 12 k-steps x 8). A non-power-of-two vector width
  scalarizes during legalization (`v_cvt_pk_bf16_f32 vD, vF32, s0`, one bf16 per VGPR); the
  MFMA packs are then rebuilt with `v_perm_b32`, and the allocator spills the unpacked elements
  (two scratch dword reloads + `vmcnt(0)` + `v_perm` per Q dword in the epilogue). Truncating
  per 8-element pack: spill 90 -> 0, +6.21 % geomean (4K +14 %), output bitwise unchanged.
  Grep the ISA for `v_perm_b32` / `, s0` conversions after changing a head dim. Moving the Q
  definition instead (reload from global for the epilogue) made it worse (-8.34 %): the
  epilogue re-spilled the reloaded Q.
- 2026-11-09 (same campaign, rounds 8-11): at block_m 256 a WG's waves launch on SIMD 0, 2, 1, 3
  slot 0 then slot 1 (ATT), so wave w and w + 4 share a SIMD and are barrier-locked into the
  same cluster. The dual-wave `s_setprio` around P*V was built for block_m 128, where the SIMD
  partner is another, unsynchronized WG; splitting the bump by wave half cost -2.29 % (later
  half) and -1.33 % (earlier half). Also neutral or negative once spill-free: k_head 2 (-0.35 %)
  and letting the row-sum MFMA chain interleave with QK (-0.03 %). The remaining gap to aiter
  (same 32-row / 64-KV tiling, same double buffers) is its 8-stage pipeline with a one-stage
  wave-group offset, which this kernel's one-barrier-per-tile protocol cannot absorb.
- 2026-11-09 (swp8 campaign, rounds 0-1): to compare an 8-stage kernel against aiter, run ATT
  on both at the same shape and sum per s_barrier segment latency / visits (visits = barrier
  hits / 2). Stage visits on the traced CU matched aiter, so stage costs compare directly and
  the barrier wait shows which side of each (lead stage k, lag stage k-1) pair is critical.
  Seeding QK with -reference costs ~34 v_mov per tile (16-VGPR splat per half, copied again by
  the allocator); a zero srcC is free. Removing it alone was neutral (-0.16 %) because the
  64-bit loop-exit compare of the index IV (VALU `v_mov_b64` + `v_cmp_lt_u64`, no 64-bit
  s_cmp_lt on gfx9) then stalled 211 units in the trailing group's mem stage, and a broadcast
  scalar subtract compiled to scalar `v_add_f32` rather than `v_pk_add_f32`. Check the loop
  head's VALU before judging a compute-stage VALU cut.
- 2026-11-09 (swp8 campaign, rounds 2-3): FlyDSL `scf_range` always converts bounds to index,
  so an i32 `range` still gives a 64-bit VALU exit test; a dynamic `while j < n:` on an i32
  counter lowers to `s_cmp_lt_u32` (+1.32 % geomean, bitwise unchanged). In MFMA-interleaved
  code, `v_pk_add_f32` issued right after a `v_mfma_f32_32x32x16_bf16` stalls ~40 ATT units
  (until the MFMA finishes) while scalar `v_add_f32` / `v_max3_f32` issue freely: forcing a
  per-tile shift into 16 packed adds cost -2.92 % vs leaving it as 32 scalar adds. Separately,
  the trailing group's first mem-stage VALU (even a 32-bit v_add_u32) stalled ~200 units when
  the partner wave's stage started directly with MFMA + exp, delaying the in-order LDS reads
  behind it; the round-1 '64-bit compare' stall was this position effect, not the width.
- 2026-11-09 (round 12): on MI355X at B1 S16K both this kernel and aiter sit at the ~1.4 kW socket
  cap, so L2 traffic shows up as clock, not cycles. Head-fastest grids put only ~2 co-resident
  readers on each K/V stream per XCD (TCC hit 0.37). For Hkv = Hq (MLA / MHA), relabel CTAs so each
  XCD (lin % 8) owns every 8th head and walks 8 q ranks of one head adjacently: hit 0.37 -> 0.80,
  same TCC requests, +5.6 % cycles yet ~4-5 % higher clock, +6.8 % to +9.6 % at S >= 16K, bitwise
  unchanged. Keep the group small relative to the q-rank count: at S4K (16 ranks) an 8-rank group
  cost -3.9 % (cycles +25 %, clock +17 %: idle tail). Compare amd-smi clock / power during a
  sustained loop, not only PMC cycles, before judging a memory-side change.
- 2026-11-09 (round 13): with an XCD-grouped causal order, dispatch the last few q ranks (here 8)
  rank-major across all of the XCD's heads again (longest-first); this removed the S4K idle tail
  (cycles -15.7 %, +3.5 %) and kept the S >= 16K L2 gain. Grouping and LPT conflict only in the
  final wave, so restore LPT there rather than shrinking the group everywhere.
- 2026-11-09 (round 14): XCD group sweep for the MLA forward (Hq = Hkv = 128, block_m 256): 16 q
  ranks per group beat 8 (+0.4 % on a GPU pinned at the 1.4 kW cap, +1.6 % on one that was not) and
  4 lost ~2.5 %; the tail length (4 / 8 / 16) was noise at group 16. Results also moved ~5-8 %
  between two MI355X nodes, so re-measure the baseline on the same node before comparing rounds.
- 2026-11-09 (round 16): on MI355X the 16x16x32 bf16 MFMA runs cooler than 32x32x16: moving the
  MLA QK (60 % of the FLOPs) to it raised the sustained clock 1930 -> 2030 MHz at ~25 W less
  socket power (B1 S16K), net +0.7 % geomean (+2.1 % at S64K, ~0 at S <= 8K) despite +3-5 %
  cycles. What it takes to keep the 32x32 score layout downstream: (a) 8 `v_permlane16_swap` per
  32-kv half (A-operand kv order [0, 8, 4, 12] + r % 4), issued at the score consumer in a later
  region; putting them right after the QK MFMAs made the LLVM scheduler hold all K packs live (51
  spills vs 9). (b) Conflict-free K reads. Measured gfx950 LDS rules: ds_read_b128 is served in
  16-lane phases, each 32-lane half split by quad parity (quads {0,3,5,6} / {1,2,4,7}), bank =
  dword address % 64; ds_read_b64 phases are lanes 0-31 / 32-63; ds_read2_b64 behaves like a
  16-byte read. LLVM merges same-base ds_read_b64 pairs into ds_read2(st64)_b64, so splitting a
  b128 read does not avoid its phase rule. With the 16x16x32 A layout the k-chunk groups 0 / 1
  (and 2 / 3) share a phase and must hit the same d % 64: pair d chunks 128 apart, and for d with
  no partner XOR-swizzle the 16-byte chunk index (m ^ 4) on half the rows through the per-lane
  global address of the direct-to-LDS DMA (bank conflicts 8.1e8 -> 0, free on the write side).
- 2026-11-09 (round 16, measurement): back-to-back quick runs of an unchanged package drifted
  ~1.5 % (the later run slower), enough to flip a 1 % verdict; alternate the candidate and the
  baseline in one session (2-3 rotations) instead. Upper-bound variants that point loads at
  addresses already read elsewhere overstate the gain (the compiler merges the duplicate LDS
  reads: 144 -> 108 ds_read_b128): a "conflict-free" fake promised +2.7 %, the real swizzle
  delivered +0.7 %. Count the loads in the fake's ISA before trusting its ceiling.
- 2026-11-09 (round 17): with QK on 16x16x32, run PV on 16x16x32 too, straight off the raw QK
  accumulators instead of converting S to the 32x32 layout. With the A-operand kv order
  [0, 8, 4, 12] + r % 4, the 16x16 D tile of QK^T (kv x q) is the B operand of O^T = V^T P^T
  without any lane movement: the pack for (32-kv half h, q tile qt) is elements
  [4 qt .. 4 qt + 3, 8 + 4 qt .. 8 + 4 qt + 3]. V comes from `ds_read_b64_tr_b16` on the usual V
  LDS rows (kv-major), with the per-lane kv offset [0, 8, 4, 12][lane // 16] matching that order.
  Keep row scalars (m, l, LSE) one q per lane and pair the two q tiles with one
  `permlane16_swap(x, x)`; the row sum is a ones-MFMA per q tile. MLA forward (B1 S16K): swaps
  101 -> 39 in the kernel (main loop 34 -> 8), clock +4.6 % at the same power, +1.22 % full-set
  geomean, every shape up. Cycles still rose (+4.2 % GRBM) even though MFMA busy cycles were
  identical: on this power-limited kernel (PPT throttle active 68 % of the time although sampled
  socket power reads ~1365 W) each 16x16x32 move paid off as clock, not as cycles.
- 2026-11-09 (post-round-17, power): on MI355X read the limiter from `amd-smi metric -v` (growth
  of PPT_ACCUMULATED over ACCUMULATION_COUNTER during the run), not from sampled SOCKET_POWER: the
  attention forward was PPT-throttled 54-68 % of the time while sampling ~1365 W (< 1400 W cap).
  An occupied GPU issuing only s_nop draws 346 W at 2396 MHz (idle 231 W), so at the power limit
  cycles saved without work saved buy almost nothing; work energy per FLOP decides. Measure
  ceilings as wrong-result fakes at the power limit (sustained clock + TFLOPS), e.g. exp2 -> fma
  (+6.7 %) and no main-loop K/V DMA (+11.1 %); with both the MLA forward hit the 2400 MHz max
  clock and became cycle-bound (1.21x aiter). aiter's opus d192 kernel keeps the MFMA pipe ~86 %
  busy (MFMA busy / GRBM vs an MFMA-only probe) but spends more energy per FLOP. Pure-MFMA loops
  are clock-capped near 2083 MHz (16x16x32) / 1880 MHz (32x32x16) without PPT throttling.
- 2026-11-09 (round 18, XCD order): with 32 CUs per XCD and 1 CTA per CU, a q-group of 32 ranks
  of one head per XCD (levels (32, 16, 8) over an 8-rank tail, remainder of each level going to the
  next) beat 16-rank groups at long S on the MLA forward: +1.2 % to +2.0 % at B1 S32K-64K,
  +0.64 % full-set geomean. Two measurement lessons. (1) Changing only the CTA-setup arithmetic
  re-scheduled the whole kernel: at S4K / S8K, where the CTA order was identical across builds, the
  three builds differed by up to 1.3 %, consistently over 5 rotated runs. Compare candidates at a
  shape where the order does not change to size the code-generation part before crediting the order.
  (2) At the power limit the L2 miss count did not predict speed: one build had 16 % fewer misses
  and ran slower at a lower clock. Use sustained ms/call and clock, not L2 counters, to decide.
  A wrong-result probe that points every CTA at kv head 0 bounds the locality gain (-3.5 % time
  at B1 S16K).
- 2026-11-09 (round 19, rollback): rewriting a hierarchical XCD map from a progressive-select
  loop into independent section maps + a select cascade did not change the main-loop ISA (same
  791 opcodes, identical sequence) and did not recover the ~1 % S4K gap vs a two-level (16, 8)
  build that shares CTA order at that shape. Diff the hot-loop opcode sequence before blaming
  setup style for a code-generation gap; if the loop already matches, the gap likely sits in the
  number of group levels / constants, not in how the selects are written.
- 2026-11-09 (round 20, accept): at BLOCK_M 256 the causal diagonal spans four KV tiles, and a
  32-col half starting at block column c is dead for every wave whose last row is < c. Predicating
  all such halves in the drain (QK and P*V, wave-uniform `chunk_live` branches; trace-time None for
  halves no wave can find dead) is bitwise neutral. It gave +2.8 % at B1 S4K and +0.25 % at S64K,
  fewer cycles at the same clock, more than the MFMA count alone suggests. Re-derive "always live"
  assumptions whenever BLOCK_M changes; they were written for BLOCK_M 128.
- 2026-11-09 (pre-round-20 probe): to price LDS -> VGPR traffic, alias odd K slots / odd P*V steps
  to the preceding packs (wrong result, half the `ds_read` bytes, same MFMA count). Here that bound
  was +9 % at the same clock, K and V costing the same per byte (so `ds_read_b64_tr_b16` is not
  dearer per byte than `ds_read_b128`). The same probe for the ones-MFMA row sum read ~0 %: don't
  move hidden MFMA work to VALU. `SQ_INSTS_LDS_LOAD_BANDWIDTH` counts 64 B units and matched the
  analytic LDS byte count within 1 %.
- 64 q rows per wave / one wave per SIMD (QT4) for MLA fwd 192/128 on gfx950: +5.8-7.3 % at S >= 16K,
  only +0.1-2 % at S = 4K (per-WG prologue and causal drain are no longer hidden by a second wave).
  O must live in AGPRs written by asm MFMAs ("=a,v,v,0", has_side_effects); the arch VGPR file is then
  full (256), so any extra live f32 (VALU row sums, seed dedupe) spills.
- IGroupLP `sched_group_barrier` does not count inline-asm MFMAs: hand-place asm sites with
  `sched_barrier(0)` after each slot; builtin MFMAs can also drift across `s_barrier` unless fenced.
- The hazard recognizer does not see inline asm: if a VALU result (e.g. a cvt) is scheduled right before
  an asm MFMA reading it, no wait states are inserted and the MFMA reads stale data (inf rows). Keep
  producers pinned away from asm sites (side-effect asm pins also stop MachineSink).
- `v_dot2c_f32_bf16` runs on the MFMA pipe on gfx950 (28 cycles after an MFMA, 4 after another dot2c):
  not a way to offload softmax row sums from the matrix core. The ones-MFMA row sum is the cheapest.
- ATT solo-wave costs on gfx950: VALU right after an MFMA +4 cycles, v_exp_f32 ~8 issue cycles (~12 after
  an MFMA), every s_waitcnt / s_setprio / s_nop >= 4 cycles of issue. With one wave per SIMD the loop is
  bound by total issue slots in the MFMA shadows (16-cycle 16x16x32 MFMA leaves 12 free).
- Solo-wave loops: measure per-MFMA-slot overflow, not instruction totals. `wip/att_bubbles.py`-style
  map from ATT stats: slot content = MFMA issue + following instructions' latency; > 16 cycles
  (16x16x32) is a pipe bubble, < 16 is spare that cannot be banked. Removing an instruction only helps
  in an overflowing slot (R5: filling an M0 hazard nop by adding a softmax piece to a full DMA slot
  cost -1.7 %).
- A `sched_barrier(0)` only after each slot's work is not enough to place work behind an asm MFMA:
  the scheduler hoists the next slot's VALU above its own asm MFMA (into the previous shadow). Fence
  both sides of the work for exact placement.
- The hazard recognizer counts inline asm as zero wait states: `s_mov m0` -> asm MFMA -> LDS DMA still
  gets an `s_nop 0`. Harmless unless the slot overflows.
- Reading the next QK's first K k-step inside the preceding P*V site (buffer already visible, not
  being refilled) removed an exposed ~48-cycle LDS wait: +0.34 %. Carrying the other one over the
  loop back edge lost -0.25 %: with a full 512-register file, extra live values cost AGPR spill moves.
- Under the power cap (B1 S16K ~1275 W), cycle savings that slightly raise energy per FLOP lose
  their clock back (R6: +0.5 % work/cycle, -0.8 % clock); pure stall removal converts best.
- Persistent job loop (FlyDSL `range(..., init=[...])` around the whole kernel body): LICM hoists every
  job-invariant lane term of the global offsets (Q-load `row * stride + d`, K / V DMA voffsets) out of
  the job loop, where they live across the KV loop next to the per-job sums. At 512 registers that was
  8 -> 77 spill slots and ~40 scratch reloads per job, which ate the whole launch saving (R9 -0.26 %).
  Re-derive those terms per job from an opaque tid (`_pin_regs([tid], "=v")` in the per-job bind):
  9 spill slots, +2.6 % (4K +4.5 %). Always diff the spill count of persistent vs one-job-per-CTA ISA.
- Persistent schedule for causal MLA fwd: a host-side LPT replay of the hardware dispatch (CTA c on
  XCD c % 8, job i to queue i % 8, cost = KV tiles + ~8 tiles fixed) keeps the per-XCD L2 grouping
  and the dynamic-dispatch balance with no atomics; a static snake / chunked split was -10 % on
  dsv2lite in simulation. Read the table with `s_buffer_load`, one job ahead.
