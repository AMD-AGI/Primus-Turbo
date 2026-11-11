# gfx950 / flash_attn_bwd_mla (d_qk 192, d_v 128) / FlyDSL tips

- 2026-11-09 (campaign flash_attn_bwd_mla_flydsl_gfx950_20261109, bring-up): Turbo's
  `PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32` defaults to 1, which makes the aiter backward use its a32
  (fp32 atomic dQ) kernel; Primus training (`runner/helpers/envs/base_env.sh`) and Turbo's
  `bench_attention_turbo.py` set it to 0 (a16). At B2 S4K H128 192/128 the two differ 7.41 vs
  4.25 ms. Set the env to 0 before the first Turbo attention forward (it is read once) when
  benchmarking against aiter, and check the kernel name (`..._a16_pssk`).
- 2026-11-09 (same, bring-up): a FlyDSL flat tensor argument's length is packed as int32
  (`struct.error: 'i' format requires ...` in `<flydsl-cabi-fill>`). At B1 S64K H128 the q_split-2
  dK slot workspace (3.2e9 elements) trips it. Size workspaces under 2**31 elements, or hand the
  kernel a view and move the descriptor base with `base_byte_offset` (64-bit) for the rest.
- 2026-11-09 (same, bring-up): the Turbo dkdv GEMM3 adds a wave's d-tile index to its pinned
  K^T LDS base by XOR. That only equals addition when the tile run per wave is a power of two
  starting at a multiple of it; D_QK 192 gives 6 tiles per wave and silently wrong dQ columns
  128-191 (-3 dB SNR on those d-tiles only). Per-d-tile SNR localizes this class of bug fast.
- 2026-11-09 (same, round 1): a16 dQ SNR falls 1.2-2 dB per doubling of S (one bf16 atomic per
  kv band per element); at block_kv 128 B1 S64K is 39.996 dB, aiter a16 41.95 dB. q_split 2 bf16
  dK/dV slots cost 1.1-1.5 dB of dK/dV SNR vs q_split 1.
- 2026-11-09 (same campaign, round 2): spreading the a16 dQ bands over K bf16 images (band b ->
  image b % K, fp32 sum in the un-permute) recovers dQ SNR as the running-sum model predicts:
  +2.09 dB at K 2 (S32K), +4.39 dB at K 4 (S64K), and cost nothing measurable (those shapes ran
  ~1 % faster). At small S (few adds per element) the gain is only +0.7-1.6 dB because the
  per-band bf16 conversion, not the sum, dominates there. A FlyDSL kernel can be handed image 0
  as its tensor argument and reach the rest with a 64-bit descriptor base offset.
- 2026-11-09 (same campaign, round 3): at block_kv 128 the fused dK/dV grid already has
  >= 4096 work-groups at S4K H128, so q_split 2 buys no balance; it only adds the bf16 slot fold
  (two torch reduces, ~9 % of the backward at 4K) and 1.1-1.6 dB of dK/dV SNR loss. q_split 1:
  +4.06 % step geomean (4K +9 %). Check the fold's share in a kernel breakdown before keeping a split.
- 2026-11-09 (same campaign, round 4): re-sweeping the Turbo dkdv schedule knobs after the MLA
  192/128 generalization found nothing large: g2d, g3d, mfma_tie, g3_dbat, g3_sb, iglp all within
  +-0.6 %; only g3_st_at 0 -> 1 held up (+1.3 % at S16K, -0.2..-0.5 % at S4K, +0.32 % step geomean).
  Burst dQ stores (g3_st_at -1) cost 3.5 %, block_kv 64 costs 48 %. iglp 1 aborts the compiler
  (AMDGPUIGroupLP.cpp:2220 DSWCounters assertion) and kills the whole Python process, so sweep it
  in a separate process or leave it out.
- 2026-11-09 (same campaign, round 5, rolled back): in the Q_PREF dkdv body the compiler sizes the
  Q/dO commit's vmcnt waits for the loop's ENTRY edge, so on the latch path they retire the previous
  head-step's dQ atomics. A vmcnt(0) at loop entry removes them (ISA confirmed) but costs -2.6 %
  step geomean (+4.2 % cycles at S16K, clock unchanged): the stall moves to the atomic issue and
  to the next fetch, which then queues behind the atomics. Do not "fix" those waits; shorten the
  atomics instead. Also: a single-CU ATT at S4K said -7.6 % cycles for this change, so check
  whole-device cycles (GRBM_GUI_ACTIVE + kernel trace) at the long shapes before trusting ATT.
- 2026-11-10 (same campaign, re-profile + round 7): on gfx950 every bf16 `buffer_atomic_pk_add`
  of the a16 dQ image is executed past L2 (TCC_EA0_ATOMIC == TCC_ATOMIC, ours and aiter), so the
  dQ atomic count is a first-order cost. aiter's hd192_128 bwd uses a 192-row kv tile
  (`hsa/gfx950/fmha_v3_bwd/fmha_bwd_dqdkdv.csv`: ts_qo 16, ts 192; hd128: ts 256) and issues 2/3
  of a 128-row kernel's atomic wavefronts. Moving the FlyDSL fused dkdv from block_kv 128 to 192
  (NT 3, dv_pin False) gave +3.25 % step geomean (+7.7 % at 32K, -2 % at 4K) even with 60 VGPRs
  spilled. A partial last kv band needs no kernel change for causal square shapes.
- 2026-11-10 (same): compare ATT stall vs issue (latency - stall) against the reference kernel,
  not latency alone: here the issue cycles matched aiter's within 3 % and the whole gap was stall.
- 2026-11-10 (same campaign, round 8): a spilling kernel's speed depends on the scratch layout of
  the process, not only on its own code. Measured on gfx950 (ROCm in the xiaoming-dev container):
  after one dispatch of any kernel with 256 B (16 KB per wave) or 512 B (32 KB per wave) private
  segment per lane, the MLA dkdv kernel (188 B, 46 spilled VGPRs) ran 5-8 % slower for the rest of
  the process; after 176, 272, 304, 400, 528 or 1024 B it did not, and a later larger
  non-power-of-two dispatch (528 B after 256 B) removed the slowdown. A kernel with 241-256 B per
  lane (rounded to 16 KB per wave) carries the slow layout on its own. Consequences: (1) never
  rank configs with different scratch sizes inside one process -- run one process per config
  (the in-process sweep reported +3.4 % for a change the per-process bench measured at +6.7 %);
  (2) read `.private_segment_fixed_size` from the final ISA next to the spill count, and treat
  16 / 32 KB per wave as a regression; (3) zero spills is the only state immune to what else the
  training process runs. Mechanism not verified (likely same-channel spill traffic at a
  power-of-two wave stride); 64 KB per wave was fast, unexplained.
- 2026-11-10 (same, round 8): at block_kv 192 (NT 3) the GEMM2 transpose-read ring depth g2d 1
  beats the D128 fused path's 2: 14 fewer spilled VGPRs and +1.5 % at equal scratch layout.
- 2026-11-10 (same campaign, round 9): with one process per config the knob ranking changed a lot
  (iglp=0: -1.8 % in-process, -27 % alone). At MLA NT 3, g1_ks_outer=False (GEMM1 contraction not
  outermost) is +2..+3.5 % at >= 8K and -1.7 % at B1 S4K even though it doubles the spill (46 -> 98
  VGPRs); for D128 the same switch was -1.1 %. That variant is bimodal per process: 1 of 13
  processes ran at the old speed on >= 8K, with no aiter, other-kernel-scratch or scratch
  re-allocation cause (rocprofv3 --scratch-memory-trace shows one allocation). Benchmark a
  spill-heavy variant in >= 3 processes and take the median.
- 2026-11-10 (same campaign, round 10): count the hot loop's scratch_load and vmcnt(0), not the
  spill total. Taking K out of the register file at MLA NT 3 (GEMM1a reading 1..6 of its 6 D-steps
  from LDS, or k_reg=False) cut spills 98 -> 55..88 VGPRs but raised the hot loop's vmcnt(0) from
  2 to 10..25, and every variant was 28-40 % slower (per-process sweep, B2 S4K / B1 S16K). The
  freed registers do not stay free: the loop still used all 256 VGPRs + 256 AGPRs, and the
  allocator spilled loop-invariant per-lane addresses (Q/dO prefetch (row, col) pairs, ds_read /
  ds_write bases), each reloaded right before its use. On gfx950 a reload's wait is in-order with
  every earlier load / store / atomic, so a reload after the Q/dO prefetch retires the prefetch.
- 2026-11-10 (same campaign, rounds 11-12 + re-profile): (1) a spill reload placed right before
  its use exposes the reload's own latency (ATT: 250 and 99 cycles per q-block at two GEMM1a
  vmcnt(0), reloads issued 10 and 150 cycles before use), independent of what else is in flight;
  moving the Q/dO prefetch issue point later (qpf_at) does not fix that and loses 1.7 %+.
  (2) Compare per-tile cycles against the reference, not per-CU windows: the traced CU ran 18 of
  our WGs and 11 of aiter's (aiter pairs kv tiles so every WG lasts the same). (3) Our causal
  WGs leave 3.4 % of the traced CU idle between WGs (aiter 0.2 %); the round-4 code with 0 spills
  had 5.2 %, so it is not scratch. Re-ordering the kv tiles (zigzag -21 %, tile-major -2.9 %) did
  not help; the in-head long-to-short order with per-head L2 sharing is best of the three.
- 2026-11-10 (same campaign, round 13): sweep LLVM cl::opts on a register-bound FlyDSL kernel
  (compile_hints llvm_options; screen by ISA first, then time per process). On the MLA dkdv
  (512 VGPR, 98 spilled) amdgpu-disable-unclustered-high-rp-reschedule=True +
  greedy-regclass-priority-trumps-globalness=True gave +1.4 % on every shape with the same spill
  count; amdgpu-use-amdgpu-trackers=True cut spills 98 -> 66 and was 22 % slower (spill count is
  not the metric); amdgpu-sched-strategy max-ilp / max-memory-clause and
  greedy-reverse-local-assignment raised spills; exhaustive-register-search, schedule-metric-bias
  0 / 100, relaxed occupancy and enable-local-reassign produced identical ISA counts.
  split-spill-mode (an enum) cannot be set through set_llvm_option_str (std::bad_alloc).
- 2026-11-10 (same campaign, rounds 14-15): look for runtime-divisor `%` / `//` in loop index
  math of FlyDSL kernels. An Index (i64) modulo by a runtime value lowers to a 64-bit
  fast/slow-path check plus a v_rcp-based 32-bit division, ~100 cycles per use with no MFMA in
  flight; the MLA dkdv q loop had two per trip (~5 % of the iteration). Replacing it is only a
  win if the loop's block structure survives: the same arithmetic as a select (or a 32-bit
  modulo, or the walk split into two loops) took the 512-register body from 98 to 132-268
  spilled VGPRs and -43 %; with the wrap as an scf.if whose arm holds an `=s,0` side-effect asm
  (so SimplifyCFG cannot fold it), spills fell to 94 and the kernel got +3.05 %. When a change
  to a register-bound loop explodes the spill count, check s_cbranch counts before blaming the
  arithmetic. Also: with iglp on, turning the pre-RA machine scheduler off (enable-misched=0)
  cost only 0.33 %, so LLVM scheduler options have little left to give on this body.
- 2026-11-10 (same campaign, round 16): re-sweep the builder knobs after every structural round,
  not once per campaign. On the MLA dkdv the GEMM3 transpose-read ring depth g3d=4 (vs the full
  band, 6) measured +0.40 % on the round-9 code and +1.13..+1.39 % (3-shape gm) after round 15
  removed the loop-tail division, with the same spill count; the changed hot loop moved the knob's
  optimum. Time the 4K shapes in their own per-process A/B before calling a full-set dip there a
  regression: B1 S4K varies by up to 5 % between processes on this kernel.
- 2026-11-10 (same campaign, round 17): before moving MFMA work across an s_barrier to fill an
  MFMA-idle rendezvous, check what that work currently covers on the far side. On the MLA dkdv the
  deferred GEMM3 sits right after the publish barrier, and the register allocator's spill reloads
  for the next GEMM are pinned there too (memory ops do not cross the barrier's acquire fence);
  GEMM3 is their only cover. Moving GEMM3 in front of the barrier cost -1.8 % (after the Q/dO
  commit) and -8.3 % (ahead of it, which also put the 8 HBM prefetch loads right before the
  reloads: gfx950's in-order vmcnt then makes the reload wait cover the prefetch latency). Also:
  with one q head per kv head, a dS "double buffer" indexed by head_local % 2 degenerates to one
  slot; check the slot index math whenever the GQA group size can be 1.
- 2026-11-10 (same campaign, round 18): a register-neutral fix can still cost through the register
  allocator. Alternating the MLA dS slots per trip (two loop-carried VGPR bases replacing two
  pinned ones, one SGPR step) added 7 instructions to a 604-instruction hot loop but moved spills
  94 -> 98, and one of GEMM1a's K-pack scratch reloads landed right before its vmcnt(0) and MFMA:
  -1.3 % step on every shape. Instruction and scratch_load counts do not show this; count the MFMAs
  between each hot-loop reload and the first `s_waitcnt vmcnt(k)` that guarantees it (vmcnt is in
  issue order on gfx950; dev/isa_reload_cover.py in the campaign) and screen variants on that.
- 2026-11-11 (same campaign, round 19): the hot-loop reload cover is a good ISA-only screen for
  knob re-sweeps after an allocator-perturbing change. Of 22 knob variants on the MLA dkdv, the only
  three that brought the minimum cover back above 0 (12-13 MFMAs) were also the only three that
  timed positive (+1.6..+2.1 % gm); the winner (GEMM2 prefetch depth g2d 1 -> 2) kept the spill count
  (98) and scratch_load count identical and still recovered the whole -1.3 % of the fix. Knob gains
  do not add: stacking a second cover-restoring knob on the winner lost 0.3-1.2 %.
