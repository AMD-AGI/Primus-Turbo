# flash_attn_bwd_mla (192/128) FlyDSL gfx950 — optimize log

Campaign: backward of the DSV3 MLA FlyDSL prototype (`megaattention_overlap_bench/flydsl_mla/flash_attn_bwd_mla.py`,
`ops_bwd.py`). Target: same-GPU Step TFLOPS geomean ratio vs Turbo AITER a16 >= 1.0.
Machine smci355-ccs-aus-n02-29, GPU 0 pinned. Gate: dq/dk/dv SNR > 40 dB vs fp32 (user decision,
absolute), finite, dK/dV bitwise repeat; a FAIL shape is excluded from the geomean.

## Bring-up (before round 1) — 2026-11-09

- Fork + head_dim_v; D128 regression bitwise vs Turbo.
- Bug: dQ columns 128-191 wrong (-3 dB on d-tiles 8-11). GEMM3 adds a wave's d-tile to the pinned
  K^T base by XOR; with 6 tiles per wave starting at tile 6, 6 ^ i != 6 + i for i >= 2. Fixed by
  taking the run as interleaved pairs (`_g3_toff`), MLA only.
- b1_s64k: kernel-arg packing overflow (`struct.error 'i' format`): a flat tensor argument's length
  is packed as int32 and the q_split-2 dK slot workspace has 3.2e9 elements. Host falls back to
  q_split 1 there.
- aiter reference trap: Turbo's default `PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=1` picks aiter's a32
  kernel (7.41 ms vs 4.25 ms a16 at b2_s4k). Primus training sets 0; the bench now does too.

## Round 1 — 2026-11-09 22:40 — BASELINE

- Full set: 6 PASS, b1_s64k FAIL (dq 39.996 dB). Geomean (6 PASS) Step 738.59 / aiter 979.16
  (0.7543), Bwd 645.45 / aiter 915.90 (0.7047).
- Main kernel 0.73-0.74 of aiter's; side kernels 980 vs 487 us at b2_s4k (dK/dV slot sum 579 us).
- Resources: dkdv 430 VGPR, 0 spill, LDS 106752 B.

## Round 2 — 2026-11-09 23:05 — ACCEPT

- Change: band-interleaved a16 dQ images (band b -> image b % K, fp32 sum in the un-permute),
  K = ceil(bands / 128) capped at 4 (K 2 at S32K, K 4 at S64K; K 1 elsewhere, code unchanged there).
- Correctness: 7 / 7 PASS; D128 regression bitwise; dK/dV bitwise vs K 1.
- dQ SNR b1_s32k 41.97 -> 44.06, b1_s64k 40.00 -> 44.39 (aiter a16 44.03 / 41.95). Matches the
  running-sum model (+2 dB per halving of adds per image at large n).
- Perf: 7-PASS Step geomean 753.256 (ratio 0.7563). Common 6-PASS step +0.156 %; K > 1 shapes
  ~+1 % bwd (no zeroing cost visible).

## Round 3 — 2026-11-09 23:20 — ACCEPT

- Change: ops_bwd default q_split 2 -> 1 (no torch dK/dV slot fold; kernel unchanged).
- Result: 7 / 7 PASS; Step geomean 783.820 (+4.058 % vs R2), ratio vs aiter a16 0.7869; 4K shapes
  +8.6 / +9.2 %, b1_s64k unchanged (already q_split 1). dK/dV SNR 48.5/49.7 -> 49.6/51.2 dB.

## Round 4 — 2026-11-09 23:20 — ACCEPT (trade-off)

- Change: config sweep of the dkdv schedule knobs (one round); default g3_st_at 0 -> 1 (kernel unchanged).
- Sweep: every knob within -0.5..+0.6 % gm at B2 S4K + B1 S16K except losers (g3_st_at -1 -3.5 %,
  g3_st_n 3 -2.4 %, block_q 64 -27.5 %, block_kv 64 -48.3 %). iglp 1 aborts LLVM (AMDGPUIGroupLP
  assertion); block_q 16 trips a builder assert.
- Result: 7 / 7 PASS; Step geomean 786.352 (+0.323 % vs R3), ratio vs aiter a16 0.7888; 8K/16K
  +0.8-1.0 %, 4K -0.4/-0.5 %. SNR unchanged.

## Round 5 — 2026-11-09 23:48 — ROLLBACK

- Change: `u_entry_drain` -- vmcnt(0) before the unmasked QDESC q-loop so the compiler drops the
  Q/dO commit's vmcnt(7) x6 / vmcnt(8) (ISA confirmed: exactly those 7 waits gone).
- Result: 7 / 7 PASS; Step geomean 766.00 (-2.588 % vs R4); long shapes -3..-4.3 %, 4K -0.6/-1.0 %.
- Diagnosis: same clock (2.34 GHz at S16K), +4.2 % GRBM cycles at S16K; single-CU ATT at S4K
  showed -7.6 % cycles, so S4K ATT does not represent the long shapes. Stall moved onto atomic
  issue and the body's fetch wait. The commit waits order atomics ahead of the next fetch.

## Round 6 — 2026-11-10 00:05 — ROLLBACK (null sweep)

- Change: `qdesc_r` sweep 1..32 (descending q-walk band rotation period). All within +-0.13 %.
  Same-address dQ contention is not the limiter. Full set not run (no candidate).

## Re-profile after rounds 5-6 (round-4 code) — 2026-11-10 00:05

- ATT B1 S8K CU 1: issue cycles equal to aiter's (36.5M vs 35.3M); the gap is stall (30.7M vs
  11.1M): waitcnt +10.5M (Q/dO commit retiring dQ atomics), ds_read +4.3M, mfma +1.9M, barrier +1.3M.
- PMC B1 S16K: atomic wavefronts 2.04e8 vs aiter 1.36e8 (exactly 2/3); every atomic goes to EA on
  both; EA write stalls 3.7e7 vs 3.9e5. aiter config csv: ts_qo 16, ts 192 (kv tile 192 vs our 128).
- Direction: 192-row kv tile.

## Round 7 — 2026-11-10 00:35 — ACCEPT (trade-off)

- Change: default block_kv 128 -> 192 (NT 3), dv_pin False, host accepts a partial last band.
- Resources: 512 VGPR, 60 VGPR spilled, LDS 147712 B. Partial band verified per row segment.
- Result: 7 / 7 PASS; Step geomean 811.921 (+3.252 % vs R4), ratio vs aiter a16 0.8145; 8K..64K
  +2.4..+7.7 %, 4K -2.1 / -1.1 %. dQ SNR +0.3..+1.0 dB.

## Round 8 — 2026-11-10 01:05 — ACCEPT

- Change: config sweep at block_kv 192; defaults g2d 2 -> 1 (GEMM2 transpose-read ring depth),
  g3_sb 1 -> 0 (kernel unchanged). Spill 60 -> 46 VGPRs, scratch 244 -> 188 B per lane.
- Result: 7 / 7 PASS, SNR unchanged; Step geomean 866.046 (+6.666 % vs R7), ratio vs aiter a16
  0.8690 (bwd 0.8384); every shape +6.1..+7.8 %. A-B-A full bench: 813.89 / 868.06 / 811.97.
- Measurement finding: the in-process sweep said +3.4 %. Later dispatches in a process run with
  the scratch of the largest private segment dispatched before them; R8's dkdv is 5-8 % slower
  after any 16 KB or 32 KB per-wave scratch (256 / 512 B per lane; R7's 244 B rounds to 16 KB),
  not after 11, 17, 19, 25, 33 or 64 KB. R7 ran on 16 KB itself. In-process sweeps since round 7
  timed every config under R7's 16 KB; new harness dev/sweep_proc.py runs one process per config.

## Round 9 — 2026-11-10 01:50 — ACCEPT (trade-off)

- Change: per-process knob re-sweep (dev/sweep_proc.py); default g1_ks_outer True -> False
  (kernel unchanged). Spill 46 -> 98 VGPRs, scratch 188 -> 396 B (25 KB per wave).
- Sweep: only g1_ks_outer=False held up (+3.25 % at S16K, -0.05 % at S4K); g3d 2/3/4 and
  g2_half within +-0.8 %; iglp 0/3, g3_st_n 2, g2_fill, mfma_tie 1, g3_sb 2/3 lose 6-45 %.
  iglp=0: -1.8 % in the round-8 in-process sweep, -27 % alone.
- Result: 7 / 7 PASS, SNR unchanged; 3 full-bench processes 866.17 / 878.97 / 876.92, median
  876.918 (+1.255 % vs R8; same-session R8 864.96), ratio vs aiter a16 0.8843; >= 8K +1.0..+3.5 %,
  4K -1.7 / -0.3 %.
- Finding: this config is bimodal per process (1 of 13 processes at R8 speed on >= 8K); not aiter,
  not another kernel's scratch, not a scratch re-allocation (scratch memory trace). Unexplained.

## Round 10 — 2026-11-10 02:13 — ROLLBACK

- Change: builder knob `k_lds_ks` -- GEMM1a reads its last k_lds_ks D-steps of K from a raw LDS
  tile (one opaque per-lane base) instead of the band-resident register packs; dQ reduce takes sm.
  Also timed k_reg=False (round-9 MLA assert lifted).
- Resources: spills 98 -> 55..88 VGPRs, but hot-loop scratch_load / vmcnt(0) 3 / 2 -> 14..35 /
  10..25 (artifacts/isa_loop_counts.txt). The in-loop reloads become per-lane addresses.
- Result: SNR gate passed; per-process sweep -28..-40 % at B2 S4K / B1 S16K (k_lds_ks=6 least
  slow, -31 % gm); k_reg=False -37 % / -40 % (round 0). Full set not run.
- Finding: hot-loop vmcnt(0) count orders performance, not the spill total. Round 9's 2 hot-loop
  vmcnt(0) sit in GEMM3 after the QPF_AT 0 Q/dO prefetch, so they retire it.

## Round 11 — 2026-11-10 02:21 — ROLLBACK

- Change: builder knob `qpf_at` (Q/dO prefetch issue point, hard-coded 0 for D >= 128); swept 0..3.
- Resources: qpf_at 1 spill 125 / 504 B (32 KB per wave), hot loop 16 / 11 reloads / vmcnt(0);
  qpf_at 2 spill 75 / 304 B, hot loop 6 / 1; qpf_at 3 does not compile (dominance, masked path).
- Result: SNR gate passed; per-process sweep (3 rounds) qpf_at 2 -2.22 % / -1.07 % (B2 S4K /
  B1 S16K), qpf_at 1 -0.84 % / -36.99 %. Full set not run.
- Finding: one fewer hot-loop vmcnt(0) does not repay the shorter prefetch cover; round 9's two
  remaining waits are not the steady-state limiter. Two consecutive rollbacks -> re-profile.

## Re-profile after rounds 10-11 (round-9 code) — 2026-11-10 02:30

- ATT B1 S8K CU 1, r9 dkdv vs aiter a16 main kernel, same session (profiles/att_r11_{r9,aiter}_s8k,
  att_r11_hot_breakdown.txt, att_r11_hot_timeline.txt, att_r11_wg_gaps.txt).
- Per wave per 192x32 tile: ours (hot-loop instructions, ATT latency sum) 5101 cycles = 2786
  issue + 2315 stall; aiter (whole kernel latency sum / 44 waves / 524 16-row iterations per WG,
  x 2) 4012 = 3058 issue + 953 stall, or 4352 from its wave span (1.14M per WG). Same MFMA count
  per tile (156 vs 155). Measured end-to-end bwd at B1 S8K: aiter / ours = 0.864 (round 9 run 3).
- Stall delta per tile: MFMA +739 (ours 887: runs of back-to-back MFMAs, 12 cycles each; aiter
  interleaves MFMA with VALU / ds_read), waitcnt +526 (two vmcnt(0) in GEMM1a = 250 + 99 cycles:
  spill reloads of 3 K packs issued 10 and 150 cycles before use; the Q/dO prefetch was issued
  ~1300 cycles earlier), ds_read +162, salu +146 (s_nop 5 MFMA hazards).
- Correction to rounds 10-11: those two waits expose the scratch reload latency itself, not the
  Q/dO prefetch drain, which is why moving the prefetch issue point (round 11) did not help.
- WG gaps on CU 1: ours 18 WGs, 643K idle cycles between WGs (3.4 % of the CU window), WG spans
  0.23-1.56M (causal); aiter 11 WGs of uniform 1.12-1.18M (kv tiles paired), 23K idle (0.2 %).
  The round-4 code (0 spills) had 5.2 % gaps at S8K, so the gaps are not a scratch effect.
  Hypothesis (not verified): in-order dispatch stalls on a full XCD while other XCDs have free CUs,
  which non-uniform WG lengths make frequent.
- Direction: (1) dispatch order of the kv tiles (round 12); (2) the GEMM1a K-pack reloads;
  (3) MFMA / VALU interleave in GEMM1 and GEMM3.

## Round 12 — 2026-11-10 02:38 — ROLLBACK

- Change: builder knob `kv_order` (bijective kv tile remap in the XCD-major decode): 0 in-head
  order (r9), 1 zigzag, 2 tile axis outermost.
- Result: SNR gate passed (dK / dV identical); per-process sweep (3 rounds, B2 S4K / B1 S8K /
  B1 S16K) zigzag -19.2 / -18.8 / -24.7 %, tile-major -5.0 / -1.3 / -2.3 %. Full set not run.
- Finding: r9's order is best. Tile-major also gives up per-head L2 sharing, so the WG-gap
  hypothesis from the re-profile is not resolved.

## Round 13 — 2026-11-10 03:05 — ACCEPT

- Change: builder knob `llvm_opts`; MLA `ops_bwd` passes amdgpu-disable-unclustered-high-rp-
  reschedule=True and greedy-regclass-priority-trumps-globalness=True (kernel source unchanged).
- Screening: 13 cl::opts by ISA; trackers cut spills 98 -> 66 but is -21.8 %; max-ilp and
  greedy-reverse-local-assignment raise spills to 137-140; most others give identical ISA counts.
- Sweep (per-process, 3 rounds): U +1.29..+1.47 %, G +1.12..+1.15 %, U+G +2.02 % gm.
- Result: 7 / 7 PASS, SNR unchanged; 3 full-bench processes 896.18 / 892.38 / 892.66, median
  892.658 (+1.795 % vs R9; +1.429 % vs same-session r9 ref 880.08), ratio vs aiter a16 0.8964
  (bwd 0.8722); every shape +1.1..+1.7 % vs the reference.

## Round 14 — 2026-11-10 03:18 — ROLLBACK

- Change: llvm_opts (U, G, X) for 17 more options X (config sweep, no source change).
- Screening: 8 options give byte-identical ISA (incl. amdgpu-mfma-vgpr-form=False,
  amdgpu-spill-vgpr-to-agpr=False, misched-limit, dag-maps-huge-region); iterative-* strategies and
  misched-regpressure=False raise spills to 131-219; igrouplp exact solver does not finish.
- Sweep (per-process, 3 rounds): misched-cluster=False -1.08 %, enable-misched=False -0.33 %,
  misched-fusion=False +0.02 %, rewrite-partial-reg-uses=False +0.05 %.
- Finding: the pre-RA scheduler off costs only 0.33 %; the loop order is FlyDSL's emission order.
  Rollback count 1.

## Round 15 — 2026-11-10 08:36 — ACCEPT

- Change: builder knob `qdesc_csub`; MLA `ops_bwd` on. The descending q walk's
  `(t + rot) % _u_nb` (runtime divisor; two per unmasked trip, 131 + 100 cycles of fast-path
  division + 73 of 64-bit checks per 5249-cycle iteration in the r9 ATT) becomes a band-hoisted
  rotation plus one 32-bit conditional subtract, with the wrap kept as a real branch.
- Six equivalent forms screened by ISA: as a select / opaque select / two stride loops / 32-bit
  modulo they all lose the division's fast/slow-path blocks and spill 132-268 VGPRs (select form
  timed: -43 %); with the wrap as an scf.if (asm in the arm) the branch count stays 13 and the
  spills fall 98 -> 94. The r13 layout depends on the loop's block structure.
- Result: 7 / 7 PASS, dK/dV bitwise equal to the modulo walk; full set 918.97 / 922.13 / 922.41,
  median 922.132 (+3.302 % vs R13; +3.052 % vs same-session r13 ref 894.82), step ratio vs aiter
  0.9263 (bwd 0.9091); every shape +1.8..+5.2 %. Rollback counter reset.

## Round 16 — 2026-11-10 09:40 — ACCEPT (trade-off)

- Change: per-process knob re-sweep on the r15 base; MLA `ops_bwd` g3d 6 -> 4 (GEMM3 kstep
  transpose-read ring depth; kernel source unchanged). Spills 94, scratch 380 B unchanged.
- Screening: 17 configs; g3d 2..5, g2_half, g3_sb=1 keep 94 spills; g3_st_at 2/3, iglp=3,
  g3_dbat=6 spill 124-169; g3_dbat=4, dv_pin, g3_kreg fail MLA asserts.
- Sweep (3 shapes): g3d=4 +1.39 % / +1.13 % gm in two sweeps (round 9 on r9 code: +0.40 %);
  g3d 1/2/3/5 -0.71..+0.12 %; g1_ks_outer=True -3.77 %; g3_st_n=2 -25 %. Focused B1/B2 S4K A/B
  (6 rounds): +2.54 / +1.28 %.
- Result: 7 / 7 PASS, SNR unchanged; full set 931.61 / 925.93 / 927.33, median 927.329
  (+0.564 % vs R15; +0.405 % vs same-session r15 ref 923.59), step ratio vs aiter 0.9311
  (bwd 0.9148); b1_s4k -1.2 % in the median run (per-process spread; +2.5 % in the focused A/B),
  other shapes +0.14..+1.15 %.

## Round 17 — 2026-11-10 10:20 — ROLLBACK

- Change: builder knob `g3_rdv`: the deferred GEMM3 between the WAR and the publish barrier on
  the Q_PREF route (1 = ahead of the Q/dO commit, 2 = after it). Spills unchanged (94).
- Sweep (4 rounds): mode 1 -8.30 % gm, mode 2 -1.76 % gm. GEMM1a's first K-pack scratch reloads
  are pinned right after the publish barrier; r16's GEMM3 is their cover, and with mode 1 the
  in-order vmcnt also makes them wait on the 8 Q/dO prefetch loads issued just before it.
- Finding: on the MLA path (one q head per kv head) the deferred GEMM3 and the trip's own dS
  writes share dS slot 0 with no barrier between them; ~2200-cycle margin in the r15 ATT against
  ~100 cycles of barrier skew. Formal race, not observed. Rollback count 1.
