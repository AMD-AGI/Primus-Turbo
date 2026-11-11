# optimize.md — flash_attn_mla_fwd_r20_flydsl_gfx950_20261110

## DEFINE_TARGET — 2026-11-10 15:03 (UTC+8)
- Continue the gfx950 FlyDSL MLA 192/128 forward from `dev/xiaoming/flydsl_mla_fwd_gfx950` round 20 (user
  decision after the same-session comparison in `../mla192_flydsl_gfx950_20261109/logs/optimize.md`).
- User target: forward >= 1.40x aiter. Re-measure aiter once, then iterate (no per-round aiter re-run).
- Workspace mode: kernel in `src/branch_mla/`, A/B harness `ab_bench.py` (cand vs last accepted snapshot,
  alternating), sustained probe `power_probe.py` (amd-smi GPU 3 = HIP device 0), `prof.py` for rocprofv3.
- Tips read: `agent/historical_experience/gfx950/flash_attn_mla/flydsl/tips.md` and the source campaign's
  `gfx950/flash_attn_fwd_mla/flydsl/tips.md` (on the source branch).

## Round 1 — BASELINE — 15:06
R20 1098.25 vs aiter 1005.24 geomean (1.0925x), 11/11 PASS. Sustained B1 S16K: R20 1899 MHz / 1278 W /
PPT 0.05 / 0.620 TFLOPS/MHz; aiter 1266 MHz / 1332 W / PPT 0.30 / 0.827. ATT (dsv3): loop MFMA busy
~71 % vs ~92 %, R20 waves stall at MFMA issue (75K vs 17K per 1e8 FLOP) because SIMD-sharing waves
w / w + 4 run in phase.

## Round 2 — ROLLED BACK — 15:20
32x32x16 QK / P*V: -6.33 % geomean. Sustained B1 S16K 1559 MHz at the same 1278 W (0.682 TFLOPS/MHz):
-9 % cycles, -18 % clock. Correction to round 1: the node is power-limited (a ~1278 W limit that does
not show as PPT), not cycle-bound; throughput = power / energy per FLOP, as in the source campaign.

### Feasibility note — 15:25
Needed for 1.40x: ~1466 TFLOPS at B1 S16K within ~1278 W, i.e. dynamic energy per FLOP down ~20 %
(R20 ~0.81 pJ/FLOP, of which ~0.47 is 16x16x32 MFMA, so non-MFMA energy roughly halved). The measured
bounds of the known levers (source campaign, wrong-result probes): no main-loop K/V DMA +11 %, no exp
+6.7 %, LDS reads halved +9 % (needs 64 q rows per wave, one wave per SIMD: a kernel rewrite). Even
removing all of them only roughly reaches the target, so 1.40x is not expected from incremental rounds.

### Upper-bound probes on n14-17 — 15:40 (UTC+8)
Wrong-result variants of R20 (`profiles/make_fakes.py`, `profiles/fakes/fakes_probe.log`), sustained 10 s,
R20 run first and last:

| variant | B1 S16K TFLOPS | clock / power / PPT | dsv3 TFLOPS | clock / power / PPT |
|---|---|---|---|---|
| R20 (mean of 2) | 1175.0 | 1900 MHz / 1280 W / 0.08 | 1010.5 | 1999 MHz / 1323 W / 0.18 |
| exp2 -> one FMA | 1267.9 (+7.9 %) | 2125 / 1265 / 0.02 | 1057.4 (+4.6 %) | 2194 / 1292 / 0.06 |
| no main-loop K/V DMA | 1260.6 (+7.3 %) | 1956 / 1218 / 0.00 | 1048.4 (+3.8 %) | 2032 / 1222 / 0.00 |
| both | 1349.4 (+14.8 %) | 2186 / 1205 / 0.00 | 1109.3 (+9.8 %) | 2254 / 1207 / 0.00 |

- Even with neither exp nor main-loop K/V traffic the kernel reaches 1.29x aiter at B1 S16K and 1.15x at
  dsv3 (~1.22x on the full-set geomean). At that point it is no longer power-limited (PPT 0, 1205 W) but
  clock-capped (~2190-2250 MHz) and cycle-bound at the same 0.62 TFLOPS/MHz.
- 1.40x therefore needs both ~-20 % energy per FLOP (about what deleting exp and DMA gives) and
  ~+10 % work per cycle. With this kernel structure neither is available from known single changes;
  the remaining measured lever is LDS bytes per FLOP (64 q rows per wave, one wave per SIMD: rewrite).

### ldshalf upper-bound probe on n14-17 — 18:55 (UTC+8)
Wrong-result R20 variant, odd K slots / odd V steps reuse the previous packs (half the ds_read bytes,
same MFMAs), 10 s sustained (`profiles/fakes/fakes_probe.log`):
B1 S16K 1287.5 vs 1180.0 (+9.1 %, 1943 vs 1898 MHz, 0.663 vs 0.622 TFLOPS/MHz, same 1285 W);
dsv3 1091.6 vs 1010.8 (+8.0 %, 2019 vs 2000 MHz, 0.541 vs 0.505). Mostly work per cycle, not energy.

## Round 3 — ACCEPTED — 18:50
QT4 rewrite (64 q rows per wave, one wave per SIMD, 4 waves x 64 rows): geomean 1129.25 vs R20 1095.84
in the same session (+3.05 %), 11/11 bitwise identical. Long S +5.8 to +7.3 %, 4K shapes +0.1 to +2.1 %.
- Sub-steps on B1 S16K: layout 1096 -> DMA per-copy thunks interleaved in the P*V site 1185.8 ->
  slot-fenced QK with softmax finish spread over the slots 1221.9 -> interleave hints + running-max
  fold 1239.0 -> early cluster sync + V head reads inside the QK 1245.6.
- Power: B1 S16K 1972 MHz / 1281 W / 0.634 TFLOPS/MHz (R20 1911 / 1291 / 0.619); dsv3 2053 MHz /
  1277 W / 0.495 (R20 2000 / 1323 / 0.505), i.e. dsv3 is now near the clock cap, not power.
- ATT dsv3: prologue 11.6 % (Q load issue stalls), loop 72.3 %, drain + epilogue 16.1 % (barrier stalls
  from causal diagonal imbalance, store stalls). One wave per SIMD hides none of it; ldshalf's +8 % on
  dsv3 shows how much of the LDS lever the fixed cost is eating at 4K.
- Loop: 6855 cycles per tile vs 5376 MFMA cycles; non-MFMA issue ~3978 cycles vs ~4032 free slots in
  the MFMA shadows, so the loop is issue-slot bound.
- Inert / failed inside the round: `llvm_options` scheduler knobs; IGroupLP `sched_group_barrier`
  (cannot see the asm P*V MFMAs); deferring the rescale; seed dedupe (512 VGPR, spills); spread V head
  reads (-0.3 %); Q loads before the K0 wait (-0.4 %); row sum on `v_dot2c_f32_bf16` (-0.5 to -2.4 %,
  shares the MFMA pipe); row sum on `v_pk_add_f32` (-1.5 %, spills, not bitwise).

## Round 4 — ACCEPTED — 19:10
`s_setprio` off for 64 rows per wave: +0.26 % geomean (1133.69 vs 1130.79), bitwise 11/11.

## Round 5 — ROLLED BACK — 19:35
Merging each P*V DMA slot with a softmax piece (to fill the M0 -> LDS-DMA hazard nop the recognizer
inserts because inline asm counts as no wait state): -1.73 %. A DMA slot already fills the ~12-cycle
MFMA shadow; the extra piece overflows it. Next: ATT per-slot bubble map.

## Round 6 — ACCEPTED (trade-off) — 20:05
ATT bubble map (`wip/att_bubbles.py`): R4 loses 1214 cycles/iteration to MFMA bubbles (399 issue
overflow, 815 stall) with 1355 spare cycles elsewhere. Spreading the P*V-site V reads one d chunk per
slot (+ two-sided `sched_barrier(0)` fences: the scheduler otherwise hoists a slot's VALU above its
asm MFMA into the previous shadow): +0.10 % geomean, 4K +0.3-0.6 %, long S -0.3-0.7 % (B1 S16K +0.5 %
work/cycle, -0.8 % clock at 1274 W). Kept for exact slot placement.

## Round 7 — ACCEPTED — 20:20
C5's K head read moved into the C3 P*V site (slot 56): +0.34 % geomean, all shapes up, bitwise.

## Round 8 — ROLLED BACK — 20:40
C1's K head carried over the back edge from C7: -0.25 %. The C1-side wait is small and the extra
live VGPRs add AGPR spill reads. Register pressure keeps showing up (R5, R6, R8).

## Integration — 2026-11-11 08:30 (UTC+8)
Round 7 merged into `primus_turbo/flydsl/attention/gfx950_mla/` on `dev/xiaoming/flydsl_mla_fwd_gfx950`
(`attn_helper_mla.py`, `flash_attn_fwd_mla.py`, `ops.py`; `CODEGEN_REV` 1044). ruff 0.15.12 format +
check (import order fix only; AST otherwise identical to the round-7 snapshot). In-tree package vs the
round-7 snapshot, full set: 11/11 PASS, bitwise 11/11, geomean 1136.30 vs 1136.28. Committed records
keep only the forward files in `rounds/*/kernel_snapshot/branch_mla/`; copy `flash_attn_bwd_mla.py` and
`ops_bwd.py` (unchanged) from `gfx950_mla/` next to them to load a snapshot as a package.
Non-causal is wrong on both R20 and round 7 (`wip/noncausal_check.py`, fp32 reference: o rel 0.22-0.42,
LSE max 0.07-0.18 at B1 S1K H4 / B2 S2K H8 / B1 S4K H2; the same script with `CAUSAL=1` gives 2.5e-3).
Pre-existing on the branch and outside this campaign's causal-only validation set; not addressed here.

## Round 9 — ROLLED BACK — 08:45
Persistent job loop (one CTA per CU, host LPT replay of the per-XCD dispatch, shape-cached table):
-0.26 % geomean (kimik2 -2.5 %), bitwise 11/11. ISA: spill slots 8 -> 77. LICM hoisted the job-
invariant lane terms of the Q / K / V global offsets out of the job loop; they spill across the KV
loop and each job reloads ~40 dwordx2, partly behind `s_waitcnt vmcnt(0)` ahead of the first DMA.

## Round 10 — ACCEPTED — 09:20
Round 9 + lane terms re-derived per job from an opaque `tid` (`_pin_regs` in `bind_item`): spill
slots 9, +2.62 % geomean (1168.05 vs 1138.20), 4K +4.2-4.5 %, B1 S64K -0.34 %, bitwise 11/11, edge
smoke 8/8 bitwise. `persistent=False` vs round 7: 1.0000, bitwise (the kept path is neutral).
1.162x aiter. Next: overlap the next job's K / V DMAs and Q load with the current job's drain.

## Round 11 — ACCEPTED — 09:55
Persistent drain C10 issues the next job's K tiles 0 / 1 (buffers free after C9); the prologue drops
its own two K DMAs (`prefetch_k_tile`, `_bind_lanes`): +0.32 % geomean (1172.27 vs 1168.53), repeat
+0.29 %, bitwise 11/11, edge smoke 8/8, spill slots 9. 4K only +0.1-0.2 % (the prologue `vmcnt(0)`
still waits for the previous job's O stores); B1 S64K +1.7 % / B2 S16K -0.35 % are a KV-loop register
allocation shift, not the prefetch. `persistent=False` ISA identical to round 10. 1.166x aiter.
Next: wait only for the prefetch in the prologue (`vmcnt(<O stores>)`), leaving the O stores in flight.

## Round 12 — ACCEPTED (trade-off) — 10:30
The whole lane / wave layout (`_bind_lane_layout`: wave_q_offset, LDS read bases, PV16 lane terms)
re-derived per job from the opaque tid: spill slots 9 -> 3 (H=128: 11 -> 3), no scratch reload left in
the prologue. +0.36 % geomean (1175.99 vs 1171.82), bitwise 11/11, smoke 8/8. H=128 binary: S4K
+1.6-1.9 %, S8K +1.8 %, S64K -2.4 % (loop register allocation: the parked K / V DMA voffset AGPR reads
split 8 + 8 instead of one block); kimik2 -0.5 %, kimik3 -1.0 % (other binaries). `persistent=False`
ISA identical. 1.170x aiter. Next: one DMA voffset VGPR per chunk pattern (row rep in soffset, d in
the immediate) to take the voffsets out of the loop's register budget.

## Round 13 — ACCEPTED — 11:05
Correction to round 12: the loop's parked AGPR values are the V buffer-1 `ds_read_b64_tr_b16`
addresses, not DMA voffsets. V buffer 1 starts at byte 65536, past the 16-bit DS offset field, so
none of its reads folded an offset (96 / 192 transposed reads had one). An opaque base for buffer 1's
two read bases (`v16_read_off`, `_pin_regs`): 186 / 192 fold, VGPRs 512 -> 497, spills 3 -> 0, loop
AGPR reads 16 -> 0. +1.80 % geomean (1196.85 vs 1175.70), every shape up (+1.1-3.3 %), bitwise 11/11,
smoke 8/8. `persistent=False`: +1.57 %, bitwise. 1.191x aiter. Next: leave V0 in flight before QK0.

## Round 14 — ACCEPTED (trade-off) — 10:05
Note: the clock times on the round 10-13 entries above are wrong (ahead of the real clock). From the
artifact times, round 10 finished ~08:55, 11 ~09:30, 12 ~09:45, 13 ~09:55. The round summaries'
headers were corrected.
Persistent prologue: dropped the pre-QK0 `vmcnt(VM_DRAIN_V)`. K1 has been prefetched since round 11,
so the wait only held half of V0, and V0 is first read after the next cluster's drain. +0.06 % /
+0.20 % geomean in two processes, bitwise. Short-job shapes +0.12-0.28 % in both runs; B1 S16K -0.2 %
in both runs; B1 S64K bistable (-0.4 % / +1.2 %). Non-persistent ISA identical. Next: job-top
`vmcnt(16)` so the O stores drain under the Q loads.

## Round 15 — ACCEPTED (trade-off) — 10:15
Persistent job top: `vmcnt(o_stores_per_lane)` (16) + `lgkmcnt(0)` instead of `s_waitcnt 0`, so the
previous job's O stores drain under the Q loads (the first job waits `s_waitcnt 0` before the loop).
+0.09 % / +0.10 % geomean in two processes, bitwise, per-shape deltas reproduce. H16 +0.2 %, H64
+0.4 %, long S +0.1-0.8 %; H128 4K -0.10 to -0.20 % in both runs (not explained; no counter run).
The O stores were mostly off the critical path. Non-persistent ISA identical. 1.192x aiter.
Next: the next job's Q in the drain, or back to the KV loop (round 8 with 15 VGPRs free).

## Round 16 — ACCEPTED (trade-off) — 11:05
`_store_final_o16x4` wrote 64 q rows x 16 B per `buffer_store` (64 lines per instruction; the
`nostore` fake bounded it at +5.1 % on dsv3). Now each wave writes its O rows to its own LDS tile
(272 B rows), reads back 16 B per lane and stores four whole rows per instruction, still 16 stores
per lane, no barrier. +0.92 % / +1.02 % geomean in two processes, bitwise, smoke 8/8. 4K shapes
+1.8-2.3 %; B1 S64K -1.2 % / -1.3 %, S32K -0.4 % in both runs: identical loop IR, but the QK halves
reschedule (V head ds_reads behind two MFMAs) at the lower register pressure (497 -> 488 VGPRs).
`persistent=False`: +1.55 %, every shape up (S64K +0.4 %). LDS 154368 B. 1.203x aiter.
Inside the round: staging lane terms from the job's own lane values cost 14 VGPRs + loop AGPR reads
(CSE across the KV loop), fixed with a fresh `_job_tid()`; 16 `i * stride` products hoisted into
SGPRs, fixed with a running sum. Sustained 10 s loops (`profiles/power_r16.log`) show no S64K loss
(+0.3 % / -0.3 %, R16 15-25 W lower), so the -1.3 % is ab_bench's short S64K windows (3 calls).
Next: balance the causal diagonal across waves (interleave the 16-row q tiles).

## Round 17 — ACCEPTED — 11:25
Persistent: the next job's 24 Q loads issue in drain C10 right after its K prefetch (this job's last
QK was C9) and the raw packs ride the job loop's iter_args; the first job's Q loads before the loop.
`load_all` split into `bind_q_rows` / `load_q_raw` / `assemble_q`, plus `prefetch_q_raw`. Job-top
`vmcnt(16)` now also retires the Q loads; ISA has no wait between the loads and it (half land in
AGPRs directly). +0.36 % / +0.35 % geomean in two processes, bitwise, no shape down, 4K +0.4-0.7 %.
`noq` fake (+8 % dsv3, +3.2 % S64K) overstated it: constant Q also removes the scale and MFMA
switching power. Non-persistent ISA identical. 1.208x aiter.
Next: measure the drain's barrier wait on dsv3 (ATT) before attempting the q-tile interleave.

## Between rounds 17 and 18 — HipKittens survey, ATT, fakes — 12:00
Read HK's gfx950 attention (`dev/mi455x_v4` in Primus-Turbo: `hk_attn_fwd_d128_gfx950.cu` etc.).
Not applicable: fixed zero max (overflows above ~88 logits), third K/V buffer (LDS 148 / 160 KB),
V read swizzle (ATT: LDS bank conflicts 1.2 % of LDS-active cycles). ATT on R17 dsv3: loop ~71 % of
cycles, MFMA ~84 % busy in it; `s_nop` 2.64 % (MFMA SrcC WAR, cvt RAW, M0 before DMA); drain ~31K
cycles per job per wave regardless of job length; C10 (next job's Q gathers) 5.8-7K of it, TA bound
(16 rows x 16 B per quarter-wave). Fakes: `coalq` +1.5 % quick geomean (+2.8 % dsv3), `warmq`
+0.8 %, `warmqh` +0.9 %: the access pattern costs, not the miss.

## Round 18 — ACCEPTED — 12:15
Persistent: the next job's paired-k-step Q (d blocks 0 and 2, 16 of 24 packs) goes by 16 LDS DMAs
per wave into the wave's O stage tile at C10 (rotation swizzle, conflict-free read back), read back
in the epilogue after `vmcnt(8)` and before the O stage; the middle block stays 8 gathers. ISA:
prefetch block 12 K + 16 Q DMAs and 8 gathers, VGPR 490 -> 492. +0.45 % / +0.68 % geomean in two
processes (quick run 3 +0.88 %), bitwise, smoke 8/8, no shape down; 4K +0.2-1.2 %. ~1/4 of the
`coalq` bound on dsv3. S64K bistable (+0.2 / +2.7 / +2.8 %). Non-persistent ISA identical.
1.210x aiter.
Next: issue the Q stage DMA earlier than C10 (it holds no registers; the O stage is free all job).

## Round 19 — ACCEPTED — 12:32
R18 ATT: C10 still 4.5-5.1K cycles per job (12 K + 16 Q DMAs + 8 gathers, VMEM queue full); fakes
`nostage` +2.6 % dsv3, `noqc10` +3.3 % (stale-LDS Q, may overstate). In drain C9 only the last
wave's QK is live and the others wait ~2.7K at the barrier, and the staged Q holds no registers,
so `stage_q_dma` now runs at the start of C9 on waves 0-2 (own tiles; waves 0 / 1 split the last
wave's). C10's `vmcnt(0)` + sync publish it; the epilogue wait is gone. +1.05 % / +1.14 % geomean
in two processes, bitwise, smoke 8/8, every shape up; 4K +0.9-1.6 %. ATT: drain -2.1K cycles per
job, C10 -> 2.2-2.8K. VGPR 492 -> 508 (loop rescheduled). Non-persistent ISA identical. 1.222x
aiter.
Next: the next job's K tile 0 (K buf0 free after C5) from the off-diagonal waves too.

## Round 20 — ROLLBACK — 12:53
R19 ATT: job top 4.3K cycles per job, ~2.0K of it Q scaling VALU (~500 instructions) that QK0
waits on. Moved the scale to the previous job's epilogue right after the O store (same block,
scheduler's choice), carrying scaled AGPR packs over the job edge. -0.36 % / -0.32 % geomean,
bitwise. ATT: job top -2.06K, O store +1.96K per job: no overlap. With one wave per SIMD a
stalled ds_write / buffer_store blocks the VALU behind it. Rolled back to R19.
Next: interleave the scale with QK0's MFMAs at the job top (sched_group_barrier pairs).

## Round 21 — ROLLBACK
Scaled the job's Q under QK0's MFMAs (k-step 0 at the job top, k-step ks+1 inside k-step ks's
slot with (MFMA, 5 VALU) interleave). Bitwise; -0.27 % / -0.35 % geomean, S64K -2.4 / -2.9 %.
ATT: job top 4286 -> 4036 per job. The MFMAs are hidden but VALU-issue bound now: 8 cycles per
post-MFMA slot, and v_pk_mul_f32 stalls 15-20 cycles within 3 instructions of an MFMA. The KV
loop's slot pattern was perturbed: only R19 has every slot `RR w (M e)x4 MMMM`, and S64K follows it
(R19 ~1365 vs ~1330 otherwise). Rolled back to R19.
Next: pin the full loop slot pattern (K reads, pairs, remaining MFMAs).

## Round 22 — ROLLBACK
Pinned the whole KV-loop QK slot (2 K reads, 4 (MFMA, EXP) pairs, 4 MFMAs) via `_qk16_slot`.
R19's slots 0-6 unchanged and the last exp slot regularized: +0.003 % / -0.03 %, bitwise, so it
does not improve on R19. Kept on record (snapshot) as a neutral pin for later rounds that perturb
the loop. Third consecutive rollback: escalate to ISA-level comparison / drain balance.

## Round 23 — ROLLBACK
Drain balance. Interleaved q rows across waves (`ROW_INTERLEAVE`: q tile qt of wave w = block
rows 64 qt + 16 w). Every wave now has the same diagonal share, and on a full band tile t's dead
q tiles are qt < t on all waves, so `_qk_diag` / `_pv_diag` skip them statically
(MFMAs 1008 -> 768). `block_row0` pulls a sequence-end block back so its band is full; the
shared rows are written twice with identical values. A clipped-band fix-up branch spilled 64
VGPRs at the merge and was dropped. Bitwise, smoke 10/10. ATT: drain 27.8K -> 23.1K per job
(barrier wait 2-8.8K -> 0.2-3.3K). But IGroupLP re-chose 2-3 KV-loop QK slots: S64K -3.1 /
-3.8 %, 4K +0.6-1.0 %, geomean -0.21 % / -0.24 %. Rolled back per Rule 3 (fourth consecutive).
Next: R23 + R22's whole-slot `_qk16_slot` pin (neutral on R19) to keep R19's loop schedule.

## Round 24 — ROLLBACK
R23 + R22's whole-slot QK pin. One hot-block slot stays irregular. -0.17 % / -0.26 %, S64K
-3.2 / -3.6 %, 4K +0.7-1.1 %. New: the S64K loss is clock, not cycles. ATT CU 0 +0.2 % cycles
only. power_probe S64K: R19 / R22 ~1860 MHz at the PPT limit (~1345 W, throttle 0.9). R21 / R23 /
R24 1750-1840 MHz at ~1300 W with PPT throttle 0.2-0.3, i.e. another, unexposed cap that tracks
the loop schedule. The loop IR is identical (R19 vs R23, `wip/ir_loop_cmp.py`). The pre-RA
schedule keeps source order (`dd M x8 e x4`; pins only act post-RA), so the RA can hand an exp a
register freed by its slot's MFMA chain and the post-RA pin becomes unsatisfiable (R24 slot 5).
Fifth consecutive rollback.
Next: emit the QK slot in pin order in the IR (exp j after MFMA j via an after_mfma hook).

## Round 25 — ACCEPTED WITH NOTE
Each pinned SOLO_WAVE QK k-slot is now emitted in the pin's order in the IR (`qk(...,
after_mfma=)`, exp i right after MFMA i), so the RA's in-slot register reuse always agrees with
the post-RA pin. First hot block 8/8 regular slots (R19 7, R24 6), second 7/8; VGPR 499, MFMA 768.
Bitwise, smoke 10/10, non-persistent bitwise vs np19. +0.19 % / +0.27 % geomean vs R19. 4K
+1.2-1.8 % (R23's drain balance comes through). S64K -3.3 / -3.0 %, S32K -0.7 / -1.0 %: power_probe
S64K ~1805 MHz / ~1310 W / PPT 0.37 vs R19 1863 MHz / 1350 W / 0.9, with the same TFLOPS/MHz. So
the clock cap does not follow the QK slot pattern (corrects R24); its trigger is open. New best:
1232.09 = 1.226x aiter. Synced to the fwd worktree.
Next: bisect R19 -> R25 on S64K clock (row mapping only / + block_row0 / + drain skip), and check
the PV / softmax blocks' VALU order.
