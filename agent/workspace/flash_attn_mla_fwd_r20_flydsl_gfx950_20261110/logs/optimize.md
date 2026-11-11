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
