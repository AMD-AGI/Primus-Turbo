# Human hints: gfx1250 FlyDSL flash-attention FORWARD (op-evolve)

Prod shape: b4 s8192 hq32 hkv8 d128 bf16, causal-BR, return_lse. One gfx1250 card.
Baseline: vendored aiter FlyDSL fwd `fmha_fwd_prefill_a16w16_m32x8` (~916-942 TF/s, 2.33-2.40 ms).
Bar: aiter ASM `fmha_bf16_pertokenBf16_hd128_128x256_mask` **1398.67 TF/s** (1.5724 ms). Gap ~1.53x.
Read this file every round. Entries are `fN`. The index comes first. Detail follows.

Source abbreviations used below:
- `ISA` = `output/0925__flydsl/fwd-isa/` (ASM vs FlyDSL ISA diff; `asm/*.s`, `fly/fwd_m32x8_prod_causal_lse.s`; REPORT.md not yet saved -- f4 carries its tables; **nothing measured**)
- `SWEEP` = `output/0923__flydsl/STAGE2-FWD-SWEEP.md` (on-card, 2026-09-23, flydsl 0.3.2)
- `BWD` = `output/0923__flydsl/hint.md` h54/h55 (backward campaign, gfx1250)
- `KYLE` = Kyle's gfx950 results (MI355X, different arch, different kernel)
- `AUDIT` = `output/0925__flydsl/api-audit/` (FlyDSL 0.3.4.1 API stability)
- `K` = `fmha_fwd_prefill_a16w16_m32x8.py` in the op tree (knobs: `DEFAULT_N_BLOCK` :147,
  `N_KV_PP` :150, `MIN_KV_BLK_BYTES` :153, `ENABLE_DEFER_RESCALE` :168, `USE_TDM_LOADER` :180,
  `O_VARIANT` :184 -- line numbers from `fwd341/op_clean`; re-grep in your copy)

---

## DECISION INDEX

Status: ❌ measured-dead · ⚠ conditional · 🔬 predicted-only (ISA reading, no card number) · ✅ open-untried

| # | lever | status | one-line reason | ptr |
|---|---|---|---|---|
| L1 | `O_VARIANT` v3 -> v1 | ⚠ | +1.90% over 101 reps, two sessions, on flydsl **0.3.2**; re-verify on 0.3.4.1 (V1 paths hit `ptr_load` breakage) | f3 |
| L2 | Grid: longest-first causal q-tile order | ✅ | FlyDSL dispatches light->heavy; KYLE +2.34% fwd (gfx950); BWD g40 (k_dq longest-first, round 12) +4.91% prod | f5 |
| L3 | Grid: in-WG pairing t with N-1-t (ASM does this) | 🔬 | equal work per WG + epilogue overlap; est. 2-5% | f5 |
| L4 | Grid: XCD-major (b,kv_head) remap | ⚠ | KYLE +2.81% (dkdv, gfx950); BWD g43 **null** on gfx1250 (one device-wide L2). Cheap; bijection check mandatory | f5 |
| L5 | Grid: causal-aligned q-tile origin | ⚠ | KYLE +0.96% (gfx950), re-verify | f5 |
| L6 | QK(j) / softmax(j-1) software pipelining, 2 S register sets | 🔬 | #1 ISA difference, est. 20-35%; never tried on fwd | f6 |
| L7 | Real LO/HI ping-pong (named barrier + `s_setprio`) | 🔬 | `_named_barrier_pair` is a no-op in the ISA today | f6 |
| L8 | Split barrier (signal early / wait late, work in gap) | ⚠ | ASM does it; BWD S2 measured **null** (gap 1->233 bought nothing, 4 barrier mechanisms refuted) -- bwd-measured, fwd may differ | f7 |
| L9 | `sched_barrier` / `sched_group_barrier` patterns | ⚠ | BWD: 0 wins 5 losses -- bwd-measured, fwd may differ; only as part of L6, never alone | f6 |
| L10 | `n_block` 64 -> 128/256 alone | ❌ | SWEEP: 3.2x / 9.9x slower, VGPR spill (2x584 > 1024). gfx1250, 0.3.2 | f8 |
| L11 | joint (BLOCK_M, n_block) retile | ❌ | SWEEP: shipped point is a local optimum (5 points). gfx1250 | f8 |
| L12 | 4 waves / 1 wave per SIMD | ⚠ | **roadblock, not a result**: V2/V3 managers raise on `num_waves != 8`. Never record "4 waves is slower" | f8 |
| L13 | TDM prefetch depth 3 (`N_KV_PP=3`, `tensor_wait(2)`) | ⚠ | ISA est. 3-10%; BWD prefetch depth/position >2 all lost (-10..-30%) -- bwd-measured, fwd may differ; needs `MIN_KV_BLK_BYTES` lowered | f9 |
| L14 | Deep KV-loop unroll (x2, ASM-style) | ✅ | never tried on fwd; ASM unrolls 2x. Tail must be predicated (spill risk) | f9 |
| L15 | Packed FP32 softmax (`v_pk_fma_f32` exp arg) | 🔬 | ISA est. 2-4% | f10 |
| L16 | `v_pk_add_f32` row-sum | ❌ | KYLE dead on gfx950 -- gfx950-measured, re-verify only paired with L17 | f10 |
| L17 | Per-lane partial row-sum, one cross-lane reduce at end | 🔬 | ISA: FlyDSL permlanes every tile, ASM only in epilogue; est. 1-3% | f10 |
| L18 | WMMA ones-row row-sum | ⚠ | KYLE +1.96% (gfx950 MFMA), re-verify | f10 |
| L19 | raw `v_exp` log2 path | ⚠ | KYLE +0.6% gfx950; FlyDSL already uses exp2 with scale folded into Q | f10 |
| L20 | Rescale: drop branch (`ENABLE_DEFER_RESCALE=False` or select) | 🔬 | branch splits basic blocks; est. 1-3%; lazy-rescale alone dead on gfx950 | f11 |
| L21 | Fixed max `_FMAX0` (no running max) | ⚠ | KYLE +13.5% fwd gfx950; **requires adversarial large-logit test**, see f16 | f11 |
| L22 | lgkmcnt/dscnt drain later; delete hand drain before barrier | ⚠ | KYLE +1.42% / +0.54% (gfx950); gfx1250 has split counters, `rocdl.s_waitcnt` raises | f12 |
| L23 | Delete `sched_barrier` pair around `s_barrier` | ⚠ | KYLE +0.96% gfx950 | f12 |
| L24 | `llvm_options` max-memory-clause + post-misched | ⚠ | KYLE +0.6-0.7% gfx950 | f12 |
| L25 | GQA sharer merge | ❌/⚠ | KYLE +2.04% was via clock; FlyDSL fwd already packs 4 q-heads per KV (its advantage over ASM) | f12 |
| L26 | 4-deep LDS ring + unroll2 | ⚠ | KYLE +0.34% / -2.2% elsewhere | f9 |
| L27 | WMMA operand reuse bits | ❌ | BWD g59 +0.08% (null) -- bwd-measured | f13 |
| L28 | Occupancy forcing / `s_sleep` phase offsets | ❌ | KYLE dead gfx950; BWD single-wave occupancy closed | f13 |
| L29 | Epilogue: TDM O-store overlapping next Q tile | 🔬 | est. <=1-2%; free with L3 | f5 |
| L30 | Static ISA metrics as ranking signal | ❌ | BWD: wrong four times. Rank by card numbers only | f13 |

---

## f1. Round 1 instruction -- do this first

1. Start from the ASM-vs-FlyDSL delta table in f4. Do **not** re-derive it.
2. **Grid layer before body.** Grid changes are cheap, orthogonal, and do not touch VGPR budget.
   In order, each its own A/B arm:
   a. longest-first causal q-tile order (L2): `block_x = grid_x - 1 - block_idx.x` (or the packed-q
      equivalent). Verify the heavy tiles really are high x at causal-BR with GQA packing.
   b. XCD-major (b, kv_head) remap (L4) -- write the mapping as a pure function, then run an
      **offline bijection check** (CPU, enumerate all grid ids, assert every (b, kvh, qtile) hit
      exactly once) before any card run.
   c. causal-aligned q-tile origin (L5).
3. Land L1 (`O_VARIANT=v1`) as its own arm if it compiles on 0.3.4.1; it is the only known win.
4. Only after the grid arms are measured, go to the body (f6 pipelining is the big one).

## f2. Why the gap is efficiency, not structure

Gap is ~1.53x, block-causal extra work is 0.78%; no structural term (unlike bwd's 1.386 k_dq
recompute). Both kernels issue ~the same useful instructions and read LDS at the same rate
(`ISA`). The 1.53x is matrix-pipe occupancy: overlap, sync, prefetch depth, load balance.
FlyDSL fwd is mature vendor code: expect single-digit-percent wins, not integer factors (BWD h55).
0.3.4.1 vs 0.3.2 of the same kernel: 942.3/942.9 vs 938.5/940.1 TF/s, same session (`fwd341/ab_prod.log`) -- the version bump is performance-neutral.

## f3. `O_VARIANT` v1 (L1)

`SWEEP` "O_VARIANT 确认": v3 2.356 ms / 947.8 TF/s, v1 2.313 ms / 965.8 TF/s, 101 reps, sd 1.17-1.21%,
+1.90%, reproduced across two sessions (20-rep run gave +2.01%). Descriptors verified distinct via
`--pmc` (VGPR 224 vs 232). The file's own comment calls v1 "fastest so far". Caveat: measured on
flydsl 0.3.2; `AUDIT/fwd-bufmgr.md` had to patch `ptr_load` to compile the V1 path on 0.3.4.1.
O v1 alone (with V2 TDM loaders) must be compile-checked first.

## f4. ASM-vs-FlyDSL delta (from the ISA diff, per 256 KV per wave)

| item | ASM | FlyDSL |
|---|---|---|
| waves/WG, waves/SIMD | 4, 1 | 8, 2 |
| KV tile per step | 256 (loop unrolled 2x) | 64 |
| VGPR | 1024 (`s_set_vgpr_msb`) | 445 (`waves_per_eu=2`) |
| TDM ops / wait | 4 / `tensorcnt 0x4` | 8 / `tensorcnt 0x0` every iter |
| barriers | 1.5, split with WMMA in gap | 4, signal+wait back to back |
| `s_wait_dscnt` | 16-19 loads kept in flight | drains to 0 |
| softmax vs WMMA | QK(j) interleaved with exp of j-1 | phases fenced by `sched_barrier(0)`, ~250 VALU/trans with no WMMA |
| exp argument | packed `v_pk_fma_f32` | scalar `v_fmamk_f32` |
| row-sum cross-lane | epilogue only | every tile |
| O rescale | every tile, branch-free | deferred (thr 8.0), ballot + `s_cbranch_vccz` |
| `v_nop` / SALU | 13 / 58 | 156 / 188 |
| grid | x = head, t paired with N-1-t (equal work) | x = packed q tile, light->heavy |
| WMMA / LDS / trans | 256 / 256 / 260 | 256 / 256 / 264 (same) |

Not part of the gap: GQA packing (FlyDSL advantage, 4x less LDS fill), LDS read rate, masking
scheme (loop split == in-loop branch), `s_setprio` (neither), reuse bits (neither),
`s_prefetch_inst` (ASM only needs it for its 43 KB loop). WAVE_MODE bit 24 vs 25: unverified.
All ISA percentage estimates below are **predictions**; BWD trap 1 says instruction counts are not costs.

## f5. Grid-layer levers (L2-L5, L29)

- Longest-first (L2): one line. KYLE +2.34% fwd. BWD g40 (`k_dq` longest-first, round 12) gave +4.91% prod
  (`0924__flydsl/facts.md` r12.i2.g40; the "14%" in BWD h55 is ~14% of `k_dq`'s own time, not of the op).
  After g40 the bwd prod grids measured 100% dispatch efficiency (`facts.md` r12.i3.g41), so order pays only on a tail imbalance.
- Pairing (L3): loop (t, N-1-t) inside the WG, halve `grid_x`. Gives equal work and lets the O store of
  tile t overlap tile N-1-t's first loads (L29). More code; do after L2 shows order matters.
- XCD remap (L4): on gfx950 it paid via L2 locality; on gfx1250 BWD g43 found **no** locality effect
  (single device-wide L2). May still pay via balance -- measure, do not assume.
- Any remap: offline bijection check is a gate, not optional. A lost tile is a silent wrong `o`.

## f6. Softmax/WMMA pipelining (L6, L7, L9) -- the big body lever

- (a) Carry previous tile's S (or P + m/l) in `_run_tiles` loop state; emit `_qk_gemm(j)` and
  `_softmax(j-1)` in one region. Register cost: +1 S set (~64 VGPR/wave at n_block 64); check
  budget against 512 (`waves_per_eu=2`). Report VGPR/spill every arm.
- (b) Real ping-pong: `rocdl.s_barrier_signal_var` / named barriers + `rocdl.s_setprio`, drop the
  per-iteration WG barrier that keeps the two waves on a SIMD in phase.
- BWD warning: `sched_barrier(0)` is a BOUNDARY, not a clamp; `sched_group_barrier` went 0/5 on bwd.
  Use it only to express the interleave of (a), and A/B it against (a) without it.
- If (a) compiles to the same ISA, you did nothing (BWD g22/g65: source reorder was ISA-identical).
  Diff the `.s` before spending card time.

## f7. Barriers (L8)

BWD S2 (commit 56cab2b0): split barrier with prefetch in a 1->233 instruction gap was a null; four
barrier mechanisms, four refutations. Forward has 4 barriers per 256 KV vs 1.5 in ASM, so it may
differ -- but try it only after L6, since the gap needs independent work to fill.

## f8. Retiling is closed (L10-L12)

- `n_block` 128: 7.741 ms (0.310x); 256: 23.753 ms (0.101x). Cause: VGPR, 2 x 584 > 1024 -> scratch.
  Note `n_block` is a def-time keyword default -- changing the module constant alone does nothing
  (`_ensure_bshd_kernel` does not pass it through). SWEEP found this the hard way.
- Joint (BLOCK_M, n_block), 5 points: shipped (256, 64) is a local optimum; halving BLOCK_M cuts the
  n_block 128 penalty to 4.7% but loses overall (R=1 n=128: 0.718x).
- n_block >= 128 is only viable combined with something that frees VGPRs; do not retry alone.
- 4-wave: blocked by V2/V3 managers raising on `num_waves != 8` and V1 loader not compiling
  (fixed only by patch). Record as ROADBLOCK. BWD h53: its own 4-wave BLOCK_KV=128 cost 45% for
  unexplained reasons -- a 4-wave fwd is a multi-round project, not a knob.

## f9. Prefetch depth and unroll (L13, L14, L26)

- `N_KV_PP=3` + `tensor_wait(2)`: generalize the hard-coded `% 2` and curr/next rotation; lower
  `MIN_KV_BLK_BYTES` (64-row tile is 16 KB but slots are floored to 64+64 KB; 3 slots do not fit 320 KB).
- BWD: every prefetch arm beyond depth 2 lost 10-30% -- but bwd staging was dual-use; fwd TDM is not.
- Deep unroll x2 of the KV loop: never tried. The remainder MUST be a predicated/masked iteration,
  not a separate tail loop (KYLE: tail loop spilled, -19%; spill can wedge the card, f14).

## f10. Softmax arithmetic (L15-L19)

- Packed exp argument: `fmath.fma` on `vector<2xf32>` so LLVM picks `v_pk_fma_f32`.
- Row-sum: per-lane partial across tiles, one cross-lane `peer()` before normalization/LSE (valid:
  m is already reduced across lane pairs). `v_pk_add_f32` row-sum alone was dead on gfx950.
- `v_nop` after each `v_exp` (156/256KV) is a dependency stall; it disappears only with L6 interleave.

## f11. Rescale and max (L20, L21)

- Branch-free rescale: removes the `s_cbranch_vccz` block split. Lazy-rescale alone was dead on gfx950.
- Fixed-max `_FMAX0`: KYLE +13.5% fwd on gfx950, the biggest single number on record -- and the most
  dangerous. Mandatory gate in f16.

## f12. Wait/barrier hygiene and compiler flags (L22-L25)

gfx950 wins, all small, all need re-measure here: lgkmcnt drain later (+1.42%), delete hand drain
before barrier (+0.54%), delete `sched_barrier` pair around `s_barrier` (+0.96%), `llvm_options`
max-memory-clause + post-misched (+0.6-0.7%). On gfx1250 `rocdl.s_waitcnt` raises (split counters);
use `fx.barrier()` and let the backend derive dscnt. GQA sharer merge: already packed in this kernel.

## f13. Closed, do not rebuild (L27, L28, L30)

WMMA reuse bits (bwd g59 +0.08%), forcing occupancy / `s_sleep` offsets (gfx950 dead), static ISA
metrics as a ranking signal (bwd: wrong 4 times; "improved every metric" arms lost hardest).
BWD closed axes (compute ILP, issue roof, LDS port pressure) are bwd-measured; fwd ISA reading
agrees LDS rate is not the gap.

---

## OPERATIONAL RULES

## f14. Card safety

- **One shape per process.** Never run two benchmark processes on the card at once.
- `rocm-smi --showpids` before trusting any number; a stray process (e.g. an abandoned scoring
  session, commit 6029b430) contends silently. Kill only by explicit PID, never by name/pattern.
- Spill can wedge the card; a wedge costs a human AC power cycle. Check `scratch=0` / spill 0/0 in
  the compile output before any card run. Remainders: predicated, never a tail loop.
- Compile-only first (`COMPILE_ONLY=1 ARCH=gfx1250`), diff the `.s`, then run.

## f15. Measurement discipline

- Record `sclk` on every A/B (prod runs sit at ~1042-1049 MHz; a drop invalidates the pair).
- Only same-session, interleaved A/B counts. Cross-session drift of the same code is ~1.5%.
- Noise floor: per-rep sd 1.17-1.39% (101 reps); median-of-101 SE is a few tenths of a percent;
  `min_gain` = 0.005. Claims under ~0.5% are noise. Claim with n >= 101.
- A probe that reports success may not have run (bwd `pw.sh` trap): assert the subject executed and
  that the arm's descriptor (VGPR/SGPR) differs from the control.
- A number from another arch/kernel is not a property of the mechanism (bwd trap 2). Every
  KYLE/BWD row above is a prior, not a result.

## f16. Correctness gates

- Fixed-max / defer-rescale / any change to max tracking: add an **adversarial large-logit test**
  (e.g. scores ~ +-1e4 after scale, one huge outlier per row, rows where the first tile's max is far
  below a later tile's) and compare o and lse against fp32 reference. Pass on random inputs means nothing.
- LSE stays **natural log** and must feed the backward unchanged (bwd consumes it). If you move to
  log2 internally, convert once at the end; test lse against reference, not just o.
- o and lse must be **bitwise deterministic** run to run. No atomics, no split-k.
- Keep the agree-dB vs ASM in the log (currently 49.93 dB); a drop is a correctness signal.

## f17. FlyDSL version and API policy

- FlyDSL **0.3.4.1** at `~/.local/flydsl0341` (prepend to sys.path). 0.3.2 and 0.3.4.1 are perf-equal (f2).
- Prefer stable `fx.*` APIs (catalog: FlyDSL `scripts/list_stable_apis.py`, `docs/api_stability.md`).
  New code must not add new `_`-private / `flydsl._mlir` usages when a stable API exists.
  See `AUDIT/fwd-kernel.md`, `fwd-bufmgr.md`, `fwd-helpers.md` for what is already stable and
  what (gfx1250 WMMA, tr16 loads, TDM, sched_barrier, `rocdl.exp2`) has no stable wrapper yet.
- Known traps: TDM needs aiter's `tdm_ops_gfx1250` shim with explicit `pad_interval`/`pad_amount`
  (else 256 B row stride, 64-way bank conflict); `BufferAtomicAdd` emits SCOPE_CU (not used in fwd;
  keep it that way); `s_barrier_signal`/`s_barrier_wait` live in `flydsl._mlir.dialects.rocdl`.
- Skill: `flydsl-gfx1250` in the FlyDSL repo (`/home/lihuzhan/code/2026_0925__flydsl/FlyDSL`).

---

## REPORT FORMAT (every round)

```
kb_read:  [fN, fM, ...]  -- which entries you read and used
tried:
  - H1 <lever Lx>: <one-line change>  VGPR/SGPR/spill  ->  <ms>, <TF/s>, <delta% vs control>,
       n=<reps>, sclk=<a,b>, agree_dB=<x>, lse ok?, bitwise ok?
  - H2 ...
  (every hypothesis gets a measured number or the reason it never ran: compile fail / roadblock)
kb_check:
  - "hint says <X> (fN/Lx), measured <Y>"  -- one line per index row you touched;
    say explicitly when a row should change status (e.g. L4 ⚠ -> ❌ on gfx1250).
```
