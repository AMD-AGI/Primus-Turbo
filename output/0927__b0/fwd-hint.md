# Human hints: gfx1250 FlyDSL flash-attention FORWARD (op-evolve)

Prod shape: b4 s8192 hq32 hkv8 d128 bf16, causal-BR, return_lse. One gfx1250 card.
Baseline: vendored aiter FlyDSL fwd `fmha_fwd_prefill_a16w16_m32x8` (~916-942 TF/s, 2.33-2.40 ms).
Bar: aiter ASM `fmha_bf16_pertokenBf16_hd128_128x256_mask` **1398.67 TF/s** (1.5724 ms). Gap ~1.53x.
Read this file every round. Entries are `hN`. The index comes first. Detail follows.

Source abbreviations used below:
- `ISA` = `output/0925__flydsl/fwd-isa/` (ASM vs FlyDSL ISA diff; `asm/*.s`, `fly/fwd_m32x8_prod_causal_lse.s`; REPORT.md = full ISA diff; **nothing measured**)
- `SWEEP` = `output/0923__flydsl/STAGE2-FWD-SWEEP.md` (on-card, 2026-09-23, flydsl 0.3.2)
- `BWD` = `output/0923__flydsl/hint.md` h54/h55 (backward campaign, gfx1250)
- `KYLE` = Kyle's gfx950 results (MI355X, different arch, different kernel)
- `AUDIT` = `output/0925__flydsl/api-audit/` (FlyDSL 0.3.4.1 API stability)
- `K` = `fmha_fwd_prefill_a16w16_m32x8.py` in the op tree (knobs: `DEFAULT_N_BLOCK` :147,
  `N_KV_PP` :150, `MIN_KV_BLK_BYTES` :153, `ENABLE_DEFER_RESCALE` :168, `USE_TDM_LOADER` :180,
  `O_VARIANT` :184 -- line numbers from `fwd341/op_clean`; re-grep in your copy)

---

| id | type | title | status |
| --- | --- | --- | --- |
| h1 | must note | Round 1 instruction -- do this first | done r1-r2 (grid first: longest-first landed) |
| h2 | standing note | Why the gap is efficiency, not structure | open |
| h3 | advise note | `O_VARIANT` v1 (L1) | open |
| h4 | standing note | ASM-vs-FlyDSL delta (from the ISA diff, per 256 KV per wave) | open |
| h5 | advise note | Grid-layer levers (L2-L5, L29) | open |
| h6 | advise note | Softmax/WMMA pipelining (L6, L7, L9) -- the big body lever | open |
| h7 | standing note | Barriers (L8) | open |
| h8 | standing note | Retiling is closed (L10-L12) | open |
| h9 | advise note | Prefetch depth and unroll (L13, L14, L26) | open |
| h10 | advise note | Softmax arithmetic (L15-L19) | open |
| h11 | advise note | Rescale and max (L20, L21) | open |
| h12 | advise note | Wait/barrier hygiene and compiler flags (L22-L25) | open |
| h13 | standing note | Closed, do not rebuild (L27, L28, L30) | open |
| h14 | must standing note | Card safety | open |
| h15 | must standing note | Measurement discipline | open |
| h16 | must standing note | Correctness gates | open |
| h17 | must standing note | FlyDSL version and API policy | open |
| h18 | standing note | Decision index -- check it before choosing a lever | open |
| h19 | standing note | Report format (every round) | open |
| h20 | must note | Re-land r1.i1.g01 (longest-first dispatch) -- round 1 lost it to two operator-side gate bugs, now fixed | done r2 (accepted, prod 1018) |
| h21 | advise note | L6 QK(i+1)/softmax(i) software pipeline -- compiled, gated, 508 VGPR / 0 spill (proto `pipeline`) | a0-owned (A0 r5: L6 pipeline lost 15% at prod) |
| h22 | must note | L15+L17+L20 softmax arithmetic -- QK->PV serial span -34%, 0 spill (proto `softmax`) | a0-owned (A0 r6: lost 2% at prod) |
| h23 | advise note | L13+L14 TDM prefetch depth 3 and clean-loop unroll x2 -- 4 compiled arms (proto `unroll`) | open (A0 paused at r13 without building it; reclaimed by B0 2026-09-27) |
| h24 | advise note | L8 barrier per 2 KV tiles + split signal/wait -- p22s, 439 VGPR (proto `barriers`) | a0-owned (A0 r7: lost 3.5-6%; barrier ceiling +5%) |
| h25 | advise note | L3+L29 in-WG q-tile pairing -- O-store overlap only, dispatch gain is zero (proto `pairing`) | open (A0 paused at r13 without building it; reclaimed by B0 2026-09-27) |
| h26 | standing note | Closed levers from the 2026-09-27 compile-only prototypes | open |
| h27 | must standing note | Prototypes h21-h25 are based on ROUND 2 -- port their diff, never copy their files over the round-4+ champion | open |
| h28 | must note | A0 results for rounds 5-7 (their fork of this job): three losses and one harness artifact -- read before choosing | open |
| h29 | must standing | B0 host, ONE card of four: physical GPU 0 via container fa-g0 only; other cards are busy by design | open |
| h30 | must standing | Division of labour with A0 and the GPU-2 lab: this fork owns the softmax max/rescale and row-sum structure levers and the 4-wave roadblock | open |
| h31 | must standing note | Rank candidate vs champion in a process WITHOUT beat; beat only in its own process (A0 h28, 2026-09-27) | open |
| h32 | standing note | A0 r8: nodelay (amdgpu-enable-delay-alu=False) +0.7% prod -- A0 re-lands it; do not duplicate, stack on it only if it reaches our champion via a must hint | open |
| h33 | must standing note | GPU-2 lab measured L22/L23/L24/L4/L5: no win (15 arms, all null or loss; itilp -6.6%) -- closed; L12 and L21 now belong to the labs | open |
| h34 | must note | Stack A0 round 11 onto our round-6 champion: nodelay (1 line) + shape-gated BLOCK_M=128 for under-filled grids -- orthogonal to speculative softmax | open |
| h35 | must standing note | RULER ALERT: identical champion code measures bimodally at prod (~1.44 vs ~1.52 ms per process/arm path). Every ranking process must carry an in-process A/A (a byte-identical champion copy); results where A/A is off by > 0.5% are void | done -- superseded by h40 (cause found and fixed) |
| h36 | standing note | A0 fwd job paused at round 13 (~15:30 UTC, machine handover): champion r11 = r4 + nodelay + BLOCK_M=128 gate; h23 and h25 are open to this job again | open |
| h37 | must note | Re-land round 8 arm B ALONE (m16x8 BLOCK_M=128 gate for under-filled grids on top of op/current): fast +29%, prod/proxy ISA byte-identical to r6; rejected only by proxy noise. nodelay (arm A) is dead on this body | open |
| h38 | must standing | DAY 2 CARD MOVE: this job now runs on PHYSICAL GPU 2 via container fa-g2 ONLY (overrides h29 GPU 0 / fa-g0) | open |
| h39 | must refactor | Adopt round 9 arm BG verbatim as op/current (h37 m16x8 gate + r7.i2.g18): beats r6 on every shape in-session; rejected only by best-ever records of unshipped arms | done (r10) |
| h40 | must standing note | RULER FIXED: benchmark.py now times BLOCKED (lead 4 + block 9 per arm, palindromic rounds); the old call-by-call interleave biased prod 3-7% FlyDSL-vs-FlyDSL and ~25% vs ASM | open |
| h41 | standing note | Re-measured under the fixed ruler: r6 vs r4 +4.6% prod (real), L12 4-wave occ2 DEAD (-0.4% prod, -7% proxy), bnegg/fixed-max vs r6 +0.6% (unconfirmed), ASM beat steady-state ~1.43-1.46 ms (r6/beat 1.035) | open |
| h42 | advise standing note | E2E (Llama-3.1-8B training): our fwd runs 1.34-1.52x ASM per layer inside training (1.64-1.85 vs 1.22 ms) and is clock-sensitive (+13% across clock states); ASM is not -- prefer changes that cut work/power, and check candidates at a low clock state too | open |
| h43 | must standing note | Speculative softmax (SPEC_STALE_MAX, r6/g14) LOSES on real training data: 13-25% of tiles recompute (score std 21-53 vs randn 1.0); the randn ruler mis-ranks it. Do not build more speculation variants; an operator nospec refactor is being validated | superseded by h44/h45 |
| h44 | must refactor | Adopt r13ns verbatim as op/current (r13 with SPEC_STALE_MAX=False): real data + training clock 12-23% faster, e2e fwd 53-60 -> 49 ms/step; output bitwise = r13; randn ruler reads it 4.8% slower (ruler bias) | open |
| h45 | must standing note | Speculative softmax is DEAD on real data -- never re-add stale-max / guessed-max / trigger-and-redo; any data-dependent candidate must also beat the champion on the real q/k/v dumps in the same process | open |

---

### h1 -- Round 1 instruction -- do this first

1. Start from the ASM-vs-FlyDSL delta table in h4. Do **not** re-derive it.
2. **Grid layer before body.** Grid changes are cheap, orthogonal, and do not touch VGPR budget.
   In order, each its own A/B arm:
   a. longest-first causal q-tile order (L2): `block_x = grid_x - 1 - block_idx.x` (or the packed-q
      equivalent). Verify the heavy tiles really are high x at causal-BR with GQA packing.
   b. XCD-major (b, kv_head) remap (L4) -- write the mapping as a pure function, then run an
      **offline bijection check** (CPU, enumerate all grid ids, assert every (b, kvh, qtile) hit
      exactly once) before any card run.
   c. causal-aligned q-tile origin (L5).
3. Land L1 (`O_VARIANT=v1`) as its own arm if it compiles on 0.3.4.1; it is the only known win.
4. Only after the grid arms are measured, go to the body (h6 pipelining is the big one).

### h2 -- Why the gap is efficiency, not structure

Gap is ~1.53x, block-causal extra work is 0.78%; no structural term (unlike bwd's 1.386 k_dq
recompute). Both kernels issue ~the same useful instructions and read LDS at the same rate
(`ISA`). The 1.53x is matrix-pipe occupancy: overlap, sync, prefetch depth, load balance.
FlyDSL fwd is mature vendor code: expect single-digit-percent wins, not integer factors (BWD h55).
0.3.4.1 vs 0.3.2 of the same kernel: 942.3/942.9 vs 938.5/940.1 TF/s, same session (`fwd341/ab_prod.log`) -- the version bump is performance-neutral.

### h3 -- `O_VARIANT` v1 (L1)

`SWEEP` "O_VARIANT 确认": v3 2.356 ms / 947.8 TF/s, v1 2.313 ms / 965.8 TF/s, 101 reps, sd 1.17-1.21%,
+1.90%, reproduced across two sessions (20-rep run gave +2.01%). Descriptors verified distinct via
`--pmc` (VGPR 224 vs 232). The file's own comment calls v1 "fastest so far". Caveat: measured on
flydsl 0.3.2; `AUDIT/fwd-bufmgr.md` had to patch `ptr_load` to compile the V1 path on 0.3.4.1.
O v1 alone (with V2 TDM loaders) must be compile-checked first.

### h4 -- ASM-vs-FlyDSL delta (from the ISA diff, per 256 KV per wave)

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

### h5 -- Grid-layer levers (L2-L5, L29)

- Longest-first (L2): one line. KYLE +2.34% fwd. BWD g40 (`k_dq` longest-first, round 12) gave +4.91% prod
  (`0924__flydsl/facts.md` r12.i2.g40; the "14%" in BWD h55 is ~14% of `k_dq`'s own time, not of the op).
  After g40 the bwd prod grids measured 100% dispatch efficiency (`facts.md` r12.i3.g41), so order pays only on a tail imbalance.
- Pairing (L3): loop (t, N-1-t) inside the WG, halve `grid_x`. Gives equal work and lets the O store of
  tile t overlap tile N-1-t's first loads (L29). More code; do after L2 shows order matters.
- XCD remap (L4): on gfx950 it paid via L2 locality; on gfx1250 BWD g43 found **no** locality effect
  (single device-wide L2). May still pay via balance -- measure, do not assume.
- Any remap: offline bijection check is a gate, not optional. A lost tile is a silent wrong `o`.

### h6 -- Softmax/WMMA pipelining (L6, L7, L9) -- the big body lever

- (a) Carry previous tile's S (or P + m/l) in `_run_tiles` loop state; emit `_qk_gemm(j)` and
  `_softmax(j-1)` in one region. Register cost: +1 S set (~64 VGPR/wave at n_block 64); check
  budget against 512 (`waves_per_eu=2`). Report VGPR/spill every arm.
- (b) Real ping-pong: `rocdl.s_barrier_signal_var` / named barriers + `rocdl.s_setprio`, drop the
  per-iteration WG barrier that keeps the two waves on a SIMD in phase.
- BWD warning: `sched_barrier(0)` is a BOUNDARY, not a clamp; `sched_group_barrier` went 0/5 on bwd.
  Use it only to express the interleave of (a), and A/B it against (a) without it.
- If (a) compiles to the same ISA, you did nothing (BWD g22/g65: source reorder was ISA-identical).
  Diff the `.s` before spending card time.

### h7 -- Barriers (L8)

BWD S2 (commit 56cab2b0): split barrier with prefetch in a 1->233 instruction gap was a null; four
barrier mechanisms, four refutations. Forward has 4 barriers per 256 KV vs 1.5 in ASM, so it may
differ -- but try it only after L6, since the gap needs independent work to fill.

### h8 -- Retiling is closed (L10-L12)

- `n_block` 128: 7.741 ms (0.310x); 256: 23.753 ms (0.101x). Cause: VGPR, 2 x 584 > 1024 -> scratch.
  Note `n_block` is a def-time keyword default -- changing the module constant alone does nothing
  (`_ensure_bshd_kernel` does not pass it through). SWEEP found this the hard way.
- Joint (BLOCK_M, n_block), 5 points: shipped (256, 64) is a local optimum; halving BLOCK_M cuts the
  n_block 128 penalty to 4.7% but loses overall (R=1 n=128: 0.718x).
- n_block >= 128 is only viable combined with something that frees VGPRs; do not retry alone.
- 4-wave: blocked by V2/V3 managers raising on `num_waves != 8` and V1 loader not compiling
  (fixed only by patch). Record as ROADBLOCK. BWD h53: its own 4-wave BLOCK_KV=128 cost 45% for
  unexplained reasons -- a 4-wave fwd is a multi-round project, not a knob.

### h9 -- Prefetch depth and unroll (L13, L14, L26)

- `N_KV_PP=3` + `tensor_wait(2)`: generalize the hard-coded `% 2` and curr/next rotation; lower
  `MIN_KV_BLK_BYTES` (64-row tile is 16 KB but slots are floored to 64+64 KB; 3 slots do not fit 320 KB).
- BWD: every prefetch arm beyond depth 2 lost 10-30% -- but bwd staging was dual-use; fwd TDM is not.
- Deep unroll x2 of the KV loop: never tried. The remainder MUST be a predicated/masked iteration,
  not a separate tail loop (KYLE: tail loop spilled, -19%; spill can wedge the card, h14).

### h10 -- Softmax arithmetic (L15-L19)

- Packed exp argument: `fmath.fma` on `vector<2xf32>` so LLVM picks `v_pk_fma_f32`.
- Row-sum: per-lane partial across tiles, one cross-lane `peer()` before normalization/LSE (valid:
  m is already reduced across lane pairs). `v_pk_add_f32` row-sum alone was dead on gfx950.
- `v_nop` after each `v_exp` (156/256KV) is a dependency stall; it disappears only with L6 interleave.

### h11 -- Rescale and max (L20, L21)

- Branch-free rescale: removes the `s_cbranch_vccz` block split. Lazy-rescale alone was dead on gfx950.
- Fixed-max `_FMAX0`: KYLE +13.5% fwd on gfx950, the biggest single number on record -- and the most
  dangerous. Mandatory gate in h16.

### h12 -- Wait/barrier hygiene and compiler flags (L22-L25)

gfx950 wins, all small, all need re-measure here: lgkmcnt drain later (+1.42%), delete hand drain
before barrier (+0.54%), delete `sched_barrier` pair around `s_barrier` (+0.96%), `llvm_options`
max-memory-clause + post-misched (+0.6-0.7%). On gfx1250 `rocdl.s_waitcnt` raises (split counters);
use `fx.barrier()` and let the backend derive dscnt. GQA sharer merge: already packed in this kernel.

### h13 -- Closed, do not rebuild (L27, L28, L30)

WMMA reuse bits (bwd g59 +0.08%), forcing occupancy / `s_sleep` offsets (gfx950 dead), static ISA
metrics as a ranking signal (bwd: wrong 4 times; "improved every metric" arms lost hardest).
BWD closed axes (compute ILP, issue roof, LDS port pressure) are bwd-measured; fwd ISA reading
agrees LDS rate is not the gap.

---


### h14 -- Card safety

- **One shape per process.** Never run two benchmark processes on the card at once.
- `rocm-smi --showpids` before trusting any number; a stray process (e.g. an abandoned scoring
  session, commit 6029b430) contends silently. Kill only by explicit PID, never by name/pattern.
- Spill can wedge the card; a wedge costs a human AC power cycle. Check `scratch=0` / spill 0/0 in
  the compile output before any card run. Remainders: predicated, never a tail loop.
- Compile-only first (`COMPILE_ONLY=1 ARCH=gfx1250`), diff the `.s`, then run.

### h15 -- Measurement discipline

- Record `sclk` on every A/B (prod runs sit at ~1042-1049 MHz; a drop invalidates the pair).
- Only same-session, interleaved A/B counts. Cross-session drift of the same code is ~1.5%.
- Noise floor: per-rep sd 1.17-1.39% (101 reps); median-of-101 SE is a few tenths of a percent;
  `min_gain` = 0.005. Claims under ~0.5% are noise. Claim with n >= 101.
- A probe that reports success may not have run (bwd `pw.sh` trap): assert the subject executed and
  that the arm's descriptor (VGPR/SGPR) differs from the control.
- A number from another arch/kernel is not a property of the mechanism (bwd trap 2). Every
  KYLE/BWD row above is a prior, not a result.

### h16 -- Correctness gates

- Fixed-max / defer-rescale / any change to max tracking: add an **adversarial large-logit test**
  (e.g. scores ~ +-1e4 after scale, one huge outlier per row, rows where the first tile's max is far
  below a later tile's) and compare o and lse against fp32 reference. Pass on random inputs means nothing.
- LSE stays **natural log** and must feed the backward unchanged (bwd consumes it). If you move to
  log2 internally, convert once at the end; test lse against reference, not just o.
- o and lse must be **bitwise deterministic** run to run. No atomics, no split-k.
- Keep the agree-dB vs ASM in the log (currently 49.93 dB); a drop is a correctness signal.

### h17 -- FlyDSL version and API policy

- FlyDSL **0.3.4.1** at `~/.local/flydsl0341` (prepend to sys.path). 0.3.2 and 0.3.4.1 are perf-equal (h2).
- Prefer stable `fx.*` APIs (catalog: FlyDSL `scripts/list_stable_apis.py`, `docs/api_stability.md`).
  New code must not add new `_`-private / `flydsl._mlir` usages when a stable API exists.
  See `AUDIT/fwd-kernel.md`, `fwd-bufmgr.md`, `fwd-helpers.md` for what is already stable and
  what (gfx1250 WMMA, tr16 loads, TDM, sched_barrier, `rocdl.exp2`) has no stable wrapper yet.
- Known traps: TDM needs aiter's `tdm_ops_gfx1250` shim with explicit `pad_interval`/`pad_amount`
  (else 256 B row stride, 64-way bank conflict); `BufferAtomicAdd` emits SCOPE_CU (not used in fwd;
  keep it that way); `s_barrier_signal`/`s_barrier_wait` live in `flydsl._mlir.dialects.rocdl`.
- Skill: `flydsl-gfx1250` in the FlyDSL repo (`/home/lihuzhan/code/2026_0925__flydsl/FlyDSL`).

---

### h19 -- Report format (every round)

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

### h18 -- Decision index -- check it before choosing a lever


Status: ❌ measured-dead · ⚠ conditional · 🔬 predicted-only (ISA reading, no card number) · ✅ open-untried

| # | lever | status | one-line reason | ptr |
|---|---|---|---|---|
| L1 | `O_VARIANT` v3 -> v1 | ⚠ | +1.90% over 101 reps, two sessions, on flydsl **0.3.2**; re-verify on 0.3.4.1 (V1 paths hit `ptr_load` breakage) | h3 |
| L2 | Grid: longest-first causal q-tile order | ✅ | FlyDSL dispatches light->heavy; KYLE +2.34% fwd (gfx950); BWD g40 (k_dq longest-first, round 12) +4.91% prod | h5 |
| L3 | Grid: in-WG pairing t with N-1-t (ASM does this) | 🔬 | equal work per WG + epilogue overlap; est. 2-5% | h5 |
| L4 | Grid: XCD-major (b,kv_head) remap | ⚠ | KYLE +2.81% (dkdv, gfx950); BWD g43 **null** on gfx1250 (one device-wide L2). Cheap; bijection check mandatory | h5 |
| L5 | Grid: causal-aligned q-tile origin | ⚠ | KYLE +0.96% (gfx950), re-verify | h5 |
| L6 | QK(j) / softmax(j-1) software pipelining, 2 S register sets | 🔬 | #1 ISA difference, est. 20-35%; never tried on fwd | h6 |
| L7 | Real LO/HI ping-pong (named barrier + `s_setprio`) | 🔬 | `_named_barrier_pair` is a no-op in the ISA today | h6 |
| L8 | Split barrier (signal early / wait late, work in gap) | ⚠ | ASM does it; BWD S2 measured **null** (gap 1->233 bought nothing, 4 barrier mechanisms refuted) -- bwd-measured, fwd may differ | h7 |
| L9 | `sched_barrier` / `sched_group_barrier` patterns | ⚠ | BWD: 0 wins 5 losses -- bwd-measured, fwd may differ; only as part of L6, never alone | h6 |
| L10 | `n_block` 64 -> 128/256 alone | ❌ | SWEEP: 3.2x / 9.9x slower, VGPR spill (2x584 > 1024). gfx1250, 0.3.2 | h8 |
| L11 | joint (BLOCK_M, n_block) retile | ❌ | SWEEP: shipped point is a local optimum (5 points). gfx1250 | h8 |
| L12 | 4 waves / 1 wave per SIMD | ⚠ | **roadblock, not a result**: V2/V3 managers raise on `num_waves != 8`. Never record "4 waves is slower" | h8 |
| L13 | TDM prefetch depth 3 (`N_KV_PP=3`, `tensor_wait(2)`) | ⚠ | ISA est. 3-10%; BWD prefetch depth/position >2 all lost (-10..-30%) -- bwd-measured, fwd may differ; needs `MIN_KV_BLK_BYTES` lowered | h9 |
| L14 | Deep KV-loop unroll (x2, ASM-style) | ✅ | never tried on fwd; ASM unrolls 2x. Tail must be predicated (spill risk) | h9 |
| L15 | Packed FP32 softmax (`v_pk_fma_f32` exp arg) | 🔬 | ISA est. 2-4% | h10 |
| L16 | `v_pk_add_f32` row-sum | ❌ | KYLE dead on gfx950 -- gfx950-measured, re-verify only paired with L17 | h10 |
| L17 | Per-lane partial row-sum, one cross-lane reduce at end | 🔬 | ISA: FlyDSL permlanes every tile, ASM only in epilogue; est. 1-3% | h10 |
| L18 | WMMA ones-row row-sum | ⚠ | KYLE +1.96% (gfx950 MFMA), re-verify | h10 |
| L19 | raw `v_exp` log2 path | ⚠ | KYLE +0.6% gfx950; FlyDSL already uses exp2 with scale folded into Q | h10 |
| L20 | Rescale: drop branch (`ENABLE_DEFER_RESCALE=False` or select) | 🔬 | branch splits basic blocks; est. 1-3%; lazy-rescale alone dead on gfx950 | h11 |
| L21 | Fixed max `_FMAX0` (no running max) | ⚠ | KYLE +13.5% fwd gfx950; **requires adversarial large-logit test**, see h16 | h11 |
| L22 | lgkmcnt/dscnt drain later; delete hand drain before barrier | ⚠ | KYLE +1.42% / +0.54% (gfx950); gfx1250 has split counters, `rocdl.s_waitcnt` raises | h12 |
| L23 | Delete `sched_barrier` pair around `s_barrier` | ⚠ | KYLE +0.96% gfx950 | h12 |
| L24 | `llvm_options` max-memory-clause + post-misched | ⚠ | KYLE +0.6-0.7% gfx950 | h12 |
| L25 | GQA sharer merge | ❌/⚠ | KYLE +2.04% was via clock; FlyDSL fwd already packs 4 q-heads per KV (its advantage over ASM) | h12 |
| L26 | 4-deep LDS ring + unroll2 | ⚠ | KYLE +0.34% / -2.2% elsewhere | h9 |
| L27 | WMMA operand reuse bits | ❌ | BWD g59 +0.08% (null) -- bwd-measured | h13 |
| L28 | Occupancy forcing / `s_sleep` phase offsets | ❌ | KYLE dead gfx950; BWD single-wave occupancy closed | h13 |
| L29 | Epilogue: TDM O-store overlapping next Q tile | 🔬 | est. <=1-2%; free with L3 | h5 |
| L30 | Static ISA metrics as ranking signal | ❌ | BWD: wrong four times. Rank by card numbers only | h13 |

---


### h20 -- Re-land r1.i1.g01 (longest-first dispatch) -- round 1 lost it to two operator-side gate bugs, now fixed

Round 1's shipped code (`rounds/001/op/`, r1.i1.g01 alone) measured prod 1015.37 TF/s vs champion
935.04 (1.086x) and proxy 1.479x in the same session, and was rejected ONLY because (a) the
precision floor was 50 dB while the unmodified baseline itself reads o 49.82-49.99 dB on 7 edge
cases, and (b) the fast shape's median (launch-bound, moves with arm position after ASM) read 0.934
against a 0.993 band. The operator has fixed both in the resolved spec: `op.precision_sqnr_db: 49`,
`evolve.shape_band: {fast: 0.90}`. Do not rebuild it from scratch: start from `rounds/001/op/`
(diff it against `op/current/`), re-measure, and ship it -- then use the remaining time of the round
for the next lever (h5 grid levers: in-WG t/N-1-t pairing L3, XCD remap L4, causal origin L5).
O_VARIANT v1 (r1.i2.g02) did NOT reproduce under this harness (proxy -3/-4%) -- do not merge it.
Discharge this hint once r1.i1.g01 is accepted.

### h21 -- L6 QK(i+1)/softmax(i) software pipeline -- compiled, gated, 508 VGPR / 0 spill (proto `pipeline`)

- **Path:** `$P=.../proto/pipeline`; diff `pipeline.patch`; review `REPORT.md`; configuration matrix
  `review/run.sh` (`GQA`, `CAUSAL`, `D`, `DT`); LDS drain check `review/dscnt_cfg.py`.
- **What it changes:** S is double-buffered in registers. Iteration i runs QK(i+1) and softmax(i) in one
  region, so the softmax VALU/TRANS of tile i sits under QK(i+1)'s 32 WMMAs. The LDS ring is re-phased
  so V lags K by one tile: a slot holds [K(i+1) | V(i)]. It is still 2 slots with one WG barrier per
  tile. The last tile's softmax+PV is peeled after the loop.
  Toggle `PIPELINE_QK_SOFTMAX` (:192).
  A compile-time gate `_PIPE` (~:757) enables it only for `USE_TDM_LOADER and mask_right and not mask_left
  and gqa_ratio in PIPE_GQA_OK (2,4) and qk_hdim==128 and v_hdim==128`. Every other config compiles to
  the champion body byte for byte.
- **ISA evidence (prod = fast, same ISA):**
  - Resources: VGPR 508/512, SGPR 96, spill 0/0, scratch 0, LDS 327680, 6310 instructions.
  - Clean-LO loop, 694 instructions: WMMA 64, v_exp 66, v_nop 25, other VALU 173, ds_load 64, TDM 2,
    barrier 2, s_wait_dscnt 15, SALU 173, v_mov 39 (32 are S copies), s_set_vgpr_msb 103.
    Clean-HI is 751 (v_nop 54). Masked loops are 950/977.
  - v_exp within 8 instructions after a QK WMMA: 59/66 (clean LO), 54/66 (clean HI), 0/66 (masked).
  - thd causal: 508/0. causal gqa2: 508/0. fp16 causal gqa4: 511/0 (not scored). Non-causal, gqa1 and
    win_sink are byte-identical to the champion.
  - Every `s_barrier_signal` has 0 LDS ops outstanding on every CFG path (dscnt_cfg.py), with
    `s_wait_tensorcnt 0` before each barrier. LO/HI both do 1+n_iter WG barriers, same as the champion.
- **Expected effect:** prod +3 to +12%, point estimate about +6%. Round 3 put exp2 on the critical
  path (13.4%), and about 250 softmax VALU/TRANS per clean tile that overlapped no WMMA are now mostly
  hidden.
  Partly offsetting: +32 v_mov_b64 and about +70 s_set_vgpr_msb per iteration; late V LDS reads on
  LO warps; no overlap in the masked loops; PV(i) still serial after softmax(i).
  fast: flat or slightly negative. Could be near zero.
- **Risks:**
  - 4 VGPRs of headroom. Stacking any other lever on this body needs the full `review/run.sh`
    6-config matrix (gqa 1/2/4 x causal 0/1) first.
  - Before the review gate, non-causal (all gqa) compiled to 512 VGPR / 92 spill / 372 B scratch and
    causal gqa1 to 512 / 85 / 344 B. **Never remove or widen `_PIPE`.**
  - Expected bitwise equal to the champion (the per-tile op sequence is unchanged). Gate:
    - bitwise vs champion on toy (gqa2 takes the pipeline), short sequences n_iter=1/2,
      unequal_seqlen*, short_q, kv_len % 64 != 0;
    - the same shape repeated 80 times (mode-2 history);
    - the 200-run determinism gate.
  - `var/sgbA`, `var/sgbB` and `var/vearly` predate the gate, so **do not copy them**. Their settings are
    toggles on `op/`: `PIPE_SGB_ROUNDS=32` (sgbA), `PIPE_SGB_ROUNDS=32, PIPE_SGB_DSRD=1` (sgbB).
- **How the round starts:**
  1. Copy per the common recipe.
  2. Arm A = `PIPELINE_QK_SOFTMAX=True` (as shipped, `PIPE_SGB_ROUNDS=0`). Control = the same file with
     `False`. Measure prod/proxy/fast.
  3. Only if A wins: optional arm B = A + `PIPE_SGB_ROUNDS=32`. Compile-only first; if the ISA equals A's,
     it is not an arm (h6). sched_group_barrier went 0/5 on bwd, so a low prior.

### h22 -- L15+L17+L20 softmax arithmetic -- QK->PV serial span -34%, 0 spill (proto `softmax`)

- **Path:** `$P=.../proto/softmax`; diff `softmax_proto.diff`; review `REPORT.md`; tools `tools/dyn.py`,
  `tools/crit.py`, `tools/cpu_algebra_check.py` (max rel 4.0e-6).
- **What it changes (toggles at :179-185, read at trace time):**
  - `SOFTMAX_PK_EXP` (L15): the exp2 argument is one v8 fma per sub-tile, emitted as `v_pk_fma_f32`
    instead of `v_fmamk`.
  - `SOFTMAX_LANE_ROWSUM` (L17): per-lane partial row-sum carried across tiles, with one cross-lane
    `peer()` in the epilogue instead of a permlane every tile.
  - `SOFTMAX_PK_ROWSUM`: the in-lane sum as a packed v8 tree (`False` = scalar tree).
  - `SOFTMAX_BRANCHFREE_RESCALE` (L20): no ballot and no branch; `o *= corr` every tile.
  - `SOFTMAX_PROTO` sets all four together.
- **ISA evidence (default PK+LANE(pk)+BF; prod = fast, md5 `6b286ce7`):**
  - Resources: VGPR 446, SGPR 98, spill 0/0, scratch 0, 4070 instructions.
  - LO-clean iteration: 537 instructions in one basic block. WMMA 64, v_exp 66, VALU 278, v_pk_fma 33,
    v_pk_add 32, v_pk_mul 65 (unconditional rescale), permlane 2, v_nop 24, vgpr_msb 47, 3 branches.
    The base is 551 (622 when a rescale fires).
  - QK->PV serial span 337 -> 222 (-34%). LO-masked 496 -> 347 (-30%). Whole-kernel permlanex16 16 -> 10.
  - Ablation, all 0 spill (LO-clean total / serial span):

    | arm | total | serial span |
    | --- | --- | --- |
    | PK only | 477 | 263 |
    | LANE only | 527 | 240 |
    | BF only | 627 | 233 |
    | PK+LANE+defer | 506 | 218 |
    | PK+BF | 583 | 193 |
    | PK+LANE(scalar)+defer | 535 | 220 |
    | PK+LANE(scalar)+BF | 606 | 222 |

  - Masked `v_cndmask ..., 0xff800000` counts equal base: prod 126, non-causal 128, win_sink 256.
- **Expected effect:** prod +2 to +5%. Round 3 showed softmax VALU on the critical path, but 2 waves per
  SIMD hide part of the shorter span.
  BF vs defer: at n_block=64, BF does not shorten the span (222 vs 218) and adds 31 VALU per iteration,
  because it runs 64 `v_pk_mul_f32` every 64 KV (ASM rescales once per 256 KV). BF's only upside is the
  single basic block, with rescale interleaved into PV. Only the card can say.
- **Risks:**
  - Non-causal SGPR spill 2 -> 4 and window+sink 6 -> 35. Both go to VGPR lanes with 0 scratch; neither
    path is scored.
  - PV span has 17 WAR `v_nop`; `s_set_vgpr_msb` 32 -> 47.
  - The L15 fma must stay **without** `fastmath=fast`. With `ninf`, LLVM may fold away the causal -inf
    mask. The review already fixed this with byte-identical ISA; keep it fixed when porting.
  - Any arm that keeps defer (`SOFTMAX_BRANCHFREE_RESCALE=False`) needs the h16 adversarial large-logit
    test for o and lse. o/lse SQNR >= 49 dB on all 3 shapes.
- **How the round starts:**
  1. Copy per the common recipe.
  2. Three separate arms against control `SOFTMAX_PROTO=False` (champion bytes):
     - C1 = default, all True (md5 6b286ce7);
     - C2 = PK+LANE+defer (`SOFTMAX_BRANCHFREE_RESCALE=False`, `ENABLE_DEFER_RESCALE=True`);
     - C3 = PK only (`SOFTMAX_LANE_ROWSUM=False`, `SOFTMAX_BRANCHFREE_RESCALE=False`).
  3. C2 vs C1 is the BF question; C3 isolates L15.
  4. Land the best one. If the winner keeps defer, the large-logit gate is mandatory.

### h23 -- L13+L14 TDM prefetch depth 3 and clean-loop unroll x2 -- 4 compiled arms (proto `unroll`)

- **Path:** `$P=.../proto/unroll`; diff `proto.diff`; logs `logs/`, `review/`; tools `tools/`
  (`compile_isa.py`, `loops.py`, `loopops.py`).
- **What it changes:**
  - `N_KV_PP` (:150; 2 or 3), L13: with 3, tile t+2 is issued at tile t, and the top-of-tile wait only
    drains tile t (`s_wait_tensorcnt 0x2`; 0x0 only on the last tile).
  - `KV_UNROLL` (:164; 1 or 2), L14: the clean loop runs two tiles per `scf.for` iteration. The in-pair
    ring rotation is SSA renaming. An odd clean count is absorbed by the masked (predicated) loop,
    with no tail loop.
  - A local `n_kv_pp` falls back to 2 when 3 slots do not fit (qk_hdim=256, n_block=256) or with the V1
    loader. Slot size is 54272 B (0x1a800).
- **ISA evidence** (all 0 spill, 0 scratch; VGPR 445, LDS 327680; prod = fast):

  | arm | SGPR | kernel instr | per tile | v_nop | SALU | branch | wait |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | U1PP2 (both off) | = champion bytes | | | | | | 0x0 |
  | U2PP2 | 95 | 5556 | 557 | 48.5 | 40 | 4 | 0x0, 0 swap moves |
  | U1PP3 | 104 | 4559 | 573 | 37 | 49 | 7 | 0x2 |
  | U2PP3 (default) | 104 | 5852 | 576.5 | 42 | 51 | 6 | 0x2, 6 moves per pair |

  thd: 445/105, 0 spill. win_sink: 453 VGPR, with SGPR spill down from the champion's 6 to 0.
- **Expected effect:**
  - **U1PP3 vs U1PP2 is the core number of the round.** If the champion really stalls on
    `s_wait_tensorcnt 0` on the card: +2 to 8%. If one tile of compute already hides TDM latency:
    -0.5 to -1.5% (about +15 SALU/move/branch per tile). The static ISA cannot tell which.
  - U2 alone: null (-1 to +1%). It is scaffolding for L6 two-tile interleave and for "N_KV_PP=4 + one
    barrier per pair".
  - U2 hands odd remainders to the masked loop: prod about -0.2%, fast -1 to -2% (inside shape_band 0.90).
- **Risks:**
  - BWD: every prefetch depth > 2 lost 10-30%. That staging was dual-use; fwd TDM is not, so it is only
    a prior.
  - qk_hdim=192 with U2PP3: 8 VGPR spill, 14 SGPR spill, scratch 36. The champion there is 0/8/0, and
    U1PP3 and U2PP2 alone are 0 VGPR spill at d192. **If U2 ships, gate it to qk_hdim==128.**
  - Non-causal SGPR spill 2 -> 6, to VGPR lanes.
  - The CPU schedule proof (`prove_schedule.py`) covers only bshd causal, mask_left=False, sq==skv.
    Add sq!=skv and kv_len % 64 != 0 shapes to the gate.
- **How the round starts:**
  1. Copy per the common recipe.
  2. Arms in one session:
     - A = `N_KV_PP=2, KV_UNROLL=1` (control, champion bytes);
     - C = `N_KV_PP=3, KV_UNROLL=1`;
     - D = `N_KV_PP=3, KV_UNROLL=2`;
     - optional B = `N_KV_PP=2, KV_UNROLL=2`.
  3. Report C vs A first. D only earns its place if D > C by more than the floor.

### h24 -- L8 barrier per 2 KV tiles + split signal/wait -- p22s, 439 VGPR (proto `barriers`)

- **Path:** `$P=.../proto/barriers`; diff `proto.diff`; review `REPORT.md`; tools `tools/sync_model.py`
  (2484 configs, 0 failures), `tools/isa_loops.py`, extra ISA in `review/isa/`.
- **What it changes (:167-175):**
  - `KV_TILES_PER_BARRIER` (G) = 2: one WG barrier per 128 KV instead of per 64. The SIMD partner waves
    (i / i+4) may drift up to about 1 tile apart, and TDM lead grows from about 1 to 1.9 tiles.
  - `KV_BARRIER_GROUPS` (NG) = 2.
  - `SPLIT_BARRIER` = True: `s_barrier_signal` goes right after the last tile's PV `s_wait_dscnt(0)`, and
    `s_barrier_wait` at the next group top, so 32 PV WMMAs sit in the gap.
  - `KV_SLOT_ALIGN=1024` replaces the 64 KB slot floor.
  - `BARRIER_FENCE=True` puts WG fences around the raw intrinsics.
  - `BARRIER_PROTO=False` gives champion bytes.
  - **Remove the `PROTO_ENV` env override (:177ff) and hard-code the winner** in the op-evolve copy.
- **ISA evidence** (default p22s = G2 NG2 split; prod = fast, md5 `a12d1565`):
  - Resources: VGPR 439, SGPR 98, spill 0/0, scratch 0. LDS 327680 is still allocated; the ring uses 140 KB.
  - Per 256 KV: WMMA/v_exp/ds_load/TDM same as the champion. 2 barrier pairs. signal->wait gap 82-84
    instructions, with 32 PV WMMAs. tensorcnt waits 2x (0x0). dscnt 36 (champion 44). SALU 216,
    branch 28, v_nop 96, VALU 740.
  - Other variants, all 0 spill:

    | variant | VGPR/SGPR | barriers | tensorcnt | SALU | branch |
    | --- | --- | --- | --- | --- | --- |
    | p12s | 439/100 | 4 | 0x0 | 184 | 24 |
    | p13s | 439/98 | 4 | 0x2 | 240 | 32 |
    | p22n | 438/98 | 2 | 0x0 | 184 | 22 |
    | p23s | 439/106 | 2 | 0x4 (= ASM) | 270 | 36 |
    | p23n | 439/105 | 2 | 0x4 | 240 | 30 |

  - thd VGPR 445 -> 439. win_sink 455 -> 449, SGPR spill 4 -> 1.
- **Expected effect:** prod 0 to +4%, median +1%, low confidence. Barriers are 0 for 4 on bwd (h7), so
  this is a prior only. The mechanism is halved rendezvous count and partner-wave phase freedom.
  The split gap (bwd S2 null) and deep tensorcnt (p23s; bwd depth > 2 lost) have low priors.
  Cost: +48 to 100 SALU and +12 to 20 branches per 256 KV.
- **Risks:**
  - p23s and p13s fail at qk_hdim=256 (compile-time assert, ring 2x3x67584 B > LDS). If either wins,
    NG must drop to 2 at d256. p22s compiles at d128, d192 and d256.
  - More SGPR spill on unscored paths (non-causal 2 -> 6, d192 8 -> 9). It goes to VGPR lanes with no
    scratch.
  - Add a **d192** shape to the correctness gate: it is the only path with ops_per_tile=3.
  - The raw split intrinsics have no LLVM memory semantics. Re-check that no ds op crosses the gap in
    any port (`tools/isa_loops.py`).
- **How the round starts:**
  1. Copy per the common recipe, then hard-code the arm constants and delete `PROTO_ENV`.
  2. Arms vs `BARRIER_PROTO=False`:
     - p22s (default);
     - p22n (`SPLIT_BARRIER=False`), which separates barrier count from the split gap;
     - p23s only if p22s wins (`KV_BARRIER_GROUPS=3`, with the d256 fallback).
  3. Do this after h21/h22: the gap needs independent work to fill (h7).

### h25 -- L3+L29 in-WG q-tile pairing -- O-store overlap only, dispatch gain is zero (proto `pairing`)

- **Path:** `$P=.../proto/pairing`; diff `pairing.diff`; CPU dispatch/bijection model
  `bijection_check.py` (`bijection_check.out`: ALL OK); ISA `isa/HASHES.txt`, `review/`.
- **What it changes:**
  - `PAIR_Q_TILES` (:192): the launcher halves grid.x, and WG rank p runs light q-tile p, then heavy N-1-p.
    Tile 1's async O store (LDS -> VRAM) stays in flight while tile 2's Q/K/V TDM prologue issues;
    its O ring moves to the LDS tail.
  - The kernel infers paired mode at runtime from grid.x < N, so one binary serves both grids.
  - `PAIR_POLICY="exact"` pairs only when n_qt is even and the paired grid is whole 256-WG rounds:
    prod (2048 WGs) and proxy (256) pair, fast (32) does not.
  - `PAIR_PIN_LANE=True` stops LICM from hoisting about 25 lane values into v256+ (+100 msb per iteration).
  - Gated to qk_hdim==128.
- **ISA evidence** (prod = fast, md5 `bccf82e8d5b3`):
  - Resources: VGPR 447, SGPR 107, VGPR spill 0, scratch 0, 4853 instructions.
  - SGPR spill 5, to v256 lanes: 5 writelanes at entry and 5 readlanes per pass, none in KV loops.
  - Clean loops 620/620 with loop-top sync identical to the champion. The right-mask loop is 910/899
    (msb 162/158), once per pass.
  - One more barrier per pass and one `s_wait_asynccnt`.
  - Off -> `23e8ba8d3892` (champion); thd, non-causal and win_sink unchanged.
- **Expected effect:** prod +0 to +1.5%, most likely near min_gain (0.5-0.7%), possibly null.
  - Dispatch model: the champion's longest-first order already hits the lower bound (prod makespan
    1064 = 1064, proxy 69 = 69), so pairing gains **nothing on balance**.
  - The only upside is fixed-cost overlap: half as many WG launches, and the O-store drain overlapping
    the next tile's prologue (1 WG/CU because LDS is full).
  - fast runs the new code unpaired, so it should be neutral; it is the sentinel.
- **Risks:**
  - Before review, d256 failed to compile and d192 spilled to scratch. Both are now gated (hdim==128).
    **Never set `PAIR_POLICY="overfill"`**: the model shows odd n_qt or partial rounds losing 5-23%.
  - Only prod and proxy exercise the paired path on the card, and both have sq==skv, even n_qt and no
    tail rows. sq!=skv paired runs are covered by the CPU bijection check only.
  - Bitwise equality with the champion is likely but not guaranteed. A tile-localized large diff means
    a sync bug; uniform ULP diffs mean codegen. The 200-run determinism gate is the race check.
- **How the round starts:**
  1. Copy per the common recipe.
  2. Arm = `PAIR_Q_TILES=True`; control = `False` (champion bytes). Measure prod, proxy and fast.
  3. Lowest expected gain of the five: run it last or in a round's spare slot.

### h26 -- Closed levers from the 2026-09-27 compile-only prototypes

None of the five prototypes is dead. These sub-levers are closed by compile-only or CPU-model evidence;
do not spend a round rebuilding them:

- **L6 on non-causal / gqa=1 / win_sink / qk_hdim 192-256** at the current body:
  - non-causal: 512 VGPR, 92 spill, 372 B scratch;
  - causal gqa1: 512, 85 spill, 344 B scratch.
  These are gated to champion bytes in h21. Reopen only after something frees 60+ VGPRs.
- **L3 pairing as a load-balance lever:** the CPU dispatch model shows longest-first (h20, landed round 2)
  already at the makespan lower bound on prod and proxy. Pairing odd n_qt or partial dispatch rounds
  loses 5-23% (`mha_odd` 0.80x, `even_frac` 0.81x under overfill). Only "exact" policy, and only for
  overlap (h25).
- **L14 unroll x2 as a stand-alone speed lever:** it keeps the per-tile wait, barrier and
  `sched_barrier(0)`, and only saves 4 moves and loop overhead for about +10 v_nop. Expected null. Keep
  it only as scaffolding (h23).
- **U2 together with PP3 at qk_hdim=192:** 8 VGPR spill and 36 B scratch (h23). Gate U2 to d128.
- **Deep tensorcnt (NG=3 / PP3) at qk_hdim=256:** 3 slots do not fit 320 KB LDS (compile-time assert).
  Always fall back to 2 there.
- **L20 branch-free rescale at n_block=64 as a VALU saving:** it adds 31 VALU per iteration and does not
  shorten the serial span (222 vs 218 with defer). If h22's C1 loses to C2 on the card, close BF for
  good.
- **L15 with `fastmath=fast` on the exp-argument fma:** it licenses LLVM to delete the causal -inf mask.
  Closed as a correctness hazard.

### h27 -- Prototypes h21-h25 are based on ROUND 2 -- port their diff, never copy their files over the round-4+ champion

The compile-only prototypes under `Primus-Turbo/output/0927__flydsl/proto/<name>/` were built from
round 2's `op/current`. Round 4 then changed the softmax (packed exp argument v_pk_fma_f32 + packed
row-sum v_pk_add_f32). Copying a prototype's `flydsl_fwd/*.py` onto the working copy would silently
REVERT round 4 (-6.3% prod). Always: `diff -ru proto/<name>/base proto/<name>/op` and apply that diff
to the current working copy (resolve conflicts by hand -- h22 overlaps round 4's packed changes most),
then re-check ISA (VGPR <= 512, 0 spill) before measuring.


## h28 -- A0 results for rounds 5-7 (their fork of this job): three losses and one harness artifact -- read before choosing

This job was cloned on 2026-09-27 from `gfx1250-flydsl-attn-fwd-20260925-114644` at round 4 (champion r4, prod
1082 TF/s on A0, 1514 TF/s on B0 = 0.787x ASM). The original job keeps running on A0 and pushes its rounds to
`output/0925__flydsl/fwd-job/rounds/` of the Primus-Turbo branch `dev/lhz/flydsl-attn` (read with
`git -C <PT> show origin/dev/lhz/flydsl-attn:<path>`; never pull, never push). What A0 measured after the fork:

| A0 round | lever | result (prod, same-session vs r4) |
|---|---|---|
| 5 (deep) | h21 / L6 QK(i+1)/softmax(i) software pipeline, clean ISA 508 VGPR | **-15%** (gain 0.835) |
| 6 | h22 / L15+L17+L20 softmax arithmetic (permlanex16 16 -> 10) | **-2%** (0.981/0.985/0.982 over 3 arm orders) |
| 7 | h24 / L8 barrier per 2 KV tiles + split signal/wait | **-3.5..-6%**; barrier removal ceiling measured at **+5%**; split signal/wait itself worth ~3 points over no-split |

**Harness artifact (A0 r6)**: a FlyDSL arm measured AFTER the ASM `beat` arm in the same process pays an I-cache
penalty (I-cache misses 0 -> 808): proxy +31%, prod +1.7%, fast worse. Rank candidate vs champion in a process
where both FlyDSL arms see the same position relative to `beat`, or without `beat` between them; rotate arm orders.
A0 also notes `ut` hard-codes 50 dB while the spec floor is 49 dB (the champion itself fails ut at 50).

## h29 -- B0 host, ONE card of four: physical GPU 0 via container fa-g0 only; other cards are busy by design

From round 5 this fork runs on **B0 = ctheliosp-1b112-a37-1**, a 4-card gfx1250 node (one XGMI hive). This overrides
every "one card / the only GPU / fa-repro / heliosr" statement in the spec prose, facts.md and earlier hints.

- **Your card is physical GPU 0** (rocm-smi GPU[0], PCI 0001:04:00.0, `/sys/class/drm/card0`, kfd gpu_id 34992).
  Container **`fa-g0`** exposes ONLY that card (`/dev/dri/renderD128`); inside it `torch.cuda.device_count()==1`
  and logical device 0 IS physical GPU 0. Env `OE_PHYS_GPU=0` is set in the container.
- **Every GPU command goes through `docker exec fa-g0 ...`.** Never `fa-repro` (parked; it saw all 4 cards), never
  `fa-g1/fa-g2/fa-g3` (bwd job and operator labs), never python on the host, never `docker run`. Do not set
  HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES.
- **Idle / contention checks look at card 0 only**: `rocm-smi -d 0 --showuse`, and KFD processes whose
  `/sys/class/kfd/kfd/proc/<pid>/queues/*/gpuid` == 34992. Processes on other cards are expected -- do not wait
  for them, do not kill them.
- **Clock witness**: `benchmark.py` reads card 0's own `pp_dpm_sclk` when `OE_PHYS_GPU` is set.
- **Power-wall / node-idle analyses**: the node is never idle. Record `node shared (4-card campaign)`, use only
  card-0 telemetry, do not retry-wait.
- B0 absolute numbers are not comparable with A0's (B0 prod: ASM 1.144 ms, r4 1.453 ms). Only same-process ratios.
- The FlyDSL JIT cache in fa-g0 is `/tmp/flycache` (container-local). The aiter JIT module lives in the shared
  aiter-src tree and is pre-built; never delete it (a rebuild takes a lock other cards' processes wait on).
- A wedge on any card of the hive can take down all four: compile-only first, 0 spill / 0 scratch, bounds-prove
  every new index expression on CPU.

## h30 -- Division of labour with A0 and the GPU-2 lab: this fork owns the softmax max/rescale and row-sum structure levers and the 4-wave roadblock

- **A0 owns** the h21-h25 families (software pipelining, packed softmax arithmetic, TDM prefetch depth / unroll,
  barrier restructuring, in-WG pairing) and whatever it derives from them. Their status column here says
  `a0-owned`; do not build them. Their bodies stay readable as context.
- **The operator lab on GPU 2 owns cheap sweeps**: L22-L24 (waitcnt/dscnt placement, sched_barrier pair around
  s_barrier, `llvm_options` max-memory-clause / post-misched) and grid remaps L4/L5. Winners arrive here as new
  `must` hints with a measured tree; do not spend a round on compiler-flag sweeps.
- **This fork, candidate order** (decision index in h18; each a separate arm, compile-only first):
  1. **L20 / L21** -- the running-max / rescale structure: drop the rescale branch (select, or fixed defer), and
     the fixed-max `_FMAX0` variant. L21 REQUIRES the adversarial large-logit correctness test (f16/h16) before
     any card number is believed.
  2. **L18** -- WMMA ones-row row-sum (moves the row-sum off VALU; A0 r3 showed softmax VALU is on the critical path).
  3. **L19** -- raw `v_exp` log2 path, if L18/L20 leave exp on the critical path.
  4. **L12** -- the 4-wave roadblock: the V2/V3 managers raise on `num_waves != 8`; the lever is making a 4-wave
     build legal (a structural change, not a flag). Never record "4 waves is slower" without a legal build.
- New ideas outside this list are welcome if they do not overlap A0's families; say so in opt.md.

## h31 -- Rank candidate vs champion in a process WITHOUT beat; beat only in its own process

A0 round 8 (their h28) showed the post-beat I-cache artifact (our h28) is large enough to reject a real +0.7% win.
From now on every ranking A/B runs `--arm-path cand=... --arm-path champ=...` with NO `beat` arm in that process;
beat (ASM) is measured in its own separate process only to report % of ASM.

## h32 -- A0 r8: nodelay (amdgpu-enable-delay-alu=False) +0.7% prod

A0 measured `amdgpu-enable-delay-alu=False` at +0.7% prod (3/3, fast/proxy neutral without beat) and is re-landing
it. It is an A0 lever (and overlaps the GPU-2 lab's L24 option sweep). Do not spend a round on it here.

## h33 -- GPU-2 lab measured L22/L23/L24/L4/L5: no win -- closed; L12 and L21 now belong to the labs

Source: `PT/output/0927__b0/lab2/MEASURE.md` (same-process A/B vs champion r4, no beat in process, 3 processes x
n=101 prod, every arm bitwise-equal output to champion). Ratio champ_ms/cand_ms, mean of 3:
- L24 compile options: iterative-ilp 0.934, expert-sched-mode2 off 0.973, max-memory-clause(+no post-misched)
  0.977-0.980, max-ilp 0.987, no post-misched 0.994, disable-machine-sink 0.999 (null). coexec spills (dropped).
  The champion's scheduler options are already the best measured; static ISA wins (itilp: -17 VGPR) lost most.
- L22 (graded dscnt in PV) 1.000 null; L23 (drop sched_barrier pair around the split barrier) 0.988 loss;
  L22+L23 0.986.
- L4 XCD grid remaps: bmajor 1.000, spread 1.003 -- null; grid locality does not matter at prod.
- L5 causal-aligned origin: null at prod (pad = 0 there by construction).
All of L4/L5/L22/L23/L24 are CLOSED for this job. Also re-assigned from h30: **L12 (4-wave legalisation) goes to
the GPU-2 lab and L21 (fixed max `_FMAX0` + adversarial large-logit test) to the GPU-3 lab.** This job's list
is now L20, then L18, then L19; winners from the labs arrive as `must` hints with a measured tree.

## h34 -- Stack A0 round 11 onto our round-6 champion: nodelay + shape-gated BLOCK_M=128

A0's fork accepted round 11 (their prod 1111 TF/s on A0 clocks; fast now beats ASM). Their champion is r4 +
two changes, both orthogonal to our round-6 speculative softmax. The full A0 tree is copied read-only at
`/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/fwd/a0_r11_champion/`.

1. **nodelay**: `"amdgpu-enable-delay-alu": False` added to the kernel's `llvm_options` (2 lines in
   `flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py`). A0: prod +0.7..+0.83% in 3/3 and 4/4 sessions without beat.
2. **Shape-gated BLOCK_M=128** (`r11.i1.g30`): new file `flydsl_fwd/fmha_fwd_prefill_a16w16_m16x8.py` and an
   `impl.py` dispatch: use the m16x8 kernel only when `d == 128` and the BLOCK_M=256 grid
   (`ceil(sq*(hq//hkv)/BLOCK_M) * hkv * b`) is smaller than the CU count. prod/proxy grids are far larger, so
   they never take it; fast gains 1.075-1.194x on A0. This also recovers our round-6 fast regression (-3.6%).

Arms (each vs our op/current, no beat in the ranking process, h31):
- A = op/current + nodelay (port the 2-line diff; our speculative code path must also get the option);
- B = A + the m16x8 gate (copy A0's m16x8 file as-is; it does NOT carry speculative softmax, which is fine because
  only launch-bound small grids take it -- but run the full correctness gate, incl. causal/non-causal small shapes).
Our round-6 speculative softmax passed a 256-case adversarial suite (`PT/output/0927__b0/verify_r6/REPORT.md`);
any change to it needs that suite re-run (scripts there).

## h35 -- RULER ALERT: identical champion code measures bimodally at prod

Found 2026-09-27 by the GPU-3 lab verifier (`PT/output/0927__b0/lab3-L21/VERIFY.md`): in one process, `current`
and a byte-identical `champ_copy` measured 1.44 ms and 1.525 ms (A/A 0.944, twice). Across processes the champion
lands at ~1.44 or ~1.52 ms. A +6% "win" (L21 bnegg) turned out to be exactly this and is null. The cause is under
investigation on GPU 2 (`PT/output/0927__b0/ruler/`; suspect: module names from hash(path) with
PYTHONHASHSEED unset). Until that report lands:
- every ranking process carries `--arm-path champ=<op/current>` AND `--arm-path champ2=<a byte-identical copy at a
  different path>`; report both. If champ vs champ2 differ by more than 0.5%, the process is void;
- set `PYTHONHASHSEED=0` in every benchmark process (costs nothing, may remove the effect);
- a gain near 6% is not believable without >= 4 processes where the champion copies agree.
Round 6's acceptance (+5.7%) and the L12 lab win (+6.4%) are being re-measured under a fixed ruler.

## h36 -- A0 fwd job paused at round 13; h23 and h25 are open to this job again

A0's fork stopped ~15:30 UTC 2026-09-27 (machine handover), round 13 interrupted mid-opt. Its final champion is
round 11 = r4 + nodelay + shape-gated BLOCK_M=128 (h34 already asks us to stack both). Its last rounds:
r12 split-KV for small grids = null; h22 closed (row-sum -1.8%, branch-free rescale -8.3%).
A0 did NOT build h23 (TDM prefetch depth 3 + unroll x2) or h25 (in-WG q-tile pairing); their status here is `open`
again -- remember h27: those prototypes are based on round 2, port the diff onto our champion.
A0 also left a CPU study of an ASM-structure 4-wave port (`PT/output/0927__flydsl/asm-structure/`, A1 = 4 waves,
A2 = 4 waves + n_block 128, none measured on a card) that overlaps the GPU-2 lab's L12 result
(`PT/output/0927__b0/lab2-L12/`, 4-wave x 2 WG/CU, +6.4% prod but proxy -6.8%, pending the ruler audit h35).

## h37 -- Re-land round 8 arm B alone

Round 8 measured arm B (h34 part 2: A0's m16x8 kernel for grids smaller than the CU count, `impl.py` dispatch) at
fast 1.294x r6 (1.288/1.378/1.221), prod 1.0014, proxy 0.995 -- and its prod/proxy dispatch is byte-identical ISA to
r6, so those two are the A/A floor. It was rejected only because proxy read 98.7% of its best-ever (r7) against a
0.993 band; the operator has since set `evolve.shape_band.proxy: 0.98` (multi-arm proxy spread is +-6%).
Round 9 (or the next fast round): rebuild arm B from `rounds/008` (its tree is in the round's scratch/raw per opt.md)
on top of op/current, run the full gate (incl. the verify_r6 adversarial suite on the small shapes, as round 8
did), and measure it with the h35 A/A protocol. Arm A (nodelay) is DEAD on the speculative body (prod -1.65% 3/3):
do not stack it.

## h38 -- DAY 2 CARD MOVE: physical GPU 2 via container fa-g2 only

From 2026-09-28 (round 9 onward) this job's card is **physical GPU 2** (rocm-smi GPU[2], PCI 0003:04:00.0,
`/sys/class/drm/card16`, kfd gpu_id **57865**), container **`fa-g2`** (only `/dev/dri/renderD144`; logical device 0
inside it IS GPU 2; `OE_PHYS_GPU=2`). Everything h29 says about GPU 0 / fa-g0 / gpu_id 34992 now applies to
GPU 2 / fa-g2 / 57865 instead. Never touch fa-g0 (end-to-end runs), fa-g3 (bwd job), and never GPU 1: it is
**wedged** (MES failed to respond, 2026-09-28) and has no container. Idle checks look at card 2 only
(`/sys/class/drm/card16/device/gpu_busy_percent`, KFD queues with gpuid 57865). The benchmark's sclk witness
follows `OE_PHYS_GPU` automatically. h35 (A/A champion copy in every ranking process) still applies.

## h39 -- Adopt round 9 arm BG verbatim as op/current

Operator decision (2026-09-28). Round 9 measured arm BG (= h37's m16x8 BLOCK_M=128 gate + r7.i2.g18 no-speculation on
masked tiles), no beat in process, PYTHONHASHSEED=0, 3 rotated sessions, against the ACTUAL champion r6 in the same
process: prod 1500.58 vs 1494.28 (+0.42%), proxy 1587.21 vs 1570.80 (+1.04%), fast 139.30 vs 109.24 (+27.5%); the
in-process A/A copy champ2 read prod 1.0012 / proxy 0.9958. Correctness passed (ut determinism 200, full gate,
adversarial suite on the small shapes). It was rejected only because acceptance compares each shape to its best-ever
record, and those records (fast 139.09 = r8's unshipped arm B, proxy 1596.26 = r7's unshipped arm) belong to code that
never became the champion -- so h37 could never land.

**Refactor task**: copy `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-fwd-b0-20260927/rounds/009/op/`
(identical copy at `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/fwd/r9_BG/`) into the working copy
VERBATIM -- no edits -- and let the correctness gate run. Do not change any kernel code. Promotion clears the
per-shape champions, which also removes the stale best-ever records.

## h40 -- RULER FIXED: benchmark.py times BLOCKED

Audit on GPU 0, 2026-09-28 (`PT/output/0927__b0/ruler/REPORT.md`, evidence `ruler/runs0/`). The "bimodal champion"
of h35 was r4 vs r6 (op/current was re-installed mid-session), and the hash(path)/PYTHONHASHSEED theory is refuted
(binaries byte-identical across copies, seeds and processes). The real defect: the old harness alternated arms call
by call, and at prod each call inherits the card state left by the previous few calls (power/clock controller memory,
survives a 2 GB L2 flush). Identical code then read 2-3.5% apart by position; r4 read 1.65 ms in {r4,beat} but 1.50 ms
in {r4,baseline,beat}; r6/beat read 1.29x interleaved vs 1.035x steady-state. It rewards the higher-power arm, and it
is what earlier hints called the "post-beat I-cache penalty".

Installed: `job_context/op/benchmark.py` now defaults to BLOCKED timing -- per round each arm runs 4 untimed then 9 timed
calls back to back, rounds palindromic (`--block 1` restores the old mode; backup `benchmark.py.bak.pre-blocked`).
Validated: A/A within +-0.19% over 9 prod processes; r4 no longer depends on the arm set (1.549 vs 1.544); within 1% of
a back-to-back stream mode. validation.py calls it, so the gate uses it too.
- Absolute ms/TF/s from before this change are NOT comparable with new ones; always re-measure the champion (and beat)
  in the same process. h35's A/A copy is no longer required but stays cheap insurance.
- Any earlier +-3-7% conclusion measured interleaved (especially with beat in the process) is discounted.

## h41 -- Re-measured under the fixed ruler

| comparison | prod (per process) | proxy | verdict |
|---|---|---|---|
| r6 vs r4 (round 6) | 1.0463 / 1.0461 / 1.0468 | +2.3..2.9% | real, +4.6% (not +5.7%) |
| L12 `r6_occ2_lo` vs r6 | mean 0.9964 over 6 | 0.928 | DEAD -- do not re-land; the lab's +6.4% was the ruler |
| bnegg (L21 guarded fixed max) vs r6 | 1.0059 / 1.0064 | not measured | same mechanism as r6's speculative softmax; +0.6% unconfirmed |
| r6 vs ASM beat | blocked 1.035 / 1.038 (r6 FASTER), stream 1.022 | -- | beat steady-state 1.43-1.46 ms at prod; old "% of ASM" figures were too low |

## h42 -- End-to-end reality check: our fwd loses to ASM inside training, and is clock-sensitive

E2E on GPU 0, 2026-09-28 (`PT/output/0927__b0/e2e/RESULT.md`, same-process ABBA/BAAB step pairing): with fwd r6 +
bwd r20 FlyDSL vs aiter ASM, FlyDSL is 0.42-0.45% slower per step (71-76 ms/step), matching the trace's FA-path delta.
Per layer inside training: our fwd 1.64-1.85 ms vs ASM 1.22 ms (1.34-1.52x), although the blocked op-level ruler reads
r6 at 1.035x FASTER than ASM. The difference is the operating point: in training the card sits at other clock/power
states between attention calls (GEMMs in between), and our fwd moves +13% between clock states while ASM does not.
Implications for this job: (1) a candidate that cuts issued work / power (fewer VALU ops, fewer instructions) is worth
more end to end than one that only wins at the steady high clock; (2) when a candidate wins, also measure it vs the
champion with the process pinned to a lower clock state (e.g. interleave a GEMM between calls, or `--block 1` which
reproduces the interleaved state) and report both. The prod ranking stays the blocked ruler (h40).

## h43 -- Speculative softmax loses on real training data; the randn ruler mis-ranks it

Profiling inside the e2e training on GPU 0 (`PT/output/0927__b0/profile/REPORT.md`): real q/k/v from the model (dumps
`/home/lihuzhan/_prof_dump/qkv_call0*.pt`, [B,S,H,D] bf16) have attention-score std 21-53, randn has ~1.0. On real data
the stale-max speculation recomputes on 13-25% of (32-row, 64-col) tile steps (0% on randn). Measured on real inputs at
the training clock: r6 1.83-2.08 ms vs r6 with `SPEC_STALE_MAX=False` 1.59-1.60 ms (and 8-12% faster than ASM at 2350 MHz);
on randn the nospec variant is 5% slower -- which is why every round's ruler keeps speculation. In training our fwd runs
1.45-1.72x ASM per layer; the gap is clock sensitivity x data (0.84 x 1.43 x 1.32).
Until the operator's nospec validation lands (it will arrive as a `must refactor`): do not build new speculative /
stale-max variants, and prefer levers that reduce issued work and clock sensitivity (v_nop, SALU, barriers).

## h44 -- Adopt r13ns verbatim as op/current

Operator lab on GPU 0, 2026-09-28 (`PT/output/0927__b0/fwd-nospec/REPORT.md`). r13ns = round 13 with `SPEC_STALE_MAX`
True -> False in m32x8 (line 173) and m32x2 (line 178); nothing else. Compile-only: 16 configs 0 spill / 0 scratch;
prod m32x8 5024 -> 4130 instructions. Gate-style checks (copy of gates.py): all PASS, 200/200 deterministic, output
BITWISE EQUAL to r13 on every gate shape and on 6 real training dumps. Timing (prod, same process, 3 rotated):
- randn blocked (this job's ruler): r13ns/r13 time 1.048 (4.8% slower) -- the ruler's bias, see h43;
- real q/k/v dumps, blocked: 0.846-0.919 (8-15% faster);
- real dumps right after a GEMM burst (training operating point): 0.774-0.880 (12-23% faster); ASM 1.27-1.30 ms,
  r13 1.85-2.11 ms, r13ns 1.62-1.64 ms;
- e2e Llama-3.1-8B (nkfix, 32 layers): fwd 53-60 -> 48-49 ms/step and no longer drifts with training step;
  direct e2e r13ns vs r13 per-step 0.9966 (5/5 pairs < 1).
**Refactor task**: copy `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/fwd-nospec/arms/fwd_r13ns/` into the working copy VERBATIM (md5 list `arms/fwd_r13ns.md5` next to it) -- no edits --
and let the correctness gate run. Do not re-measure it against the randn ruler to decide anything.

## h45 -- Speculative softmax is dead on real data

Never re-introduce stale-max / guessed-max / trigger-and-redo softmax variants: the randn ruler will always say +4.8%
for them and training will always lose (real score std 21-53 -> 13-25% of tiles recompute). Any candidate whose cost
depends on the data must ALSO be compared with the champion on the real dumps
`/home/lihuzhan/_prof_dump/qkv_call0{672,673,674,680,688,703}.pt` ([B,S,H,D] bf16) in the same process (tools in
`PT/output/0927__b0/fwd-nospec/tools/`), and a loss there is a loss. In the training operating point r13ns is still
1.25-1.29x ASM: prefer reducing issued work and clock sensitivity (v_nop, SALU, barriers).
