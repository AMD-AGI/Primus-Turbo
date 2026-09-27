# Round 11 -- fast round, opt (route design)

Working copy `rounds/011/op` == `op/current` (round-4 champion, delay-ALU on). Diff vs `rounds/010/op`:
only the `amdgpu-enable-delay-alu` llvm option (nodelay, r8.i1.g20).

## Step 1 -- findings read

- facts: bottleneck = on-chip per-wave issue/latency at prod; nodelay is a repeatable prod +0.83%
  (4 sessions) but the round-10 three-shape gate gain was only 1.0022 < 1.007.
- dead_ends: 14 ids, incl. split-KV (g04), fixed max (g27), barrier variants (g21), lock_simd (g25).
- pool: g28 (pairing on nodelay), g19 (ping-pong on depth-3 ring), g26 (diagnostic), g15 (carrier only).
- route: round 10 ordered h29 (nodelay, done: not accepted) > g28 > g19 > g26.
- round-10 reflect: route_pattern "rounds 5-10 repeatedly targeted KV-loop softmax/barriers/scheduling;
  orthogonal mechanisms remained behind them".

### What the acceptance arithmetic says (state.yaml round 10)

`gain` is the geomean over fast/proxy/prod of candidate/incumbent and must reach 1.007. A prod-only
+0.83% contributes ~+0.28% to the geomean; that is exactly why nodelay (1.0022) was rejected. Any
candidate that only moves the per-tile loop pays mostly at prod and proxy, and must be ~+1% on BOTH
to clear the bar. The fast shape is one third of the score and has never had its own lever.

## Step 2 -- look

### Static census (no card): clean loop of op/current vs nodelay (reused round-8 tool `loops_census.py`)

| build | clean loop instr/tile | s_set_vgpr_msb | SALU | v_nop | s_delay_alu |
| --- | --- | --- | --- | --- | --- |
| current (r4) | 513 | 40 | 116 | 15 | 14 |
| nodelay (r10/op) | 499 | 40 | 102 | 15 | 0 |

msb is 8% of the loop; round 8 (2c) already found no llvm option that lowers it (mmc 38, max-ilp 51)
and FlyDSL has no register-placement API. Not re-opened.

### The fast shape is a latency problem, not a throughput one

fast = b1 s1024 hq8 hkv2 d128 causal. BLOCK_M=256 packed rows (64 seq x 4 heads) -> 16 q tiles per
kv head -> **32 workgroups on a 256-CU device**. Time is set by the single heaviest WG (16 KV tiles
of 64) plus prologue/epilogue: 38.8 us one-image vs beat ~33 us. Every lever tried so far is a
per-tile throughput lever; split-KV (g04) was killed by its extra combine dispatch.

Untested lever: **a smaller BLOCK_M for under-filled grids** (`WMMA_ROW_PER_WAVE=1`, BLOCK_M=128).
64 WGs instead of 32 (still < 256 CUs, so no queueing), and the heaviest WG's per-tile work halves
(32 instead of 64 WMMAs, half the softmax) while tile count stays 16. The h8 retile sweep
("(256,64) is a local optimum"; "R=1 n=128: 0.718x") was taken at prod, where the grid is full and
halving BLOCK_M only doubles K/V refetch and per-tile fixed cost. Nothing in the record measures
R=1 n=64 at an under-filled grid.

Prediction before measuring: fast time 38.8 -> 28-32 us (per-tile body ~0.55-0.65x, fixed cost
unchanged). If R=1 fast is >= 37 us the per-tile fixed cost dominates and the lever is dead.

### p1 -- measured this round (`raw/p1_summary.md`, bulk in `raw/p1/`, arm tree `raw/arms/r1`)

One-line throwaway arm: `WMMA_ROW_PER_WAVE = 1` on op/current. Compile-only first: VGPR 300, 0 VGPR
spill, **3 SGPR spills to VGPR lanes** (private segment 0, no scratch instruction), loop 513 -> 366
instr/tile but msb 40 -> 61. Judgement on h14: the wedge mechanism on record is scratch; lane spills
with a zero private segment have none, and round 10 already queued g28 with 5 such lane spills. I ran
the correctness gate first, alone, and only then the timing.

- Correctness, validation's own `check_correctness` at 49 dB, fast causal: o 51.23 dB, lse 85.48 dB, PASS.
- Timing, one implementation per process (h28 + round-10 one-image rule), 2 sessions, sclk 1100 throughout,
  GPU idle before and after:
  r1 36.73 / 32.81 us median vs current 39.50 / 39.18 us -> **1.075x and 1.194x**.
- Against my prediction (28-32 us): short. Session spread of r1 is 12% while current's is 0.8% -- not
  resolved; the next step must run >=3 sessions before quoting a number.

Reading: the fast shape IS latency-bound on the heaviest WG and the per-tile body is most of it; halving
the rows per wave removes ~30% of loop instructions and buys 7-19% wall. It is the first lever in this
job aimed at the fast third of the score. At a full grid (proxy 512 WGs, prod 4096) R=1 doubles
K/V traffic and per-tile fixed cost, and h8 recorded R=1 losing at prod, so the lever must be gated
on grid fill.

## Step 3 -- sources

- (a) pool/route: g28 (pairing, unpriced <=1.5% prod, deep-born r10), g19 (needs a ~3% ring first),
  g26 (diagnostic), g15 (carrier). All per-tile or epilogue levers on the prod path.
- (b) survey: census above + p1.
- (c) corpus: `optimization/routes/1-metrics-to-techniques.md` (skimmed for issue/latency),
  `arch/gfx1250/isa.md` (WMMA hazard / co-exec slot rules -- explains the 15 v_nop per tile, nothing to
  take), `backends/flydsl/attention/README.md` + `recipes/hd128.md` (gfx950 config, poor match: took
  nothing), `optimization/techniques/6-gfx1250-cdna5-mechanisms.md` (no msb lever).
- (d) cross-backend: `backends/hipkittens/attention/recipes/gqa_d128.md` s6 -- rank 1 is hand-placed
  registers (1.72x via v_accvgpr traffic); the gfx1250 analogue here is s_set_vgpr_msb (40/tile, 61 at
  R=1), and FlyDSL has no placement API, as round 8 found. Rank 7's "readfirstlane hoisting" ~1.8% is
  not separated in that card. Nothing buildable in one round.

## Step 4 -- choice

- **Arm A = r11.i1.g30**: shape-gated R=1 (BLOCK_M=128) when the R=2 grid does not fill the device.
- **Arm B = r8.i1.g20** (nodelay, keeps its id): measured prod +0.83% in 4 sessions; rejected in round
  10 only because the geomean needs more than one shape to move.
- They touch different shapes (A: fast only by construction; B: every shape, mostly prod), so they are
  close to orthogonal; the merge must still be measured.
- Expectation: A alone fast +7-19% (call it +10%), proxy/prod bit-identical (gated off) -> geomean
  ~+3%. B alone ~+0.3% geomean. Merge ~+3.3%, clearing the 1.007 bar -- IF the gate's fast measurement
  (which interleaves beat and two images) does not swallow it; round 10's gate read fast 54.61 vs 54.87.
- Passed over: r10.i3.g28 (deep-born) -- outranked: unpriced, <=1.5% on prod only, while g30 has a
  measured 7-19% on a shape no lever has touched. It stays at row 3.

## Step 5 -- route written

`findings/route.md` `## Route` rewritten: g30 > r8.i1.g20 (nodelay) > g28 > g19, then the nine advise
items (conflict noted there: rule 4 caps at four rows, rule 2 wants every outstanding item). Pool gains
one entry, r11.i1.g30. No build failures this round; the only card work was p1 (correctness then four
one-image fast timings), GPU idle before and after, no process left in the container.

Consulted: knowledge/INDEX.md (not opened -- went by directory), optimization/routes/1-metrics-to-techniques.md,
arch/gfx1250/isa.md, backends/flydsl/attention/README.md, backends/flydsl/attention/recipes/hd128.md,
backends/hipkittens/attention/recipes/gqa_d128.md, optimization/techniques/6-gfx1250-cdna5-mechanisms.md.

# Round 11 -- steps 5-7: build and measure the route

Route rows taken from the top: row 1 `h29` (nodelay = Arm B, tree = `rounds/010/op` copied to
`raw/arms/B`), row 2 `r11.i1.g30` (Arm A). They were built from op/current independently of each other.

## Build

- Compile cache `/root/.flydsl/cache` cleared before the first build (`raw/cc2/run.sh`).
- Arm A = working copy `rounds/011/op` (mirrored in `raw/arms/A`):
  - new `flydsl_fwd/fmha_fwd_prefill_a16w16_m16x8.py` = m32x8 copied with `WMMA_ROW_PER_WAVE = 1`
    (BLOCK_M 128). It is a separate file because FlyDSL keys its cache on source text; a runtime
    patch of the global would reuse the R=2 binary.
  - One fix in the copy: the LSE zero-fill path wrote `tid < BLOCK_SIZE (256)` rows, which is
    the next WG's rows once BLOCK_M = 128. It is now masked `tid < BLOCK_M`.
  - `impl.py` gate: `d == 128 and ceil(sq*G/256)*hkv*b < multi_processor_count` -> m16x8, else
    m32x8. Grids: fast 32 WGs -> R=1; proxy 512 and prod 4096 -> R=2, untouched.
- Compile-only (`raw/cc2`, rc 0 on all four):

| build | md5 | VGPR | SGPR | vgpr spill | sgpr spill | private seg | scratch instr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A m16x8 @fast | 05d288f6 (= p1 r1) | 300 | 107 | 0 | 3 (lanes) | 0 | 0 |
| A m32x8 @prod | 7b078261 | 456 | 100 | 0 | 0 | 0 | 0 |
| current @prod | 7b078261 | 456 | 100 | 0 | 0 | 0 | 0 |
| B (nodelay) @prod | 07ae061a | 456 | 100 | 0 | 0 | 0 | 0 |

  A's prod/proxy kernel is byte-identical to current. A's fast kernel is byte-identical to the p1
  probe; the LSE mask sits on a path the bshd kernel does not emit. Loop: 366 instr/tile, msb 61.
- Merge tree `raw/arms/M` = A + `amdgpu-enable-delay-alu: False` in both llvm_options dicts of
  both kernel files. Prepared but not built: it is built only if neither arm lost.

## Correctness of A (`raw/c1`, one process at a time, GPU idle before, between and after)

- `ut/test_correctness.py --impl arms/A --determinism 200`: rc 0. All 16 cases PASS (gate 49 dB):
  - fast o 51.23 / lse 85.48 dB
  - proxy 50.89 / 88.24 dB
  - prod 50.83 / 89.20 dB
  - lowest is short_q full o 49.82 dB. short_q (sq 128) now runs R=1 through the gate, so this
    49.82 comes from R=1; I did not re-run current's ut, so there is no R=2 figure beside it.
  - Determinism 200/200 bitwise.
- `validation.py arms/A`: rc 2. Correctness 16/16 PASS and determinism PASS. The only failure is
  speed vs beat (geomean 0.7152 < 1.0, fast 0.6151 / proxy 0.7701 / prod 0.7724), the same bar
  every round of this job has failed.
- Warning sign in that run: A's fast median was 53.08 us (min 37.7, max 111) with beat in the
  same process. The p1 one-image runs read 33-37 us. So the measurement plan gained a
  "gate-style" fast run (arm + beat in one process) per session, for A and for INC.

## Measure -- benchmark.py, unprofiled, one implementation image per process

Setup:
- Every shape runs in its own process; beat runs in its own process (h28).
- Session order is rotated.
- INC = `rounds/004/op` (the only per-shape champion; best_round 4), re-run back to back in the same
  session. GPU idle (no KFD PIDs, 0%) before and after every batch; no amdgpu fault in dmesg.
- sclk: fast 1100 throughout; proxy/prod 1037-1100 for every arm alike (VR throttle).

b1 (`raw/b1`) -- A, B, INC, 3 sessions (+ beat once):

| shape | A (TF/s, 3 sess) | B | INC | beat | A/INC | B/INC |
| --- | --- | --- | --- | --- | --- | --- |
| fast | 65.1 / 66.2 / 65.4 | 53.9 / 55.1 / 55.3 | 54.3 / 54.0 / 55.1 | 64.5 | 1.204 | 1.005 |
| proxy | 930.8 / 929.5 / 901.6 | 930.7 / 923.5 / 928.5 | 926.2 / 930.5 / 932.5 | 1021.9 | 0.990 * | 0.998 |
| prod | 1101.3 / 1101.7 / 1102.3 | 1110.9 / 1111.1 / 1111.3 | 1103.1 / 1102.5 / 1103.2 | 1407.8 | 0.999 | 1.0074 |

(*) A's proxy kernel is byte-identical to INC's. The 901.6 (152.5 us) session is noise and sets the
session floor on proxy at ~3%.

Neither arm lost. A is +20% on fast. B is +0.74% on prod, flat elsewhere, and its prod
median sat 1979.0-1979.7 us in all three sessions. So the merge was built.

b2 (`raw/b2`) -- merge M, A, INC, 3 sessions:

| shape | M | A | INC | M/INC |
| --- | --- | --- | --- | --- |
| fast | 65.8 / 64.7 / 65.8 | 62.3 / 65.0 / 65.8 | 55.2 / 55.2 / 55.1 | **1.186** |
| proxy | 932.0 / 934.0 / 932.5 | 929.5 / 930.0 / 930.0 | 930.5 / 927.7 / 930.0 | **1.0037** |
| prod | 1111.4 / 1110.3 / 1111.8 | 1102.8 / 1102.1 / 1102.8 | 1102.4 / 1101.8 / 1101.8 | **1.0083** |

M/INC geomean **1.0627** (bar 1.007). A alone (b1) 1.0599. Over the six sessions, A's fast
single-image median was 32.45-34.49 us against INC's 38.94-39.82 us. p1's 1.075 was the low
outlier; the p1 spread question is answered by six sessions at 1.17-1.23x.

Gate-style fast (arm + beat in ONE process, the way validation measures):
- INC 63.5 / 65.0 / 64.7 us
- A 53.2 / 53.8 / 49.4 us
- M 44.7 us
- validation's own run of the working copy: 41.5 us
The ordering is the same as single-image, so the win does not depend on the measurement method.
Absolute fast times are much worse with two images in the process for every arm; I did not
investigate that.

Acceptance check against the rule (vs INC re-measured this session):
- Throughput improves on the best ever: geomean 1.063 > 1.007. Yes.
- Every below-target shape is at least 95% of its best ever: proxy 1.004 and prod 1.008 of INC. Yes.
- No shape that had reached its target falls below it: none had reached it before.
  - fast now reads 65.4 TF/s single-image vs beat 64.5 in its own process, i.e. at the target.
  - In validation's one-process pairing it reads 0.758 of beat.

validation.py on the shipped working copy (`raw/c2`): exit 2.
- Correctness 16/16 PASS (min o 49.82 dB short_q full, gate 49).
- Determinism 200/200 PASS.
- Speed geomean vs beat 0.7735 < 1.0 (fast 0.7583, proxy 0.7829, prod 0.7795). Round 10 was 0.6832.
  Speed is the only failure, and the job has failed this bar every round.

## Result

Shipped M (= r11.i1.g30 + h29 nodelay) in `rounds/011/op`:
- `impl.py`: grid-fill gate.
- `flydsl_fwd/fmha_fwd_prefill_a16w16_m16x8.py` (new): R=1, LSE row mask, nodelay.
- `flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py`: nodelay, byte-identical ISA to round 10's B.

Expected vs measured:
- Said before building: fast +7-19% (call it +10%), proxy/prod flat, merge ~+3.3% geomean.
- Measured: fast +18.6%, proxy +0.4%, prod +0.8%, geomean +6.3%.

Mechanism, in one line: the 32-WG fast grid is latency-bound on the heaviest WG. R=1 halves that
WG's per-tile WMMA/softmax work (loop 513 -> 366 instr/tile), doubles the WG count while staying
under 256 CUs, and so costs no queueing.

Not done: the gate threshold (grid < CU count) has not been swept. A grid between 256 and ~512
WGs might also gain from R=1; proxy (512) was not tried at R=1, and pool.md says to price it first.
