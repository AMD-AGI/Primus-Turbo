# Round 1 (fast) -- opt log

Written as it happened.

## Step 1 -- findings
facts.md / dead_ends.md / pool.md absent (fresh job, round 1). route.md has only framework
tables, all `_None._`. No deep profile exists. Prior knowledge comes from the spec prose
(STAGE2-FWD-SWEEP.md, 2026-09-23): n_block 64->128/256 spills (3.2x/9.9x slower); (BLOCK_M,n_block)
shipped point is a local optimum of 5 points; 4-wave arms BLOCKED (not slower); O_VARIANT v1
beats shipped v3 by 1.90% at prod (not in baseline); never tried: KV-loop deep unroll /
software pipelining, dispatch order.

## Step 2 -- survey

Environment note: the container's /tmp is NOT the host's /tmp (only /home/lihuzhan is
bind-mounted). The first detached launch into the assigned scratch dir silently did nothing
(redirect target did not exist in-container; pgrep confirmed no process). Bulk output moved to
/home/lihuzhan/.cache/op-evolve-r001-scratch (symlinked from the scratch dir).

GPU before: 0% use, no KFD pids, sclk 1100 MHz (VR throttle).

### Benchmark survey, current (== baseline) vs beat, 51 iters, one process, palindromic
Prediction before: prod ~1.51x gap (spec), proxy/fast unknown -- guessed similar.

| shape | current ms | beat ms | current/beat TF ratio |
|---|---|---|---|
| fast  | 0.06197 (min 0.04174) | 0.03093 | 0.499 |
| proxy | 0.25827 | 0.13376 | 0.518 |
| prod  | 2.3639  | 1.5718  | 0.665 |

geomean ratio ~0.557. **Surprise: proxy's gap (1.93x) is much larger than prod's (1.50x).**
Same kernel, same per-tile body, so the difference has to be in how the grid maps onto 256 CUs.

### rocprofv3 --kernel-trace returns nothing in fa-repro (instrument finding)
`rocprofv3 --kernel-trace --stats [-f csv]` exits 0 and writes NO files -- not even for a
one-line `torch.ones(10)+1`. Two attempts (default format; `-f csv`), same result. `--pmc`
in the same container DOES write `run_counter_collection.csv`. So the survey's `--stats` step
is unavailable here; kernel durations below come from CUDA events. Corpus
`arch/gfx1250/profiling-surface.md` documents the same silent-zero failure for PMC counters;
this is the kernel-trace flavour of it. Any harness must assert the CSV exists.

### Fast shape: the 2x gap is mostly not the kernel (raw/fast_*.txt, raw/interference_*.txt)
Prediction before: fast is launch/underfill dominated; expected current ~= beat back-to-back.
- back-to-back 200 calls, no flush: current 0.0361 ms/call, beat 0.0322 (1.12x). Host cost
  per call 0.016 ms for BOTH arms -> host overhead is not the difference.
- benchmark.py `--arms current` ALONE, fast: **0.0406 ms**. `--arms current,beat`: 0.060-0.071
  (3 repeats + 1 no-warmup). Warmup length irrelevant (0.2 s vs 8 s same).
- Controlled probe (sync; flush; 400k-cycle GPU sleep so host launch latency is hidden;
  then timed call): FlyDSL alone 0.0404 ms; **FlyDSL right after an ASM call 0.0691**;
  ASM after FlyDSL 0.0322 vs ASM alone 0.0313; FlyDSL after a torch fp32 matmul 0.0402 (clean).
- **Penalty is a fixed ~22-29 us per call, device-side, at every shape**:
  fast +0.0287 (0.0404->0.0691), proxy +0.0270 (0.2485->0.2755), prod +0.0222 (2.3406->2.3627).
- It is NOT cleared by a different FlyDSL kernel (toy shape, gqa=2 build) launched between
  the ASM call and the timed call (0.0685). It is NOT allocator placement: fixed pre-allocated
  out/lse passed straight to flash_attn_batch_m32x8 shows the same 0.0405 -> 0.0738, and
  torch.cuda.empty_cache() between changes nothing.
- Under `rocprofv3 --pmc` (dispatches serialized) the signature vanishes: SQC_ICACHE_MISSES
  read 0 for the FlyDSL kernel in both alone and interleaved runs, GRBM_GUI_ACTIVE equal
  (384.7k vs 381.5k). So the counters do not reach it.
- Working hypothesis (UNCONFIRMED): the ASM kernel's code (84 KB .co) evicts the FlyDSL
  kernel's code (~4300 instructions, ~25-30 KB) from the SQC instruction cache; the L2 flush
  in benchmark.py never touches I$, which is why "alone" is clean. Fits every observation
  above, but the one counter that should show it reads 0.
- **Why it matters for scoring**: validation.py measures candidate and beat interleaved in one
  process, so this penalty is IN the score: 42% of FlyDSL's fast time, ~10% of proxy, ~1% of prod.
  Isolated, the fast gap is 0.0404 vs 0.0313 = 1.29x, not 2x.
- SQ_WAVES (pmc): FlyDSL 256 waves = 32 WGs x 8; ASM 128 waves at fast. Both underfill 256 CUs
  at fast; underfill is not a FlyDSL-specific disadvantage.

### Proxy: dispatch-order tail (raw/dispatch_sim.txt) -- zero-GPU instrument
Kernel facts (ISA dump, raw not kept): LDS 327680 B/WG, scratch 0, next_free_vgpr 445,
waves_per_eu=2 hint, 8 waves/WG -> **1 WG per CU**. Grid = (ceil(sq*gqa/256), hkv, b), x fastest;
under causal, WG cost grows with block_x, so the hardware dispatches the heaviest tiles LAST in
every (y,z) group. Greedy list-scheduling of (kv tiles + fixed) per WG onto 256 CUs:

| shape | WGs | in-order / LPT makespan | reverse-x-only / LPT |
|---|---|---|---|
| fast  | 32   | 1.000 | 1.000 |
| proxy | 512  | 1.60-1.66 | 1.52-1.57 |
| prod  | 4096 | 1.074-1.078 | 1.052-1.056 |

This matches the survey's shape of surprise: proxy (2 WGs/CU) is exactly where in-order is worst
and proxy is where FlyDSL is furthest behind. Model is an upper bound (assumes FIFO dispatch,
uniform CUs, perfectly linear tile cost).

## Step 3 -- sources
- (a) pool/route: empty.
- (b) survey: proxy tail -> dispatch order; post-ASM penalty -> open mechanism.
- (c) corpus: flydsl/attention README, techniques.md (gfx950: issue-bound softmax, barrier
  co-residency, vmcnt placement -- none priced here without counters), dead-ends.md (skimmed),
  arch/gfx1250/gfx1250.md (LDS/WGP$ split, TDM), arch/gfx1250/profiling-surface.md (silent-zero
  counters). optimization/techniques/1-grid-and-cache-locality.md is the family for g01.
- (d) cross-backend recipes: all attention recipes (flydsl hd64/hd128, hipkittens gqa_d64/d128,
  aiter fmha_v3_bwd) are gfx950; only their arch fields were read. Not opened further because
  (a) was empty but (b) produced two measured leads this round; the next round that stalls
  should open hipkittens gqa_d128 section 6 for mechanisms.

## Step 4 -- choice
Two arms, independent lines (grid index remap vs O epilogue constant), so measured apart then merged:
- **A = r1.i1.g01 LPT dispatch order.** Expect prod +4-7%, proxy +20-45% (model says up to 1.6x;
  the real dispatcher is not strict FIFO and the ~27 us post-ASM penalty is untouched), fast 0.
  Geomean vs beat from ~0.557 to ~0.62-0.66.
- **B = r1.i2.g02 O_VARIANT v1.** Expect prod +1-2% (1.90% measured 2026-09-23), proxy/fast
  within noise. Risk: the 1.90% was taken before this job's harness; could be inside 0.7% floor.
- Merge only if neither lost.
Not pooled (reply caps pool additions at two), recorded here for the next round:
  * post-ASM penalty mechanism: ~27 us fixed, 42% of fast. Next test: put a large-code kernel
    that is not ASM (e.g. a hipBLASLt/Triton GEMM, or the FlyDSL bwd kernel) between, and build
    a FlyDSL variant with the 2x KV unroll removed to see if the penalty scales with code bytes.
  * fast isolated gap 1.29x at equal WG count (32): per-WG latency (prologue Q load, 16 KV tiles,
    epilogue); split-KV for grids < #CU would raise parallelism for both short shapes.

GPU after: idle, no KFD pids, no MES/fault lines in dmesg during the round's measurements.

---

## Resume (session 2, 2026-09-25 12:16 UTC)

The first opt session ended at 12:14 after writing pool.md/route.md and the log above; the
framework resumed the round and moved its dir to `1-opt.stale-20260925T120405`. Its opt.md and
raw/ are copied here unchanged (the section above is session 1's log, verbatim).

Checks this session, before touching anything:
- GPU: 0% use, `No KFD PIDs`, `docker exec fa-repro pgrep -af python` empty -> no orphan from
  session 1 holds the device. dmesg: no MES/fault lines, only apparmor noise.
- Re-read the g01 premise in code: `_packed_tile_indices` (:262-263) sets
  `seq = (block_x*BLOCK_M + ...) // gqa_ratio`, so under causal a WG's KV-tile count rises with
  block_x, and x is the fastest dispatch axis -> heavy tiles last. Line refs in the pool entry
  (:262-263, :778, :792, :1648) match the working copy. O_VARIANT is at :181 = "v3". Confirmed.
- No new card measurement this session: the survey numbers above are same-day, same container,
  and nothing changed in the code or the machine since.

Session 1 capped its pool additions at two and left two measured leads only in this file. They
are written into pool.md now as r1.i3.g03 (post-ASM ~22-29 us penalty -- a scoring-harness
term, 42% of fast's scored time) and r1.i4.g04 (split-KV for grids < #CU, fast shape 1.29x
isolated). Neither is the round's candidate; g03 is a zero-code instrument that the next round
should take alongside its build.

Route rewritten: must h1 on top; the two ideas that execute h1's steps 2a/3 directly below it
(rows 2-3 are what gets built); the advise rows follow, each with its gating condition. The
table exceeds the 4-row cap because rule 2 requires every outstanding operator item.

Expectation (checked by reflect): g01 alone -> proxy +20-40% (model upper bound 1.6x; real
dispatcher not strict FIFO), prod +3-6%, fast ~0; g02 alone -> prod +1-2%, proxy/fast in noise.
Merge only if neither lost beyond the session floor.

explored.consulted (both sessions): knowledge/backends/flydsl/attention/README.md,
techniques.md, dead-ends.md; knowledge/arch/gfx1250/gfx1250.md, profiling-surface.md;
knowledge/optimization/techniques/1-grid-and-cache-locality.md; arch fields of
backends/{flydsl,hipkittens,aiter}/attention/recipes/* (all gfx950 -> not opened further).

---

## Step 5 -- build (session 2)

Compile cache cleared first (`rm -rf /root/.flydsl/cache` in fa-repro; FlyDSL's key does include
a source hash -- jit_function.py:335 -- but cleared anyway).

**Arm A = r1.i1.g01 (working copy rounds/001/op).** Added `_lpt_block_id(axis)` in
fmha_fwd_prefill_a16w16_m32x8.py; every one of the 11 `fx.Int32(gpu.block_id(..))` reads
(packed tile indices, Q load block_x, KV range block_x, zero-fill, thd batch, bshd batch)
goes through it, so all consumers see the same triple. Uses the stable `fx.grid_dim`.
- Offline bijection gate (raw/g01_bijection.txt): OK on all 10 ut shapes; first dispatched
  WGs are x = gx-1 of every (y,z).
- COMPILE_ONLY descriptor (raw/descriptors.txt): VGPR 445 (=ctrl), SGPR 99 (ctrl 100),
  scratch 0, LDS 327680; +87 ISA lines (the runtime 32-bit divides). Descriptor differs from
  ctrl -> the arm really is a different binary.
- ut (raw/ut_A.txt): spec shapes PASS (51.23/50.89/50.83 dB), determinism 200/200 PASS,
  **7 edge cases FAIL at 49.82-49.99 dB -- identical to the 0.01 dB to the baseline's own
  failures recorded in history/review/op_setup_v000.md** (baseline fails the same 7 cases, same
  numbers). A block permutation cannot change a tile's arithmetic; this is the incumbent's
  pre-existing precision floor, not this arm's. Not weakened, not worked around: reported as
  precision fail, same as the incumbent.
  Attempts: 1, no build failure.

**Arm B = r1.i2.g02 (scratch copy of op/current, O_VARIANT "v1").** Compiles on 0.3.4.1 with
the V2 TDM loaders (the h3 `ptr_load` caveat did not bite). VGPR 449, SGPR 101, scratch 0.
ut (raw/ut_B.txt): same 7 edge failures at the same dB; determinism PASS with the SAME hash as
A (o=26a89a2db0cd) -> v1 epilogue writes bitwise-identical o/lse at fast.

## Step 6 -- measure

### Session 1 (raw/bench1.txt): benchmark.py, one process per shape, arms ctrl=rounds/000/op, A, B, beat
GPU before: 0% use, no KFD pids. dmesg monitor armed for the whole run: no events.

| shape | ctrl TF/s | A (g01) | A/ctrl | B (g02) | B/ctrl | beat |
|---|---|---|---|---|---|---|
| fast  | 38.58  | 35.40   | 0.918 | 38.88  | 1.008 | 68.36 |
| proxy | 530.07 | 850.05  | **1.604** | 513.87 | 0.969 | 1017.39 |
| prod  | 920.12 | 1018.30 | **1.107** | 934.86 | 1.016 | 1395.45 |
sclk fast 1100/1100, proxy 1029->1066, prod 1028->1047.

Read: g01 lands at the dispatch model's upper bound at proxy (model 1.60-1.66, measured 1.604)
and ABOVE it at prod (model 1.074-1.078, measured 1.107) -- the prod excess is not explained
by the model; candidates: tile cost not linear in KV tiles (epilogue/prologue), or L2 reuse
from concurrent WGs now sharing the same high-x Q/K ranges. B: prod +1.6% (prior said +1.9%),
proxy -3.1%, fast +0.8%. Fast is the noisy shape (post-ASM penalty, g03); A's -8% at fast has
no mechanism in the model (32 WGs all co-resident) beyond ~87 SALU for the divides (<1 us), so
it needs the repeat. B's geomean vs ctrl ~0.997: not resolved as a loss -> merge qualifies.
Merge AB built (armAB = working copy + O_VARIANT v1): VGPR 449, SGPR 99, scratch 0.

### Session 2 (raw/bench2.txt): same protocol, arms ctrl, A, B, AB (merge), beat
AB ut first (raw/ut_AB.txt): same 7 pre-existing edge failures at the same dB, determinism PASS
(same hash). GPU idle before and after.

| shape | ctrl | A | A/ctrl | B | B/ctrl | AB | AB/ctrl | beat |
|---|---|---|---|---|---|---|---|---|
| fast  | 47.32  | 44.20   | 0.934 | 44.75  | 0.946 | 43.63  | 0.922 | 66.99 |
| proxy | 549.77 | 813.38  | **1.479** | 525.76 | 0.956 | 749.43 | 1.363 | 1031.46 |
| prod  | 935.04 | 1015.37 | **1.086** | 943.52 | 1.009 | 1018.89 | 1.090 | 1397.30 |
sclk fast 1100, proxy 1029->1063, prod 1026->1043.

Spread (ctrl across the two sessions): fast 38.58 vs 47.32 (**22%**), proxy 3.7%, prod 1.6%.
- **A (g01) wins, twice**: proxy 1.604 / 1.479, prod 1.107 / 1.086.
- **B (g02) loses at proxy in both sessions** (-3.1%, -4.4%, same sign, above the prod-level
  floor), prod +1.6% / +0.9% (inside the prod cross-session spread; the prior +1.90% is not
  reproduced as a clear effect). Net: not a win at this harness.
- **AB (merge) is worse than A alone**: proxy 1.363 vs 1.479 (-7.9%), prod 1.090 vs 1.086 (+0.35%,
  noise), fast 0.922 vs 0.934. v1's proxy loss carries into the merge, amplified. Not shipped.
- **Fast**: every FlyDSL arm is below ctrl in session 2 and the ctrl itself moved 22% between
  sessions -- fast's median in benchmark.py is dominated by the post-ASM penalty (g03), which
  lands on whichever arm the palindromic order puts next to beat. Controlled probe
  (raw/fast_ab_probe.txt; flush + 400k-cycle sleep, arms alternated, 151 reps each):
  fast A/ctrl time 0.9989 (alone), 0.9972 (after beat) -> **g01 is neutral at fast**, as the
  model predicted (32 WGs, all co-resident). Same probe at proxy (cold L2 each call):
  A/ctrl time 0.763 / 0.744.
  Risk for acceptance: the benchmark-median fast ratio (0.918 / 0.934) is below the 95%-of-best
  rule; that is a harness position effect, and this round cannot fix it in code without a
  different mechanism (g03 is the instrument that would).

**Shipped: A (r1.i1.g01) alone** -- the working copy rounds/001/op, O_VARIANT left at v3.
Prediction vs measured: proxy predicted +20-40%, measured +48/+60% (beat my number; the model's
upper bound held). Prod predicted +3-6%, measured +8.6/+10.7% (above the model's 7.8% bound --
unexplained excess, see session 1 note). Fast predicted ~0, controlled probe 0; benchmark median -7%
(harness). g02 predicted prod +1-2%, measured +0.9/+1.6% at prod but -3/-4% at proxy.

### Own validation.py run on the working copy (raw/validation.txt): exit 2
- correctness: the same 7 edge-case o misses as the incumbent (49.82-49.99 dB vs the 50 dB bar;
  history/review/op_setup_v000.md records identical numbers for the baseline). Spec shapes pass,
  determinism 200/200 PASS.
- speed: fast 0.5286, proxy 0.7580, prod 0.7219 of beat -> geomean 0.6613 < bar 1.0000 (was ~0.557
  at the round-1 survey for the incumbent).
GPU after: 0% use, no KFD pids, no orphan processes; dmesg monitor: no events for the whole round.

## Step 7 -- verdict
Candidate r1.i1.g01, outcome delivered; left in the working copy (one file changed:
flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py). The precision gate fails exactly as it fails for
op/current -- this round neither caused nor fixed it; the operator decision the op-setup review
asked for (its section 7) is still outstanding and blocks any validation pass regardless of speed.
For the next round: h5's L3 pairing is now justified ("order pays" is measured: 1.48-1.60x at proxy);
the unexplained prod excess over the model (1.086-1.107 vs <=1.078) and the fast-shape harness
position effect (g03) are both worth a probe; g02/L1 should be recorded as not paying on this harness.
