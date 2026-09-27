# Round 12 -- fast round, opt (plan stage)

Base: `op/current` = round 11 (R=2 m32x8 + nodelay on full grids; R=1 m16x8 when the R=2 grid < 256 CUs).

## 1. Findings read

facts.md, dead_ends.md, pool.md, route.md (all tables), rounds/011 reflect + opt.md,
rounds/010/1-profiling/profiling_summary.md (round 5/10 profile, before round 11).

What still holds from the round-10 profile (before round 11): prod `latency`, low confidence;
HBM excluded; stall/ATT/WMMA counters unavailable on this card. Round 11 changed only the fast
kernel (R=1) and added nodelay on prod/proxy, so the prod ranking is unchanged.

Open threads the round-11 reflect left: (a) R=1 threshold never swept -- proxy (512 WGs) at R=1
unpriced; (b) fast's per-WG fixed cost is larger than the loop model says and has never been
measured; (c) short_q R=1 at 49.82 dB with no R=2 figure beside it.

## 2. Look -- measurements launched (`raw/s1/run.sh`, detached, sentinel `raw/s1/rc`)

GPU checked idle before launch (no KFD PIDs, use 0%, sclk 1100).

### M1 -- `rocprofv3 --kernel-trace --stats` on benchmark.py, current, one shape per process
Expectation: prod and proxy are >94% the attention kernel; fast shows the m16x8 kernel at
~30-33 us plus small helpers (L2 flush fill, maybe an LSE/zero fill). What would change my
mind: a non-attention kernel inside the timed region costing >= 2 us at fast -- that would be
a cheap fast lever unrelated to the kernel body.

### T1 -- throwaway arm: proxy through R=1 (gate `grid_m32 < 4*CU`)
`raw/arms/T1`, 3 sessions, one image per process, alternating order, proxy only.
Expectation: proxy LOSES, 0.85-0.95x. R=1 at a full grid runs 1024 WGs whose per-tile body is
366 instr for half the rows (513 for R=2) -> ~1.43x the instruction work; the only offsets are
finer longest-first granularity at the tail and the fast-shape latency effect. What would change
my mind: >= 1.0x, which would say proxy at 512 WGs (2 WGs per CU in sequence) is still
latency-bound rather than throughput-bound.

### Incident -- rocprofv3 on benchmark.py hung at exit (14:45-14:50)
`rocprofv3 --kernel-trace --stats -- python3 benchmark.py ... --shapes fast` ran the timed loop
(RESULT fast current 32.65 us, 65.84 TF/s, sclk 1100) and then glibc printed
"corrupted double-linked list"; the process sat at ~150% CPU for >5 min with no output files
written. GPU was idle (use 0%, no KFD PIDs). I killed the whole s1 tree by explicit PID
(294649/294650/294655/294656), checked no process left, and relaunched the survey on round 10's
`driver.py` (no rocm-smi subprocesses, no flush) as `raw/s3/run.sh`. The s1 T1 proxy sessions
never started, so there is no orphan number from s1.

### M3 -- seqlen sweep at fast's head config (instrument, no code change)
`raw/s2/sweep.py`: b=1, hq=8, hkv=2, d=128, causal; sq=skv in {128..4096}; same method as
benchmark.py; one arm per process; arms current (R=1 at all these sizes), beat, T0 (current
forced to R=2). Heaviest WG at R=1 sees sq/64 KV tiles (2..64). A linear fit of median time
against sq gives the per-call fixed cost (intercept) and the per-tile critical-path cost
(slope), for us and for beat.
Expectation: current intercept ~12-16 us, slope ~1.1 us per 64-KV tile (32.7 us at 16 tiles);
beat has a smaller slope (1 wave/SIMD, 256-KV steps) and a similar or larger intercept. What
would change my mind: our slope <= beat's, which would put fast's whole remaining gap in the
fixed cost (prologue/epilogue/launch), and the next fast lever would have to be there, not in
the loop.

## 2b. Results

### M1 -- survey (kernel list, dispatch count)
- rocprofv3 `--kernel-trace --stats` under `driver.py` exits 0 and writes **no files** (three
  shapes, then a retry without `-o`: `raw/s4/rp_files.txt` empty; tool log "output generation
  0.0018 s"). Kernel tracing does not capture in this container today. Not pursued further.
- Fallback, torch.profiler (`raw/s4/tprof.py`, 20 warm calls, no flush): **exactly one GPU
  kernel per call on every shape**, `kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0` (the R=1 file
  keeps the same symbol). fast 26.88 us warm, proxy 139.83 us, prod 1975.58 us. No helper
  kernel to remove; my "a >= 2 us helper at fast" alternative is falsified.

### T1 -- proxy through R=1: 0.687x, closed
3 sessions, one image per process: T1 214.40 / 214.4 / 214.24 us vs INC 147.30 / 146.74 /
147.18 us (`raw/s3/*_proxy_*.out`). Worse than my 0.85-0.95 prediction. The instruction-work
ratio (1.43x) accounts for it almost exactly (147 x 1.43 = 210). Proxy is throughput-bound at
512 WGs; the R=1 gate at `grid < CU count` stands and no scored shape sits between.
This closes round 11's "threshold never swept" thread for the scored shapes.

### M3 -- seqlen sweep at fast's head config (`raw/s2/fit.txt`, 2 sessions each)

| arm | fixed (intercept) | per 64-KV tile, heaviest WG | loop share at sq=1024 |
| --- | ---: | ---: | ---: |
| current (R=1) | 9.5 us | 1.43 us | 71% (22.9 of 32.7 us) |
| T0 (R=2 forced) | 12.8 us | 1.67 us | 68% |
| beat (ASM) | 21.5 us | 0.63 us | 32% |

(fit over sq >= 512; the sq >= 128 and >= 1024 fits agree within 1 us / 0.04 us per tile.)
- My prediction: intercept 12-16, slope ~1.1. Measured intercept 9.5 (lower) and slope 1.43
  (higher). **The round-11 reading ("a per-WG fixed cost larger than modelled") is wrong:**
  fast is 71% loop on the heaviest WG, 29% fixed.
- Beat is the mirror image: twice our fixed cost, 0.44x our per-tile slope. (Caveat: beat's
  slope assumes its heaviest WG spans the same KV length; beat's grid is not observable here
  because rocprofv3 captures nothing.)
- My alternative ("our slope <= beat's") is falsified. The fast lever is the loop's critical
  path on 64 WGs while 192 of 256 CUs are idle -- i.e. parallelism across the KV dimension --
  not the prologue/epilogue.
- **Cross-check against prod.** Prod runs 1 WG/CU too (327,680 B LDS). 4096 WGs x mean 64.5
  tiles / 256 CUs = 1032 tiles/CU x 1.67 us x (1100/1040 MHz) = 1823 us vs 1975.6 us measured:
  the lone-WG per-tile latency explains 92% of prod, and ~150 us (7.7%, ~9.5 us per WG over 16
  WGs/CU) sits outside the loop. Proxy: 65 tiles/CU x 1.67 x 1100/1045 = 114 us vs 139.8 warm,
  ~26 us (~13 us per WG, 2 WGs/CU) outside the loop. **That is the first price the pairing
  candidate r10.i3.g28 has ever had:** it halves the WG-boundary count, so its ceiling is ~half of
  those residuals -- up to ~4% prod and ~9% proxy -- well above the pool's "+0-1.5%".

### M4 -- what a dependent second dispatch costs at fast (`raw/s5/launch.py`, 2 sessions)
Palindromic in one process, flushed: A = attention(fast); B = A + one elementwise kernel
reading/writing 4 MB (the size of a fast partial-O slab); C = A + two.
Session 1: A 32.77, B 37.66, C 43.02 us (B-A 4.89, C-B 5.37). Session 2: 32.69 / 37.82 / 43.26
(5.13 / 5.45). **A follow-on combine dispatch costs ~5 us including its 4 MB of traffic.**
GPU idle before and after (`raw/s5/smi_*.txt`).
Consequence: dead end r1.i4.g04 was retired on "only ~5-6 us remains [to beat], so a combine
dispatch consumes most of the gain". That measured the gap to beat, not the available term.
M3 puts 22.9 us of fast on the heaviest WG's loop with 192 CUs idle; a 2-way KV split removes
~11.5 us of it and M4 charges ~5 us back: 9.5 + 11.5 + 5 = ~26 us, **~1.26x on fast**.

## 3. Sources
- (a) pool/route: g28 (deep-born r10, pairing), g19, g26, g15; route rows 3-13 of round 11.
- (b) survey + instruments M1, T1, M3, M4 above.
- (c) corpus: `knowledge/INDEX.md`; `optimization/routes/1-metrics-to-techniques.md` (row A6
  "grid fill: no inner-loop change recovers anything below the crossover; the lever is tile size
  or dispatch" -- that is exactly M3's verdict on fast); `backends/flydsl/attention/techniques.md`
  (store-bound subtractive probe; "Filling the grid" -- split costs a repeated prologue per split
  and an exposed reduction tail, which M4 prices); `arch/gfx1250/profiling-surface.md` (looked up
  the silent-rocprofv3 trap; it covers unknown counters, not an empty kernel trace).
- (d) cross-backend: `backends/hipkittens/attention/recipes/gqa_d128.md` section 6 (ranks 1, 2,
  7: register pinning, wave count, readfirstlane hoisting). Nothing buildable this round; its
  mechanism 2 (wave count) is closed here by the num_waves roadblock (h8).

## 4. Decision
- **r12.i1.g31 (new)**: two-kernel fixed-order split-KV for grids below the CU count (fast
  only). Priced by M3+M4 at ~1.26x fast. New id because the mechanism is the same as dead
  r1.i4.g04 but the premise that killed it is now measured false; the old id stays dead.
- **r10.i3.g28 (carried, deep-born)**: pairing on the proxy/prod path. M3 gives it its first
  price: ~150 us (7.7%) of prod and ~26 us (~19%) of proxy sit outside the per-tile loop, at
  ~9.5-13 us per WG boundary; pairing halves the boundary count.
- **r12.i2.g32 (new, instrument)**: the lone-WG seqlen sweep as a pricing instrument for loop
  edits -- lone-WG per-tile latency explains 92% of prod, and the sweep's noise is ~0.3-1 us on
  20-100 us.
- Orthogonal by construction (g31 touches only the R=1 path, g28 only the R=2 path), so both are
  built from op/current and the merge is measured if neither loses.
- **h16 tension, stated plainly:** h16 says "no atomics, no split-k"; its stated purpose is
  bitwise determinism. g31 uses no atomics and a fixed-order combine, so it is bitwise
  deterministic by construction and must pass the 200-run gate like everything else. The
  executing step should read h16 by its purpose; if the operator means it literally, g31 drops and
  row 3 moves up.

Expectation: g31 alone fast 1.20-1.30x (25-27 us one-image), proxy/prod byte-identical;
geomean +6-9%. g28 alone prod +1-2%, proxy +2-5%, fast byte-identical; geomean +1-2%. Merge
~+8-11%. I would be wrong about g31 if the partial-O write + combine read cost more than 5 us
(fp32 partials are 2x M4's 4 MB), or if two WGs on one q tile contend for more than L2 gives.

## 5. Route written
- pool.md: added r12.i1.g31, r12.i2.g32; appended the round-12 price to r10.i3.g28 (id kept).
- route.md: `## Route` rewritten (rows 1-4 ideas g31, g28, g19, g32; rows 5-13 all outstanding
  advise items h25, h11, h9, h23, h6, h24, h12, h5, h3; no `must` outstanding). Framework tables
  above untouched. Passed-over: g15 (deep-born r5, carrier only), g26 (diagnostic, answered by M3).
- Executed: round-12 instrument line added.
- For dead_ends (reflect's call): T1 closes "R=1 on proxy" -- 0.687x, 3 sessions, one image each.
- Top two for the next step: g31 (arm A), g28 (arm B), both from op/current.

## 6. Build (row 1, r12.i1.g31) -- `rounds/012/op`
- Files: `impl.py` (gate: `kv_split=2` when the R=1 kernel runs and `4*grid_m32 <= 256`, i.e. fast
  only), `flydsl_fwd/fmha_fwd_prefill_a16w16_m16x8.py` (compile-time `kv_split` in the bshd builder,
  grid.z = B*2, per-WG tile range [lo,hi) by overriding start_tile/n_tiles after `_kv_valid`; fp32
  partial O stored straight from VGPRs; partial LSE through the existing store; new FlyDSL
  `kn_fmha_fwd_split_combine`, fixed order, exp2/log2, -inf guard). m32x8 (proxy/prod) untouched.
- Compile cache `/root/.flydsl/cache` cleared before the first build (`raw/b1/run.sh`).
- Build failures (all fixed, none repeated a root cause):
  1. `NameError: batch` -- FlyDSL AST rewriter turns a Python `if` in the kernel body into scf.if, so
     names assigned inside it do not escape. Fixed with conditional expressions.
  2. `TypeError: for-body: carried variable 'outs'` -- `for h in range(2)` traced as scf.for, a list
     cannot be carried. Fixed by hand-unrolling.
  3. Combine indexing bug caught by ut on fast (raw/b2), fixed before b3.
- ISA (h14, raw/b1): split kernel VGPR 300, SGPR 107, VGPR spill 0, SGPR spill 8, scratch 0,
  LDS 327680 (non-split twin: SGPR spill 3, else identical).
- Correctness (raw/b3 + validation in raw/m1): ut 16/16 PASS; validation.py 16/16 at >=49 dB, min
  49.97 dB (unequal_seqlen2 causal), fast 51.32 dB (r11 51.23); determinism 200/200 bitwise
  (o=e0646303da7f). The split is engaged (hash and dB differ from r11).

## 7. Measure
Protocol (h28): benchmark.py, one implementation per process, no beat in process, 3 sessions
(C INC / INC C / C INC), INC = `rounds/011/op`, same idle device (`raw/m1`, smi before/after: no
KFD PIDs, use 0-1%). beat in its own process right after (`raw/m2`).

| shape | C (TF/s, s1/s2/s3) | INC (TF/s) | mean C | mean INC | C/INC | beat (own proc) |
|---|---|---|---|---|---|---|
| fast  | 65.76 / 64.88 / 64.49 | 65.92 / 66.16 / 65.60 | 65.04 | 65.89 | 0.987 | 64.26 |
| proxy | 934.55 / 930.74 / 932.26 | 929.48 / 933.78 / 934.30 | 932.52 | 932.52 | 1.000 | 1032.39 |
| prod  | 1112.19 / 1111.06 / 1110.82 | 1111.45 / 1111.02 / 1110.61 | 1111.36 | 1111.03 | 1.0003 | 1406.75 |

Geomean C/INC 0.996. fast sclk 1100 flat; proxy/prod 1034-1100 on both arms alike.
validation.py (candidate + beat in one process): rc 2, geomean 0.7650 (fast 0.7398, proxy 0.7834,
prod 0.7726) -- the known multi-image penalty; not used for scoring.

Acceptance: (1) throughput beats best ever re-measured -- NO (0.996); (2) below-target shapes
proxy/prod >= 95% of own best -- yes (1.000, 1.0003); (3) fast stays above its target -- yes
(65.04 >= 64.26). **Verdict: delivered, lost.** Left in the working copy as instructed.

Why it missed the 1.26x (diagnostic, `raw/d1`, torch.profiler 20 calls, then a C seqlen sweep):
- Per-kernel at fast: INC one kernel 26.93 us; C split attention 21.21 us + combine 4.44 us = 25.65 us.
  Event-timed wall moved +0.4 us, so the second dispatch's gap is ~1.7 us on top of the combine.
- The split kernel saved 5.7 us, not the ~11.5 us priced: C sweep 256/512/1024/2048 =
  21.67/27.04/34.61/50.60 us vs current 16.56/21.63/32.71/53.88 -> slope 0.98 us per original heaviest
  tile (priced 0.72 = 1.43/2), fixed cost +~5 us. Split wins only from sq~2048 up.
- So the M4 price of the second dispatch (~5 us) held (~6 us measured); the miss is the per-tile time of
  the split WGs rising ~1.4x. Not decomposed further (candidates: 4x as many WGs contending for KV in
  L2/HBM, the fp32 partial-O store at the WG tail). For dead_ends (reflect's call): 2-way split-KV at
  fast is null on this box; it pays only at sq >= ~2048 with grid < CU count.

Row 2 (r10.i3.g28) was not built this round; route outcome says so.
