# Round 6 (fast) -- opt

Written as the round went. Raw data is in `raw/` next to this file; bulk data is in the scratch dir. Every run
went through container `fa-repro` on GPU0. Before and after each run, `rocm-smi --showpids` read "No KFD PIDs".
sclk read 1100 MHz at fast.

## 1. Reading

- I read `findings/facts.md`, `dead_ends.md`, `pool.md`, `route.md`, `rounds/005/4-reflect/reflect.md`,
  `rounds/005/1-profiling/profiling_summary.md`, its `kernel.yaml` and `2-kernel-profiling/.../analysis.md`,
  `rounds/005/2-plan/raw/warm_flush/{summary.md,probe.py,probe_host.py}`, and `rounds/003/1-opt/opt.md` (g07).
- Route state: round 5 executed only `r5.i1.g13`, which lost. The must item `h22` (-> `r5.i2.g14`) has never
  been built.
- The pool's fast item `r1.i3.g03` described an "idle-sensitive, current-only device-side penalty" with no
  mechanism. Round 5 profiling recorded "I-cache is not a factor". Round 3 recorded that the "post-ASM penalty is
  not a cold code fetch", based on `INST_PREF_SIZE=216`.
- I chose to spend this round's own measurement on g03. It is the one open question whose answer might be large.
  Its premise ("idle-sensitive") came from probes that never controlled which kernel ran before the measured one.

## 2. Survey

- `rocprofv3 --kernel-trace` / `--stats` wrote **no output files** on this stack: 3 attempts, rc=0, no CSV.
  Round 5 recorded the same (`kernel_trace_used: false  # forbidden on this stack`). Per-dispatch start/end
  timestamps come from `--pmc` mode instead.
- Round 5's kernel list stands, since `op/current` has not changed: one kernel per call; 94.4% of prod time and
  85.6% of fast time.

## 3. Instrument: the fast penalty (r1.i3.g03)

**Prediction before measuring:** the penalty sits before the kernel starts (dispatch or launch), and
`End-Start` stays near 37 us. **What would change my mind:** `End-Start` near 70 us.

### 3a. PMC, both arms interleaved in one process (`raw/idle/`)

- Every dispatch is serialized by the profiler.
- current: 63.5 us `End-Start` and 658k `GRBM_GUI_ACTIVE`, in every mode.
- beat: 26.9 us.
- **My prediction was wrong.** The penalty is inside the kernel's own begin..end, not before it.

### 3b. Is it idle? (`raw/idle3/`)

`_sleep` backlog hides host launch in every mode.

- **Current-only process:**
  - 39.1-39.5 us for every gap, from 20k cycles to 2M cycles (1.84 ms), and after a 2 ms host idle.
  - **There is no idle penalty.**
- **Mixed process (beat also run):**
  - After a short gap following beat's calls: 69.2 us.
  - Decaying with gap: 43.3 / 42.9 / 41.7 / 39.6 us.
  - Beat: 34.6-34.9 us everywhere.
- `raw/idle2/` is **void.** Single-call events there include host launch; beat read 56 us back-to-back.

### 3c. Which process factor? (`raw/idle6/`)

fast, 101 reps, pattern `sync; [beat]; [256MB zero_]; _sleep(1e5); current`.

| loadbeat | runbeat | big | current dev median us | p10 | p90 | host us |
|---|---|---|---:|---:|---:|---:|
| 0 | 0 | 0 | 39.9 | 39.2 | 40.9 | 15.5 |
| 1 | 0 | 0 | 39.7 | 39.0 | 42.1 | 15.6 |
| 1 | 1 | 0 | **73.4** | 45.9 | 74.2 | 16.7 |
| 0 | 0 | 1 | 38.9 | 38.5 | 41.7 | 15.5 |
| 1 | 1 | 1 | **69.1** | 47.3 | 72.4 | 18.2 |
| 0 | 0 | 0 | 39.6 | 39.1 | 42.2 | 15.6 |

**Reading:**

- Running beat is what matters. Importing it (the aiter module) does not.
- A 256 MB write (the benchmark's L2 flush) does not matter. Host launch does not matter.
- Output addresses do not matter either (`raw/idle7/`): 72-73 us across every o/lse address pair, with beat
  outputs kept alive or freed.
- Predecessor WG count is not the cause (`raw/idle5/`). In a current-only process, a `zero_` of 0-40 x 1024
  elements before the call leaves current at 38.6-39.5 us throughout.

### 3d. Counters: current-only vs beat-before-each-call (`raw/pmc/summary.txt`)

| process | current `End-Start` us | SQC_ICACHE_MISSES / dispatch | SQC_ICACHE_REQ | SQ_BUSY_CYCLES |
|---|---:|---:|---:|---:|
| current only | 33.1 | **0** | 152k | 12.5M |
| beat before each call | 63.6 | **808** | 152k | 20.9M |
| (beat itself, same run) | 25.7 | 180 | 174k | 10.7M |

- `SPI_RA_REQ_NO_ALLOC`, `CHC_REQ_READ` and `GL1C_*` read 0. Per `knowledge/arch/gfx1250/profiling-surface.md`,
  those are dead-zero counters on this card, **not measurements**.
- The only counter that moves with the state is `SQC_ICACHE_MISSES`: 0 -> 808. Instruction requests are
  unchanged.

**Mechanism reading.** This is inferred, not proven:

- Beat's 79 KB code image evicts current's 27.6 KB image from the per-WGP instruction cache. Current then
  refetches its code on demand.
- If the SQC counter samples one XCD of 8, then 808 x 8 = about 6.5k lines. Over 32 WGPs, that is about 200
  lines per WGP, roughly the whole 216-line image.
- At about 140 ns per serial L2 fill, 216 lines x 140 ns = about 30 us. That matches the penalty.
- Beat does not pay the same cost. Its misses are only 180, so current's smaller image does not evict it.

This **corrects** two earlier records:

- Round 5's "I-cache is not a factor" was measured in a current-only process: 0 misses, the wrong condition.
- Round 3's "`INST_PREF_SIZE` prefetches the whole image, so it is not a cold code fetch" was an inference. The
  miss count says the descriptor prefetch does not keep the misses away. g07's measured time also says explicit
  prologue prefetch does not hide them: proxy penalty 29 -> 39 us, noise about 10 us.

**Open parts:**

- The decay with gap in 3b is not explained. Code eviction does not decay with time.
- 32 WGs land on different WGPs from dispatch to dispatch, so which WGPs still hold current's code depends on
  placement history. That fits the decay and the unstable fast medians, but it is not measured.
- The 1-of-8-XCD sampling of the SQC counter is an assumption.

### 3e. How big is it at the scored shapes? (`raw/idle8/`)

Pattern as in 3c, 41 reps, two processes per state, alternating.

| shape | current-only us | beat-run process us | penalty | as TFLOP/s |
|---|---|---|---|---|
| fast | 39.2-39.9 | 69.1-73.4 | +30-34 us (1.8x) | 54.5 vs about 30 |
| proxy | 152.1 / 147.9 | 199.0 / 194.9 | **+47 us (+31%)** | about 916 vs about 697 |
| prod | 1987.8 / 1988.2 | 2025.7 / 2019.4 | **+34 us (+1.7%)** | about 1106 vs about 1088 |

**Consequences for the job:**

1. **`benchmark.py` measures current in the beat-run state.** Both arms share one process, in palindromic order,
   with a flush before each call. So every scored current number carries this penalty in part:
   - fast: one mode is 39 us and the other 73 us. The median of a bimodal set depends on slot order. This fits
     round 3's +-8% slot swings at fast and proxy.
   - proxy: about 25% of its time.
   - prod: about 1.7%.
2. **Beat does not pay the equivalent cost.** It is a real property of the two kernels and not a harness
   artefact. In production, attention also follows other kernels.
3. **g04 (split-KV at fast) is stale by its own agreed line.** Round 5's line: current warm within about 15% of
   beat warm means stale. Clean warm current is 39.2-39.9 us against beat's 34.6-34.9 us, a ratio of 1.13.
   Round 5's 1.19-1.24 was measured in the beat-run state.
4. **The biggest remaining scored term at fast and proxy is not the softmax chain; it is this penalty.** At prod
   it is about 1.7%, comparable to g14's predicted gain.

## 4. Choice

Pool state:

- `r5.i2.g14` (h22, must, deep-born in round 5): open, never built. It goes first; the must rule requires it.
- `r5.i3.g15` (deep-born, L13 closure, predicted null): still outranked.
- `r1.i3.g03`: rewritten with this round's evidence. The diagnostic question "which state and which counter" is
  answered.
- `r1.i4.g04`: stale by its agreed line (see 3e.3). I recommend retiring it to dead ends at reflect.
- New `r6.i1.g16`: a code-refetch lever. Build-first, but it has a discriminating instrument (below).

**Expectations:**

- g14: prod +1% (planner +1-2%, reviewer >=2%). Fast and proxy unresolvable, because both sit on the bimodal
  penalty; judge by prod only.
- g16: if its instrument confirms that the penalty scales with code lines fetched, a 30-40% smaller hot image
  should cut about 10-15 us at proxy (+5-8% proxy) and about 10 us at fast. That is my number to be wrong about.
  If the penalty does not scale with code size, g16 closes and g03 is parked with a named mechanism that has no
  kernel lever.

No code change was built this round. It was a design round, and the instrument used its budget.

## Explored / consulted

- `knowledge/INDEX.md`
- `knowledge/backends/flydsl/attention/README.md`
- `knowledge/backends/flydsl/attention/recipes/hd128.md` (grep)
- `knowledge/backends/flydsl/attention/techniques.md` (grep)
- `knowledge/backends/hipkittens/attention/recipes/gqa_d128.md` (grep, row-sum)
- `knowledge/optimization/routes/1-metrics-to-techniques.md` (grep)
- `knowledge/optimization/techniques/6-gfx1250-cdna5-mechanisms.md` (grep)
- `knowledge/arch/gfx1250/profiling-surface.md` (counter availability; the zero counters)
- `knowledge/pitfalls/measurement-traps.md` (arm order / interleave section)

## 5. Written

- `findings/pool.md` changes:
  - added `r6.i1.g16`, which carries the discriminator instrument and the prediction;
  - rewrote `r1.i3.g03` (answered, superseded);
  - marked `r1.i4.g04` stale (1.13 < 1.15);
  - updated the order comment.
- `findings/route.md` `## Route`: first written as 4 rows (advise items grouped in one cell). The Python checker reads one id per row, so it was rewritten as 13 rows: h22, g16, g15, then one row per advise item.
  1. h22 -> g14 (must)
  2. g16
  3. g15
  4. the grouped advise items, each with its reason
- Framework tables above line 480 are byte-identical to before.
- Top two for the next step: h22 (g14) and r6.i1.g16.

## Failures this round

- `rocprofv3 --kernel-trace` / `--stats` wrote nothing (3 tries). Used `--pmc` timestamps instead.
- Host /tmp is not visible in the container. Worked around with mkdir and `docker cp`.
- A foreground `sleep` was blocked by the harness. Replaced it with `timeout ... until test -f rc` polling.
- `raw/idle2` timing is void: it included host launch.
- The first `raw/idle8` run raised IndexError from a fixed percentile index on 41 samples. Fixed with len-based
  percentiles and re-ran; rc=0.
- idle4 results are inconsistent in a mixed process with mixed predecessors. Attributed to placement history;
  not used for conclusions.

## 6. Build and measure (steps 5-6): row 1, h22 -> r5.i2.g14

**Change:** hand port of h22's `SOFTMAX_LANE_ROWSUM` onto the round-4 champion.
- The champion's R=2 packed in-lane tree (g10) is kept as is.
- The per-tile `peer()` add is removed from `d_new`. `d` is now a per-lane partial row sum.
- In the epilogue, `d_red = d + _peer_x16(d)` is computed once and used for both O normalisation and LSE.
- For the sink path, `d_init` = 1 on the khalf==0 lane only.
- There is no toggle. File: `op/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py`, diff 82 lines.

**Build** (`raw/build/`):
- `rm -rf /root/.flydsl/cache` first, then a compile-only build of the prod and fast shapes.
- The ISA is the same for both shapes (md5 c88e0803).
- permlanex16: 16 -> **10**.
- VGPR 455, SGPR 99, spill 0/0, private segment 0.
- 4235 instructions.
- All gates met on the first build. No build failures.

**ut** (`raw/ut/`): rc 2. The 7 lines that fail ut's hard-coded 50 dB gate are the same lines, with the same dB,
as round 4's shipped champion (`rounds/004/1-opt/raw/ut_M.txt`).
- All 16 o/lse dB values are identical to champion at 2 decimals.
- Worst o is 49.82 dB (short_q full).
- Determinism: 200/200 at hash o=b50b3239f5d0 lse=c9ca02299945. The hash differs from champion, as expected,
  because the summation is reassociated.

**validation.py** (`raw/val/`): correctness 16/16 PASS at the spec's 49 dB; determinism PASS. rc 2 comes only from
the speed bar: single run, fast 30.9, proxy 688.4, prod 1062.7 TF/s.

**Measure** (`raw/measure/`): the compile cache was cleared, so r4 was rebuilt in this session. Each run is one
`benchmark.py` process with arms r6 = `rounds/006/op`, r4 = `rounds/004/op` (the incumbent, which is also the
champion on every shape) and beat. The GPU was idle before and after (no KFD PIDs), and the dmesg monitor
reported no events.

| shape | arm order | r6 TF/s | r4 TF/s | beat TF/s | r6/r4 | r6/beat |
|---|---|---:|---:|---:|---:|---:|
| fast | r6, r4, beat | 38.60 | 37.66 | 65.20 | 1.025 | 0.592 |
| proxy | r6, r4, beat | 794.55 | 803.11 | 1023.77 | 0.989 | 0.776 |
| prod | r6, r4, beat | 1065.79 | 1086.52 | 1400.40 | **0.981** | 0.761 |
| prod s2 | r4, r6, beat | 1063.29 | 1079.71 | 1398.87 | **0.985** | 0.760 |
| prod s3 | beat, r4, r6 | 1070.11 | 1089.89 | 1413.30 | **0.982** | 0.757 |

- **Verdict: g14 loses at prod**, by 1.5-1.9% in all three arm orders. sclk was within 1% between arms in each
  session.
- The prediction (planner +1-2%, reviewer >=2%) is false.
- The build-identity gate held: permlanex16 went 16 -> 10. So this is a real change that removed 6 cross-lane
  ops, and prod got slower.
  - The per-tile cross-lane row-sum add is **off the critical path**, as the plan's own falsification line
    states.
  - The static ISA points in one direction: against the champion ISA (`rounds/005/2-plan/raw/vgpr_probe/isa_ctl`,
    md5 7b078261), `s_set_vgpr_msb` goes 240 -> 322 (+34%) and `s_wait_*` 100 -> 91. Register allocation
    reshuffled. That is a gate, not a price (facts: correction 3). The cause of the loss is inferred, not
    measured.
- fast and proxy sit on the bimodal g16 penalty and do not decide anything.
- Score is 0.7097. For comparison, round 5's session measured the incumbent at 0.7197.
- **Kept in the working copy (`outcome: delivered`).** Acceptance will not promote it.

## 7. Row 2, r6.i1.g16: the discriminator (`raw/g16disc/`)

**Question:** does the beat-preceded penalty scale with the kernel's code size?

**Method:**
- Shape: fast, 101 reps.
- Pattern as in idle6: 256 MB zero_, `_sleep(1e5)`; `rb=1` means beat runs before every call.
- Two sessions, alternating builds.
- Builds, all from `rounds/<n>/op`, with no new code: r4 champion (4235 instructions), r5 g13 (5906 instructions,
  1.39x, past the 32 KB INST_PREF window), and r6 g14 (4235 instructions).

| build | alone (rb=0) us | beat-run (rb=1) us | penalty us |
|---|---|---|---|
| r4 | 39.7 / 39.9 | 71.2 / 69.8 | 31.5 / 29.9 |
| r5 (1.39x code) | 44.7 / 45.2 | 80.2 / 76.9 | 35.5 / 31.7 |
| r6 (same size as r4) | 39.8 / 39.9 | 68.0 / 76.9 | 28.2 / 37.0 |

**Result:**
- Proportional scaling predicts about 43 us for r5. It measured 33.6 us, only +2.9 us above r4. That is inside
  g16's own "+-3 us" falsification line, and inside the 9 us spread between two same-size builds.
- **g16 is closed without a kernel arm:** the penalty is not proportional to code size.
- The I-cache miss correlation from s3d still stands, but "shrink the image" is not its lever.
- What remains open, not measured: the penalty tracks something beat leaves behind that is roughly independent of
  current's image size, such as per-WGP state or a fixed number of first-touch lines.

## Failures (steps 5-7)

None in build: the first compile passed every gate. ut rc 2 is the pre-existing 50 dB hard-code (same 7 lines as
round 4). validation rc 2 is the speed bar only.
