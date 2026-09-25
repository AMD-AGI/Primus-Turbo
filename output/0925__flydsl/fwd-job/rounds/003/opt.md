# Round 3 (fast) -- opt log (planning step: survey + pool + route; builds are the next step's)

## Step 1 -- findings
- facts: g01 (LPT) landed round 2 (prod 1.089x, proxy 1.41x). Now 0.73 prod / 0.76 proxy / 0.64 fast
  of beat. Current-bottleneck line is LOW confidence: "presumed per-tile body efficiency", no counter
  ever taken on the body. The only body datum is an inference from a loss (g05 always-rescale -6..-10%).
- dead_ends: r2.i1.g05 (branch-free rescale, L20) dead for speed; precision lever at 6-10% price.
- pool: g02 (O v1 -- h20 says did not reproduce, retire), g03 (post-ASM I$ penalty, positive test
  round 2; next lever `s_prefetch_inst`), g04 (split-KV fast, multi-hour), g06 (L4/L5, closed by
  construction in round 2).
- route: both musts (h20, h1) were discharged in round 2 (h20 accepted; h1 steps 2a-2c + 3 each have
  a number or a reason). They still appear in the framework's outstanding table, so they go on top
  as "discharged" rows. No deep round has run -> no deep-born entry to pass over.

## Step 2 -- look
Scratch: /home/lihuzhan/.cache/op-evolve-r003-scratch (container sees only /home/lihuzhan; the /tmp
scratch path is a symlink to it). GPU before: 0% use, no KFD pids, no python in fa-repro. dmesg
monitor armed for the whole session.

Survey choice: `rocprofv3 --kernel-trace --stats` writes nothing in fa-repro (round-1 instrument
finding), and durations of every arm are already known per shape from round 2's same-day session.
The unanswered question in facts.md is WHAT THE BODY WAITS ON; gfx1250 has no stall/VALU/WMMA
counters (arch/gfx1250/profiling-surface.md: 13 of 51 counters live, none about stalls) but
stochastic PC sampling works and reports Stall_Reason per sample. So the survey is PC sampling.

Prediction before: if the body is VALU/issue-bound in softmax (the job's presumption, h4), the
samples should be dominated by VALU (v_exp/v_fma/v_max) issued or ARBITER_NOT_WIN, with WMMA
a minority and barrier/tensorcnt waits small. What would change my mind: a large share on
s_barrier_wait or s_wait_tensorcnt -> the body is sync/latency-bound, and softmax arithmetic
levers (L15-L19) are pricing the wrong term.

### Instrument trouble (recorded as it happened)
- PC sampling at **prod** (interval 4096, 20 calls): ran > 9 min at 856 W without writing a
  sample file; killed by explicit PID (SIGTERM ignored in kfd_wait_on_events, SIGKILL worked).
  GPU idle and no KFD pids afterwards. Cause: sample volume at prod is huge; not a hang of the op.
- PC sampling of the **beat ASM kernel** at fast (interval 1024, 20 calls): did not finish in
  400 s, killed by `timeout -s KILL` (rc 137), no CSV. The ASM kernel apparently does not
  complete under the stochastic sampler (fast FlyDSL finishes in seconds under identical
  settings). No beat-side PC-sampling comparison is available on this box.
- PC sampling of current at **proxy** (interval 8192 x2 calls; 262144 x1 call): the workload
  finished (`done` printed), but rocprofv3 never exited -- killed at 400 s / 240 s (rc 137). The
  hang is in teardown/flush once the grid spans more than a few dozen CUs or the sample buffer is
  large. Fast (32 WGs, ~65k samples) completes in seconds.
- Workaround: a body-dominated shape at 32 WGs (b1, sq 1024, skv 16384, hq 8, hkv 2, causal; 1 WG
  of 8 waves per CU, the same per-CU residency as prod) -- raw/pcs_long.py.

### r3.i3.g09 -- instrument: PC sampling on this kernel (raw/pcs_current_fast.txt, raw/pcs_current_long.txt)
| run | samples (op) | issued | s_barrier_wait | WMMA | VALU (all) | v_exp |
|---|---|---|---|---|---|---|
| fast, iv 1024, 20 calls | 45 395 | 15.4% | 63.5% | 4.0% | 13.8% | 1.1% |
| long, iv 32768, 2 calls | 48 091 | 7.9% | **85.0%** | 3.4% | 4.2% | 1.0% |
Taken at face value this says "the body waits on the WG barrier 85% of the time, WMMA 3%",
the opposite of the VALU-bound presumption. **It is not usable as a stall mix**: the same long
shape runs **0.468 ms unsampled vs 6.70 ms at interval 32768 and 25.1 ms at interval 1024** --
the sampler slows the kernel 14-54x. The likely mechanism is that a sampled wave is held while
its 7 WG siblings pile up at the per-tile `s_barrier_wait`, which manufactures exactly this
signature. So on gfx1250, for any kernel with a per-iteration WG barrier, PC-sampling
Stall_Reason shares are an artefact of the instrument. (The corpus page
arch/gfx1250/profiling-surface.md calls PC sampling "the one rich source on this part"; it
was validated on barrier-free kernels. That caveat should go back to it.)
What survives: the body issues very few instructions per sample in both runs, and nothing here
refutes or confirms VALU-bound. **Changed my plan**: body pricing has to come from arms/probes
(g08 + the floor probe), not from counters or sampling. ATT thread trace not tried (cost ~40 min,
and it presumably perturbs the same way; untested).
Prediction check: I predicted VALU-dominated samples; got barrier-dominated -- but the
perturbation measurement voids the comparison rather than confirming a surprise.

GPU after all probes: 0% use, no KFD pids; the killed rocprofv3 children were reaped (zombies only);
dmesg monitor: no events.

## Step 3 -- sources
- (a) pool/route: g03 said "next lever: s_prefetch_inst in the prologue, new id when taken" -> g07.
  g04 multi-hour, g06 closed, g02 retired.
- (b) survey: PC sampling unusable (above); round 1's interference table re-read: ASM after FlyDSL
  +0.9 us vs FlyDSL after ASM +26-29 us -> the ASM kernel's own s_prefetch_inst is the named
  difference (h4). `llvm-mc -mcpu=gfx1250` accepts `s_prefetch_inst_pc_rel 0x0, null, 31` and
  `s_setprio`; FlyDSL has `rocdl.s_setprio` but no prefetch_inst wrapper (inline asm needed).
- (c) corpus: optimization/routes/1-metrics-to-techniques.md (instruction-mix / issue rows: price VALU
  by conversion coefficient, not count); backends/flydsl/attention/README.md, techniques.md
  (softmax issue, "pipes partly additive" floor probe, two ways to overlap exp with MFMA, s_setprio
  regime-dependent, barrier count not the cost); arch/gfx1250/profiling-surface.md (13 live
  counters, PC sampling recipe).
- (d) cross-backend: hipkittens/attention/recipes/gqa_d128.md §6 (hand-pinned registers 1.72x; 8
  waves > 4 waves under compiler allocation; threshold-8 lazy rescale 5.3% -- agrees with our g05
  loss of 6-10%); flydsl/attention/recipes/hd128.md §6 b5 (fixed-zero softmax reference + MFMA row
  sum present in Turbo's forward -> L18/L21 priors). All gfx950; mechanisms only.

## Step 4 -- choice
- A = r3.i1.g07 (I$ prefetch). Expect: fast in validation order 0.0748 -> ~0.050-0.055 ms
  (+35-50%), proxy +5-15% in validation order, prod +0.5-1%; in benchmark.py's palindrome fast
  +3-6%, proxy/prod small. Gate: g03 probe on the arm first.
- B = r3.i2.g08 (LO/HI s_setprio stagger), from op/current, independent lines. Expect prod
  -2..+4%. Rider: VALU floor probe (throwaway) to price softmax VALU.
- Merge only if neither lost beyond the floor (~1.6% prod, ~4% proxy cross-session; same-session
  floor to be read in the build session).
- Deferred, not buildable in a round: L6 in-wave pipelining (VGPR 445 + ~64 of 512 -- spill wedge
  risk), L3 pairing, g04 split-KV.

## Step 5 -- route
findings/route.md `## Route` rewritten: h20, h1 (musts, both discharged in round 2 -> retire), then
g07, g08 (the two the next step builds), then the advise hints with their conditions.
Builds in this step: none (this step plans; the one throwaway was the instrument above).

## Step 6 -- build (turn 2: "build and measure what the route says")
Compile cache cleared first (`rm -rf /root/.flydsl/cache` in fa-repro, no live python). All runs through
fa-repro, detached with sentinels, one shape per process, GPU idle (no KFD PIDs) before each run,
dmesg monitor armed throughout (no events).

- **Arm A = r3.i1.g07** (working copy rounds/003/op): `_prefetch_kernel_code()` at the top of the bshd
  kernel body = 8x inline-asm `s_prefetch_inst_pc_rel <0x0..0x7000>, null, 31` (32 x 128 B = 4 KB each,
  32 KB total). First build compiled; no build failures.
- **Arm B = r3.i2.g08** (scratch copy of op/current, NOT on top of A): `rocdl.s_setprio(2)` after the
  K-burst fence for LO waves only, `rocdl.s_setprio(0)` + sched_barrier after the V ds_load burst
  (before softmax). raw/armB_g08.diff. First build compiled.
- **Rider (throwaway, wrong output)**: VALU floor = `pj = exp2(fma(...))` -> `pj = fma(...)`. raw/valu_floor.diff.

COMPILE (COMPILE_ONLY=1 + FLYDSL_DUMP_IR, fast shape; raw/descriptors.txt):
| arm | VGPR | SGPR | scratch | LDS | extra ops in ISA |
|---|---|---|---|---|---|
| A | 445 | 97 | 0 | 327680 | 8 s_prefetch_inst_pc_rel at entry (lines 61-99, before any load) |
| B | 445 | 97 | 0 | 327680 | s_setprio 2 at the QK WMMA head / s_setprio 0 after the V ds_load_tr burst, both LO loop bodies |
| floor | 442 | 99 | 0 | 327680 | v_exp_f32 264 -> 8, v_wmma 256 kept |
Code size of A (llvm-mc re-assembly): s_endpgm at 0x6BD0 -> 27.6 KB, so 32 KB of prefetch covers it all.

**Discovery while checking coverage**: the dumped .s sets `.amdhsa_inst_pref_size
instprefsize(.Lfunc_end0 - kernel)`; decoding the kernel descriptor in the compiled binary
(raw/binkd.py) gives **COMPUTE_PGM_RSRC3.INST_PREF_SIZE = 216** (x 128 B = the whole 27.6 KB). The
hardware already prefetches this kernel's entire code at wave launch -- incumbent included. The ASM
beat's descriptor has INST_PREF_SIZE = 255 (the field max, 32 KB) on 79 KB of code, which is why IT
adds explicit s_prefetch_inst for the rest. The premise of g07 ("FlyDSL does not prefetch, ASM does")
was wrong for this kernel size; found after building, recorded before measuring.

## Step 7 -- correctness
- ut (op/ut/test_correctness.py --determinism 200; raw/ut_A.txt, raw/ut_B.txt): rc 2 for both, the
  SAME 7 edge cases at 49.82-49.99 dB under ut's hard-coded GATE_DB 50.0 -- dB lines identical to
  round 2's incumbent ut run, and determinism hash o=26a89a2db0cd lse=d6ac8da1e101 identical to the
  incumbent: A and B are **bitwise identical in output to op/current** (neither touches arithmetic).
  Spec shapes PASS, determinism PASS. Same status round 2 recorded as `unit_tests: pass`.
- Own validation.py on rounds/003/op (raw/validation.txt): **exit 2**. correctness all 16 cases PASS
  at the spec's 49 dB (worst o 49.82 dB short_q full), determinism PASS; speed FAIL: fast 0.5215,
  proxy 0.7289, prod 0.7274 of beat, geomean 0.6515 < 1.0.

## Step 8 -- measure
### g03 probe first, as the route gate requires (raw/g03_probe_r3.py/.txt)
flush + 400k-cycle sleep before each timed call, inc/A/B interleaved per iteration, median of 101
(prod 51). v1 was flawed ("alone" was preceded by ANOTHER arm's binary, so it was not warm: fast alone
0.059 vs 0.042 in round 2); v2 warms "alone" with the same arm:
| shape | inc alone / after beat / penalty | A | B |
|---|---|---|---|
| fast  | 0.0607 / 0.0610 / +0.3 us | 0.0630 / 0.0642 / +1.2 us | 0.0684 / 0.0681 / -0.3 us |
| proxy | 0.1558 / 0.1846 / **+28.8 us** | 0.1560 / 0.1952 / **+39.2 us** | 0.1559 / 0.2083 / +52.4 us |
| prod  | 2.1153 / 2.1348 / +19.6 us | 2.1170 / 2.1469 / +30.0 us | 2.1147 / 2.1558 / +41.1 us |
Fast "alone" stays cold in a 3-arm probe (32 WGs warm only the CUs they land on; the other two
binaries evict the rest), so fast is not readable here. Proxy (512 WGs, every CU warmed) is the clean
point: **the penalty did not shrink with A** (29 -> 39 us; v1 read 38 -> 39 us; noise ~10 us). Gate failed.
Consistent with the INST_PREF_SIZE finding: code prefetch was already there, and the penalty survives it.

### benchmark.py, 3 sessions, arm order rotated so each FlyDSL arm holds each palindrome slot once
(raw/bench{1,2,3}_*.txt, raw/bench_agg.txt). Orders: S1 inc,A,B,beat / S2 A,B,inc,beat / S3 B,inc,A,beat.
Median of 101, 8 s warmup, unprofiled, sclk 1026-1100. TFLOP/s per session and mean:
| shape | inc | A | B | beat | A/inc | B/inc |
|---|---|---|---|---|---|---|
| fast  | 35.73/38.16/37.34 = 37.08 | 36.04/37.45/35.49 = 36.33 | 29.81/37.06/38.03 = 34.97 | 66.07 | 0.980 | 0.943 |
| proxy | 746.2/753.2/785.3 = 761.6 | 857.9/758.6/718.1 = 778.2 | 743.8/761.4/718.4 = 741.2 | 1033.8 | 1.022 | 0.973 |
| prod  | 1017.6/1016.8/1017.3 = 1017.2 | 1015.8/1017.0/1008.0 = 1013.6 | 1017.5/1018.4/1014.9 = 1016.9 | 1397.6 | 0.996 | 1.000 |
- Same-session floor read here: prod ~0.4% (inc 1016.8-1017.6, A's S3 -0.9% is the outlier); proxy and
  fast swing +-8% with palindrome slot (A 858 -> 718 by slot alone). Only the rotated mean is usable.
- Score (mean of ratio vs beat, target = beat, margin 0%): **A 0.6760**, inc 0.6752, B 0.6629.
  A vs inc +0.0008 -- below min_gain (0.005-0.007) and inside the floor. **Neither arm won.**
- Validation order (2-arm candidate+beat, fast, alternating processes; raw/validation_order_fast.txt):
  inc 0.0601 / 0.0584 ms, A 0.0591 / 0.0604 ms -- identical. The validation fast figure moved from
  0.0748 (round 2) to ~0.059 for BOTH, i.e. session drift, not g07.
- Merge A+B: **not built.** Rule: only if neither lost beyond the floor. B lost 5.7% at fast (though
  driven by S1's post-beat slot) and neither arm won anywhere, so a merge has no upside to measure.

### Rider: VALU floor probe (raw/valu_floor_prod.txt), prod, 2-arm session inc vs floor
inc 2.1134 ms, floor 1.8303 ms -> **-13.4% wall** from removing 128 v_exp_f32 per tile per wave (256 in
the two loop bodies), WMMA count unchanged. Softmax exp is ON the critical path at prod, not hidden
under WMMA. That is the overlap B tried to get by priority; B's prod 1.000 says s_setprio does not
produce it (either the WMMA port is not what the two waves contend for, or the per-tile WG barrier
re-aligns them anyway). Prices route rows 5-6: in-wave pipelining (L6) or cheaper exp (L15-L19) have up
to ~13% at prod to claim. Side note: inc reads 2.113 ms in this 2-arm session without the beat vs 2.16
in the 4-arm sessions with it -- the post-beat penalty is in the prod benchmark numbers too (~2%).

## Step 9 -- verdict
- Shipped: **A (r3.i1.g07)**, left in rounds/003/op as instructed (built, measured, lost/neutral):
  `outcome: delivered`. B kept in scratch, measured, recorded in act.yaml `arms`.
- Mechanism for the record: (1) gfx1250 kernel descriptors already carry INST_PREF_SIZE = code/128 B
  up to 255 lines (32 KB); any FlyDSL kernel under 32 KB is fully prefetched at launch, so explicit
  s_prefetch_inst in the prologue is a no-op for it; the post-ASM penalty (~20-40 us) is therefore not
  a cold code fetch and remains unexplained (per-CU state that a same-kernel full-grid call clears).
  (2) s_setprio LO/HI does not create QK/softmax overlap on this kernel, though the floor probe proves
  softmax exp costs 13% of prod wall.
