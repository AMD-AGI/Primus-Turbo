# Round 8 (fast round) -- design step: read, survey attempt, pool, route

Assumption: this invocation writes the route. The reply contract names `top_two` as "the ids the next step
will build", so the next step does the A/B builds. No arm was built or measured here, and no kernel file in
rounds/008/op was edited.

## Read
- findings/{facts,dead_ends,pool,route}.md. op/current = round 6. Round 7's g18 was delivered but not
  accepted (x1.0030 < 0.70% bar).
- **New operator item h34 (must)**: stack A0 round 11 (nodelay + shape-gated BLOCK_M=128 m16x8 kernel)
  onto our champion. h28 (must) is still outstanding in the operator table.
- rounds/005/1-profiling/profiling_summary.md (round 5, before round 6): proxy/prod = **power**
  (1.0-1.27 GHz); fast = **latency / underfill** (32 WGs on 256 CUs, top clock, data-insensitive).
  Its c3 "fast per-WG critical path" is exactly what the m16x8 gate attacks: it doubles the WG count to 64
  and halves per-WG work. This holds after round 6: g14 changed the softmax, not the grid.
- Corpus: backends/flydsl/attention/ listing, backends/hipkittens/attention/recipes/gqa_d128.md section 6
  (the top mechanism is hand-pinned registers, 1.72x; not portable to FlyDSL's compiler allocation,
  not taken), optimization/routes/1-metrics-to-techniques.md (occupancy rows: "low occupancy alone is
  not a finding"; the fast shape is grid underfill, which is occupancy-drain/grid, technique
  1-grid-and-cache-locality).

## The h34 port, statically checked (no build)
A0 tree: `Primus-Turbo/output/0927__b0/fwd/a0_r11_champion/`. `diff -rq` against rounds/008/op:
- `impl.py`: +11 lines (import `_kern_m16`, `_NUM_CU` cache, gate `d==128 and grid_m32 < _NUM_CU`).
- `fmha_fwd_prefill_a16w16_m32x8.py`: the nodelay change is 2 lines (`"amdgpu-enable-delay-alu": False` in
  both `llvm_options` dicts, our :1953 and :2059). The rest of the file differs because A0's body is r4.
  It lacks our g14 speculative softmax, so the port is **2 lines, never the file**.
- new `fmha_fwd_prefill_a16w16_m16x8.py` (2315 lines, r4 body, carries nodelay itself). Copy as-is.
- buffer_ops / managers / kernels_common / tensor_shim: identical.
- Gate firing on B0 (fa-g0): `multi_processor_count` = **256**. grid_m32 = ceil(sq*4/256)*hkv*b:
  fast 16*2*1 = **32 -> m16x8**; proxy 64*8 = 512 and prod 128*8*4 = 4096 -> m32x8. So prod/proxy
  never see B, and fast never sees A's m32 change once B is in.

That makes the two arms nearly disjoint by shape:
- **Arm A = current + nodelay (2 lines)**. It moves prod/proxy (and fast, since fast is still on m32 in A).
- **Arm B = current + m16x8 gate (impl.py + new file)**, from current, NOT on A. prod/proxy stay
  byte-identical to current, and fast runs A0's m16x8 (r4 body + nodelay).
- The merge A+B = A0's exact recipe on our body. Build it only if neither arm lost beyond the floor.
Correctness for B: m16x8 has no speculative softmax, so its output is NOT bitwise == current at fast.
It must pass validation's gate (49 dB, lse) and the h16 adversarial suite at small grids, causal and
non-causal (verify_r6 scripts), and be run-to-run bitwise deterministic. A's hash should equal
current's (o=9eb9b58556bf lse=bbf23654e600): nodelay changes scheduling only, and if it does not, say so.

## Survey (rocprofv3), measured this round
- GPU 0 idle before (rocm-smi -d 0: 0% busy, no KFD pids).
- `rocprofv3 --kernel-trace --stats --output-format csv` (v1.3.2) on raw/survey_drv.py, fast, 50 calls,
  no beat, for current and for the A0 r11 tree: **rc 0, DONE printed, and no output files at all**, with
  and without `HARD_EXIT` (os._exit). So round 7's empty kernel-trace dir was NOT caused by os._exit.
  kernel-trace output generation is broken in fa-g0's rocprofv3. The `--pmc` form writes (rounds 5 and 7),
  and it is the instrument to use for dispatch lists here. Scripts: raw/survey.sh, raw/survey_drv.py.
  Bulk went to the container's /tmp/r008_survey{,2}; the host /tmp scratch is not visible inside fa-g0.
- The survey would only have named the one kernel. The descriptor/grid arithmetic above is the
  information the route needed.

## Pool
- r8.i1.g19 (new): port the g18 clean-tile speculative softmax into the m16x8 kernel. Only if B lands.
  See pool.md.

## Expectation (a number to be wrong about)
- A (nodelay): prod +0.3..+0.8% (A0 +0.7..0.83% on the r4 body; our g14 body has fewer VALU dependency
  chains for delay_alu to annotate, so less), proxy/fast within +-0.5%. Hash equal to current.
- B (m16x8 gate): fast +8..+15% no-beat vs current (A0 1.075-1.194x vs r4; our r6 lost 3.6% at fast vs
  r4, so there is a little extra room). prod/proxy exactly 1.000 (same kernel bytes). Gate form (with
  beat) could move differently, because m16x8 is a second kernel whose code beat can also evict.
- Merge: ~ A's prod + B's fast; score +2..+4% if B holds. This should clear the 0.70% bar where round 7 did not.

---

# Round 8 build step (row 1 = h34, row 2 = h28 protocol)

## Build (scratch ~/.cache/op-evolve-r008-scratch/{A,B,AB}, each rsync'd from job_context/op/current)
- A = current + `"amdgpu-enable-delay-alu": False` in both m32x8 `llvm_options` dicts (2 lines).
- B = current + A0's `fmha_fwd_prefill_a16w16_m16x8.py` (as-is) + A0's `impl.py`. A0's impl.py differs from
  ours ONLY by the gate (diff checked), so it is a drop-in.
- AB = both.
- The container's FlyDSL cache (/tmp/flycache) was cleared before the first build. Compile-only dumps use
  per-arm cache dirs.
- No build failures.

## ISA census (compile-only, raw/isa.sh -> raw/isa_census.txt)
| build@shape | VGPR | vgpr spill | scratch | sgpr spill | ISA lines | s_delay_alu | wmma |
|---|---|---|---|---|---|---|---|
| cur@prod (== cur@fast) | 448 | 0 | 0 | 60 | 6334 | 220 | 384 |
| A@prod | 448 | 0 | 0 | 60 | 6114 | **0** | 384 |
| B@fast (m16x8) | **300** | 0 | 0 | 3 | 3002 | 0 | 128 |
AB@prod is byte-identical to A@prod, and AB@fast to B@fast (cmp). So AB is exactly A on prod/proxy and
B on fast. B@fast shows VGPR 300, which proves the gate dispatches m16x8 at fast on this card.

## Correctness (raw/correct.sh; logs in container /tmp/r008_correct, copied to raw/)
- ut (`test_correctness.py --determinism 200`): **A, B, AB all PASS**, min o 49.97 dB on ut shapes.
  - A: 200/200 bitwise, hash o=9eb9b58556bf lse=bbf23654e600 = current's.
  - B/AB: 200/200 bitwise, hash **o=26a89a2db0cd lse=d6ac8da1e101 = round 4's hash** (facts r4.i1.g09). At fast,
    m16x8 reproduces r4's arithmetic exactly (r4 body, no speculative softmax; BLOCK_M does not change
    per-row op order).
- adv_bitwise (round 6's h16 script, 13 cases incl. big/ramp/outlier/late_high/big_ramp):
  - A bitwise == current on all 13.
  - B/AB bitwise differ at the small-grid cases (fast, sq_gt_skv late_high), exactly where m16x8 takes
    over, and report RESULT: FAIL on that bitwise criterion only. Their dB vs fp32 is identical to current's
    to 0.01 dB in every case, and all outputs are finite.
- verify_r6 adversarial suite (copied to scratch v8; slots: r6 := arm B, r4 := rounds/004, r6_as/r6_nvs
  := current), small grids only (toy, short_q, gqa4 x causal/full = the shapes that take m16x8):
  **192 cases, 0 FAIL** (144 PASS, 36 PASS(shared<49), 12 INFO). Max |Δo| and |Δlse| of B vs r4 = 0
  (bitwise == r4 in all scored cases). raw/adv_suite_B_results.md. proxy/prod were not run for B:
  they dispatch byte-identical m32x8 ISA (census).
- No new amdgpu dmesg lines during the build or correctness runs (monitor baseline 430).

## Measurement (raw/bench_r8.sh -> raw/bench/, raw/bench_r8_agg.txt, raw/score_r8.txt)
Batch 1: no beat in the process (h31). 6 arms (A, B, AB, r4, r6, r7), median of 101, one process per shape,
3 sessions with cyclic arm rotation. Batch 2: gate form (X + beat), one process per arm and shape.
GPU 0 idle before and after (rocm-smi -d 0: 0%; the only other KFD pid, 1378269, has queues on another
card, gpuid 57865, not 34992). sclk: fast 2305-2315, proxy 1292->1557, prod 1242->1278, the same for all
arms in a process. dmesg: no new amdgpu lines through the whole round.

| arm | fast | proxy | prod | fast vs r6 | proxy vs r6 | prod vs r6 | score (vs beat) |
|---|---|---|---|---|---|---|---|
| A (nodelay) | 107.44 | 1563.13 | 1482.81 | 0.998 | 0.986 | **0.9835** (0.986/0.985/0.980) | 0.8231 |
| **B (m16x8 gate)** | **139.33** | 1576.48 | 1509.79 | **1.294** (1.288/1.378/1.221) | 0.995 | 1.0014 | **0.9003** |
| AB | 138.50 | 1575.97 | 1487.05 | 1.286 | 0.994 | 0.986 | 0.8942 |
| r4 | 113.32 | 1550.65 | 1430.54 | 1.052 | 0.978 | 0.949 | 0.8237 |
| r6 (incumbent) | 107.69 | 1585.00 | 1507.68 | 1 | 1 | 1 | 0.8327 |
| r7 | 113.79 | 1596.47 | 1512.44 | 1.057 | 1.007 | 1.003 | 0.8491 |
beat (mean over the 6 gate-form processes): fast 152.97, proxy 1665.02, prod 1790.51.
Gate-form X/beat: B .833/.933/.776 vs r6 .622/.931/.775, AB .839/.920/.762, A .631/.908/.762.

Reading:
- **B wins fast by ~29% and costs nothing at prod.** Its prod/proxy ISA is byte-identical to r6's, so
  the x1.0014/x0.9946 are the A/A floor; proxy's per-session spread in this 6-arm process is +-6%.
  B vs the best champion per shape: fast x1.224 (r7), proxy 0.9875 (r7), prod 0.9982 (r7). All are
  >= 95% of best, and throughput geomean vs r6 is 1.088.
- **A (nodelay) loses prod 1.65% in 3/3 sessions**, beyond the 0.4% floor, and is null at fast/proxy. It
  deletes all 220 s_delay_alu (ISA 6334 -> 6114 lines) and does not help. On the r6 speculative-softmax
  body nodelay is a loss; A0 measured +0.7% on the r4 body. Mechanism unmeasured. A guess only: in the
  power regime the delay hints keep the dependent VALU from issuing into stalls that cost energy.
- By the two-arm rule the merge should have been skipped, because A lost. It was already in the same
  sweep, so it is reported: it carries A's prod loss (0.986) and B's fast gain.
- Fast now sits at 0.91 of beat with no beat in the process, and 0.83 in gate form. This is the
  underfill term round 5 named (c3). m16x8 doubles the grid to 64 WGs.

## Expectation vs measurement
- Said: A prod +0.3..0.8%, hash == current. Got: hash == current, **prod -1.65%**. Wrong sign.
- Said: B fast +8..15%, prod/proxy 1.000. Got: **fast +29%**, prod/proxy 1.000 within floor. Twice the top of the range.
- Said: merge score +2..4%. Got: B alone score .9003 vs .8327 (+8.1%); AB below B.

## Validation (own run, working copy = arm B)
`op/validation.py rounds/008/op`: rc **2**. All 16 correctness cases PASS; min o 49.82 dB (short_q full,
the same as the champion); lse >= 81 dB; determinism 200/200 (o=26a89a2db0cd). The speed bar FAILS:
geomean candidate/beat 0.8368 < 1.0 (fast 0.8176, proxy 0.9311, prod 0.7697 in gate form). This is the
same bar every round has failed. raw/val_wc.txt.

## What is in the working copy
rounds/008/op = arm B: `impl.py` (A0's, = ours + the 11-line gate) + new `flydsl_fwd/fmha_fwd_prefill_a16w16_m16x8.py`.
The m32x8 kernel is untouched, and nodelay is NOT in it (A lost).
- Note: A0's m16x8 file carries nodelay in its own llvm_options. It runs only at fast, where A's nodelay was null on m32x8, so B's fast number is m16x8 as A0 built it. Not separated this round.
