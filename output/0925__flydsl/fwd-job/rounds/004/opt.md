# Round 4 (fast) -- opt log (planning act: survey + probes + pool + route)

Note: a previous attempt at this round (2026-09-25 14:02, dir 1-opt.stale-20260925T140210) died
after 5 minutes with an empty raw/; its only trace is a scratch copy `coexec/` = op/current with
`"amdgpu-sched-strategy": "coexec"` enabled in both `_ensure_*_kernel`. Nothing measured. Reused below.

## Step 1 -- findings
- facts: bottleneck (medium conf): softmax exp VALU is ON the prod critical path (floor probe:
  exp2 removed, WMMA kept -> -13.4% wall). s_setprio stagger (g08) did not buy overlap (prod 1.000).
  Post-beat ~20-40 us penalty is not a cold code fetch (g07 dead; INST_PREF_SIZE=216 covers the code).
- dead_ends: g05 (always-rescale, -6..-10%), g07 (I$ prefetch).
- pool: g02 (O v1, lost -3/-4% proxy r1 -> retire), g03 (post-beat penalty instrument, cause open),
  g04 (split-KV fast, multi-hour), g06 (L4/L5 closed by construction), g08 (**measured lost/neutral in
  round 3, prod 1.000 -- still marked open in pool.md; this round marks it closed**).
- route: h20/h1 musts discharged in round 2; framework still lists them -> top rows as "discharged".
  No deep round has run -> no deep-born entry to pass over.

## Step 2 -- look
GPU before: 0% use, no KFD PIDs, no python in fa-repro. dmesg monitor armed all session.

### Survey: rocprofv3 --kernel-trace --stats on benchmark.py (fast, cur + beat, 21 iters)
Prediction: one FlyDSL kernel + one ASM kernel per call, nothing else on device.
Result: **the tool crashes in teardown** (`corrupted double-linked list`), process then hangs; no
output dir written; killed by explicit PID 2838 (rc 137), GPU idle afterwards. Same as round 1/3's
"kernel-trace writes nothing in fa-repro". The benchmark lines printed before the crash:
fast cur 0.04995 ms (43.0 TF/s) vs beat 0.02984 ms (72.0 TF/s), sclk 1100. Kernel list is known from
earlier rounds (one dispatch per call). Survey is not a usable instrument in this container.

### Reading the ISA (round-3 dump of the incumbent body, identical to op/current's loop)
Clean-loop blocks of the incumbent (k.s lines 566-1154), instruction histogram per block:
| block | contents |
|---|---|
| PV block (loop head after rotation) | 32 v_wmma, 32 v_cvt_pk_bf16_f32, 8 v_nop, SALU |
| barrier + TDM prefetch | s_barrier_signal/wait, 2 tensor_load_to_lds |
| QK + V-load + softmax block | 32 v_wmma, 32 ds_load_b128 (K), 32 ds_load_tr16 (V), 65 v_exp, 58 fmamk, 30 v_max3, **39 add (row-sum tree)**, 4 permlanex16, 30 v_nop, 31 s_delay_alu, ends in `s_cbranch_vccz` |
| rescale (taken rarely) | 32 v_pk_mul_f32 |
Observation: the row-sum tree (~39 VALU adds + 2 permlanex16 + d update per tile per wave) is
emitted in the softmax block, which ENDS with the deferred-rescale branch. LLVM's machine scheduler
works per basic block, so those adds can never move into the PV block where 32 independent WMMAs
could hide them in their co-exec slots (arch/gfx1250/isa.md: 4 co-exec slots after each bf16 WMMA).
The PV block has only the 32 cvt as filler. p (bf16) is all PV needs; the row sum feeds only d.
Pricing from the floor probe: 128 v_exp per 2 waves per tile (2 issue cycles each, routes table
"v_exp 2.00") = 256 VALU-cycles -> -13.4% of ~2270 SIMD-cycles per tile-pair = ~304 cycles:
conversion ~1.2 wall-cycles per VALU-cycle removed, i.e. the softmax block is fully serial at the
SIMD (the two waves of a SIMD are phase-locked by the per-tile WG barrier; 1 WG/CU so no
hardware wave switching -- flydsl attention techniques.md "two ways to overlap").
So ~80 VALU cycles/wave moved into WMMA shadow is worth up to ~5-7% at prod if the conversion holds.

## Probes taken this act (throwaways in /tmp/op-evolve-...-r004 = ~/.cache/op-evolve-r004-scratch; none in the working copy)
Compile: COMPILE_ONLY=1 + FLYDSL_DUMP_IR, fast shape, flydsl cache cleared; raw/descriptors.txt.
Timing: benchmark.py, 2 arms (cur = job_context/op/current, probe), median of 101, 8 s warmup, one
shape per process, two sessions with arm order swapped; GPU idle (no KFD PIDs) before and after
each batch; dmesg monitor armed, no events. raw/bench_probes.txt.

### r4.i3.g11 -- sumpv: row-sum tree + d update emitted after the PV GEMM (past the rescale branch)
Prediction: bitwise-identical output, the ~39 adds + 2 permlanes move into the PV block and hide in
WMMA co-exec slots: prod +2-5%. What would change my mind: a loss = the PV block has no free slack.
Compile: VGPR 456, scratch 0. **Surprise in the ISA**: LLVM sank not only the sum but the whole
p = exp2(fma) computation (64 v_exp + fma per tile) into the PV block, because after the change every
consumer of the exps lives there. The softmax block kept only the max tree + ballot (v_nop 168 -> 197).
Measured: prod cur 2.1159/2.1157 vs sumpv 2.1652/2.1632 ms -> **-2.3%** (both orders);
proxy 0.15587/0.15527 vs 0.1584/0.15747 -> **-1.5%/-1.4%**. Beyond the ~0.4% prod floor: LOST.
Reading: the PV WMMAs depend on those exps (p -> cvt -> WMMA), so moving exps next to them creates
dependency stalls, not co-issue; and the V ds_load_tr burst (issued before softmax precisely to hide
under softmax VALU) lost its cover. Intra-tile relocation cannot manufacture overlap here: every
VALU in a tile is either upstream of that tile's PV or upstream of nothing that has slack. Overlap
needs cross-tile independence (L6). Prediction failed. Not a pool entry (dead; for reflect).

### r4.i4.g12 -- LLVM `amdgpu-sched-strategy=coexec` (the stale attempt's probe), compile only
ISA: v_nop 168 -> 57, s_delay_alu 224 -> 148, fmamk re-paired into 82 v_dual_fmamk -- the strategy
does fill co-exec slots. But **sgpr_spill_count 3** (3 v_writelane / 9 v_readlane, SGPR 105) and
private_segment_fixed_size 16. No scratch_ memory instruction in the body (spills live in VGPR
lanes), yet h14 says "spill 0/0 before any card run" and a wedge costs a human power cycle, so it
was **not run on the card**. The source comment already says "revisit after named barrier".
Not taken into the pool this round (cap two); condition to run it: spill 0/0 (e.g. on top of g09,
whose packing changes SGPR/VGPR allocation -- compile-check first).

### r4.i1.g09 -- packed exp argument: v_pk_fma_f32 for s*log2e - m*log2e (L15, h10)
Change (raw/pkfma_g09.diff): in `_softmax` pass 2, the scalar `fma(s, log2e, neg_m)` per element
becomes one `fma` on vector<2xf32> per element pair (l2/n2 broadcast). 15 lines, one place.
Prediction: pk_fma is exact per lane -> bitwise identical output; 234 fmamk -> ~117 pk_fma;
at the floor probe's ~1 wall-cycle per VALU-cycle, prod +1.5-3%.
Compile: VGPR 446, SGPR 98, scratch 0, spill 0/0. ISA: 207 v_fmamk + 27 v_dual_fmamk -> **128
v_pk_fma_f32**; v_nop 168 -> 75, s_delay_alu 224 -> 191 (the packed op also removed dependency
nops between fma and exp); v_exp 264 and v_wmma 256 unchanged.
Correctness (op/ut/test_correctness.py --determinism 200; raw/ut_pkfma.txt): **bitwise identical to
the incumbent** -- determinism hash o=26a89a2db0cd lse=d6ac8da1e101 (same as round 2/3 incumbent),
every dB line identical; the 7 edge cases at 49.82-49.99 dB fail only ut's hard-coded 50 dB gate,
the spec gate is 49 dB (pass), as for the incumbent.
Measured: prod cur 2.12069 / 2.12426 vs pkfma 2.06245 / 2.06321 ms -> **+2.82% / +2.96%**;
proxy 0.15491 / 0.15591 vs 0.15099 / 0.15255 -> **+2.60% / +2.20%** (sclk 1039-1069).
Both orders agree; prod floor ~0.4% -> a resolved win at prod and proxy. Fast not measured (its
+-8% slot swing needs the 3-session rotation). Prediction held (upper end).
Coefficient: ~26 fmamk + ~23 v_nop removed per tile per wave -> ~2.9% of wall, i.e. ~0.65 wall
cycles per issue slot removed. The softmax block is issue-serial; every VALU slot removed pays ~2/3.

### Instruments not available
- rocprofv3 --kernel-trace --stats: crashes in teardown in fa-repro (raw/survey_rocprofv3.txt).
- PC sampling: void on this kernel (round 3, 14-54x perturbation). Counters: no stall/VALU/WMMA
  counters on gfx1250 (profiling-surface.md). So arms + subtractive probes are the instrument.

## Step 3 -- sources
- (a) pool/route: nothing open is buildable (g02 retired, g03 cause open, g04 multi-hour, g06 closed,
  g08 measured null in round 3). Pool effectively empty -> (c) and (d) mandatory.
- (b) survey/ISA: the softmax block is issue-serial (floor probe conversion ~1.2); scalar fmamk and
  its v_nops are the cheapest VALU to remove.
- (c) corpus: optimization/routes/1-metrics-to-techniques.md (instruction-mix row: "fewer VALU
  instructions, weighted by cost"; v_exp cost 2.0); backends/flydsl/attention/techniques.md (pipes
  partly additive / conversion coefficient; two ways to overlap exp -- hardware wave switching needs
  occupancy >= 2 from different WGs, we have 1 WG/CU so only in-wave overlap is available; "count what
  the destination is hiding before moving work into it" -- which is exactly why g11 lost);
  arch/gfx1250/isa.md (WMMA co-exec slots: 4 after bf16 WMMA, 1 v_nop with co-exec disabled);
  backends/flydsl/attention/dead-ends.md (single-buffer handoff, co-exec +13% -- bwd-context).
- (d) cross-backend: hipkittens/attention/recipes/gqa_d128.md §6 headings (hand-pinned registers
  1.72x, 8 waves > 4 under compiler allocation, lazy rescale 5.3%) -- nothing new for this step
  beyond what round 3 took; the ASM beat's own `v_pk_fma_f32` exp argument (h4) is the direct source
  of g09.

## Step 4 -- choice for the next step
- A = **r4.i1.g09** (packed exp argument). Already built and priced as a throwaway here: +2.8-3.0%
  prod, +2.2-2.6% proxy, bitwise identical. Next step: rebuild it into rounds/004/op from op/current
  (apply raw/pkfma_g09.diff), 3 rotated sessions incl. fast, validation.py.
- B = **r4.i2.g10** (packed row-sum tree, v_pk_add_f32, L16 on gfx1250), from op/current, different
  lines (sum tree, not pass 2). Same mechanism g09 just proved on this part: fewer VALU issue slots
  in a serial softmax block, ~0.65 wall-cycle per slot. ~39 adds/tile/wave -> ~20 pk_add. Caveat:
  the sum order changes if pairs are reassociated -> NOT bitwise identical, check dB (49 dB gate with
  0 headroom on short_q full 49.82). Keep the tree's pairing identical (pack two independent tree
  nodes per op) and the output stays bitwise; that is the build requirement.
  Expected +1-2% prod alone; merge with A only if neither lost.
- Deferred (cannot build in a round): L6 cross-tile pipelining. g11's loss is new evidence FOR it:
  intra-tile relocation cannot create overlap; only independent work from another tile can.
  Blocked on VGPR (446 + ~64 S regs of 512) and on K(j+1) residency (prefetch depth).
- g12 (coexec strategy): condition spill 0/0; compile-check it on top of the winner next round.

Expectation for the next step: A reproduces +2-3% prod / +2% proxy, fast ~0 (VALU-bound only in the
body); score ~0.676 -> ~0.69. B +1-2% alone; A+B ~+3-4% if additive (it may not be: once A removed the
fma nops, the sum tree may be the only filler between exps).

## Step 5 -- route

`job_context/findings/route.md` `## Route` rewritten (framework tables above `<!-- /operator-tables -->` untouched).
Order: h20, h1 (must, discharged -- retire) > r4.i1.g09 (arm A) > r4.i2.g10 (arm B) > advise h10, h6, h12 (incl. coexec g12, spill-gated), h11, h9, h5, h3.
Top two built next: g09, g10, each from op/current; merge only if neither lost. No deep-born entry exists to pass over.

## Step 6 -- build (turn 2: "build and measure what the route says")
Compile cache cleared first (`rm -rf /root/.flydsl/cache` in fa-repro). Note: another session's
compile-only `compile_isa.py` (Primus-Turbo, pid 62519, not GPU) was live in the same container at
that moment -- not mine, not touched; it may have had to recompile. All runs through fa-repro,
detached with sentinels, one shape per process, GPU idle (no KFD PIDs) before each, dmesg monitor armed.

- **Arm A = r4.i1.g09** (working copy rounds/004/op): raw/pkfma_g09.diff applied to op/current. Source
  byte-identical to the round's probe; compiled ISA (22_final_isa.s) **byte-identical** to the probe's.
  VGPR 446, SGPR 100, spill 0/0, private segment 0. 128 v_pk_fma_f32, 0 v_fmamk (was 207), v_nop 168->75,
  4056 -> 3794 VALU/SALU lines (-262).
- **Arm B = r4.i2.g10** (scratch `pkadd`, from op/current, NOT on A; raw/pkadd_g10.diff): for R==2 the two
  rows' sum trees (identical shape) are run in lockstep as ONE v2 tree of (row0, row1) leaves through
  `_tree_reduce_multi`, adds on v2<f32> with the same no-reassoc flags -> per-row association unchanged.
  ISA: 126 v_pk_add_f32, v_add_f32 86 -> 0, v_dual 259 -> 164, v_mov_b32 15 -> 23 (+8: pair-building is
  cheap, the RA placed most exp results in aligned pairs), 4056 -> 3996 lines (-60), VGPR 460, SGPR 99,
  spill 0/0. Fewer slots removed than predicted -- the scalar adds were partly already dual-issued
  (v_dual_add), so packing them buys less than A's fmamk -> pk_fma.
- Build failures: none (both compiled first time).

## Step 7 -- correctness
- ut (raw/ut_A.txt, raw/ut_B.txt): rc 2 for both, the SAME 7 edge cases at 49.82-49.99 dB under ut's
  hard-coded 50 dB, dB lines identical to the incumbent's, determinism 200/200 with hash
  o=26a89a2db0cd lse=d6ac8da1e101 == incumbent: **both arms bitwise identical in output to op/current**.
  Spec shapes and determinism PASS -- the status rounds 2-3 recorded as `unit_tests: pass`.
- validation.py on rounds/004/op (raw/val_A.txt): **exit 2**. correctness 16/16 PASS at the spec's 49 dB
  (worst o 49.82 short_q full), determinism PASS, speed FAIL: fast 0.5388, proxy 0.7563, prod 0.7537,
  geomean 0.6747 < 1.0.

## Step 8 -- measure
### Batch 1: A and B separately, 3 sessions, arm order rotated cyclically (raw/bench{1,2,3}_*, raw/bench_agg.txt)
Orders: S1 inc,A,B,r3 / S2 A,B,r3,inc / S3 B,r3,inc,A, beat last each time; inc = rounds/002/op (== op/current),
r3 = rounds/003/op, beat = op/beat. Median of 101, 8 s warmup, unprofiled, one process per shape. GPU idle before/after.
| shape | inc | r3 | A (g09) | B (g10) | beat | A/inc | B/inc |
|---|---|---|---|---|---|---|---|
| fast  | 47.24/44.60/45.32 = 45.72 | 44.57/44.83/44.13 = 44.51 | 45.28/48.30/46.18 = 46.59 | 45.13/45.13/47.03 = 45.76 | 71.55 | 1.019 | 1.001 |
| proxy | 877.4/747.6/761.6 = 795.6 | 755.7/755.7/807.3 = 772.9 | 787.1/905.5/783.8 = 825.5 | 753.7/773.4/884.0 = 803.7 | 1032.6 | 1.038 | 1.010 |
| prod  | 1025.7/1016.7/1017.3 = 1019.9 | 1014.1/1017.2/1020.1 = 1017.1 | 1050.3/1054.6/1045.4 = 1050.1 | 1030.9/1031.1/1038.7 = 1033.6 | 1399.2 | 1.030 | 1.013 |
- A: prod +3.0% (every session +2.4..+3.7%), proxy +3.8%, fast +1.9% -- as priced by the probe (+2.8-3.0%).
- B: prod +1.3% (+0.5/+1.4/+2.1% per session, all above the 0.4% floor), proxy +1.0%, fast +0.1%. Low end
  of the +1-2% expectation -- consistent with only 60 slots removed (the adds were already partly v_dual).
- **Neither lost -> merge built** (route row 4's rule). Merge M = A + B (scratch `merge`,
  raw/merge_g09_g10.diff): VGPR 456, SGPR 100, spill 0/0, 3768 lines (-26 vs A: on top of A the scalar adds
  were mostly v_dual pairs already, so B has little left to remove). ut: bitwise identical again (raw/ut_M.txt,
  same hash, same 7 edge-case dB lines).

### Batch 2: merge vs A vs the champions, 3 sessions rotated (raw/benchm{1,2,3}_*, raw/benchm_agg.txt)
Orders: S1 M,inc,A,r3 / S2 A,r3,M,inc / S3 r3,M,inc,A, beat last. Same protocol, GPU idle before/after.
| shape | inc (r2, champion) | r3 | A (g09) | M (g09+g10) | beat | M/inc | M/A |
|---|---|---|---|---|---|---|---|
| fast  | 44.27/44.60/45.02 = 44.63 | 43.31/44.83/45.75 = 44.63 | 45.86/47.91/45.24 = 46.34 | 48.21/47.24/46.50 = 47.32 | 71.00 | 1.060 | 1.021 |
| proxy | 859.9/745.4/754.7 = 786.6 | 752.7/782.1/874.8 = 803.2 | 785.3/899.3/767.7 = 817.4 | 922.2/804.2/861.8 = 862.8 | 1025.9 | 1.097 | 1.055 |
| prod  | 1020.1/1017.1/1016.6 = 1017.9 | 1015.2/1019.1/1024.8 = 1019.7 | 1046.7/1054.5/1044.7 = 1048.6 | 1087.2/1078.4/1080.7 = 1082.1 | 1398.9 | 1.063 | 1.032 |
- A reproduces batch 1 (prod x1.030 both times). M beats A in every session at prod (+3.0..+3.9%), and is
  **super-additive**: A +3.0% and B +1.3% alone, together +6.3%. B's ISA delta on top of A is only -26
  instructions, so slot-count alone does not explain it. Likely mechanism (not verified): with both
  packed, the softmax VALU stream becomes pk_fma/exp/pk_add with far fewer v_dual pairs (259 -> 148) and
  v_nop 168 -> 62, so the scheduler's dependency stalls shrink more than the instruction count does.
- Score vs beat (mean of per-shape ratios, target = beat, margin 0%): **M 0.7603** (0.6665/0.8410/0.7735);
  inc 0.7077, r3 0.7135, A 0.7330.
- Acceptance arithmetic: throughput above the best ever at every shape, re-measured beside it (prod
  +6.1% over the best, r3 1019.7; proxy +7.4% over r3's 803.2; fast +6.0%). No shape has reached its target,
  and every shape is above 95% of its best. The acceptance rule is met.
- validation.py on the SHIPPED working copy (raw/val_shipped.txt): exit 2 -- correctness 16/16 PASS at 49 dB,
  determinism PASS (same hash), speed FAIL geomean 0.6556 (fast 0.4979 in validation's 2-arm order, where
  fast is cold/noisy per round 3; proxy 0.7326; prod 0.7724).

## Step 9 -- verdict
- Shipped: **the merge r4.i1.g09 + r4.i2.g10** in rounds/004/op (diff = raw/merge_g09_g10.diff; working copy
  verified byte-identical to the measured `merge` arm). `candidate: r4.i1.g09` (the lead id; the pool has no
  merged id), with the merge named in act.yaml `arms`. `outcome: delivered`.
- Mechanism for the record: on this kernel (softmax VALU serial with WMMA per wave, round-3 floor probe),
  packing the scalar f32 VALU of softmax into v_pk_* is converted to wall time at >= the 0.65 cycle/slot the
  probe priced, and the two packings compound. Remaining scalar f32 VALU in the softmax/rescale path (max
  tree is v_max3 already; corr/rescale multiplies of O) are the next same-mechanism targets.
- Not done: coexec (g12) not re-tried on top of M (route row 7 condition: compile-check on top of the winner).
