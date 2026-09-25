# Round 2 (fast) -- opt log (planning step)

## Step 1 -- findings
- facts.md: g01 (LPT dispatch) confirmed speedup (prod 1.086-1.107, proxy 1.48-1.60, fast neutral in
  a controlled probe); round 1 rejected only on 7 edge-case precision misses at 50 dB that the
  baseline fails identically, plus the fast median reading 0.934 vs a 0.993 band.
- Spec now: `precision_sqnr_db: 49` (was 50) and `shape_band.fast: 0.90` -- both rejection
  causes are removed. Operator hint h20 (must): re-land g01 from rounds/001/op.
- dead_ends.md absent. pool: g02 (O_VARIANT v1, lost at proxy in round 1 -> h3 should retire),
  g03 (post-ASM penalty instrument), g04 (split-KV fast).
- `diff -r rounds/001/op rounds/002/op`: the ONLY difference is `_lpt_block_id` and its 7 call
  sites -> re-landing is a copy, not a rebuild.
- No deep profile exists (job_context/profiling absent).

## Step 2 -- look
- GPU: 0% busy, no KFD pids; fa-repro has no python process; dmesg clean (no MES/fault lines).
- No new card survey: round 1's survey (same day, same container, same code as op/current) is the
  survey, and round 1 recorded that `rocprofv3 --kernel-trace --stats` writes nothing in fa-repro.
  The builds of rows 1-3 re-measure everything against beat in one session.

## Step 3 -- sources
- (a) pool/route + operator hints h20/h1: g01 re-land is the top.
- (b) round 1 survey: proxy tail (fixed by g01); fast dominated by the post-ASM penalty (g03).
- (c) optimization/routes/1-metrics-to-techniques.md (A6 grid fill / occupancy drain ->
  1-grid-and-cache-locality, which g01 already is); backends/flydsl/attention/README.md.
- (d) recipes listed (flydsl hd64/hd128, hipkittens gqa_d64/d128, aiter fmha_v3_bwd): all gfx950;
  not opened this round -- pool is not empty and the last round's candidate is re-landable.
  The next round that goes to the body (h6) should open hipkittens gqa_d128 section 6.

## Step 4 -- choice
- A = h20 / r1.i1.g01 re-land. Expect prod +8-10%, proxy +45-60%, fast 0 (median possibly 0.92-0.95,
  inside the 0.90 band). Precision identical to incumbent (49.82+ dB > 49).
- B = r2.i1.g05 branch-free rescale, from op/current, independent lines. Expect -2..+3%; unpriced.
- h1 2b/2c = r2.i2.g06 (L4, L5), stacked on A as throwaway arms (same lines as g01).
- Rider: r1.i3.g03 instrument.
- Merge A+B only if neither lost beyond the session floor.

explored.consulted: knowledge/optimization/routes/1-metrics-to-techniques.md,
knowledge/backends/flydsl/attention/README.md, listing of backends/*/attention/recipes/.

## Step 5 -- build
Scratch: /home/lihuzhan/.cache/op-evolve-r002-scratch (container sees only /home/lihuzhan;
/tmp scratch dir is a symlink to it). Compile cache cleared (`rm -rf /root/.flydsl/cache` in fa-repro)
before the first build; no python process alive in the container beforehand.

- **Arm A = h20 / r1.i1.g01**: kernel file copied from rounds/001/op into the working copy;
  `diff -q` identical to round 1's. Bijection check from round 1 (raw g01_bijection.txt, all 10 ut
  grids) applies unchanged -- same code, same grids.
- **Arm B = r2.i1.g05**: scratch copy of op/current with `ENABLE_DEFER_RESCALE = False` (one line).
- **Row 2 (h1 steps 2b/2c, r2.i2.g06): decided without a build.**
  * L4 XCD-major: after g01, lin = rank*gyz + rem, rem enumerates each (b, kv_head) exactly once
    per rank. With round-robin WG->XCD dispatch, (b,kv_head)'s XCD is (rank*gyz+rem) mod nXCD,
    which is rem mod nXCD -- constant across ranks -- whenever nXCD divides gyz (proxy gyz=8,
    prod 32; any power-of-two nXCD <= 8). So g01 already puts every (b,kv_head) on a fixed XCD at
    the two shapes where it could matter; an XCD remap on top would reorder nothing that matters.
    Plus BWD g43 measured no L2 locality effect on gfx1250. (XCD count is not in arch/gfx1250.md.)
  * L5 causal-aligned origin: all three scored shapes have sq == skv and BLOCK_M/gqa = 64 rows =
    n_block, so q tiles already start on the causal diagonal; L5 can only change sq != skv cases,
    which are not scored.
  -> No arm built; h1's steps 2b/2c are closed by construction for this job's shapes.

COMPILE (FLYDSL_DUMP_IR, fast shape; raw/descriptors.txt):
| arm | VGPR | SGPR | scratch | LDS | s_cbranch_vccz | ISA lines |
|---|---|---|---|---|---|---|
| ctrl (rounds/000) | 445 | 98 | 0 | 327680 | 4 | 4525 |
| A | 445 | 97 | 0 | 327680 | 4 | 4651 |
| B | 443 | 99 | 0 | 327680 | 1 | 4508 |
B removes 3 of the 4 vccz branches (the deferred-rescale ballots); descriptors distinct from ctrl.
No build failures.

### ut (raw/ut_A.txt, raw/ut_B.txt; ut's own gates.py floor is still 50 dB)
- A: identical to round 1 to 0.01 dB -- spec shapes 51.23/50.89/50.83 dB, 7 edge cases 49.82-49.99
  (below ut's 50, above the spec's 49), determinism 200/200 PASS (o=26a89a2db0cd, same hash as r1).
- **B: every case >= 50.21 dB, all 16 PASS even at 50 dB**, determinism PASS (different hash, as
  expected -- arithmetic changed). The deferred rescale costs ~0.2-0.4 dB of o SQNR everywhere;
  always-rescale is what lifts the edge cases over 50. (A precision fact, not a speed one.)

## Step 6 -- measure
GPU before: 0% use, no KFD pids, no python in fa-repro. dmesg monitor armed for the whole
measure phase: no events. After: idle, no pids, no orphans.

### Session 1 (raw/bench1_*.txt): benchmark.py, one process per shape, palindromic, median of 101
Arms: ctrl = rounds/000/op (the incumbent; op/current is its copy), A, B, AB (merge), beat.
| shape | ctrl | A | A/ctrl | B | B/ctrl | AB | AB/ctrl | beat |
|---|---|---|---|---|---|---|---|---|
| fast  | 47.28  | 44.53   | 0.942 | 43.73  | 0.925 | 43.88  | 0.928 | 69.78 |
| proxy | 551.45 | 778.51  | **1.412** | 497.56 | **0.902** | 721.09 | 1.308 | 1026.83 |
| prod  | 934.50 | 1017.97 | **1.089** | 878.47 | **0.940** | 962.03 | 1.029 | 1395.10 |
sclk fast 1100, proxy 1028->1063, prod 1019->1046.
Floor: cross-session spread of ctrl, r1 s1/s2/this: proxy 530/550/551 (~4%), prod 920/935/935 (~1.6%).
- **A (h20 re-land) wins a third time**: proxy 1.41x, prod 1.089x (r1: 1.604/1.479, 1.107/1.086).
  Fast 0.942 is above the operator's 0.90 band (r1 read 0.918/0.934; controlled probe 0.999).
- **B (r2.i1.g05) LOST, far outside the floor**: proxy -9.8%, prod -6.0%, fast -7.5%. Prediction
  was -2..+3%; wrong in sign and size. Removing the deferral means the wide O rescale (the
  64x128 fp32 accumulator multiply per wave) runs on every KV tile; with the kernel already
  VALU/issue-limited in its softmax phase (h4), that extra VALU is paid directly. The branch
  it removes was cheap by comparison. -> L20 (h11) should move to ❌ on gfx1250 for speed.
- **Merge AB** was measured in the same process (built before B's number existed, so it cost no
  extra session); per the rule it is disqualified because B lost, and it also loses to A
  (proxy 0.926x of A, prod 0.945x). Not shipped.
- **Shipped: A.** Working copy rounds/002/op = round 1's g01 kernel, byte-identical.

### Own validation.py on the working copy (raw/validation.txt): exit 2
- correctness: **all 16 cases PASS at the spec's 49 dB** (worst o 49.82 dB, short_q full); determinism PASS.
- speed: fast 0.4258, proxy 0.7416, prod 0.7308 of beat -> geomean 0.6133 < bar 1.0 -> FAIL on speed only.
- Note the fast number: validation's 2-arm process reads candidate 0.0748 ms (min 0.0428) vs 0.0483
  in the 5-arm session -- in a 2-arm palindrome every candidate call directly follows beat, so the
  post-ASM penalty (g03) lands on nearly all of them. Fast in validation is a harness number.

### Rider: r1.i3.g03 post-ASM penalty probe (raw/g03_probe.py/.txt), working-copy kernel
Protocol as round 1 (flush + 400k-cycle GPU sleep before each timed call, median of 101).
Prediction: if I$ eviction, (a) any large-code kernel between clears nothing, (b) a full-grid call
of the same kernel re-warms and clears it, (c) a longer sleep does not.
| shape | alone | after beat | beat->fp32 GEMM | beat->same kernel | beat, 10x sleep | alone again |
|---|---|---|---|---|---|---|
| fast  | 0.0418 | 0.0683 | 0.0662 | **0.0670** | 0.0663 | 0.0419 |
| proxy | 0.1559 | 0.1941 | 0.1926 | **0.1556** | 0.1938 | 0.1558 |
- Not time/clock/power (10x sleep changes nothing). A GEMM between does not clear it. One full
  call of the same kernel clears it at proxy (512 WGs, all CUs) but NOT at fast (32 WGs = 32 CUs):
  the warm-up only helps on the CUs it ran on, so the next fast dispatch lands mostly on cold CUs.
  That is the signature of a **per-CU (per-WGP) instruction cache** being evicted by the ASM kernel
  (and any other big kernel) -- the I$ hypothesis now has a positive test, not only the absence of
  alternatives. Penalty at proxy is 38 us here (r1: 27 us on the incumbent).
- Consequence: the penalty scales with code fetched cold per CU. Mechanisms that attack it: prefetch
  the loop body into I$ in the prologue (`s_prefetch_inst`, which the ASM kernel uses -- h4) or
  shrink the loop body. A warm-up launch is not a fix (it does not reach the CUs at fast).

## Step 7 -- verdict
Candidate r1.i1.g01 (h20 re-land), outcome delivered, left in the working copy (one file).
Predicted prod +8-10% / proxy +45-60% / fast ~0; measured prod +8.9%, proxy +41% (below my range --
round 1 saw 48-60%; proxy is the noisier shape), fast 0.942. B predicted -2..+3%, measured -6..-10%:
wrong in sign; recorded as a speed dead end with a precision upside.
