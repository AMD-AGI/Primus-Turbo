# Round 10 reflection

## Outcome

Round 10 delivered `r8.i1.g20`, the two-line nodelay change, but it was not accepted and
`best_round` remains 4. The performance prediction held. Against a rebuilt round-4 champion in
four rotated, no-beat sessions, prod ratios were 1.0086, 1.0093, 1.0069 and 1.0084, for a mean of
1.0083. Proxy was 1.0026 and fast was 0.9943. The three-shape aggregate gain was only 1.0022,
below the configured 1.007 acceptance threshold (`rounds/010/3-act/act.yaml` and
`raw/s2/summary.txt`).

Correctness did not regress. `validation.py` passed all 16 specification cases at the configured
49 dB threshold and determinism passed 200/200 with the champion's output hashes. The separate unit
test still hard-codes 50 dB and reported eight 49.82-49.99 dB failures on output bitwise identical
to the champion, so `passed` is false for a pre-existing harness/spec mismatch
(`rounds/010/3-act/act.yaml`). This mismatch did not by itself decide round 10—the aggregate gain
was also below 0.7%—but it can reject a future winner and remains an operational blocker.

The result is the `faster_held` cell. Round 10 removed all fourteen `s_delay_alu` instructions per
KV tile, reducing the clean loop from 513 to 499 instructions. PMC measurements showed
`SQC_ICACHE_REQ` 122.24 M -> 118.05 M, `SQ_CYCLES` ratio 0.9912,
`GRBM_GUI_ACTIVE` ratio 0.9912, and zero I-cache misses for both arms. The claimed observable
mechanism therefore held: the executed stream became shorter and prod cycles fell about 0.9%.
Only about one third of the 2.73% static instruction reduction converted, so instruction counts
remain gates and diagnostics rather than general performance prices. The card still cannot separate
issue slots from dependency latency because stall counters are rejected and FlyDSL has no ATT trace
(`rounds/010/3-act/measurements/pmc_icache_cycles_prod/summary.txt` and round-10 profiling).

## My objections, checked first

### `r10.i4.g29`: the original A/B/C discriminator was not specific enough

This objection held. The planner accepted the change and ran write-size, read, no-write, warm and
post-write-delay controls before building anything. In one-image processes, current measured
38.50-38.74 us after a 256 MiB zero, 38.66-38.90 us after a 256 MiB read, 38.70-39.14 us after a
GPU sleep, and 38.70 us warm. The hypothesised write-drain term was zero within 0.4 us, so the
depth-3 arm was correctly not built (`rounds/010/2-plan/raw/postflush/summary.md`, section A).

The same instrument found the real source of the apparent 12-16 us excess. Current measured
38.82 us with one FlyDSL image, 38.9-41.7 us with two images, and 50.32 us with four; its minimum
stayed near 38-39 us. Thus round 9's large median was a multi-image harness effect, not flush
write-back exposed by `tensorcnt 0` (`rounds/010/2-plan/raw/postflush/summary.md`, section B).

### `r8.i1.g20`: prediction and falsification had to use the same threshold

This objection held and the corrected prediction was measured. The final plan required mean prod
ratio at least 1.007 over three or more sessions, with every session above 1.003. Round 10 measured
mean 1.0083 over four sessions, and the minimum session was 1.0069
(`rounds/010/2-plan/plan.yaml`; `rounds/010/3-act/act.yaml`). The candidate should remain alone as
the experimental base; round 9 already demonstrated why merging a winner with an independently
losing arm is invalid.

### `r10.i1.g27`: fixed zero is outside the correctness domain

This objection held without needing a build. A fixed softmax reference must be a true upper bound,
while the mandatory round-10 gate includes scaled scores around +/-1e4 and the operator contract
has no finite score bound. Reference zero therefore overflows on large positive scores and can
underflow the denominator on all-large-negative rows. A guard would recompute a row maximum and
restore the dependency the candidate proposed to remove. Both agents retired it before card time
(`rounds/010/2-plan/candidates.yaml`).

### `r10.i3.g28`: resource wording and static-price gate

This objection held as a plan correction; the arm was not executed. Round-10 profiling supports one
workgroup per LDS-owning unit at 327,680 B/WG, but explicitly leaves CU-versus-WGP ownership
unresolved. The final candidate also removed the proposed “longer clean loop means no card time”
rule. Its static gates are now only spill, exact unpaired code, expected dispatch and bijection;
performance must be priced on card (`rounds/010/1-profiling/6-bound-analysis/analysis.md` and
`rounds/010/2-plan/candidates.yaml`).

There were no disagreements left open in `plan.yaml`, and reflect took no additional measurements.

## What did not work

- `r10.i4.g29` failed before build. Its prediction required the 256 MiB flush to create the fast
  penalty; round 10 measured no difference between the large write, large read, GPU sleep and warm
  conditions. Depth three was not tested and receives no performance verdict; only the post-flush
  rationale is dead.
- `r10.i1.g27` failed the operator's correctness domain by arithmetic before build. The mechanism is
  not a valid optimization for this unbounded interface.
- The round still did not promote nodelay. Its +0.83% prod gain was diluted by proxy +0.26% and fast
  -0.57% to an aggregate 1.0022. At fast, candidate/champion ratios ranged 0.9872-1.0041 because even
  two FlyDSL images introduce order-dependent noise (`rounds/010/3-act/raw/s2/summary.txt`).

## What I got wrong

In my independent analysis I treated the round-9 foreign-image cold state as entirely outside the
h28 score. Round 10 showed a more precise hierarchy: removing beat removes the worst 66 us state,
but a two-FlyDSL-image candidate/champion process can still add 0-3 us depending on ordering, while
four images reproduce the 50-65 us medians. My conclusion was directionally right for rejecting a
30 us kernel-body optimization, but too absolute about the cleanliness of h28
(`rounds/010/2-plan/raw/postflush/summary.md`).

I also proposed fixed-zero softmax despite the already-mandated +/-1e4 score test. The corpus gave
the mechanism and a D=64 performance result, but not a valid upper bound for this operator. The
correctness contradiction was derivable before proposing the candidate; the plan review correctly
removed it without a build (`rounds/010/2-plan/dialogue.md`).

## Stall audit

The most consequential suspect is the belief that a no-beat two-arm process makes fast/proxy clean.
Round 10 refutes that stronger reading: image count itself perturbs the fast median. Because
acceptance averages all three shape ratios, this can prevent a consistent prod win from moving
`best_round`. Future fast/proxy comparisons below several microseconds need one implementation per
process or a null calibration.

The route has also circled the same KV-loop scheduling family since round 5: cross-tile softmax,
row reduction, barriers, rescaling, compiler scheduling and wave arbitration. Most arms lost; only
nodelay won, and by less than 1%. The statement that softmax is on the path remains supported, but
it should no longer rank every new schedule manipulation above orthogonal mechanisms. The next
measured candidate should be `r10.i3.g28`, which targets workgroup-boundary/O-store overlap, against
the nodelay base.

Finally, `r1.i2.g02` is now a doubted dead end. Its round-1 proxy loss was measured in a process with
four FlyDSL images plus beat, exactly the kind of protocol round 10 showed can create arm-specific
small-shape distortions; later proxy measurements also varied by more than the original 3-4% loss.
The old id remains retired, but a new O_VARIANT-v1 candidate on the nodelay base is justified under
a one-image-per-process protocol.

## Memory rewrite

- `r8.i1.g20` remains in facts and now carries the round-10 confirmation and mechanism evidence.
- `r10.i1.g27` and `r10.i4.g29` move from pool to dead ends on round-10 evidence.
- `r1.i3.g03`, `r1.i4.g04`, `r6.i1.g16` and `r8.i2.g21` move from pool to dead ends; their existing
  round-6/8 closure evidence is consolidated, with round 10's harness result added where relevant.
- `r10.i3.g28`, `r7.i3.g19`, `r9.i3.g26` and `r5.i3.g15` remain in the pool, in the order planned for
  surviving work.
- `route.md` and `state.yaml` were not edited.
