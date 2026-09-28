# Round 29 (fast) -- gfx1250 / FlyDSL / attention backward

**Written as the round ran.** Build failures and dead ends are in here at the point they
happened, not reconstructed at the end.

## 1. What I read first, and the one line that decided the round

`findings/facts.md`, `dead_ends.md`, `pool.md`, `route.md`, and
`rounds/025/1-profiling` (the last deep profile; no round accepted since, so it still
describes `op/current` byte for byte -- I opened its bound analysis for the 1.4076
issued/algorithmic ratio and the 22.36% `k_dq` ceiling, both quoted below).

Round 28's closing sentence says the executable pool is **empty**: `g87` and `g88` both
closed by measurement, `g85` blocked by h11 (deep-round sized), and prod proven to be
**neither dispatch-limited (`g88`) nor split-K-limited (`g87`) nor issue-limited at the
margin (`g86`'s 0.15x conversion)**.

But the framework's standing-constraint table carries **two rows round 28 never saw**:

- **h70 (`must`, outstanding):** *"k_dq is ISSUE-bound, not latency-bound: unroll k_dq
  kvloop_full x2 (lab arm u2n, removes 65 back-edge `v_mov_b64`) -> prod op -3.06%
  (3 procs), output bitwise = r19h; **build it next round and measure proxy too**."*
- **h71:** the k_dq lab's closure note -- 2 waves/SIMD k_dq (op +13-15% = loss),
  GQA-grouped WG (loss), `k_delta` fused (null), **`k_dkdv` kv-loop unroll x2 (ku2) op
  +6.5% = loss**, q1 stacked on u2n worse than either.

So the pool is empty but the **route is not**: h70 is an outstanding `must` with a
lab-measured -3.06% behind it. That is the round.

I read the lab itself rather than the hint:
`/home/lihuzhan/.../Primus-Turbo/output/0927__b0/lab-kdq/REPORT.md` (197 lines), plus its
`u2n.{kernels,impl}.diff`, its `bounds/bounds_u2.txt` CPU index proof, and its arm trees.

**The single most important check of the round, done first:**
`md5sum job_context/op/current/*.py` is **byte-identical to the lab's `r19h` tree**
(`_env` 7f33871b, `impl` 62c1c257, `__init__` 0024d09d, `kernels` b4f18d67). Round 28 was
not accepted, so `op/current` is pure r19h -- exactly the base the lab measured u2n on.
The lab's u2n tree can therefore be adopted **verbatim**, not re-derived, and its
`kernels.py`/`impl.py` md5s (`1fda1828`, `08cb8533`) match the REPORT's record of tree
`ed57c5aa`. This also means **`g86` is NOT in `op/current`** (round 28 shipped it but was
rejected), so the job's one twice-measured free win is sitting unshipped.

`diff -rq rounds/029/op job_context/op/current` -> identical (h67's per-round requirement).

## 2. The two arms

| arm | id | what | base |
|---|---|---|---|
| `armA` | **h70 / u2n** | `k_dq`'s `kvloop_full` unrolled x2 -- pairs of kv blocks per `scf.for` trip, so the carried K/V prefetch's 65 back-edge `v_mov_b64` rotation copies vanish (741 -> 655 instr/iter). Adopted verbatim from the lab tree. | `op/current` |
| `armB` | **r27.i3.g86** (inherited id, not re-badged) | `k_dkdv`'s bf16 dK/dV epilogue staged through the dead Q/dO LDS ring: 256 `buffer_store_b16` -> 32 `buffer_store_b128`. Built and measured twice before (+0.35% on the r26 base, +0.37% on r19h) and **rejected both times with its round**, so it has never shipped. | `op/current` |
| `armAB` | merge | both | `op/current` |
| `armC` | throwaway probe | **u2nb** = u2n + `sched_barrier(0)` between the two halves (lab compiled it at **798 VGPR** vs u2n's 991 and never ran it -- `plan.sh` item 1). Never enters the working copy. | `op/current` |

They are independent by construction: `armA` touches only `k_dq`/`impl.py`, `armB` only
`k_dkdv`'s epilogue. The merge is therefore worth measuring, and it costs no extra process
-- it is one more interleaved arm inside the same per-shape run, sharing one clock
excursion (h57).

⚠ **What the lab says NOT to merge, and I am obeying:** `q1` (fma softmax) alone is
-2.37% but **with** the unroll only -1.19%; `ku2` (the same unroll in `k_dkdv`) is +6.5%
= a loss. Both were measured on card. Neither is an arm here.

### 2b. Build note -- the merge did not splice cleanly, and why

`patch -p0 < /tmp/g86.patch` onto armAB **FAILED** ("Hunk #1 FAILED at 572",
`kernels.py.rej`). Cause, found by reading the reject: the lab's u2n `kernels.py` still
carries the lab's own switch code in `k_dkdv`'s epilogue --
`okv = ([ok_[si] * scale for si in ...] if VF_KV else [ok_[si] for si in ...])` -- so the
g86 patch context did not match. I did **not** hand-edit around it. I spliced with a
Python script that anchors on `        base_o = bat * Skv` in both files and replaces
armAB's 15-line epilogue with armB's 73-line g86 block wholesale. `VF_KV = False`
(`kernels.py:53`), so the switch is semantically inert and the splice changes nothing
else. **Verified afterwards, not assumed:** armAB's per-kernel ISA counters are identical
to armA's `k_dqg` and to armB's `k_dkdv`, kernel for kernel.

## 3. Free gates, before any timing

**Static-ISA gate (h3).** Clean for every tree -- `private_segment_fixed_size = 0`, zero
`scratch_` ops, no spills. Nothing was killed here.

| tree | kernel | VGPR | spill | scratch | body instr | `v_mov_b64` | `s_set_vgpr_msb` |
|---|---|---|---|---|---|---|---|
| base | `k_dq` | 741-instr body | 0 | 0 | 741 | **65** | -- |
| armA | `k_dqg` | **991** | 0 | 0 | **1310 = 655/iter** | **1** | -- |
| armB | `k_dkdv` | 729 | 0 | 0 | 659 | -- | 109 |
| armAB | both, byte-identical to armA's `k_dqg` + armB's `k_dkdv` | | | | | | |
| armC | `k_dqg` (u2nb) | **798** | 0 | 0 | | | |

armA's `k_dkdv` is **byte-identical to base's** -- the unroll touches only `k_dq`, as
claimed. The mechanism h70 asserts is visible in the ISA before any clock is read:
**65 back-edge `v_mov_b64` rotation copies -> 1**, and 741 instr/kv-block -> 655.

**Correctness gate.** `ut/test_correctness.py --impl <arm>` for all four arms:
`RESULT: PASS`, `UT_RC_*=0`. dB is **item for item identical to base** (52.46-52.98
against a 50 dB gate) -- consistent with the lab's claim that u2n is *bitwise* equal to
r19h, and with g86 being a pure store-path rearrangement.

**Idle-device gate (h57/h64).** `rocm-smi --showpids`/`--showuse` against the *physical*
card before and after: `GPU use (%) 0-2`, every listed PID either ours or a neighbour on a
different render node. Our card is `0004:04:00.0` = `renderD152` = `fa-g3` = logical GPU 3
(`docker inspect`). One earlier reading looked busy only because
`grep -Ei 'No KFD|PID'` was matching the table **header**; fixed by reading the full tail.

## 4. The deciding measurement

One process per shape, all five arms + `beat` interleaved inside it (blocked ruler, lead 4
+ block 9, palindromic, h66) so every arm shares one clock excursion (h57). Two passes.
us, lower is better.

```
=== prod   base    armA   armAB   armB    armC    beat    sclk(base)
p1        638.4   653.6   656.0   640.1   608.9   815.7   (1758, 1828)
p2        637.2   653.6   655.2   639.5   608.6   814.9   (1759, 1827)
MEAN      637.8   653.6   655.6   639.8   608.8   815.3
spread%    0.19   0.005   0.122   0.08    0.056   0.099

=== proxy  591.3   602.8   602.6   589.7   551.8   738.9
spread%    0.121   0.19   0.475   0.365   0.177   1.135

=== fast    99.4    98.6    99.0    99.0    98.4    57.7
spread%    1.531   1.369   1.749   2.125   1.476   1.43
```

Hold on -- **base is 637.8 and armA is 653.6, which is LARGER.** The harness reports
**TFLOP/s-like throughput here, not time**: `beat` is 815.3 and `beat` is by definition the
faster reference, and at `fast` `beat` is 57.7 against base's 99.4 where base is known to
win. So higher = better, and the ratios below are `arm/base` with >1 = win. I am stating
this explicitly because reading it the other way would inverse every verdict in this file.

| shape | armA (h70/u2n) | armAB (merge) | armB (g86) | armC (u2nb probe) | beat |
|---|---|---|---|---|---|
| **prod** | **1.0248** | **1.0279** | 1.0032 | 0.9545 | 1.2783 |
| **proxy** | **1.0194** | **1.0191** | 0.9972 | 0.9332 | 1.2496 |
| fast | 0.9923 | 0.9960 | 0.9960 | 0.9899 | 0.5807 |

**This session's floors, measured for free by derivational A/A nulls in the same
processes** (arms that by arithmetic cannot differ from base on that shape):

- `fast`: **all four arms** are same-code with base (armA/armAB/armC take `nsp_q=8` ->
  the untouched `k_dq_sp`; armB takes the `PARTIAL` fp32 path, which g86 leaves alone).
  Four nulls: 0.9899-0.9960 -> **fast floor ~1.0%**. Every `fast` column above is noise
  and I am reporting it, not using it.
- `proxy`: armB is a null -> 0.9972 -> **proxy floor 0.28%**.
- `prod`: no null available (all arms are live at prod); base's own cross-pass spread is
  **0.19%**, and round 28's prod floor was 0.09-0.44%. I use **0.44%**, the pessimistic end.

## 5. Verdicts, arm by arm -- every arm reported

**armA (h70 / u2n) -- WIN, and h70 is discharged.** prod **+2.48%** against a 0.44%
pessimistic floor (5.6x) and a 0.19% observed base spread (13x). proxy **+1.94%** against a
0.28% floor (6.9x). h70's explicit instruction was *"build it next round and **measure
proxy too**"*: proxy is now measured and **agrees with prod in sign and magnitude**, so the
win is not a prod-shape artefact. The lab saw -3.06% on op time (3 procs); I see +2.48%
throughput. Directionally identical; I am reporting **mine, as I measured it**.

**armB (r27.i3.g86) -- positive, at the floor, third consistent measurement.** prod
**+0.32%**, which is *equal* to the pessimistic floor and so **does not clear it on its
own**. But this is now the **third** independent measurement of the same tree (+0.35% on
the r26 base, +0.37% on r19h, +0.32% here) -- same sign, same magnitude, three different
sessions. proxy 0.9972 and fast 0.9960 are its own A/A nulls and confirm it is inert
exactly where the code says it must be. It did **not lose**.

**armAB (the merge) -- the best arm, and the merge is MEASURED, not inferred.** prod
**+2.79%**, proxy +1.91%. Neither component lost, so the merge was permitted; I built it
and ran it in the same process rather than adding the numbers on paper. It then turns out
superposition would have been right: predicted 1.0248 x 1.0032 = **1.0281**, measured
**1.0279**. Two further internal consistency checks that I did not design and that came
out right anyway:
- armAB vs armA at prod = 655.6/653.6 = **+0.31%**, i.e. *exactly* g86's standalone +0.32%.
- armAB vs armA at proxy = **-0.03%**, and at proxy g86 is on the inert PARTIAL path, so
  the merge must be a null there. It is, well inside the 0.28% proxy floor.

**armC (u2nb) -- a clear LOSS, and it answers the lab's open question.** prod **0.9545**
(-4.55%), proxy **0.9332** (-6.68%), both an order of magnitude outside their floors. The
lab compiled u2nb, saw VGPR fall **991 -> 798**, and left it unrun as `plan.sh` item 1 on
the theory that lower pressure would help. **It does not.** Both counts sit on the same
1-wave/SIMD rung (the threshold is 512), so the 193 registers buy no occupancy at all,
while the `sched_barrier(0)` costs 4.5-6.7%. That item is now closed by measurement, for
the cost of one interleaved arm.

## 6. The round's real finding: pool rule 1 is falsified for `k_dq`

Pool rule 1 -- *"anything that moves the last prefetch load later in the body loses"* --
was 8-for-8 and already downgraded by h60 to ordering-only. armA breaks it outright:

| | base `k_dq` | armA `k_dqg` |
|---|---|---|
| last `buffer_load_b128` | index 99 of 741 body = **13.4%** | index **1304 of 1310 = 99.5%** |
| outcome | -- | **+2.48% prod, +1.94% proxy** |

That is a **larger** displacement than `g72`, which lost 4.52%. So the rule as written is
wrong. armC, in the same round, shows what the rule was actually detecting: u2nb's
`sched_barrier(0)` pushes the **first** prefetch group from index 5 to 532 (40% of the
body), collapsing the distance between that load and its consumer in the second half --
and it loses 4.55%. In armA the first group is still at index 5 with the whole second half
covering it, and the last group at 1304 is consumed at the *top of the next trip*, which
is adjacent.

**Proposed restatement (one round of evidence, both signs, not yet a fact):** what costs is
shortening the issue-distance between a prefetch load and its **consumer**; the load's
absolute position in the body is a proxy that only holds when the body is one kv block
long. Rule 1 should be **scoped to `k_dkdv`** or retired. I am recording this in the round,
not promoting it to `facts.md` myself.

## 7. What shipped, and the validation I ran myself

`rounds/029/op` = **armAB** (`kernels.py` 37f37052, `impl.py` 08cb8533; `_env.py` and
`__init__.py` untouched from r19h). `job_context/op/current/` was **never edited**.

Launch note, recorded because it nearly produced a false result: my first two detached
launches wrote their sentinel to `/host/artifacts/...`, which **exists inside the container
but is not the bind mount** -- the only mount is `/home/lihuzhan -> /home/lihuzhan`. The
first attempt also had `&` binding the whole `&&` list, so `bash -c` exited before `mkdir`
ran. Neither launched anything; `pgrep -af` inside the container confirmed **nothing was
orphaned** before I relaunched, so no second copy ever ran against the first. I deleted the
`/host` decoy directory my own failed `mkdir -p` had created.

```
ut/test_correctness.py --impl rounds/029/op   ->  RESULT: PASS      UT_RC=0
op/validation.py rounds/029/op                ->  RESULT: FAILED    VAL_RC=2
     correctness  pass
     determinism  pass
     speed        FAIL      fast 1.707x | proxy 0.804x | prod 0.802x   (min 0.802x)
```

**Reported as I measured it, not as I would like it.** `validation.py` is a *separate
process* from my bench and it agrees with it: prod **656.6 TF/s** vs my armAB **655.6**
(0.15%), proxy **602.7** vs **602.6** (0.02%). Both well inside the floors. There is
nothing to reconcile, and the prod number is hard.

Speed FAILs under h69 because h69 requires proxy **and** prod each >= 1.00x `beat`, and we
are at 0.802x. The round does not close the 20% gap and never could: h70's whole mechanism
was worth ~3%. **The case for accepting this round is the structural-best rule**, not h69 --
prod 656.6 TF/s is the best number this job's tree has ever produced, +2.79% over an
incumbent that had not moved in three rejected rounds, with correctness and determinism
both passing and the determinism gate covering the exact dK/dV store path `g86` rewrites.

**Device hygiene.** `dmesg | grep -ci amdgpu` = **710 before and 710 after** the whole
session -- not one new kernel line. The `MES(0,0) failed to respond to msg=MISC` lines that
appear in the tails are on **`0002:04:00.0` = physical GPU 1**, wedged since 01:44Z per h65;
**no line anywhere mentions `0004:04:00.0`**, which is ours (renderD152 / fa-g3 / logical
GPU 3, from `docker inspect`). Benign by attribution, not by assumption.

## 8. Candidates I sourced and then killed before spending anything

Both mandatory sources were consulted because round 28 was not accepted.

**From `hipkittens/attention/recipes/gqa_d128.md` 6.1 -- hand-assigned registers, 1.72x
there. KILLED, and it was the obvious trap of this round.** The shipped `k_dqg` carries
**649 `s_set_vgpr_msb`** and `k_dkdv` 319, easily the largest non-arithmetic class in the
tree, and the corpus offers a 1.72x register-pinning result aimed straight at it. It is
dead three times over and I should not have had to measure to know:
`r10.i3.g30` closed the mechanism (gfx1250 has a **flat** VGPR file; the prefix encodes
high register numbers and moves no data); `r24.i1.g73` re-closed it *with a gate* (121
distinct operands, 62.1% occurring once, **zero** back-to-back pairs -- a uniform spread,
not a clusterable tax, bounded at 7.8% **in instruction count**); `dead_ends.md` states
flatly that **FlyDSL exposes no register-pinning lever**, and that the gfx950
`v_accvgpr_read/write` analogue **does not transfer**. Above all, `dead_ends.md:283`:
*"No future candidate may be justified by 'fewer `s_set_vgpr_msb`' or 'fewer
instructions'."* A candidate in `dead_ends.md` is not a candidate.

**Hazard-anchor coarsening (+1.76%, `flydsl/attention/dead-ends.md`). KILLED on
inapplicability:** the shipped kernels contain **zero `v_min_`/`v_max_` and zero `s_nop`**,
so the dead-operand anchor structure that technique operates on does not exist here. (Its
coarser variants are also *wrong* while still passing the main shape -- a trap I did not
need to enter.)

**A 124-`v_mov_b64` block I nearly filed. KILLED by its own basic-block census.** Bucketing
`k_dqg`'s `v_mov_b64` by block: `.LBB0_4` (the unrolled hot loop) **1**, `.LBB0_6` 1, and
**`.LBB0_5` 124**. `.LBB0_5` looks like a huge rotation-copy drain -- but enumerating the
back edges shows the only two are `.LBB0_4 -> .LBB0_4` and `.LBB0_8 -> .LBB0_8`; nothing
branches back to `.LBB0_5`. It is **straight-line, executed once per workgroup**, against a
1310-instruction loop run many times. Filing it would have repeated the corpus's own
*"deleting instructions that were never issued"* dead end, which measured **-6.1%** because
an ISA histogram counts text, not execution.

## 9. Verdict

**Ship `armAB`.** prod **+2.79%**, proxy **+1.91%**, both many multiples of their floors,
correctness and determinism passing, static-ISA gate clean, `validation.py` agreeing with
my bench to 0.15%. It fails h69's speed bar at 0.802x `beat` and I am not dressing that up;
it is accepted, if it is accepted, on **structural-best** -- the best prod this tree has
ever produced, after three consecutive rejected rounds left a twice-measured free win
(`g86`) sitting unshipped and an outstanding `must` (h70) unbuilt.

The round's transferable lesson is not the +2.79%. It is that **the pool being empty did
not mean the job was out of candidates** -- h70 was sitting in the framework's own standing
table with a lab-measured −3.06% behind it, and one `md5sum` showing `op/current` was
byte-identical to that lab's base meant the tree could be adopted verbatim instead of
re-derived. Read the whole ledger, not just `pool.md`.

Three things closed for free, inside processes that were running anyway: the lab's unrun
`plan.sh` item 1 (u2nb: a **loss**, and lower VGPR bought nothing because 991 and 798 are
the same occupancy rung), pool rule 1's falsification for `k_dq` (last prefetch load at
**99.5%** of the body, +2.48%), and the two-sided replacement model -- what costs is
**issue-distance from a prefetch load to its consumer**, not the load's absolute position.

**Consulted:** `findings/{facts,dead_ends,pool,route}.md`; `rounds/025/1-profiling`;
`rounds/027-028/1-opt`; `Primus-Turbo/output/0927__b0/lab-kdq/REPORT.md` + `u2n.*.diff` +
`bounds/bounds_u2.txt` + `tools/plan.sh`; `knowledge/backends/hipkittens/attention/recipes/gqa_d128.md`;
`knowledge/backends/flydsl/attention/dead-ends.md`; `knowledge/backends/flydsl/attention/techniques.md`;
`knowledge/optimization/techniques/3-pipelining-and-scheduling.md`;
`knowledge/optimization/routes/0-kernel-programming-principles.md`.

## 10. Build and measure, against the route table

### 10a. Two corrections I had to make before measuring anything

**(i) `rounds/028/op` is NOT what `op/current` is a copy of, and I checked rather than took
it on trust.** Per-file md5, all four files:

| tree | `_env` | `__init__` | `impl` | `kernels` |
|---|---|---|---|---|
| lab `r19h` reference | 7f33871b | 0024d09d | 62c1c257 | b4f18d67 |
| `job_context/op/current` | 7f33871b | 0024d09d | **62c1c257** | **b4f18d67** |
| `rounds/028/op` (incumbent) | 7f33871b | 0024d09d | **f60da2b4** | **1c487556** |
| `rounds/029/op` (this round) | 7f33871b | 0024d09d | 08cb8533 | 37f37052 |

`op/current` is **pure r19h**, byte for byte. `rounds/028/op` is r19h **+ `g88`** (the causal
tile-pairing grid change, in `impl.py`) **+ `g86`** (666 diff lines in `kernels.py`). Round 28
shipped both and was **rejected**, so `op/current` never moved. Both trees sit on the
post-refactor r19h structure, so my base was the right *structure* -- but the incumbent is
`rounds/028/op`, and it contains `g88`, which round 28 itself measured as a **loss** (0.9947
prod). I had never measured it. It is in this run.

**(ii) I had not read `rounds/028/0-refactor/refactor.md` before now.** Read in full. It
confirms the revert to r19h was verbatim (`tree_md5 a791a047`), that `_env.py`/`__init__.py`
are untouched, and its ranked "what would make it fast" list puts **`g86` re-basing at #3**
-- *"a clean, validated, prod-path-only win that this revert dropped... take it as a free
0.35%, do not route a round around it"* -- which is exactly how this round used it: as the
second arm, not as the round's thesis. The report predates h70 and does not mention `k_dq`
unrolling at all. Its #1 item (dependent latency in `k_dkdv`) remains untouched by round 29,
which is consistent: round 29 moved `k_dq`, not `k_dkdv`.

### 10b. Compile cache cleared before the deciding run

`FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache` held **697 MB / 41 compiled launch objects** and
`TRITON_CACHE_DIR=/tmp/triton` 52 KB. Both emptied to **0 entries** at the top of
`raw/final.sh`, before the first build, so every arm in the deciding run recompiled from
source. This matters more than usual here: three of the four arms share `_env.py` and
`__init__.py` byte for byte and differ only inside `kernels.py`/`impl.py`, which is exactly
the case where a key collision would serve one arm's binary to another and report it as this
round's number with no error anywhere. (My step-4 exploratory run did **not** clear it; the
per-arm ISA dumps did differ correctly, so the cache was keyed properly, but that is evidence
after the fact and not a substitute for clearing it.)

### 10c. The deciding numbers -- fresh session, cleared cache, four arms interleaved

One process per shape, all four arms inside it (blocked lead 4 + block 9, palindromic, h66),
two passes. TFLOP/s, higher is better. `cur` = `op/current` (pure r19h), `r028` =
`rounds/028/op` (**the incumbent**, r19h + `g86` + `g88`), `r029` = this round.

| shape | cur | r028 (incumbent) | **r029** | beat | spreads |
|---|---|---|---|---|---|
| **prod** | 639.0 | 637.0 | **656.5** | 817.0 | 0.05-0.07% |
| **proxy** | 586.1 | 587.5 | **603.8** | 739.8 | 0.29-0.79% |
| fast | 98.4 | 98.7 | **98.6** | 58.6 | 0.40-0.85% |

| ratio | prod | proxy | fast |
|---|---|---|---|
| **r029 / r028 (vs incumbent)** | **1.0307** | **1.0277** | 0.9985 |
| r029 / cur | 1.0275 | 1.0302 | 1.0018 |
| r028 / cur | **0.9968** | 1.0024 | 1.0033 |
| r029 x beat | 0.8035 | 0.8162 | 1.6829 |

**The incumbent is slower than the tree it was built from, at prod.** `r028 / cur = 0.9968`
-- `g88` (a measured loss, 0.9947 in round 28) and `g86` (+0.32%) together are a net
**-0.32%** at prod. This is the first time the two have been measured against pure r19h in
one session, and it is why `r029` gains **+3.07%** on the incumbent but only +2.75% on `cur`.
**Round 29 is therefore not carrying `g88` forward** -- `rounds/029/op` is built on `cur`,
not on `r028`, so the loss is dropped rather than inherited.

**`fast` is a null and I am not claiming it.** 0.9985 against the incumbent sits inside every
arm's own 0.40-0.85% cross-pass spread, and it is a null *by construction*: at `fast`,
`nsp_q = 8` routes dQ through the untouched `k_dq_sp`, so u2n's unroll is not even in the
binary, and `g86` takes the `PARTIAL` path it deliberately leaves alone. There is nothing in
`r029` that can move `fast`, and the measurement agrees.

**Agreement between the two independent processes**, which is the check that matters after a
cache clear: `validation.py` re-run on the rebuilt tree reports prod **657.7** and proxy
**603.2** against the bench's **656.5** and **603.8** -- 0.18% and 0.10%, both inside the
floors.

### 10d. Correctness, after the cache clear

```
ut/test_correctness.py --impl rounds/029/op   ->  RESULT: PASS      UT_RC=0
op/validation.py rounds/029/op                ->  RESULT: FAILED    VAL_RC=2
     correctness  pass     fast 52.61/52.65/52.83  proxy 52.52/52.57/52.67  prod 52.56/52.60/52.71 dB
     determinism  pass     dk/dv bitwise identical across 200 runs; dq bitwise
     speed        FAIL     fast 1.679x | proxy 0.802x | prod 0.803x
```

**Worst SQNR over every tensor and every scored shape: 52.52 dB against a 50.0 dB floor.**
These are **identical, to the digit, to the refactor report's r19h figures** (52.52 proxy dq,
52.61/52.65/52.83 fast) -- which is the expected result and a real check: u2n is *bitwise*
equal to r19h, and `g86` only rearranges stores. **I did not weaken the precision check**;
this is `op/validation.py`'s own gate, unmodified.

### 10e. Acceptance arithmetic, stated so it can be checked against me

Best-ever per shape is the best figure on the **current** (post-refactor, round-28-or-later)
structure, **re-measured in this session**: `cur` and `r028` are both round-28-or-later trees,
so best-ever = `max(cur, r028)` per shape.

| shape | best-ever (re-measured) | r029 | gain vs best-ever | target (= 1.00 x beat) | ratio |
|---|---|---|---|---|---|
| prod | 639.0 (`cur`) | 656.5 | **1.0274** | 817.0 | 0.8036 |
| proxy | 587.5 (`r028`) | 603.8 | **1.0277** | 739.8 | 0.8162 |
| fast | 98.7 (`r028`) | 98.6 | 0.9990 | 58.6 | 1.6826 (capped 1.0) |

1. **Throughput improves on best-ever** at both scored shapes, +2.74% and +2.77%. ✓
2. **Every below-target shape is >= 95% of its own best-ever:** prod 102.7%, proxy 102.8%. ✓
   (`fast` is above target, so it is not a below-target shape; it is 99.90% regardless.)
3. **No shape that had reached its target falls below it:** `fast` was at 1.68x and is at
   1.6826x. ✓

Margin is **1.00** (`beat` and `target` are equal in every row of `progress.md`), so `target`
= `beat`. `score` = mean of capped ratios = **0.8733**; uncapped it is 1.1008, and
`progress.md` says `score` is no longer capped -- Python recomputes it either way, so I am
recording both rather than choosing.

**Speed still FAILs `validation.py` under h69** (needs proxy *and* prod each >= 1.00x beat;
we are at 0.802/0.803). The round stands on the throughput rule above, not on h69, and
`VAL_RC=2` is reported as measured.
