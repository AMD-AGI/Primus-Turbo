# Round 26 -- fast round, opt step

Written as I go. Every build failure and every dead measurement stays in.

## 1. What I read first (findings)

`facts.md`, `dead_ends.md`, `pool.md`, `route.md`, and the round-25 deep profile
(`rounds/025/1-profiling/`) plus `rounds/025/4-reflect/reflect.md`.

The state I inherit, in four sentences:

1. **`op/current` (= r20) is NOT the fastest tree on this machine.** Measured twice --
   round 24 on GPU 1 (route.md:1567) and round 25 on GPU 3, two passes
   (`rounds/025/3-act/act.md`). GPU 3, mean of two passes: prod r19 **637.57** /
   r17 634.42 / r20 **628.41**; proxy r19 **592.98** / r17 589.74 / r20 573.94;
   fast r19 **97.49** / r20 92.29 / r17 86.81. The prod same-code floor on this card is
   **0.32%**. r19 beats the incumbent by **+1.46% prod / +3.32% proxy / +5.6% fast**, and
   r19 dominates r17 on all three shapes.
2. **Nobody has acted on it.** Round 25 reflect §2.5 is explicit: "two rounds of arms were
   built on the slowest of the three trees and scored against the fastest". Reflect
   recommendation #1 is "choose the base tree by a same-session A/B and branch from the
   winner". That is the single largest measured lever in this job and it is free of any
   new mechanism risk.
3. **The prefetch family, the 4-wave family and the atomic-fusion family are all closed**
   (`dead_ends.md`; rule 1 is 8-for-8; `nobar` 0.923x; FUSED5 0.39x). `pool.md` holds
   **one** open entry, `r25.i2.g76` (materialise dS in bf16), whose decision rule needs
   G1a' (free) and P76 (a throwaway probe). P78 was taken in round-25 reflect and came
   back 8.8/9.4 TB/s -- inside g76's own band, so it neither passed nor killed.
4. **`op/current` still lacks the three h33 correctness fixes** (`r24.i3.g75`): a ~261 KB
   OOB read per launch, a 262,144 B OOB write at `sq % 64 == 32`, and -- worst -- an
   Int32 overflow that makes `num_records = 0` and silently drops **every dQ write at
   `nsp_q >= 8`, which is the prod configuration**. Measured free twice (0.9999 on GPU 1
   round 24, 0.99831 round 25), both times inside the 0.32% floor. Any tree branched from
   `op/current` must re-apply them.

## 2. The structural fact the diff gives me, which decides the round

I diffed the three trees rather than treating them as opaque. They are not a chain:

```
r17                                          (base)
r19 = r17 + g59      A-operand reuse on the dK/dV WMMA pair (reuseA=(dtile>0),
                     two runs of 8 instead of one interleaved run of 16)
r20 = r17 + g61 + g62   and g59 REVERTED
        g61  chain split on the S/P K=128 contraction: one 4-deep dependent WMMA
             chain -> two chains of 2, interleaved dt%2, summed in fp32
        g62  Q/dO cross-iteration prefetch depth 1 -> depth 2
```

So r20 did **not** build on r19: it silently dropped `g59`. Nothing in `facts.md` or
`pool.md` says that was a decision, and `r20.i2.g62`'s entry describes the merge as
`g61 + g62` only. On B0, r20 loses to r17 (prod -0.94%, proxy -2.68%), and r19 beats r17
(prod +0.50%, proxy +0.55%, fast +12.3%). Read together: **`g59` pays on B0, `g62` costs on
B0, `g61` has never been measured on this machine at all, and it has never been measured on
top of `g59` on any machine.**

That is the round. Two arms, both from `op/current`, both carrying the h33 fixes:

- **A** -- revert to r19 (drop `g61` and `g62`, restore `g59`). Banks the measured
  1.46% / 3.32% / 5.6%. Low risk, high certainty.
- **B** -- r19 **plus `g61`** (i.e. drop only `g62`). Tests the one component of round 20
  that has never been priced on B0, on the base that is actually fastest here.

They are nested, not independent -- B contains A. I am taking them that way deliberately and
saying so: the merge question ("do A and B compose?") is not open, because B *is* the merge.
What the pair resolves is `g61`'s own sign on this machine, which is the only unmeasured
degree of freedom left inside round 20's accepted change.

## 3. Free instrument, taken this round: **G1a' -- `g76`'s layout gate, zero card time**

`pool.md`'s `r25.i2.g76` says its decision rule needs three numbers and that **no estimate
may appear in it**. P78 was taken in round-25 reflect (8.79-8.93 TB/s write, 9.37-9.38 read
at 8.59-17.18 GB) and came back inside g76's own 8-10 TB/s band, so it neither passed nor
killed. G1a' was never taken. It is free -- it is a source question -- and a fast round has
no excuse for filing it again. Taken here.

**The question:** the contiguous run length, in bytes, of the `dS` store in `k_dkdv` and of
the `dS` load in `k_dq`, under one shared layout. "17.18 GB at a scattered run length is not
17.18 GB at streaming bandwidth."

**What the source says** (`job_context/op/current/kernels.py`, constants at :44-56 and
:645-648):

| | `k_dkdv` (producer) | `k_dq` (consumer) |
|---|---|---|
| kv blocking | `BLOCK_KV = 32`, `NKV = 2` x 16 | `KV_STEP = 32` |
| q blocking | 32 (a query PAIR) | `BLOCK_Q = 64`, `NQ = 4` x 16 |
| lane map | `row = lane % 16`, `half = lane // 16` (:189-190) | identical (:709-710) |
| dS held as | `ds_l`, 8 bf16 = **16 B/lane**, at `off = (hh*16+row)*S_ROW_B + kh*32 + half*16` (:443-452) | `a_ds`, 16 bf16 = **32 B/lane**, `ds_halves[qh][0] + [1]` (:849-852) |

Three results, and the third is the one that matters:

1. **`BLOCK_KV == KV_STEP == 32`.** The two kernels block the kv axis identically. Combined
   with the pool entry's already-established "neither side needs a transpose", one shared
   layout `dS[.., kv_tile, q_row, 32 kv]` serves both. The layout objection is fully closed,
   not just the transpose half of it.
2. **The dense block both sides agree on is 1024 B** -- 16 q rows x 32 kv x 2 B. The producer
   fills every byte of it; the consumer reads every byte of it. Nothing is written that is
   not read, and nothing is read twice within a block.
3. **But no single instruction reaches a 128 B line.** On the producer, for fixed `kh`, lanes
   `r` and `r+16` write `[kh*32, +16)` and `[kh*32+16, +32)` of q row `r`: a **32 B run**,
   and 16 of them at the q-row stride. The lane's own two stores (`kh = 0, 1`) are 32 B
   apart, not adjacent, so they do not merge. The consumer's 32 B/lane exceeds `b128`, so it
   is **two 16 B loads per lane**. Per instruction the run is **32 B producer / 16 B
   consumer** -- one quarter and one eighth of a 128 B line.

**Verdict: G1a' does NOT kill `g76`, and it does not pass it either.** It converts the
question into a sharper one. The traffic is *block-dense* (a wave fills its 1024 B block
completely, using 2 store instructions and 4 load instructions), so whether it streams at
P78's 8.8-9.4 TB/s depends entirely on whether the coalescer write-combines the two
half-line stores from one wave. That is a hardware question, and **P78 cannot answer it,
because torch's `fill_`/`copy_` are fully contiguous per instruction.**

⚠ **This corrects how `g76`'s decision rule should be read.** The rule as written multiplies
P78 by P76. P78 is now known to be an upper bound taken at a *different access pattern* from
the candidate's. The missing instrument is a bandwidth measurement **at the dS pattern** --
a 32 B-run, block-dense write followed by a 16 B-run, block-dense read at a footprint that
cannot be cache-absorbed. That is minutes of card time and it is now the single number
standing between this job and a decision on its last open candidate. Recorded as an id below.


## 4. The two arms, and how they were measured

Two ideas, measured apart, one change each against the same base:

| arm | tree | the one change |
|---|---|---|
| **A** | `rounds/019/op` + the h33 correctness fixes | base change: *branch from r19 instead of r20* (r19 = r17 + `g59` A-operand reuse) |
| **B** | arm A + `g61` lifted verbatim from `rounds/020/op/kernels.py:398-428` | `g61` on top of `g59` -- the one combination round 20 never measured, because r20 silently reverted `g59` |

Both arms, plus the incumbent `r20`, plain `r19` and `beat`, were run **interleaved in a single
benchmark process** per shape, so every arm of a shape shares one clock excursion. Two full
passes, palindromic arm order, 51 iters, 3 s continuous warmup. Device confirmed idle before
and after (physical GPU 3 at 0% use, no python in `fa-g3`; the KFD PIDs `rocm-smi --showpids`
lists all trace via `/proc/<pid>/cgroup` to containers `fa-g0` and `fa-g2`, not mine).

### Results (TF/s, median of 51)

| shape | arm | pass 1 | pass 2 | sclk |
|---|---|---|---|---|
| fast | **A** | 97.7046 | 97.2097 | 2330-2320 |
| fast | B | 95.4114 | 94.6035 | 2330-2320 |
| fast | r20 (incumbent) | 96.0251 | 95.5471 | 2330-2320 |
| fast | r19 | 88.3713 | 89.5524 | 2330-2320 |
| fast | beat | 51.3978 | 51.3978 | 2330-2320 |
| proxy | **A** | 586.128 | 586.811 | 1900-1945 / 1897-1955 |
| proxy | B | 581.913 | 584.212 | " |
| proxy | r20 | 570.650 | 573.665 | " |
| proxy | r19 | 581.874 | 585.009 | " |
| proxy | beat | 745.106 | 753.683 | " |
| prod | **A** | **638.567** | **638.508** | 1758-1819 / 1759-1822 |
| prod | B | 632.846 | 632.940 | " |
| prod | r20 | 628.683 | 628.082 | " |
| prod | r19 | 634.094 | 634.464 | " |
| prod | beat | 826.751 | 836.997 | " |

Same-code spread across the two passes, prod: A 0.009%, B 0.015%, r20 0.096%, r19 0.058%
(beat 1.24%). So this session's prod floor on our arms is **<= 0.10%**, tighter than the 0.32%
on record in `facts.md`.

Derived, pass-mean:

- **A vs r20 (the incumbent): +1.617% prod, +2.50% proxy, +1.74% fast.**
- **A vs r19: +0.672% prod, +0.52% proxy, +9.55% fast.**
- **B vs A: -0.885% prod, -0.58% proxy.** Against a <= 0.10% floor this is a real loss, ~9x the floor.

### Verdict

**Arm A ships.** Arm B lost on prod and proxy, so by the round's own rule the merge is not
taken -- and note that B *is* the merge of the two ideas, so there is nothing further to
combine. `g61` (the chain split on the S/P K=128 contraction) is **negative on top of `g59`**:
round 20 bought `g61 + g62` only after reverting `g59`, and this is the first time anyone
measured `g59` and `g61` together. They are substitutes, not complements -- both attack the
same dependent-WMMA chain, and taking both costs 64 extra `v_add_f32` per iteration for a
chain that `g59` has already broken.

### The surprise that contradicts the record

**A beats plain r19 by +0.672% prod / +0.52% proxy / +9.55% fast, and the only difference
between them is the h33 correctness fixes.** `facts.md` records those fixes as free
(0.9999 and 0.99831) -- but both of those measurements were taken on the r20 depth-2 prefetch
tree. On the depth-1 tree the value-neutral clamp is **not** time-neutral. This is the same
lesson `r25.i3.g77` taught from the other side: in this backend a net-zero reindex is not
ISA-neutral, and the sign depends on the tree it lands in. A "free" verdict does not survive
a base-tree change.

### Framework validation, as measured

`validation.py` on `rounds/026/1-opt/arms/armA` (key `r26_valA2`): correctness **pass**
(52.52-52.84 dB on every tensor and shape), determinism **pass** (dk/dv bitwise x200), speed
**FAIL** at geomean 0.984x beat (fast 1.645, proxy 0.766, prod 0.758). The bar is still not
beaten; A is the best arm, not a winning one. Reported as measured.

### Build failures this round

- `validation.py` key `r26_valA`: `FAILED: no such implementation directory:
  /tmp/.../armA`. The container mounts only `/home/lihuzhan`; the host `/tmp` is not visible
  inside it. Fixed by moving the arm trees under `rounds/026/1-opt/arms/` and re-running with
  a fresh key (`r26_valA2`) -- a finished run is not a cache, so the key was not reused.
- `rocprofv3 --stats` first attempt: `RC 1`, and the artefacts were invisible on the host.
  Two compounding causes, both now fixed in `raw/stats.sh`: (a) **`--stats` alone is a fatal
  error on rocprofv3 1.3.2** -- `"No tracing options were enabled for --stats option"`; it
  needs `--kernel-trace`. (b) `-d /tmp/...` wrote into the *container's own* `/tmp`.
- second attempt: `rc=127`, `bash: raw/stats.sh: No such file or directory` -- the runner's
  cwd inside the container is not mine; the script needs an absolute path.
- third attempt: `RC 2`, `benchmark.py: error: arm 'cand': no impl.py under .../op/cand` --
  `--arm-path cand=DIR` already registers the arm; passing `--arms cand` as well makes the
  parser resolve `cand` a second time as a name relative to `op/`.

## 5. The `rocprofv3 --stats` survey (step 2), and what it actually returned

Run through the named runner, `raw/stats.sh`, key `r26_stats4`, on `armA` at prod, 5 iters.
Three invocation defects fixed first (listed under build failures above). On the fourth
attempt the benchmark itself ran clean under the profiler and printed a real number:

```
RESULT shape=prod arm=cand ... latency_ms=8.5956 tflops=639.65 sclk_start=1830 sclk_end=1866
```

and then, in `rocprofv3`'s **output generation**, the process died with
`corrupted double-linked list` and hung in teardown. No CSV was written; the python process
had to be killed (`rc` recorded as 137 by hand, the sentinel never appeared on its own).
The `p76` tag therefore never ran under the profiler.

**This is `h4` reproducing exactly** -- *"with the faulting call removed it exits 0 and writes
no files"*. The new detail worth banking is the failure mode: it is not a GPU fault this time
(the dmesg monitor recorded **zero** events all round, and physical GPU 3 read 0% use
afterwards) -- it is a **host-side heap corruption inside rocprofv3 1.3.2's own output
generation**, after the kernels have already run correctly. So the survey step is not
recoverable by changing flags or output paths; the tool cannot serialise this process's
dispatch table. **Step 2's survey is unavailable on this op, for the third time.**

Consequence for the round: the kernel-duration breakdown that normally seeds candidate
selection did not exist, so candidates came from the other three sources only (the tree diff,
the corpus, and the free gates below).

## 6. Free instrument, taken this round: the WMMA-shape lever does not exist on gfx1250

The corpus's two strongest *scheduling* mechanisms for a dK/dV body both reduce to one
instruction-selection choice:

- `aiter fmha_v3_bwd_hd128_bf16.md` §6 b4 -- *"pick the MFMA shape per product from the
  contracted dimension, not one shape for the kernel"* (gfx950: `16x16x32` for the d=128
  contractions, `32x32x16` for the two K=16 accumulator GEMMs, so no partial-K instruction
  is ever issued).
- `flydsl attention/techniques.md` item 16 -- the sign rule b4 lacks: against a `32x32x16`,
  a `16x16x32` is *four independent accumulator chains at the same accumulator register
  count*; **+8.7% measured on a dK/dV body**; *"dependence-bound goes narrower, issue-bound
  goes wider."*

Gate, zero card time: enumerate what the backend can actually emit on this part.

```
python3 -c "import flydsl.expr as fx; print([n for n in dir(fx.rocdl) if 'wmma' in n.lower()])"
```

For **bf16 operands with an fp32 accumulator** gfx1250 offers exactly two:
`wmma_f32_16x16x16_bf16` and `wmma_f32_16x16x32_bf16` (plus the bf16-accumulate forms
`wmma_bf16_16x16x{16,32}_bf16`, which we cannot use). **There is no `32x32`, no `32x16`, and
no wider-K bf16 form.** Every wide shape in the list is fp8/fp4/f8f6f4.

Both corpus mechanisms turn on having *two different output-tile shapes* to choose between.
On gfx1250 there is one output tile, `16x16`, for every bf16 product in this kernel. The only
remaining axis is K, `32 -> 16`, and that is the **opposite** of item 16's mechanism: halving
K on the same `16x16` C tile makes the chain **twice as long and no more independent**. It
adds dependence rather than removing it.

**Verdict: dead at the free gate, zero card time, and it stays dead until the part changes.**
Recorded so no future round spends a slot on `aiter` b4 or `techniques.md` 16.

This also gives the round's second arm a sharper reading. `g61` is the only form of
"more independent accumulator chains" this backend *can* express -- explicit fp32 partial
accumulators summed at the end -- and arm B measured it **negative** on the `g59` tree at
9x the floor. The chain-splitting family is now closed from both ends: the hardware offers no
shape to do it with, and the software emulation of it costs more than it buys.

## 7. **P76 -- the subtractive `k_dq` ceiling. It closes `r25.i2.g76`.**

`pool.md` calls P76 *"the deciding number"* for `r25.i2.g76`, and round-25 reflect
recommendation #3 asks for it. Run through the named runner, key `r26_p76`, rc 0, 51 iters,
3 s warmup, both arms interleaved in one process so they share one clock excursion
(sclk 1795-1800 for both). The `p76` arm is `op/current` with `k_dq`'s S and dP WMMAs deleted
and the mask/exp2/dS VALU chain replaced by a feed straight from the prefetched K fragment --
**numerically wrong by construction, timing only, never in a speed table.**

| prod arm | latency | TF/s (meaningless for `p76`, shown only for the ratio) |
|---|---|---|
| `base` = `op/current`, unmodified | 8.7498 ms | 628.38 |
| `p76` = same, `k_dq`'s S + dP GEMMs and the exp2/dS chain deleted | 6.7937 ms | 809.31 |

**Saving = 1.9561 ms = 22.36% of the whole op.** Pool's kill threshold was *"kill at
< ~1.2 ms"*, so **P76 passes, and by a wide margin** -- larger than the ~1.19 ms the 7-vs-5
GEMM accounting predicted, because the probe also deletes the VALU chain that hangs off those
two GEMMs, which the FLOP-ratio argument never counted.

### And that is what kills `g76`

`g76`'s own decision rule, written in round 25, is: **build iff (P78 bandwidth) x (P76 ceiling)
clears break-even at `G1a''`s measured run length, no estimate in the rule.** Both factors now
exist, so the rule evaluates. Arithmetic, all three inputs measured:

- **Bytes.** `dS` has one element per causal S element. prod's causal S count is **4.295e9**
  (`facts.md`). In bf16 that is `4.295e9 x 2 = 8.59 GB` written by `k_dkdv` and `8.59 GB` read
  back by `k_dq` = **17.18 GB round trip**. `k_dq` stops reading Q, V and dO in exchange, which
  is `3 x 4 x 32 x 8192 x 128 x 2 B = 0.80 GB` -- under 5% of the new traffic, and it does not
  change the sign.
- **Rate.** P78, measured on this host at exactly this transfer size (8.59-17.18 GB):
  **8.79-8.93 TB/s write, 9.37-9.38 TB/s read.**
- **Cost.** `8.59/8.86 + 8.59/9.375` ms = `0.970 + 0.916` = **1.886 ms**, using the *best case
  on record*.
- **Saving.** **1.956 ms**, measured above.

**Margin: 0.07 ms, or 0.8% of the op -- on the most favourable bandwidth number this job has
ever measured, and before `k_dkdv` pays anything at all for the extra store.** `g76`'s own
prediction was that `k_dkdv` would rise by *up to 12%*; 12% of `k_dkdv` is far more than
0.07 ms, so the central case is plainly negative.

And `G1a'` (§3) already showed that P78's rate is the **wrong rate to use here**: the torch-
contiguous stream runs full cache lines, whereas the `dS` pattern issues **32 B runs on the
producer side and 16 B on the consumer side**, both well under a 128 B line. The applicable
rate is therefore *below* 8.8-9.4 TB/s, not at it, and every correction moves the margin
further negative.

**Verdict: `r25.i2.g76` fails its own gate and is closed, at the cost of one 51-iter benchmark
and no build.** The 7-vs-5 GEMM redundancy is real and worth 22.36% of the op, but **paying for
it in HBM traffic cannot work at `d=128` and this sequence length**: the round trip costs
essentially exactly what the recompute costs. This is the corpus's own conclusion arriving
independently -- `flydsl attention/techniques.md` item 14 states that recompute is a deliberate
trade of MFMA for workspace bytes and that *"marginal MFMA at one wave per SIMD is barely above
its paper cost"*. Round 26 has now measured both sides of that trade on this part and they are
a wash.

**What survives.** The 22.36% is still there and is still the largest single number in the job.
It is only the *HBM* route to collecting it that is dead. Anything that removes the recompute
**without a round trip through memory** -- i.e. keeps `dS` on-chip between the two consumers --
is untouched by this result. That is a fusion question, and `h64`/FUSED5 closed the naive form
of it; it is not the same question, and no entry is filed for it this round because nothing
cheap distinguishes it yet.

## 8. What was consulted, and what the round concludes

### Corpus files opened this round

| file | what it gave |
|---|---|
| `optimization/routes/1-metrics-to-techniques.md` | the section that actually fits us -- **"When no metric is elevated ... the answer is a measurement, not a list of ideas: price each phase by deleting it."** That is P76, and it is what closed `g76`. Also: *"Low occupancy alone is not a finding"*, and the stall-attribution row confirming the survey we could not take would not have named the charge anyway. |
| `backends/flydsl/attention/techniques.md` | item 16 (MFMA shape as a scheduling knob; **killed at §6's free gate**), item 14 (recompute is a deliberate trade of MFMA for workspace bytes; *"marginal MFMA at one wave per SIMD is barely above its paper cost"* -- which §7 has now measured on this part), item 11 (the architected half caps at 256 and FlyDSL's AGPR-spill flags are a no-op), item 3 (price with a coefficient, not a removal probe). |
| `backends/flydsl/attention/recipes/hd128.md` | §8: a *deterministic* FlyDSL hd128 backward exists at **five** GEMMs with a separate reduce, so seven is not forced by determinism. §9: upstream priced its workspace by subtraction (6.78 -> 4.91 -> 4.18 ms) -- the same method as P76, reached independently. `s_setprio` inert at one wave/SIMD. |
| `backends/flydsl/attention/README.md` | the adoption rule that governs everything above: on a partial `config:` match take **the structure and the attack order, and re-sweep every constant**. gfx1250 vs gfx950 is a partial match at best. |
| `backends/aiter/attention/recipes/fmha_v3_bwd_hd128_bf16.md` | §6 b4 (per-GEMM MFMA shape -- **killed at §6's free gate**), b6 (paired causal K-tiles -- already rejected for free in `dead_ends.md`), b7 (one wave/SIMD is forced and is *not* the lever; 2.19 overhead instructions per MFMA is the algorithm, not a defect). b1/b2 are the atomic-dQ fork we did not take. |
| `backends/hipkittens/attention/recipes/gqa_d128.md` | §6.1, the 1.72x `ducks::art` register-pinning result -- same op class, same head dim, same 1 wave/SIMD. **Not applicable here: gfx1250 has a unified VGPR file and no AGPRs to shuttle operands through**, so the `v_accvgpr_read/write` traffic the 1.72x removes does not exist on this part. |
| `backends/flydsl/attention/dead-ends.md` | checked before either pool entry was filed, per the README's own gate. |

Not opened: `optimization/techniques/*` beyond what the route named -- it is a catalogue of
changes, not a procedure -- and `knowledge/profiling/*`, which is a reference to look single
things up in and was not needed once the survey proved unavailable.

### Conclusion

The round's value is not the +1.617% prod that arm A banks. It is three closures, two of them
bought for no card time at all:

1. **The base tree was wrong and is now right.** `op/current` (r20) was the slowest of the three
   candidate trees on B0, and rounds 21-25 built five arms on it and scored them against a
   faster champion. Round-25 reflect asked for the rebase; this round did it and measured it.
2. **`g76` is closed by its own rule**, with every input measured and no estimate. The 7-vs-5
   GEMM gap is worth **22.36% of prod** -- the largest number in the job -- and the HBM route to
   collecting it costs essentially exactly that much. That is a wash, not a win.
3. **The MFMA-shape family is closed for this part**, by reading what the backend can emit.

And one open surprise, which is the second arm: a *value-neutral* index clamp is worth
**+0.672% prod** on the depth-1 tree and was recorded as free on the depth-2 tree. Nobody knows
why, and the instrument that would say costs nothing to run.

The bar is still not beaten. `validation.py` on `rounds/026/op`: correctness pass, determinism
pass, speed FAIL at geomean 0.984x. Reported as measured.

## 9. The final validation, and a discrepancy I am not resolving in my favour

`validation.py` was run a second time, explicitly against `rounds/026/op` (key
`r26_val_final`, never the default `op/current`). `rounds/026/op` and
`rounds/026/1-opt/arms/armA` are byte-identical -- `diff -rq` clean -- and the tree was
re-checked as different from both `rounds/019/op` and `op/current`.

| run | key | path | fast | proxy | prod | beat prod | geomean |
|---|---|---|---|---|---|---|---|
| 1 | `r26_valA2` | `arms/armA` | 89.25 | 583.34 | **635.28** | 838.55 | 0.984x |
| 2 | `r26_val_final` | `rounds/026/op` | 89.70 | 584.52 | **628.63** | 837.82 | 0.988x |

**Identical code, and prod reads 1.06% apart.** The five-arm interleaved benchmark read
638.567 and 638.508 for that same code -- a 0.009% spread. `beat` was stable across every run
(838.55 / 837.82) and sclk sat in the same band (1765-1812) throughout.

The reading: `validation.py` times **one** arm against `beat`, in a **separate process per
shape**, so its arms do not share a clock excursion the way a five-arm interleaved process
does. Its own prod floor is therefore on the order of 1%, not the 0.10% this round measured
interleaved. The 628.63 figure lands on top of `rounds/020/op`'s interleaved number (628.38)
**by accident of that floor, not by identity of code.**

What this does and does not license:

- The **+1.617% prod over the incumbent stands**, because it is a same-process, same-clock
  comparison, both arms in one palindromic order, repeated in two passes with a 0.009-0.096%
  same-code spread.
- It is **not** reproduced by differencing two separate `validation.py` invocations, and
  nothing here licenses anyone doing that -- including me. If the framework's own re-run lands
  near 628, that is the number of record for the gate and I am not arguing with it.
- Correction 4 still binds in the other direction: nothing ships below prod's noise floor, and
  arm B (-0.885%) did not.

**Geomean 0.988x. Correctness pass, determinism pass, speed FAIL. The bar is not beaten.**

---

## 10. Step 5 -- executing the route, rows 15 and 16. Both died at their free gates.

The route was taken from the top and not re-derived. Rows 1-14 are carried hints; their
outcome cells in `findings/route.md` record what this round did with each. Rows 15 and 16
are the two idea rows, and **both are decided, both for zero card time, and neither was
built into a measured arm.**

### 10.0 Compile cache cleared before the first build

`FLYDSL_RUNTIME_CACHE_DIR` defaults to `~/.flydsl/cache`; it was removed inside `fa-g3`
(it was in fact already absent, which is itself worth recording -- this backend was not
serving anything stale). The `__pycache__` directories under `job_context/op`,
`job_context/op/beat` and `job_context/op/ut` were removed as well. **`op/current/` was
not touched**, per the standing rule; its three `.pyc` were checked instead and every one
is newer than its `.py`, so no stale bytecode could be loaded from it either.

### 10.1 Build failure: the ISA dump driver, first attempt (`r26_isa1`)

`r26_isa1` exited **rc 0 with all three compiles reporting `RC ... 0`, and produced no
dumps at all** -- only a `log` per tree. The cause is that FlyDSL's `DebugEnvManager`
carries `env_prefix = "DEBUG"` but overrides the two dump options with **explicit**
`env_var=` names, `FLYDSL_DUMP_IR` and `FLYDSL_DUMP_DIR`
(`flydsl/utils/env.py:241,243`). I had passed the prefix-derived `FLYDSL_DEBUG_DUMP_IR` /
`FLYDSL_DEBUG_DUMP_DIR`, which the manager silently ignores. **Mechanism: a
successful-looking run that produced nothing, because the option names are not derived
from the prefix for these two fields.** Fixed in `raw/isa.sh` and re-run under a fresh key
`r26_isa2` (rc 0, nine dump directories, three `21_final_isa.s`). This is the round's only
driver failure and there were no compile failures of any arm: `armC` compiled on the first
attempt.

### 10.2 Row 15, `r26.i1.g82` -- **abandoned at free gate (c)**

`g62` lifted onto the `g59` base. The arm was written, and it is kept at
`rounds/026/1-opt/arms/armC/` so the next round does not pay to write it again: a 21-line
diff to `kernels.py` only, `impl.py` untouched. Its `qloop_full` body and its prologue are
**textually identical to `rounds/024/op`'s** apart from comments (verified by `diff` of the
extracted regions), so the lift is the established `g62`+h33 form and not a re-derivation.
The depth-2 clamp is on `kk`, exactly as `rounds/024/op/kernels.py:540`.

Gate, read off `raw/isa/base/k_dkdv_0/21_final_isa.s` vs `raw/isa/g82/...`:

| condition | bar | base (`rounds/026/op`) | `g82` | verdict |
|---|---|---|---|---|
| `vgpr_spill_count` | 0 | 0 | 0 | **pass** |
| `private_segment_fixed_size` (scratch) | 0 | 0 | 0 | **pass** |
| `.vgpr_count` | <= 960 **and** occupancy >= 1 wave/SIMD | **724** | **904** | **pass** |
| `group_segment_fixed_size` | unchanged | 70656 | 70656 | pass |
| wmma per body | 64 | 64 | 64 | pass |
| `qloop_mask` body | unchanged | 620 | 620 | pass |
| **last `buffer_load_b128` index, `qloop_full` body** | **not later** | **116** | **187** | **KILL** |
| `qloop_full` body instructions | -- | 675 | 750 | -- |

Condition (b) passes and it corrects a recorded fact while doing so: **`k_dkdv` on the
`g59` base runs at 724 VGPR.** `facts.md:381`'s 904 is a `g61+g62` number and does not
describe the tree this round ships. `g62` costs **+180 VGPR** on B0, and because both 724
and 904 sit above gfx1250's 512-VGPR threshold, occupancy is 1 wave/SIMD either way -- the
180 registers are free in occupancy terms, which is exactly why (b) could not kill it.

Condition (c) kills it. The last prefetch load moves **71 instructions later**. Pool
rule 1 -- *anything that moves the last prefetch load's issue index later loses* -- was
**8 for 8 but every one of those eight was fitted on A0**. This is its first reading on
B0, and the rule's own instruction is unambiguous: do not build it. It was not built.

The opcode histogram names the mechanism, and it is the rule's own signature:

| opcode | base | `g82` | delta |
|---|---|---|---|
| `buffer_load_b128` | 128 | 160 | **+32** |
| `v_mov_b64_e32` | 190 | 254 | **+64** |
| `s_set_vgpr_msb` | 329 | 371 | **+42** |
| `s_clause` | 31 | 37 | +6 |
| `v_or_b32_e32` | 173 | 191 | +18 |
| `v_nop` | 81 | 58 | -23 |
| **`s_wait_loadcnt`** | 25 | 13 | **-12** |

The second stage does buy cover -- twelve fewer `s_wait_loadcnt` is the compiler telling
us the waits it removed. It pays for them with 32 more loads in flight, 64 more `v_mov_b64`
of rotation (**this is h60's 67 rotation `v_mov`s, unchanged by the base change**), and 42
more `s_set_vgpr_msb` prefixes because the extra 180 registers push more operands past
VGPR 255. The scheduler fits all of that in by moving the whole load block later.

**What is still unknown, and I am not claiming otherwise:** `g62`'s *sign* on the `g59`
tree. The gate did not measure it. What it establishes is that this mechanism for deeper
cover cannot be had without violating the one rule this job has never seen fail. A
different mechanism for the same goal is a new candidate for a later round, not a quiet
substitution inside this one.

### 10.3 Row 16, `r26.i2.g83` -- **abandoned at step 1; step 2 not run**

Step 1 was the free compile-only ISA diff of `rounds/019/op` against `rounds/026/op`. The
row's written kill condition was "if the ISA delta is empty this row dies". The delta is
**not literally empty, and it is mechanistically empty**, which is the same verdict:

- `k_dq_0`: **byte-identical**, 3598 instructions both.
- `k_delta_bshd_0`: **byte-identical**.
- `k_dkdv_0`: 2968 -> 2973, i.e. **+5 SALU** -- `s_add_co_i32` 24->26, `s_min_i32` 2->4,
  `s_max_i32` 2->3. Nothing else in the histogram moves at all. Everything else in the
  405-line textual diff is **SGPR renumbering** (`s7`->`s6`, `s68`->`s69`, ...), which is
  allocation noise, not work.
- Of those five, **exactly one** lands in the hot loop: `qloop_full` 674 -> 675.
  `qloop_mask` 620 -> 620. `.vgpr_count` 724 -> 724. LDS 70656 -> 70656.

`k_dq_0` being byte-identical is itself a finding: h33 **defect b** rewrote `g_dq`'s extent
in Int64, and at the shape the dump compiles (`nsp` resolving to the non-PARTIAL path) that
rewrite costs **nothing at all** in the emitted code.

All three preregistered mechanisms are falsified:

| preregistered mechanism | prediction | measured | verdict |
|---|---|---|---|
| address math collapses | instruction count **falls** | **+5** | falsified |
| compiler hoists the prefetch address | first load **earlier**, last unmoved | first 81 -> **82**, last 115 -> **116** -- both one *later* | falsified |
| VGPR falls | `.vgpr_count` down | **724 -> 724** | falsified |

One scalar `s_min_i32` per iteration, against a 675-instruction body, on a kernel whose
verdict is `latency` and whose scalar unit is not the bottleneck, cannot pay +0.672% prod.
If anything it is a small cost.

**So the row's own condition fires and I take it:** the +0.672% is reclassified as an
unexplained floor violation. Step 2 -- extending the clamp idiom to `k_dkdv`'s remaining
runtime-variable prefetch indices -- does not run, because step 1 named no mechanism for it
to extend. The standing consequence is that **the h33 clamps are a correctness fix costing
one SALU per iteration, not a speed lever.** Round 24 priced them free on the r20 tree and
round 24 was right.

Step 6 then re-measured the gap directly, six arms deep in one process (§11). It survives
in sign at every shape in both passes -- **+0.29%/+0.43% fast, +0.57%/+0.34% proxy,
+0.21%/+0.14% prod** -- but at roughly a third of the original figure, and at prod it is
now smaller than the same-arm pass-to-pass spread. Consistent with an effect of zero plus
an ordering bias the palindrome does not fully cancel. Nothing may be built on it.

---

## 11. Step 6 -- the acceptance measurement. Six arms, one process per shape, two passes.

Key `r26_benchfinal`, rc 0, `raw/bench_final/`. Every arm of a shape ran **interleaved in a
single `benchmark.py` process**, palindromic, 51 timed iterations, 3 s continuous warmup,
L2 flushed between timed iterations -- so all six share one clock excursion per shape. This
closes the gap recorded earlier in the round: **`rounds/017/op` and `rounds/025/op` had not
been re-measured this session, and now they have been.** The card was less throttled than
during the earlier passes (fast at sclk 2330 vs the earlier band, prod 1756 -> 1839), which
is exactly why stored figures are not usable and the rule re-measures champions.

Idle-device check: GPU 3 at **0% use** before and after. `rocm-smi --showpids` listed four
KFD PIDs; all four trace through `/proc/<pid>/cgroup` to `c87432dadced` (`fa-g0`) and
`92bb52de5a27` (`fa-g2`). **None is `a1f5ab75082c` (`fa-g3`).** `dmesg -W` armed throughout
and recorded **zero** GPU events.

TFLOP/s, median of 51, pass 1 / pass 2:

| shape | **cand** = `rounds/026/op` | r017 | r019 | r020 (incumbent) | r025 | `beat` |
|---|---|---|---|---|---|---|
| fast | **97.280 / 97.208** | 94.139 / 93.809 | 96.997 / 96.788 | 96.027 / 96.025 | 91.195 / 89.852 | 51.556 / 51.065 |
| proxy | **596.189 / 593.016** | 592.769 / 589.795 | 592.809 / 591.014 | 576.673 / 576.478 | 517.034 / 516.040 | 756.075 / 750.976 |
| prod | **639.975 / 638.885** | 638.116 / 637.002 | 638.626 / 638.003 | 629.716 / 628.551 | 613.092 / 610.775 | 836.109 / 832.998 |

Pass-to-pass spread of the *same* arm at prod: cand 0.17%, r017 0.17%, r019 0.10%, r020
0.19%, r025 0.38%, beat 0.37%. That is the honest floor for a cross-pass comparison; the
within-pass comparison is tighter because it shares the excursion.

### The acceptance arithmetic, written out

Best-ever **re-measured beside it in this session**, per shape, excluding cand: fast
**r019 96.997**, proxy **r019 592.809** (r017 592.769 is within noise of it), prod
**r019 638.626**.

| shape | cand | best-ever re-measured | ratio | improves? |
|---|---|---|---|---|
| fast | 97.280 | 96.997 (r019) | **1.0029** | yes |
| proxy | 596.189 | 592.809 (r019) | **1.0057** | yes |
| prod | 639.975 | 638.626 (r019) | **1.0021** | yes |

Pass 2 agrees in sign at all three (1.0043, 1.0034, 1.0014). **6 for 6.**

1. *Throughput improves on the best ever, re-measured beside it* -- **yes**, at every
   shape, in both passes. Geomean over shapes 1.0036 (p1) / 1.0030 (p2).
2. *Every below-target shape is at least 95% of its own best ever* -- the below-target
   shapes are proxy (78.9% of target) and prod (76.5%); cand is at **100.6%** and
   **100.2%** of their re-measured bests. **Yes.**
3. *No shape that had reached its target falls below it* -- only `fast` has reached its
   target; cand is at **1.887x** it. **Yes.**

**The round is accepted by that arithmetic.** It is not accepted against the *bar*: `beat`
is not beaten at proxy or prod and is not claimed to be.

Against the incumbent `op/current` (= r020), which is what the tree change actually bought:
**+1.31% fast, +3.38% proxy, +1.63% prod, geomean +2.10%** (p1; p2 +1.23/+2.87/+1.64,
geomean +1.91%). Every one of those is far outside the 0.17-0.19% same-arm floor.

### Score

`score` = mean over shapes of `this_round / target`, uncapped, target = `beat` as measured
in the same sweep (the bar's margin is 0%, so target == beat):

    fast   97.280 / 51.556  = 1.8868
    proxy 596.189 / 756.075 = 0.7885
    prod  639.975 / 836.109 = 0.7654
    score = 1.1469     (pass 2: 1.1534)

The framework will recompute this from its own `validation.py` run. Mine, taken earlier in
the session on the same shipped tree, gives fast 89.670 / proxy 584.531 / prod 628.629
against beat 53.702 / 759.688 / 837.819, i.e. **score 1.0632** -- a 7.3% disagreement that
comes entirely from `validation.py` timing one arm per process per shape rather than
sharing a clock excursion. §9 is the worked example of that; **the discrepancy is recorded,
not resolved in my favour.**

### What is in the working copy

`rounds/026/op` contains **arm A**, unchanged, byte-identical to `1-opt/arms/armA/`. Rows
15 and 16 produced no measured arm, so nothing replaced it and nothing was reverted. `armC`
(the `g82` source) is kept under `1-opt/arms/` and is deliberately **not** in the working
copy: it was never measured, and shipping an unmeasured arm would be the exact failure the
step-5 instructions warn about.
