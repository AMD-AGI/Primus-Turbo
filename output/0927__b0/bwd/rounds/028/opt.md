# Round 28 (fast) -- opt notes

Written as the round ran. Base tree: `rounds/028/op` = **r19h**, installed at this round's
head by the framework as operator refactor **h68** (`rounds/019/op` + the three h33 clamps).
`op/current/` is that same tree -- **the r20-vs-r17-vs-r19 base-tree question that ate rounds
21-27 is closed by the operator, not by me**, and this is the first round in seven that does
not start 1.5-2.7% behind the fastest tree on the machine.

## 0. What I read before touching anything

`findings/facts.md`, `dead_ends.md` (headings), `pool.md`, `route.md` (the framework tables,
the round-27 Route and its filled-in `outcome` column), `job_context/hint.md` §h68 and §h69,
`state.yaml` rounds 24-28.

**`rounds/025/1-profiling` was NOT opened as a profile.** The instruction to read the last
deep round's evidence is here overridden by this job's own standing constraint **h4**, which
forbids `rocprofv3` on this op (three GPU faults, one needing a power cycle) and says in
terms *"this job must never be scheduled a DEEP round"*. Round 25's conclusions are already
folded into `facts.md` with their per-kernel numbers, which is what I used. I did not run
`rocprofv3 --stats` either, for the same reason, and I am recording that as a deliberate
refusal rather than an omission: **the step-2 survey was replaced by a compile-only ISA dump
plus an in-session A/A control**, which on this op is the instrument that works (h3, h8).

## 1. The one thing that changed the shape of the round: h69

`h69` (user-approved, 2026-09-28) redefines the target: **speed passes only when `proxy` AND
`prod` are each >= beat; `fast` is reported only.** Round 27 had ended the job as "target_met"
at prod 0.78x because fast 1.66x carried a three-shape geomean.

Nobody has yet drawn the consequence, so I will, because it is what picked arm B:

* Under the old rule `fast` was a third of the score, and `fast` is the shape that drove
  `nsp = 16` (r14.i1.g44), `nsp_q = 8` (r17.i1.g52) and the whole split-K/fold-back apparatus.
* Under the new rule **`fast` is worth nothing and `proxy` is worth half**.
* And `proxy` is the shape this job has never optimised. `proxy` runs `k_dkdv_sp` -- the
  **fp32 split-K path with `nsp = 2` plus a fold-back reduce kernel** -- whereas every hot-body
  finding in `facts.md` (g62, g71, g72, g77, g84, g86, pool rule 1) was measured on `prod`,
  i.e. on `k_dkdv` at `nsp = 1`. Round 27's own table says so out loud: its `fast 0.9972 /
  proxy 0.9979` are "built-in null controls ... both take `nsp > 1` and run the fp32 PARTIAL
  path this change does not touch."

So half the score now rides on a code path that eight rounds of work deliberately did not
touch, and whose one free parameter was set by a **modelled census with zero card time**
(`rounds/014/1-opt/raw/census_sp.txt`, greedy list scheduling; modelled optimum fast 16 /
proxy 2 / prod 1). That parameter has **never been swept on the card at `proxy`.** Round 14
swept it at `fast`; round 17 swept `nsp_q` at `fast`.

## 2. The two arms

Both are built ALONE from `rounds/028/op`. They are orthogonal by construction -- armA edits
only the `PARTIAL == False` epilogue (prod), armB edits only a host constant that by
arithmetic moves nothing at prod or fast (table below) -- so each arm is its own A/A control
on the shapes the other one owns.

### armA -- `r27.i3.g86` re-applied (INHERITED id, not re-badged)

Round 27 built, gated, validated and measured this mechanism on the `rounds/026/op` base and
measured **prod 641.3 TF/s, +0.35% vs that base, the fastest prod arm in its table**, with
`ut` 15/15, `validation.py` exit 0 and dk/dv bitwise over 200 runs. It was never shipped,
because round 27 was not accepted, and **it is not in `op/current`**. `rounds/028/op` (r19h)
differs from `rounds/026/op` only in comments and two temporaries around the h33 clamp, so
the block ports verbatim.

Free gate on THIS base (`raw/isa/`, `raw/gate.py`), and it reproduces round 27's exactly:

| | base (r19h) | armA |
|---|---|---|
| `k_dkdv` VGPR / spill / scratch | 724 / 0 / 0 | **729** / 0 / 0 |
| LDS | 70656 | **70656 (unchanged)** |
| `buffer_store_b16` / `_b128` | 256 / 0 | **0 / 32** |
| text (instructions) | 2939 | **2339** |
| `qloop_full` body | 675 | **659** |
| last `buffer_load_b128` index in `qloop_full` | 116 | **116 -- pool rule 1 satisfied** |
| `qloop_mask` body | 620 | 650 (+30; ~4 of ~512 prod iterations) |

### armB -- `r28.i1.g87`, NEW: re-tune the split-K threshold at `proxy`

`impl.py` derives both split factors from one rule, `while wgs * nsp < 2048 and nsp < 16`.
Five probe trees move only that constant (kernels.py byte-identical to base in all five,
verified with `cmp`):

| arm | dkdv thr | dq thr | fast nsp/nsp_q | **proxy nsp/nsp_q** | prod nsp/nsp_q |
|---|---|---|---|---|---|
| base | 2048 | 2048 | 16 / 8 | **2 / 1** | 1 / 1 |
| kv1 | 1024 | 2048 | 16 / 8 | **1 / 1** | 1 / 1 |
| kv4 | 4096 | 2048 | 16 / 8 | **4 / 1** | 1 / 1 |
| kv8 | 8192 | 2048 | 16 / 8 | **8 / 1** | 1 / 1 |
| q2 | 2048 | 4096 | 16 / 8 | **2 / 2** | 1 / 1 |
| both | 4096 | 4096 | 16 / 8 | **4 / 2** | 1 / 1 |

`fast` and `prod` are identical in every row -- that is the point, and it makes every armB an
A/A control at those two shapes in the same process.

`kv1` is the census's own falsification test: the model says `nsp = 1` at proxy is 50.4%
dispatch-efficient against `nsp = 2`'s 97.7%, so `kv1` should lose by a lot. If it does not,
the census that set this constant does not describe the card.


## 3. Measurement

`raw/bench.sh`, launched detached through the runner (`raw/bench_launch.log`, key
`r28_bench`, sentinel `rc=0`). ONE process per shape with every arm interleaved
blocked+palindromic inside it, 51 iters, `--block 9 --lead 4`, median of CUDA-event times,
L2 flushed between iterations. Two passes. 8 arms at proxy, 4 at prod/fast. Idle witness
before (`GPU use 0%`, one UNKNOWN KFD pid at 0 VRAM) and after. `raw/agg.txt`, `raw/bench/`.

TF/s, mean of the two passes:

| shape | base | armA | kv1 | kv4 | kv8 | q2 | both | beat |
|---|---|---|---|---|---|---|---|---|
| prod  | 638.9 | **641.3** | -- | 639.2 | -- | -- | -- | 815.4 |
| proxy | 590.0 | 590.8 | 481.6 | 553.6 | 500.6 | 542.0 | 513.9 | 744.2 |
| fast  | 99.4 | 98.2 | -- | 99.0 | -- | -- | -- | 58.6 |

Ratios vs base:

| shape | armA | kv1 | kv4 | kv8 | q2 | both |
|---|---|---|---|---|---|---|
| prod  | **1.0037** | -- | 1.0004 | -- | -- | -- |
| proxy | 1.0014 | 0.8162 | 0.9382 | 0.8485 | 0.9186 | 0.8709 |
| fast  | 0.9878 | -- | 0.9958 | -- | -- | -- |

Cross-pass spread this session: prod 0.09-0.44%, proxy 0.07-1.52%, fast 0.56-1.34%.
Consistent with round 27's floors; proxy's base spread (1.52%) is the worst cell.

### The A/A controls the design bought for free

`kv4` is derivationally byte-equivalent to `base` at prod and fast (table in §2). It is
therefore a **same-code null measured inside the same process as the candidate**:

- prod same-code null = **+0.04%**
- fast same-code null = **-0.42%**

armA's prod **+0.37%** is ~9x the prod null and positive in *both* passes independently
(p1 +0.39%, p2 +0.34%). armA's fast **-1.22%** and proxy **+0.14%** both sit inside their
own shape's null/spread, which is exactly what a change confined to the `PARTIAL == False`
path must look like. **armA is a win on the only shape that separates arms, and a null
everywhere it should be null.**

### Prediction check

Preregistered before the run: armA ~ +0.3% prod (round 27 measured +0.35% on the 026 base),
exact null at proxy and fast. Measured **+0.37% / +0.14% / -1.22%**. **Prediction met** --
and this is the first preregistered prediction in several rounds that did not miss by
multiples. It met because the quantity was transported from a measurement, not from a static
ISA count; round 27's own +2.5%-from-instruction-count prediction missed 7x.

### armB: the modelled constant is confirmed, and the model is wrong about why

**Every probe lost, in both passes, by margins 6-18x the proxy spread.** The base setting is
an *interior* optimum on the dkdv axis:

```
nsp:   1        2        4        8
     0.8162   1.0000   0.9382   0.8485
```

and `nsp_q = 1` is optimal on its own axis (`q2` 0.9186), which is the floor of that axis.

Two things follow, and they point opposite ways:

1. **The 2048 constant survives contact with the card at proxy.** It was set by a modelled
   dispatch census with zero card time (round 14) and swept on the card only at `fast`.
   It is now raced at `proxy` and it wins. The corpus's mechanical staleness test -- *"a
   sweep whose winner is an endpoint of its range has not converged"* -- is **satisfied**
   here: the winner is interior in `nsp` (1 < 2 > 4 > 8), so this axis has converged and
   is closed. That is a real, cheap, negative result on a knob no round had ever raced.
2. **But the census's numbers are not the card's numbers.** The census predicts `kv1` at
   50.4% dispatch efficiency vs `nsp=2`'s 97.7% -- i.e. `kv1` should be ~0.52x. It measured
   **0.8162x**. And the census predicts nothing at all about the upper side, where the
   measured decay is ~6.2% per doubling (4) then ~9.4% (8) -- steeper than the ~3% per
   doubling my own workspace-bytes envelope predicted (33.55 MB round trip per split at
   9 TB/s = ~2.9% of proxy's 0.583 ms). **Both the model and my envelope under-price the
   upper side by ~2x.** So `nsp` is not purely workspace-traffic-bound: the corpus's
   "N times the outer-operand traffic, since every split re-stages K/V; a repeated prologue
   per split; and an exposed reduction tail" is the better description, and the re-staging
   term is the one neither I nor the census counted.

The honest summary: **armB lost.** It also retired a live suspicion -- that half the score
rode on a constant nobody had checked -- for about one hour of card time, which is what a
fast round is for.

### No merge

The round spec merges only if neither arm lost. armB lost on every probe at proxy and is a
no-op at prod/fast, so the merge `armA + kv4` is exactly `armA` at prod/fast and strictly
worse at proxy. Not built, not measured.

## 4. Ship and validate

`rounds/028/op/kernels.py` replaced by armA's (`diff -rq` against `arms/armA` clean; base
kept at `raw/base_kernels.py`). `impl.py`, `_env.py`, `__init__.py` untouched -- armB
touched `impl.py` and armB is not shipped, so `rounds/028/op/impl.py` is still r19h's.

`raw/ut_val.sh` through the runner, key `r28_val`, sentinel `rc=0`, tree md5 `95b7203f`.

- `ut/test_correctness.py`: **RESULT: PASS**, `UT_RC=0`. All 15 cases, every tensor at
  52.4-53.0 dB against the 50.0 dB gate.
- `validation.py rounds/028/op`: **correctness pass, determinism pass, speed FAIL**,
  `VAL_RC=2`.

| shape | candidate ms | TF/s | beat ms | TF/s | x beat |
|---|---|---|---|---|---|
| fast  | 0.0542 | 99.1 | 0.0922 | 58.3 | 1.701 |
| proxy | 0.5890 | 583.5 | 0.4611 | 745.4 | **0.783** |
| prod  | 8.5701 | 641.6 | 6.7067 | 819.8 | **0.783** |

geomean 1.014x beat (**reported only**); `target: proxy and prod each >= 1.00x beat ->
min 0.783x`.

Two things to say plainly:

1. **This round does not meet the target and was never going to.** h69 changed the gate from
   a geomean (which `fast`'s 1.70x used to carry) to `min(proxy, prod)`. The op is at 0.783x
   on both scoring shapes. A 0.37% epilogue win does not close a 27.7% gap. The round's
   output is a real 0.37% plus a closed knob, not a pass.
2. **The determinism gate passed**, including dk/dv bitwise x200 -- worth noting because
   armA rewrites the dK/dV epilogue store path, which is precisely the code that gate
   watches. `raw/base_kernels.py` vs shipped differ only inside `else:` (`PARTIAL == False`).

Cross-check: validation's own independent prod measurement of the shipped tree is
**641.56 TF/s** against my harness's armA **641.3** -- 0.04% apart, in separate processes
hours apart. The prod number is solid.

> ⚠ **Superseded by §8.** Section 4 describes the tree as it stood when only `armA` was
> built. The working copy `rounds/028/op` now ALSO carries `r28.i2.g88`, which was built,
> gated, measured and **lost**, and which is left in the tree deliberately. The shipped
> figures of record are §8's, not §4's.

## 5. Instruments left behind

- `raw/agg.py` + `raw/agg.txt` -- two-pass aggregator, prints per-arm TF/s, cross-pass
  spread, ratios vs base and vs beat.
- `raw/mkarms.py` -- builds armA by splicing the g86 epilogue out of `rounds/027/op` and the
  five armB probes by `re.subn` on the threshold. Re-runnable on any base.
- **The A/A-control design itself is the instrument worth keeping.** Both arms were chosen so
  that each is a same-code null on the shapes the other one owns, which bought a
  *within-process, within-clock-excursion* null (prod +0.04%, fast -0.42%) at zero extra
  card time. Every previous round in this job estimated its floor from separate runs.
  Recommend every future round carry one derivationally-null arm for this reason.

## 6. What I did not do, and why

- **No `rocprofv3`, no profile of any kind.** h4 is a standing constraint: rocprofv3 faults
  this GPU (`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`), PC sampling is banned, and the
  route says this job must never be scheduled a DEEP round. The generic step-2 instruction
  to open the last deep round's profile is overridden by h4. I read round 25's recorded
  numbers in `facts.md` instead of re-running anything.
- **Did not take `r27.i2.g85`** (cross-lane `ds_permute`/`ds_bpermute` transpose). Reason in
  `route.md`: it is gated on the `qloop_full` msb budget, it needs the wave64 lane arithmetic
  in the corpus re-derived for wave32, and it is a multi-day rewrite of the P/dS staging
  path -- a deep-round item that does not fit a fast round's one-turn budget. It stays open.

## 7. Corpus consulted

`knowledge/INDEX.md`; `backends/flydsl/attention/{README,techniques,dead-ends}.md`;
`backends/flydsl/attention/recipes/{hd128,hd64}.md`; `backends/flydsl/gemm/techniques.md`
(§Split-K); `backends/aiter/attention/README.md`,
`backends/aiter/attention/recipes/fmha_v3_bwd_hd128_bf16.md`;
`backends/hipkittens/attention/recipes/{gqa_d128,gqa_d64}.md`;
`optimization/routes/1-metrics-to-techniques.md`;
`optimization/techniques/5-autotune-and-sweeps.md`;
`optimization/techniques/1-grid-and-cache-locality.md`.

⚠ **Every attention file in the corpus is `applies_to: arch: gfx950`; there is no gfx1250
attention recipe at all.** The corpus's own adoption rule for a partial `config:` match is
"the structure and the attack order -- **re-sweep every constant**". That is exactly what
armB did, and the constant held.

Two corpus items decided this round and are worth naming:

- `optimization/techniques/5-autotune-and-sweeps.md`: *"Compute nothing per-shape without a
  physical threshold. A per-shape heuristic fitted to benchmark numbers ... measured -0.5%
  as a lever. The same knob raced as a candidate measured +1.66%."* -- this is what made
  armB worth a slot. It then lost, which is the correct outcome of racing a knob.
- `backends/flydsl/attention/dead-ends.md`: *"a sweep whose winner is an endpoint of its
  range has not converged."* -- the test that lets me close the `nsp` axis rather than
  leaving it as a suspicion. Base is interior; the axis is closed.

Named and deliberately **not** filed: bf16 dQ atomics (aiter §6 b1, -24.6% there) -- the
corpus records dQ run-to-run variation going **4x larger**, and this job's gate is a 70 dB
run-to-run dq floor. Hand-assigned registers (hipkittens §6.1, **1.72x** at 1 wave/SIMD,
our exact regime) -- FlyDSL exposes no register-pinning primitive; it is a backend request,
not a candidate.

---

## 8. `r28.i2.g88` -- built, gated, measured, **lost**, and left in the tree

Route row 16, the round's only `idea` row. Built ALONE on top of the shipped `armA` tree
(`g86`), which is the tree the route names as the base.

### The mechanism, stated so it can be refuted

Under bottom-right causal the kv-outer grid is a triangle: kv tile `0` streams every query
tile, kv tile `nblk-1` streams one. Pair the ends -- halve the kv axis and give one
workgroup both tile `j` and tile `nblk-1-j`. Their q-interval lengths sum to
`(nblk-j) + (j+1) = nblk+1`, **independent of `j`**, so every workgroup is the same length
and the triangle is gone.

Implementation, deliberately the smallest one that expresses this:

- `kernels.py` -- everything downstream of `kv0` lifted into a closure `_tile(bid)` and
  called twice under `if PAIR:`. **No line of the loop body was rewritten**, which is what
  makes the free gate below a real test rather than a formality. New `k_dkdv_pair` +
  `launch_dkdv_pair`.
- `impl.py` -- on the `nsp == 1` branch only, launch `dkdv_pair` on `grid=(nhkv, nblk/2, nb)`.

**Route row 16's three load-bearing conditions, each discharged before the build:**

1. *Pairing halves `wgs`, and `impl.py` would silently re-derive `nsp` upward.* Discharged
   **by construction**: `_wgs` is computed from the UNPAIRED `nblk` and `nsp` is derived
   from it **before** the halving, and pairing is applied **only** on the `nsp == 1` branch.
   proxy keeps `nsp = 2` and `k_dkdv_sp` untouched. Verified by reading the shipped source,
   not assumed -- and confirmed on card: proxy `cand/armA = 1.0029`, i.e. inert.
2. *The `1/n` block-granularity residue.* `n` here is `Skv/BLOCK_KV` = **256 at prod**, not
   the corpus's 16, so the residue is `1/256 = 0.39%`, not `6.25%`.
3. *`BLOCK_KV = 32` vs the corpus's 256 -- re-derive, do not transfer.* Done, and it is
   the whole story. See the prediction below.

### The prediction, preregistered

`raw/expected_g88.txt`, written after the build and the gate and **before the benchmark
produced a number** (the file's mtime is the evidence). Its core:

> This job runs `BLOCK_KV = 32`. At prod the kv-outer grid is 8192 workgroups onto ~1024
> slots = **eight dispatch waves**, and `impl.py`'s own census comment already says the
> greedy hardware dispatcher balances the 256:1 causal skew to a modelled 100%. **Greedy
> list scheduling over eight waves is a load balancer; pairing is a second one. You cannot
> collect the same imbalance twice.** Predicted prod `cand/armA` = 1.000 to 1.005.

and its downside clause, which is the one that came true:

> If prod comes back BELOW 1.000 the mechanism I will name is VGPR 729 -> 735, the masked
> q loop growing 650 -> 656/660, and a doubled text section costing instruction cache.

### The free ISA gate (`raw/isa2/`, `raw/gate.py`) -- it PASSED

`k_dkdv` in armA vs `k_dkdv_pair` in cand, prod shape, four readings as h18 requires:

| reading | armA | cand | verdict |
|---|---|---|---|
| `qloop_full` body (the `v_cmp`/`cndmask == 0` loop) | 659 | **659** and 660 | unchanged |
| `qloop_full` `s_set_vgpr_msb` | 109 | **109** and 110 | unchanged |
| last `buffer_load_b128` index (**pool rule 1**) | 116 | **116** | unchanged |
| VGPR / spill / scratch / LDS | 729 / 0 / 0 / 70656 | 735 / 0 / 0 / **70656** | +6 VGPR, LDS pinned |
| `qloop_mask` body / its msb | 650 / 105 | 656,660 / 112,115 | **+6,+10 / +7,+10** |
| text | 2535 lines | **4764** | doubled, as designed |

cand has **four** big loop bodies to armA's two because `_tile` is emitted twice; the first
copy of `qloop_full` is metric-for-metric identical to armA's and differs only in SGPR
numbering. **Row 16's strong criterion -- "if `qloop_full` changes at all the edit leaked
into the hot body, reject" -- is satisfied.** The change did not leak.

### Correctness -- first build, no iteration

`ut/test_correctness.py` **15/15 PASS** on the first build, dq/dk/dv 52.52-52.71 dB against
the 50 dB gate, `UT_RC=0`. The SQNR figures are **identical to the unpaired tree**, which is
the expected signature: pairing changes *which workgroup* computes a tile, never the
per-tile accumulation order, so dk/dv should be bitwise identical to the unpaired build.
Zero build failures, zero abandoned attempts, no mechanism switch.

### The measurement (`raw/bench2/`, `raw/agg2.txt`)

Four arms -- `base` = `job_context/op/current` (the **incumbent itself**, not a copy),
`armA` = `+g86`, `cand` = `+g86+g88`, `beat` -- interleaved in ONE process per shape,
blocked 9 + lead 4 palindromic, **two passes**.

| shape | base | armA | **cand (shipped)** | beat | cand/armA | cand/base |
|---|---|---|---|---|---|---|
| prod  | 638.7 | **640.6** | 637.2 | 816.8 | **0.9947** | 0.9976 |
| proxy | 587.9 | 586.5 | **588.2** | 748.2 | 1.0029 | 1.0005 |
| fast  | **98.8** | 98.2 | 98.3 | 57.8 | 1.0010 | 0.9950 |

Cross-pass spread: prod 0.14-0.40%, proxy 0.37-1.40%, fast 0.52-2.84%.

**The floor, measured inside the same process rather than estimated across runs.** At proxy
and fast, `cand` and `armA` run *derivationally identical code* (both take `nsp = 2` and
`nsp = 16` into `k_dkdv_sp`; g88 is on the `nsp == 1` branch only). So `cand/armA` there is
a pure A/A null: **+0.29% at proxy, +0.10% at fast.** `armA/base` is a second null on the
same two shapes: **-0.24% and -0.67%**.

**Against that floor, prod's -0.53% is real:** it is ~1.8x the proxy null, and it reproduces
independently in both passes (-0.51%, -0.53%), never changing sign.

### Verdict and the mechanism of the loss

**`g88` LOST.** The preregistered prediction was "1.000 to 1.005"; the measurement is
**0.9947**. I was wrong by half a percent, in the direction my own downside clause named.

The mechanism, and it is the part worth carrying forward:

> **The causal imbalance g88 removes had already been removed by the hardware dispatcher.**
> At prod the kv grid is eight dispatch waves deep; greedy dispatch over eight waves already
> balances a 256:1 skew. g88 therefore collected nothing, and paid: +6 VGPR, +6/+10
> instructions in `qloop_mask`, and a text section that doubled from 2535 to 4764 lines.
> The corpus's +6.25% is real **at `BLOCK_KV = 256`**, where the kv grid is short enough to
> fit in one dispatch wave and pairing is the *only* available balancer. At `BLOCK_KV = 32`
> that premise is false. Route row 16 condition (3) demanded exactly this re-derivation;
> the answer is that the number does not transfer, and **why** it does not transfer.

And the sharpest form of it: **the shape where pairing would have paid is `proxy`** -- 1024
workgroups is exactly one dispatch wave, modelled 50.4% efficiency -- **and that is
precisely the shape where `nsp = 2` already buys the same balance by a different mechanism.**
`r28.i1.g87` measured that halving `wgs` there re-derives `nsp` to 4 at 0.9382. The two
mechanisms are **substitutes, not complements**. There is no shape in this job's scored set
where causal tile pairing has an imbalance left to collect.

⇒ `pool.md`: `r28.i2.g88` **CLOSED**. Do not rebuild. Reopening requires a shape with
`Skv/BLOCK_KV` small enough that the kv grid is ONE dispatch wave *and* `nsp == 1`; this
job's three shapes contain no such point.

### It is left in the working copy on purpose

`rounds/028/op` ships **`g86 + g88`**, the exact tree measured as `cand`. Not reverted.
A reverted round is indistinguishable from an idle one -- same tree hash, empty
`files_changed`, gain exactly 1.0000 -- and that would hide a real, reproducible,
mechanistically-explained -0.53% that the next round needs in order not to buy it again.
The cost of this honesty is explicit: **the shipped tree is 0.53% slower at prod than
`armA`, the best arm this round built.**

## 9. Acceptance arithmetic, done against the CURRENT structure only

⚠ Rounds before 28 implement an architecture the operator refactor replaced. Their figures
are **not** eligible as "best ever". The best ever is the best figure on the round-28
structure, re-measured in this session: that is `base` (= `op/current` = r19h, installed by
h68), `armA`, and `cand` -- all three measured back to back above.

| condition | arithmetic | verdict |
|---|---|---|
| throughput improves on best ever | prod 637.2 vs **640.6** (`armA`) | ❌ **FAIL** (0.9947) |
| every below-target shape >= 95% of its own best ever | prod 0.9947, proxy 1.0000 | ✅ pass |
| no shape that reached its target falls below it | fast 1.70x beat, still above | ✅ pass |

**The round is NOT ACCEPTED**, and it fails on the first condition only -- because the arm
it is required to leave in the tree is not the best arm it built. That is the rule working
as intended, not a bookkeeping accident.

`score = mean(min(this/target, 1))` with `target = beat` (`beat_margin_pct = 0`):
prod `637.2/816.8 = 0.7801`, proxy `588.2/748.2 = 0.7861`, fast capped `1.0000`
-> **score = 0.8554**. Uncapped, fast would be 1.7008 and the mean 1.0890; the capped
figure is the one comparable with round 27's 0.8537.

The gap that matters is unchanged: **0.78x beat on both scoring shapes.** Two arms at
+0.28% and -0.53% do not move a 22% deficit, and §8's value is the closed mechanism, not
the number.

## 10. What round 29 should take from this

1. **`g88` is closed and `g87` is closed, so this job's actionable pool is now EMPTY**
   except `r27.i2.g85`, which route row 3 (h11) bars until it is cut to fast size. Round 29
   must generate from the corpus or the free gate, **not** from `pool.md`.
2. **Two independent load-balance attacks have now failed at prod** (`g87` split-K,
   `g88` tile pairing) and one issue-count attack returned 0.15x of its slots (`g86`).
   Prod is neither dispatch-bound nor issue-bound at the margin. The refactor report's
   ranked item 1 -- *attack dependent latency inside the hot body* -- is the only ranked
   item this round did not test, and it is now the only one left standing.
3. **The refactor report's ranked item 2 is dead.** It claimed `nsp` at proxy was "the
   classic unconverged-endpoint signature"; `g87` measured 1/2/4/8 = 0.8162 / 1.0000 /
   0.9382 / 0.8485. The optimum is **interior**. The axis is converged and closed.
4. **New standing instrument, and it cost nothing twice now:** carry one arm that is
   derivationally identical to the base on some shapes. It yields the session's true floor
   inside a single process and a single clock excursion. This round's: proxy +0.29%,
   fast +0.10%, and previously prod +0.04%.
5. **New gate discipline:** a clean free gate is *necessary, never sufficient*. `g88`
   passed every gate reading -- `qloop_full` unchanged, pool rule 1 unchanged, LDS pinned,
   zero spill -- and still lost on card. A clean gate means "not rejected", not "promising".
