# Round 30 -- the decision

**Selected: `r29.i1.g89`** -- stage `k_dqg`'s dQ epilogue through LDS so 256
`buffer_store_b16` become 32 `buffer_store_b128`. One card arm. Two instruments run around it.

Full exchange: [`dialogue.md`](dialogue.md). Long-form reasoning behind the merge:
[`synthesis_notes.md`](synthesis_notes.md). Machine-readable decision: [`plan.yaml`](plan.yaml).

---

## What was chosen, and what it predicts

`g89` transplants a mechanism this job has already measured. `g86` did byte-for-byte the same
thing to `k_dkdv`'s dK/dV epilogue and returned **+0.35% / +0.37% / +0.32%** at prod across
three sessions (`facts.md:587`). `k_dqg` still carries the pattern `g86` removed: **256
column-scattered 16-bit stores**, each covering 32 B of a 128 B line, with nothing to overlap
them at 1 wave/SIMD.

The reason it is worth more in `k_dqg` than it was in `k_dkdv` is that **the epilogue is a
per-workgroup fixed cost, and `k_dqg` pays it twice as often**. Both agents reached that
independently, from different counters:

| | `k_dkdv_0` | `k_dqg_0` | source |
| --- | --- | --- | --- |
| wave32 workgroups | 8,192 | **16,384** | `kernel.yaml` grid 262,144 / 524,288 at WG 32 |
| epilogues per SIMD | 8 | **16** | `2-kernel-profiling/analysis.md:67` |
| cycles per wave | 1,168,415 | **339,716** | `2-kernel-profiling/analysis.md` |

Exactly 2.00x, from two sources. And `k_dqg`'s waves are **3.4x shorter**, so it amortises
every fixed per-workgroup cost over 3.4x less work. Time share (34.9% vs 61.1%) is the wrong
denominator, and using it is why this looked sub-floor when round 29 filed it.

**The prediction is the reviewer's, not mine.** I had proposed a relative band (+0.4 to
+1.0%). The reviewer replaced it with per-epilogue arithmetic that is tighter and does not
depend on a percentage: `g86` saved ~30 us at prod (0.35% of 8.573 ms) over 8 epilogues per
SIMD, so **~3.75 us per epilogue per SIMD**; `k_dqg` runs 16, so **~60 us, band 30-120 us**,
which on this round's 8.352512 ms is ~+0.72%. `fast` takes `nsp_q > 1` and runs the fp32
PARTIAL path this change does not touch -- it is the built-in null control, exactly as `g86`
used it.

Three ways it can be refuted, and they are not the same failure:

1. **prod gain at or below this session's own re-measured floor**, gate passed -- then `g86`'s
   gain was not store width and did not scale with epilogues per SIMD, which reaches back and
   weakens `g86`'s own accepted justification.
2. **prod saving outside 30-120 us**, gate passed -- right sign, wrong mechanism.
3. **the gain arrives with a >= 1% sclk shift** -- this refutes *attribution*, not the gain.

Point 3 is the reviewer's best contribution and I had not weighed it. **P77**
(`facts.md:241-246`) measured a traffic-only change in `k_dkdv` moving prod sclk from
1818-1820 to **1876-1877 MHz (+3.2%)** while raw time moved 1.83% -- i.e. **~0 at matched
clock**. `g89` is a traffic change. A +0.7% result with a +1% clock behind it may be the
epilogue contributing nothing.

---

## The threshold argument, which the reviewer conceded

The reviewer filed `refuted_if: prod < +0.2%`. Prod's same-code floor reproduces at **0.32%**
on GPU 3 (`facts.md`, round 25). So an arm returning +0.25% would be "not refuted" and
**unshippable** under campaign correction 4 -- the round would bank a confirmed mechanism it
cannot ship. That gap is how round 15 shipped a prod regression and had to be reverted by
hand. The floor is also a property of a session, not a constant: round 27 measured prod
cross-pass spread at 0.02-0.34% and explicitly corrected a floor that had been carried
forward.

**Resolution: the threshold is this session's own re-measured same-code prod floor, measured
before the arm is priced.** The reviewer accepted this in turn 1.

---

## What was argued and how it resolved

### The round's largest output cost no GPU time

Three of the four profiling candidates and **one of my own two arms** died on evidence
already in the corpus.

- **profiling `c1` (raise occupancy)** -- the reviewer supplied a stronger, older kill than
  mine. `dead_ends.md:181-189` (`r18.i2.g55`): **395 live-in VGPR of 724** in `.LBB0_8`, of
  which 384 are the 256-VGPR dK/dV accumulator pair plus the 128-VGPR `g21` prefetch tuple.
  Reaching 512 needs >= 212 removed, which necessarily includes the prefetch tuple that IS
  `g21`'s +8.3% mechanism -- and **even deleting it entirely leaves 596 > 512**, so `BLOCK_KV`
  would also have to halve and undo `g06`'s 1.677x. `:286-291` closes the axis from the other
  end: `g60` gave the registers back and lost **17.27%**. My own ground (armC freed 193 VGPR
  below an unchanged rung and lost 4.55%) is now the secondary one.
- **profiling `c2` (S/P recompute)** -- answered from source. `kernels.py:1322-1334` confirms
  7/5 survives; every removal route is closed by measurements already in this job (h64's
  0.39x, h63's jurisdiction, `g76` failing its own gate by 0.07 ms, the 4-wave route's
  0.872x). The reviewer's version of this closure leaned partly on `P2.md`, which it correctly
  marked `unverified`; I re-grounded it on in-job measurements so it does not rest on a figure
  from outside the corpus. Same verdict, verified footing.
- **profiling `c3`** -- no mechanism. **`c4`** -- fast-only, and fast is at 1.665.

### I was wrong about `r29.i2.g90`, and the reviewer caught it

I filed it carrying round 29's pool text forward verbatim -- *".LBB0_8 is a real fraction of
prod's kv range, not a rare tail"* -- **without opening the kernel**. The kernel says the
opposite, in its own comment:

> `kernels.py:880` -- "The masked loop (**98.4% dead at prod**, a handful of iterations)"

I re-derived it independently: **256 masked trips of 16,512 total = 1.55%**. Deleting the loop
*entirely* caps at `0.0155 x 0.34860 = 0.54%` of prod, against a 0.32% floor; an x2 unroll is
a fraction of that. **Marked stale for reflect.** This is the single most important correction
of the exchange and it is exactly what the blind stage exists for.

### `C2` was re-specified by the reviewer and I took it in full

I proposed a locked-clock pair asking "does prod time scale as 1/f, and does flat scaling mean
power-limited?" The reviewer found three faults and all three hold:

- **the gate contradicted itself** -- `risk` said run if C1 >= 0.90 x cap, `settled_by` said do
  not run if C1 >= 0.95 x cap. Those cannot both be policy. A contradiction inside a single
  candidate I wrote, which I had not seen.
- **the reading was a non-sequitur** -- both setpoints sit below 1753 MHz, itself below cap,
  so **power cannot bind in either arm by construction**. Sub-1/f scaling there measures the
  fixed-nanosecond memory-latency fraction, not a power wall.
- **the prediction was already contradicted** -- `facts.md:254-256`: `k_dkdv` non-WMMA CPI
  rose 1.164 @ 1012 MHz to 1.759 @ 1806 MHz. My "tracks 1/f within 3%" was dead before it ran.

The re-specification makes the instrument **more** useful than I wrote it: fit `t = a + b/f`
over two pinned points and report `a/t` -- the share of prod time no instruction or
latency-hiding edit in `k_dkdv` can reach at any clock. That is a direct ceiling on the job's
largest open class (h60; h16's 0.15x coefficient still governs `k_dkdv`). Predicted 0.15-0.40.

---

## What remains open

Two disagreements survive, both narrow, both recorded verbatim in `plan.yaml` with the
reviewer's side stated as it would state it.

**1. Does the proxy channel work?** The reviewer proposed the `proxy% > prod%` sign test in
turn 1, I made it load-bearing, and the reviewer then **withdrew its own test** on the grounds
that proxy was not profiled this round and runs a dispatch path prod does not.

I reject the withdrawal, on one line of `impl.py`. **`nsp` and `_nsp_q0` are two different
split parameters**, derived by two separate loops over two separate workgroup counts, and
`use_g` -- the switch selecting `k_dqg`, the only kernel `g89` touches -- reads `_nsp_q0` and
never `nsp`. At proxy `_wgs_q0 = (4096/64)*32*1 = 2048`, the guard `while _wgs_q0*_nsp_q0 <
2048` is **false at entry**, so `_nsp_q0 = 1` and **proxy runs `k_dqg`**. Separately `_wgs =
1024 < 2048` so `nsp = 2` and proxy *does* take `k_dkdv_sp` -- which is why `facts.md:587-590`
correctly called proxy a null for `g86`, a `k_dkdv` change. `g89` is not a `k_dkdv` change.
The reviewer's own citation corroborates it: round 28 raced `nsp_q = 2` **at proxy** against
the shipped default and measured 0.9186 (`pool.md:346`), which is impossible unless proxy's
default is `nsp_q = 1`.

The reviewer's *real* objection -- that I stated the test with no threshold, so a genuine win
landing at proxy <= prod would be wrongly refuted -- is fair and I take it. The threshold is
derivable without any proxy counter data, from dispatch waves: prod 16,384/1024 = 16 waves,
proxy 2,048/1024 = 2, so `dt_proxy/dt_prod = 1/8` while `t_prod/t_proxy = 14.70`. **Predicted
ratio 1.84.** The shorter proxy wave shortens the wave *body*; the epilogue is the same 256
stores either way, which is why the ratio exceeds 1.

I keep it, **demoted to corroborating and barred from refuting on its own**. Two channels on
one denominator is one channel, and campaign correction 3 says a prod-only artifact has fooled
this job twice, most recently by 16.25%. *This settles inside this round, for free* --
`benchmark.py` already times proxy.

**2. Is `C2` gated?** Metric, prediction and reading are agreed. The reviewer says run it only
if C1 < 0.90 x cap. I say ungated, because the re-specification dissolves its own gate: by the
reviewer's own second ground, both setpoints are below cap and power cannot bind in either
branch of C1, so a quantity independent of whether the cap binds cannot be gated on whether
the cap binds. **No measurement settles this and I will not dress it as one** -- it is a
scheduling rule. Recorded as: C2 runs last, ungated, and is the first thing dropped if card
time runs out, which is the reviewer's outcome by another route in the branch where it matters
least.

---

## Why one card arm is the right round, not a thin one

I considered four second arms and declined all four rather than manufacture one: h60's
`k_dkdv` rotation drain (remedy unknown; the k_dq lab measured `ku2`, the identical transform,
at **+6.5% = a loss**, h71 -- and it targets the 0.15x kernel), fma-softmax folding in `k_dqg`
(h71 already measured it worse), `DQ_BQW` 64 -> 128 (spills at 992/1024, an **h3 kill**), and
deleting `k_delta_bshd` (`DQ_DFUSE = False`, and h71 measured the fusion null).

Rounds 27 and 28 were each spent on an arm aimed at `k_dkdv` with a metric that moved as
designed and a clock that did not. Adding a fifth is how that becomes three.

**Neither instrument is filed to the pool, and that is deliberate.** The pool rule is explicit
-- file the *changes* you could not try, *do* the measurements you would have needed -- and
`turbo-flydsl-attn-fwd-dense` is the counter-example: it filed `r6.i2.g55` and `r6.i3.g56` to
the pool, then spent rounds 7, 8 and 9 inventing instruction-count candidates against a power
wall, losing 1.0% and then 5.5%, with the instrument still unbuilt. This round has profiling
behind it, a GPU and counter access. It takes the measurements.

---

## Bookkeeping

**Round 30 consumes no global `g`.** Both new items are instruments (`r30.i0.C1`,
`r30.i0.C2`), and instrument ids `r*.i0.*` do not consume one -- the standing rule recorded in
`pool.md` and applied at rounds 14-19 and 25.

⚠ **Defect, recorded and not quietly fixed** (per `pool.md`'s own rule and round 24's
precedent): `pool.md:3` reads *"Highest global id allocated: **g86**. Next free id is
**g87**."* That is stale. `g87`/`g88` (round 28) and `g89`/`g90` (round 29) are all allocated
and carry entries in four files. **Highest is `g90`; next free is `g91`** -- which `route.md`'s
round-29 table already states correctly, so the two files disagree. Round 30 leaves `g91`
free.
