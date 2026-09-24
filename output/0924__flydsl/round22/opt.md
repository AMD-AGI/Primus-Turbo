# Round 22 -- 1-opt (fast round)

op: attention backward | backend: FlyDSL | arch: gfx1250
bulk: `/tmp/op-evolve-gfx1250-flydsl-attn-bwd-20260917-115934-r022`
parsed: `rounds/022/1-opt/raw/`

Round 21 was **not accepted**, so per the round rules the corpus and the other backends
were mandatory reading this round. `explored.consulted` at the bottom lists every file.

---

## 0. What this round is

Round 21 ended by writing down a **signed, directional fact**: that `k_dkdv` and `k_dq`
have *opposite signs* on the prefetch-spread lever (depth 1->2 = **+1.65%** on `k_dkdv`,
**-19.33%** on `k_dq`), and it opened two candidates that are that fact's two halves:

- `r22.i1.g67` -- tighten/shorten `k_dq`'s prefetch (the minus half, pushed the other way)
- `r22.i2.g68` -- spread `k_dkdv`'s further without deepening (the plus half, pushed further)

Both were route rows 10 and 11, both unexecuted. This round executed both. **The fact did
not survive.** Details in section 5.

---

## 1. Look

GPU idle confirmed before touching anything (`rocm-smi --showpids` empty, `--showuse` 0%).
`rocprofv3 --stats` on `op/benchmark.py` reproduced the standing dispatch picture: two
kernels carry the op, `k_dkdv` and `k_dq`, and nothing else is material. That is the same
shape round 17's deep profile recorded, which was read in full this round as instructed
even though round 20 was accepted after it.

Because this is a fast round, the instruments taken were the ones that cost no code and no
card time. Two of them are new to this job and are recorded under `r22.i0.*` (instrument
ids do not consume a `g`, round 14-19 precedent).

### `r22.i0.a` -- ISA census of both kernels (free)

Whole-file counts from the offline screen's `21_final_isa.s`:

| kernel | `buffer_load_b128` | `ds_store_b128` | `ds_load_tr16` | `v_wmma` | `v_nop` |
|---|---|---|---|---|---|
| `k_dkdv` | 160 | 80 | 80 | 128 | 52 |
| `k_dq`   | 160 | 32 | 32 | 192 | 41 |

Two things worth banking:

- **`k_dq` does have a global->register->LDS->register staging round trip for K/V** (32+32).
  This matters for section 6: campaign correction 6 closed the TDM / cluster-global-to-LDS
  family on the grounds that *`k_dkdv`'s Q/dO have no staging burst to replace*. That
  argument is about `k_dkdv`. It does not, on its face, reach `k_dq`'s K/V.
- The `v_nop` counts (52 and 41) are pure WMMA-hazard bubbles. The corpus is explicit that
  these must not be deleted, only filled with independent work.

### `r22.i0.b` -- power and clock trace at prod (free, new to this job) -- NEGATIVE RESULT

Nobody in twenty-one rounds has traced power. Motivation: the corpus records that on a
healthy gfx1250 the part sits **pinned at its 2500 W cap** (2497-2502 W) at 1699-1703 MHz
against a 2400 MHz maximum, and that when time is energy/P_limit, *removing work converts to
time but removing stalls does not.* If this op were power-bound at prod it would explain
the single most stable pattern in this job's ledger: **every pure-scheduling edit across
twenty-one rounds has returned either ~0 or a catastrophe, never a win above the floor.**
It was also suggestive that `benchmark.py`'s own sclk witness reads 1100 at fast but
1051-1067 at prod and proxy -- the signature of a cap being hit under load.

A 1 Hz sampler ran across the whole scored block, then a dedicated 0.5 Hz sampler
(60 points) ran across a single `cur_a` prod benchmark. Both in `raw/power_trace.txt`.

**The hypothesis cannot be tested on this box, and the suggestive sclk reading was a red
herring.** As measured:

| | idle | across a full prod benchmark |
|---|---|---|
| `Current Socket Graphics Package Power (W)` | 855.0 | 851-855, **median 853** |
| sclk | 1100 MHz | **1100 MHz, all 60 samples** |

The power telemetry **does not respond to load at all** -- a full-rate prod run moves the
reading by under 0.5%, which is not a measurement, it is a static register. The power-cap
question is unanswerable here and future rounds should not spend time on it.

The clock result is the useful half, and it is worth banking as a standing fact: **sclk is
flat at the degraded 1100 MHz ceiling under sustained load; it does not droop.** The
1051-1067 witnesses `benchmark.py` prints at prod/proxy are brief transients caught at a
sampling instant (the scored block's 1 Hz trace saw 67 samples at 1100 against 12 in
1045-1055), not a sustained throttle. **"The card was throttling" is now removed as an
available explanation for any measurement in this job, past or future.**

## 2. The two arms, and why they are these two

The round rules require two ideas, built alone from `op/current`, measured apart. Both came
from `pool.md` and therefore **keep the ids they already have**:

- **arm A = `r22.i2.g68`** -- built as `A68`.
- **arm B = `r22.i1.g67`** -- built as `B67`.

Plus one throwaway probe, which is free and unlimited:

- **`P70`** -- deletes `k_dq`'s prefetch entirely (depth 1 -> 0). This is `g67`'s *form 2*,
  which `pool.md` says explicitly must be done **before** form 1: a subtractive probe is an
  upper bound on the whole direction. Round 20's P1/P3 are the precedent.

`op/current` was not touched. Every arm is a separate copy under
`rounds/022/_scratch/arms/`, each built alone. Both incumbents (`cur_a`, `cur_b`) are
physically separate rebuilt copies, which is this job's noise-floor protocol.

### Builds

All arms compiled clean on the offline screen
(`COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250`), **verdict pass, spill 0** in every
case. No build failures this round. `h3` (spill > 0 is an instant kill) never fired.

---

## 3. `B67` (`r22.i1.g67`) -- killed at the offline gate, zero card time

The edit is one line each side: a `rocdl.sched_barrier(0)` hard fence immediately before
and after `k_dq`'s `_ldkv` prefetch group, pinning the 32 loads as one indivisible clump.
`pool.md` gave it a **load-bearing gate**: the `buffer_load_b128` index span in
`.LBB0_2` must come out **narrower** than the incumbent's.

Measured on the dumped ISA, before spending any card time:

| | incumbent `cur_a` | `B67` |
|---|---|---|
| `buffer_load_b128` idx span in the loop body | 51-100 (**49 wide**) | 29-92 (**63 wide**) |
| body length | 742 | 816 |
| mid-body partial `s_wait_loadcnt` | 0 | **5**, at 11-27% depth |
| `v_nop` | 28 | 49 |

The fence made the clump **wider**, not narrower, and grew the body by 74 instructions with
five new mid-body waits. That is precisely `r21.i4.g66`'s failure signature -- the one that
cost 21.93%. **Gate failed, arm not taken to the card.**

The finding underneath is worth more than the arm: *"tighten `k_dq`'s prefetch" is not
expressible through this lever.* `sched_barrier(0)` is a scheduling **boundary**, not a
clamp; fencing around a group gives the scheduler two smaller regions to fill and it fills
them by pulling loads earlier. Asking this backend to tighten a clump with barriers
reliably loosens it. That closes form 1 of `g67`.

This discharges route row 10's arm half for free and is the cheapest kill in the round.

---

## 4. `A68` (`r22.i2.g68`) -- the prediction, written before the card

The edit inserts into `k_dkdv._body`, immediately before the `qp` staging block:

```python
_VMEM, _MFMA = 0x020, 0x008
if const_expr(carry):
    for _g in range_constexpr(32):
        rocdl.sched_group_barrier(_VMEM, 1, 0)
        rocdl.sched_group_barrier(_MFMA, 2, 0)
```

i.e. interleave the 32 prefetch loads one-for-two against the WMMAs. It does not deepen
(the corpus rule *"staging depth pays exactly once"* forbids depth 3 and was respected).

### The cover model, and the prediction it produced

Reading the two disassemblies together produced a model that **supersedes round 21's
"opposite signs per kernel"**. The binding quantity is not the kernel's WMMA/LDS ratio. It
is the **worst-case load-to-use cover distance**: the number of instructions between a
prefetch load's issue and the next *full* `s_wait_loadcnt 0x0` that drains it.

| | last prefetch load idx | next full drain | **cover** |
|---|---|---|---|
| `cur_a` `k_dkdv` | 188 | idx 13 of the next iteration | **~603 instructions** |
| `A68` `k_dkdv` | 789 | idx 25 of the next iteration | **~41 instructions** |

Spreading the loads across the body does not hide them better. It moves the *last* one to
the far end of the body, and the drain at the top of the next iteration is then ~41
instructions away instead of ~603. Cover collapses by 15x.

**Prediction, recorded before any card time: `A68` loses, and loses large.**

### Measurement

`A68` passed correctness first, per `h1`, before any benchmark:

- unit tests **15/15 PASS**, 52.30-52.98 dB across all shapes
- `op/validation.py`'s own precision check **pass** (fast 52.61/52.64/52.84,
  proxy 52.52/52.57/52.67, prod 52.56/52.60/52.71 dB)
- determinism x200 **bitwise pass**
- speed gate **FAIL** (`RC_val_A68 2`)

Speed, as measured in the scored session, no adjustment (section 7 has the full table):

| shape | `cur_a` | `A68` | delta |
|---|---|---|---|
| prod  | 517.21 TF/s | **360.39 TF/s** | **-30.32%** |
| proxy | 438.79 TF/s | **326.70 TF/s** | **-25.55%** |
| fast  | 55.62 TF/s | 51.95 TF/s | -6.60% (inside a 12.80% floor -- uninformative) |

**Prediction confirmed.** The cover model called the sign and the rough size before the
card was touched. `r22.i2.g68` is DEAD.

---

## 5. What actually died this round: round 21's signed fact

This is the round's product, and it is a falsification rather than a win.

Round 21 wrote: *"the two kernels have OPPOSITE SIGNS on the prefetch-spread lever."* The
evidence was that depth 1->2 gained +1.65% on `k_dkdv` and lost 19.33% on `k_dq`, and the
proposed discriminant was body shape -- `k_dq` compute-dense (96 WMMA, 16+16 LDS),
`k_dkdv` memory-heavy (64 WMMA, 40+40).

`A68` is the direct test: same tool as `g66`, aimed at the kernel the model says should
have the *opposite* sign. It lost ~30%. **Spreading the load clump loses on both kernels.**
There is no sign flip. The model is false.

The cover model explains both results and `g62`'s win with one variable:

- `g66` on `k_dq`: spread the loads -> cover collapsed -> **-19.33%**
- `A68` on `k_dkdv`: spread the loads -> cover collapsed -> **-30%**
- `g62` on `k_dkdv`: depth 1->2 added a *whole extra iteration* of cover -> **+1.65%**

And it prices `g62` honestly. `g62` banked +1.65%. The corpus records this part's measured
**null-control floor at 1.57% (n=5)**. The banked win sits essentially on the floor. It
also explains why it is not larger: `k_dkdv`'s body opens with `s_wait_loadcnt 0x0` at
idx 13 followed by 128 `v_mov_b64` -- the FlyDSL non-rotating-carried-register block copy.
Depth 2 therefore buys roughly *one* iteration of cover rather than two, and pays 128 copies
per iteration for it. That is the real ceiling on the whole prefetch-depth axis, and it is
why `r11.i5.g37` (carried regs do not rotate) is the load-bearing constraint it is.

---

## 6. Free work that did not need the card

### `r22.i0.c` -- cluster / TDM API census (free)

The corpus's top-ranked untried lever for this part is TDM `TENSOR_LOAD_TO_LDS`
(**+40.68%, n=5, measured on gfx1250**), with cluster multicast (+6.53%) behind it. Before
opening a pool entry on either, the obvious zero-cost question: **is any of it expressible
in this backend at all?** Answer, from the installed FlyDSL:

- `flydsl/expr/rocdl/tdm_ops.py` -- a complete TDM surface: `make_tensor_descriptor_2d`,
  `tensor_load_2d`, `tensor_store_2d`, `tensor_wait`, gather descriptors, `l2_prefetch_tile`,
  `compute_padding_encoding`. **Expressible.**
- `flydsl/expr/rocdl/cluster.py` + `cluster_load_async_to_lds_b{8,32,64,128}` --
  cluster multicast is present, `cluster_barrier` with one-wave signal semantics,
  `compute_mcast_masks`. **Expressible.**

But the census also kills the version I wanted. **Every cluster load FlyDSL exposes is
`..._async_to_lds` -- global to LDS. There is no register-destination cluster load.** The
attractive idea (swap `k_dq`'s 32 `buffer_load_b128` for multicast loads *to registers*,
a drop-in with no LDS and no structural change) does not exist here. Anything in this family
must route through LDS, which is exactly the path campaign correction 6 and `r5.i2.g18`
closed, and which round 20's `P1` priced at **-6.98%** when the round trip was removed.

That leaves one live question, and it is a **zero-card-time** one, which is why `g69` is
opened as gated rather than as an arm: campaign correction 6's actual argument is that
`k_dkdv`'s Q/dO registers are **dual-use** -- consumed directly as WMMA operands *and*
stored to LDS for the transposed consumer -- so a direct global->LDS path does not replace
the round trip, it forces a second load. Section 1's census shows `k_dq` *does* carry a
real 32+32 staging round trip for K/V. **Whether `k_dq`'s K/V are likewise dual-use is
answerable by reading `_ldkv`'s consumers, with no build and no card.** If they are, the
family is closed on `k_dq` too and `g69` dies for free. If they are not, this is the only
untried lever on this part with a measured double-digit number behind it.

### Candidates rejected from the ledger, before building

- **Unroll-by-2 / ping-pong on `k_dkdv`.** This is `r10.i2.g28`'s recorded form:
  **-7.8% prod in three separate sessions**, with the recorded mechanism *"doubled the load
  batch covered by the drain"* -- which is the cover model again, stated in round 10's
  vocabulary. `r11.i5.g37` names it the only writable form. Not built. The ledger saved a round.
- **LDS double-buffering.** Re-confirmed dead twice independently this round: round 21
  killed it free on `backends/flydsl/attention/dead-ends.md` (Q-outer -2.3...-4.7%, KV-outer
  **-16.5%**; `k_dkdv` is KV-outer), and the corpus read re-quoted the same numbers.
- **A fifth `sched_group_barrier` arm** (fill `k_dkdv`'s 52 `v_nop` hazard slots with the
  `ds_store_b128` group). Tempting -- it touches no loads, so cover is preserved, and it
  adds work into bubbles rather than moving work. But the family is **0-5** on this op
  (`g47` -3.12%, `g49` -16.25%, `g53` -13.24%, `g66` -21.93%, `g68` ~-30%), and `g47` was
  this exact tool on this exact body. Declining to build a sixth loss is the correct use of
  a fast round's budget. Not built, and not pooled.

---

## 7. Scored session -- as measured

One process per shape, 51 iterations, median, palindromic order, both incumbent copies
rebuilt in-session, idle-device witness (`--showpids`, `--showuse`) before and after,
bounded in-container `dmesg | tail` between GPU blocks per `h15`. Launched detached through
`docker exec fa-repro` with the sentinel pattern (`setsid ...; echo $? > D/rc`), polled for
`rc`. `RC=0`, no KFD PIDs and 0% use afterwards, no dmesg events.

Raw rows in `raw/rows_{prod,proxy,fast}.json` and `raw/meas_out.txt`.

| shape | `cur_a` | `cur_b` | **floor** | `A68` | `P70` | `beat` |
|---|---|---|---|---|---|---|
| prod  | **517.21** | 515.17 | **0.40%** | 360.39 (**-30.32%**) | 461.05 (**-10.86%**) | 718.36 |
| proxy | **438.79** | 429.67 | **2.08%** | 326.70 (**-25.55%**) | 383.56 (**-12.59%**) | 580.46 |
| fast  | **55.62** | 48.50 | **12.80%** | 51.95 (-6.60%) | 50.66 (-8.92%) | 51.73 |

sclk 1051-1061 (prod), 1052-1067 (proxy), 1100-1100 (fast). Ratio `cur_a`/`beat` at prod
= **0.7200**.

Three readings of this table:

- **The `fast` shape is uninformative this session.** Two physically separate copies of
  identical code differ by **12.80%**. Nothing at `fast` is scored this round; both arms sit
  inside that band. This is exactly what two incumbent copies are carried for.
- **`prod` and `proxy` are clean** (0.40% / 2.08% floors). Both arms are far outside both.
- `cur_a` at prod reads 517.21 against the ledger's best-ever 509.15. **This is not a win
  and is not claimed as one** -- no code changed; it is session-to-session variation in the
  incumbent, and it is reported because it is what the instrument said.

### `P70` -- the subtractive probe, and the second half of `g67`'s death

`P70` deletes `k_dq`'s prefetch entirely (depth 1 -> 0). `pool.md` required this **before**
form 1, because a subtractive probe bounds the whole direction from above. It measures
**-10.86% prod / -12.59% proxy**.

That is a **negative ceiling for the "shorten `k_dq`'s prefetch" direction**: the best any
shortening could achieve is removing the prefetch, and removing it costs 11-13%. Combined
with form 1 dying at the offline gate (section 3), **`r22.i1.g67` is closed in both of its
forms, one for free and one for one arm's card time.**

It also independently corroborates the cover model: `k_dq`'s prefetch is not decoration,
it is buying ~11-13% of real load-to-use cover. The reason `g66` lost 19.33% was never that
prefetching is wrong on `k_dq` -- it is that spreading the clump destroys the cover the
prefetch exists to provide.

Correctness: `P70` passed unit tests **15/15** and `op/validation.py`'s own precision check
before any benchmark, per `h1`. It is a throwaway probe and ships nothing.

---

## 8. Outcome

**Nothing ships.** Both pool candidates are dead:

- `r22.i1.g67` -- form 1 killed at the offline gate (zero card time), form 2's ceiling
  measured negative (-10.86% / -12.59%). **Closed.**
- `r22.i2.g68` -- **-30.32% prod / -25.55% proxy**. **Dead.**

Per the round rule *"ship the best arm even when all of them lost"*, `A68` is recorded as
the round's arm. The round is rejected either way and the incumbent stands unchanged.

What the round is worth:

1. **Round 21's signed fact is falsified.** The one directional model twenty-one rounds
   produced -- *"the two kernels have opposite signs on the prefetch-spread lever"* -- was
   tested by its own prediction and broke. Spreading the clump loses on **both** kernels.
2. **The cover model replaces it.** It made a correct quantitative prediction *before* the
   card (`A68` loses large), and it retro-explains `g28` (-7.8%), `g62` (+1.65%),
   `g66` (-19.33%), `g68` (-30.32%) and now `P70` (-10.86%) with one variable: worst-case
   load-to-use distance to the next full `s_wait_loadcnt 0x0`.
3. **`g62`'s +1.65% is repriced onto the noise floor** (corpus null-control floor 1.57%,
   n=5) with a mechanism for why it cannot be larger: `k_dkdv`'s body opens with a full
   drain at idx 13 plus 128 `v_mov_b64`, the FlyDSL non-rotating-carried-register block
   copy, so depth 2 buys ~one iteration of cover and pays 128 copies for it.
4. **Three free kills and one free census.** `g67` form 1; the register-destination cluster
   load (does not exist in this backend); the power-cap hypothesis (untestable, sensor
   static); and the TDM/cluster API census that turns `g69` into a one-file reading task
   rather than a build.
5. **The throttling explanation is removed.** sclk is flat at 1100 under sustained load.

### Instruments this round (no `g` consumed)

- `r22.i0.a` -- ISA census of both kernels.
- `r22.i0.b` -- power and clock trace. Negative result, recorded as such.
- `r22.i0.c` -- FlyDSL cluster/TDM API census.

---

## 9. Champion re-measure -- same session, cache cleared

`rm -rf /root/.flydsl` plus a full `__pycache__` purge under `rounds/`, then the shipped copy
rebuilt and re-validated from cold, then every distinct per-shape champion re-run back to back
on the same idle device. `RC=0`; no KFD PIDs and 0% use afterwards; no dmesg events.

Shipped copy after the cache purge: UT **15/15 PASS** (52.30-52.98 dB), `op/validation.py`
correctness **pass**, determinism x200 **bitwise pass**, speed **FAIL** (`RC_ut 0`,
`RC_val 2`). This is the second full correctness pass on this code, per `h1`.

| shape | **r022** (this round) | r020 | r019 | r017 | `beat` | champion |
|---|---|---|---|---|---|---|
| prod  | **361.73** | **516.87** | 507.13 | 505.85 | 719.14 | r020 |
| proxy | **327.77** | 432.35 | **443.73** | 439.39 | 584.05 | r019 |
| fast  | **51.22** | 53.51 | **55.32** | 53.77 | 52.04 | r019 |

sclk 1051-1061 / 1051-1064 / 1100-1100. Raw in `raw/rows_{prod,proxy,fast}_champ.json` and
`raw/champ_out.txt`.

**Acceptance arithmetic, computed against champions re-measured beside it in this session:**

| shape | vs. champion | vs. target (`beat`) |
|---|---|---|
| prod  | 361.73 / 516.87 = **0.6998** | 0.5030 |
| proxy | 327.77 / 443.73 = **0.7387** | 0.5612 |
| fast  | 51.22 / 55.32 = **0.9259** | 0.9842 |

`score` = mean(ratio) = **0.6828**. Throughput does not improve on the best ever on any
shape, and all three are below 95% of their own best ever. **The round is rejected**, which is
what was predicted before the card and is the correct outcome for a falsification.

⚠ The champion ordering itself is worth banking: **`r017`, `r019` and `r020` are within 2.2%
of each other at prod** (505.85 / 507.13 / 516.87) and within 2.6% at proxy. Three rounds of
accepted work separate them. That is the same statement as `g62`'s +1.65% sitting on a 1.57%
null-control floor, seen from the other end: **this operator has not moved outside the noise
since round 17.** The next round should treat "beat the champion by more than the floor" as
the real bar, not "be positive".

### The change is left in place

`rounds/022/op/kernels.py` contains `A68` and is **not reverted**, md5
`8f67cd8b908d25d824711122c3a14709`, byte-identical to the validated arm. It lost 30.0% and
that is the result: `outcome: delivered`, number reported, acceptance rule declines to promote
it. `op/current/` was never touched at any point in this round.

---

## explored.consulted

**Job findings (step 1, all read first):**
- `job_context/findings/facts.md`
- `job_context/findings/dead_ends.md`
- `job_context/findings/pool.md`
- `job_context/findings/route.md`
- `rounds/017/1-profiling/` (read in full as instructed, despite round 20's acceptance)
- `rounds/020/1-opt/opt.md`, `rounds/020/1-opt/raw/probe_results.txt`
- `rounds/021/1-opt/opt.md`, `rounds/021/1-opt/raw/*`

**Corpus (mandatory this round -- round 21 was not accepted):**
- `optimization/routes/1-metrics-to-techniques.md`
- `knowledge/backends/flydsl/attention/` -- `recipes/`, `dead-ends.md`, `facts.md`
- `optimization/techniques/` -- prefetch/staging, scheduling-hints, split-k,
  LDS double-buffering, TDM / `TENSOR_LOAD_TO_LDS`, cluster multicast + `early_timeout`,
  register pinning, WMMA fragment shape
- `pitfalls/` -- WMMA RAW hazards and `v_nop` filling, `sched_barrier` mask semantics
  (0 = hard fence), gfx1250 split wait counters, power/clock chapter
- Other backends, same op: `knowledge/backends/*/attention/recipes/`

**Installed backend (census, this round):**
- `flydsl/expr/rocdl/cluster.py`
- `flydsl/expr/rocdl/tdm_ops.py`
- `flydsl/expr/rocdl/__init__.py`
