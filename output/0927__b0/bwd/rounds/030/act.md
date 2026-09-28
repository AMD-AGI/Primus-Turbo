# Round 30 -- act log. Candidate `r29.i1.g89`

Stage `k_dqg`'s dQ epilogue through LDS so 256 `buffer_store_b16` become 32
`buffer_store_b128`. Written as it happens.

---

## Step 0 -- feasibility. FEASIBLE.

**Working copy verified against `op/current` BEFORE anything else** (h67's standing half):

```
diff -rq --exclude=__pycache__ rounds/030/op job_context/op/current  -> no differences
kernels.py 37f37052eb739579555d6bdf62a829cc   impl.py 08cb8533d75e82198fabb9514fd18ba5
_env.py    7f33871b977f16ae03b95dd89a44483e   __init__.py 0024d09db2284ac48c29ab70e11a821d
```

### The atom question, which is the one the backend README says to check first

`knowledge/backends/flydsl/README.md:112` names the limit precisely: *"A matrix or copy
instruction is reachable from a kernel only if it exists as an `MmaOp`/`CopyOp` type in the
backend dialect. Adding one is TableGen plus C++ interface methods plus a rebuild."*

**Every atom this candidate needs is not merely available -- it is already called from this
file, and one of them from inside `k_dqg` itself.** That is a stronger answer than a
capability table, because an atom in use has a validated ThrVal layout, and
`README.md:120` warns that a wrong layout on a *new* atom "compiles, runs, and returns
garbage with no runtime diagnostic".

| needed | spelling in this tree | already used at |
| --- | --- | --- |
| transposing LDS read | `rocdl.ds_load_tr16_b128` | `kernels.py:467` (`k_dkdv` body), **`:1342` -- inside `k_dqg`'s own kv loop**, `:701-704` (`g86`'s epilogue) |
| 128-bit LDS write | `llvm_dialect.store(<8 x bf16>, create_llvm_ptr(.., address_space=3))` | `:688`, `:693` (`g86`'s epilogue) |
| 128-bit global store | `_stv(<8 vals>, _bv(buf, n, fx.BFloat16, 8), tile, fx.BFloat16)` | `:706-707` (`g86`), helper at `:101` |
| extra LDS | `fx.SharedAllocator().allocate(..)` | `:1220`, **already called in `_dqg_impl`** |

`fx.SharedAllocator`, LDS layout and swizzle are listed **"Yours"** in the capability table
(`README.md:29`), and a barrier is "Yours, but only explicit `gpu.barrier()`" (`:36`) -- not
needed here, see deviation (c).

### Structural check: the two epilogues have the same shape, which is why the transplant is real

`g86` works because the WMMA output gives a lane **one d column and 8 rows**, so a
**column-major** LDS image makes the lane's 8 values contiguous (one `ds_write_b128`), and
`ds_load_tr16_b128` reading a column-major image returns **row-major** -- lane `l` gets row
`l%16` and 8 *consecutive d columns* -- which is exactly a `b128` global store.

`k_dqg`'s epilogue at `kernels.py:1434-1443` has the identical index structure, with the q
axis where `k_dkdv` has kv:

```python
q_i = q0 + qh_*16 + half*8 + si          # si (the lane's 8 values) walks the Q ROW
_st1(ovs[si].to(fx.BFloat16), g_dq,
     base_o + q_i*Hq*D + dtile*16 + row,  # d column is FIXED at dtile*16 + row
     fx.BFloat16)
```

Lane varies the row across its 8 elements and holds one fixed d column. Same transpose, same
mechanism. **Counts check out exactly**: `NQW = DQ_BQW//16 = 4`, `NDO = D//16 = 8`, so
`4 x 8 x 8 = 256` `buffer_store_b16` per lane today; after the transpose each lane holds
`64 q x 128 d / 32 lanes = 256` elements = **32 stores of 8 bf16 = 32 `buffer_store_b128`**.
That is the plan's gate, derived from the source rather than assumed.

### Deviations from what the candidate assumed -- three, all recorded, none an abandonment

**(a) `g89` MUST GROW THE LDS ALLOCATION. `g86` did not, and its comment leans on that.**
`g86`'s own text (`kernels.py:669-671`) says *"It costs no LDS: the Q/dO ring (17408 B at
offset 0) is dead here"* -- it reused a hole inside an already-allocated 65,536 B segment.
`_dqg_impl` has no hole: it allocates exactly `NW * KV_STEP * X_ROW_B` = `1*32*272` =
**8,704 B**, which is `kernel.yaml`'s figure, and that is the live K staging ring.

The dQ image is 64 q rows x 128 d in bf16 = **16,384 B before padding**, which does not fit
in 8,704 B. `lds_k` *is* dead by the epilogue (`kvloop_mask` has returned at `:1432`), so it
can be overlapped rather than added to -- but the allocation still has to grow to the image
size. This makes the plan's LDS-budget argument **load-bearing rather than belt-and-braces**:
`k_dqg` is VGPR-capped at 992/1024 ISA VGPR -> 1 wave/SIMD -> <= 4 wave32 workgroups/CU ->
~40 KB per workgroup against 320 KB physical per WGP (`arch/gfx1250/gfx1250.md:51-53,83`).
~18 KB stays inside that, and `k_dkdv` ships **70,656 B** at the same rung for zero occupancy
cost (campaign correction 2). ⚠ The allocated LDS is a **reportable gate number**, not an
assumption -- it goes in the ISA gate next to VGPR and scratch.

**(b) Padding must be re-derived, not copied. `EPI_CB = 48` does not transfer.**
`g86` chose `EPI_CB = 48` = 32 B of kv row + the `g16` pad, because *"12 dwords walks
c*12 mod 64, 16 distinct bank groups"* (`:672-673`). Here the image's minor axis is **64 q
entries = 128 B = 32 dwords**, and `c*32 mod 64` hits only **2** bank groups out of 64 -- a
32-way conflict. The stride needs its own derivation against gfx1250's 64 banks x 4 B
(`gfx1250.md:83`). This is precisely the corpus rule the brief states: **take the mechanism,
not the constants.** Sized in the implementation step.

**(c) There is no `PARTIAL` branch to guard in `k_dqg`, so the plan's scope condition is
satisfied by construction.** The plan wrote *"scope is the bf16 direct path only
(`PARTIAL == False`)"*, carried across from `g86`, whose `_dkdv_impl(PARTIAL, ...)` really
does branch. **`_dqg_impl` takes no `PARTIAL` parameter at all** (`kernels.py:1177-1178`):
the fp32 `nsp>1` workspace path lives in a *different function*, `_dq_impl` (`:769`), which
serves `k_dq`/`k_dq_sp` and is what `fast` executes. So the fp32 path cannot be touched by
this edit even by accident, and no guard is needed. **This strengthens the `fast` null
control**: `fast` runs a function this change does not open.

### What this does NOT settle

Feasibility is an atom-set and layout question. It is **not** a prediction -- campaign
correction 3, and this job's own 4-for-4 ledger, say static facts gate a build and never
price it. Pricing goes on card.

---

## Step 1 -- implement

### The device, checked before anything was launched

`rocm-smi --showpids` inside `fa-g3` lists **PID 4105879** -- a `lab-kdq/tools/kbench.py
prod blk 45` launched from container **`fa-g0`**, alive, 8.5 min in. The KFD process list
is **global**, not per-visible-device, so that line alone does not say which card it holds.
Resolved from sysfs instead:

| container | render node | PCI | card | `gpu_busy_percent` |
| --- | --- | --- | --- | --- |
| fa-g0 | renderD128 | 0001:04:00.0 | card0 | 0 |
| -- | renderD136 | 0002:04:00.0 | card8 | **25** |
| fa-g2 | renderD144 | 0003:04:00.0 | card16 | 0 |
| **fa-g3 (this round)** | **renderD152** | **0004:04:00.0** | **card24** | **0**, sclk 2363 |

So the neighbour's work is on **card8, a different physical GPU**, and this round's device
is idle. ⚠ Recorded rather than waved past: a neighbour GPU under load shares the chassis
power and thermal envelope, and this job's prod numbers are clock-sensitive (h57, P77).
The card8 load is re-checked at the deciding measurement and goes in `provenance.yaml`.

### Implementation

`kernels.py` only; `impl.py`, `_env.py`, `__init__.py` untouched -- `g89` changes how
`k_dqg` writes dQ, not which kernel is dispatched.

1. `DQ_EPI_CB = 48` next to the other `DQ_*` constants.
2. `g_dq = _bv(DQ, 1<<30, fx.BFloat16)` -> `g_dq8 = _bv(DQ, 1<<30, fx.BFloat16, 8)`.
   The scalar descriptor has no remaining user, so it is replaced rather than added to.
3. Segment 0 becomes `max(KV_STEP*X_ROW_B, NQW*128*DQ_EPI_CB)` = `max(8704, 24576)` =
   **24576 B**; `_lds_w` names the per-wave base, `lds_k` keeps its old meaning.
4. The epilogue: per q block, 8 `ds_write_b128` of the column-major M[d][q] image, then
   8 `ds_load_tr16_b128` + 8 `_stv` b128. `256 -> 96`, VMEM `256 -> 32`.

### ⚠ CORRECTION to Step 0's deviation (b). I had the image geometry wrong.

Step 0 said `EPI_CB = 48` could not be copied because "the image's minor axis is 64 q =
128 B = 32 dwords, and `c*32 mod 64` hits 2 bank groups". **That is wrong, and re-reading
`g86` rather than my note is what caught it.** `g86` does not build one image per kernel;
it builds **one image per 16-row block** (`kh`), four of them. The same decomposition here
gives **NQW = 4 images of 128 d x 16 q**, so the minor axis is `16 * 2 = 32 B` -- exactly
`g86`'s -- and `DQ_EPI_CB = 48` transfers **unchanged**, with its 12-dword / 16-bank-group
argument intact. Deviation (b) is withdrawn. Total LDS is then `4 * 128 * 48 = 24576 B`,
the same figure `g86` uses, not the ~18 KB I estimated from the wrong geometry.

Deviation (a) still stands and is the real one: `k_dkdv` had a dead 17408 B hole inside an
already-allocated 65536 B segment, so `g86` cost nothing; `k_dqg` allocates 8704 B and
nothing else, so **segment 0 has to grow 8704 -> 24576 B**. That is a reported gate number.

### Build: first attempt compiled and launched. No build failures.

`raw/isa.sh`, detached with a sentinel, **a fresh compile cache per arm**
(`/tmp/flycache_r30_<arm>`, `rm -rf`'d first) so neither arm can be served a kernel built
before this round. `RC base 0`, `RC g89 0`, both logs end `COMPILED_AND_LAUNCHED_OK`.

### The free static gate (h3 / h18), from `21_final_isa.s`

| `k_dqg_0` | base | g89 | |
| --- | --- | --- | --- |
| `buffer_store_b16` | **256** | **0** | the whole point |
| `buffer_store_b128` | 0 | **32** | exactly the predicted count |
| `ds_store_b128` | 48 | 80 | +32 staging writes |
| `ds_load_tr16_b128` | 48 | 80 | +32 transposing reads |
| text instructions | 4187 | **3141** | -1046, -25.0% |
| `.vgpr_count` (ISA) | 991 | **1000** | +9 |
| `.vgpr_spill_count` | 0 | **0** | |
| `.sgpr_spill_count` | 0 | **0** | |
| `.private_segment_fixed_size` | 0 | **0** | h3 gate |
| `scratch_` ops | 0 | **0** | h3 gate |
| `.group_segment_fixed_size` | 8704 | **24576** | deviation (a), as predicted |

**h3 passes: zero scratch, zero spill.** That gate is a kill, not a cost -- a spilling
gfx1250 build hangs the card and needs a human power cycle -- so it was run before anything
was timed.

⚠ **The VGPR margin is now 24.** 991 -> 1000 against 1024 addressable. It did not spill and
1 wave/SIMD is unchanged (anything above 512 forces it), so nothing about occupancy moves.
But `k_dqg` is 24 registers from the wall, and that is a fact the next round needs before it
proposes anything else that lives in this kernel.

**The change is confined, and the ISA proves it rather than my asserting it:**
`k_dkdv_0` is identical across the two arms on every line of the table -- 2257 text, 729
VGPR, 70656 B LDS, 32 `buffer_store_b128`. `g89` touches `k_dqg` and only `k_dqg`.

Per campaign correction 3 none of this is a prediction. 256 -> 32 VMEM stores is a fact
about the build; whether it buys time is measured on card.

### Correctness: `op/ut/` PASS, `validation.py` correctness+determinism PASS

`ut/test_correctness.py --impl rounds/030/op`, run through the container: **15/15 PASS**,
every shape in both causal and full mode, full `isfinite` coverage on a NaN-poisoned
allocator (`dq 134217728/134217728` at prod). Worst dq is **52.52 dB** at proxy against the
50 dB gate; dq/dk/dv all sit in a 52.4-53.0 dB band, i.e. exactly where the incumbent sits
-- staging through LDS moves bits around, it does not change the arithmetic, and the
numbers say so.

`validation.py rounds/030/op` -- **the job's own gate, not one I wrote**:

```
  correctness fast   dq 52.61  dk 52.65  dv 52.83 dB
  correctness proxy  dq 52.52  dk 52.57  dv 52.67 dB
  correctness prod   dq 52.56  dk 52.60  dv 52.71 dB
  determinism fast: dk/dv bitwise identical across 200 runs; dq bitwise (floor 70 dB)
  correctness  pass       determinism  pass       speed  FAIL       RESULT: FAILED   (rc 2)
```

**`correctness pass`, `determinism pass`.** dq is *bitwise* across 200 runs, not merely
above the 70 dB floor -- the epilogue stays atomic-free and split-free.

⚠ `RESULT: FAILED` / rc 2 is the **speed** gate against `beat` (aiter), and it is the
standing state of this job rather than anything `g89` did: the gate needs proxy and prod
each `>= 1.00x beat` and prod reads **0.800x**, against the 0.806 this round's own
profiling recorded for the incumbent (`rounds/030/1-profiling/profiling.yaml`
`performance.shapes`: 658.27/817.07). Closing that gap is what the campaign is for. I
report it as measured; it is not evidence about this candidate either way, because it is
a candidate-vs-aiter number and the round is decided on candidate-vs-champion.

---

## Step 2 -- measure. THE PREDICTION IS REFUTED. `g89` is a small, reproducible LOSS.

`raw/bench.sh`, detached with a sentinel, all 10 runs `rc 0`. Two phases, because the
question "how much faster" and the question "did the clock move" cannot be answered by the
same run: inside an interleaved process both arms share one clock excursion **by
construction** (h57), which is exactly what makes the gain clean and the attribution
unanswerable.

### Phase B -- the deciding run. Both arms in ONE process per shape, blocked 9 + lead 4,
palindromic, two passes.

| shape | base ms | g89 ms | g89 vs base |
| --- | --- | --- | --- |
| prod pass 1 | 8.357282 | 8.372344 | **-0.180%** |
| prod pass 2 | 8.363511 | 8.381618 | **-0.216%** |
| **prod median** | **8.360396** | **8.376981** | **-0.198%  =  -16.6 us** |
| proxy pass 1 | 0.569214 | 0.564325 | +0.866% |
| proxy pass 2 | 0.567470 | 0.570174 | -0.474% |
| **proxy median** | 0.568342 | 0.567250 | **+0.193%** |
| fast pass 1 | 0.052639 | 0.052859 | -0.416% |
| fast pass 2 | 0.052719 | 0.053400 | -1.275% |
| **fast median** | 0.052679 | 0.053129 | **-0.848%** |

**Predicted: prod +30 to +120 us (+0.36% to +1.44%). Measured: -16.6 us.** Wrong sign,
and outside the band on the far side of zero. `refuted_if` fires on its first clause.

### The three control channels, each doing its job

**The `fast` null control fired, and it is a CALIBRATION rather than a failure.** I derived
the dispatch from `impl.py:130-150` against the shape table rather than asserting it:

| shape | `_wgs_q0` | `_nsp_q0` | `use_g` | is `k_dqg` (the only kernel g89 touches) dispatched? |
| --- | --- | --- | --- | --- |
| fast | 128 | 8 | **False** | **no** -- takes `k_dq_sp`, untouched |
| proxy | 2048 | 1 | **True** | **yes** |
| prod | 16384 | 1 | **True** | **yes** |

So at `fast` the two arms are **bitwise-identical dispatched code**, and they read
**-0.848%** apart, with the two passes at -0.416% and -1.275%. That is this session's fast
noise floor measured on an A/A, and it is 4x the size of the prod effect. Any future round
reading a sub-1% fast number as a result should look at this line first.

⚠ **And it settles unresolved disagreement #1 from `plan.yaml`, in the planner's favour, for
free -- exactly where the plan said it would.** The reviewer withdrew the proxy channel on
the grounds that "proxy runs `k_dkdv_sp` + `k_redsp` (nsp=2), which prod does not", i.e.
that proxy does not exercise this code. The derivation above says `use_g` is **True** at
proxy: `nsp` and `_nsp_q0` come from two separate loops over two separate workgroup counts,
and `use_g` reads only `_nsp_q0`. **proxy does exercise `g89`.** It reads +0.193% with the
two passes straddling zero (+0.866 / -0.474), so it corroborates nothing either way -- but
it was barred from refuting on its own, and it did not have to.

**The proxy/prod ratio channel is not evaluable.** Predicted 1.2-2.5 (central 1.84). With
prod negative the ratio has no meaning in the sense the prediction used. Recorded as
not-evaluable rather than as a number, and it could not have refuted anyway.

### The attribution check (P77), which is why Phase A exists -- and it comes back CLEAN

`sclk` per arm, from four solo prod processes, one arm each, palindromic:

| run | arm | prod ms | sclk start -> end |
| --- | --- | --- | --- |
| A1 | base | 8.340233 | 1789 -> 1796 |
| A2 | g89 | 8.377851 | 1790 -> 1792 |
| A3 | g89 | 8.365593 | 1783 -> 1787 |
| A4 | base | 8.373166 | 1791 -> 1791 |

base median sclk ~1790, g89 ~1787: **-0.20%**, against the 1% threshold that would have
voided attribution. P77's failure mode -- a traffic-only change moving prod sclk +3.2%
while raw time moved ~0 at matched clock -- **did not happen here**, in either direction.
The loss is a real loss and not a clock artifact. Phase A's solo medians (base 8.3567,
g89 8.3717, **-0.18%**) also reproduce Phase B's -0.198% across a completely separate set
of processes, which is the strongest thing I can say about a 16 us effect: **four
independent process-level comparisons, same sign, spread 0.036 percentage points.**

### r30.i0.C1 -- card power against cap. REFUTED, and that closes profiling's q1.

Sampler at 200 ms on card24's own hwmon (`raw/sample.py`, `raw/bench/power_*.txt`).

⚠ **Mechanism found on the way, worth more than the reading:** `power1_input` returns
**EIO** for most samples taken while the card is under sustained load at prod and proxy --
the SMU query fails exactly when it is wanted -- and `rocm-smi --showpower` returns
`energy_count_secondary_die_check, Unexpected data received` under the same conditions.
It reads fine at idle and at `fast` (which leaves gaps between calls). The sampler retries
8x4 ms and still loses the median at prod. **Any future power-based instrument on this card
must use the max over a sparse successful sample, not the median.**

Which is what I did, and it is the conservative direction for this hypothesis:

| trace | successful power samples | **max W** | **max / cap** | sclk med | T med |
| --- | --- | --- | --- | --- | --- |
| B1 prod | sparse | 1943.0 | **0.777** | 1790 | 75.8 C |
| B2 prod | sparse | 2053.7 | **0.821** | 1788 | 76.3 C |
| B1 fast | 30/30 | 1973.0 | **0.789** | 2255 | 63.7 C |
| B2 fast | 31/31 | 1988.3 | **0.795** | 2255 | 64.2 C |
| idle | all | 1143.1 | 0.457 | 2363 | 51.6 C |

C1 predicted `prod power >= 0.95 x cap while sclk <= 1800`, and `refuted_if: prod power <
0.90 x cap`. **Prod never exceeded 0.821 x cap across either trace.** Even taking the
maximum of the samples that survived -- which can only overstate -- prod is nowhere near
the 2500 W PPT. **q1 closes as "not power":** prod's 1790 MHz against a 2363 MHz idle is
not this card's PPT limit.

Two things the reading does NOT settle, stated rather than smuggled: (i) the EIO samples
could in principle be the high ones, though a 0.95-cap regime hiding entirely inside the
failures while every success sits at 0.78-0.82 is not a small ask; (ii) **`fast` draws the
same ~0.79 x cap at 2255 MHz that prod draws at 1790 MHz**, so power does not explain the
clock difference in either direction -- which is the same conclusion by a second route.
The neighbour card8 sat at a steady 25% busy through every single trace, so it is a
constant across arms, not a confound on the comparison.

### r30.i0.C2 -- the clock lock is REFUSED. knowledge_gaps item 11 closes permanently.

The candidate said a refused lock is itself the result. It is refused, and the mechanism is
not "permission denied" (which is a symptom) but this:

```
$ docker exec fa-g3 rocm-smi --setperflevel manual
GPU[0]          : set_perf_level, Permission denied
$ docker exec fa-g3 bash -c 'echo manual > /sys/class/drm/card24/device/power_dpm_force_performance_level'
bash: .../power_dpm_force_performance_level: Read-only file system      # as uid 0, CapEff a80c25fb
```

**`/sys` is mounted read-only in the container.** No capability and no uid changes that, so
no clock lock is reachable **through the runner** -- and the runner is the only legitimate
measurement path, so a lock taken on the host would not produce a number this job may use.
Item 11 has been named the highest-priority instrument for five rounds and was never
attempted; it is now closed with a mechanism instead of being renamed a sixth time.

**Card state was never modified.** `power_dpm_force_performance_level` read `auto` before,
during and after, verified from inside the container and from the host; the attempt ran
under `trap ... EXIT` calling `--resetperflevel` regardless.

⚠ **A deviation I am naming rather than burying:** even had the write been permitted, the
candidate's own constraint -- *"pin only to DPM levels the table already advertises"* and
*"do not pin above 1753 MHz"* -- is **unsatisfiable on this card**. `pp_dpm_sclk` advertises
exactly three levels: `500`, `2364`, `2400`. One is below 1753 and two are above, so there
are not two legal setpoints for a two-point fit. C2 as written could not have run even with
root on the host. That is a defect in the candidate I wrote, found by trying it.

### Device hygiene

No `amdgpu` message on **0004:04:00.0 (card24, this round's GPU)** during the round; its
last entry is at uptime 55890 s, ~11.6 h before any of this. The 161 `MES(0,0) ring buffer
is full` messages in the buffer are all on **0002:04:00.0 (card8)**, the neighbour, and stop
at 97639 s, before this round's first run. After the round: `docker ps` clean, no
`benchmark.py` / `sample.py` / `validation.py` / `dump_isa` process alive in `fa-g3`,
card24 `gpu_busy_percent` 0.

### Why it lost -- a HYPOTHESIS, labelled as one, for reflect

The static gate is not in dispute: VMEM stores 256 -> 32 and 1046 fewer instructions, and
the time went the other way. Campaign correction 3 says exactly this can happen, and this
is now the third time in this job.

What is different from `g86`, which won +0.35% with the identical transform in `k_dkdv`:

| | k_dkdv (g86, +0.35%) | k_dqg (g89, -0.20%) |
| --- | --- | --- |
| workgroups at prod | 8,192 | **16,384** |
| cycles per wave | 1,168,415 | **339,716** (3.4x shorter) |
| LDS before -> after | 65,536 -> 65,536 (a dead hole) | 8,704 -> **24,576** (real growth) |
| ISA VGPR before -> after | 724 -> 729 | 991 -> **1000** |

The transform replaces 256 independent stores -- which retire asynchronously and can drain
behind whatever follows -- with a **serialized chain**: 32 `ds_write_b128`, a `dscnt` wait,
32 transposing reads, then 32 stores. In `k_dkdv` that chain sits at the end of a
1.17 M-cycle wave with 8,192 workgroups covering each other. In `k_dqg` it sits at the end
of a wave 3.4x shorter, in a dispatch 2x wider, i.e. where per-workgroup fixed cost matters
~6.8x more, and at 1 wave/SIMD there is no second wave to hide it behind.

**I am not claiming this is measured.** It is consistent with every number above and with
h16's 0.15x issue-to-time conversion, and it is contradicted by nothing here. What would
settle it: the same epilogue with the LDS round-trip **split across the kv loop** so the
staging writes overlap real work instead of forming a tail -- if the loss is the serialized
tail, that recovers it; if the loss is the LDS growth or the 9 extra VGPRs, it does not.
That is a different hypothesis and belongs to a later round as its own candidate, not to
this one.

### What stays in the working copy

`g89` stays, exactly as built and measured. It lost; the acceptance rule will not promote
it and `op/current` will not move. Reverting would destroy the only copy of the one thing
this round established -- that the `g86` mechanism does **not** transfer to `k_dqg` -- and
would make the round hash-identical to one that built nothing.

---

## Step 2 addendum — PHASE C: the per-shape champion guard

The step-2 brief requires **each distinct per-shape champion in `state.yaml`** to be rebuilt
and re-measured *now*, not read from a record. `state.yaml`'s `champions` are:

| shape | champion round |
|---|---|
| `prod_b4_s8192_hq32_hkv8_d128` | 29 |
| `proxy_b1_s4096_hq32_hkv8_d128` | 29 |
| `fast_b1_s1024_hq8_hkv2_d128` | **28** |

Phase B already re-measured round 29 as the `base` arm — `job_context/op/current` is
byte-identical to `rounds/029/op` on all four files (md5 `08cb8533 / 37f37052 / 7f33871b /
0024d09d`), so prod and proxy are guarded against their champion rebuilt this session.

⚠ **round 28 is a genuinely different arm and had NOT been measured.** `impl.py` `08cb8533`
vs `f60da2b4` and `kernels.py` `37f37052` vs `1c487556` — two of four files differ. It is the
`fast` champion, so phase C rebuilds it from `rounds/028/op` and runs all three arms
interleaved in one process at `fast`, so base / g89 / c28 share one clock excursion (h57)
rather than being compared across processes.

It builds and runs: `# arm c28 -> rounds/028/op md5=8cec300f`, exit 0.

### ⚠ The phase-C fast reading corrects the phase-B one

| pass | base | g89 | c28 | g89 vs base |
|---|---|---|---|---|
| C1 | 100.297 | 100.222 | 100.749 | **−0.07%** |

Phase B read `fast` at −0.85% for g89 against base. Phase C, on the **same bitwise-identical
dispatched code** (`use_g = False` at fast, so `k_dqg` — the only kernel this candidate
touches — is never dispatched), reads −0.07%. Both are A/A nulls, so both are pure noise, and
the pair puts a number on how wide that noise is: **the fast channel scatters over at least a
0.8 percentage-point range between processes.** That is 4× the prod effect this round is
reporting, which is exactly why prod was measured interleaved and repeated rather than read
once.

It also means the `fast` `vs_champion` figure carried into `act.yaml` from phase B is a noise
draw and not a property of this code, and is reported as such rather than as a regression this
round caused.

### Phase C, both passes

| pass | base (r29) | g89 (r30) | c28 (r28, the fast champion) |
|---|---|---|---|
| C1 | 100.297 | 100.222 | 100.749 |
| C2 | 100.222 | 100.522 | **97.990** |
| mean | 100.259 | 100.372 | 99.369 |

**g89 vs its actual champion at fast: 1.0101.** That replaces the 0.9915 carried in from
phase B, which was computed against round 29 — not the fast champion at all.

⚠ **c28 moved 2.8% between two passes inside one process** (100.749 → 97.990), with
`sclk_start/end` 2244/2252 and 2255/2262, i.e. no clock story. Taken with the phase-B/phase-C
disagreement on bitwise-identical code, the honest statement about this shape is that **the
fast channel's spread is between 0.8 and 2.8 percentage points**, 4× to 14× the prod effect
this round reports. No conclusion about this candidate may be drawn from `fast`, in either
direction, and the +1.01% above is reported because the guard requires a number — not because
it is evidence.

⚠ **`state.yaml`'s fast champion looks stale.** r28 is recorded as the fast champion at
99.37, but r29 re-measured *now* reads 100.26 on the same shape in the same process. That is
inside the scatter just described, so it is not a claim that the record is wrong — it is a
flag that the fast champion was set from a figure the shape cannot reproduce to better than a
couple of percent.

### Device hygiene, phase C

Idle before: `GPU use 0`, `card24_busy 0`. Idle after: `GPU use 1`, `card24_busy 1` — the
sampler's own read. The two KFD PIDs (46160, 38986) are present identically before and after
and are the neighbour on card8; the KFD list inside a container is global, not per-visible
device (see `provenance.yaml`). No amdgpu message on 0004:04:00.0 during phase C. `docker ps`
clean afterwards, no orphan `benchmark.py` in fa-g3.

---

## Step 3 — did the predicted thing happen?

`measurements/k_dqg_epilogue_pmc/`. The prediction's own metric (`prod_vs_champion_pct`) is
wall-clock and was settled in step 2: predicted +0.36% to +1.44%, measured **−0.198%**. That
is `held: false` before any counter is read. What step 3 adds is the question step 2 cannot
answer — **is the regression even in the kernel I edited, and did the mechanism reach the
chip at all?**

### Instrument design

`rocprofv3 --pmc`, regex `^k_(dkdv|dqg)_0$`, prod, warmup 2 / iters 5, arms palindromic
(base/g89/g89/base), two counter passes — the two sets this chip accepts (`SQ_WAVES
SQ_CYCLES SQ_BUSY_CYCLES SQ_ITEMS GRBM_GUI_ACTIVE`, and `SQC_ICACHE_*`).

The design point is that **`k_dkdv_0` is byte-identical between the two arms** and is captured
in the *same pass*, so it is an internal control rather than a separate run. Profiled
durations are not comparable to step 2's wall clock (dispatches serialise, the clock sits near
1.95 GHz instead of 1.79); they are used only as arm-vs-arm ratios inside one profiled block.

### The transform reaches the chip, exactly and reproducibly

| | base | g89 |
|---|---|---|
| `k_dqg_0` LDS_Block_Size (from the dispatch packet) | 8,704 | **24,576** |
| `k_dqg_0` VGPR_Count column | 496 | 504 |
| `k_dqg_0` `SQC_ICACHE_REQ` | 152,237,824 | **145,315,328** (−4.55%) |
| `k_dkdv_0` `SQC_ICACHE_REQ` | 65,599,616 | 65,599,616 (**bit-identical**) |

`SQC_ICACHE_REQ` repeated to the **exact integer** across both runs of each arm, and the
control repeated to the exact integer across all four. So this is not a noisy channel: the
instruction stream k_dqg actually fetches fell 4.55%, and k_dkdv's did not move by one
request. The ISA's 25% text reduction arriving as 4.55% of fetched stream is itself
consistent — the epilogue runs once per wave while the kv loop body is fetched ~129 times.

⚠ The VGPR column also confirms **campaign correction 1 on this very build**: 496 and 504,
exactly `roundup(991/2, 8)` and `roundup(1000/2, 8)` against the ISA's 991 and 1000.
⚠ `SQC_ICACHE_MISSES` read 0 in all four captures. No miss-rate or fetch-efficiency claim is
made from this run; only REQ is used.

### ⚠ CORRECTION: at n=2 I read a localisation that n=4 does not support

A section stood here, written from the first four runs (n=2 per arm), saying the regression
localises to `k_dqg`: every g89 duration above every base one, ratio 1.0059, control 0.9972.
I then ran four more pass-A repeats specifically to tighten the cycle bound. **The separation
did not survive**, so that section is replaced by this one and its claim is withdrawn below.

| n=4 per arm, prod | base median | g89 median | ratio | base spread | g89 spread | arms separated? |
|---|---|---|---|---|---|---|
| `k_dqg_0` duration (ns) | 2,772,332 | 2,779,684 | 1.0027 | 0.69% | 0.68% | **no** |
| `k_dqg_0` `SQ_BUSY_CYCLES`/256 | 5,439,285 | 5,439,908 | **1.0001** | 0.44% | 0.16% | no |
| `k_dkdv_0` duration *(control)* | 4,915,663 | 4,916,946 | 1.0003 | 0.31% | 0.91% | no |
| `k_dkdv_0` `SQ_BUSY_CYCLES`/256 *(control)* | 9,394,859 | 9,392,680 | 0.9998 | 0.70% | 0.57% | no |

The n=2 ordering was a small-sample artifact: the intra-arm spread under the profiler is
~0.7%, which is larger than the effect step 2 measures (0.198%). **So the profiled duration
channel cannot resolve this effect at all, in either direction, and no localisation claim is
made from it.** `k_dqg` at 1.0027 against a control at 1.0003 is the right sign for the
regression sitting in the edited kernel, and that is all it is — a sign, inside noise.

The arithmetic I built on it — "+0.59% of k_dqg predicts +0.21% of prod against step 2's
+0.198%" — was the product of that artifact and is **withdrawn**. Two instruments agreeing to
0.01 points on an effect neither can resolve is a coincidence I would have reported as a
result, which is exactly campaign correction 3's failure mode in a new costume. Recorded
rather than deleted, because the sequence is the useful part.

### What the counters DO establish: the cycles did not move

The cycle bound is the one claim that gets *stronger* with n=4, because it is a null and a
null is what more samples sharpen:

**`SQ_BUSY_CYCLES` ratio 1.0001**, arms overlapping, with an intra-arm spread of 0.44% (base)
and 0.16% (g89). The prediction needed ~0.72% of prod concentrated in `k_dqg` — roughly **2%
of that kernel's cycles**. The measurement bounds any saving to well under 0.5%, and the
point estimate is zero. **The epilogue got no cheaper in cycles.**

That is not a null instrument reading a null. The same runs carry the icache evidence two
sections up: k_dqg's fetched instruction stream fell **4.55%** to the exact integer, the
untouched control did not move by one request, and the dispatch packet carries the new
24,576 B LDS — while `SQ_BUSY_CYCLES` is flat. **The instructions went away and the time did
not.**

### The fact this round actually produces

**`k_dqg`'s wave cycles are insensitive to its epilogue's store-instruction count.** Deleting
224 of 256 `buffer_store_b16`, replacing them with 32 `buffer_store_b128`, cutting 25% of the
text and 4.55% of the fetched instruction stream moved `SQ_BUSY_CYCLES` by **+0.01%**, and
prod's wall clock the *wrong* way by 16.6 µs. The identical
edit in `k_dkdv` won +0.35% three sessions running. That is a difference between the two
kernels, measured, and it belongs in `facts.md` — not in `dead_ends.md` as "g89 was slow".

---

## Step 4 — report

No new measurements. `act.yaml` and this file are written from what is on disk, and
`validation.py` is re-run once more against the working copy.

### The final `validation.py` run

Through the runner, **with the directory passed explicitly** — `validation.py
.../rounds/030/op`. Left to its default it grades `op/current/`, which is still the incumbent
because this round has not been accepted, and nothing in its output would say so. The run
prints the md5 of every file in the tree it is grading, so the record shows which code
produced the exit code.

### On the three things `act.yaml` must keep straight

* **`outcome: delivered`.** There is code and there is a measurement. It is not `incorrect` —
  the kernel computes the right answer, 52.52 dB worst against a 50 dB gate with dq bitwise
  deterministic across 200 runs. It is not `abandoned` — abandonment means nothing was built,
  and something was built, it is in the working copy, and it stays there.
* **The speed result and the prediction check are two fields and two facts.** Speed: prod
  −0.198%, a small reproducible loss. Prediction: refuted, and refuted with the wrong sign.
  Merging them into one verdict would lose the part that is worth carrying forward, which is
  neither of those numbers but the counter reading behind them.
* **`correctness` is correctness only.** Whether this round beats the incumbent is arithmetic
  Python does from step 2's table; whether the job stops is `validation.py`'s exit code. This
  round concludes neither, and the numbers for both are recorded as measured.

### What reflect should take from round 30

The round fails on speed and the prediction is refuted, so by the four-cell table it is the
plain bottom-right — but filing it as a dead end would lose the finding, because the finding
is not "g89 was slow". It is:

> **`k_dqg`'s wave cycles are insensitive to its epilogue's store-instruction count.** 224 of
> 256 `buffer_store_b16` deleted, 25% of the text gone, 4.55% less instruction stream actually
> fetched by the chip — and `SQ_BUSY_CYCLES` moved +0.01% (n=4/arm, spreads 0.44%/0.16%). The
> identical edit in `k_dkdv` won +0.35% across three sessions. **The two kernels differ, and
> the premise g86 was generalised from does not hold in `k_dqg`.**

That retires the whole "port g86's epilogue to the other kernel" family on a measurement
rather than on one arm's bad luck, which is worth more than the slot it cost. Three further
closures came free this round and are argued above: `r30.i0.C1` refutes the power-wall
explanation for prod's 1790 MHz (peak 0.821 × cap); `r30.i0.C2` closes knowledge-gaps item 11
permanently with a mechanism (`/sys` is read-only in the container, so no clock lock is
reachable through the runner, and the candidate's own DPM constraints were unsatisfiable
anyway); and unresolved disagreement #1 resolves in the planner's favour — `use_g` is `True`
at proxy, so proxy does exercise `k_dqg`.

Two corrections in this log are mine and are left in sequence rather than tidied away: step
0's deviation (b) had g86's image geometry wrong, and step 3's first localisation claim was an
n=2 artifact that n=4 destroyed. The second is the more useful one to carry: it is campaign
correction 3's failure mode — a static-looking agreement believed because two instruments
concurred — reproduced by me, on this round, at a point where I had already written the
warning into this same file.
