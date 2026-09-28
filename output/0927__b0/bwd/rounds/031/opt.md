# Round 31 (fast) -- opt

Written as the round happened. Build failures are recorded where they occurred, not
summarised afterwards.

## 1. What I read

`findings/facts.md`, `dead_ends.md`, `pool.md`, `route.md`, `job_context/hint.md`
(h71, h73 in full), and round 30's `1-profiling/profiling.yaml` + `3-act/act.yaml`.

Corpus (`explored.consulted`, every file opened):

- `knowledge/backends/flydsl/attention/dead-ends.md`
- `knowledge/backends/flydsl/attention/README.md`
- `knowledge/backends/flydsl/README.md`
- `knowledge/arch/gfx1250/isa.md`
- `knowledge/arch/gfx1250/perf.md`
- `knowledge/ops/attention/README.md`
- `knowledge/ops/attention/backward.md`
- other backend, same op: `knowledge/backends/aiter/attention/` (the `beat` backend's
  own notes) and `knowledge/backends/triton/attention/`.

The standing verdict is unchanged since round 29: prod 8.35 ms, ~657 TF/s, ratio ~0.80
of `beat`, **latency-shaped**, 1 wave/SIMD pinned by VGPR (736 / 992 ISA), no spill
anywhere, bandwidth ~40x below any plausible roof, clock 1753-1790 MHz under load and
**not** power-limited (round 30 refuted that at <=0.821x of the 2500 W cap) and not
lockable (`/sys` read-only, EROFS).  `k_dkdv` is 61.1% of prod, `k_dqg` 34.9%.

## 2. The survey -- a free opcode census nobody had taken

GPU was clean before I started: `rocm-smi --showpids` -> "No KFD PIDs currently
running", `--showuse` -> GPU[0] (= physical 3) 0%, sclk 2365 MHz.

Round 30 already dumped the base ISA of a tree byte-identical to `op/current`
(`rounds/030/3-act/raw/isa/base/`, confirmed with `diff -rq`), so the survey cost
**zero GPU seconds and zero compiles**.  In 30 rounds nobody had histogrammed the two
hot loops.  Per kv block:

| `k_dqg` `.LBB0_4` (unrolled x2, 1311 instr = 655.5/block) | n | % |
|---|---|---|
| `s_set_vgpr_msb` | 280 | 21.4 |
| `v_pk_mul_f32` | 256 | 19.5 |
| `v_wmma_*` | 192 | 14.6 |
| `v_pk_add_f32` | 128 | 9.8 |
| `v_exp_f32_e32` | 128 | 9.8 |
| `buffer_load_b128` | 64 | 4.9 |
| `v_cvt_pk_bf16_f32` | 64 | 4.9 |
| `ds_store_b128` / `ds_load_tr16_b128` | 32 / 32 | |
| `v_nop` 31, `v_or_b32` 30, `s_clause` 16, waits 30 | | |

| `k_dkdv` `.LBB0_11` (qloop_full, 660 instr) | n | % |
|---|---|---|
| `s_set_vgpr_msb` | 109 | 16.5 |
| `v_nop` | 66 | 10.0 |
| `v_wmma_*` | 64 | 9.7 |
| `v_pk_mul_f32` | 64 | 9.7 |
| `v_mov_b64` | 64 | 9.7 |
| `ds_store_b128` / `ds_load_tr16_b128` | 40 / 40 | |
| `buffer_load_b128` 32, `v_pk_add_f32` 32, `v_exp_f32` 32, `v_cvt_pk_bf16_f32` 32 | | |

**The headline.  Softmax VALU is 288 of 655 instructions = 44% of `k_dqg`'s hot loop**,
and it decomposes exactly: the source chain is

```python
masked = sv[si] * scale                                   # pk_mul
pf     = exp2((masked - lse_q[qh_]) * LOG2E)              # pk_add, pk_mul, exp
ds     = (pf * (pv_ - del_q[qh_]) * scale).to(bf16)       # pk_add, pk_mul, pk_mul, cvt
```

= 4 `v_pk_mul` + 2 `v_pk_add` + 1 `exp` + 1 `cvt` per 2 elements, which predicts
256/128/128/64 per kv block.  The ISA says 256/128/128/64.  The model is exact, so a
count removed from this chain is a count removed from the loop.

**Free negative result (instrument `r31.i0.C1`), also from the same dump:** a wait-
counter drain audit.  `s_wait_storecnt` appears **nowhere** in either hot loop, and the
loads are already graded (`s_wait_loadcnt 0x20, 0x1f, ... 0x16`), with only 2
`s_wait_dscnt 0x0` in `k_dqg .LBB0_4` and a single `0x0` of each kind in `k_dkdv
.LBB0_11`.  Corpus item **L1 (combined vm-counter over-drain) does not apply to this
op** -- gfx1250's split counters are already being used as intended by the compiler.
Closed for free; do not spend an arm on it.  (`k_dqg`'s *epilogue* `.LBB0_9` is a
different story: 120 waits, 77 of them `s_wait_xcnt 0x0`.  Noted for a later round.)

## 3. Sourcing, and two candidates I killed before building

- **Apply g59 WMMA A/B-operand reuse to `k_dqg`** (`kfr`/`vfr` are invariant across the
  `qh_` loop, so the hint is free there and `k_dqg` passes `reuseA=False` at all three
  WMMA sites).  **Killed on `dead_ends.md:253`**: g59 measured +0.08% against a 0.61%
  floor and the entry says in terms, "do not spend another arm on reuseA/reuseB on this
  op."
- **Flip `VF_Q = True`** (already implemented, dead in-tree).  **Killed on h73**:
  fma-only VF_Q on the unrolled base measured **+2.0%, a loss**, and h71 has q1 alone at
  -2.37% op / -1.19% stacked with u2n.

Both kills point at the same open question, and it is worth more than either arm: *why
did VF_Q lose?*  It removes two loop-invariant multiplies from a 44%-of-loop chain and
it still went backwards.  The only thing it does that my reading of the chain does not
require is call `fx.fma`.  So the round's design is to run the transform **without the
fma** and let the difference price the fma's lowering.

## 4. The two arms (measured apart, each built alone from `op/current`)

Mechanism, both arms: **constant-hoist out of the per-element softmax chain** -- the
general form of the corpus's measured +1.76% "hazard-anchor coarsening".  Two of the
four multiplies are by loop-invariant scalars.

- **`r31.i1.g91` -- `k_dqg` constant-hoist.**  `scale` ahead of the exponent folds into
  `_c1q = scale*log2e` (already computed in-tree); the trailing dS `* scale` moves to
  the dQ store, which is legal because dQ is linear in dS and `k_dqg` -- unlike
  `k_dkdv` -- never emits P.  Chain becomes 2 pk_mul + 2 pk_add.  **-64 `v_pk_mul` per
  kv block = -9.8% of the loop**; at the measured `k_dqg` conversion of 0.61x and its
  34.9% share, ~+2.1% prod (band +0.8 .. +2.5%).  Differs from the dead VF_Q in
  **exactly one token**: `sv[si] * _c1q + _nlq[qh_]` instead of
  `fx.fma(sv[si], _c1q, _nlq[qh_])`.
- **`r31.i2.g92` -- `k_dkdv` constant-hoist.**  Same fold; P stays **byte-identical**
  (it is the dV GEMM's A operand and must not carry the scale), so only the trailing dS
  scale moves, onto the **dK** accumulator at the epilogue store -- both store paths,
  the PARTIAL fp32 workspace and the g86 bf16 LDS-staged one.  **-32 `v_pk_mul` of 660
  = -4.8%**; at `k_dkdv`'s 0.15x conversion and 61.1% share, ~+0.45% prod, i.e. near the
  0.32% floor.  Stated honestly up front: h71 has "VF on k_dkdv alone +0.17% (null)".
  Its value is that the pair prices **one mechanism against both of this op's
  issue-to-time conversion regimes in a single session** -- if A wins and B is null,
  the conversion factor, not the instruction count, is the thing that predicts.

A third tree `armAB` is built for the merge, to be measured **only if neither arm lost**.

### Correctness bug found in the dead code (free, worth recording)

`VF_KV = True` is **wrong as it stands in `op/current`**.  Its body drops the trailing
`* scale` from `ds_l` with the comment "applied once to the dK accumulator at the
store", but **neither** `k_dkdv` epilogue branch has any `VF_KV`-guarded scale -- grep
for `scale` in `_dkdv_impl` returns nothing after line 445.  Anyone flipping that flag
would ship a silently wrong dK.  Arm B supplies the missing half under its own flag and
leaves `VF_KV` untouched.

## 5. Build log

- **Build failure 1 (harness, not code).**  First detached launch of the static gate
  died instantly (`python3 <defunct>`, no `out` file).  Cause: I had put the arm trees
  in the named scratch `/tmp/op-evolve-...-r031`, but `docker inspect fa-g3` shows the
  container's only bind is `/home/lihuzhan -> /home/lihuzhan` -- **the container's
  `/tmp` is private**, so `$S/armA` did not exist inside it.  Fixed by relocating the
  scratch to `rounds/031/1-opt/scratch/` (under the shared mount) and leaving the named
  `/tmp` path as a host-side symlink to it.  No GPU time lost; caught by the sentinel,
  which is what it is for.

## 6. The free probe that killed my own hypothesis (`r31.i0.C2`)

Before spending a GPU second I compiled four trees and histogrammed them: the two arms
as first drafted, the merge, and **`VF_Q = True`** -- the dead in-tree arm that h73
records at **+2.0% (a loss)**.  The probe was supposed to confirm that `fx.fma` lowers
to an unpacked `v_fma_f32`, which would have explained the loss as a pairing failure.

**It does not.**  `fx.fma` lowers to **`v_pk_fma_f32` -- packed**:

| `k_dqg` `.LBB0_4` | tot | msb | pk_mul | pk_add | pk_fma | exp | wmma |
|---|---|---|---|---|---|---|---|
| base | 1311 | 280 | 256 | 128 | 0 | 128 | 192 |
| `VF_Q=True` (the **loser**) | **1100** | 248 | 64 | 64 | **64** | 128 | 192 |
| first-draft armA (hoist + scale-at-store, no fma) | 1176 | 276 | 128 | 128 | 0 | 128 | 192 |

So the arm that measured -2.0% is the one with **211 fewer instructions in the hot
loop, 32 fewer `s_set_vgpr_msb`, one fewer VGPR, and no unpacked arithmetic anywhere**.
That is the round's most useful result and it cost nothing: **on `k_dqg`'s hot loop the
instruction count does not predict the time, even though `u2n` -- 86 instructions out
of the same loop -- bought +2.48% prod.**  The issue-bound model that has driven this
op since round 24 has a boundary, and the boundary is not where the count is.

What is left as the suspect, by elimination: VF_Q's *other* half, moving `scale` onto
the dQ accumulator at the store.  `k_dqg`'s dQ epilogue is 256 `buffer_store_b16` in a
block (`.LBB0_9`) that already carries 120 waits of which 77 are `s_wait_xcnt 0x0`;
VF_Q puts a dependent VALU in front of every one of them.  **Not proven, and I am not
spending this round's arms proving it** -- recorded in `pool.md` as the reason the
scale-at-store family is now suspect and as a cheap future instrument.

### Consequence: both arms were narrowed before they ever ran

I rewrote both arms to move **only** the exponent constant and to leave both epilogues
byte-identical.  That is the smaller, more surgical edit, it is the only half of the
transform not already implicated in a loss, and it keeps the confound out of the
number.  Each arm is a change to one expression and nothing else.

## 7. Static gate (h3 / h18) -- free, before any timing

| | `k_dqg` hot `.LBB0_4` | masked `.LBB0_8` | `k_dqg` VGPR | `k_dkdv` hot `.LBB0_11` | masked `.LBB0_4` | `k_dkdv` VGPR | spill / `scratch_` |
|---|---|---|---|---|---|---|---|
| base | 1311 | 827 | 991 | 660 | 653 | 729 | 0 / 0 |
| **g91** (armA) | **1240** (-71) | **790** (-37) | 993 (+2) | 660 (=) | 653 (=) | 729 (=) | 0 / 0 |
| **g92** (armB) | 1311 (=) | 827 (=) | 991 (=) | **631** (-29) | **688 (+35)** | **742 (+13)** | 0 / 0 |
| g91+g92 | 1240 | 790 | 993 | 631 | 688 | 742 | 0 / 0 |

- **g91 PASSES cleanly.**  `v_pk_mul_f32` 256 -> 192, exactly the predicted -64; `msb`
  280 -> 275; `k_dkdv` byte-identical, so the arm is confined to the kernel it names.
  VGPR 991 -> 993 buys nothing and costs nothing -- there is no occupancy rung between
  here and 1024, the kernel is already pinned at 1 wave/SIMD.
- **g92 passes the kill screen but fails its own thesis.**  The hot loop did fall (660
  -> 631) but only 16 of the predicted 32 `pk_mul` went (LLVM had already CSE'd the
  rest), and the **masked** loop went the *wrong* way, +35 instructions and +37
  `s_set_vgpr_msb`, with VGPR +13.  I moved `_c1`/`_nl` up beside `lse_q` to match
  `k_dqg`'s placement and recompiled: 635 -> 631 in the hot loop, masked loop unchanged
  at 688.  So the regression is real and is the round-27 interleaving tax again (+37
  msb on +35 instructions), not a placement mistake.
  Masked-loop trip share at prod is 1.55%, so the weighted body change is about -24 of
  660 = -3.7%; at `k_dkdv`'s 0.15x conversion and 61.1% share that is **~+0.33% prod,
  i.e. the 0.32% same-code floor**.  **I am reporting this prediction before the run,
  not after**: g92 is expected to be null.  It is still worth measuring, because it and
  g91 are the *same edit* on the two kernels and the pair is the cheapest available
  test of whether the 0.61x / 0.15x conversion factors predict anything.

Predicted for g91 from the only calibration this op has (`u2n`: -86 instructions in
this loop -> +2.48% prod): **-71 -> ~+2.0% prod**, band +0.8 .. +2.5%.  Recorded before
the measurement.

## 8. Measurement

Five arms interleaved in one process per shape so every arm shares one clock excursion
(h57): `beat` (the bar, re-measured this session), `base` (= `op/current`, the
incumbent, re-measured this session), `g91`, `g92`, `g91g92`.  Two passes x
prod/proxy/fast, blocked lead 4 / block 9 palindromic (harness defaults), fresh
`FLYDSL_RUNTIME_CACHE_DIR` (h72), sclk+power sampled alongside every process, `dmesg`
watched live for amdgpu faults throughout.

Idle witness before launch: `--showuse` 0%, sclk 2364 MHz, and one KFD row -- PID
231638, `UNKNOWN`, **0 VRAM, 0 SDMA, 0 CU occupancy** -- which is the benchmark's own
process appearing as the witness was taken.  Recorded as observed.

## 9. Result -- BOTH ARMS ARE REFUTED

Both passes, five arms interleaved in one process per shape, sclk recorded beside every
arm (identical within a shape by construction -- that is what interleaving buys). TF/s,
and the percentage is against `base` **in the same process**:

| shape | pass | sclk | beat | base | **g91** | **g92** | g91+g92 |
|---|---|---|---|---|---|---|---|
| prod | 1 | 1759->1811 | 815.74 | 655.64 | **651.70 (-0.600%)** | **633.94 (-3.309%)** | 630.12 (-3.892%) |
| prod | 2 | 1764->1809 | 817.58 | 657.37 | **653.49 (-0.591%)** | **634.68 (-3.452%)** | 630.52 (-4.084%) |
| proxy | 1 | 1884->1955 | 748.06 | 599.04 | **587.31 (-1.958%)** | **577.66 (-3.569%)** | 574.97 (-4.018%) |
| proxy | 2 | 1900->1959 | 748.75 | 605.76 | **585.41 (-3.361%)** | **576.83 (-4.777%)** | 571.24 (-5.699%) |
| fast | 1 | 2328->2314 | 58.43 | 97.74 | **97.85 (+0.109%)** | **96.86 (-0.903%)** | 96.58 (-1.189%) |
| fast | 2 | 2329->2315 | 58.09 | 98.75 | **98.20 (-0.549%)** | **96.58 (-2.195%)** | 97.21 (-1.558%) |

**prod is the decisive shape and the two passes agree to a hundredth of a point on the
arm that ships**: `g91` -0.600% and -0.591%. `g92` -3.309% and -3.452%. proxy and fast
scatter as `facts.md` says they do (fast 0.8-2.8 points), which is why neither is read as
a refutation on its own.

`beat` prod 815.74 / 817.58 against base 655.64 / 657.37 = ratio **0.804** in both passes,
essentially round 30's 0.806. **The bar did not move under us this session**, and `base`
re-measured within 0.26% of round 30's 657.65 -- the ruler is the same one.

**Every control fired exactly as declared in advance.**

- **`fast` is a strict A/A for `g91` and it reads +0.109%.** `fast` dispatches `k_dq_sp`
  and never `k_dqg`, so this is the same machine code on both sides; +0.11% is the
  scatter, and it is small. The instrument works.
- **`fast` is live for `g92`** (`k_dkdv` is dispatched at every shape) and it loses
  there too, -0.903%, in the same direction as prod and proxy.
- **The merge is additive at prod and is not at proxy.** The two arms touch disjoint
  kernels and disjoint expressions, so a plain sum was the prediction. **prod: measured
  -3.892% against a predicted -0.600 + -3.309 = -3.909 -- agreement to 0.017 points.**
  **proxy: measured -4.018% against a predicted -1.958 + -3.569 = -5.527 -- the merge is
  1.5 points BETTER than the sum of its parts.** ⚠ Two arms in disjoint kernels cannot
  help each other, so this is **recorded as a discrepancy rather than banked**: proxy is
  the shape that dispatches `k_dqg` and `k_dkdv` through the split-K path, and whatever
  couples them there is not something this round measured. The prod reading is the one
  the round is scored on and it behaves.

### What was predicted, and what happened

| | predicted (written down before the run) | measured (prod) |
|---|---|---|
| `g91` | **+2.0%**, band +0.8 .. +2.5% | **-0.596%** (both passes) |
| `g92` | **+0.33%**, i.e. null at the 0.32% floor | **-3.38%** (both passes) |

`g91` missed its band by 2.6 points and landed on the wrong side of zero. `g92` missed by
3.6 points, and **ten times its own gate's magnitude**: the masked-loop regression it was
docked for is 1.55% of prod trips and cannot produce -3.3% by any arithmetic.

### The finding, stated as narrowly as the evidence allows

Three independent measurements now exist on this op for one transform -- *fold a
loop-invariant scalar into the `exp2` exponent of the softmax chain*:

| arm | kernel | fma? | scale-at-store? | hot-loop instructions | prod |
|---|---|---|---|---|---|
| `VF_Q=True` (h73) | `k_dqg` | yes | yes | 1311 -> **1100** | **+2.0% (loss)** |
| `r31.i1.g91` | `k_dqg` | no | **no** | 1311 -> **1240** | **-0.600%** |
| `r31.i2.g92` | `k_dkdv` | no | **no** | 660 -> **631** | **-3.309%** |

**It loses in every configuration**: with the fma and without it, with the store moved
and with both epilogues byte-identical, on the 0.61x kernel and on the 0.15x kernel.
The instruction count falls every time.  That is as close to a closed family as three
arms can get, and it is closed at a cost of one session.

⚠ **It also breaks h16's two-regime prediction, in the sign as well as the size.** The
conversion factors say `k_dqg` (0.61x) should move ~4x more than `k_dkdv` (0.15x) for
the same relative cut. **The opposite happened**: `k_dkdv` moved 5.5x more than `k_dqg`,
and both moved the wrong way. Whatever governs these two loops, it is not the
issue-slot conversion factor that has been used to size candidates since round 24.

### A mechanism I can name but did NOT measure, so it is filed as a hypothesis

The obvious suspect is that the softmax chain's ordinary VALU work is **not on the
critical path at all**, and that what the edit actually did was remove slack that the
scheduler was using. The corpus supports the shape of this without settling it: on
gfx950 `v_exp_f32` is **half-rate, issue cost 2.00 against 1.00 for `v_mul_f32`**
(`knowledge/optimization/routes/0-kernel-programming-principles.md:240`, measured), and
`knowledge/ops/attention/online-softmax.md:198` says in terms: *establish where the
ceiling sits before spending anything on the transcendental*.

**There is no gfx1250 measurement of that rate anywhere in the corpus.** Cost-weighting
`k_dqg`'s hot loop at the gfx950 rate puts the transcendental at 128 of ~352 VALU cost
units per kv block -- large, but not obviously dominant, and the gap between "large" and
"dominant" is exactly what decides whether any softmax-VALU arm can ever pay. **I am not
asserting it.** It is written into `pool.md` as the next instrument with the one clean
measurement that would settle it, because this round has now spent two arms discovering
the same thing from the outside.

## 10. What ships

`g92` lost, so **the merge is not shipped** regardless of how it reads. The round ships
its best single arm, **`r31.i1.g91`**, into `rounds/031/op` -- byte-identical (md5) to
the `armA` tree that produced the number above -- and is **NOT accepted on speed**:
-0.600% prod against a 0.32% same-code floor is a real loss, not scatter.

Reported as measured. An honest failing arm costs nothing; a number adjusted to look
better would have cost this campaign the three-arm family closure above, which is the
only durable thing round 31 produced.

## 11. Correctness (h1) -- run on the ship tree, exit codes as measured

`md5sum` first, so that the tree being graded is provably the tree that was measured:
`rounds/031/op/*.py` is **byte-identical** to the `armA` tree that produced the numbers
in §9.

```
ut/test_correctness.py --impl rounds/031/op        UT_RC  0
validation.py rounds/031/op                        VAL_RC 2
  correctness fast   dq 52.61 dB  dk 52.65 dB  dv 52.83 dB
  correctness proxy  dq 52.52 dB  dk 52.57 dB  dv 52.67 dB
  correctness prod   dq 52.56 dB  dk 52.60 dB  dv 52.71 dB
  determinism fast: dk/dv bitwise identical across 200 runs; dq bitwise (floor 70 dB)
  correctness  pass        determinism  pass        speed  FAIL
  fast 1.650x beat | proxy 0.785x | prod 0.799x  -> min 0.785x, target >= 1.00x
```

**`VAL_RC 2` is the job's standing state** (speed FAIL), unchanged from round 30, and it
is reported as measured rather than as a target.

⚠ **Two things worth keeping from this.** First, the arm is an arithmetic reassociation
and is therefore *not* bitwise against the incumbent -- yet SQNR came out at 52.52-52.83
dB against a 50 dB gate, i.e. **indistinguishable from the incumbent's own 52.52 dB**.
The reassociation costs no measurable precision; it simply costs time. Second, the
determinism check still passes bitwise on dk/dv over 200 runs, which is the cheap signal
that `g91` really did leave `k_dkdv` alone.

Card health: `dmesg | grep -c amdgpu` was **855 before the round and 855 after**, across
every compile, both benchmark passes and the validation run. No fault, no reset, no soft
recovery.

## 12. Ledger

- **Shipped:** `r31.i1.g91` into `rounds/031/op`. **Not accepted on speed** (-0.596% prod
  against a 0.32% floor).
- **Refuted:** `r31.i1.g91` (-0.596% prod), `r31.i2.g92` (-3.38% prod), the merge
  (-3.99% prod, not shipped because `g92` lost).
- **Refuted for free, no card time:** `r31.i0.C2` -- the `fx.fma` hypothesis. `fx.fma`
  lowers to packed `v_pk_fma_f32`.
- **Closed for free, no card time:** corpus item L1 (combined vm-counter over-drain)
  does not apply to this op.
- **Killed before being written:** `reuseA`/`reuseB` on `k_dqg` (`dead_ends.md:253`), a
  plain `VF_Q=True` flip (h73, h71).
- **New standing prohibition:** softmax exponent reassociation, both kernels, all three
  configurations. Written to `dead_ends.md`.
- **New pool rule 6:** do not add arithmetic to `k_dqg`'s dQ store epilogue until the
  one-compile instrument prices it.
- **Next instrument, and it is cheap:** measure the **gfx1250 transcendental rate**. The
  corpus has gfx950's (half-rate, cost 2.00) and nothing for this part, and it is the
  one number that would have predicted all three of these losses in advance.

## 13. Champion re-measure -- every distinct per-shape champion, one session, one device

§8's sweep interleaved the bar, the incumbent and the three arms, but it did **not**
contain `rounds/030/op`. That is a gap in the thing step 6 says is not a formality, so it
was closed with a second interleaved sweep rather than by reading round 30's stored
figures: `beat`, `r029` (= `op/current`, kernels.py md5 `37f3705…`), `r030` (`9e50ba9…`)
and `r031` (the ship tree, `d97c64c…`) as four arms in one process per shape, two passes,
blocked 9 + lead 4 palindromic, fresh `FLYDSL_RUNTIME_CACHE_DIR` (h72).

Rounds before 28 were deliberately **not** measured: they implement the architecture the
round-28 refactor replaced, several are faster than anything on the current structure, and
that is expected of a restructuring that landed correct and slow. Best-ever here means best
on the **current** structure, i.e. round 28 or later.

Idle device: `fa-g2` was running another job's benchmark throughout, but on a different
physical card -- `0003:04:00.0` against this container's `0004:04:00.0`, one `renderD*`
node each -- so `rocm-smi --showpids` listing it inside `fa-g3` is a KFD-wide listing, not
contention. `--showuse` read 0% before and after. `dmesg | grep -c amdgpu` 855 -> 855.

Mean of two passes, TF/s (`raw/bench2/table.txt` has both passes and the sclk range):

| shape | beat | r029 | r030 | **r031 (ship)** | vs best ever | ratio to target |
|---|---|---|---|---|---|---|
| prod  | 818.55 | **657.12** | 656.18 | 653.08 | **0.9939** | 0.7979 |
| proxy | 739.82 | **603.38** | 602.64 | 588.40 | **0.9752** | 0.7953 |
| fast  |  57.27 |   98.13 |  96.41 | **98.66** | **1.0053** | 1.0000 (capped) |

**score 0.8644**, and the round is **not accepted**: condition one is throughput against
the best ever re-measured beside it, and prod and proxy are both below it. The other two
conditions hold -- every below-target shape is above its 95% floor (0.9939, 0.9752) and
`fast`, the one shape that had reached its target, stays above it.

Three things in this table are worth more than the score.

1. **The loss reproduces across two independent sessions.** prod `g91` read -0.600% /
   -0.591% against `base` in §8 and -0.649% / -0.579% against `r029` here -- four passes,
   two processes, two cache dirs, a 0.070 percentage-point spread. This is not scatter.

2. **`r030`'s stored fast champion did not survive re-measurement, which is exactly why
   step 6 exists.** Round 30 recorded fast at 100.37 (r030) vs 99.37 (r028) and took the
   champion on it. Re-measured today beside `r029`, `r030` reads **96.41 against 98.13**,
   i.e. -1.2% / -2.3% -- the opposite sign. Round 30's own act.yaml already warned that the
   fast channel scatters over 0.8-2.8 points and that its figure was an A/A null, so this
   is that warning coming true rather than a new surprise. **`r031`'s +0.5% at fast is the
   same A/A null and is claimed as nothing**: `use_g` is False at fast, `k_dqg` is never
   dispatched, and the only kernel `g91` touches therefore never runs.

3. **`r030` is behind `r029` on all three shapes today** (-0.14% prod, -0.12% proxy,
   -1.75% fast). The incumbent is still `r029` on every shape, and the pre-28 rounds remain
   out of scope by construction.

The bar moved between the two sweeps -- proxy `beat` read 748.06 / 748.75 in §8 and
729.12 / 750.52 here, a 2.9% spread within one process pair. Each sweep's ratios are
therefore computed only against the `beat` measured **inside that same sweep**, and the
`beat` figure reported to the ledger is this sweep's, not a carried-forward one.
