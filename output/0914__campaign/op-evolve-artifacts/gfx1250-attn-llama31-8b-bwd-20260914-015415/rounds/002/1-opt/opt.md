# Round 2 -- optimisation log (fast round)

Op: attention (Primus-Turbo Triton fused one-kernel backward), gfx1250, GPU[1].
Incumbent: round 1's merge (`waves_per_eu: 0` + scoped `TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1`),
8.683 ms / 633 TF/s / 1.340x `beat`. Gate needs 1.50x, i.e. ~706 TF/s, i.e. **-11.5%
from here**.

Written as the round ran.

## 0. What round 2 inherited

`route.md` row 4 said this round's work is `r1.i7.g7` ("stage the spilled state in
LDS"), with `TRITON_HIP_USE_ASYNC_COPY` and `num_stages` first. Both of those were
**already measured by round 1's own knob sweep** and both are dead there
(`ASYNC_COPY=0/1` ~10.0 ms = noise; `num_stages=2` 12.2-12.9 ms). So g7's cheap
handles were gone before this round started and only its structural half remained --
which is a kernel rewrite, not a fast-round item. Row 3 (`r1.i5.g5`,
`BLK_SLICE_FACTOR`) was still untested. This round therefore went back to the
instrument question first (`r1.i6.g6`), because every candidate above g4/g7 was being
chosen from a static census and an end-to-end time.

## 1. The instrument `r1.i6.g6` asked for, taken statically -- and it answered

Round 1 could not get `rocprofv3` to record a dispatch and left "where inside the
kernel does the time go" open. **It is answerable without a profiler**, from the
`.amdgcn` the Triton cache already writes: find the backward branches, and count what
is inside each loop body. No GPU, no counters, ~5 minutes. This is `r1.i6.g6`
answered at the resolution that actually mattered.

Census of the SHIPPED build (round 1's merge), `raw/census_base.txt`:

| | |
| --- | --- |
| `vgpr_count` | 1024 |
| `vgpr_spill_count` | **266** (was 326 before round 1's merge) |
| `sgpr_count` / spill | 107 / 67 |
| `private_segment_fixed_size` | **1064 B/lane** (was 1308) |
| static `scratch_load` / `scratch_store` | 265 / 128 |
| static `v_wmma_f32_16x16x32_bf16` | 448 |

Round 1's merge took spills 326 -> 266 (-18%) and time -13.0%. That correlation is
the calibration this round used: **on this kernel, spilled state is time.**

### Where the spill is: entirely in the dk/dv pass

Loop bodies located by backward branch, then instruction-counted
(`raw/isa_loops.txt`):

| region | wmma | scratch_ld | scratch_st | ds | global |
| --- | --- | --- | --- | --- | --- |
| dk/dv pass, masked inner loop | 128 | **76** | 0 | 108 | 16 |
| dk/dv pass, unmasked inner loop | 128 | **77** | 0 | 108 | 16 |
| dq pass, masked inner loop | 96 | **0** | 0 | 88 | 8 |
| dq pass, unmasked inner loop | 96 | **0** | 0 | 88 | 8 |

**All spill traffic is in the dk/dv pass; the dq pass has none.** And the stores are
outside the loops -- these are loop-invariant values spilled once and *re-loaded every
iteration*, 119 dwords per iteration of the 266 spilled. That is pure deletable work,
and it is the first thing this job has that says WHICH HALF of the kernel to attack.

The arithmetic behind it, which now has an instrument agreeing with it: at
`BLOCK_N1=256`, `HEAD_DIM=128`, 4 warps wave32 (128 lanes), the dk/dv pass holds
`dk` 256 + `dv` 256 + `k` 128 + `v` 128 = **768 VGPRs of loop-invariant state** before
a single temporary. The dq pass holds `dq` 256 + `q` 128 + `do` 128 = 512 and fits.

### The bigger surprise: a quarter of the loop is bank-switching, not arithmetic

Full instruction histogram of the dk/dv unmasked loop body -- which is exactly ONE
iteration (128 wmma = the four dots at this tiling) -- `raw/isa_hist.txt`:

| count | instruction | what it is |
| --- | --- | --- |
| **363** | `s_set_vgpr_msb` | **gfx1250's high-VGPR bank switch** |
| 128 | `v_wmma_f32_16x16x32_bf16` | the actual matrix work (9.1%) |
| 96 | `v_perm_b32` | in-thread transposes / bf16 packing |
| 84 | `v_nop` | hazard padding |
| 77 | `scratch_load_*` | spill reloads |
| 64 | `v_pk_mul_f32` | 2 elementwise passes over the fp32 score tile |
| 64 | `v_exp_f32` | the softmax |
| 64 | `v_cvt_pk_bf16_f32` | pT and dsT down-converts |
| 48 | `v_fma_f32` | the `*RCP_LN2 - m*RCP_LN2` fold (already fused by LLVM) |
| 41 | `v_*add_nc_u32` | **pointer-tensor increments** |
| 1400 | total | |

**`s_set_vgpr_msb` is the single largest instruction class in this kernel, at 26% of
the loop body.** gfx1250's flat 1024-VGPR file is addressed 256 at a time and an
instruction naming a register above v255 needs the MSB mode set first; at 1024 VGPRs
in use the compiler emits roughly one per two VALU instructions. Together
`s_set_vgpr_msb` + `v_nop` + `scratch_load` = **517 of 1400 instructions, 37% of the
dominant loop, is register-pressure overhead and not arithmetic.** Only 9% is wmma.

That is a *new* fact for this job, and it sharpens the bottleneck line: the cost of
1024 VGPRs on this part is not only the 266 spills, it is a bank-switch tax on every
instruction that touches the top three quarters of the file. It also explains why
549 -> 633 TF/s sits so far under a 2236-2667 TF/s healthy bf16 GEMM: the kernel is
instruction-bound on overhead.

## 2. Free checks that closed candidates without a build

**`dk_pe = dk` / `dq_pe = dq` are already eliminated.** The kernel aliases the
positional-encoding accumulator onto the real one when `PE_HEAD_DIM == 0` ("couldn't
assign None"), which would be a 256-VGPR loop-carried ghost if it survived. It does
not: the TTGIR `scf.for` carries `(dk, dv, curr_m, qT_ptrs, do_ptrs)` and no `_pe`
value. Candidate killed for free, before it was proposed.

**The loop-carried values ARE two pointer tensors.** Same TTGIR line:
`iter_args(... %qT_ptrs = tensor<128x32xi32>, %do_ptrs = tensor<32x128xi32>)` -- 32
VGPRs per lane each, in the pass that is spilling, plus the 41 `v_add` per iteration
that walk them. The dq loop carries `kT_ptrs` and `vT_ptrs` the same way. This became
arm B.

## 3. Pool item `r1.i5.g5` measured, and a new degree of freedom found and killed

`BLK_SLICE_FACTOR` splits the masked diagonal block so the masked inner loop does
half-tiles. Round 1 reasoned it should help and never measured it. It does not:
`raw/sweep1.txt`, all in one process against a DEFAULT re-read twice for a floor.

| config | bwd ms | vs default |
| --- | --- | --- |
| DEFAULT (`M1=32, N1=256, M2=256, N2=32, BSF=1`) | **8.6720** | -- |
| DEFAULT again, end of session | 8.6780 | +0.07% |
| `BLK_SLICE_FACTOR=2` (**`r1.i5.g5`**) | 8.8726 | **+2.3% WORSE** |
| `BLOCK_N2=16` | 9.4940 | +9.5% |
| `BLOCK_M1=16` | 11.2990 | +30% |
| `BLOCK_M1=16, BLOCK_N2=16` | 12.0620 | +39% |
| `BLOCK_M1=64` | 15.9580 | +84% |

Two things fell out of this.

**`r1.i5.g5` is dead.** Halving the masked tile buys fewer wasted lanes and costs an
extra loop trip with its own prologue in the pass that is already spilling. The
diagonal is 1/32 of the work at this tiling; the prologue is not.

**`_check_block_invariant`'s `BLOCK_M1 == BLOCK_N2` rule is over-strict**, and lifting
it is a genuinely new degree of freedom this job had not touched. `BLOCK_M1` is the
step of the dk/dv pass and `BLOCK_N2` the step of the dq pass; they are used in
INDEPENDENT passes and nothing couples them. So the sweep above decouples them (by
monkeypatching the invariant, not by editing it). **Every point loses**, including
the `M1=16` direction that the spill analysis predicted should help by shrinking the
dk/dv loop's live temporaries. The instrument explains why it does not: the spilled
state is `dk`+`dv`+`k`+`v` = 768 VGPRs of loop-INVARIANT state sized by `BLOCK_N1` and
`HEAD_DIM`, which `BLOCK_M1` does not touch at all -- shrinking `BLOCK_M1` only doubles
the trip count over the same reload. **The degree of freedom is real and the direction
is exhausted; recorded so nobody pays for it again.**

Build failure, recorded as it happened: `BLOCK_M1=16, BLK_SLICE_FACTOR=2` (giving
`MASK_BLOCK_M1=8`) compiles with a diagnostic and produces garbage --

```
error: no matching matrix core intrinsic for wmma version 3 with instruction
shape [0, 0, 128] and element types A='bf16', B='bf16', C='f32'
  at  qkT = tl.dot(k, qT)   and   dpT = tl.dot(v, tl.trans(do))
```

It still ran, at 39.45 ms and 81.9 dB. **`BLOCK_M < 16` is illegal on this arch** --
the wmma is 16x16x32 and there is no fragment below it. Noted so a later round does
not read the 81.9 dB as "correct but slow".

## 4. What the instrument says to do, and what it says NOT to do

The loop body is 37% register-pressure overhead. The two structural ways to remove it
wholesale are both closed, and both were closed by measurement, not by argument:

* **Shrink `BLOCK_N1`** (the thing that actually sizes `dk`/`dv`/`k`/`v`). Round 1
  measured tile-128 at 11.8-13.3 ms. Halving `BLOCK_N1` halves the accumulators but
  doubles the number of K/V blocks, and every Q and dO block is re-read by each --
  the dk/dv pass's Q/dO traffic doubles. Closed.
* **Raise `num_warps`** so each lane holds less. Round 1 measured `num_warps=8` at
  2.36x slower (the config is `num_warps=4` and the kernel's LDS/transpose path does
  not survive the change). Closed.

So the 266 spills and the 26% `s_set_vgpr_msb` tax are **structurally locked in at
this tile**, and this round cannot delete them. What it CAN do is delete instructions
from the 1400-instruction body that carries them -- which is also exactly what the
**hard power cap** rewards (round 0/1 established the part sits at 2497-2502 W of a
2500 W cap with GFX clock 29% below max: deleting work pays, overlapping work does
not). Two of the histogram's rows are pure waste:

* `64 v_pk_mul_f32` -- **two** full elementwise passes over the fp32 score tile, one
  of which only multiplies by the scalar `sm_scale`.
* `41 v_*add_nc_u32` -- walking two loop-carried `tensor<*xi32>` pointer tensors that
  TTGIR confirms are `iter_args`, i.e. 32 VGPRs/lane each of live addressing state in
  the pass that is spilling.

Those are the two arms. They are independent (one is arithmetic in the score tile,
one is addressing), they are both in the dk/dv pass the instrument localised, and
neither changes the math.

## 5. The two arms, built and measured

Both built from `rounds/002/op` (verified byte-identical to `job_context/op/current`
apart from root-owned `__pycache__`). **B is built from the same base as A, not on top
of it.** Each arm's only difference from base is one file,
`vendor/primus_turbo/triton/attention/fused_mha_bwd_kernel.py`, confirmed by `diff -r`.

* **arm A = `r2.i1.g8`** -- fold `sm_scale` into the exponent's own constant.
  `qkT * sm_scale` then `exp2(x * RCP_LN2 - m * RCP_LN2)` becomes
  `exp2(qkT * (sm_scale * RCP_LN2) - m * RCP_LN2)`: `sm_scale * RCP_LN2` is a scalar
  the compiler folds at compile time, so the whole `64 v_pk_mul_f32` elementwise pass
  over the fp32 score tile disappears into the `v_fma` that was already there. Applied
  in `_bwd_dkdv_inner` and `_bwd_dq_inner`. The ALIBI path is preserved exactly by
  keeping the separate multiply when `USE_ALIBI` (the bias is added in pre-scale
  units) via an `exp_arg_scale` constexpr.
* **arm B = `r2.i2.g9`** -- delete the loop-carried pointer tensors. The offset
  tensors are rebuilt from the in-block index only (`tl.arange(0, BLOCK_M)`), made
  loop-invariant, and the walk becomes a **scalar** added at the load:
  `tl.load(qT_ptrs + curr_m * stride_qm)`. Same for `do_ptrs`, and for `kT_ptrs` /
  `vT_ptrs` in `_bwd_dq_inner`. Removes two `tensor<128x32xi32>` `iter_args` (32
  VGPRs/lane each) and the 41 `v_add_nc_u32` per iteration.

### Measurement

One arm per process, six processes back to back in one session, palindromic
(`base A B B A base`). Speed is `benchmark.measure` -- the job's only timing loop,
the same call `validation.py` makes. Precision is `ut/correctness.check_shape`, the
gate `validation.py` itself applies. GPU[1] verified idle before and after
(`rocm-smi --showpids`: no KFD PIDs). `raw/arms1.txt`.

| arm | SQNR out/dq/dk/dv | bwd ms, slot 1 | slot 2 | mean | vs base |
| --- | --- | --- | --- | --- | --- |
| base (incumbent) | 53.7 / 52.2 / 52.3 / 52.7 | 8.6924 | 8.6899 | **8.6912** | -- |
| **A** `r2.i1.g8` | 53.7 / 52.2 / 52.3 / 52.7 | 9.3504 | 9.3427 | **9.3466** | **+7.5% WORSE** |
| **B** `r2.i2.g9` | 53.7 / 52.2 / 52.3 / 52.7 | 9.2883 | 9.3159 | **9.3021** | **+7.0% WORSE** |

Session floor: base's two slots are **0.03% apart**. Both arms lose by 200x the floor.
Both are correct (identical SQNR to the incumbent, to 0.1 dB, on all four tensors).

**Neither arm survived, so there is no merge to measure.** The step-4 rule is explicit
that a merge is only built when neither arm lost, and that a merge is measured and
never inferred; two losing arms merge into a worse one, and building it would spend
the round's remaining GPU time confirming that.

### The result is suspicious in a specific way, so it was controlled

Two *independent* edits -- one arithmetic, one addressing -- losing by nearly the same
7% is not what two independent causes look like. That pattern says "common cause", and
the obvious common cause is the directory, not the diff. So a **null arm** was built:
`rounds/002/armNull`, a byte-identical copy of `rounds/002/op` (`diff -r` clean), and
measured in the same palindromic shape against base and A.

`raw/null1.txt`, same session shape, GPU[1] verified idle:

| arm | slot 1 | slot 2 | mean |
| --- | --- | --- | --- |
| base | 8.6873 | 8.6945 | 8.6909 |
| **null** (byte-identical copy) | 8.6907 | 8.6812 | **8.6860** |
| A | 9.3423 | 9.3546 | 9.3485 |

**The null arm reads the incumbent's time to within 0.06%.** The directory is not the
cause. A's 7.5% loss is A's.

## 6. WHY both arms lost -- and the law that falls out of it

The `.amdgcn` of all three builds is in `raw/census_arms.txt`. It was taken for free
from the Triton cache the timing run already wrote -- no extra GPU time at all.

| build | `vgpr_spill_count` | scratch B/lane | `v_pk_mul_f32` | `v_add_nc/add3` | `s_set_vgpr_msb` | bwd ms |
| --- | --- | --- | --- | --- | --- | --- |
| **base** | **266** | 1064 | 448 | 115 | 1666 | **8.691** |
| **A** `r2.i1.g8` | **307** (+15%) | 1228 | **416** (-32) | 126 | 1702 | 9.347 (+7.5%) |
| **B** `r2.i2.g9` | **325** (+22%) | 1288 | 448 | **124** (+9) | 1669 | 9.302 (+7.0%) |

**Both edits did exactly what they were designed to do at the source level, and both
made the kernel slower by making the register allocator's job harder.**

* A deleted the 32 `v_pk_mul_f32` it targeted -- 448 -> 416, precisely as predicted --
  and paid 41 more spilled VGPRs for them. Folding the scale into the `exp2` argument
  merges a short-lived value into a longer live range: `qkT` must now survive to the
  fma instead of being consumed and replaced by `qkT_scaled`. At 1024/1024 VGPRs there
  is no slack to absorb that.
* B did NOT even get the instruction it was after: `v_add` went **up**, 115 -> 124.
  Triton's pointer-tensor increment is one add per iteration on a value the allocator
  can keep coalesced; `ptr + curr_m * stride` re-materialises the full address tensor
  at each of the two loads instead, which is more arithmetic AND more simultaneous
  live values. 59 more spills.

**The law, and it is the most useful thing this round produced:**

> On this kernel at this tile, **time tracks `vgpr_spill_count` and nothing else**.
> Instruction count in the loop body does not predict it, and neither does live-range
> reasoning at the source level.

Four independent points now, spanning 22% of spill range and 22% of time:

| build | spills | bwd ms |
| --- | --- | --- |
| round 0 | 326 | 9.983 |
| **B** | 325 | 9.302 * |
| **A** | 307 | 9.347 * |
| round 1 ship / base | 266 | 8.691 |

(* B and A are close in spills and close in time; the ordering between them is within
what a different allocation of the same pressure can do. The trend over the range is
the point, and it is monotone.)

This kills a whole class of candidate that this job has been generating -- "delete
instruction X from the hot loop" -- unless the edit is *also* shown to reduce spills.
And it hands over a **free pre-filter**: the spill count is in the `.amdgcn` after a
COMPILE, which needs no timing run, no idle GPU and no benchmark session. A candidate
whose census comes back at >= 266 spills is already known to be no better, in seconds,
before it is ever measured. That is the cheapest instrument this job has found and it
is this round's real deliverable (`r2.i3.g10`).

## 7. No merge, and what shipped

Neither arm survived, so per the round rule there is **no merge to build**: a merge is
measured and never inferred, and two arms that each cost ~40-60 spills would merge
into one that costs both. The best arm is **B** (9.302 vs A's 9.347, B lower in every
slot, gap 0.48% against a 0.03-0.06% session floor), so **B is what shipped into
`rounds/002/op`**, as the rule requires even when every arm lost.

**This round is a regression and should not be accepted.** It is reported as measured.

## 8. Validation of the shipped copy

`raw/validate_ship.txt` -- `op/validation.py rounds/002/op`, through the runner, on
GPU[1] verified idle, `beat` re-measured in the same run.

```
CORRECTNESS  rounds/002/op  vs op/eager/   (gate: all four tensors >= 50 dB)
b4_s8192_hq32_hkv8_d128   53.67  52.24  52.31  52.71   pass
sq_lt_skv                 53.01  52.50  52.58  52.57   pass
min_seqlen                54.09  52.04  52.08  52.77   pass
ragged_tail               53.99  52.09  52.15  52.75   pass
sq_gt_skv                 53.97 -15.95  52.10  52.72   fail  (diagnostic, not gated)

SPEED   order=op>beat>baseline>baseline>beat>op
candidate   fwd 4.9579   bwd  9.3013   591.12 TF/s   spread 0.79%
beat        fwd 3.7126   bwd 11.6628   471.43 TF/s   spread 3.17%
baseline    fwd 4.9819   bwd 10.0239   548.51 TF/s   spread 0.77%

vs beat 1.254x   required 1.50x   vs baseline 1.078x   FAIL
VALIDATION FAILED: speed -- short by 16.4% of the target
```

**This is reported as measured, and it agrees with my own number to 0.01%**: my
session read arm B at 9.3021 ms, validation reads it at 9.3013 ms. Nothing to
reconcile. The incumbent is 8.683 ms / 1.340x; this round ships 9.301 ms / 1.254x and
is a **regression of 7.0%**.

The `sq_gt_skv` `dq = -15.95 dB` is **pre-existing and unchanged** -- the same value
round 1 recorded, the same value the incumbent gives, and it is a diagnostic shape
outside the gate (`Sq > Skv` is a non-causal-aligned shape the vendored causal kernel
does not handle). Nothing this round touched it; A, B and base all read identically on
every shape.

## 9. Harness note (cost this round two runs)

Three vendored `primus_turbo` trees in ONE process **segfault** (rc 139) at the first
`attention` call. `impl.py`'s shim makes the *opaque-type* registration idempotent,
but each tree also defines the forward as a `torch.library.custom_op` under the same
qualname, and the third registration takes the process down. `validation.py` never
hits this because only ONE of its three arms (`baseline`) carries a vendored tree --
`beat` is `flex_attention`. **A multi-arm harness must therefore run one arm per
process.** This round's arms share a *session*, back to back, palindromically, rather
than a process; the base slots came back 0.03-0.06% apart, so nothing was lost by it.
Recorded in `dead_ends.md` so round 3 does not rediscover it.

## 10. What round 3 should do

The round's ideas both failed, but the round produced the thing the job has been
missing since round 0: **a cheap, falsifiable predictor of time.** Time tracks
`vgpr_spill_count`, over four points and 22% of range, and `vgpr_spill_count` is
readable from a COMPILE with no GPU and no benchmark session.

So round 3 should (a) make the census a gate that every candidate passes before it is
allowed to consume GPU time (`r2.i3.g10`), and (b) spend its arms on the only thing
that can actually move the spill count at this tile: the **768 VGPRs of loop-invariant
state in the dk/dv pass**, which `BLOCK_M1` cannot touch and `BLOCK_N1` cannot shrink
without doubling traffic. The concrete handle is splitting `HEAD_DIM` so `dk` and `dv`
are 128+128 rather than 256+256 VGPRs, paying one recomputed dot (`r2.i4.g11`) -- and
g10 says whether that is worth a measurement before a measurement is spent.

---

# Steps 5-7 -- executing the route

The `## Route` table's rows, from the top.

## Row 1 (`must`, `r2.i3.g10`) -- the spill gate, installed and run

Delivered as `rounds/002/1-opt/census_gate.py`. It loads an impl dir, runs one
fwd+bwd so Triton compiles, then reads the `.amdgcn` out of `TRITON_CACHE_DIR` and
prints `vgpr_count`, `vgpr_spill_count`, `sgpr_spill_count`,
`private_segment_fixed_size`, `group_segment_fixed_size` and an instruction census,
with a verdict against the incumbent's 266. No timing loop, no palindromic session,
no idle-device wait. It takes an optional second argument that forces
`BLOCK_N1 == BLOCK_M2` (patching `fused_backward_tile`, which otherwise overwrites
them -- `dead_ends.md`), so a tile can be censused without editing the kernel.

One build failure, recorded: the first run died with
`TypeError: unsupported operand type(s) for +: 'int' and 'tuple'` in `triton.cdiv`.
`fused_backward_tile` returns a **scalar** `BLOCK_N1` which `dense_fused_backward`
assigns to both keys; my patch returned a 2-tuple. One-line fix.

### Row 1 immediately falsified its own premise, which is the best thing it could have done

The first thing worth censusing was `BLOCK_N1 = 128`, because round 1 has a *measured
time* for it (11.8-13.3 ms against 9.98) and the round-2 law predicts what its spill
count must be. `raw/gate_tile128.txt`:

```
[gate] forcing BLOCK_N1 = BLOCK_M2 = 128
    vgpr_count=771  vgpr_spill_count=0  sgpr_spill_count=0
    scratch_B_per_lane=0   lds=0
      v_wmma 224   s_set_vgpr_msb 833   scratch_load 0   scratch_store 0
      ds_ 524      v_pk_mul_f32 256     v_add* 377
```

**Zero spills. Zero scratch. `vgpr_count` 771, not 1024 -- so no bank-switch pressure
either. And it is 20-33% SLOWER than the incumbent, which spills 266.**

That is a clean falsification, obtained for free, of the law this round wrote down
four hours earlier. The correct restatement:

> Time tracks `vgpr_spill_count` **only at a fixed tile.** Across tiles the
> correlation inverts, because changing the tile changes Q/dO traffic, and in the
> dk/dv pass traffic dominates register pressure.

The four points that looked monotone (326/325/307/266 spills -> 9.983/9.302/9.347/
8.691 ms) were **all at `BLOCK_N1=256`**. Within that tile the law still holds and the
gate is still worth running. As a cross-tile predictor it is wrong, and it would have
sent round 3 straight at a candidate that is already measured and dead. Catching that
cost one compile.

## Row 2 (`idea`, `r2.i4.g11`) -- killed by row 1, before it was built

Row 1's stated job is "a gate on rows 2-3". It gated row 2 out.

Row 2 proposed splitting `HEAD_DIM` in the dk/dv pass into two halves of 64 so the
768 VGPRs of loop-invariant state fall to 384, paying one recomputed dot. Two things
came out of executing row 1 and then reading the kernel to write it:

**1. The 384 is unreachable.** `HEAD_DIM` is the FREE axis of `dk` and `dv`, so those
do split -- but it is the **contraction** axis of `qkT = tl.dot(k, qT)` and
`dpT = tl.dot(v, tl.trans(do))`. A half-pass that only holds `k[:, :64]` cannot form
`qkT` at all. So `k` and `v` stay resident in full and the real accounting is
**768 -> 512, not 768 -> 384**, for **+50% wmma** (each half-pass recomputes the full
`qkT` and `dpT`: 2 x (1 + 0.5 + 1 + 0.5) = 6 dot-units against 4), plus **2x Q/dO
global traffic**, because each half-pass runs the whole m-loop and each needs full
`qT` and full `do`.

**2. That exact trade is already measured, and it lost.** `BLOCK_N1 = 128` halves the
accumulators AND `k`/`v` -- 768 -> 384, the number row 2 wanted and could not reach --
at the cost of 2x Q/dO traffic in the dk/dv pass. Row 1 now shows it reaches **zero
spills and `vgpr_count` 771** doing it. Round 1 measured it at **11.8-13.3 ms**.

So `BLOCK_N1=128` is a strictly BETTER version of row 2's bet -- more pressure relief
(384 vs 512), no recompute (+0% vs +50% wmma), same traffic cost -- and it loses by
20-33%. Row 2 is dominated by a measured point. Building it would have spent the
round's remaining budget rediscovering a slower version of a known answer.

**Row 2 is abandoned at `look`, on evidence, not on difficulty.** The mechanism is in
`dead_ends.md`: *in the dk/dv pass, Q/dO traffic dominates register pressure; any
candidate that buys residency with traffic is already answered by the `BLOCK_N1` sweep
and does not need building.* That kills row 2, row 3 in its "stage K/V in LDS by
re-reading" form, and the whole "recompute instead of keeping it live" family that
`r1.i4.g4` kept generating.

## What the working copy contains

`rounds/002/op` holds **arm B, `r2.i2.g9`** -- the round's measured candidate, built
and measured in step 4, shipped as the best arm after both arms lost. It is left in
place exactly as measured. It is a regression and the acceptance rule will not promote
it; that is the correct outcome and not something to hide by reverting.

## Step 6 -- measurement, candidate beside the incumbent in one session

`op/benchmark.py`, unprofiled, **every shape**, one process per arm (three vendored
trees in one process segfault -- s9), four processes back to back in a single
`docker exec`, palindromic `cand inc inc cand`. `raw/final_measure.txt`.

Device check: `rocm-smi --showpids` during the run shows exactly **one** process on
GPU[1] (2.36 GB VRAM, `GPU[1] use 99%`). Two other PIDs are on **GPU[0]**, which sat
at 100% throughout -- another tenant, and precisely why the spec pins this job to
GPU[1]. Nothing of mine touched GPU[0] and nothing of theirs touched GPU[1].

| shape | candidate `r2.i2.g9` TF/s | incumbent `rounds/001/op` TF/s | vs incumbent |
| --- | --- | --- | --- |
| **b4_s8192_hq32_hkv8_d128** (spec, scored) | 591.39 / 591.90 -> **591.6** | 633.07 / 633.05 -> **633.1** | **0.9345x** |
| sq_lt_skv (edge) | 333.86 / 334.40 -> 334.1 | 354.65 / 355.19 -> 354.9 | 0.9414x |
| min_seqlen (edge) | 136.93 / 136.71 -> 136.8 | 135.48 / 124.25 -> 129.9 | 1.0535x * |
| ragged_tail (edge) | 305.00 / 304.45 -> 304.7 | 304.72 / 302.56 -> 303.6 | 1.0036x |
| sq_gt_skv (diagnostic, never gated) | 249.71 / 249.62 -> 249.7 | 248.51 / 251.07 -> 249.8 | 0.9995x |

\* `min_seqlen` is a 0.16 ms kernel with 3.7-3.8% spread on the incumbent's two slots
(135.48 vs 124.25 TF/s, an 8% swing on the SAME build in the SAME session). That
"win" is noise and should not be read as one. Every other shape's two slots agree to
within 0.7%, and the spec shape's agree to 0.08%.

The incumbent re-measured **633.1 TF/s** here against the 633.2 round 1 reported --
a session re-measure agreeing with a stored figure to 0.02%, which is the point of
taking it rather than recalling it.

**Scored shape: 591.6 TF/s against a 706.0 TF/s target -> ratio 0.838, and 0.9345x
the incumbent.** Throughput does not improve on the best ever, so the acceptance rule
does not promote this round. Reported as measured.

Correctness is unchanged and was checked with `op/validation.py` itself (s8): all four
gated shapes pass, minimum 52.04 dB over all four tensors and all four gated shapes,
identical to the incumbent on every shape to two decimals -- `r2.i2.g9` rewrites
addressing, not arithmetic.

## What round 3 inherits from this round

1. **A gate that works, and knows its own limits** (`census_gate.py`). Use it within a
   tile. It caught its own over-generalisation in one compile, which is worth more
   than the generalisation was.
2. **A hard exchange rate, measured:** in the dk/dv pass, going from 768 to 384
   resident VGPRs *by doubling Q/dO traffic* costs 20-33%. `BLOCK_N1=128` is the clean
   experiment and it is now fully characterised -- zero spills, `vgpr_count` 771, and
   slower. Every "recompute instead of keeping it live" candidate pays that rate.
3. **Therefore the only live direction:** lower residency **at constant Q/dO traffic**
   -- inside one pass over the m-loop, not by running it twice. Nothing currently in
   the pool does that, and finding something that does is round 3's real problem.
