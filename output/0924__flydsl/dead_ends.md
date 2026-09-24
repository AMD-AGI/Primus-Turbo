# Dead ends -- gfx1250-flydsl-attn-bwd

These are mechanisms that have been built, measured, or made structurally impossible. Every entry
names the round that closed it.

## Standing walls

- Round 17: PC sampling faults this GPU and ATT captures no runtime-loaded FlyDSL kernel. Do not
  schedule either for stall attribution.
- Rounds 10-17: instruction counts, wait counts, and static schedule models do not predict time;
  use them only as build gates. The sharpest counterexample is `r16.i2.g49`, whose static metrics
  all improved while prod lost 16.25%.
- Rounds 2-17: any spill is a refusal, not a cost. Spilling gfx1250 builds can hang after launch.
- Round 17: `k_dkdv` is 740 VGPR and `k_dq` 960 after correcting rocprofv3's half-count. Both are
  one wave/SIMD; lowering LDS alone cannot raise occupancy.
- All rounds: atomic dQ accumulation violates the deterministic-output contract.

## r1.i4.g04 -- four-wave `k_dkdv` with BLOCK_KV 128

Round 15 killed this offline. Required barriers collapse the load-to-use distances, four waves still
give only one wave/SIMD, and the single-wave alternative cannot fit its accumulator floor. This
closes the BLOCK_KV/occupancy route that `r3.i6.g15` required.

## r2.i1.g08 -- recover residency by reordering `k_dkdv` operands

Round 2 reduced VGPR **680 -> 668** without crossing a residency step and measured **-4.9%**.
The accumulator floor and scheduler-batched LDS destinations, not source operand liveness, hold the
registers.

## r3.i5.g14 -- widen single-wave `k_dkdv` BLOCK_KV 32 -> 64

Round 3 built at **982 VGPR**, zero spill, yet measured **2.76x slower**. The accumulator floor
doubled while the residual batching budget grew only 11%; zero spill was not sufficient.

## r3.i6.g15 -- fuse dQ into the KV-outer body

Round 17 confirmed the destination is worth up to the exact **1.4080 = 7/5** work factor, but the
recorded implementation is structurally barred. Its BLOCK_KV/multi-wave prerequisite was killed by
`r1.i4.g04`, and the BLOCK_KV=32 form failed the round-7 accumulator/workspace screen. Do not reopen
without a new fusion mechanism that does not depend on that prerequisite.

## r5.i2.g18 -- cluster multicast/TDM for `k_dkdv` Q/dO

Round 17 campaign correction 6 closes the premise: there is no global-to-LDS staging burst.
The 32 loads are global-to-register, dual-use, and already prefetched. Adopting multicast would undo
`r2.i2.g09` and `r7.i1.g21`; the corpus also prices this direct-to-LDS path at -9.0% to -14.7%.

## r7.i2.g22 -- source reorder of the LDS transposes

Round 7 changed MLIR and LLVM IR but produced byte-identical ISA and same-session noise. The backend
re-derives the same final schedule; source motion alone cannot interleave this tail.

## r10.i1.g27 -- add LSE/delta loads to the carried prefetch without preserving a partial wait

Round 10 measured **-14.57% prod**. The consumer moved behind a 36/36 full drain. Round 16's bare
source-order retry `r16.i1.g48` reproduced the mechanism before reaching the card.

## r10.i2.g28 -- two-iteration ping-pong unroll

Round 10 removed 17% of issue slots but measured about **-7.8% prod** in three sessions. It doubled
the load batch covered by the drain and shortened useful load-to-use overlap.

## r10.i3.g30 -- eliminate a supposed `s_set_vgpr_msb` register-bank tax

Round 11 closed the mechanism offline. gfx1250 has a flat VGPR file; the prefix encodes high register
numbers and does not move data. FlyDSL exposes no register-pinning lever, and round 17 found nearly
equal prefix counts in kernels with very different matrix utilisation.

## r10.i4.g31 -- change to a narrower WMMA shape for more accumulator chains

Round 11 found the current kernel already uses the only relevant bf16 shape,
`v_wmma_f32_16x16x32_bf16`, with independent chains. The proposed wider starting shape does not
exist on gfx1250.

## r10.i5.g32 -- prefetch only half the Q/dO state

Round 10 measured **-61.5% prod**. Half the loads became same-iteration issue-and-wait operations,
exposing their full latency. Total VGPR reduction did not remove register-window prefixes.

## r11.i1.g33 -- phase-split the Q/dO LDS transposes

Round 11 measured both phase-split variants (`r11.i1.g33` and cross-reference `r11.i2.g34`)
negative at all shapes, including **-2.09% / -1.42% prod**. Moving the work destroys the per-tile
interleave that hides latency at one wave/SIMD.

## r11.i3.g35 -- replace the LDS transpose with cross-lane VALU permutations

Round 12 killed it offline. `ds_load_tr16_b128` already performs the transpose in hardware;
the proposed form used at least 256 VALU instructions to replace 64 LDS instructions.

## r11.i5.g37 -- deepen `k_dkdv` prefetch from one iteration to two

Round 11's clean build exposed a FlyDSL limitation: loop-carried registers do not rotate, so the
second level adds a full block copy and a full drain. The only writable ping-pong form is the
already-losing `r10.i2.g28`.

## r12.i1.g39 -- place the two LDS rings in different 64 KiB segments

Round 12 halved `s_wait_dscnt`, reduced instructions by 68, preserved VGPR and spill, and measured
only **+0.59% / +0.21% prod**, within the 0.86% floor. LDS read-port contention is not the limiter.

## r13.i2.g43 -- XCD-major q-head remap

Round 13 measured **0.9928-1.0065x**, within a 1.8% floor. Round 14 then established that gfx1250
has one device-wide 4 MiB L2 record rather than private per-XCD LLCs, so the proposed locality
mechanism does not exist.

## r15.i1.g46 -- replace full waits with hand-written graded waits

Round 15 found the compiler already emits descending partial waits. The candidate's premise was
false; hand-pipelining an already-pipelined body is barred.

## r15.i2.g47 -- `sched_group_barrier` interleave of LDS reads and WMMAs

Round 15 measured **fast +6.00%, proxy null, prod -3.12%**. The arithmetic score initially marked
it accepted, but `state.yaml` rejected and reverted it; it was never shipped. The discriminator
showed the `ds_store -> ds_load -> WMMA` tail is fixed by a real LDS RAW, so both requested
schedules leave the block intact.

## r16.i1.g48 -- place carried LSE/delta loads first by source order alone

Round 16 killed it at its zero-card gate. The backend moved the b128 loads below the consumer,
changed the partial wait `0x22 -> 0x0`, and raised the static stall model **777 -> 1481**. Source
order is not issue order.

## r16.i2.g49 -- hard scheduling fence after the Q/dO prefetch group

Round 16 measured **-16.25% prod, -18.4% proxy, -6.1% fast** despite fewer issue slots and one
fewer full drain. The hard fence collapsed b128 load-to-use distance; static improvements did not
predict time.

## r11.i4.g36 -- remove two runtime integer divisions from `k_dkdv`'s address chain

Round 18 ledger correction: **this was measured, in round 11, and lost.** Six rounds carried it as
"built with a clean screen but never measured", and round 17 made it step (3) of an unresolved
dialogue; that step is void. `rounds/011/1-opt/act.yaml`: `tflops {fast 18.073, proxy 248.232,
prod 420.232}`, `vs_champion 0.9087`. `rounds/011/1-opt/opt.md` L455-545 has the same-session
back-to-back table: fast 0.9312x, proxy 0.9475x, **prod 0.9087x**. Static changes were all
favourable (body 693->664, non-prefix SALU 71->27, first buffer load @88->@32); only
`s_set_vgpr_msb` 110->142 and VGPR 740->748 moved the wrong way. Another instance of the
round-18 rule: hot-body instruction count does not predict time.

## r17.i2.g53 -- weaken g51's hard scheduling fence with the weakest sufficient mask

Round 18 killed it at its own declared compile gate and then confirmed on card.
`sched_barrier(0x89)` and `sched_barrier(0xC9)` produce **byte-identical ISA**, so the VMEM-write
bit is not a live degree of freedom: weakening the fence at all lets the scheduler do exactly what
the fence exists to prevent. Gate (i) failed -- the b32 consumer's partial wait `0x23/0x22/0x20`
collapsed to `s_wait_loadcnt 0x0`, the dead `r10.i1.g27` signature. Gate (ii) failed -- first b128
moved from index 81 to 261, shortening prefetch cover. Card (prod, same session, six rotated slots,
three replicas each): cur 506.333/507.345/507.510 vs m89 438.938/439.798/440.159 = **0.8676x,
-13.24%**, against a same-code floor of 0.23%. Joins g27 (-14.6%) and g49 (-16.25%) in one pit.
Notable: the losing arm's hot body is **41 instructions shorter** with fewer `v_nop` and fewer
`s_set_vgpr_msb`. The g51 fence is load-bearing and must not be relaxed.

## r18.i1.g54 -- fuse dQ into the KV-outer body with a split workspace

Round 18 killed this on paper, at zero card cost, by pricing TRAFFIC instead of workspace shape.
With KV outer and q inner a dQ partial exists per KV tile -- 256 of them at prod, so a materialised
workspace is ~137 GB and the shape question never arises. As traffic: 16 B of dQ partial per
S-element x 4.295e9 causal S-elements = 68.7 GB written + 68.7 GB read = **137 GB round trip =
62.6 ms at 4.39 TB/s**, against a **10.82 ms** total kernel time and a **3.1 ms** saving from
going 7 -> 5 matrix products. A ~20x loss. Grouping KV tiles to cut the copy count requires holding
the whole group's inner accumulator set, which does not fit (see g55); the query-outer variant is
symmetric and identical. Consistent with h9 and with `r3.i6.g15`. Do not reopen by re-shaping the
workspace -- the workspace is not what is expensive.

## r18.i2.g55 -- reopen the occupancy axis for `k_dkdv`

Round 18 killed this on a register census of `.LBB0_8`: **395 live-in VGPR** of 724 allocated, of
which 384 are the 256-VGPR dK/dV accumulator pair plus the 128-VGPR g21 prefetch tuple. Reaching
<= 512 for 2 waves/SIMD needs >= 212 VGPR removed, which necessarily includes exactly the 128-VGPR
non-rotating prefetch tuple (`r11.i5.g37`) that is the mechanism of g21's +8.3%. Even deleting it
entirely leaves 596 > 512, so `BLOCK_KV` would also have to halve to 16, undoing g06's 1.677x. Two
shipped wins surrendered to buy an unpriced occupancy step. Reopen only via `r18.i5.g58`, which
would free that tuple as a side effect rather than by spending it.

## r18.i4 -- widen `k_dkdv`'s QUERY step 32 -> 64

Killed in round 18 before a slot was spent, on `r10.i2.g28`'s recorded mechanism. The attraction was
real: `k_dkdv` spends 0.597 non-WMMA issue slots per S-element against `k_dq`'s 0.315 (1.9x), and
the 470-cycle b32 stall plus the 92-cycle dscnt stall look like per-iteration fixed costs that halve
per unit work. They are not. The 470 comes from `@228 loadcnt N=35` popping a `buffer_load_b32`
stuck behind the 32-deep b128 prefetch in the in-order LOADcnt FIFO -- **a FIFO-ordering cost that
scales with the batch**, which is precisely what g28 measured when it "doubled the load batch
covered by the drain" and lost 7.8% prod in three sessions. Widening would make the dominant term
worse. With `r3.i5.g14` (KV 32 -> 64, 2.76x slower at 982 VGPR and zero spill) this closes the
tile-widening axis of `k_dkdv` on BOTH of its axes by measurement. No id was allocated.


## r18.i3.g56 -- LSE/delta via `global_load_async_to_lds_b32` (measured, correct, -6.84%)

BUILT and MEASURED in round 18. Correct (`op/validation.py`: dk/dv **52.60 / 52.71 dB** at prod,
determinism x200 bitwise) and every gate the route set for it passed -- no new
`s_wait_loadcnt 0x0` in `.LBB0_8` (2, same as incumbent), `s_wait_asynccnt` `0x2`/`0x0`, spill 0.
**Measured prod 469.56 vs incumbent 504.03 = -6.84%; proxy -8.40%; fast -5.53%. Same sign on all
three shapes, against a 0.36% same-code floor.** Tree left in `rounds/018/op`.

**The mechanism that kills the CANDIDATE is solid and is not about async at all:**

> **A stall attribution locates a stall. It does not price it.** This arm deletes the source
> instruction of the site that `attrib_ss.py` charged with **470 of `.LBB0_8`'s 562 stall
> cycles (83.6%)**. `buffer_load_b32` **4 -> 0**, `s_wait_loadcnt` **7 -> 3**, body
> **678 -> 667** instructions, VGPR **724 -> 718**, `v_wmma` unchanged at 64. Every static
> number moved the right way. The kernel got **6.84% slower**. This is the third time in this
> job -- after `r15.i2.g47` and `r16.i2.g49` -- but the first where the compiler's scheduling
> was NOT constrained, so "the barrier broke the scheduler" does not cover it. **Do not spend
> another arm on a candidate justified only by the static stall budget of `.LBB0_8`.**

**The mechanism that kills the IMPLEMENTATION BUG, which is worth more and kills more than one
candidate:**

> `global_load_async_to_lds_b8/b32/b64/b128` on gfx1250 takes a **per-lane** LDS destination:
> lane *l* writes its own dword at the address its own VGPR holds, with **no implicit lane
> stride**. This is NOT the gfx90a/gfx940 `global_load_lds` contract it resembles (uniform LDS
> base via `m0`, hardware strides by lane) -- losing that stride is the price of not needing
> `m0` and not requiring uniformity. Passing a lane-uniform address makes all 32 lanes collide
> on the same 4 bytes; 31 values are silently dropped. Build #2 did exactly that and produced
> dk/dv at **-56 / -73 dB**. **All four offline screens returned `verdict: pass`, `spill: 0` on
> it** -- `h3` checks allocation and lowering, never the lane->address map. Any per-lane
> addressing change must be gated on `op/validation.py` FIRST, before any benchmark.
> This applies unchanged to `r18.i5.g58` and `r16.i3.g50`.

**What this entry does NOT establish, stated plainly:** *why* it lost. The leading hypothesis --
the async path adds an HBM->LDS->VGPR round trip for 16 B/iteration that previously went
HBM->VGPR and was read out of a register, moving the wait off LOADcnt and onto DScnt, this
kernel's tightest queue (70656 B/wg, 15 `s_wait_dscnt`, `ds_load_tr16_b128` dominant) -- is
**UNPROVEN**. Round 18 deliberately published no static cycle attribution for it, because
`attrib_ss.py` files `global_load_async_to_lds_*` in the LOADcnt queue and reports a fiction
(797 cycles blamed on that op). **`r18.i5.g58` moves 32x more bytes down this same path and must
not be built until the attribution tool has an ASYNCcnt queue** (`route.md` row 5). If the
hypothesis holds, g58 may be the one form of this mechanism that wins, because its data has to
land in LDS anyway and it deletes 80 `ds_store_b128` in the process -- so this entry kills the
LSE/delta form, not the family.

Evidence: `rounds/018/1-opt/opt.md` sections 14-15, `rounds/018/1-opt/raw/val_g56_build3.txt`,
`raw/meas_g56_build3.txt`, `raw/isa_g56_build3_k_dkdv.s`.

## r19.i1.g59 -- WMMA A-operand reuse hint (`OPSEL[2]`) on `k_dkdv`'s dK/dV GEMM

Built, correct (52.52 dB, determinism x200), **null**: prod 505.29 vs incumbent 504.90
(+0.08%) against a same-code slot floor of 0.61%; proxy -0.10%; measured round 19.
It is the arm round 19 shipped into `rounds/019/op`, because it was the round's best arm and
is correct -- it carries no speed and is free to revert.

**Mechanism.** The hint reached the hardware: `matrix_a_reuse` goes 0 -> **56 of 128
`v_wmma`** in the shipped `k_dkdv` ISA (`rounds/019/1-opt/raw/isa_dkdv_{cur,armA}.s`), so this
is not a case of the request being dropped. What it buys is an operand *fetch*, and this body
does not pay for operand fetches: its cost is ~686 non-issuing cycles per `.LBB0_8` iteration
out of ~1809 measured. `s_set_vgpr_msb` 334 -> 329, `v_nop` 82 -> 81, VGPR 724 -> 724 -- the
instruction stream is otherwise untouched, which is exactly why the result is clean.

> **The corpus's first measurement of gfx1250 matrix-operand reuse, and it is zero.** Do not
> spend another arm on `reuseA` / `reuseB` on this op. Setting them is still free and still
> safe *only* when the preceding matrix op used the identical operand (otherwise the result is
> undefined, not slow), so leaving them set costs nothing -- it just buys nothing either.

## r19.i2.g60 -- remove `r7.i1.g21`'s carried prefetch tuple to kill the register-window tax

Built, correct, **dead: -17.27% prod (417.73 vs 504.90), -15.13% proxy**; measured round 19.

**Mechanism.** Every static indicator moved the right way and the kernel got much slower:
VGPR **724 -> 636**, `s_set_vgpr_msb` **334 -> 277**, `v_nop` **82 -> 24 (-71%)**, spill 0,
scratch 0, arithmetic bit-identical. So the ~82 issue slots of register-window and hazard
overhead are **not on the critical path**, and the gfx950 `v_accvgpr_read/write` window-tax
analogue (1.72x, `gqa_d128.md` 6.1) does **not** transfer to gfx1250's flat register file.
What those 128 VGPR actually buy is latency cover, and this body is latency-bound.

Three things this closes:
1. **No future candidate may be justified by "fewer `s_set_vgpr_msb`" or "fewer instructions".**
   This is the second sign reversal on body size (the first: `r17.i2.g53`, a 634-instruction
   body 13.24% slower than a 675-instruction one).
2. **The occupancy axis is closed from both ends.** `r18.i2.g55` showed you cannot reach 512
   VGPR without undoing `g06`; `g60` shows you would not want the registers back even if you
   could. With it, `h12`'s `BLOCK_KV` axis loses its last motive.
3. **`r7.i1.g21` is re-priced from +8.3% (round 7) to ~+21% on today's body** -- the largest
   surviving mechanism in `k_dkdv`. Anything that touches the prefetch tuple, `r16.i3.g50`
   included, is touching the most expensive thing in the kernel and must say so in its risk column.

## The static wait-attribution model (`attrib*.py`), retired as a ranking tool at round 19

Not a candidate, and recorded here because four rounds were spent on its advice.
Round 19 built the missing fifth queue (ASYNCcnt) that round 18 said the model needed
(`rounds/019/1-opt/raw/attrib_ss5.py`). The upgraded model prices `r18.i3.g56`'s `.LBB0_8` at
**1343 cycles against the incumbent's 1685 -- i.e. ~20% faster**, when `g56` had already
measured **6.84% slower**. The fix made the prediction worse, and round 18's "470 of 562 stall
cycles on one site" is a fiction of the model's in-order FIFO assumption.

> **Mechanism: a static model that assumes issue order == completion order cannot price a
> latency-bound body.** `attrib*.py` is demoted to a build gate (scratch / VGPR / spill) and may
> not be cited as a reason to prefer one arm over another. Fourth misprediction
> (`g47`, `g49`, `g56`, and its own upgrade). Consequence: `r18.i5.g58` was closed unbuilt.

## r20.i1.g61 -- split `k_dkdv`'s S/P WMMA accumulator chains 4 -> 2x2 (measured, correct, NULL)

Round 20, arm A. Built from `op/current`, `validation.py` pass (52.52-52.84 dB, bitwise x200),
ISA confirmed the change landed (`v_pk_add_f32` 32 -> 64, chain heads doubled, `v_wmma`
unchanged at 64, 752 VGPR, spill 0). Measured **+0.27% prod inside a 0.29% same-code floor**
and **−0.45% proxy inside a 1.04% floor**: null on every shape.

**Mechanism:** doubling accumulator-chain ILP only helps when the WMMA issue is serialised on
its own accumulator dependency. It is not here. 64 extra independent fp32 ops and 4-way matrix
ILP bought nothing, which is direct evidence that **the matrix pipe is not what this body
waits on** -- consistent with round 19's 30.4% matrix-busy census. The corpus claim it was
built on (`backends/flydsl/attention/techniques.md`, *"cutting one chain into four is a direct
win ... +8.7% on a dK/dV body"*) **does not transfer to a one-wave/SIMD latency-bound body.**

It shipped, because the two-arm rule requires merging when neither arm lost, and the merge's
VGPR came out lower than `g62` alone. **Do not cite it as a win** and do not go wider (4x1).

## The LDS-port story for `k_dkdv`'s non-issuing cycles (falsified in round 20)

Round 19's `facts.md` named one surviving mechanism for the ~686 non-issuing cycles per
`.LBB0_8` iteration: **LDS port pressure**, on the census that `k_dkdv` holds 1.25 LDS ops per
WMMA against `k_dq`'s 0.33, with 64 of the 80 being the Q/dO store/transpose-load round trip.
Round 20 built three throwaway probes (no `g` allocated), each ISA-verified to have landed
before any card time, and measured them on an idle device against the same-session incumbent:

| probe | what it removes | prod |
|---|---|---|
| P1 | the Q/dO LDS round trip outright (reads garbage, same instruction mix otherwise) | **−6.98%** |
| P2 | g50's traffic structure carried as in-body global loads instead | **−18.37%** |
| P3 | **all 80** LDS ops; `v_wmma` count unchanged | **−8.19%** |

**Mechanism:** if LDS ports were the constraint, deleting 80 LDS ops would have to be faster.
It is 8.19% slower. Those LDS ops are not the cost, they are **latency cover** -- independent
work standing between a load and its use. This also kills `r16.i3.g50` (pool, marked DEAD) and
any candidate whose argument is "fewer LDS ops in the hot body".

It is the sixth deletion in a row to lose (`g47`, `g49`, `g56` −6.84%, `g60` −17.27%, P1, P3).
**The sign of the lever on this kernel is known: add independent work and lengthen load-to-use
distance. Do not remove anything.**

## r21.i1.g63 -- deepen `k_dq`'s cross-iteration prefetch 1 -> 2 (measured, correct, -19.33%)

Round 21's candidate. The `r20.i2.g62` edit -- which was worth **+1.65% prod on `k_dkdv`** --
applied unchanged to the other kernel. It lost by a hundred times the noise floor.

| shape | incumbent (same session) | g63 | delta |
|---|---|---|---|
| prod | 509.71 TF/s | **411.21** | **-19.33%** |
| proxy | 431.87 | 363.02 | -15.94% |
| fast | 53.19 | 50.49 | (inside an 8.6% floor, not readable) |

Noise floor 0.19% prod, from two physically-separate rebuilt incumbent copies. Correct:
52.52 dB, bitwise-deterministic x200, 15/15 UT shapes. `rounds/021/1-opt/opt.md` §9.

**Every gate passed and it still lost.** The `pool.md` precondition held (`k_dq`'s `.LBB0_2`
carried four `s_wait_loadcnt`, including a full `0x0` drain at 93% of the body). The intended
ISA delta landed exactly: `s_wait_loadcnt` 4 -> 2, both hoisted to the top, the 93% drain gone.
VGPR 960 -> 992 with **spill 0, scratch 0**, 32 under the 1-wave/SIMD cap of 1024.

**Mechanism -- it is the SPREAD, not the registers.** The control arm `r21.i4.g66` (pool)
bought the same spread of `k_dq`'s 32 `buffer_load_b128` (idx 51-100 -> 37-712) via
`sched_group_barrier` alone, at **VGPR 960 -> 960, spill 0, body +2.7%** -- and lost **more**
(-21.93%). Register pressure, code size and spill are all excluded. What both arms did was
break up `k_dq`'s tight ~50-instruction prefetch clump, and that clump is load-bearing.

Consistent with the body shapes: `k_dq` is 96 `v_wmma` against 16 `ds_store_b128` + 16
`ds_load_tr16` -- compute-dense, memory-light, its loads already covered by the WMMA wall, so
spreading them only lengthens the dependence chain into the wall. `k_dkdv` is 64 `v_wmma`
against 40 + 40 -- memory-heavy, and there the same spread buys real cover.

**The signed fact for later rounds: `k_dq` and `k_dkdv` have OPPOSITE SIGNS on the prefetch-
spread lever.** "Add independent work and lengthen load-to-use distance" (facts.md, round 20)
is a `k_dkdv` result and does not transfer to `k_dq`. Follow-ups `r22.i1.g67` (tighten `k_dq`)
and `r22.i2.g68` (re-spread `k_dkdv`, the falsifier) are in the pool.

**And a retired argument:** *"VGPR has headroom and spill is 0"* may no longer be offered as a
reason an arm is worth card time. It was the whole case for g63 and it was worth nothing.

## r21.i3.g65 -- CLOSED on the offline screen, round 21. Zero card time.

Hoist `k_dq`'s 16 `ds_store_b128` out of the kt/dt loop and issue them as one group ahead
of the 64 S/P WMMAs, to widen the LDS write-to-read distance.

**The backend already does it, and the premise was wrong.** After register-renaming
normalisation the `k_dq` hot body is instruction-for-instruction identical to the incumbent
(`k_dkdv` and `k_delta_bshd` are byte-identical); the stores were already clustered at
idx 4-108 of 742 and the `ds_load_tr16` at 110-249, whose consumers (the dQ WMMAs) sit at
77-92% of the body -- **300-500 instructions of distance already**. `k_dq`'s LDS path is
not the exposure. Second time this job has been taught that **source order is not issue
order** (`r16.i1.g48` was the first).

## r21.i4.g66 -- DEAD, measured on card round 21. It is the load-bearing negative result.

`sched_group_barrier(VMEM,1)/(MFMA,3)` x32 in `k_dq._body`, spreading the 32 prefetch
`buffer_load_b128` through the 96-WMMA burst. **prod -21.93%** (397.94 TF/s), proxy -19.50%.
Correct. `spill 0`, **VGPR 960 -> 960 (unchanged)**, body 742 -> 762 (+2.7%).

**This arm exists to isolate why `g63` lost, and it succeeded.** The only ISA consequence
`g63` and `g66` share is that the 32 loads stop being one tight 50-instruction clump
(incumbent idx 51-100; `g63` 52-938; `g66` 37-712). `g66` buys that spread at **zero
register cost and near-zero size cost, and loses more than `g63` does.**

> The loss is not spill, not VGPR, not code size. **It is the spread itself.
> `k_dq`'s tight prefetch clump is load-bearing.**


## r22.i2.g68 -- DEAD, measured on card round 22. It killed round 21's model, not just itself.

Spread `k_dkdv`'s 32 prefetch loads across the body without deepening --
`sched_group_barrier(VMEM,1)/(MFMA,2)` x32 in `_body`. Correct (UT 15/15, `validation.py`
pass, determinism x200 bitwise pass, 52.30-52.98 dB), `spill 0`, VGPR unchanged from the
incumbent, every offline gate passed. **prod 361.73 = -30.0%, proxy 327.77 = -26.1%,
fast 51.22 = -7.4%**, against same-session floors of 0.40% / 2.08% / 12.80% and champions
re-measured beside it.

**Mechanism -- the cover model.** The binding quantity is the **worst-case load-to-use cover
distance**: instructions between a prefetch load's issue and the next *full*
`s_wait_loadcnt 0x0`. Incumbent `k_dkdv` issues its last prefetch load at idx 188 and drains
at idx 13 of the next iteration: **~603 instructions of cover**. `g68` pushes the last load to
idx 789 against a drain at idx 25: **~41 instructions**. A 15x collapse. Spreading loads
across the body does not hide them better -- it moves the *last* one next to the drain.

This was **written down and the sign and magnitude predicted before any card time**, then
confirmed. It is the first model in this job to do that.

**What it closes beyond itself.** Round 21 recorded that `k_dkdv` and `k_dq` have **opposite
signs** on this lever, with WMMA/LDS body shape as the discriminant. `g68` is that model's own
prediction applied to the kernel it says should gain. It lost 30%. **Spreading the load clump
loses on both kernels; there is no sign flip.** Do not reason from "this kernel is
compute-dense / memory-heavy" again.

**The filter this leaves for every future candidate:** *any edit that moves the last prefetch
load later in the body loses.* Five for five -- `g28` -7.8% (x3 sessions), `g63` -19.33%,
`g66` -21.93%, `g68` -30.0%, and `g50`'s probes negative. Before building anything on the
prefetch axis, answer: **is this a re-spread in disguise?**

**It also reprices the one positive.** `r20.i2.g62`'s +1.65% sits on a measured null-control
floor of **1.57% (n=5)**. Its ceiling is structural: `k_dkdv`'s body opens with a full drain
at idx 13 followed by **128 `v_mov_b64`** -- the FlyDSL non-rotating-carried-register block
copy (`r11.i5.g37`) -- so depth 2 buys ~*one* iteration of cover and pays 128 copies per
iteration for it. The prefetch depth/placement axis is exhausted in both directions on both
kernels.

## `sched_barrier(0)` is a scheduling BOUNDARY, not a clamp (established in `r22.i1.g67` form 1, round 22)

Fencing a load group on both sides to pin it as one tight clump **does the opposite**. A
`rocdl.sched_barrier(0)` before and after `k_dq`'s `_ldkv` group moved the
`buffer_load_b128` idx span from 51-100 (49 wide) to **29-92 (63 wide)**, grew the body
742 -> 816, added **five new mid-body partial `s_wait_loadcnt` at 11-27% depth**, and took
`v_nop` 28 -> 49. Killed on the offline screen at zero card time; its declared gate required
*narrower*.

**Mechanism:** the fence gives the scheduler two smaller regions instead of one, and it fills
each by pulling loads earlier. **Asking this backend to tighten a clump with barriers reliably
loosens it.** There is no expressible form of "tighten the prefetch" through this intrinsic.

⚠ Together with `g68`, `g66`, `g53`, `g49` and `g47`, the
`sched_barrier`/`sched_group_barrier` family is now **0-5 on card** on this op
(-3.12% / -16.25% / -13.24% / -21.93% / -30.0%). Round 22 declined to build a sixth.

## The power-bound hypothesis for this op (untestable on this box, round 22)

> ⚠ **RETRACTED 2026-09-24 (end of day).** The 60-sample trace behind this entry was taken
> with **no benchmark running** — `pw.sh:12`'s `--arms cur_a` made `benchmark.py` exit 2 into
> `/dev/null` while the script's `wait` reported rc=0. The 264-sample trace across the real
> scored block has 1100 MHz on 67 samples and <= 1030 MHz on 185. **1100 is the idle clock.**
> The power-bound hypothesis is NOT closed; it is unmeasured. Do not cite this entry as a kill.


Not a mechanism -- a **symptom with no instrument behind it**, recorded so nobody spends the
minutes again. The corpus records gfx1250 pinned at a 2500 W cap; `benchmark.py`'s sclk
witness reads 1100 at `fast` but 1051-1067 at prod/proxy, which looks like throttling.

Both readings are unusable here. `Current Socket Graphics Package Power (W)` reads **855.0 at
idle and a median of 853 across a full prod benchmark** (60 samples) -- under 0.5% movement,
i.e. a static register, not a measurement. And **sclk is flat at 1100 MHz for 60/60 samples
under sustained load**; the 1051-1067 figures are sampling-instant transients (the scored
block's 1 Hz trace: 67 samples at 1100 against 12 in 1045-1055).

**Consequence, and it is the useful half:** *"the card was throttling"* is removed as an
available explanation for any measurement in this job, past or future.

## No cluster load with a register destination in this FlyDSL (census, round 22)

Closed before any code was written. FlyDSL exposes cluster multicast **only** as
`cluster_load_async_to_lds_b{8,32,64,128}` -- global to LDS. There is no register-destination
form, so the attractive drop-in (swap `k_dq`'s 32 `buffer_load_b128` for multicast loads to
registers, no LDS, no structural change) **does not exist** and should not be proposed again.
Anything in this family must route through LDS, which is the path campaign correction 6 and
`r5.i2.g18` closed for `k_dkdv` and which `r16.i3.g50`'s `P1` priced at -6.98%.

TDM itself *is* expressible (`flydsl/expr/rocdl/tdm_ops.py`: `make_tensor_descriptor_2d`,
`tensor_load_2d`, `tensor_store_2d`, `tensor_wait`, gather, `l2_prefetch_tile`). That half
stays open as `r22.i3.g69` in the pool, gated on a free reading.
