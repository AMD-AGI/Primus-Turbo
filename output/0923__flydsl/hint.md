# Operator hints -- gfx1250-flydsl-attn-bwd

Written by hand on 2026-09-21, before resuming the job that was stopped on 2026-09-17.
This is the only file in the job a person writes; the framework applies pending entries at
the head of a round, so direction can be changed here **without stopping the job**.

| id | type | title | status |
| --- | --- | --- | --- |
| h1 | must note | Evidence armAB's correctness BEFORE reporting any speed for it | open |
| h2 | must standing | Boundary safety comes from explicit predicates, never from buffer-descriptor semantics | open |
| h3 | must standing | Screen a candidate offline before it takes a GPU slot; spill > 0 is a kill | open |
| h4 | advise standing | rocprofv3 is a dead end on this op; do not spend rounds on it | open |
| h5 | advise standing | A flat `fast` number is not evidence against a candidate | open |
| h6 | advise note | The spec's 5.8 TFLOP/s and ~93x are stale; the real gap is 12.6x | open |
| h7 | must standing | prod is the only shape that ranks a candidate; fast and proxy are sentinels | open |
| h8 | must note | The bottleneck is HBM traffic, and facts.md states it wrong by 86x | open |
| h9 | must standing | Do not ship g15 (fuse dQ) until BLOCK_KV is raised; at 32 it is net negative | open |
| h10 | advise note | Ledger bookkeeping: highest g is g18, and g03 belongs in dead_ends | open |
| h11 | must note | "Requires a deep round" is never a valid condition here; re-cut the big items to fast size | open |
| h12 | must note | Round 8 is g04 on k_dkdv ONLY, and a naive 4-wave build saves zero bytes | open |
| h13 | must standing | Round 8 deleted the barriers g04 needs; re-add them FIRST or the 4-wave build is silently racy | open |
| h14 | must note | Round 9 is the carried-state merge: ~200 VGPR are an allocation failure, not a cost | open |
| h15 | must standing | Rules for every script a round writes: cached_forward, line buffering, dmesg between blocks | open |
| h16 | must note | The bound is the ISSUE ROOF, not bandwidth. g15 and g18 close; g04 demoted; round 10 is g27 | open |
| h17 | must standing | Read the corpus before you build. h16 named a documented dead end and it cost a round | open |
| h18 | must note | The bar cannot pass our own gate. Round 12 buys an instrument, not a candidate | open |

---

## h1 -- Evidence armAB's correctness BEFORE reporting any speed for it

Round 1 built `armA` (causal tile skipping) and `armB` (32-deep contraction), measured
both, and was killed thirteen seconds after the benchmark returned, before it could write
`act.yaml`. It had also patched a merged arm, `armAB`, which was **never timed**. All of it
survives under `rounds/001/_scratch/`.

**But `benchAB.json` carries timing only.** Every row is latency / tflops / sclk. There is
no SQNR field, no determinism field, and `grep` finds no dB figure anywhere under
`_scratch/`. So `58.272 ms / 1.64x` for armA has **no correctness evidence behind it at
all**, and armAB has neither correctness nor timing.

Before any number from these arms is reported or used to rank anything:

1. Run the full `validation.py` against `rounds/001/_scratch/armAB` -- four-tensor SQNR
   and the 200-run bitwise determinism gate.
2. Additionally do an **elementwise bitwise diff of armA against baseline**. This is not
   what `validation.py` checks: its determinism gate is "the same implementation is
   bitwise identical across 200 consecutive runs", not "armA equals baseline". The claim
   that causal skipping is bit-identical rather than an approximation has to be cashed,
   and only this cashes it.
3. Confirm the NaN prefill actually fires before SQNR, by deliberately dropping one
   near-diagonal tile and checking that the isfinite coverage check catches it.

Why this ordering is not bureaucracy. The correctness gate is SQNR against `op/eager/`,
and the baseline measures ~52.5 dB against a 50 dB floor -- **2.5 dB of headroom**. A
causal skip that is off by one at the diagonal tile boundary drops a small part of a real
contribution. It gets **faster** and it **still clears 50 dB**. The typical failure mode
is "some tile was never written at all", and if an output buffer is zero-initialised that
is numerically indistinguishable from "correctly computed a value near zero" -- which,
under causal masking, is exactly what the true dk/dv look like in the skipped region.

The static screen has already cleared armAB as **safe to launch**: zero spill, zero
private segment, LDS 18432 within budget. So the risk here is a wrong answer, not a wedge.

## h2 -- Boundary safety comes from explicit predicates, never from buffer-descriptor semantics

Established on 2026-09-21 by reading the lowered IR, with no kernel launched. All 18
`make.buffer.rsrc` sites in the three kernels take flags `159744 = 0x00027000` -- the CDNA
form, bit 24 clear, `OOB_SELECT = 0` -- on an RDNA-family gfx12 part, because
`is_rdna_arch("gfx1250")` returns False. `num_records` is a flat 1 GiB constant, so the
hardware bound is not tracking per-tensor extents either way.

**Do not "fix" the descriptor as part of an optimisation round.** Its RDNA form is
documented as `OOB_SELECT = 2` meaning *no bounds checking*, i.e. the change may well
remove a backstop rather than add one -- and that documentation sits beside a transposed
constant, so it is not authoritative. The question is open and is being settled separately.

What this means for a round: causal tile skipping, tail handling and any varlen work must
clamp and predicate **in the kernel**. Then the kernel is correct whichever way the
descriptor question resolves, and nothing silently depends on it.

## h3 -- Screen a candidate offline before it takes a GPU slot; spill > 0 is a kill

`Primus-Turbo/output/0921__flydsl/bin/compile_only_driver.py` compiles a candidate to
gfx1250 ISA in a container with **no** `/dev/kfd` and **no** `/dev/dri`, and reports vgpr,
spill, LDS, WMMA count and instruction mix. It costs no card time and cannot contend.

`private_segment_fixed_size > 0` or any `scratch_` op is a **kill, not a cost**. On this
part a build that spills does not merely run slow -- it hangs after its first launch, and
recovering that needs a person to power-cycle the machine. The gfx950 corpus reaches the
same rule from the other side: *treat non-zero scratch as a refusal rather than a cost.*

Use the WMMA count to price **density** changes only. It is blind to trip-count changes:
armA has a WMMA count identical to baseline and measured 1.64x. Anything that changes how
many iterations run has to be measured.

## h4 -- rocprofv3 is a dead end on this op; do not spend rounds on it

Under rocprofv3 the benchmark faults the GPU with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`;
the same command without it runs clean. With the faulting call removed it exits 0 and
writes no files. Of 51 gfx1250 counters, the 13 that return non-zero are wall cycles, wave
count and I-cache only; `SQ_VALU_WMMA_FLOP_*` reads exactly 0 on a kernel proven to issue
4.096M WMMAs. PC sampling is categorically banned: three attempts, three GPU faults, one
needing a reboot.

There is no counter-based route to a bound verdict here. The instruments that do work are
the static screen (h3) and a careful A/B with a clock witness.

**Therefore this job must never be scheduled a DEEP round.** Added 2026-09-22. A deep round
runs modules `("profiling", "plan", "act", "reflect")` and the profiling module is built
entirely on the banned tool: `prompts/01_select.md` instructs the agent to run
`rocprofv3 --stats --kernel-trace`, `02_counters.md` to collect per-kernel counters with
`rocprofv3`, and `03_metrics.md` to run `rocprof-compute profile` once per kernel. Enabling
deep rounds here would point an agent that runs with `bypassPermissions` straight at the tool
that has faulted this card three times, once needing a power cycle.

The schedule is currently safe by accident rather than by design -- `fast_rounds: 12` with
`max_rounds: 12` means no deep round can ever be reached. **Do not "fix" that with
`op-evolve tune --fast-rounds` or `--fast-per-deep`.** If a profile is ever genuinely needed,
the substitute is the static ISA route (h3 and h8), which has already priced this kernel to
within 4% and which found the traffic model that five rounds of counters-free guessing missed.

## h5 -- A flat `fast` number is not evidence against a candidate

`k_dkdv`'s grid is `(Skv/16, Hkv, B)`, so the `fast` shape launches **128 workgroups onto
a 256-CU part** -- more than half the machine idle. proxy launches 2048, prod 16384. That
is why fast reads 3.0 TF/s against prod's 57.2: it is a different bottleneck, not a
smaller version of the same one. armA is 1.19x at fast and 1.64x at prod.

Read the shapes across a row, never a throughput column down the page.

## h6 -- The spec's 5.8 TFLOP/s and ~93x are stale; the real gap is 12.6x

`op.baseline.reported_tflops: 5.8` in the resolved spec, and the `~93x` in `LAUNCHED.md`
and `HINTS.md`, come from a single-head single-batch timing whose grid was 256 workgroups
-- under-parallel by construction. Op Setup measured the truth at prod: baseline
**96.05 ms / 57.24 TF/s**, beat **7.61 ms / 722.2 TF/s**, ratio 0.080, a **12.6x** gap.

Nothing scores off those fields, so this is not a correctness problem. It is a framing
one: the job is harder than its own description implies, and a round that plans against
93x will mis-size what it attempts.

## h7 -- prod is the only shape that ranks a candidate; fast and proxy are sentinels

Written 2026-09-22 after round 5. The acceptance rule takes an ARITHMETIC MEAN of the
per-shape gains, so all three shapes carry equal weight. `fast` is 0.35 ms and dominated by
launch overhead; `prod` is 16.4 ms and is the shape this job exists for.

Round 5 was accepted at gain 1.1539 while prod moved 1.0134x. The 1.15 came almost entirely
from `fast` gaining 1.36x on g17, which routed impl.py's launches through `flyc.compile()`
and cut host time 0.2688 -> 0.0313 ms. That is a real improvement to `fast` and worth having.
It is worth **nothing** at prod: the per-kernel split shows 16.26 ms of the 16.39 ms total is
inside the two kernels, so only 0.8% of prod is launch overhead at all.

The rule is not being changed -- changing it mid-campaign would make the score column mean
two different things either side of the change. Instead: **rank candidates by their prod
number, and say so explicitly in the Route table.** A round whose prod gain is inside the
noise has not advanced this job, whatever the mean says. Report the per-shape gains
separately and lead with prod.

Do not read this as "ignore fast". A sudden fast REGRESSION is a cheap early warning that
per-launch cost has exploded, and h5 still stands: a flat fast is not evidence against a
candidate.

## h8 -- The bottleneck is HBM traffic, and facts.md states it wrong by 86x

**`facts.md:385` ("measured 6.46 TB/s HBM roof (0.5%)") and `facts.md:718` ("HBM bandwidth
(0.5% of roof at prod)") are wrong, and have misdirected five rounds.** Both divide by the
tensors' COMPULSORY bytes (~1.2 GB). The kernels do not request 1.2 GB. They request
**~107 GB**, because `k_dkdv` re-reads Q and dO about 128.5 times (Skv/BLOCK_KV = 256 kv
blocks, halved by causality) and `k_dq` re-reads K and V about 64.5 times.

Counted from the shipped ISA and the source trip counts, at prod:

```
k_dkdv  32x buffer_load_b128 + 4x buffer_load_b32 per iter = 16,896 B
        iters = 4 * sum(256-bid, bid=0..255) * 32 = 4,210,688      ->  71.1 GB
k_dq    32x buffer_load_b128 per iter             = 16,384 B
        iters = sum(2*bid+2, bid=0..127) * 128    = 2,113,536      ->  34.6 GB
        + prologues + the three output writes                      -> ~107 GB
107 GB / 6.46 TB/s = 16.6 ms      measured: 16.39 ms      -> at the roof
```

The WMMA issue bound is **7.72 ms** (472.4M WMMA issued against the 335.5M the algorithm
needs; that 1.408x is exactly the 7-GEMM-vs-5-GEMM cost of splitting dq from dkdv).
**Compute is not binding. Bandwidth is.** aiter requests ~19-37 GB and is compute-bound at
71% of the compute roof; we request 107 GB and are memory-bound at ~100% of the HBM roof.
That, and not instruction quality, is the 2.1x.

This retrodicts two results the ledger recorded as puzzles:

* **g12's null.** It removed 16 of 48 `buffer_load_b128` from k_dq's loop (-33% of requests)
  and measured -0.34%. Those 16 re-read the same K rows immediately and were L1 hits: they
  cost zero HBM bytes. Counting HBM bytes predicts the null. Counting instructions, or
  "traffic identities", does not -- which is why the ledger has now recorded three
  "traffic identity fails to predict time" surprises. They were the same surprise.
* **round 4's +50%.** It took k_dkdv from 55% of the HBM roof to 108% without changing the
  bytes at all: it removed stalls and ran the kernel into the memory wall. That is exactly
  why round 5 could then only find +1.3%.

**What this round must do with it:**

1. Rank every candidate by the **prod requested bytes** it changes -- not by instruction
   count, not by WMMA count. It is pure arithmetic and costs no GPU:
   `(loop b128 count * 512 + b32 count * 128) * trip count`. Add it as a column to the
   offline screen (h3) and sort on it.
2. A candidate that does not reduce requested bytes **cannot** move prod, however clean its
   static screen looks. Screen it out before it takes a GPU slot. Had this column existed,
   g12 (null) and g14 (2.76x slower) would both have been rejected without a launch.
3. Do not take this hint on faith: the arithmetic above is reproducible from
   `rounds/005/_scratch/dump_armK/*/21_final_isa.s` and `rounds/005/op/kernels.py` in a few
   minutes with no GPU. Check it, and correct facts.md as part of this round -- the wrong
   0.5% figure must not survive another round.

## h9 -- Do not ship g15 (fuse dQ) until BLOCK_KV is raised; at 32 it is net negative

`r3.i6.g15` (fuse dQ into the KV-outer body and delete k_dq) is correctly identified as the
structural item, and aiter does exactly this -- its main kernel is a single fused kernel with
all five GEMMs, confirmed by disassembly.

But aiter reaches dQ through `buffer_atomic_add_f32`, and **atomics are barred here** by the
200-run bitwise determinism gate. The deterministic alternative is split-K partials written
to a workspace and folded by a separate reduce kernel, and its cost scales with the number of
kv bands:

```
BLOCK_KV = 32   ->  (8192/32)/2  = 128 bands  ->  128 x 268 MB written + read  ~= 68.8 GB
                    against the 34.6 GB of k_dq it deletes          ->  NET LOSS
BLOCK_KV = 128  ->  (8192/128)/2 =  32 bands  ->  ~8.6 GB + ~8.6 GB ~= 17.2 GB
                    against the same 34.6 GB                        ->  net saving ~17 GB
```

So the order matters and route.md currently has it backwards. **Raise BLOCK_KV first, then
fuse.** Raising BLOCK_KV on a single wave is impossible -- dK+dV need `8 * BLOCK_KV` dwords
per lane, so 128 would need 1024 dwords/lane -- but a **4-wave workgroup split along the kv
axis** keeps each wave at 32 kv rows and 256 dwords/lane, unchanged, while the workgroup
covers 128 kv rows and the four waves share one LDS copy of Q/dO. That is aiter's own
configuration (block=128 = 4 x wave32, ts=128, 80 WMMA per wave x 4 = 320 = 5 GEMM x 64), and
it is independently what the gfx950 branch converged on (`flat_wg=256`, `_fuse_blockkv_for`
returns 128 at D=128).

Note this also retires the gfx950 dead end recorded against g04 ("widening the KV block was
3-4x slower, occupancy collapse"): that was widening at a FIXED wave count, which explodes
per-wave registers. The 4-wave form does not raise per-wave accumulators at all, which is the
distinction `dead_ends.md`'s own g14 closeout draws.

## h10 -- Ledger bookkeeping: highest g is g18, and g03 belongs in dead_ends

`pool.md` line 6 still reads `Highest g allocated so far: **g16**` while g17 and g18 are both
allocated and described further down the same file. `facts.md` records that a previous job
lost two valid findings to a FastError for exactly this class of bookkeeping failure, so fix
the line as part of this round.

Also `r1.i3.g03` (AITER causal tile pairing) is still listed OPEN although `facts.md:320-331`
priced it at ~3% here rather than AITER's ~18% -- this kernel is already key-outer with grid.x
as the fastest-varying dimension, so the hardware already dispatches longest-first. Close it
into dead_ends.md. Leaving it OPEN distorts the pool ordering.

One caveat that goes with it: if BLOCK_KV rises to 128 on a 4-wave workgroup, the grid drops
to about 2048 workgroups and load imbalance becomes a first-order term again. g03 should then
be **re-priced**, not resurrected from its old number -- aiter halves its grid for precisely
this reason.

## h11 -- "Requires a deep round" is never a valid condition here; re-cut the big items

Round 6 found a deadlock and was right to refuse to work around it silently. Its own words:
the pool's deep-round-born open entries (`r3.i6.g15` fuse dQ, `r5.i2.g18` cluster multicast)
carry conditions that say *wait for a deep round*, while h4 establishes that this job must
never be scheduled one. Those conditions are permanently unreachable, so those entries can
never be taken, and the pool has been ordering itself around two items that cannot move.

That deadlock is the operator's fault, not the round's, and this hint clears it.

**Rule: a candidate's condition may not name a deep round.** If an item genuinely needs
something a deep round would have supplied, name that thing directly -- and if the thing is
counters, h4 already says it does not exist here and the item must be re-grounded on the
static route or closed. Rewrite `r3.i6.g15` and `r5.i2.g18` accordingly as part of this
round's ledger work.

**The item to take next is the 4-wave restructure of `k_dkdv`, and it is fast-round sized.**
It was never a deep-round item; it was mis-filed as one. It is a contained change to a single
kernel, comparable in size to what round 4 shipped (g09 deleted a staging loop and g16 changed
the LDS row stride, together +50% at prod):

```
k_dkdv today          block=(32,1,1), one wave, BLOCK_KV=32, grid.x=Skv/32
k_dkdv proposed       block=(128,1,1), four waves, BLOCK_KV=128, grid.x=Skv/128
                      the kv axis splits across the four waves: each wave still owns
                      32 kv rows, so dK+dV accumulators stay at 8*32 = 256 dwords/lane,
                      UNCHANGED -- this is the whole point
                      Q and dO are staged ONCE into shared LDS and read by all four waves,
                      which is where the traffic saving comes from
                      barriers appear around the LDS staging (there are none today: round 6
                      measured s_barrier == 0 in both kernels, the compiler having removed
                      them for a single wave)
```

Why this is the item, in the terms h8 established: it is the only open candidate that
**reduces prod requested bytes**, and it reduces them fourfold on the dominant kernel.
`k_dkdv` re-reads Q and dO once per kv block; going from 32 to 128 kv rows per workgroup cuts
that re-read count by 4, i.e. 71.1 GB -> ~17.8 GB. With `k_dq` unchanged the total falls from
~107 GB to ~54 GB, and the HBM bound from 16.6 ms to ~8.3 ms, at which point the WMMA bound
(7.72 ms) becomes binding instead and further byte cuts stop paying. Predicted prod is
roughly 500-580 TF/s. **Do not take that prediction on faith -- it is the one this round
should try to falsify.**

Three independent sources agree on this configuration, which is why it outranks everything
else in the pool:
1. aiter's gfx1250 kernel -- NOT from disassembly, see the correction in h18:
   `block=128` (4 x wave32), ts=128, 80 WMMA
   per wave x 4 = 320 = 5 GEMM x 64, LDS 327680 B, zero `global_load` (all TDM).
2. The colleague's gfx950 branch converged on `flat_wg=256` (NUM_WAVES=4) with
   `_fuse_blockkv_for` returning 128 at D=128.
3. Our own r2 measured the slope once already: BLOCK_KV 16 -> 32 (`armB_clamp`) was 1.677x.

**And it explains `g14`'s death rather than contradicting it.** g14 raised BLOCK_KV 32 -> 64
on a SINGLE wave, which takes dK+dV accumulators to 8*64 = 512 dwords/lane and wrecks the
register budget -- it measured 2.76x slower. The four-wave form raises the workgroup's kv
coverage without raising any wave's accumulator count. `dead_ends.md`'s own g14 closeout draws
exactly this distinction. So g14 is not evidence against this; it is evidence that the single-
wave route to a wider block is closed.

**Order within the round.** Screen it offline first (h3) and report VGPR, spill, LDS and the
prod requested-byte column (h8) for the 4-wave build before it takes a GPU slot; `spill > 0`
is still a kill. The LDS budget is the thing to check rather than assume: Q/dO staging is
shared (~17.4 KB) but the per-wave P/dS tiles multiply by four.

**The cheap complement, if a second arm fits.** Round 6's `armS` found that dispatch order
decides whether workgroups sharing Q/dO run concurrently, and cited the corpus measuring
causal backward at 7.1x the HBM of non-causal with L2 hit rate 8.39% vs 48.27% for exactly
this reason. The stronger form of that idea is to make the 256 co-resident workgroups of one
(hkv, b) group descend the q axis **in step** rather than each starting at its own `bid` --
today `qt = qp_start + ...` maximally de-phases them, so their combined Q/dO footprint is
~16 MB against a 4 MB L2. Reversing the scan so they start together is close to a one-line
change. It is not bitwise-identical to the incumbent (fp32 accumulation order changes), so
gate it on determinism + SQNR and say so.

## h12 -- Round 8 is g04 on k_dkdv ONLY, and a naive 4-wave build saves zero bytes

This round has one job: settle `r1.i4.g04` on `k_dkdv`. **Do not touch `k_dq` this round**
(its `block=(32,1,1)` stays; the two kernels' block sizes are independent). Do not bundle
other ideas. **A round that ships nothing because g04 was honestly falsified is a complete
delivery** -- say so in act.yaml and close the pool entry. What is not acceptable is a
verdict that rests on a build which never implemented the mechanism, which is the specific
trap described below.

Everything here was established offline on 2026-09-22, no GPU spent. Check it rather than
trusting it; where it is wrong, the round's own measurement wins.

### The trap: four waves doing what one wave does today saves NOTHING

`k_dkdv` today reads Q/dO from global **straight into registers** and feeds WMMA from them --
that is exactly what `r2.i2.g09` bought (+11%): the staging round-trip was deleted
(`kernels.py:293-300`). If four waves each keep calling `_ldqd` as now, then the workgroup
count drops 4x and each workgroup reads **four** copies of Q/dO: **requested bytes stay at
71.1 GB, unchanged.** Such a build compiles, validates, runs, and measures flat -- and would
close the only remaining structural item on a false negative.

To actually get 71.1 -> 18.0 GB the global load must be **quartered and shared**:

```
each wave loads 1/4 of the Q/dO tile from global   (8 of the 32 rows)
  -> ds_write into the SHARED lds_q / lds_do
  -> rocdl.s_waitcnt(WAIT_LGKM)      # gpu.barrier() is NOT a fence; retire the ds_writes
  -> fx.barrier()                    # the THIRD barrier, see below
  -> every wave reads back the FULL fragment from LDS to feed GEMM1
```

**This is `g09` being partly reversed, deliberately, at 1/4 the global cost**, and it has a
known price: GEMM1's B operands (`qfr`/`dfr`, `kernels.py:319-322`) stop coming from the
load registers and start coming from LDS via `tr()`, which adds a dependency the S/P GEMM
currently overtakes. `pool.md:315` warns about exactly this and the corpus measured it at
**9.0-14.7% slower** on gfx950. So the trade is: pay ~10% of dependency-chain cost, collect
4x of traffic. Expected to be strongly net positive -- but measure both, and if the arm
loses, that number is the finding.

Per-iteration static signature of a CORRECT build: `buffer_load_b128` **32 -> 8** in the loop
body, `ds_load_tr16_b128` roughly **80 -> 112**.

### Mechanics, verified against flydsl 0.3.2 at /home/lihuzhan/.local/flydsl032

Multi-wave is fully expressible and **this job already ships one**: `k_delta_bshd` is
`known_block_size=[256,1,1]` with `block=(256,1,1)` -- eight waves, running on gfx1250 today
and passing validation (`kernels.py:103,146`). aiter's gfx1250 forward is also 8 waves.

```
@flyc.kernel(known_block_size=[128, 1, 1])        # kernels.py:150, today [32,1,1]
... .launch(grid=(nhkv, nblk, nb), block=(128,1,1), stream=stream)   # kernels.py:460
```
Both sites must change together -- `compiler/kernel_function.py:304-317` raises on a
mismatch, which is a useful guard rather than a hazard.

```
wave = fx.Int32(rocdl.wave_id())   # flydsl/expr/rocdl/__init__.py:510-519, reads TTMP8, SGPR
lane = fx.lane_id()                # expr/gpu.py:67-69
```
Use `rocdl.wave_id()`, **not** `thread_idx.x // 32`: the former is wave-uniform and stays in
SGPRs, so `kv0` and the buffer offsets stay scalar; the latter is VGPR-derived and makes every
downstream address divergent.

`kernels.py:160` currently does `lane = fx.Int32(fx.thread_idx.x)`. At block=128 that gives
`half = lane // 16` in [0,8) instead of [0,2) at `:170-171` -- **wrong answers, no error**.
`kv0 = bid * BLOCK_KV` at `:172` becomes `kv0 = bid * WG_KV + wave * BLOCK_KV`, `WG_KV = 128`.
`impl.py:101-102,127` must switch its assert and grid from `BLOCK_KV` to `WG_KV`
(prod skv=8192 divides by 128).

**Barriers need no new source.** `fx.barrier()` is already called at `kernels.py:359` and
`:391`; round 6 measured `s_barrier == 0` only because `known_block_size=[32,1,1]` tells LLVM
there is one wave, so they are legally elided. At [128,1,1] they emit. The third barrier (the
staging one above) is the only new call. **`gpu.barrier()` is not a fence** -- the gfx950
reference precedes each with an explicit `rocdl.s_waitcnt(WAIT_LGKM)` to retire the ds_writes,
and only lgkmcnt, never a full drain. Do the same. And note `cluster.py`'s `cluster_barrier()`
is the wrong tool: that is cross-workgroup clustering, not intra-workgroup sync.

### The hang trap: keep the causal loop bounds workgroup-uniform

`kv0` feeds `qp_start` / `nqp_eff` / `nmaskp` (`kernels.py:258-261, 426-432`), and those are
the **trip counts** of the two `scf.for` loops whose bodies contain barriers (`:439-440`). If
`kv0` becomes per-wave, the four waves get different trip counts, the barrier lands in
divergent control flow, and the result is a **hang -- which on this card costs a power cycle,
not a wrong answer.** Today this is invisible because the barriers are elided.

```
kv0_wg = bid * fx.Int32(WG_KV)                    # uniform: use for qp_start / _qsf / nmaskp
kv0    = kv0_wg + wave * fx.Int32(BLOCK_KV)       # per-wave: use ONLY for addressing and mask
```
Cost: causal tile-skip granularity coarsens 32 -> 128 rows, which is +1.17% purely-masked
wave-iterations (4,210,688 -> 4,259,840). Record it; do not optimise it away this round.

### Budget, computed offline -- it closes with room

LDS today is 22528 B = `lds_p 2560 + lds_ds 2560 + lds_do 8704 + lds_q 8704` (the single
allocation at `kernels.py:206`; all four tiles are 32 **query** rows, and the "16 query rows"
comment at `:202` is stale since g07). Four-wave: Q/dO shared and unchanged at 17408 B, P/dS
per wave -- merge them into one `[32 q][128 kv]` pair with `S_ROW_B = 2*WG_KV + 16 = 272`:

```
2*32*272 (P,dS) + 2*32*272 (Q,dO) = 34816 B = 10.6% of the 327680 B limit, exactly 17*2048
```
272 B = 68 dwords is the same benign stride `g16` established; 256 B = 64 dwords is the fully
conflicting one. Four separate 80-byte-stride tiles would take 37888 B and waste 1024 B to
granularity, so merge.

**Residency does not change, and this is the discriminant against the gfx950 warning attached
to g04 in the pool.** VGPR ~636 > 512 means 1 wave/SIMD; a WGP's four SIMDs then hold one
four-wave workgroup = 4 waves/WGP, exactly the 4 waves/WGP of today's four single-wave
workgroups. The gfx950 "3-4x slower via occupancy collapse" note describes widening at fixed
wave count, where accumulators and LDS grow together. Here the accumulator floor is unchanged:
`kernels.py:434` allocates `2*NKV*NDO = 2*2*8 = 32` v8f32 = **256 dwords/lane**, and each wave
still owns 32 kv rows. That is precisely what `g14` did not have (NKV=4, floor 512, 2.76x
slower). `g21`'s carried prefetch state also shrinks 128 -> 32 dwords, freeing ~96 VGPR.

Expect 540-700 VGPR, spill 0. **Screen gate: VGPR > 700 kills it before a GPU slot.**

### Four falsification checks the offline screen MUST report

Without these a broken build looks like a clean negative:

1. `group_segment_fixed_size` == **34816** (or 38912 for the unmerged layout). **>= 69632
   means Q/dO were multiplied by four -- the sharing was never implemented and the whole
   premise is void.**
2. loop-body `buffer_load_b128` == **8** (today 32). Still 32 means zero bytes saved.
3. `known_block_size == [128,1,1]` **and** the launch `block == (128,1,1)`.
4. `s_barrier` per loop body == **3** exactly (locate the body from the backward branch, as
   round 6 did). 0 means the block size did not take; 2 means the staging barrier is missing
   and GEMM1 is reading LDS that may not be written yet.

### What to expect, and the one thing that could eat it

Traffic 71.1 -> 18.0 GB on the dominant kernel. Against that: ~10% of dependency-chain cost
from the LDS-fed GEMM1, +1.17% masked work, and an **uncovered barrier stall** -- at 1
wave/SIMD there is no second wave to hide a rendezvous. The per-iteration budget is roughly
1177 cycles; three barriers exposing 50-200 cycles each would be 13-51% of it. That is the
single number most likely to sink this, and it is not knowable statically. Measure the
per-kernel split, not just end-to-end.

`fast` will fall: k_dkdv's grid there goes from 64 workgroups to 16 on a 128-WGP part. Per h5
and h7 read that as an occupancy number and rank on **prod**.

## h13 -- Round 8 deleted the barriers g04 needs; re-add them FIRST

Round 8 shipped `r8.i1.g23+g24`, which **deletes both `fx.barrier()` calls** from `k_dkdv`
and `k_dq`. The reasoning was sound and the win is real (+6.3% at prod, bitwise identical):
at `block=(32,1,1)` the workgroup is one 32-lane wave, so `s_barrier` is semantically a
no-op and LLVM already removed it -- the shipped ISA had 0 of them. What survived was the
conservative all-counter wait the backend emits *for* the barrier before deleting it,
`s_wait_loadcnt_dscnt 0x0`, whose `loadcnt` half has no dependence here and was truncating
`r7.i1.g21`'s prefetch cover to 299 instructions instead of a full iteration.

**But those were the exact calls h12 named as the ones a 4-wave `k_dkdv` would rely on**
(`kernels.py:359` and `:391`). h12 said "barriers need no new source". **That is no longer
true.** With them gone, raising `known_block_size` to `[128,1,1]` does not produce a correct
kernel that merely needs tuning -- it produces a **silently racy** one: four waves reading an
LDS tile that may not be written yet.

Silently is the operative word. Under causal masking the true dK/dV in the skipped region are
near zero, so a race that drops or reorders staged rows can still clear the 50 dB SQNR gate
with its 2.5 dB of headroom, and it can be bitwise-stable across 200 runs if the race resolves
the same way each time under a fixed schedule. **Neither existing gate is guaranteed to catch
it.**

So, for any future g04 attempt, in this order:

1. **Re-add both `fx.barrier()` calls before changing anything else**, each preceded by an
   explicit `rocdl.s_waitcnt(WAIT_LGKM)` -- `gpu.barrier()` is not a fence, and the gfx950
   reference makes this explicit at every one of its own barriers.
2. Add the third barrier between the shared Q/dO staging store and GEMM1 (h12).
3. Only then flip `known_block_size` and the launch `block` to `[128,1,1]`.
4. Verify `s_barrier == 3` per loop body in the ISA, as h12's check 4 requires. At
   `[32,1,1]` the count will still read 0 no matter what the source says -- so **the barrier
   check is only meaningful once the block size has actually changed**, and it must be read
   after, never before.

### Two independent blockers now stand in front of g04 -- price them before spending a round

**Registers.** ~~Round 8's own analysis puts `g04` 462 registers short.~~ **FALSIFIED
2026-09-22 by measurement -- see h14.** The 462 figure was wrong three ways (it is a 2-wave
number applied to a 4-wave change, it carries an arithmetic slip, and above all it applies
`VGPR x waves <= 1024` with waves-per-WORKGROUP where the rule means waves-per-SIMD). A
compile-only probe raising only `known_block_size` and the launch block measured VGPR
essentially flat from 32 to 128 threads -- 988/984/984, 956/952/952, 632/636/634, spill 0
everywhere -- because four wave32 land on the WGP's four SIMDs at 1 wave/SIMD, exactly
today's residency. **There is no register blocker.** Note also that the "nothing between 951
and 1024" rule now has two counter-examples of its own: 956 and 988, both spill 0.

**Synchronisation.** This hint.

Both were created by rounds that were individually correct: `g21` bought +8.3% with the
prefetch, `g23+g24` bought +6.3% by deleting the dead barrier. Together they have hill-climbed
the kernel into a place where the one structural change the traffic analysis says is necessary
no longer fits. **That is worth stating plainly to the operator rather than discovering it a
third time**: reaching `g04` may now require giving back `g21`, `g23+g24`, or both, and the
round that attempts it should price that trade explicitly -- what the revert costs, against
the 71.1 -> 18.0 GB the 4-wave form buys -- instead of assuming the increments are free to keep.

## h14 -- Round 9 is the carried-state merge: ~200 VGPR are an allocation failure, not a cost

Round 9 has one target and it is not `g04`. Established offline on 2026-09-22, no GPU spent
beyond a compile-only probe. Check it rather than trusting it.

### `k_dkdv`'s 988 registers are set by a loop that runs four times

Parsing the real register numbers out of `21_final_isa.s` (gfx1250 uses `s_set_vgpr_msb` for
bank switching, so the text names stop at v255 and the true numbers are in the comments) and
doing read-before-written per region:

| region | VGPRs touched | carried-in | where the carried state lives |
|---|--:|--:|---|
| prologue | 549 | -- | -- |
| `.LBB0_2` masked body (59 `v_cmp`/`v_cndmask`) | **985** | 609 | K/V in v146-329, acc+prefetch in v513-928 plus 63 odd numbers in v387-511 |
| between (phi shuffling) | 35 | -- | -- |
| `.LBB0_6` full body (0 mask instructions) | 749 | 546 | **v2-v513 contiguous** = 256 acc + 128 prefetch + 128 K/V |
| epilogue | 296 | -- | -- |

The two loop bodies' carried state lives in **disjoint register blocks** with 35 registers of
phi shuffling between them. Logically there is one 512-dword carried state; physically it
occupies about 930.

**The control experiment is already on disk.** `rounds/007/_scratch/dump_armQ` -- 632 VGPR,
has the mask split, does NOT have g21 -- puts both bodies' carried state in the SAME block
(v1-v385 / v2-v409). The allocator merged them there and fails to here.

So **`r7.i1.g21`'s real register cost is +356, not +128** (632 -> 988): the 128 dwords it
added to the carried tuple pushed the two bodies past whatever threshold the allocator merges
below, and the entire carried state got duplicated.

And the masked body runs `G * nmaskp = 4` iterations at prod against the full body's ~508.
**0.8% of the iterations define 100% of the register ceiling.**

### The target

Recover those ~200 registers **without giving back `g21` (+8.3%) or `g23+g24` (+6.3%)**. This
is not a trade. It is an allocation artefact, and it is the prerequisite that makes every
4-wave variant comfortable rather than marginal.

One lead, offered as a lead and not a prescription: the masked body carries `g21`'s prefetch
state, and over four iterations a prefetch cannot pay for itself. If the masked body does not
carry it, the two carried tuples stop being the same size and the allocator's problem changes
shape. There may be better answers -- a single loop with a predicated tail, a different phi
structure, hoisting differently. Find the one that measures.

### What would make this round a success

* `k_dkdv` VGPR materially below 956 with spill 0 and LDS unchanged at 22528, **and**
* prod not worse. A register win that costs prod time is not a win here; say so and stop.

Registers are the deliverable, time is the constraint. It is entirely possible that freeing
200 registers buys nothing at prod today, because the kernel is at 1 wave/SIMD either way and
nothing is waiting on occupancy. **That result is a valid delivery** -- it still unblocks the
structural work, and reporting "the registers came back and the clock did not move" is worth
more than a round that quietly abandoned the target.

### What round 9 must NOT do

Do not attempt `g04` this round. Do not revert `g21` or `g23+g24` to make room -- the probe
below shows there is no room problem to solve that way.

### The probe that settles the register question, already run

Compile-only, three bases x three block sizes, zero GPU, baseline gate reproduced exactly
(956/988/632):

```
base                       w32    w64   w128   spill   LDS
B  round-7 ship            988    984    984     0    22528
N  current ship (round 8)  956    952    952     0    22528
Q  round-7 armQ (no g21)   632    636    634     0    22528
```

Raising `known_block_size` and the launch block from 32 to 128 threads leaves the per-wave
budget flat. Four wave32 occupy the WGP's four SIMDs at 1 wave/SIMD -- the same residency as
four independent single-wave workgroups today. LLVM does not cut the per-wave budget as
`flat_workgroup_size` rises. **h13's "462 registers short" is withdrawn.**

Be precise about what that probe did and did not show: it changed ONLY the block size, so
those builds compute wrong answers and save no bytes -- every wave still does the full 32-kv
work. It answers the register-budget question and nothing else. The real 4-wave form adds the
LDS-fed GEMM1 operands (+64 dwords/lane) and removes three quarters of the prefetch state
(-96), for a net near -32.

### Standing note for round 10, when `g04` is taken

Two conditions, both from today:

1. **h13 still holds.** Re-add both `fx.barrier()` calls first, each preceded by
   `rocdl.s_waitcnt(WAIT_LGKM)`, then the third barrier before GEMM1, and only then change
   the block size. Verify `s_barrier == 3` per loop body AFTER the block size changes -- at
   `[32,1,1]` the count reads 0 no matter what the source says.
2. **`g04`'s benefit is not yet priced, while its costs are.** The traffic argument in h8
   divides by a 6.46 TB/s HBM roof, but this kernel's requested-byte rate at prod is
   107 GB / 12.8 ms = **8.4 TB/s** -- above that figure, which means a substantial share of
   those requests are already being served by cache and never reach HBM. Against that
   unpriced benefit sit measured costs: the third barrier (9.0-14.7% on gfx950 for exactly
   this shape), the partial reversal of `r2.i2.g09` (+11% when it landed), and an uncovered
   rendezvous at 1 wave/SIMD. **Round 10 must first establish that some real fraction of the
   107 GB reaches HBM** -- otherwise it is paying measured costs for a modelled gain. The
   only indirect evidence so far points the other way: `g05`'s grid permutation was worth
   only +2.6%, which is what you would expect if L2 were already absorbing most of it.

## h15 -- Rules for every script a round writes

Three wedges in ten rounds. Two of the three happened inside a script the round had written
itself, under `_scratch/work/`, which is the least audited code in the job: `bitwise.py`
(round 8) and `gpu_final.sh` (round 9). The rules below are not style preferences. Each one
is there because its absence cost something measurable.

### 1. Never call `forward_reference` or `eager_attn_bwd`. Use `refcache_util.cached_forward`.

`op/eager/impl.py` and `common.forward_reference` are fp32 GEMMs that reach a Tensile kernel
which faults on this card -- dispatch `workgroup=[256,1,1]`, `group_seg_size=6144`, name
`Cijk_Ailk_Bljk_SB_MT128x64x8_*`. It aborted round 3's gate, wedged round 8 inside
`bitwise.py`, and on 2026-09-22 escalated to an unrecoverable MES state that cost a power
cycle. **None of the kernels under test is implicated**: k_dkdv is 32 threads with 22528 B of
LDS, k_dq 32 / 8704, k_delta 256 / 0. Nothing we write dispatches 256 threads with 6144 B.

The gate itself was fixed on 2026-09-22 (`validation.py`, `benchmark.py`), but the fix only
covers the framework's own files. A script the round writes bypasses it:

```python
import sys
sys.path.insert(0, str(JOB / "job_context/op"))        # for refcache_util
from refcache_util import cached_forward
o, lse = cached_forward(shape, q, k, v, causal=True)   # NOT forward_reference(...)
```

`cached_forward` falls back to computing whenever the cache does not apply -- a missing
entry, a provenance mismatch, a non-causal request -- so it is never worse than the call it
replaces. Caches exist for fast, proxy and prod.

**Better still, do not need a reference at all.** `split2.py` synthesises `o` and `lse`
(timing only) and therefore never touches that path. If a script is measuring time rather
than accuracy, synthesise.

### 2. Every pipe out of a long-running GPU command gets `--line-buffered`

`| grep -vE "$F"` is block-buffered. When the process is killed by a wedge, up to 4 KB of
output is lost -- which is exactly why round 3's `gpu5/out` is **0 bytes** and round 9's
acceptance section is **blank**. We still do not know which shape round 9 was on when it
died. One flag would have told us:

```bash
... 2>&1 | grep --line-buffered -vE "$F"
```

Also print a sentinel after each block, so the last completed step is unambiguous:
`echo "@@@@ STEP <name> rc=${PIPESTATUS[0]}"`.

### 2b. A wedge has a signature, and it is visible one step early

Four dmesg captures survived in `rounds/002/1-opt/raw/`, two from faults the card walked away
from and two from faults that escalated. The discriminant is clean:

| | kernel-under-test fault (card survives) | Tensile fault (escalates to a wedge) |
|---|---|---|
| ring / vmid | `ring:40 vmid:4` | `ring:24 vmid:3` |
| fault address | `0x00007180 2b884000`, a normal 47-bit user VA | **`0x00000000 977e6000` -- truncated, below 4 GB** |
| how consistently | armAB: 3 of 3 user VA | tensile: 5 of 5 sub-4 GB; gpu2: 9 of 9 |
| first fault | `PERMISSION_FAULTS:0x3, RW:0x0` | **`PERMISSION_FAULTS:0x5, RW:0x1`** (a WRITE fault) |
| escalation | none | `copy_context_work_handler [amdgpu] hogged CPU for >10000us N times` and `tawk_ipc ... Mailbox work was idle for too long ... (New max)` |

**`copy_context_work_handler` is the early-stop line.** In `dmesg_tensile_flake_gpu2.log` it
appears after the second fault and before the next seven, and its counter climbs across
captures (4 times, then 5 an hour later). It marks the MES queue-context copy path beginning
to stall, which is the step before an unrecoverable MES state.

**Abort rule.** If a post-block dmesg read shows a NEW `copy_context_work_handler` or
`Mailbox work was idle ... (New max)` line, stop the round -- do not start the next GPU
block. Also stop if VRAM in use is more than about 2 GB above the round's baseline with
nothing running, which means a leaked context.

**And an in-band precursor that needs no dmesg at all**, on the process's own stdout:

```
Memory access fault by GPU node-2 ... on address 0xd5b45000    <- also sub-4 GB
Queue error: HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION
... workgroup=[256, 1, 1] ... group_seg_size=6144
... kernel: Cijk_Ailk_Bljk_SB_MT128x64x8_SN_...
```

Grep the tee'd log for `Memory access fault by GPU node|HSA_STATUS_ERROR|Cijk_` after every
block and abort on a hit. Zero cost, and it fires before dmesg does.

**What is NOT a precursor:** sclk. Rounds 8 and 9 sat at 1050-1100 MHz throughout, the same
as every clean round -- that is the VR throttle, not a symptom. Nothing in any log shows MES
ring-buffer-full or queue preemption before a wedge either.

### 3. A bounded dmesg read between GPU blocks

Round 9's fault signature is **unknown** because its dmesg was lost to the power cut. On a
wedged card a bounded standalone `dmesg` read is the only safe probe -- `rocm-smi`, `pgrep`,
`ps -o wchan` and `docker stop` all hang in the driver. Between blocks:

```bash
timeout 10 dmesg | tail -60 | grep -iE "MES|APERTURE|page fault|GPU reset|REMOVE_QUEUE|Cijk_" \
  || echo "DMESG_CLEAN"
```

Cheap, safe, and it turns the next wedge from a mystery into a diagnosis.

**Never end a GPU pipeline with `| tail -N`.** `tail` emits nothing until EOF, so when a
wedge kills the process that step's output is lost in full -- round 9's UT block is blank for
exactly this reason. Use `| tee $LOG | tail -5` instead: the file keeps everything, the
terminal stays readable.

### 4. Put the acceptance sweep FIRST, on a cold card

Round 9 wedged in the last GPU block of a chain that had already run about twenty minutes
of continuous load, while a strictly heavier sweep in the same round -- five arms, 101
iterations, three shapes -- had finished clean three minutes earlier. That is correlational,
not causal, and the honest statement is that we do not know. But the irreplaceable
measurement is the one that decides the round, so take it while the card is idle and cheap
to re-run, and leave the validations and bitwise checks for after. Round 9 survived its own
wedge only because `d_s2` had already banked the numbers.

### 5. Copy `bitwise.py` from the repo, not from the previous round

`_scratch/work/bitwise.py` was **byte-identical in rounds 3 through 9** (md5
`bf5538332deb`), inherited unread each time, and every copy called `forward_reference` at
prod -- about 2000 high-risk dispatches. Round 8 wedged **inside that script**, at armB,
five rounds after the pattern was first written down.

The corrected copy is at
`/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/tools/gfx1250/bitwise.py`. One line
differs: `forward_reference` -> `cached_forward`. Output format is unchanged, so results
stay comparable with rounds 3-9. **Copy that one.**

The same applies to anything else inherited. `split2.py` (md5 `bd43817e37a4`, also identical
across all seven rounds) is already safe -- it synthesises `o` and `lse` and never touches
the reference path -- which is exactly the pattern to imitate when a script only needs
timing. Copying is fine; copying unread is what costs power cycles.

## h16 -- The bound is the ISSUE ROOF, not bandwidth

Eight rounds ordered this pool around a belief that the kernel is bandwidth bound. h8 said
so, h11 priced the 4-wave restructure at 500-580 TF/s on that basis, and h14 recorded that
round 9 falsified it. This hint replaces the model rather than patching it, and the
replacement changes which candidates are worth a round.

### The measurement that reframes everything

Issue efficiency, eta = issued FLOP / (peak x time), from our own machine code:

```
k_dkdv full body   64 WMMA x 4,210,688 iters = 269.5 M
k_dq   full body   96 WMMA x 2,113,536 iters = 202.9 M
                              issued 472.4 M x 16384 FLOP = 7.740e12
algorithm needs (5 GEMM)      335.5 M          = 5.497e12
ratio                         1.4080  ==  exactly 7/5
```

That 7/5 is the corpus's "seven matrix products against five" confirmed for the first time
from our own ISA, not quoted from a recipe. Against the measured 1002.7 TF/s bf16 roof:

| | eta | prod ms |
|---|--:|--:|
| us | **0.682** | 11.322 |
| aiter | **0.725** | 7.561 |

**We are already at 94.0% of aiter's scheduling quality.** Of the 1.464x that separates us,
**1.408x is doing 40% more matrix work and only 1.040x is scheduling.** The gap was never
mainly a scheduling gap, and eight rounds of treating it as a traffic gap were aiming at the
wrong term.

### Correct the HBM roof in facts.md: 4.39 TB/s, not 6.46

`k_delta` is the job's only pure streaming kernel -- four `buffer_load_b128`, a DPP
reduction, two `buffer_store_b32`, no loop -- and it moves COMPULSORY bytes: O + dO =
536.9 MB in, delta 4.2 MB out, 0.5411 GB in 0.1233 ms = **4.39 TB/s**. That is this card's
measured achievable rate at the 1100 MHz VR throttle, 32% below the spec figure h8 divided
by. It tightens f, the fraction of k_dkdv's 71.1 GB that reaches HBM, to **[0.011, 0.444]**.

Same ruler on k_dq: 6.69 TB/s before g26 and 8.68 after, both above 4.39 -- it was never
near the HBM roof, which is a third independent confirmation of round 9's finding.

### The ceiling, stated plainly

In the 7-GEMM shape, eta is the only lever:

| eta | prod ms | TF/s | x beat |
|---:|---:|---:|---:|
| 0.682 today | 11.322 | 481.0 | 0.668 |
| 0.725 = aiter | 10.646 | 511.6 | **0.710** |
| 0.800 never reached by anyone | 9.649 | 564.4 | 0.784 |

**h11's 500-580 TF/s range dies here.** 500 needs eta 0.708, i.e. matching hand-written ASM;
580 needs 0.823, i.e. beating hand-written ASM by 13% from compiler-generated FlyDSL at one
wave per SIMD. The bottom of that range was "draw level with aiter's scheduler" and the top
was never reachable. **All remaining scheduling headroom is +6.4%.**

Parity needs the 5-GEMM fused form, and that is barred structurally -- see g15 below. The
rest of the distance to aiter is not debt we have failed to pay. It is the price of a
contract we are keeping.

### Verdicts

**`r3.i6.g15` (fuse dQ) -- CLOSE PERMANENTLY, into dead_ends.** Round 9 removed its register
blocker (non-accumulator use 740-256 = 484 against a 640 budget; it fits on paper). The real
blocker is arithmetic and it does not move:
* atomics are barred by `DETERMINISM_RUNS = 200` in validation.py -- fp32 atomicAdd reorders
  and cannot be bitwise-stable across 200 runs;
* the deterministic alternative writes one dQ partial per kv band. dQ fp32 is 537 MB. At
  BLOCK_KV=128 that is 64 bands = 34.4 GB written and read, **7.84 ms at the measured
  4.39 TB/s**, against the two GEMMs it saves, 2.243e12 FLOP = **3.28 ms at today's eta**.
  **Net -4.6 ms, even after g04.** h9's "raise BLOCK_KV first" does not rescue it.

  Note carefully why this survives round 9's demolition: that workspace traffic is
  **compulsory** bytes, not re-read bytes. Round 9 falsified "re-read bytes == HBM bytes".
  It said nothing about compulsory bytes, and this is all compulsory.

  g15 keeps exactly one value: it is the thing that names where the 1.408x lives.

**`r5.i2.g18` (cluster multicast) -- CLOSE, into dead_ends.** Its entire case was bytes.
At least 55.6% of the bytes it removes never reach HBM anyway (f <= 0.444); gfx1250 caps a
multicast at five destinations so it can delete at most half the fill, not 75%; its landing
path `cluster_load_async_to_lds` measured **9.0-14.7% slower** on this kernel and this
staging; and g03 already counter-evidenced the premise when breaking sharing came out 2.6%
FASTER. Expected value negative with the upside squeezed by f.

**`r1.i4.g04` (4-wave) -- DEMOTE to round 11 at the earliest.** Multi-wave does not change
the WMMA count at all; it buys only eta, whose entire remaining headroom is +6.4%. Against
that sit measured costs: a third barrier (9.0-14.7% on this shape and staging), the partial
reversal of `r2.i2.g09` (+11% when it landed), and h13's racy precondition. A two-kernel
rewrite chasing a +6.4% ceiling is the wrong trade while cheaper eta is still on the table.

### Round 10's candidate: `r10.i1.g27`

The latency-exposure mechanism has paid twice -- g21 in k_dkdv (+8.3%), g26 in k_dq (+11.4%)
-- and there is exactly one hot-path site left that it has never touched. In k_dkdv's full
body (`dump_armAB/k_dkdv_0/21_final_isa.s`, `.LBB0_8`, relative instruction numbers):

```
@  9  s_wait_loadcnt 0x0     <- g21/g25's cross-iteration drain, covering ~520 instrs, fine
@ 89  buffer_load_b32 v164, v3, s[56:59]   @ 90  buffer_load_b32 v166, v2, s[56:59]    |  LSE (2) + delta (2)
@ 92  buffer_load_b32 v168, v3, s[60:63]    |  *** g21 never prefetched these four ***
@ 93  buffer_load_b32 v170, v2, s[60:63]   /
@206  s_wait_loadcnt 0x22
@207-210  v_nop v_nop v_nop v_nop          <- the scheduler could not fill it
@212  v_pk_add_f32 ... neg                 <- S - lse, the first consumer
@335  s_wait_loadcnt 0x21 + 4x v_nop
@359  s_wait_loadcnt 0x20
```

Give those four the same treatment g21 gave the Q/dO loads: issue iteration i+1's at the top
of iteration i and carry them in the full loop's scf.for state. Apply g25's lesson up front
-- **the masked body must load its own**, so the two carried tuples differ in shape and the
allocator has no identical block to duplicate. Cost is about +4 VGPR against a 740 baseline,
and it should be bitwise identical by construction.

Everything outside the two hot bodies totals **under 2.8% of runtime** (k_delta 1.09%, both
mask bodies 0.75%, both prologues and epilogues 0.94%), so there is no fourth site worth a
round. Two items for the pool as incidental fixes, not candidates: both epilogues emit
**256 `buffer_store_b16`** where 32 `b128` would do, and the address arithmetic ends in
`v_lshlrev_b32 v221, 2, v221` -- a 4-byte stride on 2-byte data, i.e. possibly half-density
cachelines. Upper bound under 1% either way.

### One instrument, if the round has budget after g27

A qmask address-pin probe would measure f directly, with no counters (h4 bars the profiler).
Add a **runtime** `fx.Int32 qmask` to k_dkdv and mask the query-pair index **absolutely** --
`q0 = (qt & qmask) * 32` at kernels.py:234, the single choke point through which all of
k_dkdv's Q/dO traffic flows (97% of the 71.1 GB). `qmask = 0x7FFFFFFF` is the identity;
`qmask = 3` pins every workgroup to query pairs {0,1,2,3}, a 2 MB working set that fits the
4 MB L2, forcing f to zero. DeltaT between the two is the time those requests actually spend
reaching HBM.

Three things about it that are easy to get wrong:
* **The mask must be ABSOLUTE, not relative to `qp_start`.** `qp_start == bid` at prod, so a
  relative mask pins each workgroup to its own four pairs and the union across resident
  workgroups is tens of megabytes -- f would not be pinned and DeltaT would badly understate.
* **Do not mask the GQA head.** At prod G=4 so `gh & 3 == gh` is the identity: pure cost.
* **kernels.py:287 is a byte-identical line** `q0 = qt * fx.Int32(32)` inside `_body`, and it
  must NOT change -- it feeds the lse/delta addresses and the causal mask comparison. A
  global `sed` breaks the probe silently. Patch by line number and read back both lines.

Leave `impl.py` alone: the resulting arity mismatch is a safety interlock that makes the
probe tree structurally uncallable from validation.py and benchmark.py, so it cannot leak
into an act.yaml speed table. `split2.py` loads kernels.py by path and drives it fine.
`screen.py` needs a third arity branch keyed on `_QMASK_PROBE`.

**The probe ships nothing. Its `qmask=3` output is numerically WRONG by construction.** It is
a one-shot instrument and must never appear in a speed table.

## h17 -- Read the corpus before you build

### What happened

h16 named `r10.i1.g27` as round 10's candidate, reasoning from the ISA: four LSE/delta
`buffer_load_b32` in k_dkdv's hot body that `g21` had never prefetched, followed by
`s_wait_loadcnt 0x22` and four `v_nop` the scheduler could not fill. It looked like free
latency.

It was already a documented dead end, in two places:
* `techniques.md:214-216` -- the same family, measured at **-5.3%**
* `dead_ends.md:154-194` -- again, at **-4.3%**

Round 10 built it anyway and measured **-14.57%**, then found both entries itself and wrote
"two build-before-read in one round, the thing most worth remembering from it". That is the
correct lesson and this hint makes it standing.

**The operator wrote h16 without checking either file. The cost was a whole round.** A hint
carries more weight than a round's own idea, which makes an unchecked hint more expensive
than an unchecked idea, not less.

**Rule: before any candidate is built -- whether it came from a hint, the pool, or the
round's own reading of the ISA -- grep `dead_ends.md`, `techniques.md` and `facts.md` for
its mechanism, not just its id.** g27 had no id in the corpus; it had a *mechanism* written
down twice. Searching for "g27" would have found nothing.

### The mechanism, which is worth more than the candidate was

Folding those four loads into the carried set does not shorten the wait. It **lengthens**
it: `s_wait_loadcnt 0x22` waits on a COUNT, and adding loads to the carried tuple moves the
consumer past a full-landing count instead of a partial one. A partial wait becomes a full
drain. That inverts the g21/g26 lesson -- those two added prefetch and won, this one added
prefetch and lost -- and the distinction is whether the consumer's wait count still permits
partial completion.

Round 10's three arms make the shape of it unmistakable, and the anti-correlation is the
finding:

| arm | issue slots cut | prod |
|---|--:|--:|
| `g28` | -17% | **-7.77%** |
| `g27` | -6.6% | **-14.57%** |
| `g32` (throwaway probe, finest-grained waits) | — | **-61.5%** |

**Instruction count is not the independent variable and `s_wait` count is a deceptive proxy.
Load-to-use distance is, and the in-place implementation is already at a local optimum on
it.** Instruction-level scheduling of k_dkdv's hot body is a CLOSED DOOR. Do not spend
another round on it, and do not take h16's qmask probe either -- round 10 declined it for
exactly this reason and was right to.

### What h16 keeps

Its eta decomposition stands and is not in question: of the 1.464x to aiter, **1.408x is
doing 40% more matrix work (seven GEMMs against five, confirmed from our own ISA) and only
1.040x is scheduling**. What round 10 refuted is the actionable reading -- that saving
instructions collects that 1.040x. The 1.040x is real but it is not reachable by cutting
slots.

### Round 11: `r10.i3.g30`, the register bank tax

Round 10's own route names the one door it found open. `s_set_vgpr_msb` exists solely
because the kernel uses more than 256 VGPRs, so every access above the window costs a bank
switch. The static count at 740 VGPR:

```
k_dkdv hot body   110 of 693 instructions = 15.9%
k_dkdv whole      449
```

This is a different mechanism from what round 10 closed. It is not load-to-use distance and
not issue-slot count -- it is the cost of *addressing* registers, and it scales with how
badly the register allocation is grouped rather than with how many registers there are.

Two things to establish before building, in that order:
1. **Is the tax reducible at constant VGPR?** 110 switches for 693 instructions suggests
   poor bank locality, not an irreducible floor. Grouping accesses so consecutive
   instructions stay within one window would cut switches without touching the allocation.
   Read the ISA and count how many of the 110 are *redundant* -- a switch to a window that
   the previous switch already selected.
2. **What is the floor?** The dK+dV accumulators alone are 256 dwords per lane
   (`2 * NKV * NDO` = 32 v8f32), so VGPR can never drop below 256 and the tax can never go
   to zero. Establish the arithmetic floor before estimating any payoff.

And apply the rule at the top of this hint first: **grep the corpus for
`s_set_vgpr_msb`, "bank", "vgpr window" before building anything.** If it is already priced
there, that number outranks any estimate made here.

## h18 -- The bar cannot pass our own gate

### The finding

**aiter's gfx1250 backward accumulates dq with fp32 atomics.** Three independent sources,
all hard:

* `aiter-src/csrc/cpp_itfs/mha_bwd.cu:617-618` -- "ASM kernel **atomically accumulates**
  into dq_accum; require zero-init"
* `:472` -- `need_post_processing` is **unconditionally true** for gfx1250
* our own launcher, `tools/gfx1250/asm_bwd_abi.py:27` -- the pipeline is
  `bwd_hd128_odo_bf16`, `bwd_hd128_bf16_causal_br_a32_pssk`, `bwd_hd128_dq_convert_bf16`,
  where **`a32` is `v3_atomic_fp32`**

So the bar would fail `DETERMINISM_RUNS = 200` if we ran it through our own gate. **The
entire 1.408x of matrix work that separates us grows out of that gate.** This is not "we
are worse than aiter"; it is "we are playing a different game", and every kernel-level
lever left to us lives inside the remaining 1.040x.

### A correction to h16, of the same kind h17 exists to prevent

h16 said aiter's five-GEMM fusion was "confirmed by disassembly". **It was not.**
`job_context/logs/planner.log:989` records that gfx1250 `.co` files do not disassemble --
`llvm-objdump` returns an empty `.text` for `e_flags 0x549`. The five-GEMM claim is
*inferred* from gfx950 disassembly plus the manifest plus `mha_bwd.cu`. Our own seven is
measured from our own ISA; their five is not. The atomics claim above, by contrast, IS
hard evidence, and it is the stronger of the two.

### What the gate actually is, and where it came from

The requirement is the operator's own words, unannotated, in
`job_context/history/v000_original.yaml:94-97`:

> The baseline is atomic-free by construction: every output element is written exactly
> once. Keep that -- it is one of the reasons to have a source-level backward at all, and
> **200-run bitwise determinism is a cheap gate on it.**

The real requirement is *every output element written exactly once*. The 200-run check is
named, by the person who set it, as **a cheap observation of that property, not the
property**. `validation.py:174-177` says the same: "Bitwise identity is the OBSERVABLE form
of 'no atomics on any output'." A constraint given with its reason is one the same person
can re-weigh against the same reason.

Three strengths, strictest first:

| | | us | aiter |
|---|---|---|---|
| A | bitwise against the fp32 reference | ✗ (nobody) | ✗ |
| B | **this kernel bitwise run-to-run** (the current gate) | ✓ 11 rounds | **✗** |
| C | fixed reduction order, "training reproducible" | ✓ | ✗ |

**The gate enforces B, and B is strictly stronger than C.** PyTorch's
`use_deterministic_algorithms` asks for C. aiter ships a `deterministic=True` path that is
C, priced in the corpus at **-22.7%** -- but that figure is gfx950 hd128, not measured here.

### If it were relaxed

Estimated, from our measured 481.0 TF/s at prod and the ISA-measured 7/5 ratio:

| | prod ms | TF/s | of bar | gap closed |
|---|--:|--:|--:|--:|
| today, 7 GEMM, eta 0.682 (**measured**) | 11.430 | 481.0 | 0.668x | -- |
| 5 GEMM, eta unchanged (est.) | 8.164 | 673.4 | 0.935x | **80%** |
| + the atomic form's own overhead (est.) | 8.470 | 649.1 | 0.902x | 70% |
| 5 GEMM at aiter's eta 0.725 (est.) | 7.680 | 715.9 | 0.994x | 98% |

The atomic form's overhead is arithmetic: a 536.9 MB fp32 `dq_accum` zero-init plus
`dq_convert` reading 536.9 MB and writing 268.4 MB = 1.342 GB at the measured 4.39 TB/s =
**0.306 ms**. Note this does NOT pay h16's -4.6 ms: that figure was for the *deterministic*
split-K alternative, whose 17.2 GB read-back disappears entirely under atomics. The
credible range is **[649, 720] TF/s**, and the top of it is a number actually measured on
this card -- by the bar.

**But the path is not a switch.** It is g04 (4-wave, BLOCK_KV 32->128) plus re-adding the
three barriers h13 requires, plus g15 (fuse dQ, delete k_dq), plus atomic dq, plus the
convert kernel -- the largest structural rewrite in the job. And per-kernel eta says
`k_dq` is already **0.79, above aiter's 0.725**, while `k_dkdv` is 0.63: the gap is 100% in
k_dkdv, and fusing would delete the better kernel into the worse one.

`evolve.max_rounds: 12` is the operator's own setting and rounds 0-11 are spent. **One
round remains.** Relaxing the gate today could not be cashed in it.

### What the gate has caught, stated fairly

**In eleven rounds it has never fired once** -- every `validation.py` record reads
`correctness pass / determinism x200 pass / speed FAIL`. That is evidence it has not
caught a defect here.

But the corpus has a case where it caught what nothing else would
(`planner.log:901`): dQ silently miscomputed at 8-14 dB on particular
`(q_split, BLOCK_KV)` combinations while throughput read fine, and **the determinism check
failed before the SNR check did**. And round 6 proved SQNR is blind to this class: `armM`
(bitwise-identical by construction) and `armN` (deliberately reordered fp32 accumulation)
both read **52.61 / 52.65 / 52.83 dB, to two decimals**.

Against that, h13 already recorded a hole the gate cannot cover: a race that drops or
reorders staged rows can clear 50 dB *and* be bitwise-stable across 200 runs if it resolves
the same way under a fixed schedule.

**A cheap middle:** drop from B to C (fixed reduction order) while widening the check from
one shape to cross-shape x cross-launch-config. The corpus precedent that actually caught
something relied on the `q_split x BLOCK_KV` sweep, not on the number 200 -- so this keeps
the half that works and unlocks the 1.408x. **Operator's call, not the round's.**

### Round 12 buys an instrument, not a candidate

Two rejections in a row, +6.4% of headroom, and both obvious doors shut from mechanism:

* **instruction-level scheduling** (round 10) -- instruction count and time anti-correlate;
  load-to-use distance is the variable and the code is at a local optimum.
* **the bank tax** (round 11) -- `s_set_vgpr_msb` is a pure encoding prefix for the next
  instruction's VGPR index bits. `g32` still showed 110 of them at 632 VGPR, so it is
  driven by which register numbers the WMMA operand slots reference, not by how many
  registers exist. FlyDSL has no register pinning and 32 accumulators are a 256-VGPR floor.
  **The source cannot reach this variable.**

And one lever found and killed for free by h17's rule: FlyDSL 0.3.2 exposes `waves_per_eu`
and `maxnreg` compile hints and this job has never set one. The corpus, same backend, same
op, a backward body: `waves_per_eu=3` -> **-32%**, `=4` -> **-5x**, "on a latency-bound
kernel a forced step is negative in every instance measured here". Spill is a cliff (35
dwords = -19%) and we would spill ~228. Elsewhere the same mechanism produced **wrong
answers**. Cost to establish: zero GPU.

So the last round should **price the one remaining window rather than guess at it**: an
ablation probe over the ~125-instruction zero-matrix stretch in `k_dkdv`'s hot body, where
all of the residual 1.040x lives. An ablation gives an upper bound, never a candidate --
the corpus is explicit about that -- and an upper bound is exactly what is needed to decide
whether to keep going.

**If the window prices under ~5% at prod: stop optimising and consolidate.** Eleven rounds,
57.2 -> 481.0 TF/s (8.4x), six consecutive losing arms, and a 7-GEMM ceiling of 511.6.

Also: do NOT spend the slot on pool item `g35` as written. It names `lds_q`/`lds_do` for
deletion, but those are the **2-consumer** transposes (16 `tr()` each feeding 2 WMMA) --
the side the precedent measured at **-77%**. The winning side there was the 4-`tr()`-feeding-8
pair, `a_p`/`a_ds`. The pool entry conflated them. Three further discounts: the precedent is
gfx950; the incumbent already uses the **hardware** transposing load `DS_LOAD_TR16_B128`
(one instruction, not a naive round trip); and `DS_BPERMUTE` still goes through LDS hardware
across only 32 lanes.

---

## h19 — the first hardware counters, and what they kill (2026-09-23, before round 13)

rocprofv3 `--pmc` works on this machine and does **not** wedge the card. Six runs, all
rc=0, KFD clean, zero fault signatures. The ban came from one rc=134 whose fault callback
names an *eager Tensile GEMM*, not the profiler. `--kernel-trace` is separately broken
here (zero dispatches recorded), so use `--pmc`; it costs about 2 minutes per counter set.
Profile every round.

Measured on the round-12 shipment, prod shape:

| kernel | VGPR | LDS/WG | SQ_BUSY/SQ_CYCLES | ICACHE miss |
|---|--:|--:|--:|--:|
| `k_dkdv_0` | 376 | 70656 | 0.9977 | 0.0002% |
| `k_dq_0`   | 480 |  8704 | 0.9976 | 0.00003% |

**Three things this settles. Do not re-litigate them; they are measured.**

1. **The icache hypothesis is dead.** 171 misses in 86.5M requests. Round 10's
   instruction-count inversion needs a different explanation, and instruction *count* is
   no longer a reason on its own to prefer or reject a candidate.

2. **`SQ_BUSY/SQ_CYCLES = 0.998` proves nothing, and I nearly wrote that it did.**
   The first draft of this hint read it as "waves are resident, so the deficit is stall,
   not starvation." That is wrong. `SQ_BUSY_CYCLES` counts cycles in which the SQ has
   *any* wave at all. `k_dkdv` runs at **1 wave/SIMD** (see 3), so a single wave stalling
   on memory for the entire kernel would still read 0.998. The counter is trivially
   saturated at this occupancy and discriminates nothing.
   To actually separate stall from starvation, collect an *issue* counter
   (`SQ_INSTS_VALU` / `SQ_INSTS` against `SQ_BUSY_CYCLES`) or a wait counter
   (`SQ_WAIT_ANY`, `SQ_WAIT_INST_LDS`). Until then, **neither hypothesis is excluded**,
   and a candidate must not cite this number in either direction.

3. **WRONG AS FIRST WRITTEN. `rocprofv3`'s VGPR column on this card is HALF the ISA
   allocation, and correcting it inverts the conclusion.**
   The first draft read `k_dkdv` VGPR 376 / `k_dq` 480 off the counter CSV and concluded
   that LDS was throwing away half of `k_dkdv`'s occupancy and that `k_dq` had headroom
   to 3 waves/SIMD. Both are void.

   **The units rule, verified independently and decisively:** aiter's ASM forward code
   object declares `.vgpr_count: 1024` in its own metadata, and `rocprofv3` reports
   **512** for that same kernel in the same run. The column is
   `roundup(next_free_vgpr / 2, 8)`. It fits every kernel measured with no free
   parameter: 740→376, 960→480, 40→24, 1024→512.
   **Never read an occupancy step off that column. Multiply by 2 first.**

   Corrected, and this agrees with `findings/facts.md:1306`, which the campaign already
   measured in round 12 and which the first draft of this hint contradicted:

   | | ISA VGPR | waves/SIMD | VGPR/SIMD used |
   |---|--:|--:|--:|
   | `k_dkdv` | **740** | 1 | 740 / 1024 |
   | `k_dq`   | **960** | 1 | 960 / 1024 |
   | aiter ASM forward (the bar) | 1024 | 1 | 1024 / 1024 |

   So: **both our kernels are already in the bar's configuration** -- one wave per SIMD on
   a nearly-full register file. `k_dkdv` is VGPR-capped at 4 WG/CU *regardless of LDS*,
   which means the 48128 B LDS hole costs **zero occupancy** and the "LDS is holding back
   half our occupancy" reading is dead. There is no register slack to spend: `k_dkdv` has
   272 VGPRs free, `k_dq` has 64.

   Two consequences that close families rather than open them:
   - **`BLOCK_KV` 32→64 is LESS reachable now than when round 3 measured it 2.76x slower.**
     It costs +256 accumulator and +128 kf/vf VGPRs on top of g21's 128-VGPR prefetch
     that did not exist then, landing near 1110-1124 against a 1024 ceiling. It would
     spill, and a spilling build hangs this card.
   - **`k_dq`'s register axis is arithmetically closed.** 3 waves/SIMD needs ≤341, a 64%
     cut; even 2 waves needs ≤512, and the dQ accumulators (256, proven by the
     256-instruction bf16 store epilogue) plus the hoisted Q/dO fragments (256) are
     already 512 before one K or V byte is loaded.

4. **There is no elementwise overhead. I misread my own profile and nearly shipped it.**
   The first draft of this hint claimed 5.8% of per-step time ran in torch elementwise
   kernels around the three attention kernels. Wrong: reading the *full* kernel names out
   of the counter CSV instead of a 20-character truncation shows they are
   `normal_and_transform` (6x, = profdrv.py's six `torch.randn`), `bfloat16_copy_kernel`
   (5x, = its five `.to(torch.bfloat16)`), and one `AbsFunctor` + one `add` (= its
   `lse.abs() + 8.0`). They are the **profiling driver's one-time input construction**,
   every one of them launches before the measured loop, and `impl.py` contains no
   elementwise op at all. The real per-step split by `GRBM_GUI_ACTIVE` is
   **`k_dkdv` 66.7% / `k_dq` 32.6% / `k_delta` 0.7%**, and nothing else.
   *Lesson for reading counter dumps: aggregate by the FULL kernel name, and check launch
   order, before attributing a cost to the thing under test.*

**And a methodological one.** Each of these three overturned a belief that had been
written into this file and used to steer rounds. The pattern in every case was the same:
a real observation, generalised one step too far, then quoted as if it were the
observation. `TORCH_BLAS_PREFER_HIPBLASLT=0` was set on the strength of one error message
and survived twelve rounds because `sitecustomize.py:18` pinned the same value anyway, so
nothing ever contradicted it. **Before building on a fact in this file, check whether
anything since could have falsified it — and prefer facts that carry their own
measurement.**


---

## h20 — corrections to h19, and the counters that actually exist (2026-09-23)

h19 was written the same day and three of its four points needed correcting within hours.
They are corrected in place above; what follows is the part that would otherwise be
rediscovered the expensive way.

**The counters h19 told you to collect do not exist on gfx1250.**
`SQ_WAIT_INST_LDS` and `SQ_WAIT_BARRIER` are not defined for this chip, and neither are
`SQ_WAIT_CNT_ANY`, `SQ_ACTIVE_INST_ANY`, `SQ_BUSY_CU_CYCLES`, `SQ_LDS_BANK_CONFLICT` or
`SQ_INSTS_VMEM`. **Naming any one of them fails the entire pass**, so a round that trusts
h19's list gets nothing back and may read the empty result as "the probe does not work".

The authority is `/opt/rocm/share/rocprofiler-sdk/config.yaml` -- **not** the legacy
`basic_counters.xml`, which has no gfx1250 section at all. It defines 224 basic counters
for gfx1250, 187 of them gfx1250-only, so it is chip evidence rather than gfx11
substitution.

What exists and answers the open question:

| counter | what it buys |
|---|---|
| `SQ_WAIT_ANY` | wave-cycles blocked on anything, `s_waitcnt` included |
| `SQ_WAIT_INST_ANY` | wave-cycles blocked waiting for an **issue slot** |
| `SQ_WAVE_CYCLES` | the denominator both of those belong over |
| `SQ_VALU_WMMA_FLOP_BF16` | **issued** bf16 FLOP, directly -- no more inferring it from source |
| `SQ_INSTS_VEC32_VALU_WMMA`, `SQ_INSTS_ALL` | instruction mix |

The discriminator is the **gap between the two wait counters**. At 1 wave/SIMD there is
nobody to contend with for an issue slot, so `SQ_WAIT_INST_ANY` should come back near
zero; a large `SQ_WAIT_ANY` beside it proves the wave is resident and blocked on data,
which is the thing `SQ_BUSY_CYCLES` could never show.

**Per-pass budget is per hardware block, not global**: SQ 8 (SQC shares that budget),
TCP 8, GRBM 2. Pack to exactly 8 SQ counters per pass.

**Normalisation.** `SQ_WAVES` is already a whole-GPU total and needs no divisor --
`k_dkdv` read 8192, exactly grid 262144 / 32. But the cycle-type SQ counters divide by
**256, not 1024**: /256 gives 1.00 GHz against this card's 1050-1100 MHz, while /1024
would imply an impossible 251 MHz. h19's hand-division by 1024 SIMDs was wrong.
**Prefer ratios of two SQ counters, which survive the divisor being wrong.**

**And `.co` files DO disassemble here.** `/opt/rocm/llvm/bin/llvm-objdump -d` returns
8253 lines for the backward bar and 11,846 for the forward. `planner.log:989` and h18's
caveat both say otherwise and are both wrong. Twelve rounds reverse-engineered a box that
was never locked.

### What this leaves for round 13

Four candidates were priced against the corrected numbers. Three closed:

- **the LDS hole** -- costs no occupancy, and g39 *was* measured in isolation: prod
  +0.59% / +0.21% against a self-reported 0.86% floor, i.e. **NULL**. Round 12's entire
  +4.91% came from g40, the `k_dq` dispatch-order change. Also: the "64 KB segment served
  by two 256 B/cycle read ports" story behind g39 is **UNVERIFIED as hardware** -- the
  project's own gfx1250 ISA readout has a full LDS section (320 KB, 2048 B granularity,
  64 banks x 4 B) that never mentions a segment or a read port, and the corpus's own
  portability table already marks it "Unknown ... Do not assume".
- **a fatter wave** -- `BLOCK_KV` 64 spills (above); 2x query-loop unroll measured -7.8%
  (g28), deeper prefetch -14.6% (g27), the half-prefetch probe -61.5% (g32), and K/V
  resident in registers has shipped since round 1.
- **`k_dq` registers** -- arithmetically closed (above).

One is open, and it is the one difference from the bar that has never been tested:
**the bar moves every global→LDS byte by TDM and issues zero `buffer_load`.**
Our `k_dkdv` hot body is exactly 36 `buffer_load` (32 b128 + 4 b32), counted in the
shipped round-12 object. The record that closed TDM does not bind: its number is a
**gfx950** A/B of `buffer_load ... lds` -- a different instruction -- attributed to
+16 `s_barrier` and -32 AGPRs, and **neither can happen here**: gfx1250 has no AGPRs, and
our workgroup is a single wave whose `s_barrier` count is already 0, so there are no
barriers to add. flydsl 0.3.2's TDM 2D surface is complete, and its padding mode
reproduces our g16-tuned `X_ROW_B = 272 B` exactly (`pad_interval=128, pad_amount=8`).

**Expected sign: genuinely UNKNOWN.** Say so in the act, and make it the round's
falsification target rather than a prediction. The precedent cuts both ways -- deleting
traffic on `k_dkdv` paid +11.0% once (g09) while the identical deletion on `k_dq` paid
nothing (g12).

### h20 addendum — the counter whitelist for this machine (validated, 2026-09-23)

Everything above about which counters "exist" was still one layer too optimistic, and the
failure mode is the worst kind: **`--list-avail` lists the counter, `rocprofv3` accepts it,
the pass exits 0, and it silently returns zero.**

This was caught only because the control was aiter's ASM forward, a kernel that provably
issues millions of WMMA. `SQ_VALU_WMMA_FLOP_BF16` came back **0** for it, in the same pass
where `SQ_WAVES` read a correct 16384. Without that control the conclusion would have been
"the bar issues no bf16 matrix work" -- absurd, and yet it would have looked exactly like
data.

**Use only these nine. They were validated against a known-nonzero control:**

```
SQ_WAVES   SQ_CYCLES   SQ_BUSY_CYCLES   SQ_ITEMS
SQC_ICACHE_REQ   SQC_ICACHE_HITS   SQC_ICACHE_MISSES   SQC_ICACHE_MISSES_DUPLICATE
GRBM_GUI_ACTIVE
```

**Accepted but silently zero -- do NOT use, and do not read a zero from them as a finding:**
`SQ_WAVE_CYCLES`, `SQ_INST_CYCLES_VALU_WMMA`, `SQ_INSTS_VEC32_VALU_WMMA`,
`SQ_VALU_WMMA_FLOP_BF16`, `SQ_VALU_WMMA_FLOP_FP16`, `SQ_INSTS_SENDMSG`.

**Rejected outright** (naming one fails the whole pass): every wait and stall counter --
`SQ_WAIT_ANY`, `SQ_WAIT_INST_ANY`, `SQ_WAIT_INST_LDS`, `SQ_WAIT_BARRIER`,
`SQ_BUSY_CU_CYCLES`, `SQ_LDS_BANK_CONFLICT`, `SQ_INSTS_VMEM`, `SQ_INSTS_ALL`.
Note this also kills the `SQ_WAIT_ANY` / `SQ_WAIT_INST_ANY` pair that the round-13 pricing
recommended: that recommendation was read out of
`/opt/rocm/share/rocprofiler-sdk/config.yaml`, which is **not** the table the container's
`rocprofv3` consults. Ask the tool, not the config file.

**Therefore: "is the wave stalling or starving" cannot be answered by counters on this
machine.** Stop proposing it. Drop it as a decision input rather than leaving it open as
something a future round might resolve.

**What per-round profiling is still worth**, and it is not nothing: every `--pmc` pass
returns the kernel descriptors for free -- `VGPR_Count`, `SGPR_Count`, `LDS_Block_Size`,
`Scratch_Size`, `Workgroup_Size`, `Grid_Size`. That is how the O_VARIANT result was
confirmed to be two genuinely different kernels rather than one cached build, and it is
the cheapest spill check available (`Scratch_Size` != 0 means the build spilled, and a
spilling build hangs this card). Run one pass per round with the nine counters above and
read the descriptors.

One more nail in the icache coffin: the ASM bar's own miss rate is
50137 / 339246799 = **0.0148%**, two orders of magnitude *worse* than our `k_dkdv`'s
0.0002%, and it is still 1.4x faster.

### h20 addendum 2 — `op/current/` is restored from the champion at every round start

Patching `op/current/` by hand does not survive. Starting round 13 restored
`op/current/_env.py` from the round-12 archive, mtime and all, silently discarding both an
edit made an hour earlier **and its `.bak` beside it**, and `rounds/013/op/` was seeded
from the restored copy.

Two consequences:

1. **A hand-patch to `op/current/` is transient.** It lives until the next round starts.
   Anything that must persist belongs in the job spec's `runtime.env`, in
   `op/baseline/`, or in a file the round-start restore does not touch -- and it should be
   verified *after* the round starts, not before.
2. **It can silently desynchronise the arms.** `op/baseline/` is *not* restored, so a patch
   applied to both `current` and `baseline` ends up applied to only one, and the candidate
   and baseline arms then run in different environments. That is a confound manufactured
   by a housekeeping edit. When this happened on 2026-09-23 the fix was to revert
   `baseline` too, so all three arms shared one environment for the measurement, rather
   than to re-patch `current` mid-flight.

**Rule: never hand-edit `op/current/` while a round is starting or running. If an
environment change must reach a measured arm, put it in the spec and let the round pick it
up, and check every arm has it.**

---

## h21 — round 13 won on the wrong shape, and said why the right one is still open

Round 13 is accepted, `gain 1.3704`, score **0.551 → 0.704**. Good round. But read the
shapes before quoting the number:

| shape | this round | same-session champion | vs champion | vs the bar |
|---|--:|--:|--:|--:|
| fast | 34.38 | 19.72 | **+73.1%** | 0.669 |
| proxy | 417.15 | 318.99 | **+31.0%** | 0.734 |
| **prod** | **497.39** | 495.26 | **+0.38%** | **0.699** |

**The entire gain is fast and proxy. prod did not move, by design** -- the host picks
`nsp=1` at prod, so prod runs the completely unmodified path and is bitwise identical.
The framework accepts on the arithmetic mean over three shapes, so a large small-shape win
carries a round on its own. That is not cheating and g42 is a real result, but it is worth
being blunt about the consequence: **prod has gone 0.693 → 0.699 of the bar across two
rounds.** The job exists to close that gap.

**So: name prod in the hypothesis.** If a candidate is expected to be a no-op at prod, say
so in `expected` *and say what the round is for* -- closing a family, building an
instrument, winning a shape that matters for its own sake. All three are legitimate. What
is not legitimate is letting the mean hide it.

*(And when comparing, use `champion_tflops` -- the same-session rebuild -- not
`best_round_tflops`, which is archived from another session. Reading round 13's prod
against the archived 499.1459 makes it look like a 0.4% regression when the same-session
comparison says +0.38%. This campaign has already lost a conclusion to exactly that
substitution.)*

### Round 13's most valuable output is not g42. It is the census.

The grid census (zero card time, zero code change, `1-opt/raw/census.txt`) established
that **prod's dispatch efficiency is already 100% for both hot kernels** -- g03 and g40
closed that door in earlier rounds. Round 13's own words:

> prod 剩下的不是空转的 CU，是在跑的 wave 在等

**prod is latency-bound, not parallelism-bound.** That matters because the counters on
this card cannot answer the stall-versus-starvation question at all (h20), and the census
answers it from the other side, by model rather than by counter.

**This makes TDM more motivated than it was this morning, not less.** TDM replaces 36
synchronous `buffer_load` per iteration with an asynchronous global→LDS engine drained by
a single `s_wait_tensorcnt`. That is a latency-exposure mechanism, and latency exposure is
now what prod is measured to be limited by. It is still the one structural difference from
the bar that has never been tested, and the record closing it still does not bind
(gfx950, a different instruction, causes that cannot occur here -- see h20).

### Two more results worth keeping

- **`num_xcc = 8`, read from `/sys/class/kfd/kfd/topology/nodes/*/properties`.** The corpus
  calls the XCD count undocumented for gfx1250. It is not; KFD prints it. Also
  `simd_count 1024`, `gfx_target_version 120500`.
- **The per-XCD L2 locality family is closed.** g43 cut `k_dq`'s per-XCD K/V footprint to a
  quarter and measured NULL (six readings 0.9928-1.0065, inside the session's own 1.8%
  floor), against a corpus entry promising +7%. Either the XCD mapping is not round-robin
  or the LLC is not per-XCD private; one zero-card-time check next round decides which.
- **h8 takes another hit.** g43 removed bytes and gained nothing; g42 *added* an fp32
  workspace round trip and gained 30%. **Byte counts still do not predict time on this op.**
  Stop using them to price candidates.

---

## h22 — the first properly-classified wedge, and it is a THIRD class (2026-09-23 13:35)

Two things had to be fixed before this could be written at all.

**`dmesg` was unreadable all day.** `/proc/sys/kernel/dmesg_restrict` was `1`, so `dmesg`
returned a single line. **Every "zero fault signatures" check in this campaign's logs
today was grepping empty output** — not evidence of health, evidence of nothing. If a
check for wedge signatures ever comes back clean, first confirm the source is readable:
`dmesg | wc -l` should be in the thousands. The fix needs root:
`sudo sysctl -w kernel.dmesg_restrict=0`.

**And the ring buffer does not survive a power cycle.** The evidence was recovered from
`/var/log/kern.log`, which persists (5.4 GB here, so `tail -n 400000` it, never grep the
whole file). `journalctl -k -b -1` did NOT have it.

### The signature

```
13:35:09  first fault, pid 138402 vmid 3 pasid 874, then: IH ring buffer overflow
13:35:42  fault storm, pid 138956 vmid 4 pasid 876, on AID0.XCD0, XCD1 AND XCD2
            in page starting at address 0x000076ac0e03c000 from IH client 27 (GC_UTCL2)
            Faulty UTCL2 client ID: TCP (0x8)
            MORE_FAULTS: 0x1   WALKER_ERROR: 0x0   PERMISSION_FAULTS: 0x3
            MAPPING_ERROR: 0x0   RW: 0x0   FED: 0x0
13:35:43  MES(0,0) failed to respond to msg=REMOVE_QUEUE
13:35:45  MES(0,0) failed to respond to msg=SUSPEND  /  failed to suspend all gangs
13:35:55  MES might be in unrecoverable state, issue a GPU reset
```

**This is neither recorded class.** Against class A (memory aperture) all three
discriminants fail: the address is high, not a sub-4 GB truncation; `RW: 0x0` is a
**read**, not the recorded write; and `copy_context_work_handler` never appears. Against
class B (TLB/queue) it fails too: class B's whole point is that `INVALIDATE_TLBS` times
out with **no preceding memory fault**, and here a fault storm precedes MES by one second.
*(This morning's 09:21 wedge — the one the operator power-cycled for — WAS class B: pure
`INVALIDATE_TLBS` → `failed to suspend all gangs` → unrecoverable, no fault before it.)*

### What it means, read field by field

- **`Faulty UTCL2 client ID: TCP`** is the vector-memory pipe. So this is a **kernel's own
  load**, not a copy engine and not a host transfer.
- **`RW: 0x0`** — a read.
- **`WALKER_ERROR: 0x0` and `MAPPING_ERROR: 0x0`** — the page-table walk *succeeded*. The
  page is mapped. The access was simply not permitted.
- **`MORE_FAULTS: 0x1`, faults on three XCDs, and an `IH ring buffer overflow`** — not one
  stray lane, a storm across the device.

**Diagnosis: a candidate kernel read out of bounds into a mapped-but-unreadable page, and
the resulting fault storm left MES unable to drain its queues.** The GPU reset then could
not complete, which is why only an AC cycle recovered it.

### Two consequences

1. **This is the failure mode a candidate build produces, so it will recur.** The round-14
   attempt that caused it had five variants under `_scratch` (`armA`, `armB16`, `armB32`,
   `armAB`, `red`); one of them read past the end of a buffer. **Screen offline
   (`COMPILE_ONLY=1`) before any dispatch, and bounds-check every new index expression** —
   especially any candidate that changes a block size, a split count, or a grid mapping,
   because those all rewrite address arithmetic.
2. **`poison_allocator` did not catch it, and we already knew it could not.** It was
   recorded as ineffective at prod (largest poison block 64 MiB against a 256 MiB
   request). That was filed as a prerequisite for a later work item. It has now cost a
   power cycle. **Fix it before the next round that changes address arithmetic**, not
   after.

### The check that actually works, and costs nothing

`timeout 20 docker exec <container> true` — when the card wedges this way, `docker exec`
stops responding before anything else the operator can see. It needs no root, no dmesg,
and no GPU call. It is now in the round monitor.

### h22 addendum — a fault is not a wedge. Only the interrupt-ring overflow is.

*(This paragraph was written once with the wrong discriminant and corrected within ten
minutes, by the monitor built from it falsifying it on its next tick. The first version
named `MORE_FAULTS` and multi-die faults. Both appear in the survivable case too. What
follows is the version that survived a direct comparison of the two events.)*

Forty minutes after the 13:35 wedge, round 14 faulted again on a fresh boot and **the card
survived**: `docker exec` answered, the GPU stayed at 100%, the loop kept running, and
every MES line in `dmesg` was boot-time initialisation. Only the faulting candidate's
process died.

| | 13:35 — **WEDGED** | 14:0x — **SURVIVED** |
|---|--:|--:|
| `no-retry page fault` records | 11 | 6 |
| `MORE_FAULTS: 0x1` | **11 of 11** | 3 of 6 |
| **`IH ring buffer overflow`** | **6** | **0** |
| dies faulting | XCD0, XCD1, XCD2 | five, across AID0 and AID1 |
| `PERMISSION_FAULTS` | 0x3 | 0x5 |
| `RW` | 0x0 (read) | 0x1 (write) |
| MES | `failed to respond` → unrecoverable | boot-time lines only |
| cost | **power cycle** | one dead process |

**The single discriminant is `IH ring buffer overflow`.** Everything else is present on
both sides:

- **`MORE_FAULTS: 0x1` does NOT predict a wedge**, in any form. The first survivable
  burst had it on 3 of 6 records; a second survivable burst forty minutes later had it on
  **8 of 8**, the same 100% as the wedge. Proportion is not a discriminant either.
- **Faults on several dies do not predict a wedge either.** The survivable burst hit five.
- **`PERMISSION_FAULTS: 0x5` with `RW: 0x1`** — the signature recorded as the
  memory-aperture class — is the one that **survived**. That signature names the class of
  bug, not the severity.

### The root cause is the fault ARRIVAL RATE, and it is measurable

Timestamping the faults inside each burst makes the mechanism quantitative:

| | median interval between faults |
|---|--:|
| **wedged** (11 faults) | **0.016 ms** |
| **survived** (14 faults, two bursts) | **96.8 ms** |

**Six thousand times faster.** The driver drains a fault every ~100 ms without noticing;
at one every 16 µs it cannot keep up, the interrupt ring overflows, fault records are
lost, MES can no longer drain its queues, and the driver's own GPU reset cannot complete
either — which is why only an AC cycle recovers it.

So `IH ring buffer overflow` is the *observable consequence*; arrival rate is the cause.

**One caveat, stated rather than papered over.** Four separate survivable bursts on
2026-09-23 measured 93.23, 96.79, 96.99 and 96.89 ms — **too regular to be a kernel's own
fault rate.** That cadence is almost certainly a driver-side reporting throttle, so the
~97 ms figure measures the throttle, not how fast the kernel actually faults. The wedge's
0.016 ms is then better read as *faults arriving faster than the throttle can be applied*
— which is consistent with the overflow but means the 6000x is a ratio between two
different things, not a clean rate comparison.
**What is safe to rely on: the overflow itself, and the observation that the survivable
bursts all sat at the throttle while the wedge did not.** Do not quote the 6000x as a
measured fault-generation ratio.

This also says what kind of bug is dangerous. A ~97 ms cadence is one fault per launch —
a single bad address, raised once, process killed. A 16 µs cadence is **many waves
faulting concurrently inside a hot loop**. So an out-of-bounds address computed
**per-iteration or per-wave** is the one that takes the card down; an out-of-bounds
computed once in a prologue or epilogue usually just kills the process.
**Weight the review of a candidate's address arithmetic accordingly.**

**So the alarm condition is:** `IH ring buffer overflow`, or any
`MES(...) failed to respond` / `failed to suspend all gangs` / `unrecoverable`, or
`timeout 20 docker exec <container> true` failing to answer. **Not** the word "fault", and
**not** `MORE_FAULTS` — a monitor keyed on either will cry wolf on every candidate that
runs past the end of a buffer, and this campaign produces those regularly.


### h22 addendum 3 — the survivable faults are NOT a candidate's bug. I attributed them wrongly.

h22 said "a candidate kernel read out of bounds" and told future rounds to review candidate
address arithmetic. **Round 14's own per-arm isolation refutes that** for the survivable
class, and its evidence is far stronger than the inference I made from "faults appeared
while a round was building candidates":

| arm, in its own process | faulted? |
|---|---|
| `armA` alone | **yes**, at `0x78f2372e3000` |
| `armB` alone | **yes**, at `0x732ffd5b3000` |
| **`armAB` — the merge of both** | **no** |
| `armA32` | no |
| `cur` — the unmodified champion | no |

A bug in armA's addressing cannot vanish when armA is merged with armB. And three separate
faults landed at **three different addresses**, every one of them **at the fast→proxy shape
transition**, with `GCVM_L2_PROTECTION_FAULT_STATUS_LO32: 0x00D040A1` each time and no ring
timeout and no GPU reset. A sweep containing only `beat` plus two copies of `cur` faulted
too, while the same set's next sweep was clean.

**So the signal is the shape transition, not the kernel under test.** Something at the
boundary where one shape's tensors are freed and the next shape's are allocated — a
lifetime or descriptor-reuse problem in the harness, not in a candidate.

Round 14 also ruled out two tempting explanations:
- **Not the reference forward.** All three refcache entries hit; `forward_reference`'s fp32
  Tensile dispatch never ran.
- **Not a static out-of-bounds.** `r1.i7.g07` gave every descriptor a real byte extent, so
  an out-of-range access returns zero **without faulting** — and if it hit live data the
  SQNR gate would fail first.

**What this changes:** stop sending rounds to audit candidate address arithmetic on the
strength of a survivable fault. Look at the shape-transition boundary in `benchmark.py`
instead. h22's "an out-of-bounds computed per-iteration is what takes the card down"
remains a reasonable *rule of thumb about severity*, but it is no longer supported by any
observed case here.

**Still unattributed: the 13:35 wedge.** Its signature differs from these
(`PERMISSION_FAULTS: 0x3` and `RW: 0x0`, a read, against `0x5`/`RW: 0x1`, a write, here),
it was the only event with an interrupt-ring overflow, and no isolation run exists for it.
Do not assume round 14's finding explains it.

### h22 addendum 4 — I do not have a wedge predictor, and I should stop inventing one

This section has now named four discriminants and measurement has falsified every one.

| named as the discriminant | falsified by |
|---|---|
| `PERMISSION_FAULTS: 0x5` + `RW: 0x1` (aperture class) | that exact signature survived, repeatedly |
| `MORE_FAULTS: 0x1` | present in survivable bursts, including 8 of 8 |
| faults across several XCDs | a survivable burst hit five |
| **`IH ring buffer overflow`** | **8 overflows on 2026-09-24, card fine, 0 MES failures** |

**So: there is no known predictor.** The only reliable signal is the MES failure sequence
itself — `MES(...) failed to respond`, `failed to suspend all gangs`,
`might be in unrecoverable state` — and that is the wedge happening, not a warning of it.

**Operational rule, which does not need a predictor:** alarm on the MES lines and on
`timeout 20 docker exec <container> true` failing to answer. Treat page faults, overflows
and `MORE_FAULTS` as information, not as alerts. Do not stop a round for them.

### Non-fatal page faults are normal here, and they do not corrupt results

Measured 2026-09-24 while probing the fp32 reference at the production shape:

- Three consecutive runs completed **3/3**, and the five output hashes were **identical**
  across repeats in one process and across separate processes
  (`o=7b0249a1cd82 lse=6fa4e0af2ed0 dq=bc919da8bcdb dk=2aee8288433b dv=d3f40069856d`).
- **Ten GPU page faults were logged during those same runs.** Exit code 0, correct results.
- The fault addresses are tiny and sub-4 GB: `0x47000`, `0x53000`, `0x55000`, `0x74000`,
  `0x83000`, `0x87000` — the *aperture* pattern, not a kernel's own high user VA.
- Occasionally one of these IS fatal: a run died with
  `Memory access fault ... on address 0x87000`, exit 134, at an address seen non-fatally
  in other runs.

**Consequences for how to read a run:**
1. **A page fault in `dmesg` is not evidence that the run was wrong.** Check the exit code
   and the output. Results were bitwise correct through ten of them.
2. **Conversely, a clean exit is not evidence of no faults.** They were invisible from
   inside the process.
3. This is the same small-address family the campaign filed as the memory-aperture class,
   and it appears in the *reference* path with no candidate kernel involved. It is
   plausibly the same defect as the sweep's fast→proxy transition faults. Do not attribute
   it to a candidate without an isolation run.

### A harness lesson that cost four commands

A probe script was named `/tmp/bisect.py`. With `/tmp` as `sys.path[0]`, it **shadowed the
standard library's `bisect`**, which torch imports transitively, and every run failed with
`partially initialized module 'torch' has no attribute 'Generator'` — an error that reads
like a GPU or environment problem and is neither. Renaming the file did not fix it; the
stale copy had to be deleted from the container. **Never name a probe script after a stdlib
module** (`bisect`, `random`, `types`, `queue`, `select`, `signal`, `copy`, `token`…).

---

## h23 — RETRACTION: TDM has no target. And what round 15 should actually do. (2026-09-24)

### I pushed TDM for two days and it was wrong

h20 called TDM "the one difference from the bar that has never been tested" and h21 said
round 13's census made it "more motivated, not less". **Both are withdrawn.**

My reasoning had one half right and skipped the other half entirely. I checked the corpus
entry that *closed* TDM and correctly found it misfiled — it is a **gfx950** A/B of
`buffer_load ... lds`, a different instruction, attributed to +16 `s_barrier` and −32 AGPRs,
neither of which can happen on gfx1250. **But I never checked whether the target still
existed in our kernel.** `facts.md:669` says "no target left", and I dismissed that half as
"a statement about our BLOCK_KV=32 shape, not a judgement on the mechanism". That dismissal
was the error.

What the ISA actually shows:

- **All 32 `buffer_load_b128` in `k_dkdv`'s hot body are global→REGISTER loads, and those
  registers are dual-use**: stored to LDS *and* shuffled straight into the WMMA A-operands
  (`kernels.py:270-281`, `:360-366`). That is exactly what **g09 built**. There is no
  global→LDS staging burst left to replace.
- The remaining **4 `buffer_load_b32` are scalar LSE/delta** feeding VALU (`kernels.py:372`),
  not stageable tiles.
- **K/V are hoisted outside the loop entirely** — not in the hot body at all.
- Adopting TDM therefore means **undoing g09** and abandoning **g21**, whose measured
  load-to-use distance is already **223–419 instructions** (`isa_lat_ship.txt`). That is
  TDM's entire mechanism — issue early, drain late — already delivered synchronously, and
  measured at +8.3%. TDM cannot buy distance that already exists.
- **aiter's call shape does not transfer.** Their TDM→LDS hop exists to broadcast one tile
  to **8 waves** (`fmha_b16_buffer_managers.py:976`). `k_dkdv` is a **single wave32
  workgroup** with nobody to broadcast to. The one aiter site using `num_warps=1` is a wave
  copying its own private tile through LDS — precisely the redundant round trip g09 deleted.

**Do not spend an arm slot on TDM.** If a round wants a verdict, record the static finding
and move the entry to `dead_ends.md`. It costs nothing.

*Generalisable lesson: refuting the reason something was closed does NOT reopen it. Check
that the target still exists before rebuilding a case on a bad closure.*

### The shape-switch fault is FIXED, and it was the harness

`benchmark.py` reloaded the impl module **once per shape** inside `measure()`. `load_impl`
replaces `sys.modules` (`ut/common.py:99-103`), so each boundary dropped the previous
shape's flydsl chain → `GpuJitModule.__del__` → `hipModuleUnload`
(`jit_executor.py:102,107`) — and because Python modules are reference **cycles**, that
unload landed at the next generational GC, possibly mid-launch of the next shape.
Intermittent, boundary-only, different address each run. Exactly the signature.

The control was already in the tree: `validation.py:117` loads the impl **once**, loops all
three shapes with *more* allocator churn (`:169` `empty_cache`), and never faults.

Fixed by hoisting `load_impl` out of the shape loop. Verified: the same
`beat + two copies of cur` sweep over `fast,proxy,prod` that used to fault now runs with
**zero new faults**.

**Round 14 nearly convicted two innocent arms over this.** When a fault appears, check
whether the harness changed something at that moment before suspecting the candidate.

### Round 15

**`g46`, `k_dq` only.** `k_dq` is **32.6% of prod time and has never had a candidate of its
own in 14 rounds**. Its hot body spends **128 `v_pk_mul_f32` per iteration** = 4 fp32
multiplies per element, **two of which are multiplications by loop-invariant scalars that
fold out by exact algebraic identity** (convert to base-2: `scale` and `lse` become a
per-row bias, and the trailing `* scale` on `ds` disappears).

⚠ **This fold is legal in `k_dq` but NOT in `k_dkdv`**, because there `pf` is dual-use and
also feeds dV (`kernels.py:397`). `g47` is the partial version for `k_dkdv` that keeps the
trailing `* scale`.

The asymmetry is already measured: **round 6 found that deleting the same share of a loop
body was worth +8.2% in `k_dq` and +1.1% in `k_dkdv`** (`facts.md:945`). That predicts g46
pays and g47 mostly does not — which is also the round's falsification target.

And note what this says about the diagnosis: round 13's census concluded prod is "waves
waiting". But `k_dq`'s two largest matrix-free windows have **`vmem=0` and `lds=0`, 91% and
97% VALU** (`windows.txt`). In those windows there is nothing to wait *for*. That is
**issue serialisation, not latency exposure** — a different bottleneck from `k_dkdv`'s, and
another reason TDM could not have helped prod.

---

## h24 — HARD RULE: a candidate that loses at prod does not ship (2026-09-24)

### What happened in round 15

`r15.i2.g47` was measured at **prod `vs_champion` = 0.9688**, a **3.1% regression** on the
production shape, and it was **shipped and promoted to champion anyway**. The operator had
to revert `best_round` by hand.

It was not a mistake by the round. It followed the rules exactly:

```
gain = mean(1.0600, 0.9816, 0.9688) = 1.0035  >  1.0 + min_gain(0.0)   -> accept
lowest shape 0.9688  >  NOISE_BAND 0.95                                -> no veto
```

**`fast`'s +6.0% paid for `prod`'s −3.1%, because acceptance is an arithmetic mean over
three shapes and `min_gain` is 0.0.**

`min_gain: 0.0` was justified in the spec by "the gap to the anchor is ~93x, so round gains
are expected as integer factors". **That justification died several rounds ago.** Gains are
now single-digit percentages and this machine's same-code spread is about 1.5%. The floor
has not been raised because raising it needs `resume --config`, which bumps `spec_version`,
re-runs `op_setup`, and would overwrite the hand-written `refcache`, `poison_util` and
`benchmark.py` fixes. So the floor stays, **and this rule stands in its place.**

### The rule

> **No candidate may be declared `shipped: true` if its `prod` `vs_champion` is below
> 1.0 − (the round's own self-reported noise floor).**
>
> Not below 0.95. Not "acceptable because the mean clears". **Below the floor at prod
> means it does not ship**, whatever `fast` and `proxy` did.

If an arm wins big on `fast` or `proxy` and loses at `prod`, that is a **real and reportable
result** — record it in `facts.md`, keep the instruments, and say which shape it belongs to.
It is not a shipment.

### Why this matters more than one round

`op/current` is what every later round is measured against. A champion that is 3.1% worse
at prod **lowers the bar permanently**: every subsequent round's `vs_champion` is inflated
by that much, and the regression never appears again in any table. It is the one kind of
error this ledger cannot self-correct, because the evidence of it is destroyed by the act
of committing it.

### Two things round 15 did right, and they should be kept

1. **It killed `g46` by reading the ISA instead of building it.** Its premise was that
   `.LBB0_8`'s waits were a coarse all-drain; the ISA shows they are already a descending
   staircase (`0x22 → 0x21 → 0x20 → 0xa → 0x2 → 0x0`). The round wrote *"I had misread my
   own instrument"*. That is exactly the right move and it cost no card time.
2. **It found the real blocker for the whole family.** `g47rev` showed the
   `ds_store x40 → ds_load x40 → WMMA x32` tail of `k_dkdv`'s hot body is locked by a
   **true LDS RAW dependence**. The scheduler hook works, is free, and genuinely moves the
   schedule — but there is nothing it can legally move *there*. Any future candidate aimed
   at that tail must break the dependence, not reschedule around it.

### And the bookkeeping failure that ended the round

Round 15 was recorded as **failed** because it dropped `g46` and `g47` without writing
either into `facts.md` or `dead_ends.md` and without declaring a merge. **Every id that is
built or priced must be closed somewhere**, including — especially — the ones killed
statically. That is what makes the next round cheap.

---

## h25 — deep rounds enabled, ATT measured, and three assumptions inverted (2026-09-24)

Round 17 is the campaign's first DEEP round. Getting there required measuring four things,
and **three of them came back opposite to the assumption they were built on.**

### ATT does not wedge this card. It also cannot see our kernels.

Probed directly, card otherwise idle, four runs. **Zero `MES ... failed to respond`, zero
`suspend all gangs`, zero `unrecoverable`, ZERO page faults, `docker exec` responsive
throughout.** The ban was inherited from **PC sampling**, which is a different mechanism —
a per-wave interrupt that writes wave state to a save area — and does not transfer.

But the operative fact is the second one:

| workload | result |
|---|---|
| a trivial `torch` elementwise kernel | **34 files, 11 MB, a real `.att` trace + `ui_output_*_dispatch_*/`** |
| this op's FlyDSL kernels, `--kernel-include-regex k_dkdv` | 9 files, **all 9 are `*_code_object_id_*.out`**, no trace |
| the same, **no** regex filter | identical: 9 files, all code-object dumps |

The filter is not the problem. **ATT captures nothing for FlyDSL JIT-compiled kernels on
this stack**, minutes after capturing a torch kernel in the same container. This also
closes the 2026-09-11 mystery: those runs produced ELF-dumps-only too. They were not
"unarmed" — they hit this same limitation.

**So: do not spend a deep step on ATT for this op.** `status: skipped`,
`reason: att_captures_no_flydsl_kernels`.

Two traps found while probing, both of which impersonate a GPU failure:
- **`--att-buffer-size` is in BYTES, not MB.** `64` aborts with
  `F core.cpp:108] Invalid buffer size: 64`, and rocprofv3 then sits in its own SIGABRT
  handler until killed. It reads **exactly** like a hung card and is not one. Use `67108864`.
- **Verify a capture ARMED before calling it a pass.** A directory holding only
  `*_code_object_id_*.out` is a NULL result, not a clean one.

### rocprof-compute: installable, and deliberately not installed

`/opt/rocm` points at a build with **no gfx1250 support at all**; `/opt/rocm-10.1.0a20260811`
beside it has full support. **Both report VERSION 3.8.0** — so a version check is not a
capability check. Profile mode is stdlib-only, so the hipBLASLt-style copy-to-`$HOME` would
work with zero installs.

It is still **not being used**, for a reason that matters more than availability: this
card's rocprofv3 defines **51 counters** for gfx1250, and mapping the gfx1250 panels onto
them gives **SoL 96→13** (two of which are the known silent-zero WMMA FLOPs), **LDS 152→1**
(only `GRBM_GUI_ACTIVE`), and **L2 / EA / UTCL1 / TXD → 0**. For an LDS-bound `k_dkdv` the
panels we would actually read are empty. Worse, `soc_base.py:660-720` **overrides the
runtime counter table** via `ROCPROFILER_METRICS_PATH` with its own 1811-entry table — so
it bypasses the 51-counter guard and requests counters the driver accepts and returns zero
for. That amplifies the silent-zero hazard **with no control**.

One positive by-product: those 51 definitions **independently confirm h20's whitelist** —
the 9 working and 6 silently-zero counters are all present, and every counter rejected
outright is absent from the table.

### The stock deep round would have aborted at step 2

`01_select` is a SPINE step and mandated `rocprofv3 --stats --kernel-trace`, which records
**zero dispatches** on this stack. A stock deep round therefore burns its setup and dies
having measured nothing. It now ranks kernels from a `--pmc` pass instead (timestamps and
`GRBM_GUI_ACTIVE`, two independent orderings, disagreement reported).

Also fixed, all in `deep_loop/profiling/`: the only counter name in the tree was
`SQ_VALU_MFMA_BUSY_CYCLES` — **gfx1250 has WMMA, not MFMA**, and one rejected name fails
the whole pass; three passages told the agent to `pip install` into the shared container,
restart it, or download a decoder, against the spec's `owned: false` and under
`bypassPermissions`; `05_power_wall` drove an 8192³ hipBLASLt GEMM, the path that has
faulted this card; and **`06_bound` — the step that forms the round's verdict — demanded a
roofline built from byte counters that do not exist at any level on this part.** It would
have failed or fabricated.

*The diff is not in this repo.* op-evolve belongs to someone else; the changes live in its
working tree and are saved as
`output/0924__flydsl/deep-enablement/deep_loop-trim.patch` so a `git checkout` there cannot
silently erase them.

### And a warning about the fix that nearly poisoned everything

Making matplotlib available (`pip install --target`) pulled **numpy 2.5.3**, which shadowed
the container's **2.4.1 that torch 2.11 was built against**. With that directory on
`PYTHONPATH`, every subsequent measurement would have run against a different numpy. It was
caught by an explicit adversarial check, pruned, and re-verified: `import numpy` resolves to
`/opt/venv`'s 2.4.1 and torch imports. **Any `--target` install must be audited for packages
the container already provides.**

Also: **`runtime.env` does not exist in op-evolve.** The docker runner wraps commands with
no `-e`, and `load_spec` silently drops unknown keys. Adding `runtime.env:` to a job spec is
a silent no-op — the exact failure class this campaign keeps losing days to.
