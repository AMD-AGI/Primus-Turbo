# Round 25 -- the decision

**Selected: `r25.i3.g77` -- reverse `k_dkdv`'s q scan so every co-resident workgroup
traverses the q axis in step.**
One arm. Four instruments, all taken this round. One change filed to the pool with the two
measurements it needs taken now rather than deferred.

Transcript: [`dialogue.md`](dialogue.md). Candidate entries: [`candidates.yaml`](candidates.yaml).
Machine-readable decision: [`plan.yaml`](plan.yaml).

---

## What was chosen, and why it was chosen over the bigger idea

The larger claim this round is `r25.i2.g76`: materialise `dS` in bf16 from `k_dkdv` so `k_dq`
becomes a single GEMM. It is the only surviving route to the 7-vs-5 GEMM gap, and that gap is
the whole prod deficit -- `1.4076x` issued over algorithmic
(`1-profiling/6-bound-analysis/analysis.md`). The two fusion routes that came before it are
measured dead on this host: FUSED5 at 0.39x the champion (`route.md` h64, same day), and the
4-wave `nobar` illegal upper bound at 0.923x (`facts.md` item 3). `g76` needs no atomics, so
it is the one form the determinism gate does not bar.

It is not this round's arm, and the reason is arithmetic rather than doubt. Its kill lives in
`hint.md` ~3211 as "17.18 GB = 5.7 ms against a 3.07 ms saving". That divides out to
**3.01 TB/s** -- the A0 VR-throttled roof at 1100 MHz, which this round's own READ FIRST block
forbids carrying to B0. But the *saving* side, 3.07 ms, is equally an estimate: it is
back-derived from `k_dq`'s 96-WMMA body share. Shipping `g76` now would price one estimate
against another for the third time. So round 25 takes the two measurements instead -- **P78**,
the first HBM streaming figure ever measured on B0, and **P76**, a subtractive probe that
deletes `k_dq`'s S and dP GEMMs and times what is left -- and `g76` carries a round-26
decision rule in which no estimate appears.

`g77` is what can be built *and* priced inside the round. Round 24's own route table reserved
it in terms: *"h11 末尾的 L2 同步扫描仍 OPEN … 留给 r25"*, *"目前唯一还活着的 L2/流量杠杆"*
(`route.md:1548`). Both agents reached it independently in the blind stage.

The mechanism, re-derived from source rather than inherited from h11's A0-era figure:
`impl.py:157` launches `grid = (nhkv*nsp, nblk, nb)` with x fastest, so the 1024 co-resident
one-wave workgroups are 8 kv-heads x 128 kv-tiles at one batch, with `qp_start` running 0..127
and every one of them marching *up* the q axis one pair at a time. They are de-phased by
construction and span 4096 queries at any instant -- **67.1 MB** of Q/dO against a **4 MiB**
device-wide L2 (`dead_ends.md:102-106`). Reverse the scan so each descends from `nqt2-1` to its
own `qp_start`, and the whole resident cohort sits on the same q-pair: **524 KB**. A ~128x cut
in the instantaneous working set from index arithmetic. h11 wrote ~16 MB; that is the
per-kv-head number. Across the 8 co-resident kv-heads it is 67 MB, so the effect is larger than
h11 priced.

What makes it worth the slot after four consecutive card losses is what it does *not* touch.
The body ISA is unchanged -- 64 WMMA, 740 VGPRs, the same prefetch loads at the same body
indices. Every loss since round 21 has come from the prefetch/issue-position axis that
`pool.md` rules 1-4 describe, and `g77` is the first arm in four rounds that does not test it.

### Prediction

| | |
|---|---|
| metric | `prod_vs_champion_same_session` |
| from | 1.000 |
| to | **>= 1.030** |
| primary falsifier | **P77 qmask delta at prod >= 5%**, reported *before* the arm |
| also | `prod_B1` delta >= 1.5x the end-to-end prod delta; body ISA identical; last prefetch load index unmoved; spill 0; WMMA 64; `fast` inside floor (h5) |

Falsified if prod lands inside the session floor **with P77 having passed**. If instead the
B=1 arm moves and B=4 does not, the premise is true and dispatch backfill eats it -- which is a
fact about the gfx1250 dispatcher worth keeping, and points at persistent workgroups for round
26 rather than closing the family.

---

## The instruments, and why they are this round's work and not pool entries

A measurement filed to the pool is a measurement that never happens: it produces no speed-up,
so every later round invents a fresh speed candidate instead and the instrument competes for a
slot it can never win. All four run in **one container, one box state, one noise floor**, with
`dmesg -W` armed throughout (h15, and the standing card-safety rule) -- four numbers that will
be compared to each other must not be taken across a re-warm.

| probe | what it settles | kills what |
|---|---|---|
| **P77** qmask address-pin | upper bound on what *any* Q/dO locality change can buy, with no counter | if inside floor: the entire L2/traffic family, free, and `g77` is not built |
| **P78** HBM stream | the denominator of `g76`'s break-even; closes `knowledge_gaps.md` item 10 | `g76` if < 8 TB/s |
| **P76** subtractive `k_dq` | the true ceiling of `g76`'s saving, replacing both estimates | `g76` if < ~1.2 ms |
| **C2** B = 1/2/4 sweep | a *bound* on the regime question (q1), which step 5 can never answer | nothing; reported as a bound or as unsettled |

P77 is the probe specified in round 10 and declined there, never built. It adds a runtime
`qmask` and masks the query-pair index **absolutely** at `kernels.py:234`. Its three recorded
traps are build gates with a read-back, not reminders: the mask is absolute and not relative to
`qp_start`; the GQA head is not masked (G=4, identity); `kernels.py:287` is not touched; and
`impl.py` is left alone so the arity mismatch stays a structural interlock. Its `qmask=3` output
is numerically wrong by construction and never enters a speed table.

P76 comes from the reviewer and is the best single item either of us produced. It replaces the
estimate at the centre of `g76`'s break-even with a measurement.

**Before any of it**, the three h33 defect fixes are re-applied to the base tree. `op/current`
still holds round 20 because round 24 was not accepted, and `diff` confirms it differs from
`rounds/024/op`. One of the three defects silently drops every dQ write at `nsp_q >= 8`, so an
unpatched base produces a correctness artefact and a timing artefact together. It is free --
prod 638.822 vs 638.895 = 0.9999 inside a 0.32% floor (`facts.md:600-603`) -- and it is
re-validated against the unpatched base first, so the arm's delta contains none of it.

---

## What was argued, and how it resolved

Three turns. Three of the four closures went against me, on source I should have read first.

**`q3` -- does `beat` issue 5 GEMMs or 7?** I held it open; the reviewer cited `route.md` h9,
"confirmed by disassembly". `route.md:1016-1019` retracts that sentence in terms -- gfx1250
`.co` files do not disassemble (`planner.log:989`) and the count was *inferred*. So the
reviewer's citation was superseded. But the reviewer then produced a better argument than
either of us had: the gfx1250 pipeline is exactly three kernels
(`odo` -> `main(a32)` -> `dq_convert`), dQ is atomically accumulated into `dq_accum`, and a
7-GEMM layout would need a fourth kernel. That is structural and needs no disassembly. **q3 is
closed as INFERRED**, recorded with that flag, by a timeboxed 20-minute census (`r25.i0.C1`)
that gates nothing.

And the reviewer was right that my exit rule was wrong: `k_dq` recomputes `k_dkdv`'s S and dP
in *our* source, so deleting that redundancy stands or falls on our own numbers. What `beat`
does changes what our 47.0% η *means*, not what the deletion is worth. **Withdrawn.**

**`g76`'s layout.** I objected that the two kernels contract `dS` over opposite axes, so `k_dq`
would have to stage it through LDS and re-import traffic P1 measured as load-bearing.
`kernels.py:849-852` refutes it in the code's own words -- *"FREE: the two kv-tile accumulators
concatenate in-lane into the dS A-operand"* -- `a_ds` is built in registers and passed straight
to `wmma` at `:874` with zero LDS ops. **Withdrawn**, and replaced with the question that was
standing behind it and does matter: the **contiguous run length** of the store and the load
under one shared layout, since 17.18 GB at a scattered run length is not 17.18 GB at streaming
bandwidth. Free, source only, runs before P78.

The reviewer conceded one number in return: `154.25 GB/s` is the *compulsory lower bound*, not a
measured load. With the requested upper bound at 12.3 TB/s over the same 8.701 ms and no byte
counter at any level, actual traffic sits inside an 80x interval and "the memory pipe is idle"
is not established by anything. That is why P78 is a hard gate.

**`g74`** (move the softmax/rescale VALU off the carried chain) is **dead at its free gate, zero
card time**. `kernels.py:12, :212` -- LSE and delta are precomputed fp32 *inputs*, loaded as
`g_lse`/`g_del`, so the backward has no running rescale for the mechanism to move. Corroborated
independently by the corpus's only *backward* recipe
(`fmha_v3_bwd_hd128_bf16.md` #6 b7: VALU utilisation 24.29%; the softmax and the
`dS = P*(dP - D)` correction *are* the algorithm). Marked in `pool.md` for reflect to retire.

---

## What remains open

One disagreement survived, and it is agent-versus-agent, not candidate-versus-data.

**`r25.i0.C2`, the clock sweep.** The reviewer holds that any B-scaled contrast confounds
power with tail fraction (dispatch waves go 8 -> 2) and with intra-window clock drift
(`benchmark-results.md:32-33`, proxy 1944 -> 1968). I hold that three points -- B = 1/2/4, with
cycles derived *per kernel* and the drift carried as an error bar -- separate the two slopes by
shape, and that a bound is worth two minutes when `profiling.yaml` calls the regime the round's
largest hole and step 5 can never run (h62: the node is never idle by design). We agree it
cannot *answer* q1.

**Settled by:** two prod runs of one binary at **locked clocks** --
`rocm-smi --setperfdeterminism` at two levels on card 1 -- with geometry, trip counts,
occupancy and dispatch-wave count held identical, so only the clock varies. That needs write
access to card 1's performance level but *not* an idle node, which is exactly what makes it
different from step 5's unreachable precondition. Whether that write is permitted inside the
`fa-g1` container on a shared node is itself unknown, and is filed as `knowledge_gaps.md` item
11. If it is permitted, it is round 26's highest-priority instrument and C2 should be skipped
in favour of it.

C2 runs anyway, because its B=1 point is *also* `g77`'s backfill test -- one launch, two
readings -- and because its result is reported as a bound or as unsettled, never as a regime
answer.

---

## Bookkeeping

The dialogue allocated `g78`-`g81` to a census, a clock sweep, the defect re-apply and a
subtractive probe. Step 5 reclassifies all four under this pool's own standing rules --
instruments `r*.i0.*` do not consume a `g`, throwaway probes do not either (P1/P2/P3 round 20,
P70 round 22) -- and the defect re-apply is `r24.i3.g75` re-applied, not a new idea. **Ids are
never reused, so `g78`-`g81` are burned, not reassigned.** Highest allocated is `g81`; next free
is `g82`. Recorded rather than quietly fixed, following round 24's precedent in `pool.md`.

Two new `g` ids this round: `r25.i2.g76` and `r25.i3.g77`.

## If everything dies

If P77 comes back inside the floor *and* P78 or P76 kills `g76`, then in one round the
L2/traffic family and the last 5-GEMM route are both closed by measurement, on top of a
prefetch family already closed on every axis and `g73`/`g74` dead at their free gates. That is
the state `route.md:1555` names in its own words -- 「应按 h18 第 12 轮的措辞考虑『停止优化并
固化』」. Recording it honestly beats manufacturing a fifth arm to lose, and `hint.md`'s own
judgement agrees on where the value has moved: the *forward*'s gap is 1.53x and worth
0.83 ms/step, against a backward residual worth ~0.05 ms/step.
