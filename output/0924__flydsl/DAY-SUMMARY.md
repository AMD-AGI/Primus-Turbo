# 2026-09-24 · gfx1250 FlyDSL attention backward — day summary

Machine handed over at end of day. Card released clean: GPU 0%, VRAM at the 175 MB boot
baseline, no KFD PIDs, dmesg zero amdgpu events, container `fa-repro` stopped.

## Where the campaign stands

| | |
|---|---|
| champion | **round 20** (`best_round: 20`, `op/current` byte-identical to `rounds/020/op`) |
| prod | **511.42 TF/s** in round 20's own sweep; **516.87** re-measured in round 22's |
| bar | **7.67 ms / 716 TF/s** (median of n=177 in-tree measurements, sd 0.47%) |
| ratio | **0.72x** (was 0.696x at start of day) |
| rounds | 22/40 settled; round 23 stopped mid-`opt`, leaves a partial `1-opt` that the next
`resume` will move aside automatically |
| deep rounds | scheduled at 25 / 30 / 35 / 40 |

Rounds 18, 19, 21, 22 rejected (gain 0.931 / 0.994 / 0.872 / 0.788). Round 20 accepted at
gain 1.0016.

## The day's real product is not the +2%

Throughput moved ~2%. What actually changed is that the operator went from five mutually
inconsistent hypotheses to **one variable with a pre-registered, confirmed prediction**:

> **worst-case load-to-use cover distance to the next full `s_wait_loadcnt 0x0`**

It explains every reorder result with one quantity, and none of these change a byte of
traffic, a WMMA count or an LDS op count:

| arm | effect on cover | prod |
|---|---|--:|
| `g62` k_dkdv prefetch depth 1→2 | +~1 iteration | **+1.65%** |
| `g28` | reduced | −7.8% |
| `g66` spread k_dq's 32 loads | collapsed | −19.33% |
| `g68` spread k_dkdv's 32 loads | ~603 → ~41 instr | **−30.32%** |
| `P70` delete k_dq's prefetch | to zero | −10.86% |

Round 22 wrote "A68 loses 20–30% of prod" **before** the card and measured −30.0%.
A pure reordering costing 30% is also the strongest remaining evidence against any
bandwidth reading of this kernel.

## Hypotheses closed by measurement today

compute-bound · HBM/bandwidth-bound · issue roof · occupancy & BLOCK_KV · **LDS port
pressure** · power/clock throttling · fusion (both orientations) · static ISA metrics as a
ranking signal · the "tighten k_dq" axis

Notable kills: `P3` deleted **all 80 LDS ops** with WMMA count unchanged and ran 8.19%
slower. `g60` deleted 82 issue slots and ran 17.27% slower. A 60-point power trace showed
the package sensor is static (855.0 W idle vs 853 W under load) and sclk flat at 1100 MHz
for 60/60 samples, retiring "the card was throttling".

## The bar was wrong all day and is now right

The recorded **10.160 ms** anchor traces to `output/0915__opt/status.json:34-38` — an
autograd measurement shim (`tune_attention.py:560-578`) driven through `out.backward()`
passing neither `hip=` nor `scratch=`. **It was retracted the same day** (09-15 09:33: shim
10.062 vs corrected path 8.13) and the retraction never propagated. Measured causes, by
controlled isolation: per-call `HipModule()` + 3× `hipModuleLoad` **+1.15 ms**, autograd
plumbing **+1.42 ms**, per-call scratch allocation **+0.01 ms**. The GQA host reduction is
+0.48 ms and sits inside *both* timed regions, so it cancels.

Consequence: the campaign is at **0.70–0.72x**, not the 0.92x that 10.160 implied.

Also verified: the bar does **not** read stale gradients — the scratch was NaN-poisoned and
zero NaN survived at fast and prod.

## The wedge

One wedge, 08:58:53, `RW=0x1 / PERMISSION_FAULTS 0x5`, MES dead 31 s later, cost one AC
cycle. A **separate** fault recorded by round 18 (`RW=0x0 / PERMISSION_FAULTS 0x3`) did not
wedge the card — do not merge the two.

The trigger is isolated by a controlled pair in the tree, not inferred:
`rounds/016/_scratch/meas2/out` (one process, three shapes) faults;
`meas3/out` (one process per shape) is rc=0 three times, same binary, same six arms.
`facts.md:241` had ratified "one benchmark process per shape" rounds ago and **it never
reached `validation.py`** — that single line is the direct source of the round 17 and 18
wedges. Fixed; six full measurement processes since, zero faults.

The wedge's write fault itself remains **unexplained**: full enumeration proves `k_dq`'s
writes are a bijection onto the real tensor (slack 0, 134M index tuples at prod, no
sampling), and `k_dkdv`'s only overrun is a *read* that its real descriptors clamp.

## Defects found, fixable at zero card time (not yet applied)

1. `impl.py:115` asserts `sq % 32 == 0` but `kernels.py:667-668` uses ceil while the
   launcher passes floor. At `sq % 64 == 32` the kernel **issues** 262,144 B of
   out-of-bounds writes through a fake `1<<30` descriptor and never dispatches query tile 0.
   All nine scored shapes happen to be safe. **One-line fix, ISA byte-identical.**
2. `kernels.py:685` computes `nsp * B_ * Sq * Hq * (D*4)` in int32; at prod dims nsp=8/16
   wrap to **exactly 0**, i.e. `num_records = 0` and every dQ write silently discarded.
3. `kernels.py:538/:545` and `:499-504` — k_dkdv's unclamped prefetch reads up to 982,272 B
   past Q/dO, held back only by its real extents. **Never convert k_dkdv's descriptors to
   the fake style.**

New ISA fact: **gfx1250's V# `num_records` is in units of 128 bytes** (`>> 7` in the
runtime path, folded in the constant path), which resolves a 128× discrepancy three earlier
reports filed as unexplainable.

## My own errors today — six, all caught by reopening the source

1. h28's fusion arithmetic (0.951x) — `slots = nsp` does not transfer to a fused kernel.
2. Round 16's `load_impl` hoist as "the fix" — six clean sweeps at a 44%/process fault rate
   gives P(0/6)=0.49, the *most likely* outcome under the null. **A clean run count is not
   evidence unless the base rate says it should have failed.**
3. "`empty_cache` is the delta" — wrong, and the line number was wrong too.
4. "Allocation costs ~1.52 ms" — measured +0.01 ms.
5. h36's "+188 VGPR won't fit" — measured +32, fits with 32 to spare.
6. h38's reason for rejecting the 1.57% floor — I said cross-session; it is
   `bench_onesession.json`. The conclusion survives for a **better** reason: that floor was
   measured on **HipKittens' bf16 GEMM ladder**, not on this operator, whose own
   same-session prod floors are 0.61% / 0.66% / 0.40%.

Errors 1, 4, 5 and 6 share one root: **taking a number measured in one context as a
property of the mechanism and carrying it to another.** That is also what killed g61's
chain split, g60's gfx950 analogue and g63's cross-kernel prefetch. A floor, a register
cost and a byte count are properties of the kernel and harness that produced them.

## For whoever picks this up

- The bar is **7.67 ms / 716 TF/s**. Do not cite 10.160 or 541 TF/s.
- Grade against **this operator's own same-session floor** (0.4–0.7% at prod), not the
  corpus's 1.57%. Round 22's prescription: the bar is "beat the champion by more than the
  floor", not "be positive". `min_gain: 0.0` is what let round 20 promote on gain 1.0016.
- `k_dq` and `k_dkdv` are **different machines**. Any "the edit that worked on X" proposal
  must say what makes Y's body the same, and be screened `COMPILE_ONLY` first.
- `sched_barrier(0)` is a scheduling **boundary**, not a clamp — fencing a clump reliably
  loosens it in this backend.
- `k_dq` ships at **960/1024 VGPR**. State a VGPR delta as measured, never as inherited.
- Still owed, needs GPU: re-measure the nine points behind the `seqlen >= 2048`
  qualification gate (`E2E-AB.md:301-302`). That gate rests on a "~0.85 ms fixed floor" that
  partly **was** the per-call `hipModuleLoad`, so it is probably set too high.

Full hint corpus: `output/0923__flydsl/hint.md` (h29–h38 plus corrections written today).
Per-round archives: `output/0924__flydsl/round{18,19,20,21,22}/`.
