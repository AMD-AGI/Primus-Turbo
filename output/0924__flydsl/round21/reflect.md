# Round 21 -- reflect

**Not accepted.** prod 412.10 TF/s is 80.9% of its best ever (509.1, round 20). `op/current/`
still holds round 20. Score 0.7376 against the bar.

## What happened

Route row 10 said: take `r20.i2.g62`'s cross-iteration prefetch deepening -- the one positive
mechanism this run has -- and apply it to `k_dq`, the kernel it had not been tried on. The
precondition in `pool.md` was explicit and it was satisfied: `k_dq`'s hot body `.LBB0_2`
carried four `s_wait_loadcnt`, one of them a full `0x0` drain at 93% of a 742-instruction body.
I built it (`r21.i1.g63`), it compiled clean, VGPR 960 -> 992 with spill 0 and scratch 0, it
passed correctness at 52.52 dB and 15/15 UT shapes, the intended ISA delta landed exactly --
waits 4 -> 2, both hoisted to the top, the 93% drain gone -- and it measured **-19.33% prod**
on a 0.19% noise floor.

Two other arms cost no card time and were right not to: `r21.i3.g65` died on the offline
screen (its ISA was instruction-identical to the incumbent after register renaming), and
`r21.i2.g64` was passed over on a 0-3 family record plus the free observation that `s_setprio`
is a no-op at one wave/SIMD.

The useful thing I did build was `r21.i4.g66`: a control arm that buys the *same* load spread
through `sched_group_barrier` alone, at VGPR 960 -> 960, spill 0, body +2.7%. It lost **more**
(-21.93%). That excludes register pressure, spill and code size and leaves the spread itself.
`k_dq`'s tight ~50-instruction prefetch clump is load-bearing. `k_dq` and `k_dkdv` have
**opposite signs on the same lever**, and the body shapes say why (96 WMMA / 16+16 LDS vs
64 / 40+40).

## What I got wrong

**1. I built two full arms where a cheap subtractive probe belonged first.** Round 20 settled
the LDS-port question for a fraction of the card time by *deleting* things (P1, P3) before
building anything. The equivalent here was `k_dq` prefetch depth 1 -> 0 -- one patch, one
measurement, and it would have told me the sign of the lever on this kernel before I spent a
build on deepening it. I had the pattern in front of me from the previous round and did not
apply it. That is the process error, and it is the one worth carrying.

**2. My `g65` premise was simply false.** I asserted `k_dq`'s LDS write-to-read distance was
~0 and wanted to hoist the stores. The ISA showed 300-500 instructions of distance already
present. The offline screen caught it at zero cost, which is the only reason this is a cheap
mistake rather than an expensive one.

**3. I let "VGPR has headroom and spill is 0" stand in for an argument.** It was the whole
pre-build case for `g63`. It is a gate, not evidence. It is now written into `facts.md` as
retired.

**4. My prediction did not hold, and not even in its failure mode.** `act.yaml` said
"roughly 515-520 if it transfers, and a fast unambiguous loss if register pressure serialises
the body instead". It was a fast unambiguous loss -- **by the other mechanism**. `g66` kept
VGPR flat and lost more. So I did not merely guess the wrong branch; the branch I named as the
loss condition was not what happened.

## What the next round reads

`facts.md`'s bottleneck line now carries the signed, per-kernel form. `r22.i1.g67` (tighten
`k_dq`) and `r22.i2.g68` (re-spread `k_dkdv` -- the falsifier) are in the pool and are the two
halves of testing it. If both come back the way the model says, the sentence is earned; if
`g68` also loses, then "spread" is not the variable and the story needs rebuilding.
