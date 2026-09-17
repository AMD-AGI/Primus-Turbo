# Round 1 — reflect

## What happened

Two knobs, built separately from `op/current`, then merged because neither lost:
`waves_per_eu: 1 -> 0` (−4.2%) and `TRITON_HIP_USE_IN_THREAD_TRANSPOSE` scoped around
the backward launch (−5.9%). Together **−13.2%**: 8.683 ms, 633.2 TF/s, **1.1529x** the
incumbent re-measured beside it in the same session. Score 0.8968. Correctness
unchanged to two decimals everywhere. The 1.50x gate still fails by 10.3%.

The merge being superadditive is the one genuinely interesting result — 13.2% where
additive would be 9.9% — and it is also the thing I am least able to explain. I have a
story (both reduce register pressure, so removing one constraint makes the other's
saving reachable) and no measurement behind it. It is a story, not a finding.

## What did not work

**`rocprofv3` recorded zero kernel dispatches** on this job's benchmark and I never
fixed it. The whole round was chosen from a static register census and end-to-end
times, which is a real limitation and not a stylistic one: I can say the kernel is
99.28% of the backward and that it spills 326 VGPRs, and I cannot say *where* inside it
the time goes. `r1.i6.g6` exists because of this and it should probably be taken before
the LDS work rather than after.

## What I misjudged

**I predicted `num_warps=8` would help and it was 2.36x slower.** The arithmetic that
predicts it — gfx1250's per-lane VGPR budget is `131072 / NUM_THREADS`, so doubling
warps halves an already-exhausted 1024 — was available before I measured, in the arch
card I had not yet read. I ran the experiment first and read the reason afterwards. On
a part with no usable counters, reading first is not optional.

**I lost four validation runs and roughly two hours to one bug that I diagnosed wrong
twice.** It presented as an internal assert, a `std::bad_alloc` and a bare SIGSEGV, and
because it was intermittent — surviving three full validations before it first fired —
I twice accepted a fix on the evidence of one passing run. Both of those theories were
about `sys.modules` bookkeeping in *my* code. The actual cause was that the vendored
tree and `op/baseline`'s installed one declare the same op name, so the second
`torch.library` fragment's destruction takes the schema out from under the first's live
handle. Two lessons, and the second is the expensive one: a segfault, a `bad_alloc` and
an internal assert in the same place are one bug, not three; and "it passed this time"
is not evidence against a use-after-free.

**The route row I wrote while deciding was the wrong one.** Row 4 said "shorten live
ranges to escape the spill" — a symptom with no mechanism, blocked on an instrument
that never worked. Reading the corpus afterwards replaced it with something concrete:
this kernel runs LDS at 20% of capacity while spilling 1308 B/lane, where AITER,
HipKittens and FlyDSL all run LDS near their limit with essentially zero spill. That
comparison was available at step 3 and I did the reading at step 6.

## For the next round

The launch-configuration surface is closed; `facts.md` says so with the numbers. The
remaining distance is the spill, and the bottleneck line in `facts.md` is deliberately
marked medium confidence — it is one agent's reading from a survey that did not work,
and the honest summary is that the speed stands while the model behind it is inferred
rather than measured.
