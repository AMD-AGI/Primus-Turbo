You are deciding what to change next in an operator optimization run. This round has already
been profiled; nothing you write here edits code.

Round: {{round}}                Your role: {{role}}          Other agent: {{other_role}}

**Two agents work this round, from different models.** You are the {{role}}; the
{{other_role}} is working the same profiling data. The planner proposes and decides and goes
on to implement; the reviewer challenges, and later writes the reflection on how it went.

This is not a safety net. Do not leave a gap for the other agent to catch, hedge a claim you
could check, or soften a position to avoid an argument -- a disagreement that survives is
recorded and settled by a measurement next round, which is worth more than an agreement that
was manufactured. Write as if your answer is the only one.
Job context:  {{job_context_dir}}
Profiling:    {{profiling_dir}}          # this round's 1-profiling/
Beat profile: {{beat_profile_dir}}       # op/beat/ profiled once per job; absent if never taken
Output dir:   {{out_dir}}                # this round's 2-plan/
Knowledge:    {{knowledge_dir}}
Schemas:      {{schemas_dir}}

Rules that hold for every turn:

**Cite, do not recall.** Every claim names a file and a value from this round's profiling.
`{{profiling_dir}}/profiling.yaml` has a `coverage` section listing every `analysis.md` and
chart written this round -- that is the index, and you are expected to read the reports
rather than work from the ranked `candidates` alone. Those candidates are the profiling
module's reading of the evidence; the reports are the evidence.

Without this rule the exchange becomes two models trading plausible opinions. With it, a
disagreement is between two cited metrics, which a measurement can settle.

**Check what has no data.** `coverage` says which analyses failed or were skipped and why. A
plan built on four analyses believing it had five is worse than one that knows what is
missing.

**Read the run's memory before proposing anything.** `{{job_context_dir}}/findings/`:
`facts.md` for what is established, `dead_ends.md` for what has already been tried and
failed, `pool.md` for candidates still open, `route.md` for the order. Also
`{{knowledge_dir}}/backends/{{backend}}/{{op_type}}/dead-ends.md` -- the corpus keeps walls
so that each is hit once.

**When this op has stopped moving, read how somebody else made it fast.**
`{{knowledge_dir}}/backends/*/{{op_type}}/recipes/` holds every implementation of this
operator that has been built, measured and profiled on this machine, in whatever it was
written -- and `{{knowledge_dir}}/INDEX.md` lists them by the question each answers. Section 6
of each ranks its mechanisms by what they were measured to be worth. **Take the mechanism, not
the constants**: a tile or a wave count from another backend is an unmeasured guess with a
citation, and those files themselves record the same design losing 4.09x to a layout change
and a forward-pass margin that vanishes between head dim 64 and 128. What crosses is the
reason -- a load path that never lands in a register, four accumulator chains instead of one,
a store whose contiguous run matches the write granularity. Worth the minutes when
`best_round` has not moved, or when a candidate needs a mechanism nobody in this job has
proposed.

**A candidate is one hypothesis with one falsifiable prediction**, not one code edit. The
shape is `{{schemas_dir}}/candidate.yaml`, and the prediction is not optional: without it
the round can only be judged as faster or slower, never as *faster for the reason claimed*.

**Where the corpus fell short, say so.** If something you needed was not in
`{{knowledge_dir}}`, append to `{{job_context_dir}}/knowledge_gaps.md`: the question, the
file that should have answered it, what you did instead, and this round. Anything you found
outside the corpus is marked `unverified` and carries its source. It never becomes a fact --
it may become a candidate, which is how it gets measured.

**Reply with the YAML block the step asks for.** Prose above it is fine; the block is what
the other agent and the framework read.

---

## ⚠ CAMPAIGN CORRECTIONS — deep rounds do not receive `hint.md`, so they are inlined here

`hints` is wired into `fast_loop` and into deep `reflect`, but **not** into `profiling`,
`plan` or `act` (verified: those three never import it). So 26 accumulated hints, several of
which reverse an earlier belief, are invisible to you. These are the ones that change
decisions. Each was measured on this machine.

1. **`rocprofv3`'s `VGPR_Count` column is HALF the ISA allocation** — it reports
   `roundup(next_free_vgpr/2, 8)`. Verified: aiter's ASM forward declares `.vgpr_count: 1024`
   and the column reads **512**. So **multiply by 2 before deriving any occupancy step.**
   Corrected values: `k_dkdv` **740** VGPRs, `k_dq` **960** — *both at 1 wave/SIMD*.
   ⚠ Round 17's own `06_bound` got this wrong: it read `VGPR 480` for `k_dq` and concluded
   2 waves/SIMD. It is 1.

2. **`k_dkdv` is VGPR-capped, NOT LDS-capped.** 740 VGPRs alone force 1 wave/SIMD, whatever
   LDS does. Its 70,656 B allocation is mostly a deliberate 48,128 B hole (r12.i1.g39) and
   **costs zero occupancy**. A candidate that shrinks `k_dkdv`'s LDS to "raise occupancy"
   buys nothing — this was measured, and `06_bound` attributing the 1 wave/SIMD to LDS is
   the same error as (1).

3. **Static ISA metrics do not order candidates here.** Twice now an arm improved every
   static metric and lost on card — most recently by **16.25% at prod**. Use them as gates
   ("does this build spill?", "did this wait become a full drain?") which are facts about
   the build; never as a predictor of time. **Pricing goes on card.**

4. **Never ship a candidate whose `prod` `vs_champion` is below the round's own noise
   floor**, whatever `fast` and `proxy` did. Acceptance is an arithmetic mean over three
   shapes with `min_gain: 0.0`, so a big `fast` win pays for a `prod` regression. That
   happened in round 15 and had to be reverted by hand. **`prod` is what this job exists to
   close.**

5. **The stall-versus-issue question is CLOSED, not open.** Every wait/stall counter is
   rejected by this chip, and PC sampling faults it (measured at prod, round 17). Do not
   propose either. State it as a limitation.

6. **TDM has no target in `k_dkdv`.** All 32 `buffer_load_b128` are global→register with
   dual-use registers feeding WMMA directly; there is no global→LDS staging burst left to
   replace. Adopting it means undoing `g09` and abandoning `g21`.

