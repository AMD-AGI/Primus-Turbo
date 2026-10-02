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

## ⚠ CAMPAIGN CORRECTIONS (bwd job, A0, rewritten 2026-09-30) — inlined because this step does not read `hint.md`

`hints` is wired into `fast_loop`, deep `reflect` and `route.md`'s operator tables, but **not** into the
`profiling`, `plan` or `act` prompts. The job's operator file is `{{job_context_dir}}/hint.md`; **read h74-h82
there before you start** (h54 is the older one-page index; h76 is the current champion's map). What changes
decisions, all measured on this card (A0, one gfx1250, container `fa-repro`, reflashed 2026-09-29):

1. **The champion is s6** (adopted by refactor h75): prod 5.30 ms = **1.038x the ASM bar**; proxy ~1.13x. The
   target is now **1.20x beat on proxy AND prod** (fast is reported only). Structure: `k_delta` -> `k_dkdv` (main
   stream) concurrent with `k_dqg` (side stream); both kernels take their big operands through a **3-stage TDM
   LDS ring whose next-iteration B operands are read back into carried VGPRs** (k_dkdv: Q/dO; k_dqg: K/V).
   Any "TDM has no target in k_dkdv" or "k_dkdv 740 / k_dq 960 VGPR" statement in older findings is obsolete.
2. **`rocprofv3`'s `VGPR_Count` column is HALF the ISA allocation** (reads 512 for aiter's 1024). Read
   `.vgpr_count` from the ISA. s6: k_dkdv **713**, k_dkdv_sp 707, k_dqg **881** -- all **1 wave/SIMD**
   (VGPR-capped, not LDS-capped). A second wave needs <= 512 VGPR per wave.
3. **Static ISA metrics do not order candidates.** Arms that improved every static metric lost on the card
   (ku2 +8.2%, trim2 +1.3%, earlier -16%). Use them as gates (spill, scratch, a wait turned into a full drain);
   **price on the card** with the blocked benchmark.
4. **`prod` ranks; fast/proxy are sentinels.** `min_gain` is 0.007 (floor 0.24-0.66%; blocked A/A on A0
   0.01-0.07%). Never ship a candidate whose prod `vs_champion` is inside the noise, whatever fast did.
5. **The stall question is OPEN again: ATT works on this card since the reflash** (per-instruction
   Hitcount/Latency/Stall/Idle for FlyDSL JIT kernels and the ASM `.co`; recipe in h81 and in profiling step 4).
   Every s6 lever was found from an ATT read. ATT tells you WHERE cycles go; it never ranks. **PC sampling
   stays forbidden** (it wedged MES; rocprofv3 now rejects every PC-sampling config anyway). `--kernel-trace`
   still records 0 dispatches: take kernel names/timestamps from the `--pmc` csv.
6. **What bounds k_dkdv was measured by ablation** (h77): Q/dO already in VGPRs -18.5%; TDM source pinned in L2
   -1.0% (**L2 bandwidth is not the limit**); +16 WMMA/iteration (fused dQ matrix work) only +4%. Size a lever with
   an ablation before building it. Closed on A0 (h78): lse_late, ku2, flip, trorder_b, trim2, TDM without
   carried operands, k_dq U2. The fused 4-wave w4f is correct but parked at 6.51 ms (h79; 4.24 ms without atomics).
7. **k_dqg only runs at proxy and prod.** toy/fast dispatch k_dq_sp, so a toy/fast pass does not test a k_dqg
   change; its first real run is a serialised proxy validation.
8. **Machine rules.** The fp32 reference (`forward_reference`, `op/eager`) is **never computed on the card**: use
   `op/refcache_util.py` (`cached_forward` / `cached_backward`) or `op/refcache/*.pt` (it faulted a card on
   2026-09-28 and cost a power cycle on 2026-09-22). One shape per process. Compile-only first with BOTH
   `ARCH=gfx1250` and `FLYDSL_GPU_ARCH=gfx1250`; spill or scratch > 0 is a kill. A fresh
   `FLYDSL_RUNTIME_CACHE_DIR` per process and a new directory per arm (kernels.py arms are module-level switches,
   which the JIT cache key ignores). Leave the card idle when you finish: no scoring process left running.
