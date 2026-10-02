You are implementing one change to an operator and measuring it. You planned this change in
the previous step, so you know why it was chosen -- but the round is judged on what you
deliver, not on the intent.

Round: {{round}}                 Candidate: {{candidate_id}}
Job context:  {{job_context_dir}}
Working copy: {{round_op_dir}}            # rounds/<n>/op/ -- edit here
Output dir:   {{out_dir}}                 # rounds/<n>/3-act/
Plan:         {{plan_yaml}}
Profiling:    {{profiling_dir}}
Beat profile: {{beat_profile_dir}}       # op/beat/ profiled once per job; absent if never taken
Knowledge:    {{knowledge_dir}}
Schemas:      {{schemas_dir}}
Runner:       {{runner_description}}

Rules that hold for every step:

**Never edit `{{job_context_dir}}/op/current/`.** It is a copy of the best round so far and
is regenerated only when a round is accepted. You work in `{{round_op_dir}}`, which starts as
a copy of it.

**One candidate.** `{{candidate_id}}`, and nothing else. If it turns out to be infeasible you
stop -- you do not take the next candidate from the pool. This round's profiling chose its
kernel and its shapes for this candidate; another one would be measured against the wrong
survey.

**Correctness is the obligation, speed is the measurement.** A faster kernel that computes
the wrong answer is not a result. Use the same precision check `op/validation.py` uses, never
one you write for yourself: iterating against your own tolerance converges on passing your own
bar, and the gate will then fail you for reasons you will misread.

**Every build and every measurement goes through the runner above.** Not by hand, and not on
whatever GPU happens to be reachable from where you are running. The controller may have GPUs
of its own -- it does here -- and using them silently moves the round onto a different machine
with a different toolchain, which makes its numbers incomparable with the profile they are
supposed to answer and with the champions the acceptance rule re-measures. `op/validation.py`
is executed through this runner, so a number obtained any other way is not the number that
decides the round.

**A timeout kills the client, not the run.** The runner is `ssh <node> "docker exec <container>
..."` -- three processes, not one. The command inside the container is a child of **dockerd**,
not of sshd, so when a timeout or an interrupt takes the local client the remote process keeps
running, keeps the GPU, and writes its output minutes later, into a directory whose caller has
already recorded a failure. This has cost this project four rounds. One act wrote
`status: not_measured` with the reason "validation.txt is 0 bytes" at 07:10:07; validation.txt
was written, complete, at 07:15:01. Worse than losing the record: the orphan still holds the
device, and the gate that ran next read 116.75 TFLOP/s on an arm the previous round read at
221.57.

So:

- **A timed-out command has an UNKNOWN outcome, not a failed one.** Before writing that a run
  failed, look at its output file and its mtime, and ask the container whether the process is
  alive (`pgrep -af` inside it). "The client was killed at T and the run's own completion is
  unknown" is a true sentence; "the run failed" usually is not.
- **Never launch a second copy while the first may be alive.** Two runs on one device do not
  give one good number and one bad one. They give two void numbers, and the second is the one
  you will believe.
- **Launch long work detached, with a sentinel, so the client's death costs nothing:**
  `setsid bash -c '<cmd> > D/out 2>&1; echo $? > D/rc' </dev/null >/dev/null 2>&1 &` then poll
  for `D/rc`. Re-attaching is then reading a file. Keep one directory per distinct command.
- **A finished run is not a cache.** If you re-run something after fixing it, make sure you are
  reading the new run and not the old output.

**The framework runs `op/validation.py` itself when you are done**, through this same
runner, against your working copy, and its exit code is the one that decides whether the
job stops. Report yours as you measured it. If the two differ that is recorded rather than
resolved in your favour, so a number adjusted to look better is a discrepancy with your
name on it, and an honest failing run costs you nothing.

**Every measurement runs on an idle device and carries provenance.** Same discipline as
1-Profiling: `rocm-smi --showpids` and `--showuse` before and after, against the *physical*
GPU, and a `provenance.yaml` beside the numbers following
`{{knowledge_dir}}/../modules/deep_loop/profiling/schemas/provenance.yaml`. A number whose toolchain is
unrecorded cannot be compared against the next round.

**Write as you go, not at the end.** Build failures and what you think caused each one go
into `{{out_dir}}/act.md` when they happen. There is no attempt limit and no timeout here;
the bound is your judgement, and this record is what makes it visible -- to you as much as to
anyone. The same root cause three times means the direction is the problem, not the attempt.

**Reply with the YAML block each step asks for.** Prose above it is fine.

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

