# Spec changelog

## v000 -- 2026-09-14T01:54:15

original job file, jobs/fa-gfx1250.yaml

## v000 -- 2026-09-14, job setup

Completed `gfx1250-attn-llama31-8b-bwd_final.yaml`. The user's job file is unchanged
above the `setup_resolution:` line; nothing [user] was edited, and no target field was
touched. What was added:

**Made explicit (previously implicit defaults):**
- `evolve.turn_timeout: null` -- unset, so each module's own 3h per-turn ceiling applies.
- `evolve.min_gain: 0.0` -- unset, and 0.0 is the framework default. Written out because
  it is the wrong default when the measurement floor is coarser than the wins, and this
  job's floor spread is not yet measurable (no `op/`). Flagged for Op Setup.
- `runtime.runner.ssh: null` -- the user named no ssh block and `type` is `docker`, so
  there are zero hops: controller, node and container host are all this machine.
- `op.shape.shapes[0].head_dim: 128` -- loop.py falls back to `op.config.q_head_dim`;
  identical here, written out so the fallback is not load-bearing.
- `op.shape.shapes[0].window_left: -1` -- op_flops.py's spelling of
  `op.config.sliding_window: null`.
- `op.shape.shapes[0].name: b4_s8192_hq32_hkv8_d128` -- inferred. The spec shape carried
  no name; matching to reported shapes is positional, so this is a label for people and
  for `op/validation.py`, not a key.

**Recorded as derived (computed by calling the framework, not by hand):**
margin 1.50 (relative bar live, both null target fields confirmed null); budgets 40
rounds / 129600 s; the schedule enumerated to deep rounds [18, 24, 30, 36] = 36 fast /
4 deep; review auto-approve 600 s; profiling GPU pool [1]; the fully expanded runner
including the container name `op-evolve-gfx1250-attn-llama31-8b-bwd`, that the runner
OWNS and provisions it, and the single mount (the whole checkout).

**Recorded as found, not fixed -- five open questions:**
1. `flop_basis` (high): `tools/op_flops.py` returns the FA-2 backward ALONE for
   `backward=True`, and loop.py uses it for a fwd+bwd measured time. Every TFLOP/s in
   the job file is on the 3.5x fwd+bwd basis; all four quoted anchors reproduce exactly
   on it and none on the framework's. The ledger will read 1.400x low. Does not affect
   pass/fail (the bar is relative), does affect every absolute and roofline statement.
   Op Setup must pin one basis and say which; setup did not change the shared tool.
2. `seed-vs-margin-rationale` (high): the SEED block says seed at the fused 24.435 ms
   kernel; the op.target block's justification for 50% assumes seeding at aiter's
   shipped 34.111 ms and warns that the tuned seed makes the margin "not a bar at all".
   Both are in the file. Needs a user decision before Op Setup builds op/baseline.
3. `min-gain-unmeasured` (medium).
4. `throttle-state-unknown` (high): the clock was NOT measured, because the job file
   says `clock_probe.measure()` over-commits the queue ~372x and must be fixed first.
5. `gpu-not-fenced-by-the-framework` (medium): two unrelated containers were already
   running on this node during setup.

Nothing in `state.yaml`, `history/v000_original.yaml` or `op/` was created or edited.

**Amended after the checks ran** (the five open questions above were written from the
job file alone; the checks added two more and settled part of a third):

6. `seed-and-harness-absent-from-the-pinned-image` (high): none of the four files the
   SEED block and `op.reference.api` mandate exist in the pinned image. They are in the
   host checkout `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo` @ `d2f75576`, which
   is not mounted. Op Setup must resolve this before it can build `op/baseline`.
7. `pinned-image-cannot-be-re-pulled` (medium): `docker pull` is access-denied; the one
   local copy is all there is.
4. `throttle-state-unknown` amended with an observation: under a 45 s load GPU 1 held
   **1701-1702 MHz**, not 1100 MHz — the header's own un-throttled figure. Necessary but
   not sufficient (the load was hipBLASLt-rate and low-power), so it is recorded as
   evidence, not as a settled answer.
5. `gpu-not-fenced-by-the-framework` amended: all four cards verified at 0% utilisation
   before and after setup's load, and this node has **four** gfx1250 cards, not the one
   `runtime.gpu_pool`'s comment claims.
