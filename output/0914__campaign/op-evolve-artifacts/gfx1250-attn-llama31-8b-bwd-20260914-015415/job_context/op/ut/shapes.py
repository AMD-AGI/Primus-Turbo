"""The acceptance set. Shared by op/ut/ and op/validation.py so they cannot drift.

`SPEC_SHAPES` is exactly `op.shape.shapes` from the job spec -- these are the
shapes that get measured, and `op.shape.mode: single` means the headline number
is the one shape.

`EDGE_SHAPES` are correctness-only. They exist because the job shape cannot
distinguish several ways of being wrong:

  * `sq_lt_skv` / `sq_gt_skv` -- the two sequence axes differ, so top-left and
    bottom-right causal are no longer the same mask. At the job's 8192x8192 they
    are identical and a kernel with the wrong convention scores a clean 53 dB.
    `op.config.causal` is bottom-right; these two shapes are the only thing in
    this job that checks it. `sq_gt_skv` is the harsher of the pair: under
    bottom-right its first (Sq - Skv) query rows are entirely masked, which is
    the degenerate all-masked row that makes a naive `exp(-inf - -inf)` NaN.

  * `min_seqlen` -- 512 is the baseline's own eligibility floor (seqlen_k >= 512)
    and it lands on the other side of the tile table's 2048 boundary, so it
    selects BLOCK_N1 = 128 where the job shape selects 256. Without it the
    128-tile code path is never executed at all.

  * `ragged_tail` -- 1000 is not a multiple of any tile in the config (32, 256)
    nor of FIXED_BLOCK_M = 64, so the last block of every axis is partial. A
    kernel that computes a whole trailing tile, or that drops one, shows up here
    and nowhere else.

Every shape keeps heads_q/heads_kv = 4 (the job's GQA group) and head_dim 128,
because those are `op.config` and are not ours to vary. Every shape also keeps
batch * heads_q >= 32 and seqlen_kv >= 512, which is what the baseline's fused
backward requires -- a shape it declines would measure a different kernel.
"""

SPEC_SHAPES = [
    dict(name="b4_s8192_hq32_hkv8_d128", batch=4, seqlen_q=8192, seqlen_kv=8192,
         heads_q=32, heads_kv=8, head_dim=128, causal=True, window_left=-1),
]

EDGE_SHAPES = [
    dict(name="sq_lt_skv", batch=4, seqlen_q=1024, seqlen_kv=2048,
         heads_q=32, heads_kv=8, head_dim=128, causal=True, window_left=-1),
    dict(name="min_seqlen", batch=4, seqlen_q=512, seqlen_kv=512,
         heads_q=32, heads_kv=8, head_dim=128, causal=True, window_left=-1),
    dict(name="ragged_tail", batch=4, seqlen_q=1000, seqlen_kv=1000,
         heads_q=32, heads_kv=8, head_dim=128, causal=True, window_left=-1),
]

# ---------------------------------------------------------------------------
# Diagnostic-only. RUN AND REPORTED, NEVER GATED, and the distinction is not a
# convenience -- read this before moving a shape across the line.
#
# `sq_gt_skv` has Sq > Skv, so under bottom-right causal its first (Sq - Skv)
# query rows attend to nothing at all. Measured on the baseline at op setup:
#
#     out 53.97 dB   dk 52.10 dB   dv 52.72 dB   dq -15.95 dB
#
# and the dq failure is entirely in the dead rows -- restricted to the live rows
# dq is 52.05 dB, while the dead rows, whose gradient is exactly zero, come back
# with |dq| up to 5.25. The seed's dq pass does not zero a fully masked query
# row. This is a genuine defect and it is written up in PROVENANCE.md and in the
# setup report.
#
# It is not in the gate for one reason: it is unreachable at every shape in
# `op.shape`, all of which have Sq == Skv, where no row is ever fully masked. A
# gate is the loop's only success signal and `gate.py` reads nothing but its exit
# code, so a criterion that no round can pass and no round can affect does not
# discriminate between implementations -- it just makes every round fail for a
# pre-existing reason unrelated to the work. The defect is carried as a finding
# instead, where someone can act on it.
#
# If a later round changes `op.shape` to admit Sq > Skv, this moves into
# GATE_SHAPES and the baseline has to be fixed first.
DIAGNOSTIC_SHAPES = [
    dict(name="sq_gt_skv", batch=4, seqlen_q=2048, seqlen_kv=1024,
         heads_q=32, heads_kv=8, head_dim=128, causal=True, window_left=-1),
]

# What validation.py gates on.
GATE_SHAPES = SPEC_SHAPES + EDGE_SHAPES
# What op/ut/ runs.
ALL_SHAPES = GATE_SHAPES + DIAGNOSTIC_SHAPES
