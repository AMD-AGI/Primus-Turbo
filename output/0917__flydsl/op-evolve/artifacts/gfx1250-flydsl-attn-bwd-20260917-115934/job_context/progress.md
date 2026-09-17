# gfx1250-flydsl-attn-bwd -- progress

Generated from `state.yaml`; do not edit. Best round: **0**. Spec v000.

Achieved TFLOP/s per shape, from `benchmark.py` unprofiled -- the round's own numbers, not the profiled ones. `score` is the mean of each shape's capped ratio to its own target, so it can hide a shape that has not moved; that is what the columns are for. `mode` says how the round was reached: a fast round's `bound` is one agent's reading, a deep round's is six analyses. Round 0's figure is setup's own measurement and is the one number here taken in a different session -- nothing compares against it, because round 1 re-measures round 0's code in its own session and uses that. `beat` is the anchor the round was graded against and `target` is `margin x beat`; both are PINNED at setup (`op/beat/FIXED_BEAT.yaml`), so a round is never penalised for the anchor drifting under it. A round is kept when its THROUGHPUT improves on the best ever -- re-measured in the same session -- not when its `score` improves; `score` is still reported, and is no longer capped. **`gain` is the number the verdict was made on** -- this round's code against the best-ever code, re-measured beside it in the SAME session -- and `changed` is how many files the round edited. Read those two, not the throughput column: throughput figures come from different sessions and drift about 1% on identical code, so comparing them down the page is invalid. **`current(retest)` is that drift made visible**: the incumbent's own code, rebuilt and re-measured in THIS round's session, which is what the verdict compared the candidate against. Read each row across -- shape against `current(retest)` -- never a shape column down the page. When the two move together between rounds, the machine moved, not the kernel. A round with `changed = 0` produced no candidate at all and its `gain` is 1.0000 by construction. **Notes are in `note.md`**, not in a column here -- they run to several lines and one of them would make this table scroll sideways. The three `current` marks the one round `op/current/` is a copy of -- which is not simply the last accepted row: a refactor promotes a round outright, and an accepted round is only current until a later one wins. The three `ms` columns are the same three numbers as kernel time -- read a windowed mask by these and by bandwidth, never by TFLOP/s, which the window deflates by removing arithmetic without removing bytes.

| round | mode | score | gain | changed | accepted | bound | current |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | -- | -- | -- | -- | yes | -- | yes |

### fast

| round | TFLOP/s | current(retest) | ratio | beat | target | ms | beat ms | target ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3.0 | -- | -- | -- | -- | -- | -- | -- |

### proxy

| round | TFLOP/s | current(retest) | ratio | beat | target | ms | beat ms | target ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 36.4 | -- | -- | -- | -- | -- | -- | -- |

### prod

| round | TFLOP/s | current(retest) | ratio | beat | target | ms | beat ms | target ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 57.2 | -- | -- | -- | -- | -- | -- | -- |
