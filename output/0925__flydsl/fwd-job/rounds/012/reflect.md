# Round 12 reflect (fast)

**Not accepted.** Python's score was 0.90181 against the incumbent's 0.90614. Throughput was 1.0002x
the best ever, short of the +0.70% bar. `op/current` is still round 11.

## What happened
- Built r12.i1.g31: 2-way split-KV on the R=1 path (fast only), with a fixed-order FlyDSL combine.
- Correctness passed once two FlyDSL AST-rewriter errors and one combine indexing bug were fixed:
  16/16 at >= 49.97 dB and 200/200 bitwise.
- Speed:

  | shape | vs champion |
  |---|---|
  | fast | 0.987x |
  | proxy | 1.000x (gated off) |
  | prod | 1.000x (gated off) |

- Row 2 (r10.i3.g28 pairing) was not built.

## What I got wrong
- **I priced the split from the lone-WG slope and assumed per-tile cost would stay put.**
  - It did not: 0.98 us per original tile against 0.72 predicted.
  - The attention kernel saved 5.7 us, not 11.5 us.
- **The combine side was priced about right.** It cost ~6 us against a ~5 us estimate.
- **The miss was in the half I treated as free.** I named "two WGs contend" as a way to be wrong,
  then gave it no weight.
- **I never found why the per-tile cost rose.** The dead-end entry says it is a symptom.
- **I mistook launch overhead for kernel fixed cost.** About 6 us of fast's wall time is outside
  the kernel (profiler 26.9 us vs event timing 32.6 us). Round 11 and my own M3 fit had counted it
  as kernel fixed cost.
- **The fast lever is smaller than the route claimed.** Anything that adds a dispatch starts ~6 us
  behind.
