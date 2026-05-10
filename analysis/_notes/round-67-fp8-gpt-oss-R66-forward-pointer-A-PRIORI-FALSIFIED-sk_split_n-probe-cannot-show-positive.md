# Round 67 — R66 forward-pointer (sk_split_n probe) A-PRIORI FALSIFIED at HK source

**Verdict**: A-PRIORI FALSIFIED. R66 forward-pointed to writing
`_probe_round_<N>_sk_split_n_inscope.py` that calls the binding with
`sk_split_n ∈ {0, 2, 4}` on the 8 in-scope cells. Source inspection of
`HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` shows the
probe **structurally cannot show a positive result** because the
device-side kernel does not yet read the K-split fields. The probe is
guaranteed to print NEUTRAL-or-NEGATIVE. R67 closes this forward-pointer
without burning the build/probe cycle.

## Source evidence (HK HEAD `49ffb984`, kernel_fp8_layouts.cpp)

`grep -n "g\.sk_split_n\|g\.sk_partial_buf" analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`
returns **only** host-side dispatcher references:

| Line | Site | Role |
|---|---|---|
| 2969-2970 | struct field declaration `int sk_split_n; int* sk_partial_buf;` | data only |
| 7892 | `if (g.sk_split_n > 0 && g.bpc > 0 && g.ki > 0 && g.sk_partial_buf == nullptr)` host-side alloc gate | R13a |
| 7896-7899 | `hipMallocAsync` + `hipMemsetAsync` of T_max·256·256·4 B partial buffer | R13a |
| 8137-8138 | `hipFreeAsync` of owned partial buffer | R13a |
| 9079, 9115 | `g.sk_partial_buf = reinterpret_cast<int*>(sk_workspace_ptr)` caller-allocated workspace bridge | R17 |

Zero device-side references. The HK comment block at lines 7827-7846
(R13a) explicitly states this:

> R12 declared the trailing struct fields (sk_split_n, sk_partial_buf*)
> but did not allocate. R13a (this) lands the dispatcher-side alloc/free
> gated on g.sk_split_n > 0; **the kernel control-flow branch +
> atomicAdd + reduce post-kernel + per-cell dispatcher rule follow in
> R13b/R14/R15** per the R11 plan…

R13b / R14 / R15 (the kernel branch landing) **have not been committed**
in HK. The most recent HK Stream-K commits are:

```
49ffb984 infra(fp8 grouped rcr): round-17 — caller-allocated workspace via sk_workspace_ptr (production NEUTRAL)
4e9f6b62 infra(fp8 grouped rcr): round-14 — pybind kwarg sk_split_n for R13a alloc-cost probe (production NEUTRAL)
43f37f8b infra(fp8 grouped rcr): round-13a — Stream-K host-side allocator (gated, NEUTRAL)
bc5df92d infra(fp8 grouped rcr): round-12 — Stream-K scaffolding fields (sk_split_n, sk_partial_buf), NEUTRAL
```

R14 in HK numbering is the **pybind kwarg** for the alloc-cost probe,
not the kernel branch landing. R15 (the per-cell dispatcher rule) and
R13b (the device-side branch) are both unwritten.

## What the probe would have measured

For each of the 8 in-scope cells × `sk_split_n ∈ {0, 2, 4}`:

* **sk_split_n=0**: host-side gate condition `g.sk_split_n > 0` is
  false → no alloc, no memset → identical to current production code
  path → measures baseline ±noise.
* **sk_split_n>0** with no caller workspace: triggers `hipMallocAsync`
  (per R14 measurement, 2.9-9.1 ms/call alloc cost) + `hipMemsetAsync`
  of a 44-88 MiB partial buffer + later `hipFreeAsync`. Kernel itself
  receives a non-null `g.sk_partial_buf` it never reads → math
  bit-identical to baseline → **slower than baseline by 2.9-9.1 ms per
  call**, dwarfing the 105 µs baseline kernel wall.
* **sk_split_n>0** with `sk_workspace_ptr` set (R17 path): skips the
  per-call alloc but still pays the `hipMemsetAsync` zero-init of the
  workspace. Kernel still doesn't read the buffer → math bit-identical
  → slower than baseline by the memset cost (44-88 MiB / HBM3e BW
  ~5 TB/s ≈ 9-18 µs per call, ~10-17% of the 105 µs baseline kernel
  wall on Down-B4-M2048).

In all three cases the lift envelope is **non-positive by construction**.
The probe is structurally a NEUTRAL-or-NEGATIVE-only experiment.

## What R66 missed

R66 wrote the forward-pointer based on the HK commit log claim that R12
through R17 shipped the "Stream-K infra" with each commit message noting
"production NEUTRAL". The R66 author read this as "infra in place,
ready to dispatcher-rule it on" — symmetric to the chunk_size /
num_slots / fuse_ktail_off pattern where the kernel branch landed
together with the pybind kwarg. For Stream-K the HK author deliberately
split alloc (R13a) from kernel branch (R13b) per the risk-isolation
rationale at lines 7836-7846, and R13b never landed.

## Why R13b/R14/R15 (HK numbering) is a real R67-class commit budget

The kernel-side branch needs:
1. K-split coord decode in the persistent loop (every CTA computes its
   own `(k_lo, k_hi)` window from `(sk_tile_id, sk_share_idx, sk_split_n)`)
2. `atomicAdd<float>` write into `g.sk_partial_buf[sk_tile_id]` per
   K-window completion
3. Post-kernel reduce launch (separate kernel) that scans
   `sk_partial_buf` and writes the FP8 result with rescale + ·s_a·s_b
4. Dispatcher rule in `select_default_config` predicating on
   `(tiles_m, tiles_n, k, m_total)` for the in-scope gpt_oss family
5. SNR>25 dB tight-verify on every in-scope cell (atomicAdd order is
   non-deterministic; accumulation-order risk is real)

That's a 200+-line kernel body landing in a 700-line MFMA kernel
already at 256 VGPR / 37 spill near LLVM's allocator ceiling — exactly
the entanglement the R12→R13a split avoided. Per SKILL.md NEW DIRECTIONS
A1, this is a 4-6 round project, exceeding the per-round commit budget
the auto_optimize daemon enforces.

## Falsification gates re-checked

1. **Primus-Turbo HEAD** (`4485e46f`): zero functional commits since R55;
   only doc notes R55-R66.
2. **HipKittens HEAD** (`49ffb984` — same as R63-R66): no functional
   change since R64. R13b/R14/R15 unwritten.
3. **No new lever** in the SKILL.md NEW DIRECTIONS A-G untried list. All
   seven directions remain closed; A1 (the only one with HK infra
   shipped) is gated on R13b which is itself a 4-6 round project.
4. **Daemon noise model** (R57-R66, n=10):
   ```
   round   metric
   R57     697
   R58     696
   R59     696
   R60     696
   R61     695
   R62     693
   R63     694
   R64     695
   R65     697
   R66     695   ← daemon print of 4485e46f
                n=10  mean=695.40  stdev=1.27  range=[693,697]
   ```
   R66 print 695 lands at the median exactly. R67 expected median 695,
   90% CI [693, 697], unchanged from the R56 noise model.

## Operator recommendation (7th repeat, escalated)

**Terminate this run.** The metric is on a stationary noise distribution
centered at 695±1 with all known per-round-budget levers exhausted. R67
adds one more piece of evidence to the saturation case: the R66
forward-pointer that looked legitimate on commit-log inspection is
falsified at source-inspection. There are now **zero remaining
single-round legitimate moves**.

**If continuing is non-negotiable**, the only known route to a higher
median is to relax the per-round commit budget (or split the work
across rounds with explicit "do not metric until R+N" gates) so the HK
author can land R13b/R14/R15 (kernel-side K-split + reduce + dispatcher
rule) as a single physical change set. Per SKILL.md A1 estimate that is
4-6 commits with no per-round metric lift until the final commit.

## Forward-pointer (revised from R66)

If a future round picks up this task, do **not** run the R66-suggested
sk_split_n probe — it is a-priori falsified per the source evidence above.
Instead, the legitimate next moves in priority order:

1. **Land HK R13b in HipKittens** (kernel-side K-split coord decode +
   atomicAdd write into sk_partial_buf). This is a kernel edit only —
   it doesn't change the metric on its own (default `sk_split_n=0` keeps
   the K-split branch unentered) but it unblocks #2.
2. **Land HK R14 in HipKittens** (post-kernel reduce that scans
   sk_partial_buf and writes the final FP8 output). Same property:
   default-off, doesn't move metric on its own.
3. **Land HK R15 / Primus-Turbo dispatcher rule** that predicates
   `sk_split_n=2` or `=4` on the in-scope cells. SNR-gate on every shape;
   if any cell loses, narrow the rule.
4. If after #1-#3 the metric still does not lift ≥+5 score, falsify A1
   for these cells and pivot to A3 (decoupled-warps) — the last NEW
   DIRECTION attack on the R21 barrier-pin etiology.

Each of #1-#3 is a single round's commit; #4 is itself a 3-5 round
project. Total: 4-8 rounds with no per-round score lift expected until
the final commit of #3.
