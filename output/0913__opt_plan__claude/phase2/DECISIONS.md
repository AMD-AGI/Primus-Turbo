# Autonomous run — decision log

Kept because the operator is away and asked for decisions to be recorded and the
recommended option taken without waiting. Each entry: what was decided, on what evidence,
and what would reverse it.

---

## D1. Keep the LSE-ABI decoupling even though it bought no speed

**Decided:** keep it.

`_bwd_kernel_dkdv` read the shared lse/delta scratch as `2*start_m + arange(0, 2*BLOCK_M)`
and split the halves with two `tl.gather`, which silently required its `BLOCK_M` to equal
`attn_fwd`'s. Now it indexes per row against a separate `LSE_ABI_BLOCK` constexpr.

**Evidence it is safe:** at the default 64x64 all four outputs are **bitwise identical** to
the pre-change kernel (out `e49c829d8e4a5809`, dq `badc32dde15caaf0`, dk `00b126f7e25873ae`,
dv `29a877549708422e`). That is the gate this change had to pass; SQNR closeness would not
have been enough.

**Why keep it with no speed gain:** it converts a silent-wrong-answer trap into a working
knob. Before, setting dkdv `BLOCK_M=128` returned lse-rows concatenated with delta-rows —
no fault, no shape error, smoothly wrong dk/dv. It also removes two cross-lane gathers per
iteration.

A second, related trap was fixed with it: `num_block_m` is dkdv's query-axis loop bound and
must follow *its* `BLOCK_M`. Left at the old `FIXED_BLOCK_M` count, a larger tile makes the
loop stop short and leaves the tail of dk/dv at its `zeros` init — **and a timing-only sweep
reports that as a speedup.**

**Reverses if:** a shape is found where the per-row indexing is not bitwise-neutral.

---

## D2. Asymmetric dk/dv tiles do NOT help turbo — hypothesis refuted

**Decided:** stop pursuing tile shape on turbo's dkdv; 64x64 stays.

The decoupling was done to test the hypothesis that turbo's backward was slow because it
could not use aiter's asymmetric tile. Measured, production shape, all SQNR-correct:

| blocks | bwd ms | vs 64x64 |
|---|---:|---:|
| **64x64** | **32.447** | — |
| 32x64 | 35.545 | +10% |
| 64x128 | 36.501 | +12% |
| 128x64 | 43.719 | +35% |
| 128x128 | 43.908 | +35% |
| **32x128 (aiter's)** | **51.563** | **+59%** |
| 64x256 | 58.233 | +79% |
| 32x256 | 125.998 | +288% |
| 16x256 | 237.886 | +633% |

64x64 was already optimal, and aiter's own tile is the worst of the practical options **in
turbo's kernel**. Tile shape is not transferable between the two implementations — it is a
property of the surrounding kernel structure, not of the shape.

**Reverses if:** the kernel structure changes (e.g. a fused backward), at which point the
tile space must be re-swept rather than assumed.

---

## D3. The real difference is kernel fusion, not tiling

**Evidence** (torch profiler, production shape, champion configs):

| | structure | backward |
|---|---|---:|
| turbo | **2 kernels**: `_bwd_kernel_dkdv` 18.96 ms + `_bwd_kernel_dq` 11.26 ms | 30.6 ms |
| aiter | **1 fused kernel** 27.55 ms | 27.7 ms |

Both flex and aiter fuse; turbo does not. Fusion accounts for roughly 3 ms of the ~4.3 ms
backward gap between turbo and aiter.

**Decided:** do not rewrite turbo's backward as a fused kernel yet. Rewriting is a large,
high-risk change for ~8% of the op, and there is a cheaper lever first (D4).

---

## D4. Chase flex's 23.8 ms backward via aiter's config, not via a rewrite

flex's backward is **23.831 ms** — the best anyone has produced on this shape, and it is
also Triton, so it is reachable in principle. Inductor's autotune winner was
`BLOCK_M1=64, BLOCK_M2=128, BLOCK_N1=128, BLOCK_N2=64`; aiter ships
`BLOCK_M1=32, BLOCK_N1=128, BLOCK_M2=128, BLOCK_N2=32`. The two differ only in M1/N2.

**Decided:** sweep aiter's fused backward toward flex's shape. This is a config change on an
already-fused kernel, i.e. minutes of work against weeks for a turbo rewrite.

**Guardrail implemented:** the harness now asserts `BLOCK_N1 == BLOCK_M2` and
`BLOCK_M1 == BLOCK_N2` before running. The launch grid is sized by `BLOCK_N1` and the *same*
grid serves the dq half, which is tiled by `BLOCK_M2`; breaking the pairing produces a
config that measures **faster** with dq silently covering part of the query axis. This is
the documented `BLOCK_N1=256` trap, and it is now rejected at dispatch rather than
discovered in a ledger.

---

## D5. Target is flex's 23.8 ms backward, not the historic 316.1 TFLOP/s

The 316.1 TFLOP/s / 24.347 ms figure from an earlier probe **does not reproduce** on current
aiter main: measured 31.207 ms / 246.6 TFLOP/s in a controlled same-session bake-off. The
gap is entirely backward (21.075 ms then vs 27.947 now) and is explained by upstream aiter
having retuned its shipped backward config since.

**Decided:** use measured, reproducible anchors only — flex at 31.337 ms total / 23.831 ms
backward. Chasing an unreproducible number would mean tuning against a target the machine
has never produced.

---

## D6. Stay on Triton; do not start a hand-written WMMA kernel

flex reaches a 23.831 ms backward **in Triton**. That is an existence proof that Triton is
not the constraint at this shape. A hand-written backend is only justified once something
measured — not argued — shows Triton cannot close the remaining gap.

**Reverses if:** a Triton config search plateaus well above 23.8 ms with the mechanism
understood and unfixable in Triton.

---

## D7. A silently-ignored override, and why my own assertion missed it

The first aiter backward sweep returned six configs spanning **0.6%** (27.75–27.92 ms).
That flatness is the documented signature of an override that never applied — the plan
warned about it, the harness has an assertion for it, and **the assertion passed anyway.**

**What happened.** The config function lives in
`aiter/ops/triton/_triton_kernels/attention/mha_onekernel_bwd.py`, but the backward
*wrapper* at `aiter/ops/triton/attention/mha_onekernel_bwd.py:10` does
`from ... import _get_config` — binding the name into its own module at import time.
Patching the source module rebinds the source's name; the wrapper keeps calling the
original. So `_get_config()` dutifully returned the new value while the launch used the old
one.

**Why the assertion was insufficient, and this is the transferable lesson.** It checked
that *the config source returns what was asked for*. That is not the same claim as *the
kernel ran with it*. The two differ exactly when something holds its own reference — an
import binding, a closure, a captured default, a cached compile.

**What actually settled it.** aiter bakes the tile sizes into the Triton kernel *name*, so
the profiler shows the ground truth:

```
asked BLOCK_M1=64 ; _get_config returns 64
  KERNEL LAUNCHED: bwd_kernel_causal_BLOCK_M1_32_BLOCK_N1_128_BLOCK_M2_128_BLOCK_N2_32_...
```

After also patching the wrapper's binding:

```
  KERNEL LAUNCHED: bwd_kernel_causal_BLOCK_M1_64_BLOCK_N1_128_BLOCK_M2_128_BLOCK_N2_64_...
```

**Rule adopted: verify at the launch, not at the source.** Where a backend encodes its
config in the kernel name, that name is the cheapest available proof and should be recorded
in the ledger. Where it does not, find another observable that changes with the config —
register count, shared-memory size, grid — and assert on that instead.

The first sweep's results are void and have been discarded rather than corrected; they were
twelve measurements of one configuration.

---

## D8. BREAKTHROUGH — 25.283 ms / 304.4 TFLOP/s, beating flex

Once the override actually reached the launch (D7), the sweep produced a real result.

| config (aiter fused backward) | bwd ms | total ms | TFLOP/s |
|---|---:|---:|---:|
| **M1=32, N1=256, M2=256, N2=32** | **22.031** | **25.283** | **304.4** |
| M1=64, N2=64 (i.e. N1/M2 stay 128) | 22.284 | 25.543 | 301.3 |
| aiter shipped | 27.852 | 31.107 | 247.4 |
| torch flex (the anchor) | 23.831 | 31.337 | 245.6 |
| turbo champion | 32.252 | 36.405 | 211.4 |
| turbo shipped | 48.969 | 59.642 | 129.0 |

Against the anchor this is **1.24x on the op**, and the **22.031 ms backward beats flex's
23.831** — flex previously owned the fastest backward measured on this card.

Against where this campaign started (turbo shipped, 59.642 ms) it is **2.36x**.

**The lever is not what the plan predicted.** The hypothesis was that M1 mattered, because
flex's autotune winner differs from aiter's shipped config only in M1/N2. M1 turns out to be
nearly irrelevant: M1=64,N2=64 at the shipped N1=128 gives 22.284, and M1=32 at N1=256 gives
22.031 — a 1% difference. What moves the number is **N1/M2 = 256**, the K-axis tile, worth
27.852 -> 22.031 = **1.26x on the backward by itself**.

That also retro-explains D2: in turbo's *unfused* kernel, larger N was catastrophic
(64x256 = 58.2 ms, 32x256 = 126.0 ms). In aiter's *fused* kernel the same direction is the
win. Tile preference is a property of the kernel structure, which is exactly what D2
concluded — now with the sign demonstrated in both directions.

**Not yet converged:** N1=256 was the top of the swept range, so an extension to 512 is
running. Per the harness rule, a winner at a range edge means the range was the constraint.

**The pairing assertion earned its place:** three configs were rejected before running,
including `M1=128,N1=64,M2=64,N2=128`. Without the gate those would have entered the ledger
as timings.

### What this changes strategically

The in-tree turbo backend is now **1.44x behind** the best measured configuration, and the
gap is structural (two kernels vs one fused). Continuing to tune turbo's kernel is no longer
the best use of GPU time. The options, in order:

1. **Ship a dispatch path to aiter's Triton MHA on gfx1250** — largest, cheapest win, but
   adds a dependency Primus-Turbo does not currently have on this arch.
2. **Vendor aiter's fused backward into Primus-Turbo** — the plan's original seed proposal,
   now with a measured 1.44x justification rather than an argument.
3. Keep tuning turbo's two-kernel backward — bounded above by the fusion gap.

Recommendation on the evidence: (2). It captures the win in-tree, which is what the
deliverable asks for, and the vendoring cost is two Python files.

---

## D9. CHAMPION: 21.672 ms / 355.1 TFLOP/s — 1.45x flex, 2.75x the starting point

```
aiter Triton MHA, fused backward:
  BLOCK_M1=32, BLOCK_N1=256, BLOCK_M2=256, BLOCK_N2=32, BLK_SLICE_FACTOR=1
  forward: num_stages=2
```

| | bwd ms | total ms | TFLOP/s | vs flex |
|---|---:|---:|---:|---:|
| **champion** | **18.416** | **21.672** | **355.1** | **1.45x** |
| prev. best (BSF shipped =2) | 22.040 | 25.300 | 304.2 | 1.24x |
| aiter shipped | 27.852 | 31.107 | 247.4 | 1.01x |
| torch flex (anchor) | 23.831 | 31.337 | 245.6 | — |
| turbo champion | 32.252 | 36.405 | 211.4 | 0.86x |
| turbo shipped (start) | 48.969 | 59.642 | 129.0 | 0.53x |

SQNR 53.67 / 52.24 / 52.31 / 52.71 — the canonical correct band, unchanged from every other
correct backend.

**Launch-verified**, per D7's rule:
`bwd_kernel_causal_BLOCK_M1_32_BLOCK_N1_256_BLOCK_M2_256_BLOCK_N2_32_BLK_SLICE_FACTOR_1`.

**It exceeds the 316.1 TFLOP/s figure** that prompted this line of work — and unlike that
figure, this one is reproducible and was produced in a controlled same-session comparison
against a measured anchor.

### Every axis is converged, with the evidence

| axis | champion | neighbours |
|---|---|---|
| `BLOCK_N1` / `BLOCK_M2` | 256 | 128 -> 27.85 ms bwd; **512 -> 84.31 ms** |
| `BLK_SLICE_FACTOR` | 1 | 2 (shipped) -> 22.04; **4 -> 109.21** |
| `num_warps` | 4 (shipped) | **2 -> 87.23**; **8 -> 34.62** |
| `waves_per_eu` | 1 (shipped) | **0 -> 25.17**; **2 -> 50.07** |
| forward `num_stages` | 2 | 1 -> 6.68 ms fwd (2.05x worse) |

No winner sits at a swept edge except `BLK_SLICE_FACTOR=1`, which is that knob's hard
minimum. The cliffs are severe and asymmetric in every direction — 512 is 4.6x worse than
256, `num_warps=2` is 4.7x worse than 4 — which is why a coarse sweep with a correctness
gate was the right instrument and why the neighbours matter as much as the winner.

### The mechanism, and why the plan predicted the wrong lever

The plan expected `BLOCK_M1` to be the lever, because flex's autotune winner differs from
aiter's shipped config only in M1/N2. **M1 is nearly irrelevant** (M1=32 vs 64 at fixed N1
differ by ~1%). The two real levers are:

- **`N1`/`M2` 128 -> 256**: the K-axis tile. 1.26x on the backward alone.
- **`BLK_SLICE_FACTOR` 2 -> 1**: a further 1.20x on top.

Together 27.852 -> 18.416 ms, **1.51x on the backward**, with the forward's num_stages
giving 2.05x on its (much smaller) half.

### Consequence for the deliverable

turbo's in-tree backward (32.252 ms) is now **1.75x behind** the best measured backward
(18.416 ms). The gap is structural: two kernels versus one fused kernel, and the fused one
prefers a tile shape (N=256) that is catastrophic in the unfused one (D2: 32x256 -> 126 ms).
Tuning turbo's existing kernel cannot close it.

**Next action: vendor aiter's fused backward into Primus-Turbo**, which is what the original
plan proposed as the seed and what the bake-off now justifies with a measured 1.75x rather
than an argument.
