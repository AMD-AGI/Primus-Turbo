# Round-47 — var-K wgrad uniform-M closed-form per-tile decode (Direction D-1) FALSIFIED

**Run:** `gpt_oss_fp8_local_20260509_143917` round 1 (auto-script numbering;
content-numbering "R47" continues from prior 46-round series).
**Direction (per task.md):** D — SALU coord-decode hoist. Sub-target #1: the
6-level binary search in `grouped_var_k_kernel_fp8`'s outer tile loop.
**Verdict:** FALSIFIED at +1-score / 5-sample threshold (≤ noise).
**Score:** baseline 692 median (n=5, range 691-693) vs patched 692 median
(n=5, range 690-693). Δ = 0.

## Hypothesis

R38 (HK ad501f0a) established as a load-bearing invariant that var-K's
LDS prefix-sum cache is uniform-by-construction:

```cpp
// kernel_fp8_layouts.cpp:8351-8362 (post-R38)
const int tiles_per_group = g.bpr * g.bpc;
if (threadIdx.x <= g.G && threadIdx.x < MAX_G_PLUS_1) {
    s_offs[threadIdx.x] = static_cast<int>(g.group_offs[threadIdx.x]);
    s_cum_tiles[threadIdx.x] =
        static_cast<int>(threadIdx.x) * tiles_per_group;
}
```

i.e. `s_cum_tiles[k] == k * tiles_per_group` exactly, for every legal
var-K dispatch (output tile-grid `[bpr, bpc]` is a kernel-arg-defined
constant, not a per-group property — see R38 note for the derivation).

Under this invariant, the 6-level binary search at the top of each
outer tile iteration:

```cpp
for (int gt = pid; gt < total_tiles; gt += slots_eff) {
    int lo = 0; int hi = MAX_G_PLUS_1 - 1;
    #pragma unroll
    for (int level = 0; level < 6; ++level) {
        const int mid = (lo + hi + 1) >> 1;
        if (gt >= s_cum_tiles[mid]) lo = mid;
        else hi = mid - 1;
    }
    const int group_idx = lo;
    const int tile_start = s_cum_tiles[lo];
    const int local_tile = gt - tile_start;
```

collapses analytically to a single integer divide-by-runtime-constant:

```cpp
const int group_idx = gt / tiles_per_group;
const int tile_start = group_idx * tiles_per_group;
const int local_tile = gt - tile_start;
```

Bit-equivalent (verified: 8/8 SNR PASS at 49.6 dB on patched build).
Replaces 6 dependent LDS-read + ALU-update chains with one runtime
divide-by-constant + one mul.

The hypothesis was that this would clip a measurable slice of var-K
wgrad's R21-PMC-observed SALU bottleneck (Down_B4_M2048 wgrad
SALU/SQ_busy = 85.4% per task.md table; SALUBusy total = 9.1% per R21
itself). Predicted lift if uniformly transferred: +3-9 score on the
wgrad section.

## Method

1. Edit `HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`
   line 8378-8389: replace binary search with closed-form divide
   (R47 patch with mechanistic comment block).
2. Build via `dbg_remote.sh` — clean compile, **VGPRs Spill: 0** (no
   regression on the var-K kernel's known 37-dw spill envelope), LDS
   unchanged.
3. Correctness gate: 8/8 PASS, SNR ≥ 49.6 dB across all shapes.
4. Score gate: 5 metric samples on remote MI355X, GPU 3 (matches
   daemon canonical-metric pin).

## Result

```
baseline (HK 3bcd248a, binary search):  693, 692, 691, 692, 691  → median 692
patched  (HK 3bcd248a + R47 divide):    690, 692, 692, 693, 692  → median 692
```

Δ median = 0. Range overlap is total. Per-section breakdown on the
patched single-sample run (sample 1):

```
fwd     1907 T (baseline 1909) → noise
dgrad   2089 T (baseline 2093) → noise
wgrad   1805 T (baseline 1801) → +4 T, also noise (per-shape σ ~30)
```

No section moves outside per-sample noise. Verdict: **change is a
no-op at metric resolution**.

## Why the prediction was wrong

Two compounding factors:

1. **Per-tile prologue is a small fraction of CTA wall**. For
   Down-B4-M2048 wgrad (worst cell): each CTA processes
   ~`total_tiles / slots_eff` = `4*8*11 / 256` ≈ 1.4 outer iterations.
   Per outer iter the K-loop runs `ki_g - 2` = `2048/128 - 2` = 14
   iterations of ~250 cycles each (4 MMA pairs + 5 barriers + 2 LDS
   loads per iter, R21 anatomy). The binary search prologue (6×~30 cy
   = 180 cy SALU) is **<2% of CTA wall**. Closing it 100% would
   contribute under 1 score, well inside the ±2 noise floor.

2. **R21's 85% SALU/SQ_busy reading is inside the K-loop, NOT in the
   prologue**. The K-loop body emits a steady stream of SALU work for
   per-iter address arithmetic (br*2+k+1 swizzle deltas, tic^toc
   register-rename moves, swizzled-offset table lookups, lambda
   inlines of `a_co`/`b_co`/`load_a`/`load_b`). The PMC counter
   `SALU_INST_COUNT / SQ_BUSY` integrates over the whole kernel; the
   prologue contributes only a tiny tail to the numerator.

The R21 forward pointer to "Direction D — SALU coord-decode" was
correct in *direction* but pointed at the wrong *site*. The right
target is the K-loop body's per-iter SALU, not the per-tile prologue.

## Forward pointer — Round-48 (Direction D-2)

The K-loop body at lines 8503-8585 has several per-iter SALU items
that *could* be hoisted out of the iter (rematerialized in CTA-static
or per-warp scalar regs):

| Item | Per-iter SALU | Hoist candidate |
|---|---|---|
| `br*2`, `bc*2`, `bc*2+1`, `br*2+1` tile-coord deltas for `global_load_a/b` | 4× scalar add, 4× shift | constant per tile — already CTA-scalar-loop-invariant; LLVM may already CSE |
| `k+1`, `k+2` for prefetch lambdas | 2× scalar add | trivially loop-variant; can't hoist |
| `tic^toc` ping-pong indices into `As[][]`, `Bs[][]` | 2× XOR, 2× scaled offset | same as above — loop-variant by design |
| `wm * RBM`, `wn * RBN` lane offsets in `load_a/load_b` lambdas | 2× scalar mul per call (×3 calls) | **loop-invariant** per CTA — top hoist candidate |
| Swizzled-offset prefill (`soA[mptA]`, `soB[mptB]`) reads inside `global_load_a/b` lambdas | per-call array deref | already prefilled outside loop, but per-call may re-read; verify with a `cs_amd_tail_block` printout |

R48 plan:
1. Compile current production with `-S` (or grep `kernel_fp8_layouts.s`
   from existing build) and inspect the K-loop body for redundant
   SALU. Specifically look for: per-iter recomputation of
   `wm * RBM` / `wn * RBN`, redundant materialization of `br*2`, etc.
2. If recomputation observed, hoist via explicit scalar-register
   variables defined *outside* the loop and refer-to-by-reference inside
   the lambdas. (The lambdas currently capture by `[&]`; the captured
   `wm`/`wn` are already loop-invariant, but the multiplications happen
   inside the lambda body.)
3. Build + 9-sample metric. Ship if median lift ≥ +1.

If R48 also FALSIFIED, escalate to Direction D-3:

* **R49 (D-3, structural)**: The K-loop's 5-barrier cadence (R21 anatomy)
  is what's keeping MFMA pipe at 32% util. Try `roc_shmem`-style
  warp-group split (warp 0-3 = LDS load producer, warp 4-7 = MFMA
  consumer) — drops the CTA-wide barrier dependency. This is the
  decoupled-warps direction (task.md A3); 4-6 round project.

## Files touched this round (Primus-Turbo)

* `analysis/_notes/round-47-vark-uniform-m-closed-form-coord-decode-FALSIFIED.md` (this doc).

## Files touched this round (HipKittens)

* None (R47 patch reverted after 5-sample falsification on GPU 3).

## Closure

Direction D sub-target #1 (per-tile prologue closed-form) closed:
correct mechanistic transformation, zero metric impact. R21 PMC-bottleneck
attribution refined: the SALU/SQ_busy = 85% reading is dominated by the
**K-loop body**, not the per-tile prologue. R48 will inspect K-loop
SALU directly via assembly or PMC slicing before the next surgery.
