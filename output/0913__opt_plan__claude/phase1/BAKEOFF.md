# Round-0 bake-off — the in-tree backend lost

**Date:** 2026-09-13. **Card:** gfx1250 / MI455X, still VR-throttled (MAX_CLK 1100 MHz).
**Shape everywhere:** `b=4 s=8192 hq=32 hkv=8 d=128 bf16 causal`, 7.697 TFLOP per fwd+bwd
call, 32 calls per training step.

**Conditions:** one session, one image, one process pool. Identical fp32 reference,
identical four-tensor SQNR gate, identical timer for every entry. This is the controlled
comparison that Phase 1 was supposed to run **first**.

---

## 1. The result

| impl | fwd ms | bwd ms | total ms | TFLOP/s | ms/step |
|---|---:|---:|---:|---:|---:|
| **aiter + 2 knobs** | **3.260** | 27.947 | **31.207** | **246.6** | 999 |
| torch flex (anchor) | 7.506 | **23.831** | 31.337 | 245.6 | 1003 |
| aiter shipped | 6.676 | 27.888 | 34.565 | 222.7 | 1106 |
| **turbo champion (ours)** | 4.152 | 32.252 | 36.405 | 211.4 | 1165 |
| turbo shipped | 10.673 | 48.969 | 59.642 | 129.0 | 1909 |

All four contenders passed the SQNR gate on `out`/`dq`/`dk`/`dv` against the fp32
reference. Nothing here is a fast-but-wrong result.

**Read the table honestly:**

- Our tuned in-tree backend, at **36.405 ms**, is **slower than aiter's shipped,
  untouched backend** (34.565 ms, 1.05x) and **1.17x slower than aiter with the same two
  knobs applied** (31.207 ms).
- Our forward is genuinely good: 4.152 ms beats flex's 7.506 ms by 1.81x. It is still
  1.27x behind aiter's tuned forward (3.260 ms).
- **The entire deficit is the backward: 32.252 ms vs aiter's 27.888 ms = 1.16x.**
- Tuning did work — 59.642 -> 36.405 ms is 1.64x over the shipped config, and that
  matches §3's 1.66x to within run-to-run spread. It simply was not enough to win.

Run-to-run spread, for calibration: this session measured the turbo champion at
36.405 ms where §3 measured 35.704 ms (2.0% apart), and flex at 31.337 ms where §1
measured 31.337 ms (exact). Treat differences under ~3% as noise; the 1.17x gap to aiter
is far outside it.

---

## 2. Sequencing error: this should have run first

The Phase 1 plan named Round 0 — "re-measure all paths in one image, one session" — as
the first step, and explicitly as *"the highest-value probe in the plan"*. It was not run
first. Tuning turbo started immediately instead.

**That was a mistake and it cost time.** Had the bake-off run on day one, the two knobs
would still have been found (they transfer), but the days spent on turbo's scheduling
knobs — `waves_per_eu` sweeps, the mask-skip edit, the `lo`-bound edit, the `log_p_scale`
fold, the register-spill investigation — would have been spent instead on the one
difference that actually separates the two backends, which is structural and is described
below. Every one of those scheduling experiments landed at neutral or worse (§7).

The general rule this earns: **never tune an implementation before establishing that it is
the right implementation to tune.** A bake-off is cheap (one session) and it re-scopes
everything downstream.

---

## 3. Why aiter's backward wins: asymmetric block shapes

This is not a scheduling difference. It is a tiling difference, and turbo currently cannot
express it.

| | turbo `_bwd_kernel_dkdv` / `_dq` | aiter backward |
|---|---|---|
| dkdv tile | `FIXED_BLOCK_M = 64`, `FIXED_BLOCK_N = 64` | `BLOCK_M1 = 32`, `BLOCK_N1 = 128` |
| dq tile | same two constants | `BLOCK_M2 = 128`, `BLOCK_N2 = 32` |
| shape | **symmetric**, hardcoded module constants | **asymmetric**, and *paired*: `N1 == M2`, `M1 == N2` |
| other | — | `waves_per_eu = 1`, `BLK_SLICE_FACTOR = 2` |

aiter's two backward halves are transposes of each other: the dkdv half walks a short
query tile (32) against a long key tile (128); the dq half walks a long query tile (128)
against a short key tile (32). Each half gets a long inner axis where it needs one. The
pairing (`N1 == M2`, `M1 == N2`) is what keeps the shared launch grid consistent across
both halves — the same coupling that produced the known "fast but wrong `dq`" trap when
`BLOCK_N1` is raised without raising `BLOCK_M2` (§9 of the plan).

turbo has one symmetric 64x64 tile for both halves and therefore serves neither well.

### Why turbo cannot just change the number

`dkdv`'s `BLOCK_M` is **not a free knob**. It is bound to a cross-kernel LSE/delta ABI:

1. `attn_fwd` writes LSE at offset `m * BLOCK_M * 2`.
2. `_bwd_preprocess_use_o` writes `delta` at `+ BLOCK_M` inside that same block.
3. Host-side `_lse_delta_views` reconstructs the two interleaved views from that layout.
4. `_bwd_kernel_dkdv` reads `2 * start_m + tl.arange(0, 2 * BLOCK_M)` and splits the
   result back into lse and delta with two `tl.gather`s.

Changing `BLOCK_M` in isolation does not fault. It returns lse rows concatenated with
delta rows and produces **smoothly wrong** `dk`/`dv` — caught only by the four-tensor
gate. So the block size is an ABI parameter shared by three kernels and one host-side
view helper, not a tuning parameter.

**Decoupling that ABI is the work currently in progress.** It is the prerequisite for
trying anything aiter-shaped.

### What the ceiling looks like if it lands

Arithmetic only, not measured: turbo's forward (4.152 ms) plus aiter's backward time
(27.888 ms) would be **32.04 ms / 240.2 TFLOP/s** — within ~3% of both flex and tuned
aiter. That is the prize, and it is the honest size of the prize: **parity, not a win.**

---

## 4. The 316.1 TFLOP/s figure did not reproduce

An earlier op-evolve measurement put aiter + the two knobs at **24.347 ms / 316.1
TFLOP/s**. That number has been carried through the plan as aiter's established
performance. **It does not reproduce.** Measured in this session: **31.207 ms / 246.6
TFLOP/s.**

The gap is entirely in the backward:

| | then | now | Δ |
|---|---:|---:|---:|
| fwd | ~3.272 | 3.260 | reproduces |
| bwd | **21.075** | **27.947** | **+6.872 ms** |
| total | 24.347 | 31.207 | +6.860 ms |

The forward reproduces exactly. The backward is 1.33x slower than the archived figure, and
that single delta accounts for the whole total.

**Mechanism: upstream aiter retuned its shipped backward config.** The old measurement was
taken against a different aiter revision. Two pieces of evidence:

- aiter's **forward** `num_stages` 1 -> 2 still reproduces exactly (6.676 -> 3.260 ms,
  2.05x). The forward knob is unchanged upstream, and so is the forward number.
- aiter's **backward** `num_warps` 4 -> 2 **no longer helps**: 27.888 -> 27.947 ms, which
  is noise. On the older revision this was a real win. Upstream has since moved the
  backward to a config where that knob is already spent.

**Action: retire 24.347 ms / 316.1 TFLOP/s from every target and comparison.** aiter's
current number on this card, at this clock, is 31.207 ms / 246.6 TFLOP/s. Any plan
milestone derived from 316.1 needs recomputing.

---

## 5. What this changes about the plan

1. **The champion config still ships.** It is 1.64x over the shipped turbo default, it is
   config-only (two integers, zero kernel edits), it is bitwise deterministic across 2,300
   invocations (§12), and it is E2E-neutral (§10). Nothing about the bake-off argues
   against landing it. It is just no longer the finish line.
2. **Stop spending rounds on turbo's scheduling knobs.** They are exhausted. `waves_per_eu`
   is 0-or-nothing (2 and 4 fall off a cliff at 54.9 / 104.4 ms); `bwd num_stages > 1` is
   worse; the three source edits tried were -9%, neutral, neutral, and were reverted. The
   register-pressure hypothesis is refuted outright — `dkdv` has `vgpr_spill_count = 0` at
   `num_warps` 2/4/8 and spills only at 1, so it is per-program-efficiency bound.
3. **The next work item is the LSE/delta ABI decoupling**, so `dkdv` and `dq` can take
   independent, asymmetric block shapes. Everything else in the backward is downstream of
   it. This is in progress.
4. **Re-anchor the target ladder.** The near-term realistic target for turbo's backward is
   aiter's **27.9 ms**, not flex's 23.8 ms. flex still owns the fastest backward measured
   on this card by 1.17x over aiter, and nothing yet explains why.
5. **A live option is to stop competing.** aiter + 2 knobs is 31.207 ms *today*, with no
   kernel work at all. If the ABI decoupling turns out to be expensive, "dispatch to aiter's
   Triton backend on gfx1250" is a legitimate outcome and should be costed against
   continuing. The in-tree backend's value was always *"a path on gfx1250 that is correct
   in both directions and that we can edit"* — that value survives losing the bake-off, but
   it is not a performance argument.

---

## 6. What this does NOT change

- **The E2E result still means only what it meant.** turbo 245 tps vs flex 244 tps shows
  that **enabling turbo attention is not a regression** — and nothing more. On this image
  the whole attention path is 36.4 ms x 32 = 1.16 s out of a 133.7 s step, **under 1%**.
  The step is GEMM-bound: rocBLAS bf16 dense GEMM measures 27.4 TFLOP/s against a measured
  1002.7 TFLOP/s roof, and hipBLASLt's gfx1250 Tensile library is absent from the image. A
  dense GEMM running 8x slower than a flash-attention kernel is backwards. **E2E on this
  image cannot validate attention work in either direction**, and it could not have told us
  that turbo lost the bake-off.
- **The throttle still dominates everything.** Every number above is at 1100 MHz against
  1699-1703 MHz measured on this same card on 2026-09-04. The platform escalation is worth
  more than any kernel result in this document.
- The refuted items stay refuted: `sequence_parallel=False` is silently wrong (it computes
  1/128 of the gradient), the XCD remap would hurt (`gridDim0` is already a multiple of 8),
  FlyDSL is unavailable and would reject `G=4` anyway, and AITER's CK fmha path is
  CDNA-only.
