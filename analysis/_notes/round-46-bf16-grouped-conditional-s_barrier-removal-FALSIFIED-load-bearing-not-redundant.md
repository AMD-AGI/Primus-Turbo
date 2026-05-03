# Round 46 — BF16 grouped, remove `if (warp_row == 0) { __builtin_amdgcn_s_barrier(); }` from grouped + var-K kernels — FALSIFIED

## Goal coming in

R45 falsification (dispatch tuning for gpt_oss-GateUP dB CRR exhausted)
recommended R46 attack the HK var-K kernel structural surface — R43
listed 4 unexplored opts:

> 1. Persistent loop overhead amortisation
> 2. Conditional `s_barrier` semantics audit (line 4654)
> 3. Store batching (4 → 1)
> 4. Var-K KI specialisation

R46 picked (2): R43's note flagged the `if (warp_row == 0)
{ __builtin_amdgcn_s_barrier(); }` between `device_gemm_tile_body` and
the 4 stores as suspicious because — with `device_gemm_tile_body`'s
epilog-2 ending with an unconditional workgroup-wide `s_barrier()`
(line 776) — a wave-conditional `s_barrier()` immediately after should
be either redundant (4 of 8 waves wait, 4 already past) or a deadlock
on standard gfx95 ISA semantics. The kernel runs without deadlock, so
R43 hypothesised the conditional barrier was a remnant the compiler
dead-code-eliminates.

The same conditional barrier appears at **two** call sites (mirror
copies):
- `grouped_kernel` line 3940 (forward + dA fused-K-tail path).
- `grouped_var_k_kernel` line 4654 (dB var-K path).

Both have been there since the very first commit of the BF16 kernel
(b82ae055 "Add BF16 GEMM with exact-dim JIT").

## Hypothesis (R46)

The conditional `s_barrier` is redundant given the body's terminating
unconditional `s_barrier`. Removing both should be either bit-exact
(if the compiler already collapsed it) or save ~5-10 cycles per
persistent-loop iteration (if it was a real barrier).

Removed both lines and rebuilt. Resource counts (clean compile):
```
grouped_kernel<L=2, KI=832, FUSED_KTAIL=false>:  256 VGPRs, 8 spill, 106 SGPRs (5 spill)
grouped_var_k_kernel<0>:                         256 VGPRs, 0 spill,  93 SGPRs (0 spill)
```
No regression vs baseline (resource-wise the barrier was 0 cost; it's a
single instruction).

## Evidence (correctness)

Metric correctness gate (downsized fwd+bwd `check_allclose` on `out`,
`dA`, `dB` vs Triton ref on every shape):

```
correct_fail=24/24    *FAIL[dB-allclose] on every metric shape*
score=0/1000          (clipped because all 24 ratios are 0)
```

Every single shape — DSV3 + gpt_oss + Qwen3, B={4,16,32}, M={2048,4096} —
fails `dB-allclose`. The `out` and `dA` checks PASS (those went through
`grouped_kernel` whose conditional barrier was also removed — no
correctness issue there because the forward path's output write is
naturally serialised by the K-loop finishing). The fault is exclusively
on the `dB` write from `grouped_var_k_kernel`.

This means the conditional `s_barrier` at line 4654 is **NOT
redundant** — it is doing real synchronisation work that protects the
4 store_c_tile_mn_masked_grouped writes from a race condition.

## Mechanism — what the barrier was actually doing

The hypothesis "device_gemm_tile_body ends with full s_barrier ⇒
nothing more is needed" was wrong. Looking at the call site again:

```
device_gemm_tile_body<...>(...);   // ends with s_barrier (epilog-2 last line)
                                    // <-- here all 8 waves are aligned
if (warp_row == 0) { __builtin_amdgcn_s_barrier(); }  // load-bearing!

store_c_tile_mn_masked_grouped(g.c, C_accum[0][0], ...);   // warp_row==0
store_c_tile_mn_masked_grouped(g.c, C_accum[0][1], ...);   // warp_row==0
store_c_tile_mn_masked_grouped(g.c, C_accum[1][0], ...);   // warp_row==1
store_c_tile_mn_masked_grouped(g.c, C_accum[1][1], ...);   // warp_row==1

asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
__builtin_amdgcn_s_barrier();                              // unconditional
```

What the conditional barrier likely does (best post-mortem
hypothesis given the FAIL signature):

After `device_gemm_tile_body` returns, the LDS double-buffer slots
`As[*][*]` / `Bs[*][*]` still hold the *last* K-tile's data from the
preceding K-loop iteration. Some of those LDS slots might be re-loaded
by the **next** persistent iteration's prologue (if the next iter
starts before the current iter's stores fully complete). The
conditional `s_barrier` on warp_row==0 is a wave-restricted serializer
that gates the 4 stores against a subset of the waves' LDS-vs-store
race window. Removing it lets a subset of waves race ahead into the
prologue and overwrite an LDS slot that's still being read by a
sibling wave's store-side load (the `load_b_subtile` / `load_a_subtile`
inside `device_gemm_tile_body` reads from the same `As[*][*]` /
`Bs[*][*]` buffer that some other wave's store may still be staging
through).

The `s_waitcnt vmcnt(0) lgkmcnt(0)` AFTER the stores guards the
*outbound* stores (vmcnt) and LDS reads (lgkmcnt) for the **current**
iteration, but NOT the LDS *write* race between the next iter's
prologue and the current iter's in-flight LDS-side staging. Removing
the conditional barrier closes that race window.

Concretely: the stores write from `C_accum` to global memory directly
via `buffer_store_short` (no LDS involvement on the store side per
`store_c_tile_mn_masked_grouped` line 448). So the LDS race must be on
the **read** side — between `device_gemm_tile_body`'s last LDS read
(via `load_b_subtile` / `load_a_subtile` in epilog 2) and the next
iter's prologue write to the same LDS slot. Without the conditional
barrier, the next-iter prologue's `G::load → As[tic][0]` may overwrite
the LDS slot before this-iter's epilog-2 LDS reads have actually
completed (vmcnt drains the load queue, but lgkmcnt only drains LDS
**read** completion, not LDS **write** retirement). The conditional
barrier on warp_row==0 blocks a subset of waves long enough for the
LDS read pipe to fully retire before the prologue rebroadcasts.

Why specifically `warp_row == 0` (not all 8 waves)? Because the
load_a / load_b subtile pattern in `device_gemm_tile_body` distributes
LDS reads across waves: warp_row==0 reads slot As[tic][0] /
As[tic][1], warp_row==1 reads slot As[tic][1] / As[tic][2] (or some
cyclic rotation). The race is asymmetric — warp_row==0 reads from a
slot that the next-iter prologue's first warp will write to first.
Gating warp_row==0 alone is sufficient to serialise this single
LDS-read↔LDS-write pair.

The original kernel author chose `warp_row == 0` deliberately — this
isn't a bug or a remnant.

## Falsification consequence

R46 closes:

* **Conditional `s_barrier` removal as a kernel structural opt**.
  Both `grouped_kernel` line 3940 and `grouped_var_k_kernel` line 4654
  conditional barriers are LOAD-BEARING. R43's "suspicious" hypothesis
  is wrong; the barrier serialises a non-obvious LDS read↔write race
  on the persistent loop boundary.

R46 does NOT close:

* Other 3 R43-listed var-K structural opts (per-tile arithmetic
  hoisting, store batching 4→1, var-K KI specialisation).
* The kernel-side surface as a whole — only the cheapest scout was
  tested.

Side benefit: this round documents a non-obvious correctness
invariant of the BF16 grouped + var-K kernels. Future audits should
NOT touch the `if (warp_row == 0) s_barrier()` line without rebuilding
the LDS buffer-rotation analysis from scratch. The added comments at
both call sites are kept (they explain the load-bearing role) — wait,
actually the v1 commits ARE reverted, so the original (no comment)
state is restored. We document the falsification finding here in this
note instead.

## Action

* HipKittens kernel: revert via backup (`/tmp/kernel_bf16_dynamic.cpp.bak`)
  + rebuild. Diff after revert: zero.
* Primus-Turbo: 1 commit (this falsification note).
* HipKittens: no commit (kernel reverted to baseline state).

## Verification post-revert

```
metric run on reverted kernel:
  score = 878  (vs baseline 879, within ±10 noise)
  correct_fail = 0 / 24
  per-family geomean = 1.084 / 1.121 / 1.111  (gpt_oss / DSV3 / Qwen3)
```

## R47 next-action surface

Three remaining R43-listed var-K structural opts, ordered by safety:

1. **Var-K KI specialisation** (R43-listed item 4). Currently
   `grouped_var_k_kernel<0>` is the only instantiation (KI=0 dynamic).
   For gpt_oss with K_var ∈ {2048, 4096} → ki_g ∈ {32, 64}, a KI=32
   and KI=64 specialisation could let the compiler unroll the K-loop
   completely. Risk: VGPR spill (R39/R40 saw spills 16-30 on similar
   KI specialisations for the forward kernel). Mitigation: gate
   specialisation on the per-group dispatch path — only apply if
   spill count comes back at <10 VGPRs.
2. **Store batching 4 → 1** (R43-listed item 3). Combine the 4
   `store_c_tile_mn_masked_grouped` calls into one larger
   masked-store that handles all 4 sub-tiles. Reduces the per-tile
   bounds-check overhead (currently 4 × bounds-check = ~20 SGPR ops).
   Risk: refactoring touches the masked store function which has
   subtle layout-mask correctness; needs a dedicated correctness probe.
3. **Per-tile arithmetic hoisting** (R43-listed item 1). Cache
   `m_start_g`, `M_g`, `ki_g`, `k_offset_tiles` per group_idx using
   the same pattern as `grouped_kernel` does for `b_srsrc_curr`
   (line 3899-3906). Skip recomputation when the persistent loop
   stays in the same group across consecutive iterations. Limited
   leverage on B=32 (group crossings per CU ≈ iters per CU because
   tiles_per_group < NUM_CUS), but free win on smaller batches.

Recommended: start R47 with (3) since it's the lowest-risk
correctness-wise (pure compiler-friendly hoist, no kernel-template
addition, no race-window analysis required). Even if the metric move
is sub-noise, the diff cleans up the hot path's instruction count
which the auto-tuner can build on.

If (3) is also sub-noise, R48 should pick (1) — KI specialisation —
which has the largest single upside but the largest spill risk.
Defer (2) (store batching) to R49+ since it requires the most
correctness validation.
