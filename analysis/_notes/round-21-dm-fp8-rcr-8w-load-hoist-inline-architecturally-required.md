# Round 21 (death-march) — `rcr_8w_load_hoist` inline-expansion is architecturally required; the 12-call footprint cannot be reduced by extracting the helper

## Round 20 plan revisited

Round-20 doc proposed Option D (extract FUSED_KTAIL block to
`__device__ __noinline__` helper) as the round-21 starting move. But on
re-examination, FUSED_KTAIL is `if constexpr`-gated so `<0,0,0>` (DSV3
spec, 67 spill, 8 metric shapes) doesn't compile that block at all —
extracting it can't help DSV3.

This round explored a related lever: add `__noinline__` to
**`rcr_8w_load_hoist` itself**. The 12 inline expansions per K-iter
(4 in main loop body + 7 prologue + 1 epilog 1) inflate the function
body and were hypothesized to drive the 67-VGPR-spill on `<0,0,0>`.
If `__noinline__` keeps the function body small while the load helper
runs as a function call, spill might drop.

## Result — build failed: helper architecturally requires inlining

1-line change (`__forceinline__` → `__attribute__((noinline))`):

```c++
template<int N_THREADS, ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>>
__device__ __attribute__((noinline)) void rcr_8w_load_hoist(
    ST& dst, const GL& src, const COORD& idx,
    const uint32_t* __restrict__ swizzled_offsets)
{ ... }
```

Build error:

```
kernel_fp8_layouts.cpp:536:32: error: invalid operand for instruction
  536 |             "s_mov_b32 m0, %0\n\t"
<inline asm>:2:27: note: instantiated into assembly here
    2 |         buffer_load_dwordx4 v13, v[16:19], s1 offen lds
                                                  ^
```

The inline asm `"s"(srsrc)` constraint expects the SRD as a 4-SGPR
group (`s[N:N+3]`). When `__noinline__`, the compiler passes `srsrc`
(an `i32x4`) by **single-SGPR reference** (`s1`) instead of expanding
it into 4 contiguous SGPRs that the asm's `buffer_load_dwordx4` can
consume. This is not a fixable constraint — the inline asm fundamentally
requires the SRD to live in a 4-SGPR pack at the call site, which only
happens when the helper is inlined into the caller and the caller's
register state propagates through the inline expansion.

Conclusion: **`rcr_8w_load_hoist` MUST be `__forceinline__`.** This is
an architectural property of the inline-asm `s` constraint pattern, not
a tunable knob.

## Implication for the spill-reduction problem

The 12 inline expansions of `rcr_8w_load_hoist` per output tile in
`grouped_rcr_kernel` are **structurally baked in**. Each expansion
contributes ~30-40 instructions to the function body (the `lds_addrs[]`
ramp, the buffer_load_dwordx4 inline asm, the leftover-warp tail).
12 × 35 = ~420 instructions of inlined load code per K-iter, plus
similar in prologue and epilog.

This explains a key part of the FP8 grouped 67-spill vs BF16 grouped
0-spill divergence:

| | FP8 grouped | BF16 grouped |
|---|---|---|
| Load helper | `rcr_8w_load_hoist` (`__forceinline__`, 12 calls/iter, inline-asm `s` constraint) | `G::load` via `device_gemm_tile_body` (`__forceinline__`, 4-8 calls/iter) |
| Inline expansion size | ~420 instructions/K-iter | ~150 instructions/K-iter (smaller helper, fewer calls) |
| Per-call inline asm complexity | `s_mov_b32 m0, ...; buffer_load_dwordx4 ... offen lds` (4-SGPR srsrc) | Standard load intrinsic, no asm-imposed SGPR layout |

So the FP8 grouped's spill is partly from the **larger inline-expansion
footprint** of its load helper, which the compiler cannot shrink.

## What CAN reduce the 67-spill (round 22+)

Three structural levers that DON'T require making the helper noinline:

**(A) Reduce the NUMBER of `rcr_8w_load_hoist` calls per K-iter.**
Currently 4 calls per main-loop iter. If the helper could batch 2
B-tiles per call (passing both LDS slot bases as parameters), that
halves the call count → halves the inline-expansion footprint →
hopefully drops spill 67 → ~50. Risk: requires modifying the helper
to take 2 swizzled-offset arrays, 2 LDS bases, 2 idx params; the
inline asm would need to issue 2× 4 = 8 buffer_load_dwordx4. Multi-round
refactor. The MFMA dependency chain still requires sequential issue.

**(B) Replace `rcr_8w_load_hoist` (LDS-staged) with `raw_buffer_load_b128`
direct-to-VGPR loads (FUSED_KTAIL pattern).** The FUSED_KTAIL=true spec
already has 19-LESS spill (48 vs 67) when this pattern is added. Porting
it to the main loop (replacing LDS-staged main-loop loads with direct
HBM→VGPR loads) might transfer the spill reduction. But the main loop
critically depends on LDS prefetch pipelining (cross-warp data sharing
through Bs/As LDS slots). Direct VGPR loads bypass LDS, losing
inter-warp sharing. This is a fundamental architectural change. Risk:
likely needs a different N-warp split to preserve inter-warp work
sharing.

**(C) Migrate to BF16's `device_gemm_tile_body` shared-lambda pattern.**
Port the entire grouped main loop to use a `__forceinline` body lambda
that takes precomputed SGPR-resident LDS slot offsets as parameters
(mirror BF16 grouped_kernel:3763). The inline body sees a smaller
inline-expansion footprint because the shared-state captures are
explicit parameters (compiler can register-allocate around them). This
is exactly what BF16 does to achieve 0 spill. Multi-round port. Risk:
must REPLACE not ADD per round-19 cliff.

## What stays untouched

* Round-17 / round-18 / round-20 rocprof + diagnostic findings still
  valid.
* Architecture: persistent single-launch, no host sync, no per-group
  loops — INVARIANT.
* All previous falsifications still hold.

## Falsified levers (cumulative, for next round's prompt)

1. BF16 K_STEP=64→32 port (round 26) — out of scope.
2. FP8 chunk_size sweep (round 22) — dead-end.
3. FP8 MFMA cell-shape migration (rounds 14-15) — falsified by rocprof.
4. FP8 main-loop barrier/setprio/vmcnt micro-knobs (rounds 13-16) —
   saturated.
5. FP8 grouped SRD hoist for SALU reduction (round 18) — compiler CSEs.
6. FP8 grouped two-tile main-loop port via runtime branch (round 19) —
   spill cliff -37%.
7. FP8 grouped `readfirstlane` on `lds_tile_base` (round 20) — compiler
   already hoists.
8. **FP8 `rcr_8w_load_hoist` `__noinline__` (THIS ROUND, round 21)** —
   inline asm "s"(srsrc) constraint requires 4-SGPR pack at call site
   which only materializes via `__forceinline` expansion. The helper is
   ARCHITECTURALLY required to inline; the 12-call footprint is
   structurally baked into grouped_rcr_kernel.

## Status

**Reverted** (build fixed). Doc-only commit in Primus-Turbo. No
HipKittens commit this round.

Round-21 metric SHA: this commit (Primus-Turbo).
HK SHA: unchanged.

Score: 823 ± 4 (mean of 3 runs after revert: 822, 827, 820), within
baseline noise.

## Round 22 starting move

Recommend **option A (batch 2 B-tiles per `rcr_8w_load_hoist` call)**
as the most direct path to reducing the 12-call footprint without
abandoning LDS staging. Specific implementation sketch:

1. Add new helper `rcr_8w_load_hoist_2b` taking
   `(ST& dst0, ST& dst1, GL src, COORD idx0, COORD idx1, swizzled_offsets)`
   that issues 8 `buffer_load_dwordx4` (vs current 4) with 2 separate
   `s_mov_b32 m0` ramps. The SRD is shared, but each tile gets its own
   `tile_byte_offset` and per-pass LDS addresses.
2. In the main loop body (lines 2174-2202), replace the 4 individual
   `rcr_8w_load_hoist` calls with 2 `rcr_8w_load_hoist_2b` calls (one
   for the 2 A prefetches, one for the 2 B prefetches).
3. Build → check spill drops on `<0,0,0>` (target: 67 → ~50).
4. If spill drops AND correctness passes (SNR ≥ 25 dB on DSV3-Down-B16-
   M2048), run metric. Expected: +5-10 score from reduced FLAT instruction
   count (round-17 rocprof said 2.99M FLAT vs Triton 1.03M; halving
   the per-K-iter inline-expansion would drop ~33% FLAT).

Risks:
* If the new helper needs to be inlined for the same SRD-constraint
  reason, the 2-tile expansion may have the SAME footprint as 2 separate
  calls. (Compiler might CSE the SRD setup but not the body.)
* If spill stays at 67 after halving call count, that confirms the
  inline-asm `s` constraint pattern is the dominant cost (not the
  per-call setup overhead) — and option B/C become the only paths
  forward.

If option A doesn't move spill, document the falsification and pivot
to option C (BF16 shared-lambda port) as a multi-round commitment.
