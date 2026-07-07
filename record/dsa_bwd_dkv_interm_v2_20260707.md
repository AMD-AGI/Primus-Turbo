# FlyDSL dKV-intermediate V2 (multi-wave) — 6x over old port, but below triton/gluon

> Implements record/DESIGN_dkv_interm_flydsl_multiwave.md: a native multi-wave FlyDSL
> dkv-intermediate to replace the Triton `_bwd_compute_dkv_intermediate` and remove the
> triton-wheel version dependency. Correct, 6x faster than the dead single-wave port,
> but at production TOPK it is 0.70-0.84x of triton and ~0.85x of gluon — a structural
> ceiling of the D_V-split layout. Kept env-gated OFF; triton interm stays default.

## What it computes
interm[t,key,d] = sum_h ( Q[t,h,d]*dS[t,h,key] + dO[t,h,d]*P[t,h,key] ), contract head H.
Output interm[T,TOPK,D_V]; dS/P reused from the dQ kernel's HBM buffers. Feeds the
(unchanged) CSR dkv gather-reduce.

## Design implemented (dsa_bwd_dkv_interm_v2_kernel.py)
- Grid (T,); each workgroup = NW waves on the same token; the NW waves SPLIT the D_V
  output (wave w owns d in [w*D_PER_WAVE, ...)). Contraction over H is inside each wave.
- MFMA mfma_f32_16x16x32_bf16, K=32 head-contraction. A = Q_T[d,head] via ds_read_tr16
  (row-major [head][d] LDS tile, +16 pad breaks bank conflicts); B = dS/P[key,head] from
  a SHARED [key][head] LDS tile (head-contiguous -> one vec8 B-read); C = interm[d,key].
- Q/dO staged into LDS by async global->LDS DMA (raw_ptr_buffer_load_lds) — bypasses VGPR.
  dS/P staged shared (once per workgroup) so all NW waves read LDS, not redundant HBM.
- i64 offsets throughout; launcher chunks T so each launch's largest tensor < 2^31 bytes.
- Swept defaults: NW=8 (512-thread CTA), BLOCK_H=32, TILE_K=64 (falls to 32 if TOPK%64).
  vec4 interm store. A-operand prefetch depth 2.

## Results (MI355X, real production TOPK = window512++pool = 1024 / 2560), us/call
| shape       | triton (head-gated) | gluon | flydsl-v2 | fly/tri | fly/gluon |
|-------------|--------------------:|------:|----------:|--------:|----------:|
| H128 K1024  | 2797                | 3206  | 4004      | 0.70x   | 0.80x     |
| H64  K1024  | 2110                | 1908  | 2738      | 0.77x   | 0.70x     |
| H128 K2560  | 7922                | 7773  | 9443      | 0.84x   | 0.82x     |
Correct: interm SNR 55.6 dB vs fp32 torch (== triton). 144/144 CSA tests pass with
V2 forced ON; also pass with default (V2 off).

Arc: old single-wave port (ca2a8f3) was 22.6 ms in-bwd (~6x slower than triton). The
multi-wave rewrite is ~4.0 ms equivalent for H128K1024 — ~6x faster than that port —
but still below triton/gluon.

## Why it can't beat triton (structural ceiling — the real finding)
- Both flydsl-v2 and triton/gluon re-read Q/dO from HBM once per key-tile (grid=(T,),
  outer tile / inner head-group). The ONLY lever that cuts that traffic is a larger
  TILE_K: triton uses TILE_K=128 (BH16, nw4), which halves the Q/dO re-read vs my TK64.
- My D_V-split layout CANNOT reach TILE_K=128 without spilling: each wave holds
  acc[DT_PER_WAVE][N_SUB] fp32 + the B-prefetch; at TK128 (N_SUB=8) this blows past 256
  VGPR (measured s162 at NW16/TK128, and NW32 needs >1024 threads). So flydsl-v2 is stuck
  at TK64 and pays ~2x the Q/dO re-read of triton's TK128 — that is the whole gap.
- PMC: MemUnitStalled ~56-61% (latency-bound, not bandwidth: interm write floor is only
  413 us for 2.15 GB). Low occupancy (NW8 = v214, ~2 CTAs/CU) can't hide the Q/dO fetch.

## Dead ends measured this session (do NOT re-try on this layout)
1. Shared dS/P as [head][key] with 8 scalar-LDS B-reads — slower than [key][head] vec8.
2. Full A-operand hoist (all 16 packs) — v512/spill380 catastrophe; depth-2 prefetch fixes it.
3. NW4/NW2 (gluon's 256-thread CTA) — D_PER_WAVE=128/256 makes acc too big -> spills.
4. 2D grid (T, NUM_TILES) — regressed (2311 vs 1958 @ TK64); fewer per-CTA reuse.
5. TILE_K=128 at any NW that fits threads — spills (the ceiling above).
6. Head-group double-buffer pipeline (NBUF=2, prefetch hg+1 DMA) — nets ~0/slightly worse:
   s_waitcnt(0) is a FULL drain (no per-op vmcnt threshold; the dS/P regular vector loads
   share the counter), so the hg+1 prefetch can't actually overlap hg's GEMM. Same
   limitation as the fwd softmax‖QK pipeline. Gated OFF (PRIMUS_DSA_INTERM_PIPE).
7. waves_per_eu 3/4 — worse than 2.

## Verdict / how it's wired
- dsa_bwd.py: `_USE_FLYDSL_INTERM` (env PRIMUS_DSA_FLYDSL_BWD_INTERM_V2, default 0/OFF).
  When ON and (heads%16==0 && TOPK%32==0), uses the flydsl interm with a T-chunk launch;
  else Triton. Triton stays default because it is faster at production TOPK.
- The kernel is a correct, triton-version-INDEPENDENT fallback (its motivation), but it
  does not beat triton/gluon, so enabling it regresses the combined bwd (~+1.4 ms on
  H128K512). Combined-bwd-beats-gluon must come from other levers (dQ / gather / the
  fact that gluon's H128K512 lead is NOT in interm), not from this interm.

## Env knobs
PRIMUS_DSA_FLYDSL_BWD_INTERM_V2 (0), PRIMUS_DSA_INTERM_DMA (1), PRIMUS_DSA_INTERM_PIPE (0),
PRIMUS_DSA_INTERM_NW (8), PRIMUS_DSA_INTERM_BH (32), PRIMUS_DSA_INTERM_TK (64/32),
PRIMUS_DSA_INTERM_APF (2).

## Files
- New: primus_turbo/flydsl/attention/kernels/sparse_mla_v2/dsa_bwd_dkv_interm_v2_kernel.py
- Wired (gated OFF): dsa_bwd.py (_get_interm_v2_kernel + T-chunk launch + (H,TOPK) gate)
- Harness: output/harness_interm.py (flydsl vs triton), output/bl_interm.py (triton vs gluon)

## Next lever if pushed further (open)
The only way past the ceiling is a layout that reaches TILE_K>=128 without spilling:
split KEYS across waves (each wave full D_V for a key-slice) so a wave holds
[D_V=512, small-key] — but that makes 32 d-tiles of acc, worse. Or a fundamentally
different tiling (persistent-CTA over tokens, register-blocked like triton's MMA layout).
High effort, uncertain payoff since triton interm already works and this only removes a
version dependency the combined-bwd win doesn't actually need.
