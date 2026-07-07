# Path B (M=16 16x16x32 QK + ds_read_tr PV) — implementation & result (2026-07-07)

> Goal: beat gluon_v2 fwd (411-441 TFLOP/s) by combining occupancy-2 (M=16) with
> the fast hardware-transpose PV pipeline (ds_read_tr) that all prior M=16
> attempts dropped.
>
> **Result: tr16 kernel is CORRECT (47.5 dB). Naked M=16 (no pipeline, no
> async_copy, redundant per-wave gather) is 131 TFLOP/s < M=32's 278.**
>
> **CORRECTION (2026-07-07): the "dead end" verdict below was PREMATURE.** This
> config differs from gluon in THREE ways, not one (see "What's still missing").
> gluon's 411 needs all three (shared gather + async_copy DMA + num_stages=3
> pipeline) TOGETHER. Judging the M=16 ROUTE on a config missing all three is
> unfair. Next step: fully align tr16 to gluon, THEN re-measure. See the
> "FULLY ALIGN" plan at the bottom.

## What was built
New kernel `primus_turbo/flydsl/attention/kernels/sparse_mla_v2/dsa_fwd_tr16_kernel.py`
(`build_dsa_fwd_tr16_module`), wired via env `PRIMUS_DSA_FLYDSL_FWD_TR16=1` in
`dsa_fwd.py`. Derived from `dsa_fwd_m16_kernel.py` but with the ONE fix that the
diff doc + memory said was the reason M=16 regressed:
- **Kept**: M=16 16x16x32 QK C-layout, softmax peer-reduce (shuffle_xor 16,32),
  p C->B-operand pack (all proven correct in dsa_fwd_m16_kernel).
- **Changed**: the AV A-operand (kv^T). m16 staged it via 128 SCALAR LDS stores
  (d-major). tr16 stores the gathered tile ROW-MAJOR V[key][d] with a single
  vec8 store, then reads it back transposed via `ds_read_tr16_b64` — the exact
  fast PV pipeline the M=32 `dsa_fwd_pipe_kernel` uses.

### ds_read_tr A-operand lane map (verified correct, 47.5 dB)
Row-major V[key][d], stride V_STRIDE. HW 4x4 transpose gives
`result[L][e] = V[src][L%4]`, `src=(L//16)*16 + e*4 + (L%16)//4`. To land the
16x16x32 A-operand `A[d=dt*16+L%16, key=(L//16)*8+e]`:
- `k_row = (L//16)*8 + (L%16)//4`, `d_col = dt*16 + (L%4)*4`
- `va` = keys g*8+0..3; second read at `+4*V_STRIDE` = keys g*8+4..7; shuffle vec8.
This produces the identical A-operand the m16 kernel proves correct — confirmed
out SNR 47.5 dB / lse 145-148 dB vs the M=32 reference (same SNR gluon gets).

## Correctness: PASS
out SNR 47.5 dB, lse 145-148 dB vs M=32 ref, across H128/H64, K512/K2048. The
lane-map derivation (the diff doc's flagged silent-error risk) is correct.

## Performance: FAILS to beat M=32 (let alone gluon)
MI355X, real adapter structured topk, sink on. Reference M=32 = production flydsl.
| shape | M32 (ref) | tr16 BLOCK_K32 | tr16 BLOCK_K64 | tr16 QLDS occ2 | gluon |
|---|---|---|---|---|---|
| H128 K512  | 278 | 131 | 82 | 118 | 412 |
| H64  K512  | 237 | 130 | 82 | 118 | 411 |
| H128 K2048 | 251/280 | 140 | 83 | 122 | 440 |

## Root cause (empirically confirmed, not speculation)
1. **Default tr16 (BLOCK_K=32): VGPR=270 -> occupancy=1.** Persistent footprint =
   acc [16,512] fp32 = 128 VGPR (irreducible) + q_packs 64 VGPR resident + ~78
   softmax/AV temps. At occ=1, M=16 processes HALF the heads/wave of M=32, so it
   needs 2x the waves -> ~2x slower. 131 ≈ 278/2. This is the whole gap.
2. **BLOCK_K=64 (fewer tiles) made it WORSE (82).** => per-tile softmax/fence
   overhead is NOT the bottleneck. Rules out the "small tile => too many tiles"
   hypothesis.
3. **Q staged in LDS + WAVES=2 => VGPR=226 -> occupancy=2, but perf DROPPED (118).**
   This is the decisive result. Reaching occ=2 did not help; the added per-tile
   LDS traffic (Q reload) + fewer waves (less latency hiding) outweighed it.
   AGPR pool stays idle (acc is VALU-rescaled by alpha every tile, so it MUST
   live in arch VGPR — cannot be pushed to AGPR; that's why the amdgpu-waves-per-eu
   passthrough backfired, pushing total VGPR to 310).

### The lesson (matches prior-session memory, now with the PV pipeline controlled)
"Occupancy is necessary but NOT sufficient; AV pipeline efficiency dominates."
Prior M=16 attempts were dismissed as "they dropped ds_read_tr." This session
KEPT ds_read_tr (fast PV, verified) and M=16 STILL loses to M=32. So the real
cause is structural: **M=16 = 16 heads/wave means the [16,512] fp32 acc + Q +
softmax state per wave is a fixed cost amortized over only 16 heads, vs 32 for
M=32.** M=32's [32,512] acc = 256 VGPR forces occ=1 too, but it does 2x the useful
head-work per wave, so its VGPR/occupancy "waste" is actually the better trade on
THIS problem (D_V=512 contraction dominates; occupancy can't hide it at 1 wave and
2 waves don't have enough independent MFMA work to fill the gap).

## Why gluon still wins (unchanged conclusion)
gluon's 411-441 is NOT from M=16 occupancy alone — it's M=16 + TILE_K=32 +
**num_stages=3 true software pipeline** (QK[t+1] MFMA ‖ softmax[t] VALU ‖
gather[t+2] DMA) + async_copy. The pipeline is what fills the occ=1/2 MFMA gaps.
Naked M=16 without that pipeline (this session) does not benefit from occupancy.

## UPDATE (2026-07-07): SHARED GATHER flips it — tr16 now BEATS M=32
The premature "dead end" was 100% the redundant per-wave gather. Fix: gather the
kv tile ONCE per workgroup into a SHARED LDS tile (topk depends only on token, so
all waves read identical latent rows), and read the QK A-operand FROM that LDS tile
(not redundant per-wave HBM loads). The same row-major shared tile feeds both the
QK A-operand (vec8 load, key=L%16 d=(L//16)*8+e) and the PV ds_read_tr read.
| shape | M32 | tr16 naked (old) | tr16 SHARED gather | gluon | tr16/M32 | tr16/gluon |
|---|---|---|---|---|---|---|
| H128 K512  | 279 | 131 | **373** | 412 | 1.34x | 0.90x |
| H64  K512  | 237 | 130 | **329** | 411 | 1.39x | 0.80x |
| H128 K2048 | 281 | 140 | **386** | 440 | 1.37x | 0.88x |
Correct (47.5 dB / 145-148 lse). Sweep: NUM_WAVES=auto (H128->8, H64->4) + BLOCK_K=32
is optimal; BLOCK_K=64 halves it (191). Diagnostics: VGPR=256, spill=8, occ=1 — the
win is eliminated redundant HBM traffic, NOT occupancy. **M=16 route VINDICATED.**
Remaining gap to gluon (~10-20%) = gluon's async_copy DMA + 2-buffer softmax‖QK pipeline.

## STATUS after shared gather (2026-07-07): tr16 = 0.90x gluon, all 138 tests pass
- Default tr16 (coop shared gather, DMA/pipeline OFF) = **373/329/385 TFLOP/s**,
  1.34-1.39x over M=32, 0.80-0.90x of gluon. out 47.5 dB. **138/138 CSA tests pass**
  with PRIMUS_DSA_FLYDSL_FWD_TR16=1.
- Step 2 (DMA gather, PRIMUS_DSA_TR16_DMA=1): implemented + CORRECT (47.5 dB) but
  SLOW standalone (118) — 2048 tiny 16B descriptors/tile with per-descriptor topk
  lookup. DMA only pays off when OVERLAPPED against compute (needs the pipeline), so
  it's gated OFF pending Step 3. This matches how gluon uses async_copy.
- Step 3 (2-buffer softmax(t)||QK(t+1) + DMA overlap pipeline, PRIMUS_DSA_TR16_PIPE):
  scaffolding in place (NBUF=2 LDS double-buffer, fits 72KB). The loop restructure
  (loop-carried QK score regs + buffer parity through scf.for, prologue/epilogue peel,
  s_waitcnt groups) is the remaining work to close the last 10-20% to gluon.
- Checkpoint of the 373 version: output/flydsl_v2_ckpt/dsa_fwd_tr16_kernel_shared373.py

## What's still missing (why the OLD naked number was NOT a fair M=16 test)
tr16 as first built differs from gluon in THREE ways, all of which gluon needs:
1. **gather is REDUNDANT per-wave.** topk depends only on (token), so every wave in
   a CTA gathers the SAME kv rows. gluon shares one kv tile across its 4 warps.
2. **gather is VGPR-staged** (load->VGPR->LDS store), not async_copy DMA
   (raw_ptr_buffer_load_lds, HBM->LDS bypassing VGPR).
3. **NO software pipeline.** gather->fence->QK->softmax->PV is fully serial. gluon
   runs num_stages=3 (QK[t+1] MFMA || softmax[t] VALU || gather[t+2] DMA).
These are LDS-coupled: 3-stage needs 3 buffers (99KB), which only fits 4 waves if
the gather is SHARED (99KB total) not per-wave (99KB/wave -> 1 wave, no hiding).

## Verdict / recommendation (REVISED)
- The naked-M=16 131 TFLOP/s number does NOT prove the M=16 route is dead. It only
  shows M=16-minus-three-things loses to M=32-with-DMA.
- **Plan: FULLY ALIGN tr16 to gluon before judging.** (1) cross-wave shared gather,
  (2) async_copy DMA gather, (3) num_stages=3 software pipeline. 47.5dB regression
  at each step via output/harness_tr16.py. This is a FlyDSL gluon-equivalent
  (diff doc "plan C"), high silent-error risk in the pipeline — harness is the guard.
- tr16 kernel is left in-tree, env-gated OFF (PRIMUS_DSA_FLYDSL_FWD_TR16, default
  0). Production path (M=32 dsa_fwd_pipe_kernel) is untouched and remains default.

## Files
- New: `primus_turbo/flydsl/attention/kernels/sparse_mla_v2/dsa_fwd_tr16_kernel.py`
- Wired (env-gated OFF): `dsa_fwd.py` (`_get_tr16_kernel`, `PRIMUS_DSA_FLYDSL_FWD_TR16`)
- Harness: `output/harness_tr16.py` (SNR vs M32 + TFLOP/s, real adapter topk)
- Knobs (env): PRIMUS_DSA_TR16_BLOCK_K (32/64), PRIMUS_DSA_TR16_QK_PF,
  PRIMUS_DSA_TR16_QLDS (Q in LDS), PRIMUS_DSA_M16_WAVES.
