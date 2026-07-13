# gfx950 / gemm_mxfp8 / FLYDSL — reusable tips

## Staging the raw E8M0 A-scale in LDS CAN get within ~13% of the preshuffled path
Plan A (stage the tile's raw E8M0 A-scale into LDS in the GEMM prologue, read from LDS in the
K-loop; no separate raw->broadcast preshuffle kernel / HBM round-trip) is viable and fast IF
two things are done — both were the real cost, and both are tunable (an earlier "structurally
2x slower" conclusion was WRONG):
1. **Pad the LDS row stride.** A `[BLOCK_M, K128]` layout has a power-of-two row stride, so the
   per-sub-tile stride-K128 read of 16 lanes collides on 2 banks (8-way conflict). Padding the
   stride to `K128+1` (odd, coprime with 32) spreads them across 16 banks. Effect: ~-40%.
2. **Vectorize the staging HBM load** (`vec_width=4`, `VEC=4 if K128%8==0 else 2 if K128%4==0
   else 1`, keep `K128/VEC` even so it tiles 512 threads; LDS writes stay scalar because the
   padded stride breaks b128 alignment). Effect: another ~-14..20%.
Result: from 2.15x slower (raw-in-LDS, no tuning) to ~1.13x of ps at the K=2048 combine shape.
- The read only needs byte-0 (MMA `opsel=0`); the 4-byte broadcast is dead work (but hidden
  under MFMA, so removing it is ~0 perf — do it for clarity only).
- Residual ~13%: ~5.5% is the LDS read vs ps's HBM-coalesced (L2-resident) read; the rest is
  the one-time stage + drain barrier (not yet overlapped — folding it into the fp8 prologue is
  blocked by that prologue's exact vmcnt `wait_barrier(count)`).

## When Plan A (LDS A-scale) actually pays off — and when it does NOT
- **Broadcast layout in LDS is impossible**: it is 4x the raw (A ~229 KB @ K=7168) and does not
  fit next to the 128 KB fp8 ping-pong. Always stage RAW + broadcast-on-read.
- **LDS budget**: MI355X = 160 KB/block, fp8 ping-pong = 128 KB -> ~32 KB free. A-only raw
  stage fits K<=4096 WITHOUT padding (256*(K/128)*4 <= 32 KB); WITH +1 padding it fits K<=~3968
  (K=4096 padded = 33.8 KB just overflows). K=7168 never fits.
- **No net win in the standard paths** even at lds==ps: standalone ps's a_sp is produced FUSED
  in the act-quant kernel; fused STEP1 (K=7168) doesn't fit; fused combine (K=2048) already
  fuses A-preshuffle into act-quant. Net win requires a fused kernel with a SEPARATE preshuffle
  role to delete AND K<=~3968 (removing the role + its ~0.7 ms L2 fences beats the ~13% GEMM
  slowdown). Not the case in STEP1/STEP3/combine today.
- Benchmark over-fit risk: none (kernel-side K1; A-scale fresh each step; no id()-keyed cache).

## Fused STEP1: preshuffle-role headroom is real (0.53 ms) but LDS scale cannot cash it
Diagnostic on the fused bwd STEP1 (dispatch(dy)+fc2 dgrad, 8xMI355X EP8, ndcu=pscu=16):
- Removing the preshuffle role (bwd kernel `two_stage=3` SKIP_PS_DIAG: gemm reads pre-baked
  data, garbage-but-timed) drops the wall 2.170 -> 1.644 ms (= comm floor 1.649): the gemm
  fully hides under comm once the preshuffle stage is gone. So the comm∥gemm overlap WAS
  limited by the preshuffle role, ~0.53 ms of headroom.
- But streaming the raw A-scale into LDS inside the gemm (`two_stage=4` LDS_STREAM, KT-window
  double-buffered, CORRECT cos PASS) measured **2.916 ms — +0.75 ms WORSE** than the 2.170
  baseline: the per-window `s_barrier` (cross-wave coherence) + scale prefetch loads polluting
  the fp8 `s_waitcnt vmcnt` perturb the tight fp8 software pipeline, blowing the gemm from
  ~0.9 ms to ~2.5 ms (> comm floor -> exposed). Double-buffering did NOT help (bottleneck is
  the barrier, not HBM latency). KT=14 (fewer refills) hit an LDS-too-tight / scoreboard hang.
- Verdict: cashing the headroom needs the gemm to read PRESHUFFLED scale (coalesced, zero-ALU,
  hides under comm). Every way to move that preshuffle out of the dedicated overlapped 3-stage
  role loses (FUSED_PS 3.66 ms; broadcast direct-push lost; streaming LDS 2.916 ms). Keep the
  3-stage preshuffle role. The real STEP1 lever is L2-fence granularity / comm∥gemm HBM
  contention (mxfp8-backward note), NOT the scale load. Do not retry LDS scale in fused STEP1.

## MI355X (gfx950) LDS budget
`torch.cuda.get_device_properties(0).shared_memory_per_block = 163840` (160 KB). The mxfp8
tile's 8-buffer fp8 ping-pong (BM=BN=256, BK=128) = 128 KB, leaving ~32 KB for extra LDS.
