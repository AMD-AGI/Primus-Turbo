# 2026-10-27 (second campaign), ctheliosr-rck-g02-e15-02, GPU1 (0002:01:00.0)

Image amdprimus/amdprimus:tas-20261019 (torch 2.14.0+rocm10.2.0a20261011, flydsl 0.3.4.1), branch
dev/lhz/hk_attn_ladder 8c3cdff6, container lhz-hkl-g1 (this GPU only). Kernel log clean for the whole session.
The ladders with the redesigned bottom rungs; `../20261027_e15-02_gpu1/` is the first campaign (first-draft
bottom rungs, now side rungs).

* `test_*.json`: stage C correctness (`hkl.test`, every rung + FlyDSL, every shape of its problem kind, production
  shape included). Every new rung passes everywhere. The only failures are the known MLA `rect32` dQ read error of
  r2d .. r7 (dq 17.7 dB at b1 Sq 4096 Skv 4128; see the ladder README), unreachable in production.
* `ladder_*.md / .json`: stage D timing (`hkl.ladder --side -r 10`, settle 1 s per cell, alternating order, flush
  512 MB per call; a step is over the rung's parent). Null controls: fwd 1.0027x +- 0.0039, bwd 0.9990x +- 0.0012,
  MLA 0.9996x +- 0.0045.
* `att_*.txt / .json`, `ktrace_*.txt`: stage E (thread traces of the hot kernel of every main-line rung, one CU,
  production shape; per-kernel times of the multi-kernel rungs).

## Main lines (ms at the production shape; step over the parent)

Llama-3.1-8B forward (b4 s8192, FlyDSL 1.339):

| f00 | f01t | f02 | f03b | f04r | f05p | f06g | f07l | f08s | f09m | f10z | v0 | p1 | v1.1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.545 | 2.403 | 2.288 | 1.930 | 1.884 | 1.877 | 1.589 | 1.481 | 1.451 | 1.416 | 1.238 | 1.252 | 1.144 | 1.130 |
| | 1.059x | 1.050x | 1.188x | 1.024x | 1.003x | 1.183x | 1.074x | 1.021x | 1.022x | 1.144x | 0.991x | | |

TDM fill, double buffer, operand bursts, a 3-slot ring, software pipelining, GQA packing, longest first,
lock_simd, mask on the diagonal block only, lazy rescale; f10z is v0's speed (v0 = the shipped form of the
same schedule), and the historical v0 -> p1 -> v1 -> v1.1 rungs continue from there.

Llama-3.1-8B backward (FlyDSL 4.362):

| b00 | b01t | b02 | b03b | b04f | b05g | b06k | b07l | b08d | b09i | b10p | b11s | b12l | v0 | p1 | v1.1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9.296 | 8.160 | 7.323 | 6.911 | 6.145 | 5.319 | 4.955 | 4.979 | 4.811 | 4.685 | 4.529 | 4.394 | 4.339 | 3.856 | 3.710 | 3.544 |
| | 1.140x | 1.114x | 1.060x | 1.125x | 1.154x | 1.074x | 0.996x | 1.036x | 1.026x | 1.034x | 1.031x | 1.014x | 1.125x | | |

TDM, double buffer, bursts, dQ fused in v0's decomposition (b04f), longest first, wave kinds with the FULL steps
first, LD rows by TDM (flat alone, the enabler of b08d), deferred dQ, dQ || softmax in one region, two pipelined
regions (co-execution scheduler), v0's split barrier, lock_simd; b12l is at FlyDSL's speed and v0 1.125x above it.

DeepSeek-V3 MLA backward (b2 s4096 h128, FlyDSL 3.576):

| m00 | m01 | m02 | m03q | m04b | m05w | m06g | m07q | hk_v0 | r1 | r2 | r7 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6.985 | 6.162 | 5.916 | 5.329 | 4.992 | 4.613 | 4.332 | 4.195 | 2.811 | 2.575 | 2.540 | 2.480 |
| | 1.133x | 1.044x | 1.110x | 1.070x | 1.080x | 1.065x | 1.031x | 1.494x | | | |

TDM, the dS export, wider dQ waves, dK / dV bursts, 32 kv rows per dK / dV wave, longest first, 64-query dQ
waves; hk_v0 remains a 1.49x step. m07q's 1.031x is mostly clock: 1.006x in cycles, and its dQ kernel alone is not
faster under the kernel trace (0.811 vs 0.780 ms; 384 VGPRs of accumulators take it to one wave per SIMD).

## Thread traces (stage E, `att_*.txt`; WMMA-pipe busy fraction of the traced CU)

* Forward: 26.7% (f00) -> 32.7% (f02) -> 42.8% (f03b, bursts: s_wait_dscnt no longer dominates) -> 47-49% (f06g ..
  f09m: one wave per SIMD, 45-47% of the time in VALU, the eager rescale of 4 heads) -> 58.0% (f10z: the lazy
  rescale drops the VALU share to 26%) = v0 (57.6%); p1 66.9%, v1 70.7%.
* Backward (dK / dV kernel): 21.7% (b00) -> 32.3% (b02) -> 33-34% (b04f .. b07l, fused, one wave per SIMD) -> 36.4%
  (b08d) -> 39.7% (b10p) -> 42.8% (b12l); v0 53.5%, p1 59.2%, v1.1 65.6%.
* MLA dK / dV: 20.3% (m00) -> 29.3% (m05w) -> 28.2% (m07q); hk_v0 41.5%, r7 56.2%. MLA dQ: m07q 28.4%, hk_v0 57.5%.
  The remaining MLA gap is in both kernels (`ktrace_bwd.txt`: m07q dK / dV 2.69 + dQ 0.81 ms, hk_v0 1.81 + 0.51).

## Negative and flat results (side rungs, same tables)

* Register-path double buffering (f01, b01): 0.86x / 0.98x -- the in-flight loads cost a wave per SIMD; the TDM
  path double-buffers for free.
* GQA packing before software pipelining (old f03..f08): 4x the work per workgroup at one wave per SIMD with nothing
  to hide the softmax under; the chain stays at 2.2-2.4 ms. On the pipelined kernel the same packing is 1.18x (f06g).
* A per-head lazy-rescale branch at every chunk end (f09z, 0.94x); one combined unlikely branch outside the
  interleaved region (f10z, 1.14x).
* Backward, before the work order and the wave kinds: split barrier with two buffers (b05s 0.96x), plus a third
  slot (b05r 0.95x), next-step prefetch in registers (b05n 0.95x; on b06k b07n 0.96x at 990 VGPRs). The split
  barrier pays only once the tr16 loads of dV / dK stay in flight past the signal (b11s, 1.03x).
* The co-execution scheduler matters: b10p in the default scheduler is 0.98x.
* MLA at m06g / m07q's geometry: LD rows by TDM (m07l), the pipelined softmax (m08p), barrier + readback (m09r), a
  3-slot ring (m10t), wave kinds (m07k): all 0.99-1.00x; 2-wave dK / dV workgroups (m08h): 0.97x.
* The first-draft backward chain (b03..b07: a GQA dQ kernel, fusion with 16 queries x 128 d per wave and K read
  from LDS, a deferred dQ on it) goes from 7.3 to 14.9 ms.
