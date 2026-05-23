# Related Work — grouped_gemm_fp8_rrr on MI355X (HK)

## Survey Objective
- Target operator: grouped_gemm_fp8_rrr (FP8 e4m3 tensorwise, A row-major / B row-major / C row-major)
- Target backend: HipKittens (`hk_grouped_rrr_fp8`)
- Target GPU: gfx950 / MI355X
- Date: 2026-05-21
- Campaign: grouped_gemm_fp8_rrr_hk_gfx950_20260521

## Search Scope
- Project-local implementations reviewed: [[project_ck_vs_hk_fp8_grouped_diff_2026_05_21]] (already done — CK_tile vs HK kernel-level diff)
- AMD / ROCm docs reviewed: gfx950 ISA summary, CDNA4 whitepaper (via `/wekafs/kyle/code_tmp/Primus-Turbo/agent/skills/hardware/gfx950/`)
- External repos / papers reviewed: composable_kernel (3rdparty), TransformerEngine grouped_gemm
- Competitor implementations: CK `grouped_gemm_quant`, Triton grouped MoE GEMM

## Relevant Implementations

| Source | Repo / Path | Backend / Lang | Hardware | Reported | Why it matters |
|---|---|---|---|---|---|
| HK dense fp8 RRR | `HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` (`gemm_rrr`) | HIP | MI355X | 1.55× hipBLASLt | Reference upper bound (no group dim) |
| HK grouped fp8 RRR | same file, `hk_grouped_rrr_fp8` | HIP | MI355X | gap 6.75-14.50% vs dense | Current SUT |
| CK_tile fp8 grouped | `Primus-Turbo/3rdparty/composable_kernel/.../grouped_gemm_quant` | HIP | MI355X | <5% gap reported (CK doc) | Structural reference |
| Triton grouped fp8 | `Primus-Turbo/primus_turbo/triton/grouped_gemm/` | Triton | MI355X | (TBD) | Algorithm baseline |

## AMD / ROCm Guidance (CDNA4)
- 256 active CUs (8 XCD × 32) → grid mapping must remap to fill all 8 XCDs (`BLOCK_SWIZZLE_NUM_XCDS=8`)
- LDS 160 KiB/CU @ 64 banks; HK current BN=256 uses As[2][2]+Bs[2][2] ≈ 128 KiB → blocks occupancy to 1 wave/CU (see [[feedback_fp8_rrr_attempt_h8]])
- MFMA F8 dense path 4096 ops/clk/CU on MI300X → 8192 on MI355X (2×) — HK 16×16×128 still leaves headroom
- HBM 8 TB/s peak; H14 measured ~1.1 TB/s wall achievable per kernel @ B=16 worst shape
- New ISA paths NOT yet exploited: `MFMA Transpose Load from LDS`, `V_MFMA_SCALE_*` (no MXFP needed here, tensorwise)

## Transferable Ideas

### CK structural deltas vs HK (line-level)
Source: [[project_ck_vs_hk_fp8_grouped_diff_2026_05_21]]

| Idea | Bucket | Why may transfer | Risk |
|---|---|---|---|
| BLK 128×128 + per-warp 64×64 (vs HK BLK 256×256 + per-warp 64×128) | K1 | Halves accumulator count per warp → spill 61→? | Halves per-CU throughput; needs more CUs to fill 256 |
| Single ping-pong (vs HK 2×2 As[2][2]) | K1 | Halves LDS footprint → enables occupancy 2 wave/CU | H8 已证伪：仅减 LDS 不够，HW 还要看 VGPR |
| sched_group_barrier batched (CK style) vs HK per-mfma sched_barrier(0) | K1 | Lets compiler reorder mfma/vmem across barriers | H9 已证伪 (单独加不行)；H17 框架下需 + LDS/VGPR 重构 |
| LocalPrefetch hoist B-tile to LDS before main loop | K1 | Reduces vmem-mfma dependence chain in inner loop | Currently HK does interleaved load_a/load_b |
| Chunked persistent + cross-group B reuse | K1 | B 是 [B,K,N]，同一 group_idx 的 B 在 L2 hot | 重写量 800-1500 LOC；改 dispatcher |
| Profile-driven: rocprof-compute SoL before more H attempts | (meta) | Confirms or refutes bandwidth-bound thesis | None — pure observation |

### Bucket-clean directions only
None of the above are W2/W3.  All structural kernel changes (K1).

## Non-Transferable / Misleading
- ❌ **W2 cache** activation/grad_out FP8 quant cache. Already documented `_h16_target6_baseline.py` 内部不要缓存; PT 不能引入 `id(activation)` cache.
- ❌ **CK fallback** for worst-shape dispatcher. User 2026-05-21 「严禁」.
- ❌ **Compile-time constant `M_per_group`** in any kernel path. MoE skew kills it (Rule 11 GroupGemm clause).
- ❌ **Uniform-only benchmark gains.** Any round whose win disappears under skewed token distribution = REJECT-as-overfit. Current target shapes are uniform — must add skewed shape before declaring victory.

## Real-training Transfer Audit

| Idea | Source | Bucket | Reported gain | Estimated real-training gain | Verdict |
|---|---|---|---|---|---|
| sched_group_barrier batch + LocalPrefetch (S1) | CK kernel | K1 | ? | ? (depends on round) | Promote pending round |
| BLK 128×128 體 + occupancy 2 (S2) | CK style | K1 | ? | ? | Promote pending round (needs LDS halve) |
| Single acc resident main loop (S3) | CK style | K1 | ? | ? | Promote pending round |
| Chunked persistent + group-local CU clustering | original | K1 | speculative ~+10% | same | Last-resort, 800-1500 LOC |
| Activation FP8 quant cache | (forbidden pattern) | W2 | n/a | 0 | Drop, do not promote |
| `M_per_group: constexpr` kernel specialization | (forbidden pattern) | (uniform-only) | n/a | 0 under skew | Drop, do not promote |

## Initial Hypothesis Shortlist (post-baseline)

1. **Profile baseline with rocprof-compute SoL**
   - Bucket: meta (not a code change; informs all subsequent rounds)
   - Why first: H14 only inferred bandwidth-bound from arithmetic; SoL gives evidence (HBM util, MFMA util, occupancy, LDS BC, scratch) for **all 6 cases** (not just B=16 worst).
   - Expected real-training gain: 0 directly; routes the campaign.

2. **S1: sched_group_barrier batched + LocalPrefetch hoist** (combined; H9 已证伪 ≠ 单独加)
   - Bucket: K1
   - Why next: line-level CK diff identifies this as the simplest topology delta.
   - Expected real-training gain: +2-5% combined-step if MFMA-vmem interleave is the bottleneck; 0 if bandwidth-bound.

3. **S2: BLK 128×128 + LDS halve (As[2][2]→As[2]) + occ=2**
   - Bucket: K1
   - Why fallback: 250x250 已建 mfma323264_agpr_inplace 基础（see [[feedback_fp8_rrr_32x32_foundation]]）but per-warp area thesis 已 partially flawed ([[feedback_fp8_rrr_32x32_flawed_premise]]). Need re-evaluate post-SoL.

## Temporary Survey Assets
- Temp repo path(s): none required — CK reference already in `3rdparty/`
- Notes preserved: this file

## Bottom Line
- **Best existing implementation found**: CK_tile fp8 grouped (structural reference, <5% gap reportedly).
- **Most relevant action to try locally first**: rocprof-compute SoL baseline on all 6 shapes — converts "physics-bound" inference into evidence.
- **Biggest risk / uncertainty before BASELINE**: that the bandwidth ceiling is real, and S1/S2/S3 kernel-internal work returns 0. If so, only chunked persistent rewrite has structural lever left, or campaign reports gap as physics-bound.
- **Confirmed bucket of the most-relevant idea**: K1 (kernel-internal).
