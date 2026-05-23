# gfx950 / grouped_gemm_fp8_rrr / HK — Reusable Tips

Append concise reusable tips after each round; one per ROCm/op/backend lesson. Do not dump full round summaries here.

## 2026-05-21 (pre-campaign distillation)
Tips harvested from 18+ ad-hoc pre-campaign attempts (see `/wekafs/kyle/code2/remote_sync/CLAUDE.md` §3 for memory index):

### Compiler/codegen quirks
- **HK grouped RRR main loop spill is structural (4-acc + 2-deep ping-pong As/Bs)**, NOT in lambdas/view-copies/uniform scalars. Removing patched gl<> views entirely yields 0 V/spill delta. (H5 diagnostic)
- **Single-b0 main loop (cA→cB→cD→cC, no b1)** drops NMASK=0 spill 61→37 scratch 248→152 in BN=256. Don't reintroduce b1 ping-pong in `grouped_rrr_kernel_body` — it costs more VGPR than scheduling saves. (H6 / H18 reverts)
- **Shifting B SRD base by group_idx + dropping group dim from b_co** (H3 pattern) helps BN=256 NMASK=1 (spill 61→58); applying same to BN=128 instead adds +6 VGPR with 0 spill benefit — pattern only transfers when dropped dim was multiplexed across already-spilling paths.

### Hardware constraints
- **MI355X LDS = 160 KiB / CU**. HK BN=256 As[2][2]+Bs[2][2] ≈ 96 KiB/wave × 2 waves = 192 KiB > LDS → HW pins to 1 wave/CU regardless of `launch_bounds(_,2)`. `_,2` hint alone = 0 effect. (H8)
- **gfx950 raw_buffer_load_lds does NOT clamp OOB when SRD has swizzle enabled.** Per-group A SRD in byte-level addressing MUST use `make_srsrc(... 3rd arg=0)` (non-swizzled) for HW clamp to be reliable. (var_k / byte-level addressing)

### Anti-pattern: lever exhaustion
- **Single sched_group_barrier additions don't help in HK** because multiple intra-loop `s_barrier`s already prevent compiler reorder. (H9) Combining with LDS halve / single-acc rewrite is required.
- **Per-warp output area for BN=128 forced worst-shape is gap 23.85% → 23.56% (no diff)**. BN=128 spill=0 + per-warp 64×64 == CK size but perf flat. Per-warp area is NOT the lever; K-loop topology + scheduling are. (H_bn128_force)
- **At B=16 worst shape, kernel is HBM-bound (~1.1 TB/s wall)**. Grouped streams 544 MB B data vs dense 364 MB → 50% volume diff explains 25% TFLOPS gap. No kernel-internal lever can close this; algorithm-level (chunked persistent + cross-group B reuse) is required.

### Build-cache hazard
- **PT build cache silently keeps stale `.cuh` after dual HK path cp**. Must `rm -f build/temp/.../HipKittens/...*_hip.cpp` + `touch .cpp` to force hipify regeneration. Symptom: source edit has 0 perf delta and 0 V/spill delta after rebuild.

### Forbidden by user / framework
- W2/W3 cache (id(activation)/id(grad_out)) — Rule 11 forbidden.
- CK fallback for worst-shape dispatcher — user 2026-05-21 「严禁」.
- `M_per_group: constexpr` kernel specialization — kills MoE skew (Rule 11 GroupGemm clause).
