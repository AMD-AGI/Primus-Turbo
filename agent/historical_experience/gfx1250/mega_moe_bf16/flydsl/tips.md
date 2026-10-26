### 2026-10-23 05:09 round-2 — ACCEPTED
- Context: gfx1250 Mega fused kernels (dispatch+GEMM, GEMM+combine), DSV3 shape, EP1/EP2
- Signal: the comm/combine roles share the GEMM launch, which is LDS-limited to one workgroup per CU, so a CU moves data with only that workgroup's waves; dispatch measured ~5 GB/s per wave (one 14 KB row in flight)
- Takeaway: 4x2 waves (8 wave32) beats 2x2 on all four fused kernels (combine phases -0.13..-0.18 ms at EP1, step +3.6% at EP2) with the same GEMM rate; 4x4 spills (4-5x slower)
- Applicability: any gfx1250 kernel that runs data movement inside a 1-WG/CU GEMM launch; re-check the GEMM tile rate when the tile shape changes
### 2026-10-23 05:13 round-3 — ROLLED BACK
- Context: gfx1250 Mega gemm_combine, EP2
- Signal: combine chunk 16/32/64 rows all within +-0.2%; dispatch remote bandwidth identical for store aux SYS|NT, SYS|RT, SYS|HT, SYS|bypass, DEV|NT
- Takeaway: CU remote-store cache policy does not move XGMI bandwidth on gfx1250; look at per-expert / per-row publish serialization (store-retire wait + barrier + flag) instead
- Applicability: any CU-driven peer push on gfx1250
### 2026-10-23 05:40 round-4 — ACCEPTED
- Context: gfx1250 Mega dispatch (dispatch_bf16_tile), EP2, 32 experts per rank, 64 comm CUs
- Signal: per-expert wait_mem_all + barrier + flag costs ~3 us each; dispatch-only 490 -> 393 us when published once
- Takeaway: batch remote publishes when nothing consumes them early; when GEMM tiles wait on per-expert flags, groups of 8+ delay the first tiles more than they save
- Applicability: any CU push that publishes per-segment flags over XGMI
### 2026-10-23 05:50 round-5 — ACCEPTED
- Context: gfx1250 Mega dispatch-only (x re-dispatch for dW1), EP2
- Signal: dispatch-only + gather 600 / 470 / 424 / 413 us at 64 / 128 / 192 / 254 comm CUs
- Takeaway: a comm-only launch should take the whole GPU; the peer push rate keeps rising (sublinearly) to ~192 CUs
- Applicability: standalone CU-driven dispatch / combine launches on gfx1250
### 2026-10-23 06:15 round-6 — ACCEPTED
- Context: gfx1250 Mega gemm_combine, EP2 (worst-case pool 544 tiles, a call fills ~273)
- Signal: per-WG realtime stamps: GEMM done ~1020 us, tail combine WGs start 1155-1510 us; masked tiles cost real time
- Takeaway: size the grid from the host-known pool rows (the prologue syncs them) with a runtime fx.Int32 grid; never let a worst-case masked grid sit in front of a role that must start promptly
- Applicability: every Mega launch with a MAX_TILES grid (dispatch+GEMM too)
### 2026-10-23 06:12 round-8 — ACCEPTED
- Context: gfx1250 Mega gemm_combine L2 forward fold sum (combine_bf16_tile, 8 top-k rows, 4 passes per fold row), EP2
- Signal: a dead `if tid == 0: read_clock()` sped the launch 60 us; forward-only ISA: the sum issued 7/1/1/1/1/1/1/20/4/19 NT loads between waits vs ~53 per burst, and 168 `v_cmpx_eq_u64` (readfirstlane waterfall, `s_wait_loadcnt 0x0` inside) — one pool row's buffer descriptor lived in VGPRs
- Takeaway: values loaded from wave-uniform indices are not provably uniform; `_readfirstlane_i32` them before building buffer descriptors. Count `v_cmpx_eq_u64` in `22_final_isa.s` to find waterfalls. When an unrelated edit changes speed, diff the ISA of the kernel you timed: both gemm_combine variants dump to one dir, the last compiled wins
- Applicability: any FlyDSL kernel that builds buffer resources from loaded row indices
- Context: FlyDSL disk cache
- Signal: an ep_intranode.py edit left the bf16_gfx1250 kernels' timing unchanged with the cache on, changed it with FLYDSL_RUNTIME_ENABLE_CACHE=0
- Takeaway: the cache key follows helper functions only from the launcher's own directory (`_is_user_function`); helpers elsewhere need a hash in a jit argument's `__cache_signature__` (SymBuffer now hashes ep_intranode.py). Before trusting a bench of a cross-directory helper edit, re-time once with the cache off
- Applicability: every launcher that calls device helpers from another directory (prims.py, buffer_ops.py too)
### 2026-10-23 06:42 round-9 — ACCEPTED
- Context: gfx1250 variable-K wgrad (dW2 / dW1) on a dispatched operand, EP2 DSV3 shapes
- Signal: two torch `vectorized_gather_kernel`s (224 + 232 us at T=8192) only reordered rows into pool order
- Takeaway: TDM gather mode (16 rows per op with 16-bit indices) lets the GEMM read rows through an index table; per wave, load a k-tile's indices one tile ahead into an rmem cell and `rocdl.readlane` them into the descriptor. Gather mode itself costs 3-5% of the kernel (identity rows as slow as random), so expect ~2/3 of the removed copy as net gain. Tail rows can re-read a valid row when the other operand's tile TDM zero-fills them; an empty group has no valid row (stale LDS), zero its accumulators
- Applicability: any TDM GEMM whose operand is a row permutation of a resident buffer (MoE wgrad, gathered A)
### 2026-10-26 03:00 EP4 round-2 — REJECTED
- Context: gfx1250 Mega gemm_combine lead combine knobs, EP4 (GPUs 0-3), DSV3 T=8192 random
- Signal: 8-48 lead workgroups at 10-40% of the GEMM tiles all within -0.03..+0.4% fwd+bwd; per-workgroup stamps show the GEMM done at 1019 us, the tail push at 1294 us median, the reduce only ~70 us
- Takeaway: a fold row is pushable only once its token's highest local expert's rows are done, so the push backlog piles up at the GEMM's end (~x^2 of the GEMM fraction x at ~2 local routes per token); adding pushers does not shorten it. Also: gfx1250 has no HW_REG_XCC_ID (HW_ID1 varies only in WGP_ID / SA_ID), and 8-wave workgroups can queue behind occupied slots while other CUs idle, so late-starting workgroups must take work from counters, not static shares
- Applicability: any fold-then-push combine fused behind a GEMM; any gfx1250 kernel that assumes blockIdx -> XCD placement
### 2026-10-26 03:25 EP4 round-3 — ROLLED BACK
- Context: gfx1250 Mega dispatch prologue host sync, EP4, DSV3
- Signal: a host-mapped rows notify (coherent hipHostMalloc, system-scope store at the end of phase C, host poll) cut steady-state GPU idle per step from ~110 to ~30 us, but same-session A/B fwd+bwd moved +0.14% / -0.09%
- Takeaway: a GPU kernel can publish a value to coherent pinned host memory mid-kernel and the host sees it before the kernel ends (25/25, no fallback); but at EP4 a per-rank idle gap of ~1% of the step is not on the critical path — the freed time reappears as cross-rank waits in gemm_combine and the prologue's barriers. Gate idle-removal rounds on an alternating A/B of the full step, not on the trace's idle
- Applicability: host-sync removal / launch-overlap work in cross-rank-coupled EP kernels
### 2026-10-26 03:31 EP4 round-4 — REJECTED
- Context: MI455X DSV3 Mega step at EP4 (T=8192), power / clock telemetry; wgrad XCD band remap
- Signal: amd-smi during the bench: socket power 2100-2430 W of the 2500 W PPT0 limit, GFX clock 2380 MHz idle -> 2000-2170 MHz loaded, 1650-1680 MHz at the peak sample, hotspot only 60-64 C; num_xcd=8 band remaps on the wgrads +0.1..+0.4%
- Takeaway: the fused MoE step on MI455X runs near the power cap with clocks well below idle, so sample `amd-smi metric -p -c -t` during a bench before attributing a null result (corrected in EP4 round 5: this does not absorb idle removal). A blockIdx-mod-8 XCD remap is not a safe default on MI455X until the real workgroup -> XCD / L2-cluster mapping is known
- Applicability: any MI455X GEMM-heavy kernel campaign; any placement / swizzle remap on gfx1250
### 2026-10-26 03:49 EP4 round-5 — REJECTED
- Context: gfx1250 Mega gemm_combine tail push (ep_intranode push constants), EP4, DSV3 T=8192
- Signal: ROW_UNROLL 2 / FOLD_CHUNK_ROWS 32 / MAX_LIF 128 all within ±0.2%; one remote chunk push takes 32 us in the lead phase and 35 us in the tail; a 100 / 200 / 400 us host delay after the prologue adds ~157 / 295 / 550 us to the step
- Takeaway: with 8-wave WGs each wave gets 2 fold rows, so the `_ROW_UNROLL=4` grouped path is never taken and each row waits for its own store retire (~7 GB/s per WG); check which path a constant reaches at the target shape before screening it. Idle is on the EP4 critical path (it is not absorbed by the power cap, correcting round 4)
- Applicability: any per-row CU push with a store-retire wait; any "the clock absorbs it" attribution
### 2026-10-26 06:48 TDM gather intermittent corruption — OPEN
- Context: gfx1250 TDM gather (`tensor_load_to_lds` gather mode, 16 rows per op) in the Mega dispatch_gemm nt tile, EP4
- Signal: one 256x256 tile wrong per event in 16-row blocks (slot not written / data shifted +0x1100 into the next slot); ~1 / 500 fused fwd launches; standalone gather GEMM with 70 OOB padding rows per expert 10 / 160000, without padding 0 / 160000; the bad tiles are full tiles of the same expert as the padded one
- Takeaway: gather ops that barely touch memory (OOB -1 rows, duplicate / hot rows, zero extent) can corrupt a neighbouring WG's TDM gather writes; waits, delays, descriptor holds and memory type only lower the rate. Per-lane `global_load_async_to_lds_b128` (ASYNCcnt, SADDR form) removes it at ~6% of the GEMM; spreading those loads over the ks sub-steps is slower. Validate any TDM-gather GEMM with a bitwise multi-launch stress that includes padding / duplicate indices
- Applicability: every gfx1250 TDM gather user (Mega dispatch_gemm, variable-K wgrad gather)
