# Mega MoE BF16 gfx1250 EP4 — L2 / AID placement campaign (2026-10-27)

Reference: `opt.md` (minimax-poc `Q&A/l2-cluster-aid-mapping/README_CN.md`): MI455X has 2 AIDs of 4 XCCs, one
GL2 per AID that caches only what its own XCCs read; TensileLite's ClusterAIDMapping gives each AID one half of
the output grid (+0.8% BF16). Primary metric: Combined Step TFLOPS geomean over the 3 EP4 DSV3 shapes.
Rounds are judged by same-session alternating A/B against the accepted-tree copy (`tools/ab.sh`).

### round-1 — BASELINE (2026-10-27 05:51)
- 8bbd8c89: 1039.651 TF step (1446.4 / 616.0 / 1261.3), all tests pass.
- Probe: workgroup `bid` runs on XCC `(bid + 7) % 8` in every launch on GPUs 0-3, no rebalancing across XCCs.
- Trace: the six GEMM launches are 10.26 of 11.07 ms busy (T=8192 random).
- Model: gemm_combine's row-major order (28 N blocks) makes both AIDs read every weight and activation block;
  dispatch_gemm's super-tiles make both read every weight block.
- `rocprofv3 --pmc` hung the SMU on this node (reboot); no counter collection in this campaign.

### round-2 — ROLLBACK (2026-10-27 06:40)
- gemm_combine AID N-half split, AID_GROUP_M 2 and 4 vs acc, same-session A/B x2: +0.005% / +0.09% step geomean.
- Kernel trace: gc2 L2 fwd gemm_combine +12.6%, L1 dgrad gemm_combine -3.9%; gc4 within 0.4%.
- Stage bench added (`tools/bench_mega_gfx1250.py`, bench_mega_moe.py's view on the gfx1250 kernels): standalone
  grouped GEMMs at 94-97% of dense; fused stages 1.19-1.49x gemm_only, 48% of the comm hidden at t8192 random
  (28% bench_dsv3), dW1's x re-dispatch fully exposed. The slack is in the overlap, not in GL2 placement.

### round-3 — ROLLBACK (2026-10-27 06:55)
- dispatch_gemm AID N-half split, always (-0.54%) and adaptive by pool vs weight blocks (-0.28%), A/B x2.
- XGMI ceiling measured (`tools/xgmi_peak.py`): one GPU injects ~1.58 TB/s over 1 or 3 links (ingress 1.65,
  full duplex); all-to-all: SDMA 793, RCCL 657-755, Mega dispatch_only at 255 CUs 1129 GB/s per GPU.
- Mega comm legs at t8192 random: dispatch at 64 CUs 519 GB/s (33% of injection), combine push 290 GB/s (18%).
- GL2 placement closed (rounds 2-3); next rounds target the overlap.

### round-4 — ACCEPT (2026-10-27 15:04)
- dW1's x re-dispatch: SDMA peer copies of all of x into a second symmetric receive buffer (rows src*T_max), on a
  side stream during the L1 dgrad gemm_combine, epoch flags + 1-WG wait kernel before dW1.
- A/B x2: +1.16% step geomean (t4096 +1.88%, t8192 +1.17%, skew1 +0.44%); full tests pass; bitwise stress
  T=256 x 60, T=8192 x 50, 0 mismatches.
- Trace: dispatch_only 304 us gone; L1 dgrad gemm_combine +65 us (XGMI shared with the copies); wait 1.8 us.

### round-5 — ROLLBACK (2026-10-27 15:45)
- gemm_combine `_FOLD_CHUNK_ROWS` 16 -> 32 / 64 (each wave pushes 4-row groups): -0.47% / -1.15%, slower on
  every shape, worst at t4096 (L2 fwd combine). Coarser chunks delay the first push and starve the tail.
- Stage bench of the round-4 tree: combine stages still the largest gaps (t8192 L2 fwd 0.45, L1 dgrad 0.49 ms;
  bench_dsv3 L2 fwd 0.73 ms). Next: a clock-stamp timeline of gemm_combine before another combine change.

### round-6 — ACCEPT (2026-10-27 15:48), EP2 on GPUs 2,3 from here on
- gemm_combine lead combine workgroups 16 @ 25% -> 0 (all combine workgroups after the GEMM tiles).
- Motivation: clock-stamp timeline (`tools/gc_trace.py`, EP4, round-4 tree): leads push 72-121 of ~320 MB before
  the GEMM ends but slow it ~89 us (L2 fwd) / ~144 us (L1 dgrad); no lead ends both launches ~100 us earlier.
- A/B x2 at EP2: l0 +0.687% (every shape, every rep), l4 +0.450%, l8 +0.045%. Full tests pass (EP4 skipped).
  Bench: 1285.765 TF step at EP2.
- EP2 baseline (16:00): same-session A/B x2, 8bbd8c89 / 3ef91841 / d97bd80c at EP2: 1260.143 / 1272.241 (+0.960%) /
  1282.641 TF (+1.785%), faster than base on every shape. Later EP2 rounds report vs this baseline.
- EP2 stage bench of the round-6 tree (`rounds/round-6/artifacts/stage_bench_ep2/`): fused/gemm_only at t8192 random
  L2 fwd 132.5% (+0.30 ms), L1 dgrad 123.9% (+0.41 ms), L2 dgrad 122.0%, L1 fwd 113.0%; step 10.921 ms.

### round-7 — STOPPED (2026-10-27 16:15)
- gemm_combine GROUP_M super-tile order of the GEMM tiles. One GPU: Mega's tile row-major is 5.0% (NT L2 fwd) /
  3.0% (NN L1 dgrad) slower than GROUP_M 16. EP2 A/B vs round 6: GROUP_M 16 +0.104%, 8 +0.356%, 4 +0.544% /
  +0.446%, 2 +0.813% (every shape). Tests pass at GROUP_M 16; GROUP_M 2 not fully validated.
- Stopped for submission: kernel reverted to round 6, change kept as `rounds/round-7/gemm_combine_group_m.patch`.
