# Round-14 — gpt_oss FP8 kernel-only ceiling: fwd RCR PMC closes launch-geometry door

**Date**: 2026-05-07 (UTC)
**Repo**: Primus-Turbo, branch `dev/kyle_hipkitten_bf16` (HEAD `a4349894` → R14)
**Scope**: gpt_oss FP8 kernel-only suite, fwd persistent RCR kernel
(`grouped_rcr_kernel<0, true, true, false>` — `<DSCALE=0, N_MASKED_STORE=true, FUSED_KTAIL=true, FUSE_ACT=false>`).
**Goal**: Diagnose what bottlenecks Down-B4-M2048 fwd at 1482 T (worst fwd shape, 1.5 wave-steps/CU)
after R4 falsified the slots-reduction lever for fwd RCR.

## Bottom line

PMC confirms `Down_B4_M2048 fwd` is launch-under-saturated (lowest SQ-busy / lowest MFMA-active
across the 3 anchors), but **LDS and VMEM are NOT bottlenecks** on any fwd RCR shape. Combined with
R4's negative slots-sweep result, this means the per-tile overhead is **inside** the persistent
K-iter loop body (per-tile prologue + K-tail fuse epilog), not in the persistent scheduling layer.
The cheap dispatcher levers (group_m / num_xcds / num_slots) and the env-hook lever (TK_RCR_NUM_CUS)
are now jointly exhausted for fwd RCR. **No metric-moving rule landed in R14**; deliverable is the
PMC characterization that retargets R15+ at per-tile-body kernel surgery.

## PMC counter pass (3 anchors)

Single-pass gfx950 counter set:
`GRBM_GUI_ACTIVE`, `SQ_BUSY_CYCLES`, `SQ_VALU_MFMA_BUSY_CYCLES`, `SQ_VALU_MFMA_COEXEC_CYCLES`,
`SQ_INSTS_VALU_MFMA_F8`, `SQ_INST_CYCLES_VMEM_RD`, `SQ_WAIT_INST_LDS`.

`rocprofv3 --pmc <counters> -- python3 _probe_fp8_kernel_rocprof.py fwd <B M N K> 30`

| shape (fwd, 30 traced calls) | wave-steps/CU | TFLOPS (R14 metric) | GUI cyc | SQ_busy/GUI | MFMA_busy/SQ | COEXEC/MFMA | VMEM_RD/GUI | LDS_wait/GUI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Down_B4_M2048    (worst)       | 1.50  | 1482 |    80.9 M | **328 %**   | **1364 %** | 10.9 % | **78.9 %**  | **119 %** |
| Down_B4_M4096    (sibling)     | 3.00  | 1895 |   111.7 M |   348 %     |   1861 %   | 10.9 % |  114 %      |  173 %    |
| GateUP_B32_M4096 (saturated)   | 46    | 2058 | 1640.5 M  |   394 %     |   1717 %   | 10.6 % |  120 %      |  186 %    |

(Ratios are summed-across-CUs / single-counter-GRBM-cycles. Same MI355X normalization as R1's
wgrad PMC pass — the **relative** per-CU stall fractions across the 3 shapes are the actionable
signal, not the absolute %.)

### Reading

1. **MFMA utilization scales monotonically with wave-steps/CU until ~3 wave-steps**. Down-B4-M2048
   (1.5 wave-steps) at 1364 % vs Down-B4-M4096 (3 wave-steps) at 1861 % — the M=4096 sibling
   extracts +36 % more MFMA per SQ-busy cycle just from doubling the wave-steps. Past 3 wave-steps,
   the GateUP-B32-M4096 (46 wave-steps) drops back to 1717 % — same ceiling regime as wgrad
   (R1 wgrad numbers showed identical pattern, peak at 3-10 wave-steps/CU). The under-saturation
   penalty for short-grid fwd RCR is real and concentrated at < 3 wave-steps/CU.
2. **VMEM and LDS stalls are LOW on the worst shape**, not high. Down-B4-M2048 has VMEM_RD/GUI=79 %
   and LDS_wait/GUI=119 % — both the LOWEST of the 3 shapes. Larger shapes accumulate proportionally
   more memory traffic per GUI clock, so the worst-shape result is not a stall problem. This kills
   the per-tile data-prefetch lever (cluster size `cp.async.cg`, prefetch-distance widening) and the
   LDS-bank-conflict lever — they would primarily lift the SHAPES THAT ALREADY WIN.
3. **MFMA-COEXEC fraction is constant ~11 % across all shapes**. Same on under-saturated and
   saturated regimes, same as R1 wgrad PMC. Means the MFMA-vs-non-MFMA pipelining is shape-
   independent — fixing the 11 % is a kernel-template change (e.g., 4-wave / 16x16 spec swap),
   not a per-shape config knob.

### What this rules out for fwd RCR

- **Slot-count reduction (R4 lever)**: already falsified by R4 sweep (Down-B4-M2048 fwd
  +1.47 % at slots=208 within ±2 % p20 noise; M=4096 −17 % at any slots<256). PMC explains why:
  the under-saturation is structural (1.5 wave-steps/CU) and even with the persistent grid trimmed
  to 192-slots each CU still does ≤ 2 tiles, with launch tail dominating the steady-state win.
- **Per-tile data-prefetch lever**: PMC says VMEM_RD is sub-80 % of GUI on the worst shape
  (versus 120 % on saturated). If we could **double** the VMEM rate on Down-B4-M2048 fwd we would
  only catch up to where saturated shapes already are — and we already know saturated shapes
  outperform by 40 %. So the VMEM ceiling lift would not unlock the under-saturated regime.
- **LDS-bank-conflict lever**: same — LDS_wait/GUI on Down-B4-M2048 is 32 % BELOW the saturated
  shapes' LDS_wait. Cannot be the bottleneck for the worst shape.

### What's left (R15+ candidates)

1. **Per-tile prologue trim** (G1c-(1) from R1 wgrad notes, ported to RCR fwd):
   the chiplet swizzle + 6-step binary search + LDS-cumsum-init (lines ~2697-2730) is fixed-cost
   per persistent loop iter. For Down-B4-M2048 with 1.5 wave-steps/CU, this prologue is paid per
   tile but amortized over only ~22.5 K-iter cycles. Hoist invariants across tiles within the same
   group (`m_per_group` constant: cumsum offset + group index + `ki_g` = `M_g/HB` are all
   group-uniform — currently re-derived each tile). Estimated: −10..−30 cycles per tile prologue,
   ~+1..+3 % on under-saturated shapes, +0 % on saturated. Cost: 1-2 rounds of invasive kernel
   surgery + correctness probe + bit-equivalence check.
2. **K-tail fuse cost on under-saturated shapes**: for K=2880 K_REM=64, FUSED_KTAIL=true adds ~0.5
   K-block worth of MFMA + extra A_row_reg → register liveness pressure (R34-dm noted FUSED_KTAIL
   templates use 28 vs 62 epilog scratch_store/mfma pairs vs 22/44 — a register-allocator effect).
   For under-saturated shapes the wave-steps don't hide this; for saturated shapes they do. Test:
   add a runtime gate to FUSED_KTAIL eligibility, keyed off `wave_steps_per_cu < 3` (computable host-
   side from `g.M_total / BLOCK_SIZE × g.bpc / NUM_CUS`). If the unfused path beats fused on
   Down-B4-M2048 fwd by > 1 % at tight verify, ship a single-rule predicate.
3. **N_MASKED_STORE under-saturation cost**: gpt_oss N=2880/5760 hits the `N_MASKED_STORE=true`
   spec — every tile pays the masked-store branch cost. Down-B4-M2048's last-col-tile is 64 cols
   wide (2880 % 256 = 64), one of the smallest masked tiles. The per-thread column-mask compare
   adds branchy stores. A persistent-loop branch hoist (mask once per (CU, tile) when the col-id
   is wave-uniform) could trim ~5-10 cycles per tile. Same EV class as #1 (1-2 rounds, +1-3 % on
   under-saturated shapes).

## R14 deliverables

### Primus-Turbo
- `scripts/_probe_round_14_fwd_rcr_pmc.py` — PMC pass driver.
- `analysis/_notes/round-14-fp8-fwd-rcr-pmc-undersaturation-launch-geometry-exhausted.md` (this file).
- **No** `select_default_config` change. **No** `grouped_gemm_fp8_impl.py` change.

### HipKittens
- No change. (Round-4 already shipped the env-hook infrastructure for `TK_RCR_NUM_CUS`; R14
  just confirms via PMC that no slot-count rule justifies pulling on that hook in the
  dispatcher.)

## R15 plan

Pick the **#1 candidate (per-tile prologue trim)** from the candidates above — it has the cleanest
falsification path (PMC already-confirmed under-saturation, hoisting work is a textbook compiler-
style optimization, bit-equivalent by construction). Concrete steps:

1. Identify per-tile invariants in `grouped_rcr_kernel` (file-scope, lines ~2697-2780):
   - `lo` (group index) — currently re-derived via 6-step binary search per outer iter
   - `m_off` (`s_offs[lo]`) — group base offset
   - `m_per_g` (`s_offs[lo+1] - s_offs[lo]`) — group M extent
   - `ki_g` (`m_per_g / BLOCK_SIZE`) — tiles per group along M
   - `n_off_within_group` (`gt - s_cum_tiles[lo]`) — tile index within group
2. Hoist when adjacent outer iters fall into the same group: cache last `lo` + `s_cum_tiles[lo+1]`,
   skip the binary search if `gt < s_cum_tiles_cached_hi`. For groups with wave-steps ≥ 2 (the
   under-saturated regime), every other iter hits the same group → ~50 % skip rate on the binary
   search.
3. Bit-equivalence: same group → same offsets/extents, no math change. SNR > 25 dB on Down-B4-M2048
   fwd output (R3 SNR was 297 dB on the corresponding wgrad; same formal property holds here).
4. Tight-verify with 1500-iter × 7-trial p20 on Down-B4-M2048 fwd; if win > 1 % with 3/3 seed
   positivity, ship; else falsify and pivot to candidate #2 (K-tail fuse runtime gate).
