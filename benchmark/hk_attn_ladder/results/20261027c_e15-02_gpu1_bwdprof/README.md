# HK MLA backward (r7 = Primus-Turbo 71bd0329): profile, power wall and roofline on e15-02 GPU1

DeepSeek-V3 MLA backward, b2 s4096 h128 (SBHD, folded to 256 heads), qk 192 / v 128, causal, bf16.
All timings: `hkl.ladder` (settled clock, 512 MB flush before every call, 10 rounds, null control), GPU1 only.
Files: `base_randn.*`, `base_zeros.*` (hk_v0 / r7 / FlyDSL with randn and all-zero inputs), `abl_randn.*`
(ablations of r7, wrong results, timing only), plus the r7 ktrace / ATT of `../20261027b_e15-02_gpu1`.

## 1. The card runs r7 at the power cap: time follows energy, not cycles

| inputs | r7 ms | sclk MHz | Mcycles | hk_v0 ms | hk_v0 sclk | FlyDSL ms | FlyDSL sclk |
|---|---:|---:|---:|---:|---:|---:|---:|
| randn | 2.472 | 1251 | 3.093 | 2.805 | 1428 | 3.559 | 1515 |
| zeros | 1.736 | 1831 | 3.176 | 2.077 | 1974 | 2.678 | 2022 |

- Same instruction stream, same cycle count (within 3%), but zero inputs (almost no bit toggling) run 1.42x
  faster at a 46% higher clock. So r7 is clock-limited by the 2500 W cap, not issue-limited.
- hk_v0 -> r7 removed 23% of the cycles (1.295x) but only 12% of the time (1.135x): the clock fell 1428 -> 1251.
- At the cap, time per call ~ energy per call / cap power. A change pays in time by the energy it removes;
  removing stall cycles pays only partly (the r7 asm patch: -4.7% cycles -> -1.8% time, ratio 0.38).
- hwmon `power1_input` cannot be read under load on gfx1250 (OSError, 49/60 reads); energy is inferred from
  time at the cap, and from the clock.
- ktrace runs (a few calls, unsettled clock) show the burst picture: delta 42 / dkdv 1489 / dq 464 us
  (zeros: 40 / 1237 / 386). Settled (scripts/mla_b0_bench.py per-launch): 57 / 1859 / 511 us.

## 2. Where the energy goes: ablations (timing only, wrong results; `HKL_ABL_*` knobs, patch a0)

Each ablation removes one component from r7np (r7 without the dK/dV asm patch, which the ablated sources
cannot carry). Step = r7np time / ablation time (10 rounds, randn); null control 1.0007x.

| removed | time step | cycle step | sclk MHz | share of the call |
|---|---:|---:|---:|---:|
| (r7np baseline: 2.5217 ms) | | | 1290 | |
| dkdv dS^T export (DPP + 6 stores / iter, 4.33 GB written) | **1.084x** | 1.014x | 1378 | ~7.6% |
| dq dS reads (4.33 GB TDM) | **1.073x** | 1.030x | 1344 | ~6.9% |
| dkdv Q / dO ring streaming (L2 -> LDS, 14.65 GB) | **1.055x** | 1.019x | 1334 | ~5.2% |
| dkdv softmax / dS VALU (fma, exp, mul; NL / ND loads) | 1.036x | 1.048x | 1276 | ~3.4% |
| dq K reads (0.40 GB HBM, 2.35 GB L2 -> LDS) | 1.025x | 1.004x | 1318 | ~2.4% |
| dkdv v_exp only | 1.016x | 1.015x | 1293 | ~1.6% |
| r7's asm patch (r7np -> r7) | 1.018x | 1.049x | 1290 -> 1253 | |

- Data movement is the largest removable energy: the dS round trip through HBM alone is ~14.5% of the call
  (~1.7% of the call per GB through HBM; L2-resident streaming costs ~0.36% per GB, ~5x less).
- Softmax VALU is ~3.4%, exp ~1.6%: VALU trimming is worth at most ~1-2%.
- Cycle-only savings (stalls, waits, barriers) convert at ~0.4: ABC's -17% loop cycles (a08 skeleton) would be
  ~-5% dkdv time, ~-4% bwd.

## 3. Roofline at the measured clock (1253 MHz, randn)

| kernel | executed GFLOP | WMMA floor @1253 MHz (1946 FLOP/clk/SIMD) | measured (settled) | % | HBM bytes | HBM floor (19 TB/s) |
|---|---:|---:|---:|---:|---:|---:|
| k_delta | - | - | 57 us | - | 0.55 GB | 29 us |
| k_dkdv96ds | 2813 | 1.13 ms | 1.859 ms | 61% | 1.35 R + 5.0 W | 0.33 ms |
| k_dqds_c | 831 | 0.33 ms | 0.511 ms | 65% | 4.73 R + 0.40 W | 0.27 ms |
| bwd | 3644 | 1.46 ms | 2.47 ms (ladder) | 59% | 11.2 GB | 0.59 ms |

The WMMA floor itself moves with the clock: at zero inputs (1831 MHz) it is 1.00 ms for the whole bwd, and
r7 then takes 1.736 ms (58%). Neither HBM bandwidth nor issue rate is the limit; energy per call is.

## 4. Ranked levers (expected time change of the whole bwd; cost)

| # | lever | evidence | expected | cost / risk |
|---|---|---|---|---|
| 1 | ABC: cross-iteration pipelining of the dkdv full loop (Phase 2; WAR-V gated) | a08 skeleton -17% loop cycles; conversion 0.38 | -3 to -5% | days; asm generator; WAR-V gate |
| 2 | cache-policy bits: dS stores / dq dS + K TDM loads streaming (no L2 / MALL allocate), dkdv Q / dO | dS round trip 14.5%, K 2.4% of energy | 0 to -1% | 1 build |
| 3 | dq head order reversed to dkdv's write order (the newest dS may still be in MALL) | MALL 64-128 MB per AID | 0 to -0.5% | small |
| 4 | dkdv C-init / softmax VALU trimming (scale folded into K / V fragments, NL / ND into the WMMA C) | VALU 3.4% | -0.5 to -1% | SNR check; asm patch regen |
| 5 | skip fully masked S / dP WMMAs on diagonal tiles (energy, not cycles) | 2.3% executed-but-masked FLOPs | -0.3 to -1% | masked loop |
| 6 | k_delta ramps / load order (l9o, l9f) | 57 us vs 29 us HBM floor | -0.1 to -0.4% | small |
| 7 | 4-wave dK / dV workgroups (192 kv rows, one ring): halves the Q / dO restream | stream 5.2%; a08: +2..6% cycles | -1 to +1% | rewrite; low priority |
| - | not viable: keeping dS in MALL by head chunks (occupancy collapses), fp32 dQ partials (4x the dS bytes), fused atomic dQ (non-deterministic; user rule) | | | |
