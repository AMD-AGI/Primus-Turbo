# 2026-10-30 e15-02 GPU2: the second HK MLA forward campaign (ladder `mla/ladder_fwd2.json`, patches 01-25)

Question: how far does `k_mla_fwd` (a08-1 E2 = rung `f0`, 0.8-1.8% slower than FlyDSL 917e240b, see
`../20261030_e15-02_gpu2_fwd0/`) go while staying bitwise equal to FlyDSL? Answer: **rung `f24a` runs the production
shape in 0.741 ms cold (FlyDSL 0.798 in the same run: 0.928x, -7.2%; f0: -7.6%) and every main-line rung is still
bitwise equal to FlyDSL**; the burst-mode best is `f19f` / `f22a` at ~0.66 ms (FlyDSL ~0.71). Primus-Turbo
`dev/lhz/dsv3-attn-hk-fwd` d930037a carries `f12a`; f14a..f24a are ladder-only so far.

All on e15-02 GPU2 (container lhz-hkl-g2) through `scripts/gpu_lock.sh 2` + `tools/run_gpu.sh`; every new code
object ran `toy` first, then `hkl.test` on s640 / b1_s4096 / b1_h32 / prod / prod_b4 (SNR vs fp32, 2-run determinism,
bitwise vs FlyDSL); timing with `hkl.ladder` (10 rounds x 20 calls, null control on the top rung,
`HKL_SCALE=0.1352337788608801`, FlyDSL from `HKL_TURBO_DIR=Primus-Turbo-hk-fwd`). cold = L2 flushed before every call;
burst = every call after 20 ms of idle GPU (`--mode burst --gap-ms 20`, the in-step proxy). The node hung at 16:02
(a hive reset started from GPU0, not our GPU): the batches after `cold_p22A` did not run.

## 1. Tree and rungs

`mla/fwd2/gfx1250_mla` is a copy of the merge package (`hsaco.py`, `manifest.json`, `hip/`, `tools/fwdgen`,
`tools/asmkit`) plus `gen/build.py` (hkl.build's generator entry). Each `fwd2_patches/NN_*.patch` adds generator
options or shell `#if` switches with the old behaviour as default; `fNN_null` rebuilds the parent byte for byte
(`sha256` pinned in the ladder) and the feature rungs set the option. Generator options used by the main line:

| option | patch | meaning |
|---|---|---|
| `--va-rule llvm_war_xdl` | 01 | the backward's va_vdst rule: no full drain for a write only an in-flight WMMA reads |
| `--resc branch` / `branch2` | 03 / 17 | O *= corr under a per-q-tile wave-uniform branch (FlyDSL's `_rescale_if`); branch2: its s_cmp is a scheduled node of its own |
| (scheduler) | 08 | LDS loads count as in flight until their modelled completion |
| `--plan 2` | 12 | VGPR plan 2: softmax state in block 2, V split over blocks 3 / 0 |
| `-D HKGEN_XBLK=1` | 14 | ring slots n..n+2 of a block load the next block's tiles 0..2; the epilogue issues only the next Q |
| `--sgpr-lat 20` | 17 | v_cmp (SGPR) -> SALU reader latency 20 in the scheduler |
| `--ds-max 63` | 18 | LDS loads in flight before an explicit s_wait_dscnt |
| `--lat WMMA=32,VALU=8` | 19 | scheduler result latencies |
| `-D HKGEN_ORES=1` | 22 | epilogue O buffer resources with 32-bit bounds (no 64-bit v_cmp -> SALU chains) |
| `-D HKGEN_FMERGE=1` | 24 | one scheduled `FINAL_A` region: last prefix + out / lse of all 4 q tiles, staged by generated ds_stores |

## 2. Main line (prod b2 s4096 h128, ms; FlyDSL measured in the same run)

| rung | adds | cold | FlyDSL cold | burst | FlyDSL burst | file |
|---|---|---:|---:|---:|---:|---|
| f0 | E2 (merge, 1d95fa84) | 0.8019 | 0.7946 | 0.7242 | 0.7150 | cold / burst |
| f1a | va_rule llvm_war_xdl | 0.7898 | 0.7946 | 0.7130 | 0.7150 | cold / burst |
| f3a | + RESC branch | 0.7803 | 0.7966 | 0.7094 | 0.7157 | cold_f3 / burst_f3 |
| f8x | + LDS in-flight model fix | 0.7769 | -- | 0.7049 | -- | cold_p8 / burst_p8 |
| f12a | + plan 2 (= Turbo d930037a) | 0.7765 | 0.7961 | 0.7031 | 0.7153 | cold_p11 / burst_p11 |
| f14a | + XBLK | 0.7641 | 0.7952 | 0.6842 | 0.7128 | cold_p12 / burst_p12 |
| f17a | + sgpr-lat 20, RESC branch2 | 0.7480 | 0.7969 | 0.6688 | 0.7162 | cold_p15 / burst_p16 |
| f18d | + DS_MAX 63 | 0.7477 | 0.7966 | 0.6684 | 0.7162 | cold_p16 / burst_p16 |
| f19f | + lat WMMA 32 / VALU 8 | 0.7457 | 0.7984 | 0.6608 | 0.7100 | cold_p18 / burst_p18 |
| f22a | + ORES (same as f19f within noise) | 0.7466 | 0.7983 | 0.6673 | 0.7176 | cold_p22A / burst_p19 |
| f24a | + FMERGE | 0.7406 | 0.7983 | -- | -- | cold_p22A |

Cycles (Mcycles = ms x median sclk) follow the same order: f0 1.143 -> f24a 1.008 (cold). The forward is power-capped
in cold mode (sclk ~1.37-1.43 GHz vs 2.21 GHz in burst), so a cycle saved converts to less than a cycle of time there.

## 3. Side rungs and dead ends

| rung(s) | what | result |
|---|---|---|
| f1b | no group edges (V / Q(3) reads anywhere) | -4.3% |
| f4a | dynamic per-XCD tickets (HKGEN_TICKET) | cycles -1%, time 1.00x (cold / burst / b4) |
| f5a / f5b | out-of-line decision + rescale; threshold T = 8 | -1%; +3% but o SNR -1 dB (rejected: a lagging max makes the dominant p != 1) |
| f7w* | WMMA stream orders | -6..-7% (K loads need the prefix PVs as cover) |
| f8a / f9a | fwdgen2: one generated LOOP region per block (+ prologue drain) | -1% (ATT: s_wait_dscnt stalls ~126 cyc/tile) |
| f8b | + K prefetch across tiles | -5% |
| pad_* | marginal cost of a redundant VALU / SALU / msb / s_nop in TILE_C | ~2 / 0.5 / 0.4 / 0 cycles |
| f13b* | msb-aware scheduling | no effect (the emitter is within 5-10% of the msb DP bound for a fixed order) |
| f15a-c | `--issue hw`: 1 issue/cycle, an LDS op holds others 4 cycles | +19% cycles (VALU chains packed back to back -> dependency stalls) |
| f16a-d | `--issue m1ds`: model1 + LDS hold | +4% |
| f17b / f17c | sgpr-lat only / branch2 only | 1.015x / 1.011x (additive: f17a 1.021x) |
| f18a-c | s_wait_dscnt merging (2 / 4 / 8) | 1.00x: the waits are real LDS-arrival stalls |
| f19a-e | WMMA 32 / LDS 400 / LDS 220 / both / VALU 8 | within +-0.3% cold; WMMA 32 and VALU 8 -1% burst |
| f20a | incremental TDM descriptors (HKGEN_INCDESC) | LLVM emits as many SALU (47 -> 50); ~1.00x |
| f21a-c, f22b, f24b | TILE_F: tile 0 without the no-op prefix, INIT folded in | -0.4..-0.8% in some ladder orders, **+7% in others** (f21c#null in cold_p18, f21c / f22b in cold_p19B); code 55 -> 61.7 KB |
| f23s / f23e / f23s2 | f22a with 7.2 / 7.2 / 2.4 KB of dead s_nop at the start / end | equal to f22a in both orders: plain code size or offset is not TILE_F's problem |

Built but not timed when the node hung: f24c-e (m1ds with LDS bursts on f22a), f25a / f25b (patch 25 HKGEN_ROWS64: K / V
descriptors bounded at 64 rows, -11 SALU per tile; + FMERGE). The cold-order-B and burst runs of f24a / f24b were cut
off at round 6 (f24a 0.739-0.749, f24b 0.761-0.764 in that order).

## 4. What the probes and traces say (f8x / f14a / f17a / f21a)

- Probe (`micro/fwd_probe.py`, `probe9_f8xp2.json`, `probe13_f14ap.json`): per wave ~1.01-1.04 Mcycles; TILE_C regions
  73-75%, shell between TILE_C regions 5.7-5.9% (~120 cyc/tile), TILE_D 9.8-10.7%, epilogue 8.5% (5.6k/block) -> 6.2%
  (3.9k/block) with XBLK, B_0 2.2% -> 1.2%, INIT 1%.
- Ablations (`abl_*`, from f3a): no softmax VALU -15% cycles in every mode (cycles); no LDS reads -16% cold / -8% burst
  / -2.6% zeros and no K/V TDM -14% / -7% / -3% (mostly power); all three -20% cycles.
- ATT (`tools/att.sh`, `--att-simd-select N` for wave N; analysis scripts in the lhz-devsync notes): every instruction
  issues in 1 cycle, but after an LDS op the next non-LDS op issues 4 cycles later (LDS -> LDS 1); s_wait_dscnt costs
  ~4; v_cmp writing an SGPR -> s_cmp reading it ~20; s_cmp -> s_cbranch 9; a taken branch ~25 (a RESC block costs
  ~34 skipped / ~48 executed, ~33% executed on randn); the WMMA interval mode is 9 cycles, not 8; s_set_vgpr_msb does not
  appear in the issue stream. The epilogue's 11 back-to-back TDM ops stalled on LLVM's `s_wait_tensorcnt 0xa`
  (fixed by XBLK). TILE_C (f17a) is 1533 cycles vs the 1280 WMMA floor: ~74 in RESC branches, ~128 in s_wait_dscnt,
  ~283 of LDS-op issue shadow.
- I-cache: hot-mode PMC (`SQC_ICACHE_MISSES`) is 0 after the first call for both 55 KB and 61.7 KB code; ATT shows
  25-40 cycle stalls per 128-B line for code beyond ~40-52 KB of .text even in builds whose timing is fine.

## 5. Files

| file | content |
|---|---|
| `test_*.json` | hkl.test per batch: SNR of out / lse vs fp32, determinism, bitwise vs FlyDSL |
| `cold_*.md / .json`, `burst_*.md / .json` | hkl.ladder timing; the batch (`p8`..`p22`) or rung group (`f3`..`f8`) in the name |
| `abl_{cold,burst,zeros}.*`, `pad_{cold,burst,zeros}.*` | ablation and padding rungs |
| `b4_f4.*` | tickets at b4 |
| `probe*.json` | `micro/fwd_probe.py` decodes of the HKGEN_PROBE rungs |
