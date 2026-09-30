# bwd levers already tried (A0 r0-r23, B0 r24-r32 + labs) -- classified 2026-09-30

Path aliases: `$O`=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output, `$BH`=.claude/skills/gfx1250-attn-campaign/references/bwd-history.md,
`$H`=$O/0923__flydsl/hint.md, `$BHINT`=$O/0927__b0/bwd-hint.md, `$R`=$O/0927__b0/bwd/rounds, `$KDQ`=$O/0927__b0/lab-kdq/REPORT.md,
`$RF`=$O/0930__roofline/REPORT.md, `$ATT`=$O/0930__bwd/probe (attsum.py output, CU-1 waves, prod).
Sign convention: "op +x%" = slower; "-x%" in a TF/s ratio = loss. B0 and pre-09-29 A0 absolute numbers are not comparable with today's A0.

Classes: **CM** closed-by-measurement / **CR** closed-by-roadblock (API, VGPR, wedge, compiler) / **O** open / **RE** reopened-by-new-evidence.

## 0. New evidence that re-prices old closures (09-29/30, A0 reflashed)

| fact | number | source |
|---|---|---|
| A0 reflash: r20 / r19h / r29 now within 0.3% of each other at prod | 6.571 / 6.575 / 6.583 ms, ASM 5.503 (r29 = 83.5%) | $O/0928__a0_repro/REPORT.md:164-174 |
| B0 wins do NOT reproduce on reflashed A0: r29's k_dq (u2n) is **+3.3% slower** than r20's k_dq | 2.290 vs 2.217 ms; op-minus-k_dq 4.255 vs 4.286 | $O/0928__a0_repro/REPORT.md:176-182 |
| WMMA<->DS switch ~29 cyc/wave, hidden only by a 2nd wave on the SIMD; k_dkdv 7 switches/iter = ~203 cyc | per-iter model WMMA 512 + issue ~450 + switch ~200 + mem wait ~600 = 1766 | $RF:18,129-143 |
| ~4 VALU / ~2 v_exp free per WMMA, then ~1 cyc each | | $RF:19 |
| ATT k_dkdv (r29): `s_wait_dscnt 0x6` (vaddr 18872) right after a 10x `ds_load_tr16_b128` burst: stall 706,737 / 3780 hits = **187 cyc/iter**; `s_wait_loadcnt 0x23` (17140) 502,310 / 3780 = **133 cyc/iter**; hot loop 1937 lat/iter of which stall 752, idle 236; VALU 4.94 per WMMA | $ATT/p2_r29/stats_ui_output_agent_27913_dispatch_10.csv; P1-RESULTS.md:18 |
| ATT k_dqg (r29): 21.7 lat/WMMA; top stalls `s_wait_loadcnt 0x20` 207,607 / 1018 hits (**204 cyc/iter**), `s_wait_xcnt 0x2` 110,970 (109/iter) | $ATT/p2_r29/..._dispatch_11.csv |
| ATT ASM main: 20.7 lat/WMMA; `s_wait_tensorcnt` ~19% of latency, `s_barrier_wait` ~5%; dq via **`buffer_atomic_add_f32 ... scope:SCOPE_DEV`** (so the buffer form CAN be SCOPE_DEV -- FlyDSL's `BufferAtomicAdd` emitting SCOPE_CU is an API gap, not HW); atomic issue idle 2.26e6 of 6.90e6 total latency, covered by 4 waves/WG | $ATT/p2_asm/stats_ui_output_agent_29978_dispatch_11.csv (vaddr 35852) |
| Profilers on new FW: ATT works for FlyDSL JIT (was "captures nothing"); `--kernel-trace` still 0 rows (rocprofv3 1.3.2 bug); PMC ok; rocprof-compute not installed; PC sampling untested (last, toy only) | $O/0930__bwd/probe/P1-RESULTS.md:5-12 |

## 1. The lever table

| # | lever | tried as | measured outcome | evidence | class |
|---|---|---|---|---|---|
| 1 | Q/dO prefetch 1 iter ahead (k_dkdv) | g21 (r7) | +9.3% round; deleting it (g60) **-17.27%** | $BH:57,74 | CM (keep) |
| 2 | k_dkdv prefetch depth 1->2 | g62 (r20) | +1.65% on old A0 (VGPR 724->912); g71 removal -1.74% (r23); but on B0 r19 (depth 1) **1.7% faster** than r20 (h67), r26 armA (g62+clamps) killed at free gate; reflashed A0: r20 = r19h within 0.1% | $BH:75,78; $BHINT:4506-4516; $R/026/act.yaml:199-206; 0928 REPORT:168-170 | CM (net null; champion lineage is depth 1) |
| 3 | g62 loop-head drain (67 rotation v_mov) -> unroll qloop_full x2 | h60 / g72 (r24) | **-4.52%** prod vs 0.16% floor, passed all free gates | $R/024/act.yaml:99-104; $BHINT:4399 | CM |
| 4 | k_dq prefetch 1->2 | g63 (r21) | **-19.33%** | $BH:76 | CM |
| 5 | spread loads (k_dq g66 / k_dkdv g68) | r21/r22 | -21.93% / **-30.0%** | $BH:76-77 | CM |
| 6 | prefetch position after barrier (4-wave S1) | h50/S1 | 0.995x; cover actually fell 389->132 (h61 correction) | $BH:144; $BHINT:4412 | CM |
| 7 | reversed q scan (L2 phase align) | g77 (r25) | **-1.9%** (blocked re-check 1.0192x time) | $O/0927__b0/bwd/progress.md:9; $BHINT:4493-4505 | CM |
| 8 | `sched_barrier` / `sched_group_barrier` | g47 (r15) 0.969, g51 one boundary (+), u2nb/u2b/u2fb (r29, lab) | 0 wins / 5 losses on A0; u2nb prod 0.9545 (r29 act); lab c_u2nb op +8.0%, k_dq +20.7%; c_ku2b +8.7% | $BH:70,119; $R/029/act.yaml:113-116; $KDQ:41-46 | CM |
| 9 | LLVM scheduling hints generally (double-buffer LDS rings to hoist loads) | g93 (r32) | 0.9370; ISA structurally identical to g94, `s_wait_dscnt 0x0` NOT removed: "third independent proof LLVM will not hoist these loads for a scheduling hint" | $R/032/act.yaml:97-107 | CR (compiler) |
| 10 | XCD remap | g43 (r13) | null (one device-wide L2) | $BH:68,121 | CM |
| 11 | LDS port pressure | P3 probe (delete all 80 LDS ops) | **-8.19%** (deletion lost) ; g39 LDS-segment split null (r12) | $BH:62,114 | CM for bandwidth (k_dkdv needs ~90 B/clk of the 256 B/clk segment [INF from $RF:17]); see #12 |
| 12 | LDS latency / WMMA<->DS switches in k_dkdv | never targeted directly; g95 (P/dS transposed operands carried across back edge, removes `s_wait_dscnt 0x0`) | g95 **-3.1%** (0.9688, r32); roofline says 7->2 switches saves ~145 cyc/iter (~8%) | $R/032/act.yaml:119-125; $RF:151-152; ATT dscnt 0x6 187 cyc/iter | **RE** (new cost model + ATT; but #9 says LLVM resists reordering -> needs explicit program order in source or inline-asm-level control) |
| 13 | Q/dO LDS round trip | P1 probe | -6.98% | $BH:115 | CM |
| 14 | occupancy, single-wave k_dkdv BLOCK_KV | g55 census; g14 982 VGPR (spill 0) | 2.76x slower | $BH:117 | CM |
| 15 | `waves_per_eu` / `maxnreg` | corpus | -32% / -5x | $BH:124 | CM (corpus, gfx950-ish prior) |
| 16 | k_dq at 2 waves/SIMD | lab o1 (BLOCK_Q=32, no prefetch, 445 VGPR), g4o | k_dq +41%, op **+14.7% / +13.5%** (K/V loads and staging per query double) | $KDQ:138,210-211,228; $BHINT:4552 | CM for that form; **RE** for the mechanism: $RF says a 2nd wave hides switch+wait (~800 cyc/iter, 45%) -- o1 paid 2x K/V traffic, not the wave; a 2-wave form with shared K/V (LDS-resident) is untested |
| 17 | 4-wave BLOCK_KV=128 k_dkdv (w4) | G1b/G2, S1, S2, nobar, X1, X2 | w4 0.664x (old A0); B0: X1 0.804, X2 0.872, nobar (illegal) **0.923** of champion; X1 +6.05% over w4 -> H-align | $BH:131-150; $R/024/act.yaml:105-122 | CM (barrier-free ceiling < champion). Note: w4 was still **1 wave/SIMD** (>512 VGPR), so it never tested switch hiding |
| 18 | barrier census | $O/0927__flydsl/bwd/BARRIER-CENSUS.md, h59 | `fx.barrier()` = fence release (loadcnt/dscnt 0 drain) + signal + wait + fence acquire; w4 R1 region = 40 tr16 loads + 1 WMMA, no cover; aiter keeps 16-55 WMMA per region, signals with dscnt 0x14/0x10 | BARRIER-CENSUS.md:28-40; $BHINT:4354-4398 | CM (diagnosis); recipe for any multi-wave build = raw `s_barrier_signal/wait` + partial dscnt + double-buffered rings (h59 "Consequence") |
| 19 | WMMA A-reuse | g59 | +0.08% | $BH:74 | CM |
| 20 | matrix ILP / chain split | g61 | +0.27% in 0.29% floor; r26 armB 0.9978 | $BH:75,112; $R/026/act.yaml:166-169 | CM |
| 21 | issue-count cuts in k_dkdv (unroll x2 = ku2/g94, hazard-anchor hoist g92) | lab ku2, r32 g94, r31 g92 | ku2 op **+6.5%** (638 VGPR, instr/iter 675->564); g94 **0.9377**; g92 **0.9662** | $KDQ:144,193; $R/032/act.yaml:108-118; $R/031/act.yaml:148-152 | CM ("improves every static metric and still lost") |
| 22 | k_dq kv-loop unroll x2 (u2n / h70) | lab + r29 | B0: op **-3.06%**, k_dq -6.1%, at training clock -3.2%; shipped in r29. **Reflashed A0: r29 k_dq +3.3% vs r20** | $KDQ:189,221; $BHINT:4539; 0928 REPORT:180-182 | **RE** (gain did not transfer; k_dqg 991 VGPR; worth an A/B of r20-style k_dq vs k_dqg at A0 clocks) |
| 23 | s_set_vgpr_msb bank-tax family | r11 g33-g37 | 0.929 round, 4th wedge | $BH:61 | CR (wedge) |
| 24 | instruction-level scheduling / source reordering | r10 g27-g32; g22, g65 | 0.959 round; ISA-identical | $BH:60,122 | CM |
| 25 | FUSED5: dQ in 1-wave k_dkdv with fp32 SCOPE_DEV atomics, k_dq deleted | h58, lab3 (B0) | full op **0.391x** (21.9 vs 8.57 ms). P1 (dQ GEMM, no atomics) k_dkdv **+30-33%** (7.00 vs 5.27-5.39 ms); P3 with atomics 21.7 ms: atomic extra 14.6 ms = **3.3x** the 4.43 ms the same address stream takes alone; P2 atomic BW 15.6 TB/s (0.92x stores). Suspects (unmeasured): 162 `s_wait_xcnt` exposing atomic issue back-pressure at 1 wave/SIMD; atomics delaying Q/dO prefetch returns. Payload 68.99 GB at BLOCK_KV=32 = 3.95x aiter | $O/0927__b0/lab3/FUSED5-RESULT.md:8-16,59-67; $BHINT:4291-4353,4473-4482 | CM for 1-wave BLOCK_KV=32 and for 4-wave gate (x=30% vs 7.6%). ATT now shows ASM also idles 2.26e6 on atomics but hides them with 4 waves -> atomics need multi-wave cover; not reopened for 1-wave |
| 26 | f5-res (K^T resident in VGPRs) | h58 | 1024 VGPR, 29 spill, 120 B scratch | $BHINT:4412-4430 | CR (VGPR/spill) |
| 27 | deterministic dQ fusion (any geometry), single-wave BLOCK_KV=128, 8-wave WG | h40, h30 | fails BW or VGPR (BLOCK_Q=512 needs 1416 > 1024 VGPR) | $BH:102-106 | CR (VGPR) |
| 28 | dS materialisation (bf16 dS to HBM, k_dq reads it) | P76 / g76 (r25-26) | saving 1.956 ms but dS round trip 17.18 GB = 1.886 ms -> margin 0.07 ms, killed at gate | $R/026/act.yaml:177-187 | CM (paper+probe) |
| 29 | k_dq GQA-grouped WG (4 or 2 q heads per kv head) | lab g4 / g2 | k_dq +7.4% / +3.4%; op +2.1% / +0.8% | $KDQ:137,208-209 | CM |
| 30 | k_dq epilogue staged through LDS (b16->b128) | lab c_epi; r30 g89 | 0.999 null; g89 **-0.2%** prod | $KDQ:45; $R/030/act.md:226-249 | CM |
| 31 | k_dkdv dK/dV epilogue through dead Q/dO ring | g86 | +0.35% / +0.37%, shipped in r29 | $R/029/act.yaml:96-99 | CM (shipped) |
| 32 | split-K dkdv q-axis (g42/g44 + k_redsp) and nsp_q (g52) | r13/r14/r17 | wins on fast only; prod takes nsp=1 / nsp_q=1 | $BH:68-72; $BH:35 | CM |
| 33 | split-K thresholds at proxy | g87 (r28) | kv1 0.8162, nsp4 0.9382, nsp8 0.8485, nspq2 0.9186 (all loss) | $R/028/act.yaml:138-141 | CM (prod unaffected by construction) |
| 34 | TDM in k_dkdv | h20-h23, h47-h49, g56 | "no target": all 32 `buffer_load_b128` in the hot body are dual-use (->LDS and ->WMMA A-operand regs, g09), K/V hoisted; g56 async global->LDS for LSE/delta **-6.84%**; TDM lowers via aiter shim only (FlyDSL `tdm_ops` cannot take a Fly shared view; must pass pad_interval/pad_amount) | $H:1704-1745; $BH:143; $BHINT:4140-4153 | CR/CM for the current register-operand design. **O** only inside an aiter-style redesign where S/P GEMM takes B from LDS (ASM: 0 buffer_load, TDM + tensorcnt waits ~19%) |
| 35 | delta (k_delta) fusion | lab n1d / g4d (into k_dq prologue, k_dq first) | op -0.3..+0.7% (null) | $KDQ:139,206-207,225 | CM (k_dq form). **O**: into k_dkdv prologue never built; ceiling k_delta 0.18e6 of 11.44e6 cyc = 1.6% ($RF:126) |
| 36 | VF / fma softmax, k_dq | q1 (VF_Q), u2 (q1+u2n), c_u2f (fma-only on u2n), g91 (exponent constant hoist) | q1 alone op **-2.37%**; stacked on u2n only -1.19% (worse than u2n alone); c_u2f **+2.0%** loss; g91 0.9940; C2: `fx.fma` lowers to packed `v_pk_fma_f32` | $KDQ:190-191,222-224; $KDQ:43; $R/031/act.yaml:137-141,172-178 | CM |
| 37 | VF / fma softmax, k_dkdv | k1 (VF_KV), c_k1/c_k1f, k_dkdv fma-softmax | +0.17% / null (1.0003-1.0115) | $KDQ:48-49,203; $BHINT:4573-4583 | CM. Latent bug: `VF_KV=True` drops `* scale` from ds_l with no compensating scale at dK store (flag is False) $R/031/act.yaml:258-262 |
| 38 | exp count / f16 exp | r32 C-analysis | exp ceiling ~32 cyc/iter = 1.4% of k_dkdv; `v_exp_f16` 2.4x worse than f32 | $R/032/opt.md:88-97 | CM |
| 39 | host launcher cache, longest-first dispatch (k_dkdv g03, k_dq g40), causal tile skip, LDS padding (g16) | r1-r12 | all shipped | $BH:51-62 | CM (shipped) |
| 40 | h33 defects a/b/c (clamps) | r24 g75, r19h | ISA-neutral, 0.9999 vs incumbent; shipped in r19h/r29 | $R/024/act.yaml:94-97; $BHINT:4517-4529 | CM (shipped, correctness) |
| 41 | g82 / g83 / g73 / aiter MFMA-shape-per-GEMM | r24/r26 | died at free ISA gates, zero card time | $R/026/act.yaml:188-257; $R/024/act.yaml:123-128 | CM (static) |

## 2. What the classification leaves OPEN / REOPENED (ranked by ATT/roofline size)

1. **Memory-wait on the prefetch (k_dkdv ~600 cyc/iter model; ATT loadcnt 0x23 = 133/iter; k_dqg loadcnt 0x20 = 204/iter)** -- all 1-wave reorderings lost (#3-#6, #9); only a second wave per SIMD or TDM-style async fill (#34 redesign) can cover it.
2. **WMMA<->DS switch / LDS latency (#12)**: 187 cyc/iter exposed at one `s_wait_dscnt 0x6`; switch model 203/iter. Needs explicit program order (LLVM ignores hints, #9) -- e.g. issue the 10 tr16 loads one phase earlier in source.
3. **2 waves/SIMD (#16)**: requires <=512 VGPR per wave (k_dkdv 729, k_dqg 991). o1's loss is attributed to doubled K/V traffic, not to the second wave.
4. **k_dq variant choice at A0 clocks (#22)**: u2n's B0 gain reversed on reflashed A0 (+3.3% k_dq).
5. **Delta into k_dkdv (#35)**: <=1.6%.
6. Parity structure (7 vs 5 GEMM, 1.386x, $BH:84-106) stays unreachable without a multi-wave atomic design (#17, #25).

## 3. Correctness gate (what a new arm must pass)

| gate | rule | source |
|---|---|---|
| compile | `COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250 FLYDSL_RUNTIME_CACHE_DIR=<fresh>`; **0 spill, 0 scratch** (spill hangs the card); record VGPR/LDS (r29: k_dkdv 729 VGPR / 70656 B LDS; k_dqg 991) | SKILL.md rule 4; $O/0930__bwd/tools/arm.sh:15-18 |
| bounds | CPU bounds proof of every new index expression; remainders predicated, never tail loops; never convert k_dkdv descriptors to fake `1<<30`; never "fix" k_dq's fake descriptors naively; V# `num_records` in 128 B units | SKILL.md rule 5; $BH:180-182 |
| coverage | NaN-poisoned allocator, `isfinite` coverage checked BEFORE SQNR | validation.py:11-14,176 (lab-kdq copy) |
| precision | dq, dk, dv each **>= 50 dB** vs cached `op/eager` reference, every shape (champion 52.5-52.8) | validation.py:42 `GATE_DB = 50.0` |
| determinism dk/dv | **bitwise identical across 200 consecutive runs** (job: fast shape) | validation.py:15,44 `DETERMINISM_RUNS = 200`, :246-252 |
| determinism dq | run-to-run SQNR **>= 70 dB** (aiter 113 dB fast / 98 dB prod) -- fp32 atomics allowed | validation.py:199 `DQ_STABILITY_DB = 70.0`, :221-226 |
| refcache | never recompute the fp32 reference on the card (09-22 MES wedge); lab accepts the recorded `common_sha` | lab_validate.py:22-25; $BHINT:4530-4538 |
| speed (job) | proxy AND prod each >= in-process beat (h69); fast reported only; promotion `min_gain` 0.007 | $BHINT:4530; $BHINT:4458-4472 |
| non-causal | unscored and not bounds-proven for fused variants | $BHINT:4340 |

## 4. Harness facts on A0 (to validate an arm today)

- Arms live under `$O/0930__bwd/arms/<name>` (copy of r29 = `0927__b0/champions/bwd_r29_r19h_u2n`, md5 c6b95c2c; `r29_aa` byte copy for A/A). `_env.py` pins flydsl 0.3.2 (`~/.local/flydsl032`) and asserts version and `__file__`.
- `tools/arm.sh <tag> <arm>...`: compile gate -> `lab_validate.py fast 3` with `AMD_SERIALIZE_KERNEL=3` -> `lab_validate.py prod 3` -> prod blocked benchmark with r29 + r29_aa + asm in one process. **Note: it runs only 3 determinism repeats; the 200-run dk/dv bitwise gate must be run separately before promotion** (arm.sh:21-26).
- Validator: `$O/0927__b0/lab-kdq/OP/lab_validate.py SHAPE NRUNS LABEL=/abs/impl ...` (one shape per process; also bitwise vs first arm).
- Benchmark: `$O/0927__b0/ruler/bwd/oe/artifacts/job/job_context/op/benchmark.py` (beat = `op/beat`, md5 2802a918), blocked 9 + lead 4, palindromic, 51->54 iters, 3 s warmup, L2 flush; A/A +-0.05%. Baseline log: `$O/0930__bwd/runs/base_prod_o1.log` (r29 6.5787, r29_aa 6.5777, asm 5.5023 ms, sclk 1920-2050).
- Per-kernel timing: `$O/0928__a0_repro/tools/kbench.py SHAPE blk ITERS ...` (dq alone, then full op); `gb` mode = GEMM-lowered training clock (use the image hipBLASLt lib, h73).
- Card wrapper `tools/run.sh`: flock `/tmp/a0-gpu0.lock`, refuses if `/sys/class/kfd/kfd/proc` non-empty, fresh `FLYDSL_RUNTIME_CACHE_DIR` per process (h72 JIT-cache hazard), image BLAS env assigned, sclk/fclk sampler, dmesg delta classified (exit 9 on GPU lines), container `fa-repro`, `HIP_VISIBLE_DEVICES=0`.
- Rules: one shape per process (h31, ~44% fault rate otherwise); judge threshold 0.5% (PLAN D4) vs same-session floor 0.24-0.66%; clock is arm-dependent (r32 C1: 1760/1790/1818 MHz per arm) -> read sclk beside time; cycles via PMC `GRBM_GUI_ACTIVE`/8 with `--warmup-seconds 0` ($RF §6/§8); ATT needs `--att-library-path .../_rocm_sdk_devel/lib --att-target-cu 1 --kernel-include-regex` (P1-RESULTS.md:10).
