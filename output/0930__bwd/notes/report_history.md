# bwd levers already tried, classified (2026-09-30)

A copy is saved at /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0930__bwd/notes/levers-tried.md.

No GPU was used for this. It comes from reading files and from stdlib python3 run over the ATT csvs.

Path aliases:
- `$O` = /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output
- `$BH` = /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/.claude/skills/gfx1250-attn-campaign/references/bwd-history.md
- `$H` = $O/0923__flydsl/hint.md
- `$BHINT` = $O/0927__b0/bwd-hint.md
- `$R` = $O/0927__b0/bwd/rounds
- `$KDQ` = $O/0927__b0/lab-kdq/REPORT.md
- `$RF` = $O/0930__roofline/REPORT.md
- `$ATT` = $O/0930__bwd/probe (waves on CU 1, prod shape)

Class codes:
- **CM**: closed by measurement
- **CR**: closed by a roadblock (API, VGPR, compiler or wedge)
- **O**: open
- **RE**: reopened by new evidence

Absolute numbers from B0 and from A0 before the 09-29 reflash cannot be compared with A0 numbers from today.

## 0. New evidence that changes old closures (A0 after the reflash)

| fact | number | source |
|---|---|---|
| r20, r19h and r29 are now within 0.3% of each other at prod | 6.571 / 6.575 / 6.583 ms; ASM 5.503 ms | $O/0928__a0_repro/REPORT.md:164-174 |
| The B0 win for u2n did **not** carry over. r29's k_dq is **3.3% slower** than r20's | 2.290 vs 2.217 ms | same file :176-182 |
| A WMMA↔DS switch costs about 29 cycles per wave. Only a second wave on the same SIMD hides it. k_dkdv has 7 switches per iteration, about 203 cycles | per-iteration model: 512 WMMA + ~450 issue + ~200 switch + ~600 memory wait = 1766 | $RF:18,129-143 |
| ATT k_dkdv: the `s_wait_dscnt 0x6` at vaddr 18872 comes right after 10 `ds_load_tr16_b128` | 706,737 stall / 3780 hits = **187 cycles per iteration** | $ATT/p2_r29/stats_ui_output_agent_27913_dispatch_10.csv |
| ATT k_dkdv: `s_wait_loadcnt 0x23` at vaddr 17140 | **133 cycles per iteration** | same csv |
| ATT k_dkdv: hot loop totals | latency 1937 per iteration (752 stall, 236 idle); 4.94 VALU per WMMA, above the 4 that are free | same csv |
| ATT k_dqg: top stalls | `s_wait_loadcnt 0x20` 204 cycles/iter, `s_wait_xcnt 0x2` 109 cycles/iter | $ATT/p2_r29/..._dispatch_11.csv |
| ATT ASM uses `buffer_atomic_add_f32 ... scope:SCOPE_DEV` (vaddr 35852) | The hardware supports device scope on buffer atomics; FlyDSL's `BufferAtomicAdd` emitting SCOPE_CU is an API gap | $ATT/p2_asm/..._dispatch_11.csv |
| ATT ASM atomics wait 2.26e6 before issue, out of 6.90e6 total latency | The 4 waves per WG hide this; `s_wait_tensorcnt` is about 19% of latency | same csv; P1-RESULTS.md:20 |
| Profiling tools on the new firmware | ATT now captures FlyDSL JIT kernels. `--kernel-trace` still returns 0 rows. PC sampling is not tested yet | $O/0930__bwd/probe/P1-RESULTS.md:5-12 |

## 1. Lever table

| # | lever | tried as | outcome | evidence | class |
|---|---|---|---|---|---|
| 1 | Prefetch Q/dO one iteration ahead | g21 | deleting it (g60) cost **-17.27%** | $BH:57,74 | CM (kept) |
| 2 | k_dkdv prefetch depth 1→2 | g62 (r20) | +1.65% on old A0. On B0, r19 (depth 1) was 1.7% faster (h67). r26 armA was killed at a free gate. Reflashed A0: r20 = r19h | $BH:75,78; $BHINT:4506; $R/026/act.yaml:203 | CM (net null; the champion uses depth 1) |
| 3 | Unroll qloop_full x2 to remove the loop-head drain | h60 / g72 | **-4.52%** | $R/024/act.yaml:99-104 | CM |
| 4 | k_dq prefetch 1→2 | g63 | **-19.33%** | $BH:76 | CM |
| 5 | Spread the loads (k_dq g66, k_dkdv g68) | | -21.93% / **-30.0%** | $BH:76-77 | CM |
| 6 | Move the prefetch after barrier-2 (4-wave S1) | | 0.995x; cover actually fell from 389 to 132 | $BHINT:4412 | CM |
| 7 | Reverse the q scan (g77) | | **-1.9%** | progress.md:9; $BHINT:4493 | CM |
| 8 | `sched_barrier` / `sched_group_barrier` | g47, u2nb/u2b/u2fb, c_ku2b | 0 wins, 5 losses on A0. u2nb 0.9545. Lab: op +8.0%, k_dq +20.7%; c_ku2b +8.7% | $BH:70,119; $R/029/act.yaml:113; $KDQ:41-46 | CM |
| 9 | Double-buffered LDS rings to get loads hoisted | g93 | 0.9370. The ISA matched g94 and `s_wait_dscnt 0x0` stayed: the third proof that LLVM ignores these hints | $R/032/act.yaml:97-107 | CR (compiler) |
| 10 | XCD remap | g43 | null; the part has one device-wide L2 | $BH:68 | CM |
| 11 | LDS port pressure | P3 deletes all 80 LDS ops; g39 segment split | P3 **-8.19%**; g39 null | $BH:62,114 | CM for bandwidth |
| 12 | LDS latency and WMMA↔DS switches in k_dkdv | g95 (P/dS carried across the back edge) | g95 **-3.1%**. Model: cutting 7 switches to 2 saves about 8%. ATT shows 187 cycles/iter at one dscnt wait | $R/032/act.yaml:119; $RF:151; ATT | **RE**. The order has to be fixed in the source, because hints do not work (#9) |
| 13 | Q/dO LDS round trip | P1 probe | -6.98% | $BH:115 | CM |
| 14 | Single-wave occupancy / BLOCK_KV | g55; g14 at 982 VGPR | 2.76x slower | $BH:117 | CM |
| 15 | `waves_per_eu` / `maxnreg` | corpus | -32% / -5x | $BH:124 | CM |
| 16 | k_dq at 2 waves/SIMD | o1 (BQ=32, no prefetch, 445 VGPR), g4o | k_dq +41%; op **+14.7% / +13.5%**. K/V traffic doubled | $KDQ:138,210,228 | CM for this form. **RE** for the mechanism: a second wave should hide ~800 cycles/iter ($RF:143), and a form with shared K/V was never tried |
| 17 | 4-wave BLOCK_KV=128 (w4) | S1, S2, X1, X2, nobar | w4 0.664x. On B0: X1 0.804, X2 0.872, nobar (illegal) **0.923** of champion | $BH:131-150; $R/024/act.yaml:105-122 | CM. w4 still ran **1 wave per SIMD**, so switch hiding was never tested |
| 18 | Barrier census | h59 | `fx.barrier()` drains loadcnt and dscnt to 0. The w4 R1 region has 40 tr16 loads and 1 WMMA, so no cover. aiter keeps 16-55 WMMA per region and signals with a partial dscnt | BARRIER-CENSUS.md:28-40; $BHINT:4354-4398 | CM. Recipe for any multi-wave build: raw signal/wait + partial dscnt + double-buffered rings |
| 19 | WMMA operand reuse (g59) | | +0.08% | $BH:74 | CM |
| 20 | Chain split (g61) | | +0.27% inside the floor; r26 armB 0.9978 | $BH:75; $R/026/act.yaml:166 | CM |
| 21 | Fewer issued instructions in k_dkdv: unroll (ku2, g94), constant hoist (g92) | | ku2 op **+6.5%**; g94 0.9377; g92 0.9662 | $KDQ:193; $R/032/act.yaml:108; $R/031/act.yaml:148 | CM |
| 22 | k_dq kv-loop unroll x2 (u2n / h70) | | B0: op **-3.06%**, shipped in r29. **Reflashed A0: k_dq +3.3%** | $KDQ:189; 0928 REPORT:180-182 | **RE** |
| 23 | `s_set_vgpr_msb` family | r11 | 0.929; caused a wedge | $BH:61 | CR (wedge) |
| 24 | Instruction scheduling / source reorder | r10, g22, g65 | 0.959; ISA identical | $BH:60,122 | CM |
| 25 | **FUSED5**: dQ inside 1-wave k_dkdv with fp32 SCOPE_DEV atomics | h58, lab3 | see the note below this table | FUSED5-RESULT.md:8-16,59-67; $BHINT:4473 | CM for 1-wave BLOCK_KV=32, and the 4-wave gate fails (x=30% vs 7.6%). Not reopened |
| 26 | f5-res (K^T kept in VGPRs) | | 1024 VGPR, 29 spills, 120 B scratch | $BHINT:4412 | CR (VGPR) |
| 27 | Deterministic fusion in any geometry; single-wave BLOCK_KV=128; 8-wave WG | h40, h30 | fails bandwidth or VGPR (1416 > 1024) | $BH:102-106 | CR (VGPR) |
| 28 | Write dS to HBM (P76 / g76) | | saves 1.956 ms but the round trip costs 1.886 ms; margin 0.07 ms | $R/026/act.yaml:177-187 | CM |
| 29 | k_dq GQA-grouped WG (g4 / g2) | | k_dq +7.4% / +3.4%; op +2.1% / +0.8% | $KDQ:208-209 | CM |
| 30 | k_dq epilogue through LDS (c_epi, g89) | | 0.999; g89 -0.2% | $KDQ:45; $R/030/act.md:226-249 | CM |
| 31 | dK/dV epilogue through LDS (g86) | | +0.35% | $R/029/act.yaml:96 | CM (shipped) |
| 32 | Split-K: g42/g44 + k_redsp, nsp_q g52 | | wins on fast only; prod runs nsp=1 | $BH:35,68-72 | CM |
| 33 | Split-K thresholds at proxy (g87) | | 0.816-0.938, all losses | $R/028/act.yaml:138 | CM |
| 34 | **TDM in k_dkdv** | h23, h47-49, g56 | see the note below this table | $H:1704-1745; $BHINT:4140 | CR/CM for the current design. **O** only as part of an aiter-style redesign where the GEMM reads B from LDS (ASM: 0 `buffer_load`, TDM) |
| 35 | Delta fusion (n1d / g4d, into the k_dq prologue) | | op -0.3 to +0.7%, null | $KDQ:206-207,225 | CM. Fusing into the k_dkdv prologue was never built, so that is **O**, capped at 1.6% (0.18e6 / 11.44e6 cycles, $RF:126) |
| 36 | VF / fma softmax in k_dq | q1, u2, c_u2f, g91 | q1 alone **-2.37%** op; on top of u2n only -1.19%; c_u2f +2.0% (loss); g91 0.994. `fx.fma` lowers to `v_pk_fma_f32` | $KDQ:43,190-191; $R/031/act.yaml:137,172 | CM |
| 37 | VF in k_dkdv | k1, c_k1, c_k1f | +0.17%, null up to +1.1% | $KDQ:48-49,203 | CM. Dormant bug: `VF_KV=True` drops the scale on dK (flag is off) — $R/031/act.yaml:258-262 |
| 38 | exp count / f16 exp | | exp ceiling 1.4% of k_dkdv; `v_exp_f16` is 2.4x slower | $R/032/opt.md:88-97 | CM |
| 39 | Causal skip, longest-first dispatch (g03/g40), LDS padding (g16), launcher cache | r1-r12 | all shipped | $BH:51-62 | CM (shipped) |
| 40 | h33 clamps a/b/c | | 0.9999, shipped in r19h/r29 | $R/024/act.yaml:94 | CM (shipped) |
| 41 | g73, g82, g83, aiter MFMA shape per GEMM | | died at free ISA gates | $R/026/act.yaml:188-257 | CM |

**Note on #25 (FUSED5):** the full op ran at **0.391x** the champion (21.9 vs 8.57 ms), though correctness was perfect.
- Adding the dQ GEMM alone (P1, no atomics) made k_dkdv 30-33% slower: 7.00 vs 5.27-5.39 ms.
- With atomics the kernel took 21.7 ms. That adds 14.6 ms, **3.3x** the 4.43 ms the same address stream takes when run alone.
- Atomic bandwidth itself was fine: 15.6 TB/s.
- Two causes are suspected but neither was measured: 162 `s_wait_xcnt` at 1 wave/SIMD, and atomics delaying the prefetch returns.
- The atomic payload is 69 GB, 3.95x aiter's.
- ASM also stalls on atomics and hides it with 4 waves, which a 1-wave kernel cannot do.

**Note on #34 (TDM):** the "no target" claim still holds for the current design.
- All 32 `buffer_load_b128` feed both LDS and the WMMA register operands (g09), and K/V are already hoisted.
- g56 (async global→LDS for LSE/delta) measured **-6.84%**.
- TDM only lowers through aiter's shim, with pad set explicitly.

## 2. What is open or reopened, ranked by size

1. **Memory wait, the largest item.** The model puts it at about 600 cycles/iter in k_dkdv; ATT shows 133/iter at the main loadcnt wait, and 204/iter in k_dqg. Every single-wave reordering lost (#3-6, #9). Only a second wave or a TDM-style redesign can cover it.
2. **WMMA↔DS switch latency (#12).** 187 cycles/iter at one dscnt wait. The order has to be written into the source, because LLVM does not act on hints.
3. **2 waves per SIMD (#16).** Needs 512 VGPR or fewer per wave; k_dkdv uses 729 and k_dqg 991.
4. **k_dq variant at A0 clocks (#22).** The r20-style k_dq should be A/B'd against k_dqg (u2n) on A0 now.
5. **Delta fused into k_dkdv (#35).** Worth 1.6% at most.
6. **The 7-vs-5 GEMM structure (1.386x, $BH:84-106).** Stays out of reach without a multi-wave atomic design.

## 3. Correctness gate

| gate | rule | source |
|---|---|---|
| compile | `COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250`, fresh cache. **0 spill, 0 scratch.** r29 reference: k_dkdv 729 VGPR / 70656 B LDS; k_dqg 991 VGPR | SKILL.md rule 4; tools/arm.sh:15-18 |
| bounds | CPU proof for every new index. Remainders predicated, no tail loops. No fake `1<<30` descriptors in k_dkdv; don't "fix" k_dq's fake descriptors naively. V# is in 128 B units | $BH:180-182 |
| coverage | NaN-poisoned allocator; `isfinite` checked before SQNR | validation.py:11-14,176 |
| precision | dq, dk, dv each **≥ 50 dB** vs cached eager, on every shape | validation.py:42 |
| dk/dv | **bitwise across 200 runs** | validation.py:15,44,246-252 |
| dq | run-to-run **≥ 70 dB**; atomics are allowed | validation.py:199,221-226 |
| refcache | never recompute the fp32 reference on the card (the 09-22 wedge) | lab_validate.py:22-25 |
| speed (job) | proxy **and** prod each ≥ the in-process beat (h69); `min_gain` 0.007 | $BHINT:4530,4471 |

The validation.py cited is the lab-kdq copy at $O/0927__b0/lab-kdq/oe/artifacts/job/job_context/op/validation.py.

## 4. Harness on A0

- **Arms** go in `$O/0930__bwd/arms/<name>`. r29 has md5 c6b95c2c and is taken from `0927__b0/champions/bwd_r29_r19h_u2n`; `r29_aa` is a byte copy for A/A. Its `_env.py` pins flydsl 0.3.2.
- **`tools/arm.sh <tag> <arms...>`** runs: compile gate → `lab_validate.py fast 3` (serialized) → `lab_validate.py prod 3` → a prod blocked bench with r29, r29_aa and asm in one process.
  - **It only does 3 determinism runs. The 200-run dk/dv bitwise gate has to be run separately before promotion.**
- **Validator:** `$O/0927__b0/lab-kdq/OP/lab_validate.py SHAPE NRUNS LABEL=/abs ...`.
- **Benchmark:** `$O/0927__b0/ruler/bwd/oe/artifacts/job/job_context/op/benchmark.py` (beat md5 2802a918). Blocked 9 + lead 4, palindromic, 54 iterations, L2 flush; A/A ±0.05%.
  - Baseline in `runs/base_prod_o1.log`: r29 6.5787 ms, r29_aa 6.5777 ms, asm 5.5023 ms, sclk 1920-2050 MHz.
- **Per-kernel timing:** `$O/0928__a0_repro/tools/kbench.py prod blk` (dq alone, then the full op). `gb` mode gives the training clock.
- **`tools/run.sh`** does the following:
  - takes the flock on `/tmp/a0-gpu0.lock` and refuses to start if any KFD process exists;
  - gives each process a fresh JIT cache (h72);
  - sets the image BLAS env;
  - samples sclk and fclk;
  - classifies the dmesg delta and exits 9 on GPU lines;
  - runs in `fa-repro`.
- **Rules:**
  - One shape per process.
  - The decision threshold is 0.5%; the same-session floor is 0.24-0.66%.
  - The clock depends on the arm (1760 / 1790 / 1818 MHz in r32 C1), so read sclk next to every time.
  - Cycles come from PMC `GRBM_GUI_ACTIVE`/8 with `--warmup-seconds 0`.
  - ATT needs `--att-library-path .../_rocm_sdk_devel/lib --att-target-cu 1 --kernel-include-regex`.