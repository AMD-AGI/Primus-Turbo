# 2026-10-29 e15-02 GPU3: ABC (cross-tile pipelining of the dK/dV loop) skeleton, WAR-V gate, issue model

Question: does the ABC software pipeline of the dK/dV full loop -- B(k-1) softmax / dS and C(k-2) dV / dK in the
first half of step k, the ring barrier, A(k) S / dP in the second half -- beat r8a's loop in cycles? Answer
measured here: **not within the register budget**. Under LLVM's va_vdst rule it is 3% slower than r8a; with
WAR-V-by-class waits it is still slower; even with C operand slots borrowed from registers the real kernel does not
have, it is only ~7% fewer cycles than r8a (about 2.7% time at the measured ~0.38 cycle-to-time conversion).
Along the way: the WAR-V-by-class wait rule passed its GPU gate, and a clean WMMA / VALU issue model.

All on e15-02 GPU3 (container lhz-hkl-g3), toy shapes first; per-step cycles from s_get_shader_cycles in the
kernel (SALU only; workgroup (0, 0, 0)).

## 1. abc0: the ABC loop in HIP (rung abc0, side)

`patches/abc0.patch` (r8a tree + `HKL_ABC`): one loop of steps k = 0 .. N+1 unrolled x2 (P / dS register sets
alternate), uniform branches pick the first-half variant (masked B, B only, C only, steady), a 4-stage Q / dO ring
(LDS 86016 B), N + 1 barriers per wave; `asm volatile` markers per variant keep SimplifyCFG from merging the
variants' shared code (without them B and C land in different blocks).

- Correct: every hkl.test shape (toy, s640, rect32, b1_s4096, prod) PASS, deterministic, **bitwise == r8a**.
- Slow: prod 3.794 ms vs r8a 2.415 ms (0.636x), 6.76 vs 3.07 Mcycles. LLVM puts every C / A operand into the
  tuple the previous WMMAs just read, so each load needs `s_wait_alu depctr_va_vdst(0)` and the loop serialises as
  `[va0] load [ds0] W W W`. Interleaving B into C, separate loops, or bursts of C operands all spill (the live set
  is ~1000 of 1024 VGPRs).

## 2. abcsk: a generated skeleton (rungs abcsk_l / abcsk_w / abcsk_wp / abcsk_c*)

`patches/abcsk.patch` adds the `abcgen/` package to the tree: a marker shell (`hip/src/k_dkdv96ds.hip -DHKGEN=1`:
TDM descriptors, ring, loop, probes; the 96-row MHA blocks only) and `gen/bwdgen.py` (register layout above the
shell's v0..v9, one STEP region per step parity, list scheduler) + `gen/build.py` (asmkit.emit derives MSB / waits
/ v_nops, splice, assemble, verify_obj, contract checks). Built through hkl.build's generator path
(`"gen_kernel": "k_dkdv96ds"`). Timing only: every step is a steady step, numerics garbage, no dK / dV.

| rung | waits | C operand slots | cycles / step toy | prod |
|---|---|---|---|---|
| r8a (r8a_pr probe) | LLVM | (bursts) | -- | 1532 - 1593 |
| abcsk_l | LLVM | 5 | 1607 | 1608 - 1641 |
| abcsk_w | WAR-V by class | 5 | 1580 | 1601 |
| abcsk_c8 | WAR-V | 8 (3 borrowed) | 1474 | |
| abcsk_c11 | WAR-V | 11 (6 borrowed) | 1420 | |

Phase probe (abcsk_wp, toy, cycles / step): start -> barrier signal (B + most of C) 800, barrier wait 75, C tail
128, A (60 WMMA) 492. A runs at 8.2 cycles / WMMA; C does not: its tr16 operands are loaded at the step start into
5 rotating slots, so the WMMAs wait on LDS latency. With 11 slots C gets ~120 cycles faster, but the 6 extra
tuples do not exist in the real kernel (acc 480 + K/V 240 + raw 96 + P / dS 96 + NL/ND 32 + dS^T 24 = 968).
Remaining losses even then: C's start-up latency (operands loaded at the step start), 24 DPP of the dS^T regroup
(not hidden by WMMAs, ~2.5 cycles each), uneven VALU per WMMA, the barrier.

## 3. WAR-V-by-class gate (the user's condition for per-class va_vdst waits)

The rule: a memory op that overwrites registers in-flight WMMAs read may issue after va_vdst(N), N = WMMAs issued
after the newest such reader, even with other VALU classes in flight (LLVM: N = 0 then); every other hazard keeps
LLVM's rule (`tools/warv_regen.py`; abcgen's asmkit `--rule warv`).

- GPU stress (`micro/warv_stress_gen.py` -> `micro/warv_stress.cpp`, 480 configs: b128 / tr16 overwriting a WMMA
  A or B operand, 0..6 younger WMMAs, VALU / TRANS / DPP fillers, dependent chains / independent backlogs in front
  of X; waits = the rule, k+1, k+2, none): **rule 0 mismatching lane-trips in 4.7e11**; controls 20384 mismatches
  (X behind an 8-long dependent WMMA chain, overwrite with no wait: the hazard is real and the detector sees it).
  Without the backlog configs no control ever failed (a DS load lands long after an undelayed WMMA has read its
  operands). Log: `20261029_035143_warv_stress_long.log` (and `..._035121_warv_stress2.log`).
- Independent static check (`tools/warv_regen.check_warv`, its own tokenizer and in-flight model, over STEP0 +
  STEP1 twice): abcsk_w 0 violations, abcsk_l 0; with every va_vdst stripped 160 violations.
- In the abcsk schedules the warv rule changes only WAR-V waits (same code object as per-class-for-everything).

## 4. Issue model (micro/wmma_mix_gen.py, one wave per SIMD, expert mode, lock_simd)

Shader cycles; W = v_wmma_f32_16x16x32_bf16 (independent accumulators), w = one dependent chain, P = v_pk_fma_f32,
E = v_exp_f32, D = v_mov_b32_dpp, M = v_mov_b32 (`20261029_040537_wmma_mix7.log`):

| pattern | cycles / WMMA | | pattern | cycles / instruction |
|---|---|---|---|---|
| W | 8.00 | | P / F / C / M bursts | 1.08 |
| WP / WPP / WPPP / WPPPP | 9.00 / 9.05 / 9.64 / 10.16 | | E / D bursts | 2.07 |
| WE / WEE / WEEE | 8.00 / 8.11 / 10.16 | | | |
| WPE / WPPE / WPEE | 9.28 / 9.41 / 9.41 | | | |
| WPPEE / WPPPPEE | 12.11 / 13.47 | | | |
| WDD / WDDD | 13.0 / 15.0 | | | |
| WWWWPPPP | 8.50 | | | |
| wP / wPP / wPPE | 14.0 | | | |

- A WMMA issues every 8 cycles; up to ~3 core VALU + 1 TRANS (or 1 VALU + 2 TRANS) right behind each WMMA are
  nearly free; DPP is not hidden (~2.5 cycles each); a dependent WMMA with anything between it and its producer
  costs ~14 cycles.
- Caveat found on the way: an earlier build let the compiler unroll the timing loop 8x around the asm body, which
  made long bodies look instruction-fetch bound (~1 byte / cycle); with `#pragma unroll 1` 16 KB bodies run at full
  speed (`..._040515_wmma_mix6.log`). Results before that fix are not used.

## Conclusion and next

ABC as a full generated loop is not worth its cost now: within the register budget the skeleton does not beat r8a,
and the best borrowed-register variant is ~7% fewer cycles (~2.7% time). r8a's own losses per iteration are the
softmax tail after the last S/dP WMMA (~30 VALU), pack + 24 DPP of the dS^T regroup (~60 cycles, DPP not hidden),
export store waits and the dK kh0 WMMAs waiting on Q^T tr16 loads. Cheaper levers that follow: the fragment-native
dS layout (no DPP regroup; dq's tr16 addresses change with it) and an r8a schedule that keeps <= 3 VALU + 1 TRANS
per WMMA.
