# 2026-10-29 e15-02 GPU3: fragment-native dS (r9a) and the generated dK/dV loop (r10a, r10b)

Question: after the ABC skeleton (results/20261029b), how far does re-scheduling r8a's own loop structure go when
the whole dK/dV loop is generated (static VGPR plan, WAR-V waits, our own schedule)? Answer: **bitwise == r9a at
every hkl.test shape, -3.5% kernel cycles, but only -1.1% bwd time at prod**: the kernel is power-capped, and the
remaining per-step overhead (~340 cycles over the 960-cycle WMMA floor) sits in the softmax / LDS region.

All on e15-02 GPU3 (container lhz-hkl-g3), toy shapes first, timing with hkl.ladder (cold L2, settled clock,
null control); per-step cycles from the SALU probe (micro.k3_probe --abc, workgroup (0, 0, 0)).

## 1. r9a: fragment-native dS layout (dslayout v2)

`patches/r9a.patch` (r8a tree): dS chunk (kv, c) at `(c>>1)*512 + ((c&1)*16 + kv)*16`, so k_dkdv96ds stores each dK
operand's two 16-B halves at `lane*16` / `512 + lane*16` (no DPP regroup: -24 DPP -24 moves per iteration);
k_dqds_c's tr16 lane addresses follow (`g=(lc/8)*16+lr -> (g/4)*80+(g%4)*16`, `ds_half(qh)=qh*640`).
Bitwise == r8a at every shape. Time-neutral: 1.000x +- 0.002 (cycles 0.994x): the DPP was hidden under the dO^T tr16
latency. Side rungs r9b (dO^T burst between hh0's and hh1's S/dP, 0.993x) and r9c (split-barrier window, 0.989x):
LLVM's own drains eat the change.

## 2. r10a: the generated loop (r8gen)

`patches/r10a.patch` = r9a's tree + the `r8gen/` package (scratch_b1/r8gen): a HIP marker shell
(`k_dkdv96ds.hip -DHKGEN=1`, markers INIT / LOOP / FINAL carrying `nkv=3` / `nkv=2`) and `gen/r8gen.py` +
`gen/build.py`, which emit the whole tile loop of both block heights (96-row blocks and the 64-row tail): K / V
fragments, the 3-stage TDM ring with its SALU descriptor math, the per-tile scalars, the unrolled-by-3 loop with
masked (M) and full (U) bodies, the epilogue; asmkit derives MSBs, expert waits (rule `warv`) and v_nops, and the
waits are chained across the loop's blocks by a fixed point. One correctness fix on the way: tile 0's NL/ND soffset
(pair qp_start, not pair 0).

| version (b2 sq=skv=4224, no tail blocks) | time vs r9a | cycles vs r9a |
|---|---|---|
| per-step markers (shell code between steps) | 0.928x | 0.855x |
| one LOOP marker (all scalar work generated) | 0.988x | 0.977x |
| + drain grouping, MIN_XDL_AFTER | 1.004x | |
| + slot scheduler (WMMA windows, LDS queue) | 1.006x | 1.033x |

With the NKV=2 tail generated too, r10a passes **every shape bitwise == r9a**; prod 1.004x +- 0.004 time,
1.027x cycles. rocprofv3 kernel time (hot, hkl.one at 4224): k_dkdv96ds 1584.8 vs 1623.9 us (0.976x) -- the hot,
short run sits at a higher clock than the ladder's settled one.

## 3. r10b: re-planned register roles

`patches/r10b.patch` (same package, newer `r8gen.py`): bf16 P / dS written in place of hh0's raw S / dP
(P[kh] = R0[2kh], dS[kh] = R0[2kh+1]: hh0's dwords 0..3, hh1's 4..7), which frees six tuples for dedicated dO^T
operands (D[0..7], loadable any time in the step); Q^T 0..9 into the hh1 operand slots once hh1's S/dP ended, Q^T
10, 11 into D[0], D[1] once dV read them; hh1's S/dP kh-major (P[0] ready early); readback in first-use order; one
VALU drain per step (dS^T stores + NL/ND loads behind the whole softmax); K / V waited before the loop; the ring
signal floats until its LDS reads are done (bar_gap 12).

- Bitwise == r9a at every shape (toy, s640, rect32, b1_s4096, prod).
- prod: r10b 2.3907 ms vs r10a 2.4064 (-0.66%), 2.951 vs 3.002 Mcycles.
- probe: ~1303 cycles per step vs r10a 1436.
- Variants (same package, generator flags): r10c (dO^T loads after every hh1 operand load) 1.00x, r10d (hh1 dt
  outer for dt 0..2, kh-major after) 0.997x, r10e (hh1 dt outer) 0.997x of r10b's time: within 0.3%.

## 4. Where r10b's per-step time goes

Ablations (timing only, garbage numerics; rungs r10y_*, cycles per step at 4224):

| variant | cycles / step | vs r10b |
|---|---|---|
| r10b_pr | 1303 | |
| no softmax VALU | 1042 | -261 |
| no LDS loads | 1101 | -202 |
| neither | 1025 | -278 |
| no TDM in the loop | 1278 | -25 |
| no NL/ND loads, no dS^T stores | 1323 | ~0 |
| none of the four | 942 | WMMA floor 960 |

Timeline (rungs r10b_pa .. ph: probe points every 5-10 WMMAs): W0-W20 run at 8.1-8.3 cycles per WMMA; the excess
is in W20-W80 (hh1's S/dP, softmax, dV), 11-15 cycles per WMMA. The generated code there has 10-14-VALU bursts in
single windows and WMMAs waiting on hh1's operand loads (`s_wait_dscnt` before most of W29-W39).

## 5. Issue costs (micro/wmma_mix, expert mode, 4 waves per WGP)

Compile with `-mllvm -amdgpu-expert-scheduling-mode` (as the kernels): without it the LDS patterns run 4.6x
slower (WL 37 instead of 8 cycles). Extra cycles of one WMMA window (8 cycles) by contents, P = v_pk_fma_f32 /
v_pk_mul_f32, E = v_exp_f32, C = v_cvt_pk_bf16_f32, L = ds_load_b128 (tr16 the same):

| contents | extra | contents | extra | contents | extra |
|---|---|---|---|---|---|
| E | 0 | C | 0 | P | 1.0 |
| EE | 0.11 | CC | 0.08 | PP | 1.05 |
| EEE | 2.16 | CCC | 0.19 | PPP | 1.64 |
| EC | 0.08 | CCCC | 0.22 | PE / PEE / PPE | 1.28 / 1.41 / 1.41 |
| ECC | 0.42 | CCCCC | 2.2 | PC / PCC / PPC | 1.03 / 1.14 / 1.39 |
| EEC | 2.36 | L, LE, LC | 0 | LP / LPP / LPPE | 1.0 / 1.27 / 2.1 |

Dependent WMMA chains (one accumulator, or 2..4 interleaved) issue every 8 cycles. `s_set_vgpr_msb` costs nothing
(r10b has 242 per step) and does **not** wait for XCNT == 0: buffer_load + MSB costs 0.3 cycles more than the load,
buffer_load + `s_wait_xcnt 0` 46. asmkit had assumed the implicit wait; `asmkit.emit.MSB_CLEARS_XCNT = False` now
(8 more `s_wait_xcnt 0`, all where the VMEM op is hundreds of cycles old). LLVM itself leaves 522 SGPR
redefinitions inside an XCNT window in r9a's k_dkdv96ds (no XNACK replay here: SGPR sources are read at issue).
Lower bound for the softmax from the table: 72 P + 48 E + 48 C per step pack into 36 PP + 48 EC windows, ~45
extra cycles; r10b spends ~260.

## 6. Incident

06:40 UTC: a replay microbenchmark (one generated loop block in an inline-asm loop, VGPR count patched to 1024)
page-faulted (no-retry fault, queue killed, card not hung): its asm clobbered v0..v255, so the compiler parked
threadIdx.x in v256, which the replayed block overwrote. logs/GPU3.FAULT written; GPU work stopped until the user
clears it. Fixes: the replay kernel keeps no VGPR value across the asm (wave id via readfirstlane into an SGPR,
SGPR-only output address) and its build refuses any compiler code touching v15+ after the first asm; gen/build.py
now also fails when the shell's own code anywhere in the function reaches the generated base (today v14 < v15).

## 7. Next

A window scheduler driven by the table above (`bwdgen.schedule_win`: VALU paired, E with C, one LDS op per window,
pressure-aware budget) and a replay tool to measure one loop block's cycles directly, then a schedule that keeps
the softmax at ~1 extra cycle per window.
