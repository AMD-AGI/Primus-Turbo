# ARM s3_dqtrim: ATT-guided trim of s3's k_dqg hot loop (two IR-order changes, bitwise)

Base: `arms/s3`. `impl.py`, `_env.py` and `__init__.py` are md5-identical to s3.
`k_dkdv` is not touched: its ISA is byte-identical to s3 (721 VGPR).
All changes are in `kernels.py` `_body` (k_dqg), behind two switches:

| switch | arm | off | what it does |
|---|---|---|---|
| `DQ_DQORD` | `"q2"` | `"d"` | Body 2 of the tailpf U2 trip (`nxt_in` set) issues its 16 `ds_load_tr16` (8 b_k) first. It then runs its 32 dQ WMMAs **qh-major** (for qh: for dtile) instead of dtile-major. `"q"` applies this to every body. |
| `DQ_HEADST` | `True` | `False` | In body 1 of the tailpf trip (`pf2` set), the 8 kt-0 K `ds_store`s are emitted before the kt-0 S/dP WMMAs, followed by `sched_barrier(0x406)`, instead of one dt at a time. |

With both switches off, the k_dqg ISA is byte-identical to s3 (compile-verified, `.var_off/`).

## 1. Where s3's k_dqg trip goes (ATT, dispatch 11, per-wave timelines)

Tool: `att_wall.py`. It computes wall(i) = issue(i+1) - issue(i), averaged per trip.
- Output: `att_s3_dqg_wall_fast.txt` (12 fast waves, 2906 cyc/trip) and `att_s3_dqg_wall_slow.txt` (wv0/4/8/12, 2998 cyc/trip).
- The loop is `.LBB0_4`, Vaddr 10636-19036: 1007 instructions, 192 WMMA, 1018 trips.
- The WMMA floor is 1536 cyc/trip. Measured is 2906, i.e. 15.1 cyc per WMMA in the loop. The whole kernel is 20.1 because of the epilogue and prologue.

Wall cycles per trip by class (fast waves):

| class | n | wall |
|---|---:|---:|
| WMMA | 192 | 1241 |
| v_pk | 384 | 577 |
| v_exp | 128 | 275 |
| s_clause (VMEM issue / switch stalls) | 21 | 150 |
| ds_store | 32 | 106 |
| v_cvt_pk_bf16 | 64 | 105 |
| buffer_load | 64 | 100 |
| v_nop | 43 | 98 |
| s_wait_loadcnt | 23 | 71 |
| ds_load_tr16 | 32 | 65 |
| SALU | 8 | 60 |
| s_wait_dscnt | 15 | 56 |

The VALU count is algorithmic and must stay bitwise: 3 v_pk, 1 exp and 0.5 cvt per score element. Only its **placement** can change.

Top exposed WMMA-free stretches (wall, fast waves):
1. **Body-2 dS tail**, Vaddr 16948→18496: **277 cyc**, 201 instructions. These are 86 pk_mul, 38 pk_add, 40 exp, 32 cvt and 4 ds_load_tr, with no WMMA among them. It is exposed because all 32 of body 2's dQ WMMAs are dtile-major, so the first dQ WMMA already needs `a_ds` of all 4 qh.
2. **Body-2 dQ tail**, 18496→19036: **540 cyc for 32 WMMA** (256 floor). It is `W W W W | ds_load_tr x2 | wait` repeated. Five WMMA→ds_load_tr switches cost 30-31 cyc each. The third WMMA of each group adds a further 15, the switch ramp. This matches REPORT §3's `8·nW + 29` fit.
3. Body-1 kt1 VALU plus the pf2 kt-1 clump plus 16 ds_load_tr, 13884→14596: 211 cyc.
4. Loop head: `s_wait_loadcnt 0x1e` (24.8, memory latency on the oldest pf2 pair, about 2350 cyc lead). Then 1 WMMA, then the 32-load head clump. The `s_clause` right after the WMMA takes 25 (switch). Then 4 ds_store, a WMMA, 4 ds_store. The `s_cmp` after that WMMA takes 29 (a second switch). The head totals 215 cyc to the third WMMA.

In total, s3 has 11 WMMA→DS/VMEM switches per trip.

Why the remaining `s_wait_loadcnt 0x1e` is not a target: it waits for the first 2 loads of the pf2 set issued ~2350 cyc earlier (dqg_ts design). A longer lead needs registers, and VGPR is 1016/1024. The slow-wave excess is now only +92 cyc/trip (VMEM issue +53, ds_store +26). It was +613 in r29.

## 2. The changes

- **`DQ_DQORD="q2"` (body 2 only).**
  - With qh-major order, qh 0's 8 dQ WMMAs need only `a_ds[0]`. The scheduler now issues the dS VALU of qh 1..3 in the shadow of those WMMAs.
  - All 8 b_k are live at once, 64 VGPR. They are loaded in 3 bursts inside body 2's S/dP phase, not inside the dQ tail.
  - Applying it to body 1 as well (`"q"`) is slightly worse (model -138 vs -153): body 1's dQ already overlaps body 2's S/dP.
- **`DQ_HEADST`.**
  - The 8 K ds_stores of body 1 kt 0 now sit directly behind the 32-load head clump, in one memory phase.
  - This removes the switch after W1 (29 cyc). The loop-head memory wait (now `0x3e`) comes after the clump issue instead of before it.

Both changes only reorder IR. Every LDS address expression, WMMA operand, loop bound, soffset/voffset and tailpf fence is unchanged.

Rejected: `B2ST`, where body 1 stores body 2's K right behind its own b_k reads plus a fence.
- With the fence it spills: 2 VGPR (74 with HEADST) at 1024 VGPR.
- Without the fence, the scheduler moves the stores back and the ISA is identical to q2/hs.
- The code was removed. Its dumps remain in `.var_b2*` (their kernels.py still has the switch).

## 3. ISA evidence (`isa_gate.py` → `isa_gate.log`; `ablation.log`)

Compile line: `tools/compile.sh arms/s3_dqtrim dkdv dqg`
- k_dkdv: VGPR 721, 0 spill, 0 scratch, ISA == s3.
- k_dqg: **VGPR 1020 / 1024, 0 spill, 0 scratch**, LDS 8704.

k_dqg hot loop `.LBB0_4`:

| | s3 | s3_dqtrim |
|---|---:|---:|
| instructions | 1007 | 1013 |
| WMMA / v_pk / v_exp / cvt | 192 / 384 / 128 / 64 | 192 / 384 / 128 / 64 (unchanged) |
| buffer_load / ds_store / ds_load_tr16 | 64 / 32 / 32 | 64 / 32 / 32 (unchanged) |
| v_nop | 43 | 50 |
| s_wait_loadcnt / dscnt | 23 / 15 | 22 / 16 |
| **WMMA→DS/VMEM switches** | **11** | **7** |
| **longest WMMA-free run** (model cyc, instrs) | **262, 201** (body-2 dS tail) | **172, 123** (body-1 kt1, unchanged in kind) |
| last 40 WMMAs: VALU / ds_load_tr interleaved | **0 / 7** | **193 / 0** |
| loop-head order | wait, W, 32 VMEM, 4 DS, W, 4 DS | 32 VMEM, wait, 8 DS, W... |

The rest of the kernel (prologue and mask loop `.LBB0_8`) has the same opcode sequence as s3. Only VGPR numbering differs.

## 4. Expected saving

The issue model is in `isa_gate.py`:
- WMMA every 8 cyc.
- VALU costs 1.6 under a busy WMMA and 1 otherwise; exp costs 2.4 / 2.
- WMMA→DS/VMEM switch is 29, from the REPORT §3 fit.

Calibrated on s3's ATT, it gives 2556 model cyc vs 2906 measured (0.880). The residual is spread evenly, except the VMEM backpressure at the head (+86).

| variant | VGPR | model cyc/trip | Δ |
|---|---:|---:|---:|
| s3 (`d`, HEADST off) | 1016 | 2556 | 0 |
| HEADST only | 1016 | 2535 | -21 |
| q2 only | 1020 | 2404 | -153 |
| **q2 + HEADST (arm)** | **1020** | **2382** | **-175 (-6.8%)** |

Scaled to measured cycles, the expectation is **about -175 to -200 cyc/trip, from 2906 to about 2710**.
- The loop is 88% of k_dqg wave time. So **k_dqg ≈ -5.5 to -6%, from 3.29e6 to about 3.10e6 cyc**.
- Nearly all of the gain is the body-2 dS tail (277 → hidden in the dQ shadow) and the removed dQ-tail switches, 5 × ~30.

End-to-end effect is between 0 and about -2%: k_dqg runs on the side stream concurrently with k_dkdv (6.06e6). At most about 0.1 ms comes off 5.769 ms, if the freed CU time is used by k_dkdv. Otherwise the gain is hidden.

The model ignores VALU→WMMA operand latencies; the compiler added 7 v_nops for them, and those are counted. Only ATT on the card can confirm the number.

## 5. Bitwise and bounds

- **Bitwise vs s3.** Each dQ accumulator still receives exactly one WMMA per body, with the same A (`a_ds[qh]`) and B (`b_k[dtile]`) and in the same body order. The LDS bytes each ds_load_tr reads come from the same (body, kt, dt, u, lane) store. The VALU dataflow is unchanged.
- **`bounds_proof.py` → `bounds_proof.log`** replays the program-order LDS/WMMA events of each body type in s3 and arm order: B1 = tailpf body 1, B2 = tailpf body 2, M = mask body. It asserts:
  - an identical writer per read chunk;
  - every chunk read was written by the same body (so induction over any trip sequence holds);
  - all chunks lie inside 8704 B;
  - an identical dQ operand map.
- It also replays `_dqg_impl`'s trip control for prod, fast, toy, gqa4_small and unequal_seqlen_2 × causal 0/1. The body-type sequences are unchanged; no loop bound was touched.
- impl.py dispatches k_dqg only at prod. The other four shapes take `k_dq_sp`, which is unchanged.
- No TDM/tensorcnt in k_dqg. No VMEM load moved. The loadcnt waits are compiler-generated for the new order: `0x3e`, then `0x3b..0x32` at the head.

## Files
- `kernels.py`: the arm.
- `bounds_proof.py/.log`, `isa_gate.py/.log`, `ablation.log`.
- `att_wall.py`, `att_s3_dqg_wall_{fast,slow}.txt`: the s3 ATT attribution.
- `.var_*/`: compile-only ablation dumps. They are root-owned (container) and could not be deleted without the container.
