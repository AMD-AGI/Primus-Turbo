# arm w4f: 4-wave k_dkdv that also computes dQ (5 GEMMs, fp32 SCOPE_DEV atomics)

Base: `arms/s1` (r29 + dqg_tailpf + dkdv_trorder + side-stream dQ). flydsl **0.3.2**, `_env.py` unchanged.
The Q/dO path reuses **dkdv_tdm3's** 3-slot TDM ring and its "read the next step's S/dP B operands back into
carried VGPRs" structure, extended to 4 waves.

**Status: compiles cleanly (RC=0, 0 spill, 0 scratch), bounds/protocol proof ALL PASS, ISA-ordering checks
pass. It has never run on a GPU.** Nothing in this arm touched the card.

## Files

| file | what |
|---|---|
| `kernels.py` | s1 verbatim, plus the new section from line 1723: `_dkdv_w4f_impl`, `k_dkdv_w4f` / `k_dkdv_w4f_sp`, `launch_dkdv_w4f(_sp)`, `k_dq_cvt` / `launch_dq_cvt`. The only other edit is the `tdm_ops` import. The s1 kernels compile to byte-identical ISA (`k_dkdv`, `k_dqg` were `cmp`-checked against `arms/s1/.dump`) |
| `impl.py` | `W4F = True` routes `flydsl_attn_bwd` to `_w4f_bwd`: k_delta, then zeroed fp32 dq workspace, then k_dkdv_w4f (or `_sp` + k_redsp), then k_dq_cvt. `W4F = False` is the s1 path, unchanged. Env `W4F_NSP_FORCE=<int>` forces the split count |
| `compile_w4f.py`, `compile_w4f.sh` | a copy of `tools/compile_arm.py` plus the jobs `w4f`, `w4f_sp`, `cvt`, `redsp`. The `.sh` wraps the same `docker exec fa-repro` line as `tools/compile.sh` (COMPILE_ONLY=1, ARCH=FLYDSL_GPU_ARCH=gfx1250, HIP_VISIBLE_DEVICES=-1, fresh cache, FLYDSL_DUMP_IR=1) |
| `bounds_proof.py` | host python3, no torch. Log in `.bounds_proof.log` |
| `isa_stats.py` | per-loop ISA census and run-length event string |

## 1. Design

**Workgroup.** 4 waves (128 threads). grid = (Hkv[*nsp], Skv/128, B). The kv block is ascending on grid.y,
so the longest blocks dispatch first (g03). Wave w owns kv rows `[kvw0+32w, +32)`.
- It keeps its K and V fragments resident exactly as k_dkdv does (`kf`, `vf`: 128 VGPR).
- It also keeps 8 K^T fragments resident for dQ (`kt`: 64 VGPR). The prologue builds them by staging the
  workgroup's K `[128][272 B]` in LDS once (each wave writes its 32 rows), then one barrier, then 16 tr16 loads.

**Step order.** A step is one q pair (32 rows) × one q head. Steps run pair-outer, head-inner, exactly as in
k_dkdv, so dK/dV accumulate in the same order. The schedule is workgroup-uniform. `qp_start`, the masked-pair
count and the split chunks come from `kvw0` (the workgroup base), never from the wave's own kv0. Global step
`s` maps to `(qt, gh) = (p_begin + s//G, s%G)`, clamped. The masked loop and the full loop are two `scf.for`
loops over one global `s` with an identical carried state.

**One step s, per wave** (full loop `.LBB0_6`):

| phase | work | LDS / sync |
|---|---|---|
| A | LSE/delta for s+1: 4 `buffer_load_b32`, carried | – |
| B+C | 32 S^T/dP^T WMMAs on the **carried** Q/dO operands, with **1 dQ atomic of step s-1 per WMMA gap** (deferred one step) | none: the operands were read back in step s-1 |
| D | softmax (s1 arithmetic verbatim). P goes to the wave's **private** P tile; dS goes to its own 64 B column band of the **shared** dS buffer `s%2` (8 `ds_store_b128`, one burst) | – |
| E | `s_wait_dscnt 0`, `s_wait_tensorcnt 0`, **`s_barrier_signal -1`** | split barrier B_s |
| F | dV/dK: 40 tr16 + 32 WMMAs. TRORDER regions; A comes from the private P tile or the wave's own dS columns; B comes from ring slot `s%3` | the signal→wait gap holds **32 WMMAs** (ASM: 3-6) |
| G | **`s_barrier_wait -1`**, then TDM of step s+2 into slot `(s+2)%3`: 2 `tensor_load_to_lds` per wave, each wave copying rows [8w, 8w+8) of dO and of Q | – |
| H | one DS burst of 48 `ds_load_b128`: 16 are the dS A fragments (all 128 kv), 32 are the **readback of slot (s+1)%3** for step s+1. Then 16 dQ WMMAs (2 q halves × 2 d tiles, 4-deep chains over the 128 kv, B = resident K^T), carried to step s+1 | – |

**Why one barrier per step suffices** (the kernels.py section header gives the argument; the L section of the
proof checks it):
- **dS RAW.** Writers wait dscnt 0 before signal B_s; readers read after wait B_s.
- **dS WAR.** Buffer `s%2` is rewritten at step s+2, after wait B_{s+1}; all dQ_s reads retire before each
  reader's signal B_{s+1}.
- **Ring RAW.** TDM(s+1) is waited by its issuer before signal B_s; the readback of slot `(s+1)%3` comes after
  wait B_s.
- **Ring WAR.** TDM(s+2) targets slot `(s-1)%3`. It is issued after wait B_s, and every read of that slot
  (readback in step s-2, tr16 in step s-1) retired before the reader's signal.
- Every `s_wait_tensorcnt` has exactly one TDM batch outstanding, so **there is no in-order-TDM assumption**
  (dkdv_tdm3 needed one).

**dQ.**
- dq_acc is fp32 **[B, Hq, Sq, D]**, head-major, zeroed by `torch.zeros`.
- Wave w owns d columns [32w, 32w+32). Per step it issues 32 `global_atomic_add_f32 ... scope:SCOPE_DEV` (32q×32d).
  They use ONE VGPR offset plus immediate offsets 0/64/512/…/11840, the same form as the ASM (`W4_ATOM_IMM`:
  `llvm.atomicrmw fadd syncscope("agent")` on a per-step base pointer plus constant inbounds GEPs).
- They are deferred one step and spread one per S/dP WMMA gap (`W4_ATOM_SPREAD`: `sched_group_barrier`
  {1 WMMA, 1 VMEM-write} × 32).
- At step 0 the carried dQ is zero, and it is added to tile(0) (harmless).
- After the loop, the last step's dQ is flushed.
- `k_dq_cvt` converts the fp32 [B,Hq,Sq,D] workspace to bf16 [B,Sq,Hq,D] in one pass.
- k_dqg is no longer launched.

**LDS (165,888 B; ≤ 320 KB; 1 WG/CU, which the VGPR count forces anyway).**

| range | content |
|---|---|
| [0, 52224), segment 0 | Q/dO ring: 3 × 17408 |
| [65536, 82944), segment 1 | dS double buffer: 2 × [32][272 B] |
| [82944, 93184) | P tiles: 4 × 2560 |
| [131072, 165888) | K image (prologue only) |
| [0, 98304) after the epilogue barrier | epilogue staging: 4 × 24576 |

**Split-K** (`k_dkdv_w4f_sp`): the same split of the q pairs as `k_dkdv_sp`. The masked pairs stay on split 0.
dK/dV are written as fp32 partials and folded by `k_redsp`; dQ atomics are unaffected. The impl rule is
`nsp = min(16, smallest power of 2 with wgs*nsp >= 512)`, which gives prod 1, proxy 2, and fast/toy/gqa4_small 16.

## 2. Deviations from the ASM (report_asm.md) and why

| ASM | w4f | reason / next step |
|---|---|---|
| S = Q·K^T, dV^T = dO^T·P, dK^T = Q^T·dS; P/dS come straight from C registers (no P/dS LDS round trip) | s1 orientation: S^T = K·Q^T; P/dS go through LDS; dV/dK A operand via tr16 | Keeps dK/dV **bitwise vs s1** and reuses proven code. The flip is lever #1 in report_asm (−250 to −400 cyc/unit) and the planned next step |
| 3-stage cross-step software pipeline (softmax of tile t+1 under the dV/dK of tile t) | pipelining only via the readback (S/dP operands one step early) and the deferred atomics; softmax still sits between S/dP and dV/dK | First version. LLVM does not pipeline for us; it has to be written in source (the history says hints do not work) |
| 2 split barriers per step, 3-6 WMMAs between signal and wait | **1** split barrier per step, 32 WMMAs between signal and wait | Fewer rendezvous. The dS double buffer makes the second barrier unnecessary |
| TDM for Q, dO, LSE and delta | TDM for Q/dO (2 ops per wave per step, 8 rows each); LSE/delta are 4 `buffer_load_b32` per wave, carried one step | 4× redundant but tiny. Could move into the ring later |
| K layout 2 resident, 64 VGPR | same (`kt`, built once through LDS) | – |
| `buffer_atomic_add_f32` SCOPE_DEV | `global_atomic_add_f32` SCOPE_DEV, same offset form | FlyDSL's BufferAtomicAdd is SCOPE_CU (known trap) |
| dQ ×scale after the dQ WMMA (16 pk_mul) | scale folded into dS before bf16, as s1/k_dqg do | Same numerics as s1's dK path |
| near/far kv-block pairing (j, 63-j) per WG | one kv block per WG, longest-first dispatch | 8 dispatch rounds at prod; the census says greedy balancing is already ~100% |
| VGPR 1024 declared | 714 (sp: 706) | – |

## 3. ISA evidence

`compile_w4f.sh w4f w4f_sp cvt redsp delta` gives RC=0. The ISA is deterministic: two compiles `cmp` equal.

| kernel | VGPR | SGPR | spill / scratch | LDS B | WMMA (whole kernel) |
|---|--:|--:|---|--:|--:|
| k_dkdv_w4f (prod) | **714** | 94 | 0 / 0 | 165888 | 160 (2 loop bodies × 80) |
| k_dkdv_w4f_sp | 706 | 100 | 0 / 0 | 165888 | 160 |
| k_dq_cvt | 11 | – | 0 / 0 | 0 | – |
| k_redsp, k_delta | 65, 40 | – | 0 / 0 | 0 | – |

Per step, hot loop `.LBB0_6` (masked loop `.LBB0_2` in brackets). s1/r29 k_dkdv and dkdv_tdm3 are shown for
reference; those two do only 64 of the 80 WMMAs, and k_dqg does the dQ separately (plus a recompute of S/dP).

| per step | **w4f** | dkdv_tdm3 k_dkdv | s1 k_dkdv |
|---|--:|--:|--:|
| instructions | **758** (847) | 583 | 679 |
| v_wmma | **80** (80) | 64 | 64 |
| global_atomic_add_f32 SCOPE_DEV | **32**, one VGPR offset plus imm, 1 per WMMA gap | 0 | 0 |
| s_barrier_signal / s_barrier_wait | **1 / 1** | 0 | 0 |
| tensor_load_to_lds / s_wait_tensorcnt | 2 / 1 (0x0, just before the signal) | 2 / 1 | 0 / 0 |
| ds_load_b128 / ds_load_tr16 / ds_store_b128 | 48 / 40 / 8 | 32 / 40 / 8 | 0 / 40 / 40 |
| buffer_load b128 / b32 | 0 / 4 | 0 / 4 | 32 / 4 |
| s_wait_dscnt / s_wait_xcnt / s_wait_storecnt | 32 / 1 / **0** | 7 / 0 / 0 | 18 / 0 / 0 |
| s_set_vgpr_msb / v_nop | 155 / 22 | 104 / 23 | 139 / 51 |
| v_exp / pk_mul / pk_add / cvt_pk | 32 / 64 / 32 / 32 | | |

Event string for `.LBB0_6` (`isa_stats.py`; W = WMMA, A = atomic, L = ds_load, S = ds_store, T = TDM,
dN = dscnt wait):

```
d0x1e W G4 x0x0 A ld0x816 W A W A ... (32 x "W A") ... S8 d0x0 t0x0 |sig| L36 d0x20 W d0x1e W ... L4 ... W8 d0x0 W8
|wait| T2 L48 d0x2e W l0x0 W d0x2c W2 ... d0x20 W2
```

- **WMMA↔DS switches: 2 per step** (after the S/dP+softmax phase, and after the wait), plus the dV/dK-phase
  tr16→WMMA dependence.
- The dQ WMMAs wait only for the 16 A-fragment loads (thresholds 0x2e..0x20). The 32 readback loads stay in
  flight across the back edge; the loop top waits `d0x1e`, i.e. only for the first two.
- The whole kernel has 96 atomics (32 masked loop + 32 full loop + 32 flush), 4 barrier pairs (prologue,
  2 loops, epilogue), 8 TDM ops, 4 tensor waits, and **0 `s_wait_storecnt`**: the atomics are fire-and-forget.
- The atomic variants were tried at compile time only:

| variant | full / masked loop instr | s_wait_xcnt | atomic placement |
|---|---|--:|---|
| UniversalAtomicAdd (index VGPR per atomic) | 759 / 886 | 3 / 23 (the masked loop serialised on `s_wait_xcnt 0x0` after every atomic, the FUSED5 failure mode) | – |
| `W4_ATOM_IMM` | 712 / 820 | 0 | one clump of 32 |
| `+W4_ATOM_SPREAD` (**shipped**) | 758 / 847 | 1 | 1 per gap; +45 `s_set_vgpr_msb` |

Keep the IMM-only variant (`W4_ATOM_SPREAD=False`) as the first card A/B.

## 4. Expected numerics

- **dK/dV at prod (nsp=1): expected bitwise identical to s1.**
  - The same WMMAs run on the same bf16 operands in the same (qt, gh) order.
  - The extra steps a wave runs are steps where its own 32 rows are fully masked. There P = exp2(−inf) = 0 and
    dS = 0 exactly, so acc(+0) + (±0) = acc. The accumulator starts at +0 and stays +0 through those steps.
  - Steps that are "masked" at block level but unmasked for this wave's rows take the select path with a
    false predicate, which gives the identical value.
- **dK/dV at fast/toy/gqa4_small (split): deterministic but not bitwise vs s1.** The split boundaries differ
  (128-row WGs, a different nsp), so the fp32 partial grouping differs. Run-to-run they must be bitwise
  (no atomics touch dK/dV).
- **dQ: not bitwise vs s1.**
  - The kv summation order changes: 4-chain WMMAs over each 128-kv block, then fp32 atomics across blocks in
    arrival order.
  - At prod each dq element receives ≤ 64 atomic adds (aiter: 64). Expect SQNR vs eager in the same band as
    k_dqg (≈ 50+ dB) and run-to-run ≥ 70 dB.
  - The atomic payload is 17.45 GB at prod, identical to aiter.

## 5. Bounds / protocol proof (`python3 bounds_proof.py` → `RESULT: ALL PASS`, about 20 s)

Shapes covered, each causal and non-causal (sq_gt_skv non-causal only): prod, fast, toy, gqa4_small, proxy,
mha, unequal_seqlen, unequal_seqlen_2, sq_gt_skv.

- **G: global accesses in bounds.** Every workgroup, split, wave and step is enumerated. Covered: K/V
  fragments, LSE/delta, every TDM descriptor, every dQ atomic (including deferred, clamped and flush tiles),
  the bf16 and fp32-partial dK/dV stores, and k_dq_cvt (bijection proven exhaustively on the small shapes,
  corner-checked on prod).
  - Every TDM has extent Sq−r0 ≥ 8, so hardware zero-fill never triggers.
  - prod maxima equal the tensor sizes exactly: kv 33554432/33554432, TDM and atomics 134217728/134217728,
    lse 1048576/1048576.
- **A1: the 4 waves' atomic sets are disjoint.** The 4×32 lanes × 32 atomics of a step hit all 4096 (q, d)
  of the tile exactly once.
- **A2: the dQ contraction is consistent.** The A (dS) and B (K^T) fragments map the kv index identically,
  and the 128 kv are covered once.
- **A3: each causal (q, kv) term reaches dQ, dK and dV exactly once.**
  - Every (b, qh, q pair, kv block) is visited at most once over all WGs and splits, and every pair holding a
    causal term is visited.
  - Full-loop steps are entirely causal; masked-loop steps use the exact per-element predicate; unvisited
    pairs hold none.
  - prod causal: 1,064,960 steps (32,768 masked), N per WG in [16, 1024].
- **S: dK/dV rows are written exactly once** (per split for `_sp`).
- **L: LDS protocol.** A happens-before checker runs over every LDS/TDM op of all 4 waves for N = 0..14 steps.
  The address pattern is periodic in s mod 6, so larger N adds nothing new.
  - 38,688 conflicting op pairs, all ordered by a split barrier (retire-at-next-signal) or by in-order LDS
    within one wave.
  - Every ring read sees the TDM of its own step from all 4 waves; every dQ read sees the same step's dS from
    all 4 waves; all ranges are inside the allocation and their regions.
- **B: barriers.**
  - N is computed from workgroup-only inputs, so all 4 waves run prologue + N + epilogue barriers.
  - ISA CFG: 4 entry→s_endpgm paths. Each has exactly 2 non-loop barrier pairs plus 1 pair per loop trip.
    Every conditional branch is an SCC loop/skip branch; there are no EXEC branches.
- **I: ISA ordering**, per loop body in both kernels:
  - no DS load before the signal;
  - `s_wait_dscnt 0x0` and `s_wait_tensorcnt 0x0` after the last DS/TDM op and before the signal;
  - only tr16 between signal and wait;
  - after the wait, exactly 2 TDM ops and 48 `ds_load_b128`;
  - 32 SCOPE_DEV atomics.
- **Counter simulation** (loops iterated 3×, all paths):
  - every signal is reached with dscnt = tensorcnt = 0;
  - tensorcnt is 0 at s_endpgm;
  - maximum outstanding is ds 48, load 36, tensor 4 (all ≤ 63);
  - all 134 waits lie in reachable blocks.

**Premises, not proven here:**
- The tr16/WMMA fragment semantics, which are the ones s1 and dkdv_tdm3 are card-proven on.
- LDS ops of one wave execute in order.
- The gfx12 split-barrier semantics: a wave that has signalled B_s is released by B_s's completion even if
  faster waves have already signalled B_{s+1}. The ASM relies on the same, and S2 ran it on the card.

**Re-run `bounds_proof.py` after every recompile.** The I section is the only guard against LLVM moving an
LDS op across the raw barrier intrinsics, which carry no memory semantics.

## 6. First card run: staged plan (one shape per process, flocked `tools/run.sh`, card idle, dmesg checked after each)

1. **toy alone, prod kernel object:**
   `AMD_SERIALIZE_KERNEL=3 W4F_NSP_FORCE=1 tools/run.sh ... toycheck.py toy s1=arms/s1 w4f=arms/w4f`.
   - This is 1 WG and 8 masked steps; it exercises k_dkdv_w4f, the TDM ring, the barriers, the atomics and
     k_dq_cvt.
   - Pass: all outputs finite; dq/dk/dv ≥ 50 dB vs the CPU fp32 reference (not bitwise vs s1: s1 splits toy).
   - On any hang or GPU dmesg line: stop and follow gfx1250-card-safety. Do not retry.
2. **toy default** (nsp=16 → `k_dkdv_w4f_sp`; 15 of 16 WGs have N=0, which exercises the empty-loop path),
   then **gqa4_small**, each in its own process with `AMD_SERIALIZE_KERNEL=3`.
3. **unequal_seqlen_2, then fast** (serialised), causal and non-causal: `lab_validate.py fast 3`.
   - Gates: 3 outputs ≥ 50 dB vs the cached eager reference; dq run-to-run ≥ 70 dB; dk/dv run-to-run bitwise.
4. **prod `lab_validate.py prod 3`.** Additional prediction: **dk/dv bitwise equal to s1**. A mismatch means a
   schedule or layout bug, not a tolerance.
5. **Timing:**
   - A prod blocked bench in one process: s1, the dkdv_tdm3 stack (the current best, 5.856 ms), w4f, w4f with
     `W4_ATOM_SPREAD=False`, and asm.
   - `kbench.py prod blk` for the per-kernel split: w4f k_dkdv, plus memset and cvt, which should be ≲ 0.1 ms
     together.
   - The kill line from the coordinator is **w4f must beat about 5.6 ms**.
   - Take an ATT of k_dkdv_w4f and read the gap decomposition exactly as report_asm §10 does: atomic
     back-pressure, barrier, dscnt and tensor waits.
6. The **200-run dk/dv bitwise gate** runs before any promotion.

**Budget expectation:** unknown until measured.
- Floor: 80 WMMA × 8 = 640 cyc/step.
- Issue: 758 instructions per step, of which ~155 are `s_set_vgpr_msb` and ~200 VALU.
- The ASM pays 1581 (ATT) / 1853 (PMC) cyc/unit.
- To reach 5.6 ms, w4f has to land at about ≤ 2350 PMC cyc/unit including atomics, since k_dqg's ~0.97k is
  gone.

## 7. Risks for the first run

1. **Wedge by barrier mismatch.** If any wave skips or adds a barrier, the WG hangs. Mitigations: the trip
   counts are proven WG-uniform, the ISA CFG check passes, and the first run is toy alone.
2. **First multi-wave TDM in the bwd kernel.** The 8-row tiles are issued by 4 waves into one slot. The
   descriptor has the same encoding as dkdv_tdm/tdm3 except tile_dim1 = 8 (decoded from the LLVM IR: group1
   dw0 0x07510000 is identical, dw4 = 8). Multi-wave TDM is card-proven in the fwd kernel (num_warps=8).
3. **Raw split barrier without LLVM memory semantics.** Ordering is guaranteed today by
   `sched_barrier(0)` + explicit waits, and verified in the ISA. Any edit needs the proof's I section re-run.
4. **Atomic throughput under full load.** This is the ASM's largest loss (430 cyc/step, 27%).
   - Here it is 32 SCOPE_DEV atomics per wave-step, fire-and-forget, spread 1 per WMMA gap.
   - The FUSED5 1-wave failure (0.391×) is not directly comparable: that version had 4× the payload, 1 wave
     per SIMD with no barrier-mates, and `s_wait_xcnt` serialisation. That serialisation is gone in this ISA
     (1 xcnt wait per step).
5. **dQ workspace and cvt cost.** At prod the workspace is 512 MiB: one memset plus one read pass and a
   256 MiB bf16 write.
6. **One-step deferral at s = 0 adds zeros to tile(0).** This is numerically neutral; the addresses are
   proven in bounds.
7. **The masked loop is 847 instructions and has the mask VALU.** It covers 3% of the prod steps.
8. **Validation tooling.** `toycheck.py`/`lab_validate.py` load `impl.py` via `load_impl`. `W4F_NSP_FORCE` is
   read from the environment at import time.

## 8. Next optimisation steps (in the order the evidence suggests)

1. Card A/B `W4_ATOM_SPREAD` on/off. ATT the atomic gaps.
2. **GEMM flip (report_asm lever 1).** S = Q·K^T, then dV^T = dO^T·P and dK^T = Q^T·dS with P/dS from C
   registers. This removes the P tile, 8 tr16 and the private-P LDS traffic. dS still has to go to LDS for dQ.
3. **Cross-step pipelining in source.**
   - Move the 16 dQ WMMAs of step s into step s+1, between the S/dP WMMAs.
   - Put the softmax of step s+1 under the dV/dK WMMAs of step s.
   - Goal: no WMMA↔DS switch is exposed.
4. Remove the 4 runtime `//G` per step (dkdv_divfree).
5. Move LSE/delta into the TDM ring: one 128 B tile each per step, shared by the 4 waves.
6. Pair near/far kv blocks per WG (j, 63−j) to balance the tail, as the ASM does.
7. Fold the dq memset into k_delta, or zero it in a prologue.

## 9. Card round 1 and variants (added after the first card run)

**Card result for w4f** (as reported by the coordinator): all shapes pass. prod dq is 52.56 dB, dK/dV are
run-to-run bitwise, dq run-to-run is 100.8 dB. But prod takes **15.86 ms**, of which k_dkdv_w4f is
3.30e7 cycles.

The ATT shows 75.7% of latency is waits, and the largest single wait is
`s_wait_tensorcnt 0x0` before the signal, about 3750 cyc/step.

**Why the TDM lead is short.** With 3 slots, TDM(s+2) is issued after wait B_s and must be retired before
signal B_{s+1}. That gives it only about one step of lead. The loop wait at signal B_s already retires only
TDM(s+1), because TDM(s+2) has not been issued yet at that point, so relaxing the threshold alone cannot help.
The fix needs a deeper ring.

**Variants.** Each sibling directory is a full copy with one constant pair set in its `kernels.py`, both
defaulting to False:

| arm dir | `W4_RELAX` | `W4_ABL_NOATOM` | what it is |
|---|---|---|---|
| `arms/w4f` | False | False | the round-1 version |
| `arms/w4f_relax` | True | False | the fix. 4-slot ring; TDM(s+3) is issued after wait B_s. The loop wait before signal B_s is `s_wait_tensorcnt 0x2`: it retires only TDM(s+1), the slot read back next, and leaves TDM(s+2) in flight, about 2 steps of lead. The prologue issues TDM(0..2) and waits 0x4; the epilogue waits 0x0. LDS layout: ring [0, 69632), dS/P [131072, 158720), K image [196608, 231424); 231424 B in total. This relies on in-order TDM retirement, the dkdv_tdm/tdm3 premise |
| `arms/w4f_noatom` | False | True | ablation, **output wrong**. No atomics. The 16 dQ WMMAs stay; the last step's dQ is sunk into dV accumulator 0 |
| `arms/w4f_relax_noatom` | True | True | both |

**Compile results.** All compiled with each directory's own `compile_w4f.sh w4f w4f_sp cvt redsp delta`: RC=0,
0 spill, 0 scratch.

| arm | VGPR (w4f / sp) | LDS | full-loop instr | atomics / step | loop tensor wait | tensor_load (whole kernel) |
|---|---|--:|--:|--:|---|--:|
| w4f | 714 / 706 | 165888 | 758 | 32 | 0x0 | 8 |
| w4f_relax | 714 / 706 | 231424 | 758 | 32 | **0x2** | 10 |
| w4f_noatom | 708 / 700 | 165888 | 647 | 0 | 0x0 | 8 |
| w4f_relax_noatom | 708 / 700 | 231424 | 649 | 0 | 0x2 | 10 |

All four keep 80 WMMA per loop step.

**Proof.** `bounds_proof.py` reads `W4_RELAX` and `W4_ABL_NOATOM` from the `kernels.py` next to it. It gives
ALL PASS in all four directories.
- The L section now models each wave's TDM FIFO: at every signal, all but the newest `tw` ops retire.
- The ISA section expects `s_wait_tensorcnt 0x{TW_LOOP}` and 0 or 32 atomics accordingly.
- Negative controls fail as they should: relax with a loop wait of 0x4, or with the TDM lead raised to 4.

## 10. Round 2 (r2a / r2b / r2ab), all built on w4f_relax

**Card round 1b** (as reported by the coordinator), prod timings from one process:

| arm | ms |
|---|--:|
| w4f | 15.83 |
| w4f_relax | 9.53 |
| w4f_noatom | 4.34 |
| w4f_relax_noatom | 4.24 |
| s4 (1-wave champion) | 5.53 |
| ASM | 5.50 |

The fused structure itself is worth about 4.2 ms. The dQ atomics cost about 5.3 ms in w4f_relax.

**Diagnosis of the `s_wait_loadcnt 0x0` stall in w4f_relax** (ATT Vaddr 21852):
- The 4 LSE/delta `buffer_load_b32` of step s+1 are loop-carried. The register allocator copies them into the
  PHI registers with a `v_mov`, and that copy needs `loadcnt 0`.
- The wait therefore sits in the dQ phase, right after the 32 atomics have been issued, so the b32 returns
  queue behind them.
- r2a also hit this in its first build: LSE carried from LDS produced a `v_dual_mov` that forced `dscnt 0` in
  the dQ burst. Hence the final design reads LSE/delta at the step top, not carried.

**New constants in kernels.py** (all default False; the other dirs are unchanged). With all r2 flags off the new
code compiles to ISA byte-identical to the card-run w4f_relax, for both w4f and w4f_sp (checked with `cmp`).

| dir | flags | change |
|---|---|---|
| `arms/w4f_r2a` | RELAX, `W4_LDS_LSE` | LSE/delta ride the TDM ring. Each wave issues 2 more `tensor_load_to_lds` per step (8 LSE + 8 delta fp32 into slot +17408/+17536; slot = 17664 B), 4 ops per wave-step in total. Tensor waits: prologue 0x8, loop 0x4, epilogue 0x0. The step reads its own slot's LSE/delta at the step top with 2 `ds_load_2addr_b32`; they are not carried |
| `arms/w4f_r2b` | RELAX, `W4_XCD` | XCD-aware workgroup order. Linear id → XCD c = id % 8, slot = id // 8; within an XCD the slot walks kv block ascending (longest-first kept), then the XCD's gpx = B·Hkv/8 groups, then the split. group = c·gpx + gl → (bat, hkv). So every workgroup adding into one (b, hkv) group's dQ rows lands on one XCD. Active only when B·Hkv % 8 == 0 (prod: 4 groups/XCD, proxy: 1); identity elsewhere. SALU only, in the prologue |
| `arms/w4f_r2ab` | RELAX, LDS_LSE, XCD | both |

**(C) buffer_atomic form: not built.** I found no concrete reason the current form is worse.
- The current `global_atomic_add_f32 v_off, v_data, s[base] offset:imm scale_offset scope:SCOPE_DEV` already
  has the ASM's addressing structure: an SGPR base, one VGPR offset per step, and immediates.
- It is no-return, device scope, one per WMMA gap, with 1 VGPR offset.
- The buffer form would only add V# range checking.
- The measured loss is chip-wide atomic throughput and memory-return ordering, and the instruction form
  changes neither.

**Compile.** Run from each dir: `./compile_w4f.sh w4f w4f_sp cvt redsp delta` (same docker line as
`tools/compile.sh`). All give RC=0, 0 spill, 0 scratch, LDS 231424.

| dir | VGPR w4f / sp | SGPR | full loop instr | loop VMEM loads | TDM/step | loop tensor wait | atomics/step |
|---|---|--:|--:|--:|--:|---|--:|
| w4f_relax | 714 / 706 | 94 | 758 | 4 b32 | 2 | 0x2 | 32 |
| w4f_r2a | 706 / 698 | 78 | 737 | **0** | 4 | 0x4 | 32 |
| w4f_r2b | 724 / 706 | 94 | 763 | 4 b32 | 2 | 0x2 | 32 |
| w4f_r2ab | 716 / 698 | 84 | 735 | **0** | 4 | 0x4 | 32 |

- **r2a/r2ab event string**: `d0x1e W L2 A ... (W A)x32 ... S8 d0x0 t0x4 |sig| L36 ... W8 |wait| T4 L48 d0x2e W2 ... d0x20 W2`.
  - The dQ burst has only partial dscnt waits, and the readback stays in flight across the back edge.
  - The loop-top `s_wait_loadcnt 0xe..0x0` waits come from the compiler's model of the prologue's 16 V-fragment
    `buffer_load_b128` (first use in the dP WMMA). No VMEM is issued in the loop, so in steady state they are
    no-ops, provided the hardware does not count TDM on LOADcnt (dkdv_tdm3 has the same pattern).

**Proof.** `bounds_proof.py` reads the flags from the `kernels.py` next to it. It gives ALL PASS in w4f,
w4f_relax, w4f_noatom, w4f_relax_noatom, w4f_r2a, w4f_r2b and w4f_r2ab.
- The noatom dirs have no `.dump` now, so their ISA section is skipped.
- New checks:
  - LSE TDM extents are at least 8 and in bounds.
  - The step-top LSE read sees TDM(s) from all 4 waves.
  - The ISA shows exactly the 2addr b32 loads before the signal and no `buffer_load` in the loop.
  - X section: the remap is a bijection onto (bat, hkv·nsp+sp, blk) for every shape, one XCD per group, the
    group count per XCD is balanced, and the kv block is ascending within each XCD.
- Proof-model fix: a TDM batch now retires only when all of its ops have. The old model credited a partial
  wait with retiring a whole 4-op batch.
- Negative controls: r2a loop waits 0x5/0x6/0x8 fail; 0x4 passes.

## 11. Round 3: atomic footprint, ASM vs w4f, and arm w4f_r3

**Card results, round 2** (as reported by the coordinator), prod timings from one process:

| arm | ms |
|---|--:|
| w4f_r2a | 9.272 |
| w4f_r2b | 9.821 |
| w4f_r2ab | 9.397 |
| w4f_relax | 9.518 |
| s4 | 5.494 |
| ASM | 5.484 |

B (the XCD remap) is dropped.

**ASM per-instruction footprint.** Reconstructed from `probe/p2_asm/stats_ui_output_agent_29978_dispatch_11.csv`,
lines 244-251 and the 192 `buffer_atomic_add_f32` rows.
- **Offset registers.** `v56 = v0*4 + s2*0x80`, where v0 is the lane id and s2 the wave id. The other three
  offset registers are offsets of v56: `v60 = v56 + s63*16`, `v64 = v56 + s63*64`, `v68 = v60 + s63*64`.
- **Instruction pattern.** Each of the 4 offset registers carries 8 atomics, with imm offsets
  0/512/1024/1536/4096/4608/5120/5632. The data registers are v204..v235, which are the four dQ WMMA C tiles,
  unpermuted.
- **Lane → address.** Lane l writes base + 128·wave + 4·l. That is 32 lanes × 4 B = **one contiguous,
  128 B-aligned line per instruction** (4 sectors, fully written).
- **Walk.** The 32 instructions step through 512 B slots (slot 0..31 in order if s63 = 128).
- **Consequence for the workspace layout.** A C tile holds rows m = 8·(l/16) + si and columns n = l%16, yet the
  address is linear in the lane. So the ASM's dq_acc cannot be a plain [q][d] matrix. It is laid out in
  **C-fragment order**: one 512 B slot per accumulator register, and 4 waves × 128 B per slot. A per-(head,
  32-row block) chunk is 16 KB. The separate dq_convert kernel de-swizzles it.
- **Caveat.** The ATT disassembly hides the VGPR-MSB bank. I identified `v0` as the lane id because lines
  193-222 use v0 that way, after a bank-1 write to "v0" at line 171.

**w4f_r2a footprint.** From the source: `base = ((b·Hq+qh)·Sq + qt·32 + half·8)·D + 32w + row`, plus
`imm = (qh16·16+si)·512 + j·64`.
- Lanes 0-15 write 16 consecutive d (64 B) of q row r; lanes 16-31 write the same d range of row r+8.
- So each instruction touches **2 lines, each half-written** (4 sectors).
- The two j instructions hit the same two lines again, so every line of the tile gets 2 atomic instructions.

| per atomic instruction | ASM | w4f..r2a | **w4f_r3** |
|---|---|---|---|
| distinct 128 B lines | 1 | 2 | **1** |
| 32 B sectors | 4 | 4 | 4 |
| bytes of lines touched / payload | 1× | 2× | **1×** |
| line-level atomic requests per step per wave | 32 | 64 | **32** |
| offset form | 4 VGPRs + imm | 1 VGPR + imm (scale_offset) | 1 VGPR + imm 0..15872 step 512 |
| walk over 32 instructions | slots 0..31 of 512 B | rows 0-7/8-15/16-23/24-31, with d halves interleaved | slots 0..31 of 512 B (same as ASM) |

**Workspace layout.**
- Ours was [B,Hq,Sq,D] fp32: 512 B rows, and a q row's 32-d wave slice is one aligned line.
- The ASM's is in C-fragment order: 512 B register slots, the 4 waves' 128 B lines adjacent, one 16 KB block
  per (head, 32-q block).
- Both are the same size and 128 B-aligned. The difference is how one instruction's lanes map onto lines, not
  padding.

**w4f_r3** = w4f_r2a + `W4_DQ_TILED = True` (flags: RELAX, LDS_LSE, DQ_TILED).
- The workspace is the ASM's C-fragment order. Element (q, d) of pair qt goes to
  `block(b,qh,qt)·4096 + ((qh16·2+j)·8+si)·128 + 32·wave + lane`.
- The atomics use one VGPR offset plus imm 0..15872.
- `k_dq_cvt_t` de-swizzles to bf16 [B,Sq,Hq,D], one 16 B load and one 8 B store per thread.
- There is no permute and no extra VGPR. The atomic data is the unmodified C registers, as in the ASM.

**Compile.** `./compile_w4f.sh w4f w4f_sp cvt_t cvt redsp delta` gives RC=0, 0 spill, 0 scratch.

| kernel | VGPR | LDS B |
|---|--:|--:|
| k_dkdv_w4f | 704 | 231424 |
| k_dkdv_w4f_sp | 698 | 231424 |
| k_dq_cvt_t | 14 | 0 |

- The hot loop `.LBB0_9` has 734 instructions and 80 WMMA, and its event string is identical to r2a's.
- It has 32 `global_atomic_add_f32 v141, vX, s[44:45] offset:{0,512,...,15872} scale_offset scope:SCOPE_DEV`,
  one per WMMA gap, and no buffer_load.

**Proof** (`bounds_proof.py` in w4f_r3): ALL PASS.
- New **T section.**
  - The atomic offsets of a step are a bijection onto the 16 KB block.
  - Each instruction's 32 lanes are exactly one aligned 128 B line.
  - The 4 waves stay disjoint.
  - `k_dq_cvt_t` reads exactly the element the atomics wrote for every (q, d), and its vec4 is 4 consecutive d
    within one line.
  - cvt_t is exhaustively a bijection on the small shapes and corner-checked on prod.
- **F section** prints the footprint table above.
- The G, A, S, L, I and B sections are as before. Atomic maxima on prod: 134217728 = size.

## 12. Round 4 (r4a / r4b / r4c and combinations, all built on w4f_r3)

**Card result for r3** (as reported by the coordinator):

| | value |
|---|--:|
| r3 op time | 6.630 ms |
| s4 / ASM | 5.516 / 5.485 ms |
| k_dkdv_w4f cycles (PMC) | 1.142e7, about 2745 cyc/step over 4160 steps per SIMD |
| ASM main kernel cycles | 7.71e6, about 1853 cyc/step |
| ATT, CU1 | 20.8 cycles per WMMA, the same as the ASM |

The ATT on CU1 therefore accounts for only about 1660 cyc/step. The rest (about 1.65× instead of the ASM's
1.12× PMC/ATT ratio) is chip-level: atomic back-pressure (740 cyc/step at issue) and memory-side effects.

**(a) ASM work decomposition.** 4096 WGs = B4 × Hq32 × 32 (near/far kv-block pairs j, 63−j) × 1 q head.
- Each WG runs exactly 260 steps. That is 16 WGs per CU, perfectly balanced.
- dK/dV are per q head in the ASM, so the GQA reduction happens outside the WG.
- Our WG (b, hkv, one 128-kv block, all 4 q heads in-register) has 1–4× longer and unequal jobs.
- A greedy 256-slot dispatch model (`w4f_r4a/order_model.py`) gives a makespan/average of **1.000** for our
  current grid, and also 1.000 for near/far pairing. With a per-WG overhead of 10 steps: 1.019 vs 1.010.
- **So there is no tail/imbalance to recover; pairing is worth ≤ 1%. Not built.**

**What the model does show is locality.**
- An LRU over the 64 KB dQ blocks (4 heads × 16 KB per (b, hkv, pair)), fed with the dispatch-model event
  stream, gives these miss rates:

| grid order / sweep | makespan | 8 MB | 16 MB | 32 MB | 64 MB |
|---|---|--:|--:|--:|--:|
| current (b, blk, hkv), ascending | 1.000 | 0.76 | 0.75 | 0.70 | 0.03 |
| **current, descending** | **1.000** | **0.28** | **0.27** | **0.23** | 0.05 |
| group-major (b, hkv, blk), descending | 1.138 | 0.13 | 0.03 | 0.03 | 0.03 |

- **Why descending helps.** Ascending, WG blk j reaches pair p at 4(p − 4j) steps after its start, so pair p's
  64 contributions are spread over up to about 1000 steps. Descending, all co-resident blocks of a group reach
  pair p at the same time (4(255 − p) after they start).
- The same applies to the Q/dO/LSE TDM reads, which every block of a group repeats.
- The group-major grid is even better on locality but loses 13.8% on balance, so it is off (`W4_GRPMAJOR`,
  not used).

**r4a** = r3 + `W4_LOCK` (descending sweep: full pairs first, masked diagonal pairs last; loops swapped).
- It adds the same contributions and atomics.
- dK/dV accumulation order is reversed, so dK/dV are deterministic but not bitwise vs s1/r3.

**r4b** = r3 + `W4_SIG_LATE` + `W4_TDM_LSE_LATE`. It targets the ATT's `s_wait_dscnt 0x0` before the signal
(55 cyc/step) and the idle on the 3rd `tensor_load` (58 cyc/step).
- **SIG_LATE:** the 18 R1 tr16 loads (a_p kh0 + b_do) are issued right after the P/dS store burst. The signal
  then waits `s_wait_dscnt 0x12` instead of 0: DS completes in order, so every store and older load has retired.
- **TDM_LSE_LATE:** the 2 LSE/delta tensor ops of the batch are issued after the dQ WMMAs. The batch and the
  wait B_s → signal B_{s+1} interval are unchanged, so the ring protocol is unchanged.
- ISA: `S8 L18 d0x12 t0x4 |sig| L22 … |wait| T2 L48 … T2`.

**r4c** = r3 + `W4_DELTA_ZERO` + `W4_CVT_SRC`.
- `k_delta_z` writes delta and zeroes the tiled workspace rows it owns: two 16 B stores per thread per row.
  `impl.py` then uses `torch.empty` and there is no memset.
- `k_dq_cvt_s` de-swizzles in source order. Each wave reads two whole 512 B slots, and each thread reads 2×16 B
  and writes 2×8 B.
- The old cvt_t read half-lines of two different q rows, 64 B apart per warp, so each 128 B line was fetched
  twice. That is the likely cause of the 0.135 ms (≈ 5.8 TB/s effective).
- k_dkdv_w4f / _sp ISA is byte-identical to r3 (`cmp`).

**Combinations:** r4ac = r4a + r4c; **r4abc** = all three.

**Compile.** Run from each dir: `./compile_w4f.sh w4f w4f_sp cvt_s cvt_t delta_z delta redsp`. All give RC=0,
0 spill, 0 scratch, LDS 231424.

| dir | VGPR w4f / sp | hot-loop instr |
|---|---|--:|
| r4a | 716 / 696 | 732 |
| r4b | 704 / 698 | 733 |
| r4c | 704 / 698 | = r3 |
| r4ac | 716 / 696 | 732 |
| r4abc | 716 / 696 | 731 |

Auxiliary kernels: k_delta_z 54 VGPR, k_dq_cvt_s 17 VGPR.

**Proof additions.**
- r4a: the descending tile map; masked steps are the last `n_mask`.
- r4b: the ISA check requires exactly 18 tr16 younger than the last P/dS store at the signal, and a
  `s_wait_dscnt 0x12` after them. The counter simulation checks dscnt ≤ 18 at every signal. Negative control:
  R1 = 17 fails.
- r4c: **Z** — the zero stores cover every workspace vec4 exactly once, in bounds. **C** — cvt_s writes every
  output vec4 exactly once, from the source vec4 that holds exactly that (q, d..d+3). Both are exhaustive on
  the small shapes and sampled plus corner-checked on prod/proxy.
