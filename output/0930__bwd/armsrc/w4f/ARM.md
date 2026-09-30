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
