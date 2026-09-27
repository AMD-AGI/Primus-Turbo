| h57 | must note | Card-time order for the two 2026-09-27 CPU studies: P2 atomic probe, then X1, then P1 | open |
| h58 | advise note | FUSED5: dQ fused into k_dkdv with fp32 SCOPE_DEV atomics, k_dq deleted -- compiles at 936 VGPR / 0 spill; atomic throughput decides it | open |
| h59 | advise note | 4-wave barrier cost: the fence drain is compiler-made, and the barriers phase-lock an LDS window with no WMMA cover -- X1/X2/X5 preregistered | open |
| h60 | advise note | Champion-side: the g62 depth-2 prefetch is force-drained at the loop head by 67 rotation v_movs | open |
| h61 | standing note | Closed or corrected by the 2026-09-27 studies | open |

---

Sources (both **CPU + COMPILE_ONLY only, no card time**, flydsl 0.3.4.1 in fa-repro; champ/w4/x1/x2 ISA
is identical under 0.3.2 and 0.3.4.1):
- `A` = `output/0927__flydsl/bwd/BARRIER-CENSUS.md`, artifacts in `bwd/census/`: `census.py`,
  `aiter_census.py`, `census032.{json,txt}`, `trees/<arm>/`, `dump032/`, `dump0341/`, `enc/`.
- `F5` = `output/0927__flydsl/bwd/FUSED5.md`, artifacts in `bwd/fused5/`: `op/`, `base/`, `fused5.patch`,
  `compile_f5.py`, `run_variant.sh`, `body_stats.py`, `bounds_proof.py`, `isa/`, `atomprobe/`.

Every number tagged [ISA] was read from compiled code. Every % or ms prediction is [INF] (h54 trap 1:
static counts are not costs).

## h57 -- Card-time order for the two 2026-09-27 CPU studies: P2 atomic probe, then X1, then P1

Precondition: the fwd job is stopped and the card is idle. Check `ls /sys/class/kfd/kfd/proc` and
`rocm-smi --showpids` for no foreign PIDs. One process, one shape, one mode per slot (h54 rule 3).
Any build with `.vgpr_spill_count > 0` or scratch > 0 never goes on the card.

1. **P2, the fp32 atomic throughput microbenchmark** (F5 §4, `fused5/atomprobe/run_probe.py`).
   - Lowest risk: no TDM, no barrier, one wave, and all addresses are proven in-bounds on CPU.
   - Run point a (KVG=1) once, confirm the process returns, then sweep points b-f.
   - It answers h58's go/no-go for both the 1-wave and the 4-wave fusion shapes.
2. **X1** (A §4, `census/trees/x1`: w4 minus barrier-1, a one-line change). Run `screen4w.py` UT
   (15 shapes, `output/0925__flydsl/g1b-4wave/`) and validation first. Same session as w4, palindromic.
3. **P1** (F5 §4: `F5_KT=lds F5_ATOM=0` k_dkdv_f5, 996 VGPR, 0 spill). Palindromic A/B against champion
   k_dkdv, kernel time only.
4. Then, as the verdicts say: X2 or X5 (h59); P3 and P4 (h58).

## h58 -- FUSED5: dQ fused into k_dkdv with fp32 SCOPE_DEV atomics, k_dq deleted -- compiles at 936 VGPR / 0 spill; atomic throughput decides it

- **Path:** `bwd/fused5/op/` is the champion (`bwd341/op0341`) plus the fusion. The `kernels.py` env knobs
  are `F5_KT` (`lds` | `res`) and `F5_ATOM` (0 | 1). The full diff is `fused5.patch`.
- **What it changes:**
  - After each 16-q half-block's dS (bf16), it does 8 extra WMMAs: dS as the A operand (free, same
    fragment idiom as k_dq's `a_ds`) times K^T, read from LDS with 2 `ds_load_tr16_b128` per d-tile. K is
    copied into the free part of segment 0, and LDS stays 70656 B.
  - Then 64 `global_atomic_add_f32 scope:SCOPE_DEV` per lane into an fp32 `dq_acc`. Each atomic covers
    2 q-rows x 64 B. There is no RETURN and no storecnt wait in the body.
  - Written as `fx.UniversalAtomicAdd(fx.Float32, rocdl.SyncScope.Agent)` on a plain global view. **Not**
    `make_buffer_tensor`: BufferAtomicAdd is SCOPE_CU, and the buffer UniversalAtomic aborts (h42).
  - `cvt_dq` reuses `k_redsp_q` with nsp=1 (41 VGPR). `k_dq`/`k_dq_sp` are no longer launched.
  - fast goes through `k_dkdv_sp_f5` (q-axis split-K, atomics add correctly).
- **ISA evidence:**

  | build | VGPR | spill / scratch | full body | v_wmma | atomics | tr16 | s_wait_dscnt |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | champion k_dkdv | 904 | 0 / 0 | 778 | 64 | 0 | 40 | 3 |
  | **f5-lds** k_dkdv_f5 | **936** | 0 / 0 | 1095 (+40.7%) | 80 | 128 | 72 | 33 |
  | f5-lds sp (fast) | 936 | 0 / 0 | 1093 | 80 | 128 | 72 | 33 |
  | f5-res (K^T in VGPRs) | 1024 | **29 / 120 B** | 1455 | 80 | 128 | 40 | 9 |
  | P1 probe (`F5_ATOM=0`) | 996 | 0 / 0 | 945 | 80 | 0 | 72 | 12 |

  - Body delta: +128 atomics, +57 `s_set_vgpr_msb`, +32 tr16, +30 dscnt, +16 WMMA, about +54
    SALU/address ops. Masked body 659 -> 987.
  - Prefetch cover 603 -> 602.
  - 1 wave/SIMD and 4 WG/CU are unchanged. The VGPR budget left is 88.
- **Expected effect** [INF]:
  - Matrix part at the measured issued rate (720 TF/s) is 7.665 ms. Adding k_delta and clear+cvt gives
    **7.913 ms + E_atom** against the champion's 10.751 ms.
  - With E_atom = 0: 0.970x bar, 1.36x champion. Break-even: E_atom <= 2.84 ms (<= 1.66 ms at k_dkdv's
    own 624 TF/s).
  - **Atomic throughput is the unknown.** BLOCK_KV=32 carries 68.99 GB of atomic payload (3.95x aiter).
    - Full overlap needs >= 9.0 TB/s. aiter has only shown >= 2.41 TB/s.
    - Fully serial, it loses below about 25 TB/s.
  - The 4-wave BLOCK_KV=128 shape cuts the payload to 17.45 GB (= aiter), which needs only 2.28 TB/s.
- **Risks:**
  - K^T tr16 reads are issued just before use (load->use about 10 instructions), so LDS latency is exposed.
    Hoisting them costs +64 VGPR and probably spills.
  - `s_wait_xcnt 0x4/0x0` after each 64-atomic clause: this is the issue back-pressure P2 measures.
  - dk/dv arithmetic is untouched. Predicted **bitwise equal to the champion**; if not, the dk/dv IR moved.
  - dq is not bitwise equal: g61 chain-split order. Same accuracy class; bound 90.3 dB worst, about
    120 dB expected, vs the 70 dB floor.
  - Non-causal is not bounds-proven (unscored).
- **Go/no-go (preregistered, F5 §6).** T_c = P1 time, T_a = P2 point-a time. 10.0 ms = champion - 0.25
  - 0.5 margin.
  - T_c + T_a <= 10.0: **GO**. Run P3 (`F5_KT=lds F5_ATOM=1`, E_atom = T_P3 - T_P1), then P4.
    P4 gates:
    - dk/dv: 200 runs bitwise self-consistent AND bitwise equal to the champion;
    - dq: run-to-run >= 70 dB, all three outputs >= 50 dB vs eager;
    - determinism checked on both fast (sp_f5, nsp=16) and prod.
  - max(T_c, T_a) <= 10.0 < sum: run P3 only. GO if T_P3 + 0.25 <= 10.0.
  - max > 10.0: **1-wave fusion dead** (write it to dead_ends). Go to 4-wave BLOCK_KV=128, and pass only if
    P2 point b (KVG=4) <= 3.0 ms and P1 x <= 7.6%. That route still carries the 45% barrier cost (h59).
  - P2 point b > 5.0 ms: the whole fusion route is NO-GO. Report 0.72x as the contract-bound ceiling.
- **How the round starts:** after the P2/P1 verdict only.
  1. Copy `bwd/fused5/op/` onto the round's working copy. First diff `fused5/base/` against the
     round's `op/current`; if the champion moved, apply `fused5.patch` instead.
  2. Set `F5_KT=lds F5_ATOM=1`.
  3. Compile-only through `compile_f5.py` and confirm 936 VGPR / 0 spill.
  4. Run P4 before any speed claim.

## h59 -- 4-wave barrier cost: the fence drain is compiler-made, and the barriers phase-lock an LDS window with no WMMA cover -- X1/X2/X5 preregistered

- **Path:** `bwd/census/`: arm trees `trees/{champ,w4,s1,s2,nobar,x1,x2}`, ISA in `dump032/`, tools
  `census.py` and `aiter_census.py`.
- **Findings [ISA]:**
  - **F1.** `fx.barrier()` lowers to `fence release; s_barrier_signal; s_barrier_wait; fence acquire`.
    The release fence is what emits `s_wait_loadcnt 0` + `s_wait_dscnt 0`. Raw
    `_mrocdl.s_barrier_signal` has no wait in front of it.
  - **F2.** w4's R1 region has 55 instructions: all 40 `ds_load_tr16_b128` and 1 WMMA, then dscnt 0 and a
    rendezvous. Every one of R2's 31 WMMAs depends on it.
    aiter has the same barrier density (about 40 WMMA per barrier) and occupancy (1 wave/SIMD), but
    every region has 16-55 WMMAs interleaved with its LDS reads. Every signal->wait gap has 3-6 WMMA
    and 3-4 atomics, and it signals with dscnt 0x14/0x10, never 0.
  - **F4.** S2 only split barrier-2. Barrier-1 was untouched and has **never been measured alone**.
    S2's barrier-2 signals with 40 tr loads still in flight, so its WAR safety relies on in-order LDS
    service.
  - **F5.** nobar and S1 have almost the same single-wave instruction stream (800 vs 765). The 1.449x gap
    therefore has to be a **cross-wave dynamic** effect.
  - **F7** [MEAS+INF]: w4 16.072 ms vs nobar 11.095 ms, about 4160 serial iterations per CU, which is
    **about 1270 cycles per iteration**, about 630 per barrier. That is far above any plausible
    hardware barrier latency.
- **Best-supported explanation [INF], H-align:** each barrier puts all 4 waves in phase, so every
  iteration's LDS burst (about 4 x (24 late st + 40 tr) x 512 B = 128 KB) lands while every SIMD is
  idle. The magnitude estimate is about 700-1000 cycles: plausible, but not closed.
  The competing explanation, H-count, is a fixed cost of about 630 cycles per rendezvous.
- **Preregistered arms** (all on w4, same session, palindromic, ranked on prod):

  | arm | change | H-align predicts | H-count predicts |
  | --- | --- | --- | --- |
  | X1 | delete barrier-1 (legal: Q/dO staging is replicated, and barrier-2 still guards cross-iteration WAR). 938 VGPR, 0 scratch | <= +8% | >= +15% (342 -> >= 395 TF/s) |
  | X2 | delete both, private per-wave B ring (LDS 148480 B). Legal, 938 VGPR, 0 scratch | about nobar (>= 1.40x w4, about 0.95x champion) | same |
  | X5 | de-replicate staging: wave w stores only dt==w (32 -> 8 st per wave), barrier-1 becomes required. Needs code + UT | +5 to 7% | about 0 |

  - X1 >= +15% and X5 about 0: the count is the cost. Go to double-buffered rings with fewer barriers.
  - X1 <= +8% and X5 >= +4%: the phase-locked window is the cost.
  - In between: inconclusive. **Do not propose a fifth mechanism** (h53/h54).
- **Consequence for any 5-GEMM or 4-wave design:** dS crosses waves, so at least one barrier per
  iteration is required. Build it the aiter way:
  1. raw `s_barrier_signal(-1)`/`s_barrier_wait(-1)` with a **partial** `s_wait_dscnt N>0` (needs
     double-buffered dS and Q/dO rings);
  2. software-pipeline the post-barrier LDS reads under the previous stage's dK/dV WMMAs;
  3. no replicated staging.
  Raw intrinsics carry no memory semantics, so check every build with `census.py` for ds ops crossing
  signal/wait. Named barriers do not help here (all 4 waves share the data).

## h60 -- Champion-side: the g62 depth-2 prefetch is force-drained at the loop head by 67 rotation v_movs

- **Evidence [ISA] (A, F8):** champion `.LBB0_8` rel 3-115 has 67 `v_mov` for the loop-carried
  `pre1 -> pre0` copy. At rel 12 they trigger `s_wait_loadcnt 0x0`, which retires 33 loads with cover
  603-728 instructions, about 0.8 iteration.
  So the effective prefetch distance is **below one iteration**, and each iteration pays 67 extra movs.
- **Lever [INF]:** unroll `qloop_full` x2 so pre0/pre1 swap by register renaming. That removes the copies
  and the drain. The remainder must go through the masked/predicated iteration, never a tail loop
  (spill risk).
- **Status:** the prefetch axis is closed by rule (h54), so this is only a lead from new evidence.
  Compile-only screen first: 0 spill, the drain gone at rel 12, the v_mov count down. Card time only
  if the ISA shows both. The prior is low (S1 cut cover 389 -> 132 at 0.995x).

## h61 -- Closed or corrected by the 2026-09-27 studies

- **h50's S1 cover number is wrong.** Cover did not rise 389 -> 529. The loads were force-retired by the
  loop-head rotation copies, so min cover fell to **132 (-66%)**, and prod still moved only 0.995x.
  Conclusion strengthened: vmem cover is not the 4-wave constraint.
- **Ruled out as the 4-wave barrier cost [ISA]:**
  - instruction cache: w4 loop body 4900 B vs nobar 4872 B; aiter's is 18.1 KB;
  - VGPR reuse: no WAW waits, slack never negative;
  - occupancy: every arm is > 512 VGPR, 1 wave/SIMD;
  - tensorcnt and storecnt drains: none exist.
- **f5-res (K^T resident in VGPRs):** 1024 VGPR, 29 spill, 120 B scratch. Dead by the spill rule, and
  so is its P1 probe (24 spill).
- **1-wave BLOCK_KV=32 fusion cannot reach parity with the bar** even at E_atom = 0: it needs 743 TF/s
  issued. It only competes with the champion, and only if P2 shows >= 9 TB/s with overlap.
- **Split barriers with the gap filled by prefetch (S2 style) as the fix:** closed. S2 left barrier-1
  intact and made barrier-2 WAR-unsafe on paper. Future split barriers need an explicit partial dscnt
  before signal.
- **Named barriers for k_dkdv:** they do not help, because all 4 waves consume the same data.
