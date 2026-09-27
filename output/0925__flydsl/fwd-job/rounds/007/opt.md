# Round 7 (fast) -- opt

Written as the round went. Every card run went through container `fa-repro`, GPU0. Small parsed outputs are
in `raw/`. The probe copies and ISA dumps are also in `raw/`, not in the /tmp scratch dir, because the
container cannot see /tmp. That is the reason they are here.

## 1. Reading

- findings: `facts.md`, `dead_ends.md`, `pool.md`, `route.md` (operator tables h2-h27 included),
  `rounds/006/2-reflect/reflect.md`, `rounds/006/1-opt/opt.md` s1-s2.
- Round-5 profile: `rounds/005/1-profiling/profiling_summary.md`. The prod verdict is `latency`, confidence low.
  - Candidate c1: KV-loop per-cycle efficiency.
  - Candidate c2: energy per cycle. Zeros vs randn: +7.2%, clock capped at +11%.
  - Stall attribution, ATT and WMMA counters are unavailable (facts, Constraints).
- State of the job:
  - The pool is effectively empty. g03 and g04 are marked "retire at reflect". g16 was closed by its
    discriminator in r6. g15 is predicted null.
  - The last two rounds were not accepted.
  - So this round goes to the corpus (3c) and to the cross-backend recipes (3d).

## 2. Look

### 2a. Survey

`rocprofv3 --stats` / `--kernel-trace` write no files on this stack. Rounds 5 and 6 recorded this
(3 attempts, rc=0). I did not spend a fourth attempt on it. `op/current` has not changed since round 5, so
round 5's kernel list stands: one kernel per call, 94.4% of prod time.

### 2b. Static census of the champion's clean KV loop (free, compile-only)

`raw/build/isa_prod` (md5 7b078261, VGPR 456, SGPR 100, spill 0/0). LO-clean loop `.LBB0_3`, one 64-KV tile:

| class | count |
| --- | --- |
| total instructions | 513 (553 when the rescale block runs) |
| v_wmma 16x16x32 bf16 | 64 |
| v_exp_f32 | 66 |
| VALU total | 331 (pk_fma 33, pk_add 32, cvt_pk_bf16 32, max3 30; pk_mul 32 only inside the rescale branch) |
| SALU total | 116, of which s_set_vgpr_msb 40, s_delay_alu 14, s_wait_* 15 |
| v_nop | 15 |
| ds_load | 64 (32 b128 + 32 tr16) |
| WG barriers | 1 (`s_wait_tensorcnt 0` -> `s_barrier_signal/wait`) |

Readings:

- h4's "156 v_nop per 256 KV" is stale. The champion has 15 per 64 KV.
  - About 8 of those follow `v_exp`. They are WAR hazards: the exp's source register is rewritten by the next
    `v_pk_fma`, and the compiler recycles 3 staging pairs (v[158:161], v[66:67], v[76:77]).
  - They are not a dependency on the exp result.
- The softmax segment (about 240 instructions, ISA lines 234-472 of the loop) contains **no WMMA**. It is
  fenced by `sched_barrier(0)` on both sides, so overlap can only come from the other wave on the SIMD.
- That other wave is wave i+4 of the **same** WG. It passes the **same per-tile WG barrier**. The two waves are
  therefore re-aligned in phase every 64 KV. That is exactly when both want the VALU for softmax, or both want
  the matrix pipe.
- This offers a reading of why `r3.i2.g08` (setprio stagger) measured prod 1.000: the barrier undoes any
  stagger within one tile. Inferred, not measured.
- This also gives h24 (one barrier per 2 tiles) a mechanism beyond "fewer rendezvous".

### 2c. Two throwaway probes, prod, 3 slot-rotated sessions, no beat in the process (`raw/measure/`)

- **P1 nobar**: `gpu.barrier()` deleted from `_drain_barrier`. The tensor wait is kept.
  - This races the LDS ring, so the output is garbage. It is a timing-only upper bound.
  - Barrier waits 6 -> 2, everything else the same (VGPR 456, 0 spill).
  - **Prediction:** +3 to 6% at prod. That is the ceiling for any barrier-count lever (h24).
  - **What would change my mind:** below +1% kills h24 before it is built. Above +8% says phase-lock is a
    big term, and h24 should be judged against that ceiling.
- **P2 nodefer**: `ENABLE_DEFER_RESCALE=False` on the current (packed) base.
  - VGPR 448, SGPR 98, 0 spill.
  - This is h22's last unmeasured component (L20 BF on the r4 base). r2.i1.g05 measured it at -6% on the
    round-2 base.
  - **Prediction:** -3 to -6%, which closes h22.
  - **What would change my mind:** at or above 0%.

### 2d. Probe results (`raw/measure/summary.txt`, `s{1,2,3}.out`)

- Setup: rocm-smi read "No KFD PIDs" and use 0% before and after. dmesg showed no amdgpu error during the runs.
- The three arms were rotated across three sessions.

| session | sclk | cur TF/s | P1 nobar | P2 nodefer |
| --- | --- | --- | --- | --- |
| s1 | 1042/1043 | 1096.57 | 1153.75 (**1.0521**) | 1006.97 (**0.9183**) |
| s2 | 1039/1041 | 1092.10 | 1141.11 (**1.0449**) | 998.49 (**0.9143**) |
| s3 | 1034/1040 | 1091.49 | 1141.11 (**1.0455**) | 1002.65 (**0.9186**) |

- Floor spread: the control ranges 1091.5-1096.6 across sessions, which is 0.47%.

#### r7.i1.g17 -- instrument: the ceiling of the per-tile barrier term (P1)

- Removing the one per-tile WG barrier buys **+4.5 to +5.2% at prod**. This lands inside my +3 to 6%
  prediction.
- It is a ceiling for any barrier-count or phase-freedom lever, not a price.
  - The deleted barrier is also what makes the LDS ring correct.
  - The output is garbage. Garbage data could move power, but sclk is identical within each session.
- What it settles:
  - h24 (one barrier per 2 tiles) has a real term to take. I estimate half the ceiling at best, about
    +2%, if the term scales with barrier count.
  - An arm of h24 above about +5% would be suspicious.

#### r7.i2.g18 -- instrument: h22's last component, L20 branch-free rescale on the packed base (P2)

- `ENABLE_DEFER_RESCALE=False` measured **-8.1 to -8.6% at prod** in 3 sessions. I predicted -3 to -6%; it is
  worse.
- r2.i1.g05 measured -6% on the round-2 base. On the packed r4 base the unconditional `v_pk_mul` rescale costs
  more.
- Status of h22's pieces:
  - L15 (packed exp argument) shipped in r4.
  - L17 (per-lane row sum) lost in r6 (r5.i2.g14).
  - L20 (branch-free rescale) lost here.
- h22 is discharged; nothing is left to build. h26's rule "close BF for good" applies.

### 2e. Compile-only: `r4.i4.g12` coexec scheduler on the current tree (`raw/probes/isa_coexec`)

- Result: SGPR 107, **sgpr_spill 1**, private_segment 16, VGPR 452, v_nop 44, `s_set_vgpr_msb` 293.
  - The champion has 240 `s_set_vgpr_msb` over the whole kernel.
- This is still not at spill 0/0, so it stays off the card (h14).
- Its condition in the route ("card only at spill 0/0") remains unmet on the r4 tree.

## 3. Sources

- (a) pool/route: nothing live. g15 is predicted null. g03, g04 and g16 are closed.
- (b) the survey above: the barrier term is +4.5-5.2% at prod. It is the largest measured single term this job
  has left that is not the softmax arithmetic.
- (c) corpus:
  - `optimization/routes/1-metrics-to-techniques.md` (instruction-mix and wave-cycle rows).
  - `backends/flydsl/attention/techniques.md`, the sections "The constraint is softmax issue" and
    "Two ways to overlap exp". The second makes this point: two co-resident waves overlap only if they are in
    different phases. Our two waves per SIMD belong to one WG and are re-phased by the barrier every tile.
- (d) cross-backend: `backends/hipkittens/attention/recipes/gqa_d128.md` s6 and "Forward".
  - The HipKittens forward gives warps 4-7 **one extra barrier in the prologue**. The two halves then run half
    a cluster out of phase for the whole loop, while barriers stay balanced.
  - That is a stagger which **survives** per-step barriers. The r3 `s_setprio` stagger (g08) did not survive
    them.
  - I take the mechanism, not the constants. It becomes pool entry r7.i3.g19.

## 4. Choice

- h22 is the only `must`, and it is now discharged by g18. It stays at row 1 as a record-only row, so the
  executor spends no build on it.
- The round's candidate is **h24** (operator advise; prototype `proto/barriers`):
  - Its diff dry-runs cleanly onto `op/current` (14 hunks, offset 19, no rejects), because it does not touch
    the softmax that round 4 changed.
  - The ceiling measured above says the term exists.
  - Two arms, from the same base:
    - **p22n**: G=2, no split. This is the barrier count alone.
    - **p22s**: G=2 plus split signal/wait. This adds the gap.
  - The arms touch the same lines. That is why this counts as one idea with two arms. p22n vs p22s is the
    split-gap question that bwd S2 answered null.
- Expected result:
  - p22n about +1.5% at prod (range +0.5 to +3).
  - p22s from +0 to +1% over p22n.
  - Best arm about +2%, and never above the 4.7% ceiling.
  - fast and proxy are judged only as sentinels (bimodal by slot).
- **Falsified if:** both arms are within the 0.47% floor of control. That would mean the barrier term is
  phase-lock that halving the count does not release, and g19 becomes the only route to it.

## 5. Route and pool

- `findings/route.md` `## Route` has been rewritten.
  - The queue: h22 (record only) > h24 (p22n, p22s) > r7.i3.g19 > r5.i3.g15.
  - The remaining advise items are rows 5-13, each with its condition.
- `findings/pool.md` has one new entry, r7.i3.g19.
  - r7.i1.g17 and r7.i2.g18 are instruments that have already been measured, so they are recorded here and not
    in the pool.

## explored.consulted

- knowledge/INDEX.md
- knowledge/optimization/routes/1-metrics-to-techniques.md
- knowledge/backends/flydsl/attention/techniques.md
- knowledge/backends/flydsl/attention/dead-ends.md (headings only)
- knowledge/backends/hipkittens/attention/recipes/gqa_d128.md (s6, Forward, s7)
- Primus-Turbo/output/0927__flydsl/proto/barriers/REPORT.md, plus proto.diff dry-run against op/current

## Build failures

None. The two probe builds and coexec all compiled with rc=0; coexec failed the spill gate, as recorded in s2e.
The working copy `rounds/007/op` is untouched and identical to `op/current`.

---

# Round 7 -- build and measure (route rows 1-2)

## 6. Row 1 h22 -- closed, no build

- r7.i2.g18 in s2d measured L20 at prod 0.918/0.914/0.919. Nothing is left to build.
- I wrote the closure in the route.md outcome cell. I spent no arm on it.

## 7. Row 2 h24 -- build (`raw/arms/{p22n,p22s}`, `raw/build/arms/`)

- **Build steps:** from `job_context/op/current`, I applied `proto/barriers/proto.diff` with `patch -p1` (clean).
  - The arm constants are hard-coded: G=2, NG=2, `KV_SLOT_ALIGN=1024`, `BARRIER_FENCE=True`, and `SPLIT_BARRIER` False for p22n and True for p22s.
  - The `PROTO_ENV`/`PROTO_FENCE` env reads are deleted.
  - I cleared `/root/.flydsl/cache` before the first compile.
- **Compile-only** (prod shape, `rc.py`, all rc=0):

| build | d | VGPR | SGPR | vgpr spill | sgpr spill | scratch | md5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r4 champion | 128 | 456 | 100 | 0 | 0 | 0 | 7b078261 |
| p22n | 128 | 448 | 99 | 0 | 0 | 0 | 4fcc1b92 |
| p22s | 128 | 448 | 99 | 0 | 0 | 0 | e4e2ed9c |
| r4 champion | 192 | 510 | 107 | 0 | 9 | 0 | |
| p22n / p22s | 192 | 502 | 107 | 0 | 10 | 0 | |
| r4 champion | 256 | 512 | 100 | 227 | 0 | (spills) | |
| p22n / p22s | 256 | 512 | 100 | 217 | 0 | 640 | |

  - The d192 and d256 spills are the champion's own, so they are not regressions. The d128 prod path is 0/0.
  - The VGPR is 448, not the proto's 439, because this base is round 4 (packed softmax), not round 3.
- **Correctness** (`raw/correct/`):
  - `validation.py` with the spec's 49 dB: 16/16 PASS for both arms.
  - Determinism 200/200 for both arms. The hashes o=26a89a2db0cd and lse=d6ac8da1e101 are **identical to the champion's** (`raw/ab/ut_inc.out`). So the arms are bitwise equal to the champion.
  - `ut/test_correctness.py` gives rc 2 for both arms *and for the champion*. It uses `gates.GATE_DB`=50 dB, not the spec's 49. The same 7 lines fail at the same dB. This is the same pre-existing gap as rounds 1 and 6; I did not touch the check.
  - validation exit 2 comes only from the speed bar: geomean vs beat is 0.6265 for p22n and 0.6485 for p22s.
- Build failures: none.

## 8. Row 2 h24 -- measure (`raw/ab/`, `summary.txt`)

- **Setup:** rocm-smi read "No KFD PIDs" and use 0% before and after. The dmesg monitor was armed for the whole run and caught no amdgpu error.
- **Sessions:**
  - s1-s3: prod only, arm order rotated, no beat in the process.
  - a1/a2: all shapes with beat, in palindromic order.

| session | sclk | inc (rounds/004/op) TF/s | p22n / inc | p22s / inc |
| --- | --- | --- | --- | --- |
| s1 prod | 1043/1041 | 1091.69 | 0.9412 | 0.9661 |
| s2 prod | 1037/1037 | 1086.72 | 0.9407 | 0.9695 |
| s3 prod | 1043/1037 | 1096.40 | 0.9343 | 0.9638 |
| a1 prod | 1022/1036 | 1079.88 | 0.9400 | 0.9667 |
| a2 prod | 1039/1037 | 1087.06 | 0.9392 | 0.9628 |

- All shapes, mean of a1 and a2:

| shape | inc | p22n | p22s | beat | p22s/inc | p22n/inc |
| --- | --- | --- | --- | --- | --- | --- |
| fast | 39.78 | 38.41 | 39.23 | 69.47 | 0.986 | 0.966 |
| proxy | 801.80 | 760.19 | 794.93 | 1024.71 | 0.991 | 0.948 |
| prod | 1083.47 | 1018.07 | 1045.27 | 1398.49 | 0.965 | 0.940 |

- **Verdict: both arms lost.**
  - p22s: prod -3.5% (5/5 sessions, 7 times the 0.47% floor).
  - p22n: prod -6%.
  - My prediction was p22n +1.5% and best arm about +2%. It is **falsified in the opposite direction**. What I wrote as the kill condition was "both within the floor"; the result is worse than that.
  - Acceptance: throughput does not improve on the champion. Every shape is still at least 95% of its best, and no shape had reached its target. Not accepted.
- **Mechanism** (static, `proto/barriers/tools/isa_loops.py` on the three ISAs):
  - The proto keeps a **one-tile loop body**. It guards the barrier and the 2-tile TDM group issue with a **runtime `tile % G` branch** (p22n `.LBB0_6`: `s_cbranch_vccnz` -> `s_wait_tensorcnt 0; s_barrier_signal/wait` -> 2 × 2 `tensor_load_to_lds` behind two more `s_cbranch_scc1`).
  - Per LO-clean tile:
    - total 553 -> 606 (p22n) / 607 (p22s), +10%;
    - SALU 42 -> 72/80;
    - branches 5 -> 7/8;
    - v_nop 16 -> 25/21.
  - WMMA, v_exp and ds_load are unchanged. LDS is still 327680.
  - The dynamic barrier count really is halved. What this measured is that 1/2 of a 4.7% ceiling is about 2.3%, less than what +10% issue slots per tile costs on a latency-bound loop.
  - Split (p22s) recovers about 3 points over p22n. The gap between signal and wait is worth something. Bwd S2 read it as null; here it is not.
  - Inferred, not measured: the ring-index SALU and the branches sit on the serial per-wave path. The two waves on a SIMD are one WG and cannot hide them.
- **What would be needed to take the term:** unroll the loop body x2 at trace time (two tiles per iteration, the barrier outside any runtime branch, slot indices as constants). That is h23's L14 shape. L14 was closed alone by h26; it would come back only as the carrier of L8. It is not built in this round.
- p23s was not run, because the route made it conditional on p22s winning.
- **Working copy:** `rounds/007/op` = p22s (the better loser), left in place as instructed. The only file changed is `flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py`.

## 9. Route outcomes

- Row 1 (h22): closed.
- Row 2 (h24): delivered, lost, mechanism as above.
- Rows 3-13 were not reached in this round, and their cells stay empty.
  - Row 3 (g19) needs a depth-3 ring, and this round's h24 build did not provide one.
  - Row 4 (g15) is unchanged.
