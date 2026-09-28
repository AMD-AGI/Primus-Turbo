# Round 10 act -- r10.i2.g24 (occ2 4-wave x 2 WG/CU, all-LO, prod-gated)

## Step 0: feasibility (2026-09-28)
- Source diff: lab2-L12/r6_occ2_lo.diff (unverified, outside the corpus).
- `patch --dry-run` against rounds/010/op applies cleanly: 11 hunks, offset +3 for hunks 7-11.
- `diff lab2-L12/base_r6 op`:
  - managers: identical.
  - m32x8: differs only by g18's clean-tile condition (:1222-1225), which no hunk of the lab diff touches.
  - So g18 and occ2 do not overlap in lines. The R%2 row-sum generalisation reduces to the old code path at R=2.
- Needs from FlyDSL:
  - TDM atom with num_warps=4. Lab verified it in MLIR: `tdm<rank = 2, warps = 4>` (RESULT.md s1).
  - SharedAllocator of 163840 B.
  - waves_per_eu=2 compile hint.
  All three are used as-is in the lab build, with the same flydsl tree on fa-g2.
- Deviation from the lab diff: the lab edits the m32x8 module constants in place, so one build serves every shape.
  A shape gate needs a second module with its own JIT cache: new file flydsl_fwd/fmha_fwd_prefill_a16w16_m32x4.py,
  the same pattern as h37's m16x8 copy, selected in impl.py.
  The manager edits are shared; for 8 waves they must be ISA byte-identical (the lab's r6_ctrl showed md5-equal ISA; re-check here).
- Deviation on "0 spill": the champion r6 already carries SGPR spills (sspill 60, to VGPR lanes, no scratch), and
  lab occ2 has 61 (RESULT.md s2). The gate is VGPR spill 0 and scratch 0. SGPR spill is recorded, not gated.

## Step 1: implement
- managers: lab hunks applied in place (5 raises -> `not in (4, 8)`, `_tdm_load_views(num_warps=...)`, K/V pass self.num_waves).
- flydsl_fwd/fmha_fwd_prefill_a16w16_m32x4.py: copy of op/current's m32x8 (which includes g18), with the lab's 11 kernel hunks applied
  (patch: hunks 7-11 offset +3, no fuzz) and a 3-line header. fmha_fwd_prefill_a16w16_m32x8.py is unchanged (diff == current).
- impl.py: `_kern_occ2` import; after the m16 gate, `if grid_m32 >= 4 * _NUM_CU: kern = _kern_occ2`.
  Of ut/common.SHAPES, only prod crosses the gate (4096 >= 1024). proxy is 512 and the edge shapes are tiny,
  so validation.py covers occ2 at prod causal only.
  Extra coverage: a forced-gate harness (_NUM_CU=0) runs gates.check_correctness on every shape and mode through occ2.
- The compile-cache FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache (fa-g2 env) is cleared before the first build (below).
- 2026-09-28T03:07:10Z cleared /tmp/flycache (440K) in fa-g2.
- compile-only attempt 1: c_g2/nc_g2 KeyError. My copy came from lab tools/, which lacks the gqa2 configs (verify/tools has them). Harness bug, not a kernel bug; added the configs.
- compile-only attempt 2 (gqa2 configs c_g2/nc_g2, all arms/stems, isa/run_g2.log): 10/10 COMPILE_OK.
- ISA identity (sha256 of 22_final_isa.s, isa/metadata.txt): op m32x8 and m16x8 == op/current at all 6 configs
  (prod, nc_g4, c_g1, nc_g1, c_g2, nc_g2), 12/12 SAME. So the manager edits are neutral for 8 waves, and proxy/fast
  run today's exact ISA.
- m32x4 (occ2) metadata, per config prod/nc_g4/c_g1/nc_g1/c_g2/nc_g2:
  - LDS 163840 at all six; max_flat_workgroup_size 128.
  - VGPR 454/448/452/448/454/448, the same as current m32x8 config for config. ≤512 is the arch VGPR count
    (.vgpr_count, not the ×2 rocprof column).
  - vgpr_spill 0 and private_segment (scratch) 0 at all six.
  - sgpr_spill 0/4/0/3/0/4, identical to current m32x8's 0/4/0/3/0/4. SGPR spills go to VGPR lanes, not scratch.
  - Gate (VGPR spill 0, scratch 0, LDS 163840, VGPR ≤ 512): PASS.
  - The lab's "VGPR 446" differs from 454 here. The lab body is r6; this one is r6+g18, and g18 already carries 454 in the 8-wave body.
- Correctness run launched 2026-09-28T03:11:32Z, detached in fa-g2 (correct/run.sh, sentinel ALL_DONE, per-check rc files):
  validation.py on rounds/010/op, then ut/test_correctness.py --determinism 200, then tools/occ2_check.py.
  occ2_check.py does three things:
  - A: bitwise vs current at every ut shape and mode, natural gate.
  - B: _NUM_CU=0 forces every shape through m32x4. SQNR at the spec 49 dB via gates.check_correctness, plus bitwise vs current.
  - C: the verify_r6 adversarial set, all 32 kinds, causal and non-causal, bitwise vs current. Prod runs through the natural gate; toy/short_q/gqa4 run forced.
  Device note: at launch rocm-smi in fa-g2 showed GPU use 0% and 3 KFD pids. Those pids belong to a Primus e2e training
  job in fa-g0 (cgroup docker-c87432dadced = fa-g0, physical GPU 0), started 03:09Z. That is not our device, but it is a
  concurrent load on the node. It matters for timing steps, not correctness.
- validation.py on rounds/010/op (correct/validation.log, rc=2):
  - correctness: 16/16 PASS at the spec 49.0 dB. prod causal, the only gate-crossing shape (it runs m32x4): o 50.83 dB, lse 89.20 dB.
    The minimum is short_q full at o 49.82 dB, a shape that runs m16x8 with today's ISA (the baseline also reads 49.82-49.99 there).
  - determinism: 200/200 bitwise at fast. PASS.
  - speed: did NOT run. benchmark.py crashed at all 3 shapes with `KeyError: getpwuid(): uid not found: 12850`.
    Cause: I ran `docker exec -u 12850:...` and that uid has no passwd entry in fa-g2; getpass.getuser() raises.
    This is a harness failure (build/run failure #2), not a kernel result. The speed check is the measurement step's job,
    and it must run as the default container user (no -u), like every earlier benchmark. It is not re-run here because
    GPU 0's e2e job currently loads the node.
- ut/test_correctness.py --impl rounds/010/op --determinism 200 (correct/ut.log): rc=0, RESULT: PASS. 16/16 at 49.0 dB, the same dB figures as validation.
- occ2_check.py attempt 1 (correct/occ2_check.log/.json): rc=2, and the failure is in the harness baseline, not the kernel.
  - A: the natural gate matches op/current bitwise at all 16 shape and mode pairs. PASS.
  - C at prod through the natural gate (occ2): 32/32 causal and 32/32 non-causal adversarial kinds bitwise == current. PASS.
  - B (forced occ2): SQNR ≥49 dB at all 16 (PASS). Bitwise == current at proxy and prod only. Every mismatch
    (fast, toy, short_q, gqa4_batch2, mha, unequal*, sq_gt_skv, plus C toy/short_q/gqa4) is at a shape where
    op/current runs **m16x8**, because its gate is `grid_m32 < _NUM_CU`. So the comparison was occ2 vs a different kernel, whose
    row reduction differs. At every shape where current runs m32x8, occ2 is bitwise.
  - Fix, attempt 2: CHAMP_FORCE=1 also sets the champion's `_NUM_CU=0`, so it runs m32x8 at every shape. Only B and small C are
    re-run (PHASES=BC). The bitwise baseline is then the body that occ2 replaces.
- occ2_check.py attempt 2 (correct2/, CHAMP_FORCE=1 PHASES=BC): rc=0, OCC2_CHECK PASS.
  - B, both forced (occ2 vs 8-wave m32x8): SQNR ≥49 dB at all 16 pairs, and o and lse **bitwise equal at all 16**.
  - C forced, toy/short_q/gqa4: 32/32 adversarial kinds bitwise, causal and non-causal (192/192).
    With attempt 1's prod 64/64, occ2 is bitwise identical to the body it replaces on every input tried.
- dmesg: 0 new non-ifoe lines over both runs. Nothing is left running in fa-g2 (pgrep python3 empty).
  Idle and provenance: correct/provenance.yaml and correct2/provenance.yaml. The node-wide KFD pids are fa-g0's e2e job (GPU 0) and fa-g3's benchmark (GPU 3).
- Step 1 verdict: delivered and correct. The speed section of validation.py is still owed. It runs in the measurement step as the
  default container user, where it will fail vs beat as every round has (current is 0.78-0.95 of beat).

## Step 2: measure
- Champions in state.yaml: fast/proxy/prod = round 10 = op/current (h39). So incumbent == champion, and it is re-measured now from job_context/op/current.
- Protocol (plan act_protocol): 3 rotated sessions. Per session and shape, one benchmark.py process with {cand=rounds/010/op, champ=op/current,
  champ2=measure/champ2 (a byte-identical copy of op/current, the A/A control)}, with no beat in the process (h31) and PYTHONHASHSEED=0. Then beat
  alone in its own process (the target). Order rotates: s1 cand,champ,champ2; s2 champ,champ2,cand; s3 champ2,cand,champ. benchmark.py defaults:
  median of 101, blocked 9+lead 4, palindromic rounds, 8 s warmup.
- Runs as uid 12850 with USER/LOGNAME set, so getpass.getuser() does not hit pwd (the step-1 speed crash).
- Launched 2026-09-28T03:16:24Z, detached (measure/run.sh, per-process rc, ALL_DONE sentinel). GPU 2: 0% use, no python in fa-g2.
  Node: fa-g0 (GPU 0) is running a Primus e2e training job and fa-g3 (GPU 3) a bwd benchmark.py. Neither is ours, so they are recorded, not stopped.
- All 18 processes rc=0 (03:16:24-03:24:30). No new non-ifoe dmesg lines, nothing left running in fa-g2. measure/summary.json and measure/provenance.yaml.
  Mean TFLOP/s over 3 sessions (per-session values in summary.json):

  | shape | cand | champ | champ2 | beat | cand/champ (per session) | A/A champ2/champ (per session) |
  |---|---|---|---|---|---|---|
  | fast | 143.80 | 143.87 | 143.86 | 155.47 | 0.9995 (1.0000, 1.0013, 0.9973) | 0.9999 (0.9999, 1.0026, 0.9973) |
  | proxy | 1618.15 | 1619.23 | 1619.92 | 1588.16 | 0.9993 (1.0038, 0.9942, 1.0000) | 1.0004 (1.0043, 0.9948, 1.0021) |
  | prod | 1491.83 | 1502.52 | 1501.05 | 1557.19 | **0.9929** (0.9936, 0.9920, 0.9931) | 0.9990 (0.9995, 0.9991, 0.9984) |

- Gate check (measure/gatecheck.py wraps each module's entry point): fast->m16x8, proxy->m32x8, prod->m32x4.
  So occ2 did run at prod, and the prod number is occ2's.
- Reading: **prod falsified.** The prediction was 1.045-1.075 and the kill threshold <1.02. The measured 0.9929 is below A/A in every session, a small
  consistent regression of about -0.6% net of A/A. Fast and proxy run byte-identical ISA and sit inside the A/A spread, as predicted.
  The lab's +6.2% (vs r6, unverified) does not reproduce on op/current (r6+g18+m16 gate) on this card today.
  Two differences from the lab setting are visible, but neither is tested yet:
  (a) prod sclk here is ~1245 MHz, above the 1100 MHz VR-throttled regime the lab ran in;
  (b) the base now carries g18.
  The cycle readout (plan part a, GRBM/8 via pmc) is what separates cycles from clock. It belongs to the next step, not this score.
- Targets = beat measured now, in its own process. Score = mean(min(cand/beat, 1)) = (0.9249 + 1.0 + 0.9580)/3 = **0.96099**.
  Note that champ proxy beats beat here (1619 vs 1588).

## Step 3: prediction check (g44 part a, GRBM cycles)
- Attempt 1 (measurements/g44a_cycles, launched 03:26:27Z as `docker exec -u 12850 -e HOME=/tmp`): rocprofv3 --pmc (time pass) on driver.py prod champ,cand,champ2.
  The driver completed (DRIVER_DONE, 39 dispatches). rocprofv3 then aborted in finalize: `ring_buffer.cpp:106 mmap failed with errno 22`, signal 6.
  It hung in its signal handler with a 0-byte CSV and GPU 2 at 0% use. I killed the hung processes explicitly:
  - My first `pkill -f g44a_cycles/run.sh` matched its own bash, so the kill -9 never ran. That was my error.
  - Then kill -9 52789/52790 at about 03:37Z. Only a zombie is left, no live rocprof or python, GPU 2 at 0%.
  Suspected cause: the uid. Every earlier pmc run on fa-g2 (1-profiling, g44b) ran as the container's default user (root) without -u.
  Attempt 2 matches that known-good invocation, the only change. The old output dirs are removed first, so no stale file is read.
- Attempt 2 (03:37:24-35Z, default user): 4/4 rc 0, 0 new dmesg lines, nothing left running. See measurements/g44a_cycles/RESULT.md.
  - Identity at prod: cand runs WG 128 / LDS 163,840; champ and champ2 run 256 / 327,680. Proxy runs 256 / 327,680 for every arm.
  - Prod GRBM/8: champ and champ2 1,864,126 (A/A 0.997 and 1.0004); cand **1,740,818 = 0.9339 (-6.6%)**.
    The prediction was ≤1,775,000, so it **HELD**.
  - GRBM ticks per ns: cand 1.382 vs champ 1.534 (**0.901**). Duration under the profiler is 1.037x (slower).
    The ~10% clock loss eats the 6.6% cycle gain, which matches the benchmark's 0.9929.
    The plan's lockstep band (≤3 points) failed: the gap is about 7 points, clock the other way.
  - Cell: **slower_held**. The mechanism (barrier decoupling, fewer cycles) is confirmed. The kernel is power- or clock-limited at prod on this card,
    so the cycle gain does not become time. This goes to facts.md, not dead_ends.md.

## Step 4: report
- validation.py rerun against rounds/010/op (validation/, 03:39:42-03:41:33Z). Detached in fa-g2 as the **default container user**,
  so the step-1 getpwuid crash cannot recur. GPU 2 at 0% before; node pids are fa-g0/fa-g3 by cgroup; 0 new dmesg lines.
  - **exit 2**
  - correctness: 16/16 PASS at 49.0 dB. Same dB as step 1; prod causal (occ2) o 50.83, lse 89.20.
  - determinism: 200/200 PASS.
  - speed: FAIL vs beat, geomean 0.9476 (fast 0.9264, proxy 0.9565, prod 0.9603).
  - So the exit code comes from the target, not from correctness. Every round so far has this failure mode.
  - The speed line prints `kernel=...m32x8.py md5=...`. That is validation.py's label for the default module, not the dispatched body.
    gatecheck.py (step 2) shows prod dispatches m32x4.
  - validation's cand/beat puts beat in the same process, which h31 excludes from the score protocol. Its proxy 0.9565 (sclk 1103->1292) vs step 2's 1.019 is that effect plus clock state.
    The round's numbers are step 2's.
- Files: act.yaml (structured, for reflect) and this narrative.
- Summary of the round:
  - Built: occ2 (4 waves × 2 WG/CU, all-LO) as a second module, gated to grid_m32 >= 4*NUM_CU, so prod only.
  - Correctness: bitwise identical to the 8-wave body on every input tried.
  - Speed: prod 0.9929 vs champion (below A/A in all 3 sessions); fast/proxy unchanged (same ISA).
  - Prediction (prod cycles -5% or more): held at -6.6%. The ~10% lower clock turned it into a net slowdown.
  - Cell: slower_held.
  - The finding is a fact about the card: at prod, cutting cycles through higher issue density does not become time.
  - Build failures: none in the kernel. All three failures were in the harness (listed in act.yaml).
