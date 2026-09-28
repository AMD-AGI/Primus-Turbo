# r15 act: r15.i5.g59, per-lane partial row-sum d with one peer fold in the epilogue (m32x8)

## Step 0 (feasibility)
- feasible: the only atom needed is rocdl.permlanex16, which _softmax.peer already uses. See the Step 0 reply.

## Step 1 (implement, build, correctness)
- Edit (rounds/015/op/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py only):
  - _softmax, speculative path: d_new = d_prev + local_sum. The per-tile peer(local_sum) is removed.
  - _softmax, slow path: d_new = corr*d_prev + local_sum. The per-tile peer(local_sum) is removed.
  - Epilogue: d_full[qt] = d + permlanex16(d) (sel 0x76543210/0xFEDCBA98, the same selectors as peer), built once per q-tile.
    It is used by both the 1/d normalization (with the d>0 guard) and the LSE log2.
  - The m32x2 and m16x8 bodies are untouched.
- 2026-09-28T07:36:49Z: cleared /tmp/flycache (FLYDSL_RUNTIME_CACHE_DIR in fa-g2, 1.4M) before the first build.
- Build attempt 1, run through fa-g2 as the default user (correct/): it compiled on the first try, so there are no build failures.
  - validation.py rounds/015/op: rc=2.
    - Correctness: 16/16 PASS at the spec 49.0 dB. The prod o/lse SQNR is 50.83/89.20 dB and the minimum is short_q full at 49.82 dB.
    - Determinism: 200/200 at fast.
    - Speed FAILED against beat: fast 1.0326, proxy 0.9386, prod 0.9683.
      This is the one-process cand/beat pair with the clock ramping (prod sclk 992->1144, proxy 1110->1312).
      It is a speed-vs-beat gate and not correctness. op/current is also below beat at prod. The cand-vs-current speed is step 2's measurement.
  - ut/test_correctness.py --determinism 200: rc=0, RESULT: PASS, with the same dB figures.
  - Caveat found: validation's determinism hash (o=9eb9b58556bf) is identical to op/current's in rounds 10-14.
    That is because `fast` dispatches the m32x2 body, which g59 does not touch.
    Validation's det check and the fast shape therefore do not exercise m32x8.
    Added m32x8check/: a bitwise diff vs op/current at fast/proxy/prod, 200x determinism at proxy (m32x8), and compile-only ISA gates.
- m32x8check/ (07:39:39-07:41:16Z, fa-g2, idle before and after):
  - fast: bitwise equal to op/current, as expected since it is the m32x2 body.
  - proxy: o differs in 271/16.8M elements, max 9.77e-4 = 1 bf16 ULP. lse max 1.9e-6.
    Determinism 200/200 for the m32x8 body.
  - prod: o differs in 2643/134M elements, max 1.95e-3. lse max 1.9e-6. Determinism 20/20.
  - The m32x8 source compiled in both runs is this round's edit: the ISA sha changed (prod 802003848f38 -> 61862e9d7fac).
    The plan predicted it would not be bitwise equal to op/current, and it is not.
- ISA gates (isa/metadata.txt, compile-only, 6 configs, op vs current):
  - VGPR spill 0 and scratch 0 everywhere.
  - SGPR spill identical to current.
  - .vgpr_count 448-455 (current 448-454; the prod value went from 454 to 455), within the 464 gate.
  - .text <= 0x7a80 = 31,360 B < 32,640, INST_PREF_SIZE 230-242 (the prod value is unchanged at 0x7a00).
  - v_permlanex16 count 20 -> 10: the per-tile sum peers are gone and the epilogue added 2.
  - These are gates, not pricing; pricing is on card in step 2.
  - Gate: PASS.
- Step 1 verdict: delivered and correct. There were 0 build failures.

## Step 2 (measure)
- Arms, all re-measured now:
  - r15 = rounds/015/op
  - r13 = incumbent (best_round) and fast champion; == op/current
  - r10 = proxy champion
  - r14 = prod champion
  - champ2 = a second on-disk copy of op/current (rounds/014/1-opt/arms/champ2), used as the A/A noise floor
- Ruler: benchmark.py defaults (blocked 9 + lead 4, median of 101, 8 s warmup), shapes fast, proxy, prod.
  - One process per session, 3 sessions with rotated arm order (measure/run.sh).
  - This is the plan's ruler. Separate per-arm processes would put the clock drift between arms into the ratio.
- Launched 2026-09-28 detached in fa-g2 with the measure/rc sentinel. The target GPU was idle (0%, no KFD pids) before launch. dmesg -W monitor armed.
- Results (measure/summary.txt, s1-3.json). All 3 sessions rc=0, 07:52:45-08:01:20Z, 0 new dmesg lines.
  - prod:
    - r15 vs r13: 1.0068 / 1.0051 / 1.0071. The A/A is 1.0000 / 0.9999 / 0.9999, so this is 3/3 above the floor.
    - This falls inside the plan's predicted x1.004-1.010.
    - r15 vs the prod champion r14: 1.0003 / 0.9993 / 1.0038, a median of 1.0003.
    - r15 is 1512.45 TF/s against a target of 1561.58, a ratio of 0.9685.
  - proxy:
    - r15 vs the proxy champion r10: 1.0055 / 1.0012 / 0.9991.
    - r15 vs r13: 1.0107 / 0.9941 / 1.0019.
    - The A/A spans 0.9894-1.0028, so there is no change beyond noise, and it is not below the A/A. This matches the plan.
  - fast: r15 vs r13 (the fast champion) 1.0001 / 1.0000 / 0.9985, with the A/A at 0.9939-1.0045. It is unchanged, as expected for the m32x2 body.
  - Regressions, all inside the 5% band:
    - fast s3 0.9985 vs r13
    - proxy s3 0.9991 vs r10
    - prod s2 0.9993 vs r14

## Step 3 (prediction check), measurements/pmc_cycles_clock/
- The headline prediction was prod wall x1.004-1.010 vs op/current. Step 2 measured x1.0068 / x1.0051 / x1.0071 (A/A about 1.0000), so it HELD.
- The plan's mechanism prediction was: cycles/XCD within x0.99-1.005, and ticks/ns up by at least half the gain (energy removed, so the clock rises).
- Measured under pmc (GRBM_GUI_ACTIVE/8, steady 10 dispatches, orders A and B):
  - g59 cycles/XCD x0.9821 (A) / x0.9858 (B). The A/A champ2 read x0.9986 / x0.9982.
    This is a 1.4-1.8% CYCLE CUT, outside the predicted band and about 10x the A/A.
  - g59 ticks/ns x0.9855 (A) / x1.0047 (B). The A/A read x0.9965 / x1.0025. There is NO clock rise, so the clock did not carry the gain.
  - So the gain came through cycles, not the clock, and the mechanism prediction FAILED in the opposite direction:
    - Conversion is partial: a cycles cut of about x1.016 became about x1.006 of benchmark wall, with the clock flat to slightly down.
    - This also contradicts the plan-era model that "cycle-only cuts do not convert at prod". g24 (cycles -6.6%, wall x0.9929) was one case, not a law.
  - The plan's falsified_if says "cycles drop >= 1% with wall flat: the per-tile VALU family is closed". Wall was not flat, so that branch is not triggered either.
- Likely cause (NOT measured; stall counters are closed on this chip):
  - The per-tile permlanex16 plus the dependent fadd sat on the softmax critical path between the S WMMA and the P->PV WMMA.
  - Removing it shortens the tile's dependent chain.
- Static fact, not pricing (isa_mnemonic_diff_prod.txt):
  - v_permlanex16 fell 20 -> 10.
  - The whole-kernel count rose 5024 -> 5062 (s_set_vgpr_msb +43, v_nop +23). So a naive op-count reading would even predict slower.
- Cell: faster_failed. The round is faster, but its recorded reason (energy removed, so a clock rise) is wrong.
  The observed mechanism is a cycle cut of about 1.6%. At prod, about 40% of it converts to wall.

## Step 4 (report)
- validation.py rounds/015/op (validation/, 08:04:49-08:06:46Z, fa-g2 default user, detached): exit 2.
  - Correctness: 16/16 PASS at 49.0 dB. The o/lse dB figures are identical to step 1's.
  - Determinism: 200/200 at fast (the m32x2 body; the m32x8 body is covered by m32x8check/).
  - Speed vs beat: FAIL. fast 1.0472, proxy 0.9491, prod 0.9714.
    - This is a one-process cand/beat pair with a ramping clock (prod sclk 994->1148).
    - It matches step 1's exit-2 verdict. The job's goal (beat at proxy and prod) is not reached.
- Summary for reflect:
  - Delivered and correct, with 0 build failures.
  - prod beats the incumbent r13 by +0.5 to +0.7% in 3/3 sessions, above an A/A of about 1.0000. It ties the prod champion r14 (1.0003).
  - fast and proxy are unchanged within the A/A.
  - Prediction check: the wall prediction held, but the mechanism prediction (energy -> clock) FAILED.
    - The gain is a 1.4-1.8% cycle cut with a flat clock, and about 40% of it converts at prod.
    - Cell: faster_failed.
  - Candidates for facts.md, for reflect to decide:
    - (a) A pure cycle cut at prod can convert partially to wall. The plan's "only energy converts" is too strong; g24 was issue-density-driven.
    - (b) Deferring the l-sum peer reduction to the epilogue is correct at 49 dB, and not bitwise equal (1 bf16 ulp).
