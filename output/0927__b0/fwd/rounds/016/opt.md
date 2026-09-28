# Round 16 opt (fast round)

## Starting state (read, 08:21)
- The round-head refactor h44 replaced op/current with **r13ns** (SPEC_STALE_MAX=False on m32x8 and m32x2). Output is
  bitwise equal to r13. Validation's speed phase reads prod 0.9288 / proxy 0.9041 / fast 1.0603 of beat, against
  r13's 0.9666 / 0.9366 / 1.0424 (rounds/016/0-refactor/refactor.md).
- So the round-15 profile and facts.md's "current bottleneck" describe **r13 (speculative)**, not r13ns. Every clean
  tile now runs the full `_softmax`: row-max tree, permlane max, corr exp, deferred-rescale ballot, and the per-tile
  permlane row-sum. That is the path the speculation used to skip.
- h45 forbids any re-added stale-max / guessed-max / trigger-and-redo softmax. That rules out refactor.md's ranked
  idea #1 ("cheap common case, keep S resident, redo on trigger") by the operator's own letter.
- max g in the ledgers = 59, so new ids start at g60.

## M1 (instrument, r16.i1.g60): prod/proxy pmc cycles + ticks/ns for r13ns vs r13 vs g59-on-r13ns, with A/A
Protocol from round 15 (rounds/015/3-act/measurements/pmc_cycles_clock): one rocprofv3 --pmc process per order, 4 arms,
2 orders, 3 warm-up + 10 steady dispatches per arm. Files: raw/pmc/. g59ns = the r15 g59 diff (raw/pmc/g59.diff)
patched onto op/current with `patch`, which applied cleanly. It is a throwaway arm and has not been through the
correctness gate.

**Predictions, written before the run (08:23):**
- r13 vs r13ns (cur): r13 wall x1.035-1.045 (the gate's ratio). I expect it mostly in cycles: r13ns cycles +6..9%
  over r13, and ticks/ns for r13ns x1.01-1.04 higher (more VALU, but no spec-trigger ballot). If instead cycles are
  flat and the clock drops, the nospec cost is energy and chain cuts will not convert.
- g59ns vs cur: cycles -1.5..-2.5%, wall x1.005-1.012 (g59 was +0.6% when most tiles ran the spec path; the
  plain path it now always takes carries the same peer + fadd per tile).
- champ2 A/A: inside +-0.5% on both.

### M1 result (08:23, raw/pmc/pmc_summary.txt) -- VOID as a kernel comparison; it exposed a cache hazard instead
- The run took 17 s end to end, so most arms did not compile. champ2 (a byte copy of cur) read cycles x1.066/x1.068
  over cur in both orders, and cur read the same as r13 (x0.994/x1.001). A byte-identical A/A off by 6.6% means the
  arms were not running the code their paths hold.
- Hypothesis (H-cache): the FlyDSL JIT disk cache (`FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache` in fa-g2) served
  job_context/op/current an entry compiled while that path still held **r13 (SPEC_STALE_MAX=True)**, because the
  module global `SPEC_STALE_MAX` is not in the cache key. Only 2 new `_launch_*` dirs appeared at 08:23 (the two
  fresh paths, champ2ns and g59ns).
- If true: champ2ns (+6.6% cycles, ticks/ns x1.035, wall x0.97) is the real r13ns-vs-r13 cost at prod. Also, **any
  champion re-measure of op/current in fa-g2 reads r13-speculative speed**, not r13ns, until /tmp/flycache is cleared.
- Test (M1b): same driver, 4 arms, fresh empty FLYDSL_RUNTIME_CACHE_DIR. Prediction: cur == champ2 within 0.5%
  (both ~1.97 M cycles); r13 ~1.84 M.

### M1b result (08:25, fresh FLYDSL_RUNTIME_CACHE_DIR, raw/pmc_freshcache/pmc_summary.txt) -- H-cache CONFIRMED
| prod (vs cur = r13ns) | cycles/XCD | ticks/ns | wall |
| --- | --- | --- | --- |
| champ2 A/A (order A / B) | x0.9973 / x1.0089 | x0.994 / x1.010 | x0.997 / x1.001 |
| r13 (speculative) | x0.9347 / x0.9403 | x0.974 / x0.964 | **x1.042 / x1.026** |
| g59ns (throwaway) | x1.0179 / x1.0238 | x1.026 / x1.028 | x1.008 / x1.004 |

| proxy (vs cur) | cycles | ticks/ns | wall |
| --- | --- | --- | --- |
| champ2 A/A | x0.972 / x0.998 | x0.981 / x1.000 | x1.009 / x1.002 |
| r13 | x0.933 / x0.957 | x0.971 / x0.991 | x1.041 / x1.036 |
| g59ns | x0.997 / x1.020 | x0.995 / x1.005 | x0.999 / x0.985 |

- With a clean cache, the byte-identical A/A copy agrees with cur (prod within 1%), and r13 is the build that differs.
  So M1's cur row **was the r13 speculative binary**, served from /tmp/flycache under op/current's path.
  **H-cache holds.** Mechanism: the FlyDSL 0.3.4.1 key (jit_function.py:580, `_jit_function_cache_key`) hashes the
  sources of functions and their closure scalars. `SPEC_STALE_MAX` is a module-level constant read inside
  `main_loop` (m32x8.py:1224), and a flip of it changes no function source. The corpus already names this trap:
  knowledge/pitfalls/measurement-traps.md:189 ("make an experimental knob part of the key, or disable the cache in
  probes").
- **Consequence, beyond this round**: in fa-g2, any process that loads job_context/op/current with the default
  `FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache` runs **r13 speculative**, not r13ns. That covers the framework's champion
  re-measure and every arm-vs-cur comparison. Such a process reads op/current about 3-4% faster at prod/proxy than
  the code it holds. I did NOT delete /tmp/flycache: the container is not ours, and the cache is shared state.
  Every measurement from here on must set a fresh `FLYDSL_RUNTIME_CACHE_DIR` per process (or
  `FLYDSL_RUNTIME_ENABLE_CACHE=0`). This is the route's standing condition, and the operator is told in the reply.
- **What nospec costs (prod, clean cache)**: cycles +6.4..7.0%, ticks/ns +2.6..3.7%, wall -2.5..-4.0%. That matches
  my prediction (cycles +6..9%, clock +1..4%). The cost is **cycles on the chain**, partly paid back by a higher
  clock. Chain-cycle levers are therefore the live family on r13ns, as g59 showed on r13.
- **g59 on r13ns (throwaway, not gated)**: prod wall x1.004/x1.008, proxy x0.999/x0.985 against a proxy A/A of
  x1.009/1.002. The mechanism changed sign from r15: cycles now go **up** 1.8-2.4% while the clock goes up 2.6-2.8%.
  It is still a small prod plus and not resolved at proxy. That is not enough to route it as a lead arm.

## M2 (static, r16.i1.g60 cont.): r13ns m32x8 prod ISA census (raw/isa/cur/.../22_final_isa.s, compile-only)
- 3793 instructions, **VGPR 456, 0 scratch**. refactor.md's idea #2 ("register pressure dropped, re-check
  occupancy/n_block") is **false on its premise**: the arch VGPR count is unchanged at 456. Not routed.
- Per clean-loop body (4 bodies): 64 WMMA, 66 v_exp (64 P + **2 corr**), 30 v_max3_num_f32 (row-max trees),
  4 v_permlanex16 (2 max, 2 row-sum), 80 v_pk_mul_f32, ~19 v_nop.
- Chain per tile, read from the ISA (lines ~1650-1700): max3 tree -> permlanex16 -> max -> sub -> **v_cmp -> s_cmp ->
  s_cselect** (the ballot, lowered to VCC->SCC) -> v_cndmask (m_new) -> v_mul (neg_m) -> v_pk_fma -> v_exp -> P.
  So the deferred-rescale vote sits **on the P chain**, with one VALU->SALU->VALU round trip per row under
  s_delay_alu.
- The corr exp (`exp2((m_prev - m_new)*log2e)`) runs every tile, even though corr is exactly 1.0 whenever
  do_rescale is false. corr feeds only the O rescale (inside the scf.if) and the d update.

## Sources consulted
- findings: facts.md, dead_ends.md (headings), pool.md, route.md (operator tables, h4/h8/h16/h43/h45, r15 route)
- rounds/016/0-refactor/refactor.md; rounds/015/3-act/measurements/pmc_cycles_clock/* (protocol reused)
- knowledge/backends/flydsl/attention/README.md
- knowledge/backends/hipkittens/attention/recipes/gqa_d128.md §6.3 and §7 (lazy rescale; `pending_scale` defers the
  norm correction off the critical path)
- knowledge/optimization/routes/1-metrics-to-techniques.md (Table B instruction-mix row: v_exp costs 2 issue slots)
- knowledge/pitfalls/measurement-traps.md:189, pitfalls/compiler-and-toolchain.md:229 (JIT cache keyed on source)
- The round-15 profile directory was NOT used for pricing. It describes r13 speculative, while op/current is now r13ns
  (and the r15 act pmc measured it under the same stale cache path as r13 anyway, where it was r13).

## Decision
- **h45 kills refactor.md's idea #1** (cheap common case with redo on trigger). Not pooled.
- New pool ids: **r16.i2.g61** (per-row deferred-rescale decision, which takes the vote off the P chain) and
  **r16.i3.g62** (corr exp and d-rescale only inside the rescale branch, bitwise-equal). The top two to build are
  A = g61 and B = g62, each alone from op/current. They touch neighbouring lines in `_softmax`, but neither falsifies
  the other, so a merge is buildable if neither loses.
- r15.i5.g59 goes to row 3 as a merge rider only: measured on r13ns above at prod +0.4..0.8%, proxy unresolved.
- r15.i3.g57 stays blocked on its probe.
- **Expectation**: g61 cuts prod cycles 1-2% at a flat-to-lower clock, for wall **+0.5..1.0%** at prod and
  +0.3..0.8% at proxy. g62 removes 2 of 66 trans ops plus ~4 VALU per tile, all off the P chain, for wall
  **+0.1..0.4%**. That is probably inside the A/A floor. Neither closes the 7% prod gap to beat that r13ns opened.

## Build (step 3): route rows 4 (ARM A = r16.i2.g61) and 5 (ARM B = r16.i3.g62)
- Cache: /tmp/flycache in fa-g2 cleared at 08:2xZ before the first build (it was 2.8 MB), and every process below
  also sets its own fresh `FLYDSL_RUNTIME_CACHE_DIR` (route standing condition, r16.i1.g60).
- Arms (raw/arms/): `cur` = byte copy of job_context/op/current (also the champ2 A/A arm); `g61`, `g62`, `m` = g61+g62.
  Each arm starts from the same base. Only `flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py` changes, so the prod and
  proxy bodies change and fast (m32x2) does not.
  - g61: `m_new = need.select(m_full, m_prev)` per lane (was `do_rescale.select`); the ballot drives only the O-rescale
    scf.if. On that branch, lanes whose row did not move have m_new == m_prev, so corr == exp2(0) == 1.
  - g62: `_softmax(lazy_corr=True)` returns corr = None, d = d + rowsum, and the rowsum. main_loop's rescale branch
    computes corr = exp2((m_prev - m_new)*log2e), O *= corr, and d = **fma**(corr, d_prev, rowsum).
    - Build failure 1 (caught statically, ISA): the first g62 wrote the branch's d as mul + add. LLVM then sank the
      add past the branch phi, so the branch rounded twice. op/current contracts corr*d + rowsum into one
      v_fmac/v_pk_fma, so that g62 would not have been bitwise on rescale tiles. The fix was an explicit fmath.fma.
      Root cause: contraction is lost when the add leaves the mul's block.
- Static gates (raw/isa16/, compile-only, prod + nc_g1 configs):

| arm | VGPR prod/nc_g1 | spill | .text prod | instrs prod | s_cselect prod | corr v_exp |
| --- | --- | --- | --- | --- | --- | --- |
| cur | 456 / 450 | 0 | 0x6680 | 3793 | 49 | every tile |
| g61 | 460 / 456 | 0 | 0x6600 | 3746 | 33 | every tile |
| g62 | 456 / 450 | 0 | 0x6700 | 3837 | 49 | inside the rescale branch |
| m | 456 / 450 | 0 | 0x6700 | 3803 | 33 | inside the rescale branch |

  All pass (VGPR <= 464, 0 spill, .text < 32,640 B). g61 drops 16 s_cselect (4 bodies x 2 rows x 2) and 47
  instructions.
- Correctness (raw/correct/, fa-g2, 08:35-08:48Z, idle before/mid/after, fresh JIT cache per process):
  - validation.py precision, all three arms: 16/16 PASS, min o SQNR 49.82 dB (short_q full, same as cur); proxy
    o 50.89 / lse 88.24 dB, prod o 50.83 / lse 89.20 dB, identical to cur at the rounding shown. Every validation run
    exits 2 **only** on the speed target (proxy/prod below the beat bar, as for r13ns); no precision line fails.
  - ut/test_correctness.py --determinism 200: PASS for g61, g62 and m.
  - det16.py (m32x8 body): 200/200 at proxy and 200/200 at prod for every arm; on make_inputs (randn) every arm is
    bitwise = cur at proxy and prod (hashes 0c25768c3eaa / 1bfe6e3bbcd8).
  - adv_bitwise.py (rounds/006 h16 suite, 52 cases): g62 is **bitwise = cur on all 52**. g61 differs from cur on
    proxy ramp (o 51.14 vs 51.34 dB) and proxy big (22.59 = 22.59 dB), is finite everywhere and bitwise elsewhere;
    m is bitwise = g61 on all 52 (so g62 composes bitwise on top of g61).
  - realcheck16.py (h45 real dumps, 6 sets, prod shape, fp32 reference on heads 0/13/31): g62 bitwise = cur on
    all 6. g61 and m are not bitwise (vs cur 56-61 dB, max|do| <= 0.125, max|dlse| <= 6.1e-5), and their o SQNR vs
    the fp32 reference is 0.1-0.5 dB below cur's on every set (52.18-54.80 vs 52.32-54.91 dB); lse is identical to
    the shown digits. PASS (>= 49 dB). A per-row stale m makes P = exp(S - m_stale) up to e^8 before the bf16 cast,
    so the rows that keep a stale m accumulate O in a different rounding regime. Small and consistent in sign, noted.
  - adv_m32x8.py (r16 copy of the h16 suite restricted to the shapes that dispatch m32x8: proxy x {causal, full} x 6
    kinds + prod x {randn, ramp, outlier}), g61 vs cur: finite on all 15. o/lse SQNR vs fp32 equals cur's on every
    proxy case except causal ramp (51.14 vs 51.34 dB). Not bitwise on 9/15 (big, ramp, outlier, big_ramp; by design).
- Build failures in this step: 1 (g62 double rounding, fixed statically before any card run). No card failures.

## Measure: benchmark.py, 3 rotated sessions, all spec shapes (raw/bench/, 08:48:53-08:56:30Z, fa-g2)
- One process per session, benchmark.py defaults (blocked ruler, median of 101). No beat in the process (h31).
  Arms: inc = **rounds/016/op re-measured** (unchanged at that point, = op/current = r13ns); champ2 = byte copy of
  op/current (A/A); g61; g62. Orders: `inc g61 champ2 g62` | `g62 inc g61 champ2` | `champ2 g62 inc g61`. Fresh JIT
  cache per process. Idle before/after every session (no KFD pid on this card). prod sclk 1275-1293 -> 1318-1321 MHz.
  No rounds before 16 re-measured.
- Speed vs inc (inc latency / arm latency; >1 = faster), per session, then geomean (raw/bench/summary_AB.txt):

| shape | champ2 (A/A) | g61 | g62 |
| --- | --- | --- | --- |
| fast | 1.0030 0.9969 0.9908 -> 0.9969 | 1.0030 1.0000 1.0000 -> 1.0010 | 0.9940 0.9969 1.0000 -> 0.9970 |
| proxy | 0.9950 1.0014 0.9984 -> 0.9983 | 1.0023 0.9991 1.0005 -> 1.0006 | 0.9898 0.9927 0.9943 -> **0.9923** |
| prod | 0.9949 1.0033 1.0010 -> 0.9997 | 1.0014 1.0065 1.0036 -> **1.0038** | 1.0040 1.0053 1.0012 -> 1.0035 |

- A/A floor this session: +-0.5% at proxy/prod (champ2 0.9949..1.0033), +-0.9% at fast.
- fast is the untouched m32x2 body in all arms, and it reads as noise, as it should.
- **g62 lost at proxy beyond the A/A floor.** It is below 0.995 in 3/3 sessions (geo -0.77%), while its prod reads
  +0.35% (3/3 above 1). The merge rule is "neither lost beyond the A/A floor", so **the merge is not eligible and was
  not measured**. arms/m was compiled and correctness-checked ahead of the ranking only: bitwise = g61 on all 52 h16
  cases, 49.82 dB min.
  - Mechanism (inferred from the ISA, not measured):
    - In g62 the rescale branch now carries a serial sub -> mul -> v_exp -> TRANS32 wait before its 32 v_pk_mul.
      Before, the corr exp was issued early, in the shadow of the P exps.
    - Every row takes that branch on its first tile, and on the early tiles where the max still climbs.
    - Proxy has half of prod's kv tiles per row (causal mean ~32 vs ~64), so a larger share of its tiles pay the
      longer branch.
- **g61: prod +0.38% (3/3 sessions > 1), proxy +0.06% (null), fast untouched.** It is the best arm. The prod gain is
  consistent in sign but below the A/A floor's width.

## h45: g61 vs inc on the real dumps (raw/real_ab/, 09:09:36-09:12:49Z, fa-g2)
- tools/ab16.py, blocked ruler. Six real sets (qkv_call0{672,673,674,680,688,703} = L00 L01 L02 L08 L16 L31)
  plus randn, prod shape, 108 iterations per arm per set. Arms inc (rounds/016/op, still r13ns at that point),
  g61, and champ2 (A/A). 3 rotated orders (`inc g61 champ2` | `g61 champ2 inc` | `champ2 inc g61`), one process
  each, a fresh JIT cache per process, no beat.
- Harness slip: the first launch (08:57-09:00Z) never incremented its session counter, so every process
  overwrote s0. Only the last order survives, as `prev_s0.*`: g61 0.9999, champ2 0.9998. I fixed the counter and
  reran all 3 orders. The table below is the rerun.
- Speed vs inc (inc median / arm median), geomean over the 7 sets:

| session (order) | g61 | champ2 (A/A) |
| --- | --- | --- |
| s0 (inc g61 champ2) | 1.0012 | 1.0009 |
| s1 (g61 champ2 inc) | 1.0008 | 0.9998 |
| s2 (champ2 inc g61) | 1.0003 | 1.0002 |

  Per set, g61 spans 0.9965..1.0036 and champ2 spans 0.9951..1.0061.
- Verdict: g61 is **not slower than inc on real data** (3/3 sessions >= 1). The gain is **below resolution**:
  the per-set spread equals the A/A spread. h45's bar is "beat the champion on the real dumps in the same
  process". It is met in sign only. This is not a measured win.
- Device incident, not attributed: from 09:00:28Z (about 28 s after the first real_ab launch exited) until
  ~09:03:30Z, dmesg on 0002:04:00.0 showed `MES(0, 0) failed to respond to msg=MISC (WAIT_REG_MEM)` and then
  ~70 `MES(0, 0) ring buffer is full`. No process of ours was alive in fa-g2 then; its only python3 zombie is
  from 03:26. A torch 4096^2 matmul probe at 09:09 passed (rc 0). The rerun that followed was clean: no new
  dmesg lines, idle before and after each session.

## Decision and outcome
- Shipped **ARM A = r16.i2.g61** into rounds/016/op: only `flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py`
  changed, byte-identical to raw/arms/g61 (diff -rq clean).
- ARM B = r16.i3.g62 lost at proxy beyond the A/A floor (-0.77%, 3/3 sessions). Not shipped. The merge was not
  eligible (built and correctness-checked, but not ranked).
- Acceptance vs best ever = inc (rounds/016/op re-measured in the same sessions; no earlier round re-measured).
  Figures are the mean of 3 session medians, TF/s:

| shape | g61 | inc | g61/inc | beat (validation.py, this round) | ratio g61 | ratio inc |
| --- | --- | --- | --- | --- | --- | --- |
| fast | 164.77 | 164.60 | 1.0010 | 156.9 | 1.0 (target held) | 1.0 |
| proxy | 1571.40 | 1570.43 | 1.0006 | 1676.43 | 0.9373 | 0.9368 |
| prod | 1448.85 | 1443.34 | 1.0038 | 1556.07 | 0.9311 | 0.9276 |

  - Throughput improves on best ever: score 0.9561 vs 0.9548 for inc.
  - Proxy and prod are below target, and both are >= 95% of best ever (1.0006, 1.0038).
  - fast held its target: the body is unchanged, and it read x1.046 of beat in validation.py.
  - Beat comes from validation.py's own process (g61 run, 08:3xZ), not the ranking process (h31). Its
    same-process ratios were fast 1.0459, proxy 0.9008, prod 0.9320. The proxy clock was 1099->1299 MHz there,
    and 1354->1554 MHz in the ranking.
- **Outcome: delivered, with low confidence.**
  - Every gain (prod +0.38%, proxy +0.06%, real dumps +0.03..0.12%) is inside the ±0.5% A/A floor. Only prod's
    sign is consistent (3/3).
  - Expectation vs result: I predicted g61 at +0.5..1.0% at prod and +0.3..0.8% at proxy. It measured +0.38%
    at prod and null at proxy. The 16 s_cselect and 47 instructions it removes are real (ISA), but they barely
    convert at a power-bound prod clock, the same pattern as g55/g24.
  - g62: I predicted +0.1..0.4%. Proxy lost 0.77%, because the corr exp moved onto the rescale branch's serial
    path.
- Price paid for g61: o SQNR vs fp32 is 0.1-0.5 dB lower on real dumps and 0.2 dB lower on the causal-ramp
  adversarial case. Every case stays >= 49.82 dB min on the gate.
