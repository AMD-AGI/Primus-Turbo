# Round 17 opt (fast round)

## Starting state (read, ~09:40Z)
- op/current = r13ns (h44). Validation in-process vs beat: fast 1.046, proxy 0.901, prod 0.932 (r16 act.yaml).
  Ranking-process figures r16: prod 1443 TF/s vs beat 1556 (0.93), proxy 1570 vs 1676 (0.94).
- Round 16 not accepted. Both chain edits (g61 -47 instr, g62 corr exp into branch) failed to move wall past A/A.
  facts: "What the +7% nospec cycles are made of is unmeasured. Count cycles before building another chain edit."
- Pool open: r1.i4.g04 (fast split-KV, fast already above beat, needs h16 ruling), r15.i3.g57 (2-tile TDM coalescing,
  blocked on a probe; its K/V premise is clock-priced per g55), r16.i3.g62 (lost proxy, still listed as open).
- r15 profile (rounds/015/1-profiling/6-bound-analysis/analysis.md) read: prod = power/clock (medium), 1.87 M cycles/XCD
  at ~1280 MHz implied; proxy = tail latency (SPI busy 0.879) with clock as co-factor. No stall/VALU counters exist.
  The r15 profile is of r13-spec; op/current is r13ns (+6.4..7.0% cycles at prod, r16 M1b), so its cycle count is stale.
- Card: fa-g2 = renderD144, GPU use 0%. A foreign process (lab-kdq kbench.py prod, PID 3992128) is running in fa-g0 on
  renderD128 (a different card, physical 0). It shows up in fa-g2's `rocm-smi --showpids` list because that list is
  node-wide. Not on our device; the node's power is shared, which is noted, not controlled.

## Corpus read (step 3)
- backends/flydsl/attention/README.md, recipes/hd128.md §5-6 (b5: gfx950 FlyDSL fwd computes the **row sum on the
  matrix pipe**, MFMA 16x16x32 with a ones A-operand; costs 5.88% of the forward's matrix FLOP, "confirmed present,
  not re-priced").
- backends/hipkittens/attention/recipes/gqa_d128.md §6.3 (lazy rescale threshold: never fires on randn at thr 8;
  we already have it), §8 (pipeline structure).
- optimization/routes/1-metrics-to-techniques.md (see below).

Mechanism taken (not constants): the per-tile row sum is VALU work on the softmax chain (16 v_pk_add per wave for the
two rows + permlanex16 + fadd per row); on the matrix pipe it becomes 2 WMMA per row per tile (+4 WMMA on 64, i.e.
+6.25% WMMA), needs **no cross-lane step** (the ones-matrix contraction runs over all 32 kv of the B operand, so every
lane holds the full row sum for q = l%16), and sums the bf16 P that PV actually consumes.

## Measurement plan (instruments; throwaway arms, never shipped)
Arms in raw/arms/ (cur = rounds/017/op copy, champ2 = byte copy for A/A, g61 = rounds/016/1-opt/raw/arms/g61,
patch script raw/arms/patch.py):
- **nosum** (THROWAWAY, wrong output): row-sum tree and its peer removed (d += p[0]). Prices the VALU row sum.
- **nomax** (THROWAWAY, wrong output): local max tree removed (row_max from s[0]; peer + vote kept). Prices the
  per-tile max tree, i.e. the bulk of what nospec (h44) put back.
- **wsum** (r17.i1.g63, a real candidate): row sum as 2 chained WMMA per row with a ones A-operand; d += acc[0];
  no peer on the sum. Output not bitwise (sums bf16 P, in WMMA order).
- **g61**: the r16 arm, to take the pmc A/B its dead_ends entry asked for (did its 47 instructions cut cycles?).

Static gates (raw/isa/, compile-only, prod cfg): all 0 scratch.
| arm | VGPR | instrs | v_wmma | v_pk_add_f32 | permlanex16 | v_max* |
| --- | --- | --- | --- | --- | --- | --- |
| cur | 456 | 4130 | 256 | 126 | 16 | 123 |
| nosum | 453 | 3914 | 256 | 0 | 8 | 128 |
| nomax | 456 | 3996 | 256 | 126 | 16 | 5 |
| wsum | 451 | 4242 | 272 | 0 | 8 | 128 |

**Predictions, written before the run (pmc, prod, vs cur):**
- A/A champ2 within +-1% cycles, +-0.5% wall.
- nosum: cycles -2..-4%, clock flat to +2%, wall +1..+3%. If wall < +0.7%, the row sum is not worth moving, and wsum
  cannot pay either.
- nomax: cycles -3..-5% (the max tree + the chain it gates sit before exp), wall +1.5..+4%. nomax > nosum in wall
  would say the max half of the chain is the bigger term (it is on the S->P path; the sum is off it).
- wsum: between cur and nosum; wall +0..+2%. Could lose if 16 extra WMMA cost energy (clock down).
- g61: cycles -0.5..-1%, wall +0.2..0.5% (r16 read +0.38% wall).
- What would change my mind: nosum/nomax both inside A/A => the chain VALU is not what prod pays for on r13ns, and the
  whole h10/h11 family should leave the route.

## pmc probe results (raw/pmc/, pmc_summary.txt; orders A and B reversed; one process each; fresh JIT cache; card idle before and after)
cycles = GRBM_GUI_ACTIVE/8 per XCD; wall = cur dur / arm dur (>1 = faster); ticks/ns = clock proxy.

| arm | prod cycles A / B | prod wall A / B | prod ticks/ns A / B | proxy cycles A / B | proxy wall A / B |
| --- | --- | --- | --- | --- | --- |
| nosum (throwaway) | x0.960 / x0.973 | x1.034 / x1.035 | x0.993 / x1.007 | x0.960 / x0.973 | x1.041 / x1.024 |
| nomax (throwaway) | x0.971 / x0.977 | x1.019 / x1.019 | x0.990 / x0.996 | x0.968 / x0.979 | x1.027 / x1.017 |
| wsum (g63) | x0.990 / x0.995 | x1.011 / x1.004 | x1.000 / x0.999 | x0.986 / x0.997 | x1.011 / x1.002 |
| g61 (r16) | x0.992 / x1.001 | x0.994 / x0.996 | x0.987 / x0.997 | x0.995 / x0.993 | x1.003 / x0.999 |
| champ2 A/A | x1.000 / x1.006 | x0.993 / x0.998 | x0.993 / x1.004 | x0.995 / x1.007 | x1.004 / x0.995 |

Reading against the predictions:
- **Chain VALU converts on r13ns at a flat clock.** nosum: cycles -2.7..-4.0%, clock flat, wall +3.4..3.5% (upper end of
  the +1..3% prediction). So the per-tile row sum (tree + peer) is the largest priced term on the chain, and ~3.4% at
  prod is the ceiling for moving it. nomax: -2.3..-2.9% cycles, wall +1.9% -- below nosum, which falsifies my
  "max is the bigger half" guess (the max tree is already v_max3 and partly overlapped by the QK WMMA tail).
- **wsum (g63) keeps about 1/3 of the row-sum ceiling**: prod wall +0.4..1.1%, proxy +0.2..1.1%, clock flat (the 16
  extra WMMA did not cost clock). Both orders beat the A/A spread, but only barely at order B. Where the rest went:
  +112 instructions (4242 vs 3914 in nosum), i.e. the WMMA hazard nops and the ones/pf shuffles are back on the
  chain; d += acc[0] now waits on a WMMA result latency.
- **g61 answer (its dead_ends asked for it)**: cycles x0.992 / x1.001, wall x0.994 / x0.996 -- inside the A/A. The 47
  removed instructions are not on the critical path; the vote hop was not what nospec paid for. The nospec cost is the
  tree reductions (sum ~3.4%, max ~1.9% wall), not the vote.
- rocprofv3 kernel-trace/--stats were not used (broken for FlyDSL, r9.i3.g22); pmc counters only.

## Ranking (raw/bench/, benchmark.py blocked ruler, 3 rotated sessions, one process each, fresh JIT cache, no beat; 09:25-09:33Z)
Speed vs cur (>1 = faster). Card GPU use 0-5% on the before/after smi reads (the after-read of session n is the
before-read of n+1, so the 2-5% readings are most likely our own process's tail).

| arm | proxy s1 / s2 / s3 | proxy geo | prod s1 / s2 / s3 | prod geo |
| --- | --- | --- | --- | --- |
| champ2 A/A | 0.988 / 0.996 / 0.993 | 0.992 | 0.999 / 1.001 / 1.005 | 1.002 |
| wsum (g63) | 1.003 / 0.988 / 1.013 | 1.001 | 1.006 / 1.012 / 1.009 | **1.009** |
| px2 (throwaway) | 0.637 / 0.618 / 0.634 | 0.630 | 0.999 / 1.001 / 1.001 | 1.000 |
| px16 (throwaway) | 0.712 / 0.726 / 0.724 | 0.721 | 0.999 / 1.005 / 1.001 | 1.002 |

- **wsum (r17.i1.g63)**: prod x1.009 with 3/3 sessions above the A/A maximum (1.0048). That matches the pmc read
  (+0.4..1.1%, clock flat). Proxy x1.001 is inside a noisy A/A (champ2 0.988..0.996 this run). It is the first
  softmax-chain edit since h44 to move prod beyond the A/A, and it recovers about 1/4 of the 3.4% nosum ceiling.
- **Proxy grid gate is dead.** Swapping the proxy grid (512 WGs, 2/CU) onto m32x2 costs -37%, and onto m16x8 -28%.
  The proxy gap is not fixed by giving each WG fewer queries: m32x8's 8-wave, K/V-shared WG is worth far more than
  the tail it creates. Prod reads 1.000 on both, which checks that the gate only touched proxy. Not pooled; goes to
  dead_ends via reflect.

## Precision (raw/val/wsum.txt, validation.py on the wsum arm, fresh cache, card 0% -> 1%)
16/16 correctness PASS at the spec's 49 dB, min o 49.82 dB (short_q full, same as g61/r15), lse >= 81.2 dB;
determinism 200/200 bitwise. rc 2 comes only from the speed-vs-beat bar (proxy 0.924, prod 0.942 in-process with beat),
which op/current also fails. The in-process beat ratios are not a ranking (h31).

## Choice
1. **r17.i1.g63** (wsum): measured, correct, prod +0.9% beyond A/A. Route row 4, port as-is.
2. **r17.i2.g64** (carried ones d-tile in PV): the same mechanism with its result off the per-tile chain. It targets
   the ~2.5% of the nosum ceiling that g63 leaves. Route row 5, built from the same base, keep the better one.
Not pooled: the proxy grid gate (px2/px16), which is dead (-37/-28%) and a dead_ends item for reflect.
Deep-born not taken: g57 (blocked on its probe, clock-priced premise), g04 (fast above beat). g62 is subsumed by g64.

## Explored / consulted
backends/flydsl/attention/README.md; backends/flydsl/attention/recipes/hd128.md (b5, ones-operand row sum);
backends/hipkittens/attention/recipes/gqa_d128.md (§6.3, §7, §8); optimization/routes/1-metrics-to-techniques.md;
rounds/015/1-profiling/6-bound-analysis/analysis.md; findings facts/dead_ends/pool/route.
Raw: raw/arms (patch.py), raw/isa, raw/pmc (pmc_summary.txt), raw/bench (agg.py), raw/val.

Card health: dmesg for 0003:04:00.0 is clean during our runs (the last fault, an SDMA0 UTCL2 permission fault, is dated 2026-09-27 21:52Z, before this round). No bench/validation/pmc process is left in fa-g2, and GPU use is 0% at the end.

# Build step (route row 4, r17.i1.g63)

## Build
- Read rounds/016/0-refactor/refactor.md first. The structure is r13ns: one `_softmax` body per sub-loop, and the
  m32x8 kernel serves proxy and prod. g63 edits only m32x8's `_softmax`. Fast dispatches to m32x2 and is untouched.
- Applied `1-opt/g63.diff` (cur -> wsum arm, one file: flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py). The working
  copy is byte-identical to the arm measured at opt, so there was no re-derivation and no fix-ups (1 attempt, 0 failures).
- Compile cache cleared before the first build: /tmp/flycache is absent, old /tmp/flyc_r17* was removed, the working
  copy's __pycache__ was removed, and each process gets a fresh FLYDSL_RUNTIME_CACHE_DIR.
- Note: rounds/016/op != op/current. rounds/016/op holds r16's shipped g61, which was not promoted; op/current is
  r13ns. Both are re-measured below.

## Correctness (raw/build/)
- ut/test_correctness.py --impl rounds/017/op: RESULT PASS (16/16), rc 0.
- validation.py rounds/017/op: 16/16 PASS, min o 49.82 dB (short_q full), prod o 50.91 / lse 88.92, proxy
  50.98 / 87.82; determinism 200/200 (same digest as op/current: fast runs m32x2). rc 2 only from the speed-vs-beat
  bar (in-process with beat: fast 1.078, proxy 0.904, prod 0.938), which op/current also fails.

## Measure (raw/measure/, benchmark.py blocked ruler, 3 rotated sessions, fast+proxy+prod, 09:39:45-09:47:21Z)
Arms: cand = rounds/017/op, cur = job_context/op/current (r13ns), r16 = rounds/016/op (r13ns + g61, the round-16
build, not promoted), champ2 = byte copy of op/current (A/A). One process per session, fresh FLYDSL_RUNTIME_CACHE_DIR
each, no beat (h31). No round before 16 was measured. GPU use 0-5% before/after (the 5% is our own previous session's
tail); dmesg clean for 0003:04:00.0 over the window. Figures are the mean of the 3 session medians (TF/s).

| shape | cand | cur | r16 | champ2 | cand vs best (r16/cur) | sessions vs best | beat (in-process, validation.py this round) | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fast | 166.99 | 166.82 | 166.81 | 166.73 | x1.0010 (cur) | 0.997 / 1.003 / 1.003 | 154.6 | 1.0 (capped) |
| proxy | 1567.52 | 1561.13 | 1566.79 | 1558.88 | x1.0005 (r16) | 1.0005 / 0.9995 / 1.0014 | 1665.8 | 0.9410 |
| prod | 1456.90 | 1443.53 | 1445.37 | 1442.60 | **x1.0080** (r16) | 1.0063 / 1.0084 / 1.0092 | 1553.3 | 0.9379 |

vs op/current: prod x1.0093 (3/3 1.0076..1.0113), proxy x1.0041, fast x1.0010; champ2 A/A prod 0.998..1.001.
Score 0.9596 (op/current re-measured: 0.9555; r16: 0.9583).
Acceptance arithmetic as I read it: throughput is above the best on the current structure on every shape (prod
+0.8% beyond A/A, proxy/fast within noise but not below); proxy and prod are below target and are >= 95% of their best
(>= 100%); fast stays above target. Python decides.
Prediction held: opt predicted wsum prod +0..2% and read +0.9%; it re-read +0.8..0.9% here.

## Not executed
Route row 5 (r17.i2.g64) was not built: row 4 took the round, and g64 changes the loop-carried state (yield list,
init, epilogue d). It stays the next row, with g63 as the base to beat.
