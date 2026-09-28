# Round 7 -- fast round, design act (read, look, choose, route)

Base: op/current = round 6 champion (r4 + r5.i2.g14); rounds/007/op is byte-identical to it
(`diff -rq` clean, pycache aside). Container fa-g0, physical GPU 0. Before the first launch: `rocm-smi
--showuse` 0%. The two KFD PIDs rocm-smi lists belong to fa-g2 / fa-g3 (docker cgroups
0f78cefe / 34d302ae, GPU-2/3 lab A/B runs). They are other cards, per h29. A dmesg monitor was armed.
It replayed the known 10:05 page faults on 0001:04:00.0 from hours earlier; nothing new appeared.

## Read
findings/{facts,dead_ends,pool,route}.md; rounds/006/{1-opt/opt.md, 2-reflect/reflect.md, gate.log};
rounds/005/1-profiling/profiling_summary.md and 2-kernel-profiling/analysis.md (the I$ rows).
Round 6's reflect names the A+B merge (g14 + g16) as "the obvious candidate". Round 5's profile has
SQC_ICACHE miss/REQ ~2e-4 at fast and 0 at proxy, but only for current ALONE. Nobody has read the
counter with beat interleaved, which is the configuration the fast penalty lives in. Every
code-size candidate in the pool rests on that reading.

## Survey
- `rocprofv3 --kernel-trace --stats` on a minimal driver (raw/survey_drv.py: beat then current,
  fast, 50 pairs). **rc 0, no crash** -- the round-4/6 crash is benchmark.py-specific (its exit path),
  not rocprofv3's. But the output dir is EMPTY: kernel-trace wrote no file ("output generation
  0.0009 s"). Round 5's `--pmc` form does write, so the kernel list comes from the pmc run below.
- **I$ with beat interleaved** (raw/icache_pmc.sh -> raw/icache_{agg,seq}.txt; `--pmc SQC_ICACHE_REQ,
  SQC_ICACHE_MISSES,SQC_ICACHE_MISSES_DUPLICATE,GRBM_COUNT`, fast, 30 calls, r6cur / r4 / g16 x alone /
  beat). Steady-state FlyDSL misses per dispatch = **0 in every cell, with or without beat**. Beat itself
  reads ~80-100, rising to ~180. So pmc serialisation hides the beat-presence penalty, as round 1 saw. The
  counter can neither confirm nor refute the eviction hypothesis. What it does show is cold behaviour:
  the first dispatches miss a constant **1332 (r6cur) vs 32 (r4, g16)**.
- **Why 1332: the prefetch cap** (raw/kd_rsrc3.py on the flydsl cache ELF -> raw/inst_pref_size.txt,
  raw/code_bytes.txt). COMPUTE_PGM_RSRC3.INST_PREF_SIZE (bits 11:4, x128 B):

  | build | code B | INST_PREF_SIZE | covered |
  | --- | --- | --- | --- |
  | r6cur | 38 956 | **255 (field max)** | 32 640 B; ~6.3 KB tail NOT prefetched |
  | r4 | 25 744 | 202 | all |
  | g16 | 17 688 | 139 | all |

  g14 was the first build past the cap. dead_ends r3.i1.g07 killed prefetch only "for kernels under
  32 KB", so this is new ground, not a reopened dead end.

## Probes (compile-only, COMPILE_ONLY=1 ARCH=gfx1250, scratch ~/.cache/op-evolve-r007-scratch)

| build | bytes | VGPR | scratch | wmma | v_exp | max3 | under cap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r6cur | 38 956 | 448 | 0 | 384 | 520 | 120 | no |
| r4 | 25 744 | 456 | 0 | 256 | 264 | 120 | yes |
| g16 | 17 688 | 444 | 0 | 128 | 132 | 60 | yes |
| g17 (one warp type) | 22 192 | 442 | 0 | 192 | 260 | 60 | yes |
| g18 (spec on clean tiles only) | 30 780 | 454 | 0 | 320 | 392 | 120 | yes |

LO and HI differ only at :1162 (the K ds_load vs TDM prefetch issue order); `_named_barrier_pair` is a
no-op. So g17 changes the op order, not the math. Diffs: raw/probe_g17.diff, raw/probe_g18.diff. Neither was
timed on the card: that is the build step's work, under h31.

## Decision
- r7.i1.g17 (arm A) and r7.i2.g18 (arm B), both from op/current, disjoint lines. They outrank round 6's
  suggested g14+g16 merge because g16 lost prod -1.7% to a per-tile branch that neither of them has.
- **Expectation**: g17 fast +3..6% no-beat (more in gate form), prod/proxy within the floor. g18 fast
  +1..4%, prod -0.2..-0.4%. Falsifier: if neither gains at fast, the cap tail was not the fast loss, and
  the beat penalty is not size-driven below 32 KB either -> row 4.
- Route rewritten (findings/route.md). Pool: +r7.i1.g17, +r7.i2.g18; update lines on r1.i3.g03, r6.i1.g16.

## Corpus consulted
knowledge/backends/hipkittens/attention/recipes/gqa_d128.md; knowledge/arch/gfx1250/profiling-surface.md;
listed only: knowledge/backends/flydsl/attention/ (README, techniques, dead-ends, recipes hd128/hd64),
knowledge/optimization/techniques/, knowledge/pitfalls/.

After the round: `rocm-smi --showuse` GPU 0 at 0%; no detached job left running; dmesg monitor quiet.

# Step 5 -- build (route rows 2 and 3)
- Compile cache cleared before the first build: fa-g0 FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache (4.9 MB) emptied;
  /root/.flydsl holds only autotune/.
- Arm A = r7.i1.g17 in the WORKING COPY (rounds/007/op, = raw/probe_g17.diff). Arm B = r7.i2.g18 in scratch
  ~/.cache/op-evolve-r007-scratch/g18 (= op/current + raw/probe_g18.diff), never on top of A.
  Both built first try (compile-only census above).
- Correctness (raw/correct.sh -> raw/{ut,adv}_{A,B}.txt, raw/val_A.txt):
  - A (g17): ut rc 0 (all PASS, min o 49.82 dB, short_q full, same as the champion); determinism 200/200,
    hash o=9eb9b58556bf lse=bbf23654e600 = **op/current's hash**; adversarial **52/52 bitwise == op/current**.
  - B (g18): same: ut rc 0, min 49.82 dB, the same hash, and 52/52 bitwise == op/current.
  - validation.py on the working copy while it held A: exit 2. Correctness and determinism PASS; the speed
    gate FAILs (geomean 0.7637 vs beat), as in rounds 4 and 6.
  - g17-HI throwaway (scratch g17hi, all waves in HI order): ut rc 0, the same hash, 52/52 bitwise.

# Step 6 -- measure (raw/bench_r7.sh -> raw/bench_r7/, raw/bench_r7_agg.txt; raw/g17hi.sh -> raw/g17hi/)
Protocol h28/h31: benchmark.py unprofiled, median of 101, one process per shape, fa-g0 GPU 0 at 0% use
before and after every batch, no other python on the device, no new dmesg event.
Batch 1: no beat in the process; arms A, B, r4 (rounds/004/op), r6 (rounds/006/op = op/current); 3 sessions
with cyclic rotation. Batch 2: gate form (X + beat), one process per X per shape.

Batch 1, TF/s mean of 3 sessions (vs r6, per-session in brackets):

| arm | prod | proxy | fast |
| --- | --- | --- | --- |
| A g17 | 1503.08 x0.9949 [.9941 .9971 .9935] | 1575.22 x1.0046 | 98.88 x1.0289 [1.035 1.024 1.028] |
| B g18 | 1513.66 x1.0019 [.9995 1.0048 1.0014] | 1592.51 x1.0156 [1.016 1.010 1.021] | 97.69 x1.0164 [1.026 1.015 1.009] |
| r4 | 1430.52 x0.9469 | 1545.11 x0.9854 | 98.53 x1.0252 |
| r6 | 1510.79 | 1568.07 | 96.11 |

Batch 2 (gate form, X/beat): prod r6 .7742, A .7756, B .7766, r4 .7363 | proxy r6 .9227, A .9298,
B .9354, r4 .9079 | fast r6 .6151, A .6201, B .6247, r4 .6316. Mean beat in this sweep: prod 1786.32,
proxy 1651.34, fast 154.19 TF/s.

g17-HI throwaway (3 arms A / AHI / r6 per process, 3 rotations, prod + fast):
prod A x0.9942 [.995 .991 .997], **AHI x0.9352**; fast A x0.9987, AHI x0.9640, where r6 = 1528.86 / 106.36.

## Reading
- **A (g17) lost prod: -0.5% in every session of both sweeps** (6/6, beyond the 0.4% floor). Its fast gain
  (+2.9% in the 4-arm process, +0.8% in gate form) is gone in the 3-arm process (x0.9987). By row 2's
  rule it does not ship. By row 3's rule the A+B merge is not built.
- **The LO/HI order is not free**: all-HI (prefetch issued before the K ds_load) costs **-6.5% at prod**,
  and all-LO costs -0.5%. The 50/50 stagger is the best of the three. The order of the K load and the TDM
  prefetch issue is a first-order prod term. This is a lead for an L7-family ordering study, not a size lever.
- **B (g18) ships**: prod x1.0019, proxy x1.0156, fast x1.0164 no-beat; gate form better than r6 on all
  three shapes. Masked tiles on the non-speculative path cost nothing measurable (predicted -0.2..-0.4%).
  So speculation on the 1-4 masked tiles per WG was paying nothing, and dropping it paid in proxy.
- **Fast is process-composition sensitive**: at the same sclk (2315), r6 runs **96.1 TF/s in the 4-arm
  process and 106.4 in the 3-arm one** (+10.7%). The under-cap arms beat r6 by 2-3% in the 4-arm process
  and in gate form, but not in the 3-arm process of near-identical kernels. The fast term is
  cross-kernel interference in the process, which the cap fix helps only partly. Its cause is not
  identified: it does not show in the I$ counters (Survey).

## Acceptance arithmetic (shipped = B, re-measured references r4 and r6 in the same sweep)
- Throughput vs best ever (r6 at prod/proxy, r4 at fast), no beat: geomean of per-shape vs-r6 = 1.0113.
  Score on this sweep's beat: B 0.8151, r6 0.8062, r4 0.7918, A 0.8122. **B gains +0.0089 over r6, clearing
  min_gain 0.007.**
- Below-target shapes vs their own best ever: fast 97.69 / r4 98.53 = 0.9915; proxy 1.0156 (vs r6);
  prod 1.0019 (vs r6). All >= 0.95. No shape had reached its target.
- Expectation check (written before the build): A fast +3..6% -> +2.9% (4-arm) / 0% (3-arm), prod expected
  within the floor -> -0.5%: **missed on prod**. B fast +1..4% -> +1.6%, prod -0.2..-0.4% -> +0.2%: **met**.
