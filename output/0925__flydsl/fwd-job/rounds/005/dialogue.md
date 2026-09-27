# Dialogue

## 00_analyse -- planner

Output is big; I'll read it in pieces.
还剩最后两件事：把知识缺口追加到 `knowledge_gaps.md`，然后写出候选 YAML。
我同意 profiling 对 prod 的判断（latency，置信度 low）。不过这个标签本身说明不了多少，关键是周期花在哪里。我提 3 个候选，排在第一的是把 h21 的跨 tile 流水移植到第 4 轮的 champion 上。它能不能跑，取决于编译后 VGPR 能否守住 512。

**约束怎么读：**
- **已排除 HBM：** prod 从 randn 换成 zeros 后，sclk 从 988 升到 1053 MHz，时间降 7.2%，每次调用的周期数基本不变（1.964→1.941 M）。见 5-power-wall-analysis。
- **时钟最多只能解释 +11%**（1100/988）。同样的卡状态下 beat 每秒做的工作量是我们的 1.30 倍，所以差距里约 23% 与时钟无关。见 6-bound-analysis。
- **beat 快在哪里：** 它在单个 wave 内让 WMMA 保持背靠背发射，有 66.7% 的间隔是 8 周期；softmax 的 VALU 被塞进 WMMA 之间的空隙。
- **我们慢在哪里：** softmax 与 WMMA 在每个 wave 内串行执行。facts.md 中的证据是：删掉 exp 快 13.4%，第 4 轮打包快 6.3%，而 tile 内部挪动 VALU（g11）反而慢 2.3%。
- **2 waves/SIMD 帮不上忙：** 每个 tile 的 WG barrier 把所有 wave 锁在同一相位，g08 的 setprio 错峰测出来是 1.000。所以重叠只能在 wave 内部做出来。
- **第 3 轮删 exp 的 13.4% 偏高：** 知识库 dead-ends 里的"A probe that deletes the exp"说明，这类探针同时降低了开关活动，因而抬高了时钟。因此所有预测都改用 `clock_probe` 测的每次调用周期数，sclk 同时报告，时钟和周期分开读。
- **fast 属于另一个 regime：** beat 同样只有 32 个 WG，所以 underfill 不是两者的差别所在。按 correction 4，这一轮不以 fast 为目标。

**候选排序：**
1. **g13：移植 h21 的 L6 跨 tile 流水。** 风险在 VGPR：第 4 轮把 champion 从 445 抬到 456，原型在 round-2 基线上已经是 508，直接叠加会到约 519，超过 512。所以先只编译，从 ISA 读回分配；超过 512 或有 spill，就记为 roadblock，不上卡。
2. **g14：h22 剩下的 L17（每条 lane 各自累计部分行和）。** h22 在 route 里是 must。BF 不做：第 2 轮测过已死（prod 0.940），h26 也写明它不缩短关键路径。PK 已经在 g09 里落地。这一臂不再逐位一致，要守 49 dB 的精度门槛，当前只有 0.8 dB 余量。
3. **g15：h23 的 TDM 预取深度 3（U1PP3）。** 它和 g13 的 LDS ring 改动冲突，本轮不能合并，只在 g13 编不过时作为替补。

以上 3 个都已对照 `dead_ends.md` 和知识库的 dead-ends 检查过，没有重复已死的方向。

**我没能验证的：** `waves_per_eu=2` 在 gfx1250 上到底卡多少寄存器，知识库没写，512 是我从 h21/h26 的编译结果推出来的。这一条连同另外两个缺口（bf16 WMMA 峰值、VALU 削减里有多少以时钟形式兑现）已追加到 `knowledge_gaps.md`。

```yaml
constraint:
  bound: latency
  shape: prod
  agrees_with_profiling: true
  grounds:
    - {claim: "not HBM: cycles/call flat while sclk moved 988->1053", value: "1.964 -> 1.941 Mcyc (-1.2%), time -7.2%", from: rounds/005/1-profiling/5-power-wall-analysis/analysis.md}
    - {claim: "clock explains at most +11%; beat does 1.30x the work in the same card state", value: "1100/988; 1410.87 vs 1083.33 TF/s", from: rounds/005/1-profiling/6-bound-analysis/analysis.md}
    - {claim: "machine is filled at prod", value: "SQ_BUSY/SQ_CYCLES 0.981, 4096 WGs", from: rounds/005/1-profiling/2-kernel-profiling/kernel-kn_fmha_fwd_prefill_a16w16_m32x8_bshd/analysis.md}
    - {claim: "the cycles beat saves are in-wave WMMA cadence with softmax VALU landing in WMMA gaps", value: "WMMA issue interval 8 cyc 66.7%; v_cvt/v_pk stalls 38% of stall", from: job_context/profiling/beat/4-thread-trace/analysis.md}
    - {claim: "softmax VALU is serial with WMMA in current, and only cross-tile moves can overlap it", value: "exp removed -13.4% prod (r3); VALU packing +6.3% (r4); intra-tile sink g11 -2.3%", from: job_context/findings/facts.md}
    - {claim: "the exp-removal price is inflated by clock: prod sclk is data/activity dependent", value: "randn 988 MHz / zeros 1053 MHz", from: rounds/005/1-profiling/5-power-wall-analysis/analysis.md}
  reading: >
    Agree on "latency, low", and the label matters less than where the cycles go. Beat's trace plus facts.md say it:
    the WMMA chain waits on the softmax dependency chain within each wave. Current has 2 waves/SIMD, but the per-tile
    WG barrier holds them in phase (h4: 4 barriers per 256 KV), and r3.i2.g08's s_setprio stagger bought nothing
    (prod 1.000). So the overlap has to be built inside the wave, as beat does (QK(j) interleaved with exp(j-1)).
    fast is a separate regime (32 WGs). Beat has the same underfill there, so underfill is not the differentiator
    (beat profile_summary: flushed/warm 1.25x beat vs 1.57-1.83x current). Per correction 4 it is not this round's target.

candidates:
  - id: r5.i1.g13
    title: "Port h21 L6 QK(i+1)/softmax(i) cross-tile software pipeline onto the round-4 champion"
    proposed_by: planner
    round_proposed: 5
    hypothesis: >
      The prod per-cycle gap to beat is softmax VALU/TRANS serial with WMMA inside each wave. Round-4 packing
      shortened that chain but did not overlap it. Double-buffering S so softmax(i) issues under QK(i+1)'s 32 WMMAs
      hides most of the remaining chain. This is the mechanism beat uses, and the only relocation that has not
      already been measured to lose (g11 intra-tile -2.3%).
    prediction:
      metric: prod_Mcycles_per_call_randn
      from: 1.964
      to: "<=1.89 (-4% or more)"
      also_expect:
        prod_vs_champion_benchmark: ">=1.03 (3 slot-rotated sessions)"
        prod_sclk: "within -1.5% of control (the extra v_mov/msb adds switching; cycles must fall more than clock)"
        fast_vs_champion: "0.98-1.02"
        isa_gate: "VGPR <= 512, spill 0/0, scratch 0; _PIPE gate intact, non-gated configs byte-identical to champion"
        output: "bitwise equal to champion (hash o=26a89a2db0cd lse=d6ac8da1e101)"
      falsified_if: "compiles clean at 0 spill and prod cycles/call falls < 1.5% -> in-wave cross-tile overlap is not the lever at 2 waves/SIMD; close L6"
    evidence:
      - {claim: "exp on the critical path (upper bound, clock-inflated)", value: "-13.4% prod", from: job_context/findings/facts.md}
      - {claim: "intra-tile relocation cannot make overlap", value: "g11 -2.3% prod", from: rounds/004/2-reflect/reflect.md}
      - {claim: "beat fills WMMA gaps with v_exp/v_pk_* from the adjacent tile, 1 wave/SIMD", value: "v_exp 54.6k in WMMA gaps; EXEC 68%", from: job_context/profiling/beat/4-thread-trace/analysis.md}
      - {claim: "prototype compiled on round-2 base: 508 VGPR, 0 spill; v_exp within 8 instr of QK WMMA 59/66", value: 508, from: job_context/findings/route.md (h21)}
      - {claim: "round 4 raised champion VGPR 445 -> 456", value: 456, from: job_context/findings/facts.md}
    justified_by_round: 5
    source: {kind: internal, repo: null, commit: null, verified: true}
    scope:
      files: [op/current/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py]
      backend_feasible: unknown
    risk: >
      VGPR is the wall. Base went 445 -> 456, so a naive port lands near 519 > 512. It must be ported as a diff
      (h27, never copy files; the conflict is with round 4's packed softmax lines). Read the allocation back from the
      ISA. If it is over 512 or spills, act stops, records "roadblock: VGPR", and does not run it on card (h14, a
      spill can wedge the card). One lever that can recover the budget: carry P as bf16 (the packed form PV needs)
      instead of f32 S across the tile boundary. It is not proposed here unless the compile needs it. Masked loops
      get no overlap, and the last tile is peeled.

  - id: r5.i2.g14
    title: "h22 remainder: per-lane partial row-sum carried across tiles (L17), defer rescale kept"
    proposed_by: planner
    round_proposed: 5
    hypothesis: >
      After round 4 the row-sum is a packed in-lane tree, but a cross-lane permlane reduce still runs every tile on
      the serial softmax chain. The sum is only needed at the epilogue, so carrying per-lane partials removes that
      reduce and its dependency wait from every tile. Round 4 found dependency stalls priced higher than slot
      count (super-additive merge, v_nop 168 -> 62), so a chain-shortening cut should pay more than its instruction count.
    prediction:
      metric: prod_Mcycles_per_call_randn
      from: 1.964
      to: "1.925-1.945 (-1% to -2%)"
      also_expect:
        prod_vs_champion_benchmark: "1.01-1.025"
        isa: "permlanex16 whole-kernel 16 -> <=10 (proto figure); VGPR <= 470; spill 0/0"
        precision: "not bitwise (sum re-association); o/lse >= 49 dB on all 16 ut cases, lse vs fp32 reference"
      falsified_if: "prod < +0.5% (floor) with permlane count down as predicted -> the per-tile cross-lane sum is off the critical path; per-instruction pricing of softmax cuts is then dead too"
    evidence:
      - {claim: "prototype: LANE alone shortens QK->PV serial span", value: "551/337 base -> 527/240 LANE only; PK+LANE+defer 506/218", from: job_context/findings/route.md (h22)}
      - {claim: "h22 is a 'must' route item; L15 and packed sum already landed, so L17 is what remains", value: "must", from: job_context/findings/route.md}
      - {claim: "dependency-shortening cuts were super-additive", value: "+3.0 and +1.3 -> +6.3%", from: job_context/findings/facts.md}
    justified_by_round: 5
    source: {kind: internal, repo: null, commit: null, verified: true}
    scope:
      files: [op/current/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py]
      backend_feasible: yes
    risk: >
      The precision headroom is 0.8 dB (worst 49.82). The row-max path is not touched, so h16's large-logit gate is not
      triggered, but running it anyway is cheap. The lines conflict with g13's softmax region, so it is built as a
      separate arm from op/current and merged only if both win. BF (L20) is excluded: dead r2 (prod 0.940), and h26
      says BF adds +31 VALU/iter with no span gain. PK-only (C3) already shipped as g09. So h22's C1/C3 arms have no
      question left, and this arm (C2 on the current base) discharges h22.

  - id: r5.i3.g15
    title: "h23 U1PP3: TDM prefetch depth 3, top-of-tile wait drains only tile t (s_wait_tensorcnt 0x2)"
    proposed_by: planner
    round_proposed: 5
    hypothesis: >
      Current waits for all TDM traffic (tensorcnt 0x0) before every tile, and that wait sits in front of the
      per-tile WG barrier, so a slow K/V arrival on any wave stalls all 8. Beat keeps 4 in flight (0x4), and TDM
      wait is still 5.6% of beat's exposed stall. Issuing tile t+2 at tile t removes the full drain.
    prediction:
      metric: prod_Mcycles_per_call_randn
      from: 1.964
      to: "<=1.93 if the drain is exposed; >=1.97 if one tile of compute already covers TDM latency"
      also_expect:
        prod_vs_champion_benchmark: ">=1.02 or <=0.99 -- a result inside +-0.5% is itself the answer (TDM latency hidden)"
        isa: "VGPR unchanged 456, SGPR ~+7, spill 0/0, LDS 327680, wait 0x2 in the clean loop"
        output: bitwise equal to champion
    evidence:
      - {claim: "champion waits tensorcnt 0x0 every iteration; ASM 0x4", value: "0x0 vs 0x4", from: job_context/findings/route.md (h4)}
      - {claim: "TDM wait is a residual stall even for beat", value: "s_wait_tensorcnt 5.6% of stall cycles", from: job_context/profiling/beat/4-thread-trace/analysis.md}
      - {claim: "prototype compiles: U1PP3 VGPR 445 SGPR 104, 0 spill", value: "573 instr/tile", from: job_context/findings/route.md (h23)}
    justified_by_round: 5
    source: {kind: internal, repo: null, commit: null, verified: true}
    scope:
      files: [op/current/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py]
      backend_feasible: yes
    risk: >
      BWD prior: every depth > 2 lost 10-30%, though that staging was dual-use. This arm also conflicts with g13's LDS
      ring re-phase ([K(i+1)|V(i)]), so the two cannot merge this round. Run it only if g13 is a roadblock, or as the
      third arm if budget allows. The CPU schedule proof covers only sq==skv, so the gate adds sq!=skv and
      kv_len % 64 != 0.

execution_order: >
  g13 compile-only first (the VGPR read-back decides everything). If it is <= 512 with 0 spill: arms = g13 and g14,
  each from op/current, 3 slot-rotated sessions, prod/proxy/fast, and prod cycles/call via clock_probe with sclk.
  If g13 is a VGPR roadblock: arms = g14 and g15. Ship nothing whose prod vs_champion is under the floor
  (correction 4).

from_pool:
  - {id: r1.i2.g02, still_valid: false, why: "O_VARIANT v1 lost proxy -3/-4% in round 1 (route row 11 says retire); the +1.9% was flydsl 0.3.2 pre-job"}
  - {id: r1.i3.g03, still_valid: true, why: "post-beat penalty is still unexplained (r3.i1.g07 excluded I$ fetch); fast-only instrument, not a prod lever -- keep as a rider, not this round"}
  - {id: r1.i4.g04, still_valid: false, why: "justified by fast underfill, but beat has the same 32-WG underfill and is 1.8x faster; the beat profile's flushed/warm 1.25x vs 1.57-1.83x points to per-call/cold-cache cost, not underfill -- stale justification"}
  - {id: r2.i2.g06, still_valid: false, why: "round-2 analysis: under g01's order, (b,kv_head) already maps to a fixed XCD, and scored shapes have sq==skv, so L4/L5 have nothing to do at scored shapes"}
  - {id: r3.i2.g08, still_valid: false, why: "measured round 3: prod 1.000 / proxy 0.973 / fast 0.943; reflect closed it -> dead_ends"}
  - {id: r4.i2.g10, still_valid: false, why: "shipped inside the round-4 merge (op/current); stale entry, retire to facts"}

knowledge_gaps:
  - {question: "on gfx1250 (1024-VGPR file) what ceiling does waves_per_eu=2 impose, and does exceeding it spill or drop occupancy", expected_in: knowledge/arch/gfx1250/gfx1250.md or knowledge/backends/flydsl/README.md, did_instead: "assumed 512 from route.md h21/h26 compile results; g13 gated on ISA read-back"}
  - {question: "gfx1250 bf16 WMMA peak FLOP per SIMD-cycle", expected_in: knowledge/arch/gfx1250/gfx1250.md, did_instead: "used beat's 1410.9 TF/s as a lower bound on the ceiling"}
  - {question: "how much of a VALU cut is paid as clock vs cycles when sclk is activity-dependent", expected_in: knowledge/backends/flydsl/attention/techniques.md, did_instead: "predictions stated in Mcycles/call from clock_probe with sclk reported beside"}
```

Files are in `rounds/005/2-plan`:
- step0_planner.yaml

## 00_analyse -- reviewer

```yaml
constraint:
  bound: latency
  shape: prod
  agrees_with_profiling: true
  grounds:
    - claim: "HBM bandwidth is excluded; elapsed time follows the shader clock while cycles per call remain nearly constant."
      value: "randn->zeros: sclk 988->1053 MHz, time -7.2%, cycles 1.964->1.941 M (-1.2%)"
      from: "rounds/005/1-profiling/5-power-wall-analysis/analysis.md"
    - claim: "The production grid fills the device, so grid underfill is not the production constraint."
      value: "4096 workgroups, 32768 waves, SQ_BUSY_CYCLES/SQ_CYCLES=0.981"
      from: "rounds/005/1-profiling/6-bound-analysis/analysis.md"
    - claim: "Clock headroom cannot close the production gap; the remaining deficit is on-chip execution efficiency."
      value: "current/beat=0.768, beat/current throughput=1.302x, maximum clock headroom=11%"
      from: "rounds/005/1-profiling/benchmark-results.md; rounds/005/1-profiling/6-bound-analysis/analysis.md"
    - claim: "Latency is necessarily a low-confidence label because current's issue-versus-dependency split cannot be measured."
      value: "stall counters rejected, PC sampling forbidden, ATT captures no FlyDSL kernels, WMMA counters read zero"
      from: "rounds/005/1-profiling/6-bound-analysis/analysis.md"

candidates:
  - id: r5.i1.g13
    title: "Carry row sums per lane across KV tiles and perform one final cross-lane reduction"
    proposed_by: reviewer
    round_proposed: 5
    hypothesis: >
      Repeating the cross-lane row-sum reduction inside every KV tile extends the
      softmax dependency chain. Keeping per-lane partial sums through the loop and
      reducing once in the epilogue will reduce production cycles without relying
      on a shader-clock increase.
    prediction:
      metric: prod_tflops
      from: 1083.33
      to: ">=1105.0"
      also_expect:
        prod_ms: "<=1.990"
        prod_sclk_change_pct: "within +/-2%"
        scratch_bytes: 0
        prod_vs_champion: ">=1.020"
    evidence:
      - claim: "Production has a 23.2% throughput deficit after HBM bandwidth has been excluded."
        value: "1083.33 versus 1410.87 TFLOP/s; current/beat=0.768"
        from: "rounds/005/1-profiling/benchmark-results.md"
      - claim: "Softmax pack/scaling dependencies occupy the largest measured stall class in the same-card beat implementation."
        value: "v_cvt/v_pk stall share=38.0%; v_exp in WMMA gaps=54.6k"
        from: "job_context/profiling/beat/4-thread-trace/analysis.md"
      - claim: "Reducing softmax VALU has already moved current on card, including a super-additive merged result."
        value: "packed row-sum alone x1.013 prod; packed-exp plus packed-row-sum x1.063 prod"
        from: "job_context/findings/facts.md"
      - claim: "The carried prototype establishes feasibility but is only a build gate, not a time predictor."
        value: "PK+LANE+defer serial span 218 instructions, zero spill"
        from: "job_context/findings/route.md"
    justified_by_round: 5
    source:
      kind: internal
      repo: null
      commit: null
      verified: true
    scope:
      files:
        - "job_context/op/current/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py"
      backend_feasible: yes
    risk: >
      The reduction order changes and may lose bitwise identity or accuracy; the
      route prototype reports max relative error 4.0e-6 and zero spill, but it was
      based on the round-2 body and must be ported rather than copied
      (job_context/findings/route.md, h22 and h27).

  - id: r5.i2.g14
    title: "Pipeline QK(i+1) with softmax(i) across KV tiles"
    proposed_by: reviewer
    round_proposed: 5
    hypothesis: >
      Current serializes each tile's QK and softmax phases. Double-buffering only
      the score registers and issuing the next tile's QK WMMAs alongside the
      previous tile's softmax will shorten the exposed per-wave critical path.
    prediction:
      metric: prod_tflops
      from: 1083.33
      to: ">=1137.5"
      also_expect:
        prod_ms: "<=1.934"
        prod_sclk_change_pct: "within +/-2%"
        scratch_bytes: 0
        prod_vs_champion: ">=1.050"
    evidence:
      - claim: "Production is full-grid and core-clock-scaled, leaving per-wave execution efficiency as the unresolved gap."
        value: "4096 workgroups, SQ_BUSY/SQ_CYCLES=0.981, current/beat=0.768"
        from: "rounds/005/1-profiling/6-bound-analysis/analysis.md"
      - claim: "Beat demonstrates that WMMA cadence and softmax work share the same per-wave timeline."
        value: "8-cycle WMMA intervals=66.7%; estimated matrix-pipe occupancy=76.5%"
        from: "job_context/profiling/beat/4-thread-trace/analysis.md"
      - claim: "Beat's long WMMA gaps contain substantial softmax work rather than an LDS-feed bottleneck."
        value: "v_exp=54.6k, v_pk_add=23.7k, v_pk_fma=22.2k in gaps; LDS-read stall share=0.5%"
        from: "job_context/profiling/beat/4-thread-trace/analysis.md"
      - claim: "The gated prototype proves the mechanism compiles on the scored causal GQA path without scratch."
        value: "508/512 VGPR, scratch=0, spill=0; 59/66 clean-LO exp instructions within 8 instructions of QK WMMA"
        from: "job_context/findings/route.md"
    justified_by_round: 5
    source:
      kind: internal
      repo: null
      commit: null
      verified: true
    scope:
      files:
        - "job_context/op/current/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py"
      backend_feasible: yes
    risk: >
      The prototype has only four VGPRs of headroom at 508/512 and previously
      spilled on non-causal and GQA=1 paths; its existing causal-D128-GQA gate
      must remain exact (job_context/findings/route.md, h21). Register spill or
      any production regression rejects the candidate.

from_pool:
  - id: r1.i2.g02
    still_valid: false
    why: >
      Its justification predates this job and route.md records proxy losses of
      3-4% with an explicit retire decision; the current profile provides no new
      epilogue-specific evidence (job_context/findings/route.md, route row 11).

  - id: r1.i3.g03
    still_valid: true
    why: >
      Valid only as a diagnostic rider, not as an implementation candidate:
      current is 35.3 us warm versus 56.16 us flushed and beat's warm time is
      still missing (rounds/005/1-profiling/5-power-wall-analysis/analysis.md;
      rounds/005/1-profiling/benchmark-results.md). The I-cache mechanism itself
      is false because steady-state misses are zero and explicit prefetch already
      lost (rounds/005/1-profiling/2-kernel-profiling/kernel-kn_fmha_fwd_prefill_a16w16_m32x8_bshd/analysis.md;
      job_context/findings/dead_ends.md). I would not execute it instead of a
      production candidate.

  - id: r1.i4.g04
    still_valid: true
    why: >
      Fast still launches only 32 workgroups on 256 CUs and reaches 0.547 of beat
      (rounds/005/1-profiling/kernel.yaml; rounds/005/1-profiling/benchmark-results.md).
      I would defer it: it is fast-only, requires a deterministic combine, and
      production remains at 0.768 of beat.

  - id: r2.i2.g06
    still_valid: false
    why: >
      The landed longest-first mapping already fixes each scored (batch,kv-head)
      group to an XCD and aligns the causal origin; route.md records L4/L5 as
      closed for the scored shapes. This round also excludes HBM as the production
      constraint (rounds/005/1-profiling/6-bound-analysis/analysis.md).

  - id: r3.i2.g08
    still_valid: false
    why: >
      It was measured at prod 1.000, proxy 0.973 and fast 0.943 of its incumbent
      and marked closed (job_context/findings/pool.md, lines 64-66). This round's
      production profile supplies no evidence to reopen manual wave staggering.

  - id: r4.i2.g10
    still_valid: false
    why: >
      It is already part of the champion: the merged packed-exp plus packed-row-sum
      implementation shipped at x1.063 prod (job_context/findings/facts.md).
      Round-4 reflect explicitly identifies this pool entry as stale
      (rounds/004/2-reflect/reflect.md).

knowledge_gaps: []
```

## 01_synthesis -- planner

流水原型的 diff 能干净地打到当前 champion 上（8/8 hunk，只有偏移）。编译一次就能直接解决 VGPR 是否超 512 的分歧，我去看看原型的 compile-only 脚本。
流水移植版 0 spill，但 VGPR 正好卡在 512，余量为零。再确认一下流水在 ISA 里确实生效了，而不是编译出和 champion 一样的代码：
我和 reviewer 各自独立提了同样两个机制，这一点算一致，不重复讨论。合并后排序是 g13 流水 → g14 LANE 行和 → g15 TDM 预取深度 3。

**先把 VGPR 的问题测掉了，没有靠推测。** 我在 champion 的副本上打了原型流水的 diff：8 个 hunk 全部成功，只有偏移。然后做了一次 compile-only 编译，GPU 不可见，没有碰卡。

| 构建 | VGPR | spill | scratch | 紧跟 WMMA（8 条指令内）的 v_exp | nop | s_set_vgpr_msb |
|---|---|---|---|---|---|---|
| champion | 456 | 0/0 | 0 | 0/264 | 62 | 240 |
| 流水移植版 | **512** | 0/0 | 0 | 103/330 | 217 | 757 |

- **我第 0 步的估计错了。** 我原来估约 519、会超 512；实际装得下。reviewer 引的"508/512，余量 4"也不是当前 champion 上的数，实际余量是 0。
- **重叠在 ISA 里确实出现了**：103 个 v_exp 紧跟在 WMMA 后面，champion 是 0 个。
- **代价的方向对我们不利。** nop 涨到 3.5 倍，msb 涨到约 3 倍。第 4 轮超可加的 +6.3% 恰恰伴随 nop 从 168 降到 62，这次反过来了。

**ID 撞了。** 双方都从 g13 起编号，而且交叉了：reviewer 的 r5.i1.g13 是 LANE，r5.i2.g14 是流水。合并后沿用我的 id，reviewer 的写成别名，不另作他用。`adopted_from_reviewer` 为空，因为 reviewer 没有提出新假设；我采纳了它的 TF/s 门槛、±2% sclk 区间，以及它对 g03 更清楚的说法。

**三处分歧，各自有测量可以裁定：**
1. **g13 的收益幅度：** 我预测 ≥ +3%，reviewer 预测 ≥ +5%。我的依据是上面的 nop/msb 开销，以及删 exp 那次 13.4% 被时钟抬高了。三个槽位轮换的 session 加上 `clock_probe` 周期数来裁定：≥1.05 算 reviewer 对，1.03–1.05 算我对，<1.015 两个都错，关掉 L6。
2. **排序：** 我主张 g13 先，因为它 VGPR 余量为零，先落 g14 可能把 g13 挤死，反过来不会。reviewer 主张 g14 先，因为改动小、风险低。两臂同一 session 测，由结果决定先上哪个，另一个要重新编译确认合并后 ≤512 且 0 spill 才能叠上去。
3. **r1.i4.g04 是否仍然有效：** 我认为它的依据已经过时，因为 beat 也只有 32 个 WG，却快 1.8 倍。用 fast 上 beat 的 warm 与 flushed 对比来裁定。

**文件：** 编译探针的原始数据在 `rounds/005/2-plan/raw/vgpr_probe/`，推理写在 `synthesis_notes.md` 的 S1/S2。`knowledge_gaps.md` 补了一条：512 能装下，但仍无法证明上限就是 512。

```yaml
candidates:
  - id: r5.i1.g13
    alias: r5.i2.g14 (reviewer step 0, same hypothesis -- see synthesis_notes.md S2)
    title: "Port h21 L6 QK(i+1)/softmax(i) cross-tile software pipeline onto the round-4 champion"
    proposed_by: planner          # independently proposed by reviewer too
    round_proposed: 5
    hypothesis: >
      The prod per-cycle gap to beat is softmax VALU/TRANS serial with WMMA inside each wave. Double-buffering S so
      softmax(i) issues under QK(i+1)'s WMMAs hides part of that chain, which is beat's mechanism. Intra-tile
      relocation already lost (g11 -2.3%), so this is the remaining form of overlap.
    prediction:
      metric: prod_Mcycles_per_call_randn     # tools/clock_probe, 5-power-wall protocol
      from: 1.964
      to: "<=1.905 (-3% or more)"
      also_expect:
        prod_vs_champion_benchmark: ">=1.03 (3 slot-rotated sessions); reviewer predicts >=1.05 -- see disagreements"
        prod_tflops: ">=1115.8 (1083.33 x 1.03)"
        prod_sclk: "within +-2% of control, reported beside every time"
        fast_vs_champion: "0.98-1.02"
        isa_gate: "MEASURED in plan: VGPR 512, SGPR 97, spill 0/0, scratch 0; 103/330 v_exp within 8 instr of a WMMA (champion 0/264). Act must reproduce md5 d530dda8 or explain the difference"
        output: "bitwise equal to champion (o=26a89a2db0cd lse=d6ac8da1e101)"
      falsified_if: "prod cycles/call falls < 1.5% at 0 spill -> in-wave cross-tile overlap does not pay at this nop/msb cost; close L6 at n_block 64"
    evidence:
      - {claim: "port compiles at exactly 512 VGPR, 0 spill, 0 scratch; overlap present", value: "512/0/0; 103 v_exp near WMMA", from: rounds/005/2-plan/synthesis_notes.md#S1}
      - {claim: "exp on the critical path (upper bound; clock-inflated per corpus dead-end 'A probe that deletes the exp')", value: "-13.4% prod", from: job_context/findings/facts.md}
      - {claim: "intra-tile relocation cannot make overlap", value: "g11 -2.3% prod", from: rounds/004/2-reflect/reflect.md}
      - {claim: "beat fills WMMA gaps with softmax VALU of the adjacent tile, at 1 wave/SIMD", value: "v_exp 54.6k in WMMA gaps; WMMA 8-cyc interval 66.7%", from: job_context/profiling/beat/4-thread-trace/analysis.md}
      - {claim: "2 co-resident waves do not supply the overlap: explicit stagger measured null", value: "r3.i2.g08 prod 1.000", from: job_context/findings/pool.md}
    justified_by_round: 5
    source: {kind: internal, repo: null, commit: null, verified: true}
    scope:
      files: [op/current/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py]
      backend_feasible: yes      # compiled in plan
    risk: >
      Zero VGPR headroom. Nothing stacks on it without a recompile, so a g13+g14 merge is presumed not to fit.
      Nops go 62 -> 217 and s_set_vgpr_msb 240 -> 757, the opposite of round 4's winning direction. The _PIPE gate
      must stay as shipped (non-gated configs spilled before the review, h21). The gate also needs the
      bitwise/determinism set from h21: n_iter=1/2, kv_len%64!=0, sq!=skv, 80x repeat, 200-run determinism.

  - id: r5.i2.g14
    alias: r5.i1.g13 (reviewer step 0, same hypothesis)
    title: "h22 remainder: per-lane partial row-sum carried across KV tiles (L17), one cross-lane reduce in the epilogue; defer rescale kept"
    proposed_by: planner          # independently proposed by reviewer too
    round_proposed: 5
    hypothesis: >
      The per-tile cross-lane row-sum reduce lengthens the serial softmax chain although the sum is needed only at
      the epilogue. Carrying per-lane partials removes it from every tile. Chain-shortening cuts were priced above
      their slot count in round 4.
    prediction:
      metric: prod_Mcycles_per_call_randn
      from: 1.964
      to: "1.925-1.945 (-1% to -2%)"
      also_expect:
        prod_tflops: ">=1105.0 (reviewer's threshold, adopted)"
        prod_vs_champion_benchmark: ">=1.02 (reviewer) / 1.01-1.025 (planner) -- same band, lower edge is the reviewer's"
        prod_sclk: "within +-2%"
        isa: "whole-kernel permlanex16 16 -> <=10; VGPR <= 470; spill 0/0; scratch 0"
        precision: "not bitwise; o/lse >= 49 dB on all 16 ut cases, lse checked against the fp32 reference"
      falsified_if: "permlane count falls as predicted but prod < +0.5% -> the per-tile cross-lane sum is off the critical path"
    evidence:
      - {claim: "LANE shortens the proto's QK->PV serial span", value: "337 -> 240 (LANE only); 218 PK+LANE+defer", from: job_context/findings/route.md (h22)}
      - {claim: "h22 is a 'must' route item; PK and packed sum already shipped", value: must, from: job_context/findings/route.md}
      - {claim: "softmax VALU cuts moved prod super-additively", value: "+3.0 and +1.3 -> +6.3%", from: job_context/findings/facts.md}
      - {claim: "proto diff conflicts with round 4 at the packed region, so this is a hand port", value: "hunk 4 FAILED at :603", from: rounds/005/2-plan/synthesis_notes.md#S1}
    justified_by_round: 5
    source: {kind: internal, repo: null, commit: null, verified: true}
    scope:
      files: [op/current/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py]
      backend_feasible: yes
    risk: >
      0.8 dB precision headroom (worst 49.82). BF (L20) is excluded: dead in r2 (prod 0.940), and h26 says it adds
      +31 VALU/iter with no span gain. This arm discharges h22's remaining question.

  - id: r5.i3.g15
    title: "h23 U1PP3: TDM prefetch depth 3, top-of-tile wait drains only tile t (s_wait_tensorcnt 0x2)"
    proposed_by: planner
    round_proposed: 5
    hypothesis: >
      The champion drains all TDM traffic (tensorcnt 0x0) before each tile's WG barrier, so the slowest wave's K/V
      arrival gates all 8 waves. Beat keeps 4 in flight. Depth 3 removes the full drain.
    prediction:
      metric: prod_Mcycles_per_call_randn
      from: 1.964
      to: "<=1.93 if the drain is exposed; >=1.97 if one tile of compute already covers TDM latency"
      also_expect:
        prod_vs_champion_benchmark: "a result inside +-0.5% is itself the answer (TDM latency hidden) and closes L13"
        isa: "VGPR 456, SGPR ~+4..7, spill 0/0, wait 0x2 in the clean loop"
        output: bitwise equal to champion
    evidence:
      - {claim: "champion waits tensorcnt 0x0 every iteration; ASM 0x4", value: "0x0 vs 0x4", from: job_context/findings/route.md (h4)}
      - {claim: "TDM wait remains a stall class even for beat", value: "s_wait_tensorcnt 5.6% of stall", from: job_context/profiling/beat/4-thread-trace/analysis.md}
    justified_by_round: 5
    source: {kind: internal, repo: null, commit: null, verified: true}
    scope:
      files: [op/current/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py]
      backend_feasible: yes
    risk: >
      Conflicts with g13's LDS ring re-phase, so it cannot merge with it. The BWD prior is negative (depth > 2 lost
      10-30%). Third arm only if the session budget allows; it is not a fallback any more, since g13 compiles.

adopted_from_reviewer: []
# No new hypothesis came from the reviewer: its two candidates are the same two mechanisms as planner g13/g14,
# proposed blind, which is agreement, recorded once. Adopted from its text into the merged entries: the
# prod_tflops thresholds (>=1105.0 for g14) and the +-2% sclk band. Its r1.i3.g03 reasoning ("diagnostic rider
# only; the I-cache mechanism itself is false") is also adopted; it is sharper than mine.

agreed:
  - "bound: latency (low), prod, HBM excluded by the data swap -- both, same citations"
  - "g13 and g14 are the two mechanisms of this round"
  - "pool: g02, g06, g08, g10 not valid; g03 a rider only"

disagreements:
  - candidate: r5.i1.g13
    planner:
      position: "Expected prod gain is >=+3% (cycles/call <=1.905), not >=+5%. The port pays for its overlap with the instruction overhead round 4 identified as expensive."
      grounds:
        - {claim: "port at 512 VGPR: nops 62 -> 217, s_set_vgpr_msb 240 -> 757, v_mov_b64 240 -> 496", from: rounds/005/2-plan/synthesis_notes.md#S1}
        - {claim: "round 4's super-additive +6.3% came with v_nop 168 -> 62; nop/dependency count tracked time better than slot count", from: job_context/findings/facts.md}
        - {claim: "the 13.4% exp-removal ceiling is clock-inflated: prod sclk moves 988 -> 1053 on switching activity alone", from: rounds/005/1-profiling/5-power-wall-analysis/analysis.md}
    reviewer:
      position: "Overlapping softmax with the next tile's QK removes most of the exposed per-wave chain, so prod gains >=+5% (>=1137.5 TF/s, <=1.934 ms)."
      grounds:
        - {claim: "prod is full-grid and core-clock-scaled; the gap is per-wave execution efficiency", from: rounds/005/1-profiling/6-bound-analysis/analysis.md}
        - {claim: "beat's WMMA gaps hold softmax work, not LDS-feed stalls (LDS-read stall 0.5%)", from: job_context/profiling/beat/4-thread-trace/analysis.md}
        - {claim: "proto places 59/66 clean-LO v_exp within 8 instructions of a QK WMMA", from: job_context/findings/route.md (h21)}
    settled_by: "g13 vs control, 3 slot-rotated sessions at prod, plus clock_probe cycles/call with sclk: >=1.05 -> reviewer; 1.03-1.05 -> planner; <1.015 -> both wrong and L6 closed at n_block 64"

  - candidate: ordering (r5.i1.g13 vs r5.i2.g14)
    planner:
      position: "Build and price g13 first. g14 runs in the same session as a separate arm, but g13's result decides whether g14 is shippable at all."
      grounds:
        - {claim: "g13 fits at exactly 512 VGPR, so g13+g14 cannot be presumed to merge; shipping g14 first could close g13 on VGPR (LANE proto +1 VGPR over its base)", from: rounds/005/2-plan/synthesis_notes.md#S1}
        - {claim: "g13 tests the premise under both: that the softmax chain is exposed and hideable in-wave. If it wins, g14's chain-shortening is worth less, because the chain is then under QK", from: job_context/findings/facts.md}
        - {claim: "g13's compile risk, the only reason to put it second, is discharged: it built clean", from: rounds/005/2-plan/synthesis_notes.md#S1}
    reviewer:
      position: "g14 (LANE) first: smaller, self-contained, lower risk; g13 second, because it had only 4 VGPRs of headroom and spilled on non-gated paths before its review."
      grounds:
        - {claim: "PK+LANE+defer proto: serial span 218, 0 spill", from: job_context/findings/route.md (h22)}
        - {claim: "pipeline proto 508/512, spilled on non-causal and gqa1 before the gate", from: job_context/findings/route.md (h21)}
    settled_by: "both arms measured in one session. If g13 >= g14 + floor at prod, ship g13 and re-queue g14 only after a recompile shows g13+g14 <= 512 at 0 spill. If g14 >= g13, ship g14 and re-open g13 only if a recompile on top of g14 fits"

  - candidate: r1.i4.g04
    planner:
      position: "still_valid false: the justification (fast grid underfill) is stale. Beat runs the same 32-WG grid and is 1.8x faster, so underfill is not what separates the two arms."
      grounds:
        - {claim: "beat underfill identical; flushed/warm ratio 1.25x beat vs 1.57-1.83x current", from: job_context/profiling/beat/profiling_summary.md}
        - {claim: "current warm 35.3 us vs flushed 56.16 us; beat flushed 30.73 us", from: rounds/005/1-profiling/5-power-wall-analysis/analysis.md}
    reviewer:
      position: "still_valid true but deferred: fast still launches 32 WGs on 256 CUs and sits at 0.547 of beat; it is fast-only and needs a deterministic combine."
      grounds:
        - {claim: "32 WGs on 256 CUs; fast 0.547", from: rounds/005/1-profiling/kernel.yaml}
    settled_by: "the profiling's cheapest next measurement: beat warm vs flushed at fast, in the same session as current. If current's warm time is within ~15% of beat's warm time, the fast gap is per-call/cold-cache and split-KV has no target (g04 -> dead_ends stale); if current is still >=1.3x slower warm, per-CU work at underfill matters and g04 stays"

ordering:
  proposed: [r5.i1.g13, r5.i2.g14, r5.i3.g15]
  why: >
    g13 first. It is the one lever that tests the premise under all three candidates (the softmax chain is exposed
    and can be hidden in-wave). It has the largest expected gain. Its only disqualifier, VGPR, was measured in
    plan: 512, 0 spill. Its fit leaves zero headroom, so the order matters: landing g14 first can close g13, but
    not the reverse. g14 runs as a separate arm from op/current in the same sessions. It discharges the 'must' h22
    and is the answer if g13 falls under its falsification line. g15 is third: it cannot merge with g13 (LDS ring),
    its prior is negative, and its value is mainly to close L13 if a session has room. Nothing ships with prod
    vs_champion under the floor (correction 4). Every arm reports prod sclk and clock_probe cycles/call.
```

Files are in `rounds/005/2-plan`:
- step1_planner.yaml
- synthesis_notes.md
- raw/vgpr_probe/summary.txt

## 02_review -- reviewer

```yaml
turn: 1
by: reviewer

verdicts:
  - candidate: r5.i1.g13
    stance: modify
    grounds:
      - claim: "The production opportunity is large enough to retain the pipeline as the first arm."
        value: "current/beat=0.768; current 1083.33 versus beat 1410.87 TFLOP/s"
        from: rounds/005/1-profiling/benchmark-results.md
      - claim: "The reference implementation demonstrates substantial softmax work inside WMMA gaps rather than an LDS-feed limit."
        value: "v_exp=54.6k in WMMA gaps; v_cvt/v_pk stall share=38.0%; LDS-read stall share=0.5%"
        from: job_context/profiling/beat/4-thread-trace/analysis.md
      - claim: "The production grid is full, so an in-wave overlap mechanism addresses the observed regime."
        value: "4096 workgroups; SQ_BUSY_CYCLES/SQ_CYCLES=0.981"
        from: rounds/005/1-profiling/6-bound-analysis/analysis.md
    risk: >
      Keep g13 first, but do not lower its predicted gain from 5% to 3% because
      nops rose 62->217 or s_set_vgpr_msb rose 240->757. Those values in
      rounds/005/2-plan/synthesis_notes.md#S1 prove that the build changed and
      identify risk; campaign correction 3 explicitly forbids using static ISA
      metrics to price time. Retain the reviewer prediction
      prod_Mcycles_per_call_randn <=1.866 (-5%) and prod_vs_champion >=1.05.
      Record the prediction as false for any result below 5%; retire L6 only if
      the measured gain is below the production noise floor, rather than
      conflating a smaller positive result with a dead mechanism.
    alternative: null

  - candidate: r5.i2.g14
    stance: modify
    grounds:
      - claim: "Production remains an on-chip execution problem after HBM is excluded."
        value: "sclk 988->1053 MHz, time -7.2%, cycles/call 1.964->1.941 M (-1.2%)"
        from: rounds/005/1-profiling/5-power-wall-analysis/analysis.md
      - claim: "Softmax packing and scaling dependencies are the largest measured residual class in beat."
        value: "v_cvt/v_pk instructions account for 38.0% of traced stall cycles"
        from: job_context/profiling/beat/4-thread-trace/analysis.md
      - claim: "There is sufficient production headroom for a 2% prediction to be meaningful."
        value: "prod current/beat=0.768; current time=2.03012 ms"
        from: rounds/005/1-profiling/benchmark-results.md
    risk: >
      The prediction is internally inconsistent: 1.925-1.945 Mcycles means
      1-2%, while prod_tflops >=1105.0 and prod_vs_champion >=1.02 require the
      2% edge. Use one falsifiable prediction:
      prod_Mcycles_per_call_randn <=1.925 and prod_vs_champion >=1.02.
      A result below 2% falsifies the prediction; only a result below the
      approximately 0.5% production floor closes the mechanism. Keep the
      precision, spill, and permlanex16 checks unchanged.
    alternative: null

  - candidate: r5.i3.g15
    stance: modify
    grounds:
      - claim: "Beat's TDM wait is a small residual and is attributed mainly to the first load of each workgroup."
        value: "s_wait_tensorcnt=5.6% of stall cycles; prologue s_wait_tensorcnt 0x1 approximately 1.17k cycles/wave"
        from: job_context/profiling/beat/4-thread-trace/analysis.md
      - claim: "There is no current-kernel measurement showing that the per-tile tensorcnt(0) drain is exposed."
        value: "ATT captures no FlyDSL kernels; stall counters rejected; WMMA counters read zero"
        from: rounds/005/1-profiling/6-bound-analysis/analysis.md
      - claim: "Production is already core-clock-scaled rather than HBM-limited, but this does not identify TDM as the on-chip limiter."
        value: "randn->zeros cycles/call changed only -1.2%"
        from: rounds/005/1-profiling/5-power-wall-analysis/analysis.md
    risk: >
      The current conditional prediction ("<=1.93 if exposed; >=1.97 if
      hidden") is not one falsifiable prediction. Keep g15 only as the optional
      third closure arm and predict no measurable win:
      prod_Mcycles_per_call_randn >=1.954, corresponding to less than 0.5%
      improvement. A measured value <=1.925 would falsify this review and
      validate the exposed-drain hypothesis. A null result closes L13 only if
      the built ISA contains the intended tensorcnt 0x2 wait, has zero spill,
      and its sclk is within 2% of control.
    alternative: null

  - candidate: r1.i4.g04
    stance: modify
    grounds:
      - claim: "Fast still has a genuine device-underfill condition."
        value: "32 workgroups on 256 CUs, at most 12.5% of CUs receiving work"
        from: rounds/005/1-profiling/6-bound-analysis/analysis.md
      - claim: "Fast is the furthest shape from target."
        value: "current/beat=0.547; current 56.16 us versus beat 30.73 us"
        from: rounds/005/1-profiling/benchmark-results.md
      - claim: "The available warm/flushed comparison is incomplete."
        value: "current warm=35.3 us, current flushed=56.16 us; beat warm time not measured"
        from: rounds/005/1-profiling/6-bound-analysis/analysis.md
    risk: >
      Do not mark g04 stale yet. Beat having the same 32-workgroup grid proves
      that underfill is not the only difference; it does not prove that
      additional parallelism cannot improve current. Retain g04 as conditional
      and deferred. The proposed matched beat/current warm-versus-flushed
      measurement is the correct settlement: retire g04 only if current warm
      time comes within approximately 15% of beat warm time.
    alternative: null

ordering:
  proposed: [r5.i1.g13, r5.i2.g14, r5.i3.g15]
  why: >
    Accept g13 first after the compile-only port measured exactly 512 VGPR,
    zero spill and zero scratch in rounds/005/2-plan/raw/vgpr_probe/summary.txt;
    its former feasibility objection is discharged. Measure g13 and g14 as
    separate arms from the champion in the same sessions. Run g15 only if the
    session budget remains, because beat attributes its 5.6% TDM-stall class
    mainly to the per-WG first load rather than the steady-state drain
    (job_context/profiling/beat/4-thread-trace/analysis.md). Do not attempt a
    g13+g14 merge without a fresh compile proving allocation <=512 and zero
    spill.

missing: []

unresolved:
  - r5.i1.g13
  - r5.i2.g14
  - r5.i3.g15
  - r1.i4.g04

control:
  turn: 1
  stances:
    support: 0
    modify: 4
    oppose: 0
  candidate_ids:
    - r5.i1.g13
    - r5.i2.g14
    - r5.i3.g15
    - r1.i4.g04
  unresolved:
    - r5.i1.g13
    - r5.i2.g14
    - r5.i3.g15
    - r1.i4.g04
  missing: 0
```

## 03_response -- planner

第 3 步的规划方回复（第 1 轮）如下。共 2 个 accept、2 个 partial、0 个 reject；三个候选的执行顺序与审阅方一致，为 [g13, g14, g15]。同样内容的英文版已存为 `rounds/005/2-plan/review_planner.yaml`，已确认能正常解析为 YAML。

- **g13（partial）：** 我接受审阅方的两点。一是 nop/msb 这类静态计数不能用来估算耗时（纠正 3）。二是只有低于噪声底才关闭 L6。但 +3% 的预测我不让步：唯一能给"藏住 exp 能省多少"定上限的实测是 exp 删除的 −13.4%，而这个数被时钟抬高了（只换数据，sclk 就从 988 升到 1053 MHz）。审阅方的 +5% 同样来自静态重叠计数，也没有经过计价。两个预测都保留，由实测裁决。
- **g14（partial）：** 我承认原预测前后不一致。我把它统一为 1–2%，没有采用审阅方的 ≥2%，因为唯一可比的实测（g10 单独的收益）只有 +1.3%。关闭条件采用审阅方的：只有低于噪声底才关闭该机制。
- **g15（accept）：** 采用审阅方的零效果预测。beat 的 TDM 等待里约 4.4% 出现在每个 WG 首次加载时，稳态部分最多约 1.2%。
- **g04（accept）：** 撤回"已过时"的判定。beat 同样填不满机器，只能说明填充不足不是两者差距的原因，不能说明切分 KV 帮不上忙。

```yaml
turn: 1
by: planner

verdicts:
  - candidate: r5.i1.g13
    stance: partial
    grounds:
      - claim: "接受：nop/msb/v_mov 计数只能作为可行性门槛，不能用来估算耗时（纠正 3），不再作为收益大小的依据"
        value: "nop 62→217，msb 240→757，只保留在 risk 字段"
        from: rounds/005/2-plan/synthesis_notes.md#S1
      - claim: "接受：'预测错误'与'机制失效'分开判定，只有低于 prod 噪声底才关闭 L6"
        value: "prod 噪声底约 0.4–0.5%"
        from: job_context/findings/facts.md
      - claim: "坚持（依据是实测，不是静态计数）：唯一能给 exp 隐藏收益定上限的上卡数据被时钟效应抬高了，支撑不了 +5%"
        value: "r3 删除 exp 后 prod −13.4%；只换数据，prod sclk 就从 988 升到 1053 MHz，时间 −7.2%，cycles 只少 1.2%"
        from: "job_context/findings/facts.md; rounds/005/1-profiling/5-power-wall-analysis/analysis.md"
      - claim: "审阅方的 +5% 依据的也是它排除掉的那类证据：静态重叠计数（proto 中 59/66）加上另一个 kernel 的 trace，同样没有经过计价"
        value: "proto 59/66；本次移植版全 kernel 103/330"
        from: "job_context/findings/route.md (h21); rounds/005/2-plan/synthesis_notes.md#S1"
    change: >
      规划方预测不变：prod_Mcycles_per_call_randn ≤ 1.905（−3%），prod_vs_champion ≥ 1.03。
      审阅方的 ≤ 1.866 / ≥ 1.05 作为另一条预测并列记录，不取平均。falsified_if 改为：prod < 1.03 则规划方预测错误，
      < 1.05 则审阅方预测错误；只有在 0 spill 下 prod 低于当场噪声底（约 0.5%），才关闭 n_block 64 上的 L6。nop/msb 改放 risk。

  - candidate: r5.i2.g14
    stance: partial
    grounds:
      - claim: "接受：原预测前后不一致。1.925–1.945 Mcycles 对应 1–2%，而 ≥1105 TF/s 需要约 2%"
        value: "1105.0 / 1083.33 = 1.020"
        from: rounds/005/1-profiling/benchmark-results.md
      - claim: "不把 2% 作为我自己的预测：这个 kernel 上唯一可比的跨 lane / 求和树削减实测，单独只有约 1.3%；LANE 削减的是同一条依赖链上更小的一段"
        value: "r4.i2.g10 packed row-sum 单独 x1.013 prod"
        from: job_context/findings/facts.md
    change: >
      统一后的规划方预测：Mcycles 1.925–1.945（−1% 至 −2%），vs_champion 1.01–1.02，TF/s ≥ 1094。
      审阅方的 ≥1.02 / ≤1.925 并列记录。接受审阅方的关闭条件：只有低于噪声底才关闭该机制。
      精度、spill、permlanex16 检查不变。

  - candidate: r5.i3.g15
    stance: accept
    grounds:
      - claim: "beat 的 TDM 等待主要出现在每个 WG 首次加载时，不是 g15 针对的稳态排空"
        value: "prologue 约 1.17k cyc/wave × 16 = 18.7k / 423,781 = 4.4%；稳态 ≤ 1.2%"
        from: job_context/profiling/beat/4-thread-trace/analysis.md
      - claim: "原来的条件式预测实际上是两个预测，接受这一批评"
        value: "'若暴露则 ≤1.93；若已被隐藏则 ≥1.97'"
        from: rounds/005/2-plan/step1_planner.yaml
    change: >
      采用零效果预测：Mcycles ≥ 1.954（收益 < 0.5%）；若 ≤ 1.925 则该预测被证伪。
      g15 改为关闭 L13 的对照实验。零结果只有同时满足以下条件才算数：ISA 的干净循环中有 s_wait_tensorcnt 0x2，
      spill 0/0，sclk 与对照相差在 2% 以内。g13、g14 跑完后若本场预算还有剩余才运行。

  - candidate: r1.i4.g04
    stance: accept
    grounds:
      - claim: "beat 的 grid 同样填不满机器，只能说明填充不足不是两者差距的原因；不能说明更多 WG 对当前 kernel 没用，因为切分 KV 会缩短每个 WG 的串行路径。原先判为'已过时'过强"
        value: "256 个 CU 上只有 32 个 WG，有活的 CU ≤ 12.5%；fast 0.547"
        from: rounds/005/1-profiling/6-bound-analysis/analysis.md
    change: "still_valid: true，deferred，conditional。只有同一场测试中当前 kernel 在 fast 上的 warm 时间与 beat 相差在约 15% 以内，才作为过时移入 dead_ends"

adopted_from_missing: []   # 审阅方 missing 为空

unresolved:
  - candidate: r5.i1.g13
    settled_by: "g13 与冠军对比，在 prod 上做 3 场轮换槽位测试，并用 clock_probe 同时记录 Mcycles/call 和 sclk。≥1.05 → 审阅方预测成立；1.03–1.05 → 规划方成立、审阅方错误；噪声底到 1.03 → 两个预测都错，但仍可 ship；低于噪声底 → 关闭 n_block 64 上的 L6"
  - candidate: r5.i2.g14
    settled_by: "同一批测试中，以 op/current 为基础单独跑 g14 一臂。≥1.02 → 审阅方预测成立；1.01–1.02 → 规划方预测成立；低于噪声底 → 每个 tile 的跨 lane 求和不在关键路径上"
  - candidate: r1.i4.g04
    settled_by: "同一场测试中，在 fast 上分别测 beat 与 current 的 warm（连续调用、不 flush）和 flushed 时间。当前 kernel warm 与 beat 相差 ≤ 约 15% → g04 过时；≥ 1.3x → 保留。判定线双方已一致，只因还没测量而未决"

control:
  turn: 1
  stances: {accept: 2, partial: 2, reject: 0}
  candidate_ids: [r5.i1.g13, r5.i2.g14, r5.i3.g15, r1.i4.g04]
  unresolved: [r5.i1.g13, r5.i2.g14, r1.i4.g04]
  resolved: [r5.i3.g15]
  ordering: [r5.i1.g13, r5.i2.g14, r5.i3.g15]   # 双方一致
  missing: 0
```

## 04_close -- reviewer

```yaml
confirmed:
  - r5.i3.g15
  - r1.i4.g04

conceded: []

unresolved:
  - candidate: r5.i1.g13
    position: >
      I retain the >=5% prod prediction: current is only 0.768 of beat
      (rounds/005/1-profiling/benchmark-results.md), while beat places 54.6k
      v_exp operations in WMMA gaps and attributes 38.0% of stalls to v_cvt/v_pk
      dependencies versus 0.5% to LDS reads
      (job_context/profiling/beat/4-thread-trace/analysis.md); neither agent has
      an on-card price for the overlap, so the competing 3% and 5% predictions
      should remain distinct.
    settled_by: >
      Measure g13 versus champion in three slot-rotated prod sessions plus
      clock_probe Mcycles/call and sclk: gain >=5% supports the reviewer;
      3% to <5% supports only the planner; noise-floor to <3% falsifies both
      magnitude predictions but remains shippable; below the measured prod
      noise floor closes L6 at n_block 64.

  - candidate: r5.i2.g14
    position: >
      I retain the >=2% prod prediction: production is core-clock-scaled with
      cycles changing only -1.2% under the data swap
      (rounds/005/1-profiling/5-power-wall-analysis/analysis.md), and softmax
      packing dependencies account for 38.0% of beat's traced stalls
      (job_context/profiling/beat/4-thread-trace/analysis.md); the planner's
      1-2% range remains a separate prediction.
    settled_by: >
      Measure g14 as a separate arm from op/current in the same sessions:
      gain >=2% supports the reviewer; 1% to <2% supports only the planner;
      noise-floor to <1% falsifies both magnitude predictions but remains
      shippable; below the measured prod noise floor closes the per-tile
      cross-lane-reduction mechanism.

control:
  turn: 3
  candidate_ids:
    - r5.i1.g13
    - r5.i2.g14
    - r5.i3.g15
    - r1.i4.g04
  confirmed:
    - r5.i3.g15
    - r1.i4.g04
  conceded: []
  unresolved:
    - r5.i1.g13
    - r5.i2.g14
  missing: 0
```

## 05_finalize -- planner

正在读 route.md 的运营项表和现有 Route 表，准备写 plan.yaml 并重写路线。
正在补做 g04 需要的仪表测量：fast 形状下 beat 与 current 的 warm 对比 flushed 计时（单进程，小形状），完成后写 plan 文件。
qwarm ≈ qflush，说明 L2 冷热几乎不影响；current 单次孤立调用比连续调用多出约 37 µs，beat 只多约 2.6 µs。看一下 current 每次调用在做什么。
Key 名写错了（sleeps 对 sleepS），这是脚本 bug，不是卡的问题。修正后重跑。
Let me write the instrument summary first, then candidates.yaml.
Now plan.md.
Next, update pool.md: add the 3 new entries, annotate g03/g04 with this round's evidence, mark the retire-ready entries, and reorder.
All 12 operator items are in the table and the must is above every idea row. Last step: append the knowledge gap.
本轮选定 **r5.i1.g13**：把 h21 的 L6 跨 tile 流水线移植到 round-4 冠军上。它在 plan 里只做过 compile-only：512 VGPR，0 spill，0 scratch。

- **预测：** prod 的 Mcycles/call（randn）从 1.964 降到 ≤1.905，prod vs_champion ≥1.03；审阅方预测 ≥1.05。只有 prod 低于本场噪声底（约 0.4%），才关闭 n_block 64 上的 L6。
- **Route 第 1 行必须是 h22：** 它是 `must`，框架要求排在所有 idea 行之上，由 r5.i2.g14 完成。act 会把 g14 和 g13 各自从 op/current 建成独立的一臂，在同一组 3 场轮换槽位的测试里分别和对照比较。两者不合并，因为 g13 已经用满 512 VGPR。所以这不算打包，收益能分开归因。
- **发布规则：** 哪一臂的 prod 更高且高于噪声底就发布哪一臂；都不过底就都不发。如果只有 g14 过线，g13 的预测记为证伪。
- **g15：** 可选的第三臂，预测为零收益，用来关闭 L13。
- **对话结果：** `ended_by: unresolved`。g13（我 ≥3%，审阅方 ≥5%）和 g14（我 1–2%，审阅方 ≥2%）两处分歧已照原文写入 plan.yaml，各附判定区间。

**本轮在卡上补测了 g04 的判定条件**（fast 形状，current 对 beat，每次只跑一个进程，dmesg 无 amdgpu 故障），数据在 `raw/warm_flush/summary.md`：
- **判定线没有落到任何一边。** 两次连续调用（warm）的比值是 1.24 和 1.19，正好在双方约定的阈值之间（≤1.15 判过时，≥1.3 保留）。
- **打分的 fast 差距主要不在 kernel 本身。** 约 41 µs 的差距里，约 36 µs 是只有 current 才付的单次调用固定开销。它与 L2 冷热无关（74.71 对 75.47 µs），也不是 host launch（16.43 对 16.60 µs）。
- **这个开销随调用前的空闲时间变长。** current 连续调用 38 µs，空闲 1.8 ms 后 71 µs；beat 始终 32–35 µs。
- **这就是 pool 里 r1.i3.g03 的惩罚。** 这两条已写进 g03 和 g04 的条目；route 里 g03 排在 g04 前面，两者都在 prod 各行之下。

**pool 标记：** r1.i2.g02 和 r2.i2.g06 标为 `stale`，由 reflect 退役。r4.i2.g10 已在 round 4 发布，标注请 reflect 移入 facts.md。

**知识缺口：** 新增一条，关于这个随空闲时间增长的开销。"CU 空闲断电导致 I$ 丢失"只是我的推断，已标 unverified。

```yaml
selected: r5.i1.g13
prediction: {metric: prod_Mcycles_per_call_randn, from: 1.964, to: "<=1.905 (prod vs_champion >=1.03; reviewer >=1.05)"}
added_to_pool: [r5.i1.g13, r5.i2.g14, r5.i3.g15]
retired_or_marked_stale: [r1.i2.g02, r2.i2.g06]   # r4.i2.g10 marked "shipped r4" for reflect to move to facts
unresolved_disagreements: [r5.i1.g13, r5.i2.g14]
route_updated: true
files: [plan.yaml, plan.md, candidates.yaml]
```

Files are in `rounds/005/2-plan/`:
- plan.yaml
- plan.md
- candidates.yaml
- raw/warm_flush/summary.md
