# Dialogue

## 00_analyse -- planner

正在阅读本轮 profiling 报告和运行记忆（facts/dead_ends/pool/route），接下来核对 round 10 的 profiling 是否为新数据，以及 round 9 的结论。
正在读 beat 的 thread-trace 分析和 recipes 第 6 节，寻找本 job 尚未提出的机制。
正在核对 round 9 的冷启动实验：benchmark.py 每次调用前的 L2 flush 本身是否就是一个“外来 kernel”，从而让冷启动计入得分。
正在读 benchmark.py 的计时循环，确认 fast 形状下 current 的中位数（52 us）与最小值（39 us）差距来自哪里。
正在检查 current 的 ISA：P 转 bf16 用的是硬件 `v_cvt_pk_bf16_f32` 还是软件舍入序列（beat 的最大 stall 类就在这里）。
## Step 0（planner）：结论

**这轮的 profiling 不是新数据。** `rounds/010/1-profiling` 和 `rounds/005/1-profiling` 用 `diff -r` 比对完全一致，`timing.yaml` 里 profiling 耗时为 0.0 s，但 `reused_from_round` 写的是 `null`。数据本身仍然适用：round 4 之后没有任何一轮被接受（`best_round: 4`），所以 op/current 就是当时被 profile 的那份代码。缺的数据有这些：3-kernel-metrics 跳过（没装 rocprof-compute），4-thread-trace 跳过（ATT 看不到 FlyDSL kernel），没有字节计数器，没有 stall 计数器，WMMA 计数器读数为 0。

**prod 的瓶颈：同意 profiling 的判断，是 latency，置信度低。** 依据有四条：
- HBM 已排除：sclk 从 988 升到 1053 MHz 时，每次调用的周期数只变了 -1.2%。
- 机器已被填满：SQ busy 0.981。
- 同一张卡上，beat 每周期完成的工作量是 current 的 1.30 倍。
- r9 的 lock_simd 实验表明，同 SIMD 上的伙伴 wave 确实在帮忙掩盖延迟，贡献至少 3.5%。

循环体内部的杠杆基本用完了：9 轮里唯一赢过的是 nodelay，只有 +0.6~1.1%。

**我的主要分歧：下一个能计分的大项在 fast 形状，不在循环体。** round 9 在不跑 beat 的进程里测过（`rounds/009/1-opt/raw/meas/*.out`）：
- current 在 fast 上的中位数是 51.2 到 55.7 us，最小值 38.7 到 39.3 us，中位数是最小值的 1.32 到 1.42 倍。
- beat 的中位数只比它自己的最小值高 4%（1.04 倍）。
- 39 us 正是 current 干净热态下的时间（r6 和 r9 都测到过），所以最小值是真实状态，不是偶然的离群点。

r6 当初得出"256 MB flush 不影响"，但那次测法是在 `zero_` 和调用之间插了 `_sleep(1e5)`。benchmark.py 在 flush 之后是直接计时的，这个计分用的调用顺序在不跑 beat 的进程里从来没测过。如果能把这一项消掉，fast 能快约 25%，几何平均约 +8%。这个量级超过 9 轮以来任何一个循环体杠杆。

```yaml
constraint:
  bound: latency
  shape: prod
  agrees_with_profiling: true
  grounds:
    - {claim: "Not HBM: sclk 988->1053 MHz, time -7.2%, cycles/call 1.964->1.941 M", value: "-1.2% cycles", from: rounds/010/1-profiling/5-power-wall-analysis/analysis.md}
    - {claim: "Grid fills the machine at prod", value: "SQ_BUSY/SQ_CYCLES 0.981, 4096 WGs", from: rounds/010/1-profiling/6-bound-analysis/analysis.md}
    - {claim: "Beat 1.30x work/cycle at 1 wave/SIMD; clock explains <=11%", value: "1.302; 1100/988", from: rounds/010/1-profiling/6-bound-analysis/analysis.md}
    - {claim: "Partner-wave overlap carries >=3.5%", value: "lock_simd prod 0.963-0.967", from: job_context/findings/dead_ends.md r9.i2.g25}
    - {claim: "Only body win in 9 rounds", value: "nodelay +0.62..+1.10% prod, 6/6", from: job_context/findings/facts.md r8.i1.g20}
    - {claim: "Scored fast carries a current-only excess, no beat in process", value: "current median 51.24/52.20/55.68 vs min 39.14/38.74/39.30 us; beat 32.53/33.13/33.29 vs 31.25/31.77/31.73", from: rounds/009/1-opt/raw/meas/*.out}
    - {claim: "That min is current's clean warm time", value: "38.9-39.9 us", from: rounds/006/1-opt/opt.md s3c; rounds/009/1-opt/opt.md s2b}
    - {claim: "r6's 'flush doesn't matter' put _sleep between zero_ and the call; benchmark.py does not", value: "zero_; _sleep(1e5); current", from: rounds/006/1-opt/opt.md s3c; job_context/op/benchmark.py:110-116}
candidates:
  - id: r10.i1.g27
    title: "Post-flush load latency at small grids: depth-3 TDM ring (u1pp3) selected only when WGs <= CUs"
    proposed_by: planner
    round_proposed: 10
    hypothesis: >-
      Dispatched right after a 256 MB zero_, current sees elevated load latency while the write-back drains. Its
      depth-2 ring drains tensorcnt to 0 every 64-KV tile, so at fast (32 WGs, 16 tiles) that latency is exposed
      every tile. Beat keeps 0x4 in flight and hides it. A depth-3 ring hides it. The launcher picks it only when
      grid <= CU count, so prod/proxy keep the champion binary.
    prediction:
      metric: fast_median_us (h28, no beat, >=3 rotated sessions)
      from: "51.2-55.7"
      to: "<=45 (fast vs_champion >= 1.12)"
      also_expect:
        discriminator_first: "current-only fast: A zero_;e0;f ~52 | B zero_;_sleep;e0;f <=41 | C _sleep;zero_;e0;f >=48; beat 31-34 in all three"
        prod: "within noise (identical binary)"
        fast_min_us: "<=40.5"
    falsified_if: "C <= 42 us -> host-launch gap, not drain (drop the kernel arm, re-route to host path); or C >= 48 but u1pp3 fast median > 49 -> not ring depth (fall back to r9.i3.g26 timestamps)"
    evidence:
      - {claim: "median/min asymmetry fast", value: "1.32-1.42 vs 1.04", from: rounds/009/1-opt/raw/meas/*.out}
      - {claim: "same asymmetry at proxy, smaller", value: "1.05/1.06/1.18 vs 1.01-1.02", from: rounds/009/1-opt/raw/meas/*.out}
      - {claim: "flush without sleep never tested current-only", value: "idle6 used _sleep(1e5)", from: rounds/006/1-opt/opt.md s3c}
      - {claim: "TDM drain depth", value: "tensorcnt 0x0 vs beat 0x4", from: job_context/findings/route.md h4}
      - {claim: "u1pp3 builds clean", value: "VGPR 456, SGPR 105, 0 spill, loop 530 vs 513", from: rounds/008/1-opt/opt.md s2d}
      - {claim: "a second own image costs nothing", value: "self_nc 39.7/40.9 vs none 39.3/41.0 us", from: rounds/009/1-opt/opt.md s2b}
    justified_by_round: 10
    source: {kind: internal, repo: null, commit: null, verified: false}
    scope: {files: [op/current/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py, op/current/flydsl_fwd/fmha_b16_buffer_managers.py], backend_feasible: unknown}
    risk: "Drain mechanism inferred; discriminator can kill it cheaply. Round-2 proto: port the diff (h27). Only small grids run the variant, so 49 dB x16 and 200-run determinism must include a <=256-WG causal and non-causal case."
  - id: r8.i1.g20
    title: "Re-land nodelay alone as the round's base"
    proposed_by: planner
    round_proposed: 8
    hypothesis: "The 14 s_delay_alu/tile sit on the serial per-wave issue path (513->499); the partner wave hides only part of it."
    prediction: {metric: prod vs_champion (h28, >=3 sessions), from: 1.000, to: "1.005-1.012 every session", also_expect: {fast: "0.99-1.01", proxy: "0.99-1.01", hash: "26a89a2db0cd"}}
    falsified_if: "any prod session < 1.003"
    evidence:
      - {claim: "prod 6/6", value: "1.0062-1.0110", from: job_context/findings/facts.md}
      - {claim: "no-beat fast/proxy neutral", value: "1.002/1.004", from: job_context/findings/facts.md}
      - {claim: "code unchanged since measured", value: "best_round 4", from: job_context/state.yaml}
    justified_by_round: 9
    source: {kind: internal, repo: null, commit: null, verified: true}
    scope: {files: [op/current/flydsl_fwd (both llvm_options dicts)], backend_feasible: yes}
    risk: "~+0.3% geomean alone, under the 0.70% bar; ships merged only with an arm that won alone (h29). Disjoint from g27."
  - id: r10.i3.g28
    title: "In-WG q-tile pairing (h25 proto): overlap O-store drain with the next tile's first TDM load"
    proposed_by: planner
    round_proposed: 10
    hypothesis: "LDS 327,680 B/WG => 1 WG/CU; prod runs 16 WGs/CU serially and each boundary exposes first-load + drain. Pairing halves boundaries and keeps tile 1's async O store under tile 2's prologue."
    prediction: {metric: prod vs_champion (h28, >=3 sessions), from: 1.000, to: "1.005-1.015", also_expect: {proxy: ">=1.00", fast: "within noise", clean_loop: "== champion after port"}}
    falsified_if: "prod <= 1.003 in any session, or ported clean loop longer than champion's (then no card time)"
    evidence:
      - {claim: "1 WG/CU", value: "lds_bytes 327680", from: rounds/010/1-profiling/kernel.yaml}
      - {claim: "beat's WG boundary cost", value: "3,133 + 2,444 of ~86.5k cycles/wave; 3.3% in two stalls", from: job_context/profiling/beat/4-thread-trace/analysis.md #5}
      - {claim: "proto clean", value: "VGPR 447, 0 VGPR spill, bijection ALL OK", from: job_context/findings/route.md h25}
    justified_by_round: 10
    source: {kind: internal, repo: null, commit: null, verified: true}
    scope: {files: [op/current/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py], backend_feasible: yes}
    risk: "Beat's 3.3% is beat's; current's boundary cost is unmeasured. +1 barrier, +1 s_wait_asynccnt per pass. Never PAIR_POLICY=overfill. Lowest priority."
from_pool:
  - {id: r5.i3.g15, still_valid: false, why: "Priced out as a prod arm (+3.3% instr/tile for <0.5%). Its build is reused by g27 under the fast post-flush justification. Retire."}
  - {id: r6.i1.g16, still_valid: false, why: "r9: cold start fixed, does not scale with image size. Retire."}
  - {id: r1.i3.g03, still_valid: false, why: "Answered by r6/r9. Retire."}
  - {id: r1.i4.g04, still_valid: false, why: "Clean warm ratio 1.13 < 1.15; underfill identical in both arms. The fast excess is g27's term."}
  - {id: r8.i2.g21, still_valid: false, why: "p12s 0.958-0.972 (r8). Retire."}
  - {id: r7.i3.g19, still_valid: true, why: "Consistent with r9 (desync, don't starve) but needs a +3.3% ring at prod and must gross >3%. Not now."}
  - {id: r9.i3.g26, still_valid: true, why: "Re-justified on the scored flushed state; g27's fallback if drain is confirmed but the ring doesn't fix it."}
knowledge_gaps:
  - {question: "How long does a 256 MB zero_ keep gfx1250 memory busy, and what latency does the next kernel see?", expected_in: "knowledge/pitfalls/measurement-traps.md or knowledge/arch/gfx1250/gfx1250.md", did_instead: "median vs min from rounds/009 raw; 3-pattern discriminator; appended to job_context/knowledge_gaps.md"}
  - {question: "Profiling reuse unmarked (byte copy of round 5, reused_from_round: null)", expected_in: rounds/010/1-profiling/profiling.yaml, did_instead: "diff -r + state.yaml best_round 4 confirm the code is unchanged"}
```

**执行顺序：**
1. **先跑 g27 的 A/B/C 三组对照**，不需要编译，约 10 分钟卡时。三组都只跑 current、fast 形状、不跑 beat：
   - A：`zero_` 后直接计时，和 benchmark.py 一样。
   - B：`zero_` 后先 sleep 再计时。
   - C：先 sleep 再 `zero_`，然后直接计时。这组把 host 启动开销藏在 sleep 后面，但不给 flush 留排空时间。

   如果 C 的中位数 ≤ 42 us，说明多出来的时间是 host 启动空隙，就不做 kernel 改动，转去优化 host 路径。
2. **合入 nodelay (g20) 作为本轮基底。** 它和 g27 改的是不相干的两处。
3. **最后才做配对 (g28)。**

分析全文已写入 `rounds/010/2-plan/step0_planner.yaml`，知识缺口已追加到 `job_context/knowledge_gaps.md`。

## 00_analyse -- reviewer

```yaml
constraint:
  bound: latency
  shape: prod
  agrees_with_profiling: true
  grounds:
    - claim: "HBM bandwidth is excluded: changing operand data raised shader clock while leaving memory volume unchanged, and elapsed time followed core clock rather than HBM clock."
      value: "sclk 988 -> 1053 MHz; time 1.9870 -> 1.8439 ms (-7.2%); cycles/call 1.964 -> 1.941 M (-1.2%)"
      from: rounds/010/1-profiling/5-power-wall-analysis/analysis.md
    - claim: "Production has enough workgroups to fill the machine, so grid underfill is not the production constraint."
      value: "4096 workgroups, 32768 waves on 256 CUs; SQ_BUSY_CYCLES/SQ_CYCLES = 0.981"
      from: rounds/010/1-profiling/6-bound-analysis/analysis.md
    - claim: "The remaining production gap is on-chip execution efficiency."
      value: "current 1083.33 TFLOP/s versus beat 1410.87 TFLOP/s; ratio 0.768"
      from: rounds/010/1-profiling/benchmark-results.md
    - claim: "Latency is only the best available label; compute issue, dependency latency, and on-chip memory cannot be separated."
      value: "no byte counters; WMMA counters read zero; stall counters rejected; PC sampling forbidden; ATT captured no FlyDSL kernels"
      from: rounds/010/1-profiling/6-bound-analysis/analysis.md

candidates:
  - id: r8.i1.g20
    title: "Re-land nodelay alone by disabling LLVM delay-ALU insertion"
    proposed_by: planner
    round_proposed: 8
    hypothesis: >
      LLVM's fourteen s_delay_alu instructions per KV tile are unnecessary for this build and
      lengthen the production kernel's serial on-chip issue path. Setting
      amdgpu-enable-delay-alu=False in both launch configurations should retain the useful
      LO/HI partner-wave arbitration while removing those delays.
    prediction:
      metric: prod_vs_champion_no_beat
      from: 1.000
      to: ">=1.007"
      also_expect:
        prod_tflops_same_session: ">= champion_tflops * 1.007"
        fast_vs_champion_no_beat: "0.995-1.010"
        proxy_vs_champion_no_beat: "0.995-1.010, interpreted only across rotated sessions because proxy is noisy"
        isa_gate: "all 14 per-tile s_delay_alu removed; VGPR <=464 corrected allocation; scratch 0"
        correctness: "bitwise equal to champion and all specification cases >=49 dB"
      falsified_if: >
        The mean prod ratio is below 1.007 across at least three no-beat, slot-rotated sessions,
        any session has a material prod regression, or correctness differs from the champion.
    evidence:
      - claim: "The current production kernel remains 23.2% below beat in a core-scaled regime."
        value: "current/beat = 0.768; 1083.33 versus 1410.87 TFLOP/s"
        from: rounds/010/1-profiling/benchmark-results.md
      - claim: "The current build has no scratch spill, so the candidate is not an occupancy-forcing proposal."
        value: "scratch_bytes = 0; corrected VGPR estimate = 464"
        from: rounds/010/1-profiling/kernel.yaml
      - claim: "Nodelay removed fourteen delay instructions per tile and previously won at prod."
        value: "loop 513 -> 499; prod ratios 1.0074/1.0069/1.0072; zero spill and bitwise-identical output"
        from: job_context/findings/facts.md
      - claim: "The production gain reproduced in round 9 when measured without beat in the candidate process."
        value: "prod ratios 1.0110/1.0084/1.0092; six winning sessions total across rounds 8-9"
        from: job_context/findings/route.md
    justified_by_round: 10
    source:
      kind: internal
      repo: null
      commit: null
      verified: true
    scope:
      files: [op/current/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py]
      backend_feasible: yes
    risk: >
      The expected gain is close to the configured 0.7% acceptance floor. Measure candidate
      versus champion without beat in the process and do not merge another arm unless that arm
      independently wins at prod.

  - id: r10.i1.g27
    title: "Replace the running softmax maximum with a fixed zero reference"
    proposed_by: reviewer
    round_proposed: 10
    hypothesis: >
      The per-tile row maximum, accumulator rebasing, and conditional O rescale form a
      loop-carried dependency on the production path. A fixed zero reference makes the rescale
      identically one and removes that recurrence; retain the current packed row-sum implementation
      so this arm tests the fixed-reference mechanism alone.
    prediction:
      metric: prod_Mcycles_per_call_randn
      from: 1.964
      to: "<=1.905 (-3% or more)"
      also_expect:
        prod_vs_nodelay_no_beat: ">=1.03"
        prod_sclk: "reported beside time; no prediction because reduced switching may raise it"
        correctness: >
          O and natural-log LSE pass the fp32 reference on the normal suite plus adversarial
          large-logit, late-maximum, all-masked-block, and fully-masked-row cases.
        build_gate: "scratch 0 and no new spill"
      falsified_if: >
        Any adversarial correctness case fails, determinism changes, prod cycles remain above
        1.905 M, or prod throughput is below 1.03 of the nodelay base in three rotated sessions.
    evidence:
      - claim: "Production time tracks on-chip cycles rather than HBM service."
        value: "randn-to-zeros cycles changed only -1.2% while time changed -7.2% with sclk 988 -> 1053 MHz"
        from: rounds/010/1-profiling/5-power-wall-analysis/analysis.md
      - claim: "The production kernel has a substantial unexplained per-cycle deficit to the reference."
        value: "current/beat = 0.768"
        from: rounds/010/1-profiling/benchmark-results.md
      - claim: "Softmax work has already produced material gains on this exact kernel family."
        value: "removing exp was a 13.4% timing ceiling; packed exp plus packed row sum improved prod 6.3%"
        from: job_context/findings/facts.md
      - claim: "A measured FlyDSL forward uses a fixed zero reference to eliminate row-max and rescale work."
        value: "fixed-reference max measured +13.5% at D=64; the mechanism is confirmed present at D=128 but not re-priced there"
        from: knowledge/backends/flydsl/attention/recipes/hd64.md
    justified_by_round: 10
    source:
      kind: external
      repo: AMD-AGI/Primus-Turbo
      commit: ed8d7af46938d8debe56fd5ff5b70fd64dde33be
      verified: true
    scope:
      files: [op/current/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py]
      backend_feasible: unknown
    risk: >
      Zero is not a proven upper bound for this job's allowed scaled QK scores. Overflow,
      incorrect fully-masked-row handling, or wrong LSE semantics rejects the candidate
      regardless of speed. This is distinct from dead branch-free rescale r2.i1.g05: it removes
      the rescale recurrence instead of executing the wide rescale every tile.

from_pool:
  - id: r5.i3.g15
    still_valid: true
    why: >
      Its round-5 justification matches the unchanged production regime, but it predicts a
      sub-noise null and is useful only as a depth-three-ring prerequisite. Do not execute before
      r8.i1.g20. Current evidence remains 1.964 M cycles/call at prod
      (rounds/010/1-profiling/5-power-wall-analysis/analysis.md).

  - id: r6.i1.g16
    still_valid: false
    why: >
      Its code-size premise was falsified: a 1.39x larger image changed the cold penalty by only
      about 3 us, within its own falsification line (job_context/findings/dead_ends.md). Round 10
      also reports warm I-cache misses only on initial dispatches
      (rounds/010/1-profiling/2-kernel-profiling/kernel-kn_fmha_fwd_prefill_a16w16_m32x8_bshd/analysis.md).

  - id: r1.i3.g03
    still_valid: false
    why: >
      The claimed idle-sensitive penalty was corrected to a foreign-kernel cold-start state and
      is outside scoring under the current protocol (job_context/findings/pool.md). It does not
      explain the round-10 prod ratio of 0.768
      (rounds/010/1-profiling/benchmark-results.md).

  - id: r1.i4.g04
    still_valid: false
    why: >
      The clean warm current/beat fast ratio was 1.13, below its agreed 1.15 stale threshold, and
      a deterministic combine dispatch would consume most of the remaining gap
      (job_context/findings/pool.md). Round 10's 0.547 flushed ratio therefore does not revive it
      (rounds/010/1-profiling/benchmark-results.md).

  - id: r7.i3.g19
    still_valid: false
    why: >
      As written it predicts against an h24 winner, but both h24 implementations lost and its
      depth-three carrier costs work before creating the phase offset
      (job_context/findings/dead_ends.md; job_context/findings/pool.md). Round 10 still has an
      on-chip production constraint, but that does not restore the missing prerequisite
      (rounds/010/1-profiling/6-bound-analysis/analysis.md).

  - id: r8.i2.g21
    still_valid: false
    why: >
      It was measured at only 0.958-0.972 of champion and is closed
      (job_context/findings/route.md). The round-10 full production grid and 0.981 SQ-busy ratio
      provide no changed regime that would reopen it
      (rounds/010/1-profiling/6-bound-analysis/analysis.md).

  - id: r9.i3.g26
    still_valid: true
    why: >
      The fixed cold-start location remains unknown, but this is a diagnostic for a state excluded
      from scoring, not an optimization candidate. Defer it while prod remains at 0.768 of beat
      (rounds/010/1-profiling/benchmark-results.md).

knowledge_gaps:
  - question: >
      What job-wide upper bound on scaled QK scores makes a fixed-zero softmax reference safe,
      including adversarial bf16 inputs and fully-masked rows?
    expected_in: >
      knowledge/ops/attention/online-softmax.md or
      knowledge/backends/flydsl/attention/recipes/hd128.md
    did_instead: >
      Recorded the gap in job_context/knowledge_gaps.md and made adversarial large-logit,
      late-maximum, all-masked-block, and fully-masked-row comparison against the fp32 reference
      a mandatory falsification gate for r10.i1.g27.
```

## 01_synthesis -- planner

整理 synthesis_notes.md 时拼接出错了，现在整份重写。
## Step 1 综合（planner）

一致的部分只说一次：
- 约束是 prod latency，置信度低。
- 以 nodelay r8.i1.g20 为基底，它是唯一的 `must`。
- g16、g03、g04、g21 失效。

reviewer 的固定零参考（fixed zero reference）我按原 id `r10.i1.g27` 采纳。两边的 step 0 都用了 `r10.i1.g27`，所以我的 post-flush 候选改编为 `r10.i4.g29`。

有两处实质分歧：
1. **g27 能否通过 h16。** 用 fp32 exp2 算，参考值固定为 0 时，+1e4 的分数会上溢成 inf；整行都在 −1e4 附近时，l 会下溢为 0。h16 用例正好覆盖这两种输入。要检测这两种情况就必须重新求行最大值，也就等于把 g27 想删掉的那条链加回来。
2. **fast 实测里的额外开销是否计分。** 这组数据来自 round 9，进程里没有 beat：current 的 median/min 比值是 1.31–1.42，beat 是 1.04。所以这不是 h28 定义的冷启动，而是会计入分数的。round 6 验证“flush 无影响”时，`zero_` 之后加了 sleep，而正式计分的流程里没有这个 sleep。

详细论证已写入 `rounds/010/2-plan/synthesis_notes.md` 的 s1–s4，数值都已对照 `rounds/009/1-opt/raw/meas/*.out` 核实过。

```yaml
candidates:
  - id: r10.i4.g29
    proposed_by: planner
    hypothesis: >
      scored fast 的 ~12-16 us current 独有超额，来自 256 MB flush.zero_() 写回排空与 kernel 重叠，
      暴露在 depth-2 TDM ring（tensorcnt 0x0/tile）上；grid WG 数 <= CU 数时改用 depth-3 ring 可隐藏。
    evidence:
      - "rounds/009/1-opt/raw/meas/s{1,2,3}_fast.out arm=inc: median/min 52.20/38.74, 51.24/39.14, 55.68/39.30 us (1.31-1.42)"
      - "beat_fast.out 32.53/31.25 (1.04); val_merged/val_lock beat 33.13/31.77, 33.29/31.73"
      - "rounds/006/1-opt/opt.md s3c 测的是 zero_; _sleep(1e5); current，benchmark.py:113-116 无 sleep"
    step_a_discriminator: "current-only, fast, 101 reps, 无 build：A zero_;e0;f  B zero_;sleep;e0;f  C sleep;zero_;e0;f"
    prediction: "A ~52 us 且 B <= 41 us 且 C >= 48 us；随后 depth-3 变体 fast median <= 44 us，prod 不变（不同 dispatch path）"
    falsified_if: "A ~= B（无 flush 项）或 C <= 42 us（是 host gap，不是 drain）"
  - id: r8.i1.g20
    proposed_by: planner
    hypothesis: "nodelay re-land (amdgpu-enable-delay-alu=False)，h29 唯一 must"
    prediction: "prod vs_champion >= 1.007（此前 6/6 为 1.0062-1.0110）"
  - id: r10.i1.g27
    proposed_by: reviewer
    hypothesis: "固定零参考，删除 loop-carried max/rescale 递推（Primus-Turbo hd64 a1）"
    prediction: "prod cycles <= 1.905 M，相对 nodelay >= 1.03"
    gate: "先跑 torch fp32 仿真 h16 用例集（+-1e4、单行 outlier、首 tile max 远低于后续），o/lse 有非有限值或 lse 误差 > 1e-3 即判死，不占卡时"
  - id: r10.i3.g28
    proposed_by: planner
    hypothesis: "h25 pairing（PAIR_POLICY=exact，只配 prod+proxy；VGPR 447、SGPR 107、5 次 SGPR->lane spill）"
    prediction: "prod +0-1.5%，且必须高于 noise floor 才可 ship"
adopted_from_reviewer:
  - id: r10.i1.g27
    note: "原样采纳 reviewer 的 id 与 prediction，前置 h16 fp32 仿真门槛"
  - id: r5.i3.g15
    note: "采纳 reviewer 的判断：不再作为 prod arm，只作为 ring 前提；具体由 g29 承接"
disagreements:
  - candidate: r10.i1.g27
    planner:
      position: "未加保护时 h16 必败；加保护就要每 tile 求 row max，退化成现有 deferred rescale 再加额外开销"
      grounds:
        - "route.md:100-104 h16 要求 +-1e4 after scale"
        - "fp32 exp2 在参数 >= 128（natural 约 88.7）时上溢；全负行 l=0 -> o=0/0，lse=-inf"
        - "hd64.md:1106-1108：+13.5% 是 gfx950 D=64 的引用值，未重测；hd64.md:1103 说明 head_dim=128 未测"
    reviewer:
      position: "去掉 critical path 上的 max/rescale 链，prod 收益空间最大；已有 softmax 类收益实测支撑"
      grounds: ["facts: row-max peer() 喂 exp，在关键路径上", "Primus-Turbo ed8d7af4 已发货"]
    settled_by: "h16 用例集 torch fp32 仿真（免费），通过后再看 prod cycles <= 1.905 M"
  - candidate: r10.i4.g29 / r9.i3.g26
    planner:
      position: "fast/proxy 的 current 独有超额在无 beat 条件下也存在，因此计入分数"
      grounds: ["上表 s1-s3 inc 与 beat 的 median/min 对比", "round 6 s3c 带 sleep，与计分流程不同"]
    reviewer:
      position: "g26 诊断的是 h28 之外、不计分的冷启动状态，应推迟"
      grounds: ["h28: 计分时进程内无 beat", "r9 s2b-s2g: full-grid current 调用可清除冷态"]
    settled_by: "A/B/C 判别实验（约 10 min，无 build）"
  - candidate: r7.i3.g19
    planner: {position: "有效，但推迟", grounds: ["作用于 prod dispatch，与 h24 G=2 barrier 失败不是同一机制"]}
    reviewer: {position: "失效", grounds: ["h24 前提（G=2）已判死"]}
    settled_by: "本轮不排期；若 g29 证实 ring 深度在 prod 有代价，g19 随之死"
ordering:
  proposed: [r10.i4.g29-step_a, r8.i1.g20, r10.i1.g27, r10.i3.g28, r10.i4.g29-step_b]
  why: >
    g29 的 A/B/C 判别实验最便宜，也会推翻其他结论的前提：它决定 g29/g26 能否成立，
    也决定所有 arm 的 fast/proxy 读数是否可信，所以排第一。
    nodelay 是唯一 must，作为之后各 arm 的基底。
    g27 的 prod 上限最高，但先过免费的 h16 仿真，不过就直接判死。
    g28 期望值 +0-1.5%，接近 noise floor，排在后面。
    depth-3 ring 实现（step_b）只在 step_a 证实 drain 假设后才做。
```

## 02_review -- reviewer

```yaml
turn: 1
by: reviewer

verdicts:
  - candidate: r10.i4.g29
    stance: modify
    grounds:
      - claim: "There is a large scored fast gap worth isolating before another kernel rewrite."
        value: "warm current 35.3 us versus flushed benchmark 56.16 us; 32 WGs on 256 CUs"
        from: rounds/010/1-profiling/6-bound-analysis/analysis.md
      - claim: "This round cannot attribute that gap to TDM or memory-system drain."
        value: "no HBM/L2/LDS byte counters, no stall counters, WMMA counters zero, and no FlyDSL ATT trace"
        from: rounds/010/1-profiling/6-bound-analysis/analysis.md
      - claim: "The scored harness really queues the 256 MiB zero immediately before the start event and operator."
        value: "flush.zero_(); ev0.record(); call(label)"
        from: job_context/op/benchmark.py:113
      - claim: "The no-beat fast measurements contain a repeatable median/minimum split."
        value: "inc 52.20/38.74, 51.24/39.14, and 55.68/39.30 us"
        from: rounds/009/1-opt/raw/meas/s1_fast.out
    risk: >
      The proposed A/B/C test can establish an order-dependent predecessor effect, but C cannot
      uniquely label it host launch: A and C both put the same 256 MiB zero directly before the
      event, and differ additionally in pre-zero GPU state. Even if a write-drain effect exists,
      that does not establish that tensorcnt depth is where the 12-16 us is exposed.
    alternative: null
    change: >
      Keep the discriminator first, but add controls that vary memory traffic without changing
      host enqueue opportunity: GPU-sleep->event->current, 8 MiB zero->event->current,
      256 MiB read->event->current, 256 MiB zero->event->current, and
      256 MiB zero->GPU-sleep->event->current. The write-drain premise holds only if the penalty
      grows with write volume, is absent or materially smaller after the matched no-write
      predecessor/read control, and decays when delay follows the write. Run beat controls in a
      separate process. Only then build the depth-3 arm. After r8.i1.g20 is re-landed, repeat the
      decisive direct-write versus no-write control on that actual base before pricing the ring.
      The small-grid dispatch must leave the prod kernel ISA-identical to nodelay, and prod must
      still be measured same-session: a large fast win cannot pay for a prod regression.

  - candidate: r8.i1.g20
    stance: modify
    grounds:
      - claim: "Production remains materially behind beat in the same on-chip regime where nodelay was measured."
        value: "1083.33 versus 1410.87 TFLOP/s; current/beat 0.768"
        from: rounds/010/1-profiling/benchmark-results.md
      - claim: "The current build remains spill-free."
        value: "scratch_bytes 0; corrected VGPR allocation estimate 464"
        from: rounds/010/1-profiling/kernel.yaml
      - claim: "Nodelay has already won production in six same-session comparisons."
        value: "prod ratios 1.0062-1.0110 across rounds 8-9"
        from: job_context/findings/facts.md
    risk: >
      The stated prediction and falsification line disagree: predicting 1.005-1.012 in every
      session but falsifying only below 1.003 leaves 1.003-1.005 uninterpretable. Requiring a
      speculative merge also repeats round 9's process error, where the winning nodelay arm was
      buried under losing lock_simd.
    alternative: null
    change: >
      Predict the aggregate that decides the result: mean prod_vs_champion >=1.007 across at least
      three rotated no-beat sessions, with every session above 1.003 and no material prod
      regression. Falsify the magnitude prediction if the mean is below 1.007. Re-land and retain
      nodelay alone as the working base; merge another arm only after that arm independently wins
      against nodelay.

  - candidate: r10.i1.g27
    stance: oppose
    grounds:
      - claim: "This round identifies only generic on-chip execution inefficiency, not the running-max recurrence specifically."
        value: "latency, confidence low; compute/issue cannot be excluded"
        from: rounds/010/1-profiling/6-bound-analysis/analysis.md
      - claim: "A fixed reference is correct only when it is a valid upper bound for every score in the row."
        value: "the exponential argument must remain <=0; otherwise range and overflow handling return"
        from: knowledge/ops/attention/online-softmax.md
      - claim: "The mandatory max-tracking gate includes scores far outside the safe range of a zero reference."
        value: "scaled scores approximately +/-1e4, late large outliers, and first-tile maxima below later maxima"
        from: job_context/findings/route.md
      - claim: "The borrowed performance number is not a D=128 measurement."
        value: "fixed reference +13.5% was cited from the older D=64 result; D=128 presence was confirmed but not re-priced"
        from: knowledge/backends/flydsl/attention/recipes/hd64.md
    risk: >
      This is not merely a low-confidence performance bet. With reference zero, a positive
      1e4 score overflows exp2 and an all-large-negative row underflows its denominator, so the
      candidate necessarily fails the required input domain. Recovering safety requires computing
      a row upper bound, reinstating the dependency it proposes to remove.
    alternative: >
      Retain the running maximum. Spend the round on the post-flush discriminator, re-land
      r8.i1.g20, and then price r10.i3.g28 or the conditionally supported depth-3 path. Do not
      build or simulate fixed-zero softmax unless the operator contract first supplies a finite
      score upper bound; no such bound is present here.

  - candidate: r10.i3.g28
    stance: modify
    grounds:
      - claim: "The resource fact supports one resident workgroup per LDS-owning unit, but not the stronger claim of exactly one WG per CU."
        value: "LDS 327680 B/WG; CU-versus-WGP ownership mapping unresolved"
        from: rounds/010/1-profiling/6-bound-analysis/analysis.md
      - claim: "Production already fills the machine, so pairing has no production grid-fill premise."
        value: "4096 WGs, 32768 waves, SQ_BUSY/SQ_CYCLES 0.981"
        from: rounds/010/1-profiling/6-bound-analysis/analysis.md
      - claim: "The current profile has no FlyDSL instruction trace with which to transfer beat's boundary-stall price."
        value: "ATT captures no FlyDSL kernels; stall attribution unavailable"
        from: rounds/010/1-profiling/profiling.yaml
      - claim: "Static improvements cannot decide whether an arm deserves card time."
        value: "r5.i1.g13 compiled at 512 VGPR with zero spill and intended overlap but measured prod 0.8485 of champion"
        from: job_context/findings/facts.md
    risk: >
      Beat's 3.3% boundary cost is not a price for current, and the additional barrier and
      asynchronous-store wait may erase the proposed overlap. The current falsification rule is
      also too sensitive to one noisy session.
    alternative: null
    change: >
      State the premise as one WG per LDS-owning unit, not one WG/CU. Measure paired+nodelay
      against nodelay, not against the old champion. Require mean prod_vs_nodelay >=1.007 over at
      least three rotated sessions and reject any prod regression beyond the round's noise floor.
      Use static ISA only as a gate: zero scratch/spill, verified exact unpaired path, expected
      paired dispatch, and bijection. Do not cancel card measurement merely because the clean
      loop becomes longer.

  - candidate: r7.i3.g19
    stance: support
    grounds:
      - claim: "The production regime that originally motivated partner-wave overlap remains present."
        value: "prod bound latency, confidence low; current/beat 0.768"
        from: rounds/010/1-profiling/6-bound-analysis/analysis.md
      - claim: "Existing partner-wave arbitration is measurably useful."
        value: "lock_simd prod 0.963-0.967 with the hot-loop instruction stream unchanged"
        from: job_context/findings/dead_ends.md
      - claim: "The candidate has a distinct surviving mechanism despite the failed h24 arms."
        value: "phase-offset LO/HI waves on a depth-3 ring; predicted +2% to +5%, with ring cost requiring more than approximately 3% gross"
        from: job_context/findings/pool.md
    risk: >
      Natural LO/HI overlap already exists, and the required depth-3 carrier may cost more than
      the forced phase offset saves.
    alternative: null
    change: >
      Keep it in the pool but do not execute it this round. The failed h24 barrier implementations
      do not logically kill phase-offset ping-pong, but g19 needs a newly priced depth-3 carrier
      and a prediction against the current nodelay base before it becomes executable.

ordering:
  proposed:
    - "r10.i4.g29 discriminator with write-size, read, no-write, and post-write-delay controls"
    - r8.i1.g20
    - "r10.i4.g29 depth-3 small-grid variant, only if the discriminator supports write drain and the effect persists on nodelay"
    - r10.i3.g28
  why: >
    The discriminator is cheap and addresses the round-10 report's measured 56.16-versus-35.3 us
    fast gap. Nodelay is then the mandatory shipping base. The ring rewrite is justified only by
    a discriminator that separates write traffic from generic predecessor or enqueue effects.
    Pairing is last because its only claimed gain is an unmeasured workgroup-boundary overlap.
    r10.i1.g27 is removed because its required correctness failure follows directly from the
    allowed score range.

missing: []

unresolved:
  - r10.i4.g29
  - r8.i1.g20
  - r10.i1.g27
  - r10.i3.g28

control:
  turn: 1
  stances:
    support: 1
    modify: 3
    oppose: 1
  candidate_ids:
    - r10.i4.g29
    - r8.i1.g20
    - r10.i1.g27
    - r10.i3.g28
    - r7.i3.g19
  unresolved:
    - r10.i4.g29
    - r8.i1.g20
    - r10.i1.g27
    - r10.i3.g28
  missing: 0
```

## 03_response -- planner

五个 verdict 我全部接受，没有部分接受，也没有拒绝。reviewer 引用的每个值我都对照原文件核过，都对得上。它指出了我计划里三处实打实的问题，我不能拿老立场去顶：

- **g29 判别实验：** A 和 C 在 event 前放的是同一个 256 MiB zero，C 就不能单独判成 host gap。我改用它的写入量/读/无写/写后延迟这一组对照，再加上本轮 profiling 建议的 warm 对照。这些要进程里没有 beat 的条件下跑。
- **g20：** 我原来的 prediction 和 falsified_if 之间留了一个判不了的区间（1.003–1.005）。现在改成看 mean ≥ 1.007。
- **g28：** 我原来写的"clean loop 变长就不上卡"，是拿静态指标当价格用，违反了 correction 3。这条删掉，对照组也换成 nodelay。

g27 这条，reviewer 反对的理由正好是我在 synthesis s2 里自己算过的：参考值固定为 0 时，+1e4 会上溢，全负的行会下溢。op 的接口（`impl.py:81`）也不提供分数上界。结果已经确定，所以 fp32 仿真也不用跑了，直接移除，记进 dead_ends。

本轮没有未解决的分歧。完整 review 已保存到 `rounds/010/2-plan/step3_planner_review.yaml`。

```yaml
by: planner
turn: 1
verdicts:
  - candidate: r10.i4.g29
    stance: accept
    grounds:
      - {claim: "A 与 C 在 event 前都直接放同一个 256 MiB zero，C <= 42 us 不能唯一判为 host gap", value: "A 与 C 只差前导 sleep", from: rounds/010/2-plan/synthesis_notes.md s4}
      - {claim: "profiling 自己把 warm 与 flushed 对比列为最便宜的下一个测量；beat 的 warm 时间未测", value: "warm 35.3 us vs flushed 56.2 us (1.59x)；beat flushed 30.7 us", from: "rounds/010/1-profiling/6-bound-analysis/analysis.md:92-98"}
      - {claim: "没有 byte/stall 计数器，即使证实 drain 也定位不到 tensorcnt", value: "no byte/stall counters, ATT sees no FlyDSL", from: rounds/010/1-profiling/profiling.yaml coverage}
    change: >
      Step a 对照组，全部 current-only、fast、101 reps、进程内无 beat：
      (1) GPU-sleep  (2) 8 MiB zero  (3) 256 MiB read  (4) 256 MiB zero  (5) 256 MiB zero 后接 GPU-sleep  (6) warm。
      另起进程对 beat 跑 (1)(4)(6)。
      Write-drain 前提成立需要三条同时满足：(4) 远大于 (2)；(1)/(3) 没有惩罚或明显更小；(5) 相对 (4) 衰减。
      (1) 或 (3) 与 (4) 惩罚相同，则 depth-3 臂作废。
      先在 nodelay 上复测 (1) vs (4)，再建 depth-3 臂。
      Depth-3 臂的门槛：prod ISA 与 nodelay 逐字节相同；prod 同 session 测量；prod 低于 noise floor 即拒。
      Prediction：depth-3 在 fast 上至少收回一半惩罚，即相对 nodelay median 下降 >= 6 us。
  - candidate: r8.i1.g20
    stance: accept
    grounds:
      - {claim: "prediction 与 falsified_if 之间有 1.003-1.005 的无法判读区间", value: "1.005-1.012 every session / below 1.003", from: "rounds/010/2-plan/step0_planner.yaml:94-96"}
      - {claim: "round 9 合并臂把 nodelay 拖下水", value: "val_merged prod ratio 0.7546；s1 nodelay 1.9933 ms < inc 2.01534 ms", from: "rounds/009/1-opt/raw/meas/val_merged.out:65, s1_prod.out:31-32"}
    change: "至少 3 个轮换 session、进程内无 beat：mean prod_vs_champion >= 1.007，且每个 session > 1.003。单独 re-land 作为基底；其他臂先单独赢过 nodelay 才能合并。"
  - candidate: r10.i1.g27
    stance: accept
    grounds:
      - {claim: "固定参考必须是整行分数的真上界", value: "argument is <= 0 only under a true upper bound", from: "knowledge/ops/attention/online-softmax.md:92,117-118"}
      - {claim: "h16 输入域必然让参考为 0 失败，仿真结果已知", value: "scores ~+-1e4 after scale", from: "job_context/findings/route.md:102-103"}
      - {claim: "op 契约没有分数上界", value: "flydsl_attn_fwd(q, k, v, softmax_scale=None, causal=True)", from: "job_context/op/current/impl.py:81"}
    change: "移除，不建也不仿真。以正确性域为由记入 dead_ends；只有契约提供有限上界时才重开。"
  - candidate: r10.i3.g28
    stance: accept
    grounds:
      - {claim: "只能说明每个 LDS-owning unit 驻留 1 个 WG，CU 与 WGP 的对应未解决", value: "1 WG per LDS-owning unit; mapping not resolved", from: "rounds/010/1-profiling/6-bound-analysis/analysis.md:57-59"}
      - {claim: "beat 的 3.3% 不能转移到 current", value: "ATT captures no FlyDSL kernels", from: rounds/010/1-profiling/profiling.yaml coverage}
      - {claim: "loop 长度 gate 是拿静态指标当价格", value: "clean loop longer -> do not spend card time", from: "rounds/010/2-plan/step0_planner.yaml:120"}
    change: >
      前提改为每个 LDS-owning unit 1 个 WG；对照组换成 nodelay。
      Prediction：>= 3 个 session，mean prod_vs_nodelay >= 1.007，回退超过 noise floor 即拒。
      静态 gate 只剩：0 spill；unpaired path 精确；dispatch 符合预期；bijection。删除 loop 长度 gate，期望值标为 unpriced。
  - candidate: r7.i3.g19
    stance: accept
    grounds:
      - {claim: "机制没有被 h24 杀掉，但需要 depth-3 carrier，而 ring 成本已测", value: "must win > ~3% gross to break even (u2pp2 +2.2%)", from: "job_context/findings/pool.md:114"}
    change: "留在 pool，本轮不执行。先由 g29 step b 在 nodelay 上给出 ring 的实测价格，再对 nodelay 写 prediction。"
adopted_from_missing: []
unresolved: []
ordering:
  proposed: ["r10.i4.g29 step a (对照组 + warm；beat 另起进程)", r8.i1.g20, "r10.i4.g29 step b (条件执行)", r10.i3.g28]
  why: "step a 无需 build，而且决定 fast/proxy 读数能否用来比较任何臂；nodelay 是唯一 must，也是对照基底；ring 臂要等对照组把 write traffic 与泛化前驱效应分开；pairing 收益未定价；g27 因正确性域移除。"
control:
  turn: 1
  stances: {accept: 5, reject: 0, partial: 0}
  candidate_ids: [r10.i4.g29, r8.i1.g20, r10.i1.g27, r10.i3.g28, r7.i3.g19]
  unresolved: []
  missing: 0
```

## 04_close -- reviewer

```yaml
confirmed:
  - r10.i4.g29
  - r8.i1.g20
  - r10.i1.g27
  - r10.i3.g28
  - r7.i3.g19

conceded: []

unresolved: []

control:
  turn: 3
  candidate_ids:
    - r10.i4.g29
    - r8.i1.g20
    - r10.i1.g27
    - r10.i3.g28
    - r7.i3.g19
  confirmed:
    - r10.i4.g29
    - r8.i1.g20
    - r10.i1.g27
    - r10.i3.g28
    - r7.i3.g19
  conceded: []
  unresolved: []
  missing: 0
```

## 05_finalize -- planner

正在准备本轮必须当场完成的 g29 判别测量（在 fa-repro 容器里跑），然后写 plan。
启动判别测量（current / beat / nodelay 各自独立进程，2 个 session），同时挂 dmesg 监控。
测量在跑，同时读 pool.md 准备更新。
测量已闭合，正在写 instrument 摘要、plan.yaml / plan.md / candidates.yaml，并更新 pool、route 和 knowledge_gaps。
Now write plan.yaml, candidates.yaml, plan.md.
Knowledge gaps: fix the earlier wrong id and record the answer.
本轮选中 `r8.i1.g20`（h29，单独 re-land nodelay）。另外，我在 plan 阶段当场跑了 g29 的判别测量，结论是它的前提不成立。两个 agent 的交换以 converged_at_close 结束，没有遗留的分歧。

**实测结果**（fast、每组 101 次，在 fa-repro 容器里跑；dmesg 全程监控，没有故障，GPU 已空闲）：
- **flush 不带来额外耗时。** 每个进程只加载一个实现时，计分用的 256 MiB zero 之后，current 为 38.50–38.74 us。这与 GPU sleep、256 MiB 读和热态背靠背调用相差在 ±0.4 us 以内。nodelay 和 beat 也是同样的情况。
  - 只有两组偏高：前面什么都不跑（48.4 us）和前面只清 8 MiB（44.7 us）。原因是前驱太短，约 15.5 us 的 host launch 落进了计时窗口，beat 在这两组也同样偏高。
- **round 9 的 fast median 超额来自测试进程里交替运行的 kernel 镜像数量。**

  | 同一进程里的镜像 | fast median（us） |
  | --- | --- |
  | 1 个 FlyDSL | 38.8 |
  | 2 个 FlyDSL | 38.9–41.7 |
  | 4 个 FlyDSL | 50.3–64.5 |
  | current + beat | 66.0 |

  4 个镜像这一组复现了 round 9 的 51–56 us，而且各组的 min 都在 38–41 us 左右。机制推测是镜像之间互相逐出 I$，和 r9.i3.g26 是同一个现象；本轮没有重新计 I$ miss 计数器。
- **对 act 的约束：** 一个测量进程里只放候选和 champion 两个 FlyDSL arm。

**本轮新增 3 个候选：**
- `r10.i3.g28`（pairing）：进 pool，是 nodelay 之后的下一个 idea。现在以 nodelay 为对照，收益还没有定价。
- `r10.i1.g27`（reviewer 提的固定零参考）：op 没有提供分数上界，h16 要求的 ±1e4 输入会让它失败。已标记 RETIRE，一次都没有 build。
- `r10.i4.g29`：上面的测量已证伪。已标记 RETIRE。

我无权写 dead_ends.md，所以 g27、g29 暂时标记在 pool 里，由 reflect 移入 dead_ends。

route 表已整体重写：h29 是第 1 行、唯一的 must；g28、g19、g26 随后；9 个 advise 项都在表里。测量摘要在 `rounds/010/2-plan/raw/postflush/summary.md`。

```yaml
selected: r8.i1.g20
prediction: {metric: prod_vs_champion, from: 1.000, to: ">=1.007 mean over >=3 rotated no-beat sessions, every session >1.003; fast/proxy 0.99-1.01; loop 513->499, VGPR 456, 0 spill, hash o=26a89a2db0cd"}
added_to_pool: [r10.i3.g28, r10.i1.g27, r10.i4.g29]
retired_or_marked_stale: [r10.i1.g27, r10.i4.g29]
unresolved_disagreements: []
route_updated: true
files: [plan.yaml, plan.md, candidates.yaml]
```
