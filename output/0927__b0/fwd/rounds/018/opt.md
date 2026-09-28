# Round 18 opt (fast round)

## Starting state (read)
- op/current = r13ns (unchanged since r16; rounds/018/op byte-identical to it, checked with diff -rq).
- r17 not accepted: g63 (ones-WMMA row sum) prod x1.0093 vs current, proxy/fast flat; a prod-only body edit needs ~+2.1%
  on prod to clear the 0.70% bar. r17 pmc throwaways price the chain: row sum ~3.4%, max tree ~1.9% of prod wall at a
  flat clock. g63 got ~1/4 of the sum ceiling because d += acc[0] reads a WMMA result every tile.
- Route (r17): next idea row is r17.i2.g64 (the same ones-WMMA sum, carried as a d-tile in PV, read only in the epilogue).
- r15 profile read (profiling_summary / analysis via facts and r17 opt): prod power/clock-bound, proxy tail. No stall or
  VALU counters on this chip; GRBM cycles + ticks/ns are the instrument (r17 pmc protocol, reused here).
- Card: fa-g2, GPU use 0%, sclk 2359 MHz idle; two node-wide KFD PIDs listed (other containers, not our device).

## Survey
rocprofv3 --stats / kernel-trace is closed for FlyDSL on this box (r9.i3.g22: hangs / zero kernels), so the survey is
benchmark.py (in the ranking sessions below, cur and champ2 arms) plus pmc GRBM cycles (r17 driver). The kernel list is
known and one-dispatch: m32x8 at proxy/prod, m32x2 at fast.

## Corpus read
- backends/flydsl/attention/README.md, techniques.md (softmax issue, conversion coefficient, exp/MFMA overlap).
- arch/gfx1250/isa.md §data hazards: a VALU reading a WMMA D needs 4 co-exec slots after a bf16 WMMA; WMMA->WMMA
  RAW on A/B needs 5. This is g63's residue (d += acc[0] every tile) and why g64 keeps the result in the accumulator.
- backends/hipkittens/attention/recipes/gqa_d128.md §6 (ranked mechanisms; lazy rescale already in).
- optimization/routes/1-metrics-to-techniques.md (Table A/B; A1 scratch gate used below).
- route.md h18 decision index (L7 named barrier is a no-op; L17/L18 are this family).

## Arms (rounds/018/1-opt/raw/arms, patch scripts raw/patch_*.py), each built alone from op/current
- **A = r17.i2.g64** (pool, keeps its id): d as a v8 loop-carried tile, `acc_d = WMMA(ones16x32, P^T frag, acc_d)` per
  kt in `_pv_gemm` after the O tiles; rescaled with O inside the existing `_maybe_rescale` branch; `_softmax` skips the
  sum tree and the d update (row_sum=False); epilogue reads acc_d[0]. Not bitwise (sums bf16 P, like g63).
- **B = r18.i1.g65** (new): the deferred-rescale ballot is taken on each lane's HALF-row max (no permlanex16). Both
  halves of a row are in one wave32, so the ballot is bit-identical to today's. The max peer, m_full, corr and neg_m are
  formed only inside one wave-uniform branch taken when any row of the wave rescales; the common path uses m_prev,
  corr = exp2(0) = 1, neg_m = -(m_prev*log2e), as today. Bitwise equal by construction. Touches only the max half of
  `_softmax`, so A and B are independent lines.
- cur = op/current copy, champ2 = byte copy (A/A).

Predictions (written before any run):
- A: prod x1.012..1.025 vs cur (between g63's +0.9% and nosum's +3.4%), proxy x1.005..1.02. VGPR +14..16 (<= 472), 0
  scratch. Could lose if the extra carried tile makes the rescale branch or RA worse (s_set_vgpr_msb / v_nop growth).
- B: prod x1.002..1.008 (the max peer is on the S->P path, like the sum peer g59 removed for +0.5..0.7%). Risk: the new
  basic block splits the softmax and blocks the scheduler from interleaving across it -> could read <= 1.000.
- Merge (only if neither loses): roughly additive at a flat clock, x1.015..1.03 prod.
- What would change my mind: A inside the A/A => the per-tile WMMA read was not g63's residue, and the sum ceiling is
  not reachable on the matrix pipe; the family should stop.

## Build log
- g64: patch_g64.py applied first time, COMPILE_OK.
- g65: **build failure 1** -- `TypeError: list indices must be integers or slices, not ArithValue`. The FlyDSL AST
  rewriter turns a Python `for r in range(R)` written *inside* a `@flyc.jit` branch body into an scf.for, so `r`
  became an SSA value. Fix: move the per-row work into a plain helper (`_full_rows`) defined outside the jit function
  and call it from the branch. Second build COMPILE_OK. (Worth knowing for any branch-local unrolled code in this DSL.)

## Static census (raw/isa/census.txt, prod cfg, compile-only)
| arm | VGPR | scratch | instrs | v_wmma | v_pk_add_f32 | permlanex16 | v_max* | v_nop | s_set_vgpr_msb | s_cbranch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cur | 456 | 0 | 4130 | 256 | 126 | 16 | 123 | 77 | 244 | 23 |
| g64 | 469 | 0 | 4055 | 272 | 0 | 8 | 138 | 73 | 253 | 23 |
| g65 | 460 | 0 | 4149 | 256 | 126 | 16 | 134 | 76 | 238 | 27 |
g64 lands 141 instructions above r17's nosum throwaway (3914) and 187 below g63 (4242). VGPR +13, under 512, 0 spill.
g65's permlanex16 count is unchanged statically (the peer moved into the branch, it did not disappear); +4 branches.

## Correctness (raw/val/)
- validation.py (fresh cache each): g64 and g65 both 16/16 PASS, min o 49.82 dB (short_q full, same as every round),
  det 200/200 at fast. rc 2 only from the speed-vs-beat bar that op/current also fails. In-process-with-beat ratios
  (not a ranking, h31): g64 fast 1.039 / proxy 0.938 / prod 0.949; g65 1.050 / 0.897 / 0.942.
- m32x8 check at proxy/prod (raw/val/chk.py, vs cur, same inputs): **g65 bitwise equal to cur at proxy and prod**
  (o and lse), as constructed; det 21/21 proxy, 11/11 prod. g64: o 60.1 / 60.4 dB vs cur, lse 98.5 / 101.7 dB,
  finite, det 21/21 proxy, 11/11 prod (one hash each).

## Ranking (raw/bench/, h31/h35 protocol: no beat in process, champ2 = byte-identical A/A, 3 rotated sessions, PYTHONHASHSEED=0, fresh cache; GPU 0-5% use before/after, rc 0)
speed vs cur per session, geo mean:
| shape | champ2 (A/A) | g64 | g65 |
| --- | --- | --- | --- |
| fast | 0.997 / 1.000 / 0.997 | 1.000 / 0.994 / 0.995 (0.9964) | 1.0021 geo |
| proxy | 1.0046 / 0.9991 / 1.0041 (1.0026) | 1.0171 / 0.9982 / 1.0088 (**1.0080**) | 0.9982 / 0.9790 / 0.9843 (**0.9871**) |
| prod | 0.9993 / 0.9978 / 0.9950 (0.9973) | 1.0217 / 1.0185 / 1.0146 (**1.0183**, 3/3 above A/A) | 1.0110 / 1.0043 / 1.0072 (**1.0075**, 3/3 above A/A) |
Fast runs m32x2 in every arm (body untouched) -> fast deltas are noise by construction.

Expected vs measured:
- g64: predicted prod x1.012..1.025 -> measured x1.018. Inside the band. Proxy also up (+0.8%, 2/3 sessions above A/A).
  The ones-WMMA sum reaches ~55% of the r17 no-sum ceiling (+3.4%), where g63 (sum every tile, acc[0] VALU read)
  reached +0.9%. Confirms g63's loss was the per-tile VALU read of the WMMA result, not the matrix-pipe cost.
- g65: predicted prod x1.002..1.008 -> measured x1.0075 (top of band). Proxy **x0.987, min 0.979, below the A/A
  floor (0.999) in 3/3 sessions** -> a loss at proxy. Likely cause (unverified): at proxy (4096 kv, fewer tiles
  per wave) a larger share of tiles are early tiles where the max still moves, so the any_do branch is taken often
  and the extra ballot + branch is pure overhead; at prod the steady state dominates. Not probed this round.

## Merge decision
Rule: skip the merge when either arm lost (below the incumbent by more than the floor spread). g65 lost proxy by
-1.3% (all three sessions below the A/A minimum) -> **merge skipped**, not built/measured. (arms/merge was patched
but never compiled; removed.) Shipping g64 alone.
Score view (fast capped): g64 is prod +1.8% and proxy +0.8%; mean over three shapes (fast capped, 0) ~ (0.8+1.8)/3 = +0.87%
-> above the 0.70% bar on the in-round ranking; the acceptance run decides.

## Cycles vs clock (raw/pmc/, r17 pmc protocol: rocprofv3 GRBM_GUI_ACTIVE/XCD + ticks/ns, one process per shape x order, no beat, fresh cache; GPU 0% before/after, rc 0 x4, 10:10:59-10:12:03Z)
| shape/order | g64 cycles | g64 ticks/ns | g65 cycles | g65 ticks/ns | champ2 cycles (A/A) |
| --- | --- | --- | --- | --- | --- |
| prod A | x0.9865 | x0.9991 | x1.0245 | x1.0265 | x0.9983 |
| prod B | x0.9946 | x0.9990 | x1.0278 | x1.0206 | x1.0037 |
| proxy A | x0.9762 | x0.9954 | x1.0100 | x1.0021 | x0.9867 |
| proxy B | x0.9916 | x1.0039 | x1.0156 | x1.0111 | x1.0044 |
Reading:
- **g64 is a cycles win at a flat clock** (prod -0.5..-1.4% cycles, ticks/ns x0.999). A real work cut, not a power
  artifact. It carries into the low clock state (h42 prefers this).
- **g65 is an energy trade, not a work cut.** It ADDS cycles (+2.5..2.8% prod, +1.0..1.6% proxy). Its prod wall gain
  comes entirely from the clock (+2.1..2.7% ticks/ns: skipping the peer permlanex16 + max ops on most tiles lowers
  power). At proxy (not power-bound) the clock barely moves (+0.2..1.1%), so the added cycles show up as the proxy loss.
  So the unverified "early tiles take the branch" story in the ranking section is at most part of it. The branch/
  basic-block split costs cycles at both shapes (s_cbranch 23 -> 27, static instr +19).
  Under h42 (training clock-sensitive, low clock states) g65's gain would shrink and its cycle cost would stay.
  Under h45 (real data, score std 21-53) the branch is taken more often. Both argue against g65 as built.

## Verdicts
- **A r17.i2.g64: WON, shipped.** rounds/018/op/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py = raw/arms/g64 (diff -rq:
  sources identical, pycache only). prod x1.0183 (3/3 above A/A max 0.9993), proxy x1.0080, fast untouched; cycles
  -0.5..-1.4% at flat clock; 16/16 PASS min 49.82 dB, det 200/200; 469 VGPR, 0 scratch. Estimated score delta ~+0.87%,
  above the 0.70% bar in-round, but only by a margin about the size of the A/A spread. The acceptance run decides.
  It subsumes r16.i3.g62 (the per-tile d update it guarded is gone).
- **B r18.i1.g65: LOST (proxy x0.987, 3/3 below the A/A floor); not shipped; kept in pool as an energy lever.** It is
  bitwise equal, and prod x1.0075 comes from the clock alone.
- **Merge: not built** (rule: B lost).

## explored.consulted
- job_context/findings/facts.md, dead_ends.md, pool.md, route.md (incl. h18, h31, h35, h42, h45, h46)
- rounds/017/1-opt/opt.md, rounds/017/1-opt/raw/g63.diff (protocol + scripts), rounds/015 profile via facts/r17 opt
- knowledge/backends/flydsl/attention/README.md
- knowledge/backends/flydsl/attention/techniques.md
- knowledge/arch/gfx1250/isa.md (data hazards)
- knowledge/backends/hipkittens/attention/recipes/gqa_d128.md (§6)
- knowledge/optimization/routes/1-metrics-to-techniques.md
- rounds/015/3-act/tools/compile_isa.py (ISA census tool)

---
# Act step (build + measure what the route says)

Route rows taken from the top:
- Row 1 h47 (must, open): applied as a measurement obligation on this round's arm. The arm is ranked on real q/k/v dumps
  right after a GEMM burst (ab.py condition iii, copied to raw/real/ab18.py with the arm paths swapped and the RATIO
  line generalized, no beat in process), and also on randn benchmark.py. Instruction count of the prod m32x8 hot loop:
  4130 -> 4055 (raw/isa/census.txt).
- Rows 2-4 (h37/h34/h28): discharged, nothing to build; protocol from h28 applied.
- Row 5 r17.i2.g64: executed (below).
- Row 6 r18.i1.g65: not executed. Its condition requires row 5 first plus a real-dump vote-rate probe and a new
  branch-free form (a new build = next round).
- Rows 7-8: blocked by their own conditions.
Read before acting, as the act prompt asks: rounds/016/0-refactor/refactor.md. g64 is not on its ranked list. Item 3
(cheaper row-max) is g65's family. Item 5 (report a real-data ruler) is what h47 row 1 applies.

## Build (rounds/018/op = g64; raw/build/)
- The compile cache was cleared first: /tmp/flycache removed in fa-g2 and __pycache__ under rounds/018/op deleted.
  Each process then got a fresh FLYDSL_RUNTIME_CACHE_DIR.
- Attempt 1, no failures. ut/test_correctness.py --impl rounds/018/op: RESULT PASS. validation.py rounds/018/op: every
  correctness row PASS, min o 49.82 dB (short_q full), determinism 200/200 (o=9eb9b58556bf, the same digest as
  current at fast: m32x2 untouched). rc 2 only from the speed bar (in-process with beat: fast 1.0462, proxy 0.9035,
  prod 0.9451). GPU use 0% before, 1% after.

## Measure (raw/measure/, benchmark.py unprofiled, all shapes)
Arms: cand = rounds/018/op (g64), r16 = rounds/016/op (the incumbent round), r17 = rounds/017/op (g63), cur =
job_context/op/current, champ2 = a byte copy of current (A/A). 3 rotated sessions, one process each, fresh JIT cache,
no beat in the ranking process (h31). fa-g2 idle (GPU use 0-5% before/after), 10:17:41-~10:25Z, rc 0 x3.
The beat ran in its own process (with cur beside it) right after, in the same sweep, rc 0.
No round before 16 was measured.

TFLOP/s (mean of the 3 session medians):
| shape | cand | r16 | r17 | cur | champ2 | beat (own process) |
| --- | --- | --- | --- | --- | --- | --- |
| fast | 164.10 | 164.62 | 164.52 | 164.94 | 164.85 | 155.31 |
| proxy | **1575.01** | 1564.96 | 1571.74 | 1567.47 | 1564.24 | 1602.45 |
| prod | **1467.80** | 1445.73 | 1454.55 | 1442.11 | 1442.58 | 1548.83 |
Speed vs r16 per session: prod cand 1.0139/1.0198/1.0121 (geo 1.0153; A/A champ2 0.9974/0.9987/0.9973);
proxy cand 1.0014/1.0091/1.0087 (geo 1.0064; champ2 0.9986/1.0064/0.9937); fast cand 0.9970/1.0031/0.9908 (geo
0.9969; champ2 1.0138/1.0000/0.9908, noise; the m32x2 body is untouched).
champ2 vs cur (A/A): prod 0.9978..1.0028, proxy 0.9884..1.0050, fast 0.9939..1.0046.

Acceptance arithmetic (champions re-measured in this session: fast r16 164.62, proxy r17 1571.74, prod r17 1454.55):
- Throughput vs best ever: prod x1.0091, proxy x1.0021. Both improve.
- Below-target shapes (proxy, prod) are >= 95% of their best: 1.0021 and 1.0091. Yes.
- Fast had reached its target (beat 155.31) and stays above it: 164.10 = 1.057x beat. Yes.
- Ratios vs target (target = beat x 1.0): fast 1.0 (capped; 1.0566 uncapped), proxy 0.9829, prod 0.9477.
  Score 0.9769 capped / 0.9957 uncapped.
  Caveat: the proxy beat moves between processes. It read 1602.45 in the beat sweep and 1676.84 inside validation's
  process (ratio 0.9035 there), so proxy's ratio depends on which beat is used.

## h47 real-data ruler (raw/real/, route row 1)
prod shape, 6 real q/k/v dump layers + randn. Each call is timed right after a 10-GEMM burst (ab.py condition iii,
bf16 F.linear at 25 ms per burst), 5 calls x 4 palindromic rounds per arm per set, 3 rotated orders, one process each,
fresh cache, no beat. rc 0 x3; GPU use 1-5% before/after.
| arm vs cur | real dumps (18 readings) geo, min..max | randn (same condition), geo |
| --- | --- | --- |
| cand (g64) | **1.0019**, 0.9955..1.0091 | **1.0136** |
| r16 | 1.0001, 0.9935..1.0072 | 1.0012 |
| champ2 (A/A) | 0.9985, 0.9937..1.0014 | 0.9981 |
Per layer, cand geo: L00 1.0044, L01 0.9985, L02 1.0055, L08 0.9967, L16 1.0044, L31 1.0021.
**Reading: g64's gain is a randn-ruler gain.** Under the post-burst clock it holds on randn (+1.4%), but on real data it
is inside the A/A band (+0.2%, with L08 and L01 slightly below). A likely mechanism (not probed): on real data the
max moves on 13-25% of tiles (h43), so the deferred-rescale branch runs far more often, and g64 adds a v8 d-tile
rescale per q-tile to every such branch. The common-path saving (no sum tree, no d update) is then partly paid back
in the rescale path. It is not a loss on real data, and it is not the >= 2% real-data win h47 item 3 asks for.

## Outcome
delivered: g64 left in rounds/018/op. It passes the acceptance arithmetic on the job's ruler (prod x1.0091 / proxy
x1.0021 vs the best on this structure, fast above target). By h47's second ruler it is neutral. Both rulers are
reported in act.yaml.
