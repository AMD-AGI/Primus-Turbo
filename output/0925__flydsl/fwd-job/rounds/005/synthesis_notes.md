# Round 5 synthesis notes (planner, step 1)

## S1 -- compile-only VGPR probe of the h21 pipeline ported onto the round-4 champion (done in plan, no card)

Why: planner step 0 predicted ~519 VGPR (456 + h21's +63), reviewer step 0 carried h21's 508/512 from the
round-2 base. Neither number was of the champion. A compile settles it without the card.

- Diff: `diff -u proto/pipeline/base/... proto/pipeline/op/...` (raw/vgpr_probe/pipe1.diff) applied to a COPY of
  job_context/op/current: 8/8 hunks, offsets only (+19 lines), no conflict with round 4's packed softmax.
  (The softmax proto diff, by contrast, fails hunk 4 at :603 -- the packed region -- so g14 is a hand port.)
- Compile: proto/pipeline/review/rc.py, COMPILE_ONLY=1, no visible GPU, prod shape, gqa4 causal d128 bf16.
- Result (raw/vgpr_probe/summary.txt):

| build | VGPR | SGPR | spill v/s | scratch | WMMA | v_exp | v_exp <=8 instr after WMMA | v_mov_b64 | s_set_vgpr_msb | v_nop+s_nop | md5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| champion (ctl) | 456 | 100 | 0/0 | 0 | 256 | 264 | 0 | 240 | 240 | 62 | 7b078261 |
| pipeline port | **512** | 97 | 0/0 | 0 | 352 | 330 | 103 | 496 | 757 | 217 | d530dda8 |

Readings (gates, not prices -- correction 3):
1. It fits: 0 spill, 0 scratch. But headroom is **zero**. Nothing can be stacked on it this round (g14's LANE
   proto was +1 VGPR over its base) without a fresh compile, and a merge g13+g14 is presumed not to fit.
2. The overlap is real in the ISA: 103 v_exp now sit within 8 instructions of a WMMA vs 0 in the champion.
3. The cost is the direction round 4 found expensive: nops 62 -> 217 (x3.5), msb 240 -> 757, v_mov_b64 x2.
   Round 4's super-additive win came with v_nop 168 -> 62 (facts.md). This is the strongest reason to hold the
   pipeline's prediction at >=+3% rather than the reviewer's >=+5%.
4. WMMA 256 -> 352 and v_exp 264 -> 330 are static counts (peeled last tile + loop copies), not issued work.

## S2 -- id collision

Both agents allocated g13/g14 independently, crossed:
reviewer r5.i1.g13 = LANE row-sum == planner r5.i2.g14; reviewer r5.i2.g14 = pipeline == planner r5.i1.g13.
Same hypotheses, so merged, not duplicated. The merged set keeps the planner ids; the reviewer's ids are
recorded as aliases and are not reused for anything else.
