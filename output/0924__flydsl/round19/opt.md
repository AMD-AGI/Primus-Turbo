# Round 19 -- fast round -- gfx1250 / FlyDSL attention backward

Written as the round ran. Card-free work first, then two arms.

## 0. State inherited

Incumbent 504.03-504.84 TF/s prod, bar 719.49-720.98 TF/s, ratio ~0.700. Bound recorded as
"latency, LOW confidence" -- and round 18 falsified its own model by measurement. Pool held
two open ids (`r16.i3.g50`, `r18.i5.g58`), highest global id g58.

## 1. Zero-card findings

### 1.1 The static model is calibrated for the first time in this job

prod total = 5.49823e12 FLOP / 504.03 TF/s = **10.909 ms**; `k_dkdv` is 62.93% of that
(round 17 `kernel.yaml`) = **6.865 ms**. Full-loop `.LBB0_8` iterations at prod =
`sum_bid 4*(255-bid) * 8 hkv * 4 batch` = **4,177,920**, over 256 CU x 4 SIMD = 1024 SIMDs
= 4080 iterations per SIMD. At the VR-throttled ~1.075 GHz that is

> **~1809 measured cycles per `.LBB0_8` iteration**, against the static model's 1685 (93%).

Issue-only floor = 611 non-WMMA issue + 64 WMMA x 8 = **1123 cycles**. So ~686 cycles (38%)
of every iteration is genuinely not issuing. The model is in the right neighbourhood on
TOTAL but, as 1.2 shows, wrong about WHERE.

### 1.2 The ASYNCcnt instrument was built -- and it kills its own family

Route row 5 / h18 owed an instrument that prices `global_load_async_to_lds_*` correctly:
round 18's 4-queue model charged those loads to LOADcnt, which is wrong on gfx1250 (six
independent counters; async has its own). `scr/attrib_ss5.py` adds a fifth queue
(`loadcnt` L=800, `dscnt` L=100, `asynccnt` L in {400,800,1600}, `kmcnt` L=300) and tests
ASYNCQ before LOADQ so an async load no longer pollutes LOADcnt.

Result on the shipped g56 ISA (`raw/attrib_async_cur_vs_g56.txt`):

| build | `.LBB0_8` modelled cycles | stall | measured prod |
|---|---|---|---|
| incumbent | 1685 | 562 (470 at one LOADcnt site) | -- |
| `r18.i3.g56` | **1343** | 228, **all DScnt, zero LOADcnt** | **-6.84%** |

The corrected model predicts g56 **~20% faster**. It measured 6.84% slower. Round 18's
"470-cycle LOADcnt site" is a **fiction of the 4-queue model**, and the 5-queue model does
not fix the direction of the error -- it makes it worse.

This is the **fourth** static mis-pricing in this job and the first where the instrument was
upgraded specifically to repair the previous failure. The conclusion is about the tool, not
the candidate:

> **The static wait model may not price any candidate on this kernel. Retired as a
> predictor; it remains a build gate (scratch, VGPR, spill) only.**

Direct corollary: **`r18.i5.g58` must not be built.** It is g56's mechanism 32x larger and
its ONLY pricing tool is the one just proven wrong in exactly this direction. Closed.

(At L_ASYNC=1600 the masked body `.LBB0_4` goes 2124 -> 3083 cycles, but it runs 1/256 of
prod iterations, so it cannot account for the 6.84% either.)

### 1.3 Instruction-class census: the LDS port, not the load queue

`scr/census.py` over the loop bodies (`raw/census_cur_dkdv.txt`):

| | `k_dkdv .LBB0_8` | `k_dq` hot body |
|---|---|---|
| WMMA | 64 | 96 |
| `ds_store_b128` | 40 | 0 |
| `ds_load_tr16_b128` | 40 | 32 |
| **LDS ops per WMMA** | **1.25** | **0.33** |
| `s_set_vgpr_msb` | 125 | -- |
| `v_nop` | 67 | -- |
| body slots | 675 | -- |

3.8x gap in LDS pressure, against the known 1.9x gap in non-WMMA issue per S-element.
LDS bytes per wave-iteration = 80 x 512 B = **40,960 B**, of which Q/dO is **32,768 B (80%)**.
At 4 waves/CU and 1809 cycles that is **~91 B/cycle/CU** of LDS traffic.

This one hypothesis retrodicts the whole ledger: `r4.i1.g16` (+10.7% from a pad constant --
bank conflicts matter a lot), `r12.i1.g39` (null -- segment split moves no bytes),
`r11.i1.g33` / `r15.i2.g47` (losses -- rescheduling a bandwidth limit does nothing),
`r18.i3.g56` (loss -- it ADDED `ds_load_2addr_b32` traffic to the tightest port), and the
`k_dq` / `k_dkdv` efficiency gap itself.

Also recorded, new to the corpus as well as to this job: `s_set_vgpr_msb` 125 + `v_nop` 67 =
**192 of 675 slots (28.4%) is pure addressing/hazard overhead**.

### 1.4 h8 is dead, again, and this time arithmetically

h8 claims ~107 GB of requested traffic. 107 GB / 4.39 TB/s = 24.4 ms against a 10.9 ms
measured total. HBM cannot be supplying it, so the traffic is being served by L1/L2 and the
question was never an HBM roof. Ninth piece of counter-evidence.

### 1.5 A stale comment, not a bug

`kernels.py:174` labels the x/y grid swap "h8 PROBE (throwaway, never shipped)". Diffing
`rounds/005/op` against `rounds/006/op` shows the swap IS `r1.i3.g03`, shipped at +2.60%.
Documentation only; recorded so the next round does not re-derive it.

### 1.6 `r18.i4`'s paper-kill rests on a falsified premise

q-step 32->64 was killed in round 18 on "the 470 comes from a FIFO-ordering cost that scales
with the batch". 1.2 disproves the 470. The kill still stands on independent evidence
(`r10.i2.g28` at -7.8%, and only ~50 of 675 body slots are fixed cost so widening the tile
buys almost nothing), so it is not re-opened -- but the stated reason is now wrong.

## 2. Corpus consulted

See `explored.consulted` in the reply. Two entries changed this round's plan:

- **`arch/gfx1250/isa.md`**: WMMA carries `OPSEL[2]` / `OPSEL_HI[2]` as **A-reuse / B-reuse
  hints**, free, and `rocdl.wmma_f32_16x16x32_bf16` already exposes them -- this job has
  passed `reuseA=False, reuseB=False` since round 1 without ever asking why. Legal only when
  the preceding matrix instruction used an identical operand; set otherwise the result is
  **undefined, not slow**. -> arm A.
- **`optimization/routes/1-metrics-to-techniques.md` + principle section 8**: for an
  issue-slot-bound kernel the levers are fewer VALU instructions or better placement of the
  rest, and "matrix instructions occupy neither wait counter". Confirms the class; the
  corpus has **no gfx1250 L1/L2 bytes-per-clock roof and no counter to derive one**, which
  is what leaves `r16.i3.g50` unpriceable.
- `backends/flydsl/attention/techniques.md` retracts its own "DMA to LDS forces vmcnt(0)"
  entry -- noted, but g58 is closed on measurement (1.2), not on that premise.
- `backends/hipkittens/attention/recipes/gqa_d128.md` section 6.1: the structurally
  identical gfx950 problem (per-matrix-op register-window shuffles at 12.8x count) cost
  **1.72x**. That is the closest analogue to this kernel's 125 `s_set_vgpr_msb`. -> arm B.

## 3. Arms

Both built separately from `op/current`, measured apart.

### 3.1 `r19.i1.g59` -- WMMA A-operand reuse hint (`OPSEL[2]`)

**Mechanism.** gfx1250 WMMA encodes A-reuse / B-reuse hints in `OPSEL[2]` / `OPSEL_HI[2]`;
`rocdl.wmma_f32_16x16x32_bf16` exposes them as `reuseA` / `reuseB` and this job has passed
`False` for both since round 1. The hint is only legal when the *preceding* matrix
instruction used an identical operand -- and the incumbent's emission order interleaves
`(a_p, b_do[d])` with `(a_ds, b_q[d])`, so A alternates every instruction and the hint could
never have been used even if someone had thought to set it.

**Change** (7 lines in `_body`): split the `dtile` loop into two runs of 8. Within a run A is
one 16x32 subtile held across all 8 instructions (corpus walk order: hold the operand that
is at least one subtile wide; tie -> prefer A), so instructions 2..8 of each run set
`reuseA=True`. B still changes every instruction. 28 of 64 WMMA carry the hint.
Nothing else moves; the arithmetic is untouched.

**Gate.** `op/validation.py`, because a reuse hint set on a non-identical operand is
**undefined, not slow** -- and the compiler is free to reschedule the run and break the
adjacency the hint assumes.

### 3.2 `r19.i2.g60` -- re-price `r7.i1.g21`'s prefetch against the register-window tax

**Mechanism.** The census (1.3) puts `s_set_vgpr_msb` at **125 of 675 body slots (18.5%)** --
the single largest identified overhead in the kernel, and the corpus's closest analogue (the
gfx950 `v_accvgpr_read/write` window tax, `gqa_d128.md` section 6.1) cost **1.72x**.
`r18.i2.g55`'s register census says 128 of the 384 structural live-in VGPR are exactly
`r7.i1.g21`'s prefetch tuple. g21 was worth +8.3% when it landed in round 7, on a kernel
that has since changed by g23, g26, g39, g40, g42, g45 and g51. If the window tax now
outweighs the prefetch, the whole `s_set_vgpr_msb` block becomes addressable.

**Change**: `qloop_full` calls `_body` with `carry=False` / `nxt=None` (the path
`qloop_mask` already uses), and both prologue `_ldqd` issues are deleted. Bit-identical
arithmetic.

**Gate.** `op/validation.py`. It is a *measurement*, not a proposal -- a loss here is the
answer, and a win would re-open the occupancy axis that `r18.i2.g55` closed.

## 4. Measurement

Idle-device witness before and after every session: `rocm-smi --showpids` = "No KFD PIDs
currently running", `--showuse` = 0%. `dmesg` clean (no `SMU: No response`). sclk 1051-1056
at prod/proxy, 1100 at fast, printed per line. One benchmark process per shape.
Ordering palindromic, 6 candidate slots + `beat`, median of 51 timed iterations.

Raw: `raw/rows_{prod,proxy,fast}.json`, `raw/isa_dkdv_{cur,armA,armB}.s`.
Validation transcript: `rounds/019/_scratch/run/val/out`.

### 4.1 Correctness (`op/validation.py`, run BEFORE any benchmark, per `h1`)

Both arms: `correctness pass` (dq/dk/dv **52.52-52.83 dB** at all three shapes, identical to
the incumbent's), `determinism pass` (bitwise identical across 200 runs at `fast`).
Both then `speed FAIL` -- geomean vs beat 0.811x (armA) and 0.761x (armB). Neither clears
the bar; nothing in this job ever has.

### 4.2 Speed, same session, bar re-measured

Two rebuilt slots per candidate, palindromic. TF/s, median of 51:

| arm | prod | proxy | fast (sentinel only, `h7`) |
|---|---|---|---|
| **incumbent** (`cur_a` / `cur_b`) | 506.42 / 503.37 -> **504.90** | 443.39 / 437.18 -> **440.29** | 55.59 / 52.57 -> 54.08 |
| **`r19.i1.g59`** (`armA` / `armA_b`) | 504.59 / 505.98 -> **505.29** | 437.80 / 441.86 -> **439.83** | 51.28 / 54.93 -> 53.11 |
| **`r19.i2.g60`** (`armB`) | **417.73** | **373.66** | 53.27 |
| **`beat`** (re-measured this session) | **719.50** | **584.97** | 52.06 |

Same-code floor this session: prod **0.61%**, proxy **1.42%**, fast **5.74%**.

- **`r19.i1.g59`: prod +0.08%, proxy -0.10%. Both inside the floor. NULL.**
- **`r19.i2.g60`: prod -17.27%, proxy -15.13%. Decisive loss, same sign both shapes.**
- prod ratio to bar: **505.29 / 719.50 = 0.7023** (round 18: 0.7002).

### 4.3 The ISA says both arms did exactly what they were asked to do

`k_dkdv` final ISA, whole kernel (both bodies), from `raw/isa_dkdv_*.s`:

| | incumbent | `g59` | `g60` |
|---|---|---|---|
| `v_wmma_f32_16x16x32_bf16` | 128 | 128 | 128 |
| ...carrying `matrix_a_reuse` | **0** | **56** | 0 |
| `s_set_vgpr_msb` | 334 | 329 | **277** |
| `v_nop` | 82 | 81 | **24** |
| `.vgpr_count` | 724 | 724 | **636** |
| spill / scratch | 0 / 0 | 0 / 0 | 0 / 0 |

This is what makes both results usable rather than inconclusive.

## 5. What the two arms actually established

### 5.1 `r19.i1.g59` -- the gfx1250 WMMA reuse hint is worth zero here, and it is emitted

The corpus records **no measurement of `OPSEL[2]`/`OPSEL_HI[2]` matrix-operand reuse on
gfx1250, from any backend.** This is the first.

The hint reached the ISA: **56 of 128 WMMA carry `matrix_a_reuse`** where the incumbent has
0, which means the backend both accepted the request and satisfied itself that the
adjacency the hint requires holds. It changed `s_set_vgpr_msb` by 5 (334 -> 329), `v_nop` by
1, VGPR by 0, and time by **+0.08% prod / -0.10% proxy**, against a 0.61% floor.

> **Finding: on gfx1250 the A-operand reuse hint does not reduce issue slots, hazard NOPs or
> register-window switches, and does not move time. It is not a lever on this kernel.**
> Cheap, correct, deterministic -- and null. Recorded so no future round spends a slot on it.

Why it cannot have paid, in hindsight: the hint saves an operand *fetch*, not an issue slot,
and this body's problem is 686 non-issuing cycles per iteration (1.1), not operand-fetch
bandwidth at the WMMA.

### 5.2 `r19.i2.g60` -- the register-window tax is NOT what costs time. Falsified on card.

This is the round's real result, and it is a negative that closes an axis.

The hypothesis (1.3 + the corpus's 1.72x gfx950 analogue): `s_set_vgpr_msb` at 125 of 675
body slots is the largest single identified overhead, and it exists because 384 of the 395
live-in VGPR are structural -- of which **128 are exactly `r7.i1.g21`'s prefetch tuple**.
Remove the prefetch, drop out of the wide window, win back the tax.

The arm did every part of that:

- `.vgpr_count` **724 -> 636** (-88, i.e. the 128-VGPR tuple minus what the in-body loads
  need back)
- `s_set_vgpr_msb` **334 -> 277** (-17%)
- `v_nop` **82 -> 24** (**-71%**)
- spill 0, scratch 0, correctness and 200-run determinism unchanged

and it measured **17.3% slower at prod, 15.1% slower at proxy.**

> **Finding: removing 88 VGPR, 57 `s_set_vgpr_msb` and 58 `v_nop` from `k_dkdv` makes it
> 17% slower. The window/hazard overhead is real in the instruction count and is NOT on the
> critical path.** The gfx950 `v_accvgpr` analogue (1.72x) does not transfer to gfx1250's
> flat file. Do not propose another candidate whose payoff is "fewer `s_set_vgpr_msb`".

And the same measurement re-prices `r7.i1.g21` **upward**: the prefetch was +8.3% when it
landed in round 7; on today's body it is worth **~+21%** (504.90 / 417.73). It is the single
largest surviving mechanism in this kernel. Twelve rounds of other changes made the body
more dependent on it, not less.

> **Corollary for `h16` and for `bound`:** a kernel whose time collapses 17% when you remove
> one load's worth of latency cover, while 82 issue slots of hazard overhead go free and buy
> nothing, is **latency-bound, not issue-bound**. Round 18 recorded `bound: latency, LOW
> confidence` from a model that has since been retired. This round it is a measurement.
> **Confidence: MEDIUM.**

### 5.3 Where that leaves the LDS-port hypothesis (1.3)

Untouched, and now the only surviving mechanistic story: 64 of `k_dkdv`'s 80 LDS ops are the
Q/dO round trip, 1.25 LDS ops per WMMA against `k_dq`'s 0.33, and `k_dq`'s body has zero
stall. Neither arm tested it. `r16.i3.g50` is the arm that does, and it is now the route's
only `idea` row.

## 6. Shipped

`rounds/019/op` = `r19.i1.g59` (`kernels.py`, the two-run WMMA split with `reuseA`).
It is the best of the two arms and it is correct, deterministic and non-negative at prod --
but it is **null inside the floor**, so the honest statement is: *this round shipped no
speed.* A later round may revert it at zero cost.

Incumbent as measured this round: **505.29 TF/s prod, 439.83 proxy** against a re-measured
bar of **719.50 / 584.97** -- **ratio 0.7023 at prod**, geomean 0.811x of the bar.

## 7. 验收算术(照报,不修饰)

| shape | this round (g59) | champion r17,本 session 重建 | beat,本 session 重测 | ratio | vs champion |
|---|---|---|---|---|---|
| prod | **505.29** | 504.90(两槽 506.42 / 503.37,地板 0.61%) | 719.50 | 0.7023 | 1.0008 |
| proxy | **439.83** | 440.29(443.39 / 437.18,地板 1.42%) | 584.97 | 0.7519 | 0.9990 |
| fast | 53.11 | 54.08(55.59 / 52.57,地板 5.74%) | 52.06 | 1.0000(未截断 1.0202) | 0.9821 |

`score` = mean(ratio) = **0.8181**,incumbent 同 sweep 为 0.8180。
throughput **没有**超过历史最好(prod +0.08% / proxy −0.10%,两者都在同码地板之内),
所以按验收规则这一轮**不会被 promote** —— 这是预期内的,并且是照实报的:
本轮出货的是一个正确、确定、prod 非负但**不带速度**的 arm,
买到的是三条机制性结论(`bound` 改写、寄存器窗口轴关闭、`g21` 重定价为 ~+21%)。

`rounds/019/op` 保持本轮实测的状态(`kernels.py` 的 18 行 g59 diff),未回退;
`op/current/` 全程未动(`diff -r` 验证)。
