<!-- operator-tables: written by the framework, not by the round -->

## Standing constraints -- read every round; never executed, never retired

_None._

## Refactors -- run by the loop at a round's head, one at a time

_None._

## Operator items awaiting a place in the route

Put each of these in the Route table below. **Every `must` one must sort above every `idea` row**, and that is checked after you plan and before you build anything.

_None._

<!-- /operator-tables -->

## Route -- designed by round 4, after both of its arms lost off a new surface and the last instrument died

| # | type | id | what it is | condition | outcome |
| --- | --- | --- | --- | --- | --- |
| 1 | must | r4.m1 | every candidate reports **`vgpr_count`, `vgpr_spill_count` AND `s_set_vgpr_msb`** from the free static census before it is given a timing slot, and dies on the census if its stated premise does not move the number it claims to move | this costs no GPU and it is the only surviving gate. It is a `must` and not advice because round 4 broke the one-number version of it twice in one session: `disable_licm` took scratch **1064 -> 944** B/lane and `scratch_load` **265 -> 156** and was **+17.5%**, while pushing `s_set_vgpr_msb` **1666 -> 1748**; and `unroll2_nolicm` has **fewer** spills than `dkdv_unroll2` (2913 vs 3678) and is **16% slower**. The two overheads trade against each other, so a candidate that reports one of them is not reporting its result. The DISCARD half of `r2.i3.g10` still holds -- everything that got much worse on spills also got much worse on time, which is how `loop_unroll_factor=2` was killed for free before it cost 21 minutes -- and the MEASURE half is dead four times over |  **applied, and it paid for itself.** It killed `tl.range(num_stages=2)` for free on the census -- spills **266 -> 2001**, scratch **1064 -> 2468** B/lane, instrs 9547 -> 14694, `s_set_vgpr_msb` 1666 -> 2338 -- so that arm never cost a timing slot. It also confirmed `flatten=True` as census-identical to base (1024/266/1064/1666/448/9547, byte-for-byte the same counts), which is why that one was cheap to time and came back inert. And it fired on row 2: g17's stated falsifier was `s_set_vgpr_msb` **below** 1666 and it went **UP, 1666 -> 1728**. That row was timed anyway -- recorded as a deviation, not a defence -- because the same census showed the first **zero-spill, zero-scratch** `bwd_kernel_causal` in four rounds and there was no other way to price that. The timing agreed with the census gate: **+7.8%**. The gate was right and the override bought one number and one round's delay |
| 2 | idea | r4.i2.g17 | split the dk/dv pass into **two sequential m-loops**, one accumulating `dv` and one accumulating `dk`, halving peak accumulator residency from 512 VGPRs to 256 and the loop-invariant set from 768 to 512 | this is the round's largest item and **the only structural one expressible in Triton**, which is the whole reason it ranks above row 4. It targets the term the corpus prices highest: `gqa_d128` s6.1 measures **1.72x** for the above-register-255 boundary on **the same operator, the same head dim, the same 1 wave/SIMD, identical MFMA counts**, and our form of that boundary is 1666 static `s_set_vgpr_msb`, 363 per dk/dv iteration, 26% of the body. Round 3 discounted it to 6-14% with a power coefficient taken from a GEMM card; **round 4 killed the coefficient** -- this kernel droops 10% off `MAX_CLK` where that GEMM droops 29%, and the `disable_licm` arm proves it is latency-bound rather than energy-bound by getting 17.5% slower when memory traffic was traded away for arithmetic. Tile, `BLOCK_N1`, fusion and launch config are all untouched, so it is none of the closed surface. It costs a second pass over Q/dO/LSE and a recompute of `p`, which is the trade row 3 of round 4 warns about -- but that arm recomputed **address arithmetic onto the critical path** and this moves **independent bulk work** a 264-trip loop can overlap, and that disagreement is the reason to measure it rather than argue it. **Falsifiable on the free census: if the split does not move `s_set_vgpr_msb` below 1666, the premise is wrong and it dies for a compile** |  **built, correct, DELIVERED AND LOST -- and it is the round's real finding.** Census: `vgpr_count` **1024 -> 973**, `vgpr_spill_count` **266 -> 0**, `private_segment_fixed_size` **1064 -> 0** B/lane, `scratch_load` **265 -> 0**; the first spill-free build of this kernel in four rounds, landing where round 3 measured the dq pass alone (971). The premise's own number went the wrong way: `s_set_vgpr_msb` **1666 -> 1728**, and `wmma` **448 -> 512** (+14%, the recomputed `pT` in phase 1). Time, five independent points on an idle GPU[1] in one session, `absum` bit-identical to base throughout: g17 **9.3689** / **9.3816** (repeatability 0.14%) against base **8.6873** / **8.6867** (spread **0.007%**), i.e. **+7.84% / +7.98%**; validation re-measured it at **9.3782** against baseline 9.9975 and beat 11.6458 -- **1.242x against a 1.50x bar, FAIL** (incumbent 1.342x). Slower on **every** shape: 0.925x / 0.996x / 0.974x / 0.786x / 0.804x. SQNR 52.0-54.1 dB, all four gated shapes pass. **So: removing all 266 spills and all 1064 B/lane of scratch costs 7.8%.** That is the fifth and by far the strongest counter-example to spill-count-as-proxy, and the first one where the proxy was driven to its ideal value. Shipped in `rounds/004/op` per the no-revert rule |
| 3 | idea | r4.i1.g16 | `tl.range(num_stages=N)` and `flatten=True` applied to the **dk/dv loop alone**, the unmeasured half of the source-level loop surface | the cheap second arm, and separable from row 2. It is not the closed launch surface: `num_stages` in `_DEFAULT_ONEKERNEL_CONFIG` is whole-kernel, which is why round 3 read it as 12.2-12.9 ms -- it also pipelines the dq loop, which the census shows runs at **zero** scratch traffic and so has no slack to buy a buffer with. Per-loop staging has never been expressible before and was never tried. It is the one family that attacks dependency latency **directly** rather than by removing work, which is what round 4 s7 says this kernel is actually paying. Run the row-1 census first and expect it to kill the `num_stages` half the way it killed `loop_unroll_factor=2` (266 -> 3678 spills); `flatten=True` changes no live set at all and survives that gate by construction |  **half DISCARDed free, half measured inert.** `tl.range(num_steps, num_stages=2)` on the dk/dv loop alone died on the row-1 census exactly as that row predicted it would (2001 spills, 2468 B/lane) and was never timed. `tl.range(num_steps, flatten=True)` produced an `.amdgcn` with **identical** counts to base on all seven census fields, and timed **8.7072** against base 8.6873 -- **+0.23%**, inside no margin anyone should defend but on the losing side of it. **The per-loop `tl.range` surface is now closed**: staging is unaffordable at this register pressure and flattening is a no-op on a loop the compiler already flattens. Not shipped |
| 4 | idea | r1.i7.g7 | stage the dk/dv pass's `k` 128 + `v` 128 loop-invariant VGPRs in LDS at constant Q/dO traffic | **rank unchanged, but its pricing premise fell this round and must be rebuilt before it is opened.** The item argues "at 1 wave/SIMD there is no second wave to cover an exposed scratch reload, so removing scratch traffic pays." Round 4 removed **41% of the static `scratch_load`s and 11% of the scratch footprint** and paid **+17.5%** -- so those reloads are not on the critical path the way the item assumes. Its structural half still stands (Triton offers no surface to say what lives in LDS across the m-loop; the `shared = 65536` reading from round 3 stands, so the headroom is real) and its gate no longer exists, because row 1 of round 3's table -- the byte-vs-issue instrument -- is dead on this part with no successor. **So this row cannot be opened on evidence; it can only be opened as a rewrite outside the backend, and row 2 reaches the same 1.72x term from inside it.** That is why it sits below row 2 despite being the larger item |  **not opened, and row 2 has now made its case worse.** g17 drove scratch traffic to **exactly zero** -- the end state LDS staging is trying to approximate through a surface Triton does not expose -- and it was **7.8% slower**. The item's headline justification ("at 1 wave/SIMD there is no second wave to cover an exposed scratch reload, so removing scratch traffic pays") is now refuted at both ends: at 41% removed (+17.5%) and at 100% removed (+7.8%). It should be rewritten or retired before it takes another slot |

Rows 2 and 3 are the round's two arms, built separately from `op/current` and merged
only if neither loses. Row 1 is a gate, not a competitor, and costs no GPU time.

**Round 4 shipped nothing, and it is the second round in a row that did.** Its two
arms came off a surface no earlier round had touched -- `tl.range` loop controls and
`tl.assume`, both present in Triton 3.6.0 and both absent from
`fused_mha_bwd_kernel.py` -- and both failed: `disable_licm` at **+17.5%** and
`tl.assume` **inert**, the latter closed permanently by an IR reading rather than a
time (301 assume ops in the TTIR, **zero `llvm.assume` in the LLIR**, so the facts
never reach the backend that decides addressing). Validation on the working copy:
candidate **8.6972 ms / 632.19 TF/s**, beat 11.6684 / 471.21, baseline 10.0024 /
549.69 -- **1.342x against a 1.50x bar**, unchanged from round 3 because nothing was
shipped.

What round 4 bought instead is the pricing, and every row above depends on it:

- **This kernel is latency-bound, not energy-bound.** The `disable_licm` arm traded
  the most energy-expensive class of work in the kernel (41% of static
  `scratch_load`) for the cheapest (VALU recompute) and got **17.5% slower**. Under
  `time = energy / P_limit` that trade is close to free; it was not free, so the
  cap is not what is setting the time.
- **The 20-45% instruction discount does not transfer.** It was calibrated on
  sustained bf16 GEMM at **-29%** off `MAX_CLK`; `bwd_kernel_causal` runs at
  **-10.0%** (2161 MHz median of 30 samples over 60 s, against 2400 idle). The
  power half of the regime test is **unreadable** -- `rocm-smi -P` omits the
  `GPU[1]` row in every load sample while printing it for the three other cards --
  so the discount can no longer be asserted, and instructions price near full.
- **`s_set_vgpr_msb` is therefore the largest priced term in the kernel**, at 26% of
  the dk/dv body, with a corpus number of 1.72x for the same boundary on the same
  operator. That is what rows 2 and 4 are both aimed at.
- **The device is clean.** `HIP_VISIBLE_DEVICES=1` is confirmed to be `rocm-smi`
  `GPU[1]` (0% use idle, 100% under load, neighbours at 0%); the foreign processes
  recorded against this job are on CPU and on `GPU[0]`, not on our card.

Items deliberately not in the table. **`r1.i3.g3`** (`schedule_hint`) -- still
excluded, and round 4 repaired the argument for excluding it: round 3's claim that
`TRITON_HIP_USE_IN_THREAD_TRANSPOSE` is inert on gfx1250 is a misreading of a
ternary (`compiler.py:29`: the `arch == "gfx942"` test is the `is None` branch, and
`impl.py` sets the knob), so `g2` is a real transform and `g3`'s measured
anti-synergy with it, 8.941 against 8.696, means what it says. **Do not delete the
context manager in `impl.py`.** **`r1.i4.g4`** -- refuted in round 2 and nothing
since rehabilitates it. **`r3.i4.g15`** -- dead for one grep in round 4's look: every
global store is already `buffer_store_b128`, 96 of them and zero `b64`. **The power
number** -- unreadable on the loaded device, fourth instrument to die that way; do
not spend a row on it, and do not re-attempt PC sampling or ATT.
