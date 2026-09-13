# TOMORROW: 10 HOURS, FOUR GPUs — ORDERED EXECUTION PLAN

## THE ANSWER ON FlyDSL, UP FRONT

**No. Do not start FlyDSL. Not the port, not the forward, not Gate A.**

Evidence, in order of finality:

1. **It is not installed and the build excludes it.** `import flydsl` → `ModuleNotFoundError`; `setup.py:529-534` drops it from any `--offload-arch=gfx1250` build. The pinned 0.2.4 does not expose `tdm_ops`, `s_wait_tensorcnt`, `ds_load_tr16_b128` or `permlanex16`, all of which the off-main FMHA imports directly. Nothing in-tree names a version that does. This is an external-dependency gate that may never clear, and it is step zero.
2. **The only thing that exists is the wrong shape and the wrong half.** The off-main 7677 lines are D_qk=192 / D_v=128 **varlen MLA forward**. Our shape is D=128 dense causal. The "backward" at `7e93607c` is 135 lines of Python looping `for b: for h:` materialising full `[8192,8192]` fp32 S and P — a correctness oracle, not a kernel. **No gfx1250 attention backward exists anywhere, in any tree, by anyone.**
3. **The forward is a trap even if everything works.** Our forward is already 521 TFLOP/s = 52% of roof. A *perfect* D=128 FlyDSL forward at the 770 TFLOP/s structural ceiling saves 1.37 ms of 24.435 — **5.6% end-to-end for 18-36 engineer-days**. And we have already refuted the 0.9 ms forward gap: it does not reproduce under profiling (4.219 vs 4.035).
4. **The port direction is the wrong way down the cost curve.** The reference kernel fits its per-WMMA budget at 6.7 cy/WMMA *because D=192 inflates the WMMA count*. At D=128 the same 1076-cycle softmax stream divides by 128 WMMAs instead of 160 → **8.4 cy against a 6-7 cy shadow**. Porting moves it from "fits, barely" to "structurally over budget," and the 96-row hand-edited GEMM1 schedule table has to be re-derived, not truncated. That table *is* the deliverable and it is 6-12 days on the forward alone.
5. **A strictly dominating alternative is already in-tree.** `3rdparty/hipkittens/include/udna1/` is a complete 72-header gfx1250 port with `wmma_f32_16x16x32_bf16`, TDM 2-D descriptors, split wait counters, cluster barriers and a working bf16 GEMM ladder — **with no external dependency to acquire**. It reaches the same primitives. If we ever spend 30-50 days hand-scheduling a gfx1250 backward, it goes there, not into FlyDSL.
6. **Opportunity cost tomorrow is decisive.** Gate A costs 1-2 hours to learn "blocked or not blocked" about a program whose payoff rests entirely on a number (the backward ceiling) that *no cheap experiment can measure*. The same 1-2 hours run Item 1 below, which can retire the entire backward problem outright.

Revisit only if Item 1 fails **and** the udna1 `transpose()` identity fails **and** someone hands us a working gfx1250 FlyDSL wheel. Three conditions; do not spend a minute on it tomorrow.

---

## GPU PARTITION (fixed for the day)

| GPU | Role | Held how long |
|---|---|---|
| **0** | **Decision engine.** Re-baseline, then the aiter gfx1250 ASM probe, then whichever follow-on the probe selects. Short, high-stakes, re-tasked at hour 2. | Re-tasked twice |
| **1** | **Occupancy & config sweep** on the fused Triton backward. Short jobs, many points. | All day |
| **2** | **VALU diet + layout**, cumulative, one change at a time with SQNR after each. Short jobs. | All day |
| **3** | **THE LONG HOLD.** udna1 register-tier unit tests (mostly CPU authoring, short GPU runs), then the GEMM ladder, then the softmax micro-kernel. Never interrupted, never shared. | All day, one owner |

Four **independent hypotheses**, not one sweep sharded four ways. A dud costs 2.5 GPU-hours, not 10.

---

## HOUR 0 — 0.0 to 0.5 — RE-BASELINE (all four GPUs, mandatory, blocks everything)

Today's numbers are VR-throttled to ~1100 MHz. Every absolute time shifts tomorrow.

Run the shipped champion (b=4 s=8192 hq=32 hkv=8 d=128 bf16 causal) **on all four GPUs**, plus `torch flex` and `pure aiter tuned`, and record the achieved clock per GPU.

- **Accept:** four GPUs within 3% of each other → one baseline number for the day.
- **Reject:** spread >3% → pin each stream's comparisons to its own GPU's baseline and say so in every result line. Do not cross-compare GPUs.
- **Abandon:** never. This is 30 minutes and without it every speedup tomorrow is attributed to the wrong cause.

**Reporting discipline, all day:** report the backward on **both** bases — nominal 5-GEMM (5.498 TFLOP) and **issued** 7-GEMM (7.697 TFLOP). The vendored kernel issues seven score-matrix passes, not five. Single-basis reporting understates real machine utilisation by 1.22× and the team will compare against the wrong roof.

---

## ITEM 1 — GPU 0 — 0.5 to 1.5 — THE aiter gfx1250 ASM v3 BACKWARD PROBE

**Rank 1. Highest information-per-second measurement available. Run before anything else is queued.**

There are six prebuilt `gfx1250` `.co` files at `/home/lihuzhan/code/aiter-src/hsa/gfx1250/fmha_v3_bwd/`, including `bwd_hd128_bf16_causal_br_a32_pssk.co` (+ `_perf.co`). `readelf -n` confirms `amdgcn-amd-amdhsa--gfx1250`, wave32, 1024 VGPRs, 320 KB LDS. The CSV row `bf16,128,128,mask=2,atomic32=1,pssk=0,mode=0,ts_qo=32,ts=128` matches our production shape in **every field**. The C++ host path (`csrc/cpp_itfs/mha_bwd.cu`) already special-cases gfx1250 in five places. **The only blocker is a missing Python arch gate** — there is no `can_impl_fmha_v3_bwd_gfx1250()`, so it silently falls through to Triton.

**Do:** in a **scratch script** (never patch the installed aiter — that corrupts the 21.684 ms reference and every other measurement in the queue), import the bare `@compile_ops` stub `aiter.ops.mha.fmha_v3_bwd` (`mha.py:1397`, no arch check) and call it with `is_causal=True, window_size_left=-1, window_size_right=0, deterministic=False, is_v3_atomic_fp32=True, how_v3_bf16_cvt=0`, preallocated dq/dk/dv, and **`dq.zero_()` first** (gfx1250 is atomic32-only; the gated path zeroes at `mha.py:2531` and a direct call bypasses it → silent garbage dq).

First verify the **installed** aiter ships `hsa/gfx1250/fmha_v3_bwd/`, not just the source clone. If the wheel omits it, the path is blocked on packaging — 15 minutes to find out.

**Accept/reject:**
- Raises, or returns −1, or warns "unsupported mask type" → **close Path 3 in one hour.** Move GPU 0 to Item 6.
- Runs, but per-tensor SQNR on dq/dk/dv is not ≥40 dB → **close it.** This explains the missing upstream gate (precedent: aiter's `mha_fused_bwd` ran clean and produced dk at −0.22 dB). Move GPU 0 to Item 6.
- Runs, SQNR clean, **and backward < 18 ms** → **this is the day.** Go to Item 2.
- Runs, SQNR clean, but ≥ 20 ms → record it, close it, move to Item 6. A vendor ASM kernel that does not beat our 20 ms is not worth the vendoring cost.

**Check the return value, not just the wall time.** A silent fallback to Triton reads as "no speedup" rather than as an error.

**Abandon trigger:** two hours elapsed without a clean SQNR number. It is a 15-minute experiment; if it has eaten two hours, the ABI is not what we think and the remaining eight hours belong to the Triton streams.

---

## ITEM 2 — GPU 0 — 1.5 to 6.0 — SHIP THE ASM BACKWARD *(conditional on Item 1 accepting)*

**Rank 2 if Item 1 accepts; deleted otherwise.**

1. Force **both** `.co` variants and measure separately — `dqdkdv_co_name()` silently prefers `_perf.co` when present, and it is present for both gfx1250 rows, so an unforced run is not measuring the kernel the CSV names. A `_perf` variant can be faster *and* less numerically careful; SQNR both.
2. Add `can_impl_fmha_v3_bwd_gfx1250()` mirroring the gfx950 one, **refusing `deterministic=True`** (fp32 atomic dq accumulation is run-to-run non-deterministic; exclude this path from the bitwise-determinism tests, commit `49257416`).
3. Fall back to the vendored Triton path for **everything the CSV cannot serve**: the gfx1250 rows are `pssk=0` only, so any `seqlen_q != seqlen_k` or `seqlen_k % 128 != 0` finds no kernel and returns −1. The 51-test suite will hit those.
4. Run the full 51 tests. Green is the gate to shipping.

**Accept:** 51/51 green, SQNR ≥40 dB on all three tensors across all tested shapes, end-to-end beats 24.435 ms (re-baselined) by ≥1.15×.
**Reject:** any test red, or any shape silently returning −1 without hitting the fallback.
**Abandon:** do **not** attempt in-tree vendoring of the `.co` loader tomorrow. `mha_bwd.cu` pulls `ck_tile::stream_config` and CK is force-disabled on gfx1250 (`setup.py:40,252-262`) — the loader must be reimplemented, not lifted. That is 1-2 days, and it is next week. Tomorrow ships against the installed aiter behind the new gate, with the vendoring as a follow-up ticket.

---

## ITEM 3 — GPU 1 — 0.5 to 4.0 — OCCUPANCY: THE CHAMPION'S REGISTER BUDGET

**Rank 3 overall, rank 1 among the Triton items. Largest single prior: 1.15-1.3×.**

**Start with the 20-minute ISA dump, before any sweep.** Dump AMDGCN metadata for the shipped champion and read actual VGPR-per-lane and spill counts. Arithmetic says: `BLOCK_N1=256`, `num_warps=4` on wave32 = 128 lanes → dk[256,128]f32 + dv[256,128]f32 = **512 VGPR/lane in accumulators alone** (L784-791), plus resident bf16 K and V [256,128] across the whole `hqid` loop (L847-853) = 256 more. **768 of 1024**, at 1 wave/SIMD with `num_stages=1` — no latency hiding from occupancy *or* pipelining, squarely in the `s_set_vgpr_msb` bank-switched stall regime that is a gfx1250-only class.

Then sweep, in this order:
1. `num_warps` 4 → 8 → 2 at `BLOCK_N1` ∈ {128, 256}. Six points. `num_warps` is in `_ONEKERNEL_KEYS`, so `PRIMUS_TURBO_FUSED_MHA_BWD_TUNE` reaches it.
2. `num_stages` 1 → 2 jointly with the winner. (num_stages 1→2 was worth **2.4×** on the forward and is still 1 here.)
3. `waves_per_eu` **only jointly with a winning `num_warps`** — at 768 VGPR/lane it is a no-op knob and any prior sweep of it measured nothing.

**Accept:** ≥1.08× on the backward with SQNR unchanged → promote to the config table immediately and re-run every other stream's baseline against it.
**Reject:** <1.03× across all six points.
**Abandon:** if the ISA dump shows VGPR/lane well under 512 with zero spills, the register-pressure thesis is dead — stop the sweep at the 20-minute mark, revise the Q4 ceiling down toward 15-16 ms backward, and move GPU 1 to reinforce Item 4.

*Hazard on the record:* provenance documents only three tuned deltas (`BLOCK_N1`, `BLOCK_M2`, `BLK_SLICE_FACTOR`); `num_warps=4` and `num_stages=1` are inherited from aiter unchanged. If they *were* in fact swept on the fused kernel, this item is already dead and the day's expected win drops from ~1.4× to ~1.15×. The ISA dump tells you which world you are in, in 20 minutes.

---

## ITEM 4 — GPU 2 — 0.5 to 6.5 — THE VALU DIET AND THE LAYOUT FIX

**Rank 4. Cumulative, one change at a time, SQNR after each. Expect 1.10-1.30× combined.**

The kernel is softmax-VALU bound by ~1.2-1.4× at D=128 and WMMA issues on the vector ALU, so this stream attacks the binding constraint directly.

**4a (30 min, do first, it may delete 4b):** dump the Triton IR and check whether LICM already hoists `m*RCP_LN2`. Do not assume.

**4b — pre-exp stream, 3 ops → 1.** Fold `sm_scale*RCP_LN2` into q at load (`L1098`; 32768 scalar muls total vs 34.6M in-loop) and hoist `m*RCP_LN2` out of the loop (`L517/L529`, and identically `L296/L313`).

**4c — one-line move.** `Di` is loaded mid-loop at `L356`, *after* the dv dot, while `m` is deliberately loaded early at `L285` with the comment "Load m before computing qk to reduce pipeline stall." The same argument applies to `Di` and was not applied.

**4d — accumulator-init form.** `tl.dot(q_prescaled, kT, acc=neg_m)` at `L513`, then `p = exp2(qk)` at `L529` with zero pre-exp VALU. **dq pass only** — in the dkdv pass `m` advances with `curr_m` and is not loop-invariant. Caveat to measure, not assume: the acc tile is [256,32]f32 = 64 VGPR/lane re-initialised every iteration, so the win is 3 ops → ~1 (the `v_mov`), not 3 → 0, unless the compiler folds the init into the WMMA D-operand write.
  *Do **not** attempt the true fixed-zero reference max (dropping `-m` and rescaling dq rows at the end). It is algebraically legal in the dq pass only — in dkdv the factor lies along the reduction dimension of both dots and applying it there produces smoothly wrong gradients, not a crash — and unshifted `exp2` overflows fp32 above `qk*log2e = 128`, which random-normal test inputs will never trigger. The accumulator-init form gets most of the win with none of the risk.*

**4e — kill the three per-iteration `tl.trans`** (`L363`, `L376`, `L562`) by carrying dual pointer sets (`do_ptrs` **and** `doT_ptrs`, etc.). Extra L2 traffic is trivial (an [32,128]bf16 q tile is 8 KiB, L2-resident across the 4-deep `hqid` loop). **Dump the ISA for one iteration first** to see whether Triton's AMD backend emits `ds_load_tr16_b128` or falls back to a permute network — that check alone decides whether this is worth 2% or 10%.

**4f — hoist K/V across the `hqid` loop in the dq pass.** `_bwd_dq_inner` re-reads *identical* kT/vT tiles on all four `hqid` iterations (`L489-492`, called per-hqid at `L1137/L1197`); `end_n` depends only on `start_m`, not `hqid` (`L1041`), so restructuring to loop n outside and hqid inside cuts dq-pass K/V traffic **4×**. This is the closest real analogue of the HipKittens "stage K/V once per workgroup" idea — the dkdv pass already does it (`L847-853`) and that is why the fused kernel wins.

**4g — `EVEN_N`/`EVEN_M` constexprs** to unpredicate the `L489-492` loads. **Measure isolated.** This looks superficially like the already-refuted "skip the causal mask on non-diagonal blocks" experiment but is not — `MASK` is already a constexpr split across two call sites (`L926` True, `L1018` False), so there is no scalar branch to defeat the pipeliner. It is the load predication, not the select. If the 9% regression's real cause was load-address divergence rather than the branch, this regresses the same way; that is exactly why it runs alone and last.

**Accept, per change:** ≥2% with SQNR unchanged → keep, rebase, proceed to the next.
**Reject, per change:** <1%, or any SQNR movement → revert, proceed.
**Abandon the stream:** if 4a shows Triton already hoisting *and* 4b+4c together give <2%, the VALU thesis is weaker than modelled — skip to 4e/4f (layout, a different mechanism) and drop 4d/4g.

*Do not spend any time on the ones-operand denominator. There is no `l_i` accumulation in this backward — normalisation arrived with the LSE. The only hot-path `tl.sum` is `L549`, gated on `ENABLE_SINK` which is off, and `L165` in `_bwd_preprocess`, a bandwidth-bound 537 MB pass where a matrix row-sum buys nothing. The trick belongs to the forward's `l_i`, and the forward is not in play.*

---

## ITEM 5 — GPU 3 — 0.5 to 10.0 — THE LONG HOLD: udna1's ONE BINARY RISK

**Rank 5. This GPU is held all day by one owner and is never re-tasked.** It is mostly CPU authoring with short GPU runs, which is exactly why it can hold a card without wasting it.

The question that gates 7-10 engineer-weeks of HipKittens work is **binary and answerable in five hours**: *does the byte-identical register tier actually compute the right thing under wave32 + WMMA?* Everything else in that estimate is schedulable work. This is the only coin flip.

**5a (0.5 h) — build wiring.** `setup.py:347-349` appends `-DKITTENS_CDNA4` **unconditionally** and `-DBUILD_HIPKITTENS_BACKEND` is gated on gfx950 only (`setup.py:360`), so a gfx1250 build today compiles HipKittens against wave64 headers. ~20 lines plus a `*_gfx1250.cu` suffix (the `filter_files_by_arch` hook at `setup.py:152` already handles the suffix). Local clang is 23.0.0git at `/opt/rocm-10.1.0a20260807` — the ladder's clang-22+/ROCm-7.2 requirement is already met, no container.

**5b (1.5 h) — GEMM ladder smoke.** Build and run rungs `gemm_naive` → `gemm_expert` under `-DKITTENS_UDNA1 --offload-arch=gfx1250`. **Skip `gemm_tdm_arrive` on the first pass and run it last, alone, under a timeout** — its own header warns it hangs on runtimes that do not model `DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64`, and it does not actually exercise the auto-arrive path it was written to prove (it falls back to manual `laneid()==0` arrive). This produces the **first TFLOP/s number any udna1 kernel has ever produced** and a free calibration of codegen quality on this part.

**5c (5 h) — THE ONE THAT MATTERS: port the missing unit tests.** `tests/unit/udna1/` is 444 lines covering only `warp/memory/tile`. `tests/unit/cdna4/` is 2839 lines covering `warp/{memory,shared,register}` + group. **The ~2400-line delta is exactly the register and shared tiers — exactly the byte-identical files whose wave32 correctness is in question, and exactly `mma_ABt`/`wmma161632`, which no test in the tree exercises.** Copy `cdna4/warp/{register,shared}/` into `udna1/`, switch the Makefile define. Mechanical, because the headers under test are byte-identical.

Priority order, strictly:
1. **`conversions.cuh::transpose`** — 11 gfx950-backward call sites ride on it, and it is a *pure register relabel with no cross-lane op*, an identity that holds only because MFMA's A-operand and C-accumulator maps are duals on CDNA wave64. WMMA `f32_16x16x32_bf16` (A v16bf16, C v8f32 across 32 lanes) has a different duality. **Asserted, never executed.**
2. `reductions.cuh` row/col max+sum — uses `__builtin_amdgcn_permlane32_swap` under a comment describing "row 2 and 3" of a **64-lane** accumulator, unchanged on wave32.
3. `mma_ABt`/`mma_AtB` shape coverage.
4. `maps.cuh::exp2`.

**Accept:** transpose + reductions green → the 7-10 week estimate becomes a plan, and `ds_load_tr16_b128` (currently wired nowhere — `grep -rn 'ds_load_tr' include/udna1/` returns zero) stays optional.
**Reject:** transpose red → **that is also a good day.** It reprices the backward by +5-8 days, makes `ds_load_tr16_b128` plumbing a hard prerequisite, and is a five-hour answer to a question that otherwise surfaces in week four.
**Abandon:** if 5a and 5b cannot produce a compiling, numerically correct `mma_ABt` GEMM by hour 3, stop. The primitive the whole port rests on does not work and no test porting will change that — write that up and give the remaining hours to Item 6.

**5d (3 h, only if 5c finishes early) — the softmax micro-kernel + co-issue calibration.** ~150 lines on the `gemm_naive` skeleton: QK^T on one 32×64 tile, `exp2` on a fixed zero max, bf16 convert, PV, denominator by mma-against-ones. No mask, no pipelining, no TDM. Check against torch, then `rocprof` the VALU-vs-MMA issue-slot ratio.

**This last measurement recalibrates every ceiling on the board.** The whole Q4 ladder assumes WMMA and softmax VALU **fully serialise** on gfx1250. If udna1 restores partial co-issue, the softmax wall drops from 9.2-9.9 ms toward 8.5 ms and every ceiling improves ~8%. **Run it before any of tomorrow's ceiling numbers are quoted to anyone.** If it can be squeezed in, promote it ahead of 5c priority 4.

---

## ITEM 6 — GPU 0 — 2.0 to 10.0 — THE STRUCTURAL BET *(runs only if Item 1 rejects)*

**Rank 6. This is the fallback owner of GPU 0, not a parallel item.**

Prototype the **5-GEMM one-pass backward**. This kernel currently issues **seven** score-matrix GEMM passes, not five, because it walks the matrix twice to avoid dq atomics — dkdv does 4 (`L291, L353, L363, L376`), dq does 3 (`L513, L544, L562`). That makes the hard MFMA floor **7.68 ms** for this structure versus **5.48 ms** for FA2's five. Every other item tomorrow closes the gap to the floor; **this is the only item that moves the floor.**

Design: the dkdv loop also emits dq via split-K into a `[4, b, hq, s, d]` fp32 partial buffer (2.1 GB, +~0.5 ms reduce pass) rather than atomics.

**Non-negotiable constraint:** preserve constant-work-per-pid. Today dkdv does `8+8*(31-p)` steps and dq does `8+8*p`, summing to **264 for every p** — perfect causal load balance across a (8, 32, 4) = 1024-workgroup grid on 256 CUs, exactly 4 full rounds with no tail. That property is the fused kernel's single best feature and a naive one-pass rewrite gives it back entirely, which costs more than the 29% MFMA reduction gains.

**This is explicitly *not* the refuted `sequence_parallel=False`** — that computed 1/128 of the gradient and timing-only measurement called it a win. The difference is the explicit partial buffer and the reduction. **Gate on gradient SQNR before any time is permitted to be reported.**

**Accept:** a half-working prototype with a measured inner-loop issue rate → fund the 1-2 week restructure next cycle.
**Reject:** inner-loop rate no better than the current 7-GEMM kernel's, or load balance visibly broken in the step counts.
**Abandon:** it will probably not be correct-and-fast by hour ten, and that is fine — **the deliverable is a funding decision, not a kernel.** But if by hour 7 it is not producing numerically sane gradients on a small shape (b=1, s=1024), stop and write up the design with the step-count analysis. A correct write-up beats a broken prototype.

---

## MIDDAY GATE — HOUR 4.0, 15 MINUTES, ALL HANDS

Re-plan on three facts:

1. **Item 1 verdict.** Accept → GPU 0 is committed to shipping through hour 6, and Item 6 never runs; the day's headline is "ASM backward shipped." Reject → Item 6 owns GPU 0 for six hours.
2. **Item 3's ISA dump.** 768 VGPR/lane confirmed → occupancy is the main line and Items 4b-4d are second-order. Refuted → the VALU diet is the main line and the Triton ceiling revises down toward 15-16 ms backward.
3. **Item 5b.** If `mma_ABt` does not produce a correct GEMM, the entire HipKittens program is on hold and tomorrow's write-up says so.

Anything that has not cleared its accept bar by hour 4 is abandoned, not extended.

---

## EXPECTED VALUE, RANKED

| # | Item | GPU | Hours | Expected win | Probability |
|---|---|---|---|---|---|
| 1-2 | aiter gfx1250 ASM v3 backward | 0 | 0.5-6.0 | **1.20-1.45× end-to-end**, and the backward leaves our maintenance surface | ~35% it lands clean |
| 3 | Occupancy / `num_warps` / `num_stages` | 1 | 0.5-4.0 | 1.15-1.3× on the backward | ~55% |
| 4 | VALU diet + layout + K/V hoist | 2 | 0.5-6.5 | 1.10-1.30× cumulative | ~70% for ≥1.10× |
| 5 | udna1 transpose/reduction binary | 3 | 0.5-10.0 | 0× tomorrow; de-risks or kills a 7-10 week program | ~90% it answers |
| 6 | 5-GEMM prototype *(fallback)* | 0 | 2.0-10.0 | 0× tomorrow; funds or kills a 1-2 week restructure | ~50% it decides |
| — | **FlyDSL** | **none** | **0** | **Not started. See above.** | — |

**Realistic end-of-day:** if Item 1 lands, ~17-20 ms end-to-end and the backward is a vendor problem. If it does not, Items 3+4 compose to roughly **1.25-1.45× → 17-19.5 ms**, which beats pure-tuned aiter (21.684 ms) with our own in-tree kernel, plus two funded/not-funded decisions on multi-week programs.

**Ceilings for reference, backward only:** 7.68 ms is the 7-GEMM MFMA floor (unreachable); **9.2-9.9 ms is the real wall** (that floor plus serialised softmax); 5.48 ms is the 5-GEMM floor. Today's 20 ms is 2.0-2.2× above the wall; aiter-tuned's 17.5 ms is 1.8-1.9×. Well-scheduled Triton on AMD lands at 1.3-1.5× of an issue-slot wall, which is **12.5-14 ms backward / 16.7-18.2 ms total**. Below ~13 ms total is not Triton.