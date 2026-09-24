# Round 18 -- opt (fast round)

Base: `job_context/op/current` (untouched). Working copy `rounds/018/op`, byte-identical at start
(`sha256` matches round 17's `source_sha256` 5f06189...). Bar re-measured this round.

Note on scoping: `rounds/018` already held a completed opt pass which the framework moved to
`1-opt.stale-20260924T085447` and whose findings are already merged into `pool.md`. I read it in
full first so this pass does not re-derive F1-F5 and does not re-kill `r18.i0.g53`. What follows is
new work on top of it.

---

## 1. Survey -- what I looked at, and why no new rocprofv3 run

Standing constraint h4 says rocprofv3 is a dead end and a deep round must never be scheduled; the
stale pass additionally recorded that `rocprofv3 --stats` *crashed*, left 2.5 GB of VRAM held and
needed a `kill -9`. On a box the memory already flags as degraded (VR-throttled, one hung MES), I
did not repeat it. The survey is therefore round 17's `1-profiling/kernel.yaml` (same
`source_sha256`, so it describes exactly this code) plus static analysis, which on this job has
been the instrument that actually moves candidates.

Expectations stated before measuring anything:
- `k_dkdv` is 62.93% of prod, `k_dq` 32.13%; both zero scratch. Any prod-moving change is in `k_dkdv`.
- I expected the `k_dkdv` body to be *issue* bound (F3's reading: we are at 94% of beat's issued
  rate, so the gap is the 7/5 structural factor). **This expectation turned out to be wrong.** See F3'.

## 2. New instrument -- `attrib_ss.py` (steady state), and a corrected WMMA cost

`rounds/018/_scratch/scr/attrib_ss.py` is a steady-state version of round 16's `attrib.py`: it runs
`ncopy=4` copies of the body carrying `vq`/`dq` and `clock` across the boundary, and reports, for
every popped op, how many iterations earlier it was issued.

Result: **every pop in both kernels is "0 iter(s) earlier."** The single-body attribution model was
already steady-state for these loops, so round 16's numbers stand unmodified. That is a negative
result and it is the useful kind -- it closes "maybe the model is wrong at the loop boundary."

Re-run at the physically correct WMMA cost (F1 pinned the rate at 2048 FLOP/SIMD-cycle, i.e.
`WCOST=8`, not the 1 that `clock.py` assumed):

| `k_dkdv` `.LBB0_8` (mask-free hot body) | cycles | share |
|---|---|---|
| WMMA issue (64 x 8)  | 512 | 30.4% |
| other issue          | 611 | 36.3% |
| **stall**            | **562** | **33.4%** |

and the stall is almost all one site:

- `@228 loadcnt N=35` popping a `buffer_load_b32` issued at slot 38 -- **470 cyc (83.6% of stall)**
- `ds_load_tr16_b128` -- 92 cyc

The 30.4% matrix-busy figure reproduces F1's independently-derived 29.5% by a completely different
route. That agreement is the strongest evidence the 2048 FLOP/SIMD-cycle rate has, and it is now
confirmed twice.

Raw: `raw/attrib_cur_dkdv.txt`, `raw/attrib_ss_cur_dkdv.txt`, `raw/attrib_ss_cur_dq.txt`.

## 3. F3' -- a correction to F3, and where the prod money actually is

`k_dq`'s hot body `.LBB0_2` at `WCOST=8`: 1413 cycles, 96 WMMA, **54.3% matrix busy, and zero
stall**. It is purely issue bound and fully latency-covered -- h16 confirmed.

Normalising by work (non-WMMA issue slots per S-element):

| kernel | non-WMMA issue / S-element |
|---|---|
| `k_dq`   | 0.315 |
| `k_dkdv` | 0.597 |

**1.9x.** If `k_dkdv` ran at `k_dq`'s per-element efficiency, prod would be ~7.29 ms ~= **754 TF/s**,
above the 709 bar -- without touching the 7/5 structural factor at all.

So F3's aggregate "we are at 94% of beat's issued rate, the gap is structural" **hides a 24-point
spread between our own two kernels**. The remaining prod money is inside `k_dkdv`'s body. This
reframes the route: fusion (7 -> 5 products) is not the only door, and it is the expensive one.

## 4. Two route rows killed on paper, at zero card cost

### `r18.i1.g54` -- fuse dQ into the KV-outer kernel (route row 1/3) -- DEAD
Priced by traffic, not by workspace shape. With KV outer and q inner, a dQ partial exists per
KV tile: 256 of them at prod, so a materialised workspace is ~137 GB -- unbuildable. Priced as
traffic instead: 16 B of dQ partial per S-element x 4.295e9 causal S-elements = 68.7 GB written +
68.7 GB read = **137 GB round trip = 62.6 ms at 4.39 TB/s**, against a **10.82 ms** total kernel
time and a **3.1 ms** saving from 7 -> 5 products. A ~20x loss. Grouping KV tiles to cut the copy
count requires holding the whole group's inner accumulator set, which does not fit (see g55 below);
the query-outer variant is symmetric and identical. Route row 1's own kill condition fires and
row 3 is cancelled. Consistent with h9 and with `r3.i6.g15`'s existing dead-end text.

### `r18.i2.g55` -- reopen the occupancy axis (route row 2) -- DEAD
Register census of `.LBB0_8`: **395 live-in VGPR** of 724 allocated. 384 of those 395 are the
256-VGPR dK/dV accumulator pair plus the 128-VGPR g21 prefetch tuple. Reaching <= 512 for 2
waves/SIMD needs >= 212 VGPR removed, which necessarily includes exactly the 128-VGPR non-rotating
prefetch tuple -- `r11.i5.g37`, i.e. the mechanism that earned g21's +8.3%. Even deleting it
entirely leaves 596 > 512, so `BLOCK_KV` would also have to halve to 16, undoing g06's 1.677x.
Two shipped wins would have to be surrendered to buy an occupancy step that is not priced. Dead.

## 5. Arm A as first formed -- "widen `k_dkdv`'s q step 32 -> 64" -- KILLED BEFORE BUILDING

The 1.9x gap in section 3 makes tile widening the obvious first reach: the 470-cycle b32 stall and
the 92-cycle dscnt stall look like *per-iteration fixed costs* that would halve per unit of work
(estimated ~+16% prod). The query axis also leaves the dK/dV accumulator floor untouched (it is
32kv x 128d either way), and the same axis paid +19.3% and +12.1% on `k_dq` (g10, g13).

I killed it on the dead-end adjacency instead of spending a slot on it:

- `r3.i5.g14` (KV 32 -> 64) built at 982 VGPR with **zero spill** and measured **2.76x slower**.
- `r10.i2.g28` (two-iteration ping-pong unroll) removed 17% of issue slots and measured **-7.8%
  prod in three sessions**, with the recorded mechanism *"it doubled the load batch covered by the
  drain and shortened useful load-to-use overlap."*

g28's mechanism applies **directly and exactly**. The 470-cycle site is `@228 loadcnt N=35` popping
a `buffer_load_b32` issued at slot 38: the b32 is stuck behind the b128 prefetch batch in the
in-order LOADcnt FIFO. Doubling the q step doubles that batch from 32 to 64 b128, so the b32's
drain prefix doubles too. **The 470 cycles are not a fixed per-iteration cost that amortises -- they
are a FIFO-ordering cost that scales with the batch.** My premise was falsified by an existing
measurement. Widening would make the dominant term worse, not cheaper.

Combined with g14, the tile-widening axis of `k_dkdv` is now closed by measurement on **both** of
its axes. No id was allocated; this is recorded here and as a note on g28 rather than as a new
dead end, because it is the same mechanism.

## 6. What that implies -- the mechanism the next arm must have

Every previous attack on this stall (g27, g48, g51, g53) tried to fix an in-order-FIFO problem by
*reordering within the same FIFO*. g51 bought +0.66%, the rest lost. The reframing: LSE and delta do
not need to be on LOADcnt at all. gfx1250 has six independent wait counters, and
`rocdl.global.load.async.to.lds.b32/b64/b128` (confirmed present in the FlyDSL rocdl inventory, and
documented there as "the operation introduced in gfx1250") retires on **ASYNCcnt**, a counter the
b128 Q/dO prefetch never touches. Staged through LDS, the consumer then waits on **DScnt**, which
the body already drains at slot 497 with ~100 cycles of cover to spare.

That removes the popped op from the FIFO rather than moving it inside it -- a different mechanism,
which is the bar `dead_ends.md` sets for reopening this area. Upper bound if the site goes to zero:
470/1685 = 27.9% of `k_dkdv`'s body, ~17.6% of prod.

## 7. Gate passed at zero card cost -- `asyncprobe` (r18.i3)

`_scratch/scr/asyncprobe.py`, COMPILE_ONLY, two kernels (wave-uniform LDS destination and an
explicitly lane-varying one). Raw: `raw/asyncprobe.txt`. Both build. Both produce:

```
global_load_async_to_lds_b32 v1, v[0:1], off
s_wait_asynccnt 0x0
ds_load_b32 v0, v2
s_wait_dscnt 0x0
```

| question | answer |
|---|---|
| lowers to a real instruction? | yes, `global_load_async_to_lds_b32` |
| which counter? | `s_wait_asynccnt` x1, **`s_wait_loadcnt` x0** |
| LDS destination uniform-only? | **no** -- the destination is a plain VGPR (`v1`/`v2`), no `m0` setup; the lane-varying kernel is byte-identical in structure |
| scratch | 0 |

So the destination is per-lane addressable and the op retires on a counter the b128 Q/dO prefetch
never touches. That is precisely the property section 6 asked for, confirmed rather than assumed.

First flydsl-level errors hit and fixed while writing the probe (recorded per "write as you go"):
- `fx.get_iter(smem.peek())` -> `ValueError: Operand 0 of operation "fly.get_iter" must be a Value`.
  LDS in this codebase is not read through a fly tensor; it is read with `llvm_dialect.load` on a
  `create_llvm_ptr(addr, address_space=3)`, the mirror of the `llvm_dialect.store` the kernel
  already uses. Fixed.
- there is no `rocdl.ds_load_b32` at the flydsl level; the `llvm_dialect.load` above is what the
  backend turns into `ds_load_b32`.
## 8. A cross-backend recipe found in-tree: aiter already does this on gfx1250

`aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py` stages Q/K/V with
`global_load_async_to_lds_b128` + `cluster_load_async_to_lds_b128` and drains them with explicit
`rocdl.s_wait_asynccnt(...)` countdowns. So the mechanism is not speculative on this part -- the
vendor's own gfx1250 attention *forward* kernel is built on it, and our backward kernel is the only
one of the two not using it. It also supplies the idioms: `fx.Int64(fx.ptrtoint(fx.get_iter(T)))`
for a global base, and an in-kernel `(seq < q_len).select(seq, 0)` clamp because an async load has
no buffer descriptor to clamp for it (h2).

This raises a second, larger candidate which I am NOT taking this round -- putting the 32 Q/dO
`buffer_load_b128` on the same path (`r18.i5.g58` in the pool). It rewrites the same `_ldqd` lines
as g56, so it cannot be measured in parallel with it.

## 9. Arm A -- `r18.i3.g56`, built

Edits, all in `_dkdv_impl`:
1. `smem` +512 B: a 2-slot ring of (32 lse + 32 delta) fp32 at `lds_ld`. Slot parity is
   `(qt*G + gh) % 2`, so producer `(qt_n, gh_n)` and consumer `(qt, gh)` always differ.
2. `_ldqd`: the 4 `_ldv` b32 become 2 `rocdl.global_load_async_to_lds_b32`. Lane `l` fetches row
   `q0 + l`, clamped in kernel to `Sq-1` (h2 -- a raw pointer has no descriptor).
3. the consumer's `lse_q = pre[hh*2][0]` becomes an `llvm_dialect.load` from the ring.
4. the carried tuple lost its 4 leading entries; `pre[4 + hh*16 ...]` -> `pre[hh*16 ...]`.

### Build failure #1, and the correctness bug it exposed
The first build screened clean (`spill 0`, all four kernels `pass`) with **6
`global_load_async_to_lds_b32`, 0 `buffer_load_b32`** -- and **0 `s_wait_asynccnt`**. The backend
does not derive the wait from the memory dependence; `inttoptr` LDS addresses are opaque to it. The
`ds_load` of the ring was free to race the async write. This would have been a silent wrong-answer
arm that the offline screen passes. The wait is now explicit, and its *count* is the whole point:

```python
rocdl.s_wait_asynccnt(2 if const_expr(carry) else 0)
```

In the full body `_ldqd` has just issued this iteration's 2 loads, so draining to 0 would be
`r10.i1.g27`'s exact failure -- a full drain of just-issued loads. In the masked body producer and
consumer are the same iteration, so it must be 0.

### Offline screen (h3), second build -- PASS
| | incumbent `k_dkdv_0` | g56 `k_dkdv_0` |
|---|---|---|
| `buffer_load_b32` | 4 | **0** |
| `global_load_async_to_lds_b32` | 0 | 6 |
| `s_wait_asynccnt` | 0 | 2 (`0x2` full body, `0x0` masked) |
| `ds_load_2addr_b32` | 0 | 4 (lse+delta in one instruction) |
| `.amdhsa_next_free_vgpr` | 724 | **718** |
| `vgpr_spill_count` | 0 | 0 |
| `group_segment_fixed_size` | 70656 | 71168 |

All four screens (`k_dkdv`, `k_dkdv_sp`, `k_redsp`, `k_dq_sp`+`k_redsp_q`) `verdict: pass`, spill 0.

### One thing the ISA shows that I do not like, recorded before measuring
`.LBB0_8` now opens with the 2 async loads at slots 49-51 and then an
**`s_wait_loadcnt 0x0` at slot 54**, *before* the body's first `buffer_load_b128` (slot ~93). The
allocator gave the async ops' address registers `v4`/`v5`, and the b128 batch later writes
`v[2:5]` -- so this is a WAR drain of the *previous* iteration's b128, newly introduced by register
reuse. It is a full LOADcnt drain at the top of the hot body. It may cost more than the 470 cycles
removed. If the card says g56 loses, that is the first thing to look at, and it is fixable by
keeping the async address out of the b128 batch's register range rather than by abandoning the
mechanism.

(The static attribution instrument cannot arbitrate this: it has no ASYNCcnt queue and files
`global_load_async_to_lds_b32` in the LOADcnt queue, which is why its `.LBB0_8` report blames 797
cycles on that op. That number is a model artifact and is not evidence either way. Fixing the
instrument is a fifth queue's worth of work and is not this round's job.)

## 10. Only one arm was built, and why

The brief asks for two arms measured apart. I have one. The honest accounting:
- the two route rows this round inherited (`g54`, `g55`) both died on paper (section 4);
- the arm I formed from the survey (widen the q step) died on the `g28` adjacency (section 5)
  -- correctly, before spending a slot;
- the two remaining pool-borne ideas, `r16.i3.g50` T2 and `r18.i5.g58` (Q/dO async), both rewrite
  the *same* `_ldqd` staging lines as g56, so neither can be built from the same base and merged
  with it. Building one as a second arm would produce two numbers I could not combine.
The candidates that *are* line-independent of g56 (`reuseA`/`reuseB` hints on the WMMA chain) are
unquantified and carry a "must be identical to the previous instruction" requirement that collides
with this job's standing lesson that source order is not issue order; pairing a green-gated arm with
a coin flip is not a better round than shipping the green-gated arm. Recorded as a shortfall against
the brief rather than papered over.
## 11. On-card measurement -- g56 LOSES on prod

Device idle before the run (`rocm-smi --showpids` -> `No KFD PIDs currently running`, `--showuse` 0%),
one benchmark process, palindromic slot rotation `cur_a g56_a g56_b cur_b cur_c g56_c` plus `beat`,
51 timed iterations, median, all in ONE session. sclk 1050-1056 MHz throughout (the box's VR cap).
Raw: `raw/meas_g56.txt`.

### prod (b4 s8192 hq32 hkv8 d128)
| arm | TFLOP/s | latency ms |
|---|---|---|
| cur_a | 505.35 | 10.880 |
| cur_b | 504.84 | 10.891 |
| cur_c | 504.30 | 10.903 |
| **cur median** | **504.84** | **10.891** |
| g56_a | 469.15 | 11.719 |
| g56_b | 471.32 | 11.666 |
| g56_c | 467.82 | 11.753 |
| **g56 median** | **469.15** | **11.719** |
| **beat (re-measured)** | **720.98** | **7.626** |

**g56 = -7.07% prod against the same-session incumbent.** The three-slot spread is 0.21% on `cur`
and 0.75% on `g56`, against this session's same-code floor of 0.23% -- the loss is ~30x the noise
and is not a slot artifact. The bar also moved: beat is 720.98 this round against 709.89 last
round, so the incumbent's ratio is **504.84 / 720.98 = 0.7002** (was 0.6964 on last round's bar).

### What lost
Exactly the thing section 9 flagged before measuring. The mechanism did what it was designed to do
-- `buffer_load_b32` is gone from `k_dkdv` and the 470-cycle `@228 loadcnt` site with it -- but
paid for it twice:
1. the **new `s_wait_loadcnt 0x0` at slot 54 of `.LBB0_8`**, a WAR full drain of the previous
   iteration's b128 batch, caused by the allocator putting the async address in `v4`/`v5` while the
   b128 batch writes `v[2:5]`. A full LOADcnt drain ahead of the body's first b128 is strictly
   worse than the partial `N=35` wait it replaced -- this is `r10.i1.g27`'s failure re-entering
   through the register allocator rather than through the source.
2. two new DScnt stalls on `ds_load_2addr_b32` (69 + 67 = 136 cycles) that did not exist before.

Removing 470 cycles of stall and taking back a full drain plus 136 cycles is a net loss, and the
card says so. **This does not falsify the mechanism** -- it falsifies this register placement of it.
The counter-argument is concrete and checkable offline: the ISA gate for the next attempt is that
`s_wait_loadcnt 0x0` must NOT appear at the top of `.LBB0_8`. That is route row 2.

### Ship decision
`cur` was itself an arm in this measurement and it won by 7.07%. Shipping g56 would regress the
job. `op/current` is unchanged; `rounds/018/1-opt/op` is the incumbent tree. The best-measuring arm
is shipped, and here that is the incumbent.

### fast (b1 s1024 hq8 hkv2 d128) -- same session, same rotation
| arm | TFLOP/s |
|---|---|
| cur_a / cur_b / cur_c | 55.94 / 55.07 / 55.55 -> **median 55.55** |
| g56_a / g56_b / g56_c | 47.91 / 53.79 / 51.30 -> **median 51.30** |
| beat | 51.56 |

g56 loses at fast too (-7.6%), though the g56 slot spread here is 12% -- fast single-arm reads
remain unusable on their own, and h7 (prod alone ranks a candidate) is what decides. Device idle
after the run (`No KFD PIDs currently running`), one benchmark process throughout, no dmesg GPU
faults during either shape.

## 12. `op/validation.py` -- g56 is INCORRECT. The -7.07% is void.

Run on an idle device after the benchmark, with the check `validation.py` performs and no other:

```
  correctness fast     dq  52.61 dB  dk -56.23 dB  dv -54.60 dB
  correctness proxy    dq  52.52 dB  dk -67.22 dB  dv -66.03 dB
  correctness prod     dq  52.56 dB  dk -73.14 dB  dv -72.07 dB
  correctness  FAIL / determinism FAIL / speed FAIL        RESULT: FAILED   (exit 2)
```

`dq` is fine at 52.6 dB on all three shapes -- `k_dq` was not touched. `dk`/`dv` are *negative*
dB, i.e. the error is larger than the signal: `k_dkdv` is not computing attention backward at all.

**So the -7.07% prod and -7.6% fast numbers above are not performance measurements of the g56
idea. They are timings of a broken kernel, and I am reporting them as such rather than as a verdict
on the mechanism.** Recorded in full above because the run happened and the brief says to report
what I measured; struck as evidence here.

Two things this round establishes about the mechanism, independent of the bug:
- it is buildable and lowers correctly (section 7), and
- **the offline screen cannot see this class of bug.** All four screens returned `verdict: pass`,
  `spill: 0`, on a kernel whose dk/dv are noise. h3 is necessary and nowhere near sufficient for an
  async-to-LDS change. That is a standing-constraint-level finding and it is the most useful thing
  round 18 produced about this candidate.

### What is already ruled out as the cause, and what is not
Checked in the shipped ISA of `.LBB0_8` (slot numbers relative to the label):
- **ordering is NOT the cause.** The two `global_load_async_to_lds_b32` are at slots 49/50, the
  `s_wait_asynccnt 0x2` is at slot 131, and the two `ds_load_2addr_b32` consumers are at 157 and
  307. The wait is after the issues and before the reads, which is what the count argument requires.
  The masked body carries `s_wait_asynccnt 0x0`. So this is not another "source order is not issue
  order" (g48/g51) failure.
- **the LDS offset is right.** `ds_load_2addr_b32 ... offset1:32` reads the two dwords 32 dwords =
  128 B apart, which is the lse/delta separation the producer writes.
- ring parity is right on paper: the consumer's index is `qt*G + gh = qt0*G + ii` and the producer
  is called with `ii+1`, so they always differ; the prologue `_ldqd(_qt0, 0)` writes exactly the
  slot the full loop's `ii = 0` reads.

Still open, in the order I would check them:
1. **`scale_offset` on the emitted instruction.** The ISA is
   `global_load_async_to_lds_b32 v4, v3, s[44:45] scale_offset`. If that modifier scales the
   voffset by the access size, and the source already multiplied the element index by 4, every
   lane's address is 4x too far apart -- which would produce exactly this signature (dq clean, dk/dv
   noise, no fault, because the buffer is large enough that the reads stay mapped). Cheapest to
   test: drop the `* 4` from `_off64` and re-validate. This is my leading hypothesis.
2. whether `s_wait_asynccnt N` counts *instructions* or *groups* (`rocdl.asyncmark` /
   `wait_asyncmark` exist precisely to group them, and aiter's FMHA uses a running
   `total - landed` count, not a fixed one). If it counts groups, `0x2` is not "2 loads".
3. the WAR interaction at slot 54: the async ops' LDS-address registers are `v4`/`v5` and the b128
   batch writes `v[2:5]`. That was diagnosed as a performance problem, but a *store* into the
   register holding an in-flight async op's LDS destination address is a correctness question too.

Each of these is answerable offline or with one validation run, and none of them requires giving up
the mechanism.

## 13. Round outcome

- **Nothing shipped.** `rounds/018/op` has been reverted to `op/current` byte-for-byte; the g56 tree
  is preserved at `_scratch/arms/g56_SAVED` for round 19 rather than deleted.
- The incumbent re-measured this round at **504.84 TF/s prod** against a re-measured bar of
  **720.98 TF/s** -> **0.7002**.
- Three candidates removed from the board at zero card cost (`g54`, `g55`, and the q-widen idea),
  one new mechanism proven to build and lower (`g56`), one new pool entry that the mechanism
  unlocked (`g58`), and one correction to a standing constraint (h3 does not cover async-to-LDS).

---

## 14. `r18.i3.g56` 的根因:per-lane LDS 目标被我当成了 uniform 的

上一节记录的是 build #2 上卡后 `op/validation.py` 判出 dk/dv **−56 / −73 dB**。
第 13 节列了三个待查假设,首要嫌疑是 ISA 上那个 `scale_offset` 修饰符。**三个都不是。**
根因读源码就能定死,不需要再上一次卡:

```
_sl = _ldslot(qt, gh)          # = lds_ld + ((qt*G + gh) % 2) * 256
```

`_ldslot` 只由 `lds_ld`、`qt`、`gh`、`G` 构成 —— **没有 `lane` 项,32 条 lane 拿到的是同一个
LDS 地址**。而 `global_load_async_to_lds_b32` 的 LDS 目标操作数是**逐 lane** 的:lane *l* 把
**自己**取回的那个 dword 写到**自己**那个 VGPR 里的地址。于是:

- 32 条 lane 撞在同一个 4 字节上,**只有一个值落地,另外 31 个丢失**;
- 消费端 `_cs = _ldslot(qt, gh) + (hh*16 + row) * 4` 在 `row ≥ 1` 时读的是**从未被写过的 LDS**。

这解释了观测的全部形状:`k_dq` 没动过 → dq 52.6 dB 正常;`k_dkdv` 的 LSE/delta 有 31/32 是垃圾
→ dk/dv 是噪声(**−56 / −73 dB**,而且 shape 越大越负,因为垃圾越多)。

⚠ **这是老式 `global_load_lds`(gfx90a/gfx940,`m0` + uniform LDS base,硬件按 lane 自动 ×4
步进)的肌肉记忆直接套到 gfx1250 的 async 版本上。** gfx1250 这条拿的是 per-lane VGPR 地址,
**没有隐式 lane 步进** —— 这正是它不需要 `m0`、也不要求 uniformity 的代价。
更难看的是:第 12 节我自己写的注释里就有 "per-lane VGPR LDS destination all confirmed by
`r18.i3.asyncprobe`"。**探针确认了这个能力,然后实现里没有用它。**

### 修复(build #3)

```python
_sl = _ldslot(qt, gh) + lane * fx.Int32(4)
```

几何核对:producer 写 lse 于 `base + lane*4`(0–124)、delta 于 `base + 128 + lane*4`
(128–252);consumer 读 lse 于 `base + (hh*16+row)*4`、delta 于同址 `+128`,`hh∈{0,1}`、
`row∈0..15` 覆盖 0..31。256 B 的 ring slot 正好装下,两区不重叠。

**这是同一机制的实现缺陷修复,不是换机制** —— 机制(把 LSE/delta 从 LOADcnt FIFO 移到 ASYNCcnt)
一字未动,build #2 已经证明它能正常 lower、零 spill。按 brief 的划分,这属于本轮的工作。

⚠ **构建前已清编译缓存**:`rm -rf /root/.flydsl/* ~/.cache/comgr/*`(253 MB + 77 MB),
并删掉 `rounds/018/op/__pycache__`。否则后端可能端出上一轮的二进制而无任何报错。

⚠ 顺带给下一轮留一条比这个候选更值钱的东西:**`h3` 的四个离线 screen 对这一类错完全失明**,
build #2 拿到四个 `verdict: pass` / `spill: 0`。离线 screen 检查的是分配与降级,
**不检查 lane→地址映射**。任何 per-lane 寻址的改动,第一道闸必须是 `op/validation.py`,
而且必须跑在 benchmark **之前** —— 这已经写进 `route.md` 第 1 行(`h1`)的 `condition`。

---

## 15. build #3:正确性通过,三道闸全过,**实测仍然输**

### 15.1 正确性(`op/validation.py`,`raw/val_g56_build3.txt`)

| shape | dq | dk | dv |
|---|---|---|---|
| fast | 52.61 dB | **52.65 dB** | **52.83 dB** |
| proxy | 52.52 dB | **52.57 dB** | **52.67 dB** |
| prod | 52.56 dB | **52.60 dB** | **52.71 dB** |

`correctness pass` / `determinism pass`(fast dq/dk/dv ×200 逐位相同)/ `speed FAIL`。
**speed FAIL 是本 job 十八轮的常态门**(geomean ≥ 1.00x beat,从未达到过),不是回归。
`exit_code 2`。对照 build #2 的 dk **−56.23 / −67.22 / −73.14 dB** —— per-lane 修复把
dk/dv 从噪声变成了与 dq 同级的 52.6 dB。**根因判对了。**

### 15.2 三道闸(`raw/isa_g56_build3_k_dkdv.s`,零卡时)

| 闸 | 要求 | incumbent | g56 build#3 | 判 |
|---|---|---|---|---|
| ② | `.LBB0_8` 顶部不得新增 `s_wait_loadcnt 0x0` | 2 条 | **2 条** | **PASS** |
| ③ | `s_wait_asynccnt` 计数 = 全体 `2` / masked 体 `0` | — | `0x2` @1497、`0x0` @588 | **PASS** |
| — | spill | 0 | **0** | PASS |

build #2 死在闸②(寄存器分配把两条 async 地址放进 `v4`/`v5`,后续 b128 批次写 `v[2:5]`,
凭空长出一条全排空)。build #3 没有这个问题:两条 async 在 `.LBB0_8` 的第 48/49 条,
第一条 `s_wait_loadcnt 0x0` 在第 52 条,**在它们之后**。

### 15.3 静态指标全面变好

| `.LBB0_8` | incumbent | g56 build#3 | Δ |
|---|---|---|---|
| body 指令数 | 678 | **667** | −11 |
| `s_wait_loadcnt` 总数 | 7 | **3** | **−4** |
| `s_wait_loadcnt 0x0` | 2 | 2 | 0 |
| `buffer_load_b32` | 4 | **0** | **−4(目标站点的源指令整个消失)** |
| `buffer_load_b128` | 32 | 32 | 0 |
| `v_wmma` | 64 | 64 | 0 |

**占 `.LBB0_8` 全部 stall 83.6% 的那个站点(`@228 loadcnt N=35`)所依赖的 4 条
`buffer_load_b32` 已经一条不剩。**

### 15.4 实测(6 槽回文,51 iters,中位数,同 session,设备前后空闲,sclk 1051–1100 MHz)

| shape | incumbent(`rounds/017/op`) | g56 build#3 | `beat` | g56/incumbent | g56/beat |
|---|---|---|---|---|---|
| prod | **504.03** TF/s(槽间极差 0.36%) | **469.56**(0.48%) | **719.49** | **0.9316(−6.84%)** | 0.6526 |
| proxy | **441.98**(0.10%) | **404.85**(1.97%) | **580.97** | 0.9160(−8.40%) | 0.6969 |
| fast | **55.27**(1.12%) | **52.22**(13.98%) | **51.38** | 0.9447(−5.53%) | 1.0163 |

`score`(margin 1.00 ⇒ target = 本轮实测 `beat`;`ratio = min(this/target, 1)`):
**0.7832**,incumbent 同口径 **0.8204**。**三个 shape 同号,远超本 session 0.23–0.36% 的同码地板。未被接受。**
⚠ fast 的 g56 三槽极差 **13.98%**(47.42 / 54.05 / 52.22),`h6` 的「fast 单臂读数不可用」
再次被印证;fast 不参与排名(`h7`)。

### 15.5 这是本 job 第三次「静态全好、实测大输」

`r15.i2.g47`、`r16.i2.g49`,现在是 `r18.i3.g56`。前两次的机制都是
**编译器重排被破坏**;这一次不是 —— 调度没被约束,指令更少,等待更少。

**领先假说(标注为假说,本轮没有仪器能判它):这条路给每次迭代 16 字节的数据凭空加了一趟 LDS 往返。**
incumbent 的 LSE/delta 是 HBM → VGPR(`g51` 提前一次迭代预取,数据落在 carried tuple 里,
body 里读它是**寄存器读,零延迟**)。g56 改成 HBM → LDS → VGPR:body 里每个 `hh` 都要
`ds_load` 一次,而 LDS 正是这个 kernel 已经最紧的资源(每 workgroup 70656 B、`s_wait_dscnt` 15 条、
`ds_load_tr16_b128` 是 DScnt 上的主角)。**把等待从 LOADcnt 挪到 ASYNCcnt,代价是把它挪进了 DScnt。**

⚠ **判这个假说需要 `route.md` 第 5 行的那件仪器(ASYNCcnt 队列进归因工具),本轮没有。**
现有 `attrib_ss.py` 会把 `global_load_async_to_lds_*` 记进 LOADcnt 队列并报出假象数字,
**因此本节不给任何静态 cycle 归因** —— 那正是第 5 行存在的理由。

### 15.6 一次 GPU page fault,归因未定,**不算在候选头上**

ISA dump 用的是 round 2 的 `screen.py`(十六轮前的老 harness)。它**打完 ISA、打完
`verdict: pass` / `spill: 0` 的完整 JSON 之后**,在 `dmesg` 上留下:

```
amdgpu 0001:01:00.0: [gfxhub0] no-retry page fault (src_id:0 ring:24 vmid:3 pasid:166)
  Faulty UTCL2 client ID: TCP (0x8)   PERMISSION_FAULTS: 0x3   RW: 0x0
amdgpu 0001:01:00.0: IH ring buffer overflow
```

**反向证据:** 同一份二进制此前已经跑完 `validation.py` 的三 shape 正确性 + **×200 确定性**,
以及 3 shape × 3 槽 × 51 iters 的完整测量,**全程零 fault**。因此倾向老 harness 的收尾路径,
而非 kernel 越界(`_qg` 已按 `h2` 在 kernel 内 clamp 到 `Sq-1`)。`h19` 记录的设施危险也在这一带。
**归因未定,如实记录,不得当作 arm 的缺陷。** 处置:显式 `pkill -9 -f screen.py`;之后
`rocm-smi` 正常响应、GPU% 回到 0%、`No KFD PIDs`、无后续 amdgpu 报文。卡未 wedge,未需断电。

### 15.7 出货决定

**`rounds/018/op` 留下的是 build #3(g56),不回滚。** 它输了 6.84%,而这是一个结果:
机制被证明可降级、零 spill、**正确**、闸全过,然后**依然输**——
这比「它编不过」对下一轮值钱得多。`op/current` 不动,晋升由 Python 从我报的数字决定。
