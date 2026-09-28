# Round 24 — attention (backward), flydsl, gfx1250 — fast round

Working copy: `rounds/024/op`  ·  scratch: `/tmp/op-evolve-gfx1250-flydsl-attn-bwd-20260917-115934-r024`
Runner: `docker exec fa-g1` (pre-existing, not owned). Physical GPU **1** per h62.

---

## 0. What changed under this job between round 23 and round 24

**The job moved hosts.** h62 (2026-09-27, standing must): the job now lives on B0 host
`ctheliosp-1b112-a37-1`, a 4-card gfx1250 node, and this job's card is **physical GPU 1**
(rocm-smi GPU[1], PCI `0002:04:00.0`, `/sys/class/drm/card8`, kfd `gpu_id 30548`).
Container `fa-g1` exposes only that card. Never `fa-repro`, never `fa-g0/g2/g3`, never
python on the host, never `docker run`. Do not set `HIP_VISIBLE_DEVICES` /
`ROCR_VISIBLE_DEVICES`.

The consequence that matters for this round's arithmetic: **B0 absolute numbers are not
comparable with the A0 history.** On B0 the same prod shape reads ASM 6.50 ms and
champion r20 8.72 ms (ratio 0.746), where the A0 ledger has prod champion 516.87 TF/s.
So every number below is a **same-process ratio**, and the champion is re-measured in the
same session as every arm — which is what h6 has required all along, now for a second
reason.

Idle check caveat, learned the hard way at the start of this round: inside `fa-g1`,
`rocm-smi -d 1 --showuse` answers `WARNING: No such device card1`, because the container
enumerates its single card as GPU[0]. Identity was instead confirmed two ways:
`rocm-smi --showid` inside `fa-g1` reports `GUID: 30548`, which is h62's physical-GPU-1
id; and occupancy was read from the host via `/sys/class/kfd/kfd/proc/*/queues/*/gpuid`.
At the start of this round the only KFD process on the node was pid 391560 on gpuid
34992 — a different card, the forward job — so **our card 30548 was idle**. A stale
`rocm-smi --showpids` row (pid 385769, GPU 0, VRAM 0) inside the container has no
matching entry under `/sys/class/kfd/kfd/proc`; the sysfs listing is treated as
authoritative.

**The work was divided.** h63 (must note): from round 24 this job owns X1/X2/X5 (h59),
h60, and the h33 defects. FUSED5 (h58) and the atomic probe (h57) run in an operator lab
on another card and are explicitly **not** this job's work. `min_gain` is now **0.007**.
h63 also fixes the order: (1) the h33 defects, (2) X1 with a compile-only proof first,
(3) X2 or X5 per X1's verdict, (4) h60.

---

## 1. Look — what was read, and what it closed

`rocprofv3 --stats` is the survey this round's brief suggests. It is **banned on this op**
by standing constraint h4: it faults the GPU here, and PC sampling has caused three faults
and one reboot. So "look" is again an offline ISA census plus targeted card A/B, the same
instrument round 23 used.

Read this round: `findings/facts.md`, `findings/pool.md`, `findings/route.md`,
`job_context/hint.md` (h57–h63, all new since round 23), round 23's `opt.md`,
`Primus-Turbo/.claude/skills/gfx1250-attn-campaign/references/bwd-history.md` §8 (the
h33 defects), `Primus-Turbo/output/0927__flydsl/bwd/BARRIER-CENSUS.md`, and
`job_context/op/ut/common.py`. A sub-agent swept the knowledge corpus in parallel
(§1.3 below).

### 1.1 State of the model coming in

`facts.md`: bottleneck is **bound latency, confidence MEDIUM**. Round 22 falsified round
21's per-kernel-signed-lever model. The power paragraph is RETRACTED — the 60-sample
trace behind it ran while nothing executed; the honest 264-sample trace shows 998–1029
MHz under load against 1100 idle, about 9% droop, which is not a lever. Best-ever per
shape (A0 units): prod r020 516.87, proxy r019 443.73, fast r019 55.32. *This operator
has not moved outside the noise since round 17.*

The prefetch family is closed on every axis this job can reach: depth (r23), placement
(r16), drain position (r23 rule 4). Round 23's own contribution was the mechanism behind
the wall: **LOADcnt is 6 bits, max 63**. `_ldqd` issues 36 loads, so depth 2 puts 72
loads in flight, `SIInsertWaitcnts` cannot encode the score range, and it emits a full
`s_wait_loadcnt 0x0`. That is not a scheduling choice to be tuned; it is an encoding
limit.

### 1.2 The one live pool entry

`pool.md` had exactly one OPEN entry, `r23.i2.g72` (it keeps its id — a pool candidate
never gets renumbered), and it is the same lever h60 writes up from the champion side:

> Champion `.LBB0_8` rel 3–115 contains 67 `v_mov` rotating `pre1 → pre0`; at rel 12
> they trigger an `s_wait_loadcnt 0x0` that retires 33 loads with cover 603–728, about
> 0.8 of an iteration. **Unroll `qloop_full` by 2 so pre0/pre1 swap by register
> renaming**, and both the rotation copies and the drain they force should disappear.

Both sources attach the same free gate, and `pool.md` is explicit that it is **all three
conditions, no partial credit, before any card time**:

1. `v_mov_b64` per unrolled iteration falls from 128 toward 0;
2. `buffer_load` stays **36 per iteration** (72 for the doubled body) — if it doubles
   per drain the edit has become `g28` and must not be built;
3. the last-prefetch-load-to-consumer distance must not shorten (round 22's cover rule 1,
   6-for-6).

`pool.md` closes with: *"⚠ If the gate fails, the pool is empty and round 24 must source
from the corpus and from the same operator in other backends."*

### 1.3 Corpus sweep

Swept `knowledge/INDEX.md`, `optimization/routes/`, `optimization/techniques/`,
`backends/flydsl/attention/{README,techniques,dead-ends}`, the flydsl hd64/hd128 recipes,
the aiter `fmha_v3_bwd_hd128_bf16` recipe, the hipkittens GQA d128/d64 recipes and
`scheduling-patterns.md`, and `arch/gfx1250/{isa,gfx1250,profiling-surface}.md`.

Mechanisms ranked highest that are **not** already on this job's closed list:

| # | mechanism | corpus price | status here |
|---|---|---|---|
| 1 | **Coarsen hazard anchors** — one `min(acc,0)` anchor per 4-element vector instead of per element | `v_min` 72 → 3, hot loop 663 → 596 issue slots, **+1.76%, bit-identical** | untested here; needs an opcode census of `.LBB0_8` to see whether the pattern exists |
| 2 | **`sched_barrier(0)` + `s_nop 1` between asm-emitted bf16 packs and matrix ops** — FlyDSL emits `rocdl.cvt_pk_bf16_f32` as an asm block, so the hazard recogniser misses the VALU-write→matrix-read hazard | "one line and free", *faster* than interleaving; allocation 168 → 150 | distinct from the closed prefetch-`sched_barrier` family |
| 5 | **Move softmax/rescale VALU off the loop-carried dependence** | statement reorder alone **+0.81%, bit-identical**; 0.46 cycles of wall per cycle of VALU removed | untested here |
| 6 | Re-derive drains for gfx1250's six separate counters; delete duplicate waits | +2.2%; split rendezvous +0.7 geomean pts | adjacent to round 23's closed drain axis, but "duplicate wait" ≠ "wait position" |
| 8 | Single-instruction bf16 pack `v_cvt_pk_bf16_f32` (rtz vs rtne) | 18.1% fewer static instructions where live | needs census |
| 12 | Stage the epilogue through LDS aliasing the finished operand ring | +5.43% of a +7.32% step | epilogue, not the hot loop |

Already closed here and correctly so: OPSEL[2] operand reuse (`r19.i1.g59`, dead), causal
K-tile pairing from both ends (`r1.i3.g03`, closed at r13), `lock_simd` ("worth nothing at
one wave per SIMD" — matches h12).

The corpus also supplies a **direct counterexample to naive issue-slot counting**, which
is worth recording because items 1 and 5 above are both issue-slot arguments: a compiler
scheduling-strategy override "removed 40 `s_waitcnt`, added 16 `s_nop`, came out ahead on
issue slots and lost on time" (`max-memory-clause` 863.0 vs post-misched 918.9). Fewer
slots is a hypothesis, not a result. The corpus's own resolution floor is 1.57%, well
above this job's measured 0.24% same-code floor, so corpus percentages transfer as
*direction*, not magnitude.

---

## 2. Decision — the two arms

The brief wants two ideas built independently from `op/current`. h63 wants the defects
first, then X1, then h60. These reconcile cleanly:

- **Arm A = X1** (h59/h63 item 2): the preregistered 4-wave diagnostic — `w4` minus
  barrier-1. X1 can never ship: h51's ceiling says the 4-wave build tops out at 0.950x of
  the champion even with free barriers. Its entire value is deciding **H-align vs
  H-count** for the whole 4-wave/5-GEMM route: h59 predicts ≤ +8% under H-align and
  ≥ +15% under H-count (342 → ≥ 395 TF/s).
- **Arm B = `r23.i2.g72` / h60**: the by-2 unroll of `qloop_full`. The only live
  *shipping* candidate in the job.
- **The h33 defects are not an arm.** They are correctness work that rides with whatever
  ships, and they must be priced separately because defect c puts clamps inside the hot
  loop.

Each is built alone from `op/current`; B is not built on top of A (they do not even share
a base file — X1 is a census tree).

### 2.1 Trees standing

| tree | contents |
|---|---|
| `cur_a`, `cur_b` | pristine `op/current`, md5 `51c4bf6077e191b3da3697a92e464b79`. Two physical incumbent slots, per the round-23 palindromic method. |
| `D` | `cur` + h33 defects a/b/c |
| `U` | `D` + the g72/h60 unroll |
| `w4` | verbatim from the 2026-09-27 census trees (4-wave baseline) |
| `x1` | census `w4` with exactly one line changed: `fx.barrier()` → `pass` |

Verified before use: `census/trees/champ/kernels.py` is byte-identical to
`op/current/kernels.py`, and `x1` differs from `w4` by that one line. The census trees are
therefore trustworthy as arms.

### 2.2 The h33 defects as applied in `D`

From `bwd-history.md` §8:

| # | defect | site | fix | what it costs today |
|---|---|---|---|---|
| a | asserts only `sq % 32`, but the kernel uses `ceil(Sq/BLOCK_Q)` while the launcher passes `sq // BLOCK_Q` | `impl.py:115` | `assert sq % _k.BLOCK_Q == 0` | at `sq % 64 == 32`: 262,144 B OOB **writes**, dQ tile 0 never computed |
| b | int32 overflow of the dQ workspace extent `nsp*B_*Sq*Hq*(D*4)` | `kernels.py:724` | compute in Int64 | `nsp_q ≥ 8` at prod → `num_records = 0`, every dQ write silently dropped |
| c | unclamped `k_dkdv` prologue prefetch and carried `kk = ii + 2` | `kernels.py:573-574, :583-584, :529` | clamp to `nqt2-1`, the `k_dq` idiom at `:898-899` | live OOB **read**, about 261 KB at prod |

Defect a's stricter assert was checked against every UT shape in `op/ut/common.py`: all
`sq` values (1024, 4096, 8192, 128, 128, 256, 512, 1024, 1024) are multiples of 64, so it
rejects nothing that used to pass. Defect a is expected to be ISA byte-identical — it is
a host-side assert.

Also carried from §8 as prohibitions, not edits: never convert `k_dkdv` descriptors to the
fake `1<<30` style, and never naively "fix" `k_dq`'s seven fake descriptors.

### 2.3 The unroll in `U`, and the one design problem it had

The unroll wants an even trip count. `n` is `G * (nqp_eff - nmaskp)` and **`G = Hq // Hkv`
is a runtime value** — the `mha` UT shape has G=1 — so `n` can be odd. Seven ways out were
weighed (predicate the second sub-body; `select` across 256 accumulator elements; a
separate tail loop; an `scf.if` holding two differently-shaped loops; peel the first
iteration; …). Most add either spill risk or a second loop body.

Taken instead: the option **h60 itself endorses** — *"the remainder must go through the
masked/predicated iteration, never a tail loop (spill risk)."* One odd trip is pushed into
`qloop_mask`, which computes an unmasked tile bit-identically (its predicate is trivially
true there) and carries no prefetch. No value, no order and no carried-shape changes. This
needed one new argument, a flat start index `i0`, on `qloop_full`.

`qloop_full` keeps its original rotating form under `if const_expr(PARTIAL):` for the split
kernel, and gets the doubled body only on the non-PARTIAL path. Each unrolled body runs
sub-iteration A against prefetch slot 0 (refilling slot 0 with `ii+2`) then sub-iteration B
against slot 1 (refilling slot 1 with `ii+3`), so the two slots alternate by **register
renaming** rather than by 128 `v_mov_b64`.

**Open, unverified as of this point:** defect b computes the extent in Int64 via
`.to(fx.Int64)` and then `_bv` calls `fx.Int64(nb)` on an already-Int64 value.
`numeric.py:408-410` shows `to()` returns `self` when the dtype already matches, so it is
probably idempotent — but that is a reading, not a compile. The screen below settles it.

---

## 3. Build log

### 3.1 Scratch had to move (B0 has no `/tmp` mount)

First attempt at the compile screen died with `bash: .../run_screen.sh: No such file or
directory`, rc 127. `docker inspect fa-g1` shows exactly one mount, `/home/lihuzhan ->
/home/lihuzhan`. On B0 the container does **not** share the host `/tmp`, so the brief's
scratch path is invisible to the runner. Resolved without leaving the runner: the real
scratch directory is `/home/lihuzhan/scratch/op-evolve-…-r024` and the brief's
`/tmp/op-evolve-…-r024` is a symlink to it, so host-side paths keep working and every
build and measurement still goes through `docker exec fa-g1`. Recorded because it will
bite round 25 too.

### 3.2 Compile-only screen

`COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250`, prod shape, run inside `fa-g1`.
COMPILE_ONLY returns before the execution engine is built, so nothing touches the card.

| tree | `k_dkdv` VGPR | `k_dkdv` instr | `k_dq` VGPR/instr | spill | verdict |
|---|---|---|---|---|---|
| `cur_a` | 904 | 3168 | 960 / 3567 | 0 | pass |
| `D` | 904 | 3172 | 960 / 3567 | 0 | pass |
| `U` | **890** | 3933 | 960 / 3567 | 0 | pass |
| `w4` (census) | 882 | 3277 | 960 / 3567 | 0 | pass |
| `x1` (census) | **938** | 3249 | 960 / 3567 | 0 | pass |

`cur_a` reproduces the 2026-09-27 census record `j0341_champ.json` **exactly** (904/3168,
960/3567, 40/256), which is what licenses reusing the census's own `w4`/`x1` records
rather than recompiling them. `x1`'s 938 VGPR / 0 scratch is exactly what h59
preregistered. **No arm spills.** Spill is the one hard kill here — a spilling gfx1250
build does not run slow, it hangs the machine — and none of these do.

`D` compiles, which settles §2.3's open question: `_bv`'s `fx.Int64(nb)` applied to an
already-Int64 expression is idempotent in practice, not merely by reading
`numeric.py:408-410`.

### 3.3 The h33 defects are ISA-neutral apart from the clamps

`D` vs `cur_a`, `k_dkdv` hot loop `.LBB0_8`, and the prologue loop `.LBB0_6`:

| | `cur_a` | `D` |
|---|---|---|
| `.LBB0_8` instr | 778 | 780 (+2) |
| `.LBB0_8` WMMA / `v_mov_b64` / `buffer_load` | 64 / 128 / 36 | 64 / 128 / 36 (identical) |
| `.LBB0_8` load index range | 58..187 | 61..189 |
| `.LBB0_6` (prologue prefetch) instr | 227 | 229 (+2) |
| `k_dkdv` VGPR | 904 | 904 |
| `k_dq` | 3567 instr | 3567 instr, byte-for-byte unchanged |

Four extra instructions in the whole kernel, all of them the defect-c clamps, and nothing
else moves: same WMMA count, same rotation-copy count, same 36 loads, same register
budget. Defect a is host-side and defect b changes a descriptor extent computed in
scalar setup. h63's compile-only proof for the defects — *0 spill, ISA of the hot loops
unchanged apart from clamps* — **is satisfied.**

One incidental shift worth noting rather than hiding: the forced `s_wait_loadcnt 0x0`
moves from body index 12 to index 6. That is the drain changing *position*, and round 23
rule 4 already established the drain's position is not the binding quantity, so this is
expected to be free. It is priced on the card below anyway, which is the point of
measuring `D` separately from `cur`.

### 3.4 The `r23.i2.g72` / h60 gate — all three conditions

`U`'s `.LBB0_8` is the doubled body: **1450 instructions, 128 WMMA** (two iterations'
worth), so every per-iteration figure below is the doubled figure halved.

| condition | champion | `U` | verdict |
|---|---|---|---|
| **1.** `v_mov_b64` per iteration falls from 128 toward 0 | 128 (256 per two iterations) | **76 per two iterations = 38 per iteration**, a 70% cut | **pass** |
| **2.** `buffer_load` stays 36 per iteration (72 for the doubled body) | 36 | **72 for the doubled body = 36 per iteration** | **pass** |
| **3.** last-prefetch-load-to-consumer distance must not shorten | see below | see below | **pass** |

Condition 3 needs care, because two different numbers can both be called "the cover" and
they disagree. Register-level liveness cannot follow a value across the champion's
rotation copy — after the `v_mov_b64` the data lives in a different physical register —
so a naive intra-body scan returns a meaningless 90 for `cur_a`. The number that governs
is the one h60 measured and that the drain evidence corroborates: *the champion's
prefetch is issued for tile `ii+2` but is forced to complete one iteration early*,
because the rotation copies at the top of the next body read it and trigger
`s_wait_loadcnt 0x0` at rel 12 — **effective cover 603–728, about 0.8 of an iteration**,
not the two iterations the depth-2 scheme nominally buys.

In `U` there are no rotation copies for these values. The slot-0 refill issues at body
indices 601–735 and is consumed by sub-iteration A of the *next* body, whose WMMA run
starts at index 50: **cover ≈ (1450 − 735) + 50 = 765 instructions**. That is longer than
the champion's effective 603–728, not shorter. Condition 3 passes on the only reading the
measurement supports.

Supporting evidence, and this is the part that makes the gate convincing rather than
arithmetic — the wait structure inverted:

| | `cur_a` `.LBB0_8` | `U` `.LBB0_8` |
|---|---|---|
| waits | 2 | 19 |
| first wait | rel 2, `0x21` | rel 49, **`0x3e`** (62 loads still outstanding) |
| full `s_wait_loadcnt 0x0` | **rel 12 of 778 = 1.5% into the body** | **rel 1406 of 1450 = 97% into the body** |

The forced early drain h60 names is **gone**, and the waits became graded — the same
shape round 23 saw on `B71`, but `B71` had to give up depth 2 to get it and `U` keeps it.
Issue slots also fall: 1450 for two iterations against 2 × 778 = 1556, a 6.8% reduction.

**All three gate conditions pass and h60's screen criteria (0 spill, drain gone at rel 12,
`v_mov` count down) are all met. Card time for `U` is authorised.** Cost so far: zero card
seconds.

### 3.5 Two corpus mechanisms priced to dead, free, from the same census

The `.LBB0_8` opcode histogram answers two of §1.3's candidates without a measurement:

```
 161 s_set_vgpr_msb     64 v_pk_add_f32      32 v_exp_f32
 128 v_mov_b64_e32      64 v_pk_mul_f32      32 v_cvt_pk_bf16_f32
  64 v_wmma_..._bf16    45 v_nop             15 v_or_b32
                        40 ds_store_b128     32 buffer_load_b128
                        40 ds_load_tr16_b128  4 buffer_load_b32
```

- **Corpus item 1, coarsen hazard anchors (+1.76% elsewhere): not applicable.** There is
  no `v_min`/`v_max` anywhere in this hot loop — the mechanism needs per-element
  `min(acc,0)` anchors to coarsen and this kernel has none. Dead here, no card time.
- **Corpus item 8, single-instruction bf16 pack (18.1% fewer static instructions where
  live): already applied.** FlyDSL already emits `v_cvt_pk_bf16_f32`, 32 of them, one per
  pair. Nothing to win.

And one thing the census turned up that is **not** in this job's ledger at all:
**`s_set_vgpr_msb` 161 + `v_nop` 45 = 206 of 778 instructions, 26.5% of the hot loop**, is
neither arithmetic nor memory — it is the gfx1250 prefix for addressing VGPRs above 255,
plus hazard padding. h14 retired *register pressure* as a lever, but h14 was about
occupancy; this is the per-instruction **issue cost of living at 904 VGPR**, a different
quantity that no round has priced. Recorded to the pool rather than chased this round.

---

## 4. Card session — physical GPU 1, `docker exec fa-g1`

### 4.0 Idle check

Host `/sys/class/kfd/kfd/proc/` was empty — no compute queues on any card on the node —
and `rocm-smi` reported `No KFD PIDs currently running`, 0% use. Card identity confirmed
from inside `fa-g1`: `GUID: 30548`, h62's physical GPU 1. `dmesg` confirms the host is
`ctheliosp-1b112-a37-1`, h62's B0 node. A `dmesg -W` monitor for `amdgpu`/`kfd`
faults/resets/hangs ran for the whole session and fired nothing.

### 4.1 Correctness first (h1, seventh consecutive round)

`U` — the unroll, which is the arm that could ship:

```
correctness fast     dq 52.61 dB  dk 52.64 dB  dv 52.84 dB
correctness proxy    dq 52.52 dB  dk 52.57 dB  dv 52.67 dB
correctness prod     dq 52.56 dB  dk 52.60 dB  dv 52.71 dB
determinism fast: dk/dv bitwise identical across 200 runs; dq bitwise (floor 70 dB)
correctness  pass      determinism  pass
```

Those dB figures are the champion's own, to the hundredth (round 23 logged 52.61 / 52.64 /
52.84 at fast for `cur`). The unroll is numerically a no-op, which is what it should be —
it reorders nothing inside a tile, it only changes which register a prefetched tile lands
in. The parity design also holds up: pushing one odd trip into `qloop_mask` computes that
tile under a predicate that is trivially true there, and the `mha` shape (G=1, the one
that can produce an odd trip count) passes.

`validation.py` reports `speed FAIL`, `RESULT: FAILED`, geomean **0.966x beat**. That is
not a finding about `U`. The gate is against `op/beat/` — aiter's prebuilt gfx1250 ASM —
and *the champion fails it too*; h62 records the B0 champion/ASM prod ratio as 0.746. The
question this round can answer is `U` against the champion, in the same process, which is
§4.2.

### 4.2 The box is no longer at 1100 MHz

Every row this session carries an sclk witness, and they read **1798–1891 MHz at prod,
1932–1963 at proxy, ~2360 at fast**. The standing note that this box is VR-throttled to
1100 MHz, and the ~9% droop measured for `facts.md`, describe a machine that is not the
one running today. Consequences, in order of how much they matter:

1. The A0 ledger's absolute TF/s are now doubly incomparable — different host *and*
   different clock. `U` reads 612 TF/s at prod against the A0 champion's best-ever 516.87,
   and that difference is the clock, not the code.
2. h62's own B0 reference numbers (ASM 6.50 ms, champion 8.72 ms at prod) were taken at
   some unrecorded clock, so even they cannot be differenced against today's `U` at
   8.979 ms. `U` looking 3% slow against 8.72 is **not** evidence.
3. Only same-process ratios survive. This is exactly what h6 has required for eight
   rounds; the clock change just removes the last temptation to shortcut it.

`D` — the defect fixes alone:

```
correctness fast     dq 52.61 dB  dk 52.64 dB  dv 52.84 dB
correctness proxy    dq 52.52 dB  dk 52.57 dB  dv 52.67 dB
correctness prod     dq 52.56 dB  dk 52.60 dB  dv 52.71 dB
determinism fast: dk/dv bitwise identical across 200 runs; dq bitwise
correctness  pass      determinism  pass
```

Identical to `U` and to the champion. Expected: the clamps only bound prefetch reads whose
results are discarded, so at every shape where the unclamped read happened to land in
mapped memory the arithmetic is unchanged. What the clamps buy is that it no longer
*depends* on landing in mapped memory — defect c is a live ~261 KB out-of-bounds read at
prod today.

Two `validation.py` runs, two different processes, both against `op/beat`:

| | fast ms | proxy ms | prod ms | geomean x beat |
|---|---|---|---|---|
| `U` | 0.0671 | 0.5929 | 8.9792 | 0.966 |
| `D` | 0.0583 | 0.5939 | 8.6008 | 0.987 |

**These two rows must not be differenced.** They are separate processes at separate
clocks, and the fast column moves 13% between them, which is fifty times this job's
0.24% same-code floor. They are recorded because they are what was run, not because they
decide anything. `D`'s 0.748 prod ratio against the beat does match h62's champion figure
of 0.746, which is a weak consistency check that `D` has not broken anything.

### 4.3 The measurement that decides: one process, four arms, palindromic

`benchmark.py --arms beat --arm-path cur_a=… --arm-path D=… --arm-path U=… --arm-path
cur_b=…`, one invocation per shape, 51 iterations, median, palindromic order, two
physically separate incumbent slots. The spread between `cur_a` and `cur_b` — the same
code twice — is this session's own floor, and nothing narrower than it may be read.

| shape | sclk | `cur_a` ms | `cur_b` ms | **floor** | `D` | `U` | beat |
|---|---|---|---|---|---|---|---|
| fast | 2360→2348 | 0.05568 | 0.06085 | **8.87%** | +0.80% | +3.97% | −44.1% |
| proxy | 1935→1983 | 0.59209 | 0.59257 | **0.08%** | −0.42% | +0.30% | +31.9% |
| prod | 1793→1849 | 8.58453 | 8.59823 | **0.16%** | **+0.14%** | **−4.52%** | +33.5% |

(percentages are against the mean of the two incumbent slots; positive is faster)

**fast is uninterpretable this session.** The same code measured twice differs by 8.87%,
which is 37x the floor the other two shapes produced and 37x this job's best-ever 0.24%.
The two `min_ms` are 0.05416 and 0.05392 — within 0.4% — so the medians are being set by
the tail, not by the kernels: 51 iterations of a 55 µs kernel is not enough samples to
place a median on this machine. No claim below rests on a fast number, and the fast
column is reported only because reporting every arm is the rule.

**proxy and prod produced honest floors** (0.08% and 0.16%), and they agree with each
other on both arms.

### 4.4 Arm B — `r23.i2.g72` / h60, the unroll — **LOST, decisively**

**−4.52% at prod against a 0.16% floor. Twenty-eight times the floor.** proxy is +0.30%,
above its 0.08% floor but below `min_gain` 0.007, so proxy is at best a non-result; prod
is unambiguous and prod is the shape that matters.

This is a clean falsification and it cost the thing it was supposed to cost. The unroll
did everything it promised at compile time:

- the forced `s_wait_loadcnt 0x0` moved from 1.5% into the body to 97% into it — **gone**;
- `v_mov_b64` fell 128 → 38 per iteration, a 70% cut, exactly the rotation copies h60
  identified;
- issue slots fell 6.8% (1450 for two iterations against 1556);
- VGPR fell 904 → 890, spill stayed 0, and the numerics are bit-comparable.

And it lost 4.5%. **Every stated mechanism fired and the op got slower.**

#### Why — and I had this wrong in §3.4

Gate condition 3 admits two readings, and in §3.4 I picked the wrong one. The champion
issues its 36 loads in a tight block at body indices **58–187, 24% of the way in**, with
the whole 64-WMMA run at 128–775 still ahead of them. `U` issues the slot-0 refill at
**601–735 — the very end of sub-iteration A, after all 64 of that half-body's WMMAs have
already issued** — and the slot-1 refill at 796–833. I computed cover-to-consumer
(≈765 instructions, longer than the champion's effective 603–728), declared condition 3
passed, and authorised card time.

Round 22's rule 1 does not say "cover must not shorten". It says **"any edit that moves
the last prefetch load later in the body loses."** It was 6-for-6. `U` moves the last
prefetch load of each sub-iteration from 24% of its half-body to 98% of it, which is the
largest such move any arm in this job has made, and it lost the largest amount.

The gate's own text is partly responsible and has been withdrawn in `pool.md`: condition 3
as round 23 wrote it says *"measure the last-load-to-consumer distance, **not the raw
index**"*. That instruction is wrong. Recording it that way rather than quietly fixing it,
because a gate that can be passed by an arm that loses 4.5% is worth more as a corrected
artefact than as a deleted one.

**Rule 1 is now 7-for-7, and its scope is sharper than it was: it is about the load's
issue position, not about the distance to its consumer.** Those two came apart here for
the first time — the champion's rotation copies make cover *shorter* than issue position
suggests, `U`'s absence of them makes cover *longer* — and when they disagreed, position
won. That is worth more to the next round than the unroll would have been.

A second reading, consistent with the first: the champion holds 36 loads in flight across
essentially the entire body, and the `s_wait_loadcnt 0x0` at rel 12 that h60 treats as the
defect is *retiring loads that have had a full body to complete*. h60 priced that drain as
pure loss. It is not — it is the point at which a deep, early-issued load stream is
collected. Deleting the drain by deleting the rotation also deleted the early issue that
made the drain cheap. **h60's lever was real, its sign was wrong.**

Round 23 closed prefetch depth, placement and drain-position. Round 24 closes the last one
open: **the rotation copies are not overhead, they are the price of early issue, and the
prefetch family is now closed on every axis this job can reach.**

### 4.5 Arm-carrier `D` — the h33 defects — free, and it ships

**+0.14% at prod against a 0.16% floor — a tie.** proxy is −0.42% against a 0.08% floor,
which is a loss by the letter of the rule but is 0.34 points beyond a floor measured on a
single pair; prod, with the larger floor and the larger signal, says free. fast is
uninterpretable.

Four extra instructions in `k_dkdv`, no change to `k_dq`, no change to VGPR, no change to
numerics, and in exchange three real defects go away: a 262,144-byte out-of-bounds *write*
and a silently-skipped dQ tile at `sq % 64 == 32`, a `num_records = 0` that silently drops
**every** dQ write once `nsp_q ≥ 8` (which is the prod configuration), and a live ~261 KB
out-of-bounds *read* on every prod launch today. Correctness is the obligation. It ships.

### 4.6 Arm A — X1 (h59 / h63 item 2) — the 4-wave barrier is **H-align**

`x1` is `w4` with barrier-1 deleted, one line. Both pass the full UT (`RESULT: PASS`), so
the deleted barrier is not producing a visible race at any tested shape — worth recording,
though X1 is a diagnostic and h59 never claimed it was legal.

Same process, palindromic, champion in two physical slots:

| shape | floor | champion ms | `w4` ms (x champ) | `x1` ms (x champ) | **X1 vs `w4`** |
|---|---|---|---|---|---|
| fast | 0.23% | 0.05330 | 0.08124 (0.656x) | 0.07824 (0.681x) | **+3.84%** |
| proxy | 0.43% | 0.55201 | 0.78578 (0.702x) | 0.69997 (0.789x) | **+12.26%** |
| prod | **0.08%** | 8.53680 | 11.26594 (0.758x) | 10.62310 (0.804x) | **+6.05%** |

h59 preregistered the discriminator: **H-align predicts ≤ +8%, H-count predicts ≥ +15%.**

**No shape reaches +15%. Two of three sit inside the H-align band, including prod, which
this session measured on a 0.08% floor — the tightest number this job has ever taken.**
proxy's +12.26% lands between the two bands and is the only figure with any ambiguity in
it, and it is nowhere near the H-count threshold.

The corroboration is stronger than the bands. h59 measured the *total* barrier cost as
`w4` 16.072 ms against `nobar` 11.095 ms, **+44.9%**. If the cost were proportional to
barrier count, deleting one of two should return roughly half of that, about +20%.
Measured: +6.05% at prod. That is not a near miss.

**Verdict: H-align.** The 4-wave barrier cost is the waves re-aligning at a rendezvous,
not a per-barrier toll. Deleting barriers one at a time does not pay, because the waves
still re-align at whichever one is left.

This is what h63 asked X1 to decide, and it routes the next step: under H-align, **X2
(delete both, private per-wave B ring) is the cheap confirmation and X5 (de-replicate the
staging, so there is less for the waves to align on) is the mechanism that actually
attacks the cost.** X2 is not merely a second barrier deletion — it is the barrier-free
*legal* kernel, so it measures the alignment-free floor rather than half a toll.

### 4.7 An incidental result on the fast shape

Phase 2 produced an 8.87% same-code floor at fast; phase 3, an hour later on the same
card, produced **0.23%** at the same shape. So fast is not intrinsically unmeasurable here
— phase 2 hit a one-off tail event. The guard that caught it was having two physical
incumbent slots in every session, which is the only reason §4.3's fast column was
discarded rather than believed. Two slots, every session, every shape.

### 4.8 Not asked for, cheap, and it closes the whole route: `nobar` and X2 today

The X1 verdict routes to X2 or X5, and both live under h51's ceiling — *even with free
barriers the 4-wave build tops out at 0.950x of the champion*. That ceiling was derived at
the old clock, and §4.2 has just shown the clock is not what it was. So one more session,
two arms, same palindromic method: `nobar` (both barriers deleted, **illegal** — the
unreachable upper bound of the whole route) and `x2` (both deleted **legally**, with the
private per-wave B ring, LDS 148480 B). `x2` passes the full UT.

| shape | floor | champion ms | `nobar` (x champ) | `x2` (x champ) |
|---|---|---|---|---|
| proxy | 0.94% | 0.55926 | 0.65434 (**0.855x**) | 0.65057 (**0.860x**) |
| prod | **0.015%** | 8.52348 | 9.23158 (**0.923x**) | 9.77563 (**0.872x**) |

prod's floor is **0.015%** — the same code twice, 8.524116 and 8.522835 ms. That is the
cleanest measurement this job has taken, by an order of magnitude, and it is what makes
the next sentence safe to write.

**The illegal, barrier-free upper bound of the 4-wave route is 0.923x of the champion at
prod. The legal one, X2, is 0.872x.** Assembling the route in one place, all against the
champion in the same session:

| arm | barriers | legal | x champion (prod) |
|---|---|---|---|
| `w4` | both | yes | 0.758x |
| `x1` (X1) | one deleted | diagnostic | 0.804x |
| `x2` (X2) | both deleted, private B ring | **yes** | 0.872x |
| `nobar` | both deleted | **no** | **0.923x** |
| champion | — | yes | 1.000x |

**h51's ceiling is confirmed, and it is tighter than h51 had it: 0.923x, not 0.950x.**
Every barrier in the 4-wave build could be made free tomorrow and the route would still
be 7.7% short of code that already exists and already ships.

The barrier cost itself, normalised through the champion so the two sessions can be
compared (the champion differs by 0.16% between them): `w4` 0.758x → `nobar` 0.923x =
**+21.8% at prod, +20.1% at proxy**. h59 measured +44.9% at the old clock. Half the
barrier cost evaporated when the clock changed, which is its own warning about carrying
absolute cycle-per-barrier figures across a machine change.

**Consequence for h63's ordering: X2 is measured and loses; X5 is bounded above by
`nobar` at 0.923x unless de-replicating the staging finds more than 8% from something
other than barriers, which nothing in h59 or the census claims it can.** The 4-wave /
5-GEMM route is dead at prod, not because barriers are expensive but because the 4-wave
body is slower than the 1-wave body even with the barriers gone. Rounds 25+ should not
spend card time on it, and the ~1270 cycles/iteration barrier number should stop being
treated as the route's blocker.

---

## 5. Verdict

| arm | id | prod vs incumbent (floor 0.16%) | outcome |
|---|---|---|---|
| **`D`** — h33 defects a/b/c | (correctness, no id) | **+0.14%** | **tie — SHIPS** |
| `U` — unroll `qloop_full` x2 | `r23.i2.g72` | **−4.52%** | **lost, DEAD** |
| `x1` — X1 4-wave diagnostic | h59 X1 | 0.804x champion | never shippable; **verdict H-align** |
| `x2` — X2 legal barrier-free | h59 X2 | 0.872x champion | **lost, route DEAD** |
| `nobar` — illegal upper bound | h59 | 0.923x champion | **ceiling, route DEAD** |

Shipped into `rounds/024/op`: `D`. Diff against `op/current` is three defect fixes and
nothing else — a stricter `sq` assert in `impl.py`, an Int64 dQ workspace extent, and four
clamp instructions. `job_context/op/current/` was not touched.

Nothing this round moved the operator's speed, which makes eight rounds since round 17.
What it did move: one pool entry to dead with its mechanism understood, one preregistered
hypothesis decided, one whole route closed against a 0.015% floor, and one live
out-of-bounds read removed from production.

---

## 6. Hygiene

**Post-card idle check** (`raw/idle_post.txt`). One KFD process exists on the node, pid
682297 running `lab3/ktime.py` on **gpuid 51359 = `fa-g3`** — the operator lab's card, not
ours (gpuid 30548). It started around 13:08; this round's last benchmark wrote its rows at
**13:07:43** and its rc at 13:07:44. **No overlap with any measurement in this round**, and
it is on a different card regardless. No python of this round's remains in `fa-g1`. All
four containers are pre-existing and still up; none was created or destroyed here, and
`docker run` was never used.

`dmesg -W` filtered for `amdgpu|kfd|reset|fault|hang|MES` ran across the entire card
session and **fired nothing**.

`job_context/op/current/` was not touched — verified by `diff -rq` before shipping (the
working copy was pristine) and the only writes since were into `rounds/024/op`.

**Evidence in `raw/`:** `screen_{cur_a,D,U}.json` (compile screens), `census_k_dkdv.txt`
(the three hot-loop histograms), `loopcensus.py` / `cover2.py` (the census tools, so the
numbers can be re-derived), `rows_{fast,proxy,prod}.json` (the deciding 4-arm session),
`rows4w_*.json` (X1), `rows4w2_*.json` (`nobar`/X2), `card_session.log` (every card
command's full output), `idle_pre.txt` / `idle_post.txt`, and `U_kernels.py` (the losing
arm, kept so `g72` never has to be rebuilt to be re-examined).

## 7. What round 25 inherits

1. **Two OPEN pool entries, `r24.i1.g73` and `r24.i2.g74`**, both corpus-sourced, both
   with free gates that can kill them before any card time. They are the only open
   direction this job has, because —
2. **the prefetch family is closed on every axis** (depth, placement, drain position,
   rotation cost) and **the 4-wave / 5-GEMM route is closed by measurement** against a
   0.923x ceiling. Two large regions of the search space are now genuinely eliminated
   rather than merely unexplored.
3. **Rule 1 measures the index, not the cover.** Seven for seven.
4. **`min_gain` 0.007 and a 0.08–0.16% floor.** The instrument is now far finer than the
   threshold, which is a good position to be in and is new this round.
5. **Absolute numbers are dead twice over** — different host *and* different clock. Any
   round that differences a B0 number against the A0 ledger is reporting a clock.

---

## §8 -- 路线表按 operator 纠正重写后的执行(步骤 5-7)

`route.md` 的 `## Route` 表在上一步被 operator 判为**顺序错误**:只写了 4 行,漏掉 h11/h12/h14/h18/
h57/h8/h6/h10/h58/h59/h60。表已重写为 **16 行**(9 条 `hint · must` → 5 条 `hint · advise` → 2 条 `idea`),
standing constraints 留在它自己那张表里被读而不被排程。**测量与结论一行未改**;本节只记录按新表从上往下
执行的结果,以及为此新做的两件事:`g73` 的免费门(零卡时)与第五次卡上 session(把三个 per-shape
champion 重测进同一个 session)。

### §8.1 `g73` 的免费门 —— 死在门上,零卡时

路线表第 15 行把门写死成:「按每条 `s_set_vgpr_msb` 之后触及的 VGPR 区间聚类 —— 若 bank 切换集中在
少数交错的活跃区间,分组编辑可以削;若均匀铺开,那就是寄存器堆的固有成本,免费死掉」,并且按 h18 的
第 11 轮结论**禁止**把门写成「降 VGPR」。

脚本 `raw/g73gate.py`,输入是本轮已经 dump 出来的冠军 ISA `dump/cur_a/k_dkdv_0/21_final_isa.s`,
输出 `raw/g73_gate_k_dkdv.txt`。`.LBB0_8`(冠军热体,778 条指令)实测:

| 量 | 值 | 门的读法 |
|---|---|---|
| `s_set_vgpr_msb` | **161** | 与 pool 记录一致 |
| `v_nop` | **45** | 合计 206 = **26.5%** |
| **distinct msb 操作数** | **121** | ⚠ 121 种编码 / 161 条前缀 |
| **只出现一次的编码** | **100(62.1%)** | ⚠ 六成是一次性的 bank 对 |
| 前缀间距 `gap==1`(背靠背成对) | **0** | 没有任何一对可以直接合并 |
| 间距直方图 | 2:70、3:25、4:17、5:12、6:10、8:9 … | 密集且均匀,中位数 2–3 |
| `v_nop` 落在前缀 ±1 条以内 | **8 / 45** | ⚠ 45 条 `v_nop` 里只有 8 条与前缀相邻 —— **它们不是前缀的伴随冒险填充,是独立的冒险填充**,`g73` 把两者打包成「26.5%」是**把两个不同的机制加在了一起** |
| **完美合并的上界** | **61 / 778 = 7.8%** | 把**每一个**重复出现的编码都合并掉(现实中办不到:它们之间隔着 2–4 条在干正事的指令,正是那些指令的操作数编号逼出了切换) |

**判决:`g73` 死在免费门上,零卡时。** 切换不是「集中在少数交错的活跃区间」,而是**均匀铺开**:121 种
不同的编码、六成只用一次、没有一对背靠背。这与 h18 第 11 轮的机理逐字吻合 —— `s_set_vgpr_msb` 由
**WMMA 操作数槽引用的寄存器编号**决定,不由寄存器总数决定(`g32` 在 632 VGPR 上仍有 110 条)。904 VGPR
之上这就是寄存器堆的固有成本,FlyDSL 也没有寄存器钉选可以改变编号分配。

并且即使门过了也不该上卡:上界 7.8% 是**指令数**上的,而 h18 的账本四中四 + 语料库反例(少发 40 条
`s_waitcnt`、多发 16 条 `s_nop`,发射槽净赢、时间净输,863.0 vs 918.9)说的就是**发射槽数不排名,卡排名**。

⇒ 按路线表第 4 行的写法,**h14 与 `g73` 一起结案**:h14 的理由(「让每一个 4-wave 变体从勉强变舒服的
前提」)本轮已随 4-wave 路线一起作废,它唯一可能的新接地是 `g73`,而 `g73` 刚刚在免费门上死了。两条都
进 `dead_ends.md`(下一步写,不是本步)。

`g74` 的免费门**没有**在本轮执行:它要求先把热体 192 条 VALU 分成「真在携带链上」与「per-tile 早已独立」
两类,而 gfx1250 的文本寄存器名在 v255 处截断、真号在 bank 前缀里(h14 已记),做 read-before-write 必须
先把 121 种前缀解码回真实编号。半成品的携带链分析会给出一个不可靠的判决,而这条 idea 的价值全在判决上。
**按路线表第 16 行留给第 25 轮,状态不变 OPEN。**

### §8.2 第五次卡上 session —— 三个 per-shape champion 重测进同一 session

前四次 session 只重建了 incumbent(`cur_a`/`cur_b` = `op/current` = `rounds/020/op`),**没有**把
`rounds/017/op` 与 `rounds/019/op` 这两个 per-shape champion 放进同一个 session。步骤 6 明文要求
「incumbent 与每一个不同的 per-shape champion,在同一个 session、同一张空闲卡上背靠背重跑」,所以
本轮补做了第五次 session。这不是形式:**它改变了本轮的验收算术**(见下)。

规程:开工前本卡(kfd `gpu_id 30548`,容器 `fa-g1`,物理 GPU 1)KFD 队列全空(另外三张卡有 lab 与
其它 job 的进程,按 h62 本来就不归我们);**整个 session 前先 `rm -rf /tmp/flycache` 清编译缓存**
(步骤 5 的硬性要求:后端喂一个陈旧 kernel 会让本轮测到上一轮的二进制而任何地方都不报错);
`validation.py` → `ut/test_correctness.py` → 每 shape 一个 benchmark 进程,51 iters 中位数,回文;
全程 `dmesg -W` 监控,**零 amdgpu 事件**;`RC` 全 0;收工后本卡 KFD 队列再次全空,`fa-g1` 内无我们的
python 残留,四个容器一个没建也一个没删。原始输出 `raw/card_session_phase5.log`、
`raw/rows5_{fast,proxy,prod}.json`、`raw/idle_post_phase5.txt`。

`r24_a` / `r24_b` 是 `rounds/024/op` 的**两份物理独立的拷贝**,同码地板由它们给出。

| shape | `r24_a` | `r24_b` | **r24 均值** | 同码地板 | `c020`(incumbent) | `c019` | `c017` | `beat` | sclk |
|---|---|---|---|---|---|---|---|---|---|
| fast | 97.774 | 89.134 | **93.454** | **9.69% —— 不可读** | 95.683 | **99.149** | 98.348 | 51.536 | 2359→2346 |
| proxy | 579.478 | 576.635 | **578.056** | 0.49% | 576.247 | 592.893 | **593.180** | 759.421 | 1923→1980 |
| prod | 639.848 | 637.797 | **638.822** | 0.32% | 638.895 | **650.246** | 648.700 | 851.971 | 1792→1835 |

(TFLOP/s。`r24` 一律取两槽**均值**并把两槽原值都放在表里 —— 两槽在三个 shape 上恰好都是 `a` 快于 `b`,
只报 `a` 会系统性地把自己报高。)

**本轮最重要的一条新事实,而且它是被步骤 6 的规程逼出来的:**

> **`rounds/019/op` 与 `rounds/017/op` 今天在三个 shape 上全部快于 incumbent `rounds/020/op`。**
> prod:`c019` 650.25、`c017` 648.70 对 `c020` 638.90 —— **+1.78% / +1.54%,是同码地板 0.32% 的五倍**。
> proxy:+2.89% / +2.94%,对 0.49% 的地板。fast:+3.62% / +2.79%,但 fast 本 session 地板 9.69%,不可读。
>
> 账本里 round 19 当时是**被拒**的(gain 0.9939),round 17 是 deep round(gain 1.1590,已被 20 取代)。
> 今天同 session 重测,两者都比现任冠军快。这正是「champion 按轮次追踪而不是按数字追踪」的理由:
> 019 与 020 之间的差是 0.6%,而 20 是靠一个 +0.16% 的 gain 被接受的 —— 在今天这台机器(B0、
> sclk 1792–2360、h62 换机)上,那个排序**不成立**。⚠ 这是给下一步(`facts.md`)的,不是我改的。

**本轮 arm(`D`,h33 缺陷 a/b/c)的验收算术,按测量报,不做任何修饰:**

| shape | r24 | 对 incumbent `c020` | **对 best-ever(同 session 重测)** | 对 target(= `beat`,margin 1.0) |
|---|---|---|---|---|
| fast | 93.454 | 0.9767 | **0.9426**(对 `c019`)⚠ **低于 95%**,但地板 9.69% | **1.8134**(未封顶) |
| proxy | 578.056 | **1.0031** | 0.9745(对 `c017`) | 0.7612 |
| prod | 638.822 | **0.9999 —— 与 incumbent 死平**(地板 0.32%) | 0.9824(对 `c019`) | 0.7498 |

- **throughput 没有超过 best-ever,所以本轮不被接受**,这是算术,不是判断。
- `fast` 的 0.9426 低于「每个未达标 shape 至少是自己 best-ever 的 95%」那一条 —— **照报**。
  它落在一个 **9.69% 的同码地板**里(两份逐字节相同的目录相差 9.69%),而 h5/h7 说 fast 是哨兵、
  prod 才排名。**不把它调好看,也不用地板去解释掉它:两个数都在表里。**
- `validation.py` 本 session 自己跑了一遍:correctness **pass**、determinism **pass**、speed **FAIL**
  (geomean 0.978x beat),`exit_code = 2`。speed 是本 job 的常设门,冠军同样过不了(`beat` 是 aiter ASM)。
- UT **15/15 PASS**,SQNR dq/dk/dv 52.52–52.84 dB(门 50.0 dB)。

### §8.3 工作副本里留下的是被测的那一份

`rounds/024/op` = 树 `D` = h33 缺陷 a/b/c 的修复,md5 与本轮测量的树逐字节相同,**没有回滚**。
它在 prod 上与 incumbent 死平(0.9999,地板 0.32%),在 proxy 上 +0.31%,这是一个**结果**,按
`outcome: delivered` 记录;是否晋升由 Python 按上表的数字决定,不由本轮决定。
`job_context/op/current/` 全程未被触碰(`kernels.py` md5 仍为 `51c4bf60…`)。

### §8.4 第 8 步收尾的账本动作

- `route.md` 的 16 行 `outcome` **全部填好**,每一行都带这一轮的实测数字或「未执行 + 为什么」。
- `pool.md`:`r24.i1.g73` 标为 **DEAD at its free gate**(原文在 `<details>` 里保留),并且把
  「45 条 `v_nop` 里只有 8 条挨着前缀」这条**记账错误**写进条目本身,而不是悄悄改掉百分比;
  `r24.i2.g74` 保持 **OPEN**;新开 `r24.i3.g75` 作为**已出货并同时关闭**的条目 —— 它不是 idea,
  只是 `act.yaml` 的 `candidate:` 必须能指到一个 id,本轮真正的两条 pool 新增仍是 `g73`/`g74`。
- `act.yaml` 按 `fast_loop/opt/schemas/act.yaml` 写好并做过 YAML 解析检查。
- `facts.md` 与 `dead_ends.md` **没有动** —— 它们归下一步。留给它们的三件事已在 `route.md` 的
  `outcome` 里写明:①`g73` 与 `h14` 进 dead_ends;②`r1.i3.g03` 的 closeout 是**丢失**而不是未写,
  须补;③`rounds/019/op` 与 `rounds/017/op` 今天在三个 shape 上全部快于 incumbent。
- 卡的收尾:本卡 KFD 队列全空,`fa-g1` 内无我们的 python,四个容器一个没建也一个没删,
  `dmesg -W` 全程零 amdgpu 事件。
