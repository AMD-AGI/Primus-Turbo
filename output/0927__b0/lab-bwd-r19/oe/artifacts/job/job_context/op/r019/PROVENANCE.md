# baseline provenance

## Branch taken

**Branch B: developed from scratch against the backend corpus, NOT adopted from a
best-practice recipe.** No recipe under `knowledge/backends/flydsl/attention/` matches
`op.config`, so there was no implementation pointer to follow.

This is the branch the step-1 instructions call "develop it from the backend's own corpus"
(`techniques.md`, `dead-ends.md`, the `recipes/`). What it was built *on* is the first-party
gfx1250 FlyDSL bring-up work the job spec itself names.

## Config diff, field by field

Both candidate recipes were diffed against `op.config`. Fields only one side states are not
compared. **Both fail, and on fields that are hard gates rather than preferences.**

### `recipes/hd64.md` — source `https://github.com/AMD-AGI/Primus-Turbo`, commit `ed8d7af46938d8debe56fd5ff5b70fd64dde33be`

| field | recipe | `op.config` | |
|---|---|---|---|
| `q_head_dim` | 64 | **128** | ✗ |
| `kv_head_dim` | 64 | **128** | ✗ |
| `arch` (`applies_to`) | gfx950 | **gfx1250** | ✗ |
| `heads_q_per_kv` | power of two in [8, 256] | **4** | ✗ |
| `format` | thd-varlen / sbhd (bshd at batch 1 only) | **bshd, batch 4** | ✗ |
| `determinism` | dQ not reproducible by default | **atomic-free, every element written once** | ✗ |
| `causal` | bottom-right | bottom-right | ✓ |
| `sliding_window` | left only or none | none (`-1`) | ✓ |
| `sparsity` | dense | dense | ✓ |
| `direction` | [fwd, bwd] | [bwd] | ✓ |
| `dtype.qkv` | bf16 | bf16 | ✓ |
| `softmax_scale` | 1/sqrt(D) | `head_dim ** -0.5` | ✓ |
| `sink` | optional | none | ✓ |

### `recipes/hd128.md` — same source repository, same commit `ed8d7af46938d8debe56fd5ff5b70fd64dde33be`

Head dimension now matches; four fields still differ, and each one alone is disqualifying.

| field | recipe | `op.config` | |
|---|---|---|---|
| `arch` (`applies_to`) | gfx950 | **gfx1250** | ✗ |
| `heads_q_per_kv` | power of two in [8, 256] — a HARD `_gqa_group_ok` gate, and a **backward correctness** constraint (`LD_VEC >= 2`), not a tiling preference | **4** | ✗ |
| `format` | dense is SBHD-only; bshd accepted at batch 1 alone and refused above it | **bshd at batch 4** | ✗ |
| `determinism` | dQ accumulated with `buffer_atomic_pk_add_bf16`, NOT bitwise reproducible by default; `deterministic=True` costs 22.7% | **atomic-free by construction, every output element written exactly once** (`op.config.determinism`, gated at 200-run bitwise identity) | ✗ |
| `q_head_dim` | 128 | 128 | ✓ |
| `kv_head_dim` | 128 | 128 | ✓ |
| `causal` | bottom-right | bottom-right | ✓ |
| `sliding_window` | left only or none | none (`-1`) | ✓ |
| `sparsity` | dense | dense | ✓ |
| `direction` | [fwd, bwd] | [bwd] | ✓ |
| `dtype.qkv` | bf16 | bf16 | ✓ |
| `softmax_scale` | 1/sqrt(D), baked | `head_dim ** -0.5` | ✓ |
| `sink` | optional fp32 [Hq] | none | ✓ |

**The arch mismatch is not a formality and the recipe is not retargetable.** Measured by
in-container LLVM probe (recorded in the `flydsl-gfx1250` skill): on gfx1250
`llvm.amdgcn.mfma.f32.32x32x16.bf16`, `llvm.amdgcn.ds.read.tr16.b64` and
`llvm.amdgcn.permlane32.swap` all **cannot select**. gfx1250 has no MFMA at all. The recipe's
math, accumulator widths and every hand-derived LDS→VGPR stride are expressed in the MFMA
32x32x16 wave64 fragment layout, and `warp_size = 64` is hardcoded at
`primus_turbo/flydsl/utils/attn_helper.py:382`. Adopting it was never an option to weigh.

## What the baseline was actually built from

Source repository: `https://github.com/AMD-AGI/Primus-Turbo` (local checkout
`/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo`).
**Commit `5690a771874a0fe8ab6277c245a411a1f746e9ff`** (2026-09-17 12:01:15 +0000).

The job spec names these three files directly and says setup copies them in and wires them
behind one callable. They are single-tile, single-wave bring-up probes, each validated
against torch on this card at 141–159 dB. They build and run in this environment — the
"stop and report if the referenced implementation will not build" rule did not trigger.

| file read | what came from it |
|---|---|
| `output/0917__flydsl/kernels/README.md` | the map: what exists (the complete backward **data path** at single-tile single-wave scope) and what does not (causal masking, the kv/query loops, GQA, multi-wave, tails, the bank-conflict swizzle, varlen, the launcher) |
| `output/0917__flydsl/kernels/odo_gfx1250.py` | `k_delta_bshd` — copied **verbatim**. Already full BSHD `[B,S,H,D]` → `[B,H,S]`. Also the proof that buffer-descriptor OOB store steering works on gfx1250 |
| `output/0917__flydsl/kernels/dkdv_loop_gfx1250.py` | the structure of `k_dkdv`: the runtime q loop as a `@flyc.jit` generator, the four-operand `ds_load_tr16_b128`-over-row-major-LDS path, causal bottom-right |
| `output/0917__flydsl/kernels/dq_gfx1250.py` | the structure of `k_dq`, and the "free operand" trick — two consecutive kv-tile dS accumulators concatenate in-lane into one v16 A-operand, no LDS round trip |
| `output/0917__flydsl/kernels/wmma_layout_probe.py` (via README) | the gfx1250 wave32 WMMA fragment layout every index in `kernels.py` is derived from |
| `knowledge/backends/flydsl/attention/{README,techniques,dead-ends}.md` | read; all gfx950. Treated as structure, never as numbers |
| `knowledge/pitfalls/measurement-traps.md` | the measurement method in `benchmark.py` |

### What was added here, and why

The three sources are single `[S,D]` tiles. Everything below correctness demanded; none of it
is an optimisation.

- **Batch, head and GQA addressing** in all three kernels (`rs_q = Hq*DV8`, `rs_kv = Hkv*DV8`).
- **A runtime kv loop in `k_dq`** (the source had `KV = 32`, exactly one WMMA contraction and
  no loop at all), with Q/dO fragments, `lse` and `delta` hoisted out of it.
- **Causal masking in `k_dq`** (the source was non-causal), and **bottom-right causal at
  unequal seqlens** in both main kernels via `cshift = Skv - Sq`.
- **`k_dkdv` flattens its runtime loop over `(q head, q tile)`**, so the GQA reduction happens
  in registers. This is a deliberate departure from the ASM anchor, which writes
  `[B, Skv, Hq, D]` and reduces on the host. It is *simpler*, not cleverer: it removes the
  scratch tensor and the host reduction, and it makes `op.config.determinism` true by
  construction — every output element is written exactly once, no atomics anywhere.

### Known weaknesses — deliberately left for the rounds to fix

- **No causal tile-skipping in either main kernel.** Both do the full `S²` work even when
  causal masks roughly half of it. This is the single largest structural inefficiency and it
  is the obvious round-1 target.
- **One wave (32 lanes) per workgroup**, inherited from the bring-up probes. No multi-wave
  scheduling, no TDM async pipeline, no `sched_barrier` placement.
- **`k_dkdv` pads a 16-wide contraction to the WMMA's 32**, wasting half of that GEMM.
- ~~**No bank-conflict swizzle** on the LDS staging.~~ **FIXED round 4 (`r4.i1.g16`)**: `X_ROW_B` 256 -> 272 B, `S_ROW_B` 64 -> 80 B. 256 B was exactly the 64-way collision stride on gfx1250's 64x4 B LDS. +10.7% alone, +50.2% merged with `g09`.
- Shape constraints asserted rather than handled: `D == 128`, `Sq % 16 == 0`,
  `Skv % 32 == 0`, `Hq % Hkv == 0`, `B*S*H % 32 == 0`. Every spec shape divides.

## Round 2 (fast), 2026-09-21

`k_dkdv` is round 1's `armA` plus two changes measured apart and then merged:

- **`r1.i7.g07`** -- every global buffer descriptor in `k_dkdv` carries its TRUE byte
  extent instead of a flat 1 GiB `num_records` (`k_dkdv` gained a `B_` argument to compute
  them). This is an address clamp, not an algorithm change, and not an h2 violation: h2
  forbids changing `OOB_SELECT`, and `k_delta_bshd` in this same file already proves the
  true-`num_records` bound works on this part. It unblocks:
- **`r1.i2.g02`** -- 32 query rows staged instead of 16, so the dK/dV GEMM's contraction
  over queries is full and no WMMA lane multiplies a padding zero. Query tiles are consumed
  in pairs (`seqlen_q % 32 == 0`).
- **`r1.i6.g06`** -- `BLOCK_KV = 32`: the workgroup owns 32 key rows as two 16-row WMMA
  accumulator sets, with Q/dO staging and transpose loads shared between them
  (`seqlen_kv % 32 == 0`).

Measured 1.893x the round-1 incumbent as a three-shape geomean, 0.169x of beat.
680 VGPR, zero spill, 20480 B LDS. Correctness 52.52-52.83 dB, 200-run bitwise determinism.

## Round 3 (fast), 2026-09-21

Shipped **armAB = `r3.i1.g10` + `r3.i2.g11`**, both in `k_dq`/`impl.py`, built from
`rounds/003/op` (= round 2's shipped tree) and merged only after both arms were measured
alone and neither lost.

* `r3.i1.g10` -- `k_dq`'s query tile `BLOCK_Q` 16 -> 32. One wave still; K/V global loads,
  the K LDS staging and its `ds_load_tr16_b128` transpose are now amortised over twice the
  queries. Screen: VGPR 269 -> 457, zero spill, LDS unchanged, WMMA density unchanged.
* `r3.i2.g11` -- the three kernels store bf16 directly; `impl.py` drops the fp32 temporaries
  and the three `.to()` conversions. Bit-identical (FlyDSL's f32->bf16 is RNE).

One-session palindromic sweep, idle card: **1.199x the incumbent, 0.2012x of beat**
(fast 0.6756 / proxy 3.1437 / prod 29.3595 ms). dq/dk/dv **bitwise identical** to the
incumbent at fast and proxy.

---

## Round 3 ship: `r3.i4.g13` -- `k_dq` `BLOCK_Q` 32 -> 64

**Everything above this line describes `armAB` (g10 + g11), which is what this directory
held until 2026-09-21T15:35:00. It is no longer what this directory holds.** At 15:35 the
restarted opt step shipped one more rung on top of it and did not update this file, so the
provenance and the tree disagreed for a day. Corrected here.

The shipped delta is a single constant in `kernels.py`'s dq section:
`BLOCK_Q = 32  # r3.i1.g10` became `BLOCK_Q = 64  # r3.i4.g13`. `NQ = BLOCK_Q // 16`
follows it, and `q0 = bid * BLOCK_Q` plus the causal limit `_lim` were already generic, so
the kernel body is unchanged; `impl.py` launches with `sq // _k.BLOCK_Q`, so the grid
tracks the constant and needed no edit. `k_dkdv` is untouched (`NKV = 2`, `BLOCK_KV = 32`).
One `k_dq` wave now owns 64 query rows instead of 32, amortising the K/V global loads and
the K LDS `ds_load_tr16_b128` transpose over twice the queries. Screen: VGPR 457 -> 826,
zero spill, zero private segment, LDS unchanged, WMMA density unchanged.

The pre-ship tree is preserved byte-for-byte at `rounds/003/_scratch/base/`
(`kernels.py` md5 `0e090d4355b766e7cf2c35bb47c40764`); this tree is byte-identical to
`rounds/003/_scratch/armQ64/` (md5 `cf497fef581993f35757c8e6f1888146`).

### Evidence

| | |
|---|---|
| correctness | `validation.py` -> pass, 52.52-52.83 dB on dq/dk/dv at fast/proxy/prod, `isfinite` coverage full. Taken twice: `_scratch/gpu3/out` (2026-09-21T15:39) and again 2026-09-22T08:44 after the AC-cycle. |
| determinism | pass -- dq/dk/dv bitwise identical across 200 consecutive runs at `fast`. |
| bitwise vs armAB | identical dq/dk/dv at fast and proxy (21 M elements). |
| one-session sweep | `1-opt/raw/bench_onesession.json` (15:33) -- +6.29% geomean over armAB, positive on all three shapes; `k_dq` -33.4%, `k_dkdv` within +0.03%. |
| **acceptance** | `1-opt/raw/bench_accept.json` (2026-09-22T08:45, 101 iters) -- **1.291x geomean over the round-2 champion**; fast 1.145x, proxy 1.288x, **prod 1.459x (142.23 -> 207.47 TF/s)**. Against baseline 3.06x geomean, 3.77x at prod. Against beat 0.279x geomean. |

The acceptance row is the one the round never got: `gpu5.sh` was detached at 15:42:21 to
produce it and the card wedged at 16:01:54 with its output still empty. It was retaken by
hand after the 2026-09-22T06:47:43 AC-cycle -- see `_provenance` inside that JSON for how
it differs from what `gpu5.sh` would have written.

---

## Round 5 (fast), 2026-09-22 -- `r3.i3.g12` + `r5.i1.g17`, shipped as the merge `armKH`

Built from `rounds/004/op` (= round 4's shipped tree). Two arms, independent by
construction -- one touches only `kernels.py`, the other only `impl.py` -- each built alone
and measured alone; the merge built only because neither lost.

* **`r3.i3.g12`** (`kernels.py`, `k_dq`) -- `k_dq` read the same 32x128 K tile from global
  **twice**: once in a lane-contiguous staging loop that filled `lds_k`, and again through
  `gfrag` inside the `kt` loop for the S = K Qᵀ operand. The staging loop is deleted and the
  LDS tile is now written from the fragment registers the S GEMM already holds, via a
  `gfrag2` copied verbatim from `k_dkdv` -- the same construction `r2.i2.g09` shipped there
  in round 4. `gfrag2`'s two halves tile the `[32][D]` image exactly once over
  `kt x half x dt x lane` (row `kt*16+row`, byte column `half*16 + dt*64`, second half
  `+32`), so it is the same bytes in the same order into the same LDS addresses. No new
  barrier -- the stores sit where the staging loop sat, ahead of the same `fx.barrier()`.
  Screen: loop-body `buffer_load_b128` **48 -> 32**, VGPR **826 -> 800**, `ds_store_b128`
  unchanged at 16, LDS unchanged at 8704 B, **zero spill, zero scratch**. `k_dkdv`
  byte-identical.
  **Measured: geomean -0.34% (prod +0.9%, proxy 0.0%, fast -1.9%), i.e. a NULL.** It ships
  because it is free, because its +0.9% at prod reproduced with the same sign and size in
  two independent cold sessions, and because it lowers `k_dq`'s register pressure for the
  structural work `r3.i6.g15` will need. See `rounds/005/1-opt/opt.md` sections 4-6.

* **`r5.i1.g17`** (`impl.py` only -- `kernels.py` is byte-identical with and without it) --
  the three per-call launches went through `@flyc.jit`'s `JitFunction.__call__`, whose
  `_resolve_and_make_cache_key` path costs a flat **0.266 ms of Python per call at every
  shape**, i.e. **51.3% of the `fast` shape's wall time** (`beat` runs all of `fast` in
  0.104 ms). `impl.py` now calls `flyc.compile(launcher, *args)` once per
  `(launcher, per-tensor dtype/rank, device)` and memoises the returned `CompiledFunction`.
  `flyc.compile` **issues the first launch itself**, so the memo must not repeat it;
  `COMPILE_ONLY` builds return `None` and fall back to the ordinary call.
  **Measured: +11.6% geomean alone, +36.2% at `fast`.** Host issue cost **0.2688 -> 0.0313
  ms (-88%)**; host share of `fast` **51.3% -> 8.3%**. No GPU code changed.

### Evidence

| | |
|---|---|
| correctness | `validation.py` run from **this directory**, cold, before any speed number was read -> `correctness pass`, 52.52-52.83 dB on dq/dk/dv at fast/proxy/prod against a 50 dB gate. |
| determinism | pass -- dq/dk/dv bitwise identical across 200 consecutive runs at `fast`. |
| bitwise vs round 4 | **0 differing bits of 21 M / 21 M / 201 M** on dq/dk/dv at fast, proxy **and prod**. The round changed no arithmetic. |
| arms, session 1 | `1-opt/raw/bench_onesession.json` -- cur/armK/armH/beat: armK 0.9966x, armH 1.1121x. |
| arms, session 2 | `1-opt/raw/bench_merge.json` -- cur/armK/armH/armKH/beat: armK 0.9938x, armH 1.1157x, **armKH 1.1190x**. Superposition predicts 1.1088; the merge is additive, not superadditive. |
| **acceptance** | `1-opt/raw/bench_accept.json` -- `rounds/005/op` vs `rounds/004/op` vs `beat`, one cold session, idle card, 51 iters, palindromic, `sclk` witnessed per row: **1.1446x geomean over the round-4 champion** (fast 1.3621x, proxy 1.0862x, prod 1.0132x). Score **0.3655x beat**, up from 0.3238x; gap **3.09x -> 2.74x**. |
| per-kernel split | `k_dkdv` / `k_dq` = 65.7/19.5 (fast), 62.4/33.1 (proxy), **62.2/36.9 (prod)**. |

`speed FAIL` / `RESULT: FAILED` from `validation.py` is the **>= 1.00x-beat ship gate**,
which this job has never met and does not claim to. Correctness and determinism both pass.
