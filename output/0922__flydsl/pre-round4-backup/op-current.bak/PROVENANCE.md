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
- **No bank-conflict swizzle** on the LDS staging.
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
