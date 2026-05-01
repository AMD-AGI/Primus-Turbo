# Round 64-dm — FP8 grouped: Lever D R-B step 1 LANDED — st_32x64 shared-memory tile type

**Status**: INFRASTRUCTURE-ONLY commit, no metric impact / clean step 1 of R37+ plan
**Score before** (R36 baseline): 957-960 noise band
**Score after** (this round): 959 (grp_FP8 geomean 1.1195)
**HK SHA**: `78415fb0` → `ecbead9a`
**PT SHA**: this commit
**Round time**: ~30 min (1 build + 2 metric runs + R56 cost-model re-audit)
**Auto-optimize round**: 37

---

## What was done

Executed step 1 of the R37+ Lever D full-port roadmap (see
`round-63-dm-fp8-grouped-plateau-ACCEPTED-and-lever-d-full-port-roadmap.md`):

1. Added `struct st_32x64` to `include/types/shared/st_shape.cuh`:
   rows=32, cols=64, subtile_padding=64, fp8 bytes_per_thread=16,
   identity swizzle (step 2 will refine), added to `ducks::st_shape::all`.

2. Added public `st_32x64_s` alias to `include/types/types.cuh`.

3. Added `__lever_d_round_b_force_instantiate_st_32x64` stub to
   `kernel_fp8_layouts.cpp` that declares a `__shared__
   st_fp8e4m3<HB, 64, st_32x64_s>` tile and static-asserts its
   static-member derivations (rows, cols, underlying_subtile_{rows,
   cols}, bytes_per_thread).

Build passes on gfx950. Spill profile BIT-IDENTICAL to R36 baseline
for all 4 grouped_rcr_kernel specs (39/43/32/39 dw). LLVM DCE trims
the force-instantiate stub at codegen time — 0 bytes added to
runtime instruction memory.

**HK commit**: `ecbead9a` — `infra(fp8 grouped): Lever D R-B step 1
(R37 / dm-R64) — add st_32x64 shared-memory tile type`. 116 lines
added across 3 files (70 lines stub + 34 lines struct + 12 lines
alias comment).

---

## Critical cost-model re-audit (performed this round)

Before blindly executing 4-6 rounds of kernel rewrite, I re-examined
the analytical cost model for the full main-loop 32x32x64 port vs
the current 16x16x128. **The math does NOT predict the +7.7 pp gain
that R56-dm claimed**.

### Per-K-iter MFMA cycle count (main loop, K=128)

| Approach | MFMAs per K-iter | cy/MFMA | Total cy |
|--|--|--|--|
| Current rt_16x16 | 32 × mfma_1616128 | ~16 | **512** |
| New rt_32x32 | 8 × mfma_323264 per K=64 iter × 2 K-iters for same K=128 | ~32 | **512** |

**Identical MFMA cycle count.** mfma_323264 does 4× the per-call
work of mfma_1616128 but takes 2× the issue time — net throughput
equal for a fixed K volume.

### Per-warp accumulator register pressure (cA + cB + cC + cD)

| Approach | Cells × dw/lane/cell | dw/lane per 64×32 acc | × 4 accumulators |
|--|--|--|--|
| Current rt_16x16 | 8 × 4 | 32 | **128** |
| New rt_32x32 | 2 × 16 | 32 | **128** |

**Identical VGPR pressure from accumulators.** The supposed
"halved register tile count" in R56-dm was misdirected — what
changes is the lane→data mapping, not the total dwords held
per lane.

### Where does the +7.7 pp claim come from?

Tracing back through R52 → R56 → R58 → R62:
- **R52 probe 2** (standalone K-tail kernel with 32x32 mfma):
  FALSIFIED at -76 pts. First empirical evidence that 32x32
  cell-shape isn't a slam dunk.
- **R56-dm table** (line 132-140): the numbers were speculative
  ceiling estimates, not based on empirical data. The table
  explicitly said "Lever D Round-B full main-loop port | ~1.10
  (+7.7 pp) | 4-5 rounds" — this is a CEILING, not a central
  estimate.
- **R58-dm** re-audited the K-tail cost model and gave corrected
  +128 cy mfma savings for K-tail only (not main loop).
- **R62-dm** (last round) measured actual K-tail port performance:
  **-9 pp regression from LDS fan-out cost**. This was the first
  empirical rejection of the 32x32 migration path at the tile
  level.

**Honest assessment**: the Lever D full-port thesis has NO empirical
backing. Every attempt to migrate part of the kernel to 32x32 cell
shape has either regressed or shown no improvement. The analytical
model shows equal MFMA cycle cost and equal VGPR pressure.

The only UNTESTED hypothesis is: **maybe the lane→data permutation
difference exposes better LDS read bandwidth or pipeline overlap
that LLVM can schedule more aggressively**. This is a weak
hypothesis with no precedent on gfx950.

### Validation gate added to R38+ plan

I've added an explicit validation gate in the HK commit message
(and in the `st_shape.cuh` code comment): before R38+ agent commits
to any further kernel rewrite, they MUST run a focused microbench
comparing mfma_323264 vs mfma_1616128 throughput on a synthetic
M=64 N=32 K=128 single-warp workload. If the microbench shows
< 3 pp single-warp throughput advantage for mfma_323264,
**ABANDON the full port** and accept the 957-962 plateau.

Running this microbench costs ~1 round (1 build + 1 rocprof +
analysis). It's a cheap way to de-risk the 4-6 round commitment.

---

## Why step 1 is safe to land anyway

The ST_32x64 type addition is:

1. **Bit-identical codegen for existing kernels**: every other
   shape struct in `st_shape.cuh` is independent. Adding a new
   struct + adding it to `ducks::st_shape::all` doesn't perturb
   any existing template instantiation.

2. **LLVM DCE removes the force-instantiate stub**: the
   `__attribute__((used))` prevents linker-level DCE but not
   LLVM's SSA-level DCE. The stub's `__shared__ dummy_st` and
   `(void)` reads are dropped at optimization time; no
   instructions reach the final binary.

3. **Reusable infrastructure if Lever D is later abandoned**:
   if R38 microbench falsifies the port, the st_32x64 type just
   sits as unused infrastructure (like `st_64x32_padded_b128` in
   BF16's RCR Route 1 — landed but not wired into mainline).
   Zero cost to keep, zero benefit to remove.

4. **Validates kittens template plumbing**: the static_asserts
   in the stub confirm that:
   - `st<fp8e4m3, 128, 64, st_32x64>` composes correctly
   - `underlying_subtile_*` derivations match the shape struct
   - `bytes_per_thread<fp8e4m3>` dispatches the size-1 branch
   - Swizzle functor is callable at compile time
   This is useful scaffolding for R38+ regardless of whether
   the full port is pursued.

---

## Metric delta this round (no kernel change)

```
R36 baseline:                957-960 noise band
R37 before infra add:        960 (1 sample)
R37 after infra add:         959 (1 sample)

grp_FP8 geomean:             1.1230 → 1.1195 (within noise)
grp_BF16 geomean:            1.1814 → 1.1823 (within noise)
```

Noise-band movement only. No shape's ratio changed by > 2 pp across
the two runs. Correctness 0/32 FAIL.

---

## What R38+ should do

### Step 1a (recommended first, ~1 round): mfma_323264 microbench

Write a standalone `.cu` file that:
- Allocates M=64, N=32, K=128 device tensors
- Runs N=10000 iterations of (a) 8 × mfma_1616128 and (b) 2 ×
  mfma_323264 per "K-iter"
- Measures wall clock per iteration
- Compares throughput

If mfma_323264 is < 3 pp faster per-iter (single-warp, no LDS
overhead), **abandon Lever D full port**.

If mfma_323264 is ≥ 3 pp faster, **proceed to step 2**.

### Step 1b alternative (if microbench is inconvenient):

Read the gfx950 ISA documentation for mfma_f32_32x32x64_f8f6f4 vs
mfma_scale_f32_16x16x128_f8f6f4. Compare issue latency,
pass-through latency, and accumulator dependency chain. If
mfma_323264's issue rate is meaningfully higher per-cycle, it
could expose more pipelining headroom.

(I didn't have time this round to pull the ISA docs; R38 agent
with access to `/opt/rocm/share/amdgpu/amdgpu_shaders/*.asm`
or `/opt/rocm/llvm/share/doc/` might find this faster than a
microbench.)

### Step 2 (if validation passes): ST_32x64 swizzle design

Derive bank-conflict-free swizzle for ST_32x64. The mfma_323264
input layout is (per kernel_fp8_layouts.cpp:2514-2517 which
describes the rt_16x128 K-tail lane map):
- A: lane L reads column `(L/16)*32 + col`, row `L%16`
- B: same pattern

For st_32x64 with 32x32x64 mfma, the input lane map is different.
Need to verify against the mfma_323264 intrinsic's A/B operand
register layout (defined in `mma.cuh` line 172-185 branch).

### Step 3+ (~4 rounds): main-loop kernel port

Follow R63-dm roadmap: load helpers, kernel skeleton, dispatch
wiring, full metric coverage.

---

## Cumulative landing progression

| Round | Infra | Status |
|--|--|--|
| R14-dm | `rt_32x64` / `rt_64x32` shape structs | landed |
| R57-dm | `rt_32x64_s` / `rt_64x32_s` public aliases | landed |
| R59-dm | `rcr_mma_32` wrapper | landed |
| R61-dm | K-tail rt_32x64 loaders | landed |
| R62-dm | K-tail port attempt | falsified, reverted |
| **R64-dm (this)** | **st_32x64 type + alias + force-instantiate** | **landed** |
| R38+ | microbench validation gate | PENDING |
| R38+ | ST_32x64 swizzle refinement | PENDING |
| R38+ | Main-loop load helpers | PENDING |
| R38+ | kernel_32 skeleton + dispatch | PENDING |
| R38+ | Full coverage + tuning | PENDING |

R29/R30/R31/R32/R33 are also considered "landed infrastructure" in
the skill file tracking, but this round is the first to reach the
**shared-memory tile** layer (vs register-tile-only layers).

---

## Repo state at end of round

- HipKittens: `ecbead9a` (was `78415fb0`)
- Primus-Turbo: advances after this doc-only commit
- Working tree: clean

DoD last run: 608 @ SHA 94fc3121. Not re-run this round (infra-only
commit, no dispatcher / can_handle / shared-code changes that could
affect DoD).

## Recommendation for R38 (concrete first step)

**Highest priority**: run the mfma_323264 vs mfma_1616128
microbench. This single round of validation de-risks the
4-6 round full port investment. Without this validation, the
entire R63-dm roadmap is speculative and could burn rounds for
zero score improvement.

If the R38 agent opts to skip validation and proceed directly to
step 2 (ST_32x64 swizzle design), they accept the risk that at
R41 (first metric test) the port may show no improvement or
regression, wasting 4 rounds of work.

**Fallback**: if R38 agent doesn't want to commit to either path,
accept the 957-962 plateau as final and spend remaining rounds on
backward-path improvements (dB / dA kernels currently at 52-76 dw
spill — significantly worse than forward 32-43 dw; correctness
validation via `bench_grouped_gemm_turbo.py` required per task
body rules). These don't move the metric but improve end-to-end
training throughput.
