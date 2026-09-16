# Tuning the dense attention kernels on gfx1250

What is safe to change, what is a cross-kernel contract, and what has already been
measured. Written for the Llama-3.1-8B shape (`b=4 s=8192 hq=32 hkv=8 d=128 bf16 causal`),
but the structure applies to any dense shape on this backend.

Scope: `primus_turbo/triton/attention/attention_kernel.py` and
`primus_turbo/pytorch/kernels/attention/attention_triton_impl.py`, reached through
`DenseAttnFwdTritonBackend`. This is the only dense attention backend that runs on
gfx1250 — aiter reaches CK (CDNA-only, and its backward rejects the call at runtime after
the forward has succeeded), FlyDSL and HipKittens are gfx950, and Gluon is gfx950 and
forward-only.

---

## 1. Where the time goes

Measured at the shape above, one forward + backward, `turbo:TRITON`:

| kernel | ms | share |
|---|---:|---:|
| `_bwd_kernel_dkdv` | 35.922 | **61.5%** |
| `_bwd_kernel_dq` | 11.745 | 20.1% |
| `attn_fwd` | 9.939 | 17.0% |
| `_bwd_preprocess_use_o` | 0.063 | 0.1% |

The backward is 73–76% of the time, and that holds across a 7× mask-sparsity range — it
is not an artifact of the dense-causal footing. `_bwd_kernel_dkdv` leads the second-worst
kernel by a factor of three.

**Why dkdv is the hard one.** dK/dV is naturally key-outer, so GQA divides the worker
count by `G = Hq/Hkv = 4`, while the axis that makes the kernel expensive — query length —
adds none back. Two consequences:

- A query-axis split is the lever the structure argues for — but **not** via
  `sequence_parallel`, which is not a knob at all (see §4). It would take a real 3-D grid
  plus an atomic or split-K reduction.
- A longest-first causal dispatch order, which pays in a query-outer kernel, is **exactly
  backwards** here.

Grid occupancy is not the lever: at this shape `_bwd_kernel_dq` launches
`(B·Hq, ⌈Sq/64⌉) = 16,384` programs (64/CU across 256 CUs) and `_bwd_kernel_dkdv` launches
`(B·Hkv, ⌈Sk/64⌉) = 4,096` (16/CU). Both amply fill the machine. Per-program efficiency
and per-CU occupancy are what move.

---

## 2. What is a contract, not a knob

### `FIXED_BLOCK_M` is a cross-kernel ABI

Three sites must agree on one value:

1. `attn_fwd` writes LSE into the shared `[B, Hq, Sq*2]` scratch at block offset
   `m * BLOCK_M * 2`.
2. `_bwd_preprocess_use_o` writes delta at `+ BLOCK_M` within the same block.
3. `attention_triton_impl._lse_delta_views` reconstructs that indexing on the host:
   `lse_idx = (row // FIXED_BLOCK_M) * (2 * FIXED_BLOCK_M) + (row % FIXED_BLOCK_M)`.

So LSE and delta **interleave per BLOCK_M, not per row**. Retiling the forward
independently of the backward preprocess does not fail loudly — it mis-reads delta, and
the gradients come out smoothly wrong, which is the failure mode a loose SQNR gate is
worst at catching.

If you want to tune this value, change all three together and add a test that reads back
a known delta. It is not in the default search space for that reason.

### What *is* free

| knob | scope | changes grid? | changes layout? |
|---|---|---|---|
| `num_warps` | per kernel | no | no |
| `num_stages` | per kernel | no | no |
| `waves_per_eu` | per kernel | no | no |
| `PRE_LOAD_V` | forward | no | no |
| forward `BLOCK_N` | inner k loop | no | no |
| `_bwd_kernel_dq` `BLOCK_N` | inner k loop | no | no |
| `_bwd_kernel_dkdv` `BLOCK_M` | inner q loop | **yes, since the per-row LSE index** | no — see below |
| `_bwd_kernel_dkdv` `BLOCK_N` | **its own grid** | yes, self-consistently | no |
| `sequence_parallel` | both bwd kernels | **do not touch — see §4** | — |

**`_bwd_kernel_dkdv`'s `BLOCK_M` used to be bound to the LSE ABI, and no longer is.**
This paragraph previously said it was NOT free; that has been stale since the per-row index
landed. History, because the trap is instructive: the old form read the packed scratch as
`2 * start_m + tl.arange(0, 2 * BLOCK_M)` and split the halves with two `tl.gather`s, which
silently required dkdv's `BLOCK_M` to equal `LSE_ABI_BLOCK`. At `BLOCK_M=128` it returned
lse-rows 0..63 concatenated with delta-rows 0..63 — no fault, no shape error, smoothly wrong
dk/dv.

It is now indexed per row against `LSE_ABI_BLOCK`
(`primus_turbo/triton/attention/attention_kernel.py:1621-1625`, with the reasoning at
`:1611-1620`), which frees `BLOCK_M` and is what permits an asymmetric dk/dv tile. The two
cross-lane gathers are gone from this kernel.

`tl.gather` still appears at `:1952-1953`, but that is `_attn_bwd_dq`, whose `BLOCK_M` is
`FIXED_BLOCK_M` and is not swept. Do not read those two lines as evidence that the dkdv tile
is still constrained.

Note that turbo's backward is **split into separate dq and dkdv kernels with their own
grids**. That is a real structural advantage over a one-kernel backward: it does not have
the `BLOCK_N1`/`BLOCK_M2` grid-sharing trap described in §4.

---

## 3. How to drive it

The autotune lists ship with exactly one config each. `PRIMUS_TURBO_ATTN_TRITON_TUNE`
opens them up without changing the default for any existing caller:

```bash
# baseline — the shipped config
python3 tools/gfx1250/tune_attention.py --shape llama31-8b

# one candidate, pinned: Triton compiles exactly this and benchmarks nothing else
python3 tools/gfx1250/tune_attention.py --shape llama31-8b \
    --tune "fwd:num_stages=2;bwd:num_warps=2"

# a sweep, one subprocess per candidate, resumable ledger
python3 tools/gfx1250/sweep_attention.py --shape llama31-8b-s4096 \
    --axis bwd:num_warps=1,2,4,8 --axis bwd:num_stages=1,2 \
    --baseline --ledger out/round4.jsonl

python3 tools/gfx1250/sweep_attention.py --report out/round4.jsonl
```

Spec grammar: `key=value,...` applies to both halves; `fwd:...;bwd:...` splits them;
`sweep` expands to the curated grid. Unknown keys raise rather than being ignored — a
typo that quietly measured the default would read as "this knob does nothing".

**Pin one config per measurement.** Letting `@triton.autotune` benchmark a list means the
number you get is the best of the list, which is what you want in production and not what
you want in a round.

**Use the proxy shape for sweeps.** `llama31-8b-s4096` is exactly ¼ the FLOPs and
preserves the causal block structure, the GQA 4:1 reduction and the occupancy regime. It
does not exercise the long K-loop trip count, so **confirm any winner on `llama31-8b`
before it becomes champion**.

---

## 4. Traps

### Fast and silently wrong

The canonical example is from aiter's one-kernel backward, where the launch grid is
`(num_k_heads, cdiv(seqlen, BLOCK_N1), batch)` and that *same* grid serves both the dk/dv
half (tiled by `BLOCK_N1`) and the dq half (tiled by `BLOCK_M2`):

| config | ms | dq | dk | dv | |
|---|---:|---:|---:|---:|---|
| baseline | 27.744 | 52.25 | 52.31 | 52.73 | ok |
| `BLOCK_N1=256` alone | **21.110** | **9.59** | 52.31 | 52.73 | **wrong — looks like 1.31×** |
| `BLOCK_M2=64, BLOCK_N2=64` | **21.773** | **9.59** | 52.31 | 52.73 | **wrong, from the other side** |
| `BLOCK_N1=256, BLOCK_M2=256` | 21.639 | 52.25 | 52.31 | 52.73 | ok — a real 1.28× |

Raising `BLOCK_N1` halves the grid, so the dq half covers half the query rows. **dk and dv
stay perfect throughout.** A check that looks only at the output, or only at dk/dv, does
not merely miss this — it *rewards* it.

turbo's split-kernel backward does not have this specific coupling, but the lesson
generalises: any knob that appears in a grid expression and in a tile expression must be
changed as a set.

**Therefore: SQNR is computed separately for `out`, `dq`, `dk` and `dv`, against an exact
fp32 reference, over the full tensor.** Threshold 50 dB. The bf16 round-trip floor at this
shape is ~55.6 dB and every correct backend lands at 52–54 dB, so 50 dB leaves about 2 dB
of margin on the tightest tensor. Sampling is not allowed: dk/dv accumulate across the GQA
group, so a sampled check under-counts group contributions.

For reference, the existing bench's max-abs check returns an identical `8.011e-03` for
every backend and never looks at gradients. It is not a weaker gate; it is no gate.

### Silently ignored config

`@triton.autotune` builds its config list **at import**, and aiter's `_get_config` is
`@functools.lru_cache`'d. A spec that arrives late, or a key that did not parse, leaves the
default in place. The signature is a sweep in which every candidate returns the same time
to within noise — which reads as "this knob does nothing", not as a broken harness. A
prior backward sweep returned 27.45–27.49 ms for *every* config including grid-changing
ones and was only caught afterwards.

`tune_attention.py` asserts the config that came back is the config that was set, before
any timing.

### Known cliffs

From aiter's kernel at this shape, as guardrails for range selection:
`BLK_SLICE_FACTOR=4` → 151.6 ms; `waves_per_eu=4` → 120.6 ms; `BLOCK_M1=16` → 91.1 ms;
`BLOCK_M2=64, BLOCK_N2=128` → compilation error. Ranges are not symmetric around the
default and the penalties are large.

### A winner at the edge of a range

Means the range was the constraint, not the hardware. `sweep_attention.py --report` says
so explicitly and names the direction to extend. Do not leave this to whoever reads the
table; an earlier campaign on this codebase left +18% on the floor exactly here.

### Benchmark over-fitting

Config caches keyed on activation tensors have a real-training hit rate of **zero** —
Q/K/V/dO are fresh every iteration. A microbenchmark win that depends on such a cache does
not transfer. Every accepted round needs a real-training transfer check.

---

## 5. Measured results on this part (2026-09-13) and the priors behind them

**Measured, production shape, VR-throttled card (1100 MHz ceiling):**

| config | fwd ms | bwd ms | total ms | TFLOP/s |
|---|---:|---:|---:|---:|
| shipped default | 10.781 | 48.537 | 59.319 | 129.7 |
| **`fwd:num_stages=2; bwd:num_warps=2`** | **4.16** | **31.5–32.2** | **35.7–36.4** | **212–216** |
| torch flex (the path to beat) | 7.506 | 23.831 | 31.337 | 245.6 |

**1.66× over the shipped config from two integers**, at unchanged SQNR. The forward is now
*ahead* of flex (4.16 vs 7.51 ms); the whole remaining gap is the backward.

Measured non-results, so nobody re-derives them:

- **`_bwd_kernel_dkdv` does not spill** at num_warps 2, 4 or 8 (`vgpr_spill_count: 0`,
  zero `scratch_*` ops). It spills catastrophically only at num_warps=1 (512 spilled VGPRs,
  326 scratch ops), which is why that arm measures 47.8 ms. So dkdv's cost is per-program
  efficiency, not register pressure — read the counts from the `.amdgcn` metadata
  (`.vgpr_count` / `.vgpr_spill_count`), not the Triton cache JSON, which no longer carries
  `n_spills` in this version.
- **Skipping the causal mask on non-diagonal blocks is a 9% REGRESSION** (bwd 31.5 → 34.4 ms).
  The condition is uniform so it compiles to a scalar branch, but the branch costs more than
  the `[BLOCK_M, BLOCK_N]` compare-and-select it avoids — presumably by breaking the
  pipeliner. Measured, reverted.
- **Tightening dkdv's `lo` and folding `log_p_scale` into the per-row term are both
  perf-neutral** (within 1–2% run-to-run). The `lo` bound really is one `BLOCK_M` block too
  loose — dkdv subtracts an extra `BLOCK_M` before flooring where the sibling `_bwd_kernel_dq`
  uses a tight `cdiv`, so every `start_n >= 1` spends one iteration on an all-`-inf` tile,
  ~1.5% of iterations — but removing it does not show above noise. Correctness-neutral
  cleanup, not a performance lever.
- `waves_per_eu`: 0 is best, 1 neutral, **2 and 4 fall off a cliff** (54.9 and 104.4 ms).
- Backward `num_stages` > 1 is consistently worse (39.3 ms at 2, 39.5 at 3), unlike the
  forward where 1 → 2 is the single biggest win in the whole campaign.

## 5b. Priors worth starting from

All measured on gfx1250 at this shape, on aiter's Triton MHA (a different kernel, same
arch, same "config copied from another arch" situation):

- forward `num_stages` 1 → 2: **6.665 → 3.234 ms, 2.06×** on its own.
- backward `num_warps` 4 → 2: 27.744 → 21.075 ms, 1.32× on its own.
- Together: 34.111 → 24.347 ms, **1.401×**, with SQNR identical to four decimal places on
  all four tensors and 33% less peak memory.
- The two knobs **interact**: the one two-knob forward combination tried
  (`PRE_LOAD_V=False` with `num_stages=2`) was slightly worse than `num_stages=2` alone.
  Do not assume separability.

Why this space is untouched on both kernels: turbo's configs were inherited verbatim from
the upstream ROCm perf-kernel, which targets **wave64 / MFMA / 64 KB LDS**. gfx1250 is
**wave32 / WMMA / ≥256 KB LDS**, so `num_warps=4` is 128 lanes here rather than 256, and
`num_stages=1` is no software pipelining at all on a part whose headline feature is an
async data mover. aiter's case is starker still: its `gfx1250-MHA-DEFAULT.json` is
byte-identical to its gfx950 file.

The AMD-Triton knobs `waves_per_eu`, `matrix_instr_nonkdim` and `kpack` appear nowhere in
turbo's kernel. Of these, `matrix_instr_nonkdim` is a CDNA MFMA knob and is expected to do
nothing here — moving it 16 → 32 on aiter changed the backward by 0.1%, which is the
copy's fingerprint rather than a result.

---

## 6. Candidates that need kernel work, not config

Ordered by expected value. None of these have been measured on this kernel; each needs its
own round.

1. **Fold `log2(e)` into the softmax scale.** The kernel is already on `exp2`, but computes
   `exp2(q_shifted * RCP_LN2)` — a per-element multiply across the whole `[BLOCK_M,
   BLOCK_N]` score tile. Folding `RCP_LN2` into the QK scale and tracking the running max
   in log2 space removes that multiply entirely. At 64×64 that is 4,096 VALU ops per tile.
   Hand-written gfx1250 kernels keep a pre-scaled constant in an SGPR for exactly this.
2. **Raise the WMMA:VALU ratio.** On gfx1250 `v_exp_f32` is a 3-cycle-issue
   transcendental and one `wmma_f32_16x16x32_bf16` gives roughly a 4–6 cycle shadow. The
   exp count per WMMA is `256 / (D_qk + D_v)`, so D=128 sits at **1.0 exp per WMMA** —
   inside the 1–2 budget on that metric alone. The pressure comes from the rest of the
   softmax VALU stream: cycle-weighting the whole thing puts D=128 at ~8.4 cycles per
   WMMA against a 6–9 window, i.e. **softmax-bound by roughly 1.2–1.4×**, with exp itself
   about 36% of it. So softmax is the likely bottleneck at this head dim, but by a modest
   factor, not an order of magnitude — budget effort accordingly.
   **Units trap, worth stating because it is easy to get wrong by 16×:** on wave32 a
   32-element row segment is *one* `v_exp_f32`, not 32. Counting score-matrix elements
   against WMMA instructions overstates the pressure by the wave width.
   Larger `BLOCK_N` raises the ratio; so does skipping the running-max rescale on
   fully-unmasked tiles. See `agent/skills/kernel-optimize/knowledge/hardware/gfx1250/`
   for the derivation.
3. **Do NOT add the standard XCD remap — it would make things worse.** The usual
   round-robin→chunked remap assumes nearby workgroups land on different XCDs. They do not
   here. HIP dispatches x-fastest, so linear pid = `pid0 + gridDim0 * pid1`, and `gridDim0`
   is a multiple of 8 in both backward kernels (`b*hq = 128`, `b*hk = 32`). With 8 XCDs that
   makes `XCD = pid % 8 = off_hz % 8`, so **all 128 tiles sharing a (batch, head) already
   land on one XCD for free** — exactly the locality the remap is meant to buy. Chunking
   would scatter them across all eight. The accidental alignment is worth protecting: if a
   future change makes `gridDim0` a non-multiple of 8 it evaporates silently.
   (`NUM_XCDS = 8` is confirmed from node topology: 8 GFX domains across 2 AIDs, 256 CUs,
   wave32, 320 KB LDS/CU, 4 MB L2.)
4. **Price the fp32 gradient buffers.** `dense_backward` passes `dq/dk/dv=None`, so all
   three are allocated fp32 and autograd casts them back to bf16 on the way out (visible
   as `bfloat16_copy_kernel` ×3, 0.182 ms). Not a correctness bug, but 2× the gradient
   bytes plus three casts plus fp32 zero-fills. Whether the fp32 buffer is load-bearing for
   the atomic dQ accumulation or merely inherited must be measured before changing it.
5. **Tile shape constraints if you do retile.** The WMMA atom is
   `wmma_f32_16x16x32_bf16`: `BLOCK_K` should be a multiple of **32**, `BLOCK_M`/`BLOCK_N`
   multiples of **16**.
6. **ISA audit, cheap and possibly upstream-fileable.** (a) Is the backend emitting
   `ds_load_tr16_b128` — gfx1250 *does* have a transposing 128-bit LDS read — for the PV
   V operand, or a software transpose? (b) Is it emitting blanket `s_wait_loadcnt 0x0` /
   `s_wait_dscnt 0x0` where a partial count would do, and is `s_wait_tensorcnt` used at all?
   gfx1250 has split wait counters, so a single full wait is a real loss.

---

## 7. Hardware and environment notes that change how you measure

- **The card may be VR-throttled.** A measured state on this node: `MAX_CLK` 1100 MHz where
  the same card held 1699–1703 MHz six days earlier — a factor of ~1.65. Measured bf16
  compute roof in that state: **1002.7 TFLOP/s**. Check the clock before trusting any
  absolute figure, and prefer relative comparisons measured in the same session.
- **Attention is clock-limited here, not power-limited**: 1067 MHz of 1100 (97%) at 1227 W
  of a 2500 W cap (49%), `GFX_ACTIVITY` 100%, `UMC_ACTIVITY` 0–1%. So latency-hiding
  optimisations should be scored at face value — the CDNA rule that "a filled slot costs
  clock" does not carry over.
- **Do not use hipBLASLt as a roof or a GEMM reference on this image**: 91.5 TFLOP/s against
  Triton's 1002.7 on the same tensors in the same process.
- **Never run `rocprofv3` PC sampling.** Three attempts, three GPU faults, zero rows of
  output; one wedged the driver to `wait for reset ack` and needed a reboot. Shrinking the
  workload 32× made it *worse*.
- **Health checks must not shell out to `rocm-smi`** — a wedged card leaves tasks stuck in
  `amdgpu_info_ioctl`, which is what `rocm-smi` calls, so it hangs rather than reporting.
  Grep `dmesg` for `MES(` / `GPU Hang` / `wait for reset ack` instead.
- **The profiling surface is thin**: 51 hardware counters, 0 derived metrics, and no byte,
  sector or cache-line counter at any level, so roofline/bound analysis is not available.
  Prefer many cheap A/B measurements over few expensive profiles. Requesting a nonexistent
  counter alongside a valid one exits 0 and writes a well-formed CSV with the missing
  counter's column simply absent — check set-equality on the returned counter names.
- **`flash_attn_func`'s backward is flaky at `B > 1` under flex**, ~2 runs in 12 (`rc=139`,
  a clean process kill). Run each candidate in its own process and treat `rc=139` as a
  retry, recording the count.

---

## 8. Testing

`tests/pytorch/ops/test_attention.py` carries the backend's gating tests and a
forward+backward accuracy test against the reference. These are marked
`@pytest.mark.gfx1250`, which exempts them from the blanket gfx1250 skip in
`tests/conftest.py` — everything else in the suite is still skipped on this arch.

```bash
pytest tests/pytorch/ops/test_attention.py -m gfx1250 -v
```

Mark any new test that is validated on gfx1250 the same way. A test that is not marked
will be collected and skipped there, which is the safe default but means it guards nothing.
