# Round 18 — Triton vs HK ISA breakdown localizes 5-34x FLAT instruction excess

## TL;DR

Round 17 confirmed all FP8 grouped main-loop micro-knobs are saturated. Round 18
runs an instruction-class breakdown (`SQ_INSTS_*` family) over HK and Triton on
three representative gpt_oss shapes. **MFMA count is identical to the lane
between HK and Triton** (perfect 1:1 match across all shapes), but HK consistently
emits **5-34x more `SQ_INSTS_FLAT`** instructions than Triton. This is the
single largest structural difference and most likely accounts for the bulk of
HK's SQ_BUSY excess and 4-8pp MfmaUtil deficit identified in round 16.

## Data — instruction breakdown per kernel invocation (mean of 70 runs each)

### FP8 — `Down-B4-M4096` (worst case, ratio 0.834)

| counter         | HK         | Triton     | Δ (HK−TRT) | HK/TRT   |
|-----------------|------------|------------|------------|----------|
| SQ_INSTS_MFMA   | 4,521,984  | 4,521,984  | **0**      | **1.00×** |
| SQ_INSTS_FLAT   | 1,256,960  | 36,864     | +1,220,096 | **34.1×** |
| SQ_INSTS_LDS    | 3,306,752  | 4,521,984  | -1,215,232 | 0.73×    |
| SQ_INSTS_SALU   | 3,855,360  | 4,124,672  | -269,312   | 0.94×    |
| SQ_INSTS_VALU   | 12,865,792 | 12,251,136 | +614,656   | 1.05×    |
| SQ_INSTS (total)| 27,008,256 | 25,040,896 | +1,967,360 | 1.08×    |

### FP8 — `GateUP-B32-M4096` (B=32 sanity, ratio 0.863)

| counter         | HK            | Triton        | HK/TRT   |
|-----------------|---------------|---------------|----------|
| SQ_INSTS_MFMA   | 69,337,088    | 69,337,088    | **1.00×** |
| SQ_INSTS_FLAT   | 18,933,760    | 3,203,072     | **5.91×** |
| SQ_INSTS_LDS    | 50,512,640    | 69,337,088    | 0.73×    |
| SQ_INSTS_SALU   | 56,175,104    | 102,291,456   | 0.55×    |
| SQ_INSTS_VALU   | 194,148,096   | 188,903,424   | 1.03×    |
| SQ_INSTS (total)| 407,026,432   | 431,380,480   | 0.94×    |

### BF16 — `Down-B32-M2048` (HK wins this case, ratio 1.089)

| counter         | HK            | Triton        | HK/TRT   |
|-----------------|---------------|---------------|----------|
| SQ_INSTS_MFMA   | 70,778,880    | 70,778,880    | **1.00×** |
| SQ_INSTS_FLAT   | 4,362,240     | 835,584       | **5.22×** |
| SQ_INSTS_LDS    | 26,774,016    | 26,542,080    | 1.01×    |
| SQ_INSTS_SALU   | 34,397,696    | 63,627,264    | 0.54×    |
| SQ_INSTS_VALU   | 83,411,968    | 101,967,872   | 0.82×    |
| SQ_INSTS (total)| 193,759,744   | 220,798,976   | 0.88×    |

## Findings

1. **MFMA count is structurally fixed** (4.52M / 69.3M / 70.8M) — it depends only
   on `M·N·K` of the GEMM. Both HK and Triton compute exactly the same MFMA count,
   confirming neither kernel does extra/redundant compute.

2. **HK's `SQ_INSTS_FLAT` is 5-34x higher than Triton's**, and the ratio scales
   with the metric ratio gap (FP8 B=4 worst case → 34x; BF16 B=32 HK-wins case
   → 5x). FLAT instructions are HBM/global-class memory ops (`global_load_*`,
   `global_store_*`, `flat_load_*`, `flat_store_*`); they do *not* include
   buffer-class loads (`buffer_load_*`/`buffer_store_*`).

3. **HK has fewer LDS instructions** in FP8 cases (3.3M vs 4.5M for Down-B4-M4096),
   yet does more total work (more SQ_BUSY_CYCLES, lower MfmaUtil). Triton is
   issuing more LDS reads but *they overlap with MFMA better* — likely because
   Triton's persistent kernel uses a software-pipelined LDS prefetch pattern
   that lets MFMAs issue back-to-back during LDS waits.

4. **HK has *fewer* SALU instructions** in B=32 cases (HK 56M vs Triton 102M for
   FP8 GateUP-B32-M4096 — Triton has nearly 2x more SALU). This is interesting
   because rocprof breaks `_grouped_fp8_persistent_gemm_kernel` into more SALU
   work. Probably address arithmetic for the persistent grid scheduler. Triton
   extra SALU is "free" because it parallel-issues with MFMA.

## Hypothesis on the FLAT excess

HK uses three load/store paths in `kernel_fp8_layouts.cpp`:

- `rcr_8w_load_hoist<>` (lines 423-x): uses
  `llvm_amdgcn_raw_buffer_load_b128` → **BUFFER instructions**, counted under
  VMEM not FLAT.
- `kittens::store(g_c, ...)` (defined in `include/ops/warp/memory/tile/global_to_register.cuh`
  line 185): uses `llvm_amdgcn_raw_buffer_store_b64/b128` → **BUFFER instructions**,
  counted under VMEM not FLAT.
- K-tail path B: `raw_buffer_load_b128` → **BUFFER**, counted under VMEM not FLAT.

So the main load, prefetch, and store paths are all BUFFER. The FLAT excess
must therefore come from **a different code path** that wasn't caught by the
above search. Candidates (left for round 19 to localize via `att-perfcounters`
PC sampling):

1. **Argument struct loads** (`grouped_layout_globals_fp8 g`) — if any field
   is read with `global_load_*` rather than `s_load_*`, it counts as FLAT.
   With 768 tiles × 50 invocations = 38,400 tile instances, even a few loads
   per tile add up.
2. **C tile load for FUSED_KTAIL or RMW** — if the K-tail epilog reads a
   partially-accumulated tile back from HBM (round 1's path A approach), those
   are FLAT-class.
3. **`num_xcds`/`group_offs` pointer dereferences in the persistent scheduler** —
   indexing into per-group offset arrays may emit `global_load_*`.
4. **Generic `kittens::load(dst, sub)` (LDS-to-register)** — re-checking the
   library implementation: this should be `ds_read_*` (counted as LDS, not
   FLAT). But if the compiler couldn't prove `sub` was an LDS pointer, it
   could fall back to `flat_load_*`. Worth verifying via objdump.

## Score impact this round

Score this round: **798** (up from 794 yesterday baseline; well within ±3pp
noise band of best 795). Notes-only commit, no kernel change shipped.

## Action plan for round 19

1. **Disassemble `tk_fp8_layouts.cpython-312-x86_64-linux-gnu.so`** with
   `llvm-objdump -d --arch=gfx950` and grep for `global_load_*`/`global_store_*`
   instructions to localize the FLAT-emitting code path.
2. **Or** run `rocprofv3 --att-perfcounters SQ_INSTS_FLAT --att-perfcounter-ctrl 4`
   to PC-sample which kernel addresses emit FLAT, then map back to source via
   line-number metadata.
3. Once localized, the fix is to replace `global_load_*` / `global_store_*` calls
   with their `buffer_*` counterparts (using `make_srsrc` + `raw_buffer_*`
   intrinsics already available in `util.cuh`).

**Estimated impact if fix lands cleanly**: closing the 1.2-15M FLAT instruction
gap at ~2-4 cycles/instruction = 2-60M SQ cycles per kernel. At 768 tiles
parallelised across 256 CUs, that's ~10-200K SQ cycles per CU, roughly 5-100µs
of wall time — bracketing the 30µs HK-vs-Triton gap measured in round 15.
This is the first lever with quantitative evidence pointing at a specific code
path; should yield ≥1.5pp metric shift if the localization succeeds.

## Probe artefacts

- `/tmp/rocprof_round16/{hk,trt}{,_b32,_bf16}_inst_counter_collection.csv` —
  raw `SQ_INSTS_*` rocprof outputs.
- `/tmp/bench_one_shape.py`, `/tmp/bench_bf16_one.py` — hot-loop drivers.

## Round-18 commits

- Primus-Turbo: this notes file. No kernel change.
- HipKittens: none (no kernel change this round; round 19 will start with
  disassembly to localize the FLAT excess).
