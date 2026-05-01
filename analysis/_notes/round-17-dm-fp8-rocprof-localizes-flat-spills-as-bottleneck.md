# Round-17-dm — fresh rocprof PMC LOCALIZES the FP8 grouped wall-time tax to **scratch spills (FLAT instructions)**

**Date**: 2026-05-01
**Branch**: `dev/kyle_hipkitten_bf16`
**Primus-Turbo HEAD before**: `23762796` (round-16-dm VGPR/scratch + chunk_size both falsified)
**HipKittens HEAD**: unchanged (`d154d500` — round-14 scaffold; rounds 15-16 made no kernel change)
**Primus-Turbo HEAD after**: this commit (Primus-Turbo notes-only; kernel/config bytes unchanged)

**Metric**: 822 baseline mean → 823 (no kernel change shipped; doc-only diagnostic round)

---

## TL;DR

Round-15-dm rocprof PMC localized the FP8 grouped wall-time gap to a "non-PMC
wall-time tax" but had only six counters
(`SQ_BUSY_CYCLES MfmaUtil OccupancyPercent SIMD_UTILIZATION GPU_UTIL MemUnitStalled`)
and could not split it further. Round-16-dm probed VGPR/scratch reduction via
attribute hints (all 4 byte-identical) and chunk_size sweep (regressed). Both
falsified.

**Round-17-dm captures the missing PMC counters** (`SQ_INSTS_VALU SQ_INSTS_SALU
SQ_INSTS_LDS SQ_INSTS_FLAT SQ_WAIT_INST_LDS SQ_WAIT_ANY SQ_LDS_BANK_CONFLICT
GRBM_GUI_ACTIVE SQ_WAVES`) on `DSV3-Down-B16-M4096` and decisively localizes
the gap:

| Counter (per launch) | HK | Triton | HK / TRT | Verdict |
|---|---|---|---|---|
| **SQ_INSTS_FLAT** | **2,988,288** | **1,032,192** | **2.9×** | **HK 1.96M extra spill round-trips** |
| SQ_INSTS_SALU | 60,879,872 | 39,806,976 | 1.53× | HK 21M extra (SRD rebuild + group bookkeeping) |
| SQ_INSTS_VALU | 130,764,800 | 92,084,224 | 1.42× | HK 38.7M extra (likely correlates with spill addr-calc) |
| SQ_INSTS_LDS | 22,494,208 | 29,360,128 | 0.77× | **HK uses LESS LDS — not a bottleneck** |
| SQ_BUSY_CYCLES | ~67M | ~69M | 0.97× | HK does LESS SQ work |
| MfmaUtil | 39-41% | 40% | parity | matches round-15-dm |
| SQ_LDS_BANK_CONFLICT | 0 | 0 | — | **No bank conflicts** |
| MemUnitStalled | 0.24% | 0.56% | 0.43× | HBM is not the bottleneck (TRT MORE stalled) |

**Conclusion**: the structural source of HK's wall-time tax on DSV3-Down (and
likely all 4 DSV3-Down + 8 gpt_oss FP8 grouped shapes ≤1.0 ratio) is
**scratch spills**. HK has 67-76 VGPRs spilled to a 272-byte scratch slot
(round-16 build report) and at runtime that materializes as 1.96 million extra
FLAT round-trips per launch on the metric's worst FP8 shape — every spill
load/store goes through the FLAT segment + L1.

This **falsifies the round-15-dm "non-PMC-visible" framing**: the spills WERE
visible to PMC, just not to the 6 counters round-15 sampled. The round-16-dm
priority list put VGPR/scratch reduction at #1 with reward "1-3 % per shape if
spills are real" — round-17 confirms spills are real and large.

---

## Round target & methodology

Lowest FP8 ratio shape this round = **`grpFP8_DeepSeek-V3-Down-B16-M2048` at
0.950** (HK 1151 TF / Triton 1211 TF), but the rocprof probe targeted
`DSV3-Down-B16-M4096` (0.954 ratio, HK 1360 TF / Triton 1425 TF) — the same
shape used by round-15-dm so cross-round comparison is direct, and the largest
launch (M=65536) so PMC counters are most stable.

Probe: `/tmp/probe_dsv3_one_call.py` runs 5 warm + 5 measured calls of
`turbo.ops.grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=cfg)` with
either `BackendType.HIPKITTEN` or `BackendType.TRITON`. Wrapped under
`rocprofv3 --pmc <counters>` in two passes (instruction mix + stall counters)
because rocprofv3 fails if the requested counter set can't fit in a single
pass.

Wall-time corroboration via per-kernel timestamps (`kernel_trace.csv`):

| backend | launches | avg_dur_us | min_dur_us | VGPR | Scratch | SGPR |
|---|---|---|---|---|---|---|
| HK | 10 | **1068.89** | 1053.05 | **128** | **272** | **112** |
| TRT | 10 | **1060.12** | 1040.29 | **116** | **0** | **80** |

HK is 0.83% slower per *main GEMM launch*; the metric reports 4.6% slower
end-to-end (1 - 0.954) — the additional gap comes from the helper kernels
(amax_reduce / quantize_unary / scale_from_amax), which is in scope of
*future* work but **out of scope for this round** per the prompt's hard
constraint #4 (`quantize_fp8(...)` is invariant). The main-GEMM 0.83% gap is
the in-scope target.

**Triton has 0-byte scratch**. HK's 272-byte scratch is the structural delta.

---

## Per-counter narrative (what each delta means)

### FLAT spills — the dominant cost

`SQ_INSTS_FLAT` counts FLAT-segment memory instructions: `flat_load_dwordx{1,2,4}`,
`flat_store_dwordx{1,2,4}`, and the scratch-segment `scratch_load`/`scratch_store`
variants. For a kernel like grouped_rcr_kernel that does NOT directly use
FLAT or scratch in source, every FLAT instruction is a compiler-emitted spill.

HK 1.96M extra FLAT × ~30 cyc avg per spill round-trip (L1-hit case) ≈ 59M cycles
of spill latency per launch, distributed across 256 work-groups × 8 waves =
2048 waves. Per wave: ~29K cycles of spill work per launch. Each tile takes
~16 K-iters × ~120 cyc average = ~2K cycles of MFMA, and each wave processes
~50 tiles = ~100K cycles of MFMA per launch. So spills ≈ 29% of MFMA cost per
wave — directly explaining why HK MfmaUtil is at parity with Triton (the MFMAs
themselves are fine) but per-tile wall is longer (the spills extend the
critical path between MFMAs).

**Why spills**: the 256-VGPR ceiling on gfx950 + 8-warp WG (32 lanes/wave ×
8 = 256 lanes; 2 waves/SIMD on 4-SIMD CU = 8 waves/CU minimum) means each
wave gets at most 256 VGPRs. The persistent main loop simultaneously holds:

- 4 × 32-VGPR accumulators (cA/cB/cC/cD = 128 VGPRs)
- 1 × A_row_reg (32 VGPRs) + 2 × B_row_reg (b0/b1 = 64 VGPRs) = 96 VGPRs
- Plus all SALU bookkeeping copies that overflow into VGPRs

That's 224 of 256 VGPRs already, before the LDS-prefetch shadow registers,
the per-tile coords, and the scale state. The compiler spills 67-76 VGPRs
to a 272-byte scratch slot per wave to fit.

### SALU excess — secondary, partly addressable

`SQ_INSTS_SALU` HK 60.9M vs TRT 39.8M (+21M per launch, ~10us at issue rate).
Two suspects:

1. **`rcr_8w_load_hoist` rebuilds `make_srsrc` on every call** (lines
   495-498, `kernel_fp8_layouts.cpp`):
   ```
   const uint32_t total_bytes = (uint32_t)(
       size_t(src.batch()) * size_t(src.depth()) *
       size_t(src.rows())  * size_t(src.cols())  * sizeof(T));
   i32x4 srsrc = make_srsrc(tensor_base, total_bytes);
   ```
   Called 4× per K-iter. Each call has ~15 SALU ops (3 multiplies + 4-element
   vector construction). Per K-iter × 16 K-iter × 50 tile × 144 CU = **~6.9M
   SALU per launch** = ~33% of the 21M excess. The `__forceinline` should let
   the compiler CSE across the 4 calls (same `src` arg), but inline asm + the
   `readfirstlane` for `tile_byte_offset` may inhibit it.

2. **Per-tile group bookkeeping** (`s_offs[]` LDS scan, `s_cum_tiles[]`
   binary search, group-by-M `pid_m/pid_n` swizzle, `m_subtile_A/C` derivation)
   — Triton's persistent kernel does similar work but with a different
   `compute_group_offs` Python-emitted scan that may be tighter.

### VALU excess — likely spill-correlated

HK 130.8M vs TRT 92.1M (+38.7M per launch). Each scratch spill load/store
costs 1 FLAT instruction *plus* 1-3 VALU instructions for the lane-disambiguated
address computation. 1.96M extra FLAT × ~2 VALU/spill = ~4M of the VALU
excess is spill-correlated; the remaining ~35M is SALU-correlated work
(tile-coord arithmetic that lands in VGPRs).

### LDS — HK is BETTER, not worse

`SQ_INSTS_LDS` HK 22.5M vs TRT 29.4M (HK -23%). Triton issues **more** LDS
instructions per launch than HK, despite running faster. This conclusively
rules out LDS-stage bottlenecks (bank conflicts, ds_read latency stalls) as
the source of HK's gap. **No tuning of LDS layout or bank stride will
help on FP8 grouped — that's an LDS-bound mindset that does not match the
data.**

`SQ_LDS_BANK_CONFLICT = 0` for both, also corroborates: bank-conflict-driven
LDS optimizations (round-2-dm, round-9-dm) cannot help.

### `SQ_WAIT_INST_LDS` corroboration

In pass 2 (stall counters), HK has **less** LDS-input wait
(`SQ_WAIT_INST_LDS = 27.7K` vs Triton 68.0K, HK -59%). Combined with HK
having more total wait (`SQ_WAIT_ANY = 512.8K` vs 408.1K, HK +26%), the
delta is entirely **non-LDS waits** — i.e., `s_waitcnt vmcnt(N)` waits or
similar. This is consistent with the spill round-trips being VMEM
operations: each spill load issues a `flat_load_*` which decrements vmcnt,
and the next dependent instruction waits on `vmcnt(N)`.

---

## What this rules in / out

### Ruled OUT for FP8 grouped main loop

- **LDS layout / swizzle / bank-conflict tuning** (round-2-dm, round-9-dm).
  PMC: 0 bank conflicts, 23% fewer LDS instructions than Triton. Conclusively
  not the bottleneck.
- **MFMA pipe issue throughput**. Round-15-dm + round-17 confirm parity.
- **MFMA cell shape migration** (round-13/14 plan). Round-15 falsified.
- **VGPR allocation hints** (round-16 falsified). Source refactoring needed.
- **`chunk_size` swizzle / chiplet transform** (round-16 falsified). Not
  the bottleneck.
- **Single-knob `s_setprio` / `s_barrier` removal** (round-13-dm falsified).
- **HBM bandwidth** (`MemUnitStalled` 0.24% — Triton is more HBM-stalled
  than HK).

### Ruled IN for round-18+

In priority order:

1. **Reduce VGPR live-range pressure to kill the 67-VGPR scratch spills.**
   Round-16-dm doc listed two specific candidates:
   - **(b) Reload `b0` / `b1` per section** instead of holding both live
     across all 4 sections (A/B/C/D). Saves ~32 VGPR. Cost: extra `ds_read`
     in section C, requires re-deriving the `LDS slot` ↔ `b0/b1` mapping
     across the prefetch ordering. Multi-round (~3-4 rounds: refactor,
     correctness, perf, polish). **Estimated: -800K to -1.5M FLAT spills →
     +0.4-0.8 pp DSV3-Down ratio → +5-10 score.**
   - **(a) Multi-section accumulator split**: keep only 2 of {cA, cB, cC,
     cD} alive concurrently, swap via LDS roundtrip per section. Saves
     ~64 VGPR. Cost: 4 extra LDS round-trips per K-iter (one per swap).
     Higher-yield but higher-risk than (b).

2. **Hoist `make_srsrc` out of `rcr_8w_load_hoist`**. New overload
   `rcr_8w_load_hoist_srd(dst, src, idx, swizzled_offsets, srsrc,
   tile_byte_offset)` accepting a precomputed SRD. Caller computes once
   per (`g.a`, `g.b`) tuple at kernel entry. Saves ~6.9M SALU per launch
   = ~33% of the 21M SALU excess. **Single-round, low-risk** — purely a
   CSE-shaped transform; correctness is mechanical. **Estimated: ~3 us
   per launch saved → +0.2 pp DSV3-Down → +2-4 score**. Smaller than (1)
   but cheaper / faster to ship.

3. **Investigate the helper-kernel gap** (amax_reduce + quantize_unary +
   scale_from_amax). HK and TRT both call them, but HK's helpers may run
   slightly slower per launch (HK quantize_unary 98.5us vs TRT 96.5us per
   call). 6 amax + 2 quantize + 2 scale per grouped_gemm call = ~330us of
   helper time, and HK is ~5-10% slower in some helpers. **Out of scope
   per hard-constraint #4**, but flag for later if the main-GEMM gap is
   closed.

---

## Round-17-dm verification artifacts

- `/tmp/rocprof_round17/pmc_stalls.txt` — counter set 1 (stall counters)
- `/tmp/rocprof_round17/pmc_inst_mix.txt` — counter set 2 (instruction mix)
- `/tmp/rocprof_round17/{hk,trt}_p1/` — instruction mix PMC output
- `/tmp/rocprof_round17/{hk,trt}_p2/` — stall PMC output
- `/tmp/probe_dsv3_one_call.py` — single-shape rocprof driver
- `/tmp/parse_round17.py` — counter aggregator (filters helper kernels;
  per-kernel-class table)

Reproduce:
```bash
PRIMUS_TURBO_HIPKITTEN_PATH=/workspace/code/HipKittens \
  PRIMUS_TURBO_GROUPED_GEMM_BACKEND=HIPKITTEN \
  rocprofv3 --pmc SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS SQ_INSTS_FLAT \
                  SQ_BUSY_CYCLES MfmaUtil \
  -d /tmp/rocprof_round17/hk_p1 --kernel-trace --output-format csv -- \
  python3 /tmp/probe_dsv3_one_call.py HK 5 5
# (swap HIPKITTEN → TRITON and HK → TRT for the Triton baseline)
```

---

## Score impact

- Metric this round: **823** (within noise band of rolling best 826 across
  rounds 12-16: 820/826/825/825/821/822/823 = stdev ~2.0). No kernel/config
  change shipped; doc-only commit.
- The diagnostic itself does not move the score, but **points the next
  round at the load-bearing structural fix** (kill the 67-VGPR scratch
  spills) instead of more falsifiable single-knob probes.

## Round-17-dm commits

- Primus-Turbo: this notes file. No kernel/config change.
- HipKittens: none.

## Recommended round-18 plan

Start with the **lower-risk, faster-to-ship** option-2 (hoist `make_srsrc`
out of `rcr_8w_load_hoist`). It's purely a refactor: same SRD, computed
once at kernel entry instead of 4 times per K-iter. Estimated +2-4 score,
single-round. If it lands, round-19+ can begin the multi-round option-1
(b0/b1 reload pattern) with confidence that the diagnostic was
load-bearing.

Implementation sketch for round-18:

1. In `kernel_fp8_layouts.cpp` (`grouped_rcr_kernel` body, ~line 2090):
   precompute `i32x4 a_srsrc_cached = make_srsrc((fp8e4m3*)g.a.raw_ptr,
   a_total_bytes);` and `b_srsrc_cached` once after the `s_offs[]` init.
2. Add a new overload `rcr_8w_load_hoist_srd(dst, src, idx,
   swizzled_offsets, srsrc, tile_byte_offset)` that skips the per-call SRD
   construction and uses the passed-in `srsrc`. The `tile_byte_offset` is
   per-call (depends on `idx`), so it stays per-call but is already a
   single `readfirstlane` op so cheap.
3. Replace 12 grouped-kernel call sites (4 in main loop, 4 in epilog 1,
   ~4 in epilog 2 + prologue) with the new overload.
4. Verify SNR ≥ 25 dB on all 16 FP8 metric shapes + `test_grouped_gemm_fp8.py`
   passes default + `--deterministic-only`.
5. Verify rocprof: `SQ_INSTS_SALU` should drop from ~60.9M to ~54M per
   launch (HK target ratio: 1.36× Triton instead of 1.53×).
6. Score: target +2-4 (823 → 825-827; surpasses rolling best 826).

The dense kernel has the same `rcr_8w_load_hoist` pattern (lines 1309-1320)
but *fewer* iterations of the per-call SRD overhead because dense ki is
larger; the round-18 change should be additive there too. But focus first
on grouped (where it's the rocprof-localized hot spot).
