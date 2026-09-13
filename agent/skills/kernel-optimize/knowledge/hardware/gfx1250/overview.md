# gfx1250 — Hardware Orientation for Kernel Authors

Target this guide when `target_gpu = gfx1250`. It is an **RDNA4-lineage, wave32, WMMA** part with a hardware **Tensor Data Mover (TDM)**, banked VGPRs, and split wait-counters — the vendored FlyDSL FMHA kernel calls it "gfx1250 (MI450)" (`fmha_kernel.py:9 @ 0c3937b6`). Build target `--offload-arch=gfx1250`.

The framing question for anyone arriving from CDNA is **"what do I have to unlearn?"** — the short answer: wave64→wave32, MFMA→WMMA (K=32, v16 operands), `buffer_load_lds`+`vmcnt`→TDM+`s_wait_tensorcnt`, a flat VGPR file→four banks with a mode-register switch, and a softmax/transcendental budget that is far tighter relative to the matrix core than on CDNA. Per-subsystem guidance and the hazard list live in [`optimization-directions.md`](optimization-directions.md).

**Provenance.** Two sources. The device-level facts come from a **platform report for the actual node** (`.../0911__fa_gfx1250_phase1/HARDWARE-ISSUE.md:43`, host `heliosr-1b114-c07-1`, PCI `0001:01:00.0`, SKU `M4500001`) — a node topology report, not a datasheet, but a direct reading of the part. Everything instruction-level is read out of an **off-main** FlyDSL gfx1250 FMHA kernel (commits `0c3937b6`, `7e93607c`, `2ace8def`, `3bdda5c2`), vendored at `primus_turbo/flydsl/attention/fmha_gfx1250/`. Those kernel files are **not in the working tree**; citations are written `file:line @ commit`, readable with `git show <commit>:<path>`. There is no public gfx1250 ISA table, so every row is tagged: **CONFIRMED** (platform report), **MEASURED** (quoted from that code), **INFERRED** (derived from it), or **UNCONFIRMED**.

## Quick Facts

| Item | Value | Basis |
|------|-------|-------|
| Wavefront size | **32** | CONFIRMED — `HARDWARE-ISSUE.md:43`; also `WAVE_SIZE = 32` (`fmha_prologue.py:31 @ 0c3937b6`) |
| Compute units | **256 CUs** | CONFIRMED — `HARDWARE-ISSUE.md:43` |
| Chiplet topology | **8 GFX domains / XCDs across 2 AIDs** (`AID0.XCD0..3`, `AID1.XCD0..3`) | CONFIRMED — `HARDWARE-ISSUE.md:43`, corroborated by per-die page-fault reports at `:225`, `:236` |
| L2 | 4 MB | CONFIRMED — `HARDWARE-ISSUE.md:43` |
| HBM | 432 GiB | CONFIRMED — `HARDWARE-ISSUE.md:43` |
| Power caps | PPT0 2500 W / PPT1 3200 W | CONFIRMED — `HARDWARE-ISSUE.md:44` |
| Waves per threadgroup (that kernel) | 4 (`BLOCK_SIZE = 128`) | MEASURED — `fmha_prologue.py:32-33 @ 0c3937b6` |
| VGPR file | **1024 per lane, exposed as 4 banks × 256**, selected by an `s_set_vgpr_msb` state register | MEASURED — "1024 shared VGPRs (256 per bank)" (`fmha_core_loop.py:14 @ 0c3937b6`); bank switching named at `fmha_core_loop.py:1228, 2076 @ 0c3937b6` |
| Matrix core | **WMMA**, atom `wmma_f32_16x16x32_bf16` (M=N=16, **K=32**) | MEASURED — `fmha_core_loop.py:395, 415 @ 0c3937b6`; `WMMA_M, WMMA_N, WMMA_K = 16, 16, 32` (`fmha_prologue.py:54 @ 0c3937b6`) |
| WMMA operand widths | A/B = `v16bf16` (32 B/lane), C/D = `v8f32` | MEASURED — `fmha_core_loop.py:288-290, 393-396 @ 0c3937b6` |
| LDS per CU | **320 KB** | CONFIRMED — `HARDWARE-ISSUE.md:43` |
| LDS actually reserved by one workgroup | 228 KiB | MEASURED — allocators at `fmha_kernel.py:155-168 @ 0c3937b6` |
| TDM segment granularity | **64 KiB**; a TDM copy may not cross a segment boundary | MEASURED — `LDS_SEGMENT = 0x10000` + comment (`fmha_kernel.py:153-155 @ 0c3937b6`) |
| Async global→LDS | **TDM** `tensor_load_2d`, two-dgroup descriptor | MEASURED — `fmha_prologue.py:642-643 @ 0c3937b6` |
| TDM completion counter | **`s_wait_tensorcnt`** — *not* `vmcnt`/`loadcnt` | MEASURED — `fmha_prologue.py:154-157 @ 0c3937b6`, `fmha_core_loop.py:675-677 @ 0c3937b6` |
| Wait counters | split: `s_wait_dscnt` / `s_wait_loadcnt` / `s_wait_tensorcnt` | MEASURED (tensorcnt, dscnt) / INFERRED (loadcnt) |
| Barriers | split `s_barrier_signal(-1)` / `s_barrier_wait(-1)` | MEASURED — `fmha_prologue.py:372-373 @ 0c3937b6` |
| LDS transpose-load | **`ds_load_tr16_b128` exists** (128-bit transposing LDS read) | MEASURED — `rocdl.ds_load_tr16_b128` (`fmha_core_loop.py:452 @ 0c3937b6`) |
| Transcendental | `v_exp_f32` ≈ **3-cycle issue**; all other VALU ≈ 1 cycle | MEASURED (as the kernel's own cost model) — `fmha_core_loop.py:171 @ 0c3937b6` |
| XCD count for the software remap | **8** | CONFIRMED — `HARDWARE-ISSUE.md:43`; matches `_NUM_XCDS = 8` (`fmha_kernel.py:819 @ 0c3937b6`) |
| Peak matrix / memory numbers | **not available** | UNCONFIRMED — do not compute a roofline against a guessed peak |

There is deliberately no peak-throughput table here. Unlike [`../gfx942/overview.md`](../gfx942/overview.md) and [`../gfx950/overview.md`](../gfx950/overview.md), no vendor figures have been confirmed for this part; use measured kernel time and the WMMA/VALU issue accounting in [`optimization-directions.md`](optimization-directions.md) instead of a fabricated denominator.

## Tile Granularity Is Fixed by the WMMA Atom

`wmma_f32_16x16x32_bf16` takes A/B as `v16bf16` — 16 bf16 = 32 B per lane, assembled from **two 128-bit loads** (`fmha_core_loop.py:338-341 @ 0c3937b6`) — and produces `v8f32`. For any tile-based kernel this forces:

- **`BLOCK_K` a multiple of 32**, **`BLOCK_M` / `BLOCK_N` multiples of 16**. The net-new WMMA GEMM asserts exactly this and zero-pads to it (`wmma_gemm_bf16.py:40, 112-113 @ 7e93607c`).
- A 16-lane-pair accumulator layout: the 8 f32 a lane holds are **8 consecutive N columns** at a tile-dependent base, and the **row is `lane & 15`** (INFERRED from the causal-mask address math: `_lane_lo = lane & 15`, column offset `(lane >> 4) * 8`, `fmha_kernel.py:906-915 @ 0c3937b6`). Consequence: a row-max needs exactly **one cross-lane op** — a `permlanex16` folding lanes 0-15 with 16-31 — not a `log2(32)` butterfly (`fmha_prologue.py:492-496 @ 0c3937b6`).

## Resource Model — What Differs From CDNA

- **VGPRs are banked, and the bank is machine state.** 1024 registers are reachable only 256 at a time through an `s_set_vgpr_msb` selector. Reading two operands from different banks forces a switch — **a stall class that does not exist on CDNA**, where the 512-Dword arch/Acc split is addressed directly. The kernel's whole schedule is organized to keep same-bank operands adjacent (`fmha_schedule.py:251-253 @ 0c3937b6`) and comments call out specific rewrites done purely to avoid a switch (`fmha_core_loop.py:1226-1230 @ 0c3937b6`).
  - **Caveat, stated plainly:** `_USE_BANK_HINTS = False` in that tree (`fmha_core_loop.py:228 @ 0c3937b6`), so every `set_vgpr_bank(...)` call is a no-op. The bank pinning is **design intent with disabled plumbing, not an achieved result** — the register-level analysis in the comments (per-bank free ranges, `SP_PAIR_BASE = 174`) describes a layout the compiler was never told about.
- **LDS is large but segmented.** 320 KB per CU (CONFIRMED). One workgroup reserves 228 KiB (`0x10000` × 3 segment-padded buffers + `0x9000`), and three of the four K/V ping-pong buffers are **padded up to a 64 KiB boundary purely so no TDM copy straddles a segment** (`fmha_kernel.py:153-168 @ 0c3937b6`). Budget segment alignment *before* tile size: the padding waste here is ~64 KiB and was accepted.
- **No AccVGPR pool, no MFMA accumulator shuttling.** WMMA accumulators are ordinary VGPRs (`v8f32`); the `V_ACCVGPR_READ/WRITE` discipline from CDNA has no analogue.
- **Occupancy:** no confirmed allocation quanta for this part. The reference kernel runs **one threadgroup of 4 waves** and spends the entire register and LDS budget on it, so it is a single-occupancy design by construction — do not port a CDNA occupancy-tier calculation across.

## Default Starting Configs

- **Threadgroup:** 128 threads (4 × wave32) is the shape the only known reference kernel uses; `block=(32,1,1)` for a one-warp-per-output-tile GEMM (`wmma_gemm_bf16.py:86 @ 7e93607c`).
- **Tiles:** start at `BLOCK_M = BLOCK_N = 128`, `BLOCK_K` a multiple of 32. Anything not on the 16/16/32 grid must be zero-padded.
- **Staging:** TDM double buffer (K ping/pong + V ping/pong), each buffer 64 KiB-segment aligned.
- **Scheduling:** expect to hand-schedule. The reference kernel needs ~119 `llvm.amdgcn.sched.barrier(0)` calls, `amdgpu-expert-scheduling-mode`, and a `WAVE_SCHED_MODE` setreg to hold its interleave (see [`optimization-directions.md`](optimization-directions.md) §6).
- **Do not import CDNA knobs.** `sched_mfma` / `s_setprio` ratios are wave64/MFMA-specific and mean nothing here.

## Relationship to gfx942 / gfx950 (CDNA)

gfx1250 is not "CDNA with fewer lanes". Porting a CDNA kernel is a rewrite of the data-movement and scheduling layers, not a retarget:

| Dimension | CDNA (gfx942/gfx950) | gfx1250 | What breaks |
|---|---|---|---|
| Wave | 64 | 32 | every lane-index formula, every cross-lane reduction depth |
| Matrix | MFMA, AccVGPR accumulators | WMMA `16x16x32`, plain-VGPR `v8f32` accumulators | tile granularity (K must be ×32), operand packing (`v16bf16`) |
| Async copy | `buffer_load_lds` + `vmcnt` | TDM `tensor_load_2d` + `s_wait_tensorcnt` | the whole prefetch/fence structure; a `vmcnt` wait does **not** observe TDM |
| Barrier | `s_barrier` | split `s_barrier_signal` / `s_barrier_wait` | you can now signal early and wait late — restructure, don't transliterate |
| Registers | flat 512-Dword pool, arch/Acc split | 1024 in 4 banks + mode-register switch | a new stall class; operand co-location matters |
| LDS transpose | gfx950 `ds_read_*_tr` only | **`ds_load_tr16_b128` present** | gates that refuse non-gfx950 on transpose-load grounds are wrong here |
| Softmax headroom | generous relative to MFMA | ~6-9 cycles of VALU shadow per WMMA | at D=128 the transcendental stream, not the GEMMs, sets the pace |

The cross-generation CDNA outline in [`../overview.md`](../overview.md) covers gfx942↔gfx950 only; nothing in its porting checklists transfers to this part unchanged.
