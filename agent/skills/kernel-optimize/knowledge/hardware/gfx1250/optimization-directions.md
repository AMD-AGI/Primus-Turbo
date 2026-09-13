# gfx1250 — Optimization Directions

Per-subsystem guidance for writing and tuning gfx1250 kernels. Read [`overview.md`](overview.md) first for the hardware budget, the provenance rules, and the citation convention.

Instruction-level observations come from an **off-main** FlyDSL gfx1250 FMHA forward kernel (`primus_turbo/flydsl/attention/fmha_gfx1250/`, commits `0c3937b6` / `7e93607c` / `2ace8def` / `3bdda5c2`) — files that are **not in the working tree**. Citations read `file:line @ commit`; open one with `git show 0c3937b6:primus_turbo/flydsl/attention/fmha_gfx1250/fmha_core_loop.py`. Device-level facts come from a platform report for the actual node (`.../0911__fa_gfx1250_phase1/HARDWARE-ISSUE.md`). Claims are tagged **CONFIRMED** (platform report), **MEASURED** (quoted), **INFERRED** (derived), **UNCONFIRMED**.

This file is written for an engineer who knows CDNA. §8 (silent-garbage hazards) has no CDNA analogue and should be read before writing any TDM code.

---

## 1. Matrix Core (WMMA)

`wmma_f32_16x16x32_bf16` is the only matrix atom the reference kernel uses (`fmha_core_loop.py:395, 415 @ 0c3937b6`). M=N=16, **K=32**, wave32.

- **Operands.** A and B are `v16bf16` — 16 bf16 = 32 B per lane — and are assembled from **two 128-bit loads** each (`_pair_k_frags`, `fmha_core_loop.py:338-341 @ 0c3937b6`). C/D are `v8f32` (`fmha_core_loop.py:288-290 @ 0c3937b6`). Do not import the RDNA3 "v16 with lanes 16-31 mirroring 0-15" model or the RDNA4-gfx120x "v8 operand" model: here the width is v16 *because* K=32.
- **Tile granularity is not negotiable.** `BLOCK_K % 32 == 0`, `BLOCK_M % 16 == 0`, `BLOCK_N % 16 == 0`. The net-new GEMM asserts it and zero-pads otherwise (`wmma_gemm_bf16.py:40, 112-113 @ 7e93607c`). Padding contributes 0 to the dot product, so it is safe but it is real work — pick tiles on the grid.
- **Accumulator layout (INFERRED, but load-bearing).** Each lane's 8 f32 are **8 consecutive N columns** starting at `(lane >> 4) * 8 + tile_base`; the **row is `lane & 15`**. Read it out of the causal mask, which compares `row_pos` against `col_base + e` with `row_pos` built from `(lane & 15) − (lane >> 4) * 8` (`fmha_kernel.py:906-915 @ 0c3937b6`, column bases tabulated at `fmha_prologue.py:69-88 @ 0c3937b6`).
- **Therefore a row reduction is one cross-lane op.** Lanes 0-15 and 16-31 hold the same 16 rows in different column groups, so folding them with a single `permlanex16` completes the row-max — no `log2(32)` butterfly (`fmha_prologue.py:492-496 @ 0c3937b6`). Coming from CDNA wave64 this is the single largest simplification available in the softmax.
- **No accumulator pool.** WMMA C/D live in ordinary VGPRs. There is nothing to shuttle, and no `V_ACCVGPR_*` equivalent.

## 2. VGPR Banks — a Stall Class CDNA Does Not Have

1024 VGPRs per lane are exposed as **4 banks of 256**, selected by an `s_set_vgpr_msb` state register (`fmha_core_loop.py:14 @ 0c3937b6`). An instruction whose operands live in different banks forces a switch.

- **Co-locate operands, then order the schedule by bank.** The reference schedule groups tokens from the same bank ("MSB") consecutively *specifically* to minimize `s_set_vgpr_msb` switches (`fmha_schedule.py:251-253 @ 0c3937b6`), and V-tile LDS loads are emitted MSB-major so "no `s_set_vgpr_msb` switch is needed within the group" (`fmha_core_loop.py:2074-2076 @ 0c3937b6`).
- **Rewrites to dodge a switch are worth real algebra.** `max3(sp[7], tmp, old_max)` was replaced by `max3(sp[7], tmp, sp[0])` — provably equivalent, chosen because `sp[0]` is guaranteed in-bank "without the cross-bank penalty" (`fmha_core_loop.py:1224-1233 @ 0c3937b6`).
- **Softmax phases were reordered for the same reason:** all 16 `pk_fma` first, then all `exp`, to eliminate "EXP(bank0)↔PK_FMA(bank_msb) alternating MSB switches" caused by operands escaping to bank 0 (`fmha_core_loop.py:993-998 @ 0c3937b6`).
- **CAVEAT — this is intent, not a result.** `_USE_BANK_HINTS = False` (`fmha_core_loop.py:228 @ 0c3937b6`), so `set_vgpr_bank()` / `set_vgpr_bank_offset()` return their input unchanged (`fmha_core_loop.py:247-268 @ 0c3937b6`). Every bank annotation in that kernel, and the per-bank free-range analysis behind `SP_PAIR_BASE = 174` (`fmha_core_loop.py:230-240 @ 0c3937b6`), is **disabled plumbing**. Treat the bank model as a correct mental model whose enforcement path is untested — if you enable it, measure, do not assume.

## 3. LDS & the 64 KiB Segment Rule

- **Capacity.** **320 KB of LDS per CU** (CONFIRMED — `HARDWARE-ISSUE.md:43`). The reference kernel spends 228 KiB of it in a single workgroup: `K_a`/`K_b`/`V_a` each padded to `0x10000` plus `V_b` at `0x9000` (MEASURED — `fmha_kernel.py:153-168 @ 0c3937b6`). That is a deliberate single-occupancy design, not a capacity limit.
- **TDM cannot cross a 64 KiB segment boundary** (`LDS_SEGMENT = 0x10000`, `fmha_kernel.py:153-155 @ 0c3937b6`). Large LDS buffers must therefore be **segment-aligned even at significant padding cost** — here three buffers are rounded up from `0xC800`/`0x9000` to `0x10000`, wasting ~40 KiB, and that was the right trade. Budget alignment before you budget tile size.
- **Row pitch comes free from the TDM descriptor** (see §4): `dim0_stride = 200` elements against `dim0_valid = 192` gives a 400 B LDS row with no extra instruction (`fmha_kernel.py:395-398 @ 0c3937b6`). The kernel notes this leaves "2-way bank conflicts" at 400 B (`fmha_prologue.py:43-45 @ 0c3937b6`) — choose the pad for banks, not just for alignment.
- **`ds_load_tr16_b128` exists.** A transposing 128-bit LDS read, used for every V tile so PV gets its operand in WMMA lane order with no shuffle pass (`fmha_core_loop.py:447-452, 1540 @ 0c3937b6`). **This matters beyond performance:** the on-main FlyDSL FA gate at `primus_turbo/flydsl/attention/flash_attn_fwd.py:71-74` refuses anything but gfx950 *because it uses `ds_read_tr16_b64`* — that gate is over-restrictive for gfx1250, which has the b128 form.

## 4. TDM — the Async Copy Engine

TDM replaces CDNA's `buffer_load_lds`. `tensor_load_2d` takes a **two-dgroup descriptor** (`fmha_prologue.py:642-643 @ 0c3937b6`):

- **dgroup0** — `v4i32` = `[predicate, LDS byte offset, addr_lo, addr_hi]`.
- **dgroup1** — `v8i32` shape descriptor: `[config, dim0_valid << 16, oob_dim1 << 16, dim0_stride << 16, dim1_rows, stride_seq_elems, 0, 0]` (`_make_kv_dg1_with_oob`, `fmha_kernel.py:341-362 @ 0c3937b6`).

Consequences to internalize:

- **`dim0_valid` and `dim0_stride` are independent fields.** `dim0_valid` = elements read from global per row; `dim0_stride` = LDS row pitch in elements. That independence is how you get **free LDS row padding** — and, misused, how you read out of bounds (§8).
- **The 64-bit global address is split lo/hi and the hi word is OR'd with `0x80000000`** — a required tagging bit: `hi = hi_raw | -2147483648` (`fmha_prologue.py:573-579 @ 0c3937b6`). MEASURED; the *meaning* of the bit is UNCONFIRMED, but omitting it is not optional.
- **Factor the descriptor.** `dgroup1` is loop-invariant; build it once (`_build_tdm_dgroup1`, `fmha_prologue.py:556 @ 0c3937b6`) and patch only `lds_off` / `addr_lo` / `addr_hi` per issue (`_build_tdm_descs`, `fmha_kernel.py:291-310 @ 0c3937b6`).
- **Per-warp row partitioning is SALU work, not lane predication.** Varlen tails shrink `oob_dim1` per warp (`_per_warp_oob_dim1`, `fmha_kernel.py:322-339 @ 0c3937b6`) rather than masking lanes.

## 5. Synchronization — Split Counters, Split Barriers

- **TDM has its own counter.** Completion is `s_wait_tensorcnt(n)` (`fmha_prologue.py:154-157 @ 0c3937b6`, `fmha_core_loop.py:675-677 @ 0c3937b6`). **`vmcnt` / `loadcnt` do not observe TDM.** A CDNA prefetch loop transliterated with `s_waitcnt vmcnt(0)` will read an unfilled LDS buffer and produce garbage with no diagnostic.
- **Barriers are split:** `s_barrier_signal(-1)` then `s_barrier_wait(-1)` (`fmha_prologue.py:372-373 @ 0c3937b6`; a `s_barrier_wait(0xFFFF)` variant at `fmha_core_loop.py:684-685 @ 0c3937b6`). The point is that **you can signal early and wait late** — the reference kernel signals immediately after issuing a TDM and waits many WMMAs later (`fmha_core_loop.py:1652-1653, 1736 @ 0c3937b6`).
- **`s_wait_dscnt` lets you wait on LDS without waiting on the DMA engine** (`WAIT_DSCNT0`, `fmha_core_loop.py:123 @ 0c3937b6`; used at `fmha_core_loop.py:1697, 1919 @ 0c3937b6`). Three independent counters mean three independent fences — do not collapse them into one conservative wait the way a single `s_waitcnt 0` does on CDNA.
- **Note on the FlyDSL docs:** the vendored kernel uses **none** of `pipeline_fence` / `pipeline_fence_signal` / `pipeline_fence_wait`. See the correction applied to `../../backend/flydsl/optimization-directions.md`.

## 6. Scheduling — You Are Fighting the LLVM Scheduler

Holding a hand-built interleave on this part takes three separate mechanisms, all present in the reference kernel:

- **~119 `llvm.amdgcn.sched.barrier(0)` calls** across the four modules (MEASURED: `grep -c 'sched_barrier('` = 99 in `fmha_core_loop.py`, 13 in `fmha_prologue.py`, 9 in `fmha_kernel.py`, but the defining wrapper is `fmha_core_loop.py:213-221 @ 0c3937b6`). `mask=0` blocks reordering of *all* instruction classes.
- **`amdgpu-expert-scheduling-mode`** passed as an LLVM option on the launch (`fmha_kernel.py:3326 @ 0c3937b6`).
- **A `WAVE_SCHED_MODE` setreg** at kernel entry: `_setreg(2074, 2)` (`fmha_kernel.py:806 @ 0c3937b6`, helper at `fmha_prologue.py:140-145 @ 0c3937b6`).

The schedule itself is a **compile-time 160-row token table** — one row per WMMA, listing the VALU/LDS/TDM tokens to emit in the gap after it (`fmha_schedule.py:1-30 @ 0c3937b6`). If you need this level of control, build the table; do not try to coax it out of the scheduler.

## 7. Address Arithmetic and Work Placement

- **`nuw` unlocks buffer-offset folding.** `arith.addi(..., overflow_flags=nuw)` is documented in-tree as what "enables gfx1250 buffer offset folding" (`fmha_prologue.py:105-119 @ 0c3937b6`). Without it LLVM cannot prove the offset fits the buffer instruction's immediate and emits a `v_add` per load. Propagate `nuw` through the multiplies too (`_mul_nuw`, `fmha_prologue.py:117-119 @ 0c3937b6`) or constant folding drops it.
- **Software XCD remap for K/V L2 locality.** The kernel converts the hardware's round-robin workgroup→XCD assignment into chunked assignment so neighbouring tiles share an L2 slice: `new_wgid = (wgid % NUM_XCDS) * (num_wgs / NUM_XCDS) + wgid / NUM_XCDS` with `NUM_XCDS = 8` (`fmha_kernel.py:814-846 @ 0c3937b6`).
  - **`NUM_XCDS = 8` is CONFIRMED** for this part: the node topology is 8 GFX domains / XCDs across 2 AIDs (`AID0.XCD0..3`, `AID1.XCD0..3`), 256 CUs, 4 MB L2 (`HARDWARE-ISSUE.md:43`; corroborated at `:225` and `:236`, where page faults are reported per-die as `AID0.XCD1` / `AID0.XCD3` / `AID1.XCD2`). It is a node topology report rather than a datasheet, but it reads the actual device. Note the 2-AID split: 8 XCDs are *not* symmetric with respect to memory, so a chunked remap is the floor, not the ceiling, of what locality tuning can do here.
  - **The guard is load-bearing and fails silently.** The remap is applied only when `num_wgs > NUM_XCDS && num_wgs % NUM_XCDS == 0`; otherwise the integer division truncates, two distinct `wgid`s alias onto one `new_wgid`, and **tiles are silently dropped** — no fault, no hang, just missing output (`fmha_kernel.py:833-846 @ 0c3937b6`). Now that the count is confirmed the guard normally *passes*, which makes it easier to delete as dead code and easier to miss when a grid shape stops being a multiple of 8. Keep it.

## 8. The Number That Decides the Kernel: exp vs WMMA

`v_exp_f32` is a **3-cycle-issue** transcendental; every other VALU op the kernel uses (`pk_add`, `pk_fma`, `cvt`, `mov`) is 1 cycle (MEASURED as the kernel's own cost model, `fmha_core_loop.py:170-173 @ 0c3937b6`).

**The shadow one WMMA casts.** The dispatcher runs `GEMM1_VALU_PER_PHASE = 3` cycles per VALU phase and **2 phases per WMMA** (`fmha_core_loop.py:170-173 @ 0c3937b6`; "10 exp/MSB × 2 phases/WMMA", `fmha_core_loop.py:167 @ 0c3937b6`), and the schedule builder targets "~7 cy/WMMA … range 6-9 cy" (`fmha_schedule.py:256-262 @ 0c3937b6`). So:

> **≈ 4-6 cycles of VALU shadow per WMMA → a sustainable budget of 1-2 `v_exp_f32` per WMMA instruction.**

**The demand, derived.** For BLOCK_M × BLOCK_N over 4 waves (each wave owns 32 rows), per wave per KV tile:

```
QK WMMAs = (32/16)·(BLOCK_N/16)·(D_qk/32)
PV WMMAs = (32/16)·(D_v /16)·(BLOCK_N/32)
exp instrs = 32·BLOCK_N / 32 lanes          (one v_exp per S element per lane)

  →  exp per WMMA  =  256 / (D_qk + D_v)
```

Checks against the reference kernel (`BLOCK_M=BLOCK_N=128`, `D_qk=192`, `D_v=128`):

- WMMAs = 2·8·6 + 2·8·4 = **96 + 64 = 160** — MEASURED, the table is literally "160-row … (96 GEMM1 + 64 GEMM2 WMMAs)" (`fmha_schedule.py:3-4 @ 0c3937b6`).
- exps = `VPS_MSB_SP(32) × NUM_MSB(4)` = **128** per wave per tile — MEASURED (`fmha_core_loop.py:134-140 @ 0c3937b6`; 32 exp/MSB = `EXP_PER_MSB_TO_G2(8)` + `GEMM1_EXP_OPS(24)`).
- 128/160 = **0.8 exp per WMMA** = `256/(192+128)` ✓.

**Now the whole VALU stream, not just exp.** `ALU_PER_STAGE = [40,52,56,168,120,120,132,132]` sums to **820 VALU tokens** per wave per tile (`fmha_core_loop.py:168 @ 0c3937b6`). Cycle-weighted: `128 exp × 3 + 692 cheap × 1 = 1076 cycles` against 160 WMMAs = **6.7 cycles of softmax VALU per WMMA** — right inside the 6-9 window, just under the 7 target. The kernel fits, barely, and only because `D_qk=192` inflates the WMMA count.

**At D=128 it does not fit.** With `D_qk = D_v = 128` the WMMA count per wave per tile falls to `2·8·4 + 2·8·4 = 128` while the softmax work is essentially unchanged (it scales with `BLOCK_M × BLOCK_N`, not with D): `1076 / 128 ≈ 8.4` cycles of VALU per WMMA against a ~6-7 cycle shadow — roughly **1.2-1.4× over budget**. **At D=128 on this part, softmax is the bottleneck, not the GEMMs.** Design accordingly: spend effort on exp placement, packed (`v_pk_*`) rescale math and cheap-op elimination before touching the matrix loop.

**Two caveats, because the magnitude matters.**
- Do **not** state this as "~32 exps per WMMA, 16× over budget". 32 is the count of S *elements* per WMMA instruction (`4096 elements / 128 WMMAs`); on wave32 those 32 elements are **one** `v_exp_f32` instruction. Comparing element counts against an instruction budget overstates the gap by 32×. The real overrun at D=128 is ~1.2-1.4× on the *whole* VALU stream, of which exp is ~36% (`384 / 1076`).
- The closed form says the ratio worsens as head dim shrinks: `256/(D_qk+D_v)` is 1.0 at D=128, 2.0 at D=64 (at the edge of the budget), 4.0 at D=32 (hopeless). It improves for MLA-style asymmetric dims. The 4-6 cycle shadow itself is INFERRED from the kernel's phase budget, not from a published issue table.

**Mitigations the reference kernel uses**, all worth copying: `exp2` in log2 space rather than `exp` (`_rocdl_exp2`, `fmha_core_loop.py:44-45 @ 0c3937b6`); moving 32 of the 128 exps from the QK stage to the PV stage purely for load balance (`EXP_PER_MSB_TO_G2 = 8`, `fmha_core_loop.py:139 @ 0c3937b6`); a 16-op gap between each `pk_fma` and its dependent `exp` to hide transcendental latency without interleaving (`fmha_core_loop.py:131-133 @ 0c3937b6`); and interleaving `cvt` with the sum-tree to break `s_delay_alu` dependency chains (`fmha_core_loop.py:136-137 @ 0c3937b6`).

---

## 9. Silent-Garbage Hazards

Four failure modes on this part produce **wrong numbers or corrupted memory with no error, no fault, and no hang**. Each has been hit and worked around in the reference kernel. Check all four before debugging anything else.

**(a) TDM hardware row padding has a continuous-stream rotation bug.** TDM can insert pad bytes at a programmed interval while streaming. If the **pad interval does not divide the row exactly once**, the engine rotates data across row boundaries and **corrupts elements, silently**. MEASURED workaround: V (`dim0 = 128` elements, `pad_interval = 128` elements → exactly one pad per row) enables it as `_V_TDM_CONFIG = (1<<20)|(5<<22)|(7<<25)`; K (`dim0 = 192`) **cannot** — "QK_HDIM=192 is not a multiple of any power-of-2 pad_interval that fits in one pad per row, so we skip padding to avoid the continuous-stream rotation bug" (`fmha_prologue.py:90-95 @ 0c3937b6`, restated at `fmha_kernel.py:385-390 @ 0c3937b6`). **D=128 is a privileged shape here.** D=192 must fall back to `dim0_stride` padding (§4) instead.

**(b) Setting `dim0_valid` to the padded pitch reads past the tensor.** `dim0_valid` is how many elements TDM reads **from global**; `dim0_stride` is the LDS pitch. Setting both to the padded value over-reads every row, which only goes out of bounds on the **last row of the tensor** — i.e. it reproduces on the last head of the last token and nowhere else. MEASURED regression: "Previous `dim0_valid=200` caused OOB reads for the last head of the last token (16 extra bytes past valid K allocation)"; the fix is `dim0_valid = 192`, `dim0_stride = 200` (`fmha_kernel.py:395-398 @ 0c3937b6`).

**(c) A negative TDM `dim1` must be clamped to 0.** Varlen tails are handled by shrinking the **per-warp row count in SALU**, and the subtraction `remaining = total_rows − wave_id*8` goes negative for warps past the end. `_per_warp_oob_dim1` clamps with `maxsi(remaining, 0)` then `minsi(·, 8)` (`fmha_kernel.py:322-339 @ 0c3937b6`). Without the clamp the negative value is packed into the descriptor's `oob_dim1` field as a large unsigned row count. Note the shape of the fix: **no per-lane predication** — the row count is the mask.

**(d) The XCD remap without its divisibility guard drops tiles.** Covered in §7: `num_wgs % NUM_XCDS != 0` makes the formula alias two `wgid`s onto one, and the un-mapped tiles are simply never computed — no fault, no hang (`fmha_kernel.py:833-846 @ 0c3937b6`). `NUM_XCDS = 8` is confirmed, so the guard passes on every well-shaped grid and is easy to mistake for dead code; it is the only thing standing between an odd grid and missing output.

A fifth, inherited from the FlyDSL TDM gather path rather than measured here: the addr-lo-only descriptor update wraps a 4 GiB page and manifests as a **hang**, not a wrong answer — use the carry-safe `update_tensor_gather_descriptor_addr64` form.

---

## 10. Pitfall Checklist

- Waiting on `vmcnt`/`loadcnt` for a TDM copy — TDM completion is `s_wait_tensorcnt` only.
- Transliterating a CDNA `s_waitcnt 0` instead of using the three split counters and the split barrier to signal early / wait late.
- Choosing `BLOCK_K` that is not a multiple of 32, or M/N not multiples of 16 — the atom is `16x16x32`.
- Importing the RDNA3 (mirrored-lane v16) or gfx120x (v8) WMMA operand model; gfx1250 A/B are `v16bf16` because K=32.
- Doing a `log2(32)` butterfly for a row reduction when one `permlanex16` suffices given the accumulator layout.
- Assuming `set_vgpr_bank` annotations are doing anything — `_USE_BANK_HINTS = False` in the reference tree.
- Sizing LDS buffers without 64 KiB segment alignment, then debugging a TDM copy that straddles a boundary.
- Setting TDM `dim0_valid` to the padded pitch (§9b), or enabling hardware row padding on a row length that is not exactly one pad interval (§9a).
- Omitting the `0x80000000` tag when splitting the 64-bit global address.
- Dropping `nuw` on address arithmetic and paying a `v_add` per load.
- Applying the XCD remap without the `num_wgs % NUM_XCDS == 0` guard, or deleting that guard as dead code now that `NUM_XCDS = 8` is confirmed — an odd grid then silently loses tiles.
- Gating a kernel off gfx1250 because it "needs `ds_read_tr16_b64`" — gfx1250 has `ds_load_tr16_b128` (see `primus_turbo/flydsl/attention/flash_attn_fwd.py:71-74`).
- Bringing `sched_mfma` / `s_setprio` ratios over from CDNA, or expecting the LLVM scheduler to hold an interleave without `sched.barrier` + `amdgpu-expert-scheduling-mode` + `WAVE_SCHED_MODE`.
- Optimizing the matrix loop at D=128 before the softmax VALU stream (§8).
- Computing a roofline against a guessed peak — no confirmed peak-throughput numbers exist for this part.
