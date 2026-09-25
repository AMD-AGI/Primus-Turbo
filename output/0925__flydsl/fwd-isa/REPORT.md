**Files produced**
- ASM disassembly: `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0925__flydsl/fwd-isa/asm/fmha_bf16_pertokenBf16_hd128_128x256_mask.s` and `.../asm/fmha_bf16_pertokenBf16_hd128.s`
- FlyDSL ISA for the prod config (bshd, gqa=4, causal, LSE on): `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0925__flydsl/fwd-isa/fly/fwd_m32x8_prod_causal_lse.s`. The full dump is in `.../fwd-isa/fly/dump/kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0/`.
- Compile-only driver: `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0925__flydsl/fwd-isa/fly/compile_fwd_isa.py`. It uses COMPILE_ONLY=1 and CPU tensors, with all HIP/ROCR devices hidden.

**Disassembly caveat:** `/opt/rocm/llvm/bin/llvm-objdump --mcpu=gfx1250` does not decode the ASM kernel's TDM instructions; the other ROCm objdumps checked (10.1 0811, 7.15, the container's) fail the same way. They show up as `.long 0xd0310000` (load, 27 of them) or `.long 0xd0314000` (store, 2 of them), followed by a bogus `v_illegal` / `v_cmp_ge_u16`. The third dword gives the descriptor registers: `0x7C7C4844` is s[68:71]/s[72:79] and `0x7C7C3C38` is s[56:59]/s[60:67]. I counted TDM ops from these words.

**Where the loops are**
- **ASM (mask kernel):** the main loop is lines 2970–9368, unrolled 2x.
  - Half A is lines 2970–6171; half B is lines 6172–9368.
  - Each half processes one 256-row KV tile (`s_addk_co_i32 s89, 0x100` once per half).
- **FlyDSL:** the body is traced twice (LO and HI warp types), each with a clean loop and a masked loop.
  - LO clean loop: lines 452–1077. LO masked loop: lines 1179–1959.

## Side-by-side

| item | ASM hd128_128x256_mask | FlyDSL m32x8 |
|---|---|---|
| waves/WG, threads | 4, 128 (`wv_tg=4`, `bdx=128`) | 8, 256 |
| grid | (hq 32, ceil(S/128)/2 = 32, b 4) = 4096 WGs; x = head | (ceil(S·gqa/256) = 128, hkv 8, b 4) = 4096 WGs |
| Q rows | 128 per Q-tile; each WG does 2 Q-tiles one after the other (t, then N−1−t); 32 rows per wave | 256 packed rows (64 seq × 4 heads, GQA packing); 32 per wave |
| KV tile per loop step | 256 | 64 |
| VGPR / SGPR | 1024 (uses `s_set_vgpr_msb` banks) / 106 | 445 / 98 (`waves_per_eu=2`) |
| LDS | 327680 | 327680 |
| waves per SIMD | 1 | 2 |
| GEMM form | S^T = K·Q^T (K = A, Q = B); O^T = V^T·P^T | same |
| WMMA per 256 KV per wave | 256 | 256 (4 iterations × 64) |
| LDS loads per 256 KV | 128 `ds_load_b128` + 128 `ds_load_tr16_b128` | same (32 + 32 per iteration) |
| TDM ops per 256 KV per wave | 4 | 8 |
| TDM wait | `s_wait_tensorcnt 0x4` | `s_wait_tensorcnt 0x0` every iteration |
| barriers per 256 KV | 1.5, split (signal early, wait later, with WMMA in between) | 4, signal immediately followed by wait |
| `s_wait_dscnt` | 17 per half, all `0x10`/`0x13` (16–19 loads kept in flight) | 11 per iteration, drains 0x0–0x1e |
| `s_setprio` | 0 | 0 |
| sched mode | `HW_REG_WAVE_SCHED_MODE = 2` | same |
| mode bits | `WAVE_MODE[24]`, plus `s_prefetch_inst` (wave 0) over ~48 KB of code | `WAVE_MODE[25]`, no prefetch |
| `v_exp` per 256 KV | 260 | 264 |
| exp argument | exp2; `v_pk_fma_f32 S·(scale·log2e) − m·(scale·log2e)`, packed | exp2; scale folded into Q in bf16; scalar `v_fmamk_f32` |
| row max | in-lane `v_max3`; 4 `v_permlanex16` per 256 KV | in-lane `v_max3`; 8 `v_permlanex16` per 256 KV |
| row sum | in-lane `v_pk_add_f32`; cross-lane reduce only in the epilogue | scalar/dual adds; `v_permlanex16` every tile |
| O rescale | every tile, ~64 `v_pk_mul_f32` per 256 KV, no branch | deferred (threshold 8.0, ballot + `s_cbranch_vccz`); 64 `v_pk_mul_f32` per 64 KV when it fires |
| causal mask | one loop; two in-loop scalar branches skip the per-element `v_cmp_lt_i32` + `v_cndmask` block on interior tiles | loop split three ways at runtime (masked-lo / clean / masked-hi); masked body has ~67 compares per iteration |
| pipelining | QK(j) interleaved with exp/cvt/rowsum of tile j−1 and the next K loads; S double-buffered in two register sets | phases fenced by `sched_barrier(0)`; ~250 VALU/trans instructions with no WMMA; the LO/HI ping-pong's `_named_barrier_pair` is a no-op |
| `v_nop` / SALU per 256 KV | 13 / 58 | 156 / 188 |
| grid order and balance | x = head, so consecutive WGs share a KV head; pairing t with N−1−t gives every WG equal work | x (packed Q tile) is fastest, dispatched light to heavy; work grows with x |
| epilogue | O: 16 `ds_store_b128` + 2 TDM stores; LSE: `v_log` + `ds_store_b32`; then branches back to start the mirrored Q-tile | O: 16 `ds_store_b128` + 16 `global_store_async_from_lds_b128`; LSE: `v_log` + 2 `buffer_store` |

## Key ISA evidence

**ASM: QK of the current tile interleaved with softmax of the previous tile** (0x68e0). v[92:99] is in half A's S register set; v56–v61 are in half B's (v[52:59], v[60:67]):
```
v_wmma_f32_16x16x32_bf16 v[92:99], v[160:167], v[8:15], 0
<TDM load s[68:71],s[72:79]>
ds_load_b128 v[192:195], v157 offset:43520
v_pk_add_f32 v[2:3], v[28:29], v[30:31]
v_exp_f32_e32 v56, v56
v_cvt_pk_bf16_f32 v28, v28, v29
```

**ASM: split barrier with work in the gap:**
```
s_wait_tensorcnt 0x4
s_barrier_signal -1
v_pk_add…; v_wmma…; v_cvt…; v_wmma…; v_wmma…
s_wait_dscnt 0x10
s_barrier_wait 0xffff
```

**ASM: rescale with no branch, packed FMA:**
```
v_fma_f32 v26, -v159, s102, v2
v_pk_mul_f32 v[224:225], v[26:27], v[224:225]
v_pk_fma_f32 v[92:93], v[92:93], s[102:103], v[2:3] neg_lo:[0,0,1] neg_hi:[0,0,1]
```

**ASM: Q-tile pairing (t with N−1−t):**
```
s_sub_co_u32 s5, s36, s34        ; s36 = ceil(S/128)-1
...
s_mov_b32 s34, s5
<2x TDM store 0xd0314000>
s_branch 46007                   ; back to +0x178 for the mirrored tile
```

**ASM: mask block skipped on interior tiles:**
```
s_cmp_lt_i32 s90, s92
s_cbranch_scc1 2383
```

**FlyDSL: one iteration of the clean loop, in order:**
- `s_wait_tensorcnt 0x0`, `s_barrier_signal -1`, `s_barrier_wait -1`
- ~30 SALU (TDM descriptor rebuild), then `tensor_load_to_lds` ×2
- `ds_load_b128` ×32
- 32 QK `v_wmma`
- `ds_load_tr16_b128` ×32
- `v_max3` ×30, then (`v_fmamk`, `v_exp`, `v_nop`) ×~64, then adds and `v_permlanex16`
- `s_cbranch_vccz`, then `v_pk_mul_f32` ×64
- 32 PV `v_wmma` with `v_cvt_pk_bf16`

**Instruction classes per 256 KV.** For ASM this is one half, including the mask block; about 1000 of its VALU are that block, which interior tiles skip.

| class | ASM | FlyDSL clean ×4 |
|---|---|---|
| WMMA | 256 | 256 |
| LDS | 256 | 256 |
| trans | 260 | 264 |
| VALU | 1776 (~750 on interior tiles) | 1008 |
| `v_nop` | 13 | 156 |
| SALU | 58 | 188 |
| TDM | 4 | 8 |

The two kernels issue about the same number of useful instructions and read LDS at the same rate. The 1.53x comes from how well the matrix pipe is kept busy: overlap, synchronization, prefetch depth and load balance.

## Top 8 differences, ranked

These estimates come from reading the ISA. None has been measured; each needs an A/B run on the card.

1. **Softmax does not overlap WMMA, and the cross-wave ping-pong is not wired (est. 20–35%).**
   - ASM runs one wave per SIMD and interleaves QK(j) WMMA with softmax of tile j−1, using two S register sets.
   - FlyDSL fences each phase with `sched_barrier(0)`. `_named_barrier_pair` is a no-op, there is no `s_setprio`, and the WG-wide barrier every 64 KV keeps the two waves on a SIMD in the same phase.
   - Fix (a): carry the previous tile's S (or P plus m/l) in the `_run_tiles` loop state. Emit `_qk_gemm(j)` and `_softmax(j−1)` in one region, and replace the fences with a `rocdl.sched_group_barrier` pattern. Masks: 0x008 WMMA, 0x002 VALU, 0x400 TRANS, 0x100 DS read.
   - Fix (b): build the ping-pong for real. Use `rocdl.s_barrier_signal_var(ptr, 2)` or `gpu.initialize_named_barrier` with `gpu.barrier(named_barrier=…)`, plus `rocdl.s_setprio`, and drop the per-iteration WG barrier.
2. **KV tile of 64 vs 256 (est. 5–10%).** FlyDSL pays for a barrier, a TDM wait of 0, descriptor SALU, permlanes, the rescale ballot and loop-carried moves every 64 KV.
   - Fix: `build_fmha_fwd_prefill_a16w16_m32x8(n_block=128/256)`; `_ensure_bshd_kernel` currently doesn't pass `n_block` through.
   - At `waves_per_eu=2` (512-VGPR cap, 445 used now), 128 is about the limit. 256 needs 4 waves and `waves_per_eu=1`, and the buffer managers currently raise when `num_waves != 8`.
3. **TDM prefetch is one 64-KV tile deep and drains to zero (est. 3–10%, depends on latency).**
   - Fix: `N_KV_PP=3` and `tdm_ops.tensor_wait(2)`.
   - Also make the hard-coded `% 2` and the curr/next pointer rotation general.
   - Lower `MIN_KV_BLK_BYTES`: a 64-row tile is 16 KB but each slot is floored to 64 KB + 64 KB, so three slots don't fit in 320 KB.
4. **Causal load imbalance and grid order (est. 2–5%).** ASM gives every WG equal work by pairing t with N−1−t; FlyDSL dispatches light to heavy.
   - Cheap fix: heavy first, `block_x = grid_x − 1 − block_id.x`.
   - Full fix: loop over the pair (t, N−1−t) inside the WG and halve `grid_x`.
5. **Packed FP32 softmax math (est. 2–4%).** Use `fmath.fma` and add on `vector<2xf32>` in `_softmax` and `_tree_reduce_multi` so LLVM selects `v_pk_fma_f32` / `v_pk_add_f32`.
6. **The rescale branch splits the loop body into separate basic blocks (est. 1–3%).** The scheduler can't move work across the `s_cbranch_vccz`. With `n_block` ≥ 128, set `ENABLE_DEFER_RESCALE=False` or use a select instead of a branch.
7. **Per-tile cross-lane row-sum and `v_nop` after each `v_exp` (est. 1–3%).**
   - Carry the per-lane partial row-sum and do one `peer()` before normalization/LSE. This is valid because m is already reduced across lane pairs.
   - The `v_nop`s go away once item 1 interleaves the exps.
8. **Epilogue overlap (est. ≤1–2%).** ASM's TDM O-store overlaps the mirrored Q-tile. This comes for free with the in-WG pairing in item 4.

**Not part of the gap**
- GQA packing is a FlyDSL advantage: 32 KB of LDS fill per 512 WMMA vs ASM's 128 KB per 1024.
- LDS read rate per WMMA is the same in both.
- Masking results are equivalent: FlyDSL's loop split does what ASM's in-loop branch does.
- Neither kernel uses `s_setprio` or WMMA operand reuse bits.
- ASM's `s_prefetch_inst` exists only because its unrolled loop is ~43 KB.
- I haven't verified what `WAVE_MODE` bit 24 (ASM) vs bit 25 (FlyDSL) do.