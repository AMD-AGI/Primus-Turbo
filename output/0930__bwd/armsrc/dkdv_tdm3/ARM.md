# arm dkdv_tdm3: TDM ring, S/dP B operands read back one iteration early into VGPRs

Base: `arms/dkdv_tdm` with `TDM_DEPTH=3`. flydsl 0.3.2, `_env.py`/`impl.py` unchanged; only
`kernels.py` and `bounds_proof.py` differ from `dkdv_tdm`.

Why this arm exists (card results for dkdv_tdm):
- dkdv_tdm depth 3 cost +1.65% on prod. Its top stalls were the `s_wait_dscnt` waits on the 32 ring readbacks issued at the start of the iteration.
- A2 says the win is having the B operands already in VGPRs when the iteration starts.

## What changed vs dkdv_tdm (line numbers in `arms/dkdv_tdm3/kernels.py`)

| Lines | Change |
|---|---|
| 370-386 | `_rdqd(stage_off)`: the 32 `ds_load_b128` that read one stage's Q/dO S/dP B operands. Addresses are identical to dkdv_tdm's inline reads, and the order is hh, {Q, dO}, dt, u |
| 388-415 | `_body` gains `qd` (carried operands) and `rb_off` (the stage to read back). In the full loop (carry=True), the top of the body issues only the LSE/delta b32 loads and the TDM for iteration it+2 into stage `(it+2)%3`. There is no tensor wait at the top any more |
| 433-440 | The S/dP B operands come from `qd` in the full loop, or from `_rdqd(stage 0)` in the masked loop, which is unchanged: own TDM, then wait(0) |
| 434, 492-502 | The 8 P/dS stores are deferred and emitted as one burst behind `sched_barrier(0)`, after all S/dP/softmax work |
| 524-548 | New DS phase order: the 8 P/dS stores; the 8 A-operand tr16 loads (`a_p`/`a_ds` for both kh, hoisted); the 32 B-operand tr16 loads; then, full loop only, `tensor_wait(2)` fenced by `sched_barrier(0)` and the 32-load readback of stage `(it+1)%3`; then `sched_barrier(0)` and the 32 dK/dV WMMAs. The WMMA order and operands are unchanged |
| 574 | Return value: accumulators + LSE/delta + the 32 readback vectors, which become the next iteration's `qd` |
| 618-626 | `qloop_full`: `rbo = ((it+1)%3)*17408`. The carried state is split into `[:NST]` accumulators, `[NST:NST+4]` LSE/delta, and `[NST+4:]` operands |
| 677-681 | `_tdm_prologue`: after the TDM for stages 0 and 1, `tensor_wait(2)` (retires stage 0), then `_rdqd(0)`, returned as iteration 0's operands |
| 725, 736 | Both prologues append those operands to the loop state |

LDS layout and allocation are unchanged from dkdv_tdm: a 3-stage ring over [0, 52224), group_segment 70656 B.

## Compile (`tools/compile.sh arms/dkdv_tdm3 dkdv dkdv_sp`, RC=0)

```
dkdv     .group_segment_fixed_size: 70656 .private_segment_fixed_size: 0 .sgpr_spill_count: 0 .vgpr_count: 721 .vgpr_spill_count: 0  wmma=128 ds=256
dkdv_sp  .group_segment_fixed_size: 70656 .private_segment_fixed_size: 0 .sgpr_spill_count: 0 .vgpr_count: 717 .vgpr_spill_count: 0  wmma=128 ds=192
```
VGPR went from 576 (dkdv_tdm) to 721, with 0 spill and 0 scratch; c1 was 729. SGPR: 69 for `k_dkdv`, 78 for `k_dkdv_sp`.

## ISA evidence: hot loop (k_dkdv `.LBB0_11`; k_dkdv_sp `.LBB0_8` is the same shape)

| per iteration | c1 (r29) | dkdv_tdm d3 | **dkdv_tdm3** |
|---|--:|--:|--:|
| instructions | 659 | 551 | 583 |
| WMMA<->DS switches | 7 | 35 | **2** (runs `W32 D80 W32`) |
| DS ops before the first S/dP WMMA | 7 | 4 (plus readback waits) | **0** |
| dscnt waits before the first S/dP WMMA | – | `s_wait_loadcnt_dscnt 0x2602`, then many 0x1/0x2 waits inside the S/dP phase | a single `s_wait_loadcnt_dscnt 0x2600` (dscnt 0) at the loop top |
| `v_mov_b64` / `v_nop` | 64 / 66 | 0 / 12 | 0 / 23 |
| `buffer_load_b128` / `b32` | 32 / 4 | 0 / 4 | 0 / 4 |
| `tensor_load_to_lds` / `s_wait_tensorcnt` | 0 / 0 | 2 / 1 (0x2 at the top) | 2 / 1 (0x2 in the DS phase) |
| `ds_store_b128` / `ds_load_b128` / `ds_load_tr16` | 40 / 0 / 40 | 8 / 32 / 40 | 8 / 32 / 40 |

Wait and issue order in the loop:
1. `tensor_load x2`
2. `s_wait_loadcnt_dscnt 0x2600`
3. 32 S/dP WMMAs; in between them, only no-op `s_wait_loadcnt` thresholds, since just 4 b32 loads are ever outstanding
4. `ds_store x8`
5. `ds_load_tr16 x40`
6. `s_wait_tensorcnt 0x2`
7. `ds_load_b128 x32`
8. `s_wait_dscnt 0x3e`: the first dK/dV WMMA waits only for the first 18 DS ops (8 stores, 8 A-operand loads, 2 `b_do[0]` loads)
9. dscnt waits 0x3a, 0x36, ... 0x20, then 26 WMMAs with no wait

- **The dscnt 0 at the loop top is the carried readback.** It was issued behind the 40 tr16 loads, and has at least 26 WMMAs plus the loop-head SALU and TDM issue to cover it. About 16 KB of reads need roughly 130-150 cycles at the measured about 110-124 B/clk.
- **The readback is not waited at all in the dK/dV tail.** The last tail wait is 0x20, which leaves the 32 readback loads in flight.
- The masked loop (`.LBB0_4`, 4 iterations per WG at prod) runs `D32 W32 D48 W32`: 3 switches.
- **The prologue readback waits on TDM.** Its 32 `ds_load_b128` come after `s_wait_tensorcnt 0x2`, at ISA lines 1331 and 1345 in k_dkdv and 1403 and 1408 in k_dkdv_sp.
- **The epilogue waits on TDM.** It has `s_wait_tensorcnt 0x0` immediately before its first LDS or global store.

## Bitwise status: expected identical to c1 (dkdv_tdm is already bitwise on the card)

- The S/dP WMMAs use the same bf16 bits. The readback reads the same addresses dkdv_tdm read, and the stage holds the same tile. `bounds_proof` V1 checks the tile.
- WMMA emission order and accumulator chains are unchanged. Only DS issue order moved: P/dS stores are deferred as a group, and `a_p`/`a_ds` are hoisted before `b_do`/`b_q`.
- The values stored and loaded are identical, and each LDS location is written before it is read within the iteration: the P/dS stores come before their tr16 reads.

## Bounds and lifetime proof: `python3 arms/dkdv_tdm3/bounds_proof.py` gives RESULT: ALL PASS

The proof models the tdm3 schedule per workgroup: prod runs `k_dkdv`; fast and toy run `k_dkdv_sp` with nsp=16; causal 0 and 1 are both covered. It tracks TDM writes (in-order FIFO) and LDS reads with explicit retire points:
- tr16 of iteration i retires by the end of iteration i.
- The readback done in iteration i retires at the top of i+1. That is after i+1's TDM issue, as the ISA shows.
- The final readback retires in LDS program order before the epilogue stores.

Checks:
- **G1, L1**: as in dkdv_tdm. For prod, 8,454,144 descriptors; max element 134217727 < 134217728.
- **L2**: no TDM targets a stage that has an unretired tr16 or readback read, and no read touches a stage with an unretired TDM write. The stage read early for i+1, `(i+1)%3`, is next written at the top of i+3.
- **C1**: at most 4 TDM ops outstanding at any wait; every wait retires the stage it guards; the counter is 0 at exit.
- **V1**: the readback for iteration i and the tr16 of iteration i both see tile(i).

Clamped cases are covered: the last iteration's prefetch and readback (the stage holds tile(n-1)), the prologue with n = 0 or 1, and empty loops.

ISA checks, run on both kernels:
- **Masked loop**: the tensor wait comes before all DS ops, and dscnt 0 before the back edge.
- **Full loop**:
  - both TDM issues come before the first DS op;
  - dscnt 0 and no DS op before the first WMMA;
  - `s_wait_tensorcnt 0x2` comes before the first readback;
  - the tr16 loads retire in-iteration (dscnt 0x20, which is at most the 32 later ops).

## Risks

1. **Premises of the proof.**
   - TDM ops retire in order; `tensor_wait(2)` depends on it, as in dkdv_tdm and the aiter gemm ring.
   - LDS operations from one wave execute in order. This matters only for the never-consumed final readback against the epilogue's stores into [0, 24576); there is a `tensor_wait(0)` before those stores but no dscnt wait.
2. **More than 63 DS ops outstanding.** The DS phase issues 80 DS ops back to back, above the 6-bit dscnt maximum of 63. The hardware stalls issue at saturation. The compiler's thresholds (0x3e and so on) are correct under in-order completion, but the last few readback issues may stall briefly. It is a performance question, not a correctness one.
3. **Loop-top dscnt 0.** If the readback does not finish within the tail's 26 unguarded WMMAs plus the loop head, the stall comes back at the loop top, where you asked for no wait. Fallback: move the readback before `b_do`/`b_q` in the DS burst. That trades a longer drain for the first dK/dV WMMA.
4. **Longer single DS drain before the dK/dV tail.** The first dK/dV WMMA still waits for 18 DS ops, but the phase issues 80 in one run, where c1 issued 63. Watch the ATT for back-pressure on stores (c1 measured 3-4.6 cycles per op after the 12th).
5. **VGPR is 721**, still 1 wave/SIMD. There is headroom to 1024 but no 2-wave path.
6. **`sched_barrier(0)` fences.** There are 3 new ones: before the P/dS store burst, around the tensor wait, and before the dK/dV WMMAs. They stop VALU from overlapping the DS burst. Any VALU left in the tail (the loop-head SALU division) is still exposed; stacking `dkdv_divfree` helps.
7. **Same launch discipline as dkdv_tdm.** Hang risk exists only on a wrong descriptor, and the descriptor code is unchanged from the version that has already run bitwise on the card. Run toy, then fast, then prod, each in its own flocked process.
8. **Validation gate:** dK/dV bitwise vs c1 on toy, gqa4_small, unequal_seqlen_2, fast and prod before timing.
