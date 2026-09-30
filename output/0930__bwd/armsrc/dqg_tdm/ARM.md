# arm dqg_tdm -- k_dqg K/V staged by TDM into a 3-stage LDS ring (compile-only screen)

Base: `arms/s4` (copied; .dump/.compile.log removed). `impl.py` is identical to s4's, and so are
k_dkdv / k_dkdv_sp / k_dq_sp. The k_dkdv final ISA matches s4 byte for byte (directives excluded).
`kernels.py` only adds code (`DQ_TDM = True`, `_dqg_tdm_impl`); `k_dqg` dispatches to it. s4's
`_dqg_impl` is still there unchanged, so `DQ_TDM = False` gives back s4's k_dqg.

## What changed
s4's k_dqg per 32-row kv step: 32 `buffer_load_b128` (K/V, U2 + tailpf carried set), 16 `ds_store_b128`
(K, so the dQ GEMM can read it transposed), 16 `ds_load_tr16`, about 11 `s_wait_loadcnt`.
dqg_tdm works the same way as dkdv_tdm3/s4's k_dkdv:
* One TDM op per tensor (K, V) writes the `[32 kv][128 d]` tile into ring stage `(i-1)%3` for tile
  `min(i+2, N-1)`. The image layout is the one s4's K ds_stores built: row r at r*272, element c at 2c.
* dQ B operands: 16 `ds_load_tr16` straight from the K region of stage `i%3`. There is no ds_store now.
* S/dP A operands for step i+1: after `s_wait_tensorcnt 2`, 32 `ds_load_b128` read stage `(i+1)%3`
  back into carried VGPRs. This happens in the DS phase of step i.
* There is one ring for both loops. The full loop covers `[0, nfull)` and the mask loop covers
  `[nfull, N)`, carrying acc + A operands + the ring offset. The mask loop no longer loads its own K/V.
* U2, tailpf, soffset and headst are all dropped. The body is one step per trip with no back-edge
  copies (0 `v_mov` in the loop).
* Body phase order `DQT_PHASE="b"`: S/dP WMMAs, then the DS phase (tr16, tensor wait, readback),
  then softmax VALU and the dQ WMMAs (qh-major). The DS latency hides under the softmax VALU.

## ISA (prod b4 s8192 hq32 hkv8, causal)
| k_dqg | VGPR | LDS | full-loop instr / kv step | VMEM ld | ds_st | loadcnt waits | s_set_vgpr_msb | nops |
|---|---|---|---|---|---|---|---|---|
| s4 (U2, per body) | 1020 | 8704 | 675.5 | 32 | 16 | 11 | 158.5 | 25 |
| **dqg_tdm** | **878** | 52224 | **574 (-15%)** | 0 | 0 | 0 | 79 | 5 |

WMMA 96 and VALU 290 per step are unchanged. The mask loop is 929 per step (s4: 803), but it runs only
about 2 of about 129 steps per WG at prod causal. 0 spill, 0 scratch. `isa_gate.py`: ALL PASS
(`isa_gate.log`). `isa_stats.log` has the per-loop mix for both kernels.

Variants screened (compile only):
* `DQT_PHASE="a"`: softmax before the DS phase. 776 VGPR, 605 instr/step (+16 nops, tr16 latency exposed).
* `DQT_BK_EARLY=True`: tr16 at the body top. 621 instr/step.
* `DQT_DEPTH=4` (`_d4/`, LDS 69632): 575 instr/step, `s_wait_tensorcnt 0x4`, one more step of TDM lead.
  This is the alternative to try on the GPU if TDM latency shows up at the readback wait.

`_pa/` and `_bke/` only hold container-owned `.dump` leftovers that could not be removed without a
container command.

LDS: every mix of 4 WGs/CU (k_dkdv 70656, k_dqg 52224 or 69632) is at most 282624 B, below 327680 B.
Occupancy is unchanged: VGPR still limits it to 1 wave per SIMD, and k_dqg (878) + k_dkdv (713) is
more than 1024, so the two kernels still cannot share a SIMD.

## Proofs
`bounds_proof.py` (host python3, `bounds_proof.log`: ALL OK). It covers DEPTH 3 and 4 at prod, fast,
toy, gqa4_small and unequal_seqlen_2, each with causal on and off:
* **T1 TDM**: origin, box and outer extent are in bounds for every (bat, hkv, tile).
* **T2 ring counters**: cur, nxo and ncur equal i%D, (i-1)%D and (i+1)%D.
* **T3 LDS addresses and immediates**: readback max immediate 13280, tr16 max 4576, which match the ISA.
* **T4 full/mask split**: the causal predicate is false on every block in the full loop, and N covers
  every attended kv.
* **T5 tensorcnt replay**: in-order FIFO. Each read sees the right tile with no TDM in flight, no TDM
  lands on a stage with an unconsumed read, and the final wait(0) empties the FIFO.

Mutation checks all fail as they should: wait count +1, prefetch index off by one, wrong target stage.

## Bitwise vs s4
Expected bitwise.
* **B1**: every readback chunk is the same 16 B that s4's `_ldkv_kt` loaded for the same (lane, kt, dt, K/V, u).
* **B2**: the TDM K image equals s4's ds_store image on every byte a tr16 reads, and the tr16 addresses
  match s4's relative to the K base.
* **Arithmetic**: the softmax is s4's non-VF code verbatim. Each dQ accumulator gets one WMMA per kv
  block, in block order.
* **Moved blocks**: blocks that move from s4's mask loop to the full loop have a predicate that is
  identically false, and this is proven. At all proven shapes `nfull` is even, so no block moves anyway.

## Expected effect / risks
* **Expected**: k_dqg cycles -8..-15% (3.07e6 -> about 2.7e6). The op is power-limited while both
  kernels run, so this should be about -1.5..-4% op time (-0.08..-0.2 ms).
* **Risks**:
  1. TDM lead is about one step, and the readback wait could stall. Mitigation: try `_d4`.
  2. k_dkdv and k_dqg now both drive TDM at the same time.
  3. The mask-loop SALU grew; it is small.
