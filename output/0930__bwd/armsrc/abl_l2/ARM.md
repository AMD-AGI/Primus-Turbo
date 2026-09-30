# arm dkdv_tdm: k_dkdv Q/dO through TDM into an LDS ring

Base: `arms/c1` (r29 champion with DQ_U2=False). Only `kernels.py` changed; `impl.py` and `_env.py` are
unchanged, so flydsl is still **0.3.2** (`~/.local/flydsl032`). The TDM API is already in 0.3.2:
`fx.rocdl.cdna5.make_tdm_atom`, `fx.copy_atom_call`, `flydsl.expr.rocdl.tdm_ops.tensor_wait`. It is the
same idiom as the fwd champion's `_tdm_load_views` (`0927__b0/champions/fwd_r16_r13ns/flydsl_fwd/fmha_b16_buffer_managers.py:939-988`)
and aiter's `gemm_a16w16_kernel_gfx1250.py:243-259`.

Status: **ready for a card test, not yet run.** It compiles with RC=0, 0 spill and 0 scratch, and the CPU
bounds proof passes (ALL PASS). Nothing has been launched on a GPU.

## What changed (line numbers in `arms/dkdv_tdm/kernels.py`)

| Lines | Change |
|---|---|
| 40 | `from flydsl.expr.rocdl import tdm_ops` |
| 57-58 | `TDM_DEPTH = 3`: 3 ring stages, so the prefetch runs 2 iterations ahead. Setting it to 2 prefetches 1 iteration ahead |
| 273-286 | `_ldqd` becomes `_ldl`: only the 4 LSE/delta `buffer_load_b32`, still carried one iteration early. The 32 Q/dO b128 loads and their `sched_barrier` are gone |
| 291-324 | New `_tdm_qdo(qt, gh, stage_off)`. It issues two TDM ops, dO to stage+0 and Q to stage+8704, each a `[32 rows][128 d]` bf16 tile. Tile origin is element `((bat*Sq+q0)*Hq+qh)*D`; outer stride is `Hq*D` elements; extents are `[Sq-q0, None]`; `num_warps=1`; `pad_interval=128` and `pad_amount=8` elements, giving LDS row stride 272 B = `X_ROW_B` |
| 370-420 | `_body`. Full loop (carry=True): prefetch the LSE/delta b32 for the next iteration, then `sched_barrier(0)`, `tensor_wait(2)` (depth 3; `tensor_wait(0)` at depth 2), then the TDM of iteration it+2 (or it+1) into stage `(it+2)%3`, then `sched_barrier(0)`. Masked loop (carry=False): TDM its own tile into stage 0, then `tensor_wait(0)`. The S/dP B operands `qfr`/`dfr` are now 2 x `ds_load_b128` per fragment, read from the current stage at the same byte offsets c1 used for its `ds_store`s. The 32 staging `ds_store_b128` are gone. `lds_do`/`lds_q` in the tr16 reads now point at the current stage |
| 565-583 | `qloop_full`: `cur = (ii%TDM_DEPTH)*17408`. The prefetch tile is `kk = min(ii+2, n-1)` (clamped the same way as the existing `jj`); the prefetch stage is `((ii+2)%3)*17408` |
| 623-634 | `_tdm_prologue(qt0, n)` fills stage 0 with iteration 0's tile and stage 1 with iteration 1's tile. Iteration 1's tile is clamped to `max(min(1,n-1),0)`, then through `_clampqt` |
| 636 | Asserts `not KV_U2`: the unused unroll-by-2 path is not wired to the ring |
| 676-693 | Both prologues (PARTIAL and non-PARTIAL) call `_ldl` + `_tdm_prologue`. After `qloop_full` there is a `tensor_wait(0)` before the epilogue, which reuses LDS [0, 24576) and overlaps stages 0 and 1 |

LDS: stage k occupies `[k*17408, (k+1)*17408)`, with dO at +0 and Q at +8704. Three stages end at 52224, which
is below 65536, so the whole ring stays in LDS segment 0. The P/dS ring stays in segment 1 at [65536, 70656).
**`group_segment_fixed_size` is unchanged at 70656.**

## Compile (compile-only, `tools/compile.sh arms/dkdv_tdm dkdv`)

```
depth 3 (shipped): RC=0  .group_segment_fixed_size: 70656 .private_segment_fixed_size: 0 .sgpr_spill_count: 0 .vgpr_count: 576 .vgpr_spill_count: 0  wmma=128 ds=224
depth 2          : RC=0  identical resource line (576 VGPR, 70656 LDS, 0 spill, 0 scratch)
c1/r29 k_dkdv    :       vgpr 729, LDS 70656, sgpr 78 -> dkdv_tdm sgpr 61
```
The depth-3 ISA is identical across two compiles (`cmp`).

## ISA evidence: hot loop (`qloop_full`), per iteration

| | c1 (r29 `.LBB0_11`) | tdm depth 2 (`.LBB0_8`) | tdm depth 3 (`.LBB0_8`) |
|---|--:|--:|--:|
| instructions | 659 | 513 | 551 |
| `tensor_load_to_lds` | 0 | 2 | 2 |
| `s_wait_tensorcnt` | 0 | 1 (0x0) | 1 (0x2) |
| `buffer_load_b128` | 32 | 0 | 0 |
| `buffer_load_b32` | 4 | 4 | 4 |
| `v_mov_b64` | 64 | 0 | 0 |
| `v_nop` | 66 | 12 | 12 |
| `ds_store_b128` | 40 | 8 (P/dS only) | 8 |
| `ds_load_b128` | 0 | 32 (S/dP B operands) | 32 |
| `ds_load_tr16_b128` | 40 | 40 | 40 |
| `v_wmma` | 64 | 64 | 64 |

- Whole kernel: `buffer_load_b128` goes from 128 to 32 (the 32 left are the invariant K/V fragments) and `tensor_load_to_lds` from 0 to 8. The 8 are: 2 in the masked loop, 4 in the prologue, 2 in the full loop.
- Depth 3 has 38 more SALU per iteration than depth 2. They come from a second runtime `//G` for `kk` and a `%3`. Stacking `dkdv_divfree` removes them.
- Decoded TDM descriptor (group1 `s[12:19]`):
  - data_size = 2 B
  - pad_enable = 1, pad_interval code 5 (64 DW = 128 elements), pad_amount code 3 (4 DW)
  - tensor_dim0 = 0x7FFFFFFF (no clamp, same as fwd)
  - tensor_dim1 = `max(Sq-q0, 0)`
  - tile_dim0 = 128, tile_dim1 = 32
  - stride0 = `Hq*D`
- Group0: pred = 1; LDS address = stage offset (+0x2200 for Q); global VA = base + (row offset << 8) with type bits [31:30] = 2.
- Order inside the loop body:
  1. 4 b32 loads
  2. `s_wait_tensorcnt 0x2`
  3. two `tensor_load_to_lds`
  4. the first `ds_load` (body index 117; the wait is at index 97)
  5. the last `ds_load` is followed by `s_wait_dscnt 0x0` before the back edge
- The masked loop does TDM, then `s_wait_tensorcnt 0x0`, then its `ds_load`s.
- The epilogue has `s_wait_tensorcnt 0x0` immediately before its first `ds_store_b128`.
- LLVM inserted no `s_wait_tensorcnt` of its own. The explicit waits are the only ones.

## Expected bitwise status: identical to c1 (not yet validated on card)

- **Same WMMA operands in the same order.** In c1, `qfr`/`dfr` were `shuffle(v8 @ col half*8+dt*32, v8 @ +16 cols)` of row `hh*16+row`, loaded from global. Here they are the same two 16 B chunks, read from the TDM image at byte `(hh*16+row)*272 + half*16 + dt*64 (+32)`.
  - TDM puts element (r, c) of the tile at byte `r*272 + 2c`. That is exactly the address c1's `ds_store` used for the same element.
  - TDM copies bf16 bits verbatim.
  - The outer extent is `Sq-q0 >= 32` for every issued tile (proven below), so hardware zero-fill never triggers.
- **tr16 reads are unchanged.** They read the same image at the same addresses. They touch bytes 0..255 of each row and never the pad, so whether TDM writes the pad does not matter.
- **Unchanged code:** LSE/delta loads (same values, still prefetched one iteration ahead), mask, softmax, dS, P/dS LDS round trip, dK/dV WMMA order and accumulation order, epilogue.
- **Loop structure:** the (qt, gh) iteration order is unchanged. The clamped prefetches load tiles that are never read.

So every fp32 accumulation happens in the same order on the same inputs, and dK/dV should match c1 bit for bit. dQ/k_dqg is not touched.

## Bounds proof: `python3 arms/dkdv_tdm/bounds_proof.py` gives RESULT: ALL PASS (about 60 s)

The script replays the exact integer control flow of `_dkdv_impl`, using impl.py's nsp rule: prod runs `k_dkdv` with nsp=1; fast and toy run `k_dkdv_sp` with nsp=16. It covers causal=1 and causal=0 at depths 3 and 2, and enumerates every TDM descriptor.

Checks:
- **G1 (global reads).** Every global element read is in `[0, B*Sq*Hq*D)`, and the outer extent is at least 32.
  - prod max element = 134217727 < 134217728
  - fast max = 1048575 < 1048576
  - toy max = 32767 < 32768
- **L1 (LDS writes).** Every LDS write stays inside its own stage, inside the ring, and below 65536 (never in the P/dS segment).
- **L2 (no read/write overlap).** No TDM op writes the stage being read in the same iteration, and every stage read happens after all TDM writes to that stage have retired. This assumes in-order retirement, the same premise as the aiter gemm ring.
- **C1 (tensorcnt).** Every wait is reachable: thresholds are 0 or 2, and at most 4 ops are ever outstanding. The counter is 0 at kernel exit on every path, including empty loops.
- **V1 (stage contents).** The stage an iteration reads holds exactly that iteration's (qt, gh) tile.
- **ISA checks.** No `ds_*` op comes before the `s_wait_tensorcnt` in either TDM loop, and `s_wait_dscnt 0x0` comes after the last `ds_load` and before the back edge. That second fact is the write-after-read premise: the next TDM into a stage cannot overwrite data still being read.

Numbers (depth 3):

| Shape | TDM descriptors | Clamped last prefetches | Clamped prologues | Empty full loops |
|---|--:|--:|--:|--:|
| prod causal | 8,454,144 | 16,320 | 32 | 32 |
| fast causal | 12,544 | 1,280 | 384 | 384 |
| toy causal | 296 | 12 | 58 | 58 |

All clamped tiles are in bounds.

## Risks for the first card launch

1. **Hang path: TDM descriptor.** This is the first TDM use in this bwd kernel, and a wrong extent or stride leaves a tensorcnt wait that never retires, which wedges the card.
   - The encoding comes from the same `make_tdm_atom` API as the working fwd, and the fields are decoded above.
   - Launch the **toy shape first, alone, in its own flocked process** (per gfx1250-card-safety), then fast, then prod.
2. **`k_dkdv_sp` has never been compiled.** It is the PARTIAL path and runs for fast and toy (nsp=16). `tools/compile_arm.py` only builds `k_dkdv`. It shares `_body`, the loops and `_tdm_prologue`; only its prologue arithmetic and its fp32 global-store epilogue differ, and the proof covers its control flow. Its first compile will happen on the card host, so run a compile-only pass there before launching it.
3. **In-order TDM retirement.** `tensor_wait(2)` at depth 3 assumes TDM ops retire in order. aiter's gemm ring relies on the same assumption. If there is any doubt, `TDM_DEPTH = 2` uses only `tensor_wait(0)` before issuing, and needs no ordering assumption.
4. **No buffer-descriptor safety net.** TDM takes a raw 64-bit VA, not a V# with `num_records`. Out-of-bounds safety rests entirely on the qt/jj/kk clamps and on the outer extent `Sq-q0`, both proven above. The inner extent is 0x7FFFFFFF, as in fwd.
5. **Correctness depends on the explicit waits.** LLVM does not model TDM-to-LDS dependences; the ISA shows it inserts no tensorcnt waits. The `sched_barrier(0)` fences and the ISA order check hold today. Re-run the ISA check in `bounds_proof.py` after any edit.
6. **Performance risks, not correctness.**
   - The S/dP B operands now cost 32 `ds_load_b128` at the top of the body: a new DS latency before the first S/dP WMMA, replacing c1's register-resident operands.
   - The DS instruction count per iteration is the same as c1 (80), but loads replace stores.
   - The masked loop (4 iterations per WG at prod) waits for its own TDM without a prefetch, as c1 did with its synchronous loads.
   - VGPR is 576: still 1 wave/SIMD, since 2 waves need 512 or fewer.
7. **Pad bytes.** Whether TDM writes the pad bytes is unknown. Nothing reads them.
8. **Validation gate.** Validate dK/dV bitwise against c1 at toy and fast before timing prod. Any mismatch means a layout or schedule bug, never an expected tolerance difference.
