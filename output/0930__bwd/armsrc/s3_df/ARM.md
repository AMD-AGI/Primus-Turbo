# arm s3_df: s3 + divide-free k_dkdv loop counters

Base: `arms/s3`. That is k_dkdv from `dkdv_tdm3`, k_dqg from `dqg_ts`, and impl.py side-stream k_dqg.
This arm ports `dkdv_divfree` and `tdm_df` onto s3's k_dkdv: the 3-stage TDM ring plus the B operands read back into carried VGPRs.
Only `kernels.py` changed. `bounds_proof.py` and `isa_stats.py` are new. `impl.py`, `_env.py`, `__init__.py` and k_dqg/k_delta are byte-identical to s3.

Status: compile-only plus CPU proof. **Not run on a GPU.**

## Switch
`DIVFREE` (kernels.py:60, default True). False gives s3's code, kept verbatim in the `else` branches.
**Verified: with DIVFREE=False, k_dkdv's ISA is `cmp`-identical to `s3/.dump/dkdv/k_dkdv_0/21_final_isa.s`.** The comparison lives in `_off/`, which also holds the s3-equivalent k_dkdv_sp reference.

## What changed (kernels.py)
- :60 `DIVFREE = True`.
- :583-596 comment and `_wrap(qc, gc)`, the same helper as dkdv_divfree: `g1 = gc+1; g1<G ? (qc,g1) : (qc+1,0)`.
- :598-614 `qloop_mask` carries (qi, gh) at `st[NST], st[NST+1]` and yields `body + [qn, gn]`.
- :616-684 `qloop_full` carries **(cur, qi, gh) at `st[-3:]`**. LSE/delta sit at `st[NST:NST+4]` and the carried B operands at `st[NST+4:-3]`.
  - jj: `ii+1<n ? wrap(qi,gh) : (qi,gh)`.
  - kk: `ii+2<n ? wrap(wrap(qi,gh)) : (qj,gj)`.
  - `nxo = cur==0 ? 34816 : cur-17408`, which is `(ii+2)%3*QDO_B`.
  - `ncur = cur==34816 ? 0 : cur+17408`, which is `(ii+1)%3*QDO_B`. It serves both as the next iteration's `cur` and as dkdv_tdm3's readback stage `rbo`, which was a separate `%3`.
- :729-741 `_tdm_prologue` stage-1 tile: `n>1 ? wrap(0,0) : (0,0)`. This replaces `max(min(1,n-1),0) // G`.
- :766-771 `_z2`/`_z3` zero start states.
- :789/:800: `qloop_mask(init + _z2)`.
- :794-795/:805-806: `list(out)[:NST] + LSE/delta + prologue readback + _z3`, for both PARTIAL and non-PARTIAL.
- KV_U2's qloop_full2/qloop_tail are unchanged. They are unreachable, because `_qloop_full_any` asserts `not KV_U2`.

## Compile (`tools/compile.sh arms/s3_df dkdv dkdv_sp`, RC=0)
| kernel | vgpr | sgpr | spill v/s | scratch | LDS | wmma | ds |
|---|--:|--:|--:|--:|--:|--:|--:|
| k_dkdv s3 | 721 | 69 | 0/0 | 0 | 70656 | 128 | 256 |
| k_dkdv **s3_df** | 721 | 68 | 0/0 | 0 | 70656 | 128 | 256 |
| k_dkdv_sp s3 (`_off`) | 717 | 78 | 0/0 | 0 | 70656 | 128 | 192 |
| k_dkdv_sp **s3_df** | 717 | 77 | 0/0 | 0 | 70656 | 128 | 192 |

## ISA evidence (`python3 isa_stats.py <isa>`)
Full loop is `.LBB0_11` (k_dkdv) or `.LBB0_8` (k_dkdv_sp). Mask loop is `.LBB0_4`.

| per iteration | s3 k_dkdv | **s3_df k_dkdv** | s3 sp | **s3_df sp** |
|---|--:|--:|--:|--:|
| full loop instructions | 583 | **538** | 584 | **543** |
| full loop SALU (excl. msb/wait/nop) | 105 | **60** | 107 | **66** |
| full loop s_abs / s_mul_hi / s_mul_i32 | 2 / 5 / 10 | **0 / 0 / 1** | 2 / 5 / 11 | **0 / 0 / 1** |
| full loop SALU before the 1st tensor_load (loop head, no WMMA to hide under) | 98 | **50** | | |
| full loop: position of first TDM issue / first WMMA | 108 / 115 | **60 / 67** | | |
| mask loop instructions / SALU | 620 / 84 | 603 / 67 | 644 / 84 | 614 / 67 |
| whole kernel s_abs / s_mul_hi / v_rcp | 5 / 9 / 2 | **0 / 0 / 0** | 8 / 12 / 3 | 3 / 3 / 1 |
| full loop VALU / v_nop | 269 / 23 | 269 / 23 | 269 / 23 | 269 / 23 |

- In the full loops of both kernels, and in the k_dkdv mask loop, the non-SALU instruction stream is **opcode-for-opcode identical** to s3. That covers VALU, WMMA, DS, VMEM, TDM, and every `s_wait_*` with its immediate. Only the SALU changed.
- The k_dkdv_sp mask loop is the one exception. Its instruction multiset is the same, but a few VALU/WMMA are scheduled in a different order, and its waits are identical.
- Remaining full-loop index SALU is: two wraps (s_add, s_cmp_ge, s_cselect, s_add_co_ci), two `s_cmp_lt` clamps with selects, and the stage ring (`s_cmp_lg_u32 s27,0 / s_cselect ...,0x8800` and `s_add 0x4400 / s_cmp_lg_u32 s27,0x8800 / s_cselect ...,0`).
- The rest of the loop-head SALU builds the TDM descriptors (`s_mul_u64`, `s_lshl`, `s_bitset1`), which is unchanged.
- The 3 s_abs and 3 s_mul_hi left in k_dkdv_sp come from the PARTIAL `//nsp` split, which runs once and sits outside every loop.

## Bitwise
Expected bitwise identical to s3. Every body receives the same `(qt, gh, qt_n, gh_n, cur, nxo, pf_qt, pf_gh, rbo)` tuple (proof E1), and the prologue issues the same tiles. That means every address, TDM descriptor and readback stage is the same, and the non-SALU ISA is unchanged.

## Bounds proof (`python3 bounds_proof.py`, about 70 s): RESULT: ALL PASS
This is dkdv_tdm3's proof (G1, L1, L2, C1, V1, plus the ISA ordering checks), now **driven by the DIVFREE counters**, with these additions:
- **E1** runs on every workgroup and asserts the DIVFREE tuple equals s3's `//G`, `%G`, `%3` formulas for:
  - every qloop_mask iteration;
  - the prologue stage-1 tile;
  - every qloop_full iteration: (qt, gh, qt_n, gh_n, cur, nxo, pf_qt, pf_gh, rbo).

  It also checks that (qt_n, gh_n) is a legal LSE/delta pair. Coverage by shape:

  | shape | E1 tuples, causal 1 | E1 tuples, causal 0 |
  |---|--:|--:|
  | prod (k_dkdv) | 4,218,880 | 8,396,800 |
  | fast (sp16) | 5,248 | 9,216 |
  | toy | 84 | 96 |
  | gqa4_small | 416 | 512 |
  | unequal_seqlen_2 (sq1024, skv2048) | 14,464 | 18,432 |
- The TDM/tensorcnt paths reached, all passing:
  - clamped prefetch;
  - empty full loop (n = 0) with the prologue clamp;
  - n = 1 prologue;
  - `tensor_wait(0/2)` with at most 4 outstanding, drained to 0 at exit.
- **E2** is an exhaustive sweep: G 1..16, n 0..96, qt0 {0,3}. That is 3104 cases and 297,984 tuples, plus a check that wrap == divmod.
- **M** is a mutation check. It builds 4 wrong variants (nxo, ncur, pf_gh clamp, prologue n<=1), and E2 rejects each one.
- The ISA ordering check passes on both kernels:
  - TDM issues before any DS op;
  - dscnt 0 before the first WMMA;
  - `tensorcnt 0x2` before the readback;
  - tr16 retired in-iteration.

## Expected gain
The removed SALU sat at the **loop head, before the first TDM issue and the first WMMA**: 48 of 98 head SALU per iteration, including 5 s_mul_hi and 2 s_abs in dependent chains. Nothing co-issues there, so this is exposed latency.
- k_dkdv per iteration is about 3000 cycles (23.4 latency/WMMA x 128), so removing roughly 50-70 cycles gives about -1.5 to -2%.
- End to end the estimate is about -1 to -1.5%, since k_dkdv is the critical path and k_dqg runs concurrently.

## Card order (when timed)
Same as s3:
1. toy alone first (TDM);
2. fast (k_dkdv_sp);
3. gqa4_small and unequal_seqlen_2;
4. prod.

Gate on bitwise dK/dV vs s3 before timing.

## Files
- `kernels.py`, `bounds_proof.py`, `isa_stats.py`.
- `.dump/{dkdv,dkdv_sp}/`: the ISA.
- `_off/`: DIVFREE=False, which gives s3's ISA byte-for-byte, used as the reference for k_dkdv_sp.
