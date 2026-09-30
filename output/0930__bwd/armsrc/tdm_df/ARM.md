# arm tdm_df: dkdv_tdm + dkdv_divfree (k_dkdv only)

Base: `arms/dkdv_tdm` (itself c1 = r29 with DQ_U2=False, plus a TDM LDS ring for Q/dO in k_dkdv).
Merged in: `arms/dkdv_divfree`'s division-free loop counters, extended to the two things the TDM
arm added to the loop: the second `//G` (prefetch tile `kk = min(ii+2, n-1)`) and the `%3`
stage index. Only `kernels.py` and `bounds_proof.py` differ from dkdv_tdm; `impl.py`, `_env.py`,
`__init__.py` are the same (flydsl 0.3.2). k_dqg / k_delta are untouched.

Status: compile-only + CPU proof. **Not run on a GPU.**

## Switches (each lever can be turned off on its own)
| constant | default | off = |
|---|---|---|
| `TDM_DEPTH` (kernels.py:57) | 3 | 2 = dkdv_tdm's depth-2 ring (also division-free under DIVFREE) |
| `DIVFREE` (kernels.py:59) | True | False = dkdv_tdm's code, kept verbatim in the `else` branches. **Verified: DIVFREE=False compiles to an ISA byte-identical (`cmp`) to dkdv_tdm for both k_dkdv and k_dkdv_sp** (`_off/` vs `_ref_tdm/`) |
| `KV_U2` | False | still asserted off (the ring is not wired to qloop_full2); divfree's qloop_full2 change was not ported because that path is unreachable here |

## What changed vs dkdv_tdm/kernels.py
- :59-62 `DIVFREE = True`.
- :543-560 comment + `_wrap(qc, gc)` (same helper as dkdv_divfree: `g1 = gc+1; g1<G ? (qc,g1) : (qc+1,0)`).
- :562-578 `qloop_mask`: carries (qi, gh) at `st[NST], st[NST+1]`, yields `body + [qn, gn]`. Single `yield`.
- :580-641 `qloop_full`: carries **(cur, qi, gh) at `st[-3:]`**, LSE/delta prefetch at `st[NST:-3]`.
  - jj decomposition: `live = ii+1 < n ? wrap(qi,gh) : (qi,gh)` (dkdv_divfree).
  - kk decomposition (depth 3): `ii+2 < n ? wrap(wrap(qi,gh)) : (qj,gj)` -- when ii+2 >= n, n-1 == jj.
  - stage: `cur` is a carried byte offset in {0, 17408, 34816}; `ncur = cur==34816 ? 0 : cur+17408`;
    prefetch stage `(ii+2)%3 == (ii-1)%3` is `nxo = cur==0 ? 34816 : cur-17408`.
  - depth 2: `nxo = ncur = 17408 - cur`, pf = (qj, gj).
  - one `final = yield res`; the old path is the `else` branch.
- :688-699 `_tdm_prologue` stage-1 tile: `j1 = max(min(1,n-1),0)` is 1 iff n > 1, and 1 = wrap(0,0),
  so `(q1, g1) = n>1 ? wrap(0,0) : (0,0)` -- no `//G` (outside the loop, but free to remove).
- :720-724 `_z2` (mask start (0,0)), `_z3` (full start cur=0, (0,0)); :742/:753 `qloop_mask(init + _z2, ...)`;
  :747/:758 `list(out)[:NST] + _ldl(...) + _z3` (PARTIAL and non-PARTIAL).

## Compile (`tools/compile.sh arms/tdm_df dkdv dkdv_sp`, RC=0)
| kernel | vgpr | sgpr | spill v/s | scratch | LDS | wmma | ds |
|---|--:|--:|--:|--:|--:|--:|--:|
| k_dkdv tdm_df | 576 | 60 | 0/0 | 0 | 70656 | 128 | 224 |
| k_dkdv dkdv_tdm | 576 | 61 | 0/0 | 0 | 70656 | 128 | 224 |
| k_dkdv_sp tdm_df | 572 | 69 | 0/0 | 0 | 70656 | 128 | 160 |
| k_dkdv_sp dkdv_tdm | 572 | 70 | 0/0 | 0 | 70656 | 128 | 160 |
TDM_DEPTH=2 + DIVFREE also compiles: 576/572 VGPR, 0 spill/scratch, LDS 70656.
(k_dkdv_sp is compiled here for the first time for the TDM ring, and dkdv_tdm's sp too, in `_ref_tdm/`.)

## ISA evidence (`isa_stats.py`, `.dump/*/21_final_isa.s`)
Hot loop = qloop_full (`.LBB0_8` k_dkdv, `.LBB0_6` k_dkdv_sp); mask loop = `.LBB0_2`.

| | dkdv_tdm k_dkdv | tdm_df k_dkdv | dkdv_tdm sp | tdm_df sp |
|---|--:|--:|--:|--:|
| full loop instructions | 551 | **522** | 537 | **507** |
| full loop SALU (excl. msb/wait/nop) | 99 | **61** | 99 | **65** |
| full loop s_abs / s_mul_hi / s_mul_i32 | 2 / 4 / 10 | **0 / 0 / 1** | 2 / 4 / 10 | **0 / 0 / 1** |
| mask loop instructions / SALU | 635 / 84 | 617 / 67 | 630 / 86 | 610 / 69 |
| mask loop s_abs / s_mul_hi | 1 / 1 | **0 / 0** | 1 / 1 | **0 / 0** |
| whole kernel s_abs / s_mul_hi / v_rcp | 5 / 8 / 2 | **0 / 0 / 0** | 8 / 11 / 3 | 3 / 3 / 1 |
| whole kernel `tensor_load_to_lds` | 8 | 8 | 8 | 8 |
| `s_wait_tensorcnt` sequence (kernel) | 0x0, 0x2, 0x0 | 0x0, 0x2, 0x0 | same | same |
| full loop: tensor_load / wait | 2 / 0x2 | 2 / 0x2 | 2 / 0x2 | 2 / 0x2 |
| full loop wmma / ds_load_b128 / tr16 / ds_store_b128 / buffer_load_b32 / b128 / v_mov_b64 | 64/32/40/8/4/0/0 | same | same | same |
| full loop VALU (incl. v_nop) | 258 (12 nop) | 265 (20 nop) | 257 | 265 |

- The 3 s_abs / 3 s_mul_hi left in k_dkdv_sp are the PARTIAL prologue's `//nsp` split arithmetic
  (outside every loop; dkdv_divfree has the same).
- Full-loop index math is now s_add / s_cmp_ge / s_cselect / s_add_co_ci (two wraps), two
  s_cmp_lt clamps, and the stage ring: `s_cmp_lg_u32 s42,0 ; s_cselect_b32 s37,s37,0x8800` (nxo) and
  `s_add 0x4400 ; s_cmp_lg_u32 s42,0x8800 ; s_cselect_b32 s41,s41,0` (ncur), carried in s42.
- TDM ring unchanged: 2 `tensor_load_to_lds` per iteration after one `s_wait_tensorcnt 0x2`; bounds_proof's ISA check
  passes on both kernels (no ds op before the tensor wait; `s_wait_dscnt 0x0` after the last ds_load before the back edge).
- Perf watch item: non-nop VALU in the full loop is 245 vs 246, but the scheduler added 8 v_nop (12 -> 20).

## Bitwise status
Expected bitwise identical to dkdv_tdm (and therefore to c1 once dkdv_tdm is validated): every
body receives the identical (qt, gh, qt_n, gh_n, cur, pf_qt, pf_gh, nxo) tuple and the prologue
issues the identical tiles (proof E1), so every address, every TDM descriptor, and the accumulation
order are unchanged. Only the SALU that computes them differs. Not validated on the card.

## Bounds proof (`python3 arms/tdm_df/bounds_proof.py`, about 85 s): RESULT: ALL PASS
It is dkdv_tdm's proof (G1 global reads, L1/L2 LDS, C1 tensorcnt, V1 stage contents, ISA WAR check) plus:
- **E1**: for every workgroup at prod (nsp 1, k_dkdv), fast and toy (nsp 16, k_dkdv_sp), causal 1/0, depth 3 and 2,
  it replays the DIVFREE counters next to dkdv_tdm's `//G`, `%G`, `%3` formulas and asserts identical tuples
  for qloop_mask, every qloop_full iteration (qt, gh, qt_n, gh_n, cur, pf_qt, pf_gh, nxo), and the prologue
  stage-1 tile. Prod depth 3 checks 4,218,880 tuples causal and 8,396,800 non-causal.
- The TDM/LDS/tensorcnt model is now **driven by the DIVFREE values**, so G1/L1/L2/C1/V1 are proven for the
  new expressions directly. It also checks that the LSE/delta prefetch pair (qt_n, gh_n) is a legal pair.
- **E2**: an exhaustive sweep over G 1..16, n 0..96, qt0 {0,3}, both depths (6208 cases), plus wrap == divmod.
- Mutation check: three deliberately wrong variants (nxo, pf_gh clamp, prologue n<=1 case) are each rejected by E1/E2.
- ISA WAR check now runs on k_dkdv and k_dkdv_sp.

## Files
- `kernels.py`, `bounds_proof.py`, `isa_stats.py` (loop stats tool), `.dump/{dkdv,dkdv_sp}/` ISA.
- `_ref_tdm/` = dkdv_tdm copy compiled for reference (incl. its first k_dkdv_sp compile); `_off/` = DIVFREE=False
  (ISA `cmp`-identical to `_ref_tdm/`); `_d2/`, `_ref_d2/` = TDM_DEPTH=2 sources (dumps removed after reading stats).

## Card order (when timed)
Same as dkdv_tdm: toy alone first (TDM hang risk), then fast (k_dkdv_sp), then prod; gate on bitwise dK/dV vs c1
(or vs dkdv_tdm) before timing.
