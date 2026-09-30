# arm tdm_df_tr: dkdv_tdm + dkdv_divfree + dkdv_trorder (k_dkdv / k_dkdv_sp)

Base: `arms/dkdv_tdm` (= c1 + TDM Q/dO ring). Stacked on top of it, all inside this directory:
- the tdm_df merge: `dkdv_divfree`'s carried (qi, gh) counters, plus a carried, wrapping ring-stage
  counter that replaces TDM's `ii % 3` / `(ii+2) % 3` / second `kk // G`;
- `dkdv_trorder` (variant a): the fenced 4-region emission of `_body`'s final DS phase.

Only `kernels.py` differs. `impl.py`, `_env.py` and `__init__.py` are byte-identical to r29, c1 and all
three source arms, so flydsl is still 0.3.2. k_dqg is untouched (DQ_U2=False as in c1).

Status: **compile-only.** RC=0, 0 spill, 0 scratch. The CPU bounds proof passes (ALL PASS). Nothing has been
run on a GPU.

## Switches (each lever can be turned off on its own)

| const (kernels.py) | default | off gives |
|---|---|---|
| `TDM_DEPTH` :57 | 3 | 2 = prefetch 1 ahead (TDM itself has no off switch, as in dkdv_tdm) |
| `DIVFREE` :59 | True | False = dkdv_tdm's division code, kept verbatim in the `else:` branches |
| `TRORDER` :61 | True | False = dkdv_tdm/r29 final DS phase, verbatim under `else:` |
| `QDO_BURST` :62 | False | trorder_b lever. N/A under TDM: there are no Q/dO staging stores. Asserted False (:746) |
| `KV_U2` | False | still asserted off (TDM ring not wired to qloop_full2, as in dkdv_tdm) |

I compiled the switch combinations in `_var/` (`d<DIVFREE><TRORDER>`). **`_var/d0t0` (both off) gives
k_dkdv ISA identical to `arms/dkdv_tdm/.dump` (cmp of the instruction text).**

## Changes vs dkdv_tdm/kernels.py

- :59-62 constants.
- :504-553 `if const_expr(TRORDER)`: trorder's R1..R4 block copied verbatim, including the `_a`/`_b`/`_mm`
  helpers and the `sched_barrier(0)` fences. `lds_do`/`lds_q` are dkdv_tdm's current-stage pointers
  (`_lds0 + cur_off`). :554-581 `else:` holds dkdv_tdm's code re-indented.
- :594-607 `_wrap(qc, gc)` (from dkdv_divfree) and a comment.
- :608-624 `qloop_mask`: under DIVFREE the counters sit at `st[NST], st[NST+1]` and the loop has a single
  `yield res` (dkdv_divfree's form).
- :626-684 `qloop_full` under DIVFREE. The carried tail is `[4 LSE/delta prefetch] + [qi, gh, cur]`.
  - `(qj, gj)`: `live = ii+1 < n ? wrap(qi,gh) : (qi,gh)` (divfree's jj clamp).
  - `cn = cur + QDO_B`, which becomes 0 at `TDM_DEPTH*QDO_B`. This is the next `cur`.
  - `nxo = cur < QDO_B ? (TDM_DEPTH-1)*QDO_B : cur - QDO_B`. That equals `(ii+DEPTH-1)%DEPTH * QDO_B`.
  - Prefetch tile at depth 3: `ii+2 < n ? wrap(wrap(qi,gh)) : (qj,gj)`. That equals `min(ii+2, n-1)`
    decomposed: once `ii+2 >= n`, `n-1` is `ii+1` or `ii`, and that is exactly the clamped jj.
  - At depth 2 it is `(qj, gj)`.
  - The `else:` branch is dkdv_tdm's code.
- :729-742 `_tdm_prologue`: under DIVFREE, j1 (which is 0 or 1) is decomposed as `1 < n ? wrap(0,0) : (0,0)`,
  replacing `j1 // G`.
- :763-768 `_z2` (qi, gh) = (0,0) and `_z1` (cur = 0).
- :786/:797: mask loops start from `init + _z2`.
- :791/:802: the full loop starts from `out[:NST] + _ldl(...) + _z2 + _z1`, in both the PARTIAL and the
  non-PARTIAL prologue.

## Compile (`tools/compile.sh arms/tdm_df_tr dkdv dkdv_sp`, compile-only)

```
tdm_df_tr   dkdv     LDS 70656  scratch 0  sgpr_spill 0  vgpr 590  vgpr_spill 0  wmma=128 ds=224
tdm_df_tr   dkdv_sp  LDS 70656  scratch 0  sgpr_spill 0  vgpr 586  vgpr_spill 0  wmma=128 ds=160
_var/d0t0   dkdv 576 / dkdv_sp 572   (= dkdv_tdm; first compile of dkdv_tdm's k_dkdv_sp, RC=0)
_var/d1t0   dkdv 576 / dkdv_sp 572   (tdm_df)
_var/d0t1   dkdv 590 / dkdv_sp 586   (tdm + trorder)
```
TRORDER costs +14 VGPR because 36 tr16 results are live at once. The kernel is already above 512, so it
stays at 1 wave/SIMD; occupancy does not change.

## ISA evidence (`isa_ev.py`, `isa_seq.py`; full loop k_dkdv `.LBB0_8`, k_dkdv_sp `.LBB0_6`, mask loop `.LBB0_2`)

| k_dkdv full loop | d0t0 (=dkdv_tdm) | d1t0 (tdm_df) | d0t1 | **tdm_df_tr** |
|---|--:|--:|--:|--:|
| instructions | 551 | 522 | 567 | **524** |
| `tensor_load_to_lds` / `s_wait_tensorcnt` | 2 / 0x2 | 2 / 0x2 | 2 / 0x2 | **2 / 0x2** |
| `buffer_load_b128` / `ds_store_b128` / `ds_load_b128` | 0/8/32 | 0/8/32 | 0/8/32 | **0/8/32** |
| SALU (excl. wait/msb/nop/branch) | 98 | 60 | 98 | **60** |
| `s_mul_hi` / `s_abs` | 4 / 2 | 0 / 0 | 4 / 2 | **0 / 0** |
| tr16 issued before the 1st dK/dV WMMA | 24 | 26 | 36 | **36** |
| `s_wait_dscnt` before the 1st dK/dV WMMA | 0x14 | 0x16 | 0x20 | **0x20** |
| first `s_wait_dscnt 0x0` among the 32 tail WMMAs | #25 | #25 | #25 | **#25** |

1. **TDM survived.** The whole kernel has 8 `tensor_load_to_lds`: 2 in the mask loop, 4 in the prologue and
   2 in the full loop. `buffer_load_b128` stays at 32 (K/V only). The explicit waits are the only ones: 0x0 in the
   mask loop, 0x2 in the full loop, and 0x0 before the epilogue (k_dkdv ISA:460/1378/1870). The ISA check in
   `bounds_proof.py` passes on both kernels:
   - no ds op comes before the tensor wait in either TDM loop;
   - `s_wait_dscnt 0x0` sits after the last ds_load and before the back edge. This is the WAR premise for the
     ring.
2. **DIVFREE survived.** k_dkdv has 0 `s_mul_hi` / 0 `s_abs` in the whole kernel (d0t0: 8 / 5). k_dkdv_sp keeps
   3 / 3, all outside the loops: the PARTIAL chunk division `(_fn+nsp-1)//nsp`, which divfree never touched.
   - Loop SALU drops 98 -> 60 (full loop) and 83 -> 66 (mask loop).
   - The full loop's index math (k_dkdv ISA:1324-1351) is two `_wrap` chains: `s_add 1 / s_cmp_ge G /
     s_cselect / s_add_co_ci`.
   - The ring counter shows up as its constants:
     `s_add s27, 0x4400 ; s_cmp_lt 0xcc00 ; s_cselect s26, s21, 0` (cn: +17408, wrap at 52224), then
     `s_add s27, -0x4400 ; s_cmp_gt s27, 0x43ff ; s_cselect s21, s21, 0x8800` (nxo: cur-17408, or 34816
     when cur == 0).
   - The back edge copies `s_mov s27, s26`.
3. **TRORDER survived** in all four loops (both kernels, mask and full loops):
   - 36 `ds_load_tr16` are issued back to back (R1+R2 fenced) before the first dK/dV WMMA, which waits at
     `s_wait_dscnt 0x20`. That is 4 of 36 retired: a_p0 and b_do[0]. The next waits step down 0x1e, 0x1c, ...
   - The 4 kh1 A loads (R3) follow the 8 dV kh0 WMMAs (k_dkdv ISA:1794 in isa_seq).
   - This is the same signature as dkdv_trorder on r29 (0x20, 36 + 4).

## Does trorder's final-phase order still make sense with TDM? Mostly yes, but expect a smaller gain than on r29

- **Correctness and dependences are unchanged.** The tr16 reads of dO/Q now read the current TDM stage, which the
  body-top `s_wait_tensorcnt` retired long before. P/dS still go through the 8 `ds_store_b128` into the P/dS ring,
  and DS returns in order, so the RAW on P/dS is served by `dscnt` exactly as before. The WAR on the ring
  (the next TDM into a stage happens only after that stage's last tr16 has retired) still holds: `s_wait_dscnt
  0x0` at tail WMMA #25, before the back edge (ISA check PASS).
- **Half of trorder's original rationale is gone.** On r29 the first dV WMMA waited behind 34 tr16 loads *plus 23
  Q/dO staging stores* (`dscnt 0x6`). TDM removed those staging stores, and with them gone LLVM already schedules
  the TDM base well without fences:
  - d1t0 issues 26 tr16 and waits at `0x16`, so it too waits for only 4 loads;
  - but its 8th WMMA then waits at `0x3` (22 of 26 retired);
  - it issues the remaining 14 loads in two later bursts.
- **What trorder still buys.** One uninterrupted burst of 36 loads, then a smooth one-per-WMMA countdown
  (0x20 -> 0x12 over the dV kh0 run) with no deep stall in the middle of the run. The LSE/delta
  `s_wait_loadcnt 0x0` also moves from after the 1st tail WMMA (d1t0 ISA:1759) to after the 17th (:1818).
  That is later consumption of the carried b32 prefetch, the same side effect trorder had on r29.
- **Cost:** +14 VGPR (590) and +2 instructions per iteration.
- **Net:** the ordering is still legal and still plausibly positive, but the measured -0.89% on r29 does not carry
  over automatically. Time `_var/d1t0` (tdm_df) against `tdm_df_tr` in the same process. If trorder
  does not pay on TDM, set `TRORDER = False`.

## Bitwise: expected identical to c1 / dkdv_tdm

- **DIVFREE.** It changes only the SALU that produces the integers. `bounds_proof.py` asserts equality with the
  division formulas for every iteration of every WG at prod, fast and toy, causal on and off, at depth 3 and at
  depth 2 (`divfree_eq` = 20.9M to 41.9M checks per prod case). The checked integers are:
  - (qt, gh) and (qt_n, gh_n);
  - the ring `cur` and `nxo`;
  - the prefetch tile;
  - the prologue j1 decomposition.
- **TRORDER.** It changes only issue order and fences. Every `new[k]` is still one WMMA on `acc[k]` with the same
  operands and the same reuseA flags.
- **TDM.** Same as dkdv_tdm, which is bitwise vs c1 on toy.

## Bounds proof: `python3 bounds_proof.py` gives RESULT: ALL PASS (~3.5 min, log in `bounds_proof.log`)

This is dkdv_tdm's proof, extended with four checks:
- **DF.** The kernel's DIVFREE arithmetic is replayed verbatim and asserted equal to dkdv_tdm's division path.
  Because every value is equal, dkdv_tdm's G1/L1/L2/C1/V1 checks all apply unchanged.
- **LD.** The LSE/delta index of every `_ldl` (the prologue's and every carried prefetch, all 32 rows) is below
  `B*Hq*Sq`.
- **TR.** For each of the 3 stages and all 32 lanes, the trorder tr16 address multiset equals the dkdv_tdm
  order's. All 3840 accesses stay inside their stage part or P/dS ring and never touch the pad.
- **ISA.** The WAR and tensor-wait checks now run on both k_dkdv and k_dkdv_sp.

I also mutation-tested the proof: an off-by-one in the `cn` wrap, a `+` for `-` in `nxo`, and a wrong `live2`
predicate are each caught with FAIL.

## Risks and notes

- This arm inherits all of dkdv_tdm's card-launch risks (first TDM use in bwd; in-order-retire premise for
  `tensor_wait(2)`). Launch order: toy, then fast, then prod, each in its own flocked process.
- `_var/*` are compile-only switch variants, kept as ISA evidence (the .dump directories are about 56 MB, and the
  compile container may own them).
