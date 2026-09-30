# abl_a3 -- k_dkdv softmax-VALU ablation (WRONG OUTPUT, timing only)

## Change (kernels.py, vs arms/r29)
- L56: `ABL_A3 = True` (set False -> byte-identical r29 path).
- L420-442 (inside `_dkdv_impl._body`, per kh tile): new first branch.
  P = cvt_bf16(S), dS = cvt_bf16(dP). Skipped: `*scale`, `-lse`, `*LOG2E`, `exp2`, `-delta`,
  `pf*(...)`, `*scale`. Causal mask select kept (on S instead of S*scale) so diagonal tiles
  keep their cndmask work. lse/delta are kept live by one v_add each, only for kh==0:
  `P[0] += lse_q`, `dS[0] += del_q` -> 4 v_add_f32 per q-iteration (2 hh x 2).
- Applies to k_dkdv and k_dkdv_sp (shared `_dkdv_impl`); k_dqg untouched.
- First attempt (hoisted `lse_q + del_q` once per hh) was rejected: LLVM moved the add
  to the prefetch-load site and emitted `s_wait_loadcnt 0x0` right after the 4 b32 issue
  (a full global-latency stall every iteration). The sink now uses the tile's WMMA result
  as the other operand, so it stays at the consumer.

## ISA evidence (.dump/dkdv/k_dkdv_0/21_final_isa.s, compile.sh RC=0)
Resources: vgpr 734 (r29 729), spill 0, scratch 0, LDS 70656 (same).
Whole kernel: v_wmma 128/128, ds_* 224/224, buffer_load 140/140, buffer_load_b32 12/12,
v_exp 64 -> 0, VALU lines 1226 -> 1012.

| loop | v_exp | wmma | ds | buf_load | v_add_f32 | v_pk_* | VALU | SALU | loadcnt waits |
|---|---|---|---|---|---|---|---|---|---|
| r29 masked (.LBB0_4)   | 32 | 64 | 80 | 36 | 0 | 96 | 327 | 207 | 10 |
| a3  masked (.LBB0_4)   | 0  | 64 | 80 | 36 | 4 | 0  | 237 | 176 | 11 |
| r29 full   (.LBB0_11)  | 32 | 64 | 80 | 36 | 0 | 96 | 380 | 163 | 9 |
| a3  full   (.LBB0_11)  | 0  | 64 | 80 | 36 | 4 | 0  | 257 | 133 | 5 |

Hot (full) loop wait sequence: r29 `0xc,0x4,0x23,0x22,0x20,0x7,0x6,0x4,0x0`;
a3 `0xc,0x4,0x21,0x20,0x0`. The lse/delta consumer waits stay PARTIAL (0x21/0x20: 32 b128
still in flight), as in r29. The tail staggered b128 waits (0x7/0x6/0x4) collapse into
the single 0x0 because the body is shorter (506 vs 660 lines) -- expected side effect;
the prefetch cover distance shrinks, so part of any gain may be eaten by load latency.

## Bitwise vs r29
No -- output is intentionally wrong (ablation). Do not run validation against it.

## Bounds proof
No index/address/predicate expression changed (mask predicate text identical; only its
select operand changed from S*scale to S). All loads/stores/LDS offsets untouched, so no
bounds_proof.py is required.

## Reads as
dkdv time(r29) - time(a3) = upper bound on the real cost of the softmax VALU
(~123 VALU + ~30 SALU per hot-loop iteration). Small delta => exposed VALU is hidden.
