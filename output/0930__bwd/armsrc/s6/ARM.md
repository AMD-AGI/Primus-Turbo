# ARM s5_vtrim: dqg_vtrim's k_dqg VALU trim ported onto s5's TDM k_dqg (compile-only screen)

Base: `arms/s5` (copied; no .dump/.compile.log). `impl.py`, `_env.py` and `__init__.py` are md5-identical
to s5. All edits are in `kernels.py` `_dqg_tdm_impl` behind new switches; `_dqg_impl`, k_dkdv, k_dkdv_sp,
k_dq, k_dq_sp are untouched.

| switch | arm | s5 | what |
|---|---|---|---|
| `DQT_VT` | `"fma"` | `None` | dqg_vtrim's DQ_VT: `p = exp2(fma(s, scale*LOG2E, -lse*LOG2E))`, `ds = bf16(p * fma(dp, scale, -delta*scale))`. Row constants hoisted above both kv loops. Full AND mask loop. **Not bitwise vs s5.** |
| `DQT_VT_SCHED` | `"ck"` | `None` | Full-loop body `_body_ck` (below). The mask loop keeps s5's `_body` order. |
| `DQT_VT_SGB` / `DQT_VT_SGB_W` | `(2, 1)` / `1` | - | In each chunk: `sched_group_barrier` {1 WMMA, 2 VALU, 1 TRANS} x 8 |
| `DQT_VT_FMASK` | `0` | - | `sched_barrier` mask between chunks |
| `DQT_VT_KEEP` | `False` | - | Variant: empty side-effect inline asm keeps the previous chunk's A operands live (see ablation) |

`DQT_VT=None, DQT_VT_SCHED=None` recompiles to s5's k_dqg and k_dkdv ISA **byte for byte** (0 diff
lines vs `arms/dqg_tdm/.dump`, which is md5-identical s5 source; dump in `.var/s5eq`).

## The change (`_body_ck`, full loop only)
1. TDM issue for tile min(i+2, N-1), fenced: **unchanged**.
2. kt0 S/dP 32 WMMAs, then `sched_barrier` (new: stops the scheduler pulling kt0 WMMAs into chunk 0,
   which put the kt0 softmax right behind its own chains and cost 17 RAW nops).
3. kt1 S/dP in 4 chunks. Chunk dt = the 8 kt1 WMMAs of dtile dt + softmax/dS of (kt0, qh = dt), fenced.
4. s5's DS phase: `sched_barrier(0)` | 16 tr16 (dQ B) | `sched_barrier(0)` | `s_wait_tensorcnt 2` |
   `sched_barrier(0)` | 32 ds_load_b128 readback of stage (i+1)%3 | `sched_barrier(0)`: **unchanged**.
5. softmax/dS (kt1, qh 0), fenced (hides the tr16 latency, as s5's phase "b" did with all the softmax).
6. dQ qh-major in 4 chunks: 8 dQ WMMAs of qh q + softmax/dS (kt1, qh q+1), fenced.

Each chunk carries 24 VALU on 8 WMMAs (16 plain + 8 exp). WMMA operands, accumulators and per-accumulator
WMMA order are s5's; only the WMMA<->VALU placement and the dS arithmetic change.

## ISA (prod b4 s8192 hq32 hkv8 causal; `isa_gate.log`, `isa_stats.log`, `nop_sites.log`)
Compile line: `tools/compile.sh arms/s5_vtrim dkdv dkdv_sp dqg` -> RC=0
- k_dkdv 713 VGPR, byte-identical to s5. k_dkdv_sp 707 VGPR, byte-identical to the None/None build.
- **k_dqg 881 VGPR (s5 878), 0 spill, 0 scratch, LDS 52224.**

Full loop `.LBB0_2` (one kv step):

| | s5 | s5_vtrim |
|---|---:|---:|
| instrs (all) | 574 | 605 |
| instrs excl. s_set_vgpr_msb / s_delay_alu | 489 | 417 |
| **VALU (excl. nop)** | **290** | **194** (-96) |
| v_pk (mul+add -> fma+mul) | 192 | 96 |
| v_exp / v_cvt_pk / WMMA | 64 / 32 / 96 | 64 / 32 / 96 |
| ds_load_b128 / tr16 / TDM / s_wait_tensorcnt | 32 / 16 / 2 / 1 | same, same order and offsets |
| s_wait_dscnt | 18 | 21 |
| **v_nop** | **5** | **26** |
| s_set_vgpr_msb | 83 | 143 |
| s_delay_alu | 2 | 45 |
| WMMA gaps with VALU / empty | 9 / 87 | 56 / 40 |
| gaps with >4 plain or >2 exp | 9 (290 VALU; worst 118 plain + 32 exp) | **1** (29 VALU: the (kt1, qh0) unit) |
| typical gap | - | 2 plain + 1 exp (44 of 96; + 0-4 nops in 11) |
| issue model cyc/step (dqg_vtrim lib) | 1206 | **926 (-23.2%)** |
| same model, s_set_vgpr_msb at 1 cyc | 1250 | **1005 (-19.6%)** |

Mask loop `.LBB0_6`: VALU 433 -> 337, instrs 929 -> 824 (fma form only; s5 order).

v_nop sites (`nop_sites.log`): 4x4 + 1x3 are WAR hazards at the end of kt1 chunks (the chunk's VALU is
register-allocated onto the kfr/vfr A operand of the chunk's last, still in-flight WMMA); 7x1 are
exp -> pk_mul. `DQT_VT_KEEP=True` removes the WAR nops (26 -> 7) but the new allocation adds 44
s_set_vgpr_msb (143 -> 187); the two models split (908 vs 926 without msb, 1016 vs 1005 with msb), so the
simpler `KEEP=False` ships. `.var/w1k` is the ready alternative if the card disagrees.

Ablation (`ablation.log`): 2-WMMA or 4-WMMA groups cut msb (104 / 83) but put 6-12 VALU behind a WMMA pair;
model -12..-9% only. No sched_group_barrier: -10%.

## Memory ops / bounds
No index, address, predicate, loop-bound, ring-counter or TDM expression is new or changed: `_body_ck`
calls s5's `_tdm_kv(pf_kv0, nxo)`, `_bks(cur)`, `tensor_wait(DQT_TW)`, `_rdkv(ncur)` with the same arguments,
and `_mkloop`/prologue/epilogue are untouched. Hence `arms/dqg_tdm/bounds_proof.py` (T1-T5) carries over
unchanged; no new proof is needed. ISA check `mem_check.py` -> `mem_check.log`: **ALL OK** -- the
program-order sequence of every ds_/tensor_/buffer_/global_ op (opcode + immediate offset) and
s_wait_tensorcnt is identical to s5 in the full loop (51), the mask loop (51) and the whole kernel (469).
`DQT_VT_KEEP` (off) would add only an empty `"; keep"` asm with 4 `v` uses, no instruction.

## Numerics
The dS formula is dqg_vtrim's `DQ_VT="fma"` expression for expression (same hoisted constants, same fma /
mul / exp2 / cvt), and s5's k_dqg is bitwise vs s4 (dqg_tdm B1/B2: same operand bits, one WMMA per
accumulator per kv block in block order). So s5_vtrim's dQ is **expected bitwise equal to dqg_vtrim's
dQ** and 81 dB from s5, with the dQ SQNR vs reference unchanged at 52.5 dB (`arms/dqg_vtrim/numerics_check.log`).
Check on the first proxy run: dq bitwise vs dqg_vtrim (only prod/proxy dispatch k_dqg).

## Expected gain
- Full-loop model -245..-280 cyc/step (-20..-23%), vs dqg_vtrim's -14.5% on s4's loop. The absolute VALU cut
  is the same (-96 per kv step); s5's loop is shorter, so the fraction is bigger.
- dqg_vtrim's model -14.5% measured -0.9% of the op on s4. Scaled: **op about -1.2..-1.5%**, prod
  5.339 -> ~5.27 ms (ASM 5.501: ~104.4%). Discount: +21 nops, +60 s_set_vgpr_msb and +43 s_delay_alu are
  only partly priced by the model, and k_dqg overlaps k_dkdv under a power/clock limit.
- Risk: earlier fenced-schedule arms lost on the card; this one fences only VALU/WMMA chunks and leaves
  every DS/TDM op, wait and offset exactly where s5 has it. If it loses, measure `.var/w1k`, then an
  fma-only build (`DQT_VT_SCHED=None`) to separate count from placement.

## Files
`kernels.py`; `isa_gate.py` (+ `isa_gate_lib.py` from dqg_vtrim) -> `isa_gate.log`; `nop_sites.py/.log`;
`mem_check.py/.log`; `isa_stats.log` (arms/dqg_tdm/isa_stats.py); `ablation.log`; `.var/<tag>/` variant ISAs,
`.var/s5eq/` the None/None dump.
