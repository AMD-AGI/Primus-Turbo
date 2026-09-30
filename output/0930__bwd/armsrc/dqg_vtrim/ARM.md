# ARM dqg_vtrim: k_dqg softmax/dS VALU trimmed and spread evenly under the WMMAs

Base: `arms/s4`. `impl.py`, `_env.py` and `__init__.py` are md5-identical to s4. All edits are in `kernels.py` k_dqg (`_dqg_impl`: `_smx`, `_sgb`, `_body_ck`, `_ndq`) behind new switches. k_dkdv / k_dkdv_sp / k_dq / k_dq_sp are untouched. The k_dkdv ISA is byte-identical to s4 (diff 0, 713 VGPR).

| switch | arm | s4 | what |
|---|---|---|---|
| `DQ_VT` | `"fma"` | `None` | `p = exp2(fma(s, scale*LOG2E, -lse*LOG2E))`, `ds = bf16(p * fma(dp, scale, -delta*scale))`. Row constants are hoisted out of the kv loop. The scale stays inside dS, so the epilogue is unchanged (this is not VF_Q). **Not bitwise.** |
| `DQ_VT_SCHED` | `"ck"` | `None` | Full-loop bodies are emitted as fenced chunks (see below). The mask body is s4's `_body`. |
| `DQ_VT_SGB` | `(2, 1)` | - | Inside each chunk: `sched_group_barrier` {1 WMMA, 2 VALU, 1 TRANS} x 8 |
| `DQ_VT_HST0` | `"all"` | - | Body 2 issues its 16 K `ds_store`s (kt0 and kt1) as one block at its head. |
| `DQ_VT_FMASK` | `0` | - | `sched_barrier` mask between chunks |
| `DQ_VT_SPREAD` | `False` | - | Tried; worse in the model (see ablation). |

## 1. Where the k_dqg VALU cycles go (s4 ATT, per-wave timelines, dispatch 11)

Wall(i) = issue(i+1) - issue(i), loop .LBB0_4, 1009 trips, 2771 wall cyc/trip.
- WMMA -> WMMA: 7.9 cyc. WMMA -> VALU: 1.25. VALU -> WMMA: 1.2. So a VALU placed right after a WMMA is hidden until the WMMA's 8-cycle slot runs out.
- VALU -> VALU runs cost 1.64 cyc per VALU, 605 cyc/trip (21.7%). exp -> exp costs 2.27, 220 cyc/trip (7.9%).
- s4 VALU per WMMA gap (`gap_stats.log`): 110 of 192 WMMA gaps have **no** VALU. 61 gaps carry more than 4 plain or more than 2 exp VALU, 526 VALU in total. The worst gap has 60 plain + 22 exp: body 1's kt1 softmax sits in front of its dtile-major dQ.
- Per 2 softmax elements, s4 issues 4 v_pk_mul + 2 v_pk_add + 2 v_exp + 1 cvt_pk. Per trip that is 256 pk_mul, 128 pk_add, 128 exp, 64 cvt.

Why history's VF/fma lost (#36, g91): those arms only cut instructions. LLVM's scheduler then re-clumped the VALU. This is visible at compile time here: fma alone (`q` / `ck` variants in `ablation.log`) keeps 7-VALU-per-gap clumps and 24-63-WMMA VALU-free runs, and gains only -2..-3% in the model. The count cut pays off only once the placement is forced.

## 2. The change

`_body_ck` (tailpf B1 and B2):
1. kt0 S/dP WMMAs as in s4. B1 keeps the pf2 kt0 clump.
2. kt1 K stores as one block, then fence.
3. kt1 in 4 chunks. Chunk dt = the 8 kt1 S/dP WMMAs of dtile dt + softmax/dS of (kt0, qh = dt). Each chunk is followed by `sched_group_barrier` {W1, V2, X1} x8 and `sched_barrier(0)`.
4. B1 only: the pf2 kt1 clump.
5. softmax/dS (kt1, qh0), plus the 16 `ds_load_tr` for b_k, then fence.
6. dQ qh-major in 4 chunks. Chunk q = the 8 dQ WMMAs of qh q + softmax/dS (kt1, qh q+1), with the same group pattern and fence.

Each chunk carries 24 VALU on 8 WMMAs: 16 plain + 8 exp, i.e. 2 plain + 1 exp per WMMA.

## 3. ISA evidence

Compile line: `tools/compile.sh arms/dqg_vtrim dkdv dkdv_sp dqg`
- k_dkdv 713 VGPR, byte-identical to s4.
- k_dkdv_sp 707 VGPR.
- **k_dqg 966 VGPR (s4: 1020), 0 spill, 0 scratch, LDS 8704.**

Hot loop `.LBB0_4` (`isa_gate.log`, `gap_stats.log`, `isa_rl.log` vs `isa_rl_s4.log`):

| | s4 | arm |
|---|---:|---:|
| loop instrs | 1013 | 853 |
| v_pk (mul+add / fma+mul) | 384 (256+128) | **192 (128 fma + 64 mul)** |
| v_exp / cvt_pk / WMMA | 128 / 64 / 192 | 128 / 64 / 192 |
| buffer_load / ds_store / ds_load_tr | 64 / 32 / 32 | 64 / 32 / 32 |
| v_nop | 50 | 65 |
| WMMA gaps with >4 plain or >2 exp | 61 (526 VALU) | **2 (54 VALU)**: the (kt1, qh0) unit per body, 18+9 |
| max VALU in one gap | 60 plain + 22 exp | 18 + 9 |
| WMMA->DS/VMEM switches | 7 | 5 |
| issue model cyc/trip | 2382 | **2037 (-14.5%)** |

The typical chunk in the ISA is `W1 v2 X1 W1 v2 X1 ...`.

## 4. Numerics (DQ_VT is not bitwise vs s4)

`numerics_check.py` -> `numerics_check.log` emulates one q tile in fp32, with bf16 inputs and bf16 dS, against an fp64 reference:
- The dQ SNR is identical to s4 in every trial: 52.52 / 52.49 / 52.58 / 53.04 dB.
- The max relative error is identical to 4 digits.
- dS bf16 differs in 0-0.07% of elements.

The fma form rounds once where s4 rounds 3 times. **Bitwise sibling (ready):** `.var_bitwise/` = this kernels.py with `DQ_VT=None`, `DQ_VT_SGB=(3, 1)`. It has 956 VGPR, 0 spill, model 2175 (-8.7%). Its ISA equals `.var_b31a`. It is bitwise vs s4 by the same proof: only the order changes.

## 5. Bounds / bitwise proof

`bounds_proof.py` -> `bounds_proof.log`, ALL OK.
- No index, address, predicate, loop-bound or soffset expression is new.
- The proof replays the program-order LDS/WMMA/softmax events of B1, B2 and M, s4 order vs arm order. It checks:
  - identical LDS writer per ds_load_tr chunk, and each chunk is written by the same body before it is read;
  - all chunks lie inside 8704 B;
  - identical dQ operand map, one WMMA per accumulator;
  - each S/dP chain is dt 0..3;
  - each softmax runs after its chain and before its dQ.
- Body-type sequences are covered for prod, fast, toy, gqa4_small and unequal_seqlen_2, causal 0/1. Only prod and proxy dispatch k_dqg.
- No TDM or tensorcnt in k_dqg.

## 6. Expected gain

- Model -345 cyc/trip. Scaled by the s4 calibration (ATT 2771 / model 2382) that is about -400 wall cyc/trip. The loop is about 88% of k_dqg.
- **k_dqg -8..-13% cycles** (3.07e6 -> ~2.7e6). Discount this: the model does not price the exp->mul TRANS32_DEP delays inside a chunk or the 15 extra v_nops.
- k_dqg runs concurrently with k_dkdv, and the chip is power- and clock-limited. The op is expected at **-1..-3%**, since fewer VALU cycles also mean less power.
- Risks:
  - Every earlier sched_(group_)barrier arm (#8) lost on the card. Those fenced memory ops; this arm fences only VALU/WMMA chunks, and all VMEM/DS placement matches s4.
  - If the fma arm loses, measure `.var_bitwise` to separate placement from count.

## Files

- Main: `kernels.py`, `bounds_proof.py/.log`, `numerics_check.py/.log`.
- ISA tooling: `isa_gate.py` (+ `isa_gate_lib.py`), `gap_stats.py/.log`, `isa_rl.py` + `isa_rl*.log`, `ablation.log`.
- Variant scaffolding: `mkvar.sh`, `.var_*` (the ablation variants; dumps are root-owned leftovers).
