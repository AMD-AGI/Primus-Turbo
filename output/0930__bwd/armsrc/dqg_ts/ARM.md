# ARM dqg_ts = dqg_tailpf + mem_soffset_q (both k_dqg levers stacked in the U2 loop)

Base: `arms/dqg_tailpf`. The k_dqg changes from `arms/mem_soffset_q` are merged in.
`k_dkdv`, `impl.py`, `_env.py` and `__init__.py` are r29. The three support files are
md5-identical to r29. The k_dkdv ISA is byte-identical to r29 after normalisation.
`DQ_U2 = True` is kept.

`kernels.py` is generated reproducibly by
`python3 .merge_patch.py ../dqg_tailpf/kernels.py kernels.py`. Every replacement in that
script is asserted to be unique.

## Switches (each lever can be turned off on its own)
| constant | default | off | what "off" gives (compile-verified) |
|---|---|---|---|
| `DQ_TAILPF` | `"b"` | `None` | k_dqg ISA == mem_soffset_q, byte-for-byte (`.sw_notail`) |
| `DQ_TAILPF_MASK` | `0x406` | - | tailpf fence mask, unchanged |
| `DQ_SOFF` | `True` | `False` | k_dqg ISA == dqg_tailpf, byte-for-byte (`.sw_nosoff`) |
| `DQ_SOFF_C` / `DQ_SOFF_CARRY` | `"a"` / `True` | as in mem_soffset_q | unchanged semantics |
| `DQ_TS_VBORDER` (new) | `None` = auto | `"g"` / `"k"` | IR order of the `soff_q` asm copies in a whole-block `_ldkv` (see below) |

## What changed vs dqg_tailpf (kernels.py)
- One new import: `arith as _arith`. There are the three `DQ_SOFF*` switches plus `DQ_TS_VBORDER`.
- The loop-invariant `rsrc_k/rsrc_v`, `voff_kv[kt]`, `rs_kv_b`, `_ovf()`, `_vb_cur` and `_NVB` are taken verbatim from mem_soffset_q.
- mem_soffset_q's `_ldkv` is split so that tailpf's per-kt clump can use it:
  - `_soff(kv0)` = `readfirstlane(kv0*rs_kv_b)`.
  - `_vb_kt(soff, kt)` is one empty inline asm `"; soff_q $0 $2", "=v,0,s"` for one kt. It carries `_vb_cur[0][kt]`.
  - `_ldkv_kt(kv0, kt, soff, vb)` issues the 16 `raw_ptr_buffer_load(v8bf16, rsrc, vb +nsw,nuw (dt*64+u*32), soff)` loads. The order is K u0, K u1, V u0, V u1 per dt, which is the gfrag2 order. It falls back to gfrag2 when `DQ_SOFF=False`.
  - `_ldkv` computes the soffset once and calls `_ldkv_kt` for each kt.
- In `_body`, when `pf2` is set and the variant is "b", `soff2 = _soff(pf2)` is computed once per body 1. Both per-kt pf2 clumps (`_ldkv_kt(pf2, kt, soff2)`, inside the 0x406 fences) share this one SGPR.
- The voffset carry (`_NVB` iter_args, from mem_soffset_q) is added to all three loops:
  - the tailpf `kvloop_full`: extract from `carried`, append after body 2;
  - the r29 U2 `elif` branch;
  - `kvloop_mask`.
- It is also added to the two loop init lists.
- In the tailpf loop, body 2 (`nxt_in`) issues no loads, so the carried voffset leaves body 1 from the pf2 asm copies.

### `DQ_TS_VBORDER` (the one merge-specific choice, compile-only evidence)
- `"g"`: all NKT asm copies are emitted first, then the loads. This is mem_soffset_q's IR order.
- `"k"`: asm kt0, loads kt0, asm kt1, loads kt1. This is the order the pf2 clump has anyway.
- `None` (auto) picks `"k"` when `DQ_TAILPF == "b"` and `"g"` otherwise. That is why turning tailpf off reproduces mem_soffset_q's ISA exactly.

| U2 loop `.LBB0_4` | "k" (default with tailpf) | "g" (`.sw_g`) |
|---|---|---|
| VGPR | 1016 | 1013 |
| loop instrs | 1270 | 1251 |
| first loop wait | `s_wait_loadcnt 0x1e` at instr 1 (the 2 oldest pf2 loads), then the pf2 set drains progressively (0x37..0x30 by WMMA 32) | **`s_wait_loadcnt 0x0` at instr 1**: all 32 pf2 loads drain before the head clump issues |

In "g", the scheduler hoists body 1's first S WMMA to the loop head. Its destination
`v[2:9]` (bank-shifted, really v[770:777]) shares an encoding with the last pf2 load's
`v[2:5]` (really v[514:517]). The only true dependency is on the first pf2 pair, which
needs 0x1e. The 0x0 looks like a false dependency in the waitcnt inserter over VGPR-MSB
banks. That is an inference and was not proven. "k" avoids it for +3 VGPR. Neither variant
has been timed.

## ISA evidence (compile.sh dkdv dqg, prod; `isa_ts.log`, `isa_gate.log`)
| | r29 | dqg_tailpf | mem_soffset_q | **dqg_ts** |
|---|---|---|---|---|
| k_dqg VGPR / SGPR | 991 / 89 | 1011 / 89 | 988 / 90 | **1016 / 90** |
| spill / scratch | 0/0/0 | 0/0/0 | 0/0/0 | **0 / 0 / 0** |
| k_dkdv VGPR | 729 | 729 | 729 | **729 (ISA == r29)** |
| U2 `.LBB0_4` instrs | 1310 | 1326 | 1183 | 1270 |
| U2 load clumps (size @ WMMAs before) | tail at WMMA 174-192 | 32@0 (split 5/24/3), **16@32, 16@64** | 32@0, 13@161, 19@186 | **32@1, 16@32, 16@64** |
| lead, last pf load to loop-head wait | 2 WMMA | 128 WMMA | - | **128 WMMA (754 instrs)** |
| U2 buffer_load with `offset:` imm | 0 | 0 | 56/64 | **56/64** (8 are imm 0) |
| U2 loads with SGPR soffset | 0 | 0 | 64/64 | **64/64** (head: s12, pf2: s3 = soff2) |
| U2 `v_or_b32` | 30 | 29 | 0 | **0** |
| U2 `s_wait_xcnt` | 3 | 0 | 0 | **0** |
| U2 `v_mov_b64` | 1 (tailpf ARM) | 1 | 1 | 1 |
| mask loop `.LBB0_8` xcnt / imm | 3 / 0 | 3 / 0 | 0 / 28 | **0 / 28** |

Lever signatures in dqg_ts:
- **tailpf**: the split per-kt pf2 clumps sit at WMMA 32 and WMMA 64, 16 loads each, the same position as dqg_tailpf. The head clump is 32 loads at the loop top. The head-clump waits `0x3e..0x20` fall at WMMA 64-142, the same as tailpf's WMMA 64-144.
- **soffset**: `soff_q` asm markers appear 4 times in the U2 loop and 2 times in the mask loop. There are 56/64 immediates, the SGPR soffset is on every load, and there are 0 v_or address rebuilds.
- **s_wait_xcnt**: 0 in the k_dqg hot loop and 0 in the mask loop. dqg_tailpf alone already had 0 in `.LBB0_4` and 3 in `.LBB0_8`. The merge removes the mask loop's 3.
- The kernel-wide xcnt count is 135. All of these waits are outside the loops (prologue and dq-store epilogue), as in mem_soffset_q.

## Bitwise
Expected to be bitwise-identical to r29, dqg_tailpf and mem_soffset_q:
- Each load reads the same 16 bytes into the same fragment slot. The proof below asserts arm byte == r29 byte.
- The set of blocks issued per trip equals r29's U2 set.
- The WMMA operands, the accumulation order and the LDS staging are unchanged.
- The two switch-off builds are ISA-identical to their source arms.

Not run on the GPU here.

## Bounds proof (host python3, run)
- `bounds_proof.py` is adapted from mem_soffset_q (`bounds_proof.log`). **ALL OK, 8,458,240 addresses.**
  - It covers prod, fast and toy, each with causal 0/1, every launched (bat, hkv), lane, kt, dt, u and kv0.
  - It checks arm byte == r29 byte, byte+16 <= numel(K)*2, and that no i32 intermediate wraps, so `nsw nuw` is truthful.
  - It checks voff+imm(+soff) < 2^30.
  - New: the U2 replay models the tailpf issue points: body 1 head (ii+1, kt 0/1), then pf2 (jj, kt0 and then kt1, one shared soff2), and no loads in body 2. For every trip it asserts that this (kv0, kt) multiset equals r29's U2 set.
- `bounds_proof_tailpf.py` is dqg_tailpf's proof, copied: **ALL IN BOUNDS**.

## Files
- `kernels.py`, `impl.py`, `_env.py`, `__init__.py`
- `.merge_patch.py`: generates kernels.py.
- `bounds_proof.py`, `bounds_proof.log`, `bounds_proof_tailpf.py`
- `isa_ts.py` / `isa_ts.log`: all the metrics above, plus the three byte-identity checks.
- `isa_gate.py` / `isa_gate.log`: tailpf's gate. The loop-end regex was loosened to any `s_cbranch_*`.
- `.sw_notail`, `.sw_nosoff`, `.sw_g`: the switch-off and variant builds.

## Caveats
- Compile-only. There is no GPU correctness or timing run.
- VGPR is 1016/1024, so there are 8 left.
- The `soff_q` asm is a compiler-steering device. Re-check the ISA after a toolchain bump.
