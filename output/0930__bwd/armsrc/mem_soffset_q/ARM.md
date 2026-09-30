# ARM mem_soffset_q (k_dqg K/V loop loads: invariant voffset + uniform soffset + imm)

Base: arms/r29. Only `_dqg_impl` (k_dqg) changes. The k_dkdv ISA is byte-identical to r29
(the diff of the non-comment lines is empty). The k_dq_sp split path is unchanged.
The k_dqg descriptors stay at the fake 1<<30 size.

## What changed (line numbers are in the arm's kernels.py)
- L39: `from flydsl._mlir.dialects import arith as _arith` (used for `addi` with overflow flags).
- L1176-1181: switches.
  - `DQ_SOFF = True`: the new load path. `False` gives back the r29 gfrag2 loads.
  - `DQ_SOFF_C = "a"`: where the column constant goes. Other values: "v" and "s" (see Variants).
  - `DQ_SOFF_CARRY = True`: carry the voffset through the loop.
- L1275-1293: loop-invariant setup.
  - `rsrc_k` and `rsrc_v` come from `rocdl.get_buffer_rsrc(fx.get_iter(g_k/g_v))`.
  - `voff_kv[kt] = 16*(base_kv + (kt*16+row)*rs_kv + half)`: one per-lane VGPR per kt.
  - `rs_kv_b = 16*rs_kv`.
  - `_ovf()` builds the `#arith.overflow<nsw, nuw>` attribute.
- L1295-1330: `_ldkv(kv0)` with DQ_SOFF.
  - `soff = readfirstlane(kv0*rs_kv_b)` goes in an SGPR.
  - Each load is `rocdl.raw_ptr_buffer_load(v8bf16, rsrc, vb[kt] +nsw,nuw (dt*64+u*32), soff)`.
  - `vb[kt]` is the output of an empty inline asm `"; soff_q $0 $2", "=v,0,s"(voff, soff)`. It has no side effects and emits no code.
  - Output order is the same as r29: K u0, K u1, V u0, V u1 for each (kt, dt).
- L1432-1447 (kvloop_full, U2) and L1474-1488 (kvloop_mask): the two `vb` values are appended to the scf.for iter_args. Each body's asm takes the previous asm output as input, so the tied output reuses the same register.
- L1502-1503 and L1510-1511: the loop init gets `voff_kv`. The results after the loop are sliced exactly as before.

### Why the inline asm and the carry are needed (compile-only iterations)
| variant | loop ISA | VGPR |
|---|---|---|
| "v": plain `voff + C` | IR LICM hoists the 16 invariant `voff+C` into 16 VGPRs. No imm. Loop has 1 xcnt (from SGPR recycle) | 1000 |
| "s": `soff + C` in soffset | 8 `s_add` per body. The SGPR recycle brings back 3 xcnt | 999 |
| "a", asm "=v,0" with no loop-variant operand | The asm is hoisted. Same as "v" | 1000 |
| "a" + soff operand, no carry | Not hoisted. BUT with plain `add`, ISel folds C into `offset:` only when soffset is null. With a non-zero soffset it emits v_add rebuilds | 996 |
| "a" + `add nsw nuw` | Imm folded (56/64). 0 xcnt, 0 v_or, but 2 v_mov copies per trip | 996 |
| "a" + nsw/nuw + carry in the full loop only | Loop clean, no copies. The mask loop still has a copy | 994 |
| **"a" + nsw/nuw + carry in both loops (default)** | see below | **988** |

LLVM honoured the intent only once all three pieces were in place: the loop-variant opaque base, the nuw/nsw flags and the carried register.

## ISA evidence (compile.sh, prod shape; .dump/dqg/k_dqg_0/21_final_isa.s)
k_dqg: vgpr 988 (r29: 991). sgpr 90 (r29: 89). spill 0 / 0. private_segment 0. LDS 8704. wmma 288. ds 96.

| loop | metric | r29 | arm |
|---|---|---|---|
| U2 full loop `.LBB0_4` | lines | 1310 | 1195 |
| | buffer_load | 64 | 64 |
| | with `offset:` imm | 0 | 56 (the other 8 are dt=0,u=0 → imm 0) |
| | s_wait_xcnt | 3 (incl. the 0x2 at the v_or v90,0xc0,v69 rebuild) | **0** |
| | v_or_b32 | 30 | **0** |
| | v_mov copies | 0 | 0 |
| | loop VALU address ops | 1 v_add (v69 += s11) | 0 (only `s_mul`/`s_add` scalar) |
| causal mask loop `.LBB0_8` | buffer_load | 32 | 32 |
| | with imm | 0 | 28 |
| | s_wait_xcnt | 3 | **0** |
| | v_or_b32 | 23 | 14 (all 14 are the causal-mask column indices `v_or vX, 6/7/16/17.., v107`, not addresses) |

Typical loop-head code:

    ; soff_q v196 s11         <- empty asm, no instruction
    buffer_load_b128 v[246:249], v196, s[72:75], s11 offen
    buffer_load_b128 v[250:253], v196, s[72:75], s11 offen offset:32
    ... offset:64/96/128/160/192/224 ...

ISA gate: 0 s_wait_xcnt in the loop (PASS), at most 2 v_or in the loop (0: PASS), VGPR at most 991 (988: PASS).

Kernel-wide xcnt count: r29 124, arm 134. All the arm's xcnt waits are outside the loops: 1 in the prologue and 133 in the dq-store epilogue. The epilogue schedule shifted a little. This is not per trip.

## Bitwise vs r29
Expected bitwise-identical. Every lane loads the same 16 bytes into the same fragment slot. The CPU proof checks arm byte == r29 byte on all addresses. The data path, the WMMA order and the LDS staging are unchanged.

## Bounds proof
`bounds_proof.py` (host python3, no torch): **ALL OK**, 8,458,240 addresses.
- Shapes: prod b4 s8192 hq32 hkv8, fast b1 s1024 hq8 hkv2, toy b1 s128 hq2 hkv1. Each is checked with causal 0 and 1.
- It replays the kv0 trips of the prologue, of kvloop_full U2 (including the clamped jj) and of kvloop_mask, for every q tile.
- It covers every launched (bat, hkv) under the XCD remap, and every lane, kt, dt and u.
- Checks:
  - arm byte == r29 byte.
  - 0 <= byte and byte+16 <= numel(K)*2.
  - Every i32 intermediate is in [0, 2^31). So the `nsw nuw` flags are truthful.
  - voffset+imm < 2^30 and voffset+imm+soffset < 2^30. So the fake num_records never clamps under either range-check convention.

The fast shape normally takes the k_dq_sp path, which is unchanged. It is proven here anyway.

## Not done / caveats
- Compile-only. There is no GPU validation or timing.
- The `DQ_SOFF=False` fallback was not re-compiled. It is the untouched r29 code path, but the new `_NVB` / `voff_kv` guards keep the loop state identical there.
- The inline asm is a compiler-steering device. A future LLVM could hoist or CSE it differently, so re-check the loop ISA after a toolchain bump.
