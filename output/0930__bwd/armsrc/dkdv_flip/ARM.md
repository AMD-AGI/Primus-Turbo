# dkdv_flip

Status: **ready** (compile-only; not run on GPU). k_dkdv now computes S = Q·K^T and dP = dO·V^T
(M = q, N = kv). P and dS go straight from the C registers into the B operand of
dV^T = dO^T·P and dK^T = Q^T·dS. The P/dS LDS round trip is gone.

Switch: `kernels.py:60 DKDV_FLIP = True`. With `False`, k_dkdv is r29 byte for byte. The ISA of
`k_dkdv_r29` (FLIP=False) is instruction-identical to `arms/r29` (diff = 0 lines).

## Diff (kernels.py line numbers in this arm)

| lines | change |
|---|---|
| 56-60 | `DKDV_FLIP = True` |
| 174 | `_dkdv_impl(PARTIAL, FLIP, ...)` |
| 222-229 | FLIP: vec4 buffer views `g_lse4/g_del4` and `Sq4 = Sq//4`. `NLD` = LSE/delta head length in `pre` (8 vs 4) |
| 260-264 | FLIP: LDS = 2*32*X_ROW_B = 17408 B (the P/dS ring at +64 KB is dropped; r29 used 70656) |
| 305-321 | `_ldqd` FLIP: 8 `buffer_load_b128` (2 hh x {lse, del} x 2 u) at vec4 index `(bat*Hq+qh)*Sq4 + qt*8 + half*2 + hh*4 + u`. Replaces the 4 b32 loads. The `sched_barrier(0)` head-of-FIFO placement is unchanged |
| 410-415 | `p_bf/ds_bf[hh][kh]` holders. `pre[4+..]` becomes `pre[NLD+..]` |
| 430-483 | FLIP S/dP. `wmma(A=qfr, B=kf)` and `wmma(A=dfr, B=vf)`. Per-element `lse_q[si]`/`del_q[si] = pre[hh*4(+2) + si//4][si%4]`. Mask `kv0+kh*16+row > q0+hh*16+half*8+cshift+si`. VF_KV branch adapted too (off). P/dS kept as v8 bf16, **no ds_store** |
| 484-538 | `else:` the r29 body, re-indented only |
| 567-583 | FLIP output GEMM: `bP = concat(p_bf[0][kh], p_bf[1][kh])`, same for bdS. `wmma(A=b_do[dt], B=bP, reuseB=dt>0)` and `wmma(A=b_q[dt], B=bdS, reuseB=dt>0)`. No a_p/a_ds tr16. b_do/b_q tr16 loads are unchanged |
| 584-604 | `else:` the r29 output GEMM, re-indented |
| 737-772 | FLIP epilogue, direct stores with row = kv. bf16: vec8 `base_kv + (kv0+kh*16+row)*rs_kv + half + dtile*2` (16 b128 per tensor, no LDS stage). PARTIAL fp32: vec4 `base_o/4 + kv*Hkv*32 + half*2 + dtile*4 + u` (2 b128 per tile) |
| 852-882 | k_dkdv / k_dkdv_sp pass `DKDV_FLIP`. New `k_dkdv_r29` / `k_dkdv_sp_r29` (FLIP=False) |
| 906-926 | `launch_dkdv_r29`, `launch_dkdv_sp_r29` |

impl.py:195-210: `_flip_ok = sq % 8 == 0`. If it is false, the host dispatches the r29 launchers.
The existing `sq % 64` assert means this is unreachable today; the fallback is there as the
spec asked. k_dqg / k_delta / k_redsp are untouched.

Fragment layouts:
- A and B of gfx1250 wave32 16x16x32 bf16 share one layout: lane%16 = M/N row; elements
  0-7 = K half*8+e and 8-15 = K 16+half*8+e. r29 already feeds the same `gfrag` as both A
  (kf) and B (qfr).
- The C layout is GPU-verified in aiter `fmha_fwd_prefill_a16w16_m32x8.py:_qk_gemm`: lane l,
  element si = C[M=(l//16)*8+si, N=l%16]. The r29 causal mask (kvb+si, q=row) also depends on it.
- So hh0 C gives K = q 0-15 (half*8+si), hh1 C gives q 16-31, and concat(hh0, hh1) is exactly
  the B fragment. bounds_proof.py asserts this identity.

## ISA evidence (prod compile, `.dump/dkdv/k_dkdv_0/21_final_isa.s`)

| | r29 | dkdv_flip |
|---|---|---|
| vgpr / spill / scratch | 729 / 0 / 0 | **794** / 0 / 0 (kill: >850) |
| LDS | 70656 | 17408 |
| full-loop body (.LBB0_8 vs r29 .LBB0_11) instrs | 660 | 688 |
| WMMA per loop | 64 | 64 |
| ds_store_b128 per loop | 40 (32 Q/dO + 8 P/dS) | **32** (Q/dO only) |
| ds_load_tr16_b128 per loop | 40 | **32** |
| buffer_load b128 / b32 per loop | 32 / 4 | 40 / 0 (8 vec4 LSE/delta) |
| v_mov_b64 / v_nop per loop | 64 / 66 | 81 / 59 |
| first tr16 issued after WMMA # | 32 | **27** (inside the S/dP phase) |
| dscnt wait before 1st dV/dK WMMA | `0x6` | **`0x1b`**, then 0x15, 0x13, 0xe, 0xa, 0x6, 0x2 |
| `s_wait_dscnt 0x0` after WMMA # | 34 | **56** |
| epilogue | 16 ds_store + 16 tr16 + 32 b128 | 0 DS + 32 buffer_store_b128 |

- The kernel-wide ds count is 128. That is 64 in the masked loop plus 64 in the full loop;
  the epilogue has none.
- The Q/dO staging ds_stores now issue early, between S/dP WMMAs 5 and 8. The in-order DS
  drain before the dV/dK GEMM is replaced by graded partial waits.
- Mid-loop `s_wait_loadcnt`: none blocking. The top-of-loop `0x20`/`0x18`/`0x28` waits come
  right after the tail's `s_wait_loadcnt 0x0`, so they are no-ops.
- Cost: the 32 carried LSE/delta VGPR are rotated with 16 extra `v_mov_b64` at the loop head.
  VGPR goes from 729 to 794.
- Split-K kernel (compiled via a scratch copy `_chk_sp`, launch_dkdv redirected to k_dkdv_sp
  nsp=2): 781 VGPR, 0 spill, 17408 LDS, 64 buffer_store_b128, 0 b32 stores. Same loop shape
  (32 tr16, 32 ds_store, 40 b128).
- `_chk_r29`: k_dkdv_r29 compiled; its ISA diff against arms/r29 is 0 lines.
- `_chk_spr29`: k_dkdv_sp_r29 compiles (724 VGPR, 0 spill).
- The `_chk_*` .dump files are root-owned from the container and could not be deleted. Ignore them.

## Bitwise vs r29

**Not bitwise.** The WMMA operand roles change: S is computed as Q·K^T, and dV^T/dK^T have P/dS
as B. Summation order inside the WMMA can differ, and so can the fp32 accumulation. It should
be deterministic run to run (fixed order, no atomics). Gate: ≥50 dB vs the reference, plus
run-to-run bitwise.

## Bounds proof

`bounds_proof.py` (host python3 + numpy), output in `bounds_proof.log`: **ALL OK**.

What it replays:
- `_dkdv_impl` control flow exactly: qp_start, nmaskp, split-K chunks, the prologue
  `_clampqt`, and qloop_full's jj clamp.
- Every `_ldqd` call.

What it checks:
- (A) LSE/delta vec4. Every element is in range. Element e of (hh, u) equals LSE[b, qh, q] for
  the C-layout q. Q/dO are re-checked on the same qt set.
- (B) Mask. Swapped-role predicate == reference. The C tiles cover the 32x32 (q, kv) pair
  exactly once. The concat equals the B-fragment K order.
- (C) bf16 epilogue. In bounds, and every dK/dV element is written exactly once.
- (D) fp32 split epilogue. Same checks, over [nsp, B, Skv, Hkv, D].
- Every intermediate fits int32.

Shapes: prod b4 s8192 hq32 hkv8, fast b1 s1024 hq8 hkv2, toy b1 s128 hq2 hkv1, and Sq≠Skv
(512/1024, 1024/512). Each runs causal and non-causal, with nsp ∈ {1, 2, rule} (fast and toy
use rule = 16).
