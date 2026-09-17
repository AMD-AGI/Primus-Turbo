# gfx1250 FlyDSL kernels we wrote

**All three backward kernels PLAN.md Stage 3 lists now exist and pass**, in the order that
plan gives (odo, then dkdv, then dq -- ordered by what can be validated independently, not
by dataflow):

| kernel | SQNR |
|---|--:|
| `odo`  (delta = rowsum(dO*O)) | 159.24 / 156.49 dB |
| `dkdv` (dV, dK)               | 150.73 / 145.41 dB |
| `dq`                          | 144.90 dB |

Every one passed on its first run against torch, with NaN-prefilled outputs and full
isfinite coverage checked before SQNR. What exists is the complete backward **data path** at
single-tile, single-wave scope; what does not exist is everything that makes it a production
kernel -- causal masking, the kv/query loops, GQA, multi-wave scheduling, tail handling, the
bank-conflict swizzle, varlen, and the launcher.

Bring-up sources, not yet integrated. **Placement is an open question, not an oversight:**
these are written against flydsl **0.3.2**, and `primus_turbo/flydsl/` is a 0.2.4 tree that
0.3.2 cannot import (`flydsl.expr.buffer_ops` was removed). Dropping them in beside it would
produce a tree that no single flydsl version can load. See `../STAGE1-FWD.md` section 4.

| file | status |
|---|---|
| `odo_gfx1250.py` | **works.** `delta = rowsum(dO*O)`, BSHD in, `[B,H,S]` fp32 out |
| `wmma_layout_probe.py` | **works.** One 16x32 @ 32x16 WMMA tile against torch. Pins the fragment layout everything else will be built on |
| `qk_tile_gfx1250.py` | **works.** S^T = K @ Q^T over d, the accumulation chain (d=128: 143.07 dB) |
| `accumulator_operand_probe.py` | **negative result.** The accumulator is NOT operand-shaped on gfx1250 (-4.28 dB), so dkdv needs a transpose |
| `dq_operand_probe.py` | **works.** dq's operand IS free -- two kv-tile accumulators concatenate in-lane (151.32 dB) |
| `tr16_semantics_probe.py` | **decoded.** `ds_load_tr16_b128` -> lane l, elem e gets `src[(l//16)*8+e, l%16]` |
| `tr16_operand_e2e.py` | **works.** LDS + two tr16 loads build a WMMA A-operand (148.00 dB) |
| `dv_gfx1250.py` | **works.** `dV = P^T . dO` -- the first complete slice of dkdv (150.75 dB) |
| `dkdv_gfx1250.py` | **works.** `dV` AND `dK` in one kernel -- the expensive half of the backward (150.73 / 145.41 dB) |
| `dq_gfx1250.py` | **works.** `dQ = dS . K`, operand free from the accumulators, no LDS round trip for it (144.90 dB) |

## odo_gfx1250.py

Ported from aiter's gfx942 `k_delta` (`fmha_bwd_gfx942/fmha_bwd_core.py`), which is almost
arch-neutral -- no MFMA, no `ds_read_tr16`, no `permlane32_swap`. Two changes:

1. **Wave size.** The row reduction is four `shuffle_xor` steps. gfx942 passes width 64;
   gfx1250 dispatches wave32 and must pass 32. The butterfly spans only 16 lanes either way,
   so a wrong width does not raise -- it reads lanes that are not there. Full `isfinite`
   coverage at both shapes is the evidence it is right.
2. **Layout.** gfx942's is varlen THD with `DEL [H, T]`; this is BSHD `[B, S, H, D]` with
   delta `[B, H, S]`, matching the LSE the gfx1250 forward emits.

Measured on `heliosr-1b114-c07-1` @1100 MHz, `AMD_SERIALIZE_KERNEL=3`, toy shape launched
first in the same process:

| shape | build + first launch | isfinite | SQNR |
|---|--:|---|--:|
| b=1 s=256 h=2 | 0.28 s | 512/512 | **159.24 dB** |
| b=4 s=8192 h=32 | cached | 1048576/1048576 | **156.49 dB** |

Passed on the first run, which matches how the 0915 ASM bring-up went (odo first, 147 dB,
first try). That is the reason to start here: the reference is three lines of torch and it
needs no main kernel, so it validates the whole toolchain against something that cannot be
subtly wrong.

**Cold JIT build for a kernel this size is 0.28 s** -- worth knowing before sizing any
unattended loop's per-turn timeout. aiter's attention forward, which is far larger, took
about 2.4 s.

Known gap: `n_rows % 32 == 0` is asserted rather than handled. Both shapes here divide; a
tail pass is needed before this is general.

```bash
docker cp output/0917__flydsl/kernels/odo_gfx1250.py fa-repro:/tmp/
docker exec -e AMD_SERIALIZE_KERNEL=3 -e TORCH_BLAS_PREFER_HIPBLASLT=0 fa-repro \
  bash -lc 'cd /tmp && python3 odo_gfx1250.py'
```


## wmma_layout_probe.py

`dkdv` is the expensive kernel and all of its correctness rests on one thing: the gfx1250
WMMA fragment layout. A wrong operand layout produces a plausible wrong matrix, not an error,
so it is worth one tile of confirmation before writing hundreds of lines on top of it.

The layout, read off aiter's working forward and **now confirmed against torch on the card**:

```
v_wmma_f32_16x16x32_bf16, wave32:  C[16,16] f32 = A[16,32] @ B[32,16] + C

operand fragment   lane l in 0..31, element e in 0..15
    16-axis  (A's M, B's N)  = l % 16
    32-axis  (contraction K) = (l // 16) * 8 + (e % 8) + (e // 8) * 16
accumulator        lane l, element si in 0..7
    C[(l // 16) * 8 + si, l % 16]
```

A and B use the **same** fragment layout; only which logical axis sits on `l % 16` differs.
A fragment is two `v8` loads 16 elements apart, shuffled into a `v16` -- the `lo`/`hi` pair
in aiter's `load_q_to_vgpr_part2`.

| | |
|---|--:|
| isfinite | 256/256 |
| SQNR vs `A.float() @ Bt.float().T` | **150.34 dB** |
| max abs error | 9.54e-07 |

Sources for the layout: `fmha_b16_buffer_managers.py:1124-1143` (fragment construction),
`fmha_fwd_prefill_a16w16_m32x8.py:251-276` (the `_wmma` wrapper and the accumulator comment,
which says "GPU-verified" -- this file is our own independent check of that claim).


## dv_gfx1250.py -- the first real slice of dkdv

Chains everything above into one kernel that is checkable end to end:

```
S^T[kv,q] = K @ Q^T * scale          the d-tile accumulation chain
P^T[kv,q] = exp(S^T - lse[q])        fp32, in the accumulator
dV[kv,d]  = sum_q P^T[kv,q] dO[q,d]  contracts over q -- BOTH operands transposed
```

| | |
|---|--:|
| isfinite | 2048/2048 |
| SQNR vs torch (P truncated to bf16 the same way) | **150.75 dB** |
| max abs error | 1.19e-07 |

The part worth reusing: **both operands of the second GEMM come from the same trick.**
`ds_load_tr16_b128` over row-major LDS gives lane `l` a column, so

```
A = P^T  (M=kv, K=q)  <- P  staged [q][kv];  tr16 -> P[q=(l//16)*8+e, kv=l%16]
B = dO   (N=d,  K=q)  <- dO staged [q][d];   tr16 -> dO[q=(l//16)*8+e, d=l%16]
```

and the P store is **one b128 per q-tile**, because the S^T accumulator hands a lane eight
values consecutive in kv for a fixed q, which is exactly one row-major `[q][kv]` run.

Scope: one wave, one 16-row kv tile, NQ=32 queries, D=128, non-causal, no GQA, no tail
handling. Everything structural is exercised; everything that makes it a production kernel
(causal masking, the kv/query loops, GQA, multi-wave, the LDS swizzle, dK) is not.

Next: `dK = dS^T . Q`, which needs `dP = dO . V^T` and the `delta` that `odo_gfx1250.py`
already produces.


## dkdv_gfx1250.py -- dV and dK in one kernel

dK is structurally dV with substitutions, which is why they belong in one pass:

```
S^T[kv,q] = K @ Q^T * scale        dP^T[kv,q] = V @ dO^T          <- same GEMM shape
P^T       = exp(S^T - lse[q])      dS^T = P^T*(dP^T - delta[q])*scale
dV[kv,d]  = sum_q P^T  dO[q,d]     dK[kv,d] = sum_q dS^T Q[q,d]   <- same GEMM shape
```

| | isfinite | SQNR | max abs err |
|---|---|--:|--:|
| dV | 2048/2048 | **150.73 dB** | 1.19e-07 |
| dK | 2048/2048 | **145.41 dB** | 2.38e-07 |

Four operands, all through the same `ds_load_tr16_b128`-over-row-major-LDS path: P and dS
staged `[q][kv]`, dO and Q staged `[q][d]`. The P and dS stores are one `b128` each per
q-tile. `delta` comes from `odo_gfx1250.py`.

**A FlyDSL gotcha worth keeping:** a statement-level `for si in range(8)` that appends to a
list is rewritten into an `scf.for` with carried variables and fails with
`carried variable 'p_list' does not match the region entry`. Use a list comprehension, or
`range_constexpr`, for compile-time unrolling.

Scope: one wave, one 16-row kv tile, NQ=32, D=128, non-causal, no GQA, no tail handling, no
bank-conflict swizzle, and no kv/query loops. What is done is the whole data path and every
layout question in it; what is left is the production wrapping around it.

Not yet written: `dq = dS . K`, whose operand the `dq_operand_probe` showed comes free.


## dq_gfx1250.py -- the half where the operand is free

dkdv stages P^T / dS^T through LDS because it contracts over the query axis. dq contracts
over kv, and `dq_operand_probe.py` showed what follows: two consecutive kv-tile dS^T
accumulators **concatenate in lane** into one v16 A-operand -- no LDS round trip, no barrier
for that operand.

```
dQ[q,d] = sum_kv dS[q,kv] K[kv,d]     A = dS  free from two accumulators
                                      B = K   tr16 over K staged [kv][d]
```

| | isfinite | SQNR | max abs err |
|---|---|--:|--:|
| dQ | 2048/2048 | **144.90 dB** | 1.19e-07 |

The only LDS traffic is staging K, and the only barrier is for that. Scope: one wave, 16
queries, KV=32 (exactly one WMMA contraction), D=128, non-causal.
