#!/usr/bin/env python3
"""Independent micro-test of packed-bf16 atomic add on this GPU.

The dkdv a16 probe says `buffer_atomic_pk_add_bf16` applies its addend TWICE (a lane-0-only
payload of (1.0, 4.0) reads back (2.0, 8.0) on exactly the elements it should touch, while a
`buffer_atomic_add_f32` in the same region counts exactly right). This asks the same question
with a completely different producer -- Triton's `tl.atomic_add`, which lowers to the GLOBAL
form -- so the answer separates "this hardware doubles packed-bf16 atomics" from "the buffer
form / our lowering does".
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _add_once(P, N, VAL: tl.constexpr):
    i = tl.program_id(0)
    o = i * 2 + tl.arange(0, 2)
    tl.atomic_add(P + o, tl.full((2,), VAL, tl.bfloat16), mask=o < N)


@triton.jit
def _add_f32(P, N, VAL: tl.constexpr):
    i = tl.program_id(0)
    o = i + tl.arange(0, 1)
    tl.atomic_add(P + o, tl.full((1,), VAL, tl.float32), mask=o < N)


N = 64
for dt, fn in ((torch.bfloat16, _add_once), (torch.float32, _add_f32)):
    x = torch.zeros(N, dtype=dt, device="cuda")
    g = (N // 2,) if dt is torch.bfloat16 else (N,)
    fn[g](x, N, 1.0)
    torch.cuda.synchronize()
    vals = sorted(set(x.float().cpu().tolist()))
    print("%-9s one atomic_add of 1.0 per element -> %s" % (str(dt).split(".")[-1], vals))

# and the asymmetric payload, which separates "applied twice" from "operand doubled"
x = torch.zeros(N, dtype=torch.bfloat16, device="cuda")


@triton.jit
def _add_pair(P, N):
    i = tl.program_id(0)
    o = i * 2 + tl.arange(0, 2)
    v = tl.where(tl.arange(0, 2) == 0, 1.0, 4.0).to(tl.bfloat16)
    tl.atomic_add(P + o, v, mask=o < N)


_add_pair[(N // 2,)](x, N)
torch.cuda.synchronize()
print("bfloat16  payload (1.0,4.0) per pair    ->", x.float().cpu()[:8].tolist())
