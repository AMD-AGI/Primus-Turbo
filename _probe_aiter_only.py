#!/usr/bin/env python3
"""aiter's backward on our ruler: its own public autograd API, min of N, this process."""
import json, sys, torch
B, Hq, Hkv, S, D, reps = (int(x) for x in sys.argv[1:7])
import aiter

# deterministic must be off: the gfx950 v3 gate is `not deterministic or seqlen_k <= 256`, and
# flash_attn_func defaults it to True, which sent the first two readings down the CK path at
# 10.7 ms. is_v3_atomic_fp32 needs no help -- the registered op already defaults it to True.

torch.manual_seed(11)
mk = lambda H: torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
q, k, v = mk(Hq), mk(Hkv), mk(Hkv)
out = aiter.flash_attn_func(
    q, k, v, causal=True, return_lse=True, deterministic=False
)[0]
go = torch.randn_like(out)
def bwd():
    for t in (q, k, v):
        t.grad = None
    out.backward(go, retain_graph=True)
for _ in range(5):
    bwd()
torch.cuda.synchronize()
best = float("inf")
for _ in range(reps):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record(); bwd(); e.record(); torch.cuda.synchronize()
    best = min(best, s.elapsed_time(e))
flops = 5 * B * Hq * S * S * D
r = {"aiter_ms": round(best, 4), "aiter_tfs": round(flops / best / 1e9, 1)}
open("/tmp/aiter_only.json", "w").write(json.dumps(r))
print(json.dumps(r))
