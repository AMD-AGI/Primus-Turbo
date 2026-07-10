"""Quantify the R3 dQ cross-kv-tile reduction cost.

R3 (dkdv emits dQ, delete fused) needs a deterministic split-K workspace for dQ
(atomics violate the bit-identical det gate). A feasible KV-outer design writes one
dQ partial per kv-tile (grouping kv-tiles into one WG blows up the dk/dv register
set), so #slots = num_kv_tiles = S/BLOCK_KV = 64 for the scored shape. This times
torch.sum over dQ-shaped [B,slots,S,Hq,D] workspaces to expose the reduction wall.
"""
import torch

dev = "cuda"
B, S, Hq, Hkv, D = 1, 8192, 64, 8, 64


def timeit(fn, warm=10, n=50):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(True)
    e1 = torch.cuda.Event(True)
    e0.record()
    for _ in range(n):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n * 1000.0  # us


print("== dQ reduce cost vs #kv-slots (R3 feasible design needs slots=64) ==")
for slots in [6, 8, 16, 32, 64]:
    ws = torch.randn(B, slots, S, Hq, D, device=dev, dtype=torch.bfloat16)
    us = timeit(lambda: torch.sum(ws, dim=1))
    gb = ws.numel() * 2 / 1e9
    print(f"  dQ reduce slots={slots:3d}  ws={gb:5.2f}GB  {us:8.1f}us")
    del ws
    torch.cuda.empty_cache()

print("== current dK/dV reduce (Hkv=8, slots=6) for reference ==")
wk = torch.randn(B, 6, S, Hkv, D, device=dev, dtype=torch.bfloat16)
us = timeit(lambda: torch.sum(wk, dim=1))
print(f"  dK reduce slots=6  ws={wk.numel()*2/1e9:.3f}GB  {us:.1f}us  (x2 for dk+dv)")
