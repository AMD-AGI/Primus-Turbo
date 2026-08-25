import torch, aiter
from aiter.ops.mha import FlashAttnFunc
B, Hq, Hkv, S, D = 2, 64, 8, 8192, 128
mk = lambda H: torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
q, k, v = mk(Hq), mk(Hkv), mk(Hkv)
out = FlashAttnFunc.apply(q, k, v, 0.0, None, True, (-1, -1, 0), None, None, False, True, False, True, False)[0]
go = torch.randn_like(out)
for _ in range(6):
    for t in (q, k, v):
        t.grad = None
    out.backward(go, retain_graph=True)
torch.cuda.synchronize()
